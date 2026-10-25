#!/usr/bin/env python3
"""gen_ygrid.py -- generate CLAUSON wall-normal grid FILES.

DESIGN: the solver never computes a stretched grid
in Fortran -- it READS the y distribution from an ASCII file (src/core/
ygrid_mod.F90). This tool GENERATES those files; the grid laws live here in
Python where they are inspectable.

Grid kinds:
  natural  P&O-natural, alpha-corrected (Pirozzoli & Orlandi JCP 2021 with an
           alpha = 0.35-0.40 log-layer exponent): spacing
           law in plus units  delta(y+) = cap * tanh(law/cap),
           law = (dwall^q + (c y+^alpha)^q)^(1/q)  -- ported VERBATIM from the
           validated model src/models/work_fd_ygrid_model.py (natural_*).
           Marched from the wall, rescaled to land exactly on retau, mirrored
           to the full channel. Specify c directly or --ny to bisect c.
  glc      Gauss-Lobatto-Chebyshev y_j = cos(pi j / n) (the Cheb backend's
           built-in grid; here for cross-checks and reader tests).

FILE FORMAT (parsed by ygrid_mod.F90; '#' lines and blank lines are comments):
    # free-form provenance comments
    ny = <N>                    (number of NODES, ny = n+1)
    columns = y [w]             (w = quadrature weight, optional)
    <y_0> [w_0]                 (ny data rows; y DESCENDING from +1 to -1,
    ...                          walls EXACTLY +-1 -- the code's storage order,
    <y_{ny-1}> [w_{ny-1}]        matching y_j = cos(pi j/n) direction)

Weights: --weights trapz (default) writes composite-trapezoid weights
(2nd order -- fine for diagnostics; use --weights fd8 for weights matched to
the kl=4 stencils). --weights none omits the column.

Usage:
  python3 tools/gen_ygrid.py natural --retau 2000 --ny 481 -o ygrid.dat
  python3 tools/gen_ygrid.py natural --retau 5200 --alpha 0.40 --c 0.6 -o g.dat
  python3 tools/gen_ygrid.py glc --n 480 -o ygrid_glc.dat
  python3 tools/gen_ygrid.py --selftest
"""
import argparse
import sys
import numpy as np


# ----- P&O-natural, alpha-corrected (VERBATIM port of work_fd_ygrid_model) ----
def natural_spacings(retau, alpha=0.375, dwall=0.6, c=0.6, cap=22.0, q=2.0):
    def delta(yy):
        law = (dwall**q + (c * yy**alpha)**q) ** (1.0 / q)
        return cap * np.tanh(law / cap)
    dyp = []
    y = 0.0
    while y < retau:
        d = delta(y + 0.5 * delta(y))            # midpoint evaluation
        dyp.append(d)
        y += d
    dyp = np.array(dyp)
    return dyp * (retau / dyp.sum())             # exact landing (factor ~1)


def natural_c_for_ny(retau, ny_half_target, alpha, dwall, cap):
    """Bisect c to hit the half-channel count. Returns the HI endpoint (the
    count <= target side): the midpoint sits ON the count discontinuity and
    which side it lands on is last-ulp/libm luck (a Re5200 --ny 1025 request
    can silently write 1027 that way). Callers must
    still VERIFY the achieved count (main() errors on mismatch)."""
    lo, hi = 0.01, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        n = len(natural_spacings(retau, alpha, dwall, mid, cap))
        if n > ny_half_target:
            lo = mid
        else:
            hi = mid
    return hi


def natural_grid(retau, alpha=0.375, dwall=0.6, c=0.6, cap=22.0):
    """Full-channel y, DESCENDING +1 -> -1 (the code's storage order)."""
    dy = natural_spacings(retau, alpha, dwall, c, cap) / retau
    half = np.concatenate([[0.0], np.cumsum(dy)])
    half[-1] = 1.0
    y = np.concatenate([half - 1.0, (1.0 - half[::-1])[1:]])   # ascending -1..+1
    return y[::-1].copy()                                       # descending +1..-1


def glc_grid(n):
    return np.cos(np.pi * np.arange(n + 1) / n)                # +1 -> -1


def trapz_weights(y):
    """Composite-trapezoid weights on a DESCENDING grid; sum(w) = 2 exactly-ish.
    w_j = (y_{j-1} - y_{j+1})/2 interior; half-cells at the walls."""
    ny = len(y)
    w = np.zeros(ny)
    w[0] = 0.5 * (y[0] - y[1])
    w[-1] = 0.5 * (y[-2] - y[-1])
    w[1:-1] = 0.5 * (y[:-2] - y[2:])
    return w


def fd8_weights(y, kl=4):
    """FD-scheme-matched quadrature on a DESCENDING grid: per interval
    [y_{i+1}, y_i], integrate the degree-2kl local Lagrange interpolant on the
    2kl+1 nodes centred on the interval (clipped at the walls) -- the same
    stencil family as the solver's kl=4 banded operators. Exact for
    polynomials of degree <= 2kl per interval (8th order on smooth grids vs
    trapezoid's 2nd); sum(w) = 2 to round-off (constants are exact per
    interval and the intervals tile [-1, 1]).

    Per-interval weights come from a moment match on a shifted/scaled
    Vandermonde (centre = interval midpoint, scale = half the stencil span;
    9x9, condition ~1e3 on the shipped grids -- benign in double precision).

    POSITIVITY IS NOT GUARANTEED by construction (interpolatory rules can go
    negative near sharp spacing changes) -- so it is CHECKED here, hard, and
    also re-checked by the solver's reader (ygrid_mod refuses w <= 0 and
    sum(w) /= 2). On the shipped P&O-natural
    campaign grids all weights are comfortably positive.
    """
    ny = len(y)
    m = 2 * kl + 1
    w = np.zeros(ny)
    for i in range(ny - 1):                      # interval [y[i+1], y[i]]
        lo = max(0, min(i - kl + 1, ny - m))     # stencil start, centred+clipped
        s = y[lo:lo + m]
        mid = 0.5 * (y[i] + y[i + 1])
        scl = 0.5 * (s[0] - s[-1])               # half the stencil span (>0, descending)
        t = (s - mid) / scl                      # scaled stencil coords
        a = (y[i] - mid) / scl                   # scaled interval endpoints
        b = (y[i + 1] - mid) / scl
        V = np.vander(t, m, increasing=True).T   # V[p, j] = t_j^p
        mom = np.array([(a ** (p + 1) - b ** (p + 1)) / (p + 1) for p in range(m)])
        w[lo:lo + m] += np.linalg.solve(V, mom) * scl
    if np.any(w <= 0.0):
        bad = np.where(w <= 0.0)[0]
        sys.exit(f"[gen_ygrid] ERROR: fd8 weights non-positive at nodes {bad[:5]}"
                 f" (min w = {w.min():.3e}) -- this grid's spacing profile breaks"
                 f" the interpolatory rule's positivity; use --weights trapz or"
                 f" smooth the grid law (the solver's reader would refuse this"
                 f" file anyway)")
    if abs(w.sum() - 2.0) > 1e-12:
        sys.exit(f"[gen_ygrid] ERROR: fd8 weights sum to {w.sum():.15f}, not 2")
    return w


def write_grid(path, y, w, header_lines):
    with open(path, "w") as f:
        f.write("# CLAUSON y-grid file (read by ygrid_mod.F90; see tools/gen_ygrid.py)\n")
        for ln in header_lines:
            f.write(f"# {ln}\n")
        f.write(f"ny = {len(y)}\n")
        f.write(f"columns = {'y w' if w is not None else 'y'}\n")
        for j in range(len(y)):
            if w is not None:
                f.write(f"{y[j]: .17e}  {w[j]: .17e}\n")
            else:
                f.write(f"{y[j]: .17e}\n")


def read_grid(path):
    """Reference reader (mirrors ygrid_mod.F90's grammar)."""
    ny = None
    cols = None
    rows = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("ny"):
                ny = int(s.split("=", 1)[1])
            elif s.startswith("columns"):
                cols = s.split("=", 1)[1].split()
            else:
                rows.append([float(x) for x in s.split()])
    a = np.array(rows)
    assert ny == len(a), f"{path}: ny={ny} but {len(a)} rows"
    y = a[:, 0]
    w = a[:, 1] if len(cols) > 1 else None
    return y, w


def validate(y, w, label, dyp_wall_max=None, retau=None):
    """The invariants ygrid_mod enforces + physics sanity; raises on failure."""
    assert abs(y[0] - 1.0) < 1e-14 and abs(y[-1] + 1.0) < 1e-14, f"{label}: walls not at +-1"
    assert np.all(np.diff(y) < 0), f"{label}: y not strictly decreasing"
    assert np.allclose(y, -y[::-1], atol=1e-13), f"{label}: not symmetric"
    if w is not None:
        assert abs(w.sum() - 2.0) < 1e-12, f"{label}: sum w = {w.sum()} != 2"
        assert abs((w * y).sum()) < 1e-13, f"{label}: int y != 0"
        tol2 = max(1e-6, np.max(-np.diff(y)) ** 2)     # trapz global error ~ h_max^2
        assert abs((w * y * y).sum() - 2.0 / 3.0) < tol2, f"{label}: int y^2 off (trapz tol)"
    if dyp_wall_max is not None and retau is not None:
        dyp1 = (y[0] - y[1]) * retau
        assert dyp1 < dyp_wall_max, f"{label}: dy+_wall = {dyp1:.3f} > {dyp_wall_max}"


def selftest():
    ok = True
    # natural at Re2000 with --ny targeting
    retau, alpha, dwall, cap = 2000.0, 0.375, 0.6, 22.0
    tgt_half = 240
    c = natural_c_for_ny(retau, tgt_half, alpha, dwall, cap)
    y = natural_grid(retau, alpha, dwall, c, cap)
    w = trapz_weights(y)
    validate(y, w, "natural Re2000", dyp_wall_max=2.0 * dwall, retau=retau)
    print(f"[PASS] natural Re2000: ny={len(y)} (target half {tgt_half}, c={c:.4f}), "
          f"dy+_wall={(y[0]-y[1])*retau:.3f}, dy+_CL={(y[len(y)//2-1]-y[len(y)//2])*retau:.2f}")
    # glc + exactness of the reference reader round-trip
    yg = glc_grid(48)
    wg = trapz_weights(yg)
    validate(yg, wg, "glc n=48")
    import tempfile, os
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "g.dat")
    write_grid(p, y, w, ["selftest natural"])
    y2, w2 = read_grid(p)
    assert np.array_equal(y, y2) and np.array_equal(w, w2), "ASCII round-trip not exact"
    print("[PASS] glc invariants + ASCII round-trip bit-exact (17-digit format)")
    p2 = os.path.join(tmp, "g2.dat")
    write_grid(p2, yg, None, ["no weights"])
    y3, w3 = read_grid(p2)
    assert w3 is None and np.array_equal(yg, y3)
    print("[PASS] weight-less file round-trip")
    print("ALL TESTS PASSED")
    return 0


def reweight(src, dst, kind):
    """Replace ONLY the w column of an existing grid file. Nodes are
    re-emitted from the parsed doubles with the same '% .17e' format the
    writer always used, so the y column is BYTE-IDENTICAL for any file this
    tool wrote (17 significant digits round-trip IEEE doubles exactly) --
    the validated physics baseline is untouched. Original '#' header lines
    are preserved; any old 'weights =' line is replaced."""
    if kind == "none":
        sys.exit("[gen_ygrid] --reweight with --weights none makes no sense")
    y, _w_old = read_grid(src)
    hdr = []
    with open(src) as f:
        for ln in f:
            s = ln.strip()
            if s.startswith("#"):
                body = s.lstrip("#").strip()
                if body.startswith("CLAUSON y-grid file"):
                    continue                      # the writer re-emits this line
                if body.startswith("weights ="):
                    continue                      # replaced below
                hdr.append(body)
    w = fd8_weights(y) if kind == "fd8" else trapz_weights(y)
    hdr.append("weights = FD-scheme-matched 9-point interpolatory (degree-8 "
               "exact per interval; positive; sum = 2)" if kind == "fd8" else
               "weights = composite trapezoid (2nd order)")
    hdr.append(f"reweighted from {src.split('/')[-1]} (nodes preserved bit-exactly)")
    validate(y, w, dst)
    write_grid(dst, y, w, hdr)
    print(f"[gen_ygrid] reweighted {dst}: ny={len(y)}  sum(w)={w.sum():.15f}"
          f"  min(w)={w.min():.3e}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", nargs="?", choices=["natural", "glc"], help="grid law")
    ap.add_argument("--retau", type=float, help="target friction Reynolds number (natural)")
    ap.add_argument("--alpha", type=float, default=0.375)
    ap.add_argument("--dwall", type=float, default=0.6, help="dy+ at the wall")
    ap.add_argument("--cap", type=float, default=22.0, help="dy+ cap (tanh, C-inf)")
    ap.add_argument("--c", type=float, help="growth coefficient (else bisected from --ny)")
    ap.add_argument("--ny", type=int, help="target TOTAL node count (bisects c; "
                    "ENFORCED -- errors if the law cannot land on it, try ny+-2)")
    ap.add_argument("--n", type=int, help="polynomial degree for glc (ny = n+1)")
    ap.add_argument("--weights", choices=["trapz", "fd8", "none"], default="trapz")
    ap.add_argument("--reweight", metavar="FILE",
                    help="read FILE's nodes (preserved BIT-EXACTLY -- the "
                         "validated physics baseline), recompute only the w "
                         "column with --weights, rewrite in place (or -o). "
                         "Original header lines are carried over; the old "
                         "'weights =' line is replaced.")
    ap.add_argument("-o", "--out", help="output file")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.reweight:
        return reweight(args.reweight, args.out or args.reweight, args.weights)
    if not args.kind or not args.out:
        ap.error("kind and -o are required (or --selftest)")

    if args.kind == "natural":
        if args.retau is None:
            ap.error("natural needs --retau")
        c = args.c
        if c is None:
            if args.ny is None:
                ap.error("natural needs --c or --ny")
            c = natural_c_for_ny(args.retau, (args.ny - 1) // 2, args.alpha,
                                 args.dwall, args.cap)
        y = natural_grid(args.retau, args.alpha, args.dwall, c, args.cap)
        if args.ny is not None and len(y) != args.ny:
            sys.exit(f"[gen_ygrid] ERROR: requested --ny {args.ny} but the law "
                     f"lands on ny = {len(y)} (count steps discretely with c); "
                     f"try --ny {len(y)} or adjust --alpha/--dwall/--cap")
        hdr = [f"kind = natural (P&O alpha-corrected)",
               f"retau = {args.retau}  alpha = {args.alpha}  dwall = {args.dwall}"
               f"  c = {c:.6f}  cap = {args.cap}",
               f"dy+_wall = {(y[0]-y[1])*args.retau:.4f}"
               f"  dy+_CL = {(y[len(y)//2-1]-y[len(y)//2])*args.retau:.3f}"]
    else:
        if args.n is None:
            ap.error("glc needs --n")
        y = glc_grid(args.n)
        hdr = [f"kind = glc  n = {args.n}"]

    w = None
    if args.weights == "trapz":
        w = trapz_weights(y)
        hdr.append("weights = composite trapezoid (2nd order; regenerate with "
                   "--weights fd8 for the scheme-matched rule)")
    elif args.weights == "fd8":
        w = fd8_weights(y)
        hdr.append("weights = FD-scheme-matched 9-point interpolatory "
                   "(degree-8 exact per interval; positive; sum = 2)")
    validate(y, w, args.out)
    write_grid(args.out, y, w, hdr)
    print(f"[gen_ygrid] wrote {args.out}: ny={len(y)}  y[0]={y[0]:+.1f} .. y[-1]={y[-1]:+.1f}"
          + (f"  sum(w)={w.sum():.15f}" if w is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
