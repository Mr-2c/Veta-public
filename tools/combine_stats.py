#!/usr/bin/env python3
"""
combine_stats.py -- pool the stats files of a restart-chained campaign into one.

Usage:   python3 combine_stats.py seg1.dat seg2.dat [...] -o combined.dat
         python3 combine_stats.py --selftest

Each Veta stats file covers ONE job segment (the accumulators live in device
memory, not in checkpoints). Segment files never collide: BOTH drivers stamp
the deck's stats_file with the segment's final step/time --
e.g. stats_Re550_sp_step00317102_t1011.869.dat. Glob them all into this
tool. This tool pools N segment files into a single file in
the SAME format, weighting every quantity by the segment's sample count
(header 'nsamp = ...') and renormalising the + columns by the POOLED window
u_tau (recomputed from the pooled <u> wall shear, the same convention as the
writer's window-mean u_tau).

Pooling rules (exact where the file stores enough, documented where not):
  <u>       exact:  U = sum n_i <u>_i / N
  uu        exact:  <u'^2> = sum n_i (uu_i + <u>_i^2)/N - U^2   (law of total
            variance; per-segment means ARE stored, col 8)
  vv, ww    sample-weighted mean of segment variances. The cross-segment
            mean-scatter term needs <v>_i/<w>_i, which the file does not
            store; those means are statistically zero and the dropped term is
            O(<v>_i^2) -- far below sampling noise.
  uv        sample-weighted mean (dropped term O(<u>_i scatter * <v>_i)).
  <p>       sample-weighted mean, but each segment carries its own k=0 GAUGE
            constant -- the combined <p> is provenance only (noted in header).
  <p'^2>    sample-weighted mean of per-segment variances (each about its own
            gauge -- the gauge-free convention).
  d<p>/dy, -d<v'v'>/dy   sample-weighted means (gauge-free / linear).

The combined u_tau / Re_tau / y+ / stress-linearity residual / uu-asymmetry
are recomputed from the pooled profiles (Chebyshev-exact wall derivative when
the y column is GLC, 2nd-order nonuniform FD otherwise -- analyze_stats.py's
convention). Segment files must share the y grid, Re, and column count.
"""
import argparse
import re as _re
import sys

import numpy as np


def parse_stats(path):
    """Returns (header dict, y, data columns array (npts, ncols))."""
    hdr = {"file": path, "ybasis": None}
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                m = _re.search(r"Re =\s*([0-9.Ee+-]+)\s+Re_tau =\s*([0-9.Ee+-]+)"
                               r"\s+u_tau =\s*([0-9.Ee+-]+)\s+nsamp =\s*(\d+)", line)
                if m:
                    hdr["Re"] = float(m.group(1))
                    hdr["Re_tau"] = float(m.group(2))
                    hdr["u_tau"] = float(m.group(3))
                    hdr["nsamp"] = int(m.group(4))
                if "ybasis =" in line:
                    hdr["ybasis"] = line.rstrip("\n")
            else:
                s = line.split()
                if s:
                    rows.append([float(x) for x in s])
    if "nsamp" not in hdr:
        sys.exit(f"error: {path}: no 'nsamp' in header (pre-stamp era file?)")
    dat = np.array(rows)
    if dat.shape[1] not in (8, 13):
        sys.exit(f"error: {path}: {dat.shape[1]} columns (expected 8 or 13)")
    return hdr, dat[:, 0], dat


def is_glc(y):
    n = len(y) - 1
    return np.max(np.abs(y - np.cos(np.pi * np.arange(n + 1) / n))) < 1e-9


def wall_dudy(y, u):
    """|dU/dy| at each wall: Chebyshev-exact on GLC nodes, else nonuniform FD."""
    n = len(y) - 1
    if is_glc(y):
        # forward cosine transform (GLC), coefficient derivative, eval at +-1
        jj = np.arange(n + 1)
        w = np.ones(n + 1); w[0] = w[-1] = 0.5
        a = np.array([2.0 / n * np.sum(w * u * np.cos(np.pi * k * jj / n))
                      for k in range(n + 1)])
        a[0] *= 0.5; a[-1] *= 0.5
        k = np.arange(n + 1)
        dhi = np.sum(k**2 * a)                       # dU/dy(+1)
        dlo = np.sum((-1.0) ** (k + 1) * k**2 * a)   # dU/dy(-1)
        return abs(dlo), abs(dhi)
    g = np.gradient(u, y)
    return abs(g[-1]), abs(g[0])


def combine(files, out):
    hs, ys, ds = [], [], []
    for p in files:
        h, y, d = parse_stats(p)
        if h["nsamp"] == 0 or h["u_tau"] <= 0.0:
            print(f"[skip] {p}: nsamp={h['nsamp']}, u_tau={h.get('u_tau')}"
                  " -- no usable samples")
            continue
        hs.append(h); ys.append(y); ds.append(d)
    if len(hs) < 1:
        sys.exit("error: no usable input files")
    ncol = ds[0].shape[1]
    for h, y, d in zip(hs, ys, ds):
        if d.shape[1] != ncol:
            sys.exit(f"error: {h['file']}: column count differs")
        if np.max(np.abs(y - ys[0])) > 1e-10:
            sys.exit(f"error: {h['file']}: y grid differs from {hs[0]['file']}")
        if abs(h["Re"] - hs[0]["Re"]) > 1e-6 * hs[0]["Re"]:
            sys.exit(f"error: {h['file']}: Re differs")
    y = ys[0]
    Re = hs[0]["Re"]
    have_p = ncol == 13
    ns = np.array([h["nsamp"] for h in hs], dtype=float)
    N = ns.sum()
    wgt = ns / N

    # denormalise each segment with ITS window u_tau^2
    ut2s = np.array([h["u_tau"] ** 2 for h in hs])
    U = sum(w * d[:, 7] for w, d in zip(wgt, ds))                       # <u>
    uu = sum(w * (d[:, 3] * t + d[:, 7] ** 2)
             for w, t, d in zip(wgt, ut2s, ds)) - U**2                  # exact
    vv = sum(w * d[:, 4] * t for w, t, d in zip(wgt, ut2s, ds))
    ww = sum(w * d[:, 5] * t for w, t, d in zip(wgt, ut2s, ds))
    uv = sum(w * d[:, 6] * t for w, t, d in zip(wgt, ut2s, ds))
    if have_p:
        mp   = sum(w * d[:, 8] for w, d in zip(wgt, ds))                # gauge!
        ppv  = sum(w * d[:, 10] for w, d in zip(wgt, ds))
        dpdy = sum(w * d[:, 11] for w, d in zip(wgt, ds))
        mdvv = sum(w * d[:, 12] for w, d in zip(wgt, ds))

    # pooled-window u_tau / Re_tau from the pooled <u> wall shear
    dlo, dhi = wall_dudy(y, U)
    ut2 = 0.5 * (dlo + dhi) / Re
    utau = np.sqrt(max(ut2, 0.0))
    ret = utau * Re
    # convergence metrics, writer's formulas (FD gradient suffices for the
    # rms stress fit -- only the WALL derivative above needed cheb precision)
    dUdy = np.gradient(U, y)
    T = dUdy / Re - uv
    resid = np.sqrt(np.mean((T + ut2 * y) ** 2)) / max(ut2, 1e-300)
    pk_lo = np.max(uu[y < 0]) if np.any(y < 0) else 0.0
    pk_hi = np.max(uu[y > 0]) if np.any(y > 0) else 0.0
    asym = 200.0 * abs(pk_lo - pk_hi) / max(pk_lo + pk_hi, 1e-300)

    s2 = ut2 if utau > 0 else 1.0
    with open(out, "w") as f:
        f.write("# channel DNS turbulence statistics (x-z-t Reynolds average)"
                " [combined]\n")
        f.write(f"#  Re = {Re:12.5f}   Re_tau = {ret:10.4f}   u_tau = "
                f"{utau:12.5e}   nsamp = {int(N)}\n")
        if hs[0]["ybasis"]:
            f.write(hs[0]["ybasis"] + "\n")
        f.write("#  normalisation: POOLED-WINDOW u_tau (wall shear of the"
                " pooled <u>) [combine_stats]\n")
        f.write(f"#  convergence: stress-linearity residual = {100*resid:8.3f}"
                f" %   uu-peak wall asymmetry = {asym:8.3f} %\n")
        f.write("#  combined from segments (sample-weighted):\n")
        for h in hs:
            f.write(f"#    {h['file']}   nsamp = {h['nsamp']}   u_tau = "
                    f"{h['u_tau']:.5e}\n")
        if have_p:
            f.write("#  pressure: prms+ = sqrt(<p'^2>)/u_tau^2 ; <p> pools"
                    " per-segment k=0 gauges -- provenance only\n")
            f.write("#  y            y+           U+           uu+          "
                    "vv+          ww+          uv+          <u>          <p>"
                    "          prms+        <p'^2>       d<p>/dy      "
                    "-d<v'v'>/dy\n")
        else:
            f.write("#  y            y+           U+           uu+          "
                    "vv+          ww+          uv+          <u>\n")
        for j in range(len(y)):
            yp = (1.0 - abs(y[j])) * ret
            row = [y[j], yp, U[j] / utau if utau > 0 else 0.0,
                   uu[j] / s2, vv[j] / s2, ww[j] / s2, uv[j] / s2, U[j]]
            if have_p:
                prmsp = np.sqrt(max(ppv[j], 0.0)) / s2 if utau > 0 else 0.0
                row += [mp[j], prmsp, ppv[j], dpdy[j], mdvv[j]]
            f.write(" ".join(f"{x:12.5e}" for x in row) + "\n")
    print(f"[ok] {out}: {len(hs)} segments, nsamp = {int(N)}, "
          f"Re_tau = {ret:.2f} (segments: "
          + ", ".join(f"{h['Re_tau']:.1f}" for h in hs) + ")")
    return {"y": y, "U": U, "uu": uu, "vv": vv, "ww": ww, "uv": uv,
            "utau": utau, "Re_tau": ret, "nsamp": int(N)}


def selftest():
    """Synthetic two-window pooling vs direct full-sample statistics."""
    import tempfile, os
    rng = np.random.default_rng(7)
    n = 64
    y = np.cos(np.pi * np.arange(n + 1) / n)
    Re = 1.0e4
    # synthetic per-sample "plane means": u ~ profile + noise, v,w ~ noise
    prof = 20.0 * (1.0 - y**2)
    NS = (400, 700)
    all_u, all_v, all_w = [], [], []
    paths = []
    for iseg, nsamp in enumerate(NS):
        us = prof[None, :] + rng.standard_normal((nsamp, n + 1))
        vs = 0.1 * rng.standard_normal((nsamp, n + 1))
        ws = 0.2 * rng.standard_normal((nsamp, n + 1))
        all_u.append(us); all_v.append(vs); all_w.append(ws)
        mu = us.mean(0); uu = us.var(0)
        vv = vs.var(0); ww = ws.var(0)
        uv = (us * vs).mean(0) - mu * vs.mean(0)
        dlo, dhi = wall_dudy(y, mu)
        ut2 = 0.5 * (dlo + dhi) / Re; utau = np.sqrt(ut2); ret = utau * Re
        fd, p = tempfile.mkstemp(suffix=f"_seg{iseg}.dat"); os.close(fd)
        with open(p, "w") as f:
            f.write("# channel DNS turbulence statistics (selftest)\n")
            f.write(f"#  Re = {Re:12.5f}   Re_tau = {ret:10.4f}   u_tau = "
                    f"{utau:12.5e}   nsamp = {nsamp}\n")
            f.write("#  y            y+           U+ ...\n")
            for j in range(n + 1):
                yp = (1.0 - abs(y[j])) * ret
                f.write(" ".join(f"{x:12.5e}" for x in
                        [y[j], yp, mu[j] / utau, uu[j] / ut2, vv[j] / ut2,
                         ww[j] / ut2, uv[j] / ut2, mu[j]]) + "\n")
        paths.append(p)
    fd, out = tempfile.mkstemp(suffix="_comb.dat"); os.close(fd)
    res = combine(paths, out)
    U, uu_c, vv_c = res["U"], res["uu"], res["vv"]
    # (a) EXACT pooling identity vs an independent pooling of the PARSED
    #     segment values (catches indexing/normalisation bugs; the file's
    #     es12.5 quantisation cancels because both sides read the same files)
    segs = [parse_stats(p) for p in paths]
    nsv = np.array([h["nsamp"] for h, _, _ in segs], dtype=float)
    wv = nsv / nsv.sum()
    Uref = sum(w * d[:, 7] for w, (_, _, d) in zip(wv, segs))
    uuref = sum(w * (d[:, 3] * h["u_tau"] ** 2 + d[:, 7] ** 2)
                for w, (h, _, d) in zip(wv, segs)) - Uref**2
    eU_id = np.max(np.abs(U - Uref))
    eUU_id = np.max(np.abs(uu_c - uuref))
    # (b) sanity vs the RAW samples: bounded by the es12.5 file round-trip
    #     (~1e-5 relative) and, for vv, the documented O(mean^2) approximation
    ufull = np.vstack(all_u); vfull = np.vstack(all_v)
    eU = np.max(np.abs(U - ufull.mean(0)))
    eUU = np.max(np.abs(uu_c - ufull.var(0)))
    eVV = np.max(np.abs(vv_c - vfull.var(0)))
    print(f"selftest: pooling identity |U|={eU_id:.2e} |uu|={eUU_id:.2e}"
          " (want ~1e-14: exact)")
    print(f"selftest: vs raw samples |U|={eU:.2e} |uu|={eUU:.2e} |vv|={eVV:.2e}"
          " (bounded by es12.5 file quantisation / documented vv approx)")
    ok = (eU_id < 1e-11 and eUU_id < 1e-11
          and eU < 5e-4 and eUU < 1e-3 and eVV < 1e-3)
    for p in paths + [out]:
        os.remove(p)
    print("selftest:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("files", nargs="*", help="segment stats files")
    ap.add_argument("-o", "--out", default="stats_combined.dat")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    if len(args.files) < 2:
        sys.exit("need at least two stats files (or --selftest)")
    combine(args.files, args.out)
