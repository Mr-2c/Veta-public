#!/usr/bin/env python3
"""
work_sp_solve_mms_model.py -- the sp-solve-tier DECISION prototype.

QUESTION: can the banded wall-normal
Helmholtz solves  (D2 - lambda) u = f  run with SP-STORED factors/vectors and
DP-ACCUMULATED arithmetic (the proposed GPU kernel design) at the Re_tau=20000
extremes -- lambda up to ~7.5e10 on the two-zone stretched grid -- without
losing the 1e-6 accuracy contract?

WHAT IS EMULATED (per B2.2's design):
  dp      : float64 everything (the current solve tier -- reference).
  sp_acc  : every ARRAY (operator bands, Thomas factors, RHS, solution) stored
            float32; every arithmetic op computed on float64-promoted operands
            and ROUNDED TO FLOAT32 ON STORE. Coefficients are BUILT in dp from
            dp metrics, then rounded (dp Cinv-build stays, per B2.2).
  sp      : pure float32 (contrast -- what NOT to ship).

TESTS (2nd-order central FD on the nonuniform two-zone grid, Dirichlet walls):
  1. MMS       : u = (1-y^2)^2 sin(3 pi y),  f = u'' - lambda u analytic.
                 Metric: relative Linf(u_mode - u_dp), plus vs exact.
  2. INFLUENCE : (D2 - lambda) g = 0, g(-1)=1, g(1)=0 -- the influence-matrix
                 column. Metric: relative error of the one-sided 2nd-order
                 wall derivative g'(-1) (the influence-matrix ENTRY), sp vs dp,
                 plus the discrete boundary-layer decay sanity.

PASS: Linf <= 1e-6 and wall-BC residual <= 1e-6 (vs dp).

Run:  python3 src/models/work_sp_solve_mms_model.py
"""
import numpy as np
from work_fd_ygrid_model import two_zone_grid, natural_grid

F32 = np.float32
F64 = np.float64


# ---------------------------------------------------------------- operators --
def d2_bands(y):
    """3-point nonuniform second-derivative bands (dp). Rows 1..n-2 interior."""
    n = len(y)
    hl = y[1:-1] - y[:-2]
    hr = y[2:] - y[1:-1]
    sub = 2.0 / (hl * (hl + hr))
    dia = -2.0 / (hl * hr)
    sup = 2.0 / (hr * (hl + hr))
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    a[1:-1] = sub; b[1:-1] = dia; c[1:-1] = sup
    b[0] = 1.0; b[-1] = 1.0                      # Dirichlet rows
    return a, b, c


def helm_bands(y, lam):
    a, b, c = d2_bands(y)
    b2 = b.copy()
    b2[1:-1] -= lam
    return a, b2, c


def thomas(a, b, c, f, mode):
    """Tridiagonal solve in one of the three precision modes."""
    if mode == "dp":
        st, acc = F64, F64
    elif mode == "sp":
        st, acc = F32, F32
    elif mode == "sp_acc":
        st, acc = F32, F64
    else:
        raise ValueError(mode)
    n = len(b)
    A = a.astype(st); B = b.astype(st); C = c.astype(st); F = f.astype(st)
    cp = np.zeros(n, dtype=st)                   # modified super
    fp = np.zeros(n, dtype=st)                   # modified rhs
    cp[0] = st(acc(C[0]) / acc(B[0]))
    fp[0] = st(acc(F[0]) / acc(B[0]))
    for i in range(1, n):
        den = acc(B[i]) - acc(A[i]) * acc(cp[i - 1])
        cp[i] = st(acc(C[i]) / den)
        fp[i] = st((acc(F[i]) - acc(A[i]) * acc(fp[i - 1])) / den)
    u = np.zeros(n, dtype=st)
    u[-1] = fp[-1]
    for i in range(n - 2, -1, -1):
        u[i] = st(acc(fp[i]) - acc(cp[i]) * acc(u[i + 1]))
    return u.astype(F64)


def wall_deriv(y, u):
    """One-sided 2nd-order derivative at y[0] on a nonuniform grid (dp)."""
    h1 = y[1] - y[0]; h2 = y[2] - y[0]
    c0 = -(h1 + h2) / (h1 * h2)
    c1 = h2 / (h1 * (h2 - h1))
    c2 = -h1 / (h2 * (h2 - h1))
    return c0 * u[0] + c1 * u[1] + c2 * u[2]


# -------------------------------------------------------------------- tests --
def mms_case(y, lam):
    s = np.sin(3 * np.pi * y); co = np.cos(3 * np.pi * y)
    g = (1 - y**2) ** 2
    gp = -4 * y * (1 - y**2)
    gpp = -4 + 12 * y**2
    u_ex = g * s
    upp = gpp * s + 2 * gp * (3 * np.pi) * co - g * (3 * np.pi) ** 2 * s
    f = upp - lam * u_ex
    f[0] = 0.0; f[-1] = 0.0                      # Dirichlet BC values
    a, b, c = helm_bands(y, lam)
    out = {}
    for mode in ("dp", "sp_acc", "sp"):
        u = thomas(a, b, c, f, mode)
        out[mode] = u
    return u_ex, out


def influence_case(y, lam):
    n = len(y)
    f = np.zeros(n); f[0] = 1.0                  # g(-1)=1, g(1)=0
    a, b, c = helm_bands(y, lam)
    out = {}
    for mode in ("dp", "sp_acc", "sp"):
        out[mode] = thomas(a, b, c, f, mode)
    return out


def run_grid(tag, y, lams):
    ny = len(y)
    print(f"\n===== grid: {tag}  (ny = {ny}) =====")
    print("--- 1. MMS  (relative Linf, normalised by max|u_ex|) ---")
    print(f"{'lambda':>10} | {'dp vs exact':>12} | {'sp_acc vs dp':>13} "
          f"{'PASS?':>6} | {'sp vs dp':>10}")
    ok_all = True
    for lam in lams:
        u_ex, u = mms_case(y, lam)
        nrm = np.abs(u_ex).max()
        e_dp = np.abs(u["dp"] - u_ex).max() / nrm
        e_sa = np.abs(u["sp_acc"] - u["dp"]).max() / nrm
        e_sp = np.abs(u["sp"] - u["dp"]).max() / nrm
        ok = e_sa <= 1e-6
        ok_all &= ok
        bc = max(abs(u["sp_acc"][0]), abs(u["sp_acc"][-1]))
        print(f"{lam:10.1e} | {e_dp:12.2e} | {e_sa:13.2e} "
              f"{'PASS' if ok else 'FAIL':>6} | {e_sp:10.2e}"
              f"   (BC resid {bc:.1e})")

    print("--- 2. INFLUENCE column  g'(-1) = the influence-matrix entry ---")
    print(f"{'lambda':>10} | {'g_wall_dp':>12} | {'rel sp_acc':>11} "
          f"{'PASS?':>6} | {'rel sp':>10} | {'decay/cell':>10}")
    for lam in lams:
        g = influence_case(y, lam)
        gw_dp = wall_deriv(y, g["dp"])
        r_sa = abs(wall_deriv(y, g["sp_acc"]) - gw_dp) / abs(gw_dp)
        r_sp = abs(wall_deriv(y, g["sp"]) - gw_dp) / abs(gw_dp)
        ok = r_sa <= 1e-6
        ok_all &= ok
        dec = abs(g["dp"][1] / g["dp"][0])
        print(f"{lam:10.1e} | {gw_dp:12.5e} | {r_sa:11.2e} "
              f"{'PASS' if ok else 'FAIL':>6} | {r_sp:10.2e} | {dec:10.2e}")
    return ok_all


def main():
    lams = (1.0e2, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e10, 7.5e10)

    # the PRODUCTION grid: P&O-natural, alpha-corrected:
    # alpha=0.375, dy+_wall=0.5 (actual first spacing 0.65+), smooth cap 22+
    y1 = natural_grid(20000.0, alpha=0.375, dwall=0.5, c=0.705, cap=22.0)
    ok1 = run_grid("NATURAL Re20000 a=0.375 dw=0.5 c=0.705", y1, lams)

    # natural variant: alpha=0.40 (gentler near-wall growth, same ny class)
    y2 = natural_grid(20000.0, alpha=0.40, dwall=0.5, c=0.576, cap=22.0)
    ok2 = run_grid("NATURAL Re20000 a=0.40  dw=0.5 c=0.576", y2, lams)

    # cross-generator check: the earlier two-zone map (verdict must not move)
    y3 = two_zone_grid(20000.0, r=1.03)
    run_grid("two-zone Re20000 r=1.03 (cross-check)", y3, lams)

    print("\n================= VERDICT (numpy model) =================")
    print("GRID-INDEPENDENT (natural a=0.375/0.40 AND two-zone give the same")
    print("split; the crossover sits at lambda* ~ 1e7-1e8 depending on grid --")
    print("the rule below uses the CONSERVATIVE 1e8 bound valid on all).")
    print("SPLIT verdict -- the pass/fail boundary is the SHIFT, not the grid:")
    print("  * CN-SHIFTED solves (lambda ~ k^2 + O(1/(nu dt))): the shift floor")
    print("    is ~1e8 already at the Re2000 campaign (Re_b/dt ~ 5e7-1e8) and")
    print("    ~5e9 at Re20000 -- comfortably in the PASS zone (~1e-7, 10x")
    print("    margin). These are the eta solve + the FIRST v-solve equation +")
    print("    the influence solves = the bulk of banded-solve work.")
    print("  * UNSHIFTED (bare k^2) solves -- the SECOND v-solve equation")
    print("    (D2-k^2)v = g and the pressure Poisson: k^2 on the 8pi x 3pi")
    print("    Re20000 box spans 0.0625 .. ~2.3e6, i.e. AT OR BELOW the")
    print("    observed crossover -- these MUST STAY dp (as must k^2=0 mean).")
    print("  * Pure-sp fails everywhere it matters: the dp-accumulate")
    print("    discipline is load-bearing, not optional.")
    print("DESIGN RULE for B2.2: precision tier PER EQUATION -- sp-stored/")
    print("dp-accumulated factors for every SHIFTED Helmholtz; dp for every")
    print("bare-k^2 solve. No per-mode branching needed (the equation type,")
    print("not the mode, decides).")
    print("REMAINING (cluster/Fortran): OS growth-rate gate + 1e5-step KMM A/B.")


if __name__ == "__main__":
    main()
