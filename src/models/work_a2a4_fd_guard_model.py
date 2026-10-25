#!/usr/bin/env python3
"""numpy-first models for two FD-y guards.

ORDER (FD wall rows taper 8th -> ~4th order, and a >8x ratio
gate would pass a 7th->3rd regression): measure the ACTUAL convergence order
of the kl=4 banded D2 on uniform and strongly stretched (tanh, natural-grid
class) grids, split into interior rows vs near-wall (one-sided) rows, so the
Fortran gates can pin measured orders instead of a loose ratio.

PIVOTING (no-pivot banded LU on a non-diagonally-dominant operator, with
only a zero-pivot guard): quantify the no-pivot risk on the actual operator
family A = -D2 + mu*I with identity wall rows, across the campaign's
(k2, lambda) corners incl. the worst case mu -> 0, and establish the
residual floor a once-per-factorisation self-check should gate on.
"""
import numpy as np

rng = np.random.default_rng(20260814)


def fornberg(x0, xs, m):
    nn = len(xs)
    c = np.zeros((nn, m + 1))
    c1, c4 = 1.0, xs[0] - x0
    c[0, 0] = 1.0
    for i in range(1, nn):
        mn = min(i, m)
        c2, c5 = 1.0, c4
        c4 = xs[i] - x0
        for j in range(i):
            c3 = xs[i] - xs[j]
            c2 *= c3
            if j == i - 1:
                for kk in range(mn, 0, -1):
                    c[i, kk] = c1 * (kk * c[i - 1, kk - 1] - c5 * c[i - 1, kk]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for kk in range(mn, 0, -1):
                c[j, kk] = (c4 * c[j, kk] - kk * c[j, kk - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c[:, m]


def build_d2(y, kl):
    """BANDED stencils, matching fd_build_bands: row i uses ONLY nodes within
    |i-j| <= kl (wall rows get FEWER points -- the deliberate order taper that
    keeps the implicit solves banded). Shifted full-width windows near the wall
    (|i-j| up to 2kl -- NOT banded) would over-predict the wall-row order
    (~9-10 vs the code's measured ~4.8) and under-predict the natural-481
    factor-residual floor."""
    ny = len(y)
    D2 = np.zeros((ny, ny))
    for i in range(ny):
        j0 = max(0, i - kl)
        j1 = min(ny - 1, i + kl)
        js = np.arange(j0, j1 + 1)
        D2[i, js] = fornberg(y[i], y[js], 2)
    return D2


def grids(ny, kind):
    s = np.linspace(0.0, 1.0, ny)
    if kind == "uniform":
        return 2.0 * s - 1.0
    b = 2.9                       # dy_mid/dy_wall ~ 29: natural-grid class
    return np.tanh(b * (2 * s - 1)) / np.tanh(b)


def order_study(kind, kl=4):
    f = lambda y: np.sin(2.3 * y + 0.4) * np.exp(0.3 * y)
    f2 = lambda y: ((0.3**2 - 2.3**2) * np.sin(2.3 * y + 0.4)
                    + 2 * 0.3 * 2.3 * np.cos(2.3 * y + 0.4)) * np.exp(0.3 * y)
    errs = {}
    for ny in (33, 65, 129, 257):
        y = grids(ny, kind)
        D2 = build_d2(y, kl)
        e = np.abs(D2 @ f(y) - f2(y))
        wall = np.r_[e[1:kl + 1], e[-kl - 1:-1]]      # one-sided rows (excl BC rows)
        inter = e[kl + 1:-kl - 1]
        errs[ny] = (np.max(inter), np.max(wall))
    print(f"-- {kind} grid, kl={kl}: refinement orders (interior | wall rows)")
    ns = sorted(errs)
    for a, b in zip(ns, ns[1:]):
        pi = np.log2(errs[a][0] / errs[b][0])
        pw = np.log2(errs[a][1] / errs[b][1])
        print(f"   n {a:4d}->{b:4d}:  interior p={pi:5.2f}   wall p={pw:5.2f}"
              f"   (err_int {errs[b][0]:.2e}, err_wall {errs[b][1]:.2e})")


def mild_rung_study(kl=4):
    """EXACT twin of test_fdops refine_block (the gated rows): descending
    tanh(b=1.2) grids, ny = n+1 for n = 16/24/32, f = sin(2.3y+0.4)exp(0.3y),
    orders p = ln(e_a/e_b)/ln(n_b/n_a) split interior vs one-sided wall rows.
    With the banded build_d2 this reproduces the hardware run
    to the printed digit: interior 7.43/7.71, wall 4.86/4.75."""
    f = lambda y: np.sin(2.3 * y + 0.4) * np.exp(0.3 * y)
    f2 = lambda y: ((0.3**2 - 2.3**2) * np.sin(2.3 * y + 0.4)
                    + 2 * 0.3 * 2.3 * np.cos(2.3 * y + 0.4)) * np.exp(0.3 * y)
    b = 1.2
    nv = (16, 24, 32)
    errs = []
    for n in nv:
        s = np.arange(n + 1) / n
        y = -np.tanh(b * (2 * s - 1)) / np.tanh(b)     # descending, as production
        D2 = build_d2(y, kl)
        e = np.abs(D2 @ f(y) - f2(y))
        inter = e[kl + 1:-kl - 1]
        wall = np.r_[e[1:kl + 1], e[-kl - 1:-1]]
        errs.append((np.max(inter), np.max(wall)))
    print(f"-- A2 gated rows: tanh(b={b}) mild rungs n=16->24->32 (banded, kl={kl})")
    for (na, nb), (ea, eb) in zip(zip(nv, nv[1:]), zip(errs, errs[1:])):
        r = np.log(nb / na)
        pi = np.log(ea[0] / eb[0]) / r
        pw = np.log(ea[1] / eb[1]) / r
        print(f"   n {na:2d}->{nb:2d}:  interior p={pi:5.2f}   wall p={pw:5.2f}")


def lu_nopivot_banded(A):
    """No-pivot LU (dense emulation of the banded algorithm's arithmetic)."""
    n = A.shape[0]
    U = A.astype(float).copy()
    L = np.eye(n)
    for k in range(n - 1):
        if U[k, k] == 0.0:
            return None, None
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])
    return L, U


def a4_study():
    print("-- A4: no-pivot LU on A = -D2 + mu*I (identity wall rows),")
    print("   NATURAL-CLASS grid ny=481, dy ratio ~29 (b=2.37) -- the anchor the")
    print("   test gate uses. mu->0 = the non-diagonally-dominant corner; its")
    print("   floor is eps*|A|-scaled by 1/dy_wall^2 (HW 5.63e-12, model 1.25e-11).")
    ny, kl = 481, 4
    s2 = np.linspace(0.0, 1.0, ny)
    y = -np.tanh(2.37 * (2 * s2 - 1)) / np.tanh(2.37)
    D2 = build_d2(y, kl)
    for mu in (0.0, 1.0, 1.84e5, 2.6e8):
        A = -D2 + mu * np.eye(ny)
        A[0, :] = 0; A[0, 0] = 1
        A[-1, :] = 0; A[-1, -1] = 1
        L, U = lu_nopivot_banded(A)
        growth = np.max(np.abs(U)) / np.max(np.abs(A))
        x = rng.standard_normal(ny)
        b = A @ x
        # solve via the no-pivot factors
        z = np.linalg.solve_triangular if False else None
        yv = np.zeros(ny)
        for i in range(ny):
            yv[i] = b[i] - L[i, :i] @ yv[:i]
        xs = np.zeros(ny)
        for i in range(ny - 1, -1, -1):
            xs[i] = (yv[i] - U[i, i + 1:] @ xs[i + 1:]) / U[i, i]
        rel = np.max(np.abs(A @ xs - b)) / np.max(np.abs(b))
        print(f"   mu={mu:8.2e}:  element growth={growth:8.2e}   "
              f"solve rel residual={rel:8.2e}")


print("=" * 74)
mild_rung_study()
order_study("uniform")
order_study("tanh")
print("=" * 74)
a4_study()
print("=" * 74)
print("Gate guidance (from hardware runs): interior order >= 6.5,")
print("wall-row order >= 4.0 (banded taper measures 4.75-4.86 -- model ==")
print("hardware after the banded-stencil correction); factor residual gate")
print("1e-10 (HW floor 5.63e-12 at mu=0 on natural-481).")
