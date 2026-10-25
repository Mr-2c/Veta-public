#!/usr/bin/env python3
"""TRUE-BASIS model of the v-solve chain and of the candidate runtime
residual probes.

Mirrors the REAL algorithm, structure for structure:
  * project_shen      ftil_k = (pi/2)(c_k f_k - f_{k+2}),  c_0=2 else 1
  * Shen-Dirichlet Galerkin system (S + mu B) uhat = ftil, per parity:
        B(k,k) = 3pi/2 (k=0) else pi,   B(k,k+-2) = -pi/2
        S(k,k) = 2pi(k+1)(k+2),         S(k,j) = 4pi(k+1)  for j>k same parity
    (helm_band_mod's documented assembly; its O(n) row-differencing solve is an
    exact restructuring of this system -- dense float64 solve is faithful.)
  * shen_to_cheb      a_j = u_j - u_{j-2}
  * vif chain (vif_solve_gpu_ws):  H1: mu=k2, rhs=P(f)/a;  H2: mu=k2+1/a,
    rhs=-P(g);  closure v = vp + s1 vm + s2 vp_hom with s = -Cinv[Dv(-1);Dv(+1)]
  * vif setup (vif_gpu_setup_dev_ws): lifts l+- = (T0 +- T1)/2,
    g+-: H1 with rhs=-k2*l+- then +l+-;  v+-: H2 with rhs=-g+-;  Cinv from
    wall derivs  Du(+1)=sum j^2 a_j,  Du(-1)=sum (-1)^(j+1) j^2 a_j.

Questions this model answers (all float64, like the code's dp):
  W1  wiring: clamped-polynomial data -> chain reproduces v_true; per-solve
      Galerkin residuals ~ eps.  MUST pass before anything else is believed.
  Q1  the SHIPPED strong probe  |(L-aL^2)v - f|_max / |f|_max : reproduce its
      flow-dependence (young vs developed RHS wall content) at deck params.
  Q2  the PROPOSED cheb probes r1/r2 = per-solve Shen-Galerkin residuals
      (computed the way the runtime will: strong apply + project recompute):
      floor level, deck- and flow-independence.
  Q3  closure residual  max|Dv(walls)| of the closed v over max|Dv| of vp.
  Q4  FD: full strong residual vs the wall-window-excluded interior residual
      (window = 2*kl+1 rows per wall; the probe applies L twice, so wall-row
      defects spread <= 2*kl rows), floors for both.

RESULTS (float64):
  W1  |v-vtrue| <= 3.3e-16, r1 <= 2.6e-16, r2 <= 8.5e-16 at all four
      (n, k2) corners incl. production stiffness -> wiring PASSES.
  Q2  r1/r2 = 3.6e-16..8.2e-16, Q3 closure = 7.6e-16..2.3e-15 -- FLAT across
      young/developed RHS and across all three cheb decks, while the strong
      probe on the SAME solves reads O(10) (synthetic RHS overdrives the
      strong number vs the measured 0.4-1.4; direction/contrast is the point).
      -> the Galerkin/closure probes are flow- and deck-independent
      backward-error meters.  Runtime floors on real
      slab sizes sit a little higher (max over ~1e7 modes): expect
      ~1e-14..1e-13; WARN set at 1e-11 (measured*10 -- transferable because
      the quantity is flow-independent).
  Q4  FD interior (2*kl+1 rows cut per wall) = 2.8e-12..9.2e-12, STABLE
      across a 5x wall-spacing change (dy_wall/delta 5.9 -> 1.0) AND a 100x
      RHS wall-content change -- the gated quantity is insensitive to
      grid/RHS details, as a solver-roundoff floor must be.  Runtime WARN
      1e-8 (calibration).  The FULL number is NOT predictive here (model
      2.4e2..1e3 vs measured ~1e-2): the wall rows were REPLACED by BCs in
      the solve, so the strong recompute there reads the lambda-amplified
      one-sided-stencil truncation of the closure's sqrt(a3) layer,
      a·L^2(sigma·vm) ~ A_layer*(1/a)*(truncation fraction) -- a floorless
      quantity that depends delicately on how the (sub-grid on the real
      decks) layer sits on the grid.  Surrogate and real grid/flow sit in
      different truncation regimes; the same is true of the cheb strong
      number (model O(10) vs measured 0.5).  Strong/full = character
      illustration + resolution meter (hence UNGATED in the shipped probe);
      floors (r1/r2/closure/interior) = the predictive, gated quantities.
      Support note: the exact defect algebra gives contamination <= kl+1
      rows per wall (r = -a*d_w - a*L*d_v, defects on the 2 identity rows,
      one banded spread); the shipped 2*kl+1 window carries kl rows of
      margin on top.
"""
import numpy as np

rng = np.random.default_rng(20260807)
PI = np.pi

# ---------------------------------------------------------------- cheb pieces
def cheb_deriv_coeffs(a):
    """Chebyshev coefficient derivative by the standard backward recurrence
    (what cheb_ops' apply path does)."""
    n = len(a) - 1
    b = np.zeros_like(a)
    if n >= 1:
        b[n - 1] = 2.0 * n * a[n]
        for k in range(n - 2, -1, -1):
            b[k] = b[k + 2] + 2.0 * (k + 1) * a[k + 1] if k + 2 <= n - 1 \
                else 2.0 * (k + 1) * a[k + 1]
        b[0] *= 0.5
    return b


def apply_L(a, k2):
    """L a = D2 a - k2 a in coefficient space (apply_L twin)."""
    return cheb_deriv_coeffs(cheb_deriv_coeffs(a)) - k2 * a


def project_shen(f):
    n = len(f) - 1
    ck = np.ones(n - 1); ck[0] = 2.0
    return 0.5 * PI * (ck * f[: n - 1] - f[2: n + 1])


def shen_to_cheb(u, n):
    a = np.zeros(n + 1, dtype=u.dtype)
    a[: n - 1] += u
    a[2: n + 1] -= u
    return a


def helm_matrix(n, mu):
    """Dense (S + mu B), size (n-1), the system helm_band solves."""
    m = n - 1
    k = np.arange(m)
    M = np.zeros((m, m))
    M[k, k] = 2.0 * PI * (k + 1) * (k + 2) + mu * np.where(k == 0, 1.5 * PI, PI)
    for kk in range(m):
        for j in range(kk + 2, m, 2):
            M[kk, j] += 4.0 * PI * (kk + 1)
    idx = k[k + 2 < m]
    M[idx, idx + 2] += mu * (-0.5 * PI)
    M[idx + 2, idx] += mu * (-0.5 * PI)
    return M


def helm_solve(n, mu, ftil):
    return np.linalg.solve(helm_matrix(n, mu), ftil)


def wall_derivs(a):
    n = len(a) - 1
    j = np.arange(1, n + 1)
    dhi = np.sum(a[1:] * j**2)
    dlo = np.sum(a[1:] * (-1.0) ** (j + 1) * j**2)
    return dlo, dhi


class VifCheb:
    """Per-(k2, alpha) setup + solve, mirroring vif_gpu_setup/vif_solve_gpu_ws."""

    def __init__(self, n, k2, alpha):
        self.n, self.k2, self.a = n, k2, alpha
        inva = 1.0 / alpha
        lp = np.zeros(n + 1); lp[0] = 0.5; lp[1] = 0.5     # l+ = (T0+T1)/2
        lm = np.zeros(n + 1); lm[0] = 0.5; lm[1] = -0.5    # l- = (T0-T1)/2
        gs = {}
        for tag, l in (("p", lp), ("m", lm)):
            g = shen_to_cheb(helm_solve(n, k2, project_shen(-k2 * l)), n) + l
            gs[tag] = g
        self.vhp = shen_to_cheb(helm_solve(n, k2 + inva, project_shen(-gs["p"])), n)
        self.vhm = shen_to_cheb(helm_solve(n, k2 + inva, project_shen(-gs["m"])), n)
        dvm_lo, dvm_hi = wall_derivs(self.vhm)
        dvp_lo, dvp_hi = wall_derivs(self.vhp)
        det = dvm_lo * dvp_hi - dvp_lo * dvm_hi
        self.Cinv = np.array([[dvp_hi, -dvp_lo], [-dvm_hi, dvm_lo]]) / det

    def solve(self, f):
        """Returns (v, diagnostics dict). Mirrors vif_solve_gpu_ws + the
        PROPOSED probe recomputations (strong apply + project, float64)."""
        n, k2, a = self.n, self.k2, self.a
        inva = 1.0 / a
        ftil1 = project_shen(f) * inva
        u1 = helm_solve(n, k2, ftil1)
        gc = shen_to_cheb(u1, n)
        # r1 = P((-D2+k2) g) - ftil1 = P(-L g) - ftil1   (runtime recomputation)
        r1 = project_shen(-apply_L(gc, k2)) - ftil1
        ftil2 = -project_shen(gc)
        u2 = helm_solve(n, k2 + inva, ftil2)
        vpc = shen_to_cheb(u2, n)
        # r2 = P((-D2+k2+1/a) vpc) - ftil2 = P(-L vpc + inva*vpc) - ftil2
        r2 = project_shen(-apply_L(vpc, k2) + inva * vpc) - ftil2
        dlo, dhi = wall_derivs(vpc)
        s = -self.Cinv @ np.array([dlo, dhi])
        v = vpc + s[0] * self.vhm + s[1] * self.vhp
        dlo_v, dhi_v = wall_derivs(v)
        return v, {
            "rel1": np.max(np.abs(r1)) / np.max(np.abs(ftil1)),
            "rel2": np.max(np.abs(r2)) / np.max(np.abs(ftil2)),
            "dv_vp": max(abs(dlo), abs(dhi)),
            "dv_v": max(abs(dlo_v), abs(dhi_v)),
        }


# ------------------------------------------------------------------ FD pieces
def fornberg(x0, xs, m):
    """Fornberg weights for the m-th derivative at x0 from nodes xs."""
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


class VifFD:
    """FD twin: banded D2 (half-width kl), identity wall rows, same chain."""

    def __init__(self, y, kl, k2, alpha):
        self.y, self.kl, self.k2, self.a = y, kl, k2, alpha
        ny = len(y)
        D2 = np.zeros((ny, ny))
        for i in range(ny):
            j0 = min(max(i - kl, 0), ny - 2 * kl - 1)
            js = np.arange(j0, j0 + 2 * kl + 1)
            D2[i, js] = fornberg(y[i], y[js], 2)
        self.D2 = D2
        # one-sided first-derivative wall rows (fd_wall_weights analog)
        self.wlo = np.zeros(ny); self.whi = np.zeros(ny)
        self.wlo[ny - 2 * kl - 1:] = fornberg(y[-1], y[ny - 2 * kl - 1:], 1)
        self.whi[: 2 * kl + 1] = fornberg(y[0], y[: 2 * kl + 1], 1)
        # y sorted +1 -> -1 like GLC?  Use: y[0]=+1 wall (hi), y[-1]=-1 wall (lo)
        inva = 1.0 / alpha
        self.H1 = self._helm(k2)
        self.H2 = self._helm(k2 + inva)
        lp = 0.5 * (1.0 + y); lm = 0.5 * (1.0 - y)
        gp = self._solve_dirichlet(self.H1, -k2 * lp) + lp
        gm = self._solve_dirichlet(self.H1, -k2 * lm) + lm
        self.vhp = self._solve_dirichlet(self.H2, -gp)
        self.vhm = self._solve_dirichlet(self.H2, -gm)
        dvm = (self.wlo @ self.vhm, self.whi @ self.vhm)
        dvp = (self.wlo @ self.vhp, self.whi @ self.vhp)
        det = dvm[0] * dvp[1] - dvp[0] * dvm[1]
        self.Cinv = np.array([[dvp[1], -dvp[0]], [-dvm[1], dvm[0]]]) / det

    def _helm(self, mu):
        H = -self.D2 + mu * np.eye(len(self.y))
        H[0, :] = 0.0; H[0, 0] = 1.0
        H[-1, :] = 0.0; H[-1, -1] = 1.0
        return H

    def _solve_dirichlet(self, H, f):
        b = f.copy(); b[0] = 0.0; b[-1] = 0.0
        return np.linalg.solve(H, b)

    def solve(self, f):
        inva = 1.0 / self.a
        g = self._solve_dirichlet(self.H1, f * inva)
        vpc = self._solve_dirichlet(self.H2, -g)
        s = -self.Cinv @ np.array([self.wlo @ vpc, self.whi @ vpc])
        return vpc + s[0] * self.vhm + s[1] * self.vhp

    def strong_res(self, v, f):
        L = lambda q: self.D2 @ q - self.k2 * q
        return L(v) - self.a * L(L(v)) - f


# --------------------------------------------------------------- experiments
def cheb_rhs(n, decay, eta, fwall):
    j = np.arange(n + 1)
    mag = np.maximum(np.exp(-decay * j / n), eta)
    f = mag * rng.choice([-1.0, 1.0], n + 1)
    fp, fm = np.sum(f), np.sum(f * (-1.0) ** j)
    f[0] += fwall - (fp + fm) / 2
    f[1] += fwall - (fp - fm) / 2
    return f


def run_cheb_deck(label, n, a3, k2max, decay, eta, fwall):
    out = {"strong": 0.0, "r1": 0.0, "r2": 0.0, "dv": 0.0}
    fmax_g = 0.0
    for k2 in (1.0, k2max / 100.0, k2max):
        amp = max(1.0 / (1.0 + k2 / 100.0), 1e-7)
        f = amp * cheb_rhs(n, decay, eta, fwall)
        vc = VifCheb(n, k2, a3)
        v, d = vc.solve(f)
        rs = np.max(np.abs(apply_L(v, k2) - a3 * apply_L(apply_L(v, k2), k2) - f))
        out["strong"] = max(out["strong"], rs)
        fmax_g = max(fmax_g, np.max(np.abs(f)))
        out["r1"] = max(out["r1"], d["rel1"])
        out["r2"] = max(out["r2"], d["rel2"])
        out["dv"] = max(out["dv"], d["dv_v"] / max(d["dv_vp"], 1e-300))
    print(f"{label:24s} strong={out['strong']/fmax_g:9.2e}  r1={out['r1']:9.2e}"
          f"  r2={out['r2']:9.2e}  closureDv={out['dv']:9.2e}")


print("=" * 78)
print("W1 -- wiring: clamped poly (1-y^2)^2 = 3/8 - T2/2 + T4/8, exact f")
print("=" * 78)
for (n, k2, a3) in ((160, 1.0, 4.48e-8), (160, 1.84e5, 4.48e-8),
                    (480, 1.0, 3.83e-9), (480, 5.12e5, 3.83e-9)):
    vtrue = np.zeros(n + 1)
    vtrue[0], vtrue[2], vtrue[4] = 0.375, -0.5, 0.125
    f = apply_L(vtrue, k2) - a3 * apply_L(apply_L(vtrue, k2), k2)
    vc = VifCheb(n, k2, a3)
    v, d = vc.solve(f)
    print(f"  n={n:4d} k2={k2:9.3g}: |v-vtrue|={np.max(np.abs(v-vtrue)):.2e}"
          f"  r1={d['rel1']:.2e}  r2={d['rel2']:.2e}"
          f"  (want all ~1e-12 or below)")

print()
print("=" * 78)
print("Q1/Q2/Q3 -- cheb decks, turbulent-ish RHS (worst over 3 k2 columns)")
print("=" * 78)
print("-- young flow (smoke): steep tail, tiny wall content --")
run_cheb_deck("smoke n=480", 480, 3.83e-9, 5.12e5, 30.0, 1e-12, 1e-4)
print("-- developed flow: occupied tail, O(1) wall content --")
run_cheb_deck("Re550  n=160", 160, 4.48e-8, 1.84e5, 7.0, 1e-4, 1.0)
run_cheb_deck("Re1000 n=240", 240, 2.72e-8, 3.13e5, 7.0, 1e-4, 1.0)
print("   MEASURED strong: smoke 4-8e-5, Re550 0.42-0.65, Re1000 0.66-1.4")
print("   -> r1/r2/closure are the probe candidates: must be SMALL and FLAT")

print()
print("=" * 78)
print("Q4 -- FD deck: full strong residual vs interior (2*kl+1 wall rows cut)")
print("=" * 78)
kl = 4


def fd_case(label, ny, a3, k2max, beta, fwall):
    """One FD deck surrogate.  beta sets the tanh wall clustering: beta=2.2
    gives dy_wall ~ 1.2e-3 (~5x COARSER at the wall than the real natural
    grids); beta=3.2 gives dy_wall ~ 2.4e-4 ~ 1.1*sqrt(a3) -- the real decks'
    engineered dy_wall/delta.  fwall scales the RHS wall content.  The FULL
    residual is one-sided-stencil truncation on the wall layer -- it is
    hypersensitive to BOTH knobs (that is exactly why it is ungated); the
    INTERIOR number must stay at solver roundoff regardless."""
    s = np.linspace(0.0, 1.0, ny)
    y = -np.tanh(beta * (2 * s - 1)) / np.tanh(beta)   # y[0]=+1 ... y[-1]=-1
    dyw = abs(y[0] - y[1])
    full = interior = fmax_g = fmax_i = 0.0
    for k2 in (1.0, k2max / 100.0, k2max):
        amp = max(1.0 / (1.0 + k2 / 100.0), 1e-7)
        f = amp * ((1.0 - y * y) * np.cos(3.0 * y) + fwall
                   + 1e-6 * rng.standard_normal(ny))
        fd = VifFD(y, kl, k2, a3)
        v = fd.solve(f)
        r = fd.strong_res(v, f)
        w = slice(2 * kl + 1, ny - 2 * kl - 1)
        full = max(full, np.max(np.abs(r)))
        interior = max(interior, np.max(np.abs(r[w])))
        fmax_g = max(fmax_g, np.max(np.abs(f)))
        fmax_i = max(fmax_i, np.max(np.abs(f[w])))
    print(f"  {label}: dy_wall/delta={dyw/np.sqrt(a3):5.2f}  fwall={fwall:5.2f}"
          f"  ->  full={full/fmax_g:9.2e}   interior={interior/fmax_i:9.2e}")


print("-- coarse-wall surrogate grid + O(1) RHS wall content (overdriven) --")
fd_case("Re550_fd  n=178", 179, 4.48e-8, 1.84e5, beta=2.2, fwall=1.0)
fd_case("Re1000_fd n=260", 261, 2.72e-8, 3.13e5, beta=2.2, fwall=1.0)
print("-- wall-resolved grid (real decks: dy_wall ~ 1.1*delta) + small wall RHS --")
fd_case("Re550_fd  n=178", 179, 4.48e-8, 1.84e5, beta=3.2, fwall=0.01)
fd_case("Re1000_fd n=260", 261, 2.72e-8, 3.13e5, beta=3.2, fwall=0.01)
print("   MEASURED full ~1e-2 on the real natural grids/flow. The FULL number")
print("   is stencil truncation on the sqrt(a3) layer -> unbounded when the")
print("   surrogate under-resolves it; the INTERIOR floor is the probe and")
print("   must be (and is) insensitive to grid/RHS details.")
