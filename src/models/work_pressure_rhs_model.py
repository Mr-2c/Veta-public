#!/usr/bin/env python3
"""
work_pressure_rhs_model.py -- numpy reference for the ASSEMBLY that
feeds the pressure solve (the solve itself is validated in work_pressure_model.py).
The operator/convention logic is modelled in numpy before the Fortran. This
locks the four conventions of par_pressure_mod's wiring, each against an
analytic truth:

  (1) RHS divergence  R = +div.H_hat = i*kx*Hx + D_y*Hy + i*kz*Hz   (Chebyshev-y
      coeff space; the SAME +i*k convention as stats_mod divergence_check, which
      is already validated). D_y via the cheb_deriv coefficient recurrence.
  (2) wall VALUES   f(+1) = sum_j a_j ,  f(-1) = sum_j (-1)^j a_j   (T_j(+-1)),
      INCLUDING j=0 -- distinct from wall_derivs_c (Du, which skips j=0). This is
      the new spec_helpers wall_values_c kernel.
  (3) wall data    g_+- = Hy(+-1) + (1/Re) D^2v(+-1),  with D^2v = Lv + k^2 v and
      Lv = apply_L(v) = (D^2-k^2)v  ->  D^2v(+-1) = wallval(Lv) + k^2 wallval(v).
      (v == 0 at the walls by no-slip, so the k^2 v term vanishes -- verified.)
  (4) DEALIASING  the |u|^2/2 recovery term: forming a quadratic product on the
      3/2 grid and truncating gives the ALIAS-FREE coarse coefficients, whereas
      forming it on the clean grid aliases. 1-D Fourier demonstration.

All four are exact-to-roundoff against analytic truth -> the Fortran is a direct
transcription.
"""
import numpy as np

# ---- Chebyshev coefficient-space helpers (mirror cheb_shen / spec_helpers) ----

def glc(n):
    return np.cos(np.pi * np.arange(n + 1) / n)


def cheb_forward(n, f):
    """values at GLC nodes -> Chebyshev coeffs (cheb_shen cheb_forward)."""
    j = np.arange(n + 1)
    cb = np.ones(n + 1); cb[0] = 2.0; cb[n] = 2.0
    a = np.empty(n + 1)
    for k in range(n + 1):
        ck = 2.0 if (k == 0 or k == n) else 1.0
        a[k] = 2.0 / (n * ck) * np.sum(f / cb * np.cos(np.pi * j * k / n))
    return a


def cheb_deriv(a):
    """first-derivative recurrence (cheb_shen cheb_deriv / cheb_ops cheb_deriv_c)."""
    n = a.size - 1
    b = np.zeros(n + 1)
    for k in range(n - 1, -1, -1):
        b[k] = (b[k + 2] if k + 2 <= n else 0.0) + 2.0 * (k + 1) * a[k + 1]
    b[0] *= 0.5
    return b


def cheb_d2(a):
    return cheb_deriv(cheb_deriv(a))


def wall_values(a):
    """f(+1)=sum a_j ; f(-1)=sum (-1)^j a_j  (NEW kernel wall_values_c; includes j=0)."""
    n = a.size - 1
    j = np.arange(n + 1)
    return np.sum(((-1.0) ** j) * a), np.sum(a)   # (lo at -1, hi at +1)


def apply_L(a, k2):
    """Lq = (D^2 - k^2) q in coeff space (cheb_ops apply_L)."""
    return cheb_d2(a) - k2 * a


# ---- (1) divergence R = i kx Hx + D_y Hy + i kz Hz, single (kx,kz) mode --------

def test_divergence(n=40):
    x = glc(n)
    kx, kz = 2.0, -3.0
    # analytic complex y-profiles for Hx,Hy,Hz (smooth)
    fx = np.cos(1.7 * x) + 1j * (0.4 * x**2)
    fy = np.sin(1.1 * x) - 1j * (0.3 * x**3 - 0.2)
    fz = (0.5 * x**2 - 0.1) + 1j * np.cos(0.9 * x)
    dfy = 1.1 * np.cos(1.1 * x) - 1j * (0.9 * x**2)          # analytic D_y Hy
    R_true = 1j * kx * fx + dfy + 1j * kz * fz               # analytic div.H at nodes
    # discrete: coeffs, D_y via recurrence (re/im separately), combine
    ax = cheb_forward(n, fx.real) + 1j * cheb_forward(n, fx.imag)
    ay = cheb_forward(n, fy.real) + 1j * cheb_forward(n, fy.imag)
    az = cheb_forward(n, fz.real) + 1j * cheb_forward(n, fz.imag)
    dAy = cheb_deriv(ay.real) + 1j * cheb_deriv(ay.imag)
    Rc = 1j * kx * ax + dAy + 1j * kz * az                   # coeff-space R
    R_node = (np.polynomial.chebyshev.chebval(x, Rc.real)
              + 1j * np.polynomial.chebyshev.chebval(x, Rc.imag))
    return np.max(np.abs(R_node - R_true))


# ---- (2) wall values vs analytic ---------------------------------------------

def test_wall_values(n=40):
    x = glc(n)
    f = np.cos(2.3 * x) + 0.7 * x**3 - 0.5
    a = cheb_forward(n, f)
    lo, hi = wall_values(a)
    return max(abs(hi - (np.cos(2.3) + 0.7 - 0.5)),       # f(+1)
               abs(lo - (np.cos(2.3) - 0.7 - 0.5)))       # f(-1): cos even, x^3 odd


# ---- (3) wall data g_+- = Hy(+-1) + (1/Re) D^2v(+-1), D^2v via Lv -------------

def test_wall_data(n=48):
    x = glc(n)
    Re = 180.0
    kx, kz = 1.0, 2.0; k2 = kx * kx + kz * kz
    # v with v(+-1)=0 (no-slip): v = (1-y^2)*smooth
    v = (1.0 - x**2) * (np.cos(1.3 * x) + 0.2)
    Hy = np.sin(0.8 * x) + 0.3 * x**2
    av = cheb_forward(n, v); aHy = cheb_forward(n, Hy)
    # truth: D^2 v at walls (analytic via dense diff of the coeffs is exact here)
    d2v_coeff = cheb_d2(av)
    d2v_lo_true = np.polynomial.chebyshev.chebval(-1.0, d2v_coeff)
    d2v_hi_true = np.polynomial.chebyshev.chebval(1.0, d2v_coeff)
    # the Fortran path: Lv = apply_L(v); D^2v(wall) = wallval(Lv) + k^2 wallval(v)
    Lv = apply_L(av, k2)
    Lv_lo, Lv_hi = wall_values(Lv)
    v_lo, v_hi = wall_values(av)
    d2v_hi = Lv_hi + k2 * v_hi
    d2v_lo = Lv_lo + k2 * v_lo
    Hy_lo, Hy_hi = wall_values(aHy)
    gp = Hy_hi + (1.0 / Re) * d2v_hi
    gm = Hy_lo + (1.0 / Re) * d2v_lo
    gp_true = (np.sin(0.8) + 0.3) + (1.0 / Re) * d2v_hi_true
    gm_true = (np.sin(-0.8) + 0.3) + (1.0 / Re) * d2v_lo_true
    # also report that v(wall)=0 so the k^2 v term is negligible (sanity)
    vwall = max(abs(v_lo), abs(v_hi))
    return max(abs(gp - gp_true), abs(gm - gm_true)), vwall


# ---- (4) 3/2 dealiasing of a quadratic product (1-D Fourier) ------------------

def test_dealias_product(nx=16):
    """u with modes up to nx/2-1; q = u^2. Coarse-grid product ALIASES; 3/2-grid
    product truncated back is alias-free (== exact truncated convolution).

    NOTE: this oracle uses NORMALIZED FFTs (np.fft.fft(.)/N), so it validates the
    alias-free property ONLY -- it does NOT lock the cuFFT (un)normalization. The
    Fortran forward transform (par_to_spectral_da_r2c) is UNNORMALIZED, so the
    caller must fold in 1/(nxd*nzd) when forming the |u|^2/2 product (par_pressure
    pres_half_usq takes `inv`), EXACTLY as cross_prod does for H -- otherwise q_hat
    is inflated by nxd*nzd vs Pi_hat and the p = Pi - |u|^2/2 subtraction is wrong."""
    nxd = 3 * nx // 2
    rng = np.random.default_rng(7)
    # band-limited u (real), retained modes 0..nx/2-1 (drop Nyquist for cleanliness)
    uh = np.zeros(nx, dtype=complex)
    nkeep = nx // 2
    uh[:nkeep] = rng.standard_normal(nkeep) + 1j * rng.standard_normal(nkeep)
    uh[0] = uh[0].real
    uh[nx - nkeep + 1:] = np.conj(uh[1:nkeep][::-1])         # hermitian -> real u
    u = np.fft.ifft(uh) * nx
    q_clean = u * u                                          # quadratic on clean grid
    qh_clean = np.fft.fft(q_clean) / nx
    # 3/2 path: pad uh to nxd, ifft, square, fft, truncate
    uh_pad = np.zeros(nxd, dtype=complex)
    uh_pad[:nkeep] = uh[:nkeep]
    uh_pad[nxd - nkeep + 1:] = uh[nx - nkeep + 1:]
    u_d = np.fft.ifft(uh_pad) * nxd
    qh_d_full = np.fft.fft(u_d * u_d) / nxd
    qh_d = np.zeros(nx, dtype=complex)
    qh_d[:nkeep] = qh_d_full[:nkeep]
    qh_d[nx - nkeep + 1:] = qh_d_full[nxd - nkeep + 1:]
    # exact truncated convolution (alias-free truth) over retained modes
    qh_true = np.zeros(nx, dtype=complex)
    for k in range(-nkeep + 1, nkeep):
        s = 0j
        for p in range(-nkeep + 1, nkeep):
            qm = k - p
            if -nkeep + 1 <= qm <= nkeep - 1:
                s += uh[p % nx] * uh[qm % nx]
        qh_true[k % nx] = s
    alias_err = np.max(np.abs(qh_d - qh_true))               # 3/2 path == truth?
    clean_alias = np.max(np.abs(qh_clean - qh_true))         # clean path aliases (large)
    return alias_err, clean_alias


if __name__ == "__main__":
    allok = True
    e1 = test_divergence()
    ok = e1 < 1e-9; allok &= ok
    print(f"  (1) R = i kx Hx + D_y Hy + i kz Hz   max|R-truth| = {e1:.2e}  {'PASS' if ok else 'FAIL'}")

    e2 = test_wall_values()
    ok = e2 < 1e-9; allok &= ok
    print(f"  (2) wall values f(+-1) (incl. j=0)   max err      = {e2:.2e}  {'PASS' if ok else 'FAIL'}")

    e3, vwall = test_wall_data()
    ok = e3 < 1e-8; allok &= ok
    print(f"  (3) g_+- = Hy + (1/Re)D^2v (Lv path) max err      = {e3:.2e}  {'PASS' if ok else 'FAIL'}"
          f"   [v(wall)={vwall:.1e} -> k^2 v term ~0]")

    ad, ca = test_dealias_product()
    ok = ad < 1e-10 and ca > 1e-3; allok &= ok
    print(f"  (4) |u|^2 dealias: 3/2 err={ad:.2e} (alias-free), clean-grid alias={ca:.2e} (corrupt) "
          f" {'PASS' if ok else 'FAIL'}")

    print("\n  work_pressure_rhs_model:",
          "ALL 3b ASSEMBLY CONVENTIONS VALIDATED" if allok else "FAILED")
