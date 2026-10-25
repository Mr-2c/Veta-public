#!/usr/bin/env python3
"""
work_pressure_model.py -- numpy reference model for the CLAUSON pressure solve.
The operator/closure logic is modelled in numpy before it is written in Fortran.

PHYSICS (docs/kmm_formulation.pdf, pressure-Poisson appendix). Pressure is a
DIAGNOSTIC in the v-eta formulation. Solve for the MODIFIED pressure Pi = p + |u|^2/2:
    (D^2 - k^2) Pi_hat(y) = R(y),   R = +div.H_hat = i kx H_x + D H_y + i kz H_z
    Neumann wall BC:  D Pi_hat(+-1) = g_+- = H_y + (1/Re) D^2 v_hat   at the walls
then recover p = Pi - |u|^2/2 (a pointwise physical-space subtraction; trivial -> not
modelled here). H = u x omega is the ROTATIONAL nonlinear term the code already forms
(NOT N=(u.grad)u) -> note the +div.H sign.

This script validates the SOLVE for Pi: the inhomogeneous-Neumann (D^2-k^2) BVP via the
INFLUENCE-MATRIX closure (Route A: a Dirichlet particular solve + two homogeneous
solutions + a 2x2 wall-Neumann match -- exactly what the Fortran will do reusing the
existing Dirichlet helm_band solver + wall_derivs), and the k=0 singular gauge. The
underlying 1-D solver here is Chebyshev collocation (Trefethen cheb), standing in for
helm_band; the influence-matrix MECHANICS are what we are checking.

Validation = manufactured solution (MMS): pick Pi_exact, set R and g_+- from it
analytically, solve, and require recovery (to spectral accuracy for k>0; up to the
gauge constant for k=0).
"""
import numpy as np


def cheb(N):
    """Trefethen's Chebyshev diff matrix + GLC nodes. x[0]=+1 ... x[N]=-1."""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def solve_dirichlet(L, N, rhs, bc_p1, bc_m1):
    """Solve L u = rhs with u(+1)=bc_p1 (node 0), u(-1)=bc_m1 (node N)."""
    A = L.copy(); b = rhs.copy()
    A[0, :] = 0.0; A[0, 0] = 1.0; b[0] = bc_p1
    A[N, :] = 0.0; A[N, N] = 1.0; b[N] = bc_m1
    return np.linalg.solve(A, b)


def neumann_poisson(N, k2, R, gp, gm):
    """(D^2 - k^2) Pi = R with D Pi(+1)=gp, D Pi(-1)=gm. Route-A influence matrix."""
    D, x = cheb(N)
    L = D @ D - k2 * np.eye(N + 1)
    if k2 > 0.0:
        Pi_p  = solve_dirichlet(L, N, R,              0.0, 0.0)   # particular (Dirichlet 0)
        phi_p = solve_dirichlet(L, N, np.zeros(N + 1), 1.0, 0.0)  # homog: u(+1)=1, u(-1)=0
        phi_m = solve_dirichlet(L, N, np.zeros(N + 1), 0.0, 1.0)  # homog: u(+1)=0, u(-1)=1
        dPi_p = D @ Pi_p; dphi_p = D @ phi_p; dphi_m = D @ phi_m
        # 2x2 wall-Neumann influence matrix: [c_p, c_m] s.t. D(Pi_p + c_p phi_p + c_m phi_m)=g
        M   = np.array([[dphi_p[0], dphi_m[0]],      # at +1
                        [dphi_p[N], dphi_m[N]]])     # at -1
        rhs = np.array([gp - dPi_p[0], gm - dPi_p[N]])
        c = np.linalg.solve(M, rhs)
        return Pi_p + c[0] * phi_p + c[1] * phi_m, x
    else:
        # k=0: D^2 Pi = R, pure Neumann -> singular up to a constant. Solvability:
        # integral(R) = gp - gm. Pin the gauge Pi(node 0 at y=+1)... use mid node = 0.
        A = L.copy(); b = R.copy()
        A[0, :] = D[0, :]; b[0] = gp            # D Pi(+1) = gp
        A[N, :] = D[N, :]; b[N] = gm            # D Pi(-1) = gm  (one of these is redundant)
        mid = N // 2
        A[mid, :] = 0.0; A[mid, mid] = 1.0; b[mid] = 0.0   # gauge: Pi(y=x[mid]) = 0
        return np.linalg.solve(A, b), x


def mms_check(N, k2):
    """Manufactured solution: Pi_exact = cos(2y) + 0.3 y^3."""
    D, x = cheb(N)
    Pi_ex = np.cos(2.0 * x) + 0.3 * x**3
    dPi   = -2.0 * np.sin(2.0 * x) + 0.9 * x**2          # analytic D Pi
    d2Pi  = -4.0 * np.cos(2.0 * x) + 1.8 * x             # analytic D^2 Pi
    R  = d2Pi - k2 * Pi_ex
    gp = dPi[0]      # D Pi at y=+1
    gm = dPi[N]      # D Pi at y=-1
    Pi, _ = neumann_poisson(N, k2, R, gp, gm)
    if k2 > 0.0:
        err = np.max(np.abs(Pi - Pi_ex))
    else:
        mid = N // 2                                     # gauge-invariant compare (up to const)
        err = np.max(np.abs((Pi - Pi[mid]) - (Pi_ex - Pi_ex[mid])))
    return err


# =============================================================================
# SECOND MODEL: the EXACT Fortran construction (lift + Dirichlet-helm + Neumann
# match), in Chebyshev COEFFICIENT space. This is what par_pressure_mod actually
# does, so it de-risks the Fortran 1:1 (the collocation model above only checked
# the influence-matrix *mechanics*). The homogeneous solutions are NOT analytic
# cosh/sinh here: they are built by the SAME lift trick the v-solve uses
# (vsolve_if_mod vif_setup, lines "g_+ = l_+ + ghat"), so a sign/normalisation
# bug in that path is caught now, in numpy, not on the cluster.
#
#   phi_+ : (D^2-k^2) phi_+ = 0,  phi_+(+1)=1, phi_+(-1)=0   (== gpc in vif_setup)
#   phi_- : (D^2-k^2) phi_- = 0,  phi_-(+1)=0, phi_-(-1)=1   (== gmc)
# built as  phi = l + ghat,  l the linear lift,  (D^2-k^2)ghat = k^2 l, ghat(+-1)=0.
# Interior Dirichlet-0 solves stand in for helm_band; the COEFFICIENT-space
# derivative recurrence (cheb_deriv1) stands in for cheb_deriv_c / wall_derivs_c.
# =============================================================================

def vals_to_coeffs(v):
    """Chebyshev coeffs a[0..N] from values at the GLC nodes x[j]=cos(pi j/N)."""
    N = v.size - 1
    # a_k = (2/(N c_k)) sum_j (1/cbar_j) v_j cos(pi j k / N),  c_0=c_N=2 else 1
    j = np.arange(N + 1)
    cbar = np.ones(N + 1); cbar[0] = 2.0; cbar[N] = 2.0
    a = np.zeros(N + 1)
    for k in range(N + 1):
        ck = 2.0 if (k == 0 or k == N) else 1.0
        a[k] = (2.0 / (N * ck)) * np.sum(v / cbar * np.cos(np.pi * j * k / N))
    return a


def cheb_deriv1(a):
    """First-derivative recurrence in coeff space (matches cheb_shen cheb_deriv /
    cheb_ops_mod cheb_deriv_c): b_k = b_{k+2} + 2(k+1) a_{k+1}; b_0 *= 1/2."""
    N = a.size - 1
    b = np.zeros(N + 1)
    for k in range(N - 1, -1, -1):
        b[k] = (b[k + 2] if k + 2 <= N else 0.0) + 2.0 * (k + 1) * a[k + 1]
    b[0] *= 0.5
    return b


def wall_derivs_coeff(a):
    """Du(+1)=sum_j j^2 a_j ; Du(-1)=sum_j (-1)^{j-1} j^2 a_j  (spec_helpers wall_derivs_c)."""
    N = a.size - 1
    j = np.arange(1, N + 1)
    dhi = np.sum(j * j * a[1:])
    dlo = np.sum(((-1.0) ** (j - 1)) * j * j * a[1:])
    return dlo, dhi


def helm_dir0_coeff(N, k2, R_coeff):
    """Solve (D^2-k^2) u = R, u(+-1)=0, returning Chebyshev coeffs. Collocation
    stand-in for the Shen helm_band solve (which is what the Fortran uses)."""
    D, x = cheb(N)
    L = D @ D - k2 * np.eye(N + 1)
    R_vals = np.polynomial.chebyshev.chebval(x, R_coeff)
    u_vals = solve_dirichlet(L, N, R_vals, 0.0, 0.0)
    return vals_to_coeffs(u_vals)


def pressure_lift_path(N, k2, R_coeff, gp, gm):
    """par_pressure_mod's exact recipe, in coeff space. Returns Pi (Cheb coeffs)."""
    _, x = cheb(N)
    # lift functions l_+ = (1+y)/2, l_- = (1-y)/2  in Cheb coeffs: [0.5, +-0.5]
    lp = np.zeros(N + 1); lp[0] = 0.5; lp[1] = 0.5
    lm = np.zeros(N + 1); lm[0] = 0.5; lm[1] = -0.5
    # phi_+ = l_+ + ghat,  (D^2-k^2) ghat = k^2 l_+
    ghat = helm_dir0_coeff(N, k2, k2 * lp); phi_p = lp + ghat
    ghat = helm_dir0_coeff(N, k2, k2 * lm); phi_m = lm + ghat
    # particular: (D^2-k^2) Pi_p = R, Pi_p(+-1)=0
    Pi_p = helm_dir0_coeff(N, k2, R_coeff)
    dPi_lo, dPi_hi = wall_derivs_coeff(Pi_p)
    dphip_lo, dphip_hi = wall_derivs_coeff(phi_p)
    dphim_lo, dphim_hi = wall_derivs_coeff(phi_m)
    if k2 > 0.0:
        # Neumann 2x2:  [Dphi_+(+1) Dphi_-(+1); Dphi_+(-1) Dphi_-(-1)] [c+;c-] = [gp-DPi_p(+1); gm-DPi_p(-1)]
        M = np.array([[dphip_hi, dphim_hi], [dphip_lo, dphim_lo]])
        rhs = np.array([gp - dPi_hi, gm - dPi_lo])
        c = np.linalg.solve(M, rhs)
        return Pi_p + c[0] * phi_p + c[1] * phi_m
    else:
        # k=0: M singular (Dphi_+ = +1/2, Dphi_- = -1/2 at both walls). Use only the
        # symmetric homogeneous mode l_+ + l_- = const? No -- pin the gauge directly:
        # the antisymmetric combination phi_+ - phi_- = y carries the net wall slope.
        # D(Pi_p + c*(phi_+ - phi_-))(+1) = gp  with phi_+ - phi_- = y (D=1) => c = gp - dPi_hi.
        # (Compatibility gp==gm is assumed; constant gauge left free, pinned at mid node.)
        c = gp - dPi_hi
        Pi = Pi_p + c * (phi_p - phi_m)
        return Pi


def mms_check_lift(N, k2):
    """MMS for the lift path: Pi_exact = cos(2y)+0.3 y^3 (in Cheb coeffs)."""
    _, x = cheb(N)
    Pi_ex_vals = np.cos(2.0 * x) + 0.3 * x**3
    Pi_ex = vals_to_coeffs(Pi_ex_vals)
    d1 = cheb_deriv1(Pi_ex); d2 = cheb_deriv1(d1)
    R_coeff = d2 - k2 * Pi_ex
    gp = np.sum(np.arange(0, N + 1) ** 2 * Pi_ex)               # DPi(+1) = sum j^2 a_j
    j = np.arange(0, N + 1); gm = np.sum(((-1.0) ** (j - 1)) * (j * j) * Pi_ex) * 0 + \
        np.sum(((-1.0) ** (np.arange(1, N + 1) - 1)) * (np.arange(1, N + 1) ** 2) * Pi_ex[1:])
    Pi = pressure_lift_path(N, k2, R_coeff, gp, gm)
    if k2 > 0.0:
        err = np.max(np.abs(Pi - Pi_ex))
    else:
        # gauge-invariant compare in COEFF space: the gauge constant is purely the
        # a_0 (T_0) coefficient, so compare all coeffs k>=1 (a_0 is free).
        err = np.max(np.abs(Pi[1:] - Pi_ex[1:]))
    return err


if __name__ == "__main__":
    N = 48
    print(f"  [model 1] Neumann-Poisson MMS (Pi=cos(2y)+0.3y^3, N={N} collocation):")
    allok = True
    for k2 in [0.0, 1.0, 9.0, 100.0, 2500.0]:
        e = mms_check(N, k2)
        ok = e < 1e-8
        allok = allok and ok
        print(f"    k^2={k2:9.1f}  max|Pi - Pi_exact|{' (up to gauge)' if k2==0 else '              '} = {e:.2e}  {'PASS' if ok else 'FAIL'}")

    print(f"\n  [model 2] EXACT Fortran path (lift + Dirichlet-helm + Neumann match), coeff space:")
    for k2 in [0.0, 1.0, 9.0, 100.0, 2500.0]:
        e = mms_check_lift(N, k2)
        ok = e < 1e-7
        allok = allok and ok
        print(f"    k^2={k2:9.1f}  max|Pi - Pi_exact|{' (up to gauge)' if k2==0 else '              '} = {e:.2e}  {'PASS' if ok else 'FAIL'}")

    print("\n  work_pressure_model:", "RECIPE + influence-matrix closure + k=0 gauge VALIDATED"
          if allok else "FAILED")
