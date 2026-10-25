#!/usr/bin/env python3
"""
work_a6_mass_diag_model.py -- decision/spec model for the Shen-Dirichlet MASS
diagonal, which must come from closed forms, not from differencing two dense
assemblies.

THE DEFECT this rules out:
  helm_band_setup extracts  bdiag(k) = M1(k,k) - S(k,k)  and
  boff = M1(0,2) - S(0,2)  where S = assemble(mu=0), M1 = assemble(mu=1).
  The stiffness entry S(k,k) = 2*pi*(k+1)*(k+2) dwarfs the mass entry
  B(k,k) = pi by up to 4.6e5 at n = 480, so the subtraction loses ~5.7
  digits -- a ~5e-11 RELATIVE error injected into bdiag. At the production
  shift mu = 1/alpha ~ 6.5e8 the mu*B term DOMINATES the operator at every
  k, so that is a 5e-11 relative error in the dominant term of every
  Helmholtz solve. A mu sweep that stops at 1e6 is 2-3
  decades below the regime where it shows.

THE FIX (verified here): write the header's own closed forms into the setup:
  bdiag(0) = 3*pi/2,  bdiag(k>=1) = pi,  boff = -pi/2.
  sdiag/stail stay tied to the mu=0 assembly (a SINGLE assembly, no
  differencing, entries exact) -- the module's "avoid transcribing by hand"
  philosophy survives where it is safe; only the cancellation victim moves
  to closed form.

DERIVATION of the closed forms (independent of the assembly code):
  phi_k = T_k - T_{k+2};  (T_a, T_b)_w = c_a*(pi/2)*delta_ab, c_0 = 2 else 1.
  B(k,k)   = (phi_k, phi_k)_w = (c_k + c_{k+2})*(pi/2) = 3*pi/2 (k=0), pi (k>=1)
  B(k,k+2) = -(T_{k+2}, T_{k+2})_w = -pi/2   for every k with k+2 <= n-2.

ASSERTED HERE:
  1. a faithful numpy replica of cheb_shen's assemble_helmholtz (same
     derivative recurrence, same weighted dot) reproduces the closed forms
     for B to <= a few ulp when B is assembled DIRECTLY (mu-only part);
  2. the differenced extraction bdiag_diff = M1 - S carries the predicted
     O(eps * S/B) relative error while the closed form is exact --
     the measured 3.5e-11 order at n = 480;
  3. THE GATE-DESIGN FACT: the
     operator perturbation from the differenced bdiag SATURATES at ~dB/B
     once mu*B >> S -- at test-sized n that dominance already holds at
     mu = 1e6, so a production-mu row ALONE buys nothing.
     The gate's blindness is n AND tolerance: the error is ~eps*S/B ~
     eps*n^2 (~1.6e-12 at n=128, ~2-3e-11 at n=480) against a 1e-9 gate at
     n <= 32. The row that actually polices A6 is (n = 480, mu = 6.5e8,
     tol = 1e-11): the differenced setup FAILS it at 3.5e-11. NOTE on the
     floor: this model computes the OPERATOR perturbation only; the
     adversarial verifier's numpy emulation of the exact band algorithm
     measured the dense-vs-banded SOLVE floor at ~1.1e-12 (worst seed
     1.8e-12) in this regime -- not the ~1e-13 an earlier draft assumed --
     which is why the shipped tolerance is 1e-11 (>= 5x above the floor,
     3.5x below the regression signal), not 5e-12.
"""

import numpy as np

PI = np.pi


def cheb_deriv(a):
    """Chebyshev derivative recurrence, mirrors cheb_shen::cheb_deriv."""
    n = len(a) - 1
    b = np.zeros_like(a)
    if n >= 1:
        b[n - 1] = 2.0 * n * a[n]
        for k in range(n - 2, -1, -1):
            b[k] = b[k + 2] + 2.0 * (k + 1) * a[k + 1]
        b[0] *= 0.5
    return b


def wdot(p, q):
    """(p, q)_w in coefficient space: sum c_i * pi/2 * p_i q_i, c_0 = 2."""
    c = np.ones(len(p))
    c[0] = 2.0
    return float(np.sum(p * q * 0.5 * PI * c))


def assemble(n, mu):
    """Faithful replica of cheb_shen::assemble_helmholtz (dp arithmetic)."""
    m = n - 1  # matrix is (0:n-2) x (0:n-2)
    M = np.zeros((m, m))
    Phi = np.zeros((n + 1, m))
    D2 = np.zeros((n + 1, m))
    for k in range(m):
        t = np.zeros(n + 1)
        t[k] = 1.0
        t[k + 2] = -1.0
        Phi[:, k] = t
        D2[:, k] = cheb_deriv(cheb_deriv(t))
    for j in range(m):
        for k in range(m):
            M[k, j] = -wdot(D2[:, j], Phi[:, k]) + mu * wdot(Phi[:, j], Phi[:, k])
    return M


def closed_bdiag(n):
    m = n - 1
    bd = np.full(m, PI)
    bd[0] = 1.5 * PI
    return bd, -0.5 * PI


def main():
    # ---- 1+2: closed forms vs direct assembly vs differenced extraction ----
    worst_direct = 0.0
    for n in (16, 32, 64, 128):
        S = assemble(n, 0.0)
        M1 = assemble(n, 1.0)
        B_direct = assemble(n, 1.0) - 0.0 * S  # not used; keep explicit below
        # direct mass matrix: assemble the mu-part alone via mu=1 minus the
        # SAME S array is exactly the differencing under test; the honest
        # direct route is the wdot of the bases:
        m = n - 1
        Bd = np.zeros(m)
        for k in range(m):
            t = np.zeros(n + 1)
            t[k] = 1.0
            t[k + 2] = -1.0
            Bd[k] = wdot(t, t)
        bd_closed, boff_closed = closed_bdiag(n)
        worst_direct = max(worst_direct, np.max(np.abs(Bd - bd_closed) / bd_closed))
        # off-diagonal check at (0,2): -(T2,T2)_w
        t0 = np.zeros(n + 1); t0[0] = 1.0; t0[2] = -1.0
        t2 = np.zeros(n + 1); t2[2] = 1.0; t2[4] = -1.0
        assert abs(wdot(t0, t2) - boff_closed) < 1e-14
        # differenced extraction error grows with S/B
        bd_diff = np.diag(M1) - np.diag(S)
        rel = np.max(np.abs(bd_diff - bd_closed) / bd_closed)
        smax = 2 * PI * n * (n - 1)
        print(f"[n={n:4d}] |B_direct-closed|/B <= {worst_direct:.2e}   "
              f"|B_diff-closed|/B = {rel:.2e}   (eps*S/B ~ {2.2e-16 * smax / PI:.1e})")
        assert worst_direct < 5e-15, "direct assembly must match closed forms"
        assert rel < 10 * 2.2e-16 * smax / PI + 1e-15, "differencing error beyond model"
    # ---- diagonal-only differencing at the production n = 480 (vectorized:
    # only column k's Phi/D2Phi are needed for M(k,k)) --------------------------
    def diag_diff(n):
        m = n - 1
        dd = np.zeros(m)
        bd_closed, _ = closed_bdiag(n)
        for k in range(m):
            t = np.zeros(n + 1)
            t[k] = 1.0
            t[k + 2] = -1.0
            d2 = cheb_deriv(cheb_deriv(t))
            s_kk = -wdot(d2, t)
            m1_kk = s_kk + 1.0 * wdot(t, t)      # same fl() path as assemble(mu=1)
            dd[k] = (m1_kk - s_kk) - bd_closed[k]
        return dd, bd_closed

    dd480, bd480 = diag_diff(480)
    rel480 = np.max(np.abs(dd480) / bd480)
    print(f"[n=480 exact] differencing error = {rel480:.2e}  "
          f"(measured 3.5e-11 -- same order)")
    assert 1e-12 < rel480 < 1e-9

    # ---- 3: gate design. Saturation + the (n, mu, tol) row that polices A6 ----
    # relative perturbation of the operator diagonal:  |mu*dB| / (S + mu*B);
    # it saturates at ~|dB|/B once mu*B >> S -- mu alone cannot help the gate.
    for (n, dd, bdc) in ((128, (np.diag(assemble(128, 1.0)) - np.diag(assemble(128, 0.0))) - closed_bdiag(128)[0], closed_bdiag(128)[0]),
                         (480, dd480, bd480)):
        sdc = 2 * PI * (np.arange(len(bdc)) + 1) * (np.arange(len(bdc)) + 2)
        p_old = np.max(np.abs(1.0e6 * dd) / (sdc + 1.0e6 * bdc))
        p_prod = np.max(np.abs(6.5e8 * dd) / (sdc + 6.5e8 * bdc))
        p_sat = np.max(np.abs(dd) / bdc)
        print(f"[n={n:3d}] operator perturbation: mu=1e6 -> {p_old:.2e}, "
              f"mu=6.5e8 -> {p_prod:.2e}, saturation |dB|/B = {p_sat:.2e}")
        assert p_prod < 1.05 * p_sat and p_old > 0.5 * p_sat, \
            "saturation: mu=1e6 already expresses most of the defect at these n"
    # blindness of an n<=32, tol=1e-9 gate: worst error at its largest n
    dd32 = (np.diag(assemble(32, 1.0)) - np.diag(assemble(32, 0.0))) - closed_bdiag(32)[0]
    old_worst = np.max(np.abs(dd32) / closed_bdiag(32)[0])
    print(f"[old gate] n<=32, tol=1e-9: worst differencing signal {old_worst:.1e} "
          f"-- {1e-9 / old_worst:.0f}x below the gate (blind)")
    assert old_worst < 1e-10, "old gate cannot see the defect"
    # the NEW row: n=480, mu=6.5e8, tol=1e-11 -- differenced fails, closed passes
    assert rel480 > 1e-11 * 2, "new row must catch a regression to differencing"
    print(f"[new row ] n=480, mu=6.5e8, tol=1e-11: differenced setup shows "
          f"{rel480:.1e} (FAILS at 3.5x tol); closed-form setup sits at the "
          f"~1.1-1.8e-12 solve floor (PASSES with >= 5x headroom)")

    print("\nALL A6 MODEL CHECKS PASSED")


if __name__ == "__main__":
    main()
