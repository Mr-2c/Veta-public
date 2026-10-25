!===============================================================================
! mean_mod.F90  --  mean-profile (0,0)-mode stepper (CPU, dp).
!
! At (k_x,k_z)=(0,0) the v/eta formulation degenerates (v == 0 and omega_y == 0
! by continuity), so the mean velocity profiles u0(y), w0(y) are carried OUTSIDE
! (v,eta) and advanced by their own x-z-averaged momentum equations:
!
!     d u0/dt = (1/Re) D^2 u0 + beta + N_u0 ,   u0(+/-1) = 0   (no-slip)
!     d w0/dt = (1/Re) D^2 w0 +  0   + N_w0 ,   w0(+/-1) = 0
!
! beta = -dP/dx is the mean (streamwise) pressure gradient; N_* are the (0,0)
! modes of the rotational nonlinear term (the hx0/hz0 that nl_evaluate outputs).
! The viscous operator is the Helmholtz operator at k^2 = 0 with DIRICHLET BCs --
! structurally identical to the eta equation -- so this reuses the validated
! Shen-Dirichlet banded machinery (helmholtz_rhs / helm_band_solve / shen_to_cheb)
! at k^2 = 0, mu = lambda.
!
! Time scheme: the same SMR RKW3 / symmetric-CN stage as stepper_mod, per stage k:
!     [1 - a_k D^2] q0^k = [1 + a_k D^2] q0^{k-1}
!                          + gamma_k dt N^{k-1} + zeta_k dt N^{k-2}
!                          + c_k dt beta
! with a_k = beta_k dt/Re, beta_k = c_k/2 (symmetric CN), c_k = gamma_k + zeta_k.
! The constant-in-y source beta integrates over the substep as c_k dt beta (added
! to the T_0 coefficient). Mapping to the cheb_shen convention (-D^2 + mu)q = f:
! mu = lambda = 1/a_k and f = lambda * project(RHS). Sum_k c_k = 1, so a full
! 3-stage step advances time by dt and injects dt*beta.
!
! Kept in dp regardless of wp: the mean profile accumulates the Reynolds-stress
! balance over an entire run (it IS the U(y) compared to MKM), so it is solved on
! the host in dp. Build: plain `nvfortran -c mean_mod.F90` (depends on cheb_shen,
! helm_band_mod, types).
!
! The mass-flux constraint (solving for beta to hold the bulk flux) is added in
! 6(d-ii); here beta is an explicit argument (constant-pressure-gradient mode,
! and the building block the constraint will reuse).
!===============================================================================
module mean_mod
  use kinds,         only: dp, ip
  use cheb_shen,     only: cheb_d2, helmholtz_rhs, shen_to_cheb
  use helm_band_mod, only: helm_band_solve
  use yops_mod,      only: yop_mean_flux          ! the y-op facade
  implicit none
  private
  public :: mean_stage, step_mean_lin, MGAMMA, MZETA, MCSUB
  public :: mean_flux, bulk_velocity, mean_flux_setup, mean_stage_cfr, step_mean_cfr

  ! SMR RKW3 weights (same as stepper_mod): gamma, zeta, and c = gamma + zeta.
  real(dp), parameter :: MGAMMA(3) = [ 8.0_dp/15.0_dp,  5.0_dp/12.0_dp, 3.0_dp/4.0_dp ]
  real(dp), parameter :: MZETA(3)  = [ 0.0_dp,        -17.0_dp/60.0_dp, -5.0_dp/12.0_dp ]
  real(dp), parameter :: MCSUB(3)  = [ 8.0_dp/15.0_dp,  2.0_dp/15.0_dp,  1.0_dp/3.0_dp ]

  ! Channel FULL height: walls at y = +/-1, so the domain [-1,1] has length 2.
  ! The bulk velocity is U_b = (1/H_CH) * int_{-1}^{1} u dy, and the CFR target
  ! Q_target is the BULK VELOCITY (not the raw mass-flux integral). Laminar
  ! Poiseuille then has centreline u_c = (3/2) U_b (the usual 1.5 for U_b = 1).
  real(dp), parameter, public :: H_CH = 2.0_dp

contains

  !-----------------------------------------------------------------------------
  ! One SMR/CN stage for a mean profile q0(y) at k^2 = 0, Dirichlet q0(+/-1)=0.
  !   ks       : stage index 1..3
  !   beta     : constant-in-y source (mean dp/dx term for u0; 0 for w0)
  !   q0       : Chebyshev coefficients (0:n), updated IN PLACE
  !   Nq,Nq_prev: (0,0)-mode nonlinear forcing at this / previous stage (Cheb coeffs)
  ! sdiag/stail/bdiag/boff are the helm_band structure from helm_band_setup(n,...).
  !-----------------------------------------------------------------------------
  subroutine mean_stage(n, ks, dt, Re, beta, sdiag, stail, bdiag, boff, q0, Nq, Nq_prev)
    integer(ip), intent(in)    :: n, ks
    real(dp),    intent(in)    :: dt, Re, beta, boff
    real(dp),    intent(in)    :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2)
    real(dp),    intent(in)    :: Nq(0:n), Nq_prev(0:n)
    real(dp),    intent(inout) :: q0(0:n)
    real(dp) :: d2(0:n), rhs(0:n), ftil(0:n-2), qsh(0:n-2)
    real(dp) :: bk, a, lam, g, z, ck

    bk  = 0.5_dp*MCSUB(ks)       ! symmetric-CN beta_k
    a   = bk*dt/Re               ! per-stage viscous coefficient
    lam = 1.0_dp/a               ! Helmholtz shift (mu = k^2 + lambda, k^2 = 0)
    g   = MGAMMA(ks)*dt
    z   = MZETA(ks)*dt
    ck  = MCSUB(ks)

    call cheb_d2(n, q0, d2)                       ! D^2 q0  (= L q0 at k^2=0)
    rhs = q0 + a*d2 + g*Nq + z*Nq_prev
    rhs(0) = rhs(0) + ck*dt*beta                  ! constant source -> T_0 coeff
    call helmholtz_rhs(n, rhs, ftil)              ! project onto Shen-Dirichlet test space
    ftil = lam*ftil                               ! map [1 - a D^2] = RHS  ->  (-D^2 + lam) = lam*RHS
    call helm_band_solve(n, lam, sdiag, stail, bdiag, boff, ftil, qsh)
    call shen_to_cheb(n, qsh, q0)                 ! Shen -> Cheb (Dirichlet by construction)
  end subroutine mean_stage

  !-----------------------------------------------------------------------------
  ! One full step (3 SMR stages), LINEAR (N = 0), constant beta.  6(d-i) helper /
  ! constant-pressure-gradient driver.
  !-----------------------------------------------------------------------------
  subroutine step_mean_lin(n, dt, Re, beta, sdiag, stail, bdiag, boff, q0)
    integer(ip), intent(in)    :: n
    real(dp),    intent(in)    :: dt, Re, beta, boff
    real(dp),    intent(in)    :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2)
    real(dp),    intent(inout) :: q0(0:n)
    real(dp) :: zeroN(0:n)
    integer(ip) :: ks
    zeroN = 0.0_dp
    do ks = 1, 3
       call mean_stage(n, ks, dt, Re, beta, sdiag, stail, bdiag, boff, q0, zeroN, zeroN)
    end do
  end subroutine step_mean_lin

  !=============================================================================
  ! Constant-mass-flux (CFR) machinery.
  !
  ! Hold the bulk velocity U_b = (1/H) int_{-1}^{1} u0 dy = Q_target fixed (H = 2,
  ! the channel full height) by solving for the mean pressure gradient beta each
  ! stage. Since u0 is linear in beta, split the stage into a beta=0 particular
  ! solve U_p and the (state-independent) response U_h to a unit constant source,
  ! then choose beta to hit the target:
  !     u0 = U_p + (c_k dt beta) U_h ,  beta = (H*Q_target - int U_p)/(c_k dt int U_h).
  ! U_h and int U_h depend only on (dt,Re,stage) -> precomputed once by
  ! mean_flux_setup. The flux integral uses the exact Chebyshev weights
  !     int_{-1}^{1} T_k dy = 2/(1-k^2)  (even k),   0 (odd k).
  ! Constant pressure gradient (CPG) is the special case: skip the constraint and
  ! pass beta to mean_stage directly (the 6(d-i) path).
  !=============================================================================

  ! Bulk integral int_{-1}^{1} q(y) dy from Chebyshev coefficients q(0:n).
  ! The implementation lives in yops_mod (yop_mean_flux); this public name is
  ! kept so the many callers (driver, tests, CFR) can use it directly.
  pure function mean_flux(n, q) result(qint)
    integer(ip), intent(in) :: n
    real(dp),    intent(in) :: q(0:n)
    real(dp) :: qint
    qint = yop_mean_flux(n, q)
  end function mean_flux

  !-- bulk velocity = (1/H) int_{-1}^{1} u dy  (the channel normalisation) --------
  pure function bulk_velocity(n, q) result(ub)
    integer(ip), intent(in) :: n
    real(dp),    intent(in) :: q(0:n)
    real(dp) :: ub
    ub = mean_flux(n, q)/H_CH
  end function bulk_velocity

  ! Precompute, per stage, the unit-source response U_h (solving [1 - a_k D^2]U_h = 1)
  ! and its bulk integral int U_h. State-independent; recompute only when dt or Re
  ! changes. Uh_all(0:n,3) are Chebyshev coeffs; Ih_all(3) their integrals.
  subroutine mean_flux_setup(n, dt, Re, sdiag, stail, bdiag, boff, Uh_all, Ih_all)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: dt, Re, boff
    real(dp),    intent(in)  :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2)
    real(dp),    intent(out) :: Uh_all(0:n,3), Ih_all(3)
    real(dp) :: rhs(0:n), ftil(0:n-2), qsh(0:n-2)
    real(dp) :: bk, a, lam
    integer(ip) :: ks
    do ks = 1, 3
       bk  = 0.5_dp*MCSUB(ks)
       a   = bk*dt/Re
       lam = 1.0_dp/a
       rhs = 0.0_dp; rhs(0) = 1.0_dp          ! constant source "1" -> T_0 coeff
       call helmholtz_rhs(n, rhs, ftil)
       ftil = lam*ftil
       call helm_band_solve(n, lam, sdiag, stail, bdiag, boff, ftil, qsh)
       call shen_to_cheb(n, qsh, Uh_all(:,ks))
       Ih_all(ks) = mean_flux(n, Uh_all(:,ks))
    end do
  end subroutine mean_flux_setup

  ! One SMR/CN stage for u0 with the constant-mass-flux constraint. Solves the
  ! beta=0 particular profile, then sets beta so int u0 = Q_target and returns it
  ! (the instantaneous mean pressure gradient). Uh/Ih are the stage's precomputed
  ! unit-source response and its integral from mean_flux_setup.
  subroutine mean_stage_cfr(n, ks, dt, Re, Q_target, sdiag, stail, bdiag, boff, &
                            Uh, Ih, q0, Nq, Nq_prev, beta_out)
    integer(ip), intent(in)    :: n, ks
    real(dp),    intent(in)    :: dt, Re, Q_target, boff, Ih
    real(dp),    intent(in)    :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2)
    real(dp),    intent(in)    :: Uh(0:n), Nq(0:n), Nq_prev(0:n)
    real(dp),    intent(inout) :: q0(0:n)
    real(dp),    intent(out)   :: beta_out
    real(dp) :: d2(0:n), rhs(0:n), ftil(0:n-2), qsh(0:n-2), Up(0:n)
    real(dp) :: bk, a, lam, g, z, ck, Ip

    bk  = 0.5_dp*MCSUB(ks)
    a   = bk*dt/Re
    lam = 1.0_dp/a
    g   = MGAMMA(ks)*dt
    z   = MZETA(ks)*dt
    ck  = MCSUB(ks)

    ! particular (beta = 0) solve
    call cheb_d2(n, q0, d2)
    rhs = q0 + a*d2 + g*Nq + z*Nq_prev
    call helmholtz_rhs(n, rhs, ftil)
    ftil = lam*ftil
    call helm_band_solve(n, lam, sdiag, stail, bdiag, boff, ftil, qsh)
    call shen_to_cheb(n, qsh, Up)

    ! choose beta so that the BULK VELOCITY (int/H_CH) equals Q_target, i.e.
    ! int(Up + c_k dt beta Uh) = H_CH * Q_target
    Ip       = mean_flux(n, Up)
    beta_out = (H_CH*Q_target - Ip)/(ck*dt*Ih)
    q0       = Up + (ck*dt*beta_out)*Uh
  end subroutine mean_stage_cfr

  ! One full step (3 stages), LINEAR (N = 0), constant mass flux. beta_out returns
  ! the final-stage mean pressure gradient (all stages coincide at steady state).
  subroutine step_mean_cfr(n, dt, Re, Q_target, sdiag, stail, bdiag, boff, &
                           Uh_all, Ih_all, q0, beta_out)
    integer(ip), intent(in)    :: n
    real(dp),    intent(in)    :: dt, Re, Q_target, boff
    real(dp),    intent(in)    :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2)
    real(dp),    intent(in)    :: Uh_all(0:n,3), Ih_all(3)
    real(dp),    intent(inout) :: q0(0:n)
    real(dp),    intent(out)   :: beta_out
    real(dp) :: zeroN(0:n)
    integer(ip) :: ks
    zeroN = 0.0_dp
    do ks = 1, 3
       call mean_stage_cfr(n, ks, dt, Re, Q_target, sdiag, stail, bdiag, boff, &
                           Uh_all(:,ks), Ih_all(ks), q0, zeroN, zeroN, beta_out)
    end do
  end subroutine step_mean_cfr

end module mean_mod
