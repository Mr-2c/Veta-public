!===============================================================================
! fd_mean_mod.F90 -- the FD-y MEAN (k^2 = 0) machinery: the
! point-value twin of mean_mod, mirrored statement-for-statement with the
! basis primitives swapped at the FD-y seam:
!   Shen projection / conversion  ->  identity (state = point values)
!   helm_band_solve               ->  fdops Dirichlet-Helmholtz band solve
!   cheb_d2                       ->  banded D2 matvec
!   T_0 constant source           ->  pointwise constant source
!   Chebyshev exact flux weights  ->  the ygrid FILE's quadrature weights w_j
!                                     (trapz by default from gen_ygrid -- the
!                                     2nd-order caveat is the FILE's, gated in
!                                     test_fd_mean; swap the file's w column
!                                     for a higher-order rule any time, no
!                                     code change)
! Mean profiles are REAL point values q0(1:ny) on the DESCENDING file grid,
! walls q0(1) = q0(ny) = 0 (no-slip). Same SMR/CN weights as mean_mod (shared
! constants imported -- one source of truth). HOST REFERENCE: this is the
! oracle for the round-2 device mean (composed from the batched FD primitives
! at nx = nz = 1). Deps: kinds + fdops + mean_mod (constants only).
!===============================================================================
module fd_mean_mod
  use kinds,     only: dp, ip
  use fdops_mod, only: fd_apply_band, fd_helm_dirichlet, fd_band_factor, fd_band_solve
  use mean_mod,  only: MGAMMA, MZETA, MCSUB, H_CH
  implicit none
  private
  public :: fd_mean_flux, fd_mean_stage, fd_step_mean_lin
  public :: fd_mean_flux_setup, fd_mean_stage_cfr

contains

  !-- bulk integral int_{-1}^{1} q dy from point values: the file's weights ----
  pure function fd_mean_flux(ny, wh, q) result(qint)
    integer(ip), intent(in) :: ny
    real(dp),    intent(in) :: wh(ny), q(ny)
    real(dp) :: qint
    qint = dot_product(wh, q)
  end function fd_mean_flux

  !-----------------------------------------------------------------------------
  ! One SMR/CN stage for a mean profile q0(y) at k^2 = 0, Dirichlet walls.
  ! Mirrors mean_mod::mean_stage: rhs = q0 + a D2 q0 + g Nq + z Nq_prev + ck dt
  ! beta (pointwise constant), then (-D2 + lam) q0 = lam*rhs with identity wall
  ! rows (rhs walls zeroed = homogeneous Dirichlet).
  !-----------------------------------------------------------------------------
  subroutine fd_mean_stage(ny, kl, d2b, ks, dt, Re, beta, q0, Nq, Nq_prev, ok)
    integer(ip), intent(in)    :: ny, kl, ks
    real(dp),    intent(in)    :: d2b(2*kl+1, ny), dt, Re, beta
    real(dp),    intent(in)    :: Nq(ny), Nq_prev(ny)
    real(dp),    intent(inout) :: q0(ny)
    logical,     intent(out)   :: ok
    real(dp) :: ab(2*kl+1, ny), d2(ny), rhs(ny)
    real(dp) :: bk, a, lam, g, z, ck

    bk  = 0.5_dp*MCSUB(ks)
    a   = bk*dt/Re
    lam = 1.0_dp/a
    g   = MGAMMA(ks)*dt
    z   = MZETA(ks)*dt
    ck  = MCSUB(ks)

    call fd_apply_band(ny, kl, d2b, q0, d2)             ! D2 q0 (= L q0 at k^2=0)
    rhs = q0 + a*d2 + g*Nq + z*Nq_prev
    rhs = rhs + ck*dt*beta                              ! constant source, pointwise
    rhs = lam*rhs
    rhs(1) = 0.0_dp;  rhs(ny) = 0.0_dp                  ! homogeneous Dirichlet walls
    call fd_helm_dirichlet(ny, kl, d2b, lam, ab)
    call fd_band_factor(ny, kl, ab, ok)
    if (.not. ok) return
    call fd_band_solve(ny, kl, ab, rhs)
    q0 = rhs
  end subroutine fd_mean_stage

  !-- one full step (3 SMR stages), LINEAR (N = 0), constant beta ---------------
  subroutine fd_step_mean_lin(ny, kl, d2b, dt, Re, beta, q0, ok)
    integer(ip), intent(in)    :: ny, kl
    real(dp),    intent(in)    :: d2b(2*kl+1, ny), dt, Re, beta
    real(dp),    intent(inout) :: q0(ny)
    logical,     intent(out)   :: ok
    real(dp) :: zeroN(ny)
    integer(ip) :: ks
    zeroN = 0.0_dp
    ok = .true.
    do ks = 1, 3
       call fd_mean_stage(ny, kl, d2b, ks, dt, Re, beta, q0, zeroN, zeroN, ok)
       if (.not. ok) return
    end do
  end subroutine fd_step_mean_lin

  !-----------------------------------------------------------------------------
  ! Per-stage unit-source response U_h ([1 - a_k D2] U_h = 1, Dirichlet walls)
  ! and its bulk integral (the file's weights). Mirrors mean_flux_setup; state-
  ! independent, recompute on dt/Re change.
  !-----------------------------------------------------------------------------
  subroutine fd_mean_flux_setup(ny, kl, d2b, wh, dt, Re, Uh_all, Ih_all, ok)
    integer(ip), intent(in)  :: ny, kl
    real(dp),    intent(in)  :: d2b(2*kl+1, ny), wh(ny), dt, Re
    real(dp),    intent(out) :: Uh_all(ny,3), Ih_all(3)
    logical,     intent(out) :: ok
    real(dp) :: ab(2*kl+1, ny), rhs(ny)
    real(dp) :: bk, a, lam
    integer(ip) :: ks
    ok = .true.
    do ks = 1, 3
       bk  = 0.5_dp*MCSUB(ks)
       a   = bk*dt/Re
       lam = 1.0_dp/a
       rhs = lam                                        ! constant source "1", lam-scaled
       rhs(1) = 0.0_dp;  rhs(ny) = 0.0_dp
       call fd_helm_dirichlet(ny, kl, d2b, lam, ab)
       call fd_band_factor(ny, kl, ab, ok)
       if (.not. ok) return
       call fd_band_solve(ny, kl, ab, rhs)
       Uh_all(:,ks) = rhs
       Ih_all(ks)   = fd_mean_flux(ny, wh, rhs)
    end do
  end subroutine fd_mean_flux_setup

  !-----------------------------------------------------------------------------
  ! One CFR stage: beta=0 particular solve, then beta closes the flux constraint
  !     q0 = U_p + (c_k dt beta) U_h,  beta = (H*Q_target - int U_p)/(c_k dt Ih).
  ! Identical algebra to mean_mod::mean_stage_cfr with the FD flux. Because beta
  ! is chosen with the SAME quadrature that measures the flux, int q0 = H*Q to
  ! rounding REGARDLESS of the weights' order (the constraint is exact; the
  ! weights' order only affects what "int" means physically) -- gated.
  !-----------------------------------------------------------------------------
  subroutine fd_mean_stage_cfr(ny, kl, d2b, wh, ks, dt, Re, Q_target, Uh, Ih, &
                               q0, Nq, Nq_prev, beta_out, ok)
    integer(ip), intent(in)    :: ny, kl, ks
    real(dp),    intent(in)    :: d2b(2*kl+1, ny), wh(ny), dt, Re, Q_target, Ih
    real(dp),    intent(in)    :: Uh(ny), Nq(ny), Nq_prev(ny)
    real(dp),    intent(inout) :: q0(ny)
    real(dp),    intent(out)   :: beta_out
    logical,     intent(out)   :: ok
    real(dp) :: ck
    ck = MCSUB(ks)
    call fd_mean_stage(ny, kl, d2b, ks, dt, Re, 0.0_dp, q0, Nq, Nq_prev, ok)  ! U_p
    if (.not. ok) return
    beta_out = (H_CH*Q_target - fd_mean_flux(ny, wh, q0))/(ck*dt*Ih)
    q0 = q0 + (ck*dt*beta_out)*Uh
  end subroutine fd_mean_stage_cfr

end module fd_mean_mod
