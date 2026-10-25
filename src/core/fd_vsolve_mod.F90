!===============================================================================
! fd_vsolve_mod.F90 -- the FD-y v-solve: the clamped fourth-order
! problem by the SAME two-Helmholtz factorization + 2x2 influence matrix as the
! validated Chebyshev machine (vsolve_if_mod), with the four basis primitives
! swapped exactly as the seam prescribes:
!     Helmholtz backend      : tau/Shen banded  ->  fd_helm_dirichlet (point)
!     wall-derivative row    : T_k'(+-1) sums   ->  fd_wall_weights stencils
!     lift functions         : (T0 +- T1)/2     ->  point values (1 +- y)/2
!     Shen<->Cheb converters :                  ->  identity (state = values)
! The factorization, the homogeneous-solution construction, the influence
! matrix layout, and every sign MIRROR vsolve_if_mod::vif_setup_band/vif_solve
! statement-for-statement -- correctness is gated CROSS-DISCRETIZATION in
! test_fd_vsolve (FD solution vs the Chebyshev oracle evaluated at the FD
! nodes, over (k2, alpha) sweeps incl. k2 = 0 and production-small alpha).
!
! Per mode (k2 = kx^2+kz^2, alpha = beta dt/Re):
!   setup:  g_+- = l_+- + ghat,  (-D^2+k2) ghat = -k2 l_+-  (walls 0)
!           (-D^2 + k2 + 1/alpha) v_+- = -g_+-              (walls 0)
!           C = [Dv_-(-1) Dv_+(-1); Dv_-(+1) Dv_+(+1)],  Cinv = C^-1
!   solve:  (-D^2+k2) g_p = f/alpha        (walls 0)
!           (-D^2+k2+1/alpha) v_p = -g_p   (walls 0)
!           sigma = -Cinv [Dv_p(-1); Dv_p(+1)];  v = v_p + s1 v_- + s2 v_+
!
! HOST REFERENCE (real-valued; per-mode application is Re/Im): this is the
! CPU oracle for the batched device twin (next round), the same role
! vif_setup_band plays for vif_gpu_setup_dev. Clarity over speed: the solve
! refactors its two bands per call (the device path is build-at-solve-time
! anyway). Host-only; deps: kinds + fdops_mod.
!===============================================================================
module fd_vsolve_mod
  use kinds,     only: dp, ip
  use fdops_mod, only: fd_helm_dirichlet, fd_band_factor, fd_band_solve
  implicit none
  private
  public :: fd_vif_setup, fd_vif_solve

contains

  !---------------------------------------------------------------------------
  ! Homogeneous solutions v_-, v_+ (POINT VALUES, 1..ny on the descending grid)
  ! and the inverse influence matrix, for one (k2, alpha).
  !---------------------------------------------------------------------------
  subroutine fd_vif_setup(ny, kl, y, d2b, wlo, whi, k2, alpha, vm, vp, Cinv, ok)
    integer(ip), intent(in)  :: ny, kl
    real(dp),    intent(in)  :: y(ny), d2b(2*kl+1, ny), wlo(ny), whi(ny)
    real(dp),    intent(in)  :: k2, alpha
    real(dp),    intent(out) :: vm(ny), vp(ny), Cinv(2,2)
    logical,     intent(out) :: ok
    real(dp) :: ab1(2*kl+1, ny), ab2(2*kl+1, ny)
    real(dp) :: lp(ny), lm(ny), gh(ny), gp(ny), gm(ny)
    real(dp) :: mu2, dvm_lo, dvm_hi, dvp_lo, dvp_hi, det
    integer(ip) :: j
    logical :: ok1, ok2

    ok = .false.
    mu2 = k2 + 1.0_dp/alpha
    do j = 1, ny                              ! lifts as point values
       lp(j) = 0.5_dp*(1.0_dp + y(j))
       lm(j) = 0.5_dp*(1.0_dp - y(j))
    end do

    call fd_helm_dirichlet(ny, kl, d2b, k2, ab1)
    call fd_band_factor(ny, kl, ab1, ok1)
    call fd_helm_dirichlet(ny, kl, d2b, mu2, ab2)
    call fd_band_factor(ny, kl, ab2, ok2)
    if (.not. (ok1 .and. ok2)) return

    ! g_+ = l_+ + ghat,  (-D^2+k2) ghat = -k2 l_+  (walls 0)
    gh = -k2*lp;  gh(1) = 0.0_dp;  gh(ny) = 0.0_dp
    call fd_band_solve(ny, kl, ab1, gh)
    gp = gh + lp
    ! g_-
    gh = -k2*lm;  gh(1) = 0.0_dp;  gh(ny) = 0.0_dp
    call fd_band_solve(ny, kl, ab1, gh)
    gm = gh + lm

    ! v_+ : (-D^2+mu2) v_+ = -g_+  (walls 0)
    vp = -gp;  vp(1) = 0.0_dp;  vp(ny) = 0.0_dp
    call fd_band_solve(ny, kl, ab2, vp)
    ! v_-
    vm = -gm;  vm(1) = 0.0_dp;  vm(ny) = 0.0_dp
    call fd_band_solve(ny, kl, ab2, vm)

    ! influence matrix  C = [ Dv_-(-1)  Dv_+(-1) ; Dv_-(+1)  Dv_+(+1) ]
    dvm_lo = dot_product(wlo, vm);  dvm_hi = dot_product(whi, vm)
    dvp_lo = dot_product(wlo, vp);  dvp_hi = dot_product(whi, vp)
    det = dvm_lo*dvp_hi - dvp_lo*dvm_hi
    if (abs(det) < 1.0e3_dp*tiny(1.0_dp)) return       ! singular closure: refuse
    Cinv(1,1) =  dvp_hi/det;  Cinv(1,2) = -dvp_lo/det
    Cinv(2,1) = -dvm_hi/det;  Cinv(2,2) =  dvm_lo/det
    ok = .true.
  end subroutine fd_vif_setup

  !---------------------------------------------------------------------------
  ! Solve for v (point values) given the point-value RHS f.
  !---------------------------------------------------------------------------
  subroutine fd_vif_solve(ny, kl, d2b, wlo, whi, k2, alpha, vm, vp, Cinv, f, v, ok)
    integer(ip), intent(in)  :: ny, kl
    real(dp),    intent(in)  :: d2b(2*kl+1, ny), wlo(ny), whi(ny)
    real(dp),    intent(in)  :: k2, alpha
    real(dp),    intent(in)  :: vm(ny), vp(ny), Cinv(2,2), f(ny)
    real(dp),    intent(out) :: v(ny)
    logical,     intent(out) :: ok
    real(dp) :: ab1(2*kl+1, ny), ab2(2*kl+1, ny), g(ny)
    real(dp) :: dvp_lo, dvp_hi, s1, s2
    logical  :: ok1, ok2

    ok = .false.
    call fd_helm_dirichlet(ny, kl, d2b, k2, ab1)
    call fd_band_factor(ny, kl, ab1, ok1)
    call fd_helm_dirichlet(ny, kl, d2b, k2 + 1.0_dp/alpha, ab2)
    call fd_band_factor(ny, kl, ab2, ok2)
    if (.not. (ok1 .and. ok2)) return

    ! (-D^2+k2) g_p = f/alpha   (walls 0)
    g = f/alpha;  g(1) = 0.0_dp;  g(ny) = 0.0_dp
    call fd_band_solve(ny, kl, ab1, g)
    ! (-D^2+mu2) v_p = -g_p     (walls 0)
    v = -g;  v(1) = 0.0_dp;  v(ny) = 0.0_dp
    call fd_band_solve(ny, kl, ab2, v)

    ! closure:  sigma = -Cinv [Dv_p(-1); Dv_p(+1)] ;  v = v_p + s1 v_- + s2 v_+
    dvp_lo = dot_product(wlo, v)
    dvp_hi = dot_product(whi, v)
    s1 = -( Cinv(1,1)*dvp_lo + Cinv(1,2)*dvp_hi )
    s2 = -( Cinv(2,1)*dvp_lo + Cinv(2,2)*dvp_hi )
    v = v + s1*vm + s2*vp
    ok = .true.
  end subroutine fd_vif_solve

end module fd_vsolve_mod
