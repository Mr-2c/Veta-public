!===============================================================================
! vsolve_if_mod.F90  --  clamped wall-normal v-solve by the two-Helmholtz
! factorization + influence (capacitance) matrix.  CPU / dp.
!
! Solves  (L - alpha L^2) v = f,  L = D^2 - k^2,  v = Dv = 0 at y = +-1,
! by factoring  L - alpha L^2 = -alpha L (L - 1/alpha)  into two Helmholtz
! operators  H1 = D^2 - k^2  and  H2 = D^2 - (k^2 + 1/alpha), each solved with
! the validated Shen-Dirichlet Helmholtz solve (cheb_shen). The two missing
! clamped conditions Dv(+-1)=0 are recovered with a per-mode 2x2 influence matrix.
! See docs/influence_matrix.pdf for the full derivation and sign conventions.
!
! cheb_shen's helmholtz_solve(n,mu,fcheb) solves (-D^2 + mu) u = fcheb, u(+-1)=0,
! so each H_i w = r maps to helmholtz_solve(mu_i, -r). Setup precomputes the two
! homogeneous solutions v_-, v_+ (Chebyshev coeffs) and C^{-1}; per solve does two
! Helmholtz solves + a 2x2 apply + a combine.
!
! Build:
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran -c vsolve_if_mod.F90
!===============================================================================
module vsolve_if_mod
  use kinds,     only: dp, ip
  use cheb_shen, only: helmholtz_solve, shen_to_cheb, helmholtz_rhs
  use helm_band_mod, only: helm_band_solve
  use yops_mod,  only: wall_derivs => yop_wall_deriv   ! the y-op facade
  implicit none
  private
  public :: vif_t, vif_setup, vif_setup_band, vif_solve

  type :: vif_t
     integer(ip) :: n
     real(dp)    :: k2, alpha, mu1, mu2
     real(dp), allocatable :: vm(:), vp(:)   ! homogeneous v_-, v_+ in Cheb coeffs (0:n)
     real(dp)    :: Cinv(2,2)                 ! inverse influence matrix
  end type vif_t

contains

  !---------------------------------------------------------------------------
  ! Precompute the homogeneous solutions and inverse influence matrix for a
  ! given (k2, alpha). Recompute when alpha (i.e. dt) or k2 changes.
  !---------------------------------------------------------------------------
  subroutine vif_setup(n, k2, alpha, d)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: k2, alpha
    type(vif_t), intent(out) :: d
    real(dp) :: lp(0:n), lm(0:n)        ! lift functions, Cheb coeffs
    real(dp) :: ghat(0:n-2)             ! Shen coeffs of lift correction
    real(dp) :: gpc(0:n), gmc(0:n)      ! g_+, g_- in Cheb coeffs
    real(dp) :: vsh(0:n-2)
    real(dp) :: dvm_lo, dvm_hi, dvp_lo, dvp_hi, det

    d%n = n; d%k2 = k2; d%alpha = alpha
    d%mu1 = k2
    d%mu2 = k2 + 1.0_dp/alpha
    allocate(d%vm(0:n), d%vp(0:n))

    ! lift functions  l_+ = (1+y)/2 = (T0+T1)/2 ,  l_- = (1-y)/2 = (T0-T1)/2
    lp = 0.0_dp; lm = 0.0_dp
    lp(0) = 0.5_dp; lp(1) =  0.5_dp
    lm(0) = 0.5_dp; lm(1) = -0.5_dp

    ! g_+ = l_+ + ghat,  H1 ghat = k2 l_+  ->  helmholtz_solve(mu1, -k2 l_+)
    call helmholtz_solve(n, d%mu1, -k2*lp, ghat)
    call shen_to_cheb(n, ghat, gpc)
    gpc = gpc + lp
    ! g_-
    call helmholtz_solve(n, d%mu1, -k2*lm, ghat)
    call shen_to_cheb(n, ghat, gmc)
    gmc = gmc + lm

    ! v_+ : H2 v_+ = g_+  ->  helmholtz_solve(mu2, -g_+)
    call helmholtz_solve(n, d%mu2, -gpc, vsh)
    call shen_to_cheb(n, vsh, d%vp)
    ! v_-
    call helmholtz_solve(n, d%mu2, -gmc, vsh)
    call shen_to_cheb(n, vsh, d%vm)

    ! influence matrix  C = [ Dv_-(-1)  Dv_+(-1) ; Dv_-(+1)  Dv_+(+1) ]
    call wall_derivs(n, d%vm, dvm_lo, dvm_hi)
    call wall_derivs(n, d%vp, dvp_lo, dvp_hi)
    det = dvm_lo*dvp_hi - dvp_lo*dvm_hi
    d%Cinv(1,1) =  dvp_hi/det;  d%Cinv(1,2) = -dvp_lo/det
    d%Cinv(2,1) = -dvm_hi/det;  d%Cinv(2,2) =  dvm_lo/det
  end subroutine vif_setup

  !---------------------------------------------------------------------------
  ! vif_setup_band  --  identical math to vif_setup, but every Helmholtz solve
  ! goes through the validated BANDED path (helmholtz_rhs + helm_band_solve,
  ! O(n) each) instead of the dense assemble+LU (O(n^3) each).
  !
  ! WHY: the GPU precompute (vif_gpu_setup_host) loops this over nx*nz modes x
  ! 3 RK stages -- at 64^3 that is 12288 setups x 4 dense solves ~ 50k dense
  ! assemble+LU at startup (the "long time before stepping begins"), scaling to
  ! ~30+ min at the 128^2 production case. Banded: the whole precompute drops
  ! to well under a second. Agreement with the dense path is the validated
  ! banded-vs-dense relationship (~1e-13; test_helm_band), far inside
  ! test_vif_gpu's 1e-8 tolerance. The dense vif_setup is RETAINED as the
  ! reference (used by the CPU vif tests and the stepper xchecks' CPU side).
  ! Band structure (sd, st, bd, boff) from helm_band_setup, computed ONCE by
  ! the caller and shared across all modes/stages (it is mu-independent).
  !---------------------------------------------------------------------------
  subroutine vif_setup_band(n, k2, alpha, sd, st, bd, boff, d)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: k2, alpha
    real(dp),    intent(in)  :: sd(0:n-2), st(0:n-2), bd(0:n-2), boff
    type(vif_t), intent(out) :: d
    real(dp) :: lp(0:n), lm(0:n)        ! lift functions, Cheb coeffs
    real(dp) :: ft(0:n-2)               ! projected (Shen test space) RHS
    real(dp) :: ghat(0:n-2)             ! Shen coeffs of lift correction
    real(dp) :: gpc(0:n), gmc(0:n)      ! g_+, g_- in Cheb coeffs
    real(dp) :: vsh(0:n-2)
    real(dp) :: dvm_lo, dvm_hi, dvp_lo, dvp_hi, det

    d%n = n; d%k2 = k2; d%alpha = alpha
    d%mu1 = k2
    d%mu2 = k2 + 1.0_dp/alpha
    allocate(d%vm(0:n), d%vp(0:n))

    ! lift functions  l_+ = (1+y)/2 = (T0+T1)/2 ,  l_- = (1-y)/2 = (T0-T1)/2
    lp = 0.0_dp; lm = 0.0_dp
    lp(0) = 0.5_dp; lp(1) =  0.5_dp
    lm(0) = 0.5_dp; lm(1) = -0.5_dp

    ! g_+ = l_+ + ghat,  H1 ghat = k2 l_+  ->  (-D^2+mu1) ghat = -k2 l_+
    call helmholtz_rhs(n, -k2*lp, ft)
    call helm_band_solve(n, d%mu1, sd, st, bd, boff, ft, ghat)
    call shen_to_cheb(n, ghat, gpc)
    gpc = gpc + lp
    ! g_-
    call helmholtz_rhs(n, -k2*lm, ft)
    call helm_band_solve(n, d%mu1, sd, st, bd, boff, ft, ghat)
    call shen_to_cheb(n, ghat, gmc)
    gmc = gmc + lm

    ! v_+ : H2 v_+ = g_+  ->  (-D^2+mu2) v_+ = -g_+
    call helmholtz_rhs(n, -gpc, ft)
    call helm_band_solve(n, d%mu2, sd, st, bd, boff, ft, vsh)
    call shen_to_cheb(n, vsh, d%vp)
    ! v_-
    call helmholtz_rhs(n, -gmc, ft)
    call helm_band_solve(n, d%mu2, sd, st, bd, boff, ft, vsh)
    call shen_to_cheb(n, vsh, d%vm)

    ! influence matrix  C = [ Dv_-(-1)  Dv_+(-1) ; Dv_-(+1)  Dv_+(+1) ]
    call wall_derivs(n, d%vm, dvm_lo, dvm_hi)
    call wall_derivs(n, d%vp, dvp_lo, dvp_hi)
    det = dvm_lo*dvp_hi - dvp_lo*dvm_hi
    d%Cinv(1,1) =  dvp_hi/det;  d%Cinv(1,2) = -dvp_lo/det
    d%Cinv(2,1) = -dvm_hi/det;  d%Cinv(2,2) =  dvm_lo/det
  end subroutine vif_setup_band

  !---------------------------------------------------------------------------
  ! Solve for v (Chebyshev coeffs) given the Chebyshev-coeff RHS fcheb.
  !---------------------------------------------------------------------------
  subroutine vif_solve(n, d, fcheb, vcheb)
    integer(ip), intent(in)  :: n
    type(vif_t), intent(in)  :: d
    real(dp),    intent(in)  :: fcheb(0:n)
    real(dp),    intent(out) :: vcheb(0:n)
    real(dp) :: gsh(0:n-2), gpc(0:n), vsh(0:n-2), vpc(0:n)
    real(dp) :: dvp_lo, dvp_hi, s1, s2

    ! H1 g_p = F = -f/alpha   ->  helmholtz_solve(mu1, +f/alpha)
    call helmholtz_solve(n, d%mu1, fcheb/d%alpha, gsh)
    call shen_to_cheb(n, gsh, gpc)
    ! H2 v_p = g_p            ->  helmholtz_solve(mu2, -g_p)
    call helmholtz_solve(n, d%mu2, -gpc, vsh)
    call shen_to_cheb(n, vsh, vpc)
    ! boundary closure:  sigma = -Cinv [Dv_p(-1); Dv_p(+1)]
    call wall_derivs(n, vpc, dvp_lo, dvp_hi)
    s1 = -( d%Cinv(1,1)*dvp_lo + d%Cinv(1,2)*dvp_hi )
    s2 = -( d%Cinv(2,1)*dvp_lo + d%Cinv(2,2)*dvp_hi )
    ! v = v_p + s1 v_- + s2 v_+
    vcheb = vpc + s1*d%vm + s2*d%vp
  end subroutine vif_solve

  ! wall_derivs lives in yops_mod (yop_wall_deriv, use-renamed above):
  ! one implementation serves this module, par_pressure_mod and stats_mod.

end module vsolve_if_mod
