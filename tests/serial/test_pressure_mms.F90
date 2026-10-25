!===============================================================================
! test_pressure_mms.F90  --  manufactured-solution test for the diagnostic
! Neumann-Poisson pressure solve (par_pressure_mod), ASSERTING.
!
! Pick a smooth exact MODIFIED pressure Pi_exact(y) (complex: independent real /
! imag profiles, to exercise the complex solver), form the exact data per mode
!     R    = (D^2 - k^2) Pi_exact          (via the validated cheb_d2)
!     g_+- = D Pi_exact(+-1)               (wall first-derivative from coeffs)
! analytically, solve with par_pressure_mod, and require recovery of Pi_exact:
!   * k^2 > 0 : Pi recovered fully (Neumann BVP is non-singular)            -> all coeffs
!   * k^2 = 0 : recovered up to the pure-Neumann gauge constant (a_0 free)  -> coeffs k>=1
! Both reach spectral accuracy (~1e-12 by n=16 for these smooth profiles;
! src/models/work_pressure_model.py "model 2" is the numpy twin). NaN/Inf hard-fail.
!
! The (kx,kz) grid includes the (0,0) mode (k^2=0), so the gauge branch is
! exercised every run. This validates the ALGORITHM (closure + k=0 gauge); the
! lift/helm building blocks are already validated (test_helm_band, test_vif_gpu).
!
! Build (run_all_tests.sh registers it):
!   nvfortran -c types.F90 cheb_shen.F90 helm_band_mod.F90
!   nvfortran -cuda -c helm_band_gpu_mod.F90 spec_helpers_mod.F90 par_pressure_mod.F90
!   nvfortran -cuda -c test_pressure_mms.F90
!   nvfortran -cuda -o test_pressure_mms test_pressure_mms.o par_pressure_mod.o \
!             spec_helpers_mod.o helm_band_gpu_mod.o helm_band_mod.o cheb_shen.o types.o
!   ./test_pressure_mms
!===============================================================================
program test_pressure_mms
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo root
!  (sources in ../src, parallel solve in ../parallel, output to ../build).
  use kinds,           only: dp, ip
  use cheb_shen,       only: glc_nodes, cheb_forward, cheb_d2
  use helm_band_mod,   only: helm_band_setup
  use par_pressure_mod, only: pres_gpu_setup_host, pres_solve_gpu
  use cudafor
  implicit none

  integer(ip) :: nv(3) = [16, 32, 48]
  integer(ip) :: xv(3) = [12, 16, 10]
  integer(ip) :: zv(3) = [ 8, 12, 10]
  integer(ip) :: ic
  logical :: all_pass

  all_pass = .true.
  do ic = 1, size(nv)
     write(*,'(/,a,i0,a,i0,a,i0,a)') '================ n=',nv(ic),'  nx=',xv(ic),'  nz=',zv(ic),' ================'
     call run_case(nv(ic), xv(ic), zv(ic), all_pass)
  end do
  if (all_pass) then
     print *, 'ALL TESTS PASSED'
  else
     print *, 'SOME TESTS FAILED'
  end if

contains

  subroutine run_case(n, nx, nz, all_pass)
    integer(ip), intent(in)    :: n, nx, nz
    logical,     intent(inout) :: all_pass
    real(dp) :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
    real(dp) :: kxh(nx), kzh(nz), x(0:n)
    real(dp) :: fre(0:n), fim(0:n), pre(0:n), pim(0:n)   ! Pi_exact values / coeffs
    real(dp) :: d2re(0:n), d2im(0:n)                     ! cheb_d2 of Pi_exact (re/im)
    real(dp) :: gpre, gpim, gmre, gmim, k2, err, errk0, den, dd
    complex(dp), allocatable :: Pex(:,:,:), Rc(:,:,:), gp(:,:), gm(:,:), Pg(:,:,:)
    integer(ip) :: ix, iz, k, kk, j, s, istat
    logical :: bad
    real(dp),    device, allocatable :: kx_d(:), kz_d(:), sd(:), st(:), bd(:)
    real(dp),    device, allocatable :: pp_d(:,:,:), pm_d(:,:,:), Cinv_d(:,:,:,:)
    complex(dp), device, allocatable :: Rc_d(:,:,:), gp_d(:,:), gm_d(:,:), Pi_d(:,:,:)

    ! signed wavenumbers (cuFFT order), include (0,0)
    do ix = 1, nx
       kk = ix-1; if (kk > nx/2) kk = kk - nx; kxh(ix) = real(kk, dp)
    end do
    do iz = 1, nz
       kk = iz-1; if (kk > nz/2) kk = kk - nz; kzh(iz) = real(kk, dp)
    end do
    call helm_band_setup(n, sdiag, stail, bdiag, boff)
    call glc_nodes(n, x)

    ! manufactured Pi_exact: smooth, distinct real/imag profiles -> Cheb coeffs
    do k = 0, n
       fre(k) = cos(2.0_dp*x(k)) + 0.3_dp*x(k)**3
       fim(k) = sin(1.5_dp*x(k)) - 0.4_dp*x(k)**2 + 0.1_dp
    end do
    call cheb_forward(n, fre, pre)
    call cheb_forward(n, fim, pim)
    call cheb_d2(n, pre, d2re)
    call cheb_d2(n, pim, d2im)

    ! wall derivatives of Pi_exact (same for every mode): D(+1)=sum j^2 a_j,
    ! D(-1)=sum (-1)^(j-1) j^2 a_j
    gpre = 0.0_dp; gpim = 0.0_dp; gmre = 0.0_dp; gmim = 0.0_dp
    do j = 1, n
       gpre = gpre + pre(j)*real(j*j,dp)
       gpim = gpim + pim(j)*real(j*j,dp)
       if (mod(j,2) == 1) then; s = 1; else; s = -1; end if
       gmre = gmre + real(s,dp)*pre(j)*real(j*j,dp)
       gmim = gmim + real(s,dp)*pim(j)*real(j*j,dp)
    end do

    allocate(Pex(0:n,nx,nz), Rc(0:n,nx,nz), gp(nx,nz), gm(nx,nz), Pg(0:n,nx,nz))
    do iz = 1, nz; do ix = 1, nx
       k2 = kxh(ix)**2 + kzh(iz)**2
       do k = 0, n
          Pex(k,ix,iz) = cmplx(pre(k), pim(k), dp)
          Rc(k,ix,iz)  = cmplx(d2re(k) - k2*pre(k), d2im(k) - k2*pim(k), dp)
       end do
       gp(ix,iz) = cmplx(gpre, gpim, dp)
       gm(ix,iz) = cmplx(gmre, gmim, dp)
    end do; end do

    ! GPU solve
    allocate(kx_d(nx), kz_d(nz), sd(0:n-2), st(0:n-2), bd(0:n-2))
    allocate(pp_d(0:n,nx,nz), pm_d(0:n,nx,nz), Cinv_d(2,2,nx,nz))
    allocate(Rc_d(0:n,nx,nz), gp_d(nx,nz), gm_d(nx,nz), Pi_d(0:n,nx,nz))
    kx_d = kxh; kz_d = kzh; sd = sdiag; st = stail; bd = bdiag
    call pres_gpu_setup_host(n, nx, nz, kxh, kzh, pp_d, pm_d, Cinv_d)
    Rc_d = Rc; gp_d = gp; gm_d = gm
    call pres_solve_gpu(n, nx, nz, kx_d, kz_d, sd, st, bd, boff, &
                        pp_d, pm_d, Cinv_d, Rc_d, gp_d, gm_d, Pi_d)
    istat = cudaDeviceSynchronize()
    Pg = Pi_d

    ! compare: full coeffs for k^2>0; coeffs k>=1 for the k=0 gauge mode
    err = 0.0_dp; errk0 = 0.0_dp; den = 0.0_dp; bad = .false.
    do iz = 1, nz; do ix = 1, nx
       k2 = kxh(ix)**2 + kzh(iz)**2
       do k = 0, n
          dd = abs(Pg(k,ix,iz) - Pex(k,ix,iz))
          if (.not. (dd == dd)) bad = .true.
          if (abs(real(Pg(k,ix,iz))) > huge(1.0_dp)) bad = .true.
          den = max(den, abs(Pex(k,ix,iz)))
          if (k2 == 0.0_dp) then
             if (k >= 1) errk0 = max(errk0, dd)        ! gauge: a_0 free
          else
             err = max(err, dd)
          end if
       end do
    end do; end do
    if (den == 0.0_dp) den = 1.0_dp

    call report(n, merge(1.0e30_dp, err/den,   bad), merge(1.0e30_dp, errk0/den, bad), all_pass)

    deallocate(Pex, Rc, gp, gm, Pg, kx_d, kz_d, sd, st, bd, pp_d, pm_d, Cinv_d, &
               Rc_d, gp_d, gm_d, Pi_d)
  end subroutine run_case

  subroutine report(n, e, ek0, all_pass)
    integer(ip), intent(in)    :: n
    real(dp),    intent(in)    :: e, ek0
    logical,     intent(inout) :: all_pass
    real(dp), parameter :: tol = 1.0e-8_dp
    character(6) :: tag
    if (e <= tol .and. ek0 <= tol) then; tag = '[PASS]'; else; tag = '[FAIL]'; all_pass = .false.; end if
    write(*,'(a,2x,a,i0,2x,a,es12.4,2x,a,es12.4,2x,a,es9.2)') &
         tag, 'n=', n, 'rel err(k^2>0)=', e, 'rel err(k=0,gauge)=', ek0, 'tol=', tol
  end subroutine report

end program test_pressure_mms
