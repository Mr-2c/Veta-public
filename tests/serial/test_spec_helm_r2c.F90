!===============================================================================
! test_spec_helm_r2c.F90  --  [R2C] confirm the eta-solve spectral
! layer is EXTENT-AGNOSTIC under R2C: spec_helpers_mod (project_shen,
! shen_to_cheb_c, wall_derivs_c) and helm_band_gpu_mod (helm_band_solve_gpu)
! reproduce their full-nx results when called with nkx=nx/2+1 as the x-extent
! (and nkx-length kx), on the retained half kx in [0,nkx-1], to bit identity.
!
! All four routines are PER-MODE and local in (kx,kz): helm_band_kernel is one
! thread per (ix,iz) with mu=kx(ix)^2+kz(iz)^2+lambda and a y-column-local banded
! solve; project_shen/shen_to_cheb_c/wall_derivs_c loop local (ix,iz) with a
! y-recurrence. No ix+-1 / nx-ix / cross-x reduction (verified by inspection,
! BUG_GUIDE section 4). So the R2C path reuses the SAME source with nx->nkx;
! this test proves it, so increment 4b/4c can pass nkx with no change here.
!
! The routines are chained as in the real eta-solve:
!   project_shen (Cheb f -> Shen-test ftilde)  ->  helm_band_solve_gpu (banded
!   Helmholtz, ftilde -> Shen uhat)  ->  shen_to_cheb_c (Shen uhat -> Cheb a)
!   ->  wall_derivs_c (Cheb a -> wall derivatives), at full nx and at nkx.
!
! Checks (each: nkx result == full-nx result on kx<nkx, exact 0.0 expected since
! same code/same operands; PLUS a nonzero guard so an all-zero shared bug cannot
! false-pass -- BUG_GUIDE section 1, the legitimate-vs-masked 0.0 distinction).
!
! dp throughout (the wall-normal solve layer is dp, not wp). NaN/Inf-hardened.
!
! Build & run ON A GPU NODE:
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -c spec_helpers_mod.F90 helm_band_gpu_mod.F90
!   nvfortran -cuda test_spec_helm_r2c.F90 helm_band_gpu_mod.o spec_helpers_mod.o types.o \
!             -o test_spec_helm_r2c
!   ./test_spec_helm_r2c
!===============================================================================
program test_spec_helm_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,            only: dp, ip
  use spec_helpers_mod, only: project_shen, shen_to_cheb_c, wall_derivs_c
  use helm_band_gpu_mod, only: helm_band_solve_gpu
  use cudafor
  implicit none

  integer(ip), parameter :: n = 16, nx = 12, nz = 8
  integer(ip), parameter :: nkx = nx/2 + 1
  real(dp),    parameter :: lambda = 250.0_dp
  real(real64) :: tol
  logical :: all_ok = .true.
  integer :: istat
  integer(ip) :: ix, iz, k

  real(dp) :: kxf_h(nx), kzf_h(nz), kxr_h(nkx), re, im
  real(dp) :: sdiag_h(0:n-2), stail_h(0:n-2), bdiag_h(0:n-2), boff_h
  ! host result buffers (full-nx and nkx) for each stage
  complex(dp) :: ftf_h(0:n-2,nx,nz), uhf_h(0:n-2,nx,nz), achf_h(0:n,nx,nz)
  complex(dp) :: dlof_h(nx,nz), dhif_h(nx,nz)
  complex(dp) :: ftr_h(0:n-2,nkx,nz), uhr_h(0:n-2,nkx,nz), achr_h(0:n,nkx,nz)
  complex(dp) :: dlor_h(nkx,nz), dhir_h(nkx,nz)
  complex(dp) :: fc_h(0:n,nx,nz)        ! the shared Chebyshev-coeff input

  ! device, full nx
  real(dp),    device :: kxf(nx), kzf(nz), sd(0:n-2), st(0:n-2), bd(0:n-2)
  complex(dp), device :: fcf(0:n,nx,nz), ftf(0:n-2,nx,nz), uhf(0:n-2,nx,nz), achf(0:n,nx,nz)
  complex(dp), device :: dlof(nx,nz), dhif(nx,nz)
  ! device, nkx
  real(dp),    device :: kxr(nkx)
  complex(dp), device :: fcr(0:n,nkx,nz), ftr(0:n-2,nkx,nz), uhr(0:n-2,nkx,nz), achr(0:n,nkx,nz)
  complex(dp), device :: dlor(nkx,nz), dhir(nkx,nz)

  tol = 1.0e-12_real64

  ! wavenumbers; kxr = first nkx of kxf
  do ix = 1, nx;  kxf_h(ix) = real(ix-1, dp) * 0.6_dp;  end do
  do iz = 1, nz;  kzf_h(iz) = real(iz-1, dp) * 1.1_dp;  end do
  do ix = 1, nkx; kxr_h(ix) = kxf_h(ix);                end do

  ! a plausible banded-Helmholtz structure (arbitrary but nonsingular: diag
  ! dominant). The numbers need not be the true S/B -- the test checks nkx-vs-nx
  ! equivalence of the SOLVE, not its physical correctness (that is
  ! test_helm_band_gpu's job).
  do k = 0, n-2
     sdiag_h(k) = 2.0_dp + real(k, dp)
     stail_h(k) = 0.3_dp
     bdiag_h(k) = 1.0_dp + 0.1_dp*real(k, dp)
  end do
  boff_h = 0.05_dp

  ! shared random Chebyshev-coeff input
  do iz = 1, nz; do ix = 1, nx; do k = 0, n
     call random_number(re); call random_number(im)
     fc_h(k,ix,iz) = cmplx(re-0.5_dp, im-0.5_dp, dp)
  end do; end do; end do

  ! ===== full-nx chain =========================================================
  kxf = kxf_h; kzf = kzf_h; sd = sdiag_h; st = stail_h; bd = bdiag_h; fcf = fc_h
  call project_shen(n, nx, nz, fcf, ftf)
  uhf = ftf                                   ! helm_band_solve_gpu is in-place on uhat
  call helm_band_solve_gpu(n, nx, nz, lambda, kxf, kzf, sd, st, bd, boff_h, &
                           ftf, uhf)
  call shen_to_cheb_c(n, nx, nz, uhf, achf)
  call wall_derivs_c(n, nx, nz, achf, dlof, dhif)
  istat = cudaDeviceSynchronize()
  ftf_h = ftf; uhf_h = uhf; achf_h = achf; dlof_h = dlof; dhif_h = dhif

  ! ===== nkx chain on the first nkx x-columns ==================================
  kxr = kxr_h
  do iz = 1, nz; do ix = 1, nkx; do k = 0, n
     fcr(k,ix,iz) = fc_h(k,ix,iz)
  end do; end do; end do
  call project_shen(n, nkx, nz, fcr, ftr)
  uhr = ftr
  call helm_band_solve_gpu(n, nkx, nz, lambda, kxr, kzf, sd, st, bd, boff_h, &
                           ftr, uhr)
  call shen_to_cheb_c(n, nkx, nz, uhr, achr)
  call wall_derivs_c(n, nkx, nz, achr, dlor, dhir)
  istat = cudaDeviceSynchronize()
  ftr_h = ftr; uhr_h = uhr; achr_h = achr; dlor_h = dlor; dhir_h = dhir

  ! ===== compare on the retained half + nonzero guards =========================
  call cmp3('project_shen      (nkx vs nx)', n-2, ftf_h,  ftr_h,  tol)
  call cmp3('helm_band_solve   (nkx vs nx)', n-2, uhf_h,  uhr_h,  tol)
  call cmp3('shen_to_cheb_c    (nkx vs nx)', n,   achf_h, achr_h, tol)
  call cmp2('wall_derivs Du(-1)(nkx vs nx)', dlof_h, dlor_h, tol)
  call cmp2('wall_derivs Du(+1)(nkx vs nx)', dhif_h, dhir_h, tol)

  call nz3('  helm_band solution nonzero', n-2, uhf_h)
  call nz3('  shen_to_cheb output nonzero', n, achf_h)
  call nz2('  wall_derivs nonzero',         dhif_h)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine cmp3(label, mtop, full, half, tol)
    character(*), intent(in) :: label
    integer(ip),  intent(in) :: mtop
    complex(dp),  intent(in) :: full(0:mtop,nx,nz), half(0:mtop,nkx,nz)
    real(real64), intent(in) :: tol
    real(real64) :: dm, d
    integer(ip)  :: i, j, kk
    dm = 0.0_real64
    do kk=1,nz; do j=1,nkx; do i=0,mtop
       d = abs(full(i,j,kk) - half(i,j,kk))
       if (.not. (d == d)) d = huge(1.0_real64)
       dm = max(dm, d)
    end do; end do; end do
    if (dm /= dm .or. dm > tol) then
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|diff|=', dm;  all_ok = .false.
    else
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|diff|=', dm
    end if
  end subroutine cmp3

  subroutine cmp2(label, full, half, tol)
    character(*), intent(in) :: label
    complex(dp),  intent(in) :: full(nx,nz), half(nkx,nz)
    real(real64), intent(in) :: tol
    real(real64) :: dm, d
    integer(ip)  :: j, kk
    dm = 0.0_real64
    do kk=1,nz; do j=1,nkx
       d = abs(full(j,kk) - half(j,kk))
       if (.not. (d == d)) d = huge(1.0_real64)
       dm = max(dm, d)
    end do; end do
    if (dm /= dm .or. dm > tol) then
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|diff|=', dm;  all_ok = .false.
    else
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|diff|=', dm
    end if
  end subroutine cmp2

  subroutine nz3(label, mtop, f)
    character(*), intent(in) :: label
    integer(ip),  intent(in) :: mtop
    complex(dp),  intent(in) :: f(0:mtop,nx,nz)
    real(real64) :: mx, a
    integer(ip)  :: i, j, kk
    mx = 0.0_real64
    do kk=1,nz; do j=1,nkx; do i=0,mtop
       a = abs(f(i,j,kk));  if (.not. (a == a)) a = huge(1.0_real64)
       mx = max(mx, a)
    end do; end do; end do
    if (mx > 1.0e-6_real64 .and. mx < huge(1.0_real64)) then
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|f|=', mx
    else
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|f|=', mx;  all_ok = .false.
    end if
  end subroutine nz3

  subroutine nz2(label, f)
    character(*), intent(in) :: label
    complex(dp),  intent(in) :: f(nx,nz)
    real(real64) :: mx, a
    integer(ip)  :: j, kk
    mx = 0.0_real64
    do kk=1,nz; do j=1,nkx
       a = abs(f(j,kk));  if (.not. (a == a)) a = huge(1.0_real64)
       mx = max(mx, a)
    end do; end do
    if (mx > 1.0e-6_real64 .and. mx < huge(1.0_real64)) then
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|f|=', mx
    else
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|f|=', mx;  all_ok = .false.
    end if
  end subroutine nz2

end program test_spec_helm_r2c
