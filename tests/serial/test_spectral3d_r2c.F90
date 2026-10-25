!===============================================================================
! test_spectral3d_r2c.F90  --  [R2C] validate the REAL-x
! non-dealiased transforms (to_physical_r2c / to_spectral_r2c) against the
! trusted C2C path, at machine precision.
!
! Two checks:
!   1. ROUND-TRIP: real physical P -> spectral S -> physical, /(nx*nz) = P.
!   2. C2C CROSS-CHECK (the real test): take a random REAL physical field; form
!      its spectrum two ways --
!         (a) the validated C2C path: load P as complex (imag=0), to_spectral;
!         (b) the new R2C path: to_spectral_r2c.
!      Require S_r2c(:, 0:nkx-1, :) == S_c2c(:, 0:nkx-1, :) to round-off. (The
!      C2C negative-kx half is the Hermitian conjugate of the positive half and
!      is exactly what R2C omits, so equality on the retained half is the precise
!      correctness statement -- verified numerically before coding.)
!
! NaN/Inf-hardened. Build/run ON A GPU NODE:
!   nvfortran -cuda -c types.F90 kinds is in types
!   nvfortran -cuda -cudalib=cufft -c transforms_mod.F90 cheb_fft_mod.F90 fft3d_mod.F90 spectral3d_mod.F90
!   nvfortran -cuda -cudalib=cufft test_spectral3d_r2c.F90 \
!             spectral3d_mod.o fft3d_mod.o cheb_fft_mod.o transforms_mod.o types.o -o test_spectral3d_r2c
!   ./test_spectral3d_r2c
!===============================================================================
program test_spectral3d_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,          only: tp, ip
  use spectral3d_mod, only: s3d_plans, s3d_setup, s3d_destroy, to_spectral, &
                            s3d_setup_r2c, s3d_destroy_r2c, to_physical_r2c, to_spectral_r2c
  use cudafor
  implicit none

  integer(ip), parameter :: ny = 5, nx = 8, nz = 6     ! ny = Cheb coeff count; nx even
  integer(ip), parameter :: nkx = nx/2 + 1
  real(real64) :: tol
  logical :: all_ok = .true.
  integer :: istat

  real(tp)    :: Ph(ny,nx,nz), Rh(ny,nx,nz), re(ny,nx,nz)
  complex(tp) :: Sh(ny,nkx,nz), Cc_h(ny,nx,nz)
  real(tp),    device :: Pd(ny,nx,nz), Rd(ny,nx,nz)
  complex(tp), device :: Sd(ny,nkx,nz)
  complex(tp), device :: Ac(ny,nx,nz), workc(ny,nx,nz)
  type(s3d_plans) :: p, pr

  tol = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)

  call random_number(re)
  Ph = re - 0.5_tp

  call s3d_setup_r2c(pr, ny, nx, nz)

  ! ===== Test 1: R2C round-trip identity =======================================
  Pd = Ph
  call to_spectral_r2c(pr, ny, nx, nz, Pd, Sd)         ! P -> S (scaled by nx*nz)
  call to_physical_r2c(pr, ny, nx, nz, Sd, Rd)         ! S -> P (consumes Sd)
  istat = cudaDeviceSynchronize()
  Rh = Rd
  call report_real('R2C round-trip = (nx*nz)*identity', Ph, Rh/real(nx*nz, tp), tol)

  ! ===== Test 2: R2C spectrum == C2C spectrum on the retained half =============
  ! (a) C2C path on the same real field loaded as complex
  Ac = cmplx(Ph, 0.0_tp, tp)
  call s3d_setup(p, ny, nx, nz)
  call to_spectral(p, ny, nx, nz, Ac, workc)           ! full (ny,nx,nz) spectrum
  istat = cudaDeviceSynchronize()
  Cc_h = Ac
  ! (b) R2C path
  Pd = Ph
  call to_spectral_r2c(pr, ny, nx, nz, Pd, Sd)
  istat = cudaDeviceSynchronize()
  Sh = Sd
  call report_half('R2C spectrum == C2C on kx in [0,nkx-1]', Cc_h, Sh, tol)

  call s3d_destroy(p)
  call s3d_destroy_r2c(pr)
  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine report_real(label, a, b, tol)
    character(*), intent(in) :: label
    real(tp),     intent(in) :: a(ny,nx,nz), b(ny,nx,nz)
    real(real64), intent(in) :: tol
    real(real64) :: dm, d
    integer(ip)  :: i, j, k
    dm = 0.0_real64
    do k=1,nz; do j=1,nx; do i=1,ny
       d = abs(real(a(i,j,k),real64) - real(b(i,j,k),real64))
       if (.not. (d == d)) d = huge(1.0_real64)
       dm = max(dm, d)
    end do; end do; end do
    if (dm /= dm .or. dm > tol) then
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|diff|=', dm;  all_ok = .false.
    else
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|diff|=', dm
    end if
  end subroutine report_real

  subroutine report_half(label, Cfull, Sr2c, tol)
    character(*), intent(in) :: label
    complex(tp),  intent(in) :: Cfull(ny,nx,nz), Sr2c(ny,nkx,nz)
    real(real64), intent(in) :: tol
    real(real64) :: dm, d
    integer(ip)  :: i, k, kk
    dm = 0.0_real64
    do k=1,nz; do kk=1,nkx; do i=1,ny
       d = abs(Cfull(i,kk,k) - Sr2c(i,kk,k))
       if (.not. (d == d)) d = huge(1.0_real64)
       dm = max(dm, d)
    end do; end do; end do
    if (dm /= dm .or. dm > tol) then
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|diff|=', dm;  all_ok = .false.
    else
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|diff|=', dm
    end if
  end subroutine report_half

end program test_spectral3d_r2c
