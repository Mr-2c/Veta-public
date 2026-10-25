!===============================================================================
! test_spectral3d.F90  --  GPU validation of the full 3D spectral<->physical
!                          assembly (chaining test, no dealiasing yet).
!
! Build & run (ON A GPU NODE):
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -cudalib=cufft -c transforms_mod.F90
!   nvfortran -cuda -cudalib=cufft -c cheb_fft_mod.F90
!   nvfortran -cuda -cudalib=cufft -c fft3d_mod.F90
!   nvfortran -cuda -cudalib=cufft -c spectral3d_mod.F90
!   nvfortran -cuda -cudalib=cufft test_spectral3d.F90 \
!     spectral3d_mod.o fft3d_mod.o cheb_fft_mod.o transforms_mod.o types.o \
!     -o test_spectral3d
!   ./test_spectral3d
!
! to_spectral(to_physical(A)) = (nx*nz) * A   (the two Fourier round-trips;
! the Chebyshev pair is self-inverse). Divide by nx*nz and compare to A.
!===============================================================================

program test_spectral3d
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,          only: tp, ip
  use spectral3d_mod, only: s3d_plans, s3d_setup, s3d_destroy, to_physical, to_spectral
  use cudafor
  implicit none

  integer(ip), parameter :: ny = 17, nx = 8, nz = 6     ! degree 16 in y
  real(real64) :: tol
  logical :: all_ok = .true.
  integer :: istat

  type(s3d_plans) :: p
  complex(tp) :: Ah(ny,nx,nz), Rh(ny,nx,nz)
  real(tp)    :: re(ny,nx,nz), im(ny,nx,nz)
  complex(tp), device :: A(ny,nx,nz), work(ny,nx,nz)

  tol = merge(1.0e-9_real64, 1.0e-4_real64, tp == real64)

  call random_number(re);  call random_number(im)
  Ah = cmplx(re, im, tp)

  call s3d_setup(p, ny, nx, nz)

  A = Ah
  call to_physical(p, ny, nx, nz, A, work)
  call to_spectral(p, ny, nx, nz, A, work)
  istat = cudaDeviceSynchronize()
  Rh = A
  Rh = Rh / real(nx*nz, tp)

  call s3d_destroy(p)

  call check('3D spectral round-trip', real(maxval(abs(Rh - Ah)), real64), tol)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine check(name, e, tolerance)
    character(*), intent(in) :: name
    real(real64), intent(in) :: e, tolerance
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', e <= tolerance)
    print '(a,a,a,es12.4,a,es10.2,a)', '[',tag,'] '//name//'  err=', e, '  tol=', tolerance
    if (e > tolerance) all_ok = .false.
  end subroutine check

end program test_spectral3d
