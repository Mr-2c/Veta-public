!===============================================================================
! test_spectral3d_da.F90  --  GPU validation of the DEALIASED 3D assembly.
!
! Build & run (ON A GPU NODE):
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -cudalib=cufft -c transforms_mod.F90
!   nvfortran -cuda -cudalib=cufft -c cheb_fft_mod.F90
!   nvfortran -cuda -cudalib=cufft -c fft3d_mod.F90
!   nvfortran -cuda -cudalib=cufft -c spectral3d_mod.F90
!   nvfortran -cuda -cudalib=cufft test_spectral3d_da.F90 \
!     spectral3d_mod.o fft3d_mod.o cheb_fft_mod.o transforms_mod.o types.o \
!     -o test_spectral3d_da
!   ./test_spectral3d_da
!
! Test 1 -- dealiased round-trip: to_spectral_da(to_physical_da(A)) = nxd*nzd*A
!           (exercises pad/truncate in both x and z; band-limited random field).
! Test 2 -- 3D product aliasing: u = e^{i2x}+e^{-ix}, v = e^{i3x} (constant in
!           y and z). u*v = e^{i5x}+e^{i2x}; the +5 mode would alias to freq -3
!           on the coarse grid. Dealiased, the retained spectrum keeps +2 and
!           leaves the -3 slot clean:  |A(-3)| / |A(+2)| ~ 0.
!===============================================================================

program test_spectral3d_da
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,          only: tp, ip
  use spectral3d_mod, only: s3d_plans, s3d_setup_da, s3d_destroy, &
                            to_physical_da, to_spectral_da
  use cudafor
  implicit none

  integer(ip), parameter :: ny = 17, nx = 8, nz = 6
  integer(ip), parameter :: nxd = 3*nx/2, nzd = 3*nz/2      ! 12, 9
  real(real64) :: tol, ratio
  logical :: all_ok = .true.
  integer :: istat
  integer(ip) :: iy, ix, iz

  type(s3d_plans) :: pl
  complex(tp) :: Ah(ny,nx,nz), Rh(ny,nx,nz), Awh(ny,nx,nz)
  real(tp)    :: re(ny,nx,nz), im(ny,nx,nz)
  complex(tp), device :: A(ny,nx,nz), Au(ny,nx,nz), Av(ny,nx,nz), Aw(ny,nx,nz)
  complex(tp), device :: P(ny,nxd,nzd), Pv(ny,nxd,nzd), W(ny,nxd,nzd)

  tol = merge(1.0e-9_real64, 1.0e-4_real64, tp == real64)
  call s3d_setup_da(pl, ny, nx, nz)

  !-- Test 1: dealiased round-trip ------------------------------------------
  call random_number(re);  call random_number(im)
  Ah = cmplx(re, im, tp)
  A = Ah
  call to_physical_da(pl, ny, nx, nz, nxd, nzd, A, P, W)
  call to_spectral_da(pl, ny, nx, nz, nxd, nzd, P, W, A)
  istat = cudaDeviceSynchronize()
  Rh = A
  Rh = Rh / real(nxd*nzd, tp)
  call check('dealiased 3D round-trip', real(maxval(abs(Rh - Ah)), real64), tol)

  !-- Test 2: 3D product aliasing -------------------------------------------
  Ah = (0.0_tp,0.0_tp)
  Ah(1,3,1) = (1.0_tp,0.0_tp)        ! cheb0, x=+2, z=0
  Ah(1,nx,1) = (1.0_tp,0.0_tp)       ! cheb0, x=-1, z=0
  Au = Ah
  Ah = (0.0_tp,0.0_tp)
  Ah(1,4,1) = (1.0_tp,0.0_tp)        ! cheb0, x=+3, z=0
  Av = Ah

  call to_physical_da(pl, ny, nx, nz, nxd, nzd, Au, P,  W)   ! u -> P
  call to_physical_da(pl, ny, nx, nz, nxd, nzd, Av, Pv, W)   ! v -> Pv
  !$cuf kernel do(3) <<<*,*>>>
  do iz = 1, nzd
     do ix = 1, nxd
        do iy = 1, ny
           P(iy,ix,iz) = P(iy,ix,iz) * Pv(iy,ix,iz)
        end do
     end do
  end do
  call to_spectral_da(pl, ny, nx, nz, nxd, nzd, P, W, Aw)
  istat = cudaDeviceSynchronize()
  Awh = Aw
  ! +2 retained at (cheb0, x-index 3), alias target -3 at (cheb0, x-index 6)
  ratio = real(abs(Awh(1,6,1)) / abs(Awh(1,3,1)), real64)
  call check('3D product: x-alias slot clean', ratio, tol)

  call s3d_destroy(pl)

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

end program test_spectral3d_da
