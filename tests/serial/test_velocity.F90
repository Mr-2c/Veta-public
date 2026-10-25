!===============================================================================
! test_velocity.F90  --  GPU validation of velocity/vorticity reconstruction.
!
! Build & run (ON A GPU NODE):
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -c cheb_shen.F90
!   nvfortran -cuda -c cheb_ops_mod.F90
!   nvfortran -cuda -c velocity_mod.F90
!   nvfortran -cuda test_velocity.F90 \
!     velocity_mod.o cheb_ops_mod.o cheb_shen.o types.o -o test_velocity
!   ./test_velocity
!
! v̂, η̂ are arbitrary (random) finite Chebyshev series, so all y-derivatives are
! exact and the following identities must hold to machine precision:
!   1. divergence-free velocity:  i kx û + Dv̂ + i kz ŵ = 0
!   2. eta round-trip:            i kz û - i kx ŵ = η̂
!   3. divergence-free vorticity: i kx ω̂_x + Dη̂ + i kz ω̂_z = 0
! The (0,0) mean column has v̂ = η̂ = 0 and û,ŵ = u0,w0; all three identities
! hold trivially there (every i*k factor vanishes and Dv̂ = 0).
!===============================================================================

program test_velocity
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,        only: tp, wp, ip
  use cheb_ops_mod, only: cheb_deriv_c
  use velocity_mod, only: recover_uw, compute_vorticity
  use cudafor
  implicit none

  integer(ip), parameter :: ny = 9, nx = 8, nz = 6
  real(real64) :: tol
  logical :: all_ok = .true.
  integer :: istat
  integer(ip) :: iy, ix, iz, m

  real(wp) :: kxh(nx), kzh(nz)                          ! wavenumbers feed wp dummies
  real(wp) :: re(ny,nx,nz), im(ny,nx,nz)
  real(wp) :: ru(ny), iu(ny), rw(ny), iw(ny)
  complex(wp) :: vh(ny,nx,nz), eh(ny,nx,nz)
  complex(tp) :: uh(ny,nx,nz), wh(ny,nx,nz), dvh(ny,nx,nz)   ! uhat,what,dv are tp
  complex(tp) :: oxh(ny,nx,nz), oyh(ny,nx,nz), ozh(ny,nx,nz) ! vorticity is tp
  complex(wp) :: deh(ny,nx,nz)                          ! cheb_deriv_c (wp) output
  complex(wp), parameter :: ii = (0.0_wp,1.0_wp)

  real(wp),    device :: kxd(nx), kzd(nz)
  complex(wp), device :: vd(ny,nx,nz), ed(ny,nx,nz)         ! vhat,ehat are wp
  complex(tp), device :: ud(ny,nx,nz), wd(ny,nx,nz)         ! uhat,what are tp
  complex(tp), device :: dvd(ny,nx,nz), oxd(ny,nx,nz), oyd(ny,nx,nz), ozd(ny,nx,nz)
  complex(wp), device :: ded(ny,nx,nz)                     ! cheb_deriv_c (wp) output
  complex(tp), device :: du(ny,nx,nz), dw(ny,nx,nz)        ! compute_vorticity tp scratch
  complex(wp), device :: u0d(ny), w0d(ny)                  ! u0,w0 are wp

  real(real64) :: e1, e2, e3

  tol = merge(1.0e-9_real64, 1.0e-4_real64, tp == real64)

  ! cuFFT-order integer wavenumbers (Lx=Lz=2*pi)
  do ix = 1, nx
     m = ix-1;  if (m > nx/2) m = m - nx;  kxh(ix) = real(m,wp)
  end do
  do iz = 1, nz
     m = iz-1;  if (m > nz/2) m = m - nz;  kzh(iz) = real(m,wp)
  end do
  kxd = kxh;  kzd = kzh

  ! random finite Chebyshev series for v̂, η̂; zero the (0,0) column
  call random_number(re);  call random_number(im);  vh = cmplx(re,im,wp)
  call random_number(re);  call random_number(im);  eh = cmplx(re,im,wp)
  vh(:,1,1) = (0.0_wp,0.0_wp);  eh(:,1,1) = (0.0_wp,0.0_wp)
  call random_number(ru); call random_number(iu); call random_number(rw); call random_number(iw)

  vd = vh;  ed = eh;  u0d = cmplx(ru,iu,wp);  w0d = cmplx(rw,iw,wp)

  call recover_uw(ny, nx, nz, kxd, kzd, vd, ed, u0d, w0d, ud, wd, dvd)
  call compute_vorticity(ny, nx, nz, kxd, kzd, vd, ed, ud, wd, oxd, oyd, ozd, du, dw)
  call cheb_deriv_c(ny, nx*nz, ed, ded)               ! Dη̂ for the vorticity divergence
  istat = cudaDeviceSynchronize()

  uh = ud;  wh = wd;  dvh = dvd
  oxh = oxd;  oyh = oyd;  ozh = ozd;  deh = ded

  e1 = 0.0_real64;  e2 = 0.0_real64;  e3 = 0.0_real64
  do iz = 1, nz
     do ix = 1, nx
        do iy = 1, ny
           ! 1. divergence of velocity
           e1 = max(e1, abs(cmplx( ii*kxh(ix)*uh(iy,ix,iz) + dvh(iy,ix,iz) &
                                 + ii*kzh(iz)*wh(iy,ix,iz), kind=real64)))
           ! 2. eta round-trip
           e2 = max(e2, abs(cmplx( ii*kzh(iz)*uh(iy,ix,iz) - ii*kxh(ix)*wh(iy,ix,iz) &
                                 - eh(iy,ix,iz), kind=real64)))
           ! 3. divergence of vorticity
           e3 = max(e3, abs(cmplx( ii*kxh(ix)*oxh(iy,ix,iz) + deh(iy,ix,iz) &
                                 + ii*kzh(iz)*ozh(iy,ix,iz), kind=real64)))
        end do
     end do
  end do

  call check('recovered velocity is divergence-free', e1, tol)
  call check('eta of recovered field matches input',  e2, tol)
  call check('vorticity is divergence-free',           e3, tol)

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

end program test_velocity
