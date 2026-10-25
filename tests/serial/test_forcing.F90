!===============================================================================
! test_forcing.F90  --  GPU validation of the v/eta forcing assembly.
!
! Build & run (ON A GPU NODE):
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -c cheb_shen.F90
!   nvfortran -cuda -c cheb_ops_mod.F90
!   nvfortran -cuda -c forcing_mod.F90
!   nvfortran -cuda test_forcing.F90 \
!     forcing_mod.o cheb_ops_mod.o cheb_shen.o types.o -o test_forcing
!   ./test_forcing
!
! Test 1 -- gradient annihilation: H = grad(phi), i.e. Ĥ_x=i kx phî,
!           Ĥ_y=D phî, Ĥ_z=i kz phî. Then ĥ_v = ĥ_eta = 0 (machine precision):
!           the discrete pressure-elimination identity.
! Test 2 -- reference match: for random H, the GPU forcings equal an independent
!           host computation that uses the CPU cheb_deriv for the y-derivative.
!===============================================================================

program test_forcing
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,        only: tp, wp, ip, dp
  use cheb_shen,    only: cheb_deriv
  use cheb_ops_mod, only: cheb_deriv_c_tp
  use forcing_mod,  only: assemble_hv_heta
  use cudafor
  implicit none

  integer(ip), parameter :: ny = 9, nx = 8, nz = 6, n = ny-1
  real(real64) :: tol
  logical :: all_ok = .true.
  integer :: istat
  integer(ip) :: iy, ix, iz, m

  real(wp) :: kxh(nx), kzh(nz)                          ! wavenumbers feed wp dummies
  real(wp) :: re(ny,nx,nz), im(ny,nx,nz)
  complex(tp) :: phi(ny,nx,nz), Hxh(ny,nx,nz), Hyh(ny,nx,nz), Hzh(ny,nx,nz)   ! Hx,Hy,Hz tp
  complex(wp) :: hvh(ny,nx,nz), heh(ny,nx,nz)          ! hv,heta wp
  complex(wp), parameter :: ii = (0.0_wp,1.0_wp)

  real(wp),    device :: kxd(nx), kzd(nz)
  complex(tp), device :: Hxd(ny,nx,nz), Hyd(ny,nx,nz), Hzd(ny,nx,nz)
  complex(wp), device :: hvd(ny,nx,nz), hed(ny,nx,nz)               ! hv,heta wp
  complex(tp), device :: gd(ny,nx,nz), phid(ny,nx,nz)              ! g scratch / phi are tp
  complex(tp), device :: dgd(ny,nx,nz)             ! D(g) scratch (tp) for assemble_hv_heta

  tol = merge(1.0e-9_real64, 1.0e-4_real64, tp == real64)

  do ix = 1, nx
     m = ix-1;  if (m > nx/2) m = m - nx;  kxh(ix) = real(m,wp)
  end do
  do iz = 1, nz
     m = iz-1;  if (m > nz/2) m = m - nz;  kzh(iz) = real(m,wp)
  end do
  kxd = kxh;  kzd = kzh

  !-- Test 1: H = grad(phi) -> zero forcing -------------------------------
  call random_number(re);  call random_number(im);  phi = cmplx(re,im,wp)
  phid = phi
  call cheb_deriv_c_tp(ny, nx*nz, phid, Hyd)         ! Ĥ_y = D phî  (tp tier)
  do iz = 1, nz
     do ix = 1, nx
        do iy = 1, ny
           Hxh(iy,ix,iz) = ii*kxh(ix)*phi(iy,ix,iz)  ! Ĥ_x = i kx phî
           Hzh(iy,ix,iz) = ii*kzh(iz)*phi(iy,ix,iz)  ! Ĥ_z = i kz phî
        end do
     end do
  end do
  Hxd = Hxh;  Hzd = Hzh
  call assemble_hv_heta(ny, nx, nz, kxd, kzd, Hxd, Hyd, Hzd, hvd, hed, gd, dgd)
  istat = cudaDeviceSynchronize()
  hvh = hvd;  heh = hed
  call check('gradient H -> hv = 0',  real(maxval(abs(hvh)),real64), tol)
  call check('gradient H -> heta = 0', real(maxval(abs(heh)),real64), tol)

  !-- Test 2: random H vs independent host reference ----------------------
  call random_number(re);  call random_number(im);  Hxh = cmplx(re,im,wp)
  call random_number(re);  call random_number(im);  Hyh = cmplx(re,im,wp)
  call random_number(re);  call random_number(im);  Hzh = cmplx(re,im,wp)
  Hxd = Hxh;  Hyd = Hyh;  Hzd = Hzh
  call assemble_hv_heta(ny, nx, nz, kxd, kzd, Hxd, Hyd, Hzd, hvd, hed, gd, dgd)
  istat = cudaDeviceSynchronize()
  hvh = hvd;  heh = hed
  call check('hv  vs host reference', ref_err_hv(), tol)
  call check('heta vs host reference', ref_err_he(), tol)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  ! independent host recomputation of ĥ_v using the CPU cheb_deriv
  function ref_err_hv() result(e)
    real(real64) :: e
    real(dp) :: gre(0:n), gim(0:n), dgre(0:n), dgim(0:n)
    complex(wp) :: gcol, dg, ref
    real(wp) :: k2
    e = 0.0_real64
    do iz = 1, nz
       do ix = 1, nx
          do iy = 1, ny                       ! g = kx Hx + kz Hz
             gcol = kxh(ix)*Hxh(iy,ix,iz) + kzh(iz)*Hzh(iy,ix,iz)
             gre(iy-1) = real(real(gcol,wp),dp);  gim(iy-1) = real(aimag(gcol),dp)
          end do
          call cheb_deriv(n, gre, dgre);  call cheb_deriv(n, gim, dgim)
          k2 = kxh(ix)*kxh(ix) + kzh(iz)*kzh(iz)
          do iy = 1, ny
             dg  = cmplx(real(dgre(iy-1),wp), real(dgim(iy-1),wp), wp)
             ref = -k2*Hyh(iy,ix,iz) - ii*dg
             e = max(e, abs(cmplx(hvh(iy,ix,iz) - ref, kind=real64)))
          end do
       end do
    end do
  end function ref_err_hv

  function ref_err_he() result(e)
    real(real64) :: e
    complex(wp) :: ref
    e = 0.0_real64
    do iz = 1, nz
       do ix = 1, nx
          do iy = 1, ny
             ref = ii*(kzh(iz)*Hxh(iy,ix,iz) - kxh(ix)*Hzh(iy,ix,iz))
             e = max(e, abs(cmplx(heh(iy,ix,iz) - ref, kind=real64)))
          end do
       end do
    end do
  end function ref_err_he

  subroutine check(name, e, tolerance)
    character(*), intent(in) :: name
    real(real64), intent(in) :: e, tolerance
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', e <= tolerance)
    print '(a,a,a,es12.4,a,es10.2,a)', '[',tag,'] '//name//'  err=', e, '  tol=', tolerance
    if (e > tolerance) all_ok = .false.
  end subroutine check

end program test_forcing
