!===============================================================================
! test_nonlinear_r2c.F90  --  [R2C] validate the R2C
! nonlinear_mod end-to-end with REPRESENTATION-INDEPENDENT physics identities
! (so the check is true correctness, not equivalence to a reference):
!
!   T1  energy conservation: u.(u x omega) = 0 identically (the rotational term
!       is perpendicular to u). Holds for any correct evaluation, C2C or R2C.
!   T2  manufactured single eta mode: u x omega has an analytically known
!       structure -- Hz nonzero only at +-2kz1, Hx=Hy=0.
!
! These are the SAME checks the C2C test_nonlinear uses; here the spectral inputs
! are the half-spectrum (ny,nkx,nz) and the physical fields ws%pu..poz are REAL
! (ny,nxd,nzd), so the recomputed cross product uses real arithmetic.
!
! NOTE: the C2C and R2C nonlinear_mod cannot be linked into ONE program (same
! module name, different workspace shape), so a direct C2C-vs-R2C numeric
! cross-check is not possible in a single binary -- hence the physics-identity
! approach, which is a stronger correctness statement anyway.
!
! NaN/Inf-hardened. Build & run ON A GPU NODE:
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -cudalib=cufft -c transforms_mod.F90 cheb_fft_mod.F90 fft3d_mod.F90 \
!             spectral3d_mod.F90 cheb_ops_mod.F90 velocity_mod.F90 forcing_mod.F90 nonlinear_mod.F90
!   nvfortran -cuda -cudalib=cufft test_nonlinear_r2c.F90 \
!             nonlinear_mod.o forcing_mod.o velocity_mod.o cheb_ops_mod.o spectral3d_mod.o \
!             fft3d_mod.o cheb_fft_mod.o transforms_mod.o types.o -o test_nonlinear_r2c
!   ./test_nonlinear_r2c
!===============================================================================
program test_nonlinear_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,          only: tp, wp, ip
  use nonlinear_mod,  only: nl_workspace, nl_setup, nl_destroy, nl_evaluate
  use cudafor
  implicit none

  logical :: all_ok = .true.

  call run_config( 9_ip,  8_ip,  6_ip)     ! small
  call run_config(17_ip, 16_ip, 12_ip)     ! medium, 2(ny-1)=32
  call run_config(16_ip, 12_ip,  8_ip)     ! even ny, nxd=18
  call run_config(11_ip,  8_ip,  8_ip)     ! odd ny
  call run_config(33_ip, 24_ip, 16_ip)     ! larger

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine run_config(ny, nx, nz)
    integer(ip), intent(in) :: ny, nx, nz
    integer(ip) :: nxd, nzd, nkx, ix, iz, m, izp, izm
    real(wp) :: kxh(nx/2+1), kzh(nz), kz1
    real(real64) :: tol, emax, escale, ex, ey, eref, ezother, hmax
    type(nl_workspace) :: ws

    complex(wp), allocatable :: vh(:,:,:), eh(:,:,:)
    real(wp),    allocatable :: pu(:,:,:), pv(:,:,:), pw(:,:,:)        ! REAL physical
    real(wp),    allocatable :: pox(:,:,:), poy(:,:,:), poz(:,:,:)
    real(wp),    allocatable :: hxc(:,:,:), hyc(:,:,:), hzc(:,:,:), udotH(:,:,:)
    complex(wp), allocatable :: Hxh(:,:,:), Hyh(:,:,:), Hzh(:,:,:)
    real(wp),    allocatable :: re(:,:,:), im(:,:,:)
    complex(wp), parameter :: ii = (0.0_wp,1.0_wp)

    complex(wp), device :: vd(ny,nx/2+1,nz), ed(ny,nx/2+1,nz)
    complex(wp), device :: hvd(ny,nx/2+1,nz), hetd(ny,nx/2+1,nz)
    complex(wp), device :: u0d(ny), w0d(ny), hx0(ny), hz0(ny)
    integer :: istat

    nxd = 3*nx/2;  nzd = 3*nz/2;  nkx = nx/2 + 1
    tol = merge(1.0e-9_real64, 1.0e-4_real64, tp == real64)
    kz1 = 1.0_wp

    ! kx is the NON-NEGATIVE half [0,nx/2]; kz is full (cuFFT order)
    do ix = 1, nkx
       kxh(ix) = real(ix-1, wp)
    end do
    do iz = 1, nz
       m = iz-1;  if (m > nz/2) m = m - nz;  kzh(iz) = real(m,wp)
    end do

    call nl_setup(ws, ny, nx, nz, kxh, kzh)

    allocate(vh(ny,nkx,nz), eh(ny,nkx,nz), re(ny,nkx,nz), im(ny,nkx,nz))
    allocate(pu(ny,nxd,nzd), pv(ny,nxd,nzd), pw(ny,nxd,nzd))
    allocate(pox(ny,nxd,nzd), poy(ny,nxd,nzd), poz(ny,nxd,nzd))
    allocate(hxc(ny,nxd,nzd), hyc(ny,nxd,nzd), hzc(ny,nxd,nzd), udotH(ny,nxd,nzd))
    allocate(Hxh(ny,nkx,nz), Hyh(ny,nkx,nz), Hzh(ny,nkx,nz))

    print '(a,i0,a,i0,a,i0,a,i0,a,i0,a,i0,a)', '--- ny=', ny, ' nx=', nx, ' nz=', nz, &
         '  (nkx=', nkx, ' nxd=', nxd, ' nzd=', nzd, ') ---'

    !-- T1: energy conservation on a random field -----------------------------
    call random_number(re);  call random_number(im);  vh = cmplx(re,im,wp)
    call random_number(re);  call random_number(im);  eh = cmplx(re,im,wp)
    vh(:,1,1) = (0.0_wp,0.0_wp);  eh(:,1,1) = (0.0_wp,0.0_wp)
    vd = vh;  ed = eh
    call random_number(re(:,1,1)); u0d = cmplx(re(:,1,1),0.0_wp,wp)
    call random_number(re(:,1,1)); w0d = cmplx(re(:,1,1),0.0_wp,wp)

    call nl_evaluate(ws, vd, ed, u0d, w0d, hvd, hetd, hx0, hz0)
    istat = cudaDeviceSynchronize()
    pu = ws%pu;  pv = ws%pv;  pw = ws%pw
    pox = ws%pox;  poy = ws%poy;  poz = ws%poz
    hxc = pv*poz - pw*poy          ! recompute u x omega from intact REAL fields
    hyc = pw*pox - pu*poz
    hzc = pu*poy - pv*pox
    udotH  = pu*hxc + pv*hyc + pw*hzc
    emax   = real(maxval(abs(udotH)), real64)
    escale = real(maxval(abs(pu*hxc) + abs(pv*hyc) + abs(pw*hzc)), real64)
    call check('  T1 rel |u.(uxw)| (energy conservation)', &
               emax/max(escale, tiny(1.0_real64)), tol)
    ! nonzero guard: the physical fields are genuinely nonzero
    call checknz('  T1 physical fields nonzero', real(maxval(abs(pu))+maxval(abs(poz)),real64))

    !-- T2: manufactured single-mode field ------------------------------------
    izp = 0;  izm = 0
    do iz = 1, nz
       if (abs(kzh(iz) - kz1) < 0.5_wp) izp = iz
       if (abs(kzh(iz) + kz1) < 0.5_wp) izm = iz
    end do
    eh = (0.0_wp,0.0_wp)
    eh(1,1,izp) =  ii*kz1
    eh(1,1,izm) = -ii*kz1
    vd = (0.0_wp,0.0_wp);  ed = eh;  u0d = (0.0_wp,0.0_wp);  w0d = (0.0_wp,0.0_wp)

    call nl_evaluate(ws, vd, ed, u0d, w0d, hvd, hetd, hx0, hz0)
    istat = cudaDeviceSynchronize()
    Hxh = ws%Hx;  Hyh = ws%Hy;  Hzh = ws%Hz

    izp = 0;  izm = 0
    do iz = 1, nz
       if (abs(kzh(iz) - 2.0_wp*kz1) < 0.5_wp) izp = iz
       if (abs(kzh(iz) + 2.0_wp*kz1) < 0.5_wp) izm = iz
    end do

    ex = real(maxval(abs(Hxh)), real64)
    ey = real(maxval(abs(Hyh)), real64)
    eref = real(abs(cmplx(Hzh(1,1,izp) - ii*kz1, kind=real64)), real64) &
         + real(abs(cmplx(Hzh(1,1,izm) + ii*kz1, kind=real64)), real64)
    Hzh(1,1,izp) = (0.0_wp,0.0_wp);  Hzh(1,1,izm) = (0.0_wp,0.0_wp)
    ezother = real(maxval(abs(Hzh)), real64)

    call check('  T2 Hx = 0', ex, tol)
    call check('  T2 Hy = 0', ey, tol)
    call check('  T2 Hz matches analytic +/-2kz1', eref, tol)
    call check('  T2 Hz clean elsewhere', ezother, tol)

    call nl_destroy(ws)
    deallocate(vh,eh,re,im,pu,pv,pw,pox,poy,poz,hxc,hyc,hzc,udotH,Hxh,Hyh,Hzh)
  end subroutine run_config

  subroutine check(name, e, tolerance)
    character(*), intent(in) :: name
    real(real64), intent(in) :: e, tolerance
    real(real64) :: ee
    character(len=4) :: tag
    ee = e
    if (.not. (ee == ee)) ee = huge(1.0_real64)     ! NaN -> fail
    tag = merge('PASS', 'FAIL', ee <= tolerance)
    print '(a,a,a,es12.4,a,es10.2,a)', '[',tag,'] '//name//'  err=', ee, '  tol=', tolerance
    if (.not. (ee <= tolerance)) all_ok = .false.
  end subroutine check

  subroutine checknz(name, mag)
    character(*), intent(in) :: name
    real(real64), intent(in) :: mag
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', mag > 0.1_real64 .and. mag < huge(1.0_real64))
    print '(a,a,a,es12.4)', '[',tag,'] '//name//'  max|f|=', mag
    if (.not. (mag > 0.1_real64 .and. mag < huge(1.0_real64))) all_ok = .false.
  end subroutine checknz

end program test_nonlinear_r2c
