! test_stats.F90  --  statistics module.  GPU, dp.
!
!  (A) WALL SHEAR / Re_tau.  Feed (0,0) Chebyshev coeffs with known wall slope
!      and check |du0/dy|_wall and Re_tau = sqrt(Re*|du0/dy|) to machine precision:
!        * laminar Poiseuille u0 = 0.75 T0 - 0.75 T2  (U_b=1): slope 3, both walls;
!        * u0 = T3:  slope k^2 = 9 at each wall.
!
!  (B) REYNOLDS STRESSES.  Feed a physical field with an exactly known plane mean
!      and variance,  u(j,x,z) = A(j) + B_s(j) cos(2 pi (ix-1)/nxd)  (constant in
!      z), for two samples s=1,2 with the SAME mean A but different B. The x-z
!      plane mean of cos is 0 and of cos^2 is 1/2 (exact for nxd>=3), so
!        mean_u(j) = A(j),   <u'u'>(j) = (B1^2 + B2^2)/4   (time average of B^2/2).
!      u and v share the cos phase, so <u'v'>(j) = (B1u B1v + B2u B2v)/4.
!
!  (C) DIVERGENCE CHECK.  Single mode (kx=2, kz=0), v = 1 - y^2 = (T0 - T2)/2 so
!      D_y v = -2 T1.  With u chosen as u = (i/kx) D_y v the mode is divergence
!      free -> max|div| ~ 0.  With u = w = 0 the divergence is D_y v, whose largest
!      spectral coefficient is |-2| = 2 -> max|div| = 2.
!
! NaN/Inf-hardened. Build/link (after kinds, cheb_ops_mod):
!   nvfortran -cuda -c types.F90 cheb_ops_mod.F90 stats_mod.F90
!   nvfortran -cuda -o test_stats test_stats.F90 types.o cheb_ops_mod.o stats_mod.o
!   ./test_stats

program test_stats
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,        only: tp, wp, dp, ip
  use cudafor
  use stats_mod,    only: stats_t, stats_init, stats_accumulate, stats_finalize, &
                          stats_destroy, wall_shear, re_tau, divergence_check
  implicit none

  integer(ip), parameter :: n = 16, nx = 8, nz = 8, nxd = 12, nzd = 12
  real(dp),    parameter :: PI = acos(-1.0_dp)
  logical :: all_pass = .true., bad
  integer(ip) :: j, ix, iz, k
  real(dp) :: g, ret, dmax, expv

  ! ---- (A) wall shear / Re_tau ------------------------------------------------
  real(dp) :: u0(0:n)
  write(*,'(a)') '=== stats_mod: wall shear, Reynolds stresses, divergence ==='

  u0 = 0.0_dp;  u0(0) = 0.75_dp;  u0(2) = -0.75_dp        ! laminar U_b=1
  g   = wall_shear(n, u0)
  ret = re_tau(n, 10.0_dp, u0)
  bad = (g /= g) .or. (ret /= ret)
  write(*,'(/,a)')             '--- (A) wall shear / Re_tau ---'
  write(*,'(a,f12.8,a)')       '   laminar |du0/dy|_wall = ', g,   '   (expect 3.0)'
  write(*,'(a,f12.8,a)')       '   Re_tau(Re=10)         = ', ret, '   (expect sqrt(30)=5.47722558)'
  call verdict('laminar wall slope = 3', (.not. bad) .and. abs(g-3.0_dp) < 1.0e-13_dp, all_pass)
  call verdict('laminar Re_tau = sqrt(30)', (.not. bad) .and. abs(ret-sqrt(30.0_dp)) < 1.0e-12_dp, all_pass)

  u0 = 0.0_dp;  u0(3) = 1.0_dp                            ! pure T3
  g   = wall_shear(n, u0)
  ret = re_tau(n, 2.0_dp, u0)
  call verdict('T3 wall slope = 9', abs(g-9.0_dp) < 1.0e-12_dp, all_pass)
  call verdict('T3 Re_tau = sqrt(18)', abs(ret-sqrt(18.0_dp)) < 1.0e-12_dp, all_pass)

  call check_stresses(all_pass)
  call check_divergence(all_pass)

  write(*,'(/,a)') '=============================================================='
  if (all_pass) then
     write(*,'(a)') ' ALL STATS TESTS PASSED'
  else
     write(*,'(a)') ' SOME TESTS FAILED'
  end if

contains

  ! ---- (B) Reynolds stresses --------------------------------------------------
  subroutine check_stresses(all_pass)
    logical, intent(inout) :: all_pass
    type(stats_t) :: st
    real(tp), allocatable :: hu(:,:,:), hv(:,:,:), hw(:,:,:)   ! physical vel -> tp (stats_accumulate)
    real(tp), allocatable, device :: du(:,:,:), dv(:,:,:), dw(:,:,:)
    real(dp) :: Au(0:n), Av(0:n), Aw(0:n)
    real(dp) :: B1u(0:n), B2u(0:n), B1v(0:n), B2v(0:n), B1w(0:n), B2w(0:n)
    real(dp) :: mu(0:n), mv(0:n), mw(0:n), uu(0:n), vv(0:n), ww(0:n), uv(0:n)
    real(dp) :: euu, evv, eww, euv, emu, dmx
    logical  :: bad
    integer  :: ist

    allocate(hu(0:n,nxd,nzd), hv(0:n,nxd,nzd), hw(0:n,nxd,nzd))
    allocate(du(0:n,nxd,nzd), dv(0:n,nxd,nzd), dw(0:n,nxd,nzd))

    do j = 0, n                                            ! deterministic A,B(j)
       Au(j)  = 1.0_dp + 0.05_dp*real(j,dp);  Av(j) = 0.0_dp;  Aw(j) = -0.2_dp + 0.02_dp*real(j,dp)
       B1u(j) = 0.5_dp + 0.01_dp*real(j,dp);  B2u(j) = 0.3_dp + 0.02_dp*real(j,dp)
       B1v(j) = 0.2_dp + 0.02_dp*real(j,dp);  B2v(j) = 0.4_dp - 0.01_dp*real(j,dp)
       B1w(j) = 0.1_dp + 0.00_dp*real(j,dp);  B2w(j) = 0.3_dp + 0.015_dp*real(j,dp)
    end do

    call stats_init(st, n, nxd, nzd)
    call build(hu, Au, B1u);  call build(hv, Av, B1v);  call build(hw, Aw, B1w)
    du = hu;  dv = hv;  dw = hw
    call stats_accumulate(st, n, nxd, nzd, du, dv, dw)     ! sample 1
    call build(hu, Au, B2u);  call build(hv, Av, B2v);  call build(hw, Aw, B2w)
    du = hu;  dv = hv;  dw = hw
    call stats_accumulate(st, n, nxd, nzd, du, dv, dw)     ! sample 2
    ist = cudaDeviceSynchronize()
    call stats_finalize(st, n, mu, mv, mw, uu, vv, ww, uv)

    emu=0.0_dp; euu=0.0_dp; evv=0.0_dp; eww=0.0_dp; euv=0.0_dp; bad=.false.
    do j = 0, n
       emu = max(emu, abs(mu(j)-Au(j)), abs(mw(j)-Aw(j)), abs(mv(j)-Av(j)))
       euu = max(euu, abs(uu(j) - 0.25_dp*(B1u(j)**2 + B2u(j)**2)))
       evv = max(evv, abs(vv(j) - 0.25_dp*(B1v(j)**2 + B2v(j)**2)))
       eww = max(eww, abs(ww(j) - 0.25_dp*(B1w(j)**2 + B2w(j)**2)))
       euv = max(euv, abs(uv(j) - 0.25_dp*(B1u(j)*B1v(j) + B2u(j)*B2v(j))))
    end do
    dmx = max(emu,euu,evv,eww,euv)
    if (.not. (dmx == dmx) .or. dmx > huge(1.0_dp)) bad = .true.

    write(*,'(/,a)')        '--- (B) Reynolds stresses (2 samples, known field) ---'
    write(*,'(a,es10.2)')   '   max|mean - A|         = ', emu
    write(*,'(a,es10.2)')   '   max|<uu> - (B1+B2)/4| = ', euu
    write(*,'(a,es10.2)')   '   max|<vv> - .|         = ', evv
    write(*,'(a,es10.2)')   '   max|<ww> - .|         = ', eww
    write(*,'(a,es10.2)')   '   max|<uv> - .|         = ', euv
    call verdict('mean + Reynolds stresses match analytic', (.not. bad) .and. dmx < merge(1.0e-12_dp, 1.0e-5_dp, tp == dp), all_pass)

    call stats_destroy(st)
    deallocate(hu,hv,hw,du,dv,dw)
  end subroutine check_stresses

  ! physical field  q(j,ix,iz) = A(j) + B(j) cos(2 pi (ix-1)/nxd)   (const in z)
  subroutine build(h, A, B)
    real(tp),    intent(out) :: h(0:n,nxd,nzd)
    real(dp),    intent(in)  :: A(0:n), B(0:n)
    real(dp) :: c
    do iz = 1, nzd
       do ix = 1, nxd
          c = cos(2.0_dp*PI*real(ix-1,dp)/real(nxd,dp))
          do j = 0, n
             h(j,ix,iz) = real(A(j) + B(j)*c, tp)
          end do
       end do
    end do
  end subroutine build

  ! ---- (C) divergence check ---------------------------------------------------
  subroutine check_divergence(all_pass)
    logical, intent(inout) :: all_pass
    complex(tp), allocatable :: huh(:,:,:), hwh(:,:,:)        ! -> uhat, what (tp)
    complex(wp), allocatable :: hvh(:,:,:)                    ! -> vhat (wp)
    complex(tp), allocatable, device :: uh(:,:,:), wh(:,:,:)        ! uhat, what (tp)
    complex(wp), allocatable, device :: vh(:,:,:), dvh(:,:,:)       ! vhat, scratch (wp)
    real(wp),    allocatable :: hkx(:), hkz(:)
    real(wp),    allocatable, device :: kxd(:), kzd(:)
    integer(ip), parameter :: ix0 = 3, iz0 = 1            ! kx0 = 2, kz0 = 0
    real(wp) :: kx0
    real(dp) :: maxdiv
    logical  :: bad
    integer  :: ist

    allocate(huh(0:n,nx,nz), hvh(0:n,nx,nz), hwh(0:n,nx,nz))
    allocate(uh(0:n,nx,nz), vh(0:n,nx,nz), wh(0:n,nx,nz), dvh(0:n,nx,nz))
    allocate(hkx(nx), hkz(nz), kxd(nx), kzd(nz))

    do ix = 1, nx                                          ! cuFFT-order integer wavenumbers
       if (ix-1 <= nx/2) then; hkx(ix) = real(ix-1, wp); else; hkx(ix) = real(ix-1-nx, wp); end if
    end do
    do iz = 1, nz
       if (iz-1 <= nz/2) then; hkz(iz) = real(iz-1, wp); else; hkz(iz) = real(iz-1-nz, wp); end if
    end do
    kxd = hkx;  kzd = hkz;  kx0 = hkx(ix0)                 ! = 2

    ! v = (T0 - T2)/2 = 1 - y^2  at the single mode; D_y v = -2 T1 (coeff1 = -2).
    huh = (0.0_wp,0.0_wp); hvh = (0.0_wp,0.0_wp); hwh = (0.0_wp,0.0_wp)
    hvh(0,ix0,iz0) =  cmplx(0.5_wp, 0.0_wp, wp)
    hvh(2,ix0,iz0) =  cmplx(-0.5_wp,0.0_wp, wp)

    !-- div-free: u = (i/kx) D_y v  =>  coeff1 of u = (i/kx)(-2) = -2i/kx ----------
    huh(1,ix0,iz0) = cmplx(0.0_wp, -2.0_wp/kx0, wp)        ! = (i/kx0)*(-2)
    uh = huh;  vh = hvh;  wh = hwh
    call divergence_check(n, nx, nz, kxd, kzd, uh, vh, wh, dvh, maxdiv)
    ist = cudaDeviceSynchronize()
    bad = (maxdiv /= maxdiv) .or. (maxdiv > huge(1.0_dp))
    write(*,'(/,a)')      '--- (C) divergence check ---'
    write(*,'(a,es10.2)') '   div-free mode:  max|div| = ', maxdiv
    call verdict('divergence-free mode -> ~0', (.not. bad) .and. maxdiv < 1.0e-12_dp, all_pass)

    !-- non-zero: u = w = 0  =>  div = D_y v,  largest coeff |-2| = 2 -------------
    huh(1,ix0,iz0) = (0.0_wp,0.0_wp)
    uh = huh;  vh = hvh;  wh = hwh
    call divergence_check(n, nx, nz, kxd, kzd, uh, vh, wh, dvh, maxdiv)
    ist = cudaDeviceSynchronize()
    write(*,'(a,es12.4,a)') '   v-only mode:    max|div| = ', maxdiv, '   (expect 2.0)'
    call verdict('non-div-free detected (=2)', abs(maxdiv-2.0_dp) < 1.0e-12_dp, all_pass)

    deallocate(huh,hvh,hwh,uh,vh,wh,dvh,hkx,hkz,kxd,kzd)
  end subroutine check_divergence

  subroutine verdict(label, ok, all_pass)
    character(*), intent(in)    :: label
    logical,      intent(in)    :: ok
    logical,      intent(inout) :: all_pass
    if (ok) then
       write(*,'(a,a)') '   [ok]   ', label
    else
       write(*,'(a,a)') '   [FAIL] ', label
       all_pass = .false.
    end if
  end subroutine verdict

end program test_stats
