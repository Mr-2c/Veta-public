!===============================================================================
! test_orr_sommerfeld_fd.F90 -- the FD-y LINEAR-PHYSICS rung: the
! Orr-Sommerfeld gate of test_orr_sommerfeld_r2c rerun with the WHOLE y-basis
! flipped to banded FD on a GLC-as-file grid (band_ops_fd_init + yderiv/yops/
! s3d setters + FD mean + y_vif_setup), evolved by the SAME production
! step_full. Orszag (1971), plane Poiseuille Re_c=1e4, alpha=1:
!     c = 0.23752649 + 0.00373967 i  ->  sigma = 0.00373967, omega = 0.23752649
!
! Differences from the Chebyshev rung (each forced by the basis change):
!  * n = 128 (vs 64): the TS critical layer (~(alpha*Re)^{-1/3} ~ 0.046) needs
!    more FD points than spectral; at GLC-128 the local spacing there is ~0.012.
!  * the state is POINT VALUES: the seed epsa*(1-y^2)^2 is evaluated at the
!    nodes; Q samples v at the y=0 NODE (j = n/2; cos(pi/2) = 0 exactly).
!  * TOLERANCES are backend-order (Cheb ~ eigenvalue-exact, FD = stencil-order
!    rate), set from the FD-discrete eigenproblem solved in numpy on the actual
!    fd_build_bands operators -- see the gate lines.
! Everything else -- box, Re, dt, seed mode m=+2, sampling window t in
! [300,400], per-interval complex log, Squire control -- is the rung verbatim.
!===============================================================================
program test_orr_sommerfeld_fd
  use kinds,           only: dp, ip
  use cudafor
  use cheb_shen,       only: glc_nodes
  use wavenum_mod,     only: build_wavenumbers, build_wavenumbers_r2c
  use mean_mod,        only: MCSUB
  use fd_mean_mod,     only: fd_mean_flux_setup
  use fdops_mod,       only: fd_build_bands, fd_wall_weights
  use nonlinear_mod,   only: nl_workspace, nl_setup, nl_destroy
  use stepper_mod,     only: step_full
  use spectral3d_mod,  only: s3d_set_ybasis_fd
  use yderiv_gpu_mod,  only: yderiv_set_fd
  use yops_mod,        only: yops_set_fd
  use solver_ops_mod,  only: band_ops_t, vsolve_ops_t, mean_ops_t, &
                             band_ops_alloc, band_ops_free, band_ops_fd_init, &
                             vsolve_ops_alloc, vsolve_ops_free, &
                             mean_ops_alloc, mean_ops_free, y_vif_setup
  implicit none

  integer(ip), parameter :: n = 128, ny = n + 1, kl = 4, nx = 8, nz = 2
  integer(ip), parameter :: nkx = nx/2 + 1
  integer(ip), parameter :: NT = 20000, IS0 = 15000, NSKIP = 50, NREC = 101
  real(dp),    parameter :: Re = 1.0e4_dp, dt = 2.0e-2_dp
  real(dp),    parameter :: Qtar = 2.0_dp/3.0_dp        ! u_c = 1.5*Qtar = 1
  real(dp),    parameter :: epsa = 1.0e-6_dp
  real(dp),    parameter :: PI = 3.141592653589793238462643_dp
  real(dp),    parameter :: SIG_REF = 0.00373967_dp     ! alpha*c_i (Orszag 1971)
  real(dp),    parameter :: OMG_REF = 0.23752649_dp     ! alpha*c_r
  integer(ip), parameter :: IXM = 3                     ! mode m = +2 column
  real(dp),    parameter :: DS = real(NSKIP,dp)*dt
  ! FD-backend eigenvalue gates, from a solver-faithful bordered
  ! eigenproblem on the actual kl=4 GLC-128 bands, INCLUDING the trapz-CFR
  ! base-flow amplitude bias u_c = 1.0001004): discrete offsets sigma 7.6e-7,
  ! omega 1.95e-5; gates = 2x(Cheb rung's time floor 5e-5/5e-4 + offsets).
  ! n=128 is LOAD-BEARING: at n=64 the sigma offset (1.3e-5) breaks the 2x.
  real(dp),    parameter :: SIG_TOL = 1.0e-4_dp
  real(dp),    parameter :: OMG_TOL = 1.0e-3_dp
  real(dp),    parameter :: SPR_TOL = 1.5e-3_dp

  integer(ip) :: it, irec, k, j, ks
  real(dp) :: kxh(nkx), kzh(nz)
  real(dp) :: yA(ny), wA(ny), wloA(ny), whiA(ny)
  real(dp) :: d1A(2*kl+1,ny), d2A(2*kl+1,ny), UhA(ny,3), IhA(3), aal

  type(nl_workspace) :: ws
  type(band_ops_t)   :: bo
  type(vsolve_ops_t) :: vo
  type(mean_ops_t)   :: mo
  real(dp),    device, allocatable :: kx(:), kz(:)
  complex(dp), device, allocatable :: v(:,:,:), eta(:,:,:), u0(:), w0(:)
  complex(dp), device, allocatable :: Nv_prev(:,:,:), Neta_prev(:,:,:)
  complex(dp), device, allocatable :: t1(:,:,:), t2(:,:,:), t3(:,:,:), t4(:,:,:)
  complex(dp), device, allocatable :: t5(:,:,:), t6(:,:,:), t7(:,:,:)

  complex(dp), allocatable :: vh(:,:,:), vcol(:), ecol(:), u0h(:)
  complex(dp) :: qs(NREC), lam(NREC-1), lbar
  real(dp) :: spread, vmax, emax, sig_err, omg_err, env
  logical  :: bad, okf

  bad = .false.

  ! ===== FD backend on the GLC(128)-as-file grid ==============================
  call glc_nodes(n, yA)
  yA(1) = 1.0_dp;  yA(ny) = -1.0_dp
  wA(1) = 0.5_dp*(yA(1) - yA(2))
  do j = 2, ny-1
     wA(j) = 0.5_dp*(yA(j-1) - yA(j+1))
  end do
  wA(ny) = 0.5_dp*(yA(ny-1) - yA(ny))
  call fd_build_bands(ny, kl, yA, d1A, d2A)
  call fd_wall_weights(ny, kl, yA, wloA, whiA)

  call build_wavenumbers_r2c(nx, 4.0_dp*PI, kxh)       ! m=2 -> alpha=1
  call build_wavenumbers(nz, PI, kzh)
  call band_ops_alloc(bo, n);  call mean_ops_alloc(mo, n)
  call band_ops_fd_init(bo, n, kl, yA, wA, .true.)
  call yderiv_set_fd(ny, kl, d1A)
  call yops_set_fd(n, wloA, whiA, wA)
  call s3d_set_ybasis_fd(.true.)
  call fd_mean_flux_setup(ny, kl, d2A, wA, dt, Re, UhA, IhA, okf)
  if (.not. okf) then
     write(*,'(a)') ' [FAIL] fd_mean_flux_setup refused';  error stop 1
  end if
  mo%Uhm(0:n,:) = UhA(1:ny,:);  mo%Ih = IhA
  call nl_setup(ws, ny, nx, nz, kxh, kzh)

  allocate(kx(nkx), kz(nz))
  call vsolve_ops_alloc(vo, n, nkx, nz)
  allocate(v(0:n,nkx,nz), eta(0:n,nkx,nz), u0(0:n), w0(0:n))
  allocate(Nv_prev(0:n,nkx,nz), Neta_prev(0:n,nkx,nz))
  allocate(vh(0:n,nkx,nz), vcol(0:n), ecol(0:n), u0h(0:n))
  kx = kxh;  kz = kzh
  allocate(t1(0:n,nkx,nz), t2(0:n-2,nkx,nz), t3(0:n-2,nkx,nz), t4(0:n,nkx,nz))
  allocate(t5(0:n,nkx,nz), t6(0:n,nkx,nz), t7(0:n,nkx,nz))
  do ks = 1, 3
     aal = 0.5_dp*MCSUB(ks)*dt/Re
     call y_vif_setup(bo, n, nkx, nz, kx, kz, aal, vo%vm(:,:,:,ks), &
                      vo%vp(:,:,:,ks), vo%Cinv(:,:,:,:,ks), t1, t2, t3, t4, t5, t6, t7)
  end do

  ! ===== IC: exact parabola VALUES (u_c = 1) + TS seed as POINT VALUES ========
  do j = 0, n
     u0h(j) = cmplx(1.5_dp*Qtar*(1.0_dp - yA(j+1)*yA(j+1)), 0.0_dp, dp)
  end do
  u0 = u0h;  w0 = (0.0_dp, 0.0_dp)
  eta = (0.0_dp, 0.0_dp)
  vh = (0.0_dp, 0.0_dp)
  do j = 0, n
     env = 1.0_dp - yA(j+1)*yA(j+1)
     vh(j,IXM,1) = cmplx(epsa*env*env, 0.0_dp, dp)     ! clamped BCs exact
  end do
  v = vh
  Nv_prev = (0.0_dp,0.0_dp);  Neta_prev = (0.0_dp,0.0_dp)

  ! ===== evolve; sample Q = v(y=0 NODE, m=+2) every DS in t in [300,400] ======
  irec = 0
  do it = 1, NT
     call step_full(ws, n, nkx, nz, dt, Re, kx, kz, bo, vo, mo, &
                    Qtar, u0, w0, eta, v, Nv_prev, Neta_prev)
     if (it >= IS0 .and. mod(it, NSKIP) == 0) then
        irec = irec + 1
        vcol = v(:,IXM,1)
        qs(irec) = vcol(n/2)                           ! y = cos(pi/2) = 0 exactly
        call chk_finite(abs(qs(irec)), bad)
     end if
  end do
  if (irec /= NREC) then
     write(*,'(a,i6,a,i6)') ' [FAIL] sampling logic: got ', irec, ' records, expected ', NREC
     error stop 1
  end if

  do k = 1, NREC-1
     lam(k) = log(qs(k+1)/qs(k))/DS
  end do
  lbar = sum(lam)/real(NREC-1, dp)
  spread = 0.0_dp
  do k = 1, NREC-1
     spread = max(spread, abs(lam(k) - lbar))
  end do
  sig_err = abs(real(lbar)  - SIG_REF)
  omg_err = abs(aimag(lbar) + OMG_REF)
  call chk_finite(real(lbar), bad);  call chk_finite(aimag(lbar), bad)
  call chk_finite(spread, bad)

  vcol = v(:,IXM,1);  ecol = eta(:,IXM,1)
  vmax = maxval(abs(vcol));  emax = maxval(abs(ecol))
  call chk_finite(vmax, bad);  call chk_finite(emax, bad)

  write(*,'(a)') '=== FD-y Orr-Sommerfeld rung: plane Poiseuille, Re_c = 1e4, alpha = 1 ==='
  write(*,'(a,i0,a,i0)')         ' backend: banded FD kl=', kl, ' on GLC-as-file, n=', n
  write(*,'(a,es14.6,a,es14.6)') ' measured  sigma = ', real(lbar), '   omega = ', -aimag(lbar)
  write(*,'(a,es14.6,a,es14.6)') ' Orszag    sigma = ', SIG_REF,    '   omega = ',  OMG_REF
  write(*,'(a,es10.2,a,es10.2)') ' abs error sigma = ', sig_err,    '   omega = ',  omg_err
  write(*,'(a,es10.2)')          ' interval spread |lam_k - mean|  = ', spread
  write(*,'(a,es10.2,a,es10.2)') ' |Q| window start = ', abs(qs(1)), '   end = ', abs(qs(NREC))
  write(*,'(a,es10.2)')          ' Squire control max|eta|/max|v|  = ', emax/max(vmax, tiny(1.0_dp))

  if (bad) then
     write(*,'(a)') ' [FAIL] non-finite value encountered';  error stop 1
  end if
  if (abs(qs(1)) <= 1.0e-7_dp .or. abs(qs(1)) >= 1.0e-2_dp) then
     write(*,'(a)') ' [FAIL] sharpness guard: mode amplitude out of range';  error stop 1
  end if
  if (abs(qs(NREC)) <= abs(qs(1))) then
     write(*,'(a)') ' [FAIL] mode not growing (sigma <= 0)';  error stop 1
  end if
  if (spread > SPR_TOL) then
     write(*,'(a)') ' [FAIL] growth rate not settled (transient/noise in window)';  error stop 1
  end if
  if (sig_err > SIG_TOL .or. omg_err > OMG_TOL) then
     write(*,'(a)') ' [FAIL] eigenvalue mismatch vs Orszag at FD-backend tolerance';  error stop 1
  end if
  if (emax > 1.0e-4_dp*vmax) then
     write(*,'(a)') ' [FAIL] eta not decoupled at kz = 0 (Squire coupling error)';  error stop 1
  end if
  write(*,'(a)') ' [PASS] FD-y Orr-Sommerfeld: growth rate and phase speed match Orszag'

  call nl_destroy(ws)
  deallocate(kx, kz)
  call band_ops_free(bo);  call vsolve_ops_free(vo);  call mean_ops_free(mo)

contains

  subroutine chk_finite(x, flag)
    real(dp), intent(in)    :: x
    logical,  intent(inout) :: flag
    if (.not. (x == x) .or. x > huge(1.0_dp)) flag = .true.
  end subroutine chk_finite

end program test_orr_sommerfeld_fd
