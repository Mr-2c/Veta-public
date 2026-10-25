!===============================================================================
! test_orr_sommerfeld_r2c.F90  --  [R2C] the LINEAR-PHYSICS validation rung.
! The R2C analogue of test_orr_sommerfeld -- the analytic-eigenvalue gate of the
! verification ladder (operators -> LINEAR PHYSICS -> laminar -> turbulence);
! never skip this rung silently.
!
! Measures the complex growth rate of a small Tollmien-Schlichting wave evolved
! by the FULL R2C solver (nonlinear stepper, CFR mean, banded/vif solves) on the
! laminar parabola, vs Orszag (1971) for plane Poiseuille at Re_c=1e4, alpha=1:
!     c = 0.23752649 + 0.00373967 i   (least-stable mode, UNSTABLE)
! Exercises against an ANALYTIC answer: the v biharmonic/vif solve, recover_uw,
! the mean<->fluctuation coupling in u x omega, wall BCs, wavenumber scaling, SMR/CN.
!
! WHAT CHANGED FROM C2C (the ONE real change -- the rest is extent threading):
!  * The TS seed. C2C seeds m=+2 (col IXM) AND its conjugate at m=-2 (col nx-IXM+2)
!    to make the physical field real. In R2C the m=-2 column is NOT stored (it is
!    the IMPLIED conjugate of the kx>0 column m=+2), so we seed ONLY vh(:,IXM,1)
!    and the irfft supplies the conjugate. Modelled: the R2C physical TS wave and
!    the sampled Q=vhat(m=+2,y=0) are IDENTICAL to the C2C two-column seed (diff 0).
!    m=+2 is a kx>0 FREE column (no Hermitian constraint), so no partner line.
!
! DESIGN (unchanged from C2C, each load-bearing):
!  * Units: Qtar=2/3 => u_c=1.5*Qtar=1 => Re_code=1e4 IS Re_c; eigenvalue maps 1:1,
!    d ln(vhat)/dt = sigma - i*omega, sigma=alpha*c_i, omega=alpha*c_r.
!  * Box: Lx=4*pi, wave in mode m=+2 => alpha=2*(2pi/Lx)=1 (a wrong wavenumber
!    scaling would shift alpha and miss the eigenvalue by O(1)). nz=2 (kz=0 only):
!    Squire decouples at kz=0, so eta must stay at round-off -- asserted as control.
!  * Exactly linear: nx=8 => the first harmonic m=4 IS the x-Nyquist (col nkx),
!    masked; m2 x conj(m2) feeds only the (0,0) mean (O(eps^2), absorbed by CFR).
!  * Seed v=eps*(1-y^2)^2=eps*[(3/8)T0-(1/2)T2+(1/8)T4] (clamped BCs exact).
!  * Measure Q=vhat(m=+2,y=0) every DS=1.0 over t in [300,400] (101 samples),
!    lambda_k=log(Q_{k+1}/Q_k)/DS; per-interval phase 0.2375 rad < pi (no unwrap).
!  * Tolerances 5e-5 sigma / 5e-4 omega: discriminating but with margin.
!
! Runtime ~20-40 s (20,000 steps, tiny grid): the slowest test, by design.
! Build + run (GPU node; same stack as test_realness_r2c):
!   nvfortran -c types.F90 cheb_shen.F90 cheb_bih.F90 helm_band_mod.F90 \
!             mean_mod.F90 vsolve_if_mod.F90 wavenum_mod.F90
!   nvfortran -cuda -c transforms_mod.F90 cheb_fft_mod.F90 dealias_mod.F90 \
!             fft3d_mod.F90 spectral3d_mod.F90 cheb_ops_mod.F90 velocity_mod.F90 \
!             forcing_mod.F90 nonlinear_mod.F90 helm_band_gpu_mod.F90 \
!             spec_helpers_mod.F90 vsolve_if_gpu_mod.F90 stepper_lin_mod.F90 \
!             stepper_mod.F90 stats_mod.F90 init_mod.F90
!   nvfortran -cuda -cudalib=cufft -o test_orr_sommerfeld_r2c test_orr_sommerfeld_r2c.F90 \
!     types.o cheb_shen.o cheb_bih.o helm_band_mod.o mean_mod.o vsolve_if_mod.o \
!     wavenum_mod.o transforms_mod.o cheb_fft_mod.o dealias_mod.o fft3d_mod.o \
!     spectral3d_mod.o cheb_ops_mod.o velocity_mod.o forcing_mod.o nonlinear_mod.o \
!     helm_band_gpu_mod.o spec_helpers_mod.o vsolve_if_gpu_mod.o stepper_lin_mod.o \
!     stepper_mod.o stats_mod.o init_mod.o
!===============================================================================
program test_orr_sommerfeld_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,           only: tp, dp, ip
  use cudafor
  use wavenum_mod,     only: build_wavenumbers, build_wavenumbers_r2c
  use helm_band_mod,   only: helm_band_setup
  use mean_mod,        only: mean_flux_setup
  use nonlinear_mod,   only: nl_workspace, nl_setup, nl_destroy
  use stepper_lin_mod, only: stepper_ev_setup
  use stepper_mod,     only: step_full
  use solver_ops_mod,  only: band_ops_t, vsolve_ops_t, mean_ops_t, &
                             band_ops_alloc, band_ops_free, vsolve_ops_alloc, &
                             vsolve_ops_free, mean_ops_alloc, mean_ops_free
  use init_mod,        only: init_laminar
  implicit none

  integer(ip), parameter :: n = 64, nx = 8, nz = 2
  integer(ip), parameter :: nkx = nx/2 + 1
  integer(ip), parameter :: NT = 20000, IS0 = 15000, NSKIP = 50, NREC = 101
  real(dp),    parameter :: Re = 1.0e4_dp, dt = 2.0e-2_dp
  real(dp),    parameter :: Qtar = 2.0_dp/3.0_dp        ! u_c = 1.5*Qtar = 1
  real(dp),    parameter :: epsa = 1.0e-6_dp
  real(dp),    parameter :: PI = 3.141592653589793238462643_dp
  real(dp),    parameter :: SIG_REF = 0.00373967_dp     ! alpha*c_i  (Orszag 1971)
  real(dp),    parameter :: OMG_REF = 0.23752649_dp     ! alpha*c_r
  integer(ip), parameter :: IXM = 3                     ! mode m = +2 column
  real(dp),    parameter :: DS = real(NSKIP,dp)*dt      ! sample interval = 1.0

  integer(ip) :: it, irec, k
  real(dp) :: kxh(nkx), kzh(nz)

  type(nl_workspace) :: ws
  type(band_ops_t)   :: bo        ! y-operator band structure
  type(vsolve_ops_t) :: vo        ! shifted v-solve lift + influence
  type(mean_ops_t)   :: mo        ! CFR mean precompute
  real(dp),    device, allocatable :: kx(:), kz(:)
  complex(dp), device, allocatable :: v(:,:,:), eta(:,:,:), u0(:), w0(:)
  complex(dp), device, allocatable :: Nv_prev(:,:,:), Neta_prev(:,:,:)

  complex(dp), allocatable :: vh(:,:,:), vcol(:), ecol(:)
  complex(dp) :: qs(NREC), lam(NREC-1), lbar
  real(dp) :: spread, vmax, emax, sig_err, omg_err
  logical  :: bad

  bad = .false.

  ! ===== setup: 4*pi x 2 x (pi) box, kz inert =================================
  call build_wavenumbers_r2c(nx, 4.0_dp*PI, kxh)       ! [R2C] x half-axis; m=2 -> alpha=1
  call build_wavenumbers(nz, PI,        kzh)
  call band_ops_alloc(bo, n);  call mean_ops_alloc(mo, n)
  call helm_band_setup(n, bo%sdh, bo%sth, bo%bdh, bo%boff)
  call mean_flux_setup(n, dt, Re, bo%sdh, bo%sth, bo%bdh, bo%boff, mo%Uhm, mo%Ih)
  call nl_setup(ws, n+1, nx, nz, kxh, kzh)             ! physical nx, kxh nkx-long

  allocate(kx(nkx), kz(nz))
  call vsolve_ops_alloc(vo, n, nkx, nz)
  allocate(v(0:n,nkx,nz), eta(0:n,nkx,nz), u0(0:n), w0(0:n))
  allocate(Nv_prev(0:n,nkx,nz), Neta_prev(0:n,nkx,nz))
  allocate(vh(0:n,nkx,nz), vcol(0:n), ecol(0:n))
  kx = kxh;  kz = kzh
  bo%sd = bo%sdh;  bo%st = bo%sth;  bo%bd = bo%bdh      ! H2D band structure
  call stepper_ev_setup(n, nkx, nz, dt, Re, kxh, kzh, vo%vm, vo%vp, vo%Cinv)

  ! ===== IC: exact parabola (u_c = 1) + TS seed in mode m = +2 ================
  call init_laminar(n, nkx, nz, Qtar, v, eta, u0, w0)
  vh = (0.0_dp, 0.0_dp)
  ! v = epsa*(1-y^2)^2 = epsa*[(3/8)T0 - (1/2)T2 + (1/8)T4]; clamped BCs exact.
  ! [R2C] seed ONLY the m=+2 column (IXM); the m=-2 conjugate is IMPLIED (not
  ! stored -- kx>0 free column), the irfft supplies it. No Hermitian-partner line.
  vh(0,IXM,1) = cmplx(epsa*0.375_dp, 0.0_dp, dp)
  vh(2,IXM,1) = cmplx(-epsa*0.5_dp,  0.0_dp, dp)
  vh(4,IXM,1) = cmplx(epsa*0.125_dp, 0.0_dp, dp)
  v = vh
  Nv_prev = (0.0_dp,0.0_dp);  Neta_prev = (0.0_dp,0.0_dp)

  ! ===== evolve; sample Q = vhat(m=+2, y=0) every DS in t in [300,400] ========
  irec = 0
  do it = 1, NT
     call step_full(ws, n, nkx, nz, dt, Re, kx, kz, bo, vo, mo, &
                    Qtar, u0, w0, eta, v, Nv_prev, Neta_prev)
     if (it >= IS0 .and. mod(it, NSKIP) == 0) then
        irec = irec + 1
        vcol = v(:,IXM,1)                              ! D2H, one column
        qs(irec) = vmid(n, vcol)
        call chk_finite(abs(qs(irec)), bad)
     end if
  end do
  if (irec /= NREC) then
     write(*,'(a,i6,a,i6)') ' [FAIL] sampling logic: got ', irec, ' records, expected ', NREC
     error stop 1
  end if

  ! ===== growth rate: per-interval complex log, mean + spread =================
  do k = 1, NREC-1
     lam(k) = log(qs(k+1)/qs(k))/DS
  end do
  lbar = sum(lam)/real(NREC-1, dp)
  spread = 0.0_dp
  do k = 1, NREC-1
     spread = max(spread, abs(lam(k) - lbar))
  end do
  sig_err = abs(real(lbar)  - SIG_REF)
  omg_err = abs(aimag(lbar) + OMG_REF)                 ! Q ~ e^{(sigma - i*omega)t}
  call chk_finite(real(lbar), bad);  call chk_finite(aimag(lbar), bad)
  call chk_finite(spread, bad)

  ! ===== controls: Squire decoupling (eta ~ 0 at kz = 0), mode alive ==========
  vcol = v(:,IXM,1);  ecol = eta(:,IXM,1)
  vmax = maxval(abs(vcol));  emax = maxval(abs(ecol))
  call chk_finite(vmax, bad);  call chk_finite(emax, bad)

  ! ===== report ================================================================
  write(*,'(a)') '=== R2C Orr-Sommerfeld rung: plane Poiseuille, Re_c = 1e4, alpha = 1 ==='
  write(*,'(a,es14.6,a,es14.6)') ' measured  sigma = ', real(lbar), '   omega = ', -aimag(lbar)
  write(*,'(a,es14.6,a,es14.6)') ' Orszag    sigma = ', SIG_REF,    '   omega = ',  OMG_REF
  write(*,'(a,es10.2,a,es10.2)') ' abs error sigma = ', sig_err,    '   omega = ',  omg_err
  write(*,'(a,es10.2)')          ' interval spread |lam_k - mean|  = ', spread
  write(*,'(a,es10.2,a,es10.2)') ' |Q| window start = ', abs(qs(1)), '   end = ', abs(qs(NREC))
  write(*,'(a,es10.2)')          ' Squire control max|eta|/max|v|  = ', emax/max(vmax, tiny(1.0_dp))

  if (bad) then
     write(*,'(a)') ' [FAIL] non-finite value encountered'
     error stop 1
  end if
  if (abs(qs(1)) <= 1.0e-7_dp .or. abs(qs(1)) >= 1.0e-2_dp) then
     write(*,'(a)') ' [FAIL] sharpness guard: mode amplitude out of range (seed/decay issue)'
     error stop 1
  end if
  if (abs(qs(NREC)) <= abs(qs(1))) then
     write(*,'(a)') ' [FAIL] mode not growing (sigma <= 0): subcritical instability missed'
     error stop 1
  end if
  if (spread > merge(5.0e-4_dp, 1.5e-3_dp, tp == dp)) then
     write(*,'(a)') ' [FAIL] growth rate not settled (transient/noise in window)'
     error stop 1
  end if
  if (sig_err > 5.0e-5_dp .or. omg_err > 5.0e-4_dp) then
     write(*,'(a)') ' [FAIL] eigenvalue mismatch vs Orszag (linear dynamics wrong)'
     error stop 1
  end if
  if (emax > 1.0e-4_dp*vmax) then
     write(*,'(a)') ' [FAIL] eta not decoupled at kz = 0 (Squire coupling error)'
     error stop 1
  end if
  write(*,'(a)') ' [PASS] R2C Orr-Sommerfeld: growth rate and phase speed match Orszag'

  call nl_destroy(ws)
  deallocate(kx, kz)
  call band_ops_free(bo);  call vsolve_ops_free(vo);  call mean_ops_free(mo)
  deallocate(v, eta, u0, w0, Nv_prev, Neta_prev, vh, vcol, ecol)

contains

  !-- v(y=0) from Chebyshev coefficients: T_m(0) = 0 (m odd), (-1)^(m/2) (m even)
  pure function vmid(nn, a) result(q)
    integer(ip), intent(in) :: nn
    complex(dp), intent(in) :: a(0:nn)
    complex(dp) :: q
    integer(ip) :: mm, sgn
    q = (0.0_dp, 0.0_dp);  sgn = 1
    do mm = 0, nn, 2
       q = q + real(sgn,dp)*a(mm)
       sgn = -sgn
    end do
  end function vmid

  !-- NaN/Inf hardening ---------------------------------------------------------
  subroutine chk_finite(x, flag)
    real(dp), intent(in)    :: x
    logical,  intent(inout) :: flag
    if (.not. (x == x) .or. x > huge(1.0_dp)) flag = .true.
  end subroutine chk_finite

end program test_orr_sommerfeld_r2c
