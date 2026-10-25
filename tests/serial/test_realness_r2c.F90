!===============================================================================
! test_realness_r2c.F90  --  [R2C] the realness capstone.
! The R2C analogue of test_realness. Builds the full R2C driver stack (init +
! stepper + mean) and pins the realness invariant that SURVIVES under R2C.
!
! HOW THIS DIFFERS FROM THE C2C TEST, AND WHY:
!   In C2C the "Lyapunov ghost" was the imaginary part of the COMPLEX physical
!   field (max|Im u|), amplified at the flow's Lyapunov rate (log_turb4). In R2C
!   the physical field is a REAL array (irfft of any half-spectrum), so THAT ghost
!   is STRUCTURALLY IMPOSSIBLE -- there is no imaginary part to carry it, and
!   cross_prod is a pure real product (the F4 fix became structural, 3a-iii).
!   The ONLY residual realness DOF in half storage is the kx=0 plane's
!   Hermitian-in-z symmetry: q(0,kz) = conj(q(0,-kz)) ("the kx=0 plane stays C2C
!   in z"). kx>0 columns are FREE (their -kx conjugate partner is implied, not
!   stored), so an asymmetry there is legitimate content, NOT a ghost.
!
!   Hence the R2C ghost = a violation of kx=0 Hermitian-in-z, injected at (kx=0,
!   +kz) WITHOUT its conjugate partner at (kx=0,-kz), and measured as
!   ghost = max_{k,kz} |q(0,kz) - conj(q(0,-kz))| on the kx=0 (ix=1) column.
!   (Modelled in numpy before coding: injection registers ~gins; a kx>0
!   perturbation correctly does NOT register; the forcing's kx=0 plane stays
!   Hermitian to round-off DESPITE the state ghost.)
!
! Phases:
!  (A) IC INTENSITY: eps IS the volume-rms, re-measured with an INDEPENDENT
!      spectral measure and the R2C kx>0-DOUBLED half-Parseval (interior kx>0
!      weighted 2; kx=0 weighted 1). Assert |q-eps|/eps < 1e-10.
!  (B) kx=0 HERMITIAN FORCING: inject a kx=0 ghost into the state, run one
!      nl_evaluate, assert the forcing hv is Hermitian-in-z ON THE kx=0 PLANE to
!      round-off. In R2C this holds STRUCTURALLY (real physical field => Hermitian
!      forcing), so the ghost has NO growth path through the nonlinear term. NOT
!      vacuous: a transform bug breaking realness would make this nonzero.
!      (B2) the injected ghost is VISIBLE in the kx=0 state residual (g0 > 1e-5),
!      so (C) cannot pass vacuously.
!  (C) GHOST DECAY THROUGH step_full: after 100 steps at Re=1, the kx=0 ghost
!      residual has decayed (g1 < 0.02*g0). Honest scope note (as in C2C): at
!      Re=1 the base flow is non-chaotic so a ghost decays by viscosity anyway;
!      (C) pins stability/decay THROUGH THE FULL R2C STEPPER, while (B) is the
!      structural guarantee that the forcing can't feed it.
!
! All assertions NaN-hardened. Qtar=0 so the state IS the fluctuation.
! n=16,nx=8,nz=8 (=>nkx=5),Re=1,dt=5e-3. Build + run (GPU node):
!   nvfortran -c types.F90 cheb_shen.F90 cheb_bih.F90 helm_band_mod.F90 \
!             mean_mod.F90 vsolve_if_mod.F90 wavenum_mod.F90
!   nvfortran -cuda -c transforms_mod.F90 cheb_fft_mod.F90 dealias_mod.F90 \
!             fft3d_mod.F90 spectral3d_mod.F90 cheb_ops_mod.F90 velocity_mod.F90 \
!             forcing_mod.F90 nonlinear_mod.F90 helm_band_gpu_mod.F90 \
!             spec_helpers_mod.F90 vsolve_if_gpu_mod.F90 stepper_lin_mod.F90 \
!             solver_ops_mod.F90 stepper_mod.F90 stats_mod.F90 init_mod.F90
!   nvfortran -cuda -cudalib=cufft -o test_realness_r2c test_realness_r2c.F90 \
!     types.o cheb_shen.o cheb_bih.o helm_band_mod.o mean_mod.o vsolve_if_mod.o \
!     wavenum_mod.o transforms_mod.o cheb_fft_mod.o dealias_mod.o fft3d_mod.o \
!     spectral3d_mod.o cheb_ops_mod.o velocity_mod.o forcing_mod.o nonlinear_mod.o \
!     helm_band_gpu_mod.o spec_helpers_mod.o vsolve_if_gpu_mod.o stepper_lin_mod.o \
!     solver_ops_mod.o stepper_mod.o stats_mod.o init_mod.o
!===============================================================================
program test_realness_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,           only: tp, dp, ip
  use cudafor
  use wavenum_mod,     only: build_wavenumbers, build_wavenumbers_r2c
  use cheb_shen,       only: cheb_deriv
  use helm_band_mod,   only: helm_band_setup
  use mean_mod,        only: mean_flux_setup
  use nonlinear_mod,   only: nl_workspace, nl_setup, nl_destroy, nl_evaluate
  use stepper_lin_mod, only: stepper_ev_setup
  use stepper_mod,     only: step_full
  use solver_ops_mod,  only: band_ops_t, vsolve_ops_t, mean_ops_t, &
                             band_ops_alloc, band_ops_free, vsolve_ops_alloc, &
                             vsolve_ops_free, mean_ops_alloc, mean_ops_free
  use init_mod,        only: init_turbulent
  implicit none

  integer(ip), parameter :: n = 16, nx = 8, nz = 8, Ns = 100
  integer(ip), parameter :: nkx = nx/2 + 1
  real(dp),    parameter :: Re = 1.0_dp, dt = 5.0e-3_dp, Qtar = 0.0_dp
  real(dp),    parameter :: eps = 5.0e-2_dp, gins = 1.0e-3_dp
  integer(ip), parameter :: pseed = 777, kcut = 3
  real(dp),    parameter :: PI = 3.141592653589793238462643_dp

  integer(ip) :: is, ix, iz, icz
  real(dp) :: kxh(nkx), kzh(nz)

  type(nl_workspace) :: ws
  type(band_ops_t)   :: bo        ! y-operator band structure (host sdh/sth/bdh + device sd/st/bd)
  type(vsolve_ops_t) :: vo        ! shifted v-solve lift + influence
  type(mean_ops_t)   :: mo        ! CFR mean precompute
  real(dp),    device, allocatable :: kx(:), kz(:)
  complex(dp), device, allocatable :: v(:,:,:), eta(:,:,:), u0(:), w0(:)
  complex(dp), device, allocatable :: Nv_prev(:,:,:), Neta_prev(:,:,:)
  complex(dp), device, allocatable :: hvd(:,:,:), hetd(:,:,:), hx0(:), hz0(:)

  complex(dp), allocatable :: vh(:,:,:), eh(:,:,:), hvh(:,:,:)
  real(dp) :: avr(0:n), avi(0:n), dvr(0:n), dvi(0:n)
  real(dp) :: k2p, E2, qmeas, qerr, hvmax, asym_hv_kx0, g0, g1, wgt
  logical  :: bad

  bad = .false.

  ! ===== setup (full R2C driver stack, 2*pi x pi box) =========================
  call build_wavenumbers_r2c(nx, 2.0_dp*PI, kxh)     ! [R2C] physical nx; kxh nkx-long, non-neg half
  call build_wavenumbers(nz,  PI,        kzh)
  call band_ops_alloc(bo, n);  call mean_ops_alloc(mo, n)
  call helm_band_setup(n, bo%sdh, bo%sth, bo%bdh, bo%boff)
  call mean_flux_setup(n, dt, Re, bo%sdh, bo%sth, bo%bdh, bo%boff, mo%Uhm, mo%Ih)
  call nl_setup(ws, n+1, nx, nz, kxh, kzh)            ! physical nx; kxh nkx-long

  allocate(kx(nkx), kz(nz))
  call vsolve_ops_alloc(vo, n, nkx, nz)
  allocate(v(0:n,nkx,nz), eta(0:n,nkx,nz), u0(0:n), w0(0:n))
  allocate(Nv_prev(0:n,nkx,nz), Neta_prev(0:n,nkx,nz))
  allocate(hvd(0:n,nkx,nz), hetd(0:n,nkx,nz), hx0(0:n), hz0(0:n))
  allocate(vh(0:n,nkx,nz), eh(0:n,nkx,nz), hvh(0:n,nkx,nz))
  kx = kxh;  kz = kzh;  bo%sd = bo%sdh;  bo%st = bo%sth;  bo%bd = bo%bdh
  call stepper_ev_setup(n, nkx, nz, dt, Re, kxh, kzh, vo%vm, vo%vp, vo%Cinv)

  call init_turbulent(n, nkx, nz, kxh, kzh, Qtar, eps, pseed, kcut, v, eta, u0, w0)
  Nv_prev = (0.0_dp,0.0_dp);  Neta_prev = (0.0_dp,0.0_dp)

  ! ===== (A) IC intensity: independent re-measure, R2C kx>0-doubled half sum ===
  vh = v;  eh = eta
  E2 = 0.0_dp
  do iz = 1, nz
     do ix = 1, nkx
        if (ix == 1 .and. iz == 1) cycle
        k2p = kxh(ix)**2 + kzh(iz)**2
        if (k2p <= 0.0_dp) cycle
        wgt = 1.0_dp
        if (ix >= 2 .and. ix <= nkx-1) wgt = 2.0_dp    ! [R2C] interior kx>0 doubled
        avr = real(vh(:,ix,iz));  avi = aimag(vh(:,ix,iz))
        call cheb_deriv(n, avr, dvr)
        call cheb_deriv(n, avi, dvi)
        E2 = E2 + wgt*( (tl2int_ri(n, dvr, dvi) + tl2int_c(n, eh(:,ix,iz)))/k2p &
                       +  tl2int_c(n, vh(:,ix,iz)) )
     end do
  end do
  qmeas = sqrt(max(E2, 0.0_dp)/2.0_dp)
  qerr  = abs(qmeas - eps)/eps
  call chk_finite(qmeas, bad)

  ! ===== (B) inject a kx=0 ghost (break kx=0 Hermitian-in-z), evaluate once ====
  ! Put content at (kx=0, +kz) WITHOUT the conjugate partner at (kx=0, -kz).
  ! iz=3 is a +kz mode; its -kz partner is iz = nz-1 = 7 (untouched) => asymmetry.
  vh(3,1,3) = vh(3,1,3) + cmplx(0.0_dp, gins,        dp)
  eh(4,1,3) = eh(4,1,3) + cmplx(0.0_dp, 0.5_dp*gins, dp)
  v = vh;  eta = eh

  ! (B2) measure the injected kx=0 state ghost (sharpness for phase C)
  g0 = kx0_herm_resid(vh) 
  g0 = max(g0, kx0_herm_resid(eh))
  call chk_finite(g0, bad)

  call nl_evaluate(ws, v, eta, u0, w0, hvd, hetd, hx0, hz0)

  ! (B1) the forcing hv must be Hermitian-in-z on the kx=0 plane to round-off,
  ! DESPITE the state ghost (structural in R2C: real physical field => Hermitian
  ! forcing; a transform bug breaking realness would make this nonzero).
  hvh = hvd
  hvmax = maxval(abs(hvh))
  asym_hv_kx0 = 0.0_dp
  do iz = 1, nz
     icz = mod(nz-(iz-1), nz) + 1
     asym_hv_kx0 = max(asym_hv_kx0, maxval(abs(hvh(:,1,iz) - conjg(hvh(:,1,icz)))))
  end do
  call chk_finite(hvmax, bad);  call chk_finite(asym_hv_kx0, bad)

  ! ===== (C) 100 steps: the kx=0 ghost must DECAY (no growth path) ============
  do is = 1, Ns
     call step_full(ws, n, nkx, nz, dt, Re, kx, kz, bo, vo, mo, Qtar, &
                    u0, w0, eta, v, Nv_prev, Neta_prev)
  end do
  vh = v;  eh = eta
  g1 = kx0_herm_resid(vh)
  g1 = max(g1, kx0_herm_resid(eh))
  call chk_finite(g1, bad)

  ! ===== report ================================================================
  write(*,'(a,es12.4,a,es12.4,a,es10.2)') ' (A)  q_rms measured = ', qmeas, &
       '   target eps = ', eps, '   rel err = ', qerr
  write(*,'(a,es12.4,a,es12.4)') ' (B1) hv kx=0 Hermitian-in-z asym = ', asym_hv_kx0, &
       '   vs max|hv| = ', hvmax
  write(*,'(a,es12.4,a)')        ' (B2) injected kx=0 state ghost g0 = ', g0, '   (must be > 1e-5)'
  write(*,'(a,es12.4,a,es10.2)') ' (C)  kx=0 ghost after 100 steps g1 = ', g1, &
       '   ratio g1/g0 = ', g1/max(g0, tiny(1.0_dp))

  if (bad) then
     write(*,'(a)') ' [FAIL] non-finite value encountered'
     error stop 1
  end if
  if (qerr >= 1.0e-10_dp) then
     write(*,'(a)') ' [FAIL] IC intensity does not match eps (normalisation broken)'
     error stop 1
  end if
  if (asym_hv_kx0 > merge(1.0e-11_dp, 1.0e-6_dp, tp == dp)*max(hvmax, tiny(1.0_dp))) then
     write(*,'(a)') ' [FAIL] forcing not Hermitian on kx=0 plane (ghost has a growth path)'
     error stop 1
  end if
  if (g0 <= 1.0e-5_dp) then
     write(*,'(a)') ' [FAIL] sharpness guard: injected kx=0 ghost not visible (test vacuous)'
     error stop 1
  end if
  if (g1 >= 2.0e-2_dp*g0) then
     write(*,'(a)') ' [FAIL] kx=0 ghost did not decay'
     error stop 1
  end if
  write(*,'(a)') ' [PASS] R2C realness: eps-exact IC, Hermitian kx=0 forcing, kx=0 ghost decays'

  call nl_destroy(ws)
  call band_ops_free(bo);  call vsolve_ops_free(vo);  call mean_ops_free(mo)
  deallocate(kx, kz)
  deallocate(v, eta, u0, w0, Nv_prev, Neta_prev, hvd, hetd, hx0, hz0)
  deallocate(vh, eh, hvh)

contains

  !-- kx=0 plane Hermitian-in-z residual: max|q(0,kz)-conj(q(0,-kz))| (ix=1) ----
  pure function kx0_herm_resid(q) result(r)
    complex(dp), intent(in) :: q(0:n,nkx,nz)
    real(dp) :: r
    integer(ip) :: iz, icz, k
    r = 0.0_dp
    do iz = 1, nz
       icz = mod(nz-(iz-1), nz) + 1
       do k = 0, n
          r = max(r, abs(q(k,1,iz) - conjg(q(k,1,icz))))
       end do
    end do
  end function kx0_herm_resid

  !-- exact int_{-1}^{1} |f|^2 dy from Chebyshev coeffs (INDEPENDENT measure) ----
  pure function tl2int_c(m, a) result(s)
    integer(ip), intent(in) :: m
    complex(dp), intent(in) :: a(0:m)
    real(dp) :: s
    s = tl2int_ri(m, real(a), aimag(a))
  end function tl2int_c

  pure function tl2int_ri(m, ar, ai) result(s)
    integer(ip), intent(in) :: m
    real(dp),    intent(in) :: ar(0:m), ai(0:m)
    real(dp) :: s, w
    integer(ip) :: i, l
    s = 0.0_dp
    do i = 0, m
       do l = i, m
          if (mod(i+l, 2) /= 0) cycle
          w = 0.5_dp*(twint(i+l) + twint(l-i))
          if (l == i) then
             s = s + (ar(i)*ar(i) + ai(i)*ai(i))*w
          else
             s = s + 2.0_dp*(ar(i)*ar(l) + ai(i)*ai(l))*w
          end if
       end do
    end do
  end function tl2int_ri

  pure function twint(j) result(w)            ! int_{-1}^{1} T_j dy
    integer(ip), intent(in) :: j
    real(dp) :: w
    if (mod(j,2) == 0) then
       w = 2.0_dp/(1.0_dp - real(j,dp)**2)
    else
       w = 0.0_dp
    end if
  end function twint

  subroutine chk_finite(x, flag)
    real(dp), intent(in)    :: x
    logical,  intent(inout) :: flag
    if (.not. (x == x) .or. x > huge(1.0_dp)) flag = .true.
  end subroutine chk_finite

end program test_realness_r2c
