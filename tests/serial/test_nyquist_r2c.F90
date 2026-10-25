!===============================================================================
! test_nyquist_r2c.F90  --  [R2C] pins the "Nyquist zeroed and KEPT zeroed"
! invariant on the nkx half-spectrum. The R2C analogue of test_nyquist.
!
! LOAD-BEARING FOR R2C: the asymmetric 3/2 pad/truncate only
! round-trips if the coarse Nyquist mode is zero. The R2C mask_nyquist is a
! DIFFERENT code path from C2C (nkx-aware: it zeros the LAST x-column = the
! x-Nyquist, and the kz-Nyquist row over the STORED x range 1..nkx; the C2C
! "do ix=1,nx" loop would write OOB on the half-width array).
!
! WHAT CHANGED FROM C2C:
!  * x-Nyquist is a SINGLE column (the last, ix=nkx), not a folded middle column
!    with a conjugate partner. z-Nyquist (row nz/2+1) is unchanged (z is full).
!  * The C2C assertion 3 (max|Im(physical u,v,w)|) is DROPPED: the R2C physical
!    fields are REAL arrays (irfft), so aimag() is identically zero by type, and
!    the realness SYMPTOM of a Nyquist leak (a one-sided e^{+iKx} mode breaking
!    Hermitian symmetry) is DECOUPLED -- a real field is real regardless. The
!    realness DOF that DOES survive (kx=0 Hermitian-in-z) is pinned fully by
!    test_realness_r2c; here it is kept only as a cheap WITNESS that masking
!    preserves it. The direct invariant -- Nyquist columns EXACTLY zero -- is
!    what this test asserts.
!
! ASSERTED (NaN-hardened, exact-zero where claimed):
!   0. After init_turbulent, the state's Nyquist columns are EXACTLY zero.
!   1. SHARPNESS: after one nl_evaluate, the UNMASKED product ws%Hx has NONZERO
!      Nyquist content (seeded |m|<=kcut band reaches Nyquist via 2+2,1+3 --
!      modelled: x-Nyquist 0.074, z-Nyquist 0.047), while hv/heta Nyquist columns
!      are EXACTLY zero and hv is nonzero off-Nyquist (non-trivial forcing).
!   2. INVARIANT: after Ns full steps (step_full), the v/eta Nyquist columns are
!      still EXACTLY zero (zero RHS there => exactly-zero banded/vif solve), finite.
!   3. WITNESS: the final state's kx=0 plane stays Hermitian-in-z at the tp
!      transform tier's ROUNDOFF-ACCUMULATION scale (masking did not perturb the
!      realness DOF). Not the primary assertion (test_realness_r2c pins the DOF
!      with an injected-ghost DECAY test). Tier-aware tolerance: the residual is
!      Ns steps of z-FFT round-trip rounding -- ~1e-16 at tp=dp, ~1e-12 at tp=sp
!      (draw-dependent; one IC draw measured 1.2e-12). A REAL
!      masking leak shows at FIELD scale (>= the ~1e-6 unmasked sharpness), 4+
!      orders above either threshold, so discrimination is preserved.
!
! Build + run (GPU node; same stack as test_realness_r2c):
!   nvfortran -c types.F90 cheb_shen.F90 cheb_bih.F90 helm_band_mod.F90 \
!             mean_mod.F90 vsolve_if_mod.F90 wavenum_mod.F90
!   nvfortran -cuda -c transforms_mod.F90 cheb_fft_mod.F90 dealias_mod.F90 \
!             fft3d_mod.F90 spectral3d_mod.F90 cheb_ops_mod.F90 velocity_mod.F90 \
!             forcing_mod.F90 nonlinear_mod.F90 helm_band_gpu_mod.F90 \
!             spec_helpers_mod.F90 vsolve_if_gpu_mod.F90 stepper_lin_mod.F90 \
!             stepper_mod.F90 stats_mod.F90 init_mod.F90
!   nvfortran -cuda -cudalib=cufft -o test_nyquist_r2c test_nyquist_r2c.F90 \
!     types.o cheb_shen.o cheb_bih.o helm_band_mod.o mean_mod.o vsolve_if_mod.o \
!     wavenum_mod.o transforms_mod.o cheb_fft_mod.o dealias_mod.o fft3d_mod.o \
!     spectral3d_mod.o cheb_ops_mod.o velocity_mod.o forcing_mod.o nonlinear_mod.o \
!     helm_band_gpu_mod.o spec_helpers_mod.o vsolve_if_gpu_mod.o stepper_lin_mod.o \
!     stepper_mod.o stats_mod.o init_mod.o
!===============================================================================
program test_nyquist_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,           only: dp, tp, ip
  use cudafor
  use wavenum_mod,     only: build_wavenumbers, build_wavenumbers_r2c
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

  integer(ip), parameter :: n = 16, nx = 8, nz = 8, Ns = 5
  integer(ip), parameter :: nkx = nx/2 + 1
  real(dp),    parameter :: Re = 10.0_dp, dt = 2.0e-3_dp, Qtar = 1.0_dp
  real(dp),    parameter :: eps = 1.0e-2_dp
  integer(ip), parameter :: pseed = 4242, kcut = 3
  real(dp),    parameter :: TWOPI = 2.0_dp*acos(-1.0_dp)

  integer(ip) :: is
  real(dp) :: kxh(nkx), kzh(nz)

  type(nl_workspace) :: ws
  type(band_ops_t)   :: bo   ! y-operator band structure (host mirrors + device)
  type(vsolve_ops_t) :: vo   ! shifted v-solve lift + influence
  type(mean_ops_t)   :: mo   ! CFR mean precompute
  real(dp),    device, allocatable :: kx(:), kz(:)
  complex(dp), device, allocatable :: v(:,:,:), eta(:,:,:), u0(:), w0(:)
  complex(dp), device, allocatable :: Nv_prev(:,:,:), Neta_prev(:,:,:)
  complex(dp), device, allocatable :: hv(:,:,:), heta(:,:,:), hx0(:), hz0(:)

  complex(dp), allocatable :: vh(:,:,:), eh(:,:,:), hvh(:,:,:), hetah(:,:,:), Hxh(:,:,:)
  real(dp) :: z0, z1, sharp, nontriv, fldmax, herm0
  real(dp)    :: herm_tol
  logical  :: bad

  bad = .false.

  ! ===== setup ================================================================
  call build_wavenumbers_r2c(nx, TWOPI, kxh)    ! [R2C] x half-axis (value irrelevant here)
  call build_wavenumbers(nz, TWOPI, kzh)
  call band_ops_alloc(bo, n);  call mean_ops_alloc(mo, n)
  call helm_band_setup(n, bo%sdh, bo%sth, bo%bdh, bo%boff)
  call mean_flux_setup(n, dt, Re, bo%sdh, bo%sth, bo%bdh, bo%boff, mo%Uhm, mo%Ih)
  call nl_setup(ws, n+1, nx, nz, kxh, kzh)      ! physical nx, kxh nkx-long

  allocate(kx(nkx), kz(nz))
  call vsolve_ops_alloc(vo, n, nkx, nz)
  allocate(v(0:n,nkx,nz), eta(0:n,nkx,nz), u0(0:n), w0(0:n))
  allocate(Nv_prev(0:n,nkx,nz), Neta_prev(0:n,nkx,nz))
  allocate(hv(0:n,nkx,nz), heta(0:n,nkx,nz), hx0(0:n), hz0(0:n))
  allocate(vh(0:n,nkx,nz), eh(0:n,nkx,nz), hvh(0:n,nkx,nz), hetah(0:n,nkx,nz), Hxh(0:n,nkx,nz))
  kx = kxh;  kz = kzh
  bo%sd = bo%sdh;  bo%st = bo%sth;  bo%bd = bo%bdh      ! H2D band structure
  call stepper_ev_setup(n, nkx, nz, dt, Re, kxh, kzh, vo%vm, vo%vp, vo%Cinv)

  call init_turbulent(n, nkx, nz, kxh, kzh, Qtar, eps, pseed, kcut, v, eta, u0, w0)
  Nv_prev = (0.0_dp,0.0_dp);  Neta_prev = (0.0_dp,0.0_dp)

  ! ===== 0. IC invariant ======================================================
  vh = v;  eh = eta
  z0 = max(nyq_cols_max(n, vh), nyq_cols_max(n, eh))
  call chk_finite(maxval(abs(vh)), bad);  call chk_finite(maxval(abs(eh)), bad)

  ! ===== 1. one nl_evaluate: mask active, and the test is sharp ===============
  call nl_evaluate(ws, v, eta, u0, w0, hv, heta, hx0, hz0)
  hvh = hv;  hetah = heta;  Hxh = ws%Hx
  sharp   = nyq_cols_max(n, Hxh)                 ! UNMASKED product: must be /= 0
  z1      = max(nyq_cols_max(n, hvh), nyq_cols_max(n, hetah))   ! masked: exact 0
  nontriv = maxval(abs(hvh))                     ! forcing nonzero off-Nyquist
  call chk_finite(sharp, bad);  call chk_finite(nontriv, bad)

  ! ===== 2. Ns full steps: state Nyquist stays EXACTLY zero ===================
  do is = 1, Ns
     call step_full(ws, n, nkx, nz, dt, Re, kx, kz, bo, vo, mo, Qtar, &
                    u0, w0, eta, v, Nv_prev, Neta_prev)
  end do
  vh = v;  eh = eta
  z0 = max(z0, nyq_cols_max(n, vh), nyq_cols_max(n, eh))
  fldmax = max(maxval(abs(vh)), maxval(abs(eh)))
  call chk_finite(fldmax, bad)

  ! ===== 3. WITNESS: final-state kx=0 plane stays Hermitian-in-z ==============
  ! (the realness DOF that survives R2C; masking must not perturb it). The
  ! complex-Im physical check of the C2C test is meaningless here (real fields).
  herm0 = max(kx0_herm(n, vh), kx0_herm(n, eh))
  call chk_finite(herm0, bad)
  ! re-confirm the mask on the EVOLVED state via one more nl_evaluate
  call nl_evaluate(ws, v, eta, u0, w0, hv, heta, hx0, hz0)
  hvh = hv;  hetah = heta
  z1 = max(z1, nyq_cols_max(n, hvh), nyq_cols_max(n, hetah))

  ! ===== report ================================================================
  write(*,'(a,es12.4)') ' [0,2] state Nyquist columns (IC + after steps; must be 0): ', z0
  write(*,'(a,es12.4)') ' [1]   hv/heta Nyquist columns (masked; must be 0)        : ', z1
  write(*,'(a,es12.4)') ' [1]   sharpness: UNMASKED Hx Nyquist content (must be >0): ', sharp
  write(*,'(a,es12.4)') ' [1]   non-trivial: max|hv| off-Nyquist                   : ', nontriv
  write(*,'(a,es12.4)') ' [2]   max|field| after steps (finite)                    : ', fldmax
  write(*,'(a,es12.4)') ' [3]   witness: final kx=0 Hermitian-in-z residual        : ', herm0

  if (bad) then
     write(*,'(a)') ' [FAIL] non-finite value encountered'
     error stop 1
  end if
  if (z0 /= 0.0_dp .or. z1 /= 0.0_dp) then
     write(*,'(a)') ' [FAIL] Nyquist columns not exactly zero'
     error stop 1
  end if
  if (sharp <= 1.0e-16_dp .or. nontriv <= 1.0e-16_dp) then
     write(*,'(a)') ' [FAIL] sharpness/triviality guard (test would be vacuous)'
     error stop 1
  end if
  ! tier-aware witness bound (see header, assertion 3): roundoff accumulation
  ! at the tp transform tier, NOT a structural pin -- dp keeps the tight bound,
  ! sp gets 1e-10 (~100x the observed accumulation, ~4 orders under a real leak).
  herm_tol = merge(1.0e-12_dp, 1.0e-10_dp, tp == real64)
  if (herm0 >= herm_tol) then
     write(*,'(a)') ' [FAIL] kx=0 Hermitian-in-z perturbed (realness DOF leak?)'
     error stop 1
  end if
  write(*,'(a)') ' [PASS] R2C Nyquist invariant: masked forcings, state exactly zero, realness held'

  call nl_destroy(ws)
  deallocate(kx, kz)
  call band_ops_free(bo);  call vsolve_ops_free(vo);  call mean_ops_free(mo)
  deallocate(v, eta, u0, w0, Nv_prev, Neta_prev, hv, heta, hx0, hz0)
  deallocate(vh, eh, hvh, hetah, Hxh)

contains

  !-- max |.| over the Nyquist columns: x-Nyquist = LAST col (nkx); z-Nyquist row
  pure function nyq_cols_max(m, a) result(s)
    integer(ip), intent(in) :: m
    complex(dp), intent(in) :: a(0:m, nkx, nz)
    real(dp) :: s
    s = max(maxval(abs(a(:, nkx, :))), maxval(abs(a(:, :, nz/2+1))))
  end function nyq_cols_max

  !-- kx=0 plane Hermitian-in-z residual: max|q(0,kz)-conj(q(0,-kz))| (ix=1) ----
  pure function kx0_herm(m, a) result(r)
    integer(ip), intent(in) :: m
    complex(dp), intent(in) :: a(0:m, nkx, nz)
    real(dp) :: r
    integer(ip) :: iz, icz, k
    r = 0.0_dp
    do iz = 1, nz
       icz = mod(nz-(iz-1), nz) + 1
       do k = 0, m
          r = max(r, abs(a(k,1,iz) - conjg(a(k,1,icz))))
       end do
    end do
  end function kx0_herm

  !-- NaN/Inf hardening: flag if x is non-finite --------------------------------
  subroutine chk_finite(x, flag)
    real(dp), intent(in)    :: x
    logical,  intent(inout) :: flag
    if (.not. (x == x) .or. x > huge(1.0_dp)) flag = .true.
  end subroutine chk_finite

end program test_nyquist_r2c
