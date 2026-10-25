!===============================================================================
! test_stepper_vardt.F90 -- validate VARIABLE-dt time stepping.
!
! (A) eta eigenmode vs analytic SMR/CN amplification (~1e-8 floor).
!     Power-iterate the eta scheme to its dominant (slowest-decaying) eigenmode
!     phi_d, back out the eigenvalue mu_d from the measured per-step amplification
!     at dt_pi, then step phi_d with dt RAMPED down-then-up and check the result
!     against  phi_d * prod_i f(mu_d, dt_i),  where
!       f(mu,dt) = prod_{k=1}^3 (1 - 0.5 c_k dt mu/Re)/(1 + 0.5 c_k dt mu/Re).
!     This pins the on-the-fly lambda_k = Re/(beta_k dt) that step_eta_linear
!     rebuilds on every call (a wrong SMR/CN coefficient misses by ~1e-2).
!     NOTE: the scheme is NOT exactly diagonalised by a single dt-independent
!     eigenvector -- the RHS uses apply_L (strong-form Chebyshev) while the LHS
!     solve uses helm_band (assembled Galerkin); these are consistent but not
!     bit-identical, so the eigenvector drifts slightly with dt and leaves a
!     residual ~1e-8 between the calibration dt and the ramp dt's. Hence the 1e-6
!     gate, not machine precision.
!
! (B) eta+v variable-dt vs a FINE constant-dt reference (mild damping).
!     A ramped-dt run, recomputing the dt-dependent vif precompute (stepper_ev_setup)
!     on every change, must match a fine fixed-dt run to truncation. A stale or
!     mis-tuned recompute gives the wrong per-step amplification and diverges O(1).
!     This exercises the v recompute that (A) does not (the clamped 4th-order v
!     operator has no eta-style eigenmode trick).
!
! (C) rest-start clamp: dt = min(max(cfl_dt(fmax->0), dt_min), dt_max) = dt_max,
!     finite -- a zero-velocity start cannot produce dt = inf.
!
! Auto-discovered by run_all_tests.sh. Standalone:
!   nvfortran -cuda -cudalib=cufft -o test_stepper_vardt test_stepper_vardt.F90 \
!     stepper_lin_mod.o spec_helpers_mod.o helm_band_gpu_mod.o cheb_ops_mod.o \
!     vsolve_if_gpu_mod.o vsolve_if_mod.o helm_band_mod.o cheb_bih.o cheb_shen.o types.o
!===============================================================================
program test_stepper_vardt
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,           only: dp, ip
  use helm_band_mod,   only: helm_band_setup
  use stepper_lin_mod, only: step_eta_linear, step_ev_linear, stepper_ev_setup
  use cfl_mod,         only: cfl_dt
  use cudafor
  implicit none
  logical :: all_pass
  all_pass = .true.
  call test_eta_eigenmode(all_pass)
  call test_ev_fineref(all_pass)
  call test_clamp(all_pass)
  if (all_pass) then
     print *, 'ALL TESTS PASSED'
  else
     print *, 'SOME TESTS FAILED'
  end if

contains

  !-- SMR/CN per-step amplification of an eigenmode (eigenvalue mu) -------------
  real(dp) function amp(mu, dt, Re) result(R)
    real(dp), intent(in) :: mu, dt, Re
    real(dp) :: CSUB(3), hk
    integer  :: k
    CSUB = [8.0_dp/15.0_dp, 2.0_dp/15.0_dp, 1.0_dp/3.0_dp]
    R = 1.0_dp
    do k = 1, 3
       hk = 0.5_dp*CSUB(k)*dt/Re
       R = R * (1.0_dp - hk*mu)/(1.0_dp + hk*mu)
    end do
  end function amp

  !-- 2-norm of a complex column (0:m) -----------------------------------------
  real(dp) function cnorm(a, m) result(s)
    integer(ip), intent(in) :: m
    complex(dp), intent(in) :: a(0:m)
    integer(ip) :: k
    s = 0.0_dp
    do k = 0, m; s = s + real(a(k))**2 + aimag(a(k))**2; end do
    s = sqrt(s)
  end function cnorm

  !=== (A) ====================================================================
  subroutine test_eta_eigenmode(all_pass)
    logical, intent(inout) :: all_pass
    integer(ip), parameter :: n=32, nx=4, nz=4, imx=2, imz=1     ! mode kx=1,kz=0 (k2=1)
    real(dp),    parameter :: Re=180.0_dp, dt_pi=2.0_dp          ! strong-ish: fast power iter
    integer(ip), parameter :: Npit=400, Nr=20
    real(dp) :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
    real(dp) :: kxh(nx), kzh(nz)
    real(dp),    device, allocatable :: kx(:), kz(:), sd(:), st(:), bd(:)
    complex(dp), device, allocatable :: eta(:,:,:), Leta(:,:,:), rhs(:,:,:), scr(:,:,:)
    complex(dp), device, allocatable :: ftil(:,:,:), etash(:,:,:)
    complex(dp) :: phid(0:n), col(0:n), expect(0:n)
    real(dp) :: dt_ramp(Nr), nrm, Rpi, mu_lo, mu_hi, mu_mid, mu_d, prod, frac, d, dmax
    integer(ip) :: ix, iz, k, i, kk
    logical :: bad

    do ix=1,nx; kk=ix-1; if (kk>nx/2) kk=kk-nx; kxh(ix)=real(kk,dp); end do
    do iz=1,nz; kk=iz-1; if (kk>nz/2) kk=kk-nz; kzh(iz)=real(kk,dp); end do
    call helm_band_setup(n, sdiag, stail, bdiag, boff)
    allocate(kx(nx),kz(nz),sd(0:n-2),st(0:n-2),bd(0:n-2))
    allocate(eta(0:n,nx,nz),Leta(0:n,nx,nz),rhs(0:n,nx,nz),scr(0:n,nx,nz))
    allocate(ftil(0:n-2,nx,nz),etash(0:n-2,nx,nz))
    kx=kxh; kz=kzh; sd=sdiag; st=stail; bd=bdiag

    do k=0,n; phid(k)=cmplx(1.0_dp/real(k+1,dp), 0.0_dp, dp); end do   ! smooth seed
    nrm = cnorm(phid, n); phid = phid/nrm

    Rpi = 0.0_dp                                                       ! power iteration
    do i=1,Npit
       eta = (0.0_dp,0.0_dp); eta(:,imx,imz) = phid
       call step_eta_linear(n, nx, nz, dt_pi, Re, kx, kz, sd, st, bd, boff, eta, &
                            Leta, rhs, ftil, etash, scr)
       col = eta(:,imx,imz)
       nrm = cnorm(col, n); Rpi = nrm; phid = col/nrm
    end do

    mu_lo = 0.0_dp; mu_hi = 600.0_dp                                   ! mu_d from Rpi=f(mu,dt_pi)
    do i=1,100
       mu_mid = 0.5_dp*(mu_lo+mu_hi)
       if (amp(mu_mid, dt_pi, Re) > Rpi) then; mu_lo=mu_mid; else; mu_hi=mu_mid; end if
    end do
    mu_d = 0.5_dp*(mu_lo+mu_hi)

    do i=1,Nr                                                          ! dt ramp down-then-up (dt0=0.04)
       if (i<=Nr/2) then
          frac=real(i-1,dp)/real(Nr/2,dp);        dt_ramp(i)=0.04_dp*(1.0_dp-0.6_dp*frac)
       else
          frac=real(i-Nr/2,dp)/real(Nr-Nr/2,dp);  dt_ramp(i)=0.04_dp*(0.4_dp+0.6_dp*frac)
       end if
    end do

    eta = (0.0_dp,0.0_dp); eta(:,imx,imz) = phid                       ! step phi_d with the ramp
    prod = 1.0_dp
    do i=1,Nr
       call step_eta_linear(n, nx, nz, dt_ramp(i), Re, kx, kz, sd, st, bd, boff, eta, &
                            Leta, rhs, ftil, etash, scr)
       prod = prod * amp(mu_d, dt_ramp(i), Re)
    end do
    col = eta(:,imx,imz); expect = phid * prod

    bad=.false.; dmax=0.0_dp
    do k=0,n
       d = abs(col(k)-expect(k))
       if (.not. (d==d)) bad=.true.
       if (d > huge(1.0_dp)) bad=.true.
       dmax = max(dmax, d)
    end do
    if (bad) dmax = 1.0e30_dp
    write(*,'(a,f9.5,a,es12.5,a,es11.3)') &
       '  (A) eta eigenmode: mu_d=', mu_d, '  Pi R=', prod, '  max|eta-phi*PiR|=', dmax
    if (dmax < 1.0e-6_dp) then     ! floor ~1e-8: apply_L(strong RHS) vs helm_band(Galerkin
       write(*,'(a)') '       [ok] variable-dt eta amplification matches analytic SMR/CN'
    else                           ! LHS) make the scheme eigenvector weakly dt-dependent
       write(*,'(a)') '       [FAIL]'; all_pass=.false.
    end if
    deallocate(kx,kz,sd,st,bd,eta,Leta,rhs,scr,ftil,etash)
  end subroutine test_eta_eigenmode

  !=== (B) ====================================================================
  subroutine test_ev_fineref(all_pass)
    logical, intent(inout) :: all_pass
    integer(ip), parameter :: n=32, nx=4, nz=4
    real(dp),    parameter :: Re=180.0_dp, Tend=0.2_dp
    integer(ip), parameter :: Na=20, Nb=200
    real(dp) :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
    real(dp) :: kxh(nx), kzh(nz)
    real(dp), device, allocatable :: kx(:), kz(:), sd(:), st(:), bd(:)
    real(dp), device, allocatable :: vm(:,:,:,:), vp(:,:,:,:), Ci(:,:,:,:,:)
    complex(dp), device, allocatable :: etaA(:,:,:), vA(:,:,:), etaB(:,:,:), vB(:,:,:)
    complex(dp), allocatable :: eta0(:,:,:), v0(:,:,:)
    complex(dp), allocatable :: eA(:,:,:), eB(:,:,:), vAh(:,:,:), vBh(:,:,:)
    real(dp) :: dtA(Na), dtfine, ssum, de, dv, demax, dvmax
    integer(ip) :: ix, iz, k, i, kk
    real(dp) :: PI2
    logical :: bad

    PI2 = 8.0_dp*atan(1.0_dp)
    do ix=1,nx; kk=ix-1; if (kk>nx/2) kk=kk-nx; kxh(ix)=real(kk,dp); end do
    do iz=1,nz; kk=iz-1; if (kk>nz/2) kk=kk-nz; kzh(iz)=real(kk,dp); end do
    call helm_band_setup(n, sdiag, stail, bdiag, boff)
    allocate(kx(nx),kz(nz),sd(0:n-2),st(0:n-2),bd(0:n-2))
    allocate(vm(0:n,nx,nz,3),vp(0:n,nx,nz,3),Ci(2,2,nx,nz,3))
    allocate(etaA(0:n,nx,nz),vA(0:n,nx,nz),etaB(0:n,nx,nz),vB(0:n,nx,nz))
    allocate(eta0(0:n,nx,nz),v0(0:n,nx,nz))
    allocate(eA(0:n,nx,nz),eB(0:n,nx,nz),vAh(0:n,nx,nz),vBh(0:n,nx,nz))
    kx=kxh; kz=kzh; sd=sdiag; st=stail; bd=bdiag

    ! spectrally smooth IC (steep exp damping -> only low wall-normal modes, so the
    ! stiff biharmonic modes are negligible and both runs stay asymptotic); (0,0) zeroed.
    do iz=1,nz; do ix=1,nx; do k=0,n
       eta0(k,ix,iz) = cmplx(exp(-0.7_dp*real(k,dp))/real(ix+iz,dp), &
                             0.3_dp*exp(-0.7_dp*real(k,dp))/real(1+iz,dp), dp)
       v0(k,ix,iz)   = cmplx(0.5_dp*exp(-0.7_dp*real(k,dp))/real(ix,dp), &
                             0.2_dp*exp(-0.7_dp*real(k,dp)), dp)
    end do; end do; end do
    eta0(:,1,1) = (0.0_dp,0.0_dp); v0(:,1,1) = (0.0_dp,0.0_dp)

    ! ramp dt (down then up via 1+0.5cos), rescaled so sum(dtA) = Tend exactly
    do i=1,Na; dtA(i) = 1.0_dp + 0.5_dp*cos(PI2*real(i-1,dp)/real(Na,dp)); end do
    ssum = sum(dtA(1:Na)); dtA = dtA*(Tend/ssum)
    dtfine = Tend/real(Nb,dp)

    ! Run A: variable dt, recompute the vif precompute on every step
    etaA = eta0; vA = v0
    do i=1,Na
       call stepper_ev_setup(n, nx, nz, dtA(i), Re, kxh, kzh, vm, vp, Ci)
       call step_ev_linear(n, nx, nz, dtA(i), Re, kx, kz, sd, st, bd, boff, vm, vp, Ci, etaA, vA)
    end do

    ! Run B: fine constant dt reference
    etaB = eta0; vB = v0
    call stepper_ev_setup(n, nx, nz, dtfine, Re, kxh, kzh, vm, vp, Ci)
    do i=1,Nb
       call step_ev_linear(n, nx, nz, dtfine, Re, kx, kz, sd, st, bd, boff, vm, vp, Ci, etaB, vB)
    end do

    eA = etaA; eB = etaB; vAh = vA; vBh = vB          ! device -> host
    bad=.false.; demax=0.0_dp; dvmax=0.0_dp
    do iz=1,nz; do ix=1,nx; do k=0,n
       de = abs(eA(k,ix,iz)-eB(k,ix,iz)); dv = abs(vAh(k,ix,iz)-vBh(k,ix,iz))
       if (.not. (de==de)) bad=.true.
       if (.not. (dv==dv)) bad=.true.
       demax = max(demax, de); dvmax = max(dvmax, dv)
    end do; end do; end do
    if (bad) then; demax=1.0e30_dp; dvmax=1.0e30_dp; end if
    write(*,'(a,es11.3,a,es11.3)') '  (B) eta+v variable vs fine ref: max|d eta|=', demax, &
                                   '  max|d v|=', dvmax
    if (demax < 1.0e-3_dp .and. dvmax < 1.0e-3_dp) then
       write(*,'(a)') '       [ok] vif recompute under ramped dt matches the fine run'
    else
       write(*,'(a)') '       [FAIL]'; all_pass=.false.
    end if
    deallocate(kx,kz,sd,st,bd,vm,vp,Ci,etaA,vA,etaB,vB,eta0,v0,eA,eB,vAh,vBh)
  end subroutine test_ev_fineref

  !=== (C) ====================================================================
  subroutine test_clamp(all_pass)
    logical, intent(inout) :: all_pass
    real(dp) :: dt_max, dt_min, C, fmax, dt
    dt_max = 2.0e-2_dp; dt_min = 1.0e-4_dp; C = 1.0_dp
    fmax = 0.0_dp                                   ! rest start
    dt = min(max(cfl_dt(fmax, C), dt_min), dt_max)
    write(*,'(a,es12.4,a,es10.3)') '  (C) rest-start clamp: cfl_dt=', cfl_dt(fmax,C), &
                                   '  -> clamped dt=', dt
    if (dt == dt_max .and. dt == dt .and. dt < huge(1.0_dp)) then
       write(*,'(a)') '       [ok] zero-velocity start clamps to dt_max (finite)'
    else
       write(*,'(a)') '       [FAIL]'; all_pass=.false.
    end if
  end subroutine test_clamp

end program test_stepper_vardt
