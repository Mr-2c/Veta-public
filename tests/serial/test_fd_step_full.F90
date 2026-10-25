!===============================================================================
! test_fd_step_full.F90 -- the END-TO-END gate: FD-y time steps
! through the PRODUCTION stepper (step_full: SMR/CN
! + CFR mean + dealiased nonlinear evaluation), plus the identity-leg proof.
!
!   (1) IDENTITY LEGS: with the FD flag up, a (kx,kz)=(0,0)-only spectral field
!       must arrive in physical space with its y-column UNTOUCHED (values pass
!       through; the x/z inverse of a k=0-only spectrum replicates the column).
!       If the Chebyshev y-leg ran, P(y) would be the T-series SYNTHESIS of the
!       column -- an O(1) difference. Round trip back scales by nxd*nzd (cuFFT
!       unnormalised) on the retained modes.
!   (2) FD LAMINAR STEADY STATE (exactness class): u0 = 1.5*Q*(1-y^2) point
!       values, v = eta = 0. The parabola is degree-2, every FD band row is
!       >= degree-4 exact, and the flow is parallel (all convective products
!       vanish mode-exactly), so the DISCRETE steady state is the parabola
!       itself. Qtar is chosen as the IC's own discrete flux / H so CFR starts
!       exactly satisfied (otherwise stage 1 legitimately shifts the profile
!       by the trapz-vs-exact flux gap ~1e-4). 5 full steps: profile drift
!       <= 1e-9 rel, flux == H*Qtar to 1e-12 rel after every step.
!   (3) FD PERTURBED RUN (structural class): ic_fill_slab_fd perturbation
!       (envelope-BC-exact, eps-normalised via the FD energy), 10 full steps:
!       finite every step, u0 EXACTLY real (the forced-real mean contract),
!       flux conserved to 1e-10 rel, and the wall BCs held end-to-end through
!       the nonlinear stepper: |v|,|eta| at both walls <= 1e-14 (solves force
!       walls exactly), |Dv| at both walls <= 1e-8 abs (influence-matrix
!       enforcement at solver tolerance; field scale eps = 1e-4).
!   (4) CHEB REGRESSION GUARD: a 2-step Chebyshev mini-run executed before ANY
!       FD state exists and repeated after all FD rows + resets -- fields equal
!       to the BIT (proves the global backend flags leak nothing when down).
! Grid GLC(48)-as-file (descending, exact walls) + trapz weights, kl=4;
! nx=16, nz=12, Re=100, dt=2e-3.
!===============================================================================
program test_fd_step_full
  use kinds,           only: tp, dp, ip
  use cudafor
  use cheb_shen,       only: glc_nodes
  use helm_band_mod,   only: helm_band_setup
  use mean_mod,        only: mean_flux_setup, MCSUB, H_CH
  use fd_mean_mod,     only: fd_mean_flux_setup
  use fdops_mod,       only: fd_build_bands, fd_wall_weights
  use init_mod,        only: ic_fill_slab_fd, ic_mean_u0_fd
  use nonlinear_mod,   only: nl_workspace, nl_setup, nl_destroy
  use stepper_lin_mod, only: stepper_ev_setup
  use stepper_mod,     only: step_full, step_full_ws, stepper_ws, &
                             stepper_ws_alloc, stepper_ws_free   ! stub-ws leg
  use spectral3d_mod,  only: s3d_plans, s3d_set_ybasis_fd, s3d_setup_da_r2c, &
                             to_physical_da_r2c, to_spectral_da_r2c
  use yderiv_gpu_mod,  only: yderiv_set_fd, yderiv_set_cheb
  use yops_mod,        only: yops_set_fd, yops_set_cheb
  use solver_ops_mod,  only: band_ops_t, vsolve_ops_t, mean_ops_t, &
                             band_ops_alloc, band_ops_free, band_ops_fd_init, &
                             vsolve_ops_alloc, vsolve_ops_free, &
                             mean_ops_alloc, mean_ops_free, y_vif_setup
  implicit none
  integer(ip), parameter :: n = 48, ny = n + 1, kl = 4
  integer(ip), parameter :: nx = 16, nz = 12, nkx = nx/2 + 1
  integer(ip), parameter :: nxd = 3*nx/2, nzd = 3*nz/2, nkxd = nxd/2 + 1
  real(dp),    parameter :: RE = 100.0_dp, DT = 2.0e-3_dp, EPS = 1.0e-4_dp
  real(dp) :: yA(ny), wA(ny), d1A(2*kl+1,ny), d2A(2*kl+1,ny)
  real(dp) :: wloA(ny), whiA(ny), UhA(ny,3), IhA(3)
  real(dp) :: kxh(nkx), kzh(nz)
  real(dp) :: Qrun, fx, e, w, dre, dim_, a
  complex(dp), allocatable :: h1(:,:,:), h2(:,:,:), u0h(:), vch(:,:,:), ech(:,:,:)
  complex(dp), allocatable :: vch2(:,:,:), ech2(:,:,:)
  complex(dp), allocatable :: hv0(:,:,:), he0(:,:,:), u00(:)   ! saved IC
  complex(dp), allocatable :: hvF(:,:,:), heF(:,:,:), u0F(:)   ! saved finals
  type(stepper_ws) :: sws2                                     ! fd=.true. stub ws
  real(dp) :: E2, scal, u0ref(ny)
  ! device state
  type(band_ops_t)   :: bofd, boc
  type(vsolve_ops_t) :: vo
  type(mean_ops_t)   :: mops
  type(nl_workspace) :: ws
  type(s3d_plans)    :: pl
  real(dp),    device, allocatable :: kx_d(:), kz_d(:)
  complex(dp), device, allocatable :: v_d(:,:,:), e_d(:,:,:), u0_d(:), w0_d(:)
  complex(dp), device, allocatable :: Nvp_d(:,:,:), Nep_d(:,:,:)
  complex(dp), device, allocatable :: t1(:,:,:), t2(:,:,:), t3(:,:,:), t4(:,:,:)
  complex(dp), device, allocatable :: t5(:,:,:), t6(:,:,:), t7(:,:,:)
  complex(tp), device, allocatable :: A_d(:,:,:), Wf_d(:,:,:)
  real(tp),    device, allocatable :: P_d(:,:,:)
  complex(tp), allocatable :: Ah(:,:,:)
  real(tp),    allocatable :: Ph(:,:,:)
  logical :: allok, ok, okf
  integer(ip) :: j, ix, iz, ks, it, kk, istat
  allok = .true.

  ! ----- grid-as-file: GLC(48) descending, exact walls, trapz weights ---------
  call glc_nodes(n, yA)
  yA(1) = 1.0_dp;  yA(ny) = -1.0_dp
  wA(1) = 0.5_dp*(yA(1) - yA(2))
  do j = 2, ny-1
     wA(j) = 0.5_dp*(yA(j-1) - yA(j+1))
  end do
  wA(ny) = 0.5_dp*(yA(ny-1) - yA(ny))
  call fd_build_bands(ny, kl, yA, d1A, d2A)
  call fd_wall_weights(ny, kl, yA, wloA, whiA)

  do ix = 1, nkx; kxh(ix) = real(ix-1, dp); end do
  do iz = 1, nz; kk = iz-1; if (kk > nz/2) kk = kk - nz; kzh(iz) = real(kk, dp); end do

  allocate(kx_d(nkx), kz_d(nz));  kx_d = kxh;  kz_d = kzh
  allocate(v_d(0:n,nkx,nz), e_d(0:n,nkx,nz), u0_d(0:n), w0_d(0:n))
  allocate(Nvp_d(0:n,nkx,nz), Nep_d(0:n,nkx,nz))
  allocate(h1(0:n,nkx,nz), h2(0:n,nkx,nz), u0h(0:n))
  allocate(vch(0:n,nkx,nz), ech(0:n,nkx,nz), vch2(0:n,nkx,nz), ech2(0:n,nkx,nz))
  call nl_setup(ws, ny, nx, nz, kxh, kzh)
  call vsolve_ops_alloc(vo, n, nkx, nz)
  call mean_ops_alloc(mops, n)

  ! ===== (4a) CHEB mini-run, PRISTINE (before any FD state exists) ============
  call band_ops_alloc(boc, n)
  call helm_band_setup(n, boc%sdh, boc%sth, boc%bdh, boc%boff)
  boc%sd = boc%sdh;  boc%st = boc%sth;  boc%bd = boc%bdh
  call cheb_mini_run(vch, ech)

  ! ===== FD backend flip =======================================================
  call band_ops_fd_init(bofd, n, kl, yA, wA, .true.)
  call yderiv_set_fd(ny, kl, d1A)
  call yops_set_fd(n, wloA, whiA, wA)
  call s3d_set_ybasis_fd(.true.)

  ! ===== (1) identity-leg proof on the dealiased R2C path ======================
  allocate(A_d(0:n,nkx,nz), Wf_d(0:n,nkxd,nzd), P_d(0:n,nxd,nzd))
  allocate(Ah(0:n,nkx,nz), Ph(0:n,nxd,nzd))
  call s3d_setup_da_r2c(pl, ny, nx, nz)
  Ah = (0.0_tp, 0.0_tp)
  do j = 0, n
     Ah(j,1,1) = cmplx(hv(j,1), 0.0_dp, tp)      ! (0,0)-only column, real values
  end do
  A_d = Ah
  call to_physical_da_r2c(pl, ny, nx, nz, nxd, nzd, A_d, P_d, Wf_d)
  istat = cudaDeviceSynchronize();  Ph = P_d
  e = 0.0_dp;  w = 0.0_dp
  do iz = 1, nzd; do ix = 1, nxd; do j = 0, n
     w = max(w, abs(real(Ah(j,1,1), dp)))
     e = max(e, abs(real(Ph(j,ix,iz), dp) - real(Ah(j,1,1), dp)))
  end do; end do; end do
  call report('fd: y-column passes to physical UNTOUCHED (identity leg)', &
              e <= 100.0_dp*real(epsilon(1.0_tp), dp)*max(1.0_dp, w))
  call to_spectral_da_r2c(pl, ny, nx, nz, nxd, nzd, P_d, Wf_d, A_d)
  istat = cudaDeviceSynchronize()
  block
    complex(tp) :: A2h(0:n,nkx,nz)
    real(dp) :: sc
    A2h = A_d;  sc = real(nxd*nzd, dp)
    e = 0.0_dp
    do iz = 1, nz; do ix = 1, nkx; do j = 0, n
       e = max(e, abs(cmplx(A2h(j,ix,iz), kind=dp) - sc*cmplx(Ah(j,ix,iz), kind=dp)))
    end do; end do; end do
    call report('fd: round trip == nxd*nzd * input (identity leg, fwd)', &
                e <= 100.0_dp*real(epsilon(1.0_tp), dp)*max(1.0_dp, sc*w))
  end block

  ! ===== FD vif + mean apparatus ==============================================
  allocate(t1(0:n,nkx,nz), t2(0:n-2,nkx,nz), t3(0:n-2,nkx,nz), t4(0:n,nkx,nz))
  allocate(t5(0:n,nkx,nz), t6(0:n,nkx,nz), t7(0:n,nkx,nz))
  do ks = 1, 3
     a = 0.5_dp*MCSUB(ks)*DT/RE
     call y_vif_setup(bofd, n, nkx, nz, kx_d, kz_d, a, vo%vm(:,:,:,ks), &
                      vo%vp(:,:,:,ks), vo%Cinv(:,:,:,:,ks), t1, t2, t3, t4, t5, t6, t7)
  end do
  call fd_mean_flux_setup(ny, kl, d2A, wA, DT, RE, UhA, IhA, okf)
  if (.not. okf) then
     call report('fd: fd_mean_flux_setup ok', .false.)
  end if
  mops%Uhm(0:n,:) = UhA(1:ny,:)          ! FD mean_ops repopulation
  mops%Ih = IhA

  ! ===== (2) FD laminar steady state ==========================================
  call ic_mean_u0_fd(n, 1.0_dp, yA, u0h)
  do j = 1, ny
     u0ref(j) = real(u0h(j-1))
  end do
  Qrun = dot_product(wA, u0ref)/H_CH     ! discrete flux of the IC (see header)
  u0_d = u0h;  w0_d = (0.0_dp, 0.0_dp)
  v_d = (0.0_dp, 0.0_dp);  e_d = (0.0_dp, 0.0_dp)
  Nvp_d = (0.0_dp, 0.0_dp);  Nep_d = (0.0_dp, 0.0_dp)
  ok = .true.;  e = 0.0_dp
  do it = 1, 5
     call step_full(ws, n, nkx, nz, DT, RE, kx_d, kz_d, bofd, vo, mops, Qrun, &
                    u0_d, w0_d, e_d, v_d, Nvp_d, Nep_d)
     istat = cudaDeviceSynchronize();  u0h = u0_d
     fx = 0.0_dp
     do j = 1, ny
        fx = fx + wA(j)*real(u0h(j-1))
        e  = max(e, abs(real(u0h(j-1)) - u0ref(j)))
     end do
     if (abs(fx - H_CH*Qrun) > 1.0e-12_dp*H_CH*Qrun) ok = .false.
  end do
  call report('fd laminar: parabola is the discrete steady state (rel 1e-9)', &
              e <= 1.0e-9_dp*maxval(abs(u0ref)))
  call report('fd laminar: CFR flux == H*Q after every step (rel 1e-12)', ok)

  ! ===== (3) FD perturbed nonlinear run =======================================
  call ic_fill_slab_fd(n, nkx, nz, 0_ip, nkx, kxh, kzh, 4242_ip, 4_ip, &
                       yA, wA, kl, d1A, h1, h2, E2)
  scal = 0.0_dp
  if (E2 > 0.0_dp) scal = EPS/sqrt(E2/2.0_dp)
  h1 = scal*h1;  h2 = scal*h2
  allocate(hv0, source=h1);  allocate(he0, source=h2)   ! save the IC
  v_d = h1;  e_d = h2
  ! mean base = the same parabola: Qrun was DEFINED as its discrete flux / H,
  ! so the CFR constraint starts exactly satisfied here too (CFR would retarget
  ! any mismatch at stage 1 anyway; the gate below checks conservation after
  ! every step against H*Qrun, which CFR enforces at every stage).
  call ic_mean_u0_fd(n, 1.0_dp, yA, u0h)
  allocate(u00, source=u0h)                              ! save the mean IC
  u0_d = u0h;  w0_d = (0.0_dp, 0.0_dp)
  Nvp_d = (0.0_dp, 0.0_dp);  Nep_d = (0.0_dp, 0.0_dp)
  ok = .true.
  do it = 1, 10
     call step_full(ws, n, nkx, nz, DT, RE, kx_d, kz_d, bofd, vo, mops, Qrun, &
                    u0_d, w0_d, e_d, v_d, Nvp_d, Nep_d)
     istat = cudaDeviceSynchronize()
     h1 = v_d;  h2 = e_d;  u0h = u0_d
     ! finiteness (sum propagates NaN/Inf), realness, flux
     dre = sum(abs(real(h1))) + sum(abs(real(h2))) + sum(abs(aimag(h1))) &
           + sum(abs(aimag(h2)))
     if (.not. (dre == dre) .or. dre > huge(1.0_dp)) ok = .false.
     do j = 0, n
        if (aimag(u0h(j)) /= 0.0_dp) ok = .false.
     end do
     fx = 0.0_dp
     do j = 1, ny
        fx = fx + wA(j)*real(u0h(j-1))
     end do
     if (abs(fx - H_CH*Qrun) > 1.0e-10_dp*H_CH*Qrun) ok = .false.
  end do
  call report('fd perturbed: 10 steps finite, u0 exactly real, flux held', ok)
  ! wall BCs end-to-end: v, eta walls exact; Dv walls at influence tolerance
  e = 0.0_dp;  w = 0.0_dp
  do iz = 1, nz
     do ix = 1, nkx
        e = max(e, abs(h1(0,ix,iz)), abs(h1(n,ix,iz)), &
                   abs(h2(0,ix,iz)), abs(h2(n,ix,iz)))
        dre = 0.0_dp;  dim_ = 0.0_dp
        do j = 0, n
           dre  = dre  + whiA(j+1)*real(h1(j,ix,iz))
           dim_ = dim_ + whiA(j+1)*aimag(h1(j,ix,iz))
        end do
        w = max(w, abs(cmplx(dre, dim_, dp)))
        dre = 0.0_dp;  dim_ = 0.0_dp
        do j = 0, n
           dre  = dre  + wloA(j+1)*real(h1(j,ix,iz))
           dim_ = dim_ + wloA(j+1)*aimag(h1(j,ix,iz))
        end do
        w = max(w, abs(cmplx(dre, dim_, dp)))
     end do
  end do
  call report('fd perturbed: v/eta walls exact (<=1e-14)', e <= 1.0e-14_dp)
  call report('fd perturbed: Dv walls at influence tolerance (<=1e-8)', &
              w <= 1.0e-8_dp)

  ! ===== (4a) the fd=.true. STUB workspace, bit-identical ================
  ! Rerun the SAME 10 perturbed steps from the SAME IC through step_full_ws
  ! with a caller-owned stepper_ws allocated fd=.true. -- the 3 Shen buffers
  ! are 1-element stubs passed through the assumed-size seam. This exercises
  ! the production FD memory configuration. The comparison is BIT-IDENTITY vs the
  ! wrapper run above: valid because zeta_1 = 0 makes cross-step history
  ! weightless (wrapper-per-step == persistent-ws, the documented restart
  ! property), and no FD branch reads or writes the stubbed buffers -- any
  ! difference means a stub was touched.
  allocate(hvF, source=h1);  allocate(heF, source=h2)    ! finals of the wrapper run
  allocate(u0F, source=u0h)
  v_d = hv0;  e_d = he0;  u0_d = u00;  w0_d = (0.0_dp, 0.0_dp)
  Nvp_d = (0.0_dp, 0.0_dp);  Nep_d = (0.0_dp, 0.0_dp)
  call stepper_ws_alloc(sws2, n, nkx, nz, fd=.true.)
  do it = 1, 10
     call step_full_ws(ws, sws2, n, nkx, nz, DT, RE, kx_d, kz_d, bofd, vo, mops, &
                       Qrun, u0_d, w0_d, e_d, v_d, Nvp_d, Nep_d)
  end do
  istat = cudaDeviceSynchronize()
  call stepper_ws_free(sws2)
  h1 = v_d;  h2 = e_d;  u0h = u0_d
  call report('fd stub-ws (fd=.true.): 10 steps BIT-IDENTICAL to full-size ws', &
              all(h1 == hvF) .and. all(h2 == heF) .and. all(u0h == u0F))

  ! ===== (4b) reset + CHEB mini-run again: BIT-IDENTICAL ======================
  call s3d_set_ybasis_fd(.false.)
  call yderiv_set_cheb()
  call yops_set_cheb()
  call cheb_mini_run(vch2, ech2)
  call report('cheb guard: mini-run BIT-IDENTICAL before/after FD state', &
              all(vch == vch2) .and. all(ech == ech2))

  call band_ops_free(boc);  call band_ops_free(bofd)
  call vsolve_ops_free(vo);  call mean_ops_free(mops)
  call nl_destroy(ws)
  if (allok) then; print '(a)', 'ALL TESTS PASSED'
  else;            print '(a)', 'SOME TESTS FAILED'; end if
  if (.not. allok) error stop 1

contains

  !-- 2 Chebyshev step_full's from a deterministic hash IC; returns fields ----
  subroutine cheb_mini_run(vout, eout)
    complex(dp), intent(out) :: vout(0:n,nkx,nz), eout(0:n,nkx,nz)
    integer(ip) :: jj, ixx, izz
    call stepper_ev_setup(n, nkx, nz, DT, RE, kxh, kzh, vo%vm, vo%vp, vo%Cinv)
    call mean_flux_setup(n, DT, RE, boc%sdh, boc%sth, boc%bdh, boc%boff, &
                         mops%Uhm, mops%Ih)
    do izz = 1, nz; do ixx = 1, nkx; do jj = 0, n
       h1(jj,ixx,izz) = 1.0e-4_dp*exp(-0.5_dp*jj) &
                        *cmplx(hv(jj,ixx+3*izz), hv(jj,ixx+7*izz), dp)
       h2(jj,ixx,izz) = 1.0e-4_dp*exp(-0.5_dp*jj) &
                        *cmplx(hv(jj,ixx+11*izz), hv(jj,ixx+13*izz), dp)
    end do; end do; end do
    h1(:,1,1) = (0.0_dp,0.0_dp);   h2(:,1,1) = (0.0_dp,0.0_dp)
    h1(:,nkx,:) = (0.0_dp,0.0_dp); h2(:,nkx,:) = (0.0_dp,0.0_dp)
    h1(:,:,nz/2+1) = (0.0_dp,0.0_dp); h2(:,:,nz/2+1) = (0.0_dp,0.0_dp)
    v_d = h1;  e_d = h2
    u0h = (0.0_dp,0.0_dp)
    u0h(0) = cmplx(0.75_dp, 0.0_dp, dp);  u0h(2) = cmplx(-0.75_dp, 0.0_dp, dp)
    u0_d = u0h;  w0_d = (0.0_dp,0.0_dp)
    Nvp_d = (0.0_dp,0.0_dp);  Nep_d = (0.0_dp,0.0_dp)
    do it = 1, 2
       call step_full(ws, n, nkx, nz, DT, RE, kx_d, kz_d, boc, vo, mops, 1.0_dp, &
                      u0_d, w0_d, e_d, v_d, Nvp_d, Nep_d)
    end do
    istat = cudaDeviceSynchronize()
    vout = v_d;  eout = e_d
  end subroutine cheb_mini_run

  real(dp) function hv(a1, b1) result(vv)
    integer(ip), intent(in) :: a1, b1
    integer(ip) :: h
    h = modulo(a1*1009_ip + b1*2027_ip + a1*b1*17_ip, 100003_ip)
    vv = real(h, dp)/100003.0_dp - 0.5_dp
  end function hv

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. okk) allok = .false.
  end subroutine report

end program test_fd_step_full
