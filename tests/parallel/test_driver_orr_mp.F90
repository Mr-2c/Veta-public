!===============================================================================
! test_driver_orr_mp.F90 -- distributed driver: the ORR-SOMMERFELD
! LINEAR-PHYSICS rung, distributed: evolve a small Tollmien-Schlichting wave on the
! laminar parabola with the FULL distributed solver (par_step_full: nonlinear +
! CFR mean + banded/vif solves on the kx-slab) and measure its complex growth rate
! vs Orszag (1971) for plane Poiseuille at Re_c=1e4, alpha=1:
!     c = 0.23752649 + 0.00373967 i   (least-stable mode, UNSTABLE).
!
! This is the distributed analogue of tests/serial/test_orr_sommerfeld_r2c.F90 and the
! FIRST distributed run where a real fluctuation mode lives off the (0,0) column,
! so the distributed transposes + the tp/sp tier actually bite (the laminar case
! has an identically-zero nonlinear and is dp==sp bit-identical).
!
! TWO checks (bit-exact THEN physics):
!  (1) BIT-EXACT: a short Ns_short-step run of par_step_full on this rank's kx-slab
!      == the single-GPU step_full oracle (eta,v per slab; u0,w0 on the kx=0 owner;
!      tol merge(1e-9,1e-3) -- the nonlinear carries tp). Confirms the assembled
!      distributed stepper reproduces the oracle for THIS IC/grid.
!  (2) PHYSICS: evolve NT=20000 steps; sample Q = vhat(m=+2, y=0) every DS=1.0 over
!      t in [300,400] (101 samples) ON THE RANK OWNING global kx=2; growth rate
!      lambda_k = log(Q_{k+1}/Q_k)/DS; the mean lambda gives sigma = alpha*c_i and
!      omega = alpha*c_r, matched to Orszag (5e-5 sigma, 5e-4 omega). Controls:
!      mode grows, |Q| in range, per-interval spread settled (merge 5e-4/1.5e-3),
!      Squire decoupling (max|eta|/max|v| ~ 0 at kz=0). The owner broadcasts its
!      scalars to rank 0 (par_reduce_sum of owner-only values) for printing.
!
! GRID (vs the single-GPU nx=8/nz=2): nx=16 (nkx=9) and nz=6 (nzd=9) so NO rank
! has an empty kx- or dealiased-z slab at P<=9 (the stepper path needs only
! nkx>=P and nzd>=P). Box Lx=4*pi keeps mode m=+2 -> alpha=1 (the wavenumber
! scaling is Lx/mode based, NOT nx based). The extra x-modes (m=+4 retained, not
! Nyquist as at nx=8) and all kz/=0 modes stay at EXACTLY zero for a kz=0 seed
! (the only nonlinear products are m2*m2 -> m4/(mean), no kz generated), so the
! seeded-mode eigenvalue is unchanged to O(eps^3). n=64 (eigenvalue resolution).
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_driver_orr_mp   Run: bash run_all_par_tests.sh run -k test_driver_orr_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!   module load nvhpc openmpi nccl
! Build: TP=dp|sp bash scripts/build/build_par.sh test_driver_orr_mp   Run: bash run_all_par_tests.sh run -k test_driver_orr_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_driver_orr_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,             only: tp, dp, ip
  use cudafor
  use decomp_mod,        only: decomp_t, decomp_setup
  use helm_band_mod,     only: helm_band_setup
  use mean_mod,          only: mean_flux_setup
  use stepper_lin_mod,   only: stepper_ev_setup
  use init_mod,          only: init_laminar
  use nonlinear_mod,     only: nl_workspace, nl_setup, nl_destroy
  use stepper_mod,       only: step_full
  use par_nonlinear_mod, only: par_nl_workspace, par_nl_setup, par_nl_destroy
  use par_stepper_mod,   only: par_step_full
  use solver_ops_mod,    only: band_ops_t, vsolve_ops_t, mean_ops_t, &
                               band_ops_alloc, band_ops_free, vsolve_ops_alloc, &
                               vsolve_ops_free, mean_ops_alloc, mean_ops_free
  use par_mod
  implicit none

  integer(ip), parameter :: n = 64, nx = 16, nz = 6, nkx = nx/2+1, ny = n+1
  integer(ip), parameter :: NT = 20000, IS0 = 15000, NSKIP = 50, NREC = 101, Ns_short = 10
  real(dp),    parameter :: Re = 1.0e4_dp, dt = 2.0e-2_dp, Qtar = 2.0_dp/3.0_dp
  real(dp),    parameter :: epsa = 1.0e-6_dp, PI = acos(-1.0_dp), Lx = 4.0_dp*PI
  real(dp),    parameter :: SIG_REF = 0.00373967_dp, OMG_REF = 0.23752649_dp
  integer(ip), parameter :: GKX = 2                          ! TS mode m=+2 -> global kx index 2 (0-based)
  real(dp),    parameter :: DS = real(NSKIP,dp)*dt           ! = 1.0

  type(decomp_t) :: d
  integer(ip) :: bnkx, kxlo, kxhi, ix, iz, kk, k, ts_col
  integer     :: istat
  logical     :: owns_ts, ok_e, ok_v, ok_m, ok_phys, all_ok

  real(dp) :: kxh(nkx), kzh(nz)
  type(band_ops_t) :: bo          ! y-operator band structure (host+device)
  type(mean_ops_t) :: mo          ! CFR mean precompute
  real(dp) :: buf(8)
  real(real64) :: tol

  complex(dp), allocatable :: eF(:,:,:), vF(:,:,:), eS(:,:,:), vS(:,:,:)
  complex(dp) :: u0F(0:n), w0F(0:n), u0S(0:n), w0S(0:n)
  complex(dp) :: qs(NREC)

  call par_init()
  tol = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)
  if (par_rank == 0) print '(a,i0,a)', '[driver-orr] world size = ', par_nproc, ' rank(s)'

  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
  bnkx = d%b_nkx_loc;  kxlo = d%b_kx0 + 1;  kxhi = d%b_kx0 + bnkx
  owns_ts = (bnkx > 0) .and. (GKX >= d%b_kx0) .and. (GKX <= d%b_kx1)
  ts_col  = GKX - d%b_kx0 + 1                                ! local 1-based col of m=+2 (valid iff owns_ts)

  ! physical wavenumbers: kx = (ix-1)*2pi/Lx -> m=2 gives alpha=1; kz folded (kz=0 used)
  do ix = 1, nkx
     kxh(ix) = real(ix-1, dp) * (2.0_dp*PI/Lx)
  end do
  do iz = 1, nz
     kk = iz-1;  if (kk > nz/2) kk = kk - nz;  kzh(iz) = real(kk, dp)
  end do
  call band_ops_alloc(bo, n);  call mean_ops_alloc(mo, n)
  call helm_band_setup(n, bo%sdh, bo%sth, bo%bdh, bo%boff)
  call mean_flux_setup(n, dt, Re, bo%sdh, bo%sth, bo%bdh, bo%boff, mo%Uhm, mo%Ih)

  allocate(eF(0:n,nkx,nz), vF(0:n,nkx,nz), eS(0:n,bnkx,nz), vS(0:n,bnkx,nz))

  ! ===== (1) BIT-EXACT: short run, distributed slab == single-GPU oracle ========
  call run_full(Ns_short, eF, vF, u0F, w0F)
  ok_e = .true.;  ok_v = .true.;  ok_m = .true.
  u0S = (0.0_dp,0.0_dp);  w0S = (0.0_dp,0.0_dp)
  if (bnkx > 0) then
     call run_slab(Ns_short, .false., eS, vS, u0S, w0S, qs)
     ok_e = close_c(eS, eF(:, kxlo:kxhi, :), tol)
     ok_v = close_c(vS, vF(:, kxlo:kxhi, :), tol)
  end if
  if (d%owns_mode00) ok_m = close_c1(u0S, u0F, tol) .and. close_c1(w0S, w0F, tol)

  ! ===== (2) PHYSICS: long run, growth rate on the kx=2 owner vs Orszag =========
  qs = (0.0_dp,0.0_dp)
  call run_slab(NT, .true., eS, vS, u0S, w0S, qs)            ! qs filled on the owner; eS/vS = final field
  ok_phys = .true.;  buf = 0.0_dp
  if (owns_ts) call assess_growth(qs, eS, vS, buf, ok_phys)  ! owner-only; fills buf, ok_phys
  call par_reduce_sum_vec(buf, 8)                            ! broadcast the owner's scalars to all ranks

  call par_reduce_and(ok_e)
  call par_reduce_and(ok_v)
  call par_reduce_and(ok_m)
  call par_reduce_and(ok_phys)
  all_ok = ok_e .and. ok_v .and. ok_m .and. ok_phys

  if (par_rank == 0) then
     print '(a)', '   --- Orr-Sommerfeld: plane Poiseuille, Re_c=1e4, alpha=1 (mode m=+2, kz=0) ---'
     print '(a,es14.6,a,es14.6)', '   measured  sigma = ', buf(1), '   omega = ', buf(2)
     print '(a,es14.6,a,es14.6)', '   Orszag    sigma = ', SIG_REF, '   omega = ', OMG_REF
     print '(a,es10.2,a,es10.2)', '   abs error sigma = ', buf(3), '   omega = ', buf(4)
     print '(a,es10.2,a)',        '   interval spread = ', buf(5), '   (tol 5e-4 dp / 1.5e-3 sp)'
     print '(a,es10.2,a,es10.2)', '   |Q| start = ', buf(6), '   end = ', buf(7)
     print '(a,es10.2)',          '   Squire control max|eta|/max|v| = ', buf(8)
     call report('eta:  distributed step_full == single-GPU oracle (short, per kx-slab)', ok_e)
     call report('v:    distributed step_full == single-GPU oracle (short, per kx-slab)', ok_v)
     call report('mean: u0,w0 == single-GPU oracle (short, on the kx=0 owner)',           ok_m)
     call report('PHYSICS: growth rate + phase speed match Orszag 1971',                   ok_phys)
     if (all_ok) then
        print '(a)', 'ALL TESTS PASSED'
     else
        print '(a)', 'SOME TESTS FAILED'
     end if
  end if

  call band_ops_free(bo);  call mean_ops_free(mo)
  call par_finalize()
  if (.not. all_ok) error stop 1

contains

  ! seed the TS wave v = epsa*(1-y^2)^2 = epsa*[(3/8)T0 - (1/2)T2 + (1/8)T4] into
  ! column `col` (kz=0 = iz 1) of a device v array; clamped wall BCs are exact.
  subroutine seed_ts(m, nxl, nzl, vdev, col)
    integer(ip), intent(in) :: m, nxl, nzl, col
    complex(dp), device     :: vdev(0:m,nxl,nzl)
    vdev(0,col,1) = cmplx( epsa*0.375_dp, 0.0_dp, dp)
    vdev(2,col,1) = cmplx(-epsa*0.5_dp,   0.0_dp, dp)
    vdev(4,col,1) = cmplx( epsa*0.125_dp, 0.0_dp, dp)
  end subroutine seed_ts

  ! single-GPU oracle: laminar parabola + TS seed in col IXM(=GKX+1), nsteps steps.
  subroutine run_full(nsteps, eout, vout, u0out, w0out)
    integer(ip), intent(in)  :: nsteps
    complex(dp), intent(out) :: eout(0:n,nkx,nz), vout(0:n,nkx,nz), u0out(0:n), w0out(0:n)
    type(nl_workspace) :: ws
    real(dp),    device, allocatable :: kx(:), kz(:)
    type(vsolve_ops_t) :: vo                                 ! local v-solve apparatus
    complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:), Nvp(:,:,:), Nep(:,:,:)
    integer(ip) :: it
    call nl_setup(ws, n+1, nx, nz, kxh, kzh)
    allocate(kx(nkx), kz(nz))
    call vsolve_ops_alloc(vo, n, nkx, nz)
    allocate(eta(0:n,nkx,nz), v(0:n,nkx,nz), u0(0:n), w0(0:n), Nvp(0:n,nkx,nz), Nep(0:n,nkx,nz))
    kx = kxh;  kz = kzh
    bo%sd = bo%sdh;  bo%st = bo%sth;  bo%bd = bo%bdh         ! H2D band structure
    call stepper_ev_setup(n, nkx, nz, dt, Re, kxh, kzh, vo%vm, vo%vp, vo%Cinv)
    call init_laminar(n, nkx, nz, Qtar, v, eta, u0, w0)
    call seed_ts(n, nkx, nz, v, GKX+1)                       ! full field: m=+2 is col GKX+1
    Nvp = (0.0_dp,0.0_dp);  Nep = (0.0_dp,0.0_dp)
    do it = 1, nsteps
       call step_full(ws, n, nkx, nz, dt, Re, kx, kz, bo, vo, mo, &
                      Qtar, u0, w0, eta, v, Nvp, Nep)
    end do
    istat = cudaDeviceSynchronize()
    eout = eta;  vout = v;  u0out = u0;  w0out = w0
    call vsolve_ops_free(vo)
    deallocate(kx, kz, eta, v, u0, w0, Nvp, Nep)
    call nl_destroy(ws)
  end subroutine run_full

  ! distributed: laminar + TS seed (on the owner) on this rank's kx-slab; nsteps
  ! steps. If sample, record Q = vhat(m=+2,y=0) on the owner every NSKIP in [IS0,NT].
  subroutine run_slab(nsteps, sample, eout, vout, u0out, w0out, qsout)
    integer(ip), intent(in)  :: nsteps
    logical,     intent(in)  :: sample
    complex(dp), intent(out) :: eout(0:n,bnkx,nz), vout(0:n,bnkx,nz), u0out(0:n), w0out(0:n)
    complex(dp), intent(out) :: qsout(NREC)
    type(par_nl_workspace) :: ws
    real(dp),    device, allocatable :: kx(:), kz(:)
    type(vsolve_ops_t) :: vo                                 ! local v-solve apparatus
    complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:), Nvp(:,:,:), Nep(:,:,:)
    complex(dp) :: vcol(0:n)
    integer(ip) :: it, irec
    qsout = (0.0_dp,0.0_dp)
    call par_nl_setup(ws, d, n+1, nx, nz, kxh, kzh)
    allocate(kx(bnkx), kz(nz))
    call vsolve_ops_alloc(vo, n, bnkx, nz)
    allocate(eta(0:n,bnkx,nz), v(0:n,bnkx,nz), u0(0:n), w0(0:n), Nvp(0:n,bnkx,nz), Nep(0:n,bnkx,nz))
    kx = kxh(kxlo:kxhi);  kz = kzh
    bo%sd = bo%sdh;  bo%st = bo%sth;  bo%bd = bo%bdh         ! H2D band structure
    call stepper_ev_setup(n, bnkx, nz, dt, Re, kxh(kxlo:kxhi), kzh, vo%vm, vo%vp, vo%Cinv)
    call init_laminar(n, bnkx, nz, Qtar, v, eta, u0, w0)
    if (.not. d%owns_mode00) then
       u0 = (0.0_dp,0.0_dp);  w0 = (0.0_dp,0.0_dp)           ! mean is rank-0-local
    end if
    if (owns_ts) call seed_ts(n, bnkx, nz, v, ts_col)        ! TS lives on the kx=2 owner's slab
    Nvp = (0.0_dp,0.0_dp);  Nep = (0.0_dp,0.0_dp)
    irec = 0
    do it = 1, nsteps
       call par_step_full(ws, n, bnkx, nz, dt, Re, kx, kz, bo, vo, mo, &
                          Qtar, u0, w0, eta, v, Nvp, Nep)
       if (sample .and. it >= IS0 .and. mod(it, NSKIP) == 0) then
          irec = irec + 1
          if (owns_ts .and. irec <= NREC) then
             vcol = v(:, ts_col, 1)                          ! D2H one column
             qsout(irec) = vmid(n, vcol)
          end if
       end if
    end do
    istat = cudaDeviceSynchronize()
    eout = eta;  vout = v;  u0out = u0;  w0out = w0
    call vsolve_ops_free(vo)
    deallocate(kx, kz, eta, v, u0, w0, Nvp, Nep)
    call par_nl_destroy(ws)
  end subroutine run_slab

  ! owner-only: complex growth rate from the Q samples + the original test's gates.
  ! buf = [sigma, omega, sig_err, omg_err, spread, |Q|start, |Q|end, max|eta|/max|v|].
  subroutine assess_growth(qsamp, efin, vfin, bufo, ok)
    complex(dp), intent(in)  :: qsamp(NREC), efin(0:n,bnkx,nz), vfin(0:n,bnkx,nz)
    real(dp),    intent(out) :: bufo(8)
    logical,     intent(out) :: ok
    complex(dp) :: lam(NREC-1), lbar
    real(dp) :: spread, sig, omg, sig_err, omg_err, q1, qN, vmax, emax
    integer(ip) :: kk
    logical :: fin
    do kk = 1, NREC-1
       lam(kk) = log(qsamp(kk+1)/qsamp(kk))/DS
    end do
    lbar = sum(lam)/real(NREC-1, dp)
    spread = 0.0_dp
    do kk = 1, NREC-1
       spread = max(spread, abs(lam(kk) - lbar))
    end do
    sig = real(lbar, dp);  omg = -aimag(lbar)
    sig_err = abs(sig - SIG_REF);  omg_err = abs(omg - OMG_REF)
    q1 = abs(qsamp(1));  qN = abs(qsamp(NREC))
    vmax = maxval(abs(vfin(:, ts_col, 1)));  emax = maxval(abs(efin(:, ts_col, 1)))
    bufo(1) = sig;  bufo(2) = omg;  bufo(3) = sig_err;  bufo(4) = omg_err
    bufo(5) = spread;  bufo(6) = q1;  bufo(7) = qN
    bufo(8) = emax/max(vmax, tiny(1.0_dp))
    fin = all(bufo == bufo) .and. all(abs(bufo) < huge(1.0_dp))
    ok = fin &
         .and. (spread <= merge(5.0e-4_dp, 1.5e-3_dp, tp == dp)) &
         .and. (sig_err <= 5.0e-5_dp) .and. (omg_err <= 5.0e-4_dp) &
         .and. (q1 > 1.0e-7_dp) .and. (q1 < 1.0e-2_dp) &      ! sharpness / seed range
         .and. (qN > q1) &                                    ! mode growing (sigma>0)
         .and. (bufo(8) <= 1.0e-4_dp)                         ! Squire: eta decoupled at kz=0
  end subroutine assess_growth

  !-- v(y=0) from Chebyshev coefficients: T_m(0)=0 (m odd), (-1)^(m/2) (m even) ---
  pure function vmid(nn, a) result(q)
    integer(ip), intent(in) :: nn
    complex(dp), intent(in) :: a(0:nn)
    complex(dp) :: q
    integer(ip) :: mm, sgn
    q = (0.0_dp,0.0_dp);  sgn = 1
    do mm = 0, nn, 2
       q = q + real(sgn,dp)*a(mm)
       sgn = -sgn
    end do
  end function vmid

  logical function close_c(a, b, tolc) result(okk)
    complex(dp),  intent(in) :: a(:,:,:), b(:,:,:)
    real(real64), intent(in) :: tolc
    okk = all(a == a) .and. all(b == b) .and. (real(maxval(abs(a - b)), real64) <= tolc)
  end function close_c

  logical function close_c1(a, b, tolc) result(okk)
    complex(dp),  intent(in) :: a(:), b(:)
    real(real64), intent(in) :: tolc
    okk = all(a == a) .and. all(b == b) .and. (real(maxval(abs(a - b)), real64) <= tolc)
  end function close_c1

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

end program test_driver_orr_mp
