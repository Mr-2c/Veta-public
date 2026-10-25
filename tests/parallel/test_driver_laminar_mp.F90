!===============================================================================
! test_driver_laminar_mp.F90 -- distributed driver: a DISTRIBUTED
! LAMINAR POISEUILLE SOLVE: assemble the validated pieces (setup + time loop +
! mean/constant-mass-flux + watchdog reduction) into a real run, and check both
!   (1) distributed == single-GPU oracle  (bit-exact / round-off ceiling), and
!   (2) the PHYSICS: the evolved mean stays the analytic Poiseuille parabola.
!
! The laminar Poiseuille profile u0(y) = 1.5*Qtar*(1 - y^2) is the exact steady
! state of the constant-mass-flux mean equation (the CN viscous operator preserves
! a degree-2 polynomial exactly and the CFR holds the bulk velocity), with zero
! fluctuations. So starting from init_laminar (u0 = QC0*Qtar*(T0 - T2),
! QC0 = 3/4; v = eta = w0 = 0) and stepping must REPRODUCE it.  This exercises the
! whole driver back-end: per-rank setup (helm_band / stepper_ev with the kx-slice /
! mean_flux), the rank-0-local mean + mass-flux solve inside par_step_full, the
! distributed nonlinear (which must return ~0 for a x,z-uniform mean), and a
! per-run finiteness watchdog (field_sumabs_c -> par_reduce_SUM, the NaN-propagating
! safety reduction the driver uses).
!
! Setup is per-(kx,kz)-column or rank-0-local (no communication): each rank calls
! stepper_ev_setup with its own kxh(kxlo:kxhi) slice; the mean lives only on the
! owns_mode00 rank.  This is the same assembly main.F90 uses, minus I/O (laminar
! needs none: the IC is generated in-code, the output is a profile to stdout).
!
! Checks (Ns steps, Re=10, dt=5e-3, Qtar=1):
!   eta,v : slab columns == oracle, per kx-slab, every rank (tol merge(1e-9,1e-3));
!   mean  : u0,w0 == oracle, on the owns_mode00 rank (same ceiling);
!   PHYS  : on the owner, the evolved real(u0) == the analytic parabola
!           (QC0*Qtar at T0, -QC0*Qtar at T2), bulk velocity == Qtar (the CFR is
!           dp-exact, tol 1e-9), Re_tau == sqrt(3*Re*Qtar)=sqrt(30) (tol ceiling);
!   WATCH : the |v| watchdog sum is finite (NaN/Inf guard).
! Margins are PRINTED.  The nonlinear term carries the tp tier so run dp AND sp.
! par_reduce_and fails the job on any rank.  Precondition P <= nkx = 17.
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_driver_laminar_mp   Run: bash run_all_par_tests.sh run -k test_driver_laminar_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!   module load nvhpc openmpi nccl
! Build: TP=dp|sp bash scripts/build/build_par.sh test_driver_laminar_mp   Run: bash run_all_par_tests.sh run -k test_driver_laminar_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_driver_laminar_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,             only: tp, ip, dp
  use cudafor
  use decomp_mod,        only: decomp_t, decomp_setup
  use helm_band_mod,     only: helm_band_setup
  use mean_mod,          only: mean_flux_setup, bulk_velocity
  use stepper_lin_mod,   only: stepper_ev_setup
  use init_mod,          only: init_laminar
  use stats_mod,         only: re_tau, wall_shear, field_sumabs_c
  use nonlinear_mod,     only: nl_workspace, nl_setup, nl_destroy
  use stepper_mod,       only: step_full
  use par_nonlinear_mod, only: par_nl_workspace, par_nl_setup, par_nl_destroy
  use par_stepper_mod,   only: par_step_full
  use solver_ops_mod,    only: band_ops_t, vsolve_ops_t, mean_ops_t, &
                               band_ops_alloc, band_ops_free, vsolve_ops_alloc, &
                               vsolve_ops_free, mean_ops_alloc, mean_ops_free
  use par_mod
  implicit none

  integer(ip), parameter :: n = 16, nx = 32, nz = 16, Ns = 20
  integer(ip), parameter :: ny = n + 1, nkx = nx/2+1
  real(dp),    parameter :: Re = 10.0_dp, dt = 5.0e-3_dp, Qtar = 1.0_dp
  real(dp),    parameter :: QC0 = 0.75_dp     ! laminar T0/T2 coeff (matches init_mod)
  real(real64) :: tol, tol_prof, tol_bulk

  type(decomp_t)  :: d
  integer(ip) :: bnkx, kxlo, kxhi, ix, iz, kk, k
  integer     :: istat
  logical     :: ok_e, ok_v, ok_m, ok_phys, ok_watch, all_ok

  real(dp) :: kxh(nkx), kzh(nz)
  type(band_ops_t) :: bo          ! y-operator band structure (host+device)
  type(mean_ops_t) :: mo          ! CFR mean precompute
  real(dp) :: u0re(0:n), u0an(0:n), ub, wsh, ret, watch_loc
  real(dp) :: ub_an, wsh_an, ret_an          ! analytic targets (sqrt -> runtime, not a parameter)
  real(real64) :: prof_margin, bulk_margin, retau_margin, watch_sum

  complex(dp), allocatable :: eF(:,:,:), vF(:,:,:), eS(:,:,:), vS(:,:,:)
  complex(dp) :: u0F(0:n), w0F(0:n), u0S(0:n), w0S(0:n)

  call par_init()
  tol      = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)   ! slab==oracle ceiling (nonlinear tier)
  tol_prof = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)   ! profile/Re_tau vs analytic (sp drifts via NL)
  tol_bulk = 1.0e-9_real64                                       ! bulk velocity: CFR is dp-exact (tp-independent)
  ub_an = Qtar;  wsh_an = 3.0_dp*Qtar;  ret_an = sqrt(3.0_dp*Re*Qtar)   ! parabola: Re_tau=sqrt(30)
  if (par_rank == 0) print '(a,i0,a)', '[driver-laminar] world size = ', par_nproc, ' rank(s)'

  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
  bnkx = d%b_nkx_loc;  kxlo = d%b_kx0 + 1;  kxhi = d%b_kx0 + bnkx

  do ix = 1, nkx
     kxh(ix) = real(ix-1, dp)
  end do
  do iz = 1, nz
     kk = iz-1
     if (kk > nz/2) kk = kk - nz
     kzh(iz) = real(kk, dp)
  end do
  call band_ops_alloc(bo, n);  call mean_ops_alloc(mo, n)
  call helm_band_setup(n, bo%sdh, bo%sth, bo%bdh, bo%boff)
  call mean_flux_setup(n, dt, Re, bo%sdh, bo%sth, bo%bdh, bo%boff, mo%Uhm, mo%Ih)

  allocate(eF(0:n,nkx,nz), vF(0:n,nkx,nz))

  ! ===== ORACLE (single-GPU step_full, full field, redundant per rank) =========
  call run_full(eF, vF, u0F, w0F)

  ! ===== DISTRIBUTED (par_step_full on this rank's kx-slab) ====================
  ok_e = .true.;  ok_v = .true.;  ok_m = .true.;  ok_phys = .true.
  u0S = (0.0_dp,0.0_dp);  w0S = (0.0_dp,0.0_dp);  watch_loc = 0.0_dp
  if (bnkx > 0) then
     allocate(eS(0:n,bnkx,nz), vS(0:n,bnkx,nz))
     call run_slab(eS, vS, u0S, w0S, watch_loc)
     ok_e = close_c(eS, eF(:, kxlo:kxhi, :), tol)
     ok_v = close_c(vS, vF(:, kxlo:kxhi, :), tol)
     deallocate(eS, vS)
  end if

  ! the mean lives ONLY on the kx=0 owner: compare it there and check the physics.
  prof_margin = 0.0_real64;  bulk_margin = 0.0_real64;  retau_margin = 0.0_real64
  ub = ub_an;  wsh = wsh_an;  ret = ret_an                       ! defaults (overwritten on the owner)
  if (d%owns_mode00) then
     ok_m = close_c1(u0S, u0F, tol) .and. close_c1(w0S, w0F, tol)
     u0re = real(u0S, dp)
     u0an = 0.0_dp;  u0an(0) = QC0*Qtar;  u0an(2) = -QC0*Qtar     ! analytic parabola coeffs
     do k = 0, n
        prof_margin = max(prof_margin, real(abs(u0re(k) - u0an(k)), real64))
     end do
     ub  = bulk_velocity(n, u0re)
     wsh = wall_shear(n, u0re)
     ret = re_tau(n, Re, u0re)
     bulk_margin  = real(abs(ub  - ub_an),  real64)
     retau_margin = real(abs(ret - ret_an), real64)
     ok_phys = (prof_margin <= tol_prof) .and. (bulk_margin <= tol_bulk) &
               .and. (retau_margin <= tol_prof) &
               .and. (prof_margin == prof_margin) .and. (wsh == wsh)
  end if

  ! finiteness watchdog: |v| sum across ranks must be finite (laminar -> ~0).
  watch_sum = real(watch_loc, real64)
  call par_reduce_sum(watch_sum)
  ok_watch = (watch_sum == watch_sum) .and. (watch_sum < huge(1.0_real64))

  call par_reduce_and(ok_e)
  call par_reduce_and(ok_v)
  call par_reduce_and(ok_m)
  call par_reduce_and(ok_phys)
  call par_reduce_and(ok_watch)
  all_ok = ok_e .and. ok_v .and. ok_m .and. ok_phys .and. ok_watch

  if (par_rank == 0) then
     ! rank 0 is always the kx=0 owner (block 0 starts at index 0), so ub/ret are set
     print '(a,i0,a,f6.3,a,f7.4)', '   laminar solve: Ns=', Ns, ' steps, Re=', Re, ', Qtar=', Qtar
     print '(a,es10.2,a)',        '   max|u0 - parabola|         = ', prof_margin, '   (tol 1e-9/1e-3)'
     print '(a,es10.2,a,f9.6,a)', '   |bulk - Qtar|              = ', bulk_margin, '   (bulk=',   ub,  ', tol 1e-9)'
     print '(a,es10.2,a,f8.4,a)', '   |Re_tau - sqrt(3*Re*Qtar)| = ', retau_margin, '   (Re_tau=', ret, ', =sqrt(30)=5.4772)'
     print '(a,es10.2)',          '   watchdog sum|v| (finite)   = ', watch_sum
     call report('eta:  distributed step_full == single-GPU oracle (per kx-slab)', ok_e)
     call report('v:    distributed step_full == single-GPU oracle (per kx-slab)', ok_v)
     call report('mean: u0,w0 == single-GPU oracle (on the kx=0 owner)',           ok_m)
     call report('PHYSICS: evolved mean == analytic Poiseuille parabola + bulk/Re_tau', ok_phys)
     call report('watchdog: |v| reduction finite (NaN/Inf guard)',                 ok_watch)
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

  ! single-GPU oracle: laminar IC, evolve Ns full steps on the FULL field.
  subroutine run_full(eout, vout, u0out, w0out)
    complex(dp), intent(out) :: eout(0:n,nkx,nz), vout(0:n,nkx,nz), u0out(0:n), w0out(0:n)
    type(nl_workspace) :: ws
    real(dp),    device, allocatable :: kx(:), kz(:)
    type(vsolve_ops_t) :: vo                ! local v-solve apparatus
    complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:), Nvp(:,:,:), Nep(:,:,:)
    integer(ip) :: it
    call nl_setup(ws, n+1, nx, nz, kxh, kzh)
    allocate(kx(nkx), kz(nz))
    call vsolve_ops_alloc(vo, n, nkx, nz)
    allocate(eta(0:n,nkx,nz), v(0:n,nkx,nz), u0(0:n), w0(0:n), Nvp(0:n,nkx,nz), Nep(0:n,nkx,nz))
    kx = kxh;  kz = kzh;  bo%sd = bo%sdh;  bo%st = bo%sth;  bo%bd = bo%bdh
    call stepper_ev_setup(n, nkx, nz, dt, Re, kxh, kzh, vo%vm, vo%vp, vo%Cinv)
    call init_laminar(n, nkx, nz, Qtar, v, eta, u0, w0)     ! v=eta=w0=0, u0=parabola
    Nvp = (0.0_dp,0.0_dp);  Nep = (0.0_dp,0.0_dp)
    do it = 1, Ns
       call step_full(ws, n, nkx, nz, dt, Re, kx, kz, bo, vo, mo, &
                      Qtar, u0, w0, eta, v, Nvp, Nep)
    end do
    istat = cudaDeviceSynchronize()
    eout = eta;  vout = v;  u0out = u0;  w0out = w0
    call vsolve_ops_free(vo)
    deallocate(kx, kz, eta, v, u0, w0, Nvp, Nep)
    call nl_destroy(ws)
  end subroutine run_full

  ! distributed: laminar IC, evolve Ns full steps on THIS rank's kx-slab.
  subroutine run_slab(eout, vout, u0out, w0out, watch_out)
    complex(dp), intent(out) :: eout(0:n,bnkx,nz), vout(0:n,bnkx,nz), u0out(0:n), w0out(0:n)
    real(dp),    intent(out) :: watch_out
    type(par_nl_workspace) :: ws
    real(dp),    device, allocatable :: kx(:), kz(:)
    type(vsolve_ops_t) :: vo                ! local v-solve apparatus (slab)
    complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:), Nvp(:,:,:), Nep(:,:,:)
    integer(ip) :: it
    call par_nl_setup(ws, d, n+1, nx, nz, kxh, kzh)
    allocate(kx(bnkx), kz(nz))
    call vsolve_ops_alloc(vo, n, bnkx, nz)
    allocate(eta(0:n,bnkx,nz), v(0:n,bnkx,nz), u0(0:n), w0(0:n), Nvp(0:n,bnkx,nz), Nep(0:n,bnkx,nz))
    kx = kxh(kxlo:kxhi);  kz = kzh;  bo%sd = bo%sdh;  bo%st = bo%sth;  bo%bd = bo%bdh
    call stepper_ev_setup(n, bnkx, nz, dt, Re, kxh(kxlo:kxhi), kzh, vo%vm, vo%vp, vo%Cinv)
    call init_laminar(n, bnkx, nz, Qtar, v, eta, u0, w0)     ! slab v=eta=0, u0=parabola
    if (.not. d%owns_mode00) then
       u0 = (0.0_dp,0.0_dp);  w0 = (0.0_dp,0.0_dp)            ! mean is rank-0-local
    end if
    Nvp = (0.0_dp,0.0_dp);  Nep = (0.0_dp,0.0_dp)
    do it = 1, Ns
       call par_step_full(ws, n, bnkx, nz, dt, Re, kx, kz, bo, vo, mo, &
                          Qtar, u0, w0, eta, v, Nvp, Nep)
    end do
    istat = cudaDeviceSynchronize()
    call field_sumabs_c(ny, bnkx, nz, v, watch_out)          ! |v| watchdog (laminar -> ~0)
    eout = eta;  vout = v;  u0out = u0;  w0out = w0
    call vsolve_ops_free(vo)
    deallocate(kx, kz, eta, v, u0, w0, Nvp, Nep)
    call par_nl_destroy(ws)
  end subroutine run_slab

  logical function close_c(a, b, tol) result(okk)
    complex(dp),  intent(in) :: a(:,:,:), b(:,:,:)
    real(real64), intent(in) :: tol
    okk = all(a == a) .and. all(b == b) .and. &
          (real(maxval(abs(a - b)), real64) <= tol)
  end function close_c

  logical function close_c1(a, b, tol) result(okk)
    complex(dp),  intent(in) :: a(:), b(:)
    real(real64), intent(in) :: tol
    okk = all(a == a) .and. all(b == b) .and. &
          (real(maxval(abs(a - b)), real64) <= tol)
  end function close_c1

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

end program test_driver_laminar_mp
