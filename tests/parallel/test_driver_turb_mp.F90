!===============================================================================
! test_driver_turb_mp.F90 -- distributed driver: the TURBULENT
! validation rung: the FULL MKM machinery on a genuinely 3-D turbulent field,
! distributed vs the single-GPU oracle.
!
! init_turbulent (laminar parabola + a divergence-free, BC-satisfying random
! perturbation seeded across ALL (kx,kz) modes within |k|<=kcut, exact volume-rms
! eps, kx=0 plane Hermitian-symmetrised) -> evolve Ns steps with par_step_full ->
! recover_uw -> par_to_physical x3 -> stats_accumulate -> par_stats_finalize. It
! composes every distributed piece (stepper, transforms, statistics) on the
! hardest input: a fully populated turbulent spectrum, the IC the
! production MKM run (examples/input_kmm.nml) uses. Same Re/eps/kcut/seed as the
! headline case, on a smaller/cheaper grid (a correctness rung, not converged
! statistics).
!
! Checks (vs the single-GPU oracle, every rank builds the SAME full IC by fixed
! seed then slices its kx-slab):
!   eta,v : par_step_full slab == step_full oracle, per kx-slab (tol merge(1e-9,1e-3));
!   mean  : u0,w0 == oracle on the kx=0 owner;
!   STATS : mean_u/v/w, <u'u'>,<v'v'>,<w'w'>,<u'v'>(y) and Re_tau from the
!           distributed (slab + Allreduce SUM) path == the single-GPU stats, on the
!           REAL evolved turbulent physical field. The stats go through the TRANSFORM
!           (tp tier) so they match to the round-off ceiling merge(1e-9,1e-3) (dp
!           ~1e-11, sp ~1e-4 -- the field values differ at transform round-off; the
!           cross-rank SUM itself is exact-to-roundoff). Margins are PRINTED.
!
! GRID n=32 / nx=32 (nkx=17) / nz=16 (nzd=24): all modes active, no empty kx- or
! dealiased-z slab at P<=16 (stepper/stats need nkx>=P and nzd>=P). Re=2800,
! Qtar=1, dt=5e-3, eps=0.10, kcut=8, seed=12345 (input_kmm.nml values). Ns=5 (the
! field stays bounded -- transition needs t~100, not 5 steps). par_reduce_and fails
! the job on any rank. dp AND sp. Precondition P<=16.
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_driver_turb_mp   Run: bash run_all_par_tests.sh run -k test_driver_turb_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!   module load nvhpc openmpi nccl
! Build: TP=dp|sp bash scripts/build/build_par.sh test_driver_turb_mp   Run: bash run_all_par_tests.sh run -k test_driver_turb_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_driver_turb_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,              only: tp, wp, dp, ip
  use cudafor
  use decomp_mod,         only: decomp_t, decomp_setup
  use helm_band_mod,      only: helm_band_setup
  use mean_mod,           only: mean_flux_setup
  use solver_ops_mod,     only: band_ops_t, vsolve_ops_t, mean_ops_t, &
                                band_ops_alloc, band_ops_free, vsolve_ops_alloc, &
                                vsolve_ops_free, mean_ops_alloc, mean_ops_free
  use stepper_lin_mod,    only: stepper_ev_setup
  use init_mod,           only: init_turbulent
  use velocity_mod,       only: recover_uw
  use stats_mod,          only: stats_t, stats_init, stats_accumulate, stats_finalize, &
                                stats_destroy, re_tau
  use spectral3d_mod,     only: s3d_plans, s3d_setup_da_r2c, s3d_destroy_r2c, to_physical_da_r2c
  use par_spectral3d_mod, only: par_s3d_plans, par_s3d_setup_da_r2c, &
                                par_s3d_destroy_da_r2c, par_to_physical_da_r2c
  use nonlinear_mod,      only: nl_workspace, nl_setup, nl_destroy
  use stepper_mod,        only: step_full
  use par_nonlinear_mod,  only: par_nl_workspace, par_nl_setup, par_nl_destroy
  use par_stepper_mod,    only: par_step_full
  use par_stats_mod,      only: par_stats_finalize
  use par_mod
  implicit none

  integer(ip), parameter :: n = 32, nx = 32, nz = 16, Ns = 5
  integer(ip), parameter :: ny = n+1, nkx = nx/2+1, nxd = 3*nx/2, nzd = 3*nz/2, nkxd = nxd/2+1
  real(dp),    parameter :: Re = 2800.0_dp, Qtar = 1.0_dp, dt = 5.0e-3_dp
  real(dp),    parameter :: epsp = 0.10_dp, PI = acos(-1.0_dp), Lx = 4.0_dp*PI, Lz = 4.0_dp*PI/3.0_dp
  integer(ip), parameter :: kcut = 8, pseed = 12345

  type(decomp_t) :: d
  integer(ip) :: bnkx, kxlo, kxhi, anz, ix, iz, kk, j
  integer     :: istat
  logical     :: ok_e, ok_v, ok_m, ok_st, all_ok

  real(dp) :: kxh(nkx), kzh(nz)
  type(band_ops_t) :: bo          ! y-operator band structure (host+device)
  type(mean_ops_t) :: mo          ! CFR mean precompute
  real(real64) :: tol, tol_st, prof_margin, retau_margin

  complex(dp), allocatable :: he0(:,:,:), hv0(:,:,:)        ! full turbulent IC (host, rank-invariant)
  complex(dp) :: hu00(0:n), hw00(0:n)                       ! mean columns of the IC
  complex(dp), allocatable :: eF(:,:,:), vF(:,:,:), eS(:,:,:), vS(:,:,:)
  complex(dp) :: u0F(0:n), w0F(0:n), u0S(0:n), w0S(0:n)
  real(dp) :: muF(0:n), mvF(0:n), mwF(0:n), uuF(0:n), vvF(0:n), wwF(0:n), uvF(0:n), retF
  real(dp) :: muS(0:n), mvS(0:n), mwS(0:n), uuS(0:n), vvS(0:n), wwS(0:n), uvS(0:n), retS

  call par_init()
  tol    = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)    ! field: slab==oracle (nonlinear tier)
  tol_st = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)    ! stats: transform tier (dp~1e-11/sp~1e-4)
  if (par_rank == 0) print '(a,i0,a)', '[driver-turb] world size = ', par_nproc, ' rank(s)'

  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
  bnkx = d%b_nkx_loc;  kxlo = d%b_kx0 + 1;  kxhi = d%b_kx0 + bnkx;  anz = d%ad_nz_loc

  ! physical wavenumbers (production box); init_turbulent + recover_uw use them
  do ix = 1, nkx
     kxh(ix) = real(ix-1, dp) * (2.0_dp*PI/Lx)
  end do
  do iz = 1, nz
     kk = iz-1;  if (kk > nz/2) kk = kk - nz;  kzh(iz) = real(kk, dp) * (2.0_dp*PI/Lz)
  end do
  call band_ops_alloc(bo, n);  call mean_ops_alloc(mo, n)
  call helm_band_setup(n, bo%sdh, bo%sth, bo%bdh, bo%boff)
  call mean_flux_setup(n, dt, Re, bo%sdh, bo%sth, bo%bdh, bo%boff, mo%Uhm, mo%Ih)

  allocate(he0(0:n,nkx,nz), hv0(0:n,nkx,nz))
  allocate(eF(0:n,nkx,nz), vF(0:n,nkx,nz), eS(0:n,bnkx,nz), vS(0:n,bnkx,nz))
  call build_ic(he0, hv0, hu00, hw00)         ! the SAME full turbulent IC on every rank

  call run_full(muF, mvF, mwF, uuF, vvF, wwF, uvF, retF, eF, vF, u0F, w0F)
  ok_e = .true.;  ok_v = .true.;  ok_m = .true.
  u0S = (0.0_dp,0.0_dp);  w0S = (0.0_dp,0.0_dp)
  if (bnkx > 0) then
     call run_slab(muS, mvS, mwS, uuS, vvS, wwS, uvS, retS, eS, vS, u0S, w0S)
     ok_e = close_c(eS, eF(:, kxlo:kxhi, :), tol)
     ok_v = close_c(vS, vF(:, kxlo:kxhi, :), tol)
  end if
  if (d%owns_mode00) ok_m = close_c1(u0S, u0F, tol) .and. close_c1(w0S, w0F, tol)

  ! statistics: every rank holds the global profiles (par_stats_finalize Allreduces);
  ! compare to the single-GPU oracle's profiles (max abs diff over all 7).
  prof_margin = 0.0_real64
  do j = 0, n
     prof_margin = max(prof_margin, &
          real(abs(muS(j)-muF(j)),real64), real(abs(mvS(j)-mvF(j)),real64), &
          real(abs(mwS(j)-mwF(j)),real64), real(abs(uuS(j)-uuF(j)),real64), &
          real(abs(vvS(j)-vvF(j)),real64), real(abs(wwS(j)-wwF(j)),real64), &
          real(abs(uvS(j)-uvF(j)),real64))
  end do
  ok_st = (prof_margin <= tol_st) .and. (prof_margin == prof_margin) &
          .and. (maxval(uuF) > 1.0e-6_dp)               ! profiles are global -> checked on every rank
  ! Re_tau comes from the (0,0) MEAN column, which lives ONLY on the kx=0 owner
  ! (non-owners zero u0 -> retS=0); so compare it there, like the mean field.
  retau_margin = 0.0_real64
  if (d%owns_mode00) then
     retau_margin = real(abs(retS - retF), real64)
     ok_st = ok_st .and. (retau_margin <= tol_st)
  end if

  call par_reduce_and(ok_e)
  call par_reduce_and(ok_v)
  call par_reduce_and(ok_m)
  call par_reduce_and(ok_st)
  all_ok = ok_e .and. ok_v .and. ok_m .and. ok_st

  if (par_rank == 0) then
     print '(a,i0,a,f7.1,a,f5.2)', '   turbulent IC (init_turbulent): Ns=', Ns, ' steps, Re=', Re, ', eps=', epsp
     print '(a,es10.2,a,f8.5,a)', '   max|stat_dist - stat_ref|  = ', prof_margin, &
          '   (max uu=', maxval(uuF), ', tol 1e-9/1e-3)'
     print '(a,es10.2,a,f8.4,a)', '   |Re_tau_dist - Re_tau_ref| = ', retau_margin, &
          '   (Re_tau=', retF, ', tol 1e-9/1e-3)'
     call report('eta:  distributed step_full == single-GPU oracle (per kx-slab)',  ok_e)
     call report('v:    distributed step_full == single-GPU oracle (per kx-slab)',  ok_v)
     call report('mean: u0,w0 == single-GPU oracle (on the kx=0 owner)',            ok_m)
     call report('STATS: distributed Reynolds stresses + Re_tau == single-GPU',     ok_st)
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

  ! Build the full turbulent IC once on host (identical on every rank: init_turbulent
  ! is seeded by the fixed pseed). Done via a throwaway full device array -> D2H.
  subroutine build_ic(he, hv, hu0, hw0)
    complex(dp), intent(out) :: he(0:n,nkx,nz), hv(0:n,nkx,nz), hu0(0:n), hw0(0:n)
    complex(wp), device, allocatable :: vd(:,:,:), ed(:,:,:), u0d(:), w0d(:)
    allocate(vd(0:n,nkx,nz), ed(0:n,nkx,nz), u0d(0:n), w0d(0:n))
    call init_turbulent(n, nkx, nz, kxh, kzh, Qtar, epsp, pseed, kcut, vd, ed, u0d, w0d)
    istat = cudaDeviceSynchronize()
    hv = vd;  he = ed;  hu0 = u0d;  hw0 = w0d                ! D2H (wp=dp)
    deallocate(vd, ed, u0d, w0d)
  end subroutine build_ic

  ! complex(dp) -> complex(tp) elementwise demote (transform input must be tp).
  subroutine demote(m, nxl, nzl, src, dst)
    integer(ip), intent(in) :: m, nxl, nzl
    complex(dp), device :: src(0:m,nxl,nzl)
    complex(tp), device :: dst(0:m,nxl,nzl)
    integer(ip) :: i, jj, kk2
    !$cuf kernel do(3) <<<*,*>>>
    do kk2 = 1, nzl;  do jj = 1, nxl;  do i = 0, m
       dst(i,jj,kk2) = src(i,jj,kk2)
    end do; end do; end do
  end subroutine demote

  ! single-GPU oracle: evolve the full turbulent IC Ns steps, then one stats sample.
  subroutine run_full(mu, mv, mw, uu, vv, ww, uv, ret, eout, vout, u0out, w0out)
    real(dp),    intent(out) :: mu(0:n), mv(0:n), mw(0:n), uu(0:n), vv(0:n), ww(0:n), uv(0:n), ret
    complex(dp), intent(out) :: eout(0:n,nkx,nz), vout(0:n,nkx,nz), u0out(0:n), w0out(0:n)
    type(nl_workspace) :: ws
    type(s3d_plans)    :: orpl
    type(stats_t)      :: sa
    real(dp),    device, allocatable :: kx(:), kz(:)
    type(vsolve_ops_t) :: vo                ! local v-solve apparatus
    real(wp),    device, allocatable :: kxt(:), kzt(:)            ! wp
    real(tp),    device, allocatable :: pu(:,:,:), pv(:,:,:), pw(:,:,:)
    complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:), Nvp(:,:,:), Nep(:,:,:)
    complex(tp), device, allocatable :: uhat(:,:,:), what(:,:,:), dvt(:,:,:), vc(:,:,:), Wf(:,:,:)
    complex(dp) :: u0h(0:n)
    integer(ip) :: it
    call nl_setup(ws, n+1, nx, nz, kxh, kzh)
    call s3d_setup_da_r2c(orpl, ny, nx, nz)
    call stats_init(sa, n, nxd, nzd)
    allocate(kx(nkx), kz(nz), kxt(nkx), kzt(nz))
    call vsolve_ops_alloc(vo, n, nkx, nz)
    allocate(eta(0:n,nkx,nz), v(0:n,nkx,nz), u0(0:n), w0(0:n), Nvp(0:n,nkx,nz), Nep(0:n,nkx,nz))
    allocate(uhat(0:n,nkx,nz), what(0:n,nkx,nz), dvt(0:n,nkx,nz), vc(0:n,nkx,nz))
    allocate(Wf(ny,nkxd,nzd), pu(ny,nxd,nzd), pv(ny,nxd,nzd), pw(ny,nxd,nzd))
    kx = kxh;  kz = kzh;  kxt = kxh;  kzt = kzh
    bo%sd = bo%sdh;  bo%st = bo%sth;  bo%bd = bo%bdh
    call stepper_ev_setup(n, nkx, nz, dt, Re, kxh, kzh, vo%vm, vo%vp, vo%Cinv)
    eta = he0;  v = hv0;  u0 = hu00;  w0 = hw00
    Nvp = (0.0_dp,0.0_dp);  Nep = (0.0_dp,0.0_dp)
    do it = 1, Ns
       call step_full(ws, n, nkx, nz, dt, Re, kx, kz, bo, vo, mo, &
                      Qtar, u0, w0, eta, v, Nvp, Nep)
    end do
    call recover_uw(ny, nkx, nz, kxt, kzt, v, eta, u0, w0, uhat, what, dvt)
    call demote(n, nkx, nz, v, vc)
    call to_physical_da_r2c(orpl, ny, nx, nz, nxd, nzd, uhat, pu, Wf)
    call to_physical_da_r2c(orpl, ny, nx, nz, nxd, nzd, vc,   pv, Wf)
    call to_physical_da_r2c(orpl, ny, nx, nz, nxd, nzd, what, pw, Wf)
    call stats_accumulate(sa, n, nxd, nzd, pu, pv, pw)
    call stats_finalize(sa, n, mu, mv, mw, uu, vv, ww, uv)
    istat = cudaDeviceSynchronize()
    u0h = u0;  ret = re_tau(n, Re, real(u0h, dp))
    eout = eta;  vout = v;  u0out = u0;  w0out = w0
    call s3d_destroy_r2c(orpl);  call nl_destroy(ws);  call stats_destroy(sa)
    call vsolve_ops_free(vo)
    deallocate(kx, kz, kxt, kzt, eta, v, u0, w0, &
               Nvp, Nep, uhat, what, dvt, vc, Wf, pu, pv, pw)
  end subroutine run_full

  ! distributed: evolve this rank's kx-slab Ns steps, then one stats sample
  ! (recover -> par_to_physical -> stats_accumulate -> par_stats_finalize).
  subroutine run_slab(mu, mv, mw, uu, vv, ww, uv, ret, eout, vout, u0out, w0out)
    real(dp),    intent(out) :: mu(0:n), mv(0:n), mw(0:n), uu(0:n), vv(0:n), ww(0:n), uv(0:n), ret
    complex(dp), intent(out) :: eout(0:n,bnkx,nz), vout(0:n,bnkx,nz), u0out(0:n), w0out(0:n)
    type(par_nl_workspace) :: ws
    type(par_s3d_plans)    :: pp
    type(stats_t)          :: sa
    real(dp),    device, allocatable :: kx(:), kz(:)
    type(vsolve_ops_t) :: vo                ! local v-solve apparatus
    real(wp),    device, allocatable :: kxt(:), kzt(:)            ! wp
    real(tp),    device, allocatable :: pu(:,:,:), pv(:,:,:), pw(:,:,:)
    complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:), Nvp(:,:,:), Nep(:,:,:)
    complex(tp), device, allocatable :: uhat(:,:,:), what(:,:,:), dvt(:,:,:), vc(:,:,:)
    complex(tp), device, allocatable :: Bz(:,:,:,:), Az(:,:,:,:), Af(:,:,:)
    complex(dp) :: u0h(0:n)
    integer(ip) :: it
    call par_nl_setup(ws, d, n+1, nx, nz, kxh, kzh)
    call par_s3d_setup_da_r2c(pp, d, ny, nx, nz)
    call stats_init(sa, n, nxd, nzd)                          ! GLOBAL geometry
    allocate(kx(bnkx), kz(nz), kxt(bnkx), kzt(nz))
    call vsolve_ops_alloc(vo, n, bnkx, nz)
    allocate(eta(0:n,bnkx,nz), v(0:n,bnkx,nz), u0(0:n), w0(0:n), Nvp(0:n,bnkx,nz), Nep(0:n,bnkx,nz))
    allocate(uhat(0:n,bnkx,nz), what(0:n,bnkx,nz), dvt(0:n,bnkx,nz), vc(0:n,bnkx,nz))
    allocate(pu(ny,nxd,anz), pv(ny,nxd,anz), pw(ny,nxd,anz))
    allocate(Bz(ny,bnkx,nzd,1), Az(ny,nkx,anz,1), Af(ny,nkxd,anz))
    kx = kxh(kxlo:kxhi);  kz = kzh
    bo%sd = bo%sdh;  bo%st = bo%sth;  bo%bd = bo%bdh
    kxt = kxh(kxlo:kxhi);  kzt = kzh
    call stepper_ev_setup(n, bnkx, nz, dt, Re, kxh(kxlo:kxhi), kzh, vo%vm, vo%vp, vo%Cinv)
    eta = he0(:,kxlo:kxhi,:);  v = hv0(:,kxlo:kxhi,:)
    if (d%owns_mode00) then
       u0 = hu00;  w0 = hw00
    else
       u0 = (0.0_dp,0.0_dp);  w0 = (0.0_dp,0.0_dp)            ! mean is rank-0-local
    end if
    Nvp = (0.0_dp,0.0_dp);  Nep = (0.0_dp,0.0_dp)
    do it = 1, Ns
       call par_step_full(ws, n, bnkx, nz, dt, Re, kx, kz, bo, vo, mo, &
                          Qtar, u0, w0, eta, v, Nvp, Nep)
    end do
    call recover_uw(ny, bnkx, nz, kxt, kzt, v, eta, u0, w0, uhat, what, dvt)
    call demote(n, bnkx, nz, v, vc)
    call par_to_physical_da_r2c(pp, d, ny, nx, nz, nxd, nzd, bnkx, anz, uhat, Bz, Az, Af, pu)
    call par_to_physical_da_r2c(pp, d, ny, nx, nz, nxd, nzd, bnkx, anz, vc,   Bz, Az, Af, pv)
    call par_to_physical_da_r2c(pp, d, ny, nx, nz, nxd, nzd, bnkx, anz, what, Bz, Az, Af, pw)
    call stats_accumulate(sa, n, nxd, anz, pu, pv, pw)        ! LOCAL plane count
    call par_stats_finalize(sa, n, mu, mv, mw, uu, vv, ww, uv)
    istat = cudaDeviceSynchronize()
    u0h = u0;  ret = re_tau(n, Re, real(u0h, dp))             ! pure fn of the (0,0) column (owner)
    eout = eta;  vout = v;  u0out = u0;  w0out = w0
    call par_s3d_destroy_da_r2c(pp);  call par_nl_destroy(ws);  call stats_destroy(sa)
    call vsolve_ops_free(vo)
    deallocate(kx, kz, kxt, kzt, eta, v, u0, w0, &
               Nvp, Nep, uhat, what, dvt, vc, pu, pv, pw, Bz, Az, Af)
  end subroutine run_slab

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

end program test_driver_turb_mp
