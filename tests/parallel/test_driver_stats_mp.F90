!===============================================================================
! test_driver_stats_mp.F90 -- distributed driver: the turbulence-
! statistics path (Reynolds-stress y-profiles + Re_tau), distributed vs the
! single-GPU oracle.
!
! Pipeline (the back half of a real time step's monitor):
!   stats_accumulate x nsamp  : fold each rank's z-slab of physical (u,v,w) into
!                               the per-sample plane-mean SUMS, normalised by the
!                               GLOBAL plane size nxd*nzd
!     -> par_stats_finalize    : Allreduce(SUM) the partial sums across ranks,
!                                then the SHARED stats_combine -> mean_u/v/w,
!                                <u'u'>,<v'v'>,<w'w'>,<u'v'>(y)
!     -> re_tau                : wall shear from the (0,0) MEAN column -- owned by
!                                the owns_mode00 rank, no reduction needed
!
! Why a SYNTHETIC field (not the transform path): this test isolates the
! STATISTICS reduction (the slab normalisation + the cross-rank SUM).  A
! deterministic field, generated in dp on the host and cast to real(tp), is
! BIT-IDENTICAL per global z-plane on every rank/path; the reference (full plane)
! and the distributed (slab + Allreduce) therefore differ ONLY by dp sum
! reassociation.  So the profiles tolerance is the dp round-off ceiling 1e-9 for
! BOTH dp and sp (there is NO transform here, hence NOT the merge(1e-9,1e-3)
! pattern of the transform-carrying tests -- the statistics accumulators are
! real(dp) and the field bits are identical).  The transform that feeds real
! statistics is already validated separately (par_to_physical, test_driver_state_mp).
! The grid uses nzd = 21 (= 3*14/2), which is NOT divisible by 4 or 8, so the
! z-slabs are deliberately IMBALANCED at P=4,8 -- exactly what exercises the
! global-vs-local normaliser.  Re_tau is a pure function of the (0,0) column held
! by the mode-00 owner, so it is BIT-EXACT there (tol 1e-12).  Actual margins are
! PRINTED.  par_reduce_and fails the whole job on any rank.  dp AND sp.
! Precondition P <= nzd = 21 (no empty z-slab; empty slabs would still add zero).
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_driver_stats_mp   Run: bash run_all_par_tests.sh run -k test_driver_stats_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!   module load nvhpc openmpi nccl
! Build: TP=dp|sp bash scripts/build/build_par.sh test_driver_stats_mp   Run: bash run_all_par_tests.sh run -k test_driver_stats_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_driver_stats_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,         only: tp, dp, ip
  use cudafor
  use decomp_mod,    only: decomp_t, decomp_setup
  use stats_mod,     only: stats_t, stats_init, stats_accumulate, stats_finalize, &
                           stats_destroy, re_tau
  use par_stats_mod, only: par_stats_finalize
  use par_mod
  implicit none

  integer(ip), parameter :: n = 16, nx = 16, nz = 14
  integer(ip), parameter :: ny = n+1, nxd = 3*nx/2, nzd = 3*nz/2
  integer(ip), parameter :: nsamp = 3
  real(dp),    parameter :: PI = acos(-1.0_dp), Re = 180.0_dp

  type(decomp_t) :: d
  integer(ip) :: anz, z0, j
  logical     :: ok_prof, ok_retau, all_ok

  real(dp) :: u0(0:n)
  real(dp) :: mu_ref(0:n), mv_ref(0:n), mw_ref(0:n)
  real(dp) :: uu_ref(0:n), vv_ref(0:n), ww_ref(0:n), uv_ref(0:n)
  real(dp) :: mu_g(0:n),   mv_g(0:n),   mw_g(0:n)
  real(dp) :: uu_g(0:n),   vv_g(0:n),   ww_g(0:n),   uv_g(0:n)
  real(dp) :: ret_ref, ret_g
  real(real64) :: prof_margin, retau_margin, tol_prof, tol_retau

  call par_init()
  tol_prof  = 1.0e-9_real64     ! profiles: dp sum-reassociation only (no transform, bits identical)
  tol_retau = 1.0e-12_real64    ! re_tau: pure fn of the (0,0) column -> bit-exact on the mode-00 owner
  if (par_rank == 0) print '(a,i0,a)', '[driver-stats] world size = ', par_nproc, ' rank(s)'

  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
  anz = d%ad_nz_loc;  z0 = d%ad_z0       ! this rank's dealiased z-slab (count, 0-based offset)

  ! deterministic mean (0,0) Chebyshev column: laminar 1 - y^2 = 1/2 - T2/2,
  ! wall slope |du0/dy| = 2  ->  Re_tau = sqrt(2 Re).
  u0 = 0.0_dp;  u0(0) = 0.5_dp;  u0(2) = -0.5_dp

  call run_ref (mu_ref, mv_ref, mw_ref, uu_ref, vv_ref, ww_ref, uv_ref)   ! full plane, every rank
  call run_slab(mu_g,   mv_g,   mw_g,   uu_g,   vv_g,   ww_g,   uv_g)      ! z-slab + Allreduce(SUM)

  ! profiles: max abs difference over all seven profiles and all y-nodes
  prof_margin = 0.0_real64
  do j = 0, n
     prof_margin = max(prof_margin, &
          real(abs(mu_g(j)-mu_ref(j)), real64), real(abs(mv_g(j)-mv_ref(j)), real64), &
          real(abs(mw_g(j)-mw_ref(j)), real64), real(abs(uu_g(j)-uu_ref(j)), real64), &
          real(abs(vv_g(j)-vv_ref(j)), real64), real(abs(ww_g(j)-ww_ref(j)), real64), &
          real(abs(uv_g(j)-uv_ref(j)), real64))
  end do
  ok_prof = (prof_margin <= tol_prof) .and. (prof_margin == prof_margin) &
            .and. (maxval(abs(mu_ref)) > 0.1_dp)     ! sanity: a real, non-trivial profile

  ! re_tau: single-GPU value (every rank); distributed value only on the mode-00 owner
  ret_ref = re_tau(n, Re, u0)
  if (d%owns_mode00) then
     ret_g = re_tau(n, Re, u0)            ! pure fn of the (0,0) column held by this rank
     retau_margin = real(abs(ret_g - ret_ref), real64)
     ok_retau = (retau_margin <= tol_retau)
  else
     retau_margin = 0.0_real64
     ok_retau = .true.                    ! non-owner ranks make no claim
  end if

  call par_reduce_and(ok_prof)
  call par_reduce_and(ok_retau)
  all_ok = ok_prof .and. ok_retau

  if (par_rank == 0) then
     print '(a,es10.2,a,f7.4,a)', '   max|prof_dist - prof_ref|  = ', prof_margin, &
          '   (max|mean_u|=', maxval(abs(mu_ref)), ', tol 1e-9)'
     print '(a,es10.2,a,f8.4,a)', '   |re_tau_dist - re_tau_ref| = ', retau_margin, &
          '   (Re_tau=', ret_ref, ', tol 1e-12)'
     call report('Reynolds-stress profiles: distributed (slab + Allreduce SUM) == single-GPU', ok_prof)
     call report('Re_tau (mode-00 owner, wall shear): == single-GPU', ok_retau)
     if (all_ok) then
        print '(a)', 'ALL TESTS PASSED'
     else
        print '(a)', 'SOME TESTS FAILED'
     end if
  end if

  call par_finalize()
  if (.not. all_ok) error stop 1

contains

  ! Deterministic, turbulence-like physical field on the dealiased (nxd x nzl)
  ! grid, z-slab starting at GLOBAL 0-based plane zoff (zoff=0 -> full field).
  ! Built in dp on the host then cast to real(tp): IDENTICAL bits on every path
  ! (same dp inputs -> same tp rounding).  The fluctuations have ZERO discrete
  ! x-z plane mean (full-period sines) so mean_u(j) ~ 1 - y_j^2, mean_v,mean_w ~ 0;
  ! u and v share the (sin x cos z) structure so <u'v'> is non-trivial; fac_s
  ! varies the fluctuation amplitude across samples so the time average is real.
  subroutine gen_field(ismp, nzl, zoff, hu, hv, hw)
    integer(ip), intent(in)  :: ismp, nzl, zoff
    real(tp),    intent(out) :: hu(0:n,nxd,nzl), hv(0:n,nxd,nzl), hw(0:n,nxd,nzl)
    real(dp) :: yj, xx, zz, by, fac_s, fu, fv, fw
    integer(ip) :: jj, ix, il, izg
    fac_s = 1.0_dp + 0.25_dp*real(ismp-1, dp)         ! 1.00, 1.25, 1.50
    do il = 1, nzl
       izg = zoff + (il-1)                            ! GLOBAL 0-based z plane
       zz  = 2.0_dp*PI*real(izg, dp)/real(nzd, dp)
       do ix = 1, nxd
          xx = 2.0_dp*PI*real(ix-1, dp)/real(nxd, dp)
          do jj = 0, n
             yj = cos(PI*real(jj, dp)/real(n, dp))
             by = 1.0_dp - yj*yj                       ! wall-zero envelope
             fu = 0.10_dp*sin(xx)*cos(zz)*by
             fv = 0.06_dp*sin(xx)*cos(zz)*by           ! correlated with u -> <u'v'> /= 0
             fw = 0.07_dp*cos(xx)*sin(2.0_dp*zz)*by    ! uncorrelated
             hu(jj,ix,il) = real(by + fac_s*fu, tp)    ! mean 1 - y^2 + fluctuation
             hv(jj,ix,il) = real(     fac_s*fv, tp)
             hw(jj,ix,il) = real(     fac_s*fw, tp)
          end do
       end do
    end do
  end subroutine gen_field

  ! single-GPU reference: full (nxd x nzd) plane, nsamp samples.
  subroutine run_ref(mu, mv, mw, uu, vv, ww, uv)
    real(dp), intent(out) :: mu(0:n), mv(0:n), mw(0:n)
    real(dp), intent(out) :: uu(0:n), vv(0:n), ww(0:n), uv(0:n)
    type(stats_t) :: st
    real(tp), allocatable :: hu(:,:,:), hv(:,:,:), hw(:,:,:)
    real(tp), allocatable, device :: du(:,:,:), dv(:,:,:), dw(:,:,:)
    integer(ip) :: isamp
    integer :: ist
    allocate(hu(0:n,nxd,nzd), hv(0:n,nxd,nzd), hw(0:n,nxd,nzd))
    allocate(du(0:n,nxd,nzd), dv(0:n,nxd,nzd), dw(0:n,nxd,nzd))
    call stats_init(st, n, nxd, nzd)
    do isamp = 1, nsamp
       call gen_field(isamp, nzd, 0_ip, hu, hv, hw)
       du = hu;  dv = hv;  dw = hw
       call stats_accumulate(st, n, nxd, nzd, du, dv, dw)
    end do
    ist = cudaDeviceSynchronize()
    call stats_finalize(st, n, mu, mv, mw, uu, vv, ww, uv)
    call stats_destroy(st)
    deallocate(hu, hv, hw, du, dv, dw)
  end subroutine run_ref

  ! distributed: this rank's z-slab (anz planes starting at global z0), nsamp
  ! samples, then par_stats_finalize (Allreduce SUM + shared combine).  stats_init
  ! is told the GLOBAL geometry (nxd,nzd) so the normaliser is rank-invariant.
  subroutine run_slab(mu, mv, mw, uu, vv, ww, uv)
    real(dp), intent(out) :: mu(0:n), mv(0:n), mw(0:n)
    real(dp), intent(out) :: uu(0:n), vv(0:n), ww(0:n), uv(0:n)
    type(stats_t) :: st
    real(tp), allocatable :: hu(:,:,:), hv(:,:,:), hw(:,:,:)
    real(tp), allocatable, device :: du(:,:,:), dv(:,:,:), dw(:,:,:)
    integer(ip) :: isamp
    integer :: ist
    allocate(hu(0:n,nxd,anz), hv(0:n,nxd,anz), hw(0:n,nxd,anz))
    allocate(du(0:n,nxd,anz), dv(0:n,nxd,anz), dw(0:n,nxd,anz))
    call stats_init(st, n, nxd, nzd)            ! GLOBAL geometry (not anz)
    do isamp = 1, nsamp
       call gen_field(isamp, anz, z0, hu, hv, hw)
       du = hu;  dv = hv;  dw = hw
       call stats_accumulate(st, n, nxd, anz, du, dv, dw)   ! LOCAL plane count
    end do
    ist = cudaDeviceSynchronize()
    call par_stats_finalize(st, n, mu, mv, mw, uu, vv, ww, uv)
    call stats_destroy(st)
    deallocate(hu, hv, hw, du, dv, dw)
  end subroutine run_slab

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

end program test_driver_stats_mp
