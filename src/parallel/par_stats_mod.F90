!===============================================================================
! par_stats_mod.F90 -- distributed turbulence-statistics finalisation for the
! 1-D z-slab decomposition.  The ONLY new collective on the statistics path.
!
! stats_accumulate (src/core/stats_mod.F90) is already slab-correct: each rank folds
! its LOCAL z-planes into the per-sample plane-mean SUMS su..suv, normalised by
! the GLOBAL plane size nxd*nzd (x is never split; only z is).  The full-plane
! time-mean profiles and Reynolds stresses are therefore the cross-rank SUM of
! those partials.  This module does that ONE MPI_Allreduce(SUM) -- packed into a
! single 7*(n+1) collective (par_reduce_sum_vec) -- and then reuses the SHARED
! stats_combine, so the finalisation arithmetic is the SAME source of truth as
! the single-GPU stats_finalize.
!
! Result: every rank gets the SAME global profiles (Allreduce), matching the
! single-GPU oracle to dp sum-reassociation round-off -- the per-plane field
! values are identical on both paths; only the ORDER of the dp additions (which
! planes a rank sums, then the cross-rank sum) differs.  This is a dp-tier round-
! off independent of the tp tier (the statistics accumulators are real(dp)).
!
! Re_tau / wall_shear are PURE functions of the (0,0) MEAN Chebyshev column,
! which lives only on the owns_mode00 rank (Layout B) -- so they need NO
! reduction: that rank evaluates them exactly as the single-GPU code does.
!
! nsamp is identical on every rank (all ranks accumulate the same samples), so
! it is NOT reduced.
!
! Build (after kinds, cheb_ops_mod, stats_mod, par_mod):
!   nvfortran -cuda -c par_stats_mod.F90
!===============================================================================
module par_stats_mod
  use kinds,     only: dp, ip
  use stats_mod, only: stats_t, stats_combine, stats_combine_p
  use par_mod,   only: par_reduce_sum_vec
  implicit none
  private
  public :: par_stats_finalize, par_stats_finalize_p

contains

  !-----------------------------------------------------------------------------
  ! Distributed finalize: D2H the per-rank partial sums (proven whole-array
  ! pattern), pack the seven profiles, ONE Allreduce(SUM) over all ranks, then
  ! the shared combine.  All ranks return the SAME global result.
  !-----------------------------------------------------------------------------
  subroutine par_stats_finalize(st, n, mean_u, mean_v, mean_w, uu, vv, ww, uv)
    type(stats_t), intent(in)  :: st
    integer(ip),   intent(in)  :: n
    real(dp),      intent(out) :: mean_u(0:n), mean_v(0:n), mean_w(0:n)
    real(dp),      intent(out) :: uu(0:n), vv(0:n), ww(0:n), uv(0:n)
    real(dp) :: su(0:n), suu(0:n), sv(0:n), svv(0:n), sw(0:n), sww(0:n), suv(0:n)
    real(dp) :: buf(7*(n+1))
    integer  :: m
    su  = st%su;  suu = st%suu          ! D2H copies (host(0:n) = device(0:n))
    sv  = st%sv;  svv = st%svv
    sw  = st%sw;  sww = st%sww
    suv = st%suv
    m = n + 1
    buf(      1:  m) = su;   buf(  m+1:2*m) = suu
    buf(2*m+1:3*m) = sv;     buf(3*m+1:4*m) = svv
    buf(4*m+1:5*m) = sw;     buf(5*m+1:6*m) = sww
    buf(6*m+1:7*m) = suv
    call par_reduce_sum_vec(buf, 7*m)   ! -> global plane-mean sums on every rank
    call stats_combine(st%nsamp, n, buf(1:m), buf(m+1:2*m), buf(2*m+1:3*m), &
                       buf(3*m+1:4*m), buf(4*m+1:5*m), buf(5*m+1:6*m), buf(6*m+1:7*m), &
                       mean_u, mean_v, mean_w, uu, vv, ww, uv)
  end subroutine par_stats_finalize

  !-----------------------------------------------------------------------------
  ! Distributed PRESSURE finalize: same pattern as above for the two
  ! pressure profiles -- ONE Allreduce(SUM) over (sp, spp), then the shared
  ! stats_combine_p. nsamp is identical on every rank (not reduced). mean_p carries
  ! the k=0 gauge constant; p_var is gauge-independent.
  !-----------------------------------------------------------------------------
  subroutine par_stats_finalize_p(st, n, mean_p, p_var)
    type(stats_t), intent(in)  :: st
    integer(ip),   intent(in)  :: n
    real(dp),      intent(out) :: mean_p(0:n), p_var(0:n)
    real(dp) :: sp(0:n), spp(0:n)
    real(dp) :: buf(2*(n+1))
    integer  :: m
    sp = st%sp;  spp = st%spp            ! D2H copies
    m = n + 1
    buf(    1:  m) = sp
    buf(  m+1:2*m) = spp
    call par_reduce_sum_vec(buf, 2*m)    ! -> global pressure plane-mean sums on every rank
    call stats_combine_p(st%nsamp, n, buf(1:m), buf(m+1:2*m), mean_p, p_var)
  end subroutine par_stats_finalize_p

end module par_stats_mod
