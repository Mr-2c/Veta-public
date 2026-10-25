!===============================================================================
! test_par_reduce_mp.F90 -- validate the driver global-reduction primitives
! par_reduce_max / par_reduce_sum / par_reduce_sum_vec (par_mod) against
! closed-form expected values across whatever world size the launcher provides.
!
! These are the reductions the distributed time-stepping driver needs beyond the
! transpose + nonlinear all-to-alls: CFL frequency MAX (-> adaptive dt),
! divergence/imag MAX, and turbulence-statistics SUM. They are plain host-side
! MPI_Allreduce in real64 (= dp; the accuracy-critical accumulation tier), so this
! test is dp-only (no tp tier) and the chosen contributions are exact integers /
! half-integers, hence the comparisons are BIT-EXACT. par_reduce_and aggregates
! the verdict so any rank's failure fails the whole job. Run at P=1/4/8.
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_par_reduce_mp   Run: bash run_all_par_tests.sh run -k test_par_reduce_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_par_reduce_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use par_mod
  implicit none

  integer, parameter :: m = 4
  logical :: all_ok = .true.
  integer :: P, r, j
  real(real64) :: xmax, xsum, expected_max, expected_sum
  real(real64) :: vsum(m), exp_vsum(m)

  call par_init()
  P = par_nproc;  r = par_rank
  if (r == 0) print '(a,i0,a)', '[reduce-mp] world size = ', P, ' rank(s)'

  ! --- MAX of a scalar: rank r contributes (r - 1.5); global max = (P-1) - 1.5 ---
  xmax = real(r, real64) - 1.5_real64
  call par_reduce_max(xmax)
  expected_max = real(P-1, real64) - 1.5_real64
  call check('par_reduce_max  (scalar)', (xmax == xmax) .and. (xmax == expected_max))

  ! --- SUM of a scalar: rank r contributes r; global sum = P(P-1)/2 ---
  xsum = real(r, real64)
  call par_reduce_sum(xsum)
  expected_sum = real(P*(P-1)/2, real64)            ! P*(P-1) is always even -> exact
  call check('par_reduce_sum  (scalar)', (xsum == xsum) .and. (xsum == expected_sum))

  ! --- SUM of a profile: element j on rank r = 10*r + j;
  !     global sum_j = 10*P(P-1)/2 + j*P ---
  do j = 1, m
     vsum(j) = real(10*r + j, real64)
  end do
  call par_reduce_sum_vec(vsum, m)
  do j = 1, m
     exp_vsum(j) = 10.0_real64*real(P*(P-1)/2, real64) + real(j*P, real64)
  end do
  call check('par_reduce_sum_vec (profile)', all(vsum == vsum) .and. all(vsum == exp_vsum))

  call par_reduce_and(all_ok)
  if (r == 0) then
     if (all_ok) then
        print '(a)', 'ALL TESTS PASSED'
     else
        print '(a)', 'SOME TESTS FAILED'
     end if
  end if

  call par_finalize()
  if (.not. all_ok) error stop 1

contains

  subroutine check(name, ok)
    character(*), intent(in) :: name
    logical,      intent(in) :: ok
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', ok)
    print '(a,i0,a)', '[', par_rank, '] ['//tag//'] '//name
    if (.not. ok) all_ok = .false.
  end subroutine check

end program test_par_reduce_mp
