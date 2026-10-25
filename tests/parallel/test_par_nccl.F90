!===============================================================================
! test_par_nccl.F90  --  smoke test for the MPI/NCCL scaffold (par_mod + nccl_mod).
!
! This does NOT test the transpose (see test_par_transpose).
! It proves the whole binding + build + launch chain works on the real GPUs:
!   1. MPI_Init + world rank/size           (par_init)
!   2. node-local device bind + CUDA stream  (par_init)
!   3. NCCL communicator init                (par_init)
!   4. a REAL NCCL collective on DEVICE buffers: ncclAllReduce(SUM) of
!      (rank+1) across all ranks must equal P*(P+1)/2  -- BIT-EXACT (integers
!      up to a few thousand are exact in float32), NaN-hardened.
!   5. MPI collective sanity (par_reduce_and == MPI_Allreduce) aggregates the
!      verdict so a failure on ANY rank fails the whole job.
!   6. clean teardown (par_finalize).
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_par_nccl   Run: bash run_all_par_tests.sh run -k test_par_nccl in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!   srun -N1 -n4 --gpus-per-task=1 --gpu-bind=closest ./build_par/test_par_nccl
!   srun -N2 -n8 --gpus-per-task=1 --gpu-bind=closest ./build_par/test_par_nccl
!===============================================================================
program test_par_nccl
  use iso_c_binding, only: c_float, c_size_t
  use cudafor
  use nccl_mod
  use par_mod
  implicit none

  logical :: all_ok = .true.
  integer :: istat, ncclrc, ver
  real(c_float), allocatable, device :: ds(:), dr(:)
  real(c_float) :: hr(1), expected

  call par_init()

  if (par_rank == 0) then
     ncclrc = ncclGetVersion(ver)
     print '(a,i0)',  '[info] NCCL version int = ', ver
     print '(a,i0,a,i0)', '[info] world size = ', par_nproc, &
                          '   (rank0 local device = ', par_device
  end if

  ! ---- NCCL AllReduce(SUM) on device buffers --------------------------------
  allocate(ds(1), dr(1))
  ds = real(par_rank + 1, c_float)
  dr = 0.0_c_float
  ncclrc = ncclAllReduce(c_devloc(ds), c_devloc(dr), 1_c_size_t, ncclFloat, &
                         ncclSum, par_nccl_comm, par_stream)
  if (ncclrc /= ncclSuccess) then
     print '(a,i0,a)', '[rank ', par_rank, '] ncclAllReduce: '//nccl_err_str(ncclrc)
     all_ok = .false.
  end if
  istat = cudaStreamSynchronize(par_stream)
  if (istat /= 0) then
     print '(a,i0,a)', '[rank ', par_rank, '] cudaStreamSynchronize failed'
     all_ok = .false.
  end if
  hr = dr

  expected = real(par_nproc * (par_nproc + 1) / 2, c_float)
  ! NaN-hardened (hr==hr) AND bit-exact equality to the expected integer sum
  call check('NCCL AllReduce(SUM)=P(P+1)/2', (hr(1) == hr(1)) .and. (hr(1) == expected))

  deallocate(ds, dr)

  ! ---- aggregate verdict across ranks (this IS the MPI-collective sanity) ----
  call par_reduce_and(all_ok)
  if (par_rank == 0) then
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

end program test_par_nccl
