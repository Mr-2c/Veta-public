!===============================================================================
! par_mod.F90  --  parallel context for the multi-GPU port: MPI world + one
!                  NCCL communicator + this rank's GPU + a CUDA stream.
!
! Bootstrap only; no transpose/physics.
!   par_init():  MPI_Init -> world rank/size -> node-local rank (for device
!                binding, via MPI_Comm_split_type SHARED -- launcher-agnostic)
!                -> cudaSetDevice -> par_stream (BLOCKING -- load-bearing, see
!                par_init) + thread default stream = par_stream -> NCCL bootstrap
!                (rank-0 ncclGetUniqueId, MPI_Bcast the id, ncclCommInitRank).
!   par_finalize(): destroy the NCCL comm, MPI_Finalize.
!   par_reduce_and(): MPI_Allreduce(.and.) -- lets a test fail the WHOLE job if
!                any rank fails (also a live MPI-collective sanity).
!
! MPI BINDING: we use `include 'mpif.h'`, NOT `use mpi`, because
! nvfortran cannot read the system OpenMPI's gfortran-built `.mod`.  The F77
! interface (mpif.h + -lmpi -lmpi_mpifh) is compiler-agnostic.  Build flags are
! in parallel/build_run_smoke.sh.  (NVHPC's bundled CUDA-aware
! OpenMPI/HPC-X mpifort is the alternative -- see that script's header.)
!
! Progress prints (rank 0) make a blind first-run failure diagnosable: you see
! how far init got before any abort.
!===============================================================================
module par_mod
  use iso_c_binding, only: c_ptr, c_null_ptr, c_associated
  use, intrinsic :: iso_fortran_env, only: real64
  use cudafor
  use nccl_mod
  implicit none
  include 'mpif.h'
  private

  integer,                   save, public :: par_rank       = -1        ! world rank
  integer,                   save, public :: par_nproc      = 0         ! world size
  integer,                   save, public :: par_local_rank = -1        ! node-local rank
  integer,                   save, public :: par_ndev       = 0         ! GPUs visible
  integer,                   save, public :: par_device     = -1        ! GPU this rank uses
  type(c_ptr),               save, public :: par_nccl_comm  = c_null_ptr
  integer(cuda_stream_kind), save, public :: par_stream     = 0
  integer(cuda_stream_kind), save, public :: par_comm_stream = 0   ! NCCL-only overlap stream
  logical,                   save, public :: par_on         = .false.
  character(len=64),         save, public :: par_hostname   = 'unknown'  ! for fatal-error attribution

  public :: par_init, par_finalize, par_reduce_and
  public :: par_reduce_max, par_reduce_sum, par_reduce_sum_vec   ! driver reductions

contains

  subroutine par_init()
    integer :: ierr, nodecomm, istat, ncclrc, nlen
    character(len=MPI_MAX_PROCESSOR_NAME) :: pname
    type(ncclUniqueId) :: id

    call MPI_Init(ierr)
    if (ierr /= MPI_SUCCESS) error stop 'par_init: MPI_Init failed'
    call MPI_Comm_rank(MPI_COMM_WORLD, par_rank,  ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, par_nproc, ierr)
    call MPI_Get_processor_name(pname, nlen, ierr)
    if (ierr == MPI_SUCCESS .and. nlen > 0) par_hostname = pname(1:min(nlen, len(par_hostname)))
    if (par_rank == 0) print '(a,i0,a)', '[par] MPI up: ', par_nproc, ' rank(s)'

    ! node-local rank -> which GPU to bind (launcher-agnostic, no env vars)
    call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, par_rank, &
                             MPI_INFO_NULL, nodecomm, ierr)
    call MPI_Comm_rank(nodecomm, par_local_rank, ierr)
    call MPI_Comm_free(nodecomm, ierr)

    istat = cudaGetDeviceCount(par_ndev)
    if (istat /= 0)     error stop 'par_init: cudaGetDeviceCount failed'
    if (par_ndev < 1)   error stop 'par_init: no CUDA devices visible'
    par_device = mod(par_local_rank, par_ndev)
    istat = cudaSetDevice(par_device)
    if (istat /= 0)     error stop 'par_init: cudaSetDevice failed'
    ! ONE stream for ALL GPU work (kernels via cudaforSetDefaultStream, cuFFT
    ! via cufftSetStream at plan creation, NCCL by passing par_stream). Single stream
    ! = single total order, so no host sync is needed around a transpose. The
    ! arena invariant (one transform in flight) holds: still one stream. The
    ! serial build never calls par_init, so the oracle stays on the legacy stream.
    !
    ! BLOCKING (plain cudaStreamCreate) is LOAD-BEARING -- do NOT make this stream
    ! non-blocking: the nvfortran runtime performs device ALLOCATE/DEALLOCATE and
    ! module-symbol copyin traffic with legacy-default-stream semantics, and a
    ! blocking user stream interlocks with that traffic while a non-blocking one
    ! does not. With cudaStreamNonBlocking here, any dealloc/realloc of a device
    ! allocatable while kernels were in flight freed live memory (a sticky
    ! error 700 across every step_full path via cheb_ops' scratch resize). The
    ! COMM stream can be non-blocking (it carries only NCCL + pack/unpack on
    ! preallocated buffers, no runtime alloc/symbol traffic).
    istat = cudaStreamCreate(par_stream)
    if (istat /= 0)     error stop 'par_init: cudaStreamCreate failed'
    istat = cudaforSetDefaultStream(par_stream)
    if (istat /= 0)     error stop 'par_init: cudaforSetDefaultStream failed'
    ! The COMM stream: NON-blocking is safe HERE (and only here) because it
    ! carries ONLY ncclSend/ncclRecv on the preallocated transpose buffers -- no
    ! kernels, no runtime ALLOCATE/DEALLOCATE/symbol traffic, so the allocator
    ! interlock (see par_stream note above) does not apply. Ordering against
    ! par_stream is by cudaEventRecord/cudaStreamWaitEvent inside
    ! par_transpose_mod's pipelined phases. NOTHING else may use this stream.
    istat = cudaStreamCreateWithFlags(par_comm_stream, cudaStreamNonBlocking)
    if (istat /= 0)     error stop 'par_init: cudaStreamCreateWithFlags(comm) failed'
    if (par_rank == 0) print '(a)', '[par] device bound + streams created (compute=par_stream, A2A overlap=par_comm_stream)'

    ! NCCL bootstrap: rank 0 mints the id, everyone shares it, then init
    if (par_rank == 0) then
       ncclrc = ncclGetUniqueId(id)
       if (ncclrc /= ncclSuccess) error stop 'par_init: ncclGetUniqueId: '//nccl_err_str(ncclrc)
    end if
    call MPI_Bcast(id%internal, NCCL_UNIQUE_ID_BYTES, MPI_BYTE, 0, MPI_COMM_WORLD, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'par_init: MPI_Bcast(id) failed'
    ncclrc = ncclCommInitRank(par_nccl_comm, par_nproc, id, par_rank)
    if (ncclrc /= ncclSuccess) error stop 'par_init: ncclCommInitRank: '//nccl_err_str(ncclrc)
    if (.not. c_associated(par_nccl_comm)) error stop 'par_init: NCCL comm is null after init'
    if (par_rank == 0) print '(a)', '[par] NCCL communicator initialised'

    par_on = .true.
  end subroutine par_init

  subroutine par_finalize()
    integer :: ierr, ncclrc, istat
    if (par_on) then
       ncclrc = ncclCommDestroy(par_nccl_comm)        ! best-effort teardown
       istat  = cudaforSetDefaultStream(0_cuda_stream_kind)  ! back to legacy BEFORE destroy
       istat  = cudaStreamDestroy(par_comm_stream)
       istat  = cudaStreamDestroy(par_stream)
       par_nccl_comm = c_null_ptr
       par_on = .false.
    end if
    call MPI_Finalize(ierr)
  end subroutine par_finalize

  !-- global AND across ranks: turns any single-rank failure into a job-wide one
  subroutine par_reduce_and(flag)
    logical, intent(inout) :: flag
    integer :: ierr, sloc, sall
    sloc = merge(1, 0, flag)
    call MPI_Allreduce(sloc, sall, 1, MPI_INTEGER, MPI_MIN, MPI_COMM_WORLD, ierr)
    ! ierr checked (parity with the hardened I/O chain): benign under
    ! MPI_ERRORS_ARE_FATAL, but a changed error handler must not turn a failed
    ! reduction into a silent wrong answer.
    if (ierr /= MPI_SUCCESS) error stop 'par_reduce_and: MPI_Allreduce failed'
    flag = (sall == 1)
  end subroutine par_reduce_and

  !=============================================================================
  ! Driver global reductions. The ONLY communication beyond the transpose +
  ! nonlinear all-to-alls: the time-stepping driver reduces a few scalars/profiles
  ! computed from each rank's slab -- CFL frequency (MAX -> adaptive dt),
  ! divergence/imag check (MAX), turbulence statistics (SUM). All in real64 (=dp)
  ! regardless of the tp/wp tier: these are accuracy-critical accumulations (the
  ! dp policy -- see src/core/types.F90). MPI_Allreduce so every rank gets the result
  ! (the watchdog / adaptive-dt decision must be identical on all ranks).
  ! Reductions are host-side scalars/small profiles -> plain MPI (not NCCL).
  !=============================================================================

  !-- global MAX of a real64 scalar across ranks (CFL fmax, max divergence, ...).
  subroutine par_reduce_max(x)
    real(real64), intent(inout) :: x
    real(real64) :: xloc
    integer :: ierr
    xloc = x
    call MPI_Allreduce(xloc, x, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'par_reduce_max: MPI_Allreduce failed'
  end subroutine par_reduce_max

  !-- global SUM of a real64 scalar across ranks (e.g. a scalar statistic).
  subroutine par_reduce_sum(x)
    real(real64), intent(inout) :: x
    real(real64) :: xloc
    integer :: ierr
    xloc = x
    call MPI_Allreduce(xloc, x, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'par_reduce_sum: MPI_Allreduce failed'
  end subroutine par_reduce_sum

  !-- global elementwise SUM of a real64 profile across ranks (stats profiles).
  subroutine par_reduce_sum_vec(x, m)
    integer,      intent(in)    :: m
    real(real64), intent(inout) :: x(m)
    real(real64) :: xloc(m)
    integer :: ierr
    xloc = x
    call MPI_Allreduce(xloc, x, m, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'par_reduce_sum_vec: MPI_Allreduce failed'
  end subroutine par_reduce_sum_vec

end module par_mod
