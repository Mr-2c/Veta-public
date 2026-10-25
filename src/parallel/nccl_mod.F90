!===============================================================================
! nccl_mod.F90  --  thin Fortran (ISO_C_BINDING) interface to the subset of NCCL
!                   the multi-GPU transpose needs (NCCL is the transpose
!                   backend).
!
! NCCL binding only; no transpose logic.
!   The C ABI details (by-value struct, device pointers, stream handle) cannot
!   be compile-checked without a real NCCL toolchain, so the smoke test
!   (tests/parallel/test_par_nccl.F90) fails LOUDLY at compile / link / init if
!   any signature here is wrong -- fail at compile, not silently later.
!
! Binding conventions (NVHPC nvfortran + cudafor):
!   * ncclComm_t (opaque pointer)        -> type(c_ptr)
!   * ncclUniqueId  {char internal[128]} -> bind(c) derived type, passed BY VALUE
!     to ncclCommInitRank (matches the C by-value struct argument)
!   * device buffers (const void* / void*) -> type(c_devptr), via c_devloc()
!   * cudaStream_t                       -> integer(cuda_stream_kind), BY VALUE
!   * size_t count                       -> integer(c_size_t)   [NO 32-bit limit,
!                                            unlike MPI_Alltoallv]
!   * enums (ncclResult_t/DataType/RedOp) -> integer(c_int) parameters
!
! Compiles with `nvfortran -cuda` ALONE (no nccl.h needed -- the interface is
! pure Fortran); NCCL is pulled in only at LINK (-lnccl).
!===============================================================================
module nccl_mod
  use iso_c_binding
  use cudafor, only: cuda_stream_kind, c_devptr
  implicit none
  private

  !-- result codes (we only special-case success) ------------------------------
  integer(c_int), parameter, public :: ncclSuccess = 0_c_int

  !-- data types (enum values from nccl.h; aliases share a value) ---------------
  integer(c_int), parameter, public :: ncclInt8    = 0_c_int, ncclChar  = 0_c_int
  integer(c_int), parameter, public :: ncclUint8   = 1_c_int
  integer(c_int), parameter, public :: ncclInt32   = 2_c_int, ncclInt   = 2_c_int
  integer(c_int), parameter, public :: ncclUint32  = 3_c_int
  integer(c_int), parameter, public :: ncclInt64   = 4_c_int
  integer(c_int), parameter, public :: ncclUint64  = 5_c_int
  integer(c_int), parameter, public :: ncclFloat16 = 6_c_int, ncclHalf  = 6_c_int
  integer(c_int), parameter, public :: ncclFloat32 = 7_c_int, ncclFloat = 7_c_int
  integer(c_int), parameter, public :: ncclFloat64 = 8_c_int, ncclDouble= 8_c_int

  !-- reduction ops (only needed by AllReduce; transpose uses send/recv) --------
  integer(c_int), parameter, public :: ncclSum=0_c_int, ncclProd=1_c_int, &
                                       ncclMax=2_c_int, ncclMin=3_c_int, ncclAvg=4_c_int

  integer, parameter, public :: NCCL_UNIQUE_ID_BYTES = 128

  !-- the unique-id handshake struct (interoperable; passed by value on init) ---
  type, bind(c), public :: ncclUniqueId
     character(kind=c_char) :: internal(NCCL_UNIQUE_ID_BYTES)
  end type ncclUniqueId

  public :: ncclGetVersion, ncclGetUniqueId, ncclCommInitRank, ncclCommDestroy
  public :: ncclGroupStart, ncclGroupEnd, ncclSend, ncclRecv, ncclAllReduce
  public :: nccl_err_str

  interface
     integer(c_int) function ncclGetVersion(version) bind(c, name='ncclGetVersion')
       import :: c_int
       integer(c_int) :: version                       ! int*  (out, by reference)
     end function ncclGetVersion

     integer(c_int) function ncclGetUniqueId(uniqueId) bind(c, name='ncclGetUniqueId')
       import :: c_int, ncclUniqueId
       type(ncclUniqueId) :: uniqueId                  ! ncclUniqueId* (out, by reference)
     end function ncclGetUniqueId

     integer(c_int) function ncclCommInitRank(comm, nranks, commId, rank) &
                                              bind(c, name='ncclCommInitRank')
       import :: c_int, c_ptr, ncclUniqueId
       type(c_ptr)               :: comm               ! ncclComm_t* (out, by reference)
       integer(c_int), value     :: nranks
       type(ncclUniqueId), value :: commId             ! BY VALUE (struct)
       integer(c_int), value     :: rank
     end function ncclCommInitRank

     integer(c_int) function ncclCommDestroy(comm) bind(c, name='ncclCommDestroy')
       import :: c_int, c_ptr
       type(c_ptr), value :: comm                      ! ncclComm_t (by value)
     end function ncclCommDestroy

     integer(c_int) function ncclGroupStart() bind(c, name='ncclGroupStart')
       import :: c_int
     end function ncclGroupStart

     integer(c_int) function ncclGroupEnd() bind(c, name='ncclGroupEnd')
       import :: c_int
     end function ncclGroupEnd

     integer(c_int) function ncclSend(sendbuff, count, datatype, peer, comm, stream) &
                                      bind(c, name='ncclSend')
       import :: c_int, c_size_t, c_devptr, c_ptr, cuda_stream_kind
       type(c_devptr),            value :: sendbuff     ! const void*
       integer(c_size_t),         value :: count
       integer(c_int),            value :: datatype
       integer(c_int),            value :: peer
       type(c_ptr),               value :: comm
       integer(cuda_stream_kind), value :: stream
     end function ncclSend

     integer(c_int) function ncclRecv(recvbuff, count, datatype, peer, comm, stream) &
                                      bind(c, name='ncclRecv')
       import :: c_int, c_size_t, c_devptr, c_ptr, cuda_stream_kind
       type(c_devptr),            value :: recvbuff     ! void*
       integer(c_size_t),         value :: count
       integer(c_int),            value :: datatype
       integer(c_int),            value :: peer
       type(c_ptr),               value :: comm
       integer(cuda_stream_kind), value :: stream
     end function ncclRecv

     integer(c_int) function ncclAllReduce(sendbuff, recvbuff, count, datatype, op, &
                                           comm, stream) bind(c, name='ncclAllReduce')
       import :: c_int, c_size_t, c_devptr, c_ptr, cuda_stream_kind
       type(c_devptr),            value :: sendbuff
       type(c_devptr),            value :: recvbuff
       integer(c_size_t),         value :: count
       integer(c_int),            value :: datatype
       integer(c_int),            value :: op
       type(c_ptr),               value :: comm
       integer(cuda_stream_kind), value :: stream
     end function ncclAllReduce

     !-- const char* ncclGetErrorString(ncclResult_t) -- wrapped below ----------
     type(c_ptr) function ncclGetErrorString_c(result) bind(c, name='ncclGetErrorString')
       import :: c_int, c_ptr
       integer(c_int), value :: result
     end function ncclGetErrorString_c
  end interface

contains

  !-- turn NCCL's C error string into a Fortran string (for messages) -----------
  function nccl_err_str(res) result(s)
    integer(c_int), intent(in)    :: res
    character(len=:), allocatable :: s
    type(c_ptr)                   :: p
    character(kind=c_char), pointer :: arr(:)
    integer :: i, n
    p = ncclGetErrorString_c(res)
    if (.not. c_associated(p)) then
       s = '<null>'
       return
    end if
    call c_f_pointer(p, arr, [256])          ! upper bound; scan to the NUL
    n = 0
    do i = 1, 256
       if (arr(i) == c_null_char) exit
       n = n + 1
    end do
    allocate(character(len=n) :: s)
    do i = 1, n
       s(i:i) = arr(i)
    end do
  end function nccl_err_str

end module nccl_mod
