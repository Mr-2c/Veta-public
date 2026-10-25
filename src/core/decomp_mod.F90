!===============================================================================
! decomp_mod.F90  --  1D-slab domain-decomposition descriptor + index/count
!                     arithmetic for the multi-GPU port.
!
! MPI-FREE BY DESIGN.
!   decomp_setup takes (nprocs, myrank) as ARGUMENTS, not from MPI calls, so the
!   block/index/count arithmetic is fully unit-testable at P = 1,2,4,8,... with
!   NO MPI and NO change to the nvfortran toolchain -- which is what preserves
!   the per-step bit-identity oracle.  `comm` is a plain
!   integer the caller stores for later; it is UNUSED until the MPI scaffold
!   lands.  No `use mpi` here.
!
! DECOMPOSITION: split ONE axis (1D slab).
!   Layout B (kx-slab): rank owns a contiguous block of kx in [0,nkx), all z.
!                       Solve- and z-FFT-friendly.
!   Layout A (z-slab):  rank owns a contiguous block of z, all kx.
!                       Physical- and x-FFT-friendly.
!   The A<->B transpose (transpose_mod) is a single all-to-all that swaps
!   which of {kx,z} is split; y (leading, length ny) rides inside the messages.
!
! Two z extents matter downstream: the truncated spectral nz (a pure-spectral
! transpose) and the dealiased nzd (the nonlinear-path transpose); likewise two
! kx extents, the coarse nkx and the fine nkxd.  This descriptor stores the
! balanced-block slab bounds for ALL FOUR (each is just block_range applied to a
! known global extent -- unambiguous and testable NOW).  It deliberately does
! NOT bake the A<->B all-to-all COUNT/DISPLACEMENT arrays: those depend on the
! exact (nkx, nzd) pairing the nonlinear path uses, which is re-derived
! carefully where that pairing is known (premature/guessed extents are
! the #1 bug source).  Counts are instead produced on demand by the PURE,
! separately-tested helper alltoall_counts_A_to_B below.
!
! INDEXING: 0-based global indices [0, n-1], matching the solver's existing
! 0-based kx/kz indexing (grid%kx(0:nkx-1), grid%kz(0:nkz-1)).
!
! Modelled in numpy first (src/models/work_decomp_model.py); the asserts in
! tests/serial/test_decomp.F90 mirror that model exactly.
!
! CPU-only module (host integer arithmetic; no device code).  Depends only on
! `kinds`.  Build: nvfortran -c decomp_mod.F90  (no -cuda needed).
!===============================================================================
module decomp_mod
  use kinds, only: ip
  implicit none
  private
  public :: decomp_t, decomp_setup
  public :: block_range, alltoall_counts_A_to_B
  public :: g2l, l2g, owner

  !=============================================================================
  ! decomp_t  --  per-rank decomposition descriptor.  A DUMB RECORD (like the
  ! types.F90 aggregates): decomp_setup fills it; everything else only reads it.
  !=============================================================================
  type :: decomp_t
     ! --- communicator + rank (comm is UNUSED until the MPI scaffold) ----------
     integer(ip) :: comm    = -1     ! caller-stored MPI communicator handle (opaque)
     integer(ip) :: nprocs  = 1      ! number of ranks
     integer(ip) :: myrank  = 0      ! this rank, in [0, nprocs-1]
     ! --- global spectral / dealiased extents (derived from nx,ny,nz with the
     !     SAME formulas as grid_mod build_grid -- these MUST stay in sync) ------
     integer(ip) :: nx  = 0, ny = 0, nz = 0
     integer(ip) :: nkx = 0          ! = nx/2 + 1    (real-to-complex in x)
     integer(ip) :: nxd = 0, nzd = 0 ! = 3*nx/2, 3*nz/2   (dealiased, 3/2 rule)
     integer(ip) :: nkxd = 0         ! = nxd/2 + 1   (fine real-to-complex in x)
     ! --- Layout A (z-slab) bounds for THIS rank ------------------------------
     !     coarse z (truncated spectral nz):
     integer(ip) :: a_z0 = 0, a_z1 = -1, a_nz_loc = 0
     !     dealiased z (nzd):
     integer(ip) :: ad_z0 = 0, ad_z1 = -1, ad_nz_loc = 0
     ! --- Layout B (kx-slab) bounds for THIS rank -----------------------------
     !     coarse kx (nkx):
     integer(ip) :: b_kx0 = 0, b_kx1 = -1, b_nkx_loc = 0
     !     fine kx (nkxd):
     integer(ip) :: bd_kx0 = 0, bd_kx1 = -1, bd_nkx_loc = 0
     ! --- convenience flag: does THIS rank own kx-index 0 -- the column that
     !     carries the (0,0) mean mode -- in layout B?  (Only the kx=0 owner runs
     !     the mean (0,0) solve once the solves are distributed.) ---------------
     logical     :: owns_mode00 = .true.
  end type decomp_t

contains

  !-----------------------------------------------------------------------------
  ! decomp_setup -- fill the descriptor for (nprocs, myrank) on a (nx,ny,nz)
  ! grid.  MPI-FREE: nprocs/myrank are ARGUMENTS.  Derives the spectral and
  ! dealiased extents with the SAME formulas as grid_mod (build_grid), then the
  ! balanced-block slab bounds for this rank on each split axis.
  !-----------------------------------------------------------------------------
  subroutine decomp_setup(d, comm, nprocs, myrank, nx, ny, nz)
    type(decomp_t), intent(out) :: d
    integer(ip),    intent(in)  :: comm, nprocs, myrank, nx, ny, nz
    if (nprocs < 1) error stop 'decomp_setup: nprocs must be >= 1'
    if (myrank < 0 .or. myrank >= nprocs) error stop 'decomp_setup: myrank out of [0,nprocs-1]'
    d%comm   = comm
    d%nprocs = nprocs
    d%myrank = myrank
    ! global extents -- MUST mirror grid_mod build_grid exactly
    d%nx   = nx;          d%ny  = ny;          d%nz  = nz
    d%nkx  = nx/2 + 1
    d%nxd  = 3*nx/2;      d%nzd = 3*nz/2
    d%nkxd = d%nxd/2 + 1
    ! layout-A z-slabs (coarse nz, then dealiased nzd)
    call block_range(d%nz,  nprocs, myrank, d%a_z0,  d%a_z1,  d%a_nz_loc)
    call block_range(d%nzd, nprocs, myrank, d%ad_z0, d%ad_z1, d%ad_nz_loc)
    ! layout-B kx-slabs (coarse nkx, then fine nkxd)
    call block_range(d%nkx,  nprocs, myrank, d%b_kx0,  d%b_kx1,  d%b_nkx_loc)
    call block_range(d%nkxd, nprocs, myrank, d%bd_kx0, d%bd_kx1, d%bd_nkx_loc)
    ! kx-index 0 lives on the rank whose coarse-kx block contains 0
    d%owns_mode00 = (owner(d%nkx, nprocs, 0_ip) == myrank)
  end subroutine decomp_setup

  !-----------------------------------------------------------------------------
  ! block_range -- balanced 1D block decomposition of [0, n-1] across P ranks
  ! ("first `rem` ranks get one extra").  Returns (lo, hi, cnt) for rank r.
  ! An EMPTY block (when P > n) has cnt == 0, lo == n, hi == n-1, so lo:hi is an
  ! empty range and lo still marks the (past-the-end) start.  Tiles [0,n-1]
  ! exactly with max imbalance <= 1.  PURE; mirrored by
  ! src/models/work_decomp_model.py:block_range.  (P must be >= 1; callers guard.)
  !-----------------------------------------------------------------------------
  pure subroutine block_range(n, P, r, lo, hi, cnt)
    integer(ip), intent(in)  :: n, P, r
    integer(ip), intent(out) :: lo, hi, cnt
    integer(ip) :: base, rem
    base = n / P
    rem  = mod(n, P)
    if (r < rem) then
       cnt = base + 1
       lo  = r * (base + 1)
    else
       cnt = base
       lo  = rem * (base + 1) + (r - rem) * base
    end if
    hi = lo + cnt - 1
  end subroutine block_range

  !-----------------------------------------------------------------------------
  ! alltoall_counts_A_to_B -- per-destination send/recv counts (in COMPLEX
  ! elements) and exclusive-prefix displacements for an A->B transpose
  ! (z-slab -> kx-slab), as rank pr, batched over nfield.  Layout A holds
  ! (ny, all-kx, z-slab_pr); layout B will hold (ny, kx-slab_q, all-z).
  !   pr sends q the block (ny, kx-slab_q, z-slab_pr):
  !       sendcnt(q) = ny * b_nkx_loc(q) * a_nz_loc(pr) * nfield
  !   pr recvs from q the block (ny, kx-slab_pr, z-slab_q):
  !       recvcnt(q) = ny * b_nkx_loc(pr) * a_nz_loc(q) * nfield
  ! Displacements are exclusive prefix sums (contiguous per-destination buffers,
  ! the MPI_Alltoallv layout).  nz_e / nkx_e are the EFFECTIVE extents: pass
  ! (nkx, nz) for a pure-spectral transpose, (nkxd, nzd) for the dealiased
  ! nonlinear path.  PURE; mirrors
  ! src/models/work_decomp_model.py:alltoall_counts_A_to_B.
  !-----------------------------------------------------------------------------
  pure subroutine alltoall_counts_A_to_B(ny, nkx_e, nz_e, P, pr, nfield, &
                                         sendcnt, senddsp, recvcnt, recvdsp)
    integer(ip), intent(in)  :: ny, nkx_e, nz_e, P, pr, nfield
    integer(ip), intent(out) :: sendcnt(0:P-1), senddsp(0:P-1)
    integer(ip), intent(out) :: recvcnt(0:P-1), recvdsp(0:P-1)
    integer(ip) :: q, lo, hi, b_nkx_pr, a_nz_pr, b_nkx_q, a_nz_q, acc
    ! this rank's own coarse slab sizes (the fixed factor in each direction)
    call block_range(nkx_e, P, pr, lo, hi, b_nkx_pr)
    call block_range(nz_e,  P, pr, lo, hi, a_nz_pr)
    do q = 0, P - 1
       call block_range(nkx_e, P, q, lo, hi, b_nkx_q)
       call block_range(nz_e,  P, q, lo, hi, a_nz_q)
       sendcnt(q) = ny * b_nkx_q * a_nz_pr * nfield
       recvcnt(q) = ny * b_nkx_pr * a_nz_q * nfield
    end do
    ! exclusive prefix-sum displacements
    acc = 0_ip
    do q = 0, P - 1
       senddsp(q) = acc
       acc = acc + sendcnt(q)
    end do
    acc = 0_ip
    do q = 0, P - 1
       recvdsp(q) = acc
       acc = acc + recvcnt(q)
    end do
  end subroutine alltoall_counts_A_to_B

  !-- global -> local index on a block-split axis; -1 if rank r does not own g. -
  ! PURE; mirrors src/models/work_decomp_model.py:g2l.
  pure function g2l(n, P, r, g) result(il)
    integer(ip), intent(in) :: n, P, r, g
    integer(ip) :: il, lo, hi, cnt
    call block_range(n, P, r, lo, hi, cnt)
    if (cnt > 0 .and. g >= lo .and. g <= hi) then
       il = g - lo
    else
       il = -1_ip
    end if
  end function g2l

  !-- local -> global index on a block-split axis (assumes 0 <= il < cnt). ------
  ! PURE; mirrors src/models/work_decomp_model.py:l2g.
  pure function l2g(n, P, r, il) result(g)
    integer(ip), intent(in) :: n, P, r, il
    integer(ip) :: g, lo, hi, cnt
    call block_range(n, P, r, lo, hi, cnt)
    g = lo + il
  end function l2g

  !-- owning rank of global index g (0-based); -1 if out of range. --------------
  ! PURE; mirrors src/models/work_decomp_model.py:owner.
  pure function owner(n, P, g) result(r)
    integer(ip), intent(in) :: n, P, g
    integer(ip) :: r, rr, lo, hi, cnt
    r = -1_ip
    do rr = 0, P - 1
       call block_range(n, P, rr, lo, hi, cnt)
       if (cnt > 0 .and. g >= lo .and. g <= hi) then
          r = rr
          return
       end if
    end do
  end function owner

end module decomp_mod
