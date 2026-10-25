!===============================================================================
! par_transpose_mod.F90  --  the REAL multi-GPU A<->B transpose: compose
! transpose_mod's validated device pack/unpack kernels with one grouped NCCL
! all-to-all.
!
! Lives in parallel/ (NOT in the single-GPU
! suite) so that suite stays MPI/NCCL-independent.  The data-movement index
! arithmetic is the one validated in src/models/work_decomp_model.py and exercised
! single-GPU by tests/serial/test_transpose_pack.F90; here we only add the wire:
!
!   pack (GPU)  ->  ncclGroupStart{ ncclSend/ncclRecv to every peer }  ->  unpack (GPU)
!
! WIRE DETAILS
!   * complex(tp) is sent as RAW BYTES (ncclChar, count = elems*sizeof(complex)).
!     A transpose only MOVES data (no reduction), so bytes are exact and
!     kind-agnostic -- no NCCL complex type needed, bit-exact for tp = sp or dp.
!   * counts/displacements (in complex elements) come from decomp_mod's PURE,
!     separately-tested alltoall_counts_A_to_B.  B->A is the SAME permutation
!     with send<->recv swapped (sendcnt_B->A == recvcnt_A->B, etc.).
!   * the NCCL communicator + CUDA stream come straight from par_mod (set by
!     par_init).  No need to thread them through the signature or stash them in
!     decomp_t yet -- the solver calls this after par_init, so the globals are
!     live.  (Stashing in decomp_t is a later cleanup if the call sites want it.)
!
! THE LAYOUT-B SIDE IS STAGED NOWHERE.  The per-(src,dst) wire order
! [iy, local-kx, local-z, field] makes the Layout-B side's staging a PURE
! IDENTITY per (peer, field): the A->B message from src q / field f lands in
! out_B(:, :, az0(q)+1 : az0(q)+anz_q, f) -- one contiguous column-major run
! whose array offset EQUALS the wire displacement (block_range tiles z
! ascending/contiguous, so the prefix sum of anz is az0). Mirror statement for
! the B->A send side. So NCCL now sends DIRECTLY from in_B (B->A) and receives
! DIRECTLY into out_B (A->B), one message per (peer, field) on BOTH ends (NCCL
! p2p matching is one-to-one in posting order with equal counts; both ends
! iterate f ascending). The kx side (Layout A) is genuinely strided and keeps
! its gather/scatter kernel + the ONE persistent staging buffer -- ptbuf serves
! as A->B send stage and B->A recv stage.  A single staging buffer saves
! max(slen,rlen) complex(tp) elements/rank = 9 elements per COARSE SPECTRAL
! point at the nfield=6 dealiased transpose [6 fields x the 3/2 fine-z factor;
! the crossing data is coarse-kx]: 72 B/pt sp / 144 B/pt dp, ~5.7 GB/GPU on a
! 16-node frame, along with one full HBM pass + P kernel launches per transpose
! (the identity side's copy).
! Bit-exact by construction: same bytes, same final addresses, no staging hop
! on the identity side.  Executable spec + bit-identity sweep:
! src/models/work_m1_direct_a2a_model.py. nfield==1 (every pipelined ptp leg
! and the r2c single-field transposes) degenerates to one message per peer.
! Batched nfield>1 transposes post nfield messages
! per peer (2*P*nf p2p ops per ncclGroup) -- the per-field SIZE equals what
! the pipelined flights move, while the DEFAULT (a2a_overlap=0) monolithic
! path groups nf x more, smaller ops; correctness is count-matched either way.
!
! STREAMS / ORDERING (correctness-first)
!   pack/unpack run as CUF kernels on the DEFAULT stream; the NCCL ops run on
!   par_stream.  We serialise the three phases with host syncs:
!     pack -> cudaDeviceSynchronize -> NCCL group -> cudaStreamSynchronize(par_stream)
!          -> unpack -> cudaDeviceSynchronize.
!   This is not the fastest (no comm/compute overlap) but it is unambiguously
!   correct across the stream boundary.
!
! P == 1: layout A == layout B, so the transpose is the bit-exact identity copy
! (same shapes there) -- matches transpose_mod's P==1 path; no NCCL is touched.
!
! Build: nvfortran -cuda  (needs kinds, decomp_mod, transpose_mod, nccl_mod,
! par_mod .mod files; links -lnccl -lmpi via parallel/build_run_transpose.sh).
!===============================================================================
module par_transpose_mod
  use kinds,         only: tp, ip, dp
  use iso_c_binding, only: c_size_t, c_int
  use cudafor
  use decomp_mod,    only: decomp_t, alltoall_counts_A_to_B, block_range
  ! Only the strided Layout-A side's kernels are used here: pack_A_to_B (A->B
  ! gather) and unpack_B_to_A (B->A scatter). The Layout-B side's pack_B_to_A /
  ! unpack_A_to_B would be identity copies -- NCCL reads/writes the field
  ! arrays directly (they stay in transpose_mod for the single-GPU pack test).
  use transpose_mod, only: pack_A_to_B, unpack_B_to_A
  use nccl_mod,      only: ncclSend, ncclRecv, ncclGroupStart, ncclGroupEnd, &
                           ncclChar, ncclSuccess, nccl_err_str
  use par_mod,       only: par_nccl_comm, par_stream, par_comm_stream, par_reduce_max
  implicit none
  private
  public :: par_transpose_A_to_B, par_transpose_B_to_A
  public :: par_transpose_warmup        ! pre-size persistent buffers + open NCCL channels
  public :: par_transpose_timing        ! per-phase wall (pack/NCCL/unpack) -- comm vs overhead probe
  public :: par_transpose_set_timing    ! enable per-phase barriers+clocks (time_every>0)
  ! Pipelined-transpose phases: the per-field chunked all-to-all that
  ! par_spectral3d's overlapped nf paths interleave with their per-field FFT work.
  ! One pipelined leg at a time (single host thread, legs never nested -- the same
  ! assumption the pt_* clocks already make). Phase contract per leg:
  !   ptp_begin(dir, ...)                          once
  !   A2B: for f: <local FFTs> ; ptp_pack_field(f, fld) ; ptp_a2a_field(f, fld_B)
  !   B2A: for f: <local FFTs> ;                         ptp_a2a_field(f, fld_B)
  !   for f: ptp_unpack_field(f, fld) ; <local FFTs>     (A2B: wait-only)
  !   ptp_end()                                    once
  ! fld_B is the field's Layout-B-side slice: the DIRECT recv target (A2B) or
  ! the DIRECT send source (B2A) -- the identity side is never staged, so the
  ! B2A leg has no pack call and the A2B unpack is only the ev_cm(f) wait.
  ! Ordering is ALL device-side: ptp_a2a_field records ev_pk(f) on par_stream
  ! (= the position right after the pack kernel / the leg's local FFT work for
  ! field f), the A2A waits
  ! it on par_comm_stream, then records ev_cm(f); consumers wait ev_cm(f) via
  ! ptp_unpack_field. No host syncs in the leg. The DIRECT sides inherit the
  ! same edges: the A2A's read of fld_B (B2A) / write of fld_B (A2B) is ordered
  ! after every prior par_stream toucher of fld_B through ev_pk(f), and every
  ! later par_stream toucher is ordered after it through the ev_cm(f) waits;
  ! comm-stream ops are FIFO, so cross-leg reuse of fld_B slices or ptbuf slots
  ! on the comm side is issue-ordered for free.
  ! Under pt_timing the A2A occupancy is EVENT-timed (no barriers -> overlap is
  ! preserved in timed runs); pack/unpack are not separable and report 0 -- the
  ! nccl column then reads the true comm-stream occupancy per step.
  public :: ptp_begin, ptp_pack_field, ptp_a2a_field, ptp_unpack_field, ptp_end
  integer(ip), parameter, public :: PTP_A2B = 1, PTP_B2A = 2

  ! With ALL GPU work on the one (BLOCKING -- see par_mod) par_stream (pack/unpack CUF
  ! kernels via cudaforSetDefaultStream, NCCL by explicit stream arg), in-stream
  ! order makes the four per-transpose host syncs redundant for CORRECTNESS.
  ! They remain necessary for MEASUREMENT: the pack/NCCL/unpack wall-time split
  ! (the comm-vs-compute columns every scaling analysis reads) is only meaningful
  ! if each phase is drained before its clock is read. So the barriers+clocks
  ! run only when the driver enables them (time_every > 0); an untimed
  ! production run is barrier-free. Only the pack+nccl+unpack SUM is
  ! comparable across runs: there are no identity-side copies, so the pack
  ! column reads ~0 on every B->A and the unpack column ~0 on every A->B.
  logical, save :: pt_timing = .false.

  ! Persistent transpose buffer (grow-only). Hoisted out of the per-call path: at
  ! large grids this is MULTI-GB and was cudaMalloc/cudaFree'd 12x/step
  ! (cudaFree synchronizes -> serialization + jitter, and the alloc cost grows with
  ! the grid). Sized once to the largest (nfield) transpose -- ideally by
  ! par_transpose_warmup before the time loop -- and reused by every call after.
  ! ONE buffer, not two: only the strided Layout-A side is ever staged
  ! (A->B stages its sends here, B->A stages its recvs here); the Layout-B side
  ! goes direct. A->B and B->A never overlap in flight (monolithic: in-stream
  ! order on par_stream; pipelined: one leg at a time + the ev_pk/ev_cm edges),
  ! so the two roles can share the allocation.
  complex(tp), allocatable, device :: ptbuf(:)

  ! Per-phase wall-time accumulators (PERF DIAGNOSTIC, not on any correctness path).
  ! A host sync is taken BEFORE pack AND after each of pack / NCCL / unpack, so every
  ! system_clock delta = that phase's PURE wall time -- in particular pt_pack EXCLUDES
  ! the per-substage solver compute queued ahead of it (that work shows up as
  ! step - comm in main_par, not as "pack"). par_transpose_timing reads+resets; main_par
  ! splits these out of the step timer to test whether the transpose is BANDWIDTH-bound
  ! (NCCL ~2x sp->dp, ~1/P strong scaling) or overhead/latency-bound (NCCL ~flat).
  ! pt_c0..3 are reused module scratch (single host thread; transposes never run
  ! concurrently). system_clock is ~ns vs the ms phases -> negligible cost.
  real(dp),    save :: pt_pack_s = 0.0_dp, pt_nccl_s = 0.0_dp, pt_unpack_s = 0.0_dp
  integer(ip), save :: pt_ncalls = 0
  integer(8),  save :: pt_rate = 0, pt_c0 = 0, pt_c1 = 0, pt_c2 = 0, pt_c3 = 0

  ! Pipelined-leg state (one leg at a time; see the phase contract above).
  integer(ip), parameter :: PTP_MAXF = 8
  integer(ip), save :: ptp_dir = 0, ptp_nf = 0, ptp_P = 0, ptp_pr = 0
  integer(ip), save :: ptp_ny = 0, ptp_nkx = 0, ptp_nz = 0, ptp_anz = 0, ptp_bnkx = 0
  integer(ip), save :: ptp_slen1 = 0, ptp_rlen1 = 0
  integer(ip), allocatable, save :: ptp_sc(:), ptp_sd(:), ptp_rc(:), ptp_rd(:)
  type(cudaEvent), save :: ptp_ev_pk(PTP_MAXF), ptp_ev_a0(PTP_MAXF), ptp_ev_cm(PTP_MAXF)
  logical, save :: ptp_ev_live = .false.

contains

  !-----------------------------------------------------------------------------
  ! ensure_ptbuf -- grow the persistent staging buffer to at least blen.
  ! Grow-only: once sized for the nfield=6 transpose, nf=3/1 calls reuse it.
  !-----------------------------------------------------------------------------
  subroutine ensure_ptbuf(blen)
    integer(ip), intent(in) :: blen
    integer(ip) :: istat
    ! DRAIN before any grow-dealloc: the non-blocking par_comm_stream has
    ! NO alloc-interlock with the runtime's free, so a
    ! regrow while a pipelined A2A is in flight would free live memory. Warm-up
    ! pre-sizing makes this branch unreachable mid-loop; the sync makes the
    ! contract structural instead of assumed (grow paths only -- zero hot cost).
    if (allocated(ptbuf)) then
       if (size(ptbuf) < blen) then
          istat = cudaDeviceSynchronize()
          deallocate(ptbuf)
       end if
    end if
    if (.not. allocated(ptbuf)) allocate(ptbuf(max(blen,1)))
  end subroutine ensure_ptbuf

  !-----------------------------------------------------------------------------
  ! par_transpose_warmup -- call ONCE before the time loop with the production
  ! extents and the LARGEST nfield used (=6):
  !   (1) pre-sizes the persistent buffers so no transpose grow-reallocs mid-loop;
  !   (2) issues one all-to-all so NCCL establishes every peer channel here (lazy
  !       connection setup, the one-off multi-ms first-call cost) -- not in step 1;
  !   (3) runs a small BANDWIDTH SELF-TEST on the warm channels and prints the
  !       achieved all-to-all throughput. If it is implausibly low (P>1), NCCL has
  !       almost certainly fallen back to TCP SOCKETS instead of InfiniBand RDMA
  !       (e.g. a wrong/absent NCCL_IB_HCA, or a libibverbs ABI clash from a module
  !       shadowing the system provider). A big warning
  !       is printed; set CLAUSON_REQUIRE_RDMA=1 to hard-abort instead.
  ! The exchanges move uninitialised bytes that are discarded -- a transpose only
  ! MOVES data, so this is harmless and does not affect results. No-op at P==1.
  !-----------------------------------------------------------------------------
  subroutine par_transpose_warmup(d, ny, nkx, nz, a_nz_loc, b_nkx_loc, nfield)
    type(decomp_t), intent(in) :: d
    integer(ip),    intent(in) :: ny, nkx, nz, a_nz_loc, b_nkx_loc, nfield
    integer(ip) :: P, pr, slen, rlen, mx, istat, rep
    integer(ip) :: sc(0:d%nprocs-1), sd(0:d%nprocs-1)
    integer(ip) :: rc(0:d%nprocs-1), rd(0:d%nprocs-1)
    integer(ip), parameter :: nrep = 3
    real(dp) :: min_gbps                          ! endpoint-aware RDMA floor (set below)
    integer(8) :: crate, c0, c1
    real(dp)   :: secs, gbps, gbytes, cbytes
    real(dp)   :: gneg                            ! -gbps -> max-reduce = min over ranks
    character(len=8) :: req
    ! Warm-up-local recv side: production stages only ONE side (ptbuf), but
    ! the self-test needs disjoint send/recv regions to keep the exchange -- and
    ! the [comm] GB/s metric -- comparable across runs.
    ! Freed right after, so steady state is one buffer.
    complex(tp), allocatable, device :: wrecv(:)

    P = d%nprocs;  pr = d%myrank
    if (P == 1) return
    call pt_guard(ny, nkx, nz, P, nfield)
    ! Endpoint-aware RDMA floor: legit per-rank all-to-all efficiency falls
    ! with endpoint count; a TCP-socket fallback is ~1 GB/s/rank flat. Hold 5 GB/s
    ! up to P=16, scale ~1/P beyond, floor 1.5 GB/s (still well above sockets) --
    ! avoids CLAUSON_REQUIRE_RDMA false-aborts on healthy O(10^2-10^3)-rank jobs.
    min_gbps = max(1.5_dp, 5.0_dp*min(1.0_dp, 16.0_dp/real(P,dp)))
    call alltoall_counts_A_to_B(ny, nkx, nz, P, pr, nfield, sc, sd, rc, rd)
    slen = sum(sc);  rlen = sum(rc)
    mx = max(slen, rlen)                   ! A->B stages slen, B->A stages rlen -> size to max
    call ensure_ptbuf(mx)
    allocate(wrecv(max(rlen,1)))

    ! (1)+(2) establish all peer channels (first all-to-all pays lazy connection setup)
    call exchange_bytes(par_stream, P, slen, rlen, ptbuf, sc, sd, wrecv, rc, rd)
    istat = cudaStreamSynchronize(par_stream)

    ! (3) bandwidth self-test on WARM channels -> detect TCP-socket fallback
    istat = cudaDeviceSynchronize()
    call system_clock(count_rate=crate);  call system_clock(c0)
    do rep = 1, nrep
       call exchange_bytes(par_stream, P, slen, rlen, ptbuf, sc, sd, wrecv, rc, rd)
    end do
    istat = cudaStreamSynchronize(par_stream)
    call system_clock(c1)
    deallocate(wrecv)                      ! drained above (all wrecv ops ran on par_stream)

    cbytes = real(2*storage_size(0.0_tp)/8, dp)                ! bytes per complex(tp)
    secs   = real(c1 - c0, dp) / real(crate, dp)
    gbytes = real(nrep, dp) * real(slen + rlen, dp) * cbytes / 1.0e9_dp
    gbps   = 0.0_dp;  if (secs > 0.0_dp) gbps = gbytes / secs  ! per-rank send+recv throughput

    if (pr == 0) then
       write(*,'(a,f9.2,a,i0,a)') '   [comm] warm all-to-all throughput = ', gbps, &
            ' GB/s/rank  (P=', P, ', nfield transpose)'
       if (gbps < min_gbps) then
          write(*,'(a)') ' !!=========================================================================!!'
          write(*,'(a,f7.2,a)') ' !! WARNING: transpose throughput is only ', gbps, ' GB/s/rank --'
          write(*,'(a)') ' !! NCCL is almost certainly on the TCP-SOCKET fallback, NOT InfiniBand RDMA.'
          write(*,'(a)') ' !! The run will be COMM-BOUND and several-x slower than it should be.'
          write(*,'(a)') ' !! Verify:  grep "Using network" <out>   (want "IB", not "Socket")'
          write(*,'(a)') ' !! Fix (Helma): NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5 +'
          write(*,'(a)') ' !!   LD_PRELOAD=/lib64/libibverbs.so.1 (see scripts/sbatch/run_main_par.sbatch).'
          write(*,'(a)') ' !!   Set CLAUSON_REQUIRE_RDMA=1 to hard-abort instead of warn.'
          write(*,'(a)') ' !!=========================================================================!!'
       end if
    end if

    ! optional hard stop. gbps is a PER-RANK measurement (own clock),
    ! so the abort verdict is min-reduced first: every rank then evaluates the
    ! SAME condition and the stop is genuinely uniform.
    gneg = -gbps;  call par_reduce_max(gneg)
    if (-gneg < min_gbps) then
       call get_environment_variable('CLAUSON_REQUIRE_RDMA', req)
       if (trim(req) == '1') error stop 'par_transpose_warmup: slow comm (TCP fallback?) + CLAUSON_REQUIRE_RDMA=1'
    end if
  end subroutine par_transpose_warmup

  !-----------------------------------------------------------------------------
  ! exchange_bytes -- one grouped all-to-all of complex(tp) sent as raw bytes.
  ! Sends this rank's per-destination chunks and receives the incoming chunks in
  ! one ncclGroupStart/End.  Self (peer == this rank) is a valid in-group NCCL
  ! send/recv (it becomes a device-local copy).  Empty messages are skipped.
  ! Device buffers are explicit-shape, no intent (project style).
  ! WARM-UP ONLY: production transposes use the direct exchanges below.
  ! DO NOT read the printed [comm] GB/s as the production per-message rate:
  ! this warm-up posts ONE aggregated nfield-wide message
  ! per peer, 6x/3x larger than the per-(peer,field) messages production posts.
  ! The metric's actual job -- TCP-socket-fallback detection (~1 GB/s/rank vs
  ! the min_gbps floor) -- is message-size
  ! insensitive; a small-message NCCL-efficiency study needs its own probe.
  !-----------------------------------------------------------------------------
  subroutine exchange_bytes(strm, P, slen, rlen, sendbuf, sendcnt, senddsp, &
                            recvbuf, recvcnt, recvdsp)
    integer(cuda_stream_kind), intent(in) :: strm     ! par_stream (monolithic) or par_comm_stream (pipelined)
    integer(ip), intent(in) :: P, slen, rlen
    complex(tp), device     :: sendbuf(slen), recvbuf(rlen)
    integer(ip), intent(in) :: sendcnt(0:P-1), senddsp(0:P-1)
    integer(ip), intent(in) :: recvcnt(0:P-1), recvdsp(0:P-1)
    integer(ip)       :: q, ncrc
    integer(c_size_t) :: cb
    cb = int(2*storage_size(0.0_tp)/8, c_size_t)        ! bytes per complex(tp)
    ncrc = ncclGroupStart()
    if (ncrc /= ncclSuccess) error stop 'par_transpose: ncclGroupStart: '//nccl_err_str(ncrc)
    do q = 0, P-1
       if (sendcnt(q) > 0) then
          ncrc = ncclSend(c_devloc(sendbuf(senddsp(q)+1)), &
                          int(sendcnt(q), c_size_t)*cb, ncclChar, &
                          int(q, c_int), par_nccl_comm, strm)
          if (ncrc /= ncclSuccess) error stop 'par_transpose: ncclSend: '//nccl_err_str(ncrc)
       end if
       if (recvcnt(q) > 0) then
          ncrc = ncclRecv(c_devloc(recvbuf(recvdsp(q)+1)), &
                          int(recvcnt(q), c_size_t)*cb, ncclChar, &
                          int(q, c_int), par_nccl_comm, strm)
          if (ncrc /= ncclSuccess) error stop 'par_transpose: ncclRecv: '//nccl_err_str(ncrc)
       end if
    end do
    ncrc = ncclGroupEnd()
    if (ncrc /= ncclSuccess) error stop 'par_transpose: ncclGroupEnd: '//nccl_err_str(ncrc)
  end subroutine exchange_bytes

  !-----------------------------------------------------------------------------
  ! exchange_direct_a2b -- the A->B wire: per-(peer, field) sends from the
  ! PACKED stage (field f's block of q's chunk sits at sd(q) + (f-1)*sc(q)/nf,
  ! the pack kernel's own layout), per-(peer, field) recvs DIRECTLY into
  ! out_B(:, :, az0(q)+1.., f) -- the identity landing (see the module header;
  ! spec/sweep: src/models/work_m1_direct_a2a_model.py). sc/sd/rc are THIS
  ! rank's A->B send counts/disps and recv counts (alltoall_counts_A_to_B).
  ! Both ends iterate f ascending -> NCCL's one-to-one posting-order matching
  ! pairs (q,f) with (q,f). 4-D subscripts keep the address arithmetic 64-bit
  ! (no int32 linearised offset; pt_guard bounds the int32 counts/disps).
  ! Empty per-field messages are skipped (sc(q)==0 <=> sc1==0; ditto rc).
  !-----------------------------------------------------------------------------
  subroutine exchange_direct_a2b(strm, P, nf, ny, bnkx_pr, nz_e, slen, stage, &
                                 sc, sd, rc, out_B)
    integer(cuda_stream_kind), intent(in) :: strm
    integer(ip), intent(in) :: P, nf, ny, bnkx_pr, nz_e, slen
    complex(tp), device     :: stage(slen)
    complex(tp), device     :: out_B(ny, bnkx_pr, nz_e, nf)
    integer(ip), intent(in) :: sc(0:P-1), sd(0:P-1), rc(0:P-1)
    integer(ip)       :: q, f, ncrc, sc1, rc1, az0, az1, anz_q
    integer(c_size_t) :: cb
    cb = int(2*storage_size(0.0_tp)/8, c_size_t)        ! bytes per complex(tp)
    ncrc = ncclGroupStart()
    if (ncrc /= ncclSuccess) error stop 'par_transpose: ncclGroupStart: '//nccl_err_str(ncrc)
    do q = 0, P-1
       call block_range(nz_e, P, q, az0, az1, anz_q)    ! q's z-slab = the landing rows
       sc1 = sc(q)/nf;  rc1 = rc(q)/nf                  ! exact: counts carry *nf
       do f = 1, nf
          if (sc1 > 0) then
             ncrc = ncclSend(c_devloc(stage(sd(q) + (f-1)*sc1 + 1)), &
                             int(sc1, c_size_t)*cb, ncclChar, &
                             int(q, c_int), par_nccl_comm, strm)
             if (ncrc /= ncclSuccess) error stop 'par_transpose: ncclSend: '//nccl_err_str(ncrc)
          end if
          if (rc1 > 0) then
             ncrc = ncclRecv(c_devloc(out_B(1, 1, az0+1, f)), &
                             int(rc1, c_size_t)*cb, ncclChar, &
                             int(q, c_int), par_nccl_comm, strm)
             if (ncrc /= ncclSuccess) error stop 'par_transpose: ncclRecv: '//nccl_err_str(ncrc)
          end if
       end do
    end do
    ncrc = ncclGroupEnd()
    if (ncrc /= ncclSuccess) error stop 'par_transpose: ncclGroupEnd: '//nccl_err_str(ncrc)
  end subroutine exchange_direct_a2b

  !-----------------------------------------------------------------------------
  ! exchange_direct_b2a -- the B->A wire, mirror of the above: per-(peer,
  ! field) sends DIRECTLY from in_B(:, :, az0(q)+1.., f) (q's z-slab is what q
  ! gets), per-(peer, field) recvs into the stage at rdB(q) + (f-1)*rcB(q)/nf --
  ! exactly the buffer layout the unpack kernel expects. Args are the B->A-role
  ! counts/disps: scB = B->A send counts (= A->B recv counts), rcB/rdB = B->A
  ! recv counts/disps (= A->B send counts/disps) -- the same swap the monolithic
  ! caller has always done.
  !-----------------------------------------------------------------------------
  subroutine exchange_direct_b2a(strm, P, nf, ny, bnkx_pr, nz_e, rlen, in_B, &
                                 scB, rcB, rdB, stage)
    integer(cuda_stream_kind), intent(in) :: strm
    integer(ip), intent(in) :: P, nf, ny, bnkx_pr, nz_e, rlen
    complex(tp), device     :: in_B(ny, bnkx_pr, nz_e, nf)
    complex(tp), device     :: stage(rlen)
    integer(ip), intent(in) :: scB(0:P-1), rcB(0:P-1), rdB(0:P-1)
    integer(ip)       :: q, f, ncrc, sc1, rc1, az0, az1, anz_q
    integer(c_size_t) :: cb
    cb = int(2*storage_size(0.0_tp)/8, c_size_t)        ! bytes per complex(tp)
    ncrc = ncclGroupStart()
    if (ncrc /= ncclSuccess) error stop 'par_transpose: ncclGroupStart: '//nccl_err_str(ncrc)
    do q = 0, P-1
       call block_range(nz_e, P, q, az0, az1, anz_q)    ! q's z-slab = the send rows
       sc1 = scB(q)/nf;  rc1 = rcB(q)/nf                ! exact: counts carry *nf
       do f = 1, nf
          if (sc1 > 0) then
             ncrc = ncclSend(c_devloc(in_B(1, 1, az0+1, f)), &
                             int(sc1, c_size_t)*cb, ncclChar, &
                             int(q, c_int), par_nccl_comm, strm)
             if (ncrc /= ncclSuccess) error stop 'par_transpose: ncclSend: '//nccl_err_str(ncrc)
          end if
          if (rc1 > 0) then
             ncrc = ncclRecv(c_devloc(stage(rdB(q) + (f-1)*rc1 + 1)), &
                             int(rc1, c_size_t)*cb, ncclChar, &
                             int(q, c_int), par_nccl_comm, strm)
             if (ncrc /= ncclSuccess) error stop 'par_transpose: ncclRecv: '//nccl_err_str(ncrc)
          end if
       end do
    end do
    ncrc = ncclGroupEnd()
    if (ncrc /= ncclSuccess) error stop 'par_transpose: ncclGroupEnd: '//nccl_err_str(ncrc)
  end subroutine exchange_direct_b2a

  !-----------------------------------------------------------------------------
  ! pt_guard -- fail LOUDLY (not silently/corruptly) on the two latent
  ! scale hazards: (i) the slab dimensional precondition P <= min(nkx_e, nz_e)
  ! (beyond it some rank owns zero columns/planes); (ii) int32 overflow of the
  ! per-rank transpose counts/displacements (ip = int32; the per-rank send/recv
  ! TOTAL is ~ny*nkx_e*nz_e*nfield/P elements and overflows for big-grid/small-P
  ! combos, e.g. the Re2000 5120x481x4096 grid at P <= 21).
  !-----------------------------------------------------------------------------
  subroutine pt_guard(ny, nkx_e, nz_e, P, nfield)
    integer(ip), intent(in) :: ny, nkx_e, nz_e, P, nfield
    integer(8) :: bmax, amax, smax, rmax
    if (P > nkx_e .or. P > nz_e) &
       error stop 'par_transpose: slab precondition violated (P > min(nkx, nzd)): '// &
                  'reduce ranks or use a bigger grid'
    bmax = int((nkx_e + P - 1)/P, 8);  amax = int((nz_e + P - 1)/P, 8)
    smax = int(ny,8)*int(nkx_e,8)*amax*int(nfield,8)     ! per-rank send total (A->B view)
    rmax = int(ny,8)*bmax*int(nz_e,8)*int(nfield,8)      ! per-rank recv total (A->B view)
    if (max(smax, rmax) > int(huge(1_ip),8)) &
       error stop 'par_transpose: int32 overflow in transpose counts/displacements '// &
                  '(ny*nkx*nzd*nfield/P too large) -- increase P or reduce nfield'
  end subroutine pt_guard

  !-----------------------------------------------------------------------------
  ! par_transpose_A_to_B -- z-slab (layout A) -> kx-slab (layout B), batched.
  !   in_A : (ny, nkx,       a_nz_loc, nfield)   full kx, this rank's z-slab
  !   out_B: (ny, b_nkx_loc, nz,       nfield)   this rank's kx-slab, full z
  !-----------------------------------------------------------------------------
  subroutine par_transpose_A_to_B(d, ny, nkx, nz, a_nz_loc, b_nkx_loc, nfield, in_A, out_B)
    type(decomp_t), intent(in) :: d
    integer(ip),    intent(in) :: ny, nkx, nz, a_nz_loc, b_nkx_loc, nfield
    complex(tp), device :: in_A (ny, nkx,       a_nz_loc, nfield)
    complex(tp), device :: out_B(ny, b_nkx_loc, nz,       nfield)
    integer(ip) :: P, pr, slen, istat
    integer(ip) :: sendcnt(0:d%nprocs-1), senddsp(0:d%nprocs-1)
    integer(ip) :: recvcnt(0:d%nprocs-1), recvdsp(0:d%nprocs-1)

    P = d%nprocs;  pr = d%myrank
    if (P == 1) then
       out_B = in_A                                  ! identity, bit-exact
       istat = cudaDeviceSynchronize()
       return
    end if
    call pt_guard(ny, nkx, nz, P, nfield)
    call alltoall_counts_A_to_B(ny, nkx, nz, P, pr, nfield, &
                                sendcnt, senddsp, recvcnt, recvdsp)
    slen = sum(sendcnt)
    call ensure_ptbuf(slen)                          ! persistent stage (send side only)

    ! pack -> NCCL are in-stream ordered on par_stream; host barriers run
    ! only under pt_timing (they serialize the phase clocks). No unpack:
    ! the recvs land directly in out_B (identity side) -- the unpack column of a
    ! timed run reads ~0.
    if (pt_timing) then
       if (pt_rate == 0) call system_clock(count_rate=pt_rate)
       istat = cudaDeviceSynchronize()               ! drain pre-transpose compute -> pt_pack is PURE pack
       call system_clock(pt_c0)
    end if
    call pack_A_to_B(ny, nkx, nz, a_nz_loc, nfield, P, pr, in_A, slen, ptbuf)
    if (pt_timing) then
       istat = cudaDeviceSynchronize()               ! pack done -> clock the pack phase
       call system_clock(pt_c1)
    end if
    call exchange_direct_a2b(par_stream, P, nfield, ny, b_nkx_loc, nz, slen, &
                             ptbuf, sendcnt, senddsp, recvcnt, out_B)
    if (pt_timing) then
       istat = cudaStreamSynchronize(par_stream)     ! NCCL done -> clock the NCCL phase
       call system_clock(pt_c2)
       istat = cudaDeviceSynchronize()
       call system_clock(pt_c3)
       call pt_accum(pt_c0, pt_c1, pt_c2, pt_c3)
    end if
  end subroutine par_transpose_A_to_B

  !-----------------------------------------------------------------------------
  ! par_transpose_B_to_A -- kx-slab (layout B) -> z-slab (layout A), batched.
  ! Same permutation as A->B with send<->recv swapped: B->A send counts/disps are
  ! A->B RECV counts/disps, and B->A recv counts/disps are A->B SEND ones.
  !-----------------------------------------------------------------------------
  subroutine par_transpose_B_to_A(d, ny, nkx, nz, a_nz_loc, b_nkx_loc, nfield, in_B, out_A)
    type(decomp_t), intent(in) :: d
    integer(ip),    intent(in) :: ny, nkx, nz, a_nz_loc, b_nkx_loc, nfield
    complex(tp), device :: in_B (ny, b_nkx_loc, nz,       nfield)
    complex(tp), device :: out_A(ny, nkx,       a_nz_loc, nfield)
    integer(ip) :: P, pr, rlen, istat
    integer(ip) :: a2b_sc(0:d%nprocs-1), a2b_sd(0:d%nprocs-1)
    integer(ip) :: a2b_rc(0:d%nprocs-1), a2b_rd(0:d%nprocs-1)

    P = d%nprocs;  pr = d%myrank
    if (P == 1) then
       out_A = in_B                                  ! identity, bit-exact
       istat = cudaDeviceSynchronize()
       return
    end if
    call pt_guard(ny, nkx, nz, P, nfield)
    call alltoall_counts_A_to_B(ny, nkx, nz, P, pr, nfield, &
                                a2b_sc, a2b_sd, a2b_rc, a2b_rd)
    ! B->A: send <- A->B recv ; recv <- A->B send
    rlen = sum(a2b_sc)
    call ensure_ptbuf(rlen)                          ! persistent stage (recv side only)

    ! NCCL -> unpack are in-stream ordered on par_stream; host barriers
    ! run only under pt_timing (they serialize the phase clocks). No pack:
    ! the sends read directly from in_B (identity side) -- the pack column of a
    ! timed run reads ~0.
    if (pt_timing) then
       if (pt_rate == 0) call system_clock(count_rate=pt_rate)
       istat = cudaDeviceSynchronize()               ! drain pre-transpose compute
       call system_clock(pt_c0)
       call system_clock(pt_c1)                      ! no pack phase
    end if
    call exchange_direct_b2a(par_stream, P, nfield, ny, b_nkx_loc, nz, rlen, &
                             in_B, a2b_rc, a2b_sc, a2b_sd, ptbuf)
    if (pt_timing) then
       istat = cudaStreamSynchronize(par_stream)     ! NCCL done -> clock the NCCL phase
       call system_clock(pt_c2)
    end if
    call unpack_B_to_A(ny, nkx, nz, a_nz_loc, nfield, P, pr, rlen, ptbuf, out_A)
    if (pt_timing) then
       istat = cudaDeviceSynchronize()
       call system_clock(pt_c3)
       call pt_accum(pt_c0, pt_c1, pt_c2, pt_c3)
    end if
  end subroutine par_transpose_B_to_A

  !-----------------------------------------------------------------------------
  ! pt_accum / par_transpose_timing -- PERF DIAGNOSTIC accumulators (host wall).
  ! pt_accum: add one transpose's pack/NCCL/unpack wall to the running totals.
  ! par_transpose_timing: hand back the accumulated ms (pack/NCCL/unpack) + the call
  ! count, optionally zeroing for the next interval. main_par calls it around
  ! par_step_full_ws to split comm out of the step timer (bandwidth-vs-overhead probe).
  !-----------------------------------------------------------------------------
  subroutine pt_accum(c0, c1, c2, c3)
    integer(8), intent(in) :: c0, c1, c2, c3
    real(dp) :: r
    r = real(max(pt_rate, 1_8), dp)
    pt_pack_s   = pt_pack_s   + real(c1 - c0, dp)/r
    pt_nccl_s   = pt_nccl_s   + real(c2 - c1, dp)/r
    pt_unpack_s = pt_unpack_s + real(c3 - c2, dp)/r
    pt_ncalls   = pt_ncalls + 1
  end subroutine pt_accum

  subroutine par_transpose_timing(pack_ms, nccl_ms, unpack_ms, ncalls, reset)
    real(dp),    intent(out) :: pack_ms, nccl_ms, unpack_ms
    integer(ip), intent(out) :: ncalls
    logical,     intent(in)  :: reset
    pack_ms   = 1.0e3_dp*pt_pack_s
    nccl_ms   = 1.0e3_dp*pt_nccl_s
    unpack_ms = 1.0e3_dp*pt_unpack_s
    ncalls    = pt_ncalls
    if (reset) then
       pt_pack_s = 0.0_dp;  pt_nccl_s = 0.0_dp;  pt_unpack_s = 0.0_dp;  pt_ncalls = 0
    end if
  end subroutine par_transpose_timing

  ! Driver opt-in for the per-phase barriers+clocks (see pt_timing note above).
  subroutine par_transpose_set_timing(on)
    logical, intent(in) :: on
    pt_timing = on
  end subroutine par_transpose_set_timing

  !=============================================================================
  ! Pipelined-transpose phases (contract in the header). The per-field
  ! counts are the nfield=1 counts; a staged field's bytes live in slot f of
  ! the persistent buffer -- only ONE side is ever staged (A2B: the packed
  ! send at offset (f-1)*slen1; B2A: the landed recv at offset (f-1)*rlen1; the
  ! Layout-B side goes direct against the caller's fld_B slice), so the
  ! footprint is at most the monolithic nfield=nf stage and the warm-up's
  ! nfield=6 pre-sizing covers every production leg (no mid-loop growth).
  !=============================================================================
  subroutine ptp_begin(dir, d, ny, nkx, nz, a_nz_loc, b_nkx_loc, nf)
    integer(ip),    intent(in) :: dir
    type(decomp_t), intent(in) :: d
    integer(ip),    intent(in) :: ny, nkx, nz, a_nz_loc, b_nkx_loc, nf
    integer(ip) :: a2b_sc(0:d%nprocs-1), a2b_sd(0:d%nprocs-1)
    integer(ip) :: a2b_rc(0:d%nprocs-1), a2b_rd(0:d%nprocs-1)
    integer(ip) :: f, istat

    if (d%nprocs <= 1) error stop 'ptp_begin: pipelined path requires P > 1 (caller guards)'
    if (nf < 1 .or. nf > PTP_MAXF) error stop 'ptp_begin: nf out of range'
    if (ptp_dir /= 0) error stop 'ptp_begin: previous pipelined leg not ended'
    call pt_guard(ny, nkx, nz, d%nprocs, nf)

    ptp_dir = dir;  ptp_nf = nf
    ptp_P   = d%nprocs;  ptp_pr = d%myrank
    ptp_ny  = ny;  ptp_nkx = nkx;  ptp_nz = nz
    ptp_anz = a_nz_loc;  ptp_bnkx = b_nkx_loc

    if (allocated(ptp_sc)) then
       if (size(ptp_sc) /= ptp_P) deallocate(ptp_sc, ptp_sd, ptp_rc, ptp_rd)
    end if
    if (.not. allocated(ptp_sc)) &
       allocate(ptp_sc(0:ptp_P-1), ptp_sd(0:ptp_P-1), ptp_rc(0:ptp_P-1), ptp_rd(0:ptp_P-1))

    call alltoall_counts_A_to_B(ny, nkx, nz, ptp_P, ptp_pr, 1_ip, &
                                a2b_sc, a2b_sd, a2b_rc, a2b_rd)
    if (dir == PTP_A2B) then
       ptp_sc = a2b_sc;  ptp_sd = a2b_sd;  ptp_rc = a2b_rc;  ptp_rd = a2b_rd
    else if (dir == PTP_B2A) then
       ! B->A: send <- A->B recv ; recv <- A->B send (same swap as the monolithic path)
       ptp_sc = a2b_rc;  ptp_sd = a2b_rd;  ptp_rc = a2b_sc;  ptp_rd = a2b_sd
    else
       error stop 'ptp_begin: bad direction'
    end if
    ptp_slen1 = sum(ptp_sc);  ptp_rlen1 = sum(ptp_rc)
    ! A2B stages sends (nf*slen1), B2A stages recvs (nf*rlen1); size to the
    ! max so a direction flip never grow-reallocs mid-loop either.
    call ensure_ptbuf(nf*max(ptp_slen1, ptp_rlen1))

    if (.not. ptp_ev_live) then
       do f = 1, PTP_MAXF
          istat = cudaEventCreate(ptp_ev_pk(f))
          if (istat /= 0) error stop 'ptp_begin: cudaEventCreate(pk) failed'
          istat = cudaEventCreate(ptp_ev_a0(f))
          if (istat /= 0) error stop 'ptp_begin: cudaEventCreate(a0) failed'
          istat = cudaEventCreate(ptp_ev_cm(f))
          if (istat /= 0) error stop 'ptp_begin: cudaEventCreate(cm) failed'
       end do
       ptp_ev_live = .true.
    end if
  end subroutine ptp_begin

  !-- pack field f (compute stream). A2B legs ONLY -- the B2A send side is
  !   the identity and goes direct from fld_B in ptp_a2a_field; a B2A call here
  !   is a contract violation and stops loudly. fld is the CONTIGUOUS f-th
  !   slice of the caller's nf-field array, A2B shape (ny, nkx, a_nz_loc, 1)
  !   (Layout A, coarse kx). The pack-done event moved into ptp_a2a_field
  !   (called back-to-back at both call sites -- identical stream position).
  subroutine ptp_pack_field(f, fld)
    integer(ip), intent(in) :: f
    complex(tp), device     :: fld(*)
    integer(ip) :: off
    if (ptp_dir /= PTP_A2B) error stop 'ptp_pack_field: only the A2B leg packs'
    off = (f-1)*ptp_slen1
    call pack_A_to_B(ptp_ny, ptp_nkx, ptp_nz, ptp_anz, 1_ip, ptp_P, ptp_pr, &
                     fld, ptp_slen1, ptbuf(off+1:off+ptp_slen1))
  end subroutine ptp_pack_field

  !-- all-to-all of field f on the comm stream ----------------------------------
  ! fld_B is field f's Layout-B-side slice, shape (ny, bnkx, nz, 1): the DIRECT
  ! recv target (A2B) or the DIRECT send source (B2A). Records ev_pk(f) on
  ! par_stream first (the position right after the pack kernel / the leg's
  ! local FFT work for f), so the comm stream's touch of fld_B is ordered after
  ! every prior par_stream toucher.
  subroutine ptp_a2a_field(f, fld_B)
    integer(ip), intent(in) :: f
    complex(tp), device     :: fld_B(*)
    integer(ip) :: off, istat
    istat = cudaEventRecord(ptp_ev_pk(f), par_stream)
    if (istat /= 0) error stop 'ptp_a2a_field: cudaEventRecord(pk) failed'
    istat = cudaStreamWaitEvent(par_comm_stream, ptp_ev_pk(f), 0)
    if (istat /= 0) error stop 'ptp_a2a_field: cudaStreamWaitEvent failed'
    if (pt_timing) then
       istat = cudaEventRecord(ptp_ev_a0(f), par_comm_stream)   ! best-effort (timing only; elapsed guarded)
    end if
    if (ptp_dir == PTP_A2B) then
       off = (f-1)*ptp_slen1
       call exchange_direct_a2b(par_comm_stream, ptp_P, 1_ip, ptp_ny, ptp_bnkx, &
                                ptp_nz, ptp_slen1, ptbuf(off+1:off+ptp_slen1), &
                                ptp_sc, ptp_sd, ptp_rc, fld_B)
    else
       ! B2A roles: scB = ptp_sc (= a2b_rc), rcB = ptp_rc (= a2b_sc), rdB = ptp_rd
       off = (f-1)*ptp_rlen1
       call exchange_direct_b2a(par_comm_stream, ptp_P, 1_ip, ptp_ny, ptp_bnkx, &
                                ptp_nz, ptp_rlen1, fld_B, &
                                ptp_sc, ptp_rc, ptp_rd, ptbuf(off+1:off+ptp_rlen1))
    end if
    istat = cudaEventRecord(ptp_ev_cm(f), par_comm_stream)
    if (istat /= 0) error stop 'ptp_a2a_field: cudaEventRecord failed'
  end subroutine ptp_a2a_field

  !-- unpack field f (compute stream; waits on its A2A-done event) --------------
  ! fld slice shapes: A2B: (ny, bnkx, nz, 1) Layout B; B2A: (ny, nkx, a_nz_loc, 1).
  ! A2B is WAIT-ONLY (the recv landed direct in fld = fld_B). The wait is
  ! NOT optional: a leg MUST call this for EVERY f before ptp_end -- the
  ! ev_cm(f) wait is the ONLY edge ordering later par_stream work after both
  ! the fld_B landing AND the group's ptbuf-slot reads (a future caller that
  ! skips a field silently races both).
  subroutine ptp_unpack_field(f, fld)
    integer(ip), intent(in) :: f
    complex(tp), device     :: fld(*)
    integer(ip) :: off, istat
    istat = cudaStreamWaitEvent(par_stream, ptp_ev_cm(f), 0)
    if (istat /= 0) error stop 'ptp_unpack_field: cudaStreamWaitEvent failed'
    if (ptp_dir == PTP_B2A) then
       off = (f-1)*ptp_rlen1
       call unpack_B_to_A(ptp_ny, ptp_nkx, ptp_nz, ptp_anz, 1_ip, ptp_P, ptp_pr, &
                          ptp_rlen1, ptbuf(off+1:off+ptp_rlen1), fld)
    end if
  end subroutine ptp_unpack_field

  !-- end of leg: event-timed A2A occupancy under pt_timing, then reset ---------
  subroutine ptp_end()
    integer(ip) :: f, istat
    real(4)     :: ms
    if (pt_timing .and. ptp_nf > 0) then
       istat = cudaEventSynchronize(ptp_ev_cm(ptp_nf))
       do f = 1, ptp_nf
          istat = cudaEventElapsedTime(ms, ptp_ev_a0(f), ptp_ev_cm(f))
          if (istat == 0) pt_nccl_s = pt_nccl_s + real(ms, dp)/1.0e3_dp
       end do
       pt_ncalls = pt_ncalls + 1
    end if
    ptp_dir = 0;  ptp_nf = 0
  end subroutine ptp_end

end module par_transpose_mod
