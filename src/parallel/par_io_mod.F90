!===============================================================================
! par_io_mod.F90  --  DISTRIBUTED spectral checkpoint I/O.
!
! Writes/reads the restart state -- v, eta (0:n, kx-slab, nz) + the mean (0,0)
! profiles u0, w0 (0:n) -- across all ranks into ONE file via MPI-IO, in a
! P-INDEPENDENT global-kx order so a checkpoint written at P1 ranks restarts at
! any P2 (and is readable by the single-GPU oracle / a Python reader).
!
! THE "SINGLE-SHIFT" LAYOUT  (validated in
!                             src/models/work_io_layout_model.py)
!   The split axis (kx) is the MIDDLE axis of the in-memory slab (0:n, bnkx, nz).
!   To make every rank's payload ONE contiguous block we store the field on disk
!   with kx OUTERMOST -- canonical order (ny, nz, nkx) -- and rotate the local
!   slab (ny, bnkx, nz) -> (ny, nz, bnkx) on write (the mirror on read). Then
!       byte offset(rank) = HDR + field_base + (ny*nz*kx0)*elem_bytes
!       element count      = ny*nz*bnkx
!   block_range tiles [0,nkx) exactly, so the per-rank blocks partition the file
!   with no gaps/overlap and a different-P read just takes its own band's block.
!   No MPI derived datatypes (sidesteps the Alltoallv INT_MAX hazard); each rank
!   does ONE MPI_File_write_at_all of a contiguous buffer; offsets are 64-bit.
!
! FORMAT  -- mirrors io_mod.F90 v2 (1024-byte ASCII KEY=VALUE header + binary
!   payload) so the same Python reader + refuse-on-mismatch contract apply. The
!   ONLY differences from the serial io_mod are layout=slab-kx and the on-disk
!   axis order (ny,nz,nkx) recorded in v.shape/eta.shape. (The shared header
!   helpers are duplicated here rather than factored out, to keep the
!   parallel build's dependency set unchanged.)
!
! ROBUSTNESS: write to <fname>.tmp, barrier, rank-0 atomically renames to <fname>
!   (a crash mid-write never yields a half "latest"), then rank-0 updates the
!   pointer file <dir>/latest_checkpoint.txt. A no-clobber guard refuses an
!   existing <fname> unless allow_overwrite (the rewind case).
!
! TIMING: rank 0 prints "[io] wrote/read <f>: <GB> in <s> (<GB/s>)" -- UX info
!   (barrier + system_clock; good enough to see, not for scaling studies).
!
! MPI BINDING: include 'mpif.h' (NOT use mpi) -- same as par_mod.
! Build:  nvfortran -O2 -Mnoinline -cuda <ompi-incdir> -c parallel/par_io_mod.F90
!===============================================================================
module par_io_mod
  use kinds,      only: tp, wp, dp, sp, ip
  use cudafor
  use decomp_mod, only: decomp_t, block_range
  use par_mod,    only: par_rank, par_nproc
  implicit none
  include 'mpif.h'
  private
  public :: par_chk_write, par_chk_read
  public :: par_snap_write                                 ! physical u,v,w snapshot
  public :: par_io_set_hints                               ! MPI-IO write hints
  public :: par_io_set_ybasis                              ! y-discretisation stamp

  integer(ip), parameter :: IO_HDR_BYTES = 1024            ! fixed ASCII header (mult. of 8)
  character(*), parameter :: IO_MAGIC     = 'ClOpDNS'
  ! v3: coordinate records between the header and the fields, declared by
  ! the header keys 'coords' (comma list; each name is one float64 array of that
  ! axis's length: x->nx, y->ny, z->nz) and 'coord_bytes' (always 8 -- the grid
  ! definition is never degraded by an fp32 payload knob). Checkpoints store
  ! 'coords = y' (spectral in x,z; the y NODES are what the FD backend needs);
  ! snapshots store 'coords = x,y,z'. Readers of v2 files see no 'coords' key and
  ! use the old offsets unchanged; par_chk_read accepts BOTH v2 and v3.
  integer(ip), parameter :: IO_VERSION    = 3
  integer(ip), parameter :: IO_COORD_BYTES = 8

  ! Deck-settable MPI-IO hint string 'key=val[,key=val...]', applied to the
  ! WRITE opens (chk + snap). Measured: 161-GB writes at 2.1-3.6 GB/s while
  ! the same file READS back at 9.3 GB/s -- write-side headroom exists. The right
  ! keys are filesystem-specific (cb_nodes / romio_cb_write / striping_* ...), so
  ! they are a deck knob for on-cluster A/B, not hardcoded guesses. Unknown keys
  ! are ignored by the MPI library by design. Empty (default) = MPI_INFO_NULL.
  character(len=512) :: pio_hints = ''
  logical            :: pio_hints_logged = .false.

  ! y-discretisation stamp, written into every header and gated on
  ! restart. Defaults = the Chebyshev basis;
  ! the FD wiring flips them ONCE at init, WITH the node vector: 'ygrid = file'
  ! alone cannot name WHICH file, so an FD restart additionally verifies the
  ! checkpoint's v3 y record against the run's nodes.
  character(len=16)     :: pio_ybasis = 'cheb-T'
  character(len=16)     :: pio_ygrid  = 'glc'
  real(dp), allocatable :: pio_ynode(:)

contains

  subroutine par_io_set_hints(str)
    character(*), intent(in) :: str
    pio_hints = str
  end subroutine par_io_set_hints

  ! rank-uniform verdict: logical AND of a local condition across ranks.
  ! EVERY abort/success decision in the collective I/O routines routes through
  ! this, so no rank can take a different branch around a collective call (the
  ! desync class: rank-local iostat/ierr folded into an early return).
  logical function pio_all(local_ok) result(g)
    logical, intent(in) :: local_ok
    integer :: ierr
    call MPI_Allreduce(local_ok, g, 1, MPI_LOGICAL, MPI_LAND, MPI_COMM_WORLD, ierr)
  end function pio_all

  subroutine par_io_set_ybasis(basis, grid, ynodes)
    character(*), intent(in) :: basis, grid
    real(dp), intent(in), optional :: ynodes(:)
    pio_ybasis = basis;  pio_ygrid = grid
    if (allocated(pio_ynode)) deallocate(pio_ynode)
    if (present(ynodes)) then
       allocate(pio_ynode(size(ynodes)));  pio_ynode = ynodes
    end if
  end subroutine par_io_set_ybasis

  ! Build an MPI_Info from pio_hints (empty -> MPI_INFO_NULL). Caller frees
  ! it after MPI_File_open (the standard allows that; the file keeps its own copy).
  subroutine pio_make_info(info)
    integer, intent(out) :: info
    integer :: ierr, p0, p1, peq, lh
    ! len matches pio_hints (512) so long keys/values (cb_config_list or
    ! striping specs) can never be TRUNCATED: a truncated hint would apply a
    ! different hint than written, invisibly (MPI ignores unknown keys by
    ! design). At 512 truncation is impossible.
    character(len=512) :: key, val
    lh = len_trim(pio_hints)
    if (lh == 0) then
       info = MPI_INFO_NULL;  return
    end if
    call MPI_Info_create(info, ierr)
    p0 = 1
    do
       p1 = index(pio_hints(p0:lh), ',')
       if (p1 == 0) then;  p1 = lh + 1
       else;               p1 = p0 + p1 - 1
       end if
       peq = index(pio_hints(p0:p1-1), '=')
       if (peq > 1) then
          key = adjustl(pio_hints(p0:p0+peq-2))
          val = adjustl(pio_hints(p0+peq:p1-1))
          call MPI_Info_set(info, trim(key), trim(val), ierr)
          if (par_rank == 0 .and. .not. pio_hints_logged) &
             write(*,'(a,a,a,a)') '   [io] hint: ', trim(key), ' = ', trim(val)
       end if
       if (p1 > lh) exit
       p0 = p1 + 1
    end do
    pio_hints_logged = .true.
  end subroutine pio_make_info

  !=============================================================================
  ! par_chk_write -- collective distributed checkpoint write.
  !   d        : decomposition descriptor (kx-slab bounds for THIS rank)
  !   v, eta   : device spectral state (0:n, d%b_nkx_loc, nz), complex(wp)
  !   u0, w0   : device mean (0,0) profiles (0:n), complex(wp) (owner-meaningful)
  !   ok       : .true. on success (every rank gets the same verdict)
  !   allow_overwrite (optional, default .false.): permit clobbering an existing
  !              <fname> (the rewind-restart case); else an existing file -> abort.
  !=============================================================================
  subroutine par_chk_write(fname, d, n, nx, nz, Re, t, dt, Qtar, step, &
                           v, eta, u0, w0, yh, Lx, Lz, ok, allow_overwrite, fp32)
    character(*),   intent(in)  :: fname
    type(decomp_t), intent(in)  :: d
    integer(ip),    intent(in)  :: n, nx, nz, step
    real(dp),       intent(in)  :: Re, t, dt, Qtar, Lx, Lz
    complex(wp), device, intent(in) :: v(0:n,d%b_nkx_loc,nz), eta(0:n,d%b_nkx_loc,nz)
    complex(wp), device, intent(in) :: u0(0:n), w0(0:n)
    real(dp),       intent(in)  :: yh(0:n)      ! wall-normal NODES (host) -> v3 coord record
    logical,        intent(out) :: ok
    logical, optional, intent(in) :: allow_overwrite
    logical, optional, intent(in) :: fp32       ! write payload as complex(sp) (half volume)
    complex(wp), allocatable :: hbuf(:,:,:), rbuf(:,:,:), m0(:)
    complex(sp), allocatable :: r4(:,:,:), m4(:) ! downcast-on-write staging
    logical :: wr4
    integer :: rbf                               ! file real_bytes (4 or wp bytes)
    character(len=IO_HDR_BYTES) :: hdr
    character(len=len(fname)+4) :: ftmp
    integer :: pinfo, ierr2                  ! write-open hints
    integer :: ny, nkx, bnkx, kx0, fh, ierr, istat, mpitype, cnt, st(MPI_STATUS_SIZE)
    integer(kind=MPI_OFFSET_KIND) :: disp, eb, fld_elems, hdrb, total, cshift
    integer(8) :: cnt64, t0, t1, trate
    real(dp) :: secs, gb
    logical :: clob, exq
    logical :: wok                               ! accumulated write verdict

    ok = .false.
    ny = n + 1;  nkx = nx/2 + 1;  bnkx = d%b_nkx_loc;  kx0 = d%b_kx0
    cshift = int(ny,MPI_OFFSET_KIND)*int(IO_COORD_BYTES,MPI_OFFSET_KIND)  ! y(ny) record
    clob = .false.;  if (present(allow_overwrite)) clob = allow_overwrite
    ! checkpoint-precision knob: the PAYLOAD precision is independent of
    ! the run tier -- header-recorded (real_bytes), reader promotes on restart.
    wr4 = .false.;  if (present(fp32)) wr4 = fp32
    rbf = storage_size(1.0_wp)/8;  if (wr4) rbf = 4
    mpitype = merge(MPI_COMPLEX, MPI_DOUBLE_COMPLEX, storage_size(1.0_wp) == 32)
    if (wr4) mpitype = MPI_COMPLEX
    eb   = int(2*rbf, MPI_OFFSET_KIND)                                ! bytes per complex on DISK
    hdrb = int(IO_HDR_BYTES, MPI_OFFSET_KIND)
    fld_elems = int(ny,MPI_OFFSET_KIND)*int(nz,MPI_OFFSET_KIND)*int(nkx,MPI_OFFSET_KIND)

    ! per-rank element count (one big field) -- guard the default-int MPI count arg.
    ! the verdict is ALLREDUCED: block_range makes cnt64 differ across ranks
    ! when mod(nkx,P) /= 0, so a rank-local return here could strand the others
    ! in the collective open below.
    cnt64 = int(ny,8)*int(nz,8)*int(bnkx,8)
    if (.not. pio_all(cnt64 <= int(huge(0),8))) then
       if (par_rank == 0) write(*,'(a,i0,a)') &
          ' !! par_chk_write: per-rank count ', cnt64, ' exceeds INT_MAX -- raise P (or add chunked I/O).'
       return
    end if
    cnt = int(cnt64)

    ! no-clobber guard on the FINAL name (rank 0 decides; broadcast the verdict)
    if (par_rank == 0) then
       inquire(file=trim(fname), exist=exq)
       if (exq .and. .not. clob) then
          write(*,'(a,a,a)') ' !! par_chk_write: ', trim(fname), ' exists (set allow_overwrite for rewind). Aborting write.'
       end if
    end if
    call MPI_Bcast(exq, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, ierr)
    if (exq .and. .not. clob) return

    ftmp = trim(fname)//'.tmp'
    call system_clock(count_rate=trate)
    call MPI_Barrier(MPI_COMM_WORLD, ierr);  call system_clock(t0)

    ! collective create+open of the .tmp (deck hints, empty -> INFO_NULL)
    ! verdict allreduced: a rank-local open failure must not leave the other
    ! ranks in the collective writes below.
    call pio_make_info(pinfo)
    call MPI_File_open(MPI_COMM_WORLD, trim(ftmp), IOR(MPI_MODE_CREATE, MPI_MODE_WRONLY), &
                       pinfo, fh, ierr)
    if (pinfo /= MPI_INFO_NULL) call MPI_Info_free(pinfo, ierr2)
    if (.not. pio_all(ierr == MPI_SUCCESS)) then
       if (par_rank == 0) write(*,'(a,a)') ' !! par_chk_write: MPI_File_open failed for ', trim(ftmp)
       ! deliberately NO subset close: MPI_File_close is collective
       ! over the file group, and on a MIXED open verdict only some ranks hold a
       ! valid handle -- a subset close is non-conforming and can itself hang.
       ! Leaking one handle on this already-fatal path is the safe choice.
       return
    end if
    ! set the exact final size up front -> a stale/larger .tmp leaves no trailing
    ! garbage. NOTE this is why a file-SIZE check can never validate a
    ! checkpoint: even a fully failed payload write leaves a correct-size sparse
    ! file. Validity = the write ierr/count verdict below, nothing else.
    total = hdrb + cshift + 2_MPI_OFFSET_KIND*fld_elems*eb + 2_MPI_OFFSET_KIND*int(ny,MPI_OFFSET_KIND)*eb
    call MPI_File_set_size(fh, total, ierr)
    wok = (ierr == MPI_SUCCESS)

    ! from here to the close, EVERY MPI-IO call feeds wok (ierr AND the
    ! transferred count from the status object -- a short write is a failure)
    ! and NOTHING returns early: the writes are collective, so all ranks stay
    ! in lockstep and the verdict is reduced once at the end.

    ! header (rank 0 only) at offset 0, then the y coordinate record
    ! (float64 regardless of the payload knob) right after it
    if (par_rank == 0) then
       call build_header(hdr, n, nx, nz, Re, Qtar, t, dt, step, Lx, Lz, rbf)
       call MPI_File_write_at(fh, 0_MPI_OFFSET_KIND, hdr, IO_HDR_BYTES, MPI_CHARACTER, st, ierr)
       wok = wok .and. pio_xfer_ok(ierr, st, MPI_CHARACTER, IO_HDR_BYTES)
       call MPI_File_write_at(fh, hdrb, yh, ny, MPI_DOUBLE_PRECISION, st, ierr)
       wok = wok .and. pio_xfer_ok(ierr, st, MPI_DOUBLE_PRECISION, ny)
    end if

    ! --- v, then eta: rotate (ny,bnkx,nz)->(ny,nz,bnkx) and write the rank's block
    allocate(hbuf(0:n,bnkx,nz), rbuf(0:n,nz,bnkx))
    if (wr4) allocate(r4(0:n,nz,bnkx))
    ! v  (field index 0)
    hbuf = v;  istat = cudaDeviceSynchronize()
    call rotate_fwd(ny, bnkx, nz, hbuf, rbuf)
    disp = hdrb + cshift + int(ny,MPI_OFFSET_KIND)*int(nz,MPI_OFFSET_KIND)*int(kx0,MPI_OFFSET_KIND)*eb
    if (wr4) then
       r4 = cmplx(rbuf, kind=sp)                 ! downcast-on-write
       call MPI_File_write_at_all(fh, disp, r4, cnt, mpitype, st, ierr)
    else
       call MPI_File_write_at_all(fh, disp, rbuf, cnt, mpitype, st, ierr)
    end if
    wok = wok .and. pio_xfer_ok(ierr, st, mpitype, cnt)
    ! eta (field index 1)
    hbuf = eta;  istat = cudaDeviceSynchronize()
    call rotate_fwd(ny, bnkx, nz, hbuf, rbuf)
    disp = hdrb + cshift + fld_elems*eb + int(ny,MPI_OFFSET_KIND)*int(nz,MPI_OFFSET_KIND)*int(kx0,MPI_OFFSET_KIND)*eb
    if (wr4) then
       r4 = cmplx(rbuf, kind=sp)
       call MPI_File_write_at_all(fh, disp, r4, cnt, mpitype, st, ierr)
    else
       call MPI_File_write_at_all(fh, disp, rbuf, cnt, mpitype, st, ierr)
    end if
    wok = wok .and. pio_xfer_ok(ierr, st, mpitype, cnt)
    deallocate(hbuf, rbuf)
    if (wr4) deallocate(r4)

    ! --- means u0, w0 (tiny, global): the kx=0 owner writes them once
    if (d%owns_mode00) then
       allocate(m0(0:n))
       if (wr4) allocate(m4(0:n))
       m0 = u0;  istat = cudaDeviceSynchronize()
       disp = hdrb + cshift + 2_MPI_OFFSET_KIND*fld_elems*eb
       if (wr4) then
          m4 = cmplx(m0, kind=sp)
          call MPI_File_write_at(fh, disp, m4, ny, mpitype, st, ierr)
       else
          call MPI_File_write_at(fh, disp, m0, ny, mpitype, st, ierr)
       end if
       wok = wok .and. pio_xfer_ok(ierr, st, mpitype, ny)
       m0 = w0;  istat = cudaDeviceSynchronize()
       disp = disp + int(ny,MPI_OFFSET_KIND)*eb
       if (wr4) then
          m4 = cmplx(m0, kind=sp)
          call MPI_File_write_at(fh, disp, m4, ny, mpitype, st, ierr)
       else
          call MPI_File_write_at(fh, disp, m0, ny, mpitype, st, ierr)
       end if
       wok = wok .and. pio_xfer_ok(ierr, st, mpitype, ny)
       deallocate(m0)
       if (wr4) deallocate(m4)
    end if

    call MPI_File_close(fh, ierr)
    wok = wok .and. (ierr == MPI_SUCCESS)        ! close flushes -- a failure here is a failure
    call MPI_Barrier(MPI_COMM_WORLD, ierr);  call system_clock(t1)

    ! ONE reduced verdict over every write on every rank. Only a fully
    ! verified .tmp is renamed onto the final name; a bad .tmp is left in place
    ! (evidence) and ok=.false. goes back to the driver, which then keeps its
    ! existing checkpoints instead of rolling them.
    wok = pio_all(wok)
    if (par_rank == 0) then
       if (wok) then
          call execute_command_line('mv -f '//trim(ftmp)//' '//trim(fname), wait=.true., exitstat=istat)
          if (istat == 0) call write_pointer(fname)
          ok = (istat == 0)
          if (ok) then
             secs = real(t1-t0,dp)/real(max(trate,1_8),dp)
             gb   = real(total, dp)/1.0e9_dp
             write(*,'(a,a,a,f8.3,a,f7.2,a,f6.1,a)') '   [io] wrote ', trim(fname), ' : ', gb, ' GB in ', &
                  secs, ' s (', gb/max(secs,1.0e-9_dp), ' GB/s)'
          else
             write(*,'(a,a,a)') ' !! par_chk_write: rename .tmp -> ', trim(fname), ' FAILED -- checkpoint NOT committed'
          end if
       else
          ok = .false.
          write(*,'(a,a,a)') ' !! par_chk_write: MPI-IO write/count FAILURE on ', trim(ftmp), &
             ' (disk full / quota / FS error?) -- checkpoint NOT committed, .tmp left for inspection'
       end if
    end if
    call MPI_Bcast(ok, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, ierr)
  end subroutine par_chk_write

  ! one transfer verdict: MPI success AND the status count matches what was
  ! requested (a short transfer against a pre-sized sparse file is the silent
  ! failure mode MPI_File_set_size creates -- ierr alone does not catch it).
  logical function pio_xfer_ok(ierr, st, dtype, want) result(g)
    integer, intent(in) :: ierr, st(MPI_STATUS_SIZE), dtype, want
    integer :: got, ierr2
    g = (ierr == MPI_SUCCESS)
    if (g) then
       call MPI_Get_count(st, dtype, got, ierr2)
       g = (ierr2 == MPI_SUCCESS) .and. (got == want)
    end if
  end function pio_xfer_ok

  !=============================================================================
  ! par_chk_read -- collective distributed restart read. Validates the header
  ! (refuse on grid/precision/xtype/layout/fields mismatch), then each rank pulls
  ! ITS band's contiguous block (this run's P -> cross-P restart) and un-rotates
  ! into the device state. Scalars come back from the header.
  !=============================================================================
  subroutine par_chk_read(fname, d, n, nx, nz, Re, t, dt, Qtar, step, &
                          v, eta, u0, w0, ok, Lxf, Lzf)
    character(*),   intent(in)  :: fname
    type(decomp_t), intent(in)  :: d
    integer(ip),    intent(in)  :: n, nx, nz
    real(dp),       intent(out) :: Re, t, dt, Qtar
    integer(ip),    intent(out) :: step
    complex(wp), device, intent(out) :: v(0:n,d%b_nkx_loc,nz), eta(0:n,d%b_nkx_loc,nz)
    complex(wp), device, intent(out) :: u0(0:n), w0(0:n)
    logical,        intent(out) :: ok
    real(dp), optional, intent(out) :: Lxf, Lzf  ! header box (-1 = absent)
    complex(wp), allocatable :: hbuf(:,:,:), rbuf(:,:,:), m0(:)
    complex(sp), allocatable :: r4(:,:,:), m4(:) ! fp32-payload staging
    real(dp), allocatable :: ycheck(:)           ! FD y-record verify
    logical :: rd4, hasy
    character(len=IO_HDR_BYTES) :: hdr
    character(len=64) :: val
    integer :: ny, nkx, bnkx, kx0, fh, ierr, istat, mpitype, cnt, st(MPI_STATUS_SIZE)
    integer(kind=MPI_OFFSET_KIND) :: disp, eb, fld_elems, hdrb, cshift
    integer(8) :: cnt64, t0, t1, trate
    integer :: fnx, fny, fnz, fstep, rb, fver, cbyt, ierr2
    logical :: rok                               ! accumulated read verdict
    logical :: okp, gate
    real(dp) :: secs, gb

    ok = .false.;  Re = 0; t = 0; dt = 0; Qtar = 0; step = 0
    ny = n + 1;  nkx = nx/2 + 1;  bnkx = d%b_nkx_loc;  kx0 = d%b_kx0
    mpitype = merge(MPI_COMPLEX, MPI_DOUBLE_COMPLEX, storage_size(1.0_wp) == 32)
    eb   = int(2 * (storage_size(1.0_wp)/8), MPI_OFFSET_KIND)
    hdrb = int(IO_HDR_BYTES, MPI_OFFSET_KIND)
    fld_elems = int(ny,MPI_OFFSET_KIND)*int(nz,MPI_OFFSET_KIND)*int(nkx,MPI_OFFSET_KIND)
    ! allreduced: cnt64 differs across ranks when mod(nkx,P) /= 0 -- a
    ! rank-local return would strand the rest in the collective open (10.1a).
    cnt64 = int(ny,8)*int(nz,8)*int(bnkx,8)
    if (.not. pio_all(cnt64 <= int(huge(0),8))) then
       if (par_rank == 0) write(*,'(a,i0,a)') &
          ' !! par_chk_read: per-rank count ', cnt64, ' exceeds INT_MAX -- raise P (or add chunked I/O).'
       return
    end if
    cnt = int(cnt64)

    call system_clock(count_rate=trate)
    call MPI_Barrier(MPI_COMM_WORLD, ierr);  call system_clock(t0)

    call MPI_File_open(MPI_COMM_WORLD, trim(fname), MPI_MODE_RDONLY, MPI_INFO_NULL, fh, ierr)
    if (.not. pio_all(ierr == MPI_SUCCESS)) then
       if (par_rank == 0) write(*,'(a,a)') ' !! par_chk_read: cannot open ', trim(fname)
       ! deliberately NO subset close: MPI_File_close is collective
       ! over the file group, and on a MIXED open verdict only some ranks hold a
       ! valid handle -- a subset close is non-conforming and can itself hang.
       ! Leaking one handle on this already-fatal path is the safe choice.
       return
    end if

    ! every rank reads + validates the 1 KB header (independent reads, same bytes)
    call MPI_File_read_at(fh, 0_MPI_OFFSET_KIND, hdr, IO_HDR_BYTES, MPI_CHARACTER, st, ierr)
    gate = (ierr == MPI_SUCCESS)
    call hdr_get(hdr,'magic',val,okp);       gate = gate .and. okp .and. (trim(val)==IO_MAGIC)
    ! endian gate -- refuse-don't-misread parity with the serial
    ! reader (the header is ASCII, so a big-endian file passed every other gate
    ! while its binary payload would be silently misread). build_header has
    ! always stamped the key, so absent = malformed, refuse.
    call hdr_get(hdr,'endian',val,okp);      gate = gate .and. okp .and. (trim(val)==trim(endian_tag()))
    ! accept v2 (no coordinate records) AND v3 (coord records after the
    ! header, declared by 'coords'/'coord_bytes'); refuse anything newer than we
    ! understand rather than misplacing every payload offset.
    fver = 2                                            ! absent key = legacy v2
    call hdr_get(hdr,'version',val,okp);     if (okp) read(val,*,iostat=istat) fver
    gate = gate .and. (fver == 2 .or. fver == 3)
    call hdr_get(hdr,'layout',val,okp);      gate = gate .and. okp .and. (trim(val)=='slab-kx')
    call hdr_get(hdr,'xtype',val,okp);       gate = gate .and. okp .and. (trim(val)=='r2c')
    call hdr_get(hdr,'space',val,okp);       gate = gate .and. okp .and. (trim(val)=='spectral')
    call hdr_get(hdr,'fields',val,okp);      gate = gate .and. okp .and. (trim(val)=='v,eta,u0,w0')
    ! y-discretisation gate, against the RUN's stamp (cheb-T/glc
    ! by default; the FD wiring sets fd-point/file). A cross-read in EITHER
    ! direction would pass every other gate and silently corrupt (T_k coefficients
    ! vs point values have the same shape). Absent keys = legacy file = cheb-T/glc
    ! -> acceptable ONLY when the run itself is Chebyshev.
    call hdr_get(hdr,'ybasis',val,okp)
    if (okp) then;  gate = gate .and. (trim(val) == trim(pio_ybasis))
    else;           gate = gate .and. (trim(pio_ybasis) == 'cheb-T');  end if
    call hdr_get(hdr,'ygrid',val,okp)
    if (okp) then;  gate = gate .and. (trim(val) == trim(pio_ygrid))
    else;           gate = gate .and. (trim(pio_ygrid) == 'glc');  end if
    call hdr_get(hdr,'real_bytes',val,okp);  if (okp) read(val,*,iostat=istat) rb
    ! accept the native precision OR an fp32 payload (promoted on read;
    ! restart-from-sp-checkpoint is the documented half-volume trade). Anything
    ! else (e.g. an fp64 file in a hypothetical wp=sp build) is refused.
    gate = gate .and. okp .and. (rb == storage_size(1.0_wp)/8 .or. rb == 4)
    call hdr_get(hdr,'nx',val,okp); if (okp) read(val,*,iostat=istat) fnx
    call hdr_get(hdr,'ny',val,okp); if (okp) read(val,*,iostat=istat) fny
    call hdr_get(hdr,'nz',val,okp); if (okp) read(val,*,iostat=istat) fnz
    gate = gate .and. (fnx==nx .and. fny==ny .and. fnz==nz)
    ! the gate folds a rank-local read ierr -> ALLREDUCE before acting, so
    ! one node's I/O error cannot desync the collective close/reads (10.1b).
    gate = pio_all(gate)
    if (.not. gate) then
       if (par_rank == 0) write(*,'(a,a)') ' !! par_chk_read: header mismatch/invalid in ', trim(fname)
       call MPI_File_close(fh, ierr);  return
    end if
    ! scalar header values must PARSE, not silently default: a
    ! header whose gate keys are intact but whose 't = ...' value string is
    ! corrupted would otherwise restart the campaign clock from t=0 on real
    ! field data. Absent OR unparseable -> refuse (rank-uniform via pio_all).
    gate = .true.
    call hdr_get(hdr,'Re',  val,okp); gate = gate .and. okp
    if (okp) then; read(val,*,iostat=istat) Re;   gate = gate .and. istat==0; end if
    call hdr_get(hdr,'Qtar',val,okp); gate = gate .and. okp
    if (okp) then; read(val,*,iostat=istat) Qtar; gate = gate .and. istat==0; end if
    call hdr_get(hdr,'t',   val,okp); gate = gate .and. okp
    if (okp) then; read(val,*,iostat=istat) t;    gate = gate .and. istat==0; end if
    call hdr_get(hdr,'dt',  val,okp); gate = gate .and. okp
    if (okp) then; read(val,*,iostat=istat) dt;   gate = gate .and. istat==0; end if
    call hdr_get(hdr,'step',val,okp); gate = gate .and. okp
    if (okp) then
       read(val,*,iostat=istat) fstep;  gate = gate .and. istat==0
       if (istat == 0) step = fstep
    end if
    gate = pio_all(gate)
    if (.not. gate) then
       if (par_rank == 0) write(*,'(a,a)') &
          ' !! par_chk_read: scalar header value absent/unparseable (Re/Qtar/t/dt/step) in ', trim(fname)
       call MPI_File_close(fh, ierr);  return
    end if
    ! box identity back to the caller (optional; -1 = key absent, caller
    ! skips its gate). PRESENT-but-unparseable refuses: treating a corrupt
    ! Lx as absent would silently skip the caller's fatal box-mismatch gate.
    gate = .true.
    if (present(Lxf)) then
       Lxf = -1.0_dp
       call hdr_get(hdr,'Lx',val,okp)
       if (okp) then; read(val,*,iostat=istat) Lxf; gate = gate .and. istat==0; end if
    end if
    if (present(Lzf)) then
       Lzf = -1.0_dp
       call hdr_get(hdr,'Lz',val,okp)
       if (okp) then; read(val,*,iostat=istat) Lzf; gate = gate .and. istat==0; end if
    end if
    gate = pio_all(gate)
    if (.not. gate) then
       if (par_rank == 0) write(*,'(a,a)') &
          ' !! par_chk_read: box key (Lx/Lz) present but unparseable in ', trim(fname)
       call MPI_File_close(fh, ierr);  return
    end if

    ! file element size/type follow the HEADER (rb), not the run tier
    rd4 = (rb == 4 .and. storage_size(1.0_wp)/8 /= 4)
    eb  = int(2*rb, MPI_OFFSET_KIND)
    if (rb == 4) mpitype = MPI_COMPLEX

    ! v3 coordinate block between header and fields: SKIP it (the run's
    ! own grid is authoritative here; the ybasis/ygrid gates above already refuse
    ! a different discretization). Size = sum over the 'coords' names of that
    ! axis's length, times coord_bytes. v2: no key -> cshift = 0, old offsets.
    cshift = 0_MPI_OFFSET_KIND
    hasy = .false.;  cbyt = IO_COORD_BYTES   ! defined even for v2 (.and. is non-short-circuit)
    if (fver >= 3) then
       call hdr_get(hdr,'coord_bytes',val,okp);  if (okp) read(val,*,iostat=istat) cbyt
       call hdr_get(hdr,'coords',val,okp)
       if (okp) then
          if (index(val,'x') > 0) cshift = cshift + int(nx,MPI_OFFSET_KIND)
          if (index(val,'y') > 0) then
             cshift = cshift + int(ny,MPI_OFFSET_KIND)
             ! checkpoints store 'coords = y' -> the y record sits FIRST, at hdrb
             hasy = (index(val,'x') == 0)
          end if
          if (index(val,'z') > 0) cshift = cshift + int(nz,MPI_OFFSET_KIND)
          cshift = cshift * int(cbyt,MPI_OFFSET_KIND)
       end if
    end if

    ! FD restart: the fd-point/file stamp cannot name WHICH grid file,
    ! so verify the checkpoint's y record against the run's nodes. Every rank
    ! checks independently (a ny*8-byte read; same bytes -> same verdict). The
    ! 1e-12 slack tolerates a reformatted-but-identical grid file; any actually
    ! different grid differs at >= 1e-4.
    if (trim(pio_ybasis) == 'fd-point') then
       gate = hasy .and. allocated(pio_ynode) .and. (cbyt == IO_COORD_BYTES)
       if (gate) gate = (size(pio_ynode) == ny)
       if (gate) then
          allocate(ycheck(ny))
          call MPI_File_read_at(fh, hdrb, ycheck, ny, MPI_DOUBLE_PRECISION, st, ierr)
          gate = (ierr == MPI_SUCCESS)
          if (gate) gate = (maxval(abs(ycheck - pio_ynode)) <= 1.0e-12_dp)
          deallocate(ycheck)
       end if
       gate = pio_all(gate)                      ! rank-local read ierr -> uniform verdict
       if (.not. gate) then
          if (par_rank == 0) write(*,'(a,a)') &
             ' !! par_chk_read: FD y-grid mismatch (checkpoint nodes /= run nodes) in ', trim(fname)
          call MPI_File_close(fh, ierr);  return
       end if
    end if

    ! each rank reads ITS band's contiguous block and un-rotates into device memory.
    ! every read feeds rok (ierr AND transferred count): the header of a
    ! payload-truncated file is intact and passes every gate above, and the
    ! MPI_File_set_size pre-sizing on the write side means even the SIZE is
    ! right -- short/failed reads here are the only place truncation shows.
    rok = .true.
    allocate(hbuf(0:n,bnkx,nz), rbuf(0:n,nz,bnkx))
    if (rd4) allocate(r4(0:n,nz,bnkx))
    disp = hdrb + cshift + int(ny,MPI_OFFSET_KIND)*int(nz,MPI_OFFSET_KIND)*int(kx0,MPI_OFFSET_KIND)*eb
    if (rd4) then
       call MPI_File_read_at_all(fh, disp, r4, cnt, mpitype, st, ierr)
       rbuf = r4                                  ! promote sp -> wp
    else
       call MPI_File_read_at_all(fh, disp, rbuf, cnt, mpitype, st, ierr)
    end if
    rok = rok .and. pio_xfer_ok(ierr, st, mpitype, cnt)
    call rotate_inv(ny, bnkx, nz, rbuf, hbuf);  v = hbuf;  istat = cudaDeviceSynchronize()
    disp = hdrb + cshift + fld_elems*eb + int(ny,MPI_OFFSET_KIND)*int(nz,MPI_OFFSET_KIND)*int(kx0,MPI_OFFSET_KIND)*eb
    if (rd4) then
       call MPI_File_read_at_all(fh, disp, r4, cnt, mpitype, st, ierr)
       rbuf = r4
    else
       call MPI_File_read_at_all(fh, disp, rbuf, cnt, mpitype, st, ierr)
    end if
    rok = rok .and. pio_xfer_ok(ierr, st, mpitype, cnt)
    call rotate_inv(ny, bnkx, nz, rbuf, hbuf);  eta = hbuf;  istat = cudaDeviceSynchronize()
    deallocate(hbuf, rbuf)
    if (rd4) deallocate(r4)

    ! means: this run's kx=0 owner reads them; everyone else holds zero (IC convention)
    u0 = (0.0_wp,0.0_wp);  w0 = (0.0_wp,0.0_wp)
    if (d%owns_mode00) then
       allocate(m0(0:n))
       if (rd4) allocate(m4(0:n))
       disp = hdrb + cshift + 2_MPI_OFFSET_KIND*fld_elems*eb
       if (rd4) then
          call MPI_File_read_at(fh, disp, m4, ny, mpitype, st, ierr);  m0 = m4
       else
          call MPI_File_read_at(fh, disp, m0, ny, mpitype, st, ierr)
       end if
       rok = rok .and. pio_xfer_ok(ierr, st, mpitype, ny)
       u0 = m0
       disp = disp + int(ny,MPI_OFFSET_KIND)*eb
       if (rd4) then
          call MPI_File_read_at(fh, disp, m4, ny, mpitype, st, ierr);  m0 = m4
       else
          call MPI_File_read_at(fh, disp, m0, ny, mpitype, st, ierr)
       end if
       rok = rok .and. pio_xfer_ok(ierr, st, mpitype, ny)
       w0 = m0
       istat = cudaDeviceSynchronize();  deallocate(m0)
       if (rd4) deallocate(m4)
    end if

    call MPI_File_close(fh, ierr)
    call MPI_Barrier(MPI_COMM_WORLD, ierr);  call system_clock(t1)

    ! ONE reduced verdict; ok=.false. on any failed/short read anywhere.
    ! (was: unconditional ok=.true. -- a truncated checkpoint restarted from
    ! whatever the staging buffers held.)
    ok = pio_all(rok)
    if (par_rank == 0) then
       if (ok) then
          secs = real(t1-t0,dp)/real(max(trate,1_8),dp)
          gb   = real(hdrb + cshift + 2_MPI_OFFSET_KIND*fld_elems*eb + 2_MPI_OFFSET_KIND*int(ny,MPI_OFFSET_KIND)*eb, dp)/1.0e9_dp
          write(*,'(a,a,a,f8.3,a,f7.2,a,f6.1,a)') '   [io] read  ', trim(fname), ' : ', gb, ' GB in ', &
               secs, ' s (', gb/max(secs,1.0e-9_dp), ' GB/s)'
       else
          write(*,'(a,a,a)') ' !! par_chk_read: payload read/count FAILURE in ', trim(fname), &
             ' (truncated checkpoint?) -- restart REFUSED'
       end if
    end if
  end subroutine par_chk_read

  !=============================================================================
  ! par_snap_write -- collective PHYSICAL u,v,w snapshot (clean nx*ny*nz grid).
  ! Each field is a z-slab (Layout A): the split axis z is ALREADY the outermost
  ! (slowest) dim, so each rank's payload is ONE contiguous block at
  ! offset ny*nx*a_z0 -- NO rotation (the easy half of the single-shift scheme).
  ! Fields are real(tp); written at snap_bytes precision (4=fp32 default, 8=fp64)
  ! by downcast-on-write. Atomic .tmp->rename + latest_snapshot.txt pointer; this
  ! is OUTPUT (numbered keep-all in the driver) so there is no no-clobber guard.
  !   u, v, w : real(tp) device (ny, nx, a_nz_loc), ny=n+1, COARSE z-slab.
  !=============================================================================
  ! Optional 4th field p (pressure): same shape/grid as u,v,w.
  ! Present -> nfields=4, written after w; the header records the manifest so the
  ! Python reader (read_snapshot.py) loops the declared fields. Absent -> the
  ! original 3-field snapshot, byte-for-byte unchanged.
  subroutine par_snap_write(fname, d, n, nx, nz, Re, t, step, snap_bytes, u, v, w, yh, Lx, Lz, ok, p)
    character(*),   intent(in)  :: fname
    type(decomp_t), intent(in)  :: d
    integer(ip),    intent(in)  :: n, nx, nz, step, snap_bytes
    real(dp),       intent(in)  :: Re, t, Lx, Lz
    real(tp), device, intent(in) :: u(n+1,nx,d%a_nz_loc), v(n+1,nx,d%a_nz_loc), w(n+1,nx,d%a_nz_loc)
    real(dp),       intent(in)  :: yh(0:n)   ! wall-normal NODES (host) -> v3 coord record
    real(tp), device, optional, intent(in) :: p(n+1,nx,d%a_nz_loc)
    logical,        intent(out) :: ok
    real(tp), allocatable :: ht(:,:,:)
    real(sp), allocatable :: hs(:,:,:)
    real(dp), allocatable :: hd(:,:,:)
    real(dp), allocatable :: xco(:), zco(:)  ! uniform periodic x,z (no endpoint)
    character(len=IO_HDR_BYTES) :: hdr
    character(len=len(fname)+4)  :: ftmp
    integer :: ny, anz, z0, fh, ierr, istat, mpitype, cnt, st(MPI_STATUS_SIZE), f, nfields
    integer :: pinfo, ierr2                  ! write-open hints
    integer(kind=MPI_OFFSET_KIND) :: disp, eb, fld_elems, hdrb, total, cshift
    integer(8) :: cnt64, t0, t1, trate
    real(dp) :: secs, gb
    logical :: wok                               ! accumulated write verdict

    ok = .false.
    ny = n+1;  anz = d%a_nz_loc;  z0 = d%a_z0
    cshift = int(nx+ny+nz,MPI_OFFSET_KIND)*int(IO_COORD_BYTES,MPI_OFFSET_KIND)  ! x,y,z records
    nfields = 3;  if (present(p)) nfields = 4
    eb = int(snap_bytes, MPI_OFFSET_KIND)
    mpitype = merge(MPI_REAL, MPI_DOUBLE_PRECISION, snap_bytes == 4)
    hdrb = int(IO_HDR_BYTES, MPI_OFFSET_KIND)
    fld_elems = int(ny,MPI_OFFSET_KIND)*int(nx,MPI_OFFSET_KIND)*int(nz,MPI_OFFSET_KIND)
    ! allreduced: anz differs across ranks when mod(nz,P) /= 0 -- same
    ! non-uniform class as the checkpoint guards.
    cnt64 = int(ny,8)*int(nx,8)*int(anz,8)
    if (.not. pio_all(cnt64 <= int(huge(0),8))) then
       if (par_rank == 0) write(*,'(a)') ' !! par_snap_write: per-rank count exceeds INT_MAX -- raise P.'
       return
    end if
    cnt = int(cnt64)

    ftmp = trim(fname)//'.tmp'
    call system_clock(count_rate=trate)
    call MPI_Barrier(MPI_COMM_WORLD, ierr);  call system_clock(t0)
    call pio_make_info(pinfo)                ! deck hints, empty -> INFO_NULL
    call MPI_File_open(MPI_COMM_WORLD, trim(ftmp), IOR(MPI_MODE_CREATE, MPI_MODE_WRONLY), &
                       pinfo, fh, ierr)
    if (pinfo /= MPI_INFO_NULL) call MPI_Info_free(pinfo, ierr2)
    if (.not. pio_all(ierr == MPI_SUCCESS)) then
       if (par_rank == 0) write(*,'(a,a)') ' !! par_snap_write: MPI_File_open failed for ', trim(ftmp)
       ! deliberately NO subset close: MPI_File_close is collective
       ! over the file group, and on a MIXED open verdict only some ranks hold a
       ! valid handle -- a subset close is non-conforming and can itself hang.
       ! Leaking one handle on this already-fatal path is the safe choice.
       return
    end if
    total = hdrb + cshift + int(nfields,MPI_OFFSET_KIND)*fld_elems*eb
    call MPI_File_set_size(fh, total, ierr)
    wok = (ierr == MPI_SUCCESS)              ! accumulate every transfer verdict
    if (par_rank == 0) then
       call build_snap_header(hdr, n, nx, nz, Re, t, step, Lx, Lz, snap_bytes, nfields)
       call MPI_File_write_at(fh, 0_MPI_OFFSET_KIND, hdr, IO_HDR_BYTES, MPI_CHARACTER, st, ierr)
       wok = wok .and. pio_xfer_ok(ierr, st, MPI_CHARACTER, IO_HDR_BYTES)
       ! coordinate records x(nx), y(ny), z(nz), float64, right after the
       ! header (always, independent of snap_bytes). x,z are the uniform periodic
       ! grids x_i = (i-1)*Lx/nx (no repeated endpoint); y is the caller's nodes.
       allocate(xco(nx), zco(nz))
       do f = 1, nx;  xco(f) = real(f-1,dp)*Lx/real(nx,dp);  end do
       do f = 1, nz;  zco(f) = real(f-1,dp)*Lz/real(nz,dp);  end do
       disp = hdrb
       call MPI_File_write_at(fh, disp, xco, nx, MPI_DOUBLE_PRECISION, st, ierr)
       wok = wok .and. pio_xfer_ok(ierr, st, MPI_DOUBLE_PRECISION, nx)
       disp = disp + int(nx,MPI_OFFSET_KIND)*int(IO_COORD_BYTES,MPI_OFFSET_KIND)
       call MPI_File_write_at(fh, disp, yh, ny, MPI_DOUBLE_PRECISION, st, ierr)
       wok = wok .and. pio_xfer_ok(ierr, st, MPI_DOUBLE_PRECISION, ny)
       disp = disp + int(ny,MPI_OFFSET_KIND)*int(IO_COORD_BYTES,MPI_OFFSET_KIND)
       call MPI_File_write_at(fh, disp, zco, nz, MPI_DOUBLE_PRECISION, st, ierr)
       wok = wok .and. pio_xfer_ok(ierr, st, MPI_DOUBLE_PRECISION, nz)
       deallocate(xco, zco)
    end if

    ! each field: D2H (real tp) -> downcast to the output precision -> contiguous write
    allocate(ht(ny,nx,anz))
    if (snap_bytes == 4) then; allocate(hs(ny,nx,anz)); else; allocate(hd(ny,nx,anz)); end if
    do f = 0, nfields-1
       if      (f == 0) then; ht = u
       else if (f == 1) then; ht = v
       else if (f == 2) then; ht = w
       else;                  ht = p; end if
       istat = cudaDeviceSynchronize()
       disp = hdrb + cshift + int(f,MPI_OFFSET_KIND)*fld_elems*eb &
              + int(ny,MPI_OFFSET_KIND)*int(nx,MPI_OFFSET_KIND)*int(z0,MPI_OFFSET_KIND)*eb
       if (snap_bytes == 4) then
          hs = real(ht, sp)
          call MPI_File_write_at_all(fh, disp, hs, cnt, mpitype, st, ierr)
       else
          hd = real(ht, dp)
          call MPI_File_write_at_all(fh, disp, hd, cnt, mpitype, st, ierr)
       end if
       wok = wok .and. pio_xfer_ok(ierr, st, mpitype, cnt)
    end do
    deallocate(ht)
    if (allocated(hs)) deallocate(hs)
    if (allocated(hd)) deallocate(hd)

    call MPI_File_close(fh, ierr)
    wok = wok .and. (ierr == MPI_SUCCESS)
    call MPI_Barrier(MPI_COMM_WORLD, ierr);  call system_clock(t1)
    ! rename only a fully verified .tmp (same contract as par_chk_write)
    wok = pio_all(wok)
    if (par_rank == 0) then
       if (wok) then
          call execute_command_line('mv -f '//trim(ftmp)//' '//trim(fname), wait=.true., exitstat=istat)
          if (istat == 0) call write_pointer_named(fname, 'latest_snapshot.txt')
          ok = (istat == 0)
          if (ok) then
             secs = real(t1-t0,dp)/real(max(trate,1_8),dp)
             gb   = real(total, dp)/1.0e9_dp
             write(*,'(a,a,a,f8.3,a,f7.2,a,f6.1,a)') '   [io] wrote ', trim(fname), ' : ', gb, ' GB in ', &
                  secs, ' s (', gb/max(secs,1.0e-9_dp), ' GB/s)'
          else
             write(*,'(a,a,a)') ' !! par_snap_write: rename .tmp -> ', trim(fname), ' FAILED -- snapshot NOT committed'
          end if
       else
          ok = .false.
          write(*,'(a,a,a)') ' !! par_snap_write: MPI-IO write/count FAILURE on ', trim(ftmp), &
             ' (disk full / quota / FS error?) -- snapshot NOT committed, .tmp left for inspection'
       end if
    end if
    call MPI_Bcast(ok, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, ierr)
  end subroutine par_snap_write

  !-- physical-snapshot header (mirrors build_header; layout=slab-z, space=physical) --
  subroutine build_snap_header(hdr, n, nx, nz, Re, t, step, Lx, Lz, snap_bytes, nfields)
    character(len=IO_HDR_BYTES), intent(out) :: hdr
    integer(ip), intent(in) :: n, nx, nz, step, snap_bytes, nfields
    real(dp),    intent(in) :: Re, t, Lx, Lz
    character(len=IO_HDR_BYTES) :: b
    character(len=64) :: shp
    b = ''
    call addln(b, 'magic = '//IO_MAGIC)
    call addln(b, 'version = '//itoa(IO_VERSION))
    call addln(b, 'endian = '//trim(endian_tag()))
    call addln(b, 'real_bytes = '//itoa(snap_bytes))        ! OUTPUT precision (4 fp32 / 8 fp64)
    call addln(b, 'int_bytes = 4')
    call addln(b, 'coords = x,y,z')                         ! float64 records after the header
    call addln(b, 'coord_bytes = '//itoa(IO_COORD_BYTES))
    call addln(b, 'layout = slab-z')                        ! z outermost on disk; z-slab tiling
    call addln(b, 'xtype = r2c')
    call addln(b, 'space = physical')                       ! clean nx*ny*nz grid, real velocities
    call addln(b, 'hdr_bytes = '//itoa(IO_HDR_BYTES))
    call addln(b, 'flow = channel')
    call addln(b, 'ygrid = '//trim(pio_ygrid))  ! glc, or 'file' (FD); nodes are in the y record
    call addln(b, 'ybc = noslip-noslip')
    call addln(b, 'nx = '//itoa(nx))
    call addln(b, 'ny = '//itoa(n+1))
    call addln(b, 'nz = '//itoa(nz))
    call addln(b, 'Lx = '//ftoa(Lx))
    call addln(b, 'Lz = '//ftoa(Lz))
    call addln(b, 'Re = '//ftoa(Re))
    call addln(b, 't = '//ftoa(t))
    call addln(b, 'step = '//itoa(step))
    call addln(b, 'nfields = '//itoa(nfields))
    if (nfields >= 4) then
       call addln(b, 'fields = u,v,w,p')
    else
       call addln(b, 'fields = u,v,w')
    end if
    write(shp,'(i0,a,i0,a,i0)') n+1, ',', nx, ',', nz       ! on-disk order (ny,nx,nz), z slowest
    call addln(b, 'u.shape = '//trim(shp));  call addln(b, 'u.kind = real')
    call addln(b, 'v.shape = '//trim(shp));  call addln(b, 'v.kind = real')
    call addln(b, 'w.shape = '//trim(shp));  call addln(b, 'w.kind = real')
    if (nfields >= 4) then
       call addln(b, 'p.shape = '//trim(shp));  call addln(b, 'p.kind = real')
    end if
    call addln(b, 'end_header')
    hdr = b
  end subroutine build_snap_header

  !-- rank-0 pointer file <dir>/<ptrbase> holding the newest filename.
  !-- ATOMIC (.tmp + rename): no restart code reads these
  !-- convenience pointers (the live pointer is the driver's <chk_file>.latest),
  !-- but an external tool that does must never see a torn write. On any step
  !-- failing, the old pointer survives intact (same rank-uniform discipline).
  subroutine write_pointer_named(fname, ptrbase)
    character(*), intent(in) :: fname, ptrbase
    character(len=:), allocatable :: dir, ptr
    integer :: u, ios
    integer :: islash
    islash = index(trim(fname), '/', back=.true.)
    if (islash > 0) then; dir = fname(1:islash); else; dir = ''; end if
    ptr = dir//trim(ptrbase)
    open(newunit=u, file=ptr//'.tmp', status='replace', action='write', iostat=ios)
    if (ios /= 0) return
    write(u,'(a)', iostat=ios) trim(fname)
    close(u, iostat=ios)
    if (ios /= 0) return
    call execute_command_line('mv -f '//ptr//'.tmp '//ptr, exitstat=ios)
  end subroutine write_pointer_named

  !================================ helpers ======================================

  !-- forward rotation: in (ny,bnkx,nz) -> out (ny,nz,bnkx) [kx -> outermost] -----
  subroutine rotate_fwd(ny, bnkx, nz, a, b)
    integer, intent(in) :: ny, bnkx, nz
    complex(wp), intent(in)  :: a(ny,bnkx,nz)
    complex(wp), intent(out) :: b(ny,nz,bnkx)
    integer :: ik, iz, iy
    do ik = 1, bnkx
       do iz = 1, nz
          do iy = 1, ny
             b(iy,iz,ik) = a(iy,ik,iz)
          end do
       end do
    end do
  end subroutine rotate_fwd

  !-- inverse rotation: in (ny,nz,bnkx) -> out (ny,bnkx,nz) ------------------------
  subroutine rotate_inv(ny, bnkx, nz, b, a)
    integer, intent(in) :: ny, bnkx, nz
    complex(wp), intent(in)  :: b(ny,nz,bnkx)
    complex(wp), intent(out) :: a(ny,bnkx,nz)
    integer :: ik, iz, iy
    do ik = 1, bnkx
       do iz = 1, nz
          do iy = 1, ny
             a(iy,ik,iz) = b(iy,iz,ik)
          end do
       end do
    end do
  end subroutine rotate_inv

  !-- assemble the fixed 1024-byte ASCII header (mirrors io_mod v2; slab-kx) -------
  subroutine build_header(hdr, n, nx, nz, Re, Qtar, t, dt, step, Lx, Lz, rb)
    character(len=IO_HDR_BYTES), intent(out) :: hdr
    integer(ip), intent(in) :: n, nx, nz, step
    real(dp),    intent(in) :: Re, Qtar, t, dt, Lx, Lz
    integer,     intent(in) :: rb              ! payload real_bytes (4 = fp32 checkpoint)
    character(len=IO_HDR_BYTES) :: b
    character(len=64) :: shp
    integer(ip) :: nkx
    nkx = nx/2 + 1
    b = ''
    call addln(b, 'magic = '//IO_MAGIC)
    call addln(b, 'version = '//itoa(IO_VERSION))
    call addln(b, 'endian = '//trim(endian_tag()))
    call addln(b, 'real_bytes = '//itoa(rb))
    call addln(b, 'int_bytes = 4')
    call addln(b, 'coords = y')                       ! float64 y(ny) record after the header
    call addln(b, 'coord_bytes = '//itoa(IO_COORD_BYTES))
    call addln(b, 'layout = slab-kx')                 ! [par] global-kx-ordered, kx outermost on disk
    call addln(b, 'xtype = r2c')
    call addln(b, 'space = spectral')
    call addln(b, 'hdr_bytes = '//itoa(IO_HDR_BYTES))
    call addln(b, 'flow = channel')
    call addln(b, 'ybc = noslip-noslip')
    ! y-discretisation identity (see io_mod.F90): cheb-T/glc =
    ! Chebyshev-T COEFFICIENTS on the GLC grid; fd-point/file = point VALUES on
    ! the ygrid-file nodes (stored in the v3 y record). Readers refuse a mismatch.
    call addln(b, 'ybasis = '//trim(pio_ybasis))
    call addln(b, 'ygrid = '//trim(pio_ygrid))
    call addln(b, 'nx = '//itoa(nx))
    call addln(b, 'ny = '//itoa(n+1))
    call addln(b, 'nz = '//itoa(nz))
    call addln(b, 'nkx = '//itoa(nkx))
    call addln(b, 'Lx = '//ftoa(Lx))
    call addln(b, 'Lz = '//ftoa(Lz))
    call addln(b, 'Re = '//ftoa(Re))
    call addln(b, 'Qtar = '//ftoa(Qtar))
    call addln(b, 't = '//ftoa(t))
    call addln(b, 'dt = '//ftoa(dt))
    call addln(b, 'step = '//itoa(step))
    call addln(b, 'drive = CFR')
    call addln(b, 'nfields = 4')
    call addln(b, 'fields = v,eta,u0,w0')
    write(shp,'(i0,a,i0,a,i0)') n+1, ',', nz, ',', nkx     ! ON-DISK order (ny,nz,nkx)
    call addln(b, 'v.shape = '//trim(shp));    call addln(b, 'v.kind = complex')
    call addln(b, 'eta.shape = '//trim(shp));  call addln(b, 'eta.kind = complex')
    call addln(b, 'u0.shape = '//itoa(n+1));   call addln(b, 'u0.kind = complex')
    call addln(b, 'w0.shape = '//itoa(n+1));   call addln(b, 'w0.kind = complex')
    call addln(b, 'end_header')
    hdr = b
  end subroutine build_header

  !-- rank-0 pointer file: <dir>/latest_checkpoint.txt holds the newest filename
  !-- (convenience only -- restart reads <chk_file>.latest; atomic via
  !-- the shared named writer)
  subroutine write_pointer(fname)
    character(*), intent(in) :: fname
    call write_pointer_named(fname, 'latest_checkpoint.txt')
  end subroutine write_pointer

  subroutine addln(s, line)
    character(len=IO_HDR_BYTES), intent(inout) :: s
    character(*), intent(in) :: line
    integer :: L
    L = len_trim(s)
    if (L == 0) then; s = line//new_line('a')
    else;             s = s(1:L)//new_line('a')//line//new_line('a'); end if
  end subroutine addln

  subroutine hdr_get(hdr, key, val, okp)
    character(len=IO_HDR_BYTES), intent(in)  :: hdr
    character(*),                intent(in)  :: key
    character(*),                intent(out) :: val
    logical,                     intent(out) :: okp
    integer :: p, q, eq, nl, ks, ke
    okp = .false.;  val = '';  p = 1
    do
       if (p > IO_HDR_BYTES) exit
       nl = index(hdr(p:), new_line('a'))
       if (nl == 0) then; q = IO_HDR_BYTES; else; q = p + nl - 2; end if
       if (q >= p) then
          eq = index(hdr(p:q), '=')
          if (eq > 0) then
             ks = p;  ke = p + eq - 2
             do while (ks <= ke .and. hdr(ks:ks) == ' '); ks = ks + 1; end do
             do while (ke >= ks .and. hdr(ke:ke) == ' '); ke = ke - 1; end do
             if (ke >= ks) then
                if (hdr(ks:ke) == key) then
                   val = adjustl(hdr(p+eq:q));  okp = .true.;  return
                end if
             end if
          end if
       end if
       if (nl == 0) exit
       p = p + nl
    end do
  end subroutine hdr_get

  pure function endian_tag() result(tag)
    character(len=8) :: tag
    integer(ip) :: i
    integer(1)  :: b(4)
    i = 1_ip;  b = transfer(i, b)
    if (b(1) == 1_1) then; tag = 'little'; else; tag = 'big'; end if
  end function endian_tag

  pure function itoa(i) result(s)
    integer(ip), intent(in) :: i
    character(len=:), allocatable :: s
    character(len=32) :: t
    write(t,'(i0)') i;  s = trim(t)
  end function itoa

  pure function ftoa(x) result(s)
    real(dp), intent(in) :: x
    character(len=:), allocatable :: s
    character(len=32) :: t
    write(t,'(es24.16e3)') x;  s = trim(adjustl(t))
  end function ftoa

end module par_io_mod
