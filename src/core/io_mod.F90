!===============================================================================
! io_mod.F90  --  checkpoint / restart of the solver state.  FORMAT v2.
!
! Writes and reads the SPECTRAL state needed to restart a run:
!     v, eta   (0:n, nkx, nz)  complex(wp)   -- wall-normal velocity & vorticity
!                                              [R2C: nkx = nx/2+1, non-negative kx half]
!     u0, w0   (0:n)           complex(wp)   -- mean profiles (the (0,0) mode)
! plus scalars t, dt, Re, Q_target, step.
!
! WHY THAT IS THE WHOLE STATE: because zeta_1 = 0 makes every SMR *step*
! self-starting, the N^{k-2} history registers are re-derived inside each step
! and need NOT be saved. A complete restart is just (v, eta, u0, w0) + scalars.
! (test_io's restart-continuation check verifies this: restart with fresh
! registers reproduces an uninterrupted run bit-for-bit.)
!
!-------------------------------------------------------------------------------
! FORMAT v2 -- ASCII self-describing header + binary payload.
!
! The file begins with a FIXED-SIZE ASCII header of exactly IO_HDR_BYTES bytes
! (a multiple of 8 so the binary payload that follows is 8-byte aligned). The
! header is plain text, newline-separated `KEY = VALUE` lines, padded with
! trailing spaces to IO_HDR_BYTES. It is human-readable:
!
!     head -c 1024 chk.bin           # dumps the whole header
!
! so any reader -- Fortran, Python, `head` -- can self-describe the file without
! prior knowledge. After the header, at byte offset IO_HDR_BYTES, comes the
! binary payload: the fields named in the `fields=` manifest, in that order,
! each a raw little-/big-endian dump in the precision given by `real_bytes`.
!
! HEADER KEYS (v2):
!   magic        = ClOpDNS               format sentinel
!   version      = 2
!   endian       = little | big          byte order of the binary payload
!   real_bytes   = 8 | 4                 bytes per real of the stored fields (wp)
!   int_bytes    = 4                     (informational; header ints are ASCII)
!   layout       = serial-yxz            payload layout tag (see LAYOUT below)
!   xtype        = r2c | c2c             x-transform type; R2C payload is nkx-wide,
!                                        C2C is nx-wide. Reader REFUSES a mismatch
!                                        (grids match but column counts differ).
!   hdr_bytes    = 1024                  total header size (== IO_HDR_BYTES)
!   flow         = channel               flow / geometry; implies the y BCs
!   ybc          = noslip-noslip         wall-normal boundary conditions
!   space        = spectral              spectral (Cheb-y, Fourier-xz) | physical
!   nx, ny, nz   = ...                   grid (ny = n+1 Chebyshev coeffs)
!   Lx, Lz       = ...                   box dimensions (Ly = 2 implied for channel)
!   Re, Qtar, t, dt = ...                physical scalars (always f.p., dp on read)
!   step         = ...                   step counter
!   drive        = CFR                   mass-flux drive mode (CFR | CPG)
!   nfields      = 4                     number of payload records
!   fields       = v,eta,u0,w0           ORDERED manifest of payload records
!   <name>.shape = ny,nx,nz | ny         per-field dims (one line each, in order)
!   <name>.kind  = complex | real        per-field datatype (one line each)
!
! WHY A MANIFEST: adding a field later (e.g. a temperature theta, or physical
! u,v,w dumps for a single shared Python reader) is a manifest entry + a payload
! record -- NO format change, NO version bump. A reader keys off `fields=` and
! the per-field `.shape`/`.kind` lines, so unknown extra fields are skippable by
! byte size and the velocities can be pulled by name. `flow`/`ybc` let one
! reader handle channel, BL, Couette, ... and know the wall conditions without
! guessing. This is the extensibility the v1 binary header could not give.
!
! LAYOUT (forward-looking for parallel I/O): `layout=serial-yxz` means a single
! contiguous y-leading dump written by one rank, exactly as v1 did. A future
! parallel writer bumps `layout` (e.g. `slab-z`, `pencil-zx`) and writes the
! payload its way; the ASCII header, the flow/field/precision/endian semantics,
! and the refuse-on-mismatch contract are UNCHANGED. Thus v2's header is already
! parallel-ready though this writer is serial.
!
! READER CONTRACT: parse the ASCII header; REFUSE (ok=.false., no read) on any
! mismatch -- wrong magic, version > this reader, real_bytes /= this build's wp,
! grid /= the run, an unsupported layout, or a fields manifest this reader does
! not know how to restart from. On endian mismatch v2 still refuses (byte-swap
! is a deliberate later option; the key is RECORDED so a converter can be added
! without another format change). No silent misreads, ever.
!
! PRECISION (see src/core/types.F90): fields are written in the
! COMPUTE precision wp; `real_bytes` records it (8 real64 / 4 a future real32).
! A file whose real_bytes /= this build's wp is REFUSED. Scalars are dp in the
! header text regardless. io stays precision-agnostic via wp.
!
! BACKWARD COMPATIBILITY: v1 files (7 leading default integers, first == the v1
! IO_MAGIC 1146312497) are detected and REFUSED with a clear message. A v1->v2
! converter is a trivial add if one is ever needed.
!
! NOT IMPLEMENTED: physical u,v,w dumps (a manifest+payload add,
! no format change -- the v2 design anticipates them); a Neumann pressure solve
! for snapshot pressure; HDF5/ADIOS2 backends (the v2 header maps 1:1 onto HDF5
! attributes + datasets, so it is a clean future swap);
! parallel collective write; endian byte-swap-on-read.
!
! Build:  nvfortran -cuda -c io_mod.F90
!===============================================================================
module io_mod
  use kinds, only: tp, wp, dp, ip
  use cudafor
  implicit none
  private
  public :: io_write_state, io_read_state
  public :: io_set_ybasis                                  ! y-discretisation stamp

  integer(ip), parameter :: IO_VERSION   = 2
  integer(ip), parameter :: IO_HDR_BYTES = 1024          ! fixed header size (mult. of 8)
  character(*), parameter :: IO_MAGIC     = 'ClOpDNS'     ! ASCII sentinel
  integer(ip), parameter :: IO_MAGIC_V1   = 1146312497    ! v1's leading integer (detect+refuse)

  ! y-discretisation stamp (defaults = Chebyshev). The FD wiring flips it
  ! once at init. v2 files
  ! carry no y record, so FD grid identity rides a scalar 'ygrid_moment' key
  ! (= sum_j j*y_j, 1-indexed -- a weak hash, but any real grid-law change
  ! shifts it by >>tol); the reader recomputes it from the RUN's nodes and
  ! refuses a mismatch.
  character(len=16)     :: iom_ybasis = 'cheb-T'
  character(len=16)     :: iom_ygrid  = 'glc'
  real(dp)              :: iom_ymom   = 0.0_dp
  logical               :: iom_has_mom = .false.

contains

  subroutine io_set_ybasis(basis, grid, ynodes)
    character(*), intent(in) :: basis, grid
    real(dp), intent(in), optional :: ynodes(:)
    integer(ip) :: j
    iom_ybasis = basis;  iom_ygrid = grid
    iom_has_mom = present(ynodes)
    if (iom_has_mom) then
       iom_ymom = 0.0_dp
       do j = 1, size(ynodes)
          iom_ymom = iom_ymom + real(j, dp)*ynodes(j)
       end do
    end if
  end subroutine io_set_ybasis

  !-- write a checkpoint; ok=.false. on any I/O error -----------------------------
  subroutine io_write_state(fname, n, nx, nz, Re, t, dt, Qtar, step, v, eta, u0, w0, ok, &
                            Lx, Lz, flow, ybc, drive)
    character(*),           intent(in)  :: fname
    integer(ip),            intent(in)  :: n, nx, nz, step
    real(dp),               intent(in)  :: Re, t, dt, Qtar
    ! [R2C] state arrays are (0:n, nkx, nz), nkx = nx/2+1 (non-negative kx half).
    ! nx is the PHYSICAL x size (recorded in the header for grid identity); the
    ! payload + the v.shape/eta.shape manifest use nkx (self-documenting).
    complex(wp),  device                :: v(0:n,nx/2+1,nz), eta(0:n,nx/2+1,nz), u0(0:n), w0(0:n)
    logical,                intent(out) :: ok
    real(dp),     optional, intent(in)  :: Lx, Lz                  ! box (default 2pi, 2pi)
    character(*), optional, intent(in)  :: flow, ybc, drive        ! metadata (defaults below)
    complex(wp), allocatable :: vh(:,:,:), eh(:,:,:), u0h(:), w0h(:)
    character(len=IO_HDR_BYTES) :: hdr
    character(len=32) :: cendian, cflow, cybc, cdrive
    real(dp) :: rLx, rLz
    integer(ip) :: u, ios, rb, tp_rb, dp_rb, nkx

    ok = .false.
    nkx = nx/2_ip + 1_ip                               ! [R2C] spectral x-extent
    rb = storage_size(1.0_wp)/8                        ! bytes per wp real (field/payload)
    tp_rb = storage_size(1.0_tp)/8                     ! bytes per tp real (transform tier)
    dp_rb = storage_size(1.0_dp)/8                     ! bytes per dp real (solve/accum tier)
    rLx = 6.283185307179586_dp; if (present(Lx)) rLx = Lx
    rLz = 6.283185307179586_dp; if (present(Lz)) rLz = Lz
    cflow  = 'channel';          if (present(flow))  cflow  = flow
    cybc   = 'noslip-noslip';    if (present(ybc))   cybc   = ybc
    cdrive = 'CFR';              if (present(drive)) cdrive = drive
    cendian = trim(endian_tag())

    call build_header(hdr, rb, tp_rb, dp_rb, cendian, cflow, cybc, cdrive, &
                      n, nx, nz, rLx, rLz, Re, Qtar, t, dt, step)

    allocate(vh(0:n,nkx,nz), eh(0:n,nkx,nz), u0h(0:n), w0h(0:n))
    vh = v;  eh = eta;  u0h = u0;  w0h = w0             ! device -> host
    ! Stage to .tmp + rename: a status='replace' open would TRUNCATE the
    ! previous checkpoint before its replacement was safely on disk, so a
    ! crash mid-write would leave ZERO restart points. Only a fully written
    ! .tmp is renamed onto fname.
    open(newunit=u, file=trim(fname)//'.tmp', access='stream', form='unformatted', &
         status='replace', action='write', iostat=ios)
    if (ios /= 0) then; deallocate(vh, eh, u0h, w0h); return; end if
    write(u, iostat=ios) hdr                            ! exactly IO_HDR_BYTES ASCII bytes
    if (ios == 0) write(u, iostat=ios) vh               ! payload at offset IO_HDR_BYTES
    if (ios == 0) write(u, iostat=ios) eh
    if (ios == 0) write(u, iostat=ios) u0h
    if (ios == 0) write(u, iostat=ios) w0h
    if (ios == 0) then
       close(u, iostat=ios)                             ! close flushes -- its failure is a failure
    else
       close(u)
    end if
    deallocate(vh, eh, u0h, w0h)
    if (ios == 0) then
       call execute_command_line('mv -f '//trim(fname)//'.tmp '//trim(fname), &
                                 wait=.true., exitstat=ios)
    end if
    ok = (ios == 0)
  end subroutine io_write_state

  !-- read a checkpoint into preallocated arrays; validate header; ok on success --
  ! n,nx,nz are the RUN's grid (the file must match). Re,t,dt,Qtar,step come back
  ! from the file. Optional metadata (flow,ybc,drive,Lx,Lz) is returned if asked.
  subroutine io_read_state(fname, n, nx, nz, Re, t, dt, Qtar, step, v, eta, u0, w0, ok, &
                           Lx, Lz, flow, ybc, drive)
    character(*),           intent(in)  :: fname
    integer(ip),            intent(in)  :: n, nx, nz
    real(dp),               intent(out) :: Re, t, dt, Qtar
    integer(ip),            intent(out) :: step
    ! [R2C] state arrays are (0:n, nkx, nz), nkx = nx/2+1; nx is the PHYSICAL size.
    complex(wp),  device                :: v(0:n,nx/2+1,nz), eta(0:n,nx/2+1,nz), u0(0:n), w0(0:n)
    logical,                intent(out) :: ok
    real(dp),     optional, intent(out) :: Lx, Lz
    character(*), optional, intent(out) :: flow, ybc, drive
    complex(wp), allocatable :: vh(:,:,:), eh(:,:,:), u0h(:), w0h(:)
    character(len=IO_HDR_BYTES) :: hdr
    character(len=64) :: val, sflow, sybc, sdrive, slayout, sendian, sspace, sfields, sxtype
    real(dp) :: rLx, rLz, rmom
    integer(ip) :: u, ios, ver, rb, fn, fnx, fnz, fstep, i4magic, nkx
    logical :: okp

    ok = .false.;  Re = 0; t = 0; dt = 0; Qtar = 0; step = 0
    nkx = nx/2_ip + 1_ip                               ! [R2C] spectral x-extent
    open(newunit=u, file=fname, access='stream', form='unformatted', &
         status='old', action='read', iostat=ios)
    if (ios /= 0) return

    ! Peek the first 4 bytes as a v1 integer magic; if it matches v1, refuse.
    read(u, pos=1, iostat=ios) i4magic
    if (ios == 0 .and. i4magic == IO_MAGIC_V1) then
       close(u); return                                 ! v1 file: refuse (none in the wild)
    end if

    ! Read the fixed ASCII header.
    read(u, pos=1, iostat=ios) hdr
    if (ios /= 0) then; close(u); return; end if

    ! Validate sentinel + version.
    call hdr_get(hdr, 'magic', val, okp)
    if (.not. okp .or. trim(val) /= IO_MAGIC) then; close(u); return; end if
    call hdr_get(hdr, 'version', val, okp)
    if (.not. okp) then; close(u); return; end if
    read(val,*,iostat=ios) ver
    if (ios /= 0 .or. ver > IO_VERSION) then; close(u); return; end if   ! newer than we understand

    ! Precision, endian, layout gates.
    call hdr_get(hdr, 'real_bytes', val, okp);  if (.not. okp) then; close(u); return; end if
    read(val,*,iostat=ios) rb
    if (ios /= 0 .or. rb /= storage_size(1.0_wp)/8) then; close(u); return; end if  ! precision mismatch
    call hdr_get(hdr, 'endian', sendian, okp);  if (.not. okp) then; close(u); return; end if
    if (trim(sendian) /= trim(endian_tag())) then; close(u); return; end if         ! endian mismatch
    call hdr_get(hdr, 'layout', slayout, okp);  if (.not. okp) then; close(u); return; end if
    if (trim(slayout) /= 'serial-yxz') then; close(u); return; end if               ! only layout we restart
    ! [R2C] x-transform-type gate: an R2C build must REFUSE a C2C file (and vice
    ! versa) -- the grids match (same physical nx) but the payload column count
    ! differs (nkx vs nx), so a silent cross-read would corrupt. v2 files written
    ! before this gate have no xtype key; treat absent as 'c2c' (the pre-R2C default)
    ! so this R2C reader refuses them, which is correct (their payload is nx-wide).
    call hdr_get(hdr, 'xtype', sxtype, okp)
    if (.not. okp) sxtype = 'c2c'
    if (trim(sxtype) /= 'r2c') then; close(u); return; end if                       ! not an R2C file

    ! Grid gate.
    call hdr_get(hdr, 'nx', val, okp); if (.not. okp) then; close(u); return; end if; read(val,*,iostat=ios) fnx
    if (ios /= 0) then; close(u); return; end if
    call hdr_get(hdr, 'ny', val, okp); if (.not. okp) then; close(u); return; end if; read(val,*,iostat=ios) fn
    if (ios /= 0) then; close(u); return; end if
    call hdr_get(hdr, 'nz', val, okp); if (.not. okp) then; close(u); return; end if; read(val,*,iostat=ios) fnz
    if (ios /= 0) then; close(u); return; end if
    if (fnx /= nx .or. fn /= n+1 .or. fnz /= nz) then; close(u); return; end if      ! grid mismatch (ny = n+1)

    ! Restartability gate: this reader knows the 4-record spectral state.
    call hdr_get(hdr, 'fields', sfields, okp)
    if (.not. okp .or. trim(sfields) /= 'v,eta,u0,w0') then; close(u); return; end if
    call hdr_get(hdr, 'space', sspace, okp)
    if (okp) then; if (trim(sspace) /= 'spectral') then; close(u); return; end if; end if
    ! y-basis gate, against the RUN's stamp (cheb-T/glc by
    ! default; the FD wiring sets fd-point/file). Cross-reads in EITHER direction
    ! refuse (same shape, garbage physics). Absent keys = legacy file = cheb-T/glc
    ! -> acceptable ONLY when the run itself is Chebyshev.
    call hdr_get(hdr, 'ybasis', val, okp)
    if (okp) then; if (trim(val) /= trim(iom_ybasis)) then; close(u); return; end if
    else;          if (trim(iom_ybasis) /= 'cheb-T')  then; close(u); return; end if; end if
    call hdr_get(hdr, 'ygrid', val, okp)
    if (okp) then; if (trim(val) /= trim(iom_ygrid)) then; close(u); return; end if
    else;          if (trim(iom_ygrid) /= 'glc')     then; close(u); return; end if; end if
    ! FD grid identity: recompute the moment from the RUN's nodes and
    ! compare to the file's. Key absent in an fd-point file (foreign writer) or
    ! moment mismatch (different grid file) -> refuse. Tolerance is relative to
    ! the moment's magnitude (~ny^2/6; a real grid-law change shifts it >>tol).
    if (trim(iom_ybasis) == 'fd-point') then
       call hdr_get(hdr, 'ygrid_moment', val, okp)
       if (.not. (okp .and. iom_has_mom)) then; close(u); return; end if
       read(val,*,iostat=ios) rmom
       if (ios /= 0) then; close(u); return; end if
       if (abs(rmom - iom_ymom) > 1.0e-8_dp*max(1.0_dp, abs(iom_ymom))) then
          close(u); return
       end if
    end if

    ! Scalars from the header text.
    call hdr_get(hdr, 'Re',   val, okp); if (okp) read(val,*,iostat=ios) Re
    call hdr_get(hdr, 'Qtar', val, okp); if (okp) read(val,*,iostat=ios) Qtar
    call hdr_get(hdr, 't',    val, okp); if (okp) read(val,*,iostat=ios) t
    call hdr_get(hdr, 'dt',   val, okp); if (okp) read(val,*,iostat=ios) dt
    call hdr_get(hdr, 'step', val, okp); if (okp) then; read(val,*,iostat=ios) fstep; step = fstep; end if
    rLx = 0; rLz = 0
    call hdr_get(hdr, 'Lx', val, okp); if (okp) read(val,*,iostat=ios) rLx
    call hdr_get(hdr, 'Lz', val, okp); if (okp) read(val,*,iostat=ios) rLz
    if (present(Lx)) Lx = rLx
    if (present(Lz)) Lz = rLz
    call hdr_get(hdr, 'flow',  sflow,  okp); if (present(flow))  flow  = trim(sflow)
    call hdr_get(hdr, 'ybc',   sybc,   okp); if (present(ybc))   ybc   = trim(sybc)
    call hdr_get(hdr, 'drive', sdrive, okp); if (present(drive)) drive = trim(sdrive)

    ! Payload at the fixed offset. [R2C] (0:n, nkx, nz).
    allocate(vh(0:n,nkx,nz), eh(0:n,nkx,nz), u0h(0:n), w0h(0:n))
    read(u, pos=IO_HDR_BYTES+1, iostat=ios) vh
    if (ios == 0) read(u, iostat=ios) eh
    if (ios == 0) read(u, iostat=ios) u0h
    if (ios == 0) read(u, iostat=ios) w0h
    close(u)
    if (ios == 0) then
       v = vh;  eta = eh;  u0 = u0h;  w0 = w0h           ! host -> device
       ok = .true.
    end if
    deallocate(vh, eh, u0h, w0h)
  end subroutine io_read_state

  !================================ helpers ======================================

  !-- assemble the fixed-size ASCII header (space-padded to IO_HDR_BYTES) ----------
  subroutine build_header(hdr, rb, tp_rb, dp_rb, cendian, cflow, cybc, cdrive, &
                          n, nx, nz, Lx, Lz, Re, Qtar, t, dt, step)
    character(len=IO_HDR_BYTES), intent(out) :: hdr
    integer(ip),  intent(in) :: rb, tp_rb, dp_rb, n, nx, nz, step
    character(*), intent(in) :: cendian, cflow, cybc, cdrive
    real(dp),     intent(in) :: Lx, Lz, Re, Qtar, t, dt
    character(len=IO_HDR_BYTES) :: b
    character(len=64) :: lvol, lcol
    integer(ip) :: nkx
    nkx = nx/2_ip + 1_ip                            ! [R2C] spectral x-extent
    b = ''
    call addln(b, 'magic = '//IO_MAGIC)
    call addln(b, 'version = 2')
    call addln(b, 'endian = '//trim(cendian))
    call addln(b, 'real_bytes = '//itoa(rb))         ! = wp (field) bytes -- the reader's precision gate
    call addln(b, 'tp_bytes = '//itoa(tp_rb))        ! [precision] transform tier (provenance; tp=sp => 4)
    call addln(b, 'dp_bytes = '//itoa(dp_rb))        ! [precision] solve/accumulation tier (provenance)
    call addln(b, 'int_bytes = 4')
    call addln(b, 'layout = serial-yxz')
    call addln(b, 'xtype = r2c')                     ! [R2C] real-x transform (payload nkx-wide)
    call addln(b, 'hdr_bytes = '//itoa(IO_HDR_BYTES))
    call addln(b, 'flow = '//trim(cflow))
    call addln(b, 'ybc = '//trim(cybc))
    call addln(b, 'space = spectral')
    ! y-discretisation identity: cheb-T/glc = CHEBYSHEV-T
    ! COEFFICIENT columns on the Gauss-Lobatto grid; fd-point/file = point
    ! VALUES on the ygrid-file nodes. Readers refuse a mismatch instead of
    ! silently treating T_k coefficients as grid-point values (same shape,
    ! garbage physics). FD adds 'ygrid_moment' (grid identity; v2 has no y record).
    call addln(b, 'ybasis = '//trim(iom_ybasis))
    call addln(b, 'ygrid = '//trim(iom_ygrid))
    if (iom_has_mom) call addln(b, 'ygrid_moment = '//ftoa(iom_ymom))
    call addln(b, 'nx = '//itoa(nx))                 ! PHYSICAL x size (grid identity)
    call addln(b, 'ny = '//itoa(n+1))               ! Chebyshev coefficient count
    call addln(b, 'nz = '//itoa(nz))
    call addln(b, 'nkx = '//itoa(nkx))               ! [R2C] spectral x-extent (= nx/2+1)
    call addln(b, 'Lx = '//ftoa(Lx))
    call addln(b, 'Lz = '//ftoa(Lz))
    call addln(b, 'Re = '//ftoa(Re))
    call addln(b, 'Qtar = '//ftoa(Qtar))
    call addln(b, 't = '//ftoa(t))
    call addln(b, 'dt = '//ftoa(dt))
    call addln(b, 'step = '//itoa(step))
    call addln(b, 'drive = '//trim(cdrive))
    call addln(b, 'nfields = 4')
    call addln(b, 'fields = v,eta,u0,w0')
    write(lvol,'(i0,a,i0,a,i0)') n+1, ',', nkx, ',', nz    ! [R2C] ny,nkx,nz
    write(lcol,'(i0)') n+1
    call addln(b, 'v.shape = '//trim(lvol));    call addln(b, 'v.kind = complex')
    call addln(b, 'eta.shape = '//trim(lvol));  call addln(b, 'eta.kind = complex')
    call addln(b, 'u0.shape = '//trim(lcol));   call addln(b, 'u0.kind = complex')
    call addln(b, 'w0.shape = '//trim(lcol));   call addln(b, 'w0.kind = complex')
    call addln(b, 'end_header')
    hdr = b                                          ! trailing chars are spaces (b init '')
  end subroutine build_header

  !-- append "line"//newline to s after its current content -----------------------
  subroutine addln(s, line)
    character(len=IO_HDR_BYTES), intent(inout) :: s
    character(*), intent(in) :: line
    integer :: L
    L = len_trim(s)
    if (L == 0) then
       s = line//new_line('a')
    else
       s = s(1:L)//new_line('a')//line//new_line('a')
    end if
  end subroutine addln

  !-- fetch VALUE for KEY from the ASCII header; okp=.false. if absent ------------
  ! Matches a line of the form "key = value" (spaces around '=' optional).
  subroutine hdr_get(hdr, key, val, okp)
    character(len=IO_HDR_BYTES), intent(in)  :: hdr
    character(*),                intent(in)  :: key
    character(*),                intent(out) :: val
    logical,                     intent(out) :: okp
    integer :: p, q, eq, nl, ks, ke
    okp = .false.;  val = ''
    p = 1
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
                   val = adjustl(hdr(p+eq:q))
                   okp = .true.
                   return
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
    i = 1_ip
    b = transfer(i, b)
    if (b(1) == 1_1) then; tag = 'little'; else; tag = 'big'; end if
  end function endian_tag

  pure function itoa(i) result(s)
    integer(ip), intent(in) :: i
    character(len=:), allocatable :: s
    character(len=32) :: t
    write(t,'(i0)') i
    s = trim(t)
  end function itoa

  pure function ftoa(x) result(s)
    real(dp), intent(in) :: x
    character(len=:), allocatable :: s
    character(len=32) :: t
    write(t,'(es24.16e3)') x          ! full-precision, round-trippable
    s = trim(adjustl(t))
  end function ftoa

end module io_mod
