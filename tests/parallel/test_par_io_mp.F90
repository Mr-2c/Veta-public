!===============================================================================
! test_par_io_mp.F90 -- distributed checkpoint I/O round-trip + on-disk layout.
!
! Validates par_io_mod on whatever P the job is launched with:
!   (1) ROUND-TRIP: build a deterministic GLOBAL spectral field, slice this rank's
!       kx-band into the device state, par_chk_write, par_chk_read it back, and
!       require bit-exact recovery of v, eta, the means (on the owner), and the
!       scalar header (Re,t,dt,Qtar,step).
!   (2) ON-DISK ORDER (rank 0): read the whole file as a plain stream and check the
!       v payload is exactly the canonical (ny,nz,nkx) global order -- i.e. what a
!       single-GPU oracle / Python reader would expect. This is the cross-P /
!       oracle-readable guarantee (cross-P itself is proven in
!       src/models/work_io_layout_model.py).
!
! Every element of the global field is a unique value, so any offset/rotation bug
! Build: TP=dp|sp bash scripts/build/build_par.sh test_par_io_mp   Run: bash run_all_par_tests.sh run -k test_par_io_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
! Legs (3)-(7) below add: fp32 round-trip, ybasis stamp refusals, the
! truncated-payload refusal, and the corrupt-scalar ('t = ????') and
! endian-mismatch header refusals (both in-place byte-preserving edits).
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_par_io_mp   Run: bash run_all_par_tests.sh run -k test_par_io_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_par_io_mp
  use kinds,      only: wp, dp, ip
  use cudafor
  use decomp_mod, only: decomp_t, decomp_setup
  use par_mod,    only: par_init, par_finalize, par_rank, par_nproc, par_reduce_and
  use par_io_mod, only: par_chk_write, par_chk_read, par_io_set_ybasis
  implicit none

  integer(ip), parameter :: n = 16, nx = 24, nz = 18          ! ny=17, nkx=13
  integer(ip) :: ny, nkx, bnkx, kx0, iy, ik, iz, ikg
  type(decomp_t) :: d
  complex(wp), device, allocatable :: v(:,:,:), eta(:,:,:), u0(:), w0(:)
  complex(wp), device, allocatable :: v2(:,:,:), eta2(:,:,:), u02(:), w02(:)
  complex(wp), allocatable :: hv(:,:,:), he(:,:,:), h2(:,:,:), hm(:), hm2(:)
  complex(wp), allocatable :: vdisk(:,:,:)
  real(dp), allocatable :: yh(:), ydisk(:)     ! v3 y coordinate record
  real(dp), allocatable :: yh2(:)              ! perturbed grid (mismatch leg)
  real(dp) :: Re, t, dt, Qtar, Re2, t2, dt2, Qtar2
  integer(ip) :: step, step2, u, ios
  real(dp) :: dv, de, dmean
  real(dp) :: dv4, de4, dmean4, vnrm, tol4      ! fp32 round-trip metrics
  real(dp) :: dvf                               ! fd round-trip metric
  logical :: ok, allok
  logical :: okfd, refA, refB, refC             ! stamp legs
  logical :: refT                               ! truncated-payload leg
  integer :: fsz                                ! staged-file size probe
  logical :: refV, refE                         ! corrupt-scalar / endian legs
  character(len=1024) :: hb1                    ! header staging buffer
  character(len=*), parameter :: fchk  = 'build_par/test_par_io_chk.bin'
  character(len=*), parameter :: fchk4 = 'build_par/test_par_io_chk_fp32.bin'
  character(len=*), parameter :: ffd   = 'build_par/test_par_io_chk_fd.bin'
  character(len=*), parameter :: ftr   = 'build_par/test_par_io_chk_trunc.bin'
  character(len=*), parameter :: fcv   = 'build_par/test_par_io_chk_badt.bin'
  character(len=*), parameter :: fen   = 'build_par/test_par_io_chk_bade.bin'

  call par_init()
  ny = n + 1;  nkx = nx/2 + 1
  call decomp_setup(d, 0_ip, int(par_nproc,ip), int(par_rank,ip), nx, ny, nz)
  bnkx = d%b_nkx_loc;  kx0 = d%b_kx0

  Re = 2800.0_dp;  t = 1.234_dp;  dt = 1.0e-3_dp;  Qtar = 1.0_dp;  step = 42

  allocate(v(0:n,bnkx,nz), eta(0:n,bnkx,nz), u0(0:n), w0(0:n))
  allocate(v2(0:n,bnkx,nz), eta2(0:n,bnkx,nz), u02(0:n), w02(0:n))
  allocate(hv(0:n,bnkx,nz), he(0:n,bnkx,nz), h2(0:n,bnkx,nz), hm(0:n), hm2(0:n))

  ! deterministic GLOBAL field: every element a unique value (global kx index ikg)
  do iz = 1, nz
     do ik = 1, bnkx
        ikg = kx0 + (ik-1)
        do iy = 0, n
           hv(iy,ik,iz) = gval(iy, ikg, iz-1)
           he(iy,ik,iz) = gval(iy, ikg, iz-1) * cmplx(2.0_wp, 3.0_wp, wp)
        end do
     end do
  end do
  v = hv;  eta = he
  do iy = 0, n
     hm(iy) = cmplx(real(iy+1,wp), real(-(iy+1),wp), wp)
  end do
  u0 = hm;  w0 = hm * cmplx(0.0_wp, 1.0_wp, wp)
  allocate(yh(0:n))                            ! deterministic "node" values
  do iy = 0, n
     yh(iy) = 1.0_dp - 2.0_dp*real(iy,dp)/real(n,dp)
  end do

  ! (1) write then read back. allow_overwrite: the test deliberately rewrites a fixed
  !     path on every run/P (the no-clobber default is for production's unique names).
  call par_chk_write(fchk, d, n, nx, nz, Re, t, dt, Qtar, step, v, eta, u0, w0, yh, &
                     6.2831853_dp, 3.1415927_dp, ok, allow_overwrite=.true.)
  if (.not. ok) call fail('par_chk_write returned ok=.false.')
  call par_chk_read(fchk, d, n, nx, nz, Re2, t2, dt2, Qtar2, step2, v2, eta2, u02, w02, ok)
  if (.not. ok) call fail('par_chk_read returned ok=.false.')

  h2 = v2;    dv = maxval(abs(h2 - hv))
  h2 = eta2;  de = maxval(abs(h2 - he))
  dmean = 0.0_dp
  if (d%owns_mode00) then
     hm2 = u02;  dmean = max(dmean, maxval(abs(hm2 - hm)))
     hm2 = w02;  dmean = max(dmean, maxval(abs(hm2 - hm*cmplx(0.0_wp,1.0_wp,wp))))
  end if

  allok = (dv == 0.0_dp) .and. (de == 0.0_dp) .and. (dmean == 0.0_dp)
  allok = allok .and. (Re2==Re) .and. (t2==t) .and. (dt2==dt) .and. (Qtar2==Qtar) .and. (step2==step)

  ! (2) rank 0: whole-file stream read -> the y coordinate record sits
  !     right after the 1024 B header (ny float64, bit-exact), then the v payload
  !     in canonical (ny,nz,nkx) order at pos 1025 + ny*8.
  if (par_rank == 0) then
     allocate(vdisk(0:n, nz, nkx), ydisk(0:n))
     open(newunit=u, file=fchk, access='stream', form='unformatted', status='old', &
          action='read', iostat=ios)
     if (ios == 0) then
        read(u, pos=1025, iostat=ios) ydisk          ! v3 coord record (coords = y)
        if (ios /= 0) then
           allok = .false.
        else if (any(ydisk /= yh)) then
           allok = .false.
        end if
        read(u, pos=1025 + (n+1)*8, iostat=ios) vdisk
        close(u)
        do iz = 1, nz
           do ikg = 0, nkx-1
              do iy = 0, n
                 if (vdisk(iy, iz, ikg+1) /= gval(iy, ikg, iz-1)) allok = .false.
              end do
           end do
        end do
     else
        allok = .false.
     end if
     deallocate(vdisk, ydisk)
  end if

  ! (3) fp32 checkpoint round-trip: write demoted (half volume), read
  !     back PROMOTED. Error bound: one sp rounding per component, relative to
  !     the LARGEST payload value (gval magnitudes are O(global index)), with a
  !     generous factor -- still 10+ orders below any wrong-file failure mode.
  call par_chk_write(fchk4, d, n, nx, nz, Re, t, dt, Qtar, step, v, eta, u0, w0, yh, &
                     6.2831853_dp, 3.1415927_dp, ok, allow_overwrite=.true., fp32=.true.)
  if (.not. ok) call fail('par_chk_write(fp32) returned ok=.false.')
  call par_chk_read(fchk4, d, n, nx, nz, Re2, t2, dt2, Qtar2, step2, v2, eta2, u02, w02, ok)
  if (.not. ok) call fail('par_chk_read(fp32 file) returned ok=.false.')
  h2 = v2;    dv4 = maxval(abs(h2 - hv));  vnrm = maxval(abs(hv))
  h2 = eta2;  de4 = maxval(abs(h2 - he));  vnrm = max(vnrm, maxval(abs(he)))
  dmean4 = 0.0_dp
  if (d%owns_mode00) then
     hm2 = u02;  dmean4 = max(dmean4, maxval(abs(hm2 - hm)))
     hm2 = w02;  dmean4 = max(dmean4, maxval(abs(hm2 - hm*cmplx(0.0_wp,1.0_wp,wp))))
  end if
  tol4 = 1.0e-6_dp * vnrm
  allok = allok .and. (dv4 <= tol4) .and. (de4 <= tol4) .and. (dmean4 <= tol4)
  allok = allok .and. (Re2==Re) .and. (t2==t) .and. (step2==step)   ! header scalars stay exact (ASCII)

  ! (4) fd-point ybasis stamp. An fd-point/file checkpoint round-trips
  !     bit-exactly under its own stamp (the y-record identity check passes on
  !     the same nodes), and every cross-read direction REFUSES:
  !       refA: fd run reading the same fd file on a DIFFERENT grid (y perturbed)
  !       refB: cheb run reading the fd file  (stamp mismatch)
  !       refC: fd run reading the cheb file  (legacy/cheb stamp mismatch)
  !     Expected refusal chatter ('header mismatch' / 'y-grid mismatch') on
  !     rank 0 is part of the test.
  call par_io_set_ybasis('fd-point', 'file', yh)
  call par_chk_write(ffd, d, n, nx, nz, Re, t, dt, Qtar, step, v, eta, u0, w0, yh, &
                     6.2831853_dp, 3.1415927_dp, ok, allow_overwrite=.true.)
  if (.not. ok) call fail('par_chk_write(fd-point) returned ok=.false.')
  call par_chk_read(ffd, d, n, nx, nz, Re2, t2, dt2, Qtar2, step2, v2, eta2, u02, w02, okfd)
  dvf = -1.0_dp
  if (okfd) then
     h2 = v2;  dvf = maxval(abs(h2 - hv))
     okfd = (dvf == 0.0_dp) .and. (step2 == step)
  end if
  allocate(yh2(0:n));  yh2 = yh;  yh2(n/2) = yh2(n/2) + 1.0e-6_dp
  call par_io_set_ybasis('fd-point', 'file', yh2)
  call par_chk_read(ffd, d, n, nx, nz, Re2, t2, dt2, Qtar2, step2, v2, eta2, u02, w02, refA)
  call par_io_set_ybasis('cheb-T', 'glc')
  call par_chk_read(ffd, d, n, nx, nz, Re2, t2, dt2, Qtar2, step2, v2, eta2, u02, w02, refB)
  call par_io_set_ybasis('fd-point', 'file', yh)
  call par_chk_read(fchk, d, n, nx, nz, Re2, t2, dt2, Qtar2, step2, v2, eta2, u02, w02, refC)
  call par_io_set_ybasis('cheb-T', 'glc')            ! restore the default stamp
  allok = allok .and. okfd .and. .not. refA .and. .not. refB .and. .not. refC

  ! (5) payload-refusal NEGATIVE leg. Every refusal above exits at the HEADER
  !     or y-record gate, so none of them exercises the short-transfer
  !     machinery (pio_xfer_ok / MPI_Get_count). Build the scenario that does:
  !     an INTACT header (first 1024 bytes) over a truncated payload -- the
  !     reader must refuse via short payload reads, not via any header check. NOT COVERED here (needs
  !     fault injection): the write-side wok=.false. no-rename path -- a
  !     failed MPI write cannot be provoked portably from a test.
  !     Expected refusal chatter ('short read' / 'truncated') on rank 0 is
  !     part of the test.
  if (par_rank == 0) then
     call execute_command_line('cp -f '//fchk//' '//ftr// &
                               ' && truncate -s 4096 '//ftr, exitstat=ios)
     ! non-vacuity guard: if cp failed, ftr is absent and the read below would
     ! refuse at the OPEN gate -- a vacuous pass that never touches the payload
     ! machinery this leg exists to test. Assert the staged file is exactly
     ! 4096 bytes and fold that into the verdict.
     inquire(file=ftr, size=fsz)
     if (ios /= 0 .or. fsz /= 4096) then
        write(*,'(a,i0,a,i0)') ' ** staging failed: exitstat=', ios, '  size=', fsz
        allok = .false.
     end if
  end if
  ok = .true.;  call par_reduce_and(ok)              ! barrier: file staged before any rank opens it
  call par_chk_read(ftr, d, n, nx, nz, Re2, t2, dt2, Qtar2, step2, v2, eta2, u02, w02, refT)
  allok = allok .and. .not. refT

  ! (6) scalar-parse refusal: every gate key intact, the 't = ' VALUE string
  !     corrupted to '?'s in place (byte-count preserving). The reader must
  !     REFUSE: returning ok=.true. with t silently reset to 0 would restart
  !     the campaign clock from zero on real field data.
  !     Expected refusal chatter ('scalar header value') on rank 0.
  if (par_rank == 0) then
     call execute_command_line('cp -f '//fchk//' '//fcv, exitstat=ios)
     if (ios /= 0) allok = .false.
     call stage_header_edit(fcv, char(10)//'t = ', 5, -1, allok)
  end if
  ok = .true.;  call par_reduce_and(ok)
  call par_chk_read(fcv, d, n, nx, nz, Re2, t2, dt2, Qtar2, step2, v2, eta2, u02, w02, refV)
  allok = allok .and. .not. refV

  ! (7) endian refusal: 'endian = little' -> first value char flipped
  !     (same length; header stays ASCII-valid). Without this gate a
  !     big-endian-written file passes every other check and its payload is
  !     silently misread.
  if (par_rank == 0) then
     call execute_command_line('cp -f '//fchk//' '//fen, exitstat=ios)
     if (ios /= 0) allok = .false.
     call stage_header_edit(fen, char(10)//'endian = ', 10, 1, allok)
  end if
  ok = .true.;  call par_reduce_and(ok)
  call par_chk_read(fen, d, n, nx, nz, Re2, t2, dt2, Qtar2, step2, v2, eta2, u02, w02, refE)
  allok = allok .and. .not. refE

  call par_reduce_and(allok)
  if (par_rank == 0) then
     write(*,'(a,i0,a)') ' --- test_par_io_mp (P=', par_nproc, ') ---'
     write(*,'(a,es10.2,a,es10.2,a,es10.2)') '   max|dv|=', dv, '  max|deta|=', de, '  max|dmean|=', dmean
     write(*,'(a,es10.2,a,es10.2,a,es10.2,a,es10.2)') '   [fp32] max|dv|=', dv4, &
          '  max|deta|=', de4, '  max|dmean|=', dmean4, '  (tol ', tol4, ')'
     write(*,'(a,es10.2,a,l1,a,l1,a,l1,a)') '   [fd]   round-trip max|dv|=', dvf, &
          '  refusals(grid,cheb<-fd,fd<-cheb) = ', .not.refA, ',', .not.refB, ',', .not.refC, &
          '  (want T,T,T)'
     write(*,'(a,l1,a)') '   truncated-payload refusal (intact header) = ', .not.refT, &
          '  (want T)'
     write(*,'(a,l1,a,l1,a)') '   corrupt-t / endian refusals = ', &
          .not.refV, ',', .not.refE, '  (want T,T)'
     if (allok) then; write(*,'(a)') ' ALL TESTS PASSED'
     else;            write(*,'(a)') ' SOME TESTS FAILED'; end if
  end if

  deallocate(v, eta, u0, w0, v2, eta2, u02, w02, hv, he, h2, hm, hm2)
  call par_finalize()
  if (.not. allok) error stop 1

contains
  ! unique global value at 0-based (iy, global-kx, iz)
  pure complex(wp) function gval(iy, ikg, iz)
    integer(ip), intent(in) :: iy, ikg, iz
    real(wp) :: r
    r = real(1 + iy + (n+1)*ikg + (n+1)*(nx/2+1)*iz, wp)
    gval = cmplx(r, -r, wp)
  end function gval

  subroutine fail(msg)
    character(*), intent(in) :: msg
    if (par_rank == 0) write(*,'(a,a)') ' SOME TESTS FAILED: ', msg
    call par_finalize();  error stop 1
  end subroutine fail

  ! in-place header edit on a STAGED copy (rank 0 only): find `pat` in the
  ! first 1024 bytes, then overwrite the value that starts `voff` bytes past the
  ! pattern start -- nch = -1 rewrites every char up to the next LF with '?'
  ! (unparseable number), nch = 1 flips one char to 'X' (same-length string
  ! mismatch). Byte count is preserved either way, so every OTHER header line
  ! (and the payload offsets) stay valid -- the refusal under test is the ONLY
  ! reason the read can fail. Pattern-absent counts as a staging failure
  ! (non-vacuity), never a silent pass.
  subroutine stage_header_edit(fnam, pat, voff, nch, okio)
    character(*), intent(in)    :: fnam, pat
    integer,      intent(in)    :: voff, nch
    logical,      intent(inout) :: okio
    integer :: u2, ios2, q1, q2
    hb1 = ''
    open(newunit=u2, file=fnam, access='stream', form='unformatted', &
         status='old', action='readwrite', iostat=ios2)
    if (ios2 /= 0) then
       okio = .false.;  return
    end if
    read(u2, pos=1, iostat=ios2) hb1
    q1 = index(hb1, pat)
    if (ios2 /= 0 .or. q1 == 0) then
       write(*,'(a,a)') ' ** header staging failed for ', trim(fnam)
       okio = .false.;  close(u2);  return
    end if
    q1 = q1 + voff                       ! first byte of the value
    if (nch == 1) then
       hb1(q1:q1) = 'X'
    else
       do q2 = q1, len(hb1)
          if (hb1(q2:q2) == char(10)) exit
          hb1(q2:q2) = '?'
       end do
    end if
    write(u2, pos=1, iostat=ios2) hb1
    close(u2)
    if (ios2 /= 0) okio = .false.
  end subroutine stage_header_edit
end program test_par_io_mp
