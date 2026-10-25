!===============================================================================
! test_ygrid.F90 -- unit gate for the ASCII y-grid file reader
! (ygrid_mod). SELF-CONTAINED: writes its own files (no Python at test time),
! reads them back, and requires:
!   (1) a well-formed file with weights + comments + blank lines round-trips
!       BIT-EXACTLY (17-significant-digit ASCII round-trips IEEE doubles);
!   (2) a weight-less file parses with have_w = .false.;
!   (3) REFUSALS: wrong ny for the run, wall not at +1, non-monotone y,
!       truncated data -- each returns ok=.false. (never a misread grid);
!   (4) further refusal classes: NaN in y, NaN in w,
!       extra numeric column, duplicate 'ny =' header, non-positive w, and
!       weights that do not sum to 2 -- the classes a plain refuse-on-true
!       gate is provably transparent to (NaN fails every ordered compare).
! Host-only. Auto-discovered by run_all_tests.sh.
!===============================================================================
program test_ygrid
  use kinds,     only: dp, ip
  use ygrid_mod, only: ygrid_read
  implicit none
  integer(ip), parameter :: n = 16, ny = n + 1
  real(dp), parameter :: PI = 3.141592653589793238462643_dp
  real(dp)    :: y0(ny), w0(ny), y1(ny), w1(ny)
  integer(ip) :: j
  integer     :: u
  logical     :: have_w, ok, allok
  character(len=256) :: msg
  character(*), parameter :: fgood = 'build/test_ygrid_good.dat'
  character(*), parameter :: fnow  = 'build/test_ygrid_now.dat'
  character(*), parameter :: fbad  = 'build/test_ygrid_bad.dat'

  allok = .true.

  ! reference grid: GLC nodes + positive weights NORMALIZED to sum(w) = 2
  ! (the reader enforces the basis-independent quadrature identity
  ! sum(w) = 2 on [-1,1]; arbitrary unnormalized weights are refused)
  do j = 1, ny
     y0(j) = cos(PI*real(j-1,dp)/real(n,dp))
     w0(j) = 0.1_dp + 1.0_dp/real(j,dp)
  end do
  w0 = w0 * (2.0_dp/sum(w0))
  y0(1) = 1.0_dp;  y0(ny) = -1.0_dp

  ! (1) well-formed file with weights, comments, blank lines ------------------
  open(newunit=u, file=fgood, status='replace', action='write')
  write(u,'(a)') '# comment line'
  write(u,'(a)') ''
  write(u,'(a,i0)') 'ny = ', ny
  write(u,'(a)') 'columns = y w'
  do j = 1, ny
     write(u,'(es24.16e3,2x,es24.16e3)') y0(j), w0(j)
  end do
  close(u)
  call ygrid_read(fgood, ny, y1, w1, have_w, ok, msg)
  call report('well-formed file parses (ok, have_w)', ok .and. have_w)
  call report('y + w round-trip BIT-EXACT through 17-digit ASCII', &
              ok .and. all(y1(1:ny) == y0) .and. all(w1(1:ny) == w0))
  call ygrid_read(fgood, 0_ip, y1, w1, have_w, ok, msg)
  call report('ny_expect = 0 accepts any count', ok)

  ! (2) weight-less file -------------------------------------------------------
  open(newunit=u, file=fnow, status='replace', action='write')
  write(u,'(a,i0)') 'ny = ', ny
  write(u,'(a)') 'columns = y'
  do j = 1, ny
     write(u,'(es24.16e3)') y0(j)
  end do
  close(u)
  call ygrid_read(fnow, ny, y1, w1, have_w, ok, msg)
  call report('weight-less file: ok, have_w = .false., y exact', &
              ok .and. (.not. have_w) .and. all(y1(1:ny) == y0))

  ! (3) refusals ---------------------------------------------------------------
  call ygrid_read(fgood, ny + 2, y1, w1, have_w, ok, msg)
  call report('REFUSED: file ny /= run ny        ['//trim(msg)//']', .not. ok)

  call write_bad(fbad, y0, w0, ny, 1_ip)          ! wall off +1
  call ygrid_read(fbad, ny, y1, w1, have_w, ok, msg)
  call report('REFUSED: wall not at +1           ['//trim(msg)//']', .not. ok)

  call write_bad(fbad, y0, w0, ny, 2_ip)          ! non-monotone
  call ygrid_read(fbad, ny, y1, w1, have_w, ok, msg)
  call report('REFUSED: y not strictly decreasing['//trim(msg)//']', .not. ok)

  call write_bad(fbad, y0, w0, ny, 3_ip)          ! truncated
  call ygrid_read(fbad, ny, y1, w1, have_w, ok, msg)
  call report('REFUSED: truncated data           ['//trim(msg)//']', .not. ok)

  ! (4) the new refusal classes ------------------------------------------
  ! (label wording: 'NaN-token', hyphenated, NOT the standalone word -- the
  ! harness's non-finite scan (run_all_tests.sh:193-199) fails any test whose
  ! stdout carries a whitespace-delimited nan/inf token, exactly the leaked-
  ! value heuristic it should apply; a bare 'NaN' in a PASS label is
  ! indistinguishable from a leak. Caught on the first gate run of round C.)
  call write_bad(fbad, y0, w0, ny, 4_ip)          ! literal 'NaN' token in y
  call ygrid_read(fbad, ny, y1, w1, have_w, ok, msg)
  call report('REFUSED: NaN-token in y column    ['//trim(msg)//']', .not. ok)

  call write_bad(fbad, y0, w0, ny, 5_ip)          ! literal 'NaN' token in w
  call ygrid_read(fbad, ny, y1, w1, have_w, ok, msg)
  call report('REFUSED: NaN-token in w column    ['//trim(msg)//']', .not. ok)

  call write_bad(fbad, y0, w0, ny, 6_ip)          ! a third numeric column
  call ygrid_read(fbad, ny, y1, w1, have_w, ok, msg)
  call report('REFUSED: extra numeric column     ['//trim(msg)//']', .not. ok)

  call write_bad(fbad, y0, w0, ny, 7_ip)          ! duplicate 'ny =' header
  call ygrid_read(fbad, ny, y1, w1, have_w, ok, msg)
  call report('REFUSED: duplicate ny header      ['//trim(msg)//']', .not. ok)

  call write_bad(fbad, y0, w0, ny, 8_ip)          ! one negative weight
  call ygrid_read(fbad, ny, y1, w1, have_w, ok, msg)
  call report('REFUSED: non-positive weight      ['//trim(msg)//']', .not. ok)

  call write_bad(fbad, y0, w0, ny, 9_ip)          ! weights scaled: sum(w) = 6
  call ygrid_read(fbad, ny, y1, w1, have_w, ok, msg)
  call report('REFUSED: weights do not sum to 2  ['//trim(msg)//']', .not. ok)

  if (allok) then; print '(a)', 'ALL TESTS PASSED'
  else;            print '(a)', 'SOME TESTS FAILED'; end if
  if (.not. allok) error stop 1

contains

  subroutine write_bad(path, yy, ww, m, mode)
    character(*), intent(in) :: path
    integer(ip),  intent(in) :: m, mode
    real(dp),     intent(in) :: yy(m), ww(m)
    real(dp) :: yb(m), wb(m)
    integer  :: uu, jj, mtop
    yb = yy;  wb = ww
    mtop = m
    select case (mode)
    case (1);  yb(1) = 0.999_dp                   ! wall off +1
    case (2);  yb(5) = yb(3)                      ! breaks monotonicity
    case (3);  mtop = m - 2                       ! fewer rows than ny
    case (8);  wb(7) = -wb(7)                     ! one negative weight
    case (9);  wb = 3.0_dp*wb                     ! sum(w) = 6, not 2
    end select
    open(newunit=uu, file=path, status='replace', action='write')
    write(uu,'(a,i0)') 'ny = ', m
    if (mode == 7) write(uu,'(a,i0)') 'ny = ', m  ! duplicate header (the old
                                                  ! silent-override quirk)
    write(uu,'(a)') 'columns = y w'
    do jj = 1, mtop
       ! modes 4/5 plant a literal 'NaN' token (parses with ios == 0 under
       ! F2003 list-directed input); mode 6
       ! appends a third numeric column the header does not declare.
       if (mode == 4 .and. jj == 6) then
          write(uu,'(a,2x,es24.16e3)') '                     NaN', wb(jj)
       else if (mode == 5 .and. jj == 6) then
          write(uu,'(es24.16e3,2x,a)') yb(jj), 'NaN'
       else if (mode == 6 .and. jj == 6) then
          write(uu,'(es24.16e3,2x,es24.16e3,2x,es24.16e3)') yb(jj), wb(jj), 0.5_dp
       else
          write(uu,'(es24.16e3,2x,es24.16e3)') yb(jj), wb(jj)
       end if
    end do
    close(uu)
  end subroutine write_bad

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. okk) allok = .false.
  end subroutine report

end program test_ygrid
