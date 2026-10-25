!===============================================================================
! ygrid_mod.F90 -- reader for the ASCII wall-normal GRID FILE.
!
! DESIGN: the solver never computes a stretched
! grid in Fortran. Grid LAWS (P&O-natural alpha-corrected, ...) live in
! tools/gen_ygrid.py, which writes a self-describing ASCII file; this module
! only reads and VALIDATES it. The Chebyshev backend keeps its built-in GLC
! nodes (glc_nodes) -- the file path is the FD backend's grid source. Grid =
! data: the same contract as the v3
! coordinate records (par_io) and the yops facade ("the backend supplies
! NUMBERS, not code paths").
!
! FILE GRAMMAR (written by gen_ygrid.py; '#' and blank lines are comments):
!     ny = <N>            number of NODES (ny = n+1)
!     columns = y [w]     w = quadrature weight column, optional
!     <y> [w]             N data rows, y DESCENDING from +1 to -1 (the code's
!                         storage order, same direction as y_j = cos(pi j/n));
!                         walls EXACTLY +-1 (to 1e-12).
!
! Every rank reads the tiny file itself (shared FS, static content ->
! rank-identical result, no broadcast -- the .latest-pointer convention).
! REFUSES rather than misreads: wrong ny, missing/garbled rows, walls off
! +-1, non-monotone y, non-finite values in
! EITHER column (all gates are acceptance-polarity, since NaN fails every
! ordered comparison and would sail through refuse-on-true forms),
! duplicate 'ny ='/'columns =' headers (which would otherwise silently
! override), extra
! numeric data columns, non-positive weights, and weights whose sum is not 2
! (the basis-independent quadrature identity on [-1,1]; the w column IS the
! discrete bulk flux the CFR constraint enforces). All return ok=.false.
! with a rank-0-printable message in ymsg. Host-only; depends on kinds alone.
!===============================================================================
module ygrid_mod
  use kinds, only: dp, ip
  implicit none
  private
  public :: ygrid_read

contains

  !---------------------------------------------------------------------------
  ! ygrid_read -- parse + validate a grid file.
  !   fname     : path
  !   ny_expect : required node count (the run's n+1); 0 = accept any
  !   y(:)      : nodes, size >= the file's ny (filled 1..ny)
  !   w(:)      : quadrature weights (same size); untouched unless have_w
  !   have_w    : file carried a weight column
  !   ok        : .false. on ANY parse/validation failure (y/w then undefined)
  !   ymsg      : one-line reason on failure ('' on success)
  !---------------------------------------------------------------------------
  subroutine ygrid_read(fname, ny_expect, y, w, have_w, ok, ymsg)
    character(*), intent(in)  :: fname
    integer(ip),  intent(in)  :: ny_expect
    real(dp),     intent(out) :: y(:), w(:)
    logical,      intent(out) :: have_w, ok
    character(*), intent(out) :: ymsg
    character(len=512) :: line
    character(len=64)  :: val
    integer :: u, ios, nyf, nrow, ieq, j
    real(dp) :: a, b, c, wsum
    logical :: saw_cols

    ok = .false.;  have_w = .false.;  ymsg = '';  nyf = -1;  nrow = 0
    saw_cols = .false.
    open(newunit=u, file=trim(fname), status='old', action='read', iostat=ios)
    if (ios /= 0) then
       ymsg = 'cannot open '//trim(fname);  return
    end if

    do
       read(u, '(a)', iostat=ios) line
       if (ios /= 0) exit
       line = adjustl(line)
       if (len_trim(line) == 0) cycle
       if (line(1:1) == '#') cycle
       ieq = index(line, '=')
       if (line(1:2) == 'ny' .and. ieq > 0) then
          if (nrow > 0) then                     ! header after data = corrupt file
             ymsg = 'header line after data rows in '//trim(fname);  close(u);  return
          end if
          if (nyf > 0) then                      ! duplicate ny header: a second
             ymsg = 'duplicate ny line in '//trim(fname);  close(u);  return
          end if                                 ! would silently override the first
          val = adjustl(line(ieq+1:))
          read(val, *, iostat=ios) nyf
          if (ios /= 0 .or. nyf < 3) then
             ymsg = 'bad ny line in '//trim(fname);  close(u);  return
          end if
          if (ny_expect > 0 .and. nyf /= ny_expect) then
             write(ymsg, '(a,i0,a,i0,a)') 'grid file has ny = ', nyf, &
                  ' but the run needs ', ny_expect, ' ('//trim(fname)//')'
             close(u);  return
          end if
          if (nyf > size(y)) then
             ymsg = 'y array too small for '//trim(fname);  close(u);  return
          end if
          if (have_w .and. nyf > size(w)) then   ! columns line may precede ny
             ymsg = 'w array too small for '//trim(fname);  close(u);  return
          end if
       else if (line(1:7) == 'columns' .and. ieq > 0) then
          if (nrow > 0) then                     ! header after data = corrupt file
             ymsg = 'header line after data rows in '//trim(fname);  close(u);  return
          end if
          if (saw_cols) then                     ! duplicate columns header:
             ymsg = 'duplicate columns line in '//trim(fname);  close(u);  return
          end if                                 ! same silent-override class as ny
          saw_cols = .true.
          val = adjustl(line(ieq+1:))
          have_w = (index(val, 'w') > 0)
          if (have_w .and. nyf > 0 .and. nyf > size(w)) then
             ymsg = 'w array too small for '//trim(fname);  close(u);  return
          end if
       else
          ! data row (list-directed: 1 or 2 reals)
          if (nyf < 0) then
             ymsg = 'data before ny line in '//trim(fname);  close(u);  return
          end if
          if (nrow >= nyf) then
             ymsg = 'more data rows than ny in '//trim(fname);  close(u);  return
          end if
          ! huge() sentinels catch list-directed NULL fields (a bare ',' or a
          ! '/' terminator returns ios=0 with the item UNTOUCHED -- without the
          ! sentinel that would silently pass stale values through). c is
          ! an extra-column probe: a row carrying MORE numbers than the header
          ! declared would be silently truncated by the list-directed read
          ! (incl. the header-says-'y'-but-data-has-w mismatch) -- it is refused.
          a = huge(1.0_dp);  b = huge(1.0_dp);  c = huge(1.0_dp)
          if (have_w) then
             read(line, *, iostat=ios) a, b, c
             if (ios == 0 .and. c /= huge(1.0_dp)) then
                ymsg = 'more columns than declared in '//trim(fname);  close(u);  return
             end if
             read(line, *, iostat=ios) a, b
          else
             read(line, *, iostat=ios) a, b
             if (ios == 0 .and. b /= huge(1.0_dp)) then
                ymsg = 'data has a column the header does not declare in '//trim(fname)
                close(u);  return
             end if
             read(line, *, iostat=ios) a
             b = 0.0_dp
          end if
          if (ios /= 0 .or. a == huge(1.0_dp) .or. b == huge(1.0_dp)) then
             ymsg = 'garbled data row in '//trim(fname);  close(u);  return
          end if
          ! refuse non-finite values AT READ TIME, acceptance-polarity:
          ! NaN fails every ordered comparison, so tests must be written as
          ! "accept iff provably good" -- (x==x) is false for NaN, and the
          ! huge() bound catches +-Inf. Refuse-on-true wall/monotone gates
          ! would let a NaN y column through every one of them.
          if (.not. (a == a .and. abs(a) < huge(1.0_dp))) then
             ymsg = 'non-finite y value in '//trim(fname);  close(u);  return
          end if
          if (have_w) then
             if (.not. (b == b .and. abs(b) < huge(1.0_dp))) then
                ymsg = 'non-finite w value in '//trim(fname);  close(u);  return
             end if
          end if
          nrow = nrow + 1
          y(nrow) = a
          if (have_w) w(nrow) = b
       end if
    end do
    close(u)

    if (nyf < 0) then
       ymsg = 'no ny line in '//trim(fname);  return
    end if
    if (nrow /= nyf) then
       write(ymsg, '(a,i0,a,i0,a)') 'expected ', nyf, ' rows, found ', nrow, &
            ' ('//trim(fname)//')'
       return
    end if

    ! --- validation: walls exactly +-1, strictly DESCENDING ------------------
    ! ACCEPTANCE polarity throughout: every gate states the condition a
    ! GOOD file satisfies and refuses otherwise. Refuse-on-true forms
    ! (abs(..) > tol, y(j) >= y(j-1)) all evaluate FALSE against NaN operands,
    ! so a NaN-carrying column would sail through the lot. Non-finite values
    ! are already refused at read time above; the polarity here is defence in
    ! depth.
    if (.not. (abs(y(1) - 1.0_dp) <= 1.0e-12_dp .and. &
               abs(y(nyf) + 1.0_dp) <= 1.0e-12_dp)) then
       ymsg = 'walls not at +-1 in '//trim(fname);  return
    end if
    do j = 2, nyf
       if (.not. (y(j) < y(j-1))) then
          ymsg = 'y not strictly decreasing in '//trim(fname);  return
       end if
    end do

    ! --- w column: inspected, because w defines the discrete bulk flux the
    ! CFR constraint enforces. Any
    ! consistent quadrature on [-1,1] integrates constants exactly, so
    ! sum(w) = 2 is basis-independent (trapezoid today, scheme-matched later);
    ! positivity is required of every rule gen_ygrid.py may emit.
    if (have_w) then
       wsum = 0.0_dp
       do j = 1, nyf
          if (.not. (w(j) > 0.0_dp)) then
             ymsg = 'non-positive quadrature weight in '//trim(fname);  return
          end if
          wsum = wsum + w(j)
       end do
       if (.not. (abs(wsum - 2.0_dp) <= 1.0e-8_dp)) then
          ymsg = 'quadrature weights do not sum to 2 in '//trim(fname);  return
       end if
    end if
    ok = .true.
  end subroutine ygrid_read

end module ygrid_mod
