!===============================================================================
! test_cfl.F90  --  unit test for cfl_mod (convective CFL dt limit).
!
! Validates, on KNOWN physical velocity fields (no transform needed -- the
! spectral->physical path is exercised elsewhere; here we isolate the CFL logic):
!   1. cfl_inv_dy  -- local GLC node spacings; near-wall clustering => max inv_dy
!      at the walls (smallest gap).
!   2. cfl_freq_max -- the pointwise frequency |u|*inv_dx + |v|*inv_dy + |w|*inv_dz
!      reduced to its grid maximum. Checked two ways: against the analytic answer
!      for structured fields (uniform-u, uniform-v, single spike) AND against an
!      independent HOST replication of the same reduction for a random field
!      (catches device-kernel/reduction bugs vs. a formula bug, respectively).
!   3. cfl_dt -- dt = C / fmax.
! NaN/Inf-hardened. Prints CFL_STAB (the max stable CFL).
!
! Build/link:  nvfortran -cuda -c cfl_mod.F90
!              nvfortran -cuda -o test_cfl test_cfl.F90 kinds.o cfl_mod.o
! (kinds.o comes from types.F90.) Needs a GPU node.
!===============================================================================
program test_cfl
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,   only: tp, dp, ip
  use cfl_mod, only: CFL_STAB, cfl_inv_dy, cfl_freq_max, cfl_dt
  use cudafor
  implicit none

  integer(ip), parameter :: n = 8, nx = 4, nz = 4, ny = n+1
  real(dp),    parameter :: PI = 3.141592653589793238462643_dp
  real(dp),    parameter :: Lx = 2.0_dp*PI, Lz = 2.0_dp*PI
  real(dp),    parameter :: tol = 1.0e-12_dp

  real(dp) :: y(0:n), inv_dy(0:n), inv_dx, inv_dz
  real(tp) :: uh(ny,nx,nz), vh(ny,nx,nz), wh(ny,nx,nz)   ! velocity -> tp (cfl_freq_max dummies)
  real(tp), device :: ud(ny,nx,nz), vd(ny,nx,nz), wd(ny,nx,nz)   ! tp device velocity
  real(dp), device :: inv_dy_d(ny)                              ! inv_dy stays dp
  real(dp) :: fmax, fhost, expect, dt, rr
  integer(ip) :: j, ix, iz
  logical :: ok

  ! GLC nodes and reciprocal spacings
  do j = 0, n
     y(j) = cos(PI*real(j,dp)/real(n,dp))
  end do
  call cfl_inv_dy(n, y, inv_dy)
  inv_dx = real(nx,dp)/Lx
  inv_dz = real(nz,dp)/Lz
  inv_dy_d = inv_dy            ! host (0:n) -> device (ny), 1-based

  ok = .true.
  write(*,'(a)') '=== cfl_mod: convective CFL dt limit ==='
  write(*,'(a,f10.6)') '  CFL_STAB (max stable CFL, RK3 imaginary-axis) = ', CFL_STAB

  ! --- 1. inv_dy: near-wall clustering => largest inv_dy at the walls -----------
  if (abs(inv_dy(0) - maxval(inv_dy)) > tol*maxval(inv_dy)) then
     write(*,'(a)') '  [FAIL] inv_dy not maximal at the wall node (clustering missed)'
     ok = .false.
  else
     write(*,'(a,es12.4,a,es12.4)') '  [ok] inv_dy max at wall: inv_dy(0)=', inv_dy(0), &
          '  core inv_dy(n/2)=', inv_dy(n/2)
  end if
  ! spot-check one interior value against a direct recompute
  expect = 1.0_dp/min(abs(y(2)-y(3)), abs(y(3)-y(4)))
  if (abs(inv_dy(3) - expect) > tol*expect) then
     write(*,'(a)') '  [FAIL] inv_dy(3) disagrees with direct min-gap recompute'
     ok = .false.
  end if

  ! --- 2a. uniform u = 2, v = w = 0  => fmax = 2*inv_dx -------------------------
  uh = 2.0_dp; vh = 0.0_dp; wh = 0.0_dp
  ud = uh; vd = vh; wd = wh
  call cfl_freq_max(ny, nx, nz, ud, vd, wd, inv_dx, inv_dz, inv_dy_d, fmax)
  expect = 2.0_dp*inv_dx
  call check('uniform u: fmax = 2*inv_dx', fmax, expect, tol, ok)

  ! --- 2b. uniform v = 1, u = w = 0  => fmax = max_j inv_dy (a wall node) -------
  uh = 0.0_dp; vh = 1.0_dp; wh = 0.0_dp
  ud = uh; vd = vh; wd = wh
  call cfl_freq_max(ny, nx, nz, ud, vd, wd, inv_dx, inv_dz, inv_dy_d, fmax)
  expect = maxval(inv_dy)
  call check('uniform v: fmax = max(inv_dy) (near-wall sets it)', fmax, expect, tol, ok)

  ! --- 2c. single spike on a small background  => spike point dominates ---------
  uh = 0.01_dp; vh = 0.01_dp; wh = 0.01_dp
  uh(4,2,3) = 5.0_dp; vh(4,2,3) = 3.0_dp; wh(4,2,3) = 4.0_dp
  ud = uh; vd = vh; wd = wh
  call cfl_freq_max(ny, nx, nz, ud, vd, wd, inv_dx, inv_dz, inv_dy_d, fmax)
  ! inv_dy_d(4) corresponds to host inv_dy(3)
  expect = 5.0_dp*inv_dx + 3.0_dp*inv_dy(3) + 4.0_dp*inv_dz
  call check('spike: fmax = spike-point frequency', fmax, expect, tol, ok)

  ! --- 2d. random field: device reduction vs independent host replication ------
  block   ! [repro] fixed seed, applied once per run -> the suite is a bit-identity oracle
          ! across the precision phases (replaces a bare call random_seed())
    integer              :: rs_n_, rs_j_
    integer, allocatable :: rs_seed_(:)
    logical, save        :: rs_done_ = .false.
    if (.not. rs_done_) then
       call random_seed(size = rs_n_)
       allocate(rs_seed_(rs_n_))
       rs_seed_ = [ (1789 + 101*rs_j_, rs_j_ = 0, rs_n_ - 1) ]
       call random_seed(put = rs_seed_)
       rs_done_ = .true.
    end if
  end block
  do iz = 1, nz; do ix = 1, nx; do j = 1, ny
     call random_number(rr); uh(j,ix,iz) = rr - 0.5_dp
     call random_number(rr); vh(j,ix,iz) = rr - 0.5_dp
     call random_number(rr); wh(j,ix,iz) = rr - 0.5_dp
  end do; end do; end do
  ud = uh; vd = vh; wd = wh
  call cfl_freq_max(ny, nx, nz, ud, vd, wd, inv_dx, inv_dz, inv_dy_d, fmax)
  fhost = 0.0_dp
  do iz = 1, nz; do ix = 1, nx; do j = 1, ny
     fhost = max(fhost, abs(uh(j,ix,iz))*inv_dx + abs(vh(j,ix,iz))*inv_dy(j-1) &
                       + abs(wh(j,ix,iz))*inv_dz)
  end do; end do; end do
  call check('random field: device fmax = host reduction', fmax, fhost, tol, ok)

  ! --- 3. cfl_dt = C / fmax -----------------------------------------------------
  dt = cfl_dt(fmax, 1.2_dp)
  call check('cfl_dt = C/fmax', dt, 1.2_dp/fmax, tol, ok)

  if (ok) then
     write(*,'(/,a)') 'ALL TESTS PASSED'
  else
     write(*,'(/,a)') 'SOME TESTS FAILED'
  end if

contains

  subroutine check(label, got, want, rtol, ok)
    character(*), intent(in)    :: label
    real(dp),     intent(in)    :: got, want, rtol
    logical,      intent(inout) :: ok
    real(dp) :: d
    logical  :: bad
    d = abs(got - want)
    bad = .not. (d == d)                         ! NaN
    if (abs(got) > huge(1.0_dp) .or. abs(want) > huge(1.0_dp)) bad = .true.
    if (bad .or. d > rtol*max(abs(want), tiny(1.0_dp))) then
       write(*,'(a,a,a,es14.6,a,es14.6,a,es10.2)') '  [FAIL] ', label, &
            ': got=', got, ' want=', want, ' |d|=', d
       ok = .false.
    else
       write(*,'(a,a,a,es14.6)') '  [ok] ', label, ' = ', got
    end if
  end subroutine check

end program test_cfl
