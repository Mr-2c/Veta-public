!===============================================================================
! test_fdops.F90 -- oracle for the FD-y operator kit (fdops_mod).
! Host-only; auto-discovered by run_all_tests.sh.
!
!   (1) POLYNOMIAL EXACTNESS, every row, p = 0..kl: with row stencils of at
!       least kl+1 nodes, D1 y^p and D2 y^p must be exact to round-off at ALL
!       rows including the tapered wall rows. This is a complete algebraic
!       characterization of the Fornberg weights -- a transcription error in
!       the recurrence CANNOT pass it.
!   (2) Interior rows (full 2kl+1 stencils): exactness through p = 2kl.
!   (3) Smooth-function accuracy: D1/D2 of sin(2y) on GLC(64) to tight abs tol.
!   (4) Helmholtz MMS: (-D2 + mu) u = f with u = (1-y^2) sin(3y), analytic f,
!       Dirichlet 0 walls; solve via fd_helm_dirichlet + no-pivot LU at
!       ny = 33 and 65 -- error small AND shrinking by > 8x (order >= 3 gate;
!       the scheme is ~7th order, the loose gate avoids brittleness).
!   (5) Zero-pivot guard: a singular system must return ok = .false.
!   (6) THE REAL GRID: read the committed ygrid file (B1.1's Re2000 natural
!       grid, ny = 481) through ygrid_mod and require polynomial exactness on
!       it -- the B1.1 file contract and the B1.2 operators gate each other.
!===============================================================================
program test_fdops
  use kinds,     only: dp, ip
  use fdops_mod, only: fd_weights, fd_build_bands, fd_apply_band, &
                       fd_helm_dirichlet, fd_band_factor, fd_band_solve
  use ygrid_mod, only: ygrid_read
  implicit none
  integer(ip), parameter :: kl = 4
  real(dp), parameter :: PI = 3.141592653589793238462643_dp
  logical :: allok
  allok = .true.

  call poly_block(glc(32_ip), 'glc(33)')
  call sin_block(glc(64_ip))
  call mms_block()
  call pivot_block()
  call realgrid_block()
  call refine_block()      ! measured-order gates, stretched grids
  call facres_block()      ! no-pivot factor residual, production shape

  if (allok) then; print '(a)', 'ALL TESTS PASSED'
  else;            print '(a)', 'SOME TESTS FAILED'; end if
  if (.not. allok) error stop 1

contains

  function glc(n) result(y)
    integer(ip), intent(in) :: n
    real(dp) :: y(n+1)
    integer(ip) :: j
    do j = 0, n
       y(j+1) = cos(PI*real(j,dp)/real(n,dp))
    end do
  end function glc

  !-- (1)+(2): polynomial exactness on an arbitrary node set -------------------
  subroutine poly_block(y, label)
    real(dp),     intent(in) :: y(:)
    character(*), intent(in) :: label
    integer(ip) :: ny, i, j, p
    real(dp), allocatable :: d1b(:,:), d2b(:,:), f(:), g1(:), g2(:), sc1(:), sc2(:)
    real(dp) :: e1, e2, ref1, ref2
    logical  :: ok_all, ok_int
    ny = size(y)
    allocate(d1b(2*kl+1,ny), d2b(2*kl+1,ny), f(ny), g1(ny), g2(ny), sc1(ny), sc2(ny))
    call fd_build_bands(ny, kl, y, d1b, d2b)
    do i = 1, ny                                  ! per-row weight scales
       sc1(i) = 0.0_dp;  sc2(i) = 0.0_dp
       do j = max(1_ip,i-kl), min(ny,i+kl)
          sc1(i) = sc1(i) + abs(d1b(kl+1+i-j, j))
          sc2(i) = sc2(i) + abs(d2b(kl+1+i-j, j))
       end do
    end do
    ok_all = .true.;  ok_int = .true.
    do p = 0, 2*kl
       f = y**p
       call fd_apply_band(ny, kl, d1b, f, g1)
       call fd_apply_band(ny, kl, d2b, f, g2)
       do i = 1, ny
          ref1 = 0.0_dp;  ref2 = 0.0_dp
          if (p >= 1) ref1 = real(p,dp)*y(i)**(p-1)
          if (p >= 2) ref2 = real(p,dp)*real(p-1,dp)*y(i)**(p-2)
          e1 = abs(g1(i) - ref1);  e2 = abs(g2(i) - ref2)
          if (p <= kl) then
             if (e1 > 1.0e-10_dp*max(1.0_dp, sc1(i))) ok_all = .false.
             if (e2 > 1.0e-10_dp*max(1.0_dp, sc2(i))) ok_all = .false.
          end if
          if (i > kl .and. i <= ny-kl) then       ! full central stencils
             if (e1 > 1.0e-10_dp*max(1.0_dp, sc1(i))) ok_int = .false.
             if (e2 > 1.0e-10_dp*max(1.0_dp, sc2(i))) ok_int = .false.
          end if
       end do
    end do
    call report('poly exactness p<=kl, ALL rows incl. tapered walls ['//label//']', ok_all)
    call report('poly exactness p<=2*kl, interior rows              ['//label//']', ok_int)
  end subroutine poly_block

  !-- (3): smooth-function accuracy -------------------------------------------
  subroutine sin_block(y)
    real(dp), intent(in) :: y(:)
    integer(ip) :: ny, i
    real(dp), allocatable :: d1b(:,:), d2b(:,:), f(:), g(:)
    real(dp) :: e1, e2
    logical  :: ok
    ny = size(y)
    allocate(d1b(2*kl+1,ny), d2b(2*kl+1,ny), f(ny), g(ny))
    call fd_build_bands(ny, kl, y, d1b, d2b)
    do i = 1, ny
       f(i) = sin(2.0_dp*y(i))
    end do
    call fd_apply_band(ny, kl, d1b, f, g)
    e1 = 0.0_dp
    do i = 1, ny
       e1 = max(e1, abs(g(i) - 2.0_dp*cos(2.0_dp*y(i))))
    end do
    call fd_apply_band(ny, kl, d2b, f, g)
    e2 = 0.0_dp
    do i = 1, ny
       e2 = max(e2, abs(g(i) + 4.0_dp*sin(2.0_dp*y(i))))
    end do
    ok = (e1 < 1.0e-8_dp) .and. (e2 < 1.0e-6_dp)
    if (.not. ok) print '(a,es10.2,a,es10.2)', '   [sin] max|D1 err|=', e1, '  max|D2 err|=', e2
    call report('sin(2y) on glc(65): D1 err < 1e-8, D2 err < 1e-6', ok)
  end subroutine sin_block

  !-- (4): Helmholtz MMS solve at two resolutions ------------------------------
  subroutine mms_block()
    real(dp) :: err33, err65
    logical  :: ok
    call mms_one(32_ip, err33, ok)
    if (ok) call mms_one(64_ip, err65, ok)
    if (.not. ok) then
       call report('Helmholtz MMS: factor ok', .false.);  return
    end if
    call report('Helmholtz MMS (-D2+25)u = f, Dirichlet: err(33) < 1e-5', err33 < 1.0e-5_dp)
    call report('Helmholtz MMS: err(65) < err(33)/8 (order gate)', err65*8.0_dp < err33)
    if (.not. (err33 < 1.0e-5_dp .and. err65*8.0_dp < err33)) &
       print '(a,es10.2,a,es10.2)', '   [mms] err33=', err33, '  err65=', err65
  end subroutine mms_block

  subroutine mms_one(n, err, ok)
    integer(ip), intent(in)  :: n
    real(dp),    intent(out) :: err
    logical,     intent(out) :: ok
    real(dp), parameter :: MU = 25.0_dp
    real(dp), allocatable :: y(:), d1b(:,:), d2b(:,:), ab(:,:), b(:)
    real(dp) :: s, cx, uex
    integer(ip) :: ny, i
    ny = n + 1
    allocate(y(ny), d1b(2*kl+1,ny), d2b(2*kl+1,ny), ab(2*kl+1,ny), b(ny))
    y = glc(n)
    call fd_build_bands(ny, kl, y, d1b, d2b)
    call fd_helm_dirichlet(ny, kl, d2b, MU, ab)
    do i = 1, ny        ! f = -u'' + mu u for u = (1-y^2) sin(3y)
       s  = sin(3.0_dp*y(i));  cx = cos(3.0_dp*y(i))
       b(i) = 2.0_dp*s + 12.0_dp*y(i)*cx + (9.0_dp + MU)*(1.0_dp - y(i)**2)*s
    end do
    b(1) = 0.0_dp;  b(ny) = 0.0_dp
    call fd_band_factor(ny, kl, ab, ok)
    if (.not. ok) then; err = huge(1.0_dp); return; end if
    call fd_band_solve(ny, kl, ab, b)
    err = 0.0_dp
    do i = 1, ny
       uex = (1.0_dp - y(i)**2)*sin(3.0_dp*y(i))
       err = max(err, abs(b(i) - uex))
    end do
  end subroutine mms_one

  !-- (5): zero-pivot guard ----------------------------------------------------
  subroutine pivot_block()
    integer(ip), parameter :: ny = 12
    real(dp) :: ab(2*kl+1, ny)
    integer(ip) :: j
    logical :: ok
    ab = 0.0_dp
    do j = 1, ny
       ab(kl+1, j) = 1.0_dp                       ! identity band
    end do
    do j = max(1_ip, 5-kl), min(ny, 5+kl)         ! kill ROW 5 entirely
       ab(kl+1+5-j, j) = 0.0_dp
    end do
    call fd_band_factor(ny, kl, ab, ok)
    call report('zero-pivot guard: singular band REFUSED (ok=.false.)', .not. ok)
  end subroutine pivot_block

  !-- (6): the committed B1.1 natural grid through ygrid_mod -------------------
  subroutine realgrid_block()
    integer(ip), parameter :: nyg = 481
    real(dp) :: yg(nyg), wg(nyg)
    logical  :: have_w, ok
    character(len=256) :: msg
    call ygrid_read('examples/for_testing/ygrid_Re2000_natural481.dat', nyg, &
                    yg, wg, have_w, ok, msg)
    if (.not. ok) then
       print '(a)', '   [realgrid] '//trim(msg)
       call report('committed ygrid file reads (Re2000 natural, ny=481)', .false.)
       return
    end if
    call report('committed ygrid file reads (Re2000 natural, ny=481)', .true.)
    call poly_block(yg, 'Re2000 natural 481')
  end subroutine realgrid_block

  !-- tanh-stretched grid, DESCENDING +1..-1 (production orientation); b sets
  !   the clustering: b=1.2 ~ dy ratio 3 (mild), b=2.9 ~ 83 (natural-grid class)
  function tanh_grid(n, b) result(y)
    integer(ip), intent(in) :: n
    real(dp),    intent(in) :: b
    real(dp) :: y(n+1), s
    integer(ip) :: j
    do j = 0, n
       s = real(j,dp)/real(n,dp)
       y(j+1) = -tanh(b*(2.0_dp*s - 1.0_dp))/tanh(b)
    end do
  end function tanh_grid

  !-- max |D2 f - f''| split into interior rows and wall (one-sided) rows,
  !   f = sin(2.3y+0.4) exp(0.3y) (non-polynomial, non-symmetric)
  subroutine d2_err_tanh(n, b, ei, ew)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: b
    real(dp),    intent(out) :: ei, ew
    real(dp) :: y(n+1), fv(n+1), g(n+1), ref
    real(dp) :: d1b(2*kl+1, n+1), d2b(2*kl+1, n+1)
    integer(ip) :: i, ny
    ny = n + 1
    y = tanh_grid(n, b)
    do i = 1, ny
       fv(i) = sin(2.3_dp*y(i) + 0.4_dp)*exp(0.3_dp*y(i))
    end do
    call fd_build_bands(ny, kl, y, d1b, d2b)
    call fd_apply_band(ny, kl, d2b, fv, g)
    ei = 0.0_dp;  ew = 0.0_dp
    do i = 1, ny
       ref = ((0.3_dp**2 - 2.3_dp**2)*sin(2.3_dp*y(i) + 0.4_dp) &
              + 2.0_dp*0.3_dp*2.3_dp*cos(2.3_dp*y(i) + 0.4_dp))*exp(0.3_dp*y(i))
       if (i > kl+1 .and. i < ny-kl) then
          ei = max(ei, abs(g(i) - ref))
       else if (i > 1 .and. i < ny) then           ! one-sided rows, excl BC rows
          ew = max(ew, abs(g(i) - ref))
       end if
    end do
  end subroutine d2_err_tanh

  !-- measured-ORDER refinement gates on STRETCHED grids. The
  !   mms '>8x' ratio gate would pass a 7th->3rd-order regression; these rows
  !   pin the measured order itself. Model + HARDWARE
  !   (src/models/work_a2a4_fd_guard_model.py, on the
  !   BANDED wall stencils fd_build_bands actually uses -- wall rows get
  !   FEWER points so implicit solves stay banded): mild
  !   stretch b=1.2, rungs n=16->24->32: interior p=7.43/7.71, wall
  !   p=4.86/4.75 (model == hardware to the digit) -- the 8th->~4th
  !   taper, pinned: gates interior 6.5, wall 4.0 (a drop to 3rd fails).
  !   STRONG stretch b=2.9 (natural-grid class): wall-row truncation sits
  !   BELOW the eps/dy^2 roundoff envelope at any usable n (orders
  !   unmeasurable there) -> gate ABSOLUTE error at n=48 instead:
  !   interior <= 3e-6 (measured 2.8e-7), wall <= 1e-8 (measured 1.5e-10).
  subroutine refine_block()
    real(dp) :: ei(3), ew(3), e48i, e48w, p1i, p1w, p2i, p2w, r1, r2
    integer(ip), parameter :: nv(3) = [16_ip, 24_ip, 32_ip]
    integer(ip) :: r
    do r = 1, 3
       call d2_err_tanh(nv(r), 1.2_dp, ei(r), ew(r))
    end do
    r1 = log(real(nv(2),dp)/real(nv(1),dp));  r2 = log(real(nv(3),dp)/real(nv(2),dp))
    p1i = log(ei(1)/ei(2))/r1;  p1w = log(ew(1)/ew(2))/r1
    p2i = log(ei(2)/ei(3))/r2;  p2w = log(ew(2)/ew(3))/r2
    call report('refine tanh(b=1.2) 16->24: interior order >= 6.5', p1i >= 6.5_dp)
    call report('refine tanh(b=1.2) 16->24: wall-row order >= 4.0', p1w >= 4.0_dp)
    call report('refine tanh(b=1.2) 24->32: interior order >= 6.5', p2i >= 6.5_dp)
    call report('refine tanh(b=1.2) 24->32: wall-row order >= 4.0', p2w >= 4.0_dp)
    if (.not. (p1i >= 6.5_dp .and. p1w >= 4.0_dp .and. p2i >= 6.5_dp .and. p2w >= 4.0_dp)) &
       print '(a,4f7.2)', '   [refine] measured orders p1i/p1w/p2i/p2w =', p1i, p1w, p2i, p2w
    call d2_err_tanh(48_ip, 2.9_dp, e48i, e48w)
    call report('strong stretch (b=2.9, n=48): interior err <= 3e-6', &
                e48i == e48i .and. e48i <= 3.0e-6_dp)
    call report('strong stretch (b=2.9, n=48): wall rows <= 1e-8 (roundoff envelope)', &
                e48w == e48w .and. e48w <= 1.0e-8_dp)
  end subroutine refine_block

  !-- no-pivot banded LU factor residual at PRODUCTION shape.
  !   Model + HARDWARE: element growth <= 1 and the
  !   worst residual is at the non-diagonally-dominant mu->0 corner, where the
  !   floor is eps*|A|-scaled by the natural grid's 1/dy_wall^2 ~ 2e7 D2
  !   entries: HW measured 5.63e-12 (natural-481), model 1.25e-11 (natural-
  !   class twin); mu >= 1.84e5 sits at 1e-14..1e-16. No-pivot remains benign
  !   (growth ~1) -- the residual is roundoff-through-a-stiff-operator, not
  !   instability. Gate 1e-10 (~18x HW floor; the first-draft 1e-12 gate was
  !   calibrated on a 129-node model grid -- too coarse an anchor). Runtime
  !   coverage of the production solves = the rd2 galerkin/interior canaries;
  !   the GPU factor kernels are bit-gated against this host path.
  subroutine facres_block()
    integer(ip), parameter :: nyg = 481
    real(dp) :: yg(nyg), wg(nyg)
    real(dp) :: d1b(2*kl+1,nyg), d2b(2*kl+1,nyg), ab(2*kl+1,nyg)
    real(dp) :: fb(nyg), xs(nyg), rr(nyg), rmax, fmax, rel, muv(3)
    logical :: have_w, ok, okk
    character(len=160) :: msg
    integer(ip) :: i, im
    call ygrid_read('examples/for_testing/ygrid_Re2000_natural481.dat', nyg, &
                    yg, wg, have_w, ok, msg)
    if (.not. ok) then
       call report('factor residual: committed ygrid reads', .false.);  return
    end if
    call fd_build_bands(nyg, kl, yg, d1b, d2b)
    muv = [0.0_dp, 1.84e5_dp, 6.5e8_dp]     ! mu->0 corner .. production 1/alpha_2
    okk = .true.
    do im = 1, 3
       call fd_helm_dirichlet(nyg, kl, d2b, muv(im), ab)
       do i = 1, nyg
          fb(i) = sin(1.7_dp*yg(i)) + 0.3_dp*cos(4.1_dp*yg(i))
       end do
       fb(1) = 0.0_dp;  fb(nyg) = 0.0_dp
       xs = fb
       call fd_band_factor(nyg, kl, ab, ok)
       if (.not. ok) then
          okk = .false.;  exit
       end if
       call fd_band_solve(nyg, kl, ab, xs)
       call fd_helm_dirichlet(nyg, kl, d2b, muv(im), ab)   ! unfactored operator
       call fd_apply_band(nyg, kl, ab, xs, rr)
       rmax = 0.0_dp;  fmax = 0.0_dp
       do i = 1, nyg
          rmax = max(rmax, abs(rr(i) - fb(i)))
          fmax = max(fmax, abs(fb(i)))
       end do
       rel = rmax/max(fmax, tiny(1.0_dp))
       if (.not. (rel == rel .and. rel <= 1.0e-10_dp)) then
          okk = .false.
          print '(a,es9.2,a,es10.2)', '   [facres] mu=', muv(im), '  rel residual=', rel
       end if
    end do
    call report('no-pivot factor residual (natural 481, mu 0..6.5e8) <= 1e-10', okk)
  end subroutine facres_block

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. okk) allok = .false.
  end subroutine report

end program test_fdops
