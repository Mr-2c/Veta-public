!===============================================================================
! test_yops.F90 -- unit gate for the wall-normal operator facade.
!
! Host-only (no GPU work). Checks, at machine precision:
!   (1) yop_wall_deriv unit responses: a = e_j -> Du(+1) = j^2 exactly and
!       Du(-1) = (-1)^(j-1) j^2 exactly (single-term sums, bit-exact).
!   (2) The rewired consumers on the laminar parabola u = 1.5(1-y^2)
!       (coeffs 0.75 T0 - 0.75 T2): mean flux = 2 (=> U_b = 1), and
!       stats_mod wall_shear = 3 (du/dy(+-1) = -+3) THROUGH the facade path.
!   (3) yop_quad_w (Clenshaw-Curtis on GLC nodes), BOTH parities of n:
!       sum w = 2, sum w y = 0, sum w y^2 = 2/3, and the grid-space integral of
!       a general degree-n polynomial matches the exact coefficient-space
!       weights (yop_int_tj) to round-off.
!   (4) yop_l2int: |T_0|^2 -> 2, |T_1|^2 -> 2/3, and a general complex series
!       (degree <= n/2, so |f|^2 has degree <= n) matches the quadrature
!       integral of |f(y_j)|^2 -- coefficient space vs grid space agree.
!
! Build/run: auto-discovered by run_all_tests.sh (host code; -cuda flag benign).
!===============================================================================
program test_yops
  use kinds,     only: dp, ip
  use yops_mod,  only: yop_wall_deriv, yop_mean_flux, yop_int_tj, yop_l2int, &
                       yop_quad_w, YOPS_BACKEND, YOPS_YGRID
  use stats_mod, only: wall_shear
  implicit none
  real(dp), parameter :: PI = 3.141592653589793238462643_dp
  integer(ip) :: j, k, np, icase
  real(dp)    :: a(0:40), dlo, dhi, ref
  real(dp)    :: par(0:2)
  real(dp), allocatable :: w(:), yg(:), ac(:), fg(:)
  complex(dp), allocatable :: cc(:)
  real(dp)    :: s0, s1, s2, gint, cint, fj, scale
  logical     :: ok, allok

  allok = .true.
  print '(a,a,a,a)', '[yops] backend = ', YOPS_BACKEND, '  ygrid = ', YOPS_YGRID

  ! (1) wall-derivative unit responses (exact) --------------------------------
  ok = .true.
  do j = 0, 12
     a(0:12) = 0.0_dp;  a(j) = 1.0_dp
     call yop_wall_deriv(12_ip, a(0:12), dlo, dhi)
     ref = real(j*j, dp)
     if (dhi /= ref) ok = .false.
     if (dlo /= ref*real((-1)**(j-1), dp) .and. j > 0) ok = .false.
     if (j == 0 .and. dlo /= 0.0_dp) ok = .false.
  end do
  call report('wall-deriv unit responses T_j: Du(+1)=j^2, Du(-1)=(-1)^(j-1) j^2 (exact)', ok)

  ! (2) laminar parabola through the REWIRED consumers ------------------------
  par = [ 0.75_dp, 0.0_dp, -0.75_dp ]          ! u = 1.5 (1 - y^2)
  ok = abs(yop_mean_flux(2_ip, par) - 2.0_dp) <= 1.0e-14_dp     ! => U_b = 1
  call report('parabola: yop_mean_flux = 2 (U_b = 1)', ok)
  ok = abs(wall_shear(2_ip, par) - 3.0_dp) <= 1.0e-13_dp        ! |du/dy|_wall = 3
  call report('parabola: stats_mod wall_shear = 3 via the facade', ok)

  ! (3) quadrature weights, both parities -------------------------------------
  do icase = 1, 2
     np = merge(32_ip, 17_ip, icase == 1)
     allocate(w(0:np), yg(0:np), ac(0:np), fg(0:np))
     call yop_quad_w(np, w)
     do j = 0, np
        yg(j) = cos(PI*real(j,dp)/real(np,dp))
     end do
     s0 = sum(w);  s1 = sum(w*yg);  s2 = sum(w*yg*yg)
     ok = (abs(s0 - 2.0_dp) <= 1.0e-13_dp) .and. (abs(s1) <= 1.0e-13_dp) .and. &
          (abs(s2 - 2.0_dp/3.0_dp) <= 1.0e-13_dp)
     ! general degree-np polynomial: grid quadrature == exact coefficient weights
     do k = 0, np
        ac(k) = real((-1)**k, dp)/real(1 + k, dp)
     end do
     do j = 0, np
        fj = 0.0_dp
        do k = 0, np
           fj = fj + ac(k)*cos(real(k,dp)*PI*real(j,dp)/real(np,dp))   ! T_k(y_j)
        end do
        fg(j) = fj
     end do
     gint = sum(w*fg)
     cint = 0.0_dp
     do k = 0, np
        cint = cint + ac(k)*yop_int_tj(k)
     end do
     ok = ok .and. (abs(gint - cint) <= 1.0e-12_dp*max(1.0_dp, abs(cint)))
     call report(merge('quad weights n=32 (even): moments + degree-n poly == coeff integral', &
                       'quad weights n=17 (odd) : moments + degree-n poly == coeff integral', &
                       icase == 1), ok)
     deallocate(w, yg, ac, fg)
  end do

  ! (4) L2 integral: knowns + coefficient-vs-grid agreement --------------------
  np = 32_ip
  allocate(cc(0:np), w(0:np), fg(0:np))
  cc = (0.0_dp, 0.0_dp);  cc(0) = (1.0_dp, 0.0_dp)
  ok = abs(yop_l2int(np, cc) - 2.0_dp) <= 1.0e-14_dp            ! int |T0|^2 = 2
  cc = (0.0_dp, 0.0_dp);  cc(1) = (1.0_dp, 0.0_dp)
  ok = ok .and. abs(yop_l2int(np, cc) - 2.0_dp/3.0_dp) <= 1.0e-14_dp  ! int y^2
  ! general complex series of degree np/2 -> |f|^2 has degree <= np: the grid
  ! quadrature of |f(y_j)|^2 must equal the exact coefficient-space L2 integral
  cc = (0.0_dp, 0.0_dp)
  do k = 0, np/2 - 2
     cc(k) = cmplx(1.0_dp/real(1+k,dp), real((-1)**k,dp)/real(2+k,dp), dp)
  end do
  call yop_quad_w(np, w)
  do j = 0, np
     s1 = 0.0_dp;  s2 = 0.0_dp
     do k = 0, np
        fj = cos(real(k,dp)*PI*real(j,dp)/real(np,dp))
        s1 = s1 + real(cc(k))*fj
        s2 = s2 + aimag(cc(k))*fj
     end do
     fg(j) = s1*s1 + s2*s2                                       ! |f(y_j)|^2
  end do
  gint  = sum(w*fg)
  cint  = yop_l2int(np, cc)
  scale = max(1.0_dp, abs(cint))
  ok = ok .and. (abs(gint - cint) <= 1.0e-12_dp*scale)
  call report('yop_l2int: T0/T1 knowns + complex series == quadrature of |f|^2', ok)
  deallocate(cc, w, fg)

  if (allok) then; print '(a)', 'ALL TESTS PASSED'
  else;            print '(a)', 'SOME TESTS FAILED'; end if
  if (.not. allok) error stop 1

contains
  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. okk) allok = .false.
  end subroutine report
end program test_yops
