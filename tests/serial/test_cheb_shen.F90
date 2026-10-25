!===============================================================================
! test_cheb_shen.F90  --  CPU validation of the wall-normal core.
!
! Build & run (no GPU, no deps):
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran test_cheb_shen.F90 cheb_shen.o types.o -o test_cheb
!   ./test_cheb
! (gfortran -cpp works too, if you prefer.)
!
! Three layered tests, weakest dependency first:
!   1. cheb_d2 exactness on T_3  (24x) -- validates the derivative recurrence.
!   2. cheb_forward o cheb_eval round-trip -- validates the GLC transform.
!   3. Helmholtz MMS with u = sin(pi x): u(+-1)=0 exactly. f = -u'' + mu u is
!      formed from the *independent* exact derivative, so a wrong stiffness or
!      mass matrix CANNOT pass -- this pins (S + mu B) to the true operator.
!===============================================================================

program test_cheb_shen
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,     only: dp, ip
  use cheb_shen
  implicit none

  real(dp), parameter :: PI = acos(-1.0_dp)
  logical :: all_ok = .true.

  call test_derivative()
  call test_transform()
  call test_helmholtz()

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine check(name, err, tol)
    character(*), intent(in) :: name
    real(dp),     intent(in) :: err, tol
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', err <= tol)
    print '(a,a,a,es12.4,a,es10.2,a)', '[',tag,'] '//name//'  err=', err, '  tol=', tol
    if (err > tol) all_ok = .false.
  end subroutine check

  !-- 1. second derivative of T_3 is 24 x = 24 T_1 --------------------------
  subroutine test_derivative()
    integer(ip), parameter :: n = 6
    real(dp) :: a(0:n), c(0:n), ex(0:n)
    a = 0.0_dp; a(3) = 1.0_dp          ! u = T_3
    call cheb_d2(n, a, c)
    ex = 0.0_dp; ex(1) = 24.0_dp       ! u'' = 24 T_1
    call check('cheb_d2(T_3) = 24 T_1', maxval(abs(c-ex)), 1.0e-11_dp)
  end subroutine test_derivative

  !-- 2. forward transform inverts evaluation on the GLC grid ---------------
  subroutine test_transform()
    integer(ip), parameter :: n = 24
    real(dp) :: a0(0:n), a1(0:n), x(0:n), f(0:n)
    integer(ip) :: j
    call random_number(a0)
    a0(n-3:n) = 0.0_dp                  ! band-limited so it is exactly recoverable
    call glc_nodes(n, x)
    do j = 0, n
       f(j) = cheb_eval(n, a0, x(j))
    end do
    call cheb_forward(n, f, a1)
    call check('cheb_forward o cheb_eval = I', maxval(abs(a1-a0)), 1.0e-11_dp)
  end subroutine test_transform

  !-- 3. Helmholtz MMS: -u'' + mu u = f, u = sin(pi x), u(+-1)=0 ------------
  subroutine test_helmholtz()
    integer(ip), parameter :: n = 48
    real(dp),    parameter :: mu = 10.0_dp
    real(dp) :: x(0:n), uex(0:n), au(0:n), d2(0:n), af(0:n)
    real(dp) :: uhat(0:n-2), asol(0:n)
    real(dp) :: xt, err
    integer(ip) :: j, np
    ! exact solution sampled on GLC -> Chebyshev coeffs (spectrally accurate)
    call glc_nodes(n, x)
    do j = 0, n
       uex(j) = sin(PI * x(j))
    end do
    call cheb_forward(n, uex, au)
    ! true RHS in coeff space: f = -u'' + mu u (cheb_d2 is exact)
    call cheb_d2(n, au, d2)
    af = -d2 + mu*au
    ! solve and reconstruct
    call helmholtz_solve(n, mu, af, uhat)
    call shen_to_cheb(n, uhat, asol)
    ! compare to the exact sin(pi x) on a fine grid
    err = 0.0_dp; np = 200
    do j = 0, np
       xt = -1.0_dp + 2.0_dp*real(j,dp)/real(np,dp)
       err = max(err, abs(cheb_eval(n, asol, xt) - sin(PI*xt)))
    end do
    call check('Helmholtz solve vs sin(pi x)', err, 1.0e-9_dp)
  end subroutine test_helmholtz

end program test_cheb_shen
