!===============================================================================
! test_cheb_bih.F90  --  CPU validation of the wall-normal v-equation solver.
!
! Build & run:
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran -c cheb_bih.F90
!   nvfortran test_cheb_bih.F90 cheb_bih.o cheb_shen.o types.o -o test_bih
!   ./test_bih
!
! Tests:
!   1. Basis BCs: psi_k(+-1) = 0 and psi_k'(+-1) = 0 to machine precision,
!      for k = 0..7 -- independent confirmation the clamped basis is correct.
!   2. Manufactured solution v = sin^2(pi x), which satisfies v = v' = 0 at
!      both walls. RHS = (A4 D^4 + A2 D^2 + A0) v is formed from the exact
!      recurrence; solving must recover v at the walls and interior. Run for
!      two (k^2, alpha) pairs. The biharmonic operator is more ill-conditioned
!      than the Helmholtz one (D^4 ~ N^8), so the tolerance is looser; the
!      printed error shows the true accuracy -- and is why this solve
!      is the one kept in FP64.
!===============================================================================

program test_cheb_bih
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,     only: dp, ip
  use cheb_shen, only: glc_nodes, cheb_forward, cheb_eval, cheb_deriv, cheb_d2
  use cheb_bih
  implicit none

  real(dp), parameter :: PI = acos(-1.0_dp)
  logical :: all_ok = .true.

  call test_basis_bcs()
  call test_bih_mms(4.0_dp,  0.05_dp)
  call test_bih_mms(16.0_dp, 0.02_dp)

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

  !-- 1. clamped boundary conditions of the basis ---------------------------
  subroutine test_basis_bcs()
    integer(ip), parameter :: n = 20
    real(dp) :: psi(0:n), dpsi(0:n)
    real(dp) :: err
    integer(ip) :: k
    err = 0.0_dp
    do k = 0, 7
       call bih_basis(n, k, psi)
       call cheb_deriv(n, psi, dpsi)
       err = max(err, abs(cheb_eval(n, psi,  1.0_dp)))   ! psi(+1)
       err = max(err, abs(cheb_eval(n, psi, -1.0_dp)))   ! psi(-1)
       err = max(err, abs(cheb_eval(n, dpsi, 1.0_dp)))   ! psi'(+1)
       err = max(err, abs(cheb_eval(n, dpsi,-1.0_dp)))   ! psi'(-1)
    end do
    call check('biharmonic basis: v = Dv = 0 at walls', err, 1.0e-11_dp)
  end subroutine test_basis_bcs

  !-- 2. manufactured solution v = sin^2(pi x) ------------------------------
  subroutine test_bih_mms(k2, alpha)
    real(dp), intent(in) :: k2, alpha
    integer(ip), parameter :: n = 48
    real(dp) :: x(0:n), vex(0:n), av(0:n), d2(0:n), d4(0:n), fcheb(0:n)
    real(dp) :: vhat(0:n-4), asol(0:n)
    real(dp) :: A4, A2, A0, xt, err
    integer(ip) :: j, np
    A4 = -alpha
    A2 = 1.0_dp + 2.0_dp*alpha*k2
    A0 = -(k2 + alpha*k2*k2)
    ! exact v on GLC -> Chebyshev coeffs
    call glc_nodes(n, x)
    do j = 0, n
       vex(j) = sin(PI*x(j))**2
    end do
    call cheb_forward(n, vex, av)
    ! true RHS = (A4 D^4 + A2 D^2 + A0) v
    call cheb_d2(n, av, d2)
    call cheb_d2(n, d2, d4)
    fcheb = A4*d4 + A2*d2 + A0*av
    ! solve and reconstruct
    call biharmonic_solve(n, k2, alpha, fcheb, vhat)
    call shen_bih_to_cheb(n, vhat, asol)
    err = 0.0_dp; np = 200
    do j = 0, np
       xt = -1.0_dp + 2.0_dp*real(j,dp)/real(np,dp)
       err = max(err, abs(cheb_eval(n, asol, xt) - sin(PI*xt)**2))
    end do
    call check('biharmonic vs sin^2(pi x)', err, 1.0e-7_dp)
  end subroutine test_bih_mms

end program test_cheb_bih
