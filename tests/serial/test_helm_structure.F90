!===============================================================================
! test_helm_structure.F90  --  CPU diagnostic of the Shen Helmholtz matrix.
!
! Goal: confirm, straight from the validated cheb_shen assembly, the exact
! sparsity we intend to exploit for an O(n) banded GPU solve. We isolate the
! stiffness S and mass B by assembling M = S + mu*B at two values of mu:
!     S = M(mu=0),      B = M(mu=1) - M(mu=0).
!
! It checks / prints:
!   (1) B is nonzero only on diagonals 0 and +-2 (tridiagonal per parity).
!   (2) S is zero in the lower triangle (k>j)         -> upper-triangular.
!   (3) S is zero for (k-j) odd                        -> parity-decoupled.
!   (4) THE OPEN QUESTION: for each row k, are the off-diagonal entries
!       S(k, k+2), S(k, k+4), ... constant in the column index? If the spread
!       (max-min) is ~0, the "constant-tail" partial-sum structure holds and
!       row-differencing collapses M to a narrow band. Printed per row.
!   (5) Diagonal of S, and the value of the constant tail per row (if constant).
!
! Pure CPU. Build:
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran -c test_helm_structure.F90
!   nvfortran -o test_helm_structure test_helm_structure.o cheb_shen.o types.o
!   ./test_helm_structure
!===============================================================================
program test_helm_structure
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,     only: dp, ip
  use cheb_shen, only: assemble_helmholtz
  implicit none

  integer(ip) :: nn(2) = [12, 16], c
  do c = 1, 2
     call analyze(nn(c))
  end do

contains

  subroutine analyze(n)
    integer(ip), intent(in) :: n
    real(dp) :: S(0:n-2,0:n-2), B(0:n-2,0:n-2), M0(0:n-2,0:n-2), M1(0:n-2,0:n-2)
    real(dp), parameter :: tol = 1.0e-11_dp
    integer(ip) :: k, j, m, ntail
    real(dp) :: vmax, tmin, tmax, tval

    call assemble_helmholtz(n, 0.0_dp, M0)   ! S
    call assemble_helmholtz(n, 1.0_dp, M1)   ! S + B
    S = M0
    B = M1 - M0
    m = n-2

    write(*,'(/,a,i0,a,i0,a)') '================ n=',n,'  (dof = ',n-1,') ================'

    ! (1) B bandwidth: only diagonals 0, +-2
    vmax = 0.0_dp
    do j = 0, m
       do k = 0, m
          if (abs(k-j) /= 0 .and. abs(k-j) /= 2) vmax = max(vmax, abs(B(k,j)))
       end do
    end do
    call line('B nonzero only on diag 0,+-2  (max stray entry)', vmax, tol)

    ! (2) S lower triangle (k>j) zero
    vmax = 0.0_dp
    do j = 0, m
       do k = j+1, m
          vmax = max(vmax, abs(S(k,j)))
       end do
    end do
    call line('S lower triangle (k>j) is zero (max)        ', vmax, tol)

    ! (3) S zero for (k-j) odd
    vmax = 0.0_dp
    do j = 0, m
       do k = 0, m
          if (mod(abs(k-j),2) == 1) vmax = max(vmax, abs(S(k,j)))
       end do
    end do
    call line('S parity-decoupled: (k-j) odd is zero (max) ', vmax, tol)

    ! (4)+(5) constant-tail test per row, plus diagonal
    write(*,'(a)') '  row k :  S(k,k) [diag]   tail S(k,k+2..) spread   tail value'
    do k = 0, m
       tmin =  huge(1.0_dp); tmax = -huge(1.0_dp); ntail = 0; tval = 0.0_dp
       do j = k+2, m, 2
          tmin = min(tmin, S(k,j)); tmax = max(tmax, S(k,j))
          ntail = ntail + 1; tval = S(k,j)
       end do
       if (ntail == 0) then
          write(*,'(a,i3,a,es14.6,a)') '   k=',k,' : ',S(k,k),'      (no tail)'
       else
          write(*,'(a,i3,a,es14.6,a,es12.4,a,es14.6)') &
               '   k=',k,' : ',S(k,k),'   ',tmax-tmin,'   ',tval
       end if
    end do

    ! also print B's three diagonals for the solve construction
    write(*,'(a)') '  B diagonals:  k : B(k,k)   B(k,k+2)'
    do k = 0, m
       if (k+2 <= m) then
          write(*,'(a,i3,a,es14.6,es16.6)') '   k=',k,' : ',B(k,k),B(k,k+2)
       else
          write(*,'(a,i3,a,es14.6)')        '   k=',k,' : ',B(k,k)
       end if
    end do
  end subroutine analyze

  subroutine line(name, v, tol)
    character(*), intent(in) :: name
    real(dp),     intent(in) :: v, tol
    character(6) :: tag
    if (v <= tol) then; tag = '[ OK ]'; else; tag = '[ ?? ]'; end if
    write(*,'(a,2x,a,2x,es12.4)') tag, name, v
  end subroutine line

end program test_helm_structure
