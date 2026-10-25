!===============================================================================
! test_bih_structure.F90  --  CPU diagnostic of the biharmonic operator.
!
! Reproduces cheb_bih's constituent matrices in the clamped Shen basis
! psi_k = T_k + a_k T_{k+2} + b_k T_{k+4}:
!     B (k,j) = (psi_j,        psi_k)_w
!     S2(k,j) = (D^2 psi_j,    psi_k)_w
!     S4(k,j) = (D^4 psi_j,    psi_k)_w
! and the combined operator M = A4*S4 + A2*S2 + A0*B with
!     A4 = -alpha,  A2 = 1 + 2 alpha k2,  A0 = -(k2 + alpha k2^2),
! at a sample (k2, alpha). We need to know, BEFORE writing a banded solver:
!   * the lower bandwidth of M (how many subdiagonals to eliminate),
!   * that M is parity-decoupled (k-j even only),
!   * whether the UPPER tail of each row is constant in the column index j
!     (=> one differencing bands it, as for the Helmholtz stiffness) or carries
!     a j-dependence (=> needs a second differencing).
!
! For each matrix it prints, per row k, the entries on offsets -4,-2,0,+2,+4,
! +6,+8 (same parity), so the tail pattern is visible directly, plus global
! checks that nothing lives beyond |k-j|=4 below-diagonal or on odd offsets.
!
! Pure CPU. Build:
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran -c cheb_bih.F90
!   nvfortran -c test_bih_structure.F90
!   nvfortran -o test_bih_structure test_bih_structure.o cheb_bih.o cheb_shen.o types.o
!   ./test_bih_structure
!===============================================================================
program test_bih_structure
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,     only: dp, ip
  use cheb_shen, only: cheb_d2, wdot
  use cheb_bih,  only: bih_basis
  implicit none
  integer(ip) :: nn(2) = [14, 18], c
  do c = 1, 2
     call analyze(nn(c))
  end do
contains

  subroutine analyze(n)
    integer(ip), intent(in) :: n
    integer(ip) :: md, k, j
    real(dp) :: Psi(0:n,0:n-4), D2(0:n,0:n-4), D4(0:n,0:n-4)
    real(dp) :: B(0:n-4,0:n-4), S2(0:n-4,0:n-4), S4(0:n-4,0:n-4), M(0:n-4,0:n-4)
    real(dp) :: k2, alpha, A4, A2, A0
    md = n-4
    do k = 0, md
       call bih_basis(n, k, Psi(:,k))
       call cheb_d2(n, Psi(:,k), D2(:,k))
       call cheb_d2(n, D2(:,k),  D4(:,k))
    end do
    do j = 0, md
       do k = 0, md
          B (k,j) = wdot(n, Psi(:,j), Psi(:,k))
          S2(k,j) = wdot(n, D2(:,j),  Psi(:,k))
          S4(k,j) = wdot(n, D4(:,j),  Psi(:,k))
       end do
    end do
    k2 = 4.0_dp; alpha = 0.02_dp
    A4 = -alpha; A2 = 1.0_dp + 2.0_dp*alpha*k2; A0 = -(k2 + alpha*k2*k2)
    M = A4*S4 + A2*S2 + A0*B

    write(*,'(/,a,i0,a,i0,a,f0.3,a,f0.4,a)') &
         '================ n=',n,'  (dof=',n-3,')   k2=',k2,'  alpha=',alpha,' ================'
    call report_mat('B  (mass)        ', md, B)
    call report_mat('S2 (D^2 stiff)   ', md, S2)
    call report_mat('S4 (D^4 stiff)   ', md, S4)
    call report_mat('M  (A4S4+A2S2+A0B)', md, M)
  end subroutine analyze

  subroutine report_mat(name, md, X)
    character(*), intent(in) :: name
    integer(ip),  intent(in) :: md
    real(dp),     intent(in) :: X(0:md,0:md)
    real(dp) :: lowmax, parmax, tmin, tmax, spr
    integer(ip) :: k, j, ntail
    real(dp), parameter :: tol = 1.0e-10_dp

    lowmax = 0.0_dp; parmax = 0.0_dp
    do j = 0, md
       do k = 0, md
          if (mod(abs(k-j),2) == 1) parmax = max(parmax, abs(X(k,j)))
          if (k-j >= 6)             lowmax = max(lowmax, abs(X(k,j)))   ! >2 subdiagonals
       end do
    end do

    write(*,'(/,a,a)') ' --- ', name
    write(*,'(a,es10.2,a,es10.2)') '   max|entry| (k-j)>=6 below: ', lowmax, &
                                   '     max|entry| odd offset: ', parmax
    write(*,'(a)') '   row k :     [-4]        [-2]         [0]         [+2]         [+4]         [+6]         [+8]   | deeptail spread'
    do k = 0, min(5_ip, md)
       ! deep-tail spread over same-parity columns j >= k+4
       tmin = huge(1.0_dp); tmax = -huge(1.0_dp); ntail = 0
       do j = k+4, md, 2
          tmin = min(tmin, X(k,j)); tmax = max(tmax, X(k,j)); ntail = ntail + 1
       end do
       if (ntail > 0) then; spr = tmax - tmin; else; spr = 0.0_dp; end if
       write(*,'(a,i3,a,7es13.5,a,es11.3)') '   k=',k,' :', &
            gv(md,X,k,k-4), gv(md,X,k,k-2), gv(md,X,k,k), gv(md,X,k,k+2), &
            gv(md,X,k,k+4), gv(md,X,k,k+6), gv(md,X,k,k+8), '  | ', spr
    end do
  end subroutine report_mat

  pure function gv(md, X, k, j) result(v)   ! safe fetch, 0 outside range
    integer(ip), intent(in) :: md, k, j
    real(dp),    intent(in) :: X(0:md,0:md)
    real(dp) :: v
    if (j < 0 .or. j > md) then
       v = 0.0_dp
    else
       v = X(k,j)
    end if
  end function gv

end program test_bih_structure
