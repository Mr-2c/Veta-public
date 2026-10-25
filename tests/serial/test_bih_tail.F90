!===============================================================================
! test_bih_tail.F90  --  pin the polynomial degree of the S4 (D^4) row tail.
!
! For a fixed low row k0, gather the same-parity entries S4(k0, k0), S4(k0,k0+2),
! ... across all columns, and form successive forward finite differences
! Delta^1 .. Delta^5. The lowest order d at which Delta^d collapses to round-off
! in the FAR tail is the degree+1; i.e. the tail is a polynomial of degree d-1 in
! the column index. That degree sets how many differencings (or running partial
! sums) are needed to band M = A4 S4 + A2 S2 + A0 B for an O(n) solve.
! (S2 and B are already banded, so M's tail equals -alpha times S4's tail.)
!
! Pure CPU. Build:
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran -c cheb_bih.F90
!   nvfortran -c test_bih_tail.F90
!   nvfortran -o test_bih_tail test_bih_tail.o cheb_bih.o cheb_shen.o types.o
!   ./test_bih_tail
!===============================================================================
program test_bih_tail
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,     only: dp, ip
  use cheb_shen, only: cheb_d2, wdot
  use cheb_bih,  only: bih_basis
  implicit none
  integer(ip), parameter :: n = 28
  integer(ip) :: md, k, j
  real(dp) :: Psi(0:n,0:n-4), D2(0:n,0:n-4), D4(0:n,0:n-4)
  real(dp) :: S4(0:n-4,0:n-4)

  md = n-4
  do k = 0, md
     call bih_basis(n, k, Psi(:,k))
     call cheb_d2(n, Psi(:,k), D2(:,k))
     call cheb_d2(n, D2(:,k),  D4(:,k))
  end do
  do j = 0, md
     do k = 0, md
        S4(k,j) = wdot(n, D4(:,j), Psi(:,k))
     end do
  end do

  write(*,'(a,i0,a,i0,a)') 'n=',n,'  (dof=',n-3,'),  S4 (D^4) row-tail finite differences'
  call tail(0_ip, md, S4)
  call tail(1_ip, md, S4)
  call tail(2_ip, md, S4)

contains

  subroutine tail(k0, md, X)
    integer(ip), intent(in) :: k0, md
    real(dp),    intent(in) :: X(0:md,0:md)
    real(dp) :: seq(0:64), dif(0:64)
    integer(ip) :: np, i, d

    np = 0
    do j = k0, md, 2
       seq(np) = X(k0,j); np = np + 1
    end do

    write(*,'(/,a,i0,a,i0,a)') '======== row k=',k0,'   (',np,' same-parity entries) ========'
    write(*,'(a)') ' seq   :'
    write(*,'(4es17.8)') (seq(i), i=0,np-1)
    dif(0:np-1) = seq(0:np-1)
    do d = 1, 5
       do i = 0, np-1-d
          dif(i) = dif(i+1) - dif(i)
       end do
       if (np-1-d >= 0) then
          write(*,'(a,i0,a)') ' Delta^',d,' :'
          write(*,'(4es17.8)') (dif(i), i=0,np-1-d)
       end if
    end do
  end subroutine tail

end program test_bih_tail
