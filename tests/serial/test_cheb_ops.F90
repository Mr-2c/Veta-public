!===============================================================================
! test_cheb_ops.F90  --  GPU validation of the complex Chebyshev derivative.
!
! Build & run (ON A GPU NODE):
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -c cheb_shen.F90
!   nvfortran -cuda -c cheb_ops_mod.F90
!   nvfortran -cuda test_cheb_ops.F90 cheb_ops_mod.o cheb_shen.o types.o -o test_cheb_ops
!   ./test_cheb_ops
!
! A complex coefficient column is a + i*c. The real-kernel recurrence must give
! cheb_deriv(a) + i*cheb_deriv(c), so we compare the GPU complex result against
! the trusted CPU cheb_deriv applied to the real and imaginary parts.
!===============================================================================

program test_cheb_ops
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,        only: wp, ip, dp
  use cheb_shen,    only: glc_nodes, cheb_forward, cheb_deriv
  use cheb_ops_mod, only: cheb_deriv_c
  use cudafor
  implicit none

  integer(ip), parameter :: ny = 17, n = ny-1, batch = 100   ! batch>32 (multi-warp) exercises
                                                             ! the column-leading coalescing transpose
  real(real64) :: tol
  logical :: all_ok = .true.
  integer :: istat
  integer(ip) :: j, col

  real(wp) :: x(0:n), fr(0:n), fi(0:n)
  real(dp) :: are(0:n,batch), aim(0:n,batch), dre(0:n,batch), dim(0:n,batch)
  complex(wp) :: ah(ny,batch), bh(ny,batch)
  complex(wp), device :: ad(ny,batch), bd(ny,batch)
  real(real64) :: err_re, err_im

  tol = merge(1.0e-9_real64, 1.0e-3_real64, wp == real64)

  call glc_nodes(n, x)
  do col = 1, batch
     do j = 0, n
        fr(j) = sample(col,       x(j))
        fi(j) = sample(col+batch, x(j))
     end do
     call cheb_forward(n, fr, are(:,col))
     call cheb_forward(n, fi, aim(:,col))
     call cheb_deriv(n, are(:,col), dre(:,col))    ! CPU reference derivatives
     call cheb_deriv(n, aim(:,col), dim(:,col))
  end do

  do col = 1, batch
     do j = 1, ny
        ah(j,col) = cmplx(are(j-1,col), aim(j-1,col), wp)
     end do
  end do

  ad = ah
  call cheb_deriv_c(ny, batch, ad, bd)
  istat = cudaDeviceSynchronize()
  bh = bd

  err_re = 0.0_real64;  err_im = 0.0_real64
  do col = 1, batch
     do j = 1, ny
        err_re = max(err_re, abs(real(real(bh(j,col),wp),real64) - dre(j-1,col)))
        err_im = max(err_im, abs(real(aimag(bh(j,col)),  real64) - dim(j-1,col)))
     end do
  end do

  call check('cheb_deriv_c Re vs CPU cheb_deriv', err_re, tol)
  call check('cheb_deriv_c Im vs CPU cheb_deriv', err_im, tol)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  pure function sample(id, xx) result(f)
    integer(ip), intent(in) :: id
    real(wp),    intent(in) :: xx
    real(wp) :: f
    select case (id)
    case (1);  f = exp(xx)
    case (2);  f = cos(4.0_wp*xx)
    case (3);  f = 1.0_wp/(2.0_wp + xx)
    case (4);  f = sin(2.0_wp*xx)
    case (5);  f = xx*xx*xx - 2.0_wp*xx
    case (6);  f = cos(3.0_wp*xx)*exp(0.5_wp*xx)
    case (7);  f = 1.0_wp/(1.5_wp - xx)
    case default; f = sin(5.0_wp*xx)
    end select
  end function sample

  subroutine check(name, e, tolerance)
    character(*), intent(in) :: name
    real(real64), intent(in) :: e, tolerance
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', e <= tolerance)
    print '(a,a,a,es12.4,a,es10.2,a)', '[',tag,'] '//name//'  err=', e, '  tol=', tolerance
    if (e > tolerance) all_ok = .false.
  end subroutine check

end program test_cheb_ops
