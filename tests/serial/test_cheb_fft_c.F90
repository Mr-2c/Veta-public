!===============================================================================
! test_cheb_fft_c.F90  --  GPU validation of the COMPLEX Chebyshev transform.
!
! Build & run (ON A GPU NODE):
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -c cheb_shen.F90
!   nvfortran -cuda -cudalib=cufft -c transforms_mod.F90
!   nvfortran -cuda -cudalib=cufft -c cheb_fft_mod.F90
!   nvfortran -cuda -cudalib=cufft test_cheb_fft_c.F90 \
!             cheb_fft_mod.o transforms_mod.o cheb_shen.o types.o -o test_cheb_fft_c
!   ./test_cheb_fft_c
!
! Strategy: a complex profile is v = f1 + i f2. The cosine kernel is real, so
! the trusted REAL transform applied to f1 and f2 separately must reproduce the
! real and imaginary parts of the COMPLEX transform. That ties the new path to
! code we already validated to machine precision.
!   1. Re/Im of cheb_fft_c_fwd  vs  cheb_fft_fwd(f1), cheb_fft_fwd(f2).
!   2. Complex round-trip  cheb_fft_c_inv(cheb_fft_c_fwd(v)) = v.
!===============================================================================

program test_cheb_fft_c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,        only: tp, dp, ip
  use cheb_shen,    only: glc_nodes
  use transforms_mod, only: tf_r2c_plan, tf_c2c_plan, tf_destroy
  use cheb_fft_mod, only: cheb_fft_fwd, cheb_fft_c_fwd, cheb_fft_c_inv
  use cudafor
  implicit none

  integer(ip), parameter :: n = 16, batch = 4
  real(real64) :: tol
  logical :: all_ok = .true.
  integer :: istat, planR, planC

  real(dp) :: x(0:n)                              ! glc_nodes (cheb_shen) is dp
  real(tp) :: f1(0:n,batch), f2(0:n,batch)
  complex(tp) :: vh(0:n,batch), ach(0:n,batch), vrt(0:n,batch)
  real(tp) :: ar1(0:n,batch), ar2(0:n,batch)
  real(tp),    device :: r1d(0:n,batch), r2d(0:n,batch), a1d(0:n,batch), a2d(0:n,batch)
  complex(tp), device :: vcd(0:n,batch), acd(0:n,batch)
  integer(ip) :: j, col
  real(real64) :: err_re, err_im

  tol = merge(1.0e-10_real64, 1.0e-4_real64, tp == real64)

  call glc_nodes(n, x)
  do col = 1, batch
     do j = 0, n
        f1(j,col) = sample(col,        x(j))      ! real part
        f2(j,col) = sample(col+batch,  x(j))      ! imaginary part (different funcs)
     end do
  end do
  vh = cmplx(f1, f2, tp)

  call tf_r2c_plan(planR, 2_ip*n, batch)          ! real transform (for f1,f2)
  call tf_c2c_plan(planC, 2_ip*n, batch)          ! complex transform

  ! 1. complex forward, then compare re/im to real forward of f1,f2
  vcd = vh
  call cheb_fft_c_fwd(planC, n, batch, vcd, acd)
  r1d = f1;  r2d = f2
  call cheb_fft_fwd(planR, n, batch, r1d, a1d)
  call cheb_fft_fwd(planR, n, batch, r2d, a2d)
  istat = cudaDeviceSynchronize()
  ach = acd;  ar1 = a1d;  ar2 = a2d
  err_re = real(maxval(abs(real(ach,tp)  - ar1)), real64)
  err_im = real(maxval(abs(aimag(ach)    - ar2)), real64)
  call check('complex fwd Re vs real fwd(f1)', err_re, tol)
  call check('complex fwd Im vs real fwd(f2)', err_im, tol)

  ! 2. complex round-trip
  vcd = vh
  call cheb_fft_c_fwd(planC, n, batch, vcd, acd)
  call cheb_fft_c_inv(planC, n, batch, acd, vcd)
  istat = cudaDeviceSynchronize()
  vrt = vcd
  call check('complex Chebyshev round-trip', real(maxval(abs(vrt - vh)), real64), tol)

  call tf_destroy(planR);  call tf_destroy(planC)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  pure function sample(id, xx) result(f)
    integer(ip), intent(in) :: id
    real(dp),    intent(in) :: xx
    real(dp) :: f
    select case (id)
    case (1);  f = exp(xx)
    case (2);  f = cos(4.0_dp*xx)
    case (3);  f = 1.0_dp/(2.0_dp + xx)
    case (4);  f = sin(2.0_dp*xx)
    case (5);  f = xx*xx*xx
    case (6);  f = cos(3.0_dp*xx)*exp(0.5_dp*xx)
    case (7);  f = 1.0_dp/(1.5_dp - xx)
    case default; f = sin(5.0_dp*xx)
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

end program test_cheb_fft_c
