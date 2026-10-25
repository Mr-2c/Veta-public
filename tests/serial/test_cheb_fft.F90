!===============================================================================
! test_cheb_fft.F90  --  GPU validation of the Chebyshev transform.
!
! Build & run (ON A GPU NODE):
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -c cheb_shen.F90
!   nvfortran -cuda -cudalib=cufft -c transforms_mod.F90
!   nvfortran -cuda -cudalib=cufft -c cheb_fft_mod.F90
!   nvfortran -cuda -cudalib=cufft test_cheb_fft.F90 \
!             cheb_fft_mod.o transforms_mod.o cheb_shen.o types.o -o test_cheb_fft
!   ./test_cheb_fft
!
! Tests (batched over `batch` columns of smooth functions):
!   1. GPU forward  vs CPU cheb_forward  (anchors the GPU path to trusted code).
!   2. GPU inverse  vs CPU cheb_eval     (series evaluated at the GLC points).
!   3. Round-trip inv(fwd(v)) = v.
!===============================================================================

program test_cheb_fft
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,        only: tp, dp, ip
  use cheb_shen,    only: glc_nodes, cheb_forward, cheb_eval
  use transforms_mod, only: tf_r2c_plan, tf_destroy
  use cheb_fft_mod, only: cheb_fft_fwd, cheb_fft_inv
  use cudafor
  implicit none

  integer(ip), parameter :: n = 16, batch = 4    ! degree N=16 (17 points)
  real(real64) :: tol
  logical :: all_ok = .true.
  integer :: istat, plan

  ! cheb_shen (the CPU reference) is dp -> nodes, the truth, and CPU outputs are
  ! dp; the GPU path (cheb_fft is tp) gets tp host mirrors, cast at the boundary.
  real(dp) :: x(0:n)
  real(dp) :: vh(0:n,batch)                       ! input truth + CPU fwd input
  real(dp) :: a_cpu(0:n,batch), v_cpu(0:n,batch)  ! CPU reference outputs
  real(tp) :: vt(0:n,batch), at(0:n,batch)        ! tp host mirrors of vd/ad
  real(tp), device :: vd(0:n,batch), ad(0:n,batch)
  integer(ip) :: j, col

  tol = merge(1.0e-10_real64, 1.0e-4_real64, tp == real64)

  call glc_nodes(n, x)
  do col = 1, batch
     do j = 0, n
        vh(j,col) = sample(col, x(j))
     end do
  end do

  call tf_r2c_plan(plan, 2_ip*n, batch)           ! R2C of length 2N, batched

  ! 1. forward GPU vs CPU
  vt = real(vh, tp)                                ! cast truth dp -> tp for the GPU
  vd = vt
  call cheb_fft_fwd(plan, n, batch, vd, ad)
  istat = cudaDeviceSynchronize()
  at = ad                                          ! GPU forward coeffs (tp)
  do col = 1, batch
     call cheb_forward(n, vh(:,col), a_cpu(:,col))
  end do
  call check('cheb_fft_fwd vs CPU cheb_forward', maxerr(at, a_cpu), tol)

  ! 2. inverse GPU vs CPU cheb_eval (use the CPU coeffs as input)
  at = real(a_cpu, tp)                             ! cast CPU coeffs dp -> tp
  ad = at
  call cheb_fft_inv(plan, n, batch, ad, vd)
  istat = cudaDeviceSynchronize()
  vt = vd                                          ! GPU inverse values (tp)
  do col = 1, batch
     do j = 0, n
        v_cpu(j,col) = cheb_eval(n, a_cpu(:,col), x(j))
     end do
  end do
  call check('cheb_fft_inv vs CPU cheb_eval', maxerr(vt, v_cpu), tol)

  ! 3. round-trip
  do col = 1, batch
     do j = 0, n
        vh(j,col) = sample(col, x(j))
     end do
  end do
  vt = real(vh, tp)
  vd = vt
  call cheb_fft_fwd(plan, n, batch, vd, ad)
  call cheb_fft_inv(plan, n, batch, ad, vd)
  istat = cudaDeviceSynchronize()
  vt = vd                                          ! GPU round-trip values (tp)
  call check('cheb_fft round-trip', maxerr(vt, vh), tol)

  call tf_destroy(plan)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  pure function sample(col, xx) result(f)
    integer(ip), intent(in) :: col
    real(dp),    intent(in) :: xx
    real(dp) :: f
    select case (col)
    case (1);  f = exp(xx)
    case (2);  f = cos(4.0_dp*xx)
    case (3);  f = 1.0_dp/(2.0_dp + xx)
    case default; f = sin(2.0_dp*xx)
    end select
  end function sample

  pure function maxerr(p, q) result(e)
    real(tp), intent(in) :: p(0:n,batch)            ! GPU result (tp)
    real(dp), intent(in) :: q(0:n,batch)            ! CPU reference (dp)
    real(real64) :: e
    e = real(maxval(abs(real(p,dp) - q)), real64)
  end function maxerr

  subroutine check(name, err, tolerance)
    character(*), intent(in) :: name
    real(real64), intent(in) :: err, tolerance
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', err <= tolerance)
    print '(a,a,a,es12.4,a,es10.2,a)', '[',tag,'] '//name//'  err=', err, '  tol=', tolerance
    if (err > tolerance) all_ok = .false.
  end subroutine check

end program test_cheb_fft
