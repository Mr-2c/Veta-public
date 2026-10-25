!===============================================================================
! test_fft3d.F90  --  GPU validation of strided 3D C2C transforms.
!
! Build & run (ON A GPU NODE):
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -cudalib=cufft -c transforms_mod.F90
!   nvfortran -cuda -cudalib=cufft -c fft3d_mod.F90
!   nvfortran -cuda -cudalib=cufft test_fft3d.F90 \
!             fft3d_mod.o transforms_mod.o types.o -o test_fft3d
!   ./test_fft3d
!
! Tests (n1,n2,n3 deliberately distinct so a wrong stride/axis is exposed):
!   1-3. Per-axis round-trip: fwd then inv along axis a, /n_a = identity.
!   4.   Full 3D round-trip: fwd along all three axes, inv along all three,
!        /(n1*n2*n3) = identity.
!===============================================================================

program test_fft3d
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,     only: tp, ip
  use transforms_mod, only: tf_destroy
  use fft3d_mod, only: plan_axis_c2c, exec_axis_c2c, CUFFT_FORWARD, CUFFT_INVERSE
  use cudafor
  implicit none

  integer(ip), parameter :: n1 = 4, n2 = 6, n3 = 8
  real(real64) :: tol
  logical :: all_ok = .true.
  integer :: istat

  complex(tp) :: Ah(n1,n2,n3), Rh(n1,n2,n3)
  real(tp)    :: re(n1,n2,n3), im(n1,n2,n3)
  complex(tp), device :: A(n1,n2,n3), B(n1,n2,n3), C(n1,n2,n3)

  tol = merge(1.0e-10_real64, 1.0e-4_real64, tp == real64)

  call random_number(re);  call random_number(im)
  Ah = cmplx(re, im, tp)

  call axis_roundtrip(1_ip, n1)
  call axis_roundtrip(2_ip, n2)
  call axis_roundtrip(3_ip, n3)
  call full_roundtrip()

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine axis_roundtrip(axis, nlen)
    integer(ip), intent(in) :: axis, nlen
    integer :: plan
    character(len=24) :: nm
    A = Ah
    call plan_axis_c2c(plan, axis, n1, n2, n3)
    call exec_axis_c2c(plan, axis, n1, n2, n3, A, B, CUFFT_FORWARD)
    call exec_axis_c2c(plan, axis, n1, n2, n3, B, C, CUFFT_INVERSE)
    istat = cudaDeviceSynchronize()
    Rh = C
    Rh = Rh / real(nlen, tp)
    call tf_destroy(plan)
    write(nm,'(a,i0,a)') 'axis ', axis, ' round-trip'
    call check(trim(nm), real(maxval(abs(Rh - Ah)), real64), tol)
  end subroutine axis_roundtrip

  subroutine full_roundtrip()
    integer :: p1, p2, p3
    A = Ah
    call plan_axis_c2c(p1, 1_ip, n1, n2, n3)
    call plan_axis_c2c(p2, 2_ip, n1, n2, n3)
    call plan_axis_c2c(p3, 3_ip, n1, n2, n3)
    call exec_axis_c2c(p1, 1_ip, n1, n2, n3, A, B, CUFFT_FORWARD)
    call exec_axis_c2c(p2, 2_ip, n1, n2, n3, B, A, CUFFT_FORWARD)
    call exec_axis_c2c(p3, 3_ip, n1, n2, n3, A, B, CUFFT_FORWARD)
    call exec_axis_c2c(p1, 1_ip, n1, n2, n3, B, A, CUFFT_INVERSE)
    call exec_axis_c2c(p2, 2_ip, n1, n2, n3, A, B, CUFFT_INVERSE)
    call exec_axis_c2c(p3, 3_ip, n1, n2, n3, B, A, CUFFT_INVERSE)
    istat = cudaDeviceSynchronize()
    Rh = A
    Rh = Rh / real(n1*n2*n3, tp)
    call tf_destroy(p1);  call tf_destroy(p2);  call tf_destroy(p3)
    call check('full 3D round-trip', real(maxval(abs(Rh - Ah)), real64), tol)
  end subroutine full_roundtrip

  subroutine check(name, e, tolerance)
    character(*), intent(in) :: name
    real(real64), intent(in) :: e, tolerance
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', e <= tolerance)
    print '(a,a,a,es12.4,a,es10.2,a)', '[',tag,'] '//name//'  err=', e, '  tol=', tolerance
    if (e > tolerance) all_ok = .false.
  end subroutine check

end program test_fft3d
