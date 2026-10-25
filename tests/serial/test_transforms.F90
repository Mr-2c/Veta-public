!===============================================================================
! test_transforms.F90  --  GPU validation of transforms_mod (increment 4a).
!
! Build & run (ON A GPU COMPUTE NODE):
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -c cheb_shen.F90
!   nvfortran -cuda -c grid_mod.F90
!   nvfortran -cuda -cudalib=cufft -c transforms_mod.F90
!   nvfortran -cuda -cudalib=cufft test_transforms.F90 \
!             transforms_mod.o grid_mod.o cheb_shen.o types.o -o test_transforms
!   ./test_transforms
!
! Tests:
!   1. C2C round-trip:  inv(fwd(f))/nz = f.
!   2. R2C/C2R round-trip:  c2r(r2c(f))/nx = f.
!   3. C2C spectral d/dz via grid%kz (incl. a negative mode) = analytic.
!   4. R2C spectral d/dx via grid%kx = analytic (real derivative).
!
! Tests 3-4 are the important ones: they prove cuFFT's output ordering is the
! one grid_mod assumes. Tolerance adapts to tp so the same test works for an
! FP32 build.
!===============================================================================

program test_transforms
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,          only: tp, ip
  use types,          only: grid_t, params_t
  use grid_mod,       only: build_grid, destroy_grid
  use transforms_mod, only: tf_c2c_plan, tf_r2c_plan, tf_c2r_plan, tf_destroy, &
                            tf_c2c, tf_r2c, tf_c2r, CUFFT_FORWARD, CUFFT_INVERSE
  use cudafor
  implicit none

  real(real64), parameter :: PI = acos(-1.0_real64)
  real(real64) :: tol
  logical :: all_ok = .true.

  type(grid_t)   :: grid
  type(params_t) :: p
  integer(ip), parameter :: nx = 16, ny = 5, nz = 8
  integer(ip) :: nkx
  integer :: istat

  tol = merge(1.0e-10_real64, 1.0e-4_real64, tp == real64)
  nkx = nx/2 + 1

  p%lx = 2.0_real64*PI;  p%lz = 2.0_real64*PI
  call build_grid(grid, p, nx, ny, nz)

  call test_c2c_roundtrip()
  call test_r2c_roundtrip()
  call test_c2c_derivative()
  call test_r2c_derivative()

  call destroy_grid(grid)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine check(name, err, tolerance)
    character(*), intent(in) :: name
    real(real64), intent(in) :: err, tolerance
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', err <= tolerance)
    print '(a,a,a,es12.4,a,es10.2,a)', '[',tag,'] '//name//'  err=', err, '  tol=', tolerance
    if (err > tolerance) all_ok = .false.
  end subroutine check

  !-- 1. C2C round-trip -----------------------------------------------------
  subroutine test_c2c_roundtrip()
    complex(tp) :: fh(0:nz-1), gh(0:nz-1)
    complex(tp), device :: a(0:nz-1), b(0:nz-1)
    real(tp) :: re(0:nz-1), im(0:nz-1)
    integer :: plan
    call random_number(re);  call random_number(im)
    fh = cmplx(re, im, tp)
    a = fh
    call tf_c2c_plan(plan, nz, 1_ip)
    call tf_c2c(plan, a, b, CUFFT_FORWARD)
    call tf_c2c(plan, b, a, CUFFT_INVERSE)
    istat = cudaDeviceSynchronize()
    gh = a
    gh = gh / real(nz, tp)
    call tf_destroy(plan)
    call check('C2C round-trip', real(maxval(abs(gh - fh)), real64), tol)
  end subroutine test_c2c_roundtrip

  !-- 2. R2C / C2R round-trip ----------------------------------------------
  subroutine test_r2c_roundtrip()
    real(tp) :: fh(0:nx-1), gh(0:nx-1)
    real(tp),    device :: rd(0:nx-1)
    complex(tp), device :: cd(0:nkx-1)
    integer :: planf, planb
    call random_number(fh)
    rd = fh
    call tf_r2c_plan(planf, nx, 1_ip)
    call tf_c2r_plan(planb, nx, 1_ip)
    call tf_r2c(planf, rd, cd)
    call tf_c2r(planb, cd, rd)
    istat = cudaDeviceSynchronize()
    gh = rd
    gh = gh / real(nx, tp)
    call tf_destroy(planf);  call tf_destroy(planb)
    call check('R2C/C2R round-trip', real(maxval(abs(gh - fh)), real64), tol)
  end subroutine test_r2c_roundtrip

  !-- 3. C2C spectral d/dz via grid%kz --------------------------------------
  subroutine test_c2c_derivative()
    integer(ip), parameter :: nm = 3
    integer(ip) :: modes(nm) = [ 1_ip, 2_ip, -1_ip ]
    complex(tp) :: coeff(nm)
    complex(tp) :: fh(0:nz-1), ch(0:nz-1), dfh(0:nz-1)
    complex(tp), device :: a(0:nz-1), b(0:nz-1)
    real(tp) :: kzh(0:nz-1)
    complex(real64) :: analytic
    real(real64) :: z, err
    integer :: plan, q, j, m
    coeff = [ cmplx(1.0_tp,0.0_tp,tp), cmplx(0.5_tp,-0.3_tp,tp), cmplx(2.0_tp,0.0_tp,tp) ]
    kzh = grid%kz                                   ! D2H copy
    do q = 0, nz-1
       z = 2.0_real64*PI*real(q,real64)/real(nz,real64)
       fh(q) = (0.0_tp,0.0_tp)
       do m = 1, nm
          fh(q) = fh(q) + coeff(m)*exp(cmplx(0.0_tp, real(modes(m),tp)*real(z,tp), tp))
       end do
    end do
    a = fh
    call tf_c2c_plan(plan, nz, 1_ip)
    call tf_c2c(plan, a, b, CUFFT_FORWARD)
    istat = cudaDeviceSynchronize()
    ch = b                                          ! D2H
    do j = 0, nz-1
       if (j == nz/2) then
          ch(j) = (0.0_tp,0.0_tp)                   ! zero Nyquist for derivative
       else
          ch(j) = ch(j) * cmplx(0.0_tp, kzh(j), tp) ! multiply by i k
       end if
    end do
    b = ch                                          ! H2D
    call tf_c2c(plan, b, a, CUFFT_INVERSE)
    istat = cudaDeviceSynchronize()
    dfh = a
    dfh = dfh / real(nz, tp)
    call tf_destroy(plan)
    err = 0.0_real64
    do q = 0, nz-1
       z = 2.0_real64*PI*real(q,real64)/real(nz,real64)
       analytic = (0.0_real64,0.0_real64)
       do m = 1, nm
          analytic = analytic + cmplx(coeff(m),kind=real64) &
                     *cmplx(0.0_real64, real(modes(m),real64), real64) &
                     *exp(cmplx(0.0_real64, real(modes(m),real64)*z, real64))
       end do
       err = max(err, abs(cmplx(dfh(q),kind=real64) - analytic))
    end do
    call check('C2C d/dz via grid%kz', err, tol)
  end subroutine test_c2c_derivative

  !-- 4. R2C spectral d/dx via grid%kx --------------------------------------
  subroutine test_r2c_derivative()
    integer(ip), parameter :: nm = 3
    integer(ip) :: modes(nm) = [ 1_ip, 2_ip, 3_ip ]
    real(tp) :: amp(nm)
    real(tp) :: fh(0:nx-1), dfh(0:nx-1)
    real(tp),    device :: rd(0:nx-1)
    complex(tp), device :: cd(0:nkx-1)
    complex(tp) :: ch(0:nkx-1)
    real(tp) :: kxh(0:nkx-1)
    real(real64) :: x, err, analytic
    integer :: planf, planb, q, j, m
    amp = [ 1.0_tp, 0.7_tp, -0.4_tp ]
    kxh = grid%kx                                   ! D2H copy
    do q = 0, nx-1
       x = 2.0_real64*PI*real(q,real64)/real(nx,real64)
       fh(q) = 0.0_tp
       do m = 1, nm
          fh(q) = fh(q) + amp(m)*cos(real(modes(m),tp)*real(x,tp))
       end do
    end do
    rd = fh
    call tf_r2c_plan(planf, nx, 1_ip)
    call tf_c2r_plan(planb, nx, 1_ip)
    call tf_r2c(planf, rd, cd)
    istat = cudaDeviceSynchronize()
    ch = cd                                         ! D2H
    do j = 0, nkx-1
       if (j == nx/2) then
          ch(j) = (0.0_tp,0.0_tp)                   ! zero Nyquist
       else
          ch(j) = ch(j) * cmplx(0.0_tp, kxh(j), tp)
       end if
    end do
    cd = ch                                         ! H2D
    call tf_c2r(planb, cd, rd)
    istat = cudaDeviceSynchronize()
    dfh = rd
    dfh = dfh / real(nx, tp)
    call tf_destroy(planf);  call tf_destroy(planb)
    err = 0.0_real64
    do q = 0, nx-1
       x = 2.0_real64*PI*real(q,real64)/real(nx,real64)
       analytic = 0.0_real64
       do m = 1, nm
          analytic = analytic - real(amp(m),real64)*real(modes(m),real64) &
                     *sin(real(modes(m),real64)*x)
       end do
       err = max(err, abs(real(dfh(q),real64) - analytic))
    end do
    call check('R2C d/dx via grid%kx', err, tol)
  end subroutine test_r2c_derivative

end program test_transforms
