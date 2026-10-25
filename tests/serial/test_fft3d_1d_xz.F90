!===============================================================================
! test_fft3d_1d_xz.F90  --  validate the STANDALONE 1-D real-x
! transforms (plan_x_r2c / plan_x_c2r -- an x-leading reorder + ONE batched
! cuFFT) against the FUSED rank-2 R2C/C2R plan that the
! single-GPU code uses. This also guards the reorder: it must still match the fused
! transform to round-off.
!
! This is the un-fused x pass in ISOLATION: no transpose, no MPI/NCCL, one GPU.
! The parallel transform is  [1-D x] -> transpose -> [1-D z]; here we check
! that on a single rank  [1-D rfft-x ; 1-D fft-z]  reproduces the fused R2C, and
! [1-D ifft-z ; 1-D irfft-x]  reproduces the fused C2R, to round-off. The z pass
! reuses the already-validated plan_axis_c2c on the trailing axis, so what is
! really being shaken out here is the x-plane R2C/C2R stride/embed derivation.
!
! Agreement is to ROUND-OFF (not bit-exact): a single rank-2 cuFFT plan and two
! separate rank-1 plans use different internal factorisations. cuFFT is
! unnormalised, so a forward+inverse round-trip scales by nx*nz (Test 3).
!
! NaN/Inf-hardened; marker-based pass/fail ('[FAIL]' / 'error stop'). Build & run
! ON A GPU NODE via `bash run_all_tests.sh` (auto-discovered); minimal deps:
!   types.F90, transforms_mod.F90, fft3d_mod.F90  (-cuda -cudalib=cufft).
!===============================================================================
program test_fft3d_1d_xz
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,          only: tp, ip
  use transforms_mod, only: tf_destroy
  use fft3d_mod,      only: plan_xz_r2c, plan_xz_c2r, exec_xz_r2c, exec_xz_c2r, &
                            plan_x_r2c, plan_x_c2r, exec_x_r2c, exec_x_c2r, &
                            plan_axis_c2c, exec_axis_c2c, &
                            CUFFT_FORWARD, CUFFT_INVERSE
  use cudafor
  implicit none

  integer(ip), parameter :: ny = 5, nx = 12, nz = 8
  integer(ip), parameter :: nkx = nx/2 + 1
  real(real64) :: tol
  logical :: all_ok = .true.
  integer  :: istat
  integer(ip) :: iy, ix, kz
  real(tp) :: re

  real(tp)    :: P0_h(ny,nx,nz), Pf_h(ny,nx,nz), P1_h(ny,nx,nz), Prt_h(ny,nx,nz)
  complex(tp) :: Sf_h(ny,nkx,nz), S1_h(ny,nkx,nz)

  real(tp),    device :: P0(ny,nx,nz), Pf(ny,nx,nz), P1(ny,nx,nz), Prt(ny,nx,nz)
  complex(tp), device :: Sf(ny,nkx,nz), S1(ny,nkx,nz), Stmp(ny,nkx,nz)

  integer :: pr2c, pc2r        ! fused rank-2 (oracle)
  integer :: pxr2c, pxc2r, pz  ! 1-D x-R2C, x-C2R, and z-C2C (trailing axis)

  tol = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)

  ! --- a random real physical field (deterministic default RNG state) ---
  do kz = 1, nz; do ix = 1, nx; do iy = 1, ny
     call random_number(re)
     P0_h(iy,ix,kz) = re - 0.5_tp
  end do; end do; end do
  P0 = P0_h

  call plan_xz_r2c(pr2c, ny, nx, nz)
  call plan_xz_c2r(pc2r, ny, nx, nz)
  call plan_x_r2c (pxr2c, ny, nx, nz)
  call plan_x_c2r (pxc2r, ny, nx, nz)
  call plan_axis_c2c(pz, 3_ip, ny, nkx, nz)

  ! ===== Test 1: forward  [1-D rfft-x ; 1-D fft-z]  ==  fused R2C ==============
  call exec_xz_r2c(pr2c, P0, Sf)                                  ! fused (oracle)
  call exec_x_r2c (pxr2c, ny, nx, nz, P0, S1)                     ! rfft-x (loop z)
  call exec_axis_c2c(pz, 3_ip, ny, nkx, nz, S1, S1, CUFFT_FORWARD) ! fft-z (trailing)
  istat = cudaDeviceSynchronize()
  Sf_h = Sf;  S1_h = S1
  call report_c('forward [rfftx;fftz] == fused R2C', Sf_h, S1_h, tol)

  ! ===== Test 2: inverse  [1-D ifft-z ; 1-D irfft-x]  ==  fused C2R ============
  Stmp = Sf
  call exec_xz_c2r(pc2r, Stmp, Pf)                                ! fused (consumes Stmp)
  Stmp = Sf
  call exec_axis_c2c(pz, 3_ip, ny, nkx, nz, Stmp, Stmp, CUFFT_INVERSE) ! ifft-z
  call exec_x_c2r (pxc2r, ny, nx, nz, Stmp, P1)                   ! irfft-x (loop z)
  istat = cudaDeviceSynchronize()
  Pf_h = Pf;  P1_h = P1
  call report_r('inverse [ifftz;irfftx] == fused C2R', Pf_h, P1_h, tol)

  ! ===== Test 3: 1-D round-trip  ==  (nx*nz) * identity =======================
  call exec_x_r2c (pxr2c, ny, nx, nz, P0, S1)
  call exec_axis_c2c(pz, 3_ip, ny, nkx, nz, S1, S1, CUFFT_FORWARD)
  call exec_axis_c2c(pz, 3_ip, ny, nkx, nz, S1, S1, CUFFT_INVERSE)
  call exec_x_c2r (pxc2r, ny, nx, nz, S1, Prt)
  istat = cudaDeviceSynchronize()
  Prt_h = Prt
  call report_r('1-D round-trip == (nx*nz)*identity', &
                P0_h*real(nx*nz,tp), Prt_h, tol*real(nx*nz,real64))

  call tf_destroy(pr2c);  call tf_destroy(pc2r)
  call tf_destroy(pxr2c); call tf_destroy(pxc2r); call tf_destroy(pz)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine report_c(label, a, b, tol)
    character(*), intent(in) :: label
    complex(tp),  intent(in) :: a(ny,nkx,nz), b(ny,nkx,nz)
    real(real64), intent(in) :: tol
    real(real64) :: dm, d
    integer(ip)  :: i, j, k
    dm = 0.0_real64
    do k = 1, nz; do j = 1, nkx; do i = 1, ny
       d = abs(a(i,j,k) - b(i,j,k))
       if (.not. (d == d)) d = huge(1.0_real64)
       dm = max(dm, d)
    end do; end do; end do
    if (dm /= dm .or. dm > tol) then
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|diff|=', dm;  all_ok = .false.
    else
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|diff|=', dm
    end if
  end subroutine report_c

  subroutine report_r(label, a, b, tol)
    character(*), intent(in) :: label
    real(tp),     intent(in) :: a(ny,nx,nz), b(ny,nx,nz)
    real(real64), intent(in) :: tol
    real(real64) :: dm, d
    integer(ip)  :: i, j, k
    dm = 0.0_real64
    do k = 1, nz; do j = 1, nx; do i = 1, ny
       d = abs(real(a(i,j,k),real64) - real(b(i,j,k),real64))
       if (.not. (d == d)) d = huge(1.0_real64)
       dm = max(dm, d)
    end do; end do; end do
    if (dm /= dm .or. dm > tol) then
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|diff|=', dm;  all_ok = .false.
    else
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|diff|=', dm
    end if
  end subroutine report_r

end program test_fft3d_1d_xz
