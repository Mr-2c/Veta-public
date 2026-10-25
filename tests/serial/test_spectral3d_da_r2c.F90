!===============================================================================
! test_spectral3d_da_r2c.F90  --  [R2C] validate the DEALIASED
! real-x transforms (to_physical_da_r2c / to_spectral_da_r2c) against the C2C
! dealiased path.
!
! The x-axis pad/truncate is a LOW-PASS COPY (kx=0..nx/2, no +-wrap) because R2C
! stores only kx>=0; z keeps the full C2C +-wrap. Correct ONLY under the Nyquist
! invariant (solver keeps x- and z-Nyquist zero), which the test enforces.
!
! Robust construction (no manual Hermitian fill): build a real coarse field,
! get its FULL C2C spectrum with to_spectral (auto-Hermitian), and take the
! first nkx columns as the R2C input -- guaranteed consistent. Then:
!   1. ROUND-TRIP: R2C coarse -> dealiased physical -> coarse, /(nxd*nzd) = input.
!   2. C2C CROSS-CHECK: R2C dealiased physical == C2C dealiased physical (real).
!
! NaN/Inf-hardened. Build & run ON A GPU NODE:
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -cudalib=cufft -c transforms_mod.F90
!   nvfortran -cuda -cudalib=cufft -c cheb_fft_mod.F90
!   nvfortran -cuda -cudalib=cufft -c fft3d_mod.F90
!   nvfortran -cuda -cudalib=cufft -c spectral3d_mod.F90
!   nvfortran -cuda -cudalib=cufft test_spectral3d_da_r2c.F90 \
!             spectral3d_mod.o fft3d_mod.o cheb_fft_mod.o transforms_mod.o types.o \
!             -o test_spectral3d_da_r2c
!   ./test_spectral3d_da_r2c
!===============================================================================
program test_spectral3d_da_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,          only: tp, ip
  use spectral3d_mod, only: s3d_plans, s3d_setup, s3d_destroy, to_spectral, &
                            s3d_setup_da, to_physical_da, &
                            s3d_setup_da_r2c, s3d_destroy_r2c, &
                            to_physical_da_r2c, to_spectral_da_r2c
  use cudafor
  implicit none

  integer(ip), parameter :: ny = 5, nx = 8, nz = 6
  integer(ip), parameter :: nxd = 3*nx/2, nzd = 3*nz/2
  integer(ip), parameter :: nkx = nx/2+1, nkxd = nxd/2+1
  real(real64) :: tol
  logical :: all_ok = .true.
  integer :: istat
  integer(ip) :: iy, kx, kz

  real(tp)    :: re, im
  complex(tp) :: Ah(ny,nkx,nz), A2h(ny,nkx,nz), Cfull_h(ny,nx,nz), Pc_h(ny,nxd,nzd)
  real(tp)    :: Pr_h(ny,nxd,nzd)
  complex(tp), device :: Ad(ny,nkx,nz), Wf(ny,nkxd,nzd)
  real(tp),    device :: Pd(ny,nxd,nzd)
  ! buffers to get the C2C spectrum of a real field, then its dealiased physical
  complex(tp), device :: Ac(ny,nx,nz), workc(ny,nx,nz)
  complex(tp), device :: Pc(ny,nxd,nzd), Wc(ny,nxd,nzd)
  type(s3d_plans) :: pr, pcfull, pcda

  tol = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)

  ! --- build a real coarse field as a complex array with imag=0 ---
  do kz = 1, nz; do kx = 1, nx; do iy = 1, ny
     call random_number(re)
     Cfull_h(iy,kx,kz) = cmplx(re-0.5_tp, 0.0_tp, tp)
  end do; end do; end do
  ! forward C2C to get its (auto-Hermitian) full spectrum
  Ac = Cfull_h
  call s3d_setup(pcfull, ny, nx, nz)
  call to_spectral(pcfull, ny, nx, nz, Ac, workc)
  istat = cudaDeviceSynchronize()
  Cfull_h = Ac                                   ! full C2C spectrum of a real field
  ! zero the Nyquist columns (solver invariant) on the full spectrum
  Cfull_h(:, nx/2+1, :) = (0.0_tp, 0.0_tp)       ! x-Nyquist
  Cfull_h(:, :, nz/2+1) = (0.0_tp, 0.0_tp)       ! z-Nyquist
  ! R2C input = first nkx columns
  do kz = 1, nz; do kx = 1, nkx; do iy = 1, ny
     Ah(iy,kx,kz) = Cfull_h(iy,kx,kz)
  end do; end do; end do

  call s3d_setup_da_r2c(pr, ny, nx, nz)

  ! ===== Test 1: dealiased R2C round-trip ======================================
  Ad = Ah
  call to_physical_da_r2c(pr, ny, nx, nz, nxd, nzd, Ad, Pd, Wf)
  call to_spectral_da_r2c(pr, ny, nx, nz, nxd, nzd, Pd, Wf, Ad)
  istat = cudaDeviceSynchronize()
  A2h = Ad
  call report_c('dealiased R2C round-trip = (nxd*nzd)*identity', Ah, &
                A2h/cmplx(real(nxd*nzd,tp),0.0_tp,tp), tol)

  ! ===== Test 2: R2C dealiased physical == C2C dealiased physical ==============
  Ac = Cfull_h                                   ! the same Nyquist-zeroed full spectrum
  call s3d_setup_da(pcda, ny, nx, nz)
  call to_physical_da(pcda, ny, nx, nz, nxd, nzd, Ac, Pc, Wc)
  istat = cudaDeviceSynchronize()
  Pc_h = Pc                                      ! device -> host (report compares on host)
  Ad = Ah
  call to_physical_da_r2c(pr, ny, nx, nz, nxd, nzd, Ad, Pd, Wf)
  istat = cudaDeviceSynchronize()
  Pr_h = Pd
  call report_vs_c2c('R2C dealiased phys == C2C dealiased phys (real)', Pc_h, Pr_h, tol)

  call s3d_destroy(pcfull)
  call s3d_destroy(pcda)
  call s3d_destroy_r2c(pr)
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
    do k=1,nz; do j=1,nkx; do i=1,ny
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

  subroutine report_vs_c2c(label, Cc, Pr, tol)
    character(*), intent(in) :: label
    complex(tp),  intent(in) :: Cc(ny,nxd,nzd)
    real(tp),     intent(in) :: Pr(ny,nxd,nzd)
    real(real64), intent(in) :: tol
    real(real64) :: dm, d, im_max
    integer(ip)  :: i, j, k
    dm = 0.0_real64;  im_max = 0.0_real64
    do k=1,nzd; do j=1,nxd; do i=1,ny
       d = abs(real(Cc(i,j,k),real64) - real(Pr(i,j,k),real64))
       if (.not. (d == d)) d = huge(1.0_real64)
       dm = max(dm, d)
       im_max = max(im_max, abs(aimag(Cc(i,j,k))))
    end do; end do; end do
    if (dm /= dm .or. dm > tol) then
       print '(a,a,a,es12.4,a,es12.4)', '  [FAIL] ', label, '  max|diff|=', dm, ' max|imag C2C|=', im_max
       all_ok = .false.
    else
       print '(a,a,a,es12.4,a,es12.4,a)', '  [ok] ', label, '  max|diff|=', dm, ' (max|imag C2C|=', im_max, ')'
    end if
  end subroutine report_vs_c2c

end program test_spectral3d_da_r2c
