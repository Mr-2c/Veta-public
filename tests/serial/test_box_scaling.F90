!===============================================================================
! test_box_scaling.F90  --  pins the PHYSICAL wavenumber scaling k = 2*pi*m/L.
!
! The bug this guards: integer mode indices used as wavenumbers (no 2*pi/L).
! A test that builds its own integer k is blind to it (self-consistent 2*pi
! boxes), so this test uses a box with Lx, Lz != 2*pi:
!
!   PART A: build_wavenumbers (wavenum_mod, the canonical builder main.F90
!           uses) must equal the independent analytic definition 2*pi*m/L to
!           machine precision, including the Nyquist slot (+nm/2).
!   PART B: GPU apply_L with those wavenumbers must equal the CPU reference
!           cheb_d2(q) - k2_phys*q per mode, k2_phys formed independently in
!           this test. Machine precision.
!   PART C (sharpness): the same comparison against the INTEGER-k reference
!           must FAIL by a large margin -- proving Parts A/B would catch the
!           original bug, not vacuously pass.
!
! NaN/Inf-hardened. Build + run (GPU node):
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran -c wavenum_mod.F90
!   nvfortran -cuda -c cheb_ops_mod.F90
!   nvfortran -cuda -cudalib=cufft -o test_box_scaling test_box_scaling.F90 \
!       types.o cheb_shen.o wavenum_mod.o cheb_ops_mod.o
!===============================================================================
program test_box_scaling
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,        only: dp, ip
  use cudafor
  use wavenum_mod,  only: build_wavenumbers
  use cheb_shen,    only: cheb_d2
  use cheb_ops_mod, only: apply_L
  implicit none

  integer(ip), parameter :: n = 16, nx = 8, nz = 6
  real(dp),    parameter :: Lx = 3.0_dp, Lz = 1.7_dp     ! deliberately != 2*pi
  real(dp),    parameter :: TWOPI = 2.0_dp*acos(-1.0_dp)

  real(dp) :: kxh(nx), kzh(nz), kref
  real(dp) :: errA, errB, errC, denom
  complex(dp), allocatable :: qh(:,:,:), Lq_gpu(:,:,:)
  complex(dp), allocatable :: ac(:), d2(:), Lref(:), Lint(:)
  complex(dp), device, allocatable :: q_d(:,:,:), Lq_d(:,:,:), scr_d(:,:,:)
  real(dp),    device, allocatable :: kx_d(:), kz_d(:)
  real(dp)    :: k2p, k2i, r1, r2, dec
  integer(ip) :: ix, iz, m, mm
  integer,     allocatable :: sseed(:)
  integer     :: nseed
  logical     :: bad

  ! deterministic RNG (mirrors init_mod's seed_rng)
  call random_seed(size=nseed)
  allocate(sseed(nseed))
  do m = 1, nseed; sseed(m) = 7919 + 101*m; end do
  call random_seed(put=sseed)
  deallocate(sseed)

  ! ---------- PART A: builder vs independent analytic definition --------------
  call build_wavenumbers(nx, Lx, kxh)
  call build_wavenumbers(nz, Lz, kzh)
  errA = 0.0_dp
  do ix = 1, nx
     mm = ix-1; if (mm > nx/2) mm = mm - nx
     kref = TWOPI*real(mm,dp)/Lx
     errA = max(errA, abs(kxh(ix) - kref))
  end do
  do iz = 1, nz
     mm = iz-1; if (mm > nz/2) mm = mm - nz
     kref = TWOPI*real(mm,dp)/Lz
     errA = max(errA, abs(kzh(iz) - kref))
  end do
  ! Nyquist slots must carry +nm/2 (not 0): guard the convention explicitly.
  errA = max(errA, abs(kxh(nx/2+1) - TWOPI*real(nx/2,dp)/Lx))
  errA = max(errA, abs(kzh(nz/2+1) - TWOPI*real(nz/2,dp)/Lz))

  ! ---------- build a smooth random complex field ------------------------------
  allocate(qh(0:n,nx,nz), Lq_gpu(0:n,nx,nz), ac(0:n), d2(0:n), Lref(0:n), Lint(0:n))
  do iz = 1, nz
     do ix = 1, nx
        do m = 0, n
           dec = exp(-real(m,dp)/3.0_dp)
           call random_number(r1); call random_number(r2)
           qh(m,ix,iz) = cmplx(2.0_dp*r1-1.0_dp, 2.0_dp*r2-1.0_dp, dp)*dec
        end do
     end do
  end do

  allocate(q_d(0:n,nx,nz), Lq_d(0:n,nx,nz), scr_d(0:n,nx,nz), kx_d(nx), kz_d(nz))
  q_d = qh;  kx_d = kxh;  kz_d = kzh

  ! ---------- PART B: GPU apply_L vs CPU (cheb_d2 - k2_phys) -------------------
  call apply_L(n+1, nx, nz, kx_d, kz_d, q_d, Lq_d, scr_d)
  Lq_gpu = Lq_d
  errB = 0.0_dp;  errC = 1.0e30_dp;  denom = 0.0_dp;  bad = .false.
  do iz = 1, nz
     mm = iz-1; if (mm > nz/2) mm = mm - nz
     do ix = 1, nx
        m = ix-1; if (m > nx/2) m = m - nx
        k2p = (TWOPI*real(m,dp)/Lx)**2 + (TWOPI*real(mm,dp)/Lz)**2   ! physical
        k2i = real(m*m + mm*mm, dp)                                  ! the old bug
        ac = qh(:,ix,iz)
        call cheb_d2_c(n, ac, d2)
        Lref = d2 - k2p*ac
        Lint = d2 - k2i*ac
        errB  = max(errB,  maxval(abs(Lq_gpu(:,ix,iz) - Lref)))
        denom = max(denom, maxval(abs(Lref)))
        ! PART C accumulator: distance of the GPU result from the INTEGER-k ref,
        ! over the modes where the two references actually differ (k2p /= k2i).
        if (abs(k2p - k2i) > 1.0e-12_dp) &
             errC = min(errC, maxval(abs(Lq_gpu(:,ix,iz) - Lint)))
     end do
  end do
  if (denom > 0.0_dp) errB = errB/denom

  ! NaN/Inf hardening
  if (.not.(errA == errA) .or. .not.(errB == errB) .or. errA > huge(1.0_dp) &
       .or. errB > huge(1.0_dp)) bad = .true.
  if (bad) then; errA = 1.0e30_dp; errB = 1.0e30_dp; end if

  write(*,'(a,es12.4)') ' [A] builder vs analytic 2*pi*m/L : max abs err = ', errA
  write(*,'(a,es12.4)') ' [B] apply_L vs CPU, physical k    : max rel err = ', errB
  write(*,'(a,es12.4)') ' [C] sharpness: min dist to INTEGER-k result = ', errC

  if (errA < 1.0e-14_dp .and. errB < 1.0e-12_dp .and. errC > 1.0e-3_dp) then
     write(*,'(a)') ' [PASS] box scaling: physical wavenumbers, and the test is sharp'
  else
     write(*,'(a)') ' [FAIL] box scaling'
     stop 1
  end if

contains

  ! complex second derivative via the real CPU recurrence on Re and Im parts
  subroutine cheb_d2_c(n_, a, c)
    integer(ip), intent(in)  :: n_
    complex(dp), intent(in)  :: a(0:n_)
    complex(dp), intent(out) :: c(0:n_)
    real(dp) :: ar(0:n_), ai(0:n_), cr(0:n_), ci(0:n_)
    ar = real(a);  ai = aimag(a)
    call cheb_d2(n_, ar, cr)
    call cheb_d2(n_, ai, ci)
    c = cmplx(cr, ci, dp)
  end subroutine cheb_d2_c

end program test_box_scaling
