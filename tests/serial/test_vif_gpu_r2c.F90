!===============================================================================
! test_vif_gpu_r2c.F90  --  [R2C] confirm the GPU clamped v-solve
! (vsolve_if_gpu_mod) is EXTENT-AGNOSTIC under R2C: the host precompute
! (vif_gpu_setup_host) and the per-step solve (vif_solve_gpu) reproduce their
! full-nx results when called with nkx=nx/2+1 as the x-extent (and nkx-length kx),
! on the retained half kx in [0,nkx-1], to bit identity.
!
! Every routine is PER-MODE: vif_closure_kernel is one thread per (ix,iz);
! vif_gpu_setup_host loops (ix,iz) calling vif_setup_band with the scalar
! k2=kxh(ix)^2+kzh(iz)^2 (k2-LOCAL, no x-extent notion); the interior solves reuse
! the extent-agnostic helm_band/spec_helpers (proven in 4a). No ix+-1 / nx-ix /
! special x-column (the v-solve treats every mode uniformly -- NO mean-mode branch,
! unlike recover_uw). Verified by inspection, BUG_GUIDE section 4. So the R2C path
! reuses the SAME source with nx->nkx; this test proves it.
!
! IMPORTANT (BUG_GUIDE section 4, the rank-5 hazard): the arrays HERE are rank-4
! (vm/vp (0:n,nx,nz); Cinv (2,2,nx,nz)) -- the x-slot is a plain extent. The
! per-STAGE rank-5 arrays (vm_all/Cinv_all (...,3)) live in the STEPPER (incr 4c),
! not here, so the L7 under-subscript hazard is a 4c concern.
!
! Checks: full pipeline (setup -> solve) at nx and at nkx; require vm/vp/Cinv and
! the solved v agree on kx<nkx (exact 0.0 -- same code/operands) + nonzero guards.
!
! dp throughout. NaN/Inf-hardened. Build & run ON A GPU NODE:
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -c cheb_shen.F90 helm_band_mod.F90 cheb_bih.F90 vsolve_if_mod.F90 \
!             spec_helpers_mod.F90 helm_band_gpu_mod.F90 vsolve_if_gpu_mod.F90
!   nvfortran -cuda test_vif_gpu_r2c.F90 vsolve_if_gpu_mod.o helm_band_gpu_mod.o \
!             spec_helpers_mod.o vsolve_if_mod.o cheb_bih.o helm_band_mod.o cheb_shen.o \
!             types.o -o test_vif_gpu_r2c
!   ./test_vif_gpu_r2c
!===============================================================================
program test_vif_gpu_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,             only: dp, ip
  use helm_band_mod,     only: helm_band_setup
  use vsolve_if_gpu_mod, only: vif_gpu_setup_host, vif_solve_gpu
  use cudafor
  implicit none

  integer(ip), parameter :: n = 24, nx = 12, nz = 8
  integer(ip), parameter :: nkx = nx/2 + 1
  real(dp),    parameter :: alpha = 1.0e-3_dp        ! representative DNS value
  real(real64) :: tol
  logical :: all_ok = .true.
  integer :: istat
  integer(ip) :: ix, iz, k

  real(dp) :: kxf_h(nx), kzf_h(nz), kxr_h(nkx), re, im
  real(dp) :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
  complex(dp) :: fc_h(0:n,nx,nz)
  ! full-nx results
  real(dp)    :: vmf_h(0:n,nx,nz), vpf_h(0:n,nx,nz), Cinvf_h(2,2,nx,nz)
  complex(dp) :: vf_h(0:n,nx,nz)
  ! nkx results
  real(dp)    :: vmr_h(0:n,nkx,nz), vpr_h(0:n,nkx,nz), Cinvr_h(2,2,nkx,nz)
  complex(dp) :: vr_h(0:n,nkx,nz)

  ! device, full nx
  real(dp),    device :: kxf(nx), kzf(nz), sd(0:n-2), st(0:n-2), bd(0:n-2)
  real(dp),    device :: vmf(0:n,nx,nz), vpf(0:n,nx,nz), Cinvf(2,2,nx,nz)
  complex(dp), device :: fcf(0:n,nx,nz), vf(0:n,nx,nz)
  ! device, nkx
  real(dp),    device :: kxr(nkx)
  real(dp),    device :: vmr(0:n,nkx,nz), vpr(0:n,nkx,nz), Cinvr(2,2,nkx,nz)
  complex(dp), device :: fcr(0:n,nkx,nz), vr(0:n,nkx,nz)

  tol = 1.0e-12_real64

  do ix = 1, nx;  kxf_h(ix) = real(ix-1, dp) * 0.6_dp;  end do
  do iz = 1, nz;  kzf_h(iz) = real(iz-1, dp) * 1.1_dp;  end do
  do ix = 1, nkx; kxr_h(ix) = kxf_h(ix);                end do

  call helm_band_setup(n, sdiag, stail, bdiag, boff)

  do iz = 1, nz; do ix = 1, nx; do k = 0, n
     call random_number(re); call random_number(im)
     fc_h(k,ix,iz) = cmplx(re-0.5_dp, im-0.5_dp, dp)
  end do; end do; end do

  ! ===== full-nx pipeline ======================================================
  kxf = kxf_h; kzf = kzf_h; sd = sdiag; st = stail; bd = bdiag; fcf = fc_h
  call vif_gpu_setup_host(n, nx, nz, kxf_h, kzf_h, alpha, vmf, vpf, Cinvf)
  call vif_solve_gpu(n, nx, nz, alpha, kxf, kzf, sd, st, bd, boff, vmf, vpf, Cinvf, fcf, vf)
  istat = cudaDeviceSynchronize()
  vmf_h = vmf; vpf_h = vpf; Cinvf_h = Cinvf; vf_h = vf

  ! ===== nkx pipeline on the first nkx x-columns ===============================
  kxr = kxr_h
  do iz = 1, nz; do ix = 1, nkx; do k = 0, n
     fcr(k,ix,iz) = fc_h(k,ix,iz)
  end do; end do; end do
  call vif_gpu_setup_host(n, nkx, nz, kxr_h, kzf_h, alpha, vmr, vpr, Cinvr)
  call vif_solve_gpu(n, nkx, nz, alpha, kxr, kzf, sd, st, bd, boff, vmr, vpr, Cinvr, fcr, vr)
  istat = cudaDeviceSynchronize()
  vmr_h = vmr; vpr_h = vpr; Cinvr_h = Cinvr; vr_h = vr

  ! ===== compare on the retained half + nonzero guards =========================
  call cmpr3('vif precompute vm  (nkx vs nx)', n, vmf_h,  vmr_h,  tol)
  call cmpr3('vif precompute vp  (nkx vs nx)', n, vpf_h,  vpr_h,  tol)
  call cmpCinv('vif precompute Cinv(nkx vs nx)',  Cinvf_h, Cinvr_h, tol)
  call cmpc3('vif solved v       (nkx vs nx)', n, vf_h,   vr_h,   tol)

  call nzc3('  solved v nonzero',  n, vf_h)
  call nzr3('  vm nonzero',        n, vmf_h)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  ! real (0:n,nx,nz) vs (0:n,nkx,nz)
  subroutine cmpr3(label, mtop, full, half, tol)
    character(*), intent(in) :: label
    integer(ip),  intent(in) :: mtop
    real(dp),     intent(in) :: full(0:mtop,nx,nz), half(0:mtop,nkx,nz)
    real(real64), intent(in) :: tol
    real(real64) :: dm, d
    integer(ip)  :: i, j, kk
    dm = 0.0_real64
    do kk=1,nz; do j=1,nkx; do i=0,mtop
       d = abs(real(full(i,j,kk),real64) - real(half(i,j,kk),real64))
       if (.not. (d == d)) d = huge(1.0_real64)
       dm = max(dm, d)
    end do; end do; end do
    call verdict(label, dm, tol)
  end subroutine cmpr3

  ! complex (0:n,nx,nz) vs (0:n,nkx,nz)
  subroutine cmpc3(label, mtop, full, half, tol)
    character(*), intent(in) :: label
    integer(ip),  intent(in) :: mtop
    complex(dp),  intent(in) :: full(0:mtop,nx,nz), half(0:mtop,nkx,nz)
    real(real64), intent(in) :: tol
    real(real64) :: dm, d
    integer(ip)  :: i, j, kk
    dm = 0.0_real64
    do kk=1,nz; do j=1,nkx; do i=0,mtop
       d = abs(full(i,j,kk) - half(i,j,kk))
       if (.not. (d == d)) d = huge(1.0_real64)
       dm = max(dm, d)
    end do; end do; end do
    call verdict(label, dm, tol)
  end subroutine cmpc3

  ! Cinv (2,2,nx,nz) vs (2,2,nkx,nz)
  subroutine cmpCinv(label, full, half, tol)
    character(*), intent(in) :: label
    real(dp),     intent(in) :: full(2,2,nx,nz), half(2,2,nkx,nz)
    real(real64), intent(in) :: tol
    real(real64) :: dm, d
    integer(ip)  :: a, b, j, kk
    dm = 0.0_real64
    do kk=1,nz; do j=1,nkx; do b=1,2; do a=1,2
       d = abs(real(full(a,b,j,kk),real64) - real(half(a,b,j,kk),real64))
       if (.not. (d == d)) d = huge(1.0_real64)
       dm = max(dm, d)
    end do; end do; end do; end do
    call verdict(label, dm, tol)
  end subroutine cmpCinv

  subroutine verdict(label, dm, tol)
    character(*), intent(in) :: label
    real(real64), intent(in) :: dm, tol
    if (dm /= dm .or. dm > tol) then
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|diff|=', dm;  all_ok = .false.
    else
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|diff|=', dm
    end if
  end subroutine verdict

  subroutine nzc3(label, mtop, f)
    character(*), intent(in) :: label
    integer(ip),  intent(in) :: mtop
    complex(dp),  intent(in) :: f(0:mtop,nx,nz)
    real(real64) :: mx, a
    integer(ip)  :: i, j, kk
    mx = 0.0_real64
    do kk=1,nz; do j=1,nkx; do i=0,mtop
       a = abs(f(i,j,kk));  if (.not. (a == a)) a = huge(1.0_real64)
       mx = max(mx, a)
    end do; end do; end do
    call nzverdict(label, mx)
  end subroutine nzc3

  subroutine nzr3(label, mtop, f)
    character(*), intent(in) :: label
    integer(ip),  intent(in) :: mtop
    real(dp),     intent(in) :: f(0:mtop,nx,nz)
    real(real64) :: mx, a
    integer(ip)  :: i, j, kk
    mx = 0.0_real64
    do kk=1,nz; do j=1,nkx; do i=0,mtop
       a = abs(real(f(i,j,kk),real64));  if (.not. (a == a)) a = huge(1.0_real64)
       mx = max(mx, a)
    end do; end do; end do
    call nzverdict(label, mx)
  end subroutine nzr3

  subroutine nzverdict(label, mx)
    character(*), intent(in) :: label
    real(real64), intent(in) :: mx
    if (mx > 1.0e-9_real64 .and. mx < huge(1.0_real64)) then
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|f|=', mx
    else
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|f|=', mx;  all_ok = .false.
    end if
  end subroutine nzverdict

end program test_vif_gpu_r2c
