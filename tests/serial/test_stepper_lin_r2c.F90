!===============================================================================
! test_stepper_lin_r2c.F90  --  [R2C] confirm the LINEAR stepper
! (stepper_lin_mod: stepper_ev_setup + step_ev_linear, advancing eta AND v through
! the three SMR/CN substages) is EXTENT-AGNOSTIC under R2C: with nkx=nx/2+1 as the
! x-extent it reproduces the full-nx result on kx in [0,nkx-1], to bit identity.
!
! WHY THIS IS THE RANK-5 CHECK (BUG_GUIDE section 4 / L7): the per-stage precompute
! arrays are RANK-5 vm_all/vp_all (0:n,nx,nz,3) and Cinv_all (2,2,nx,nz,3), sliced
! per substage as vm_all(:,:,:,ks) [3 colons, rank-4 array] and
! Cinv_all(:,:,:,:,ks) [4 colons, rank-5 array]. The L7 bug was MIScounting these
! colons (a rank-4 slice on a rank-5 array selects an x-column as the stage and
! feeds garbage). Here nx is a DUMMY extent, so nx->nkx only resizes the x-axis --
! the colon count is UNCHANGED and the slicing stays correct (modeled before
! coding). This test exercises the full stage loop incl. that slicing, so a colon
! or extent error would show as a mismatch (or, per L7, a wrong-but-finite v that
! the bit-identity check still catches since the nx path is the reference).
!
! eta: Helmholtz; v: biharmonic via influence matrix. All sub-ops
! (apply_L, project_shen, helm_band_solve_gpu, shen_to_cheb_c, vif_solve_gpu) are
! extent-agnostic (proven 4a/4b). dp throughout. NaN/Inf-hardened.
!
! Build & run ON A GPU NODE:
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -c cheb_shen.F90 helm_band_mod.F90 cheb_bih.F90 vsolve_if_mod.F90 \
!             cheb_ops_mod.F90 spec_helpers_mod.F90 helm_band_gpu_mod.F90 \
!             vsolve_if_gpu_mod.F90 stepper_lin_mod.F90
!   nvfortran -cuda test_stepper_lin_r2c.F90 stepper_lin_mod.o vsolve_if_gpu_mod.o \
!             helm_band_gpu_mod.o spec_helpers_mod.o cheb_ops_mod.o vsolve_if_mod.o \
!             cheb_bih.o helm_band_mod.o cheb_shen.o types.o -o test_stepper_lin_r2c
!   ./test_stepper_lin_r2c
!===============================================================================
program test_stepper_lin_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,           only: dp, ip
  use helm_band_mod,   only: helm_band_setup
  use stepper_lin_mod, only: stepper_ev_setup, step_ev_linear
  use cudafor
  implicit none

  integer(ip), parameter :: n = 24, nx = 12, nz = 8
  integer(ip), parameter :: nkx = nx/2 + 1
  real(dp),    parameter :: dt = 1.0e-2_dp, Re = 180.0_dp
  integer(ip), parameter :: nsteps = 4
  real(real64) :: tol
  logical :: all_ok = .true.
  integer :: istat
  integer(ip) :: ix, iz, k, it

  real(dp) :: kxf_h(nx), kzf_h(nz), kxr_h(nkx), re_, im_
  real(dp) :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
  complex(dp) :: e0_h(0:n,nx,nz), v0_h(0:n,nx,nz)
  complex(dp) :: ef_h(0:n,nx,nz), vf_h(0:n,nx,nz)     ! full-nx final
  complex(dp) :: er_h(0:n,nkx,nz), vr_h(0:n,nkx,nz)   ! nkx final

  ! device, full nx
  real(dp),    device :: kxf(nx), kzf(nz), sd(0:n-2), st(0:n-2), bd(0:n-2)
  real(dp),    device :: vmf(0:n,nx,nz,3), vpf(0:n,nx,nz,3), Cinvf(2,2,nx,nz,3)
  complex(dp), device :: ef(0:n,nx,nz), vf(0:n,nx,nz)
  ! device, nkx
  real(dp),    device :: kxr(nkx)
  real(dp),    device :: vmr(0:n,nkx,nz,3), vpr(0:n,nkx,nz,3), Cinvr(2,2,nkx,nz,3)
  complex(dp), device :: er(0:n,nkx,nz), vr(0:n,nkx,nz)

  tol = 1.0e-12_real64

  do ix = 1, nx;  kxf_h(ix) = real(ix-1, dp) * 0.6_dp;  end do
  do iz = 1, nz;  kzf_h(iz) = real(iz-1, dp) * 1.1_dp;  end do
  do ix = 1, nkx; kxr_h(ix) = kxf_h(ix);                end do

  call helm_band_setup(n, sdiag, stail, bdiag, boff)

  ! random initial eta, v (Cheb coeffs)
  do iz = 1, nz; do ix = 1, nx; do k = 0, n
     call random_number(re_); call random_number(im_); e0_h(k,ix,iz) = cmplx(re_-0.5_dp,im_-0.5_dp,dp)
     call random_number(re_); call random_number(im_); v0_h(k,ix,iz) = cmplx(re_-0.5_dp,im_-0.5_dp,dp)
  end do; end do; end do

  ! ===== full-nx run ===========================================================
  kxf = kxf_h; kzf = kzf_h; sd = sdiag; st = stail; bd = bdiag
  ef = e0_h; vf = v0_h
  call stepper_ev_setup(n, nx, nz, dt, Re, kxf_h, kzf_h, vmf, vpf, Cinvf)
  do it = 1, nsteps
     call step_ev_linear(n, nx, nz, dt, Re, kxf, kzf, sd, st, bd, boff, &
                         vmf, vpf, Cinvf, ef, vf)
  end do
  istat = cudaDeviceSynchronize()
  ef_h = ef; vf_h = vf

  ! ===== nkx run on the first nkx x-columns ====================================
  kxr = kxr_h
  do iz = 1, nz; do ix = 1, nkx; do k = 0, n
     er(k,ix,iz) = e0_h(k,ix,iz);  vr(k,ix,iz) = v0_h(k,ix,iz)
  end do; end do; end do
  call stepper_ev_setup(n, nkx, nz, dt, Re, kxr_h, kzf_h, vmr, vpr, Cinvr)
  do it = 1, nsteps
     call step_ev_linear(n, nkx, nz, dt, Re, kxr, kzf, sd, st, bd, boff, &
                         vmr, vpr, Cinvr, er, vr)
  end do
  istat = cudaDeviceSynchronize()
  er_h = er; vr_h = vr

  ! ===== compare on the retained half + nonzero guards =========================
  call cmpc3('eta after 4 steps (nkx vs nx)', ef_h, er_h, tol)
  call cmpc3('v   after 4 steps (nkx vs nx)', vf_h, vr_h, tol)
  call nzc3 ('  eta nonzero (not decayed to 0)', ef_h)
  call nzc3 ('  v   nonzero (not decayed to 0)', vf_h)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine cmpc3(label, full, half, tol)
    character(*), intent(in) :: label
    complex(dp),  intent(in) :: full(0:n,nx,nz), half(0:n,nkx,nz)
    real(real64), intent(in) :: tol
    real(real64) :: dm, d
    integer(ip)  :: i, j, kk
    dm = 0.0_real64
    do kk=1,nz; do j=1,nkx; do i=0,n
       d = abs(full(i,j,kk) - half(i,j,kk))
       if (.not. (d == d)) d = huge(1.0_real64)
       dm = max(dm, d)
    end do; end do; end do
    if (dm /= dm .or. dm > tol) then
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|diff|=', dm;  all_ok = .false.
    else
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|diff|=', dm
    end if
  end subroutine cmpc3

  subroutine nzc3(label, f)
    character(*), intent(in) :: label
    complex(dp),  intent(in) :: f(0:n,nx,nz)
    real(real64) :: mx, a
    integer(ip)  :: i, j, kk
    mx = 0.0_real64
    do kk=1,nz; do j=1,nkx; do i=0,n
       a = abs(f(i,j,kk));  if (.not. (a == a)) a = huge(1.0_real64)
       mx = max(mx, a)
    end do; end do; end do
    if (mx > 1.0e-9_real64 .and. mx < huge(1.0_real64)) then
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|f|=', mx
    else
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|f|=', mx;  all_ok = .false.
    end if
  end subroutine nzc3

end program test_stepper_lin_r2c
