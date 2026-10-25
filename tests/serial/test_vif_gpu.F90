!===============================================================================
! test_vif_gpu.F90  --  validate the full GPU vif v-solve against the CPU vif.
!
! Both run the identical two-Helmholtz + influence-matrix algorithm; the only
! difference is the interior Helmholtz solves (banded GPU vs dense cheb_shen),
! and the homogeneous solutions / Cinv are computed by the SAME host vif_setup,
! so the two must agree to the banded-vs-dense Helmholtz level (~1e-12), for any
! RHS. (This is the port-correctness check; algorithm correctness was the MMS
! test.) Complex RHS, alpha across the DNS range. NaN/Inf hard-fail.
!
! Build:
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran -c cheb_bih.F90
!   nvfortran -c helm_band_mod.F90
!   nvfortran -cuda -c helm_band_gpu_mod.F90
!   nvfortran -cuda -c spec_helpers_mod.F90
!   nvfortran -c vsolve_if_mod.F90
!   nvfortran -cuda -c vsolve_if_gpu_mod.F90
!   nvfortran -cuda -c test_vif_gpu.F90
!   nvfortran -cuda -o test_vif_gpu test_vif_gpu.o vsolve_if_gpu_mod.o spec_helpers_mod.o \
!             helm_band_gpu_mod.o vsolve_if_mod.o helm_band_mod.o cheb_bih.o cheb_shen.o types.o
!   ./test_vif_gpu
!===============================================================================
program test_vif_gpu
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,             only: dp, ip
  use helm_band_mod,     only: helm_band_setup
  use vsolve_if_mod,     only: vif_t, vif_setup, vif_solve
  use vsolve_if_gpu_mod, only: vif_gpu_setup_host, vif_solve_gpu
  use cudafor
  implicit none

  integer(ip) :: nv(2) = [20, 32]
  integer(ip) :: xv(2) = [12, 16]
  integer(ip) :: zv(2) = [ 8, 12]
  real(dp)    :: alv(4) = [1.0_dp, 1.0e-2_dp, 1.0e-4_dp, 1.0e-6_dp]
  integer(ip) :: ic, ia
  logical :: all_pass

  all_pass = .true.
  do ic = 1, size(nv)
     write(*,'(/,a,i0,a,i0,a,i0,a)') '================ n=',nv(ic),'  nx=',xv(ic),'  nz=',zv(ic),' ================'
     do ia = 1, size(alv)
        call run_case(nv(ic), xv(ic), zv(ic), alv(ia), all_pass)
     end do
  end do
  if (all_pass) then
     print *, 'ALL TESTS PASSED'
  else
     print *, 'SOME TESTS FAILED'
  end if

contains

  subroutine run_case(n, nx, nz, alpha, all_pass)
    integer(ip), intent(in)    :: n, nx, nz
    real(dp),    intent(in)    :: alpha
    logical,     intent(inout) :: all_pass
    real(dp) :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
    real(dp) :: kxh(nx), kzh(nz)
    complex(dp), allocatable :: fc(:,:,:), vref(:,:,:), vg(:,:,:)
    real(dp) :: fre(0:n), fim(0:n), vre(0:n), vim(0:n)
    type(vif_t) :: d
    real(dp) :: k2, err, den, dd, rr, ri
    logical  :: bad
    integer(ip) :: ix, iz, k, kk, istat
    real(dp),    device, allocatable :: kx_d(:), kz_d(:), sd(:), st(:), bd(:)
    real(dp),    device, allocatable :: vm_d(:,:,:), vp_d(:,:,:), Cinv_d(:,:,:,:)
    complex(dp), device, allocatable :: fc_d(:,:,:), v_d(:,:,:)

    do ix = 1, nx
       kk = ix-1; if (kk > nx/2) kk = kk - nx; kxh(ix) = real(kk, dp)
    end do
    do iz = 1, nz
       kk = iz-1; if (kk > nz/2) kk = kk - nz; kzh(iz) = real(kk, dp)
    end do
    call helm_band_setup(n, sdiag, stail, bdiag, boff)

    allocate(fc(0:n,nx,nz), vref(0:n,nx,nz), vg(0:n,nx,nz))
    block   ! [repro] fixed seed, applied once per run -> the suite is a bit-identity oracle
            ! across the precision phases (replaces a bare call random_seed())
      integer              :: rs_n_, rs_j_
      integer, allocatable :: rs_seed_(:)
      logical, save        :: rs_done_ = .false.
      if (.not. rs_done_) then
         call random_seed(size = rs_n_)
         allocate(rs_seed_(rs_n_))
         rs_seed_ = [ (1789 + 101*rs_j_, rs_j_ = 0, rs_n_ - 1) ]
         call random_seed(put = rs_seed_)
         rs_done_ = .true.
      end if
    end block
    do iz = 1, nz; do ix = 1, nx; do k = 0, n
       call random_number(rr); call random_number(ri)
       fc(k,ix,iz) = cmplx(rr-0.5_dp, ri-0.5_dp, dp)
    end do; end do; end do

    ! CPU reference (per mode, re/im)
    do iz = 1, nz; do ix = 1, nx
       k2 = kxh(ix)**2 + kzh(iz)**2
       call vif_setup(n, k2, alpha, d)
       do k = 0, n; fre(k) = real(fc(k,ix,iz)); fim(k) = aimag(fc(k,ix,iz)); end do
       call vif_solve(n, d, fre, vre)
       call vif_solve(n, d, fim, vim)
       do k = 0, n; vref(k,ix,iz) = cmplx(vre(k), vim(k), dp); end do
    end do; end do

    ! GPU
    allocate(kx_d(nx), kz_d(nz), sd(0:n-2), st(0:n-2), bd(0:n-2))
    allocate(vm_d(0:n,nx,nz), vp_d(0:n,nx,nz), Cinv_d(2,2,nx,nz))
    allocate(fc_d(0:n,nx,nz), v_d(0:n,nx,nz))
    kx_d = kxh; kz_d = kzh; sd = sdiag; st = stail; bd = bdiag
    call vif_gpu_setup_host(n, nx, nz, kxh, kzh, alpha, vm_d, vp_d, Cinv_d)
    fc_d = fc
    call vif_solve_gpu(n, nx, nz, alpha, kx_d, kz_d, sd, st, bd, boff, &
                       vm_d, vp_d, Cinv_d, fc_d, v_d)
    istat = cudaDeviceSynchronize()
    vg = v_d

    err = 0.0_dp; den = 0.0_dp; bad = .false.
    do iz = 1, nz; do ix = 1, nx; do k = 0, n
       dd = abs(vg(k,ix,iz) - vref(k,ix,iz))
       if (.not. (dd == dd)) bad = .true.
       if (abs(vg(k,ix,iz)) > huge(1.0_dp)) bad = .true.
       err = max(err, dd); den = max(den, abs(vref(k,ix,iz)))
    end do; end do; end do
    if (den == 0.0_dp) den = 1.0_dp

    call report(alpha, merge(1.0e30_dp, err/den, bad), all_pass)

    deallocate(fc, vref, vg, kx_d, kz_d, sd, st, bd, vm_d, vp_d, Cinv_d, fc_d, v_d)
  end subroutine run_case

  subroutine report(alpha, e, all_pass)
    real(dp), intent(in)    :: alpha, e
    logical,  intent(inout) :: all_pass
    real(dp), parameter :: tol = 1.0e-8_dp
    character(6) :: tag
    if (e <= tol) then; tag = '[PASS]'; else; tag = '[FAIL]'; all_pass = .false.; end if
    write(*,'(a,2x,a,es9.2,2x,a,es12.4,2x,a,es9.2)') &
         tag, 'alpha=', alpha, 'rel err=', e, 'tol=', tol
  end subroutine report

end program test_vif_gpu
