!===============================================================================
! test_vif_setup_gpu.F90 -- equivalence check for the batched on-device v-solve
! operator setup (vif_gpu_setup_dev) against the validated host per-mode loop
! (vif_gpu_setup_host). Single GPU, no MPI/cuFFT: the setup is mode-local.
!
! Builds vm (v_-), vp (v_+) and the 2x2 inverse influence matrix Cinv for every
! (kx,kz) by BOTH paths and reports the max abs difference. They must agree to
! banded-solve round-off (host helm_band_solve vs the GPU helm_band kernel are
! already validated equivalent to ~1e-13; vif tolerance is 1e-8).
!
! Build/run:  bash run_all_tests.sh -k vif_setup_gpu   (TP=real64 default; TP=sp also)
!===============================================================================
program test_vif_setup_gpu
  use kinds,             only: dp, ip
  use cudafor
  use helm_band_mod,     only: helm_band_setup
  use vsolve_if_gpu_mod, only: vif_gpu_setup_host, vif_gpu_setup_dev, vif_gpu_setup_dev_ws
  implicit none
  integer(ip), parameter :: n = 48, nx = 24, nz = 24
  real(dp),    parameter :: PI = 3.141592653589793238462643_dp
  real(dp) :: Lx, Lz, Re, dt, alpha, boff
  real(dp) :: kxh(nx), kzh(nz), sdh(0:n-2), sth(0:n-2), bdh(0:n-2)
  real(dp), device :: kx_d(nx), kz_d(nz), sd_d(0:n-2), st_d(0:n-2), bd_d(0:n-2)
  real(dp), device :: vmH(0:n,nx,nz), vpH(0:n,nx,nz), CiH(2,2,nx,nz)
  real(dp), device :: vmG(0:n,nx,nz), vpG(0:n,nx,nz), CiG(2,2,nx,nz)
  real(dp), device :: vmW(0:n,nx,nz), vpW(0:n,nx,nz), CiW(2,2,nx,nz)
  ! caller-owned build scratch for the _ws variant, seeded with garbage
  ! (huge) to prove every temp is fully written before it is read
  complex(dp), device :: wrhs(0:n,nx,nz), wftl(0:n-2,nx,nz), wgsh(0:n-2,nx,nz)
  complex(dp), device :: wgpc(0:n,nx,nz), wgmc(0:n,nx,nz), wvpc(0:n,nx,nz), wvmc(0:n,nx,nz)
  real(dp), allocatable :: h1(:,:,:), h2(:,:,:), c1(:,:,:,:), c2(:,:,:,:)
  real(dp) :: dvm, dvp, dci, wvm, wvp, wci, tol
  integer(ip) :: ix, iz, kk, istat
  logical :: ok, okw

  Lx = 2.0_dp*PI;  Lz = PI;  Re = 4000.0_dp;  dt = 2.0e-3_dp
  do ix = 1, nx
     kxh(ix) = real(ix-1, dp)*(2.0_dp*PI/Lx)              ! R2C kx half-axis
  end do
  do iz = 1, nz
     kk = iz-1;  if (kk > nz/2) kk = kk - nz
     kzh(iz) = real(kk, dp)*(2.0_dp*PI/Lz)                ! full kz (cuFFT order)
  end do
  alpha = 0.5_dp*(8.0_dp/15.0_dp)*dt/Re                   ! a representative substage a

  call helm_band_setup(n, sdh, sth, bdh, boff)
  kx_d = kxh;  kz_d = kzh;  sd_d = sdh;  st_d = sth;  bd_d = bdh

  call vif_gpu_setup_host(n, nx, nz, kxh, kzh, alpha, vmH, vpH, CiH)
  call vif_gpu_setup_dev (n, nx, nz, kx_d, kz_d, alpha, sd_d, st_d, bd_d, boff, vmG, vpG, CiG)

  ! _ws variant with caller-owned (garbage-seeded) scratch: same kernels
  ! in the same order on the same inputs -> must be BIT-EXACT vs the wrapper.
  wrhs = cmplx(huge(1.0_dp), huge(1.0_dp), dp);  wftl = cmplx(huge(1.0_dp), huge(1.0_dp), dp)
  wgsh = cmplx(huge(1.0_dp), huge(1.0_dp), dp);  wgpc = cmplx(huge(1.0_dp), huge(1.0_dp), dp)
  wgmc = cmplx(huge(1.0_dp), huge(1.0_dp), dp);  wvpc = cmplx(huge(1.0_dp), huge(1.0_dp), dp)
  wvmc = cmplx(huge(1.0_dp), huge(1.0_dp), dp)
  call vif_gpu_setup_dev_ws(n, nx, nz, kx_d, kz_d, alpha, sd_d, st_d, bd_d, boff, &
                            vmW, vpW, CiW, wrhs, wftl, wgsh, wgpc, wgmc, wvpc, wvmc)
  istat = cudaDeviceSynchronize()

  allocate(h1(0:n,nx,nz), h2(0:n,nx,nz), c1(2,2,nx,nz), c2(2,2,nx,nz))
  h1 = vmH;  h2 = vmG;  dvm = maxval(abs(h1 - h2))
  h1 = vpH;  h2 = vpG;  dvp = maxval(abs(h1 - h2))
  c1 = CiH;  c2 = CiG;  dci = maxval(abs(c1 - c2))
  h1 = vmW;  h2 = vmG;  wvm = maxval(abs(h1 - h2))
  h1 = vpW;  h2 = vpG;  wvp = maxval(abs(h1 - h2))
  c1 = CiW;  c2 = CiG;  wci = maxval(abs(c1 - c2))

  tol = 1.0e-8_dp
  print '(a,i0,a,i0,a,i0,a)', ' vif setup equivalence  (n=', n, ' nx=', nx, ' nz=', nz, ')'
  print '(a,es11.3)', '   max|vm_gpu  - vm_host |  = ', dvm
  print '(a,es11.3)', '   max|vp_gpu  - vp_host |  = ', dvp
  print '(a,es11.3)', '   max|Cinv_gpu- Cinv_host| = ', dci
  ok = (dvm < tol) .and. (dvp < tol) .and. (dci < tol)
  if (ok) then
     print '(a)', '   [PASS] batched on-device vif setup matches the host oracle'
  else
     print '(a)', '   [FAIL] batched on-device vif setup DIFFERS from the host oracle'
  end if
  print '(a,3es11.3)', '   max|_ws - self-alloc| (vm,vp,Cinv) = ', wvm, wvp, wci
  okw = (wvm == 0.0_dp) .and. (wvp == 0.0_dp) .and. (wci == 0.0_dp)
  if (okw) then
     print '(a)', '   [PASS] workspace-scratch setup is BIT-EXACT vs the self-allocating one'
  else
     print '(a)', '   [FAIL] workspace-scratch setup differs from the self-allocating one'
  end if
  if (.not. (ok .and. okw)) stop 1
end program test_vif_setup_gpu
