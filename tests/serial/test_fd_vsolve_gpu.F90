!===============================================================================
! test_fd_vsolve_gpu.F90 -- gate for the batched on-device FD
! vif machine (fd_vsolve_gpu_mod) against the HOST oracle (fd_vsolve_mod,
! itself cross-discretization-gated vs the Chebyshev truth in test_fd_vsolve):
!   (1) setup equivalence : GPU vm/vp/Cinv vs per-mode host fd_vif_setup
!       (rel 1e-8, the Cheb test_vif_setup_gpu posture; the GPU runs the SAME
!       assemble/LU recurrence so the real gap is FMA/order-level);
!   (2) ws BIT-EXACTNESS  : the caller-scratch _ws variants vs the
!       self-allocating wrappers (identical code path, different memory --
!       any nonzero diff means scratch is read before written);
!   (3) solve equivalence : GPU fd_vif_solve_gpu on a deterministic complex
!       RHS vs the host Re/Im per-mode chain (rel 1e-8);
!   (4) exact wall zeros  : vm/vp and the closed v carry EXACT +-0 walls on
!       the GPU (solver-forced walls survive unpack + closure bitwise).
! Two alphas: 1e-3 (moderate) and 1e-6 (production-small; stiff mu2). Modes
! include kx = kz = 0 (the mu = 0 Dirichlet-Laplacian corner) and mixed-sign
! kz. Grid GLC(48) DESCENDING, kl = 4. The PACKED setup (+ family in Re,
! - family in Im, real band => componentwise-exact) is what makes GPU==host
! a real check: any Re/Im slot or sign error lands ~O(1), not in the noise.
!===============================================================================
program test_fd_vsolve_gpu
  use kinds,            only: dp, ip
  use cudafor
  use fdops_mod,        only: fd_build_bands, fd_wall_weights
  use fd_vsolve_mod,    only: fd_vif_setup, fd_vif_solve
  use fd_vsolve_gpu_mod, only: fd_vif_setup_gpu, fd_vif_setup_gpu_ws, &
                               fd_vif_solve_gpu, fd_vif_solve_gpu_ws
  implicit none
  integer(ip), parameter :: n = 48, ny = n + 1, kl = 4, nx = 24, nz = 24
  real(dp), parameter :: PI  = 3.141592653589793238462643_dp
  real(dp), parameter :: TOL = 1.0e-8_dp
  real(dp)    :: y(ny), d1b(2*kl+1, ny), d2b(2*kl+1, ny)
  real(dp)    :: wlo(ny), whi(ny), kxh(nx), kzh(nz)
  complex(dp) :: fh(0:n, nx, nz)
  ! device inputs
  real(dp),    device :: kx_d(nx), kz_d(nz), y_d(0:n), d2b_d(2*kl+1, 0:n)
  real(dp),    device :: wlo_d(0:n), whi_d(0:n)
  complex(dp), device :: f_d(0:n, nx, nz)
  ! device results (self-allocating vs _ws pair)
  real(dp),    device :: vm_d(0:n,nx,nz), vp_d(0:n,nx,nz), Ci_d(2,2,nx,nz)
  real(dp),    device :: vm2_d(0:n,nx,nz), vp2_d(0:n,nx,nz), Ci2_d(2,2,nx,nz)
  complex(dp), device :: v_d(0:n,nx,nz), v2_d(0:n,nx,nz)
  ! caller-owned scratch for the _ws variants
  complex(dp), device :: rhs_s(0:n,nx,nz), g_s(0:n,nx,nz), vh_s(0:n,nx,nz)
  complex(dp), device :: fsc_s(0:n,nx,nz), gp_s(0:n,nx,nz), vpc_s(0:n,nx,nz)
  complex(dp), device :: dlo_s(nx,nz), dhi_s(nx,nz)
  integer(ip) :: i, j, ix, iz, kk
  logical     :: allok
  allok = .true.

  do j = 0, n
     y(j+1) = cos(PI*real(j,dp)/real(n,dp))
  end do
  y(1) = 1.0_dp;  y(ny) = -1.0_dp
  call fd_build_bands(ny, kl, y, d1b, d2b)      ! d1b unused (builder emits both)
  call fd_wall_weights(ny, kl, y, wlo, whi)
  do i = 1, nx
     kxh(i) = 0.7_dp*real(i-1, dp)
  end do
  do i = 1, nz
     kk = i - 1;  if (kk > nz/2) kk = kk - nz
     kzh(i) = 1.3_dp*real(kk, dp)
  end do
  y_d = y;  d2b_d = d2b;  wlo_d = wlo;  whi_d = whi
  kx_d = kxh;  kz_d = kzh
  do iz = 1, nz; do ix = 1, nx; do j = 0, n
     fh(j, ix, iz) = cmplx(fval(j, ix, iz), -0.5_dp*fval(j, iz, ix+1), dp)
  end do; end do; end do
  f_d = fh

  print '(a,i0,a,i0,a,i0,a)', ' fd vif GPU equivalence  (n=', n, ' nx=', nx, &
        ' nz=', nz, ')'
  call run_alpha(1.0e-3_dp, 'alpha=1e-3')
  call run_alpha(1.0e-6_dp, 'alpha=1e-6')

  if (allok) then; print '(a)', 'ALL TESTS PASSED'
  else;            print '(a)', 'SOME TESTS FAILED'; end if
  if (.not. allok) error stop 1

contains

  subroutine run_alpha(alpha, lab)
    real(dp),     intent(in) :: alpha
    character(*), intent(in) :: lab
    real(dp)    :: vmg(0:n,nx,nz), vpg(0:n,nx,nz), Cig(2,2,nx,nz)
    real(dp)    :: vmg2(0:n,nx,nz), vpg2(0:n,nx,nz), Cig2(2,2,nx,nz)
    complex(dp) :: vg(0:n,nx,nz), vg2(0:n,nx,nz)
    real(dp)    :: vmh(ny), vph(ny), Cih(2,2), fre(ny), fim(ny), vre(ny), vim(ny)
    real(dp)    :: k2, dvm, dvp, dci, wvm, wvp, wci, dv, wv, e
    complex(dp) :: vhost
    integer(ip) :: ix, iz, j, istat
    logical     :: okk, wall0, hok

    ! --- seed the _ws scratch with garbage (house convention, cf.
    !     test_vif_setup_gpu): BIT-EXACT then PROVES fully-written-before-read
    rhs_s = cmplx(huge(1.0_dp), huge(1.0_dp), dp)
    g_s   = cmplx(huge(1.0_dp), huge(1.0_dp), dp)
    vh_s  = cmplx(huge(1.0_dp), huge(1.0_dp), dp)
    fsc_s = cmplx(huge(1.0_dp), huge(1.0_dp), dp)
    gp_s  = cmplx(huge(1.0_dp), huge(1.0_dp), dp)
    vpc_s = cmplx(huge(1.0_dp), huge(1.0_dp), dp)
    dlo_s = cmplx(huge(1.0_dp), huge(1.0_dp), dp)
    dhi_s = cmplx(huge(1.0_dp), huge(1.0_dp), dp)

    ! --- GPU: self-allocating and _ws setup on the same inputs ----------------
    call fd_vif_setup_gpu(n, kl, nx, nz, kx_d, kz_d, alpha, y_d, d2b_d, &
                          wlo_d, whi_d, vm_d, vp_d, Ci_d)
    call fd_vif_setup_gpu_ws(n, kl, nx, nz, kx_d, kz_d, alpha, y_d, d2b_d, &
                             wlo_d, whi_d, vm2_d, vp2_d, Ci2_d, rhs_s, g_s, vh_s)
    ! --- GPU: self-allocating and _ws solve -----------------------------------
    call fd_vif_solve_gpu(n, kl, nx, nz, alpha, kx_d, kz_d, d2b_d, wlo_d, whi_d, &
                          vm_d, vp_d, Ci_d, f_d, v_d)
    call fd_vif_solve_gpu_ws(n, kl, nx, nz, alpha, kx_d, kz_d, d2b_d, wlo_d, whi_d, &
                             vm_d, vp_d, Ci_d, f_d, v2_d, fsc_s, gp_s, vpc_s, &
                             dlo_s, dhi_s)
    istat = cudaDeviceSynchronize()
    vmg = vm_d;  vpg = vp_d;  Cig = Ci_d
    vmg2 = vm2_d;  vpg2 = vp2_d;  Cig2 = Ci2_d
    vg = v_d;  vg2 = v2_d

    ! --- ws bit-exactness ------------------------------------------------------
    e = maxval(abs(vmg2 - vmg)) + maxval(abs(vpg2 - vpg)) + maxval(abs(Cig2 - Cig))
    print '(a,a,a,es12.3)', '   [', lab, '] max|ws - self-alloc| setup  =', e
    call report(lab//': ws setup BIT-EXACT vs self-allocating', e == 0.0_dp)
    e = maxval(abs(vg2 - vg))
    print '(a,a,a,es12.3)', '   [', lab, '] max|ws - self-alloc| solve  =', e
    call report(lab//': ws solve BIT-EXACT vs self-allocating', e == 0.0_dp)

    ! --- host oracle per mode: setup + Re/Im solve -----------------------------
    dvm = 0.0_dp;  dvp = 0.0_dp;  dci = 0.0_dp
    wvm = 0.0_dp;  wvp = 0.0_dp;  wci = 0.0_dp
    dv  = 0.0_dp;  wv  = 0.0_dp
    hok = .true.
    do iz = 1, nz
       do ix = 1, nx
          k2 = kxh(ix)**2 + kzh(iz)**2
          call fd_vif_setup(ny, kl, y, d2b, wlo, whi, k2, alpha, vmh, vph, Cih, okk)
          if (.not. okk) then
             hok = .false.;  cycle
          end if
          do j = 1, ny
             wvm = max(wvm, abs(vmh(j)));  dvm = max(dvm, abs(vmg(j-1,ix,iz) - vmh(j)))
             wvp = max(wvp, abs(vph(j)));  dvp = max(dvp, abs(vpg(j-1,ix,iz) - vph(j)))
          end do
          wci = max(wci, maxval(abs(Cih)))
          dci = max(dci, maxval(abs(Cig(:,:,ix,iz) - Cih)))
          do j = 1, ny
             fre(j) = real(fh(j-1,ix,iz));  fim(j) = aimag(fh(j-1,ix,iz))
          end do
          call fd_vif_solve(ny, kl, d2b, wlo, whi, k2, alpha, vmh, vph, Cih, &
                            fre, vre, okk)
          if (.not. okk) hok = .false.
          call fd_vif_solve(ny, kl, d2b, wlo, whi, k2, alpha, vmh, vph, Cih, &
                            fim, vim, okk)
          if (.not. okk) hok = .false.
          do j = 1, ny
             vhost = cmplx(vre(j), vim(j), dp)
             wv = max(wv, abs(vhost))
             dv = max(dv, abs(vg(j-1,ix,iz) - vhost))
          end do
       end do
    end do
    call report(lab//': host oracle setup/solve accepted every mode', hok)
    print '(a,a,a,3es12.3)', '   [', lab, '] setup rel dvm/dvp/dCinv     =', &
          dvm/max(1.0e-300_dp, wvm), dvp/max(1.0e-300_dp, wvp), &
          dci/max(1.0e-300_dp, wci)
    call report(lab//': on-device vif setup matches host oracle', &
                dvm <= TOL*wvm .and. dvp <= TOL*wvp .and. dci <= TOL*wci)
    print '(a,a,a,es12.3)', '   [', lab, '] solve rel d|v|              =', &
          dv/max(1.0e-300_dp, wv)
    call report(lab//': on-device vif solve matches host oracle', dv <= TOL*wv)

    ! --- exact wall zeros on the GPU results -----------------------------------
    wall0 = .true.
    do iz = 1, nz
       do ix = 1, nx
          if (vmg(0,ix,iz) /= 0.0_dp .or. vmg(n,ix,iz) /= 0.0_dp) wall0 = .false.
          if (vpg(0,ix,iz) /= 0.0_dp .or. vpg(n,ix,iz) /= 0.0_dp) wall0 = .false.
          if (real(vg(0,ix,iz)) /= 0.0_dp .or. aimag(vg(0,ix,iz)) /= 0.0_dp) &
             wall0 = .false.
          if (real(vg(n,ix,iz)) /= 0.0_dp .or. aimag(vg(n,ix,iz)) /= 0.0_dp) &
             wall0 = .false.
       end do
    end do
    call report(lab//': GPU vm/vp/v walls exactly zero', wall0)
  end subroutine run_alpha

  real(dp) function fval(j, a, b) result(v)
    integer(ip), intent(in) :: j, a, b
    integer(ip) :: h
    h = modulo(j*1009_ip + a*2027_ip + b*3001_ip + j*a*17_ip, 100003_ip)
    v = real(h, dp)/100003.0_dp - 0.5_dp
  end function fval

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. okk) allok = .false.
  end subroutine report

end program test_fd_vsolve_gpu
