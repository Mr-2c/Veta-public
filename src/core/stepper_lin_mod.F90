!===============================================================================
! stepper_lin_mod.F90  --  LINEAR (nonlinear term off) viscous time step for the
! wall-normal vorticity eta, by the SMR three-substage Crank-Nicolson scheme.
! This is increment 6(a): it nails the stage loop + coefficient bookkeeping +
! Helmholtz wiring before v and the nonlinear forcing are added.
!
! Per substage k = 1..3 (symmetric CN: alpha_k = beta_k = c_k/2):
!     [1 - a_k L] eta^k = [1 + a_k L] eta^{k-1},     L = D^2 - k^2,
!   with a_k = beta_k * dt/Re,  beta_k = c_k/2,  c = (8/15, 2/15, 1/3),  sum c = 1.
!
! Rearranged into the cheb_shen / helm_band solve convention (-D^2 + mu)u = f:
!     (-D^2 + (k^2 + lam_k)) eta^k = lam_k * RHS,   lam_k = 1/a_k,
!     RHS = eta^{k-1} + a_k * L eta^{k-1}.
! So per substage:  Leta = apply_L(eta);  rhs = eta + a*Leta;
!   ftil = lam * project_shen(rhs);  etash = helm_band_solve(lam, ftil);
!   eta  = shen_to_cheb(etash).
!
! INDEX CARE: eta has n+1 Chebyshev coeffs (0:n) => apply_L gets ny = n+1;
! everything else (project/helm_band/shen_to_cheb) gets the degree n.
!
! Scratch is caller-owned (preallocated by the test / integrator workspace).
! kx,kz/eta are passed straight to apply_L (declared wp) and the solve helpers
! (declared dp); wp == dp == real64 here, so the kinds match. When wp is flipped
! to real32 a cast boundary will be needed (deferred, per the precision policy).
!
! Build:  nvfortran -cuda -c stepper_lin_mod.F90
!===============================================================================
module stepper_lin_mod
  use kinds,             only: dp, ip
  use cudafor
  use cheb_ops_mod,      only: apply_L
  use helm_band_gpu_mod, only: helm_band_solve_gpu
  use spec_helpers_mod,  only: project_shen, shen_to_cheb_c
  use vsolve_if_gpu_mod, only: vif_solve_gpu, vif_gpu_setup_host, vif_gpu_setup_dev, &
                               vif_gpu_setup_dev_ws
  use helm_band_mod,     only: helm_band_setup
  implicit none
  private
  public :: steplin_csub_ok                        ! [12.4] cross-module table assert
  public :: step_eta_linear, step_ev_linear, stepper_ev_setup, stepper_ev_setup_gpu, &
            stepper_ev_setup_gpu_ws

  ! SMR substep fractions c_k = gamma_k + zeta_k  (gamma=(8/15,5/12,3/4),
  ! zeta=(0,-17/60,-5/12)); they sum to 1.
  real(dp), parameter :: CSUB(3) = [ 8.0_dp/15.0_dp, 2.0_dp/15.0_dp, 1.0_dp/3.0_dp ]

contains

  !-- [12.4] cross-module SMR-table assert (see stepper_mod twin); this module
  !   carries only CSUB.
  pure logical function steplin_csub_ok(refc) result(okk)
    real(dp), intent(in) :: refc(3)
    okk = all(refc == CSUB)
  end function steplin_csub_ok

  !-- in-place complex scale by a real scalar:  x = x * s  (plain dummy => cuf-safe)
  subroutine cscale(m, nx, nz, x, s)
    integer(ip), intent(in) :: m, nx, nz
    real(dp),    intent(in) :: s
    complex(dp), device     :: x(0:m, nx, nz)
    integer(ip) :: ix, iz, k
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, m
             x(k,ix,iz) = x(k,ix,iz)*s
          end do
       end do
    end do
  end subroutine cscale

  !-- out = x + s*y  (complex x,y; real s; plain dummies => cuf-safe)
  subroutine cxpsy(m, nx, nz, x, s, y, out)
    integer(ip), intent(in) :: m, nx, nz
    real(dp),    intent(in) :: s
    complex(dp), device     :: x(0:m, nx, nz), y(0:m, nx, nz), out(0:m, nx, nz)
    integer(ip) :: ix, iz, k
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, m
             out(k,ix,iz) = x(k,ix,iz) + s*y(k,ix,iz)
          end do
       end do
    end do
  end subroutine cxpsy

  !-- one linear viscous step (three SMR/CN substages) for eta
  subroutine step_eta_linear(n, nx, nz, dt, Re, kx, kz, sd, st, bd, boff, eta, &
                             Leta, rhs, ftil, etash, scr)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: dt, Re, boff
    real(dp),    device :: kx(nx), kz(nz), sd(0:n-2), st(0:n-2), bd(0:n-2)
    complex(dp), device :: eta(0:n,nx,nz)                       ! in/out (Cheb coeffs)
    complex(dp), device :: Leta(0:n,nx,nz), rhs(0:n,nx,nz), scr(0:n,nx,nz)   ! scratch
    complex(dp), device :: ftil(0:n-2,nx,nz), etash(0:n-2,nx,nz)             ! scratch
    integer(ip) :: ks
    real(dp)    :: beta, a, lam

    do ks = 1, 3
       beta = 0.5_dp*CSUB(ks)
       a    = beta*dt/Re
       lam  = 1.0_dp/a
       call apply_L(n+1, nx, nz, kx, kz, eta, Leta, scr)    ! Leta = L eta^{k-1}
       call cxpsy(n, nx, nz, eta, a, Leta, rhs)             ! rhs  = eta + a*L eta
       call project_shen(n, nx, nz, rhs, ftil)              ! ftil = project(rhs)
       call cscale(n-2, nx, nz, ftil, lam)                  ! ftil = lam*project(rhs)
       call helm_band_solve_gpu(n, nx, nz, lam, kx, kz, sd, st, bd, boff, &
                                ftil, etash)                ! etash = eta^k (Shen)
       call shen_to_cheb_c(n, nx, nz, etash, eta)           ! eta   = eta^k (Cheb)
    end do
  end subroutine step_eta_linear

  !---------------------------------------------------------------------------
  ! stepper_ev_setup -- precompute the per-stage vif data (homogeneous solutions
  ! v_-, v_+ and inverse influence matrix Cinv) for the THREE substage values of
  ! alpha_k = beta_k*dt/Re, beta_k = c_k/2. Stored with a trailing stage index.
  ! Recompute whenever dt (or Re) changes -- this is the variable-dt hook. The
  ! Helmholtz band STRUCTURE (sdiag/stail/bdiag/boff) is alpha-independent and is
  ! set up once by the caller (helm_band_setup); only lambda varies per solve.
  ! kxh/kzh are HOST wavenumber arrays (vif_gpu_setup_host loops on the host).
  !---------------------------------------------------------------------------
  subroutine stepper_ev_setup(n, nx, nz, dt, Re, kxh, kzh, vm_all, vp_all, Cinv_all)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: dt, Re, kxh(nx), kzh(nz)
    real(dp),    device :: vm_all(0:n,nx,nz,3), vp_all(0:n,nx,nz,3), Cinv_all(2,2,nx,nz,3)
    integer(ip) :: ks
    real(dp)    :: a
    do ks = 1, 3
       a = 0.5_dp*CSUB(ks)*dt/Re
       call vif_gpu_setup_host(n, nx, nz, kxh, kzh, a, &
                               vm_all(:,:,:,ks), vp_all(:,:,:,ks), Cinv_all(:,:,:,:,ks))
    end do
  end subroutine stepper_ev_setup

  !---------------------------------------------------------------------------
  ! stepper_ev_setup_gpu -- drop-in replacement for stepper_ev_setup that builds
  ! the per-(kx,kz) v-solve operators (vm/vp/Cinv) with the FULLY ON-DEVICE batched
  ! vif_gpu_setup_dev instead of the host per-mode loop. Same signature (kxh,kzh are
  ! host); the band structure + wavenumbers are uploaded once here. This collapses
  ! the ~100-150 s host startup (and any variable-dt rebuild) to a few batched GPU
  ! kernels. Results match stepper_ev_setup to banded round-off (test_vif_setup_gpu).
  !---------------------------------------------------------------------------
  subroutine stepper_ev_setup_gpu(n, nx, nz, dt, Re, kxh, kzh, vm_all, vp_all, Cinv_all)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: dt, Re, kxh(nx), kzh(nz)
    real(dp),    device :: vm_all(0:n,nx,nz,3), vp_all(0:n,nx,nz,3), Cinv_all(2,2,nx,nz,3)
    real(dp), device, allocatable :: kx_d(:), kz_d(:), sd_d(:), st_d(:), bd_d(:)
    real(dp) :: sdh(0:n-2), sth(0:n-2), bdh(0:n-2), boff, a
    integer(ip) :: ks
    call helm_band_setup(n, sdh, sth, bdh, boff)             ! band structure (mu-independent)
    allocate(kx_d(nx), kz_d(nz), sd_d(0:n-2), st_d(0:n-2), bd_d(0:n-2))
    kx_d = kxh;  kz_d = kzh;  sd_d = sdh;  st_d = sth;  bd_d = bdh
    do ks = 1, 3
       a = 0.5_dp*CSUB(ks)*dt/Re
       call vif_gpu_setup_dev(n, nx, nz, kx_d, kz_d, a, sd_d, st_d, bd_d, boff, &
                              vm_all(:,:,:,ks), vp_all(:,:,:,ks), Cinv_all(:,:,:,:,ks))
    end do
    deallocate(kx_d, kz_d, sd_d, st_d, bd_d)
  end subroutine stepper_ev_setup_gpu

  !---------------------------------------------------------------------------
  ! stepper_ev_setup_gpu_ws -- same build, ZERO slab allocations: the
  ! seven slab-sized build temps are CALLER-OWNED (lend the stepper_ws per-step
  ! scratch, idle between steps). This is the variant for the driver's init AND
  ! mid-run variable-dt rebuilds -- the self-allocating one is a ~9 GB/GPU
  ! transient at production scale, enough to exhaust device memory on a 16-node
  ! run. Only the tiny band
  ! structure / wavenumber uploads (a few hundred KB) are allocated here.
  ! Shapes: rhs,gpc,gmc,vpc,vmc (0:n,nx,nz); ftil,gsh (0:n-2,nx,nz).
  !---------------------------------------------------------------------------
  subroutine stepper_ev_setup_gpu_ws(n, nx, nz, dt, Re, kxh, kzh, vm_all, vp_all, Cinv_all, &
                                     rhs, ftil, gsh, gpc, gmc, vpc, vmc)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: dt, Re, kxh(nx), kzh(nz)
    real(dp),    device :: vm_all(0:n,nx,nz,3), vp_all(0:n,nx,nz,3), Cinv_all(2,2,nx,nz,3)
    complex(dp), device :: rhs(0:n,nx,nz), ftil(0:n-2,nx,nz), gsh(0:n-2,nx,nz)
    complex(dp), device :: gpc(0:n,nx,nz), gmc(0:n,nx,nz), vpc(0:n,nx,nz), vmc(0:n,nx,nz)
    real(dp), device, allocatable :: kx_d(:), kz_d(:), sd_d(:), st_d(:), bd_d(:)
    real(dp) :: sdh(0:n-2), sth(0:n-2), bdh(0:n-2), boff, a
    integer(ip) :: ks
    call helm_band_setup(n, sdh, sth, bdh, boff)             ! band structure (mu-independent)
    allocate(kx_d(nx), kz_d(nz), sd_d(0:n-2), st_d(0:n-2), bd_d(0:n-2))
    kx_d = kxh;  kz_d = kzh;  sd_d = sdh;  st_d = sth;  bd_d = bdh
    do ks = 1, 3
       a = 0.5_dp*CSUB(ks)*dt/Re
       call vif_gpu_setup_dev_ws(n, nx, nz, kx_d, kz_d, a, sd_d, st_d, bd_d, boff, &
                                 vm_all(:,:,:,ks), vp_all(:,:,:,ks), Cinv_all(:,:,:,:,ks), &
                                 rhs, ftil, gsh, gpc, gmc, vpc, vmc)
    end do
    deallocate(kx_d, kz_d, sd_d, st_d, bd_d)
  end subroutine stepper_ev_setup_gpu_ws

  !---------------------------------------------------------------------------
  ! step_ev_linear -- one linear (N=0) viscous step advancing BOTH eta and v
  ! through the three SMR/CN substages.
  !   eta:  [1 - a_k L] eta^k = [1 + a_k L] eta^{k-1}                 (Helmholtz)
  !   v  :  [L - a_k L^2] v^k = L[1 + a_k L] v^{k-1}                  (biharmonic)
  ! with a_k = beta_k*dt/Re, beta_k = c_k/2. The biharmonic RHS is
  !   RHS_v = L v + a_k L^2 v   (two apply_L), solved by vif (alpha = a_k).
  ! Requires the per-stage vif precompute from stepper_ev_setup for this dt.
  ! Scratch is allocated internally (clean interface; preallocate in the
  ! integrator workspace later). vif_solve_gpu allocates its own internal scratch.
  !---------------------------------------------------------------------------
  subroutine step_ev_linear(n, nx, nz, dt, Re, kx, kz, sd, st, bd, boff, &
                            vm_all, vp_all, Cinv_all, eta, v)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: dt, Re, boff
    real(dp),    device :: kx(nx), kz(nz), sd(0:n-2), st(0:n-2), bd(0:n-2)
    real(dp),    device :: vm_all(0:n,nx,nz,3), vp_all(0:n,nx,nz,3), Cinv_all(2,2,nx,nz,3)
    complex(dp), device :: eta(0:n,nx,nz), v(0:n,nx,nz)        ! in/out (Cheb coeffs)
    complex(dp), device, allocatable :: Leta(:,:,:), rhs(:,:,:), scr(:,:,:)
    complex(dp), device, allocatable :: ftil(:,:,:), etash(:,:,:)
    complex(dp), device, allocatable :: Lv(:,:,:), L2v(:,:,:), rhsv(:,:,:)
    integer(ip) :: ks
    real(dp)    :: beta, a, lam

    allocate(Leta(0:n,nx,nz), rhs(0:n,nx,nz), scr(0:n,nx,nz))
    allocate(ftil(0:n-2,nx,nz), etash(0:n-2,nx,nz))
    allocate(Lv(0:n,nx,nz), L2v(0:n,nx,nz), rhsv(0:n,nx,nz))

    do ks = 1, 3
       beta = 0.5_dp*CSUB(ks)
       a    = beta*dt/Re
       lam  = 1.0_dp/a

       ! --- eta: Helmholtz ---
       call apply_L(n+1, nx, nz, kx, kz, eta, Leta, scr)
       call cxpsy(n, nx, nz, eta, a, Leta, rhs)
       call project_shen(n, nx, nz, rhs, ftil)
       call cscale(n-2, nx, nz, ftil, lam)
       call helm_band_solve_gpu(n, nx, nz, lam, kx, kz, sd, st, bd, boff, &
                                ftil, etash)
       call shen_to_cheb_c(n, nx, nz, etash, eta)

       ! --- v: biharmonic via influence matrix ---
       call apply_L(n+1, nx, nz, kx, kz, v,  Lv,  scr)        ! Lv  = L v^{k-1}
       call apply_L(n+1, nx, nz, kx, kz, Lv, L2v, scr)        ! L2v = L^2 v^{k-1}
       call cxpsy(n, nx, nz, Lv, a, L2v, rhsv)                ! RHS_v = L v + a L^2 v
       call vif_solve_gpu(n, nx, nz, a, kx, kz, sd, st, bd, boff, &
                          vm_all(:,:,:,ks), vp_all(:,:,:,ks), Cinv_all(:,:,:,:,ks), &
                          rhsv, v)                            ! v = v^k
    end do

    deallocate(Leta, rhs, scr, ftil, etash, Lv, L2v, rhsv)
  end subroutine step_ev_linear

end module stepper_lin_mod
