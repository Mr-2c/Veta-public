!===============================================================================
! vsolve_if_gpu_mod.F90  --  clamped v-solve on the GPU by the two-Helmholtz
! factorization + influence matrix (the GPU port of vsolve_if_mod).
!
! Per mode (kx,kz), per solve, in Chebyshev coefficients:
!   1. F = -f/alpha :  ftil = project(f)/alpha ; H1 (lambda=0, mu=k^2) banded
!      Helmholtz solve -> g_p (Shen) -> g_p (Cheb).
!   2. H2 v_p = g_p :  ftil = -project(g_p) ; H2 (lambda=1/alpha, mu=k^2+1/alpha)
!      banded Helmholtz solve -> v_p (Shen) -> v_p (Cheb).
!   3. wall derivatives Dv_p(+-1).
!   4. closure  sigma = -Cinv [Dv_p(-1); Dv_p(+1)] ;  v = v_p + s1 v_- + s2 v_+ .
!
! The two interior solves reuse the validated banded Helmholtz GPU kernel; the
! projection/reconstruction/wall-derivative steps reuse the validated spectral
! helpers; the homogeneous solutions v_-, v_+ (real Cheb coeffs per mode) and the
! inverse influence matrix Cinv (2x2 per mode) are precomputed on the host with
! the validated CPU vif_setup and uploaded. Recompute the precompute when alpha
! (dt) changes; the per-step solve never re-derives them.
!
! Scratch is allocated per call here for a clean interface; the integrator will
! hold it in a preallocated workspace.
!
! Build:  nvfortran -cuda -c vsolve_if_gpu_mod.F90
!===============================================================================
module vsolve_if_gpu_mod
  use kinds,             only: dp, ip
  use cudafor
  use helm_band_gpu_mod, only: helm_band_solve_gpu
  use spec_helpers_mod,  only: project_shen, shen_to_cheb_c, wall_derivs_c
  use vsolve_if_mod,     only: vif_t, vif_setup_band
  use helm_band_mod,     only: helm_band_setup
  implicit none
  private
  public :: vif_gpu_setup_host, vif_gpu_setup_dev, vif_gpu_setup_dev_ws, &
            vif_solve_gpu, vif_solve_gpu_ws

contains

  !---------------------------------------------------------------------------
  ! Host precompute of the per-mode homogeneous solutions and inverse influence
  ! matrix, via the validated CPU vif_setup, uploaded to device arrays.
  !---------------------------------------------------------------------------
  subroutine vif_gpu_setup_host(n, nx, nz, kxh, kzh, alpha, vm_d, vp_d, Cinv_d)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: kxh(nx), kzh(nz), alpha
    real(dp), device        :: vm_d(0:n,nx,nz), vp_d(0:n,nx,nz), Cinv_d(2,2,nx,nz)
    real(dp), allocatable :: vm_h(:,:,:), vp_h(:,:,:), Cinv_h(:,:,:,:)
    type(vif_t) :: d
    real(dp)    :: k2
    real(dp)    :: sdh(0:n-2), sth(0:n-2), bdh(0:n-2), boffh
    integer(ip) :: ix, iz
    allocate(vm_h(0:n,nx,nz), vp_h(0:n,nx,nz), Cinv_h(2,2,nx,nz))
    ! Band structure once (mu-independent), then O(n) banded setups
    ! per mode instead of 4 dense O(n^3) solves each: startup drops from
    ! minutes (and ~30+ min at 128^2) to well under a second. Same math; the
    ! banded-vs-dense agreement (~1e-13, test_helm_band) is far inside
    ! test_vif_gpu's 1e-8 tolerance.
    call helm_band_setup(n, sdh, sth, bdh, boffh)
    do iz = 1, nz
       do ix = 1, nx
          k2 = kxh(ix)**2 + kzh(iz)**2
          call vif_setup_band(n, k2, alpha, sdh, sth, bdh, boffh, d)
          vm_h(:,ix,iz)     = d%vm
          vp_h(:,ix,iz)     = d%vp
          Cinv_h(:,:,ix,iz) = d%Cinv
       end do
    end do
    vm_d = vm_h; vp_d = vp_h; Cinv_d = Cinv_h
    deallocate(vm_h, vp_h, Cinv_h)
  end subroutine vif_gpu_setup_host

  !---------------------------------------------------------------------------
  ! vif_gpu_setup_dev -- FULLY ON-DEVICE, BATCHED port of vif_gpu_setup_host:
  ! builds the per-mode homogeneous solutions v_-, v_+ (real Cheb coeffs) and the
  ! 2x2 inverse influence matrix Cinv for ALL (kx,kz) at once, replacing the host
  ! per-mode loop (the ~100-150 s startup cost) with a handful of batched kernels.
  !
  ! It mirrors vif_setup_band EXACTLY, reusing the validated per-step batched ops:
  !   l_+ = (T0+T1)/2 , l_- = (T0-T1)/2  (lift functions)
  !   g_+/- : (-D^2 + mu1) ghat = -k2 l_+/- ,  mu1 = k2          (helm lambda = 0)
  !           g = shen_to_cheb(ghat) + l
  !   v_+/- : (-D^2 + mu2) v = -g ,  mu2 = k2 + 1/alpha          (helm lambda = 1/alpha)
  !   Cinv  = inv[ Dv_-(-1) Dv_+(-1) ; Dv_-(+1) Dv_+(+1) ]
  ! The setup quantities are REAL; they are carried as complex with a zero
  ! imaginary part (exact, since every input is real and every op is linear) so the
  ! existing complex project_shen / helm_band_solve_gpu / shen_to_cheb_c / cscale /
  ! wall_derivs_c apply unchanged; the real part is extracted at the end. Result
  ! matches the host setup to banded-solve round-off (test_vif_setup_gpu).
  !
  ! kx,kz,sdiag,stail,bdiag are DEVICE arrays (caller uploads).
  !
  ! WORKSPACE variant: the SEVEN slab-sized complex(dp) build temps are
  ! CALLER-OWNED (at production scale each is ~1.3 GB/GPU -- self-allocating all
  ! seven is a ~9 GB transient, which runs a 16-node job out of memory at the
  ! first mid-run variable-dt rebuild). Every temp is fully written
  ! before it is read, so any correctly-shaped scratch (e.g. the stepper_ws
  ! per-step buffers, idle between steps) can be lent. Only the four (nx,nz)
  ! wall-derivative rows (a few MB total) are still self-allocated.
  ! Shapes: rhs,gpc,gmc,vpc,vmc (0:n,nx,nz); ftil,gsh (0:n-2,nx,nz).
  !---------------------------------------------------------------------------
  subroutine vif_gpu_setup_dev_ws(n, nx, nz, kx, kz, alpha, sdiag, stail, bdiag, boff, &
                                  vm_d, vp_d, Cinv_d, rhs, ftil, gsh, gpc, gmc, vpc, vmc)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: alpha, boff
    real(dp), device :: kx(nx), kz(nz), sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2)
    real(dp), device :: vm_d(0:n,nx,nz), vp_d(0:n,nx,nz), Cinv_d(2,2,nx,nz)
    complex(dp), device :: rhs(0:n,nx,nz), ftil(0:n-2,nx,nz), gsh(0:n-2,nx,nz)
    complex(dp), device :: gpc(0:n,nx,nz), gmc(0:n,nx,nz), vpc(0:n,nx,nz), vmc(0:n,nx,nz)
    complex(dp), device, allocatable :: dlo_m(:,:), dhi_m(:,:), dlo_p(:,:), dhi_p(:,:)
    real(dp) :: inva
    inva = 1.0_dp/alpha
    allocate(dlo_m(nx,nz), dhi_m(nx,nz), dlo_p(nx,nz), dhi_p(nx,nz))

    ! --- g_+ : H1 ghat = -k2 l_+ , then g_+ = shen_to_cheb(ghat) + l_+ ----------
    call vif_lift_rhs(n, nx, nz, kx, kz, 1.0_dp, rhs)            ! rhs = -k2 l_+
    call project_shen(n, nx, nz, rhs, ftil)
    call helm_band_solve_gpu(n, nx, nz, 0.0_dp, kx, kz, sdiag, stail, bdiag, boff, &
                             ftil, gsh)
    call shen_to_cheb_c(n, nx, nz, gsh, gpc)
    call vif_add_lift(n, nx, nz, 1.0_dp, gpc)                    ! gpc += l_+
    ! --- g_- ------------------------------------------------------------------
    call vif_lift_rhs(n, nx, nz, kx, kz, -1.0_dp, rhs)          ! rhs = -k2 l_-
    call project_shen(n, nx, nz, rhs, ftil)
    call helm_band_solve_gpu(n, nx, nz, 0.0_dp, kx, kz, sdiag, stail, bdiag, boff, &
                             ftil, gsh)
    call shen_to_cheb_c(n, nx, nz, gsh, gmc)
    call vif_add_lift(n, nx, nz, -1.0_dp, gmc)                   ! gmc += l_-
    ! --- v_+ : H2 v_+ = -g_+   (mu2 = k2 + 1/alpha) ---------------------------
    call project_shen(n, nx, nz, gpc, ftil)
    call cscale(n-2, nx, nz, ftil, -1.0_dp)                      ! -project(g_+)
    call helm_band_solve_gpu(n, nx, nz, inva, kx, kz, sdiag, stail, bdiag, boff, &
                             ftil, gsh)
    call shen_to_cheb_c(n, nx, nz, gsh, vpc)
    ! --- v_- ------------------------------------------------------------------
    call project_shen(n, nx, nz, gmc, ftil)
    call cscale(n-2, nx, nz, ftil, -1.0_dp)
    call helm_band_solve_gpu(n, nx, nz, inva, kx, kz, sdiag, stail, bdiag, boff, &
                             ftil, gsh)
    call shen_to_cheb_c(n, nx, nz, gsh, vmc)

    ! --- real homogeneous solutions + influence matrix ------------------------
    call vif_extract_real(n, nx, nz, vpc, vp_d)
    call vif_extract_real(n, nx, nz, vmc, vm_d)
    call wall_derivs_c(n, nx, nz, vmc, dlo_m, dhi_m)             ! Dv_-(-1), Dv_-(+1)
    call wall_derivs_c(n, nx, nz, vpc, dlo_p, dhi_p)             ! Dv_+(-1), Dv_+(+1)
    call vif_cinv(nx, nz, dlo_m, dhi_m, dlo_p, dhi_p, Cinv_d)

    deallocate(dlo_m, dhi_m, dlo_p, dhi_p)
  end subroutine vif_gpu_setup_dev_ws

  !---------------------------------------------------------------------------
  ! SELF-ALLOCATING wrapper (the original interface, kept for the setup tests
  ! and the small --validate builders). NOT for mid-run rebuilds at production
  ! scale: the seven slab temps below are a multi-GB transient -- use the _ws
  ! variant with lent scratch there (see vif_gpu_setup_dev_ws header).
  !---------------------------------------------------------------------------
  subroutine vif_gpu_setup_dev(n, nx, nz, kx, kz, alpha, sdiag, stail, bdiag, boff, &
                               vm_d, vp_d, Cinv_d)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: alpha, boff
    real(dp), device :: kx(nx), kz(nz), sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2)
    real(dp), device :: vm_d(0:n,nx,nz), vp_d(0:n,nx,nz), Cinv_d(2,2,nx,nz)
    complex(dp), device, allocatable :: rhs(:,:,:), ftil(:,:,:), gsh(:,:,:)
    complex(dp), device, allocatable :: gpc(:,:,:), gmc(:,:,:), vpc(:,:,:), vmc(:,:,:)
    allocate(rhs(0:n,nx,nz), ftil(0:n-2,nx,nz), gsh(0:n-2,nx,nz))
    allocate(gpc(0:n,nx,nz), gmc(0:n,nx,nz), vpc(0:n,nx,nz), vmc(0:n,nx,nz))
    call vif_gpu_setup_dev_ws(n, nx, nz, kx, kz, alpha, sdiag, stail, bdiag, boff, &
                              vm_d, vp_d, Cinv_d, rhs, ftil, gsh, gpc, gmc, vpc, vmc)
    deallocate(rhs, ftil, gsh, gpc, gmc, vpc, vmc)
  end subroutine vif_gpu_setup_dev

  !-- lift RHS: rhs = -k2 * l, with l_+ = (T0+T1)/2 (s=+1), l_- = (T0-T1)/2 (s=-1)
  subroutine vif_lift_rhs(n, nx, nz, kx, kz, s, rhs)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: s
    real(dp),    device     :: kx(nx), kz(nz)
    complex(dp), device     :: rhs(0:n,nx,nz)
    integer(ip) :: ix, iz, j
    real(dp)    :: k2
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do j = 0, n
             rhs(j,ix,iz) = (0.0_dp,0.0_dp)
          end do
       end do
    end do
    !$cuf kernel do(2) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          k2 = kx(ix)*kx(ix) + kz(iz)*kz(iz)
          rhs(0,ix,iz) = cmplx(-k2*0.5_dp,    0.0_dp, dp)
          rhs(1,ix,iz) = cmplx(-k2*0.5_dp*s,  0.0_dp, dp)
       end do
    end do
  end subroutine vif_lift_rhs

  !-- add the lift l (coeffs T0=0.5, T1=0.5*s) to the Cheb-coeff field gc
  subroutine vif_add_lift(n, nx, nz, s, gc)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: s
    complex(dp), device     :: gc(0:n,nx,nz)
    integer(ip) :: ix, iz
    !$cuf kernel do(2) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          gc(0,ix,iz) = gc(0,ix,iz) + cmplx(0.5_dp,    0.0_dp, dp)
          gc(1,ix,iz) = gc(1,ix,iz) + cmplx(0.5_dp*s,  0.0_dp, dp)
       end do
    end do
  end subroutine vif_add_lift

  !-- extract the real part (the setup quantities are real; imag is exactly 0)
  subroutine vif_extract_real(n, nx, nz, csrc, rdst)
    integer(ip), intent(in) :: n, nx, nz
    complex(dp), device :: csrc(0:n,nx,nz)
    real(dp),    device :: rdst(0:n,nx,nz)
    integer(ip) :: ix, iz, j
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do j = 0, n
             rdst(j,ix,iz) = real(csrc(j,ix,iz), dp)
          end do
       end do
    end do
  end subroutine vif_extract_real

  !-- per-mode 2x2 inverse of C = [Dv_-(-1) Dv_+(-1); Dv_-(+1) Dv_+(+1)] -> Cinv
  subroutine vif_cinv(nx, nz, dlo_m, dhi_m, dlo_p, dhi_p, Cinv)
    integer(ip), intent(in) :: nx, nz
    complex(dp), device :: dlo_m(nx,nz), dhi_m(nx,nz), dlo_p(nx,nz), dhi_p(nx,nz)
    real(dp),    device :: Cinv(2,2,nx,nz)
    integer(ip) :: ix, iz
    real(dp) :: dvm_lo, dvm_hi, dvp_lo, dvp_hi, det
    !$cuf kernel do(2) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          dvm_lo = real(dlo_m(ix,iz), dp);  dvm_hi = real(dhi_m(ix,iz), dp)
          dvp_lo = real(dlo_p(ix,iz), dp);  dvp_hi = real(dhi_p(ix,iz), dp)
          det = dvm_lo*dvp_hi - dvp_lo*dvm_hi
          Cinv(1,1,ix,iz) =  dvp_hi/det;  Cinv(1,2,ix,iz) = -dvp_lo/det
          Cinv(2,1,ix,iz) = -dvm_hi/det;  Cinv(2,2,ix,iz) =  dvm_lo/det
       end do
    end do
  end subroutine vif_cinv

  !---------------------------------------------------------------------------
  ! In-place complex scale by a real scalar (plain device dummy -> cuf-safe).
  !---------------------------------------------------------------------------
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

  !---------------------------------------------------------------------------
  ! Closure + combine:  sigma = -Cinv [Dv_p(-1); Dv_p(+1)],
  !                      v = v_p + sigma1 v_- + sigma2 v_+ .
  !---------------------------------------------------------------------------
  attributes(global) subroutine vif_closure_kernel(n, nx, nz, vpc, dlo, dhi, &
                                                    Cinv, vm, vp, v)
    integer(ip), value :: n, nx, nz
    complex(dp) :: vpc(0:n,nx,nz), v(0:n,nx,nz)
    complex(dp) :: dlo(nx,nz), dhi(nx,nz)
    real(dp)    :: Cinv(2,2,nx,nz), vm(0:n,nx,nz), vp(0:n,nx,nz)
    integer(ip) :: ix, iz, j
    complex(dp) :: s1, s2
    ix = (blockIdx%x-1)*blockDim%x + threadIdx%x
    iz = (blockIdx%y-1)*blockDim%y + threadIdx%y
    if (ix > nx .or. iz > nz) return
    s1 = -( Cinv(1,1,ix,iz)*dlo(ix,iz) + Cinv(1,2,ix,iz)*dhi(ix,iz) )
    s2 = -( Cinv(2,1,ix,iz)*dlo(ix,iz) + Cinv(2,2,ix,iz)*dhi(ix,iz) )
    do j = 0, n
       v(j,ix,iz) = vpc(j,ix,iz) + s1*vm(j,ix,iz) + s2*vp(j,ix,iz)
    end do
  end subroutine vif_closure_kernel

  !---------------------------------------------------------------------------
  ! Per-step solve, WORKSPACE variant: all scratch is caller-owned
  ! (plain device arrays, house style). Shapes: ftil,gsh,vsh (0:n-2,nx,nz);
  ! gc,vpc (0:n,nx,nz); dlo,dhi (nx,nz). No allocation in here.
  !---------------------------------------------------------------------------
  subroutine vif_solve_gpu_ws(n, nx, nz, alpha, kx, kz, sdiag, stail, bdiag, boff, &
                              vm, vp, Cinv, fc, v, &
                              ftil, gsh, gc, vsh, vpc, dlo, dhi)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: alpha, boff
    real(dp), device :: kx(nx), kz(nz), sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2)
    real(dp), device :: vm(0:n,nx,nz), vp(0:n,nx,nz), Cinv(2,2,nx,nz)
    complex(dp), device :: fc(0:n,nx,nz)     ! in : RHS (Cheb coeffs)
    complex(dp), device :: v(0:n,nx,nz)      ! out: v (Cheb coeffs)
    complex(dp), device :: ftil(0:n-2,nx,nz), gsh(0:n-2,nx,nz), gc(0:n,nx,nz)
    complex(dp), device :: vsh(0:n-2,nx,nz), vpc(0:n,nx,nz)
    complex(dp), device :: dlo(nx,nz), dhi(nx,nz)
    type(dim3) :: grid, block
    integer(cuda_stream_kind) :: strm      ! explicit stream for chevron launches
    real(dp) :: inva

    inva = 1.0_dp/alpha

    ! step 1: H1 g_p = -f/alpha   (mu = k^2, lambda = 0)
    call project_shen(n, nx, nz, fc, ftil)
    call cscale(n-2, nx, nz, ftil, inva)             ! project(f)/alpha
    call helm_band_solve_gpu(n, nx, nz, 0.0_dp, kx, kz, sdiag, stail, bdiag, boff, &
                             ftil, gsh)
    call shen_to_cheb_c(n, nx, nz, gsh, gc)

    ! step 2: H2 v_p = g_p        (mu = k^2 + 1/alpha, lambda = 1/alpha)
    call project_shen(n, nx, nz, gc, ftil)
    call cscale(n-2, nx, nz, ftil, -1.0_dp)          ! -project(g_p)
    call helm_band_solve_gpu(n, nx, nz, inva, kx, kz, sdiag, stail, bdiag, boff, &
                             ftil, vsh)
    call shen_to_cheb_c(n, nx, nz, vsh, vpc)

    ! step 3: wall derivatives of v_p
    call wall_derivs_c(n, nx, nz, vpc, dlo, dhi)

    ! step 4: closure + combine
    block = dim3(16,16,1); grid = dim3((nx+15)/16, (nz+15)/16, 1)
    strm = cudaforGetDefaultStream()
    call vif_closure_kernel<<<grid,block,0,strm>>>(n, nx, nz, vpc, dlo, dhi, Cinv, vm, vp, v)
  end subroutine vif_solve_gpu_ws

  !---------------------------------------------------------------------------
  ! Per-step solve, SELF-ALLOCATING wrapper (the original interface, kept for
  ! the existing tests / stepper_lin_mod). The hot path (stepper_mod) uses the
  ! _ws variant with preallocated scratch instead.
  !---------------------------------------------------------------------------
  subroutine vif_solve_gpu(n, nx, nz, alpha, kx, kz, sdiag, stail, bdiag, boff, &
                           vm, vp, Cinv, fc, v)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: alpha, boff
    real(dp), device :: kx(nx), kz(nz), sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2)
    real(dp), device :: vm(0:n,nx,nz), vp(0:n,nx,nz), Cinv(2,2,nx,nz)
    complex(dp), device :: fc(0:n,nx,nz)     ! in : RHS (Cheb coeffs)
    complex(dp), device :: v(0:n,nx,nz)      ! out: v (Cheb coeffs)
    complex(dp), device, allocatable :: ftil(:,:,:), gsh(:,:,:), gc(:,:,:), vsh(:,:,:), vpc(:,:,:)
    complex(dp), device, allocatable :: dlo(:,:), dhi(:,:)

    allocate(ftil(0:n-2,nx,nz), gsh(0:n-2,nx,nz), gc(0:n,nx,nz), &
             vsh(0:n-2,nx,nz), vpc(0:n,nx,nz))
    allocate(dlo(nx,nz), dhi(nx,nz))
    call vif_solve_gpu_ws(n, nx, nz, alpha, kx, kz, sdiag, stail, bdiag, boff, &
                          vm, vp, Cinv, fc, v, &
                          ftil, gsh, gc, vsh, vpc, dlo, dhi)
    deallocate(ftil, gsh, gc, vsh, vpc, dlo, dhi)
  end subroutine vif_solve_gpu

end module vsolve_if_gpu_mod
