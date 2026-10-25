!===============================================================================
! fd_vsolve_gpu_mod.F90 -- the clamped v-solve on the GPU for
! the FD-y backend: the batched device twin of fd_vsolve_mod, playing exactly
! the role vsolve_if_gpu_mod plays for the Chebyshev machine. Same anatomy:
!   setup (per alpha/dt): homogeneous solutions v_-, v_+ + 2x2 Cinv per mode;
!   solve (per step)    : H1 g = f/alpha ; H2 v_p = -g ; wall derivs ; closure
!                         v = v_p + s1 v_- + s2 v_+ .
! Both Helmholtz legs are the validated batched fd_helm_solve_gpu (lam = 0 and
! lam = 1/alpha give mu = k2 and k2 + 1/alpha mode-by-mode); wall derivatives
! are the validated fd_wall_derivs_gpu rows. No projection / basis-conversion
! legs exist in point space, so the FD chain is SHORTER than the Cheb one.
!
! PACKED SETUP (the one structural deviation from the Cheb GPU setup, which
! carries real data as complex-with-zero-imag through 4 batched solves): the
! + and - families ride the SAME complex field -- Re = the l_+ chain, Im = the
! l_- chain. The band and all LU multipliers are REAL, so complex substitution
! acts componentwise: each part sees EXACTLY the real recurrence the host
! oracle runs (no mixing, not an approximation). 2 batched solves instead of
! 4, 3 slab temps instead of 7. One complex wall-derivative row then carries
! all four C entries: Re(dlo)=Dv_+(-1), Im(dlo)=Dv_-(-1), same at +1.
!
! Contracts (house rules):
!   * setup is per-alpha: rebuild when dt changes (same as Cheb vif);
!   * caller-owned _ws scratch, fully written before read, plain device
!     dummies (0155); self-allocating wrappers for tests/validate builders;
!   * walls: fd_helm kernels force u(0)=u(n)=0 exactly and vm/vp inherit it,
!     so the closed v has EXACT +-0 walls (gated in test_fd_vsolve_gpu);
!   * no device det guard in the Cinv kernel -- mirrors the Cheb vif_cinv.
!     There is no host-oracle fallback here: --validate refuses FD mode, and
!     both drivers go y_vif_setup -> fd_vif_setup_gpu_ws, so the guarded host
!     fd_vif_setup runs only in the serial test suite. The accepted margin is
!     physical: mu >= 0 production modes sit ~300 orders above singular, and a
!     hypothetical singular closure yields Inf/NaN Cinv that the NaN watchdog
!     catches steps later -- delayed-loud, never silent-wrong;
!   * default-stream chevrons / cuf kernels (blocking default stream).
!
! Build: nvfortran -cuda -c fd_vsolve_gpu_mod.F90  (after kinds, fd_helm_gpu_mod)
!===============================================================================
module fd_vsolve_gpu_mod
  use kinds,          only: dp, ip
  use cudafor
  use fd_helm_gpu_mod, only: fd_helm_solve_gpu, fd_wall_derivs_gpu
  implicit none
  private
  public :: fd_vif_setup_gpu, fd_vif_setup_gpu_ws
  public :: fd_vif_solve_gpu, fd_vif_solve_gpu_ws

contains

  !---------------------------------------------------------------------------
  ! Batched on-device setup: v_-, v_+ (point values, 0:n per mode) and Cinv
  ! (2,2 per mode) for ALL (kx,kz), one alpha. Mirrors fd_vif_setup:
  !   rhs  = -k2 * l_pm            (packed: Re <- l_+, Im <- l_-)
  !   H1 ghat = rhs (walls 0)      (fd_helm, lam = 0)
  !   g    = ghat + l_pm
  !   H2 vhom = -g (walls 0)       (fd_helm, lam = 1/alpha)
  !   C    = [Dv_-(-1) Dv_+(-1); Dv_-(+1) Dv_+(+1)],  Cinv = C^-1
  ! Scratch (caller-owned, fully written before read): rhs, g, vhom, each
  ! complex (0:n, nx, nz). y is the grid file's DESCENDING nodes on device.
  !---------------------------------------------------------------------------
  subroutine fd_vif_setup_gpu_ws(n, kl, nx, nz, kx, kz, alpha, y, d2b, wlo, whi, &
                                 vm_d, vp_d, Cinv_d, rhs, g, vhom)
    integer(ip), intent(in) :: n, kl, nx, nz
    real(dp),    intent(in) :: alpha
    real(dp),    device     :: kx(nx), kz(nz), y(0:n), d2b(2*kl+1, 0:n)
    real(dp),    device     :: wlo(0:n), whi(0:n)
    real(dp),    device     :: vm_d(0:n,nx,nz), vp_d(0:n,nx,nz), Cinv_d(2,2,nx,nz)
    complex(dp), device     :: rhs(0:n,nx,nz), g(0:n,nx,nz), vhom(0:n,nx,nz)
    complex(dp), device, allocatable :: dlo(:,:), dhi(:,:)
    allocate(dlo(nx,nz), dhi(nx,nz))

    ! rhs = -k2 * lift (packed), then H1 ghat = rhs -> g (walls forced 0)
    call fd_lift_rhs(n, nx, nz, kx, kz, y, rhs)
    call fd_helm_solve_gpu(n, kl, nx, nz, 0.0_dp, kx, kz, d2b, rhs, g)
    call fd_add_lift(n, nx, nz, y, g)                ! g = ghat + l_pm
    call fd_cscale(n, nx, nz, g, -1.0_dp)            ! -> -g (RHS of H2)
    call fd_helm_solve_gpu(n, kl, nx, nz, 1.0_dp/alpha, kx, kz, d2b, g, vhom)

    ! unpack the homogeneous pair + influence matrix
    call fd_unpack_hom(n, nx, nz, vhom, vp_d, vm_d)
    call fd_wall_derivs_gpu(n, nx, nz, wlo, whi, vhom, dlo, dhi)
    call fd_cinv(nx, nz, dlo, dhi, Cinv_d)

    deallocate(dlo, dhi)
  end subroutine fd_vif_setup_gpu_ws

  !-- self-allocating wrapper (tests / small --validate builders) --------------
  subroutine fd_vif_setup_gpu(n, kl, nx, nz, kx, kz, alpha, y, d2b, wlo, whi, &
                              vm_d, vp_d, Cinv_d)
    integer(ip), intent(in) :: n, kl, nx, nz
    real(dp),    intent(in) :: alpha
    real(dp),    device     :: kx(nx), kz(nz), y(0:n), d2b(2*kl+1, 0:n)
    real(dp),    device     :: wlo(0:n), whi(0:n)
    real(dp),    device     :: vm_d(0:n,nx,nz), vp_d(0:n,nx,nz), Cinv_d(2,2,nx,nz)
    complex(dp), device, allocatable :: rhs(:,:,:), g(:,:,:), vhom(:,:,:)
    allocate(rhs(0:n,nx,nz), g(0:n,nx,nz), vhom(0:n,nx,nz))
    call fd_vif_setup_gpu_ws(n, kl, nx, nz, kx, kz, alpha, y, d2b, wlo, whi, &
                             vm_d, vp_d, Cinv_d, rhs, g, vhom)
    deallocate(rhs, g, vhom)
  end subroutine fd_vif_setup_gpu

  !---------------------------------------------------------------------------
  ! Per-step batched solve for the complex point-value RHS f -> v. Mirrors
  ! fd_vif_solve; f is NOT modified. Scratch (caller-owned): fsc, gp, vpc
  ! complex (0:n,nx,nz); dlo, dhi complex (nx,nz). v may not alias any of
  ! f/vm/vp/scratch (each stage reads its input whole).
  !---------------------------------------------------------------------------
  subroutine fd_vif_solve_gpu_ws(n, kl, nx, nz, alpha, kx, kz, d2b, wlo, whi, &
                                 vm, vp, Cinv, f, v, fsc, gp, vpc, dlo, dhi)
    integer(ip), intent(in) :: n, kl, nx, nz
    real(dp),    intent(in) :: alpha
    real(dp),    device     :: kx(nx), kz(nz), d2b(2*kl+1, 0:n)
    real(dp),    device     :: wlo(0:n), whi(0:n)
    real(dp),    device     :: vm(0:n,nx,nz), vp(0:n,nx,nz), Cinv(2,2,nx,nz)
    complex(dp), device     :: f(0:n,nx,nz)      ! in : RHS (point values)
    complex(dp), device     :: v(0:n,nx,nz)      ! out: v (point values)
    complex(dp), device     :: fsc(0:n,nx,nz), gp(0:n,nx,nz), vpc(0:n,nx,nz)
    complex(dp), device     :: dlo(nx,nz), dhi(nx,nz)
    type(dim3) :: grid, block
    integer(cuda_stream_kind) :: strm

    ! step 1: H1 g_p = f/alpha    (walls forced 0 by the solver)
    call fd_copy_scale(n, nx, nz, f, fsc, 1.0_dp/alpha)
    call fd_helm_solve_gpu(n, kl, nx, nz, 0.0_dp, kx, kz, d2b, fsc, gp)
    ! step 2: H2 v_p = -g_p
    call fd_cscale(n, nx, nz, gp, -1.0_dp)
    call fd_helm_solve_gpu(n, kl, nx, nz, 1.0_dp/alpha, kx, kz, d2b, gp, vpc)
    ! step 3: wall derivatives of v_p
    call fd_wall_derivs_gpu(n, nx, nz, wlo, whi, vpc, dlo, dhi)
    ! step 4: closure + combine
    block = dim3(16,16,1);  grid = dim3((nx+15)/16, (nz+15)/16, 1)
    strm = cudaforGetDefaultStream()
    call fd_vif_closure_kernel<<<grid,block,0,strm>>>(n, nx, nz, vpc, dlo, dhi, &
                                                      Cinv, vm, vp, v)
  end subroutine fd_vif_solve_gpu_ws

  !-- self-allocating wrapper ---------------------------------------------------
  subroutine fd_vif_solve_gpu(n, kl, nx, nz, alpha, kx, kz, d2b, wlo, whi, &
                              vm, vp, Cinv, f, v)
    integer(ip), intent(in) :: n, kl, nx, nz
    real(dp),    intent(in) :: alpha
    real(dp),    device     :: kx(nx), kz(nz), d2b(2*kl+1, 0:n)
    real(dp),    device     :: wlo(0:n), whi(0:n)
    real(dp),    device     :: vm(0:n,nx,nz), vp(0:n,nx,nz), Cinv(2,2,nx,nz)
    complex(dp), device     :: f(0:n,nx,nz), v(0:n,nx,nz)
    complex(dp), device, allocatable :: fsc(:,:,:), gp(:,:,:), vpc(:,:,:)
    complex(dp), device, allocatable :: dlo(:,:), dhi(:,:)
    allocate(fsc(0:n,nx,nz), gp(0:n,nx,nz), vpc(0:n,nx,nz), dlo(nx,nz), dhi(nx,nz))
    call fd_vif_solve_gpu_ws(n, kl, nx, nz, alpha, kx, kz, d2b, wlo, whi, &
                             vm, vp, Cinv, f, v, fsc, gp, vpc, dlo, dhi)
    deallocate(fsc, gp, vpc, dlo, dhi)
  end subroutine fd_vif_solve_gpu

  !-- rhs = -k2 * lift, packed: Re <- l_+ = (1+y)/2, Im <- l_- = (1-y)/2 --------
  subroutine fd_lift_rhs(n, nx, nz, kx, kz, y, rhs)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    device     :: kx(nx), kz(nz), y(0:n)
    complex(dp), device     :: rhs(0:n,nx,nz)
    integer(ip) :: ix, iz, j
    real(dp)    :: k2
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do j = 0, n
             k2 = kx(ix)*kx(ix) + kz(iz)*kz(iz)
             rhs(j,ix,iz) = cmplx(-k2*0.5_dp*(1.0_dp + y(j)), &
                                  -k2*0.5_dp*(1.0_dp - y(j)), dp)
          end do
       end do
    end do
  end subroutine fd_lift_rhs

  !-- g = g + lift (packed), all rows incl. walls -------------------------------
  subroutine fd_add_lift(n, nx, nz, y, g)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    device     :: y(0:n)
    complex(dp), device     :: g(0:n,nx,nz)
    integer(ip) :: ix, iz, j
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do j = 0, n
             g(j,ix,iz) = g(j,ix,iz) + cmplx(0.5_dp*(1.0_dp + y(j)), &
                                             0.5_dp*(1.0_dp - y(j)), dp)
          end do
       end do
    end do
  end subroutine fd_add_lift

  !-- unpack the homogeneous pair: vp = Re(vhom), vm = Im(vhom) ------------------
  subroutine fd_unpack_hom(n, nx, nz, vhom, vp_d, vm_d)
    integer(ip), intent(in) :: n, nx, nz
    complex(dp), device     :: vhom(0:n,nx,nz)
    real(dp),    device     :: vp_d(0:n,nx,nz), vm_d(0:n,nx,nz)
    integer(ip) :: ix, iz, j
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do j = 0, n
             vp_d(j,ix,iz) = real(vhom(j,ix,iz), dp)
             vm_d(j,ix,iz) = aimag(vhom(j,ix,iz))
          end do
       end do
    end do
  end subroutine fd_unpack_hom

  !-- per-mode Cinv from the PACKED wall-derivative rows:
  !   Re(dlo) = Dv_+(-1), Im(dlo) = Dv_-(-1); same packing at +1.
  !   Same expressions as the host fd_vif_setup / Cheb vif_cinv (no det guard
  !   on device -- see header).
  subroutine fd_cinv(nx, nz, dlo, dhi, Cinv)
    integer(ip), intent(in) :: nx, nz
    complex(dp), device     :: dlo(nx,nz), dhi(nx,nz)
    real(dp),    device     :: Cinv(2,2,nx,nz)
    integer(ip) :: ix, iz
    real(dp) :: dvm_lo, dvm_hi, dvp_lo, dvp_hi, det
    !$cuf kernel do(2) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          dvp_lo = real(dlo(ix,iz), dp);  dvm_lo = aimag(dlo(ix,iz))
          dvp_hi = real(dhi(ix,iz), dp);  dvm_hi = aimag(dhi(ix,iz))
          det = dvm_lo*dvp_hi - dvp_lo*dvm_hi
          Cinv(1,1,ix,iz) =  dvp_hi/det;  Cinv(1,2,ix,iz) = -dvp_lo/det
          Cinv(2,1,ix,iz) = -dvm_hi/det;  Cinv(2,2,ix,iz) =  dvm_lo/det
       end do
    end do
  end subroutine fd_cinv

  !-- in-place complex scale by a real scalar ------------------------------------
  subroutine fd_cscale(n, nx, nz, x, s)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: s
    complex(dp), device     :: x(0:n,nx,nz)
    integer(ip) :: ix, iz, j
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do j = 0, n
             x(j,ix,iz) = x(j,ix,iz)*s
          end do
       end do
    end do
  end subroutine fd_cscale

  !-- dst = src * s (src untouched; whole column -- the solver ignores walls) ----
  subroutine fd_copy_scale(n, nx, nz, src, dst, s)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: s
    complex(dp), device     :: src(0:n,nx,nz), dst(0:n,nx,nz)
    integer(ip) :: ix, iz, j
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do j = 0, n
             dst(j,ix,iz) = src(j,ix,iz)*s
          end do
       end do
    end do
  end subroutine fd_copy_scale

  !---------------------------------------------------------------------------
  ! Closure + combine (the FD twin of vif_closure_kernel, identical algebra):
  !   sigma = -Cinv [Dv_p(-1); Dv_p(+1)] ;  v = v_p + s1 v_- + s2 v_+ .
  ! vm/vp walls are exact 0 and vpc walls exact 0, so v's walls stay exact 0.
  !---------------------------------------------------------------------------
  attributes(global) subroutine fd_vif_closure_kernel(n, nx, nz, vpc, dlo, dhi, &
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
  end subroutine fd_vif_closure_kernel

end module fd_vsolve_gpu_mod
