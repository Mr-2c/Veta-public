!===============================================================================
! par_pressure_mod.F90  --  diagnostic pressure solve: the
! inhomogeneous-Neumann Helmholtz BVP per (kx,kz) mode, by the INFLUENCE-MATRIX
! closure (no new basis). Solves for the MODIFIED pressure
!     Pi = p + |u|^2/2 :   (D^2_y - k^2) Pi_hat(y) = R(y),
!     Neumann wall BC      D_y Pi_hat(+-1) = g_+- ,
! where (docs/kmm_formulation.pdf, pressure-Poisson appendix):
!     R    = +div.H_hat = i kx Hx_hat + D_y Hy_hat + i kz Hz_hat   (H = u x omega)
!     g_+- = Hy_hat + (1/Re) D^2_y v_hat   at the walls.
! p is then recovered as p = Pi - |u|^2/2 (DEALIASED -- S6 step 4; not here).
!
! This module is the SOLVE for Pi only. Assembling R and g_+- from the
! dealiased nonlinear workspace, and the |u|^2/2 recovery, are the kernels below.
!
! THE CLOSURE, modelled 1:1 in src/models/work_pressure_model.py ("model 2"):
!   particular Pi_p : (D^2-k^2) Pi_p = R, Pi_p(+-1)=0   -- the existing Dirichlet
!                     Shen helm_band solve (project_shen + helm_band_solve_gpu).
!   homogeneous     : phi_+ , phi_- with (D^2-k^2)phi=0 and unit Dirichlet wall
!                     data, built by the SAME lift trick the v-solve uses
!                     (vsolve_if_mod vif_setup: phi = l + ghat, (D^2-k^2)ghat=k^2 l).
!   Neumann closure : the 2x2 wall matrix M = [Dphi_+(+1) Dphi_-(+1);
!                     Dphi_+(-1) Dphi_-(-1)]; per mode coefficients
!                     [c+;c-] = M^{-1} [g_+ - DPi_p(+1); g_- - DPi_p(-1)],
!                     Pi = Pi_p + c+ phi_+ + c- phi_- .  (wall_derivs_c gives the
!                     Dphi(+-1); the per-mode M^{-1} = Cinv is precomputed on host.)
!   k=0 gauge       : pure-Neumann singular (Dphi_+=+1/2, Dphi_-=-1/2 at both
!                     walls -> M singular). Pin the constant: store
!                     Cinv = [[1,0],[-1,0]] so c+ = (g_+ - DPi_p(+1)),
!                     c- = -(g_+ - DPi_p(+1)) -> Pi = Pi_p + c+ (phi_+ - phi_-)
!                     = Pi_p + c+ * y, gauge constant fixed to 0. (Same UNIFORM
!                     closure kernel handles every mode -- no per-mode branch.)
!
! Reuses: helm_band_solve_gpu (Dirichlet banded Helmholtz), spec_helpers
! (project_shen / shen_to_cheb_c / wall_derivs_c), and on the host the validated
! CPU helm path (helm_band_setup/solve, cheb_shen helmholtz_rhs/shen_to_cheb).
! y-local, mode-local -> NO communication (same pattern as every existing solve).
! dp throughout (accuracy-critical wall-normal solve), exactly like vsolve_if_gpu.
!
! Build:  nvfortran -cuda -c par_pressure_mod.F90
!===============================================================================
module par_pressure_mod
  use kinds,             only: dp, tp, wp, ip
  use cudafor
  use helm_band_gpu_mod, only: helm_band_solve_gpu
  use spec_helpers_mod,  only: project_shen, shen_to_cheb_c, wall_derivs_c, wall_values_c
  use helm_band_mod,     only: helm_band_setup, helm_band_solve
  use cheb_shen,         only: helmholtz_rhs, shen_to_cheb
  use cheb_ops_mod,      only: apply_L, cheb_deriv_c
  use yops_mod,          only: wall_derivs_h => yop_wall_deriv   ! facade
  use fdops_mod,         only: fd_helm_dirichlet, fd_band_factor, fd_band_solve
  use fd_helm_gpu_mod,   only: fd_helm_solve_gpu, fd_wall_derivs_gpu, fd_apply_L_gpu
  use yderiv_gpu_mod,    only: y_deriv_c                          ! FD D_y (device)
  implicit none
  private
  public :: pres_gpu_setup_host, pres_solve_gpu, pres_solve_gpu_ws
  public :: pres_set_fd                                           ! FD-y backend flip
  ! Assembly kernels (RHS divergence, Neumann wall data, dealiased
  ! |u|^2/2 recovery, tp<->dp seam). Validated in src/models/work_pressure_rhs_model.py.
  public :: pres_promote_c, pres_div_add, pres_d2v, pres_wall_g, pres_half_usq, pres_combine_p
  ! The transform-free assemble+solve for the modified pressure Pi,
  ! shared by the driver, the single-GPU oracle, and the distributed slab so there
  ! is ONE implementation (the transforms around it differ; this core does not).
  public :: pres_assemble_solve

  ! FD-y backend state: flipped ONCE at init by the FD wiring (mirror of
  ! the yderiv/yops setters). Host copies for the per-mode setup; device copies
  ! for the solve-time legs. Cheb runs never touch any of it.
  logical               :: pres_fd = .false.
  integer(ip)           :: pfd_kl = 4
  real(dp), allocatable :: pfd_y(:), pfd_d2b(:,:), pfd_wlo(:), pfd_whi(:)
  real(dp), device, allocatable :: pfd_d2b_d(:,:), pfd_wlo_d(:), pfd_whi_d(:)
  complex(dp), device, allocatable :: pfd_scr(:,:,:)   ! (0:n,nx,nz) -R staging (grow-only)

contains

  ! wall_derivs_h lives in yops_mod (yop_wall_deriv, use-renamed above).

  ! Flip the pressure backend to FD-y (called from the mains' FD block,
  ! after fd_build_bands/fd_wall_weights). d2b/wlo/whi are the SAME arrays the
  ! v-solve uses; y is the file node vector (descending +1..-1, index 0 = +1).
  subroutine pres_set_fd(ny, kl, y, d2b, wlo, whi)
    integer(ip), intent(in) :: ny, kl
    real(dp),    intent(in) :: y(ny), d2b(2*kl+1, ny), wlo(ny), whi(ny)
    pres_fd = .true.;  pfd_kl = kl
    if (allocated(pfd_y)) deallocate(pfd_y, pfd_d2b, pfd_wlo, pfd_whi, &
                                     pfd_d2b_d, pfd_wlo_d, pfd_whi_d)
    allocate(pfd_y(ny), pfd_d2b(2*kl+1, ny), pfd_wlo(ny), pfd_whi(ny))
    allocate(pfd_d2b_d(2*kl+1, ny), pfd_wlo_d(ny), pfd_whi_d(ny))
    pfd_y = y;  pfd_d2b = d2b;  pfd_wlo = wlo;  pfd_whi = whi
    pfd_d2b_d = d2b;  pfd_wlo_d = wlo;  pfd_whi_d = whi
  end subroutine pres_set_fd

  !---------------------------------------------------------------------------
  ! FD twin of pres_setup_mode: point values on the file nodes, the
  ! fd_vif_setup shape with the NEUMANN closure. phi_+- = l_+- + g,
  ! (-D^2+k2) g = -k2 l_+- (zero Dirichlet walls); wall DERIVATIVES via the
  ! one-sided stencils; M/Cinv/k=0 gauge EXACTLY the cheb pres_setup_mode
  ! conventions (M rows = [+1 wall; -1 wall], columns = [phi_+, phi_-]).
  ! The lifts are linear, so at k2=0 Dphi_+ = +1/2 and Dphi_- = -1/2 exactly,
  ! same as cheb -- the gauge rows carry over unchanged.
  !---------------------------------------------------------------------------
  subroutine fd_pres_setup_mode(ny, kl, y, d2b, wlo, whi, k2, pp, pm, Cinv)
    integer(ip), intent(in)  :: ny, kl
    real(dp),    intent(in)  :: y(ny), d2b(2*kl+1, ny), wlo(ny), whi(ny), k2
    real(dp),    intent(out) :: pp(ny), pm(ny), Cinv(2,2)
    real(dp) :: ab(2*kl+1, ny), gh(ny)
    real(dp) :: dpp_lo, dpp_hi, dpm_lo, dpm_hi, det
    integer(ip) :: j
    logical :: okf
    do j = 1, ny
       pp(j) = 0.5_dp*(1.0_dp + y(j))          ! l_+
       pm(j) = 0.5_dp*(1.0_dp - y(j))          ! l_-
    end do
    if (k2 > 0.0_dp) then
       call fd_helm_dirichlet(ny, kl, d2b, k2, ab)
       call fd_band_factor(ny, kl, ab, okf)
       if (.not. okf) then
          ! diagonally-dominant kit at k2>0: unreachable. REFUSE LOUDLY rather
          ! than degrade -- a zeroed Cinv would silently drop the Neumann BC.
          write(*,'(a,es12.4)') ' !! fd_pres_setup_mode: band factor FAILED at k2 = ', k2
          error stop 1
       end if
       gh = -k2*pp;  gh(1) = 0.0_dp;  gh(ny) = 0.0_dp
       call fd_band_solve(ny, kl, ab, gh);  pp = pp + gh
       gh = -k2*pm;  gh(1) = 0.0_dp;  gh(ny) = 0.0_dp
       call fd_band_solve(ny, kl, ab, gh);  pm = pm + gh
    end if                                     ! k2=0: phi_+- = the bare lifts
    dpp_lo = dot_product(wlo, pp);  dpp_hi = dot_product(whi, pp)
    dpm_lo = dot_product(wlo, pm);  dpm_hi = dot_product(whi, pm)
    if (k2 > 0.0_dp) then
       det = dpp_hi*dpm_lo - dpm_hi*dpp_lo
       Cinv(1,1) =  dpm_lo/det;  Cinv(1,2) = -dpm_hi/det
       Cinv(2,1) = -dpp_lo/det;  Cinv(2,2) =  dpp_hi/det
    else
       Cinv(1,1) =  1.0_dp;  Cinv(1,2) = 0.0_dp
       Cinv(2,1) = -1.0_dp;  Cinv(2,2) = 0.0_dp
    end if
  end subroutine fd_pres_setup_mode

  !---------------------------------------------------------------------------
  ! Per-mode host setup: homogeneous solutions phi_+ (pp), phi_- (pm) in
  ! Chebyshev coeffs, and the inverse Neumann influence matrix Cinv = M^{-1}.
  ! Band structure (sd,st,bd,boff) from helm_band_setup, shared across modes.
  !---------------------------------------------------------------------------
  subroutine pres_setup_mode(n, k2, sd, st, bd, boff, pp, pm, Cinv)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: k2, sd(0:n-2), st(0:n-2), bd(0:n-2), boff
    real(dp),    intent(out) :: pp(0:n), pm(0:n), Cinv(2,2)
    real(dp) :: lp(0:n), lm(0:n), ft(0:n-2), ghat(0:n-2)
    real(dp) :: dpp_lo, dpp_hi, dpm_lo, dpm_hi, det

    ! lift functions  l_+ = (1+y)/2 = (T0+T1)/2 ,  l_- = (1-y)/2 = (T0-T1)/2
    lp = 0.0_dp; lm = 0.0_dp
    lp(0) = 0.5_dp; lp(1) =  0.5_dp
    lm(0) = 0.5_dp; lm(1) = -0.5_dp

    ! phi_+ = l_+ + ghat,  (D^2-k^2)ghat = k^2 l_+  ->  (-D^2+k^2)ghat = -k^2 l_+
    call helmholtz_rhs(n, -k2*lp, ft)
    call helm_band_solve(n, k2, sd, st, bd, boff, ft, ghat)
    call shen_to_cheb(n, ghat, pp);  pp = pp + lp
    ! phi_-
    call helmholtz_rhs(n, -k2*lm, ft)
    call helm_band_solve(n, k2, sd, st, bd, boff, ft, ghat)
    call shen_to_cheb(n, ghat, pm);  pm = pm + lm

    call wall_derivs_h(n, pp, dpp_lo, dpp_hi)
    call wall_derivs_h(n, pm, dpm_lo, dpm_hi)

    if (k2 > 0.0_dp) then
       ! M = [ Dphi_+(+1)  Dphi_-(+1) ;  Dphi_+(-1)  Dphi_-(-1) ] ; Cinv = M^{-1}.
       det = dpp_hi*dpm_lo - dpm_hi*dpp_lo
       Cinv(1,1) =  dpm_lo/det;  Cinv(1,2) = -dpm_hi/det
       Cinv(2,1) = -dpp_lo/det;  Cinv(2,2) =  dpp_hi/det
    else
       ! k=0 pure-Neumann gauge: c+ = (g_+ - DPi_p(+1)), c- = -(c+), const -> 0.
       Cinv(1,1) =  1.0_dp;  Cinv(1,2) = 0.0_dp
       Cinv(2,1) = -1.0_dp;  Cinv(2,2) = 0.0_dp
    end if
  end subroutine pres_setup_mode

  !---------------------------------------------------------------------------
  ! Host precompute over all local modes -> device pp,pm,Cinv. k=0 is detected
  ! value-based (k2==0 on the mode owning kx=kz=0), mirroring the mean special-
  ! case; recompute is unnecessary (no dt dependence -- the Neumann BVP carries
  ! none, unlike the v-solve whose alpha=Re/(beta dt) does).
  !---------------------------------------------------------------------------
  subroutine pres_gpu_setup_host(n, nx, nz, kxh, kzh, pp_d, pm_d, Cinv_d)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: kxh(nx), kzh(nz)
    real(dp), device        :: pp_d(0:n,nx,nz), pm_d(0:n,nx,nz), Cinv_d(2,2,nx,nz)
    real(dp), allocatable :: pp_h(:,:,:), pm_h(:,:,:), Cinv_h(:,:,:,:)
    real(dp) :: sd(0:n-2), st(0:n-2), bd(0:n-2), boff, k2
    integer(ip) :: ix, iz
    allocate(pp_h(0:n,nx,nz), pm_h(0:n,nx,nz), Cinv_h(2,2,nx,nz))
    if (pres_fd) then                          ! FD-y: point-value modes
       do iz = 1, nz
          do ix = 1, nx
             k2 = kxh(ix)**2 + kzh(iz)**2
             call fd_pres_setup_mode(n+1, pfd_kl, pfd_y, pfd_d2b, pfd_wlo, pfd_whi, &
                                     k2, pp_h(:,ix,iz), pm_h(:,ix,iz), Cinv_h(:,:,ix,iz))
          end do
       end do
       pp_d = pp_h; pm_d = pm_h; Cinv_d = Cinv_h
       deallocate(pp_h, pm_h, Cinv_h)
       return
    end if
    call helm_band_setup(n, sd, st, bd, boff)
    do iz = 1, nz
       do ix = 1, nx
          k2 = kxh(ix)**2 + kzh(iz)**2
          call pres_setup_mode(n, k2, sd, st, bd, boff, &
                               pp_h(:,ix,iz), pm_h(:,ix,iz), Cinv_h(:,:,ix,iz))
       end do
    end do
    pp_d = pp_h; pm_d = pm_h; Cinv_d = Cinv_h
    deallocate(pp_h, pm_h, Cinv_h)
  end subroutine pres_gpu_setup_host

  !---------------------------------------------------------------------------
  ! wall VALUES for FD point-value fields: the walls ARE the end rows
  ! (index 0 = +1 wall, index n = -1 wall on the descending file grid). The FD
  ! twin of spec_helpers wall_values_c (same _hi at +1 / _lo at -1 convention).
  !---------------------------------------------------------------------------
  subroutine fd_wall_rows(n, nx, nz, a, alo, ahi)
    integer(ip), intent(in) :: n, nx, nz
    complex(dp), device :: a(0:n, nx, nz)
    complex(dp), device :: alo(nx, nz), ahi(nx, nz)
    integer(ip) :: ix, iz
    !$cuf kernel do(2) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          ahi(ix,iz) = a(0,ix,iz)
          alo(ix,iz) = a(n,ix,iz)
       end do
    end do
  end subroutine fd_wall_rows

  !---------------------------------------------------------------------------
  ! In-place complex negate (plain device dummy -> cuf-safe), to turn the Shen
  ! projection of R into the projection of -R (helm solves (-D^2+mu)u = rhs).
  !---------------------------------------------------------------------------
  subroutine cnegate(m, nx, nz, x)
    integer(ip), intent(in) :: m, nx, nz
    complex(dp), device     :: x(0:m, nx, nz)
    integer(ip) :: ix, iz, k
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, m
             x(k,ix,iz) = -x(k,ix,iz)
          end do
       end do
    end do
  end subroutine cnegate

  !---------------------------------------------------------------------------
  ! Neumann closure + combine, IN PLACE on Pi (holds Pi_p on entry, Pi on exit):
  !   r+ = g_+ - DPi_p(+1) ,  r- = g_- - DPi_p(-1)
  !   c+ = Cinv11 r+ + Cinv12 r- ,  c- = Cinv21 r+ + Cinv22 r-
  !   Pi = Pi_p + c+ phi_+ + c- phi_- .
  ! dhi = DPi_p(+1), dlo = DPi_p(-1) (spec_helpers convention).
  !---------------------------------------------------------------------------
  attributes(global) subroutine pres_closure_kernel(n, nx, nz, dlo, dhi, gp, gm, &
                                                     Cinv, pp, pm, Pi)
    integer(ip), value :: n, nx, nz
    complex(dp) :: dlo(nx,nz), dhi(nx,nz), gp(nx,nz), gm(nx,nz)
    real(dp)    :: Cinv(2,2,nx,nz), pp(0:n,nx,nz), pm(0:n,nx,nz)
    complex(dp) :: Pi(0:n,nx,nz)
    integer(ip) :: ix, iz, j
    complex(dp) :: rp, rm, cp, cm
    ix = (blockIdx%x-1)*blockDim%x + threadIdx%x
    iz = (blockIdx%y-1)*blockDim%y + threadIdx%y
    if (ix > nx .or. iz > nz) return
    rp = gp(ix,iz) - dhi(ix,iz)
    rm = gm(ix,iz) - dlo(ix,iz)
    cp = Cinv(1,1,ix,iz)*rp + Cinv(1,2,ix,iz)*rm
    cm = Cinv(2,1,ix,iz)*rp + Cinv(2,2,ix,iz)*rm
    do j = 0, n
       Pi(j,ix,iz) = Pi(j,ix,iz) + cp*pp(j,ix,iz) + cm*pm(j,ix,iz)
    end do
  end subroutine pres_closure_kernel

  !---------------------------------------------------------------------------
  ! Per-mode pressure solve, WORKSPACE variant (caller-owned scratch, house
  ! style). Rc = R (Cheb coeffs); gp/gm = Neumann wall data DPi(+-1); Pi out
  ! (Cheb coeffs). Scratch: ftil,Pish (0:n-2,nx,nz); dlo,dhi (nx,nz).
  ! No allocation here.
  !---------------------------------------------------------------------------
  subroutine pres_solve_gpu_ws(n, nx, nz, kx, kz, sdiag, stail, bdiag, boff, &
                               pp, pm, Cinv, Rc, gp, gm, Pi, &
                               ftil, Pish, dlo, dhi)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: boff
    real(dp), device :: kx(nx), kz(nz), sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2)
    real(dp), device :: pp(0:n,nx,nz), pm(0:n,nx,nz), Cinv(2,2,nx,nz)
    complex(dp), device :: Rc(0:n,nx,nz)         ! in : RHS R (Cheb coeffs)
    complex(dp), device :: gp(nx,nz), gm(nx,nz)  ! in : Neumann wall data
    complex(dp), device :: Pi(0:n,nx,nz)         ! out: Pi (Cheb coeffs)
    ! ftil/Pish ASSUMED-SIZE (the stub seam): FD-mode callers pass
    ! 1-mode stubs (the FD branch below returns before the Shen legs, their
    ! only consumers); Cheb callers pass full-size, forwarded to the
    ! explicit-shape leaves unchanged.
    complex(dp), device :: ftil(0:n-2,nx,*), Pish(0:n-2,nx,*)
    complex(dp), device :: dlo(nx,nz), dhi(nx,nz)
    type(dim3) :: grid, block
    integer(cuda_stream_kind) :: strm      ! explicit stream for chevron launches
    integer(ip) :: ix2, iz2, j2, istat     ! FD staging loop

    ! FD-y branch: point values end-to-end, no Shen legs. Same math:
    ! (-D^2+k^2) Pi_p = -R with zero-Dirichlet walls (fd_helm_solve_gpu factors
    ! per-mode lam=0+k^2 on the fly -- pressure runs only on stats/snap steps),
    ! then one-sided wall derivatives + the SAME closure kernel (the combine
    ! Pi += c+ phi_+ + c- phi_- is pointwise in either representation).
    if (pres_fd) then
       ! EXACT-shape scratch (not grow-only): the staging kernel writes with the
       ! retained shape while fd_helm_solve_gpu reads via sequence association
       ! with the CURRENT one -- a retained-larger buffer would silently stride-
       ! mismatch every iz>=2 mode (latent until dual-shape callers).
       if (allocated(pfd_scr)) then
          if (size(pfd_scr,1) /= n+1 .or. size(pfd_scr,2) /= nx .or. size(pfd_scr,3) /= nz) then
             istat = cudaDeviceSynchronize();  deallocate(pfd_scr)
          end if
       end if
       if (.not. allocated(pfd_scr)) allocate(pfd_scr(0:n,nx,nz))
       !$cuf kernel do(3) <<<*,*>>>
       do iz2 = 1, nz
          do ix2 = 1, nx
             do j2 = 0, n
                pfd_scr(j2,ix2,iz2) = -Rc(j2,ix2,iz2)
             end do
          end do
       end do
       call fd_helm_solve_gpu(n, pfd_kl, nx, nz, 0.0_dp, kx, kz, pfd_d2b_d, &
                              pfd_scr, Pi)                    ! Pi holds Pi_p (values)
       call fd_wall_derivs_gpu(n, nx, nz, pfd_wlo_d, pfd_whi_d, Pi, dlo, dhi)
       block = dim3(16,16,1); grid = dim3((nx+15)/16, (nz+15)/16, 1)
       strm = cudaforGetDefaultStream()
       call pres_closure_kernel<<<grid,block,0,strm>>>(n, nx, nz, dlo, dhi, gp, gm, &
                                                Cinv, pp, pm, Pi)
       return
    end if

    ! particular: (D^2-k^2) Pi_p = R, Pi_p(+-1)=0  ->  (-D^2+k^2)Pi_p = -R
    call project_shen(n, nx, nz, Rc, ftil)
    call cnegate(n-2, nx, nz, ftil)                  ! project(-R)
    call helm_band_solve_gpu(n, nx, nz, 0.0_dp, kx, kz, sdiag, stail, bdiag, boff, &
                             ftil, Pish)
    call shen_to_cheb_c(n, nx, nz, Pish, Pi)         ! Pi holds Pi_p (Cheb)

    ! wall derivatives of Pi_p, then Neumann closure (in place on Pi)
    call wall_derivs_c(n, nx, nz, Pi, dlo, dhi)
    block = dim3(16,16,1); grid = dim3((nx+15)/16, (nz+15)/16, 1)
    strm = cudaforGetDefaultStream()
    call pres_closure_kernel<<<grid,block,0,strm>>>(n, nx, nz, dlo, dhi, gp, gm, &
                                             Cinv, pp, pm, Pi)
  end subroutine pres_solve_gpu_ws

  !---------------------------------------------------------------------------
  ! Self-allocating wrapper (test / non-hot-path interface; the driver uses the
  ! _ws variant with preallocated scratch).
  !---------------------------------------------------------------------------
  subroutine pres_solve_gpu(n, nx, nz, kx, kz, sdiag, stail, bdiag, boff, &
                            pp, pm, Cinv, Rc, gp, gm, Pi)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: boff
    real(dp), device :: kx(nx), kz(nz), sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2)
    real(dp), device :: pp(0:n,nx,nz), pm(0:n,nx,nz), Cinv(2,2,nx,nz)
    complex(dp), device :: Rc(0:n,nx,nz), gp(nx,nz), gm(nx,nz), Pi(0:n,nx,nz)
    complex(dp), device, allocatable :: ftil(:,:,:), Pish(:,:,:), dlo(:,:), dhi(:,:)
    allocate(ftil(0:n-2,nx,nz), Pish(0:n-2,nx,nz), dlo(nx,nz), dhi(nx,nz))
    call pres_solve_gpu_ws(n, nx, nz, kx, kz, sdiag, stail, bdiag, boff, &
                           pp, pm, Cinv, Rc, gp, gm, Pi, &
                           ftil, Pish, dlo, dhi)
    deallocate(ftil, Pish, dlo, dhi)
  end subroutine pres_solve_gpu

  !===========================================================================
  ! Assembly kernels. All operate on plain device arrays (no
  ! cross-module deps); the driver orchestrates the transforms/apply_L around
  ! them. Conventions validated in src/models/work_pressure_rhs_model.py.
  !===========================================================================

  !-- tp -> dp complex copy (promote dealiased H_hat to the dp solve tier) ----
  subroutine pres_promote_c(n, nx, nz, src, dst)
    integer(ip), intent(in) :: n, nx, nz
    complex(tp), device :: src(0:n, nx, nz)
    complex(dp), device :: dst(0:n, nx, nz)
    integer(ip) :: ix, iz, k
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, n
             dst(k,ix,iz) = cmplx(real(src(k,ix,iz),dp), aimag(src(k,ix,iz)), dp)
          end do
       end do
    end do
  end subroutine pres_promote_c

  !-- divergence RHS accumulate:  R += i*kx*Hx + i*kz*Hz   (R holds D_y Hy on
  !   entry; the +i*k convention matches stats_mod divergence_check). dp.
  subroutine pres_div_add(n, nx, nz, kx, kz, Hx, Hz, R)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    device :: kx(nx), kz(nz)
    complex(dp), device :: Hx(0:n,nx,nz), Hz(0:n,nx,nz), R(0:n,nx,nz)
    integer(ip) :: ix, iz, k
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, n
             R(k,ix,iz) = R(k,ix,iz) + cmplx(0.0_dp, kx(ix), dp)*Hx(k,ix,iz) &
                                     + cmplx(0.0_dp, kz(iz), dp)*Hz(k,ix,iz)
          end do
       end do
    end do
  end subroutine pres_div_add

  !-- D^2 v from apply_L:  d2v = Lv + k^2 v   (Lv = (D^2 - k^2) v). dp. ---------
  subroutine pres_d2v(n, nx, nz, kx, kz, Lv, v, d2v)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    device :: kx(nx), kz(nz)
    complex(dp), device :: Lv(0:n,nx,nz), v(0:n,nx,nz), d2v(0:n,nx,nz)
    integer(ip) :: ix, iz, k
    real(dp)    :: k2
    ! k2 computed INSIDE the innermost loop so the three loops are TIGHTLY nested
    ! (a statement between the do's breaks cuf do(3) -- NVFORTRAN-1050). Same idiom
    ! as cheb_ops_mod apply_L; the redundant per-k recompute is negligible.
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, n
             k2 = kx(ix)*kx(ix) + kz(iz)*kz(iz)
             d2v(k,ix,iz) = Lv(k,ix,iz) + k2*v(k,ix,iz)
          end do
       end do
    end do
  end subroutine pres_d2v

  !-- Neumann wall data:  g_+ = Hy(+1) + (1/Re) D^2v(+1),  g_- likewise at -1.
  !   Inputs are the per-mode wall VALUES (wall_values_c): _hi at +1, _lo at -1.
  subroutine pres_wall_g(nx, nz, Re, Hy_lo, Hy_hi, d2v_lo, d2v_hi, gp, gm)
    integer(ip), intent(in) :: nx, nz
    real(dp),    intent(in) :: Re
    complex(dp), device :: Hy_lo(nx,nz), Hy_hi(nx,nz), d2v_lo(nx,nz), d2v_hi(nx,nz)
    complex(dp), device :: gp(nx,nz), gm(nx,nz)
    integer(ip) :: ix, iz
    real(dp)    :: inv_re
    inv_re = 1.0_dp/Re
    !$cuf kernel do(2) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          gp(ix,iz) = Hy_hi(ix,iz) + inv_re*d2v_hi(ix,iz)
          gm(ix,iz) = Hy_lo(ix,iz) + inv_re*d2v_lo(ix,iz)
       end do
    end do
  end subroutine pres_wall_g

  !-- q = inv * 1/2 (pu^2 + pv^2 + pw^2) on the DEALIASED physical grid (real tp).
  !   inv = 1/(nxd*nzd) folds in the UNNORMALIZED-cuFFT scale of the forward
  !   transform that follows (par_to_spectral_da_r2c), EXACTLY as cross_prod does
  !   for H -> so q_hat lands on the same true-coefficient scale as H_hat / Pi_hat,
  !   making the spectral subtraction p_hat = Pi_hat - q_hat dimensionally consistent.
  subroutine pres_half_usq(ny, nxd, anz, inv, pu, pv, pw, q)
    integer(ip), intent(in) :: ny, nxd, anz
    real(tp),    intent(in) :: inv
    real(tp), device :: pu(ny,nxd,anz), pv(ny,nxd,anz), pw(ny,nxd,anz), q(ny,nxd,anz)
    integer(ip) :: iy, ix, iz
    real(tp)    :: h
    h = 0.5_tp*inv
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, anz
       do ix = 1, nxd
          do iy = 1, ny
             q(iy,ix,iz) = h*( pu(iy,ix,iz)*pu(iy,ix,iz) &
                             + pv(iy,ix,iz)*pv(iy,ix,iz) &
                             + pw(iy,ix,iz)*pw(iy,ix,iz) )
          end do
       end do
    end do
  end subroutine pres_half_usq

  !---------------------------------------------------------------------------
  ! pres_assemble_solve -- the TRANSFORM-FREE core of the pressure recovery: from
  ! the promoted dealiased H_hat (Hxd,Hyd,Hzd, dp) and the wall-normal state v,
  ! assemble the RHS R = +div.H and the Neumann wall data g_+- and solve
  ! (D^2-k^2)Pi = R for the modified pressure Pi (dp, Chebyshev-y coeffs). Works on
  ! ANY mode range nm (= bnkx slab, or nkx full single-GPU), so the driver, the
  ! --validate oracle (run_full) and slab (run_slab) all call it -> ONE source of
  ! truth. Self-allocating scratch (called per snapshot/stats sample / once per
  ! validate -> infrequent, so the alloc is not on the hot path). dp tier; requires
  ! WP=dp (the only shipped build). Conventions: src/models/work_pressure_rhs_model.py.
  !   kx,kz  (dp) : i*k for the divergence + k^2 for D^2v.
  !   kxw,kzw(wp) : for apply_L (Lv).   sd,st,bd,boff : Shen band structure.
  !   pp,pm,Cinv  : the per-mode influence-matrix setup (pres_gpu_setup_host).
  !---------------------------------------------------------------------------
  subroutine pres_assemble_solve(n, nm, nz, nxg, kx0, Re, kx, kz, kxw, kzw, &
                                 sd, st, bd, boff, &
                                 pp, pm, Cinv, Hxd, Hyd, Hzd, v, Pi)
    ! nxg = GLOBAL nx, kx0 = this slab's global kx offset (0 on the
    ! full-grid oracle): needed by the Nyquist mask below to find the owner
    ! of the global kx-Nyquist column.
    integer(ip), intent(in) :: n, nm, nz, nxg, kx0
    real(dp),    intent(in) :: Re, boff
    real(dp), device :: kx(nm), kz(nz), sd(0:n-2), st(0:n-2), bd(0:n-2)
    real(wp), device :: kxw(nm), kzw(nz)
    real(dp), device :: pp(0:n,nm,nz), pm(0:n,nm,nz), Cinv(2,2,nm,nz)
    complex(dp), device :: Hxd(0:n,nm,nz), Hyd(0:n,nm,nz), Hzd(0:n,nm,nz)
    complex(dp), device :: v(0:n,nm,nz)          ! in : wall-normal state (Cheb coeffs)
    complex(dp), device :: Pi(0:n,nm,nz)         ! out: modified pressure (Cheb coeffs)
    complex(dp), device, allocatable :: Rc(:,:,:), Lvp(:,:,:), scrLp(:,:,:), d2vp(:,:,:)
    complex(dp), device, allocatable :: ftil(:,:,:), Pish(:,:,:)
    complex(dp), device, allocatable :: gp(:,:), gm(:,:), dlo(:,:), dhi(:,:)
    complex(dp), device, allocatable :: Hy_lo(:,:), Hy_hi(:,:), d2v_lo(:,:), d2v_hi(:,:)
    ! FD mode allocates NO Cheb-only slabs: scrLp (apply_L
    ! scratch) is never referenced there, and ftil/Pish (Shen scratch) ship as
    ! 1-mode stubs through the assumed-size seam of pres_solve_gpu_ws, whose FD
    ! branch returns before the Shen legs (the stub recipe). Cuts the peak
    ! pressure transient by ~3 dp slabs (~a third) in FD runs.
    if (pres_fd) then
       allocate(ftil(0:n-2,1,1), Pish(0:n-2,1,1))
    else
       allocate(scrLp(0:n,nm,nz), ftil(0:n-2,nm,nz), Pish(0:n-2,nm,nz))
    end if
    allocate(Rc(0:n,nm,nz), Lvp(0:n,nm,nz), d2vp(0:n,nm,nz))
    allocate(gp(nm,nz), gm(nm,nz), dlo(nm,nz), dhi(nm,nz), &
             Hy_lo(nm,nz), Hy_hi(nm,nz), d2v_lo(nm,nz), d2v_hi(nm,nz))
    ! Mask the Nyquist columns of the promoted H BEFORE any assembly:
    ! par_nl_eval_H masks only
    ! hv/heta, so the S3-sourced Hxd/Hyd/Hzd would otherwise carry the
    ! asymmetric-Nyquist junk into BOTH the RHS (via pres_div_add/y_deriv) and
    ! the Neumann wall data g+- (via the wall rows). The mask touches the Nyquist
    ! columns ONLY (junk -> exact zeros); the velocity trajectory is untouched
    ! (this path is diagnostic sampling). In-place on the caller's promoted
    ! scratch -- dead copies at every call site.
    call pres_mask_nyq(n, nm, nz, nxg, kx0, Hxd)
    call pres_mask_nyq(n, nm, nz, nxg, kx0, Hyd)
    call pres_mask_nyq(n, nm, nz, nxg, kx0, Hzd)
    ! RHS  R = i*kx*Hx + D_y*Hy + i*kz*Hz
    ! y_deriv_c = cheb_deriv_c's statements verbatim until the FD wiring
    ! flips it (yderiv_set_fd) -> banded D1 on point values. Bit-identical cheb.
    call y_deriv_c(n+1, nm*nz, Hyd, Rc)                     ! Rc = D_y Hy
    call pres_div_add(n, nm, nz, kx, kz, Hxd, Hzd, Rc)     ! Rc += i kx Hx + i kz Hz
    ! Neumann wall data  g_+- = Hy(+-1) + (1/Re) D^2v(+-1),  D^2v = Lv + k^2 v
    if (pres_fd) then
       call fd_apply_L_gpu(n, pfd_kl, nm, nz, kx, kz, pfd_d2b_d, v, Lvp)
    else
       call apply_L(n+1, nm, nz, kxw, kzw, v, Lvp, scrLp)  ! Lvp = (D^2-k^2) v
    end if
    call pres_d2v(n, nm, nz, kx, kz, Lvp, v, d2vp)         ! d2vp = Lv + k^2 v
    if (pres_fd) then                        ! point values: walls ARE rows 0 / n
       call fd_wall_rows(n, nm, nz, d2vp, d2v_lo, d2v_hi)
       call fd_wall_rows(n, nm, nz, Hyd,  Hy_lo,  Hy_hi)
    else
       call wall_values_c(n, nm, nz, d2vp, d2v_lo, d2v_hi)
       call wall_values_c(n, nm, nz, Hyd,  Hy_lo,  Hy_hi)
    end if
    call pres_wall_g(nm, nz, Re, Hy_lo, Hy_hi, d2v_lo, d2v_hi, gp, gm)
    ! solve (D^2 - k^2) Pi = R, Neumann
    call pres_solve_gpu_ws(n, nm, nz, kx, kz, sd, st, bd, boff, pp, pm, Cinv, &
                           Rc, gp, gm, Pi, ftil, Pish, dlo, dhi)
    deallocate(Rc, Lvp, d2vp, ftil, Pish, &
               gp, gm, dlo, dhi, Hy_lo, Hy_hi, d2v_lo, d2v_hi)
    if (allocated(scrLp)) deallocate(scrLp)
  end subroutine pres_assemble_solve

  !-- dp Nyquist mask for the pressure assembly: zero the local
  !-- kz-Nyquist row everywhere and the global kx-Nyquist column on its owner
  !-- rank (index logic verbatim from par_nonlinear's mask_nyquist_slab;
  !-- duplicated here to keep par_pressure free of a par_nonlinear build-order
  !-- dependency). q is (0:n, nm, nz) dp -- the promoted H scratch.
  subroutine pres_mask_nyq(n, nm, nz, nxg, kx0, q)
    integer(ip), intent(in) :: n, nm, nz, nxg, kx0
    complex(dp), device     :: q(0:n, nm, nz)
    integer(ip) :: j, ix, iz, iz0, nkxg, ixN
    nkxg = nxg/2 + 1
    iz0  = nz/2 + 1
    !$cuf kernel do(2) <<<*,*>>>
    do ix = 1, nm
       do j = 0, n
          q(j, ix, iz0) = (0.0_dp, 0.0_dp)
       end do
    end do
    if (kx0 + nm == nkxg) then             ! this rank owns the kx-Nyquist column
       ixN = nm
       !$cuf kernel do(2) <<<*,*>>>
       do iz = 1, nz
          do j = 0, n
             q(j, ixN, iz) = (0.0_dp, 0.0_dp)
          end do
       end do
    end if
  end subroutine pres_mask_nyq

  !-- recover p:  phat = Pi - qhat  (Pi is dp from the solve, qhat is the
  !   dealiased tp |u|^2/2 spectral; output phat in tp for the clean inverse).
  !   Index seam: Pi is 0:n (1..ny), qhat/phat are 1..ny -- same ny rows.
  subroutine pres_combine_p(n, nx, nz, Pi, qhat, phat)
    integer(ip), intent(in) :: n, nx, nz
    complex(dp), device :: Pi(0:n, nx, nz)
    complex(tp), device :: qhat(0:n, nx, nz), phat(0:n, nx, nz)
    integer(ip) :: ix, iz, k
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, n
             phat(k,ix,iz) = cmplx(real(Pi(k,ix,iz),tp), aimag(Pi(k,ix,iz)), tp) &
                             - qhat(k,ix,iz)
          end do
       end do
    end do
  end subroutine pres_combine_p

end module par_pressure_mod
