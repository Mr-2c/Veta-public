!===============================================================================
! nonlinear_mod.F90  --  end-to-end nonlinear forcing evaluation.  [R2C path]
!
! Given the state (v̂, η̂, u0, w0) produce the explicit RK forcings (ĥ_v, ĥ_η)
! and the mean-mode forcings Ĥ_x(0,0), Ĥ_z(0,0). Pure orchestration of the
! validated parts:
!
!   recover_uw  ->  compute_vorticity
!     -> six to_physical_da_r2c  (u,v,w,ω_x,ω_y,ω_z to the 3/2-dealiased grid)
!     -> H = (u x ω)/(nxd*nzd)   (rotational nonlinear term, normalised)
!     -> three to_spectral_da_r2c (Ĥ_x,Ĥ_y,Ĥ_z)
!     -> assemble_hv_heta
!
! [R2C] x is a REAL transform: spectral arrays are (ny,nkx,nz), nkx=nx/2+1 (the
! non-negative kx half; the negative half is the Hermitian conjugate and is NOT
! stored). The dealiased PHYSICAL fields are REAL (ny,nxd,nzd). The R2C/C2R
! transforms need a complex FINE scratch (ny,nkxd,nzd), nkxd=nxd/2+1.
!
! velocity_mod (recover_uw, compute_vorticity) and forcing_mod (assemble_hv_heta)
! are EXTENT-AGNOSTIC -- local in (kx,kz), no negative-kx coupling -- so they are
! called UNCHANGED with nkx in the x-extent slot and an nkx-length kx array.
! (Proven by test_velocity_r2c / test_forcing_r2c: nkx call == nx call on the
! retained half, to bit-identity.)
!
! THE F4 REALNESS FIX IS NOW STRUCTURAL. The C2C path carried the real field in
! full-complex storage and had to take real() in cross_prod to kill the
! conjugate-asymmetric "ghost" (the t~53 Lyapunov blow-up,
! bug 9). With R2C the physical fields are REAL by construction: the ghost
! subspace cannot exist (there is no imaginary part to carry it), and Ĥ is
! Hermitian by construction of the R2C forward transform. cross_prod is now a
! pure real cross product. (Validated separately in increment 3b via the
! test_realness ghost-decay logic on this path.)
!
! NORMALISATION: unchanged -- 1/(nxd*nzd) folded into the physical product makes
! Ĥ the true coefficients (cuFFT's unnormalised nxd*nzd from the forward pair).
!
! Requires -cuda -cudalib=cufft on a GPU node.
!===============================================================================

module nonlinear_mod
  use kinds,          only: tp, wp, ip
  use cudafor
  use spectral3d_mod, only: s3d_plans, s3d_setup_da_r2c, s3d_destroy_r2c, &
                            to_physical_da_r2c, to_spectral_da_r2c
  use velocity_mod,   only: recover_uw, compute_vorticity
  use forcing_mod,    only: assemble_hv_heta
  use yderiv_gpu_mod, only: gal_shift_mean    ! basis-aware Galilean shift
  implicit none
  private
  public :: nl_workspace, nl_setup, nl_destroy, nl_evaluate, nl_eval_H
  public :: mask_nyquist, copy_c3

  type :: nl_workspace
     integer(ip) :: ny, nx, nz, nkx, nxd, nzd, nkxd
     real(wp)    :: ugal = 0.0_wp              ! [Galilean] constant frame speed U_c (0 = lab frame)
     type(s3d_plans) :: plans
     real(wp),    device, allocatable :: kx(:), kz(:)         ! kx is nkx-long; wp
     ! spectral scratch (ny,nkx,nz)
     complex(tp), device, allocatable :: uh(:,:,:), wh(:,:,:)
     complex(tp), device, allocatable :: ox(:,:,:), oy(:,:,:), oz(:,:,:)
     complex(tp), device, allocatable :: dv(:,:,:), du(:,:,:), dw(:,:,:), gg(:,:,:)
     complex(tp), device, allocatable :: Hx(:,:,:), Hy(:,:,:), Hz(:,:,:)
     ! physical scratch (ny,nxd,nzd) -- REAL [R2C]
     real(tp),    device, allocatable :: pu(:,:,:), pv(:,:,:), pw(:,:,:)
     real(tp),    device, allocatable :: pox(:,:,:), poy(:,:,:), poz(:,:,:)
     real(tp),    device, allocatable :: phx(:,:,:), phy(:,:,:), phz(:,:,:)
     ! complex FINE scratch (ny,nkxd,nzd) for the R2C/C2R intermediate [R2C]
     complex(tp), device, allocatable :: Wf(:,:,:)
  end type nl_workspace

contains

  subroutine nl_setup(ws, ny, nx, nz, kx_h, kz_h)
    type(nl_workspace), intent(out) :: ws
    integer(ip),        intent(in)  :: ny, nx, nz
    real(wp),           intent(in)  :: kx_h(nx/2+1), kz_h(nz)   ! kx_h is nkx-long
    integer(ip) :: nxd, nzd, nkx, nkxd
    nxd = 3_ip*nx/2_ip;  nzd = 3_ip*nz/2_ip
    nkx = nx/2_ip + 1_ip;  nkxd = nxd/2_ip + 1_ip
    ws%ny = ny;  ws%nx = nx;  ws%nz = nz;  ws%nxd = nxd;  ws%nzd = nzd
    ws%nkx = nkx;  ws%nkxd = nkxd
    call s3d_setup_da_r2c(ws%plans, ny, nx, nz)
    allocate(ws%kx(nkx), ws%kz(nz));  ws%kx = kx_h;  ws%kz = kz_h   ! wp, no demotion
    allocate(ws%uh(ny,nkx,nz), ws%wh(ny,nkx,nz))
    allocate(ws%ox(ny,nkx,nz), ws%oy(ny,nkx,nz), ws%oz(ny,nkx,nz))
    allocate(ws%dv(ny,nkx,nz), ws%du(ny,nkx,nz), ws%dw(ny,nkx,nz), ws%gg(ny,nkx,nz))
    allocate(ws%Hx(ny,nkx,nz), ws%Hy(ny,nkx,nz), ws%Hz(ny,nkx,nz))
    allocate(ws%pu(ny,nxd,nzd), ws%pv(ny,nxd,nzd), ws%pw(ny,nxd,nzd))
    allocate(ws%pox(ny,nxd,nzd), ws%poy(ny,nxd,nzd), ws%poz(ny,nxd,nzd))
    allocate(ws%phx(ny,nxd,nzd), ws%phy(ny,nxd,nzd), ws%phz(ny,nxd,nzd))
    allocate(ws%Wf(ny,nkxd,nzd))
  end subroutine nl_setup

  subroutine nl_destroy(ws)
    type(nl_workspace), intent(inout) :: ws
    call s3d_destroy_r2c(ws%plans)
    if (allocated(ws%kx)) deallocate(ws%kx, ws%kz)
    if (allocated(ws%uh)) deallocate(ws%uh, ws%wh, ws%ox, ws%oy, ws%oz, &
         ws%dv, ws%du, ws%dw, ws%gg, ws%Hx, ws%Hy, ws%Hz, &
         ws%pu, ws%pv, ws%pw, ws%pox, ws%poy, ws%poz, ws%phx, ws%phy, ws%phz, ws%Wf)
  end subroutine nl_destroy

  !-----------------------------------------------------------------------------
  ! cross_prod  --  H = (u x omega) / (nxd*nzd).  [R2C: pure REAL cross product]
  !
  ! The physical fields are REAL (ny,nxd,nzd), so this is a plain real cross
  ! product with no real()/cmplx() bookkeeping. The F4 Lyapunov-ghost fix (which
  ! took real() of complex physical fields in the C2C path) is now STRUCTURAL:
  ! with a real-x transform the physical field has no imaginary part to carry the
  ! conjugate-asymmetric ghost, and the R2C forward transform makes Ĥ Hermitian
  ! by construction. (Verified on this path in 3b.)
  !-----------------------------------------------------------------------------
  subroutine cross_prod(ny, nxd, nzd, inv, pu, pv, pw, pox, poy, poz, phx, phy, phz)
    integer(ip), intent(in) :: ny, nxd, nzd
    real(wp),    intent(in) :: inv
    real(tp), device :: pu(ny,nxd,nzd), pv(ny,nxd,nzd), pw(ny,nxd,nzd)
    real(tp), device :: pox(ny,nxd,nzd), poy(ny,nxd,nzd), poz(ny,nxd,nzd)
    real(tp), device :: phx(ny,nxd,nzd), phy(ny,nxd,nzd), phz(ny,nxd,nzd)
    integer(ip) :: iy, ix, iz
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nzd
       do ix = 1, nxd
          do iy = 1, ny
             phx(iy,ix,iz) = inv*(pv(iy,ix,iz)*poz(iy,ix,iz) - pw(iy,ix,iz)*poy(iy,ix,iz))
             phy(iy,ix,iz) = inv*(pw(iy,ix,iz)*pox(iy,ix,iz) - pu(iy,ix,iz)*poz(iy,ix,iz))
             phz(iy,ix,iz) = inv*(pu(iy,ix,iz)*poy(iy,ix,iz) - pv(iy,ix,iz)*pox(iy,ix,iz))
          end do
       end do
    end do
  end subroutine cross_prod

  !-- device-to-device copy of a y-column (plain dummies) ---------------------
  subroutine copy_col(ny, src, dst)
    integer(ip), intent(in) :: ny
    complex(tp), device :: src(ny)
    complex(wp), device :: dst(ny)
    integer(ip) :: iy
    !$cuf kernel do(1) <<<*,*>>>
    do iy = 1, ny
       dst(iy) = src(iy)
    end do
  end subroutine copy_col

  !-- element-wise device field copy with kind conversion (whole-array copy is a
  !   cross-kind device assignment nvfortran rejects when tp /= wp) ----------
  subroutine copy_c3(ny, nx, nz, src, dst)
    integer(ip), intent(in) :: ny, nx, nz
    complex(wp), device :: src(ny,nx,nz)
    complex(tp), device :: dst(ny,nx,nz)
    integer(ip) :: iy, ix, iz
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do iy = 1, ny
             dst(iy,ix,iz) = src(iy,ix,iz)
          end do
       end do
    end do
  end subroutine copy_c3

  !-----------------------------------------------------------------------------
  ! mask_nyquist  --  zero the kx- and kz-Nyquist columns of an (ny,nkx,nz)
  ! spectral field (R2C-aware).
  !
  ! R2C stores kx in [0,nkx-1]; the kx-Nyquist (kx=nx/2) is the LAST stored column
  ! (index nkx). The kz-Nyquist (kz=nz/2) is index nz/2+1 as before, but the row
  ! is zeroed only over the STORED x range ix=1..nkx (NOT 1..nx -- that would be
  ! out of bounds on the half-width array). Same invariant as the C2C version:
  ! with a Nyquist-zero state every RHS contribution there is zero, the solves of
  ! a zero RHS return zero, and the state stays zero. (Public so the driver can
  ! sanitise restarted/initial states.)
  !-----------------------------------------------------------------------------
  subroutine mask_nyquist(ny, nx, nz, q)
    integer(ip), intent(in) :: ny, nx, nz
    complex(wp), device     :: q(ny, (nx/2+1), nz)        ! (ny,nkx,nz)
    integer(ip) :: iy, ix, iz, ixN, iz0, nkx
    nkx = nx/2 + 1
    ixN = nkx              ! kx-Nyquist = last stored column
    iz0 = nz/2 + 1         ! kz-Nyquist row
    !$cuf kernel do(2) <<<*,*>>>
    do iz = 1, nz
       do iy = 1, ny
          q(iy, ixN, iz) = (0.0_wp, 0.0_wp)
       end do
    end do
    !$cuf kernel do(2) <<<*,*>>>
    do ix = 1, nkx                                        ! stored x range only
       do iy = 1, ny
          q(iy, ix, iz0) = (0.0_wp, 0.0_wp)
       end do
    end do
  end subroutine mask_nyquist

  !---------------------------------------------------------------------------
  ! Evaluate the nonlinear forcings. v̂,η̂ are preserved; u0,w0 are the mean
  ! profiles. Outputs: ĥ_v, ĥ_η, and the mean forcings hx0=Ĥ_x(0,0,:),
  ! hz0=Ĥ_z(0,0,:). After the call, ws holds the physical fields and products
  ! (ws%pu.. ws%phz) and the spectral Ĥ (ws%Hx,Hy,Hz) for diagnostics/tests.
  ! All spectral arrays are (ny,nkx,nz); physical scratch is REAL (ny,nxd,nzd).
  !---------------------------------------------------------------------------
  ! nl_eval_H -- steps 1-4 only: build the dealiased H = u x omega into ws%Hx/Hy/Hz
  ! (coarse spectral) and leave the dealiased physical velocities in ws%pu/pv/pw.
  ! NO projection onto v,eta (steps 5-6). nl_evaluate calls this then projects; the
  ! single-GPU pressure diagnostic (--validate) calls it alone to get the
  ! SAME dealiased H the step uses + the velocities for the |u|^2/2 recovery. Mirrors
  ! the distributed par_nl_eval_H.
  subroutine nl_eval_H(ws, vhat, ehat, u0, w0)
    type(nl_workspace), intent(inout) :: ws
    complex(wp), device :: vhat(ws%ny,ws%nkx,ws%nz), ehat(ws%ny,ws%nkx,ws%nz)
    complex(wp), device :: u0(ws%ny), w0(ws%ny)
    integer(ip) :: ny, nx, nz, nkx, nxd, nzd
    real(wp) :: inv
    ny = ws%ny;  nx = ws%nx;  nz = ws%nz;  nkx = ws%nkx;  nxd = ws%nxd;  nzd = ws%nzd
    inv = 1.0_wp / real(nxd*nzd, wp)

    ! 1. recover u,w (uh,wh; dv=Dv̂) and vorticity (ox,oy,oz). velocity_mod is
    !    extent-agnostic: pass nkx as the x-extent and the nkx-long kx array.
    call recover_uw(ny, nkx, nz, ws%kx, ws%kz, vhat, ehat, u0, w0, ws%uh, ws%wh, ws%dv)
    call compute_vorticity(ny, nkx, nz, ws%kx, ws%kz, vhat, ehat, ws%uh, ws%wh, &
                           ws%ox, ws%oy, ws%oz, ws%du, ws%dw)

    ! 1b. [Galilean] subtract U_c from the mean streamwise velocity so the cross
    !     product advects with u - U_c. Basis-aware on the yderiv seam:
    !     Cheb = T_0 coeff only, FD = every point value. Vorticity above used
    !     the lab u (curl unchanged by a constant). Serial owns kx=0.
    if (ws%ugal /= 0.0_wp) call gal_shift_mean(ny, ws%ugal, ws%uh(:,1,1))

    ! 2. transform six fields to the dealiased REAL physical grid.
    !    to_physical_da_r2c consumes its complex spectral input, so copy the
    !    preserved state vhat into scratch (dv) first; uh,wh,ox,oy,oz are scratch.
    !    copy_c3 = element-wise demote (ws%dv is tp, vhat is wp); a whole-array
    !    ws%dv=vhat is a cross-kind device copy nvfortran rejects when tp /= wp.
    call copy_c3(ny, nkx, nz, vhat, ws%dv)
    call to_physical_da_r2c(ws%plans, ny,nx,nz,nxd,nzd, ws%dv, ws%pv,  ws%Wf)
    call to_physical_da_r2c(ws%plans, ny,nx,nz,nxd,nzd, ws%uh, ws%pu,  ws%Wf)
    call to_physical_da_r2c(ws%plans, ny,nx,nz,nxd,nzd, ws%wh, ws%pw,  ws%Wf)
    call to_physical_da_r2c(ws%plans, ny,nx,nz,nxd,nzd, ws%ox, ws%pox, ws%Wf)
    call to_physical_da_r2c(ws%plans, ny,nx,nz,nxd,nzd, ws%oy, ws%poy, ws%Wf)
    call to_physical_da_r2c(ws%plans, ny,nx,nz,nxd,nzd, ws%oz, ws%poz, ws%Wf)

    ! 3. rotational nonlinear term H = (u x omega)/(nxd*nzd) -- real cross product
    call cross_prod(ny, nxd, nzd, inv, ws%pu, ws%pv, ws%pw, ws%pox, ws%poy, ws%poz, &
                    ws%phx, ws%phy, ws%phz)

    ! 4. transform H back to spectral (real physical -> complex coarse)
    call to_spectral_da_r2c(ws%plans, ny,nx,nz,nxd,nzd, ws%phx, ws%Wf, ws%Hx)
    call to_spectral_da_r2c(ws%plans, ny,nx,nz,nxd,nzd, ws%phy, ws%Wf, ws%Hy)
    call to_spectral_da_r2c(ws%plans, ny,nx,nz,nxd,nzd, ws%phz, ws%Wf, ws%Hz)
  end subroutine nl_eval_H

  subroutine nl_evaluate(ws, vhat, ehat, u0, w0, hv, heta, hx0, hz0)
    type(nl_workspace), intent(inout) :: ws
    complex(wp), device :: vhat(ws%ny,ws%nkx,ws%nz), ehat(ws%ny,ws%nkx,ws%nz)
    complex(wp), device :: u0(ws%ny), w0(ws%ny)
    complex(wp), device :: hv(ws%ny,ws%nkx,ws%nz), heta(ws%ny,ws%nkx,ws%nz)
    complex(wp), device :: hx0(ws%ny), hz0(ws%ny)
    integer(ip) :: ny, nx, nz, nkx

    ! steps 1-4: dealiased H -> ws%Hx/Hy/Hz, velocities -> ws%pu/pv/pw
    call nl_eval_H(ws, vhat, ehat, u0, w0)
    ny = ws%ny;  nx = ws%nx;  nz = ws%nz;  nkx = ws%nkx

    ! 5. project onto the v and eta equations (forcing_mod is extent-agnostic).
    !    ws%du (dead since compute_vorticity) holds D(g) -- the tp seam scratch.
    call assemble_hv_heta(ny, nkx, nz, ws%kx, ws%kz, ws%Hx, ws%Hy, ws%Hz, hv, heta, ws%gg, ws%du)

    ! 5b. enforce the Nyquist-zero convention on the forcings.
    call mask_nyquist(ny, nx, nz, hv)
    call mask_nyquist(ny, nx, nz, heta)

    ! 6. mean-mode forcings (the (0,0) column of Ĥ_x, Ĥ_z; kx=0 is column 1)
    call copy_col(ny, ws%Hx(:,1,1), hx0)
    call copy_col(ny, ws%Hz(:,1,1), hz0)
  end subroutine nl_evaluate

end module nonlinear_mod
