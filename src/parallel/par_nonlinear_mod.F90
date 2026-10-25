!===============================================================================
! par_nonlinear_mod.F90  --  multi-GPU DISTRIBUTED rotational nonlinear RHS
! H = (u x omega), projected onto the (v, eta) equations -- the distributed
! counterpart of nonlinear_mod's nl_evaluate, on the 1-D slab decomposition.
!
! KEY ENABLER (verified, not assumed): the spectral operators are EXTENT-AGNOSTIC
! and VALUE-BASED. recover_uw / compute_vorticity / assemble_hv_heta are local in
! (kx,kz) with no negative-kx coupling, and the singular (0,0) mean mode is
! detected by  k2 = kx^2 + kz^2 == 0  -- NOT by column position. So on a kx-slab
! they run UNCHANGED, given THIS rank's LOCAL kx array (the slice kx(b_kx0+1:..)):
! rank 0 owns kx=0 so its mean branch fires on column 1; every other rank's local
! kx values are all > 0 so the branch correctly never fires. cross_prod is
! pointwise so it runs on the local z-slab, and inv = 1/(nxd*nzd) stays the GLOBAL
! normalisation (the distributed transform produces the same physical values as
! the monolithic one, just on this rank's z-slab).
!
! PIPELINE (mirrors nl_evaluate exactly; the only distributed pieces are the
! transforms and two small slab-aware bits):
!   recover_uw ; compute_vorticity                         [Layout B, local kx]
!   dv := vhat ; 6x to_physical  (kx-slab -> z-slab)   [distributed transform]
!   cross_prod                                             [Layout A, local z]
!   3x to_spectral (z-slab -> kx-slab)                 [distributed transform]
!   assemble_hv_heta                                       [Layout B, local kx]
!   mask_nyquist_slab : kz-Nyquist row is local to every rank; the kx-Nyquist
!     column (global kx=nx/2) lives only on the rank owning the last coarse kx.
!   mean-mode Hx,Hz(:,1,1) -> hx0,hz0 : only the rank owning kx=0 (owns_mode00).
!
! The 6/3 transforms each do their OWN transpose (9 all-to-alls). The six
! fields and three H-fields are laid out as STACKED buffers S6(...,6)/PH(...,3)
! from the start, so a batched version is a one-line swap (six single
! calls -> one nfield=6 call) with everything else unchanged.
!
! Single-GPU nl_evaluate is untouched; this lives in parallel/ and is built only
! by parallel/build_run_nonlinear_mp.sh. PRECONDITION: P<=min(nkx,nzd).
!===============================================================================
module par_nonlinear_mod
  use kinds,              only: tp, wp, ip
  use cudafor
  use decomp_mod,         only: decomp_t
  use velocity_mod,       only: recover_uw, compute_vorticity
  use forcing_mod,        only: assemble_hv_heta
  use yderiv_gpu_mod,     only: gal_shift_mean   ! basis-aware Galilean shift
  use par_spectral3d_mod, only: par_s3d_plans, par_s3d_setup_da_r2c, &
                                par_s3d_destroy_da_r2c, &
                                par_to_physical_nf_begin, par_to_physical_nf_finish, &
                                par_to_spectral_nf_begin, par_to_spectral_nf_finish
  implicit none
  private
  public :: par_nl_workspace, par_nl_setup, par_nl_destroy, par_nl_evaluate, par_nl_eval_H
  public :: par_nl_ev_begin, par_nl_ev_mid, par_nl_ev_end   ! hoist phases

  type :: par_nl_workspace
     integer(ip) :: ny, nx, nz, nkx, nxd, nzd, nkxd
     integer(ip) :: bnkx, anz, b_kx0           ! this rank's local slab extents/offset
     logical     :: owns_mode00
     real(wp)    :: ugal = 0.0_wp              ! [Galilean] constant frame speed U_c (0 = lab frame)
     type(decomp_t)      :: d
     type(par_s3d_plans) :: plans
     real(wp),    device, allocatable :: kx(:), kz(:)        ! LOCAL bnkx-long / nz-long; wp
     ! stacked spectral scratch (ny, bnkx, nz, *) -- the six fields and three H-fields
     complex(tp), device, allocatable :: S6(:,:,:,:)         ! [dv,uh,wh,ox,oy,oz]
     complex(tp), device, allocatable :: S3(:,:,:,:)         ! [Hx,Hy,Hz]
     complex(tp), device, allocatable :: du(:,:,:), dw(:,:,:), gg(:,:,:)   ! scratch
     ! stacked physical scratch (ny, nxd, anz, *) -- REAL
     real(tp),    device, allocatable :: P6(:,:,:,:)         ! [pv,pu,pw,pox,poy,poz]
     real(tp),    device, allocatable :: PH(:,:,:,:)         ! [phx,phy,phz]
     ! transpose-endpoint scratch, sized for the BATCHED transform: nfield=6 for the
     ! inverse; the nfield=3 forward reuses the leading 1:3 field-slices.
     complex(tp), device, allocatable :: Bz(:,:,:,:), Az(:,:,:,:)
     complex(tp), device, allocatable :: Af(:,:,:)   ! ONE x-scratch slice (consumed per field)
  end type par_nl_workspace

contains

  !-----------------------------------------------------------------------------
  ! ORDERING CONTRACT: intent(out)
  ! makes every DEFAULT-INITIALIZED component take its initializer on entry
  ! (ws%ugal = 0 -- lab frame, Galilean shift OFF -- is the one that matters;
  ! members without initializers become undefined, allocatables deallocate).
  ! Any driver must therefore set ws%ugal strictly AFTER this call (main_par
  ! does, at the pws%ugal assignment right below its par_nl_setup call).
  ! Setting it before would be silently discarded.
  subroutine par_nl_setup(ws, d, ny, nx, nz, kx_h, kz_h)
    type(par_nl_workspace), intent(out) :: ws
    type(decomp_t),         intent(in)  :: d
    integer(ip),            intent(in)  :: ny, nx, nz
    real(wp),               intent(in)  :: kx_h(nx/2+1), kz_h(nz)   ! GLOBAL kx (nkx-long), kz
    integer(ip) :: nxd, nzd, nkx, nkxd, bnkx, anz
    nxd = 3_ip*nx/2_ip;  nzd = 3_ip*nz/2_ip;  nkx = nx/2_ip + 1_ip;  nkxd = nxd/2_ip + 1_ip
    bnkx = d%b_nkx_loc;  anz = d%ad_nz_loc
    ws%ny = ny;  ws%nx = nx;  ws%nz = nz;  ws%nxd = nxd;  ws%nzd = nzd
    ws%nkx = nkx;  ws%nkxd = nkxd
    ws%bnkx = bnkx;  ws%anz = anz;  ws%b_kx0 = d%b_kx0;  ws%owns_mode00 = d%owns_mode00
    ws%d = d
    call par_s3d_setup_da_r2c(ws%plans, d, ny, nx, nz)
    allocate(ws%kx(bnkx), ws%kz(nz))
    ws%kx = kx_h(d%b_kx0+1 : d%b_kx0+bnkx)                  ! THIS rank's local kx slice (wp)
    ws%kz = kz_h
    allocate(ws%S6(ny,bnkx,nz,6), ws%S3(ny,bnkx,nz,3))
    allocate(ws%du(ny,bnkx,nz), ws%dw(ny,bnkx,nz), ws%gg(ny,bnkx,nz))
    allocate(ws%P6(ny,nxd,anz,6), ws%PH(ny,nxd,anz,3))
    allocate(ws%Bz(ny,bnkx,nzd,6), ws%Az(ny,nkx,anz,6), ws%Af(ny,nkxd,anz))
  end subroutine par_nl_setup

  subroutine par_nl_destroy(ws)
    type(par_nl_workspace), intent(inout) :: ws
    call par_s3d_destroy_da_r2c(ws%plans)
    if (allocated(ws%kx)) deallocate(ws%kx, ws%kz)
    if (allocated(ws%S6)) deallocate(ws%S6, ws%S3, ws%du, ws%dw, ws%gg, &
                                     ws%P6, ws%PH, ws%Bz, ws%Az, ws%Af)
  end subroutine par_nl_destroy

  !-----------------------------------------------------------------------------
  ! par_nl_eval_H -- steps 1-5 of the nonlinear evaluation ONLY: build the
  ! dealiased rotational term H = u x omega and leave it in ws%S3 (coarse spectral
  ! [Hx,Hy,Hz], Layout B kx-slab, Chebyshev-y coeffs), with the dealiased physical
  ! velocities in ws%P6(:,:,:,1:3) = [pv,pu,pw] (z-slab). NO projection onto
  ! v,eta (steps 6-8). par_nl_evaluate calls this then projects; the pressure
  ! diagnostic calls this alone -> it gets the SAME dealiased H the
  ! step uses, plus the velocities it needs for the |u|^2/2 recovery, in one pass.
  !-----------------------------------------------------------------------------
  subroutine par_nl_eval_H(ws, vhat, ehat, u0, w0)
    type(par_nl_workspace), intent(inout) :: ws
    complex(wp), device :: vhat(ws%ny,ws%bnkx,ws%nz), ehat(ws%ny,ws%bnkx,ws%nz)
    complex(wp), device :: u0(ws%ny), w0(ws%ny)
    call par_nl_ev_begin(ws, vhat, ehat, u0, w0)
    call par_nl_ev_mid(ws)
    call par_nl_ev_land(ws)
  end subroutine par_nl_eval_H

  ! The SAME steps 1-5, split at the two flight boundaries so the
  ! stepper can issue its independent L-operator kernels while fields fly:
  !   ev_begin = steps 1,1b,2 + inverse legs UP TO the all-to-all launches
  !   ev_mid   = inverse landings + cross product + forward legs up to the a2a
  !   ev_land  = forward landings (S3 complete)      [private]
  !   ev_end   = ev_land + steps 6-8                 [public; evaluate's tail]
  ! begin+mid+land together are exactly par_nl_eval_H (the wrapper above), so
  ! eval_H callers (pressure) run the identical statement sequence.
  subroutine par_nl_ev_begin(ws, vhat, ehat, u0, w0)
    type(par_nl_workspace), intent(inout) :: ws
    complex(wp), device :: vhat(ws%ny,ws%bnkx,ws%nz), ehat(ws%ny,ws%bnkx,ws%nz)
    complex(wp), device :: u0(ws%ny), w0(ws%ny)
    integer(ip) :: ny, nx, nz, nxd, nzd, bnkx, anz

    ny = ws%ny;  nx = ws%nx;  nz = ws%nz;  nxd = ws%nxd;  nzd = ws%nzd
    bnkx = ws%bnkx;  anz = ws%anz

    ! 1. recover u,w (->S6 2,3; dv=Dv̂ in S6 1) and vorticity (->S6 4,5,6) on the
    !    LOCAL kx-slab with the LOCAL kx array (value-based, mean mode safe).
    call recover_uw(ny, bnkx, nz, ws%kx, ws%kz, vhat, ehat, u0, w0, &
                    ws%S6(:,:,:,2), ws%S6(:,:,:,3), ws%S6(:,:,:,1))
    call compute_vorticity(ny, bnkx, nz, ws%kx, ws%kz, vhat, ehat, &
                           ws%S6(:,:,:,2), ws%S6(:,:,:,3), &
                           ws%S6(:,:,:,4), ws%S6(:,:,:,5), ws%S6(:,:,:,6), ws%du, ws%dw)

    ! 1b. [Galilean] shift to a frame moving at U_c: subtract U_c from the mean
    !     streamwise velocity so the cross product below advects with u - U_c.
    !     Basis-aware on the yderiv seam: Cheb = T_0 coeff only, FD = every
    !     point value. Only the kx=0 owner carries the mean; vorticity was
    !     already formed from the lab u (curl is unchanged by a constant). v,eta
    !     and the (0,0) mean FORCING are frame-invariant, so the pressure recovered
    !     downstream (p = Pi - |u|^2/2, both moving-frame) is the exact physical p.
    if (ws%owns_mode00 .and. ws%ugal /= 0.0_wp) &
       call gal_shift_mean(ny, ws%ugal, ws%S6(:,1,1,2))

    ! 2. dv := demoted vhat (overwrites S6(:,:,:,1), recover_uw's Dv̂ scratch).
    call copy_c3(ny, bnkx, nz, vhat, ws%S6(:,:,:,1))

    ! 3 (first half). six fields: inverse legs + the all-to-all LAUNCHES.
    call par_to_physical_nf_begin(ws%plans, ws%d, ny, nx, nz, nxd, nzd, bnkx, anz, &
                                  6_ip, ws%S6, ws%Bz, ws%Az)
  end subroutine par_nl_ev_begin

  subroutine par_nl_ev_mid(ws)
    type(par_nl_workspace), intent(inout) :: ws
    integer(ip) :: ny, nx, nz, nxd, nzd, bnkx, anz
    real(wp) :: inv
    ny = ws%ny;  nx = ws%nx;  nz = ws%nz;  nxd = ws%nxd;  nzd = ws%nzd
    bnkx = ws%bnkx;  anz = ws%anz
    inv = 1.0_wp / real(nxd*nzd, wp)                       ! GLOBAL FFT normalisation

    ! 3 (second half). landings + x legs -> dealiased REAL physical (z-slab).
    call par_to_physical_nf_finish(ws%plans, ws%d, ny, nx, nz, nxd, nzd, bnkx, anz, &
                                   6_ip, ws%Az, ws%Af, ws%P6)

    ! 4. rotational product H = (u x omega)/(nxd*nzd), pointwise on the local z-slab.
    !    Stacking: pv=P6(1) pu=P6(2) pw=P6(3) pox=P6(4) poy=P6(5) poz=P6(6).
    call cross_prod(ny, nxd, anz, inv, ws%P6(:,:,:,2), ws%P6(:,:,:,1), ws%P6(:,:,:,3), &
                    ws%P6(:,:,:,4), ws%P6(:,:,:,5), ws%P6(:,:,:,6), &
                    ws%PH(:,:,:,1), ws%PH(:,:,:,2), ws%PH(:,:,:,3))

    ! 5 (first half). three H-fields: forward legs + the all-to-all LAUNCHES.
    call par_to_spectral_nf_begin(ws%plans, ws%d, ny, nx, nz, nxd, nzd, bnkx, anz, &
                                  3_ip, ws%PH, ws%Af, ws%Az(:,:,:,1:3), ws%Bz(:,:,:,1:3))
  end subroutine par_nl_ev_mid

  subroutine par_nl_ev_land(ws)
    type(par_nl_workspace), intent(inout) :: ws
    integer(ip) :: ny, nx, nz, nxd, nzd, bnkx, anz
    ny = ws%ny;  nx = ws%nx;  nz = ws%nz;  nxd = ws%nxd;  nzd = ws%nzd
    bnkx = ws%bnkx;  anz = ws%anz
    ! 5 (second half). forward landings -> coarse spectral S3.
    call par_to_spectral_nf_finish(ws%plans, ws%d, ny, nx, nz, nxd, nzd, bnkx, anz, &
                                   3_ip, ws%Bz(:,:,:,1:3), ws%S3)
  end subroutine par_nl_ev_land

  subroutine par_nl_ev_end(ws, hv, heta, hx0, hz0)
    type(par_nl_workspace), intent(inout) :: ws
    complex(wp), device :: hv(ws%ny,ws%bnkx,ws%nz), heta(ws%ny,ws%bnkx,ws%nz)
    complex(wp), device :: hx0(ws%ny), hz0(ws%ny)
    integer(ip) :: ny, nx, nz, bnkx
    call par_nl_ev_land(ws)
    ny = ws%ny;  nx = ws%nx;  nz = ws%nz;  bnkx = ws%bnkx
    ! 6. project onto v, eta (local kx-slab; gg, du are scratch).
    call assemble_hv_heta(ny, bnkx, nz, ws%kx, ws%kz, &
                          ws%S3(:,:,:,1), ws%S3(:,:,:,2), ws%S3(:,:,:,3), hv, heta, ws%gg, ws%du)
    ! 7. Nyquist-zero convention (slab-aware).
    call mask_nyquist_slab(ny, nx, nz, ws%b_kx0, bnkx, hv)
    call mask_nyquist_slab(ny, nx, nz, ws%b_kx0, bnkx, heta)
    ! 8. mean-mode forcings ((0,0) on the owning rank only).
    if (ws%owns_mode00) then
       call copy_col(ny, ws%S3(:,1,1,1), hx0)     ! Hx(:,1,1)
       call copy_col(ny, ws%S3(:,1,1,3), hz0)     ! Hz(:,1,1)
    end if
  end subroutine par_nl_ev_end

  !-----------------------------------------------------------------------------
  ! Distributed nonlinear evaluation. Inputs/outputs are THIS rank's Layout-B
  ! kx-slabs (ny, bnkx, nz) in wp; u0,w0 are the (shared) mean profiles; hx0,hz0
  ! are the (0,0) mean forcings, written only on the rank that owns kx=0.
  !-----------------------------------------------------------------------------
  subroutine par_nl_evaluate(ws, vhat, ehat, u0, w0, hv, heta, hx0, hz0)
    type(par_nl_workspace), intent(inout) :: ws
    complex(wp), device :: vhat(ws%ny,ws%bnkx,ws%nz), ehat(ws%ny,ws%bnkx,ws%nz)
    complex(wp), device :: u0(ws%ny), w0(ws%ny)
    complex(wp), device :: hv(ws%ny,ws%bnkx,ws%nz), heta(ws%ny,ws%bnkx,ws%nz)
    complex(wp), device :: hx0(ws%ny), hz0(ws%ny)

    ! phases back-to-back == the full steps 1-8 in order
    call par_nl_ev_begin(ws, vhat, ehat, u0, w0)
    call par_nl_ev_mid(ws)
    call par_nl_ev_end(ws, hv, heta, hx0, hz0)
  end subroutine par_nl_evaluate

  !=============================================================================
  ! Slab-aware Nyquist mask. q is THIS rank's kx-slab (ny, nkxl, nz), wp. Always
  ! zero the kz-Nyquist row (z is whole on every rank); zero the kx-Nyquist column
  ! (global kx=nx/2 = last coarse kx) only on the rank that owns it.
  !=============================================================================
  subroutine mask_nyquist_slab(ny, nx, nz, b_kx0, nkxl, q)
    integer(ip), intent(in) :: ny, nx, nz, b_kx0, nkxl
    complex(wp), device     :: q(ny, nkxl, nz)
    integer(ip) :: iy, ix, iz, iz0, nkx, ixN
    nkx = nx/2 + 1
    iz0 = nz/2 + 1
    !$cuf kernel do(2) <<<*,*>>>
    do ix = 1, nkxl
       do iy = 1, ny
          q(iy, ix, iz0) = (0.0_wp, 0.0_wp)
       end do
    end do
    if (b_kx0 + nkxl == nkx) then          ! this rank owns the kx-Nyquist column
       ixN = nkxl
       !$cuf kernel do(2) <<<*,*>>>
       do iz = 1, nz
          do iy = 1, ny
             q(iy, ixN, iz) = (0.0_wp, 0.0_wp)
          end do
       end do
    end if
  end subroutine mask_nyquist_slab

  !-- the three small kernels reused verbatim from nonlinear_mod (kept private
  !   here so this module depends only on the spectral ops + the distributed transform) --
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

end module par_nonlinear_mod
