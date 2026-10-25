!===============================================================================
! velocity_mod.F90  --  velocity / vorticity reconstruction (physics step 1).
!
! Given the evolved spectral fields v̂, η̂ (layout (ny,nx,nz), y=Chebyshev coeffs,
! x,z=Fourier) and the mean profiles u0(y), w0(y), reconstruct the in-plane
! velocities and the vorticity for the nonlinear evaluation. From the KMM
! formulation (see kmm_formulation.pdf), for k^2 = kx^2+kz^2 > 0:
!
!   û = (i/k^2)(kx Dv̂ - kz η̂),   ŵ = (i/k^2)(kz Dv̂ + kx η̂)
!   ω̂_x = Dŵ - i kz v̂,   ω̂_y = η̂,   ω̂_z = i kx v̂ - Dû
!
! At the mean mode (kx,kz)=(0,0) the recovery is singular: v̂ ≡ η̂ ≡ 0 there and
! û,ŵ are the separately evolved means u0,w0. The vorticity formula is regular
! at (0,0) (the i*k*v̂ terms vanish), so it needs no special case.
!
! D is the wall-normal derivative through the y-basis dispatch seam (y_deriv_tp:
! the Chebyshev coefficient-space recurrence, or the FD-y banded D1 on point
! values; the formulas here are basis-agnostic). kx,kz are the
! Fourier wavenumbers in cuFFT order. Wavenumbers and the k^2/inv
! algebra are WP: at TP=sp the kz spacing 2/3 is not fp32-representable, and a
! wrong wavenumber scales the FIELD everywhere. The k-scalars live in registers
! (free); only the final product is stored back to tp.
!
! Requires -cuda on a GPU node.
!===============================================================================

module velocity_mod
  use kinds,          only: tp, wp, ip
  use cudafor
  use yderiv_gpu_mod, only: y_deriv_tp        ! y-basis dispatch seam
  implicit none
  private
  public :: recover_uw, compute_vorticity

contains

  !---------------------------------------------------------------------------
  ! Recover û, ŵ from v̂, η̂ (+ mean u0,w0). dv returns Dv̂ (also used internally).
  !---------------------------------------------------------------------------
  subroutine recover_uw(ny, nx, nz, kx, kz, vhat, ehat, u0, w0, uhat, what, dv)
    integer(ip), intent(in) :: ny, nx, nz
    real(wp),    device     :: kx(nx), kz(nz)
    complex(wp), device     :: vhat(ny,nx,nz), ehat(ny,nx,nz)
    complex(wp), device     :: u0(ny), w0(ny)
    complex(tp), device     :: uhat(ny,nx,nz), what(ny,nx,nz)
    complex(tp), device     :: dv(ny,nx,nz)
    integer(ip) :: iy, ix, iz
    real(wp)    :: kxx, kzz, k2, inv
    complex(wp), parameter :: ii = (0.0_wp, 1.0_wp)

    ! demote v-hat (wp) into the tp uhat scratch, then differentiate in tp; uhat is
    ! read in full by y_deriv_tp before the kernel below overwrites it. The
    ! demote is an explicit element-wise kernel -- a whole-array uhat=vhat is a
    ! cross-kind device copy nvfortran rejects when tp/=wp.
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do iy = 1, ny
             uhat(iy,ix,iz) = vhat(iy,ix,iz)
          end do
       end do
    end do
    call y_deriv_tp(ny, nx*nz, uhat, dv)

    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do iy = 1, ny
             kxx = kx(ix);  kzz = kz(iz);  k2 = kxx*kxx + kzz*kzz
             if (k2 > 0.0_wp) then
                inv = 1.0_wp / k2
                uhat(iy,ix,iz) = ii*inv*(kxx*dv(iy,ix,iz) - kzz*ehat(iy,ix,iz))
                what(iy,ix,iz) = ii*inv*(kzz*dv(iy,ix,iz) + kxx*ehat(iy,ix,iz))
             else
                uhat(iy,ix,iz) = u0(iy)        ! mean mode (single (0,0) column)
                what(iy,ix,iz) = w0(iy)
             end if
          end do
       end do
    end do
  end subroutine recover_uw

  !---------------------------------------------------------------------------
  ! Vorticity components from (û, v̂, ŵ, η̂). du, dw are scratch (Dû, Dŵ).
  !---------------------------------------------------------------------------
  subroutine compute_vorticity(ny, nx, nz, kx, kz, vhat, ehat, uhat, what, &
                               ox, oy, oz, du, dw)
    integer(ip), intent(in) :: ny, nx, nz
    real(wp),    device     :: kx(nx), kz(nz)
    complex(wp), device     :: vhat(ny,nx,nz), ehat(ny,nx,nz)
    complex(tp), device     :: uhat(ny,nx,nz), what(ny,nx,nz)
    complex(tp), device     :: ox(ny,nx,nz), oy(ny,nx,nz), oz(ny,nx,nz)
    complex(tp), device     :: du(ny,nx,nz), dw(ny,nx,nz)
    integer(ip) :: iy, ix, iz
    real(wp)    :: kxx, kzz
    complex(wp), parameter :: ii = (0.0_wp, 1.0_wp)

    call y_deriv_tp(ny, nx*nz, uhat, du)
    call y_deriv_tp(ny, nx*nz, what, dw)

    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do iy = 1, ny
             kxx = kx(ix);  kzz = kz(iz)
             ox(iy,ix,iz) = dw(iy,ix,iz) - ii*kzz*vhat(iy,ix,iz)
             oy(iy,ix,iz) = ehat(iy,ix,iz)
             oz(iy,ix,iz) = ii*kxx*vhat(iy,ix,iz) - du(iy,ix,iz)
          end do
       end do
    end do
  end subroutine compute_vorticity

end module velocity_mod
