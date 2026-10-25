!===============================================================================
! forcing_mod.F90  --  curl / curl-curl projection of the nonlinear vector H
!                      onto the v and eta equations (physics step 3).
!
! Given the spectral nonlinear vector (Ĥ_x, Ĥ_y, Ĥ_z) (layout (ny,nx,nz),
! y=Chebyshev coeffs, x,z=Fourier), assemble the explicit forcings of the v-
! and eta-equations (see kmm_formulation.pdf):
!
!   ĥ_v   = -k^2 Ĥ_y - i D(kx Ĥ_x + kz Ĥ_z)
!   ĥ_eta = i (kz Ĥ_x - kx Ĥ_z)
!
! where D is the wall-normal derivative through the y-basis dispatch seam
! (y_deriv_tp: Chebyshev recurrence or FD-y banded D1) and
! k^2 = kx^2 + kz^2. At (kx,kz)=(0,0) both forcings vanish identically (every
! i*k factor is zero and the derivative argument is zero); the mean profiles
! u0,w0 are driven separately by Ĥ_x(0,0), Ĥ_z(0,0), read directly by the caller.
!
! Because the projections are exactly the operators that annihilate gradients,
! feeding H = grad(phi) yields ĥ_v = ĥ_eta = 0 -- the discrete statement that
! pressure drops out (used as a test).
!
! Requires -cuda on a GPU node.
!===============================================================================

module forcing_mod
  use kinds,          only: tp, wp, ip
  use cudafor
  use yderiv_gpu_mod, only: y_deriv_tp        ! y-basis dispatch seam
  implicit none
  private
  public :: assemble_hv_heta

contains

  subroutine assemble_hv_heta(ny, nx, nz, kx, kz, Hx, Hy, Hz, hv, heta, g, dg)
    integer(ip), intent(in) :: ny, nx, nz
    real(wp),    device     :: kx(nx), kz(nz)
    complex(tp), device     :: Hx(ny,nx,nz), Hy(ny,nx,nz), Hz(ny,nx,nz)
    complex(wp), device     :: hv(ny,nx,nz), heta(ny,nx,nz)
    complex(tp), device     :: g(ny,nx,nz)          ! scratch
    complex(tp), device     :: dg(ny,nx,nz)         ! scratch: D(g)
    integer(ip) :: iy, ix, iz
    real(wp)    :: kxx, kzz, k2
    complex(wp), parameter :: ii = (0.0_wp, 1.0_wp)

    ! g = kx Hx + kz Hz ;  heta = i (kz Hx - kx Hz)
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do iy = 1, ny
             kxx = kx(ix);  kzz = kz(iz)
             g(iy,ix,iz)    = kxx*Hx(iy,ix,iz) + kzz*Hz(iy,ix,iz)
             heta(iy,ix,iz) = ii*(kzz*Hx(iy,ix,iz) - kxx*Hz(iy,ix,iz))
          end do
       end do
    end do

    ! hv <- D(g)
    call y_deriv_tp(ny, nx*nz, g, dg)

    ! hv = -k^2 Hy - i D(g)
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do iy = 1, ny
             kxx = kx(ix);  kzz = kz(iz);  k2 = kxx*kxx + kzz*kzz
             hv(iy,ix,iz) = -k2*Hy(iy,ix,iz) - ii*dg(iy,ix,iz)
          end do
       end do
    end do
  end subroutine assemble_hv_heta

end module forcing_mod
