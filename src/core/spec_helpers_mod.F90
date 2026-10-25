!===============================================================================
! spec_helpers_mod.F90  --  batched GPU spectral helpers used by the GPU vif
! (influence-matrix) v-solve. Complex spectral fields, batched over (kx,kz).
!
!   project_shen : Chebyshev coeffs f(0:n) -> Shen-Dirichlet test projection
!                  ftil(k) = (f, phi_k)_w = (pi/2)(c_k f_k - f_{k+2}),
!                  c_0 = 2, c_k = 1 (k>=1). Matches cheb_shen's helmholtz_rhs;
!                  this is the RHS that helm_band_solve_gpu consumes.
!   shen_to_cheb_c : Shen-Dirichlet coeffs u(0:n-2) -> Chebyshev coeffs a(0:n),
!                  a_j = u_j - u_{j-2}  (phi_k = T_k - T_{k+2} basis).
!   wall_derivs_c : Chebyshev coeffs a(0:n) -> first derivative at the walls,
!                  Du(+1) = sum_j j^2 a_j,  Du(-1) = sum_j (-1)^{j-1} j^2 a_j.
!
! Real coefficients act on the complex data. One thread per mode (cuf kernels).
!
! Build:  nvfortran -cuda -c spec_helpers_mod.F90
!===============================================================================
module spec_helpers_mod
  use kinds,  only: dp, ip
  use cudafor
  implicit none
  private
  public :: project_shen, shen_to_cheb_c, wall_derivs_c, wall_values_c
  real(dp), parameter :: PI = 3.141592653589793238462643383279_dp

contains

  subroutine project_shen(n, nx, nz, fc, ftil)
    integer(ip), intent(in) :: n, nx, nz
    complex(dp), device :: fc(0:n, nx, nz)        ! in : Chebyshev coeffs
    complex(dp), device :: ftil(0:n-2, nx, nz)    ! out: Shen-test projection
    integer(ip) :: ix, iz, k
    real(dp)    :: ck
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, n-2
             if (k == 0) then; ck = 2.0_dp; else; ck = 1.0_dp; end if
             ftil(k,ix,iz) = 0.5_dp*PI*( ck*fc(k,ix,iz) - fc(k+2,ix,iz) )
          end do
       end do
    end do
  end subroutine project_shen

  subroutine shen_to_cheb_c(n, nx, nz, u, a)
    integer(ip), intent(in) :: n, nx, nz
    complex(dp), device :: u(0:n-2, nx, nz)       ! in : Shen coeffs
    complex(dp), device :: a(0:n, nx, nz)         ! out: Chebyshev coeffs
    integer(ip) :: ix, iz, j
    complex(dp) :: uj, ujm2
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do j = 0, n
             if (j <= n-2) then; uj   = u(j,ix,iz);   else; uj   = (0.0_dp,0.0_dp); end if
             if (j >= 2)   then; ujm2 = u(j-2,ix,iz); else; ujm2 = (0.0_dp,0.0_dp); end if
             a(j,ix,iz) = uj - ujm2
          end do
       end do
    end do
  end subroutine shen_to_cheb_c

  subroutine wall_derivs_c(n, nx, nz, a, dlo, dhi)
    integer(ip), intent(in) :: n, nx, nz
    complex(dp), device :: a(0:n, nx, nz)         ! in : Chebyshev coeffs
    complex(dp), device :: dlo(nx, nz), dhi(nx, nz) ! out: Du(-1), Du(+1) per mode
    integer(ip) :: ix, iz, j, s
    complex(dp) :: slo, shi
    real(dp)    :: jj
    !$cuf kernel do(2) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          slo = (0.0_dp,0.0_dp); shi = (0.0_dp,0.0_dp)
          do j = 1, n
             jj  = real(j*j, dp)
             shi = shi + a(j,ix,iz)*jj
             if (mod(j,2) == 1) then; s = 1; else; s = -1; end if
             slo = slo + real(s,dp)*a(j,ix,iz)*jj
          end do
          dlo(ix,iz) = slo
          dhi(ix,iz) = shi
       end do
    end do
  end subroutine wall_derivs_c

  ! wall VALUES (not derivatives) from Chebyshev coeffs:  T_j(+1)=1, T_j(-1)=(-1)^j
  !   f(+1) = sum_{j=0}^n a_j ,  f(-1) = sum_{j=0}^n (-1)^j a_j .
  ! INCLUDES j=0 (the mean coeff) -- unlike wall_derivs_c (Du, where T_0'=0 drops
  ! j=0). Used for the pressure Neumann wall data g_+- = Hy(+-1) + (1/Re)D^2v(+-1).
  ! flo = f(-1), fhi = f(+1), per mode (one thread per (kx,kz)).
  subroutine wall_values_c(n, nx, nz, a, flo, fhi)
    integer(ip), intent(in) :: n, nx, nz
    complex(dp), device :: a(0:n, nx, nz)            ! in : Chebyshev coeffs
    complex(dp), device :: flo(nx, nz), fhi(nx, nz)  ! out: f(-1), f(+1) per mode
    integer(ip) :: ix, iz, j, s
    complex(dp) :: slo, shi
    !$cuf kernel do(2) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          slo = (0.0_dp,0.0_dp); shi = (0.0_dp,0.0_dp)
          do j = 0, n
             shi = shi + a(j,ix,iz)
             if (mod(j,2) == 0) then; s = 1; else; s = -1; end if   ! (-1)^j
             slo = slo + real(s,dp)*a(j,ix,iz)
          end do
          flo(ix,iz) = slo
          fhi(ix,iz) = shi
       end do
    end do
  end subroutine wall_values_c

end module spec_helpers_mod
