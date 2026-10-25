!===============================================================================
! cheb_bih.F90  --  Shen recombined-basis solver for the wall-normal
!                   v-equation (fourth order, clamped BCs v = Dv = 0).
!
! Solves, per (kx,kz) mode, the operator that comes out of the KMM v-equation
! after Crank-Nicolson on the viscous term:
!
!     (L - alpha L^2) v = f,   L = D^2 - k^2,   alpha = nu * (CN weight)
!
! Expanded:  -alpha D^4 + (1 + 2 alpha k^2) D^2 - (k^2 + alpha k^4),  i.e.
!     A4 = -alpha,  A2 = 1 + 2 alpha k^2,  A0 = -(k^2 + alpha k^4).
!
! Biharmonic Shen basis:  psi_k = T_k + a_k T_{k+2} + b_k T_{k+4},  k = 0..n-4.
!   a_k, b_k are obtained HERE by solving the boundary-condition system
!     psi_k(+-1) = 0,  psi_k'(+-1) = 0
!   (a 2x2 linear solve), so the basis satisfies v = Dv = 0 by construction
!   rather than by trusting recalled constants. (This reproduces Shen's
!   a_k = -2(k+2)/(k+3), b_k = (k+1)/(k+3), which the BC test confirms.)
!
! As in cheb_shen, the Galerkin matrices are assembled by applying the
! validated Chebyshev derivative recurrence to the basis and taking exact
! weighted inner products. dp throughout; dense LU for now.
!===============================================================================

module cheb_bih
  use kinds,     only: dp, ip
  use cheb_shen, only: cheb_d2, wdot, lu_factor, lu_solve
  implicit none
  private

  public :: bih_ab, bih_basis, shen_bih_to_cheb
  public :: assemble_biharmonic, biharmonic_rhs, biharmonic_solve

contains

  !---------------------------------------------------------------------------
  ! Solve the 2x2 BC system for the basis coefficients a_k, b_k:
  !     [ 1        1      ] [a]   [ -1   ]
  !     [ (k+2)^2  (k+4)^2] [b] = [ -k^2 ]
  !---------------------------------------------------------------------------
  subroutine bih_ab(k, ak, bk)
    integer(ip), intent(in)  :: k
    real(dp),    intent(out) :: ak, bk
    real(dp) :: a11, a12, a21, a22, r1, r2, det
    a11 = 1.0_dp;  a12 = 1.0_dp
    a21 = real((k+2)*(k+2), dp);  a22 = real((k+4)*(k+4), dp)
    r1  = -1.0_dp;  r2 = -real(k*k, dp)
    det = a11*a22 - a12*a21
    ak  = (r1*a22 - a12*r2) / det
    bk  = (a11*r2 - r1*a21) / det
  end subroutine bih_ab

  !---------------------------------------------------------------------------
  ! Biharmonic basis function psi_k in Chebyshev coefficients.
  !---------------------------------------------------------------------------
  subroutine bih_basis(n, k, psi)
    integer(ip), intent(in)  :: n, k
    real(dp),    intent(out) :: psi(0:n)
    real(dp) :: ak, bk
    call bih_ab(k, ak, bk)
    psi = 0.0_dp
    psi(k)   = 1.0_dp
    psi(k+2) = ak
    psi(k+4) = bk
  end subroutine bih_basis

  !---------------------------------------------------------------------------
  ! Convert biharmonic Shen coefficients (0..n-4) to Chebyshev coeffs (0..n).
  !---------------------------------------------------------------------------
  subroutine shen_bih_to_cheb(n, vhat, a)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: vhat(0:n-4)
    real(dp),    intent(out) :: a(0:n)
    integer(ip) :: k
    real(dp)    :: ak, bk
    a = 0.0_dp
    do k = 0, n-4
       call bih_ab(k, ak, bk)
       a(k)   = a(k)   +        vhat(k)
       a(k+2) = a(k+2) + ak   * vhat(k)
       a(k+4) = a(k+4) + bk   * vhat(k)
    end do
  end subroutine shen_bih_to_cheb

  !---------------------------------------------------------------------------
  ! Assemble M = A4*S4 + A2*S2 + A0*B, size (n-3)x(n-3), where
  !   S4_kj = (D^4 psi_j, psi_k)_w,  S2_kj = (D^2 psi_j, psi_k)_w,
  !   B_kj  = (psi_j, psi_k)_w.
  !---------------------------------------------------------------------------
  subroutine assemble_biharmonic(n, k2, alpha, M)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: k2, alpha
    real(dp),    intent(out) :: M(0:n-4,0:n-4)
    real(dp) :: Psi(0:n,0:n-4), D2Psi(0:n,0:n-4), D4Psi(0:n,0:n-4)
    real(dp) :: A4, A2, A0
    integer(ip) :: k, j
    A4 = -alpha
    A2 = 1.0_dp + 2.0_dp*alpha*k2
    A0 = -(k2 + alpha*k2*k2)
    do k = 0, n-4
       call bih_basis(n, k, Psi(:,k))
       call cheb_d2(n, Psi(:,k),   D2Psi(:,k))
       call cheb_d2(n, D2Psi(:,k), D4Psi(:,k))
    end do
    do j = 0, n-4
       do k = 0, n-4
          M(k,j) = A4*wdot(n, D4Psi(:,j), Psi(:,k)) &
                 + A2*wdot(n, D2Psi(:,j), Psi(:,k)) &
                 + A0*wdot(n, Psi(:,j),   Psi(:,k))
       end do
    end do
  end subroutine assemble_biharmonic

  !---------------------------------------------------------------------------
  ! Project a Chebyshev-coeff RHS onto the biharmonic test functions.
  !---------------------------------------------------------------------------
  subroutine biharmonic_rhs(n, fcheb, ftilde)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: fcheb(0:n)
    real(dp),    intent(out) :: ftilde(0:n-4)
    real(dp) :: psik(0:n)
    integer(ip) :: k
    do k = 0, n-4
       call bih_basis(n, k, psik)
       ftilde(k) = wdot(n, fcheb, psik)
    end do
  end subroutine biharmonic_rhs

  !---------------------------------------------------------------------------
  ! Full solve: given k^2, alpha and a Chebyshev-coeff RHS, return the
  ! biharmonic Shen coefficients of v solving (L - alpha L^2) v = f.
  !---------------------------------------------------------------------------
  subroutine biharmonic_solve(n, k2, alpha, fcheb, vhat)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: k2, alpha
    real(dp),    intent(in)  :: fcheb(0:n)
    real(dp),    intent(out) :: vhat(0:n-4)
    real(dp)    :: M(0:n-4,0:n-4), rhs(0:n-4)
    integer(ip) :: piv(0:n-4), m_
    m_ = n-3
    call assemble_biharmonic(n, k2, alpha, M)
    call biharmonic_rhs(n, fcheb, rhs)
    call lu_factor(m_, M, piv)
    call lu_solve(m_, M, piv, rhs)
    vhat = rhs
  end subroutine biharmonic_solve

end module cheb_bih
