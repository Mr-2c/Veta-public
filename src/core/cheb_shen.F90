!===============================================================================
! cheb_shen.F90  --  Wall-normal core: Chebyshev coefficient-space operators
!                    and the Shen recombined-basis Helmholtz solver (Dirichlet).
!
! This is the accuracy-critical wall-normal machinery, so it works entirely in
! dp (see src/core/types.F90). It is pure host math: NO CUDA, NO cuFFT, NO LAPACK,
! NO external test framework -- compiles and runs with plain nvfortran so the
! numerics can be validated on the CPU before anything touches a GPU.
!
! KEY DESIGN CHOICE (correctness over cleverness):
!   The Galerkin stiffness (S) and mass (B) matrices are NOT hardcoded from
!   Shen's closed-form constants. They are assembled by applying the standard
!   Chebyshev derivative recurrence to the basis functions and taking exact
!   Chebyshev-weighted inner products. Every entry therefore derives from two
!   well-understood primitives: the derivative recurrence and Chebyshev
!   orthogonality (T_i,T_j)_w = (pi/2) c_i delta_ij, c_0=2 else 1.
!   Banded storage + the batched GPU solve live in helm_band_mod and
!   helm_band_gpu_mod; here the matrices are dense and solved with a small
!   in-file LU (ny is small).
!
! CONVENTIONS
!   Chebyshev coefficients: a(0:n), u(x) = sum_{k=0}^{n} a_k T_k(x),  N = n+1.
!   Shen-Dirichlet basis:    phi_k = T_k - T_{k+2},  k = 0..n-2  (M = n-1 dof).
!     phi_k(+-1) = 0, so every basis function satisfies homogeneous Dirichlet.
!   Helmholtz problem solved:  -u'' + mu u = f,  u(+-1) = 0.
!     Galerkin:  (S + mu B) uhat = ftilde,  ftilde_k = (f, phi_k)_w.
!===============================================================================

module cheb_shen
  use kinds, only: dp, ip
  implicit none
  private

  public :: glc_nodes, cheb_deriv, cheb_d2, cheb_forward, cheb_eval
  public :: shen_to_cheb, assemble_helmholtz, helmholtz_rhs, helmholtz_solve
  public :: lu_factor, lu_solve, wdot

  real(dp), parameter :: PI = acos(-1.0_dp)

contains

  !---------------------------------------------------------------------------
  ! Chebyshev-Gauss-Lobatto nodes: x_j = cos(pi j / n), j = 0..n. Includes +-1.
  !---------------------------------------------------------------------------
  subroutine glc_nodes(n, x)
    integer(ip), intent(in)  :: n
    real(dp),    intent(out) :: x(0:n)
    integer(ip) :: j
    do j = 0, n
       x(j) = cos(PI * real(j,dp) / real(n,dp))
    end do
  end subroutine glc_nodes

  !---------------------------------------------------------------------------
  ! First derivative in Chebyshev coefficient space (exact for polynomials).
  !   u = sum a_k T_k  ->  u' = sum b_k T_k
  ! Standard recurrence; b_0 is halved at the end.
  !---------------------------------------------------------------------------
  subroutine cheb_deriv(n, a, b)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: a(0:n)
    real(dp),    intent(out) :: b(0:n)
    integer(ip) :: k
    real(dp)    :: bk2
    b = 0.0_dp
    do k = n-1, 0, -1
       bk2 = 0.0_dp
       if (k+2 <= n) bk2 = b(k+2)
       b(k) = bk2 + 2.0_dp * real(k+1,dp) * a(k+1)
    end do
    b(0) = 0.5_dp * b(0)
  end subroutine cheb_deriv

  !---------------------------------------------------------------------------
  ! Second derivative = derivative applied twice.
  !---------------------------------------------------------------------------
  subroutine cheb_d2(n, a, c)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: a(0:n)
    real(dp),    intent(out) :: c(0:n)
    real(dp) :: b(0:n)
    call cheb_deriv(n, a, b)
    call cheb_deriv(n, b, c)
  end subroutine cheb_d2

  !---------------------------------------------------------------------------
  ! Forward Chebyshev transform on the GLC grid: physical values -> coeffs.
  !   a_k = (2/(n cbar_k)) sum_j (1/cbar_j) f_j cos(pi j k / n)
  ! with cbar_0 = cbar_n = 2, else 1. Naive O(n^2) (fine: ny is small).
  !---------------------------------------------------------------------------
  subroutine cheb_forward(n, f, a)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: f(0:n)
    real(dp),    intent(out) :: a(0:n)
    integer(ip) :: k, j
    real(dp)    :: s, cbk, cbj
    do k = 0, n
       cbk = merge(2.0_dp, 1.0_dp, k == 0 .or. k == n)
       s = 0.0_dp
       do j = 0, n
          cbj = merge(2.0_dp, 1.0_dp, j == 0 .or. j == n)
          s = s + (f(j)/cbj) * cos(PI*real(j,dp)*real(k,dp)/real(n,dp))
       end do
       a(k) = 2.0_dp * s / (real(n,dp) * cbk)
    end do
  end subroutine cheb_forward

  !---------------------------------------------------------------------------
  ! Evaluate a Chebyshev series at a point x via the T_k recurrence.
  !---------------------------------------------------------------------------
  pure function cheb_eval(n, a, x) result(val)
    integer(ip), intent(in) :: n
    real(dp),    intent(in) :: a(0:n)
    real(dp),    intent(in) :: x
    real(dp) :: val, t0, t1, t2
    integer(ip) :: k
    val = a(0)
    if (n >= 1) then
       t0 = 1.0_dp; t1 = x
       val = val + a(1)*t1
       do k = 2, n
          t2 = 2.0_dp*x*t1 - t0
          val = val + a(k)*t2
          t0 = t1; t1 = t2
       end do
    end if
  end function cheb_eval

  !---------------------------------------------------------------------------
  ! Convert Shen-Dirichlet coefficients (0..n-2) to Chebyshev coefficients
  ! (0..n).  phi_k = T_k - T_{k+2}.
  !---------------------------------------------------------------------------
  subroutine shen_to_cheb(n, uhat, a)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: uhat(0:n-2)
    real(dp),    intent(out) :: a(0:n)
    integer(ip) :: k
    a = 0.0_dp
    do k = 0, n-2
       a(k)   = a(k)   + uhat(k)
       a(k+2) = a(k+2) - uhat(k)
    end do
  end subroutine shen_to_cheb

  !---------------------------------------------------------------------------
  ! Chebyshev-weighted inner product of two coefficient vectors:
  !   (p,q)_w = sum_i p_i q_i (pi/2) c_i,   c_0 = 2, else 1.
  !---------------------------------------------------------------------------
  pure function wdot(n, p, q) result(s)
    integer(ip), intent(in) :: n
    real(dp),    intent(in) :: p(0:n), q(0:n)
    real(dp) :: s, ci
    integer(ip) :: i
    s = 0.0_dp
    do i = 0, n
       ci = merge(2.0_dp, 1.0_dp, i == 0)
       s = s + p(i)*q(i)*0.5_dp*PI*ci
    end do
  end function wdot

  !---------------------------------------------------------------------------
  ! Assemble the dense Helmholtz matrix  M = S + mu B,  size (n-1)x(n-1).
  !   B_kj = (phi_j, phi_k)_w
  !   S_kj = (-phi_j'', phi_k)_w
  ! Built from the derivative recurrence + weighted inner products only.
  !---------------------------------------------------------------------------
  subroutine assemble_helmholtz(n, mu, M)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: mu
    real(dp),    intent(out) :: M(0:n-2,0:n-2)
    real(dp) :: Phi(0:n,0:n-2), D2Phi(0:n,0:n-2)
    real(dp) :: tmp(0:n)
    integer(ip) :: k, j
    ! basis functions and their second derivatives, in Chebyshev coeffs
    do k = 0, n-2
       tmp = 0.0_dp
       tmp(k)   =  1.0_dp
       tmp(k+2) = -1.0_dp
       Phi(:,k) = tmp
       call cheb_d2(n, tmp, D2Phi(:,k))
    end do
    do j = 0, n-2
       do k = 0, n-2
          M(k,j) = -wdot(n, D2Phi(:,j), Phi(:,k)) + mu * wdot(n, Phi(:,j), Phi(:,k))
       end do
    end do
  end subroutine assemble_helmholtz

  !---------------------------------------------------------------------------
  ! Project a Chebyshev-coeff RHS f onto the Shen test functions:
  !   ftilde_k = (f, phi_k)_w
  !---------------------------------------------------------------------------
  subroutine helmholtz_rhs(n, fcheb, ftilde)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: fcheb(0:n)
    real(dp),    intent(out) :: ftilde(0:n-2)
    real(dp) :: phik(0:n)
    integer(ip) :: k
    do k = 0, n-2
       phik = 0.0_dp
       phik(k)   =  1.0_dp
       phik(k+2) = -1.0_dp
       ftilde(k) = wdot(n, fcheb, phik)
    end do
  end subroutine helmholtz_rhs

  !---------------------------------------------------------------------------
  ! Full convenience solve: given mu and a Chebyshev-coeff RHS, return the
  ! Shen-Dirichlet coefficients uhat of the solution of -u'' + mu u = f.
  !---------------------------------------------------------------------------
  subroutine helmholtz_solve(n, mu, fcheb, uhat)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: mu
    real(dp),    intent(in)  :: fcheb(0:n)
    real(dp),    intent(out) :: uhat(0:n-2)
    real(dp)    :: M(0:n-2,0:n-2), rhs(0:n-2)
    integer(ip) :: piv(0:n-2), m_
    m_ = n-1
    call assemble_helmholtz(n, mu, M)
    call helmholtz_rhs(n, fcheb, rhs)
    call lu_factor(m_, M, piv)
    call lu_solve(m_, M, piv, rhs)
    uhat = rhs
  end subroutine helmholtz_solve

  !---------------------------------------------------------------------------
  ! Dense LU with partial pivoting (in place). A is (m x m), 0-based.
  ! Placeholder for the eventual banded / batched GPU solver.
  !---------------------------------------------------------------------------
  subroutine lu_factor(m, A, piv)
    integer(ip), intent(in)    :: m
    real(dp),    intent(inout) :: A(0:m-1,0:m-1)
    integer(ip), intent(out)   :: piv(0:m-1)
    integer(ip) :: i, k, p
    real(dp)    :: amax, factor, trow(0:m-1)
    do k = 0, m-1
       piv(k) = k; amax = abs(A(k,k)); p = k
       do i = k+1, m-1
          if (abs(A(i,k)) > amax) then
             amax = abs(A(i,k)); p = i
          end if
       end do
       piv(k) = p
       if (p /= k) then
          trow = A(k,:); A(k,:) = A(p,:); A(p,:) = trow
       end if
       do i = k+1, m-1
          factor = A(i,k)/A(k,k)
          A(i,k) = factor
          A(i,k+1:m-1) = A(i,k+1:m-1) - factor*A(k,k+1:m-1)
       end do
    end do
  end subroutine lu_factor

  subroutine lu_solve(m, A, piv, b)
    integer(ip), intent(in)    :: m
    real(dp),    intent(in)    :: A(0:m-1,0:m-1)
    integer(ip), intent(in)    :: piv(0:m-1)
    real(dp),    intent(inout) :: b(0:m-1)
    integer(ip) :: i, k
    real(dp)    :: t
    do k = 0, m-1                 ! apply row swaps
       if (piv(k) /= k) then
          t = b(k); b(k) = b(piv(k)); b(piv(k)) = t
       end if
    end do
    do i = 1, m-1                 ! forward (unit lower)
       b(i) = b(i) - dot_product(A(i,0:i-1), b(0:i-1))
    end do
    do i = m-1, 0, -1             ! back (upper)
       b(i) = (b(i) - dot_product(A(i,i+1:m-1), b(i+1:m-1))) / A(i,i)
    end do
  end subroutine lu_solve

end module cheb_shen
