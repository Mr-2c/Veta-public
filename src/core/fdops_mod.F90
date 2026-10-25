!===============================================================================
! fdops_mod.F90 -- the FD-y OPERATOR KIT: finite-difference
! operators on an ARBITRARY node set (the ygrid file's y), banded storage,
! and a no-pivot banded LU. Host-only, dp; the GPU batched twins (one chain
! per (kx,kz) mode, chain-split anatomy) live in the fd_*_gpu modules --
! THIS layer is their oracle.
!
! Pieces:
!   fd_weights       Fornberg (1988) weights: derivatives 0..m at a point from
!                    arbitrary nodes. THE primitive everything else builds on;
!                    gated by polynomial EXACTNESS in test_fdops (weights for
!                    the d-th derivative must reproduce d/dy of y^p exactly for
!                    p <= stencil_size-1 at EVERY row -- a complete algebraic
!                    characterization, so a transcription error cannot hide).
!   fd_build_bands   D1, D2 as (2*kl+1, ny) bands, LAPACK GB convention
!                    ab(kl+1+i-j, j) = A(i,j). Row i uses nodes
!                    max(1,i-kl)..min(ny,i+kl): interior = centered 2*kl+1
!                    stencil (degree-2kl exact), wall rows = offset stencils
!                    that STAY INSIDE THE BAND (degree kl.. exact -- the
!                    standard order taper that keeps implicit solves banded;
!                    same choice as the pipe's r-operators, LDAB = 2*KL+1).
!   fd_apply_band    g = A f (banded matvec, fixed accumulation order).
!   fd_helm_dirichlet  ab = -D2 + mu*I with identity wall rows (caller puts
!                    the boundary VALUES in rhs(1)/rhs(ny)).
!   fd_band_factor / fd_band_solve   no-pivot in-place banded LU + solve.
!                    NO PIVOTING is the pipe-validated choice (their oracle
!                    sweep: no-pivot vs dense-pivoted LU agreed to 2.9e-11
!                    across the full operator set incl. a near-singular
!                    Neumann operator at kappa ~ 3e11) -- it keeps the solve a
!                    pure recurrence (chain-splittable on the GPU). A runtime
!                    ZERO-PIVOT GUARD stays in anyway (ok=.false., never NaNs).
!
! Build: nvfortran -c fdops_mod.F90   (after kinds; no other deps)
!===============================================================================
module fdops_mod
  use kinds, only: dp, ip
  implicit none
  private
  public :: fd_weights, fd_build_bands, fd_apply_band
  public :: fd_helm_dirichlet, fd_band_factor, fd_band_solve
  public :: fd_wall_weights

contains

  !---------------------------------------------------------------------------
  ! fd_weights -- Fornberg's algorithm (Mathematics of Computation 51, 1988;
  ! the canonical 'weights' routine): c(j,k) = weight of node x(j) in the
  ! k-th-derivative approximation at z, for k = 0..m, from nodes x(0:nd).
  ! Exact for polynomials of degree <= nd. Nodes need not be ordered; z need
  ! not be a node.
  !---------------------------------------------------------------------------
  pure subroutine fd_weights(z, x, nd, m, c)
    real(dp),    intent(in)  :: z
    integer(ip), intent(in)  :: nd, m
    real(dp),    intent(in)  :: x(0:nd)
    real(dp),    intent(out) :: c(0:nd, 0:m)
    real(dp)    :: c1, c2, c3, c4, c5
    integer(ip) :: i, j, k, mn
    c1 = 1.0_dp
    c4 = x(0) - z
    c  = 0.0_dp
    c(0,0) = 1.0_dp
    do i = 1, nd
       mn = min(i, m)
       c2 = 1.0_dp
       c5 = c4
       c4 = x(i) - z
       do j = 0, i-1
          c3 = x(i) - x(j)
          c2 = c2*c3
          if (j == i-1) then
             do k = mn, 1, -1
                c(i,k) = c1*(real(k,dp)*c(i-1,k-1) - c5*c(i-1,k))/c2
             end do
             c(i,0) = -c1*c5*c(i-1,0)/c2
          end if
          do k = mn, 1, -1
             c(j,k) = (c4*c(j,k) - real(k,dp)*c(j,k-1))/c3
          end do
          c(j,0) = c4*c(j,0)/c3
       end do
       c1 = c2
    end do
  end subroutine fd_weights

  !---------------------------------------------------------------------------
  ! fd_build_bands -- first/second-derivative operators for nodes y(1:ny)
  ! (any ordering the caller uses consistently; the solver passes the grid
  ! file's DESCENDING +1..-1). d1b/d2b are (2*kl+1, ny), GB convention.
  !---------------------------------------------------------------------------
  subroutine fd_build_bands(ny, kl, y, d1b, d2b)
    integer(ip), intent(in)  :: ny, kl
    real(dp),    intent(in)  :: y(ny)
    real(dp),    intent(out) :: d1b(2*kl+1, ny), d2b(2*kl+1, ny)
    real(dp)    :: c(0:2*kl, 0:2)
    integer(ip) :: i, j, j0, j1, nd
    d1b = 0.0_dp
    d2b = 0.0_dp
    do i = 1, ny
       j0 = max(1_ip, i - kl)
       j1 = min(ny, i + kl)
       nd = j1 - j0
       call fd_weights(y(i), y(j0:j1), nd, 2_ip, c(0:nd, 0:2))
       do j = j0, j1
          d1b(kl + 1 + i - j, j) = c(j - j0, 1)
          d2b(kl + 1 + i - j, j) = c(j - j0, 2)
       end do
    end do
  end subroutine fd_build_bands

  !---------------------------------------------------------------------------
  ! fd_wall_weights -- FULL-ORDER one-sided first-derivative rows at the two
  ! walls, as FULL-LENGTH weight vectors (zeros beyond the stencil): the FD
  ! replacement for the Chebyshev wall-derivative sums (yops yop_wall_deriv /
  ! spec_helpers wall_derivs_c). Wall derivative EVALUATION is not part of a
  ! banded solve, so unlike the D1/D2 band rows it is NOT band-limited: the
  ! stencil is the 2*kl+1 nodes nearest each wall (degree-2kl exact -- gated in
  ! test_fd_prims_gpu). Convention (descending grid, y(1) = +1, y(ny) = -1):
  !   whi(:) dotted with point values = Du(+1);  wlo(:) = Du(-1).
  !---------------------------------------------------------------------------
  subroutine fd_wall_weights(ny, kl, y, wlo, whi)
    integer(ip), intent(in)  :: ny, kl
    real(dp),    intent(in)  :: y(ny)
    real(dp),    intent(out) :: wlo(ny), whi(ny)
    real(dp)    :: c(0:2*kl, 0:1)
    integer(ip) :: nd, j
    nd  = min(2*kl, ny - 1)
    wlo = 0.0_dp
    whi = 0.0_dp
    call fd_weights(y(1), y(1:1+nd), nd, 1_ip, c(0:nd, 0:1))
    do j = 0, nd
       whi(1 + j) = c(j, 1)
    end do
    call fd_weights(y(ny), y(ny-nd:ny), nd, 1_ip, c(0:nd, 0:1))
    do j = 0, nd
       wlo(ny - nd + j) = c(j, 1)
    end do
  end subroutine fd_wall_weights

  !-- g = A f for a GB-stored band (fixed j-ascending accumulation) ------------
  subroutine fd_apply_band(ny, kl, ab, f, g)
    integer(ip), intent(in)  :: ny, kl
    real(dp),    intent(in)  :: ab(2*kl+1, ny), f(ny)
    real(dp),    intent(out) :: g(ny)
    integer(ip) :: i, j
    do i = 1, ny
       g(i) = 0.0_dp
       do j = max(1_ip, i - kl), min(ny, i + kl)
          g(i) = g(i) + ab(kl + 1 + i - j, j)*f(j)
       end do
    end do
  end subroutine fd_apply_band

  !---------------------------------------------------------------------------
  ! ab = -D2 + mu*I with Dirichlet wall rows (rows 1 and ny become identity;
  ! put the boundary VALUES in rhs(1)/rhs(ny) before fd_band_solve).
  !---------------------------------------------------------------------------
  subroutine fd_helm_dirichlet(ny, kl, d2b, mu, ab)
    integer(ip), intent(in)  :: ny, kl
    real(dp),    intent(in)  :: d2b(2*kl+1, ny), mu
    real(dp),    intent(out) :: ab(2*kl+1, ny)
    integer(ip) :: j
    ab = -d2b
    do j = 1, ny
       ab(kl + 1, j) = ab(kl + 1, j) + mu
    end do
    do j = 1, min(1_ip + kl, ny)                 ! row 1 -> identity
       ab(kl + 1 + 1 - j, j) = 0.0_dp
    end do
    ab(kl + 1, 1) = 1.0_dp
    do j = max(1_ip, ny - kl), ny                ! row ny -> identity
       ab(kl + 1 + ny - j, j) = 0.0_dp
    end do
    ab(kl + 1, ny) = 1.0_dp
  end subroutine fd_helm_dirichlet

  !---------------------------------------------------------------------------
  ! No-pivot in-place banded LU (ku = kl; no fill outside the band without
  ! pivoting). ok=.false. on a zero/underflow pivot -- refuse, never NaN.
  !---------------------------------------------------------------------------
  subroutine fd_band_factor(ny, kl, ab, ok)
    integer(ip), intent(in)    :: ny, kl
    real(dp),    intent(inout) :: ab(2*kl+1, ny)
    logical,     intent(out)   :: ok
    real(dp)    :: piv, l
    integer(ip) :: i, j, k
    ok = .false.
    do k = 1, ny
       piv = ab(kl + 1, k)
       if (abs(piv) < 1.0e3_dp*tiny(1.0_dp)) return       ! zero-pivot guard
       do i = k + 1, min(k + kl, ny)
          l = ab(kl + 1 + i - k, k)/piv
          ab(kl + 1 + i - k, k) = l
          do j = k + 1, min(k + kl, ny)
             ab(kl + 1 + i - j, j) = ab(kl + 1 + i - j, j) - l*ab(kl + 1 + k - j, j)
          end do
       end do
    end do
    ok = .true.
  end subroutine fd_band_factor

  !-- solve LU x = b in place (b -> x), factors from fd_band_factor ------------
  subroutine fd_band_solve(ny, kl, ab, b)
    integer(ip), intent(in)    :: ny, kl
    real(dp),    intent(in)    :: ab(2*kl+1, ny)
    real(dp),    intent(inout) :: b(ny)
    integer(ip) :: i, j, k
    do k = 1, ny                                  ! forward: L y = b
       do i = k + 1, min(k + kl, ny)
          b(i) = b(i) - ab(kl + 1 + i - k, k)*b(k)
       end do
    end do
    do k = ny, 1, -1                              ! backward: U x = y
       do j = k + 1, min(k + kl, ny)
          b(k) = b(k) - ab(kl + 1 + k - j, j)*b(j)
       end do
       b(k) = b(k)/ab(kl + 1, k)
    end do
  end subroutine fd_band_solve

end module fdops_mod
