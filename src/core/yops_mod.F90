!===============================================================================
! yops_mod.F90 -- the WALL-NORMAL OPERATOR FACADE: the single home of
! the small y-direction "boundary + integral" formulas that were duplicated
! across the solver, and THE DECIDED SEAM for the future FD-y backend.
!
! What lives here (Chebyshev-tau / GLC backend, the only one today):
!   yop_wall_deriv  : Du(+-1) from Cheb coeffs  (was triplicated: vsolve_if_mod
!                     wall_derivs / par_pressure_mod wall_derivs_h / inlined in
!                     stats_mod wall_shear -- all now route here)
!   yop_mean_flux   : int_{-1}^{1} q dy from Cheb coeffs (CFR flux; was mean_mod)
!   yop_int_tj      : int_{-1}^{1} T_j dy (was init_mod wint)
!   yop_l2int       : int |f|^2 dy from Cheb coeffs (was init_mod cheb_l2int)
!   yop_quad_w      : GRID-space quadrature weights w_j on the GLC nodes
!                     (Clenshaw-Curtis) -- sum_j w_j f(y_j) == the exact
!                     coefficient-space integral for any degree-<=n polynomial.
!                     No solver path consumes these YET (Cheb integrates in
!                     coefficient space); they exist so the FD backend supplies
!                     NUMBERS to the same call sites, not new code paths, and
!                     they are validated against yop_int_tj in test_yops.
!   YOPS_BACKEND / YOPS_YGRID : identity tags ('cheb-T' / 'glc') -- the single
!                     source of truth for the gate-tolerance switch and for the
!                     io headers' ybasis/ygrid keys.
!
! THE FD-y SWAP CONTRACT: an FD backend replaces the FORMULAS here (wall
! derivative -> one-sided stencil weights on grid values; integrals -> yop_quad_w
! dot products; tags -> 'fd'/'stretched'), while every consumer keeps calling the
! same names. The batched DEVICE twins (spec_helpers_mod wall_derivs_c /
! wall_values_c and the cheb_* kernels) are the device tier of this same seam --
! they swap kernel bodies with the backend, coordinated with this module.
!
! BIT-IDENTITY RULE (load-bearing): each routine preserves its consumer's
! HISTORICAL expression and accumulation order verbatim -- e.g. yop_mean_flux
! keeps  q(k)*2/(1-k^2)  (multiply-then-divide) rather than dotting a
! precomputed-weight array, because  (q*2)/d  and  q*(2/d)  round differently
! and the whole suite (and --validate) gates bit-exact trajectories. The one
! reshaped body (stats_mod wall_shear -> yop_wall_deriv) is bit-identical
! WITHIN ONE BUILD'S CODEGEN: every bit-exact gate compares two paths built by
! the same Makefile with the same flags, so FMA contraction (which could fuse
! (+-a)*j^2 into the accumulate) fuses IDENTICALLY on both sides. Cross-BUILD
! bit comparison still requires matching flags. Unifying the arithmetic itself
! would break bit-identity and is deliberately not done here.
!
! Host-only, depends on kinds alone; compiled early in every harness (before
! its first consumer: mean_mod / vsolve_if_mod / init_mod).
! Build:  nvfortran -c yops_mod.F90
!===============================================================================
module yops_mod
  use kinds, only: dp, ip
  implicit none
  private
  public :: yop_wall_deriv, yop_mean_flux, yop_int_tj, yop_l2int, yop_quad_w
  public :: YOPS_BACKEND, YOPS_YGRID
  public :: yops_set_fd, yops_set_cheb, yop_backend_tag, yop_ygrid_tag

  character(*), parameter :: YOPS_BACKEND = 'cheb-T'   ! y-basis identity tag
  character(*), parameter :: YOPS_YGRID   = 'glc'      ! y-grid identity tag

  ! THE FD-y SWAP: module-save
  ! weight vectors on the ygrid-file nodes (descending, index 0 <-> y = +1),
  ! set once at FD init by yops_set_fd. With the flag down (every Chebyshev
  ! run) each routine takes its Chebyshev branch untouched -- the bit-identity
  ! rule holds by construction.
  ! FD-SAFETY MAP (the FD IC does not route through the Cheb-only entries; it
  ! uses fd_apply_band + w-dots directly in init_mod):
  !   FD-safe here:  yop_wall_deriv, yop_mean_flux (both branch on yops_fd).
  !   CHEB-ONLY:     yop_l2int, yop_int_tj, yop_quad_w -- Chebyshev bodies with
  !                  NO fd branch; a new FD-mode caller must first give them FD
  !                  bodies (or route around them as the IC does).
  logical, save :: yops_fd = .false.
  real(dp), allocatable, save :: yops_wlo(:)   ! wall D1 row: dot -> Du(-1)
  real(dp), allocatable, save :: yops_whi(:)   ! wall D1 row: dot -> Du(+1)
  real(dp), allocatable, save :: yops_wq(:)    ! quadrature weights (file's w)

contains

  !-- flip the facade to FD: wall-derivative rows + quadrature weights --------
  ! (fd_wall_weights' full-order one-sided rows and the grid file's w column,
  ! passed 1-based (ny) by the caller -- sequence association onto 0:n.)
  subroutine yops_set_fd(n, wlo, whi, wq)
    integer(ip), intent(in) :: n
    real(dp),    intent(in) :: wlo(0:n), whi(0:n), wq(0:n)
    yops_wlo = wlo;  yops_whi = whi;  yops_wq = wq
    yops_fd  = .true.
  end subroutine yops_set_fd

  !-- reset to the Chebyshev default (test hygiene; production never calls) ---
  subroutine yops_set_cheb()
    yops_fd = .false.
    if (allocated(yops_wlo)) deallocate(yops_wlo, yops_whi, yops_wq)
  end subroutine yops_set_cheb

  !-- runtime identity tags: the compile-time parameters above are the Cheb
  !-- default; these are what the io headers/checkpoint stamps must consume ---
  pure function yop_backend_tag() result(tag)
    character(:), allocatable :: tag
    if (yops_fd) then; tag = 'fd-point'
    else;              tag = YOPS_BACKEND; end if
  end function yop_backend_tag

  pure function yop_ygrid_tag() result(tag)
    character(:), allocatable :: tag
    if (yops_fd) then; tag = 'file'
    else;              tag = YOPS_YGRID; end if
  end function yop_ygrid_tag

  !---------------------------------------------------------------------------
  ! First derivative at the walls from Chebyshev coefficients:
  !   Du(+1) = sum_j j^2 a_j ,   Du(-1) = sum_j (-1)^(j-1) j^2 a_j .
  ! dlo = Du(-1), dhi = Du(+1). (Verbatim body of the retired vsolve_if_mod
  ! wall_derivs == par_pressure_mod wall_derivs_h; the j=0 term is exactly 0.)
  !---------------------------------------------------------------------------
  pure subroutine yop_wall_deriv(n, a, dlo, dhi)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: a(0:n)
    real(dp),    intent(out) :: dlo, dhi
    integer(ip) :: j, s
    if (yops_fd) then            ! a = POINT VALUES at the file nodes
       dlo = dot_product(yops_wlo, a)
       dhi = dot_product(yops_whi, a)
       return
    end if
    dlo = 0.0_dp; dhi = 0.0_dp
    do j = 1, n
       dhi = dhi + a(j)*real(j*j, dp)
       if (mod(j,2) == 1) then; s = 1; else; s = -1; end if   ! (-1)^(j-1)
       dlo = dlo + real(s, dp)*a(j)*real(j*j, dp)
    end do
  end subroutine yop_wall_deriv

  !---------------------------------------------------------------------------
  ! int_{-1}^{1} q dy from Chebyshev coefficients (odd-T_k integrals vanish).
  ! (Verbatim body of the retired mean_mod mean_flux -- see the bit-identity
  ! rule in the header for why this does NOT call yop_int_tj.)
  !---------------------------------------------------------------------------
  pure function yop_mean_flux(n, q) result(qint)
    integer(ip), intent(in) :: n
    real(dp),    intent(in) :: q(0:n)
    real(dp) :: qint
    integer(ip) :: k
    if (yops_fd) then            ! q = POINT VALUES; file's quadrature
       qint = dot_product(yops_wq, q)
       return
    end if
    qint = 0.0_dp
    do k = 0, n, 2                       ! even k only (odd-k weights vanish)
       qint = qint + q(k)*2.0_dp/(1.0_dp - real(k,dp)**2)
    end do
  end function yop_mean_flux

  !-- int_{-1}^{1} T_j dy  (verbatim body of the retired init_mod wint) --------
  pure function yop_int_tj(j) result(w)
    integer(ip), intent(in) :: j
    real(dp) :: w
    if (mod(j,2) == 0) then
       w = 2.0_dp/(1.0_dp - real(j,dp)**2)
       if (j == 0) w = 2.0_dp
    else
       w = 0.0_dp
    end if
  end function yop_int_tj

  !---------------------------------------------------------------------------
  ! EXACT integral of |f(y)|^2 over y in [-1,1] from Chebyshev coefficients
  ! (UNWEIGHTED). (Verbatim body of the retired init_mod cheb_l2int.)
  !---------------------------------------------------------------------------
  pure function yop_l2int(n, a) result(s)
    integer(ip), intent(in) :: n
    complex(dp), intent(in) :: a(0:n)
    real(dp) :: s, w
    integer(ip) :: m, l
    s = 0.0_dp
    do m = 0, n
       do l = m, n
          if (mod(m+l, 2) /= 0) cycle
          w = 0.5_dp*(yop_int_tj(m+l) + yop_int_tj(l-m))
          if (l == m) then
             s = s + (real(a(m))**2 + aimag(a(m))**2)*w
          else
             s = s + 2.0_dp*(real(a(m))*real(a(l)) + aimag(a(m))*aimag(a(l)))*w
          end if
       end do
    end do
  end function yop_l2int

  !---------------------------------------------------------------------------
  ! Grid-space quadrature weights on the GLC nodes y_j = cos(pi j / n),
  ! j = 0..n: the Clenshaw-Curtis rule (Trefethen, Spectral Methods in MATLAB,
  ! clencurt). sum_j w(j) f(y_j) integrates any polynomial of degree <= n
  ! EXACTLY (matches the coefficient-space integral to round-off -- gated in
  ! test_yops for both parities of n). The FD backend replaces this fill with
  ! the weights of ITS grid; every grid-space integral call site is unchanged.
  !---------------------------------------------------------------------------
  pure subroutine yop_quad_w(n, w)
    integer(ip), intent(in)  :: n
    real(dp),    intent(out) :: w(0:n)
    real(dp), parameter :: PI = 3.141592653589793238462643_dp
    real(dp)    :: theta, v
    integer(ip) :: j, k, khalf
    if (n == 0) then
       w(0) = 2.0_dp                                   ! single node, int 1 dy = 2
       return
    end if
    if (mod(n,2) == 0) then
       w(0) = 1.0_dp/real(n*n - 1, dp);  khalf = n/2 - 1
    else
       w(0) = 1.0_dp/real(n*n, dp);      khalf = (n-1)/2
    end if
    w(n) = w(0)
    do j = 1, n-1
       theta = PI*real(j,dp)/real(n,dp)
       v = 1.0_dp
       do k = 1, khalf
          v = v - 2.0_dp*cos(2.0_dp*real(k,dp)*theta)/real(4*k*k - 1, dp)
       end do
       if (mod(n,2) == 0) v = v - cos(real(n,dp)*theta)/real(n*n - 1, dp)
       w(j) = 2.0_dp*v/real(n,dp)
    end do
  end subroutine yop_quad_w

end module yops_mod
