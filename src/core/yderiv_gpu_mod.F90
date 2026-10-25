!===============================================================================
! yderiv_gpu_mod.F90 -- the DEVICE D_y DISPATCH SEAM for the
! transform-tier derivative consumers (velocity_mod / forcing_mod / stats_mod):
! the last numerical primitives that still called the Chebyshev recurrence
! directly. y_deriv_tp / y_deriv_c keep the EXACT cheb_deriv_c_tp / cheb_deriv_c
! call as their default branch (bit-identity: same routine, same arguments --
! gated to the BIT in test_yderiv), and add an FD branch = batched banded D1
! matvec on point values when the backend is flipped by yderiv_set_fd (called
! from the FD init path only; a Chebyshev run never touches the flag).
!
! FD branch contract:
!   * a holds POINT VALUES at the ygrid-file nodes (descending, y(1) = +1); the
!     band is fdops fd_build_bands' D1 (GB convention ab(kl+1+i-j, j), interior
!     centered 2*kl+1 stencil, band-limited order-tapered wall rows) -- the
!     SAME operator the host fd_apply_band applies, with the SAME fixed
!     j-ascending row accumulation (the cross-check test_yderiv gates).
!   * dp ACCUMULATION (house policy): the band is dp; each field value is
!     promoted to dp, the row sum accumulates in dp, and ONE final store rounds
!     to the field kind. At TP=sp this bounds the error at the store rounding
!     (~eps_tp relative), NOT the stencil-conditioning amplification that a
!     tp-native accumulation would suffer near the walls (D1 rows ~ 1/h_wall).
!   * a and b MUST NOT alias in FD mode (the matvec reads neighbours of rows
!     already written) -- same contract as fd_apply_L_gpu. Every current call
!     site passes distinct arrays; the Cheb branch is alias-safe regardless
!     (it round-trips through cheb_ops' column-leading scratch).
!   * one (column, row) pair per thread, y-leading layout (UNCOALESCED, like
!     the rest of the FD device tier).
!
! Module-save state, set ONCE at init (before any y_deriv_* call, so the
! dealloc in a repeated set needs no stream drain -- nothing in flight can
! reference yd_d1b yet). yderiv_set_cheb exists for TEST state hygiene only.
! Build: nvfortran -cuda -c yderiv_gpu_mod.F90   (after cheb_ops_mod)
!===============================================================================
module yderiv_gpu_mod
  use kinds,        only: tp, wp, dp, ip
  use cudafor
  use cheb_ops_mod, only: cheb_deriv_c, cheb_deriv_c_tp
  implicit none
  private
  public :: yderiv_set_fd, yderiv_set_cheb, y_deriv_tp, y_deriv_c, gal_shift_mean

  logical,     save :: yd_fd = .false.   ! backend switch (default: Chebyshev)
  integer(ip), save :: yd_kl = 0
  real(dp), allocatable, device, save :: yd_d1b(:,:)  ! (2*kl+1, ny) D1 band (own copy)

contains

  !-- flip the backend to FD: upload this module's own copy of the D1 band ----
  subroutine yderiv_set_fd(ny, kl, d1bh)
    integer(ip), intent(in) :: ny, kl
    real(dp),    intent(in) :: d1bh(2*kl+1, ny)
    if (allocated(yd_d1b)) deallocate(yd_d1b)
    allocate(yd_d1b(2*kl+1, ny))
    yd_d1b = d1bh
    yd_kl  = kl
    yd_fd  = .true.
  end subroutine yderiv_set_fd

  !-- reset to the Chebyshev default (test hygiene; production never calls) ---
  subroutine yderiv_set_cheb()
    yd_fd = .false.
    yd_kl = 0
    if (allocated(yd_d1b)) deallocate(yd_d1b)
  end subroutine yderiv_set_cheb

  !-- [Galilean] subtract the constant frame speed U_c from the (0,0)
  !   streamwise column, IN THE ACTIVE y-BASIS. Chebyshev: the column holds
  !   T_k coefficients -- the constant lives in T_0 alone, so exactly row 1
  !   moves. FD: the column holds POINT VALUES at the ygrid nodes -- the
  !   constant lives at EVERY node. (Shifting row 1 in both bases would be
  !   wall-node-only in FD, leaving the advecting frame wrong at every
  !   interior node.) Lives on the dispatch seam because only
  !   this module knows the basis; both nonlinear twins call it, on the rank
  !   that owns the (0,0) mode only. Gated by test_yderiv (10).
  subroutine gal_shift_mean(ny, ugal, ucol)
    integer(ip), intent(in) :: ny
    real(wp),    intent(in) :: ugal
    complex(tp), device     :: ucol(ny)
    complex(tp) :: cs
    integer(ip) :: i, iend
    cs   = cmplx(ugal, 0.0_wp, kind=tp)
    iend = merge(ny, 1_ip, yd_fd)
    !$cuf kernel do(1) <<<*,*>>>
    do i = 1, iend
       ucol(i) = ucol(i) - cs
    end do
  end subroutine gal_shift_mean

  !-- b = D_y a on the transform tier (complex(tp)), batched over columns -----
  subroutine y_deriv_tp(ny, batch, a, b)
    integer(ip), intent(in) :: ny, batch
    complex(tp), device     :: a(ny, batch)
    complex(tp), device     :: b(ny, batch)
    integer(ip) :: klv, i, j, col
    real(dp)    :: sr, si, w
    if (.not. yd_fd) then
       call cheb_deriv_c_tp(ny, batch, a, b)     ! verbatim Cheb branch
       return
    end if
    klv = yd_kl
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do i = 1, ny
          sr = 0.0_dp;  si = 0.0_dp
          do j = max(1_ip, i - klv), min(ny, i + klv)
             w  = yd_d1b(klv + 1 + i - j, j)
             sr = sr + w*real(a(j, col), dp)
             si = si + w*real(aimag(a(j, col)), dp)
          end do
          b(i, col) = cmplx(sr, si, tp)
       end do
    end do
  end subroutine y_deriv_tp

  !-- b = D_y a on the wp tier (stats' D_y v), batched over columns -----------
  subroutine y_deriv_c(ny, batch, a, b)
    integer(ip), intent(in) :: ny, batch
    complex(wp), device     :: a(ny, batch)
    complex(wp), device     :: b(ny, batch)
    integer(ip) :: klv, i, j, col
    real(dp)    :: sr, si, w
    if (.not. yd_fd) then
       call cheb_deriv_c(ny, batch, a, b)        ! verbatim Cheb branch
       return
    end if
    klv = yd_kl
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do i = 1, ny
          sr = 0.0_dp;  si = 0.0_dp
          do j = max(1_ip, i - klv), min(ny, i + klv)
             w  = yd_d1b(klv + 1 + i - j, j)
             sr = sr + w*real(a(j, col), dp)
             si = si + w*real(aimag(a(j, col)), dp)
          end do
          b(i, col) = cmplx(sr, si, wp)
       end do
    end do
  end subroutine y_deriv_c

end module yderiv_gpu_mod
