!===============================================================================
! cheb_ops_mod.F90  --  GPU Chebyshev coefficient-space operators.
!
! cheb_deriv_c: first y-derivative of a complex field given in Chebyshev
!   coefficients, batched over (kx,kz) columns. This is the GPU counterpart of
!   the validated CPU cheb_deriv (cheb_shen) and uses the identical recurrence:
!
!     b = 0
!     do k = N-1, 0, -1:  b_k = [b_{k+2} if k+2<=N else 0] + 2(k+1) a_{k+1}
!     b_0 *= 1/2
!
! The recurrence is sequential in the Chebyshev index k (each b_k depends on
! b_{k+2}), but fully independent across columns, so we parallelise over the
! batch and run the k-recurrence sequentially within each thread.
!
! COALESCED COLUMN-LEADING LAYOUT (same treatment as helm_band).
! The caller's arrays are (ny, batch) with the Chebyshev index LEADING, so a warp
! of 32 consecutive columns would stride memory by ny (32 transactions/access),
! which put the two cheb_deriv kernels at ~20% of the step combined. We transpose
! to column-leading (batch, ny) on the way in, run the recurrence with the column
! index contiguous (warp = 32 consecutive columns = coalesced), and transpose back
! out. Persistent grow/shape-tracked module scratch holds the transposed buffers.
! The recurrence arithmetic is unchanged -> bit-exact (the apply_L / D^2 cross-check
! against the CPU cheb_d2 is the gate). The two transposes are bandwidth-bound,
! full-occupancy copies; net ~2x (the transpose eats ~1/3 of the coalescing win).
! FOLLOW-ON: apply_L calls cheb_deriv_c twice in series, so it round-trips the
! intermediate through y-leading scr (two redundant transposes); a column-leading
! D^2 path inside apply_L would remove them (deferred -- keep this contained+simple).
!
! Arrays are (ny, batch) with the Chebyshev index leading: a(j,col) holds the
! coefficient of T_{j-1}. For the 3D field (ny,nx,nz) pass batch = nx*nz by
! sequence association. Requires ny >= 2.
!
! Requires -cuda on a GPU node.
!===============================================================================

module cheb_ops_mod
  use kinds, only: tp, wp, ip
  use cudafor
  implicit none
  private
  public :: cheb_deriv_c, cheb_deriv_c_tp, apply_L

  ! Column-leading (batch, ny) scratch for the coalesced recurrence; one pair per
  ! kind (wp serves apply_L/stats, tp serves the transform tier). Grow/shape-tracked.
  complex(wp), allocatable, device :: cdwp_a(:,:), cdwp_b(:,:)
  complex(tp), allocatable, device :: cdtp_a(:,:), cdtp_b(:,:)

contains

  ! GROW-ONLY capacity, deliberately not exact-resize. step_full's
  ! mean block alternates cheb_deriv_c between batch = nx*nz (apply_L) and batch = 1
  ! (mean_stage_*), so an exact-resize scheme would dealloc/realloc this LIVE
  ! module device pair SIX times per step -- freeing device memory that queued
  ! kernels still reference. Under the legacy default stream that is
  ! (accidentally) serialized; on a single-stream regime it hangs or faults
  ! inside the runtime's descriptor handling.
  ! Grow-only removes the churn: the kernels index (col,k) with
  ! col <= batch, k <= ny, valid at any larger capacity. Same fix philosophy as
  ! helm_band_gpu_mod's private mean-path scratch (its header, lines 39-45).
  ! RULE: never dealloc/realloc device
  ! allocatables between kernel launches without draining the stream first.
  subroutine ensure_cd_wp(ny, batch)
    integer(ip), intent(in) :: ny, batch
    integer(ip) :: c1, c2
    c1 = batch;  c2 = ny
    if (allocated(cdwp_a)) then
       if (size(cdwp_a,1) >= batch .and. size(cdwp_a,2) >= ny) return
       c1 = max(c1, int(size(cdwp_a,1), ip));  c2 = max(c2, int(size(cdwp_a,2), ip))
       deallocate(cdwp_a, cdwp_b)               ! grow: rare (phase boundary), drained
    end if
    allocate(cdwp_a(c1, c2), cdwp_b(c1, c2))
  end subroutine ensure_cd_wp

  subroutine ensure_cd_tp(ny, batch)
    integer(ip), intent(in) :: ny, batch
    integer(ip) :: c1, c2
    c1 = batch;  c2 = ny
    if (allocated(cdtp_a)) then
       if (size(cdtp_a,1) >= batch .and. size(cdtp_a,2) >= ny) return
       c1 = max(c1, int(size(cdtp_a,1), ip));  c2 = max(c2, int(size(cdtp_a,2), ip))
       deallocate(cdtp_a, cdtp_b)               ! grow: rare (phase boundary), drained
    end if
    allocate(cdtp_a(c1, c2), cdtp_b(c1, c2))
  end subroutine ensure_cd_tp

  subroutine cheb_deriv_c(ny, batch, a, b)
    integer(ip), intent(in) :: ny, batch
    complex(wp), device     :: a(ny, batch)   ! Chebyshev coeffs of f
    complex(wp), device     :: b(ny, batch)   ! Chebyshev coeffs of df/dy
    integer(ip) :: k, col
    call ensure_cd_wp(ny, batch)
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch                              ! a (y-leading) -> cdwp_a (col-leading)
       do k = 1, ny
          cdwp_a(col, k) = a(k, col)
       end do
    end do
    !$cuf kernel do(1) <<<*,*>>>
    do col = 1, batch                              ! coalesced recurrence (col contiguous)
       cdwp_b(col, ny) = (0.0_wp, 0.0_wp)          ! coeff N = ny-1
       do k = ny-2, 0, -1                          ! coeff index k = N-1 .. 0
          if (k+2 <= ny-1) then
             cdwp_b(col, k+1) = cdwp_b(col, k+3) + real(2*(k+1), wp) * cdwp_a(col, k+2)
          else
             cdwp_b(col, k+1) =                    real(2*(k+1), wp) * cdwp_a(col, k+2)
          end if
       end do
       cdwp_b(col, 1) = 0.5_wp * cdwp_b(col, 1)    ! halve coeff 0
    end do
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch                              ! cdwp_b (col-leading) -> b (y-leading)
       do k = 1, ny
          b(k, col) = cdwp_b(col, k)
       end do
    end do
  end subroutine cheb_deriv_c

  ! cheb_deriv_c_tp -- transform-tier twin of cheb_deriv_c on the tp kind (real32
  ! at TP=sp, real64 otherwise). velocity_mod/forcing_mod call this
  ! on their tp workspace; the wp routine above keeps serving apply_L
  ! and stats_mod. These are two named routines rather than one generic because
  ! every caller sequence-associates a rank-3 (ny,nx,nz) array onto the rank-2
  ! (ny,batch) dummy, which a generic interface cannot resolve (NVFORTRAN-0155).
  subroutine cheb_deriv_c_tp(ny, batch, a, b)
    integer(ip), intent(in) :: ny, batch
    complex(tp), device     :: a(ny, batch)   ! Chebyshev coeffs of f
    complex(tp), device     :: b(ny, batch)   ! Chebyshev coeffs of df/dy
    integer(ip) :: k, col
    call ensure_cd_tp(ny, batch)
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do k = 1, ny
          cdtp_a(col, k) = a(k, col)
       end do
    end do
    !$cuf kernel do(1) <<<*,*>>>
    do col = 1, batch
       cdtp_b(col, ny) = (0.0_tp, 0.0_tp)          ! coeff N = ny-1
       do k = ny-2, 0, -1                          ! coeff index k = N-1 .. 0
          if (k+2 <= ny-1) then
             cdtp_b(col, k+1) = cdtp_b(col, k+3) + real(2*(k+1), tp) * cdtp_a(col, k+2)
          else
             cdtp_b(col, k+1) =                    real(2*(k+1), tp) * cdtp_a(col, k+2)
          end if
       end do
       cdtp_b(col, 1) = 0.5_tp * cdtp_b(col, 1)    ! halve coeff 0
    end do
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do k = 1, ny
          b(k, col) = cdtp_b(col, k)
       end do
    end do
  end subroutine cheb_deriv_c_tp

  ! apply_L: Lq = (D^2 - k^2) q  -- the linear viscous/Helmholtz operator L for a
  ! complex field given in Chebyshev coefficients, batched over the (kx,kz) modes.
  !
  ! D^2 is formed by applying the validated first-derivative recurrence
  ! cheb_deriv_c TWICE. Chebyshev differentiation is exact in coefficient space,
  ! so D(D q) equals the direct second-derivative recurrence (CPU cheb_d2) to
  ! round-off -- the test cross-checks exactly that. The -k^2 term is then
  ! subtracted per mode with k^2 = kx(ix)^2 + kz(iz)^2.
  !
  ! scr is caller-supplied scratch of the same shape (ny,nx,nz); it returns Dq.
  ! The input q is preserved (needed for the -k^2 q term and by callers). To form
  ! L^2 q (for the v biharmonic RHS) call apply_L twice: apply_L(...,q,Lq,scr)
  ! then apply_L(...,Lq,L2q,scr). kx,kz are the cuFFT-order wavenumbers (grid_mod).
  !
  ! INVARIANT (variable dt): L = D^2 - k^2 carries no time-step dependence; the
  ! per-stage Helmholtz shift lambda_k = Re/(beta_k*dt) lives in the implicit
  ! solve, not here. apply_L is therefore reused unchanged for any dt.
  subroutine apply_L(ny, nx, nz, kx, kz, q, Lq, scr)
    integer(ip), intent(in) :: ny, nx, nz
    real(wp),    device     :: kx(nx), kz(nz)
    complex(wp), device     :: q(ny,nx,nz)    ! input  : Chebyshev coeffs of q
    complex(wp), device     :: Lq(ny,nx,nz)   ! output : Chebyshev coeffs of Lq
    complex(wp), device     :: scr(ny,nx,nz)  ! scratch: returns Dq
    integer(ip) :: iy, ix, iz
    real(wp)    :: k2
    call cheb_deriv_c(ny, nx*nz, q,   scr)    ! scr = D q
    call cheb_deriv_c(ny, nx*nz, scr, Lq)     ! Lq  = D^2 q
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do iy = 1, ny
             k2 = kx(ix)*kx(ix) + kz(iz)*kz(iz)
             Lq(iy,ix,iz) = Lq(iy,ix,iz) - k2 * q(iy,ix,iz)
          end do
       end do
    end do
  end subroutine apply_L

end module cheb_ops_mod
