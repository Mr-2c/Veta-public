!===============================================================================
! helm_band_gpu_mod.F90  --  batched GPU port of the O(n) banded Shen-Helmholtz
! solve (helm_band_mod). One CUDA thread per (kx,kz) mode solves
!     (S + mu B) uhat = ftilde,   mu = kx^2 + kz^2 + lambda,
! for the complex spectral RHS, using the SAME parity-split row-differencing as
! the validated CPU routine (difference rows i <= L-3; keep the top two rows of
! each parity in-band). The dt-independent structure arrays sdiag/stail/bdiag and
! the scalar boff are computed once on the host (helm_band_setup) and uploaded;
! lambda = Re/(beta_k dt) enters per call as a scalar, so changing dt is free.
!
! COALESCED COLUMN-LEADING LAYOUT.  The solve is one
! thread per (kx,kz) column doing an O(n) sequential sweep, and the field/scratch
! arrays it walks are COLUMN-LEADING -- (col, k) with col = ix+(iz-1)*nx the
! contiguous (leading) index, k=0..n-2 the strided one -- so a warp of 32
! consecutive columns hits 32 consecutive addresses (coalesced) instead of a
! y-leading (k, ix, iz) layout whose warp stride would be (n-1) (32 separate
! transactions per access). In that layout this kernel measured ~44% of step /
! ~519us per launch with ~1% occupancy and uncoalesced strided access. The
! caller's arrays stay y-leading; helm_band_solve_gpu transposes the RHS in and
! the solution out (hb_to_cl / hb_from_cl, bandwidth-bound, full-occupancy), and
! the per-mode band scratch lives in persistent column-leading module buffers.
! This changes ONLY memory order -- the per-column arithmetic is identical, so the
! result is bit-for-bit the same (the helm/stepper xcheck tests are the gate).
! NOTE: occupancy is unchanged (still nx*nz threads); a parallel banded solver is
! the separate follow-on if coalescing alone is insufficient.
!
! Precision is dp throughout: the wall-normal solve stays double even when the
! field precision wp is single (cast at the call boundary -- deferred).
!
! Build:
!   nvfortran -cuda -c helm_band_gpu_mod.F90
!===============================================================================
module helm_band_gpu_mod
  use kinds, only: dp, ip
  use cudafor
  implicit none
  private
  public :: helm_band_solve_gpu
  ! helm_band_kernel is exposed so the on-device mean solve (mean_gpu_mod) can run
  ! the SAME validated banded solve directly at ncol=1 (one (0,0) column) using its
  ! own tiny persistent scratch -- never the module buffers below -- so the mean's
  ! ncol=1 launches don't dealloc/realloc the fluctuation solve's ncol=bnkx*nz
  ! buffers (the column-leading layout is exact-resize, not grow-only). For ncol=1
  ! the column-leading (1,0:n-2) layout is bit-identical memory to y-leading
  ! (0:n-2,1,1), so no transpose is needed -- see mean_gpu_mod.
  public :: helm_band_kernel

  ! Persistent column-leading scratch (grow/shape-tracked; one (re)alloc per size).
  ! fcl/ucl carry the transposed RHS/solution; dia/up1/up2 are the band scratch.
  complex(dp), allocatable, device :: hb_fcl(:,:), hb_ucl(:,:)
  real(dp),    allocatable, device :: hb_dia(:,:), hb_up1(:,:), hb_up2(:,:)

contains

  attributes(global) subroutine helm_band_kernel(n, nx, nz, lambda, kx, kz, &
                       sdiag, stail, bdiag, boff, ftilde, uhat, dia, up1, up2)
    integer(ip), value :: n, nx, nz
    real(dp),    value :: lambda, boff
    real(dp)    :: kx(nx), kz(nz)
    real(dp)    :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2)
    complex(dp) :: ftilde(nx*nz, 0:n-2)            ! COLUMN-LEADING (col, k)
    complex(dp) :: uhat(nx*nz, 0:n-2)
    real(dp)    :: dia(nx*nz, 0:n-2), up1(nx*nz, 0:n-2), up2(nx*nz, 0:n-2)
    integer(ip) :: ix, iz, col, m, pstart, L, i, k, kprev, k1, k2
    real(dp)    :: mu, r, fac

    col = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (col > nx*nz) return
    ix = mod(col-1, nx) + 1                         ! col = ix + (iz-1)*nx
    iz = (col-1)/nx + 1

    m  = n - 2
    mu = kx(ix)*kx(ix) + kz(iz)*kz(iz) + lambda

    do pstart = 0, 1
       L = (m - pstart)/2 + 1
       ! (this guard is DEAD for n=2 -- Fortran (-1)/2 = 0 gives L=1; harmless
       !  because helm_band_setup refuses n < 4 [r3], so no factored system
       !  with a degenerate parity class can reach this kernel)
       if (L < 1) cycle

       ! Build the banded system (parity-local index i; k = pstart+2i).
       do i = 0, L-1
          k = pstart + 2*i
          if (i <= L-3) then                              ! differenced row
             r = stail(k) / stail(k+2)
             dia(col,i)  = sdiag(k) + mu*bdiag(k) - r*(mu*boff)
             up1(col,i)  = stail(k) + mu*boff - r*(sdiag(k+2) + mu*bdiag(k+2))
             up2(col,i)  = -r*(mu*boff)
             uhat(col,k) = ftilde(col,k) - r*ftilde(col,k+2)
          else if (i == L-2) then                         ! in-band (keep k+2)
             dia(col,i)  = sdiag(k) + mu*bdiag(k)
             up1(col,i)  = stail(k) + mu*boff
             up2(col,i)  = 0.0_dp
             uhat(col,k) = ftilde(col,k)
          else                                            ! top row
             dia(col,i)  = sdiag(k) + mu*bdiag(k)
             up1(col,i)  = 0.0_dp
             up2(col,i)  = 0.0_dp
             uhat(col,k) = ftilde(col,k)
          end if
       end do

       ! Forward elimination of the constant subdiagonal sub = mu*boff (i>=1).
       do i = 1, L-1
          k     = pstart + 2*i
          kprev = pstart + 2*(i-1)
          fac = (mu*boff) / dia(col,i-1)
          dia(col,i)  = dia(col,i)  - fac*up1(col,i-1)
          up1(col,i)  = up1(col,i)  - fac*up2(col,i-1)
          uhat(col,k) = uhat(col,k) - fac*uhat(col,kprev)
       end do

       ! Back substitution, in place in uhat (positions i+1,i+2 already solved).
       k = pstart + 2*(L-1)
       uhat(col,k) = uhat(col,k) / dia(col,L-1)
       if (L >= 2) then
          k  = pstart + 2*(L-2)
          k1 = pstart + 2*(L-1)
          uhat(col,k) = (uhat(col,k) - up1(col,L-2)*uhat(col,k1)) / dia(col,L-2)
       end if
       do i = L-3, 0, -1
          k  = pstart + 2*i
          k1 = pstart + 2*(i+1)
          k2 = pstart + 2*(i+2)
          uhat(col,k) = ( uhat(col,k) - up1(col,i)*uhat(col,k1) &
                                      - up2(col,i)*uhat(col,k2) ) / dia(col,i)
       end do
    end do
  end subroutine helm_band_kernel

  !-- (re)size the column-leading scratch to (nx*nz, 0:n-2) --------------------
  subroutine ensure_hb(n, nx, nz)
    integer(ip), intent(in) :: n, nx, nz
    integer(ip) :: ncol
    ncol = nx*nz
    if (allocated(hb_fcl)) then
       if (size(hb_fcl,1) /= ncol .or. size(hb_fcl,2) /= n-1) &
            deallocate(hb_fcl, hb_ucl, hb_dia, hb_up1, hb_up2)
    end if
    if (.not. allocated(hb_fcl)) &
       allocate(hb_fcl(ncol,0:n-2), hb_ucl(ncol,0:n-2), &
                hb_dia(ncol,0:n-2), hb_up1(ncol,0:n-2), hb_up2(ncol,0:n-2))
  end subroutine ensure_hb

  !-- y-leading (k,ix,iz) <-> column-leading (col,k), col = ix+(iz-1)*nx -------
  subroutine hb_to_cl(n, nx, nz, src, dst)
    integer(ip), intent(in) :: n, nx, nz
    complex(dp), device     :: src(0:n-2, nx, nz)
    complex(dp), device     :: dst(nx*nz, 0:n-2)
    integer(ip) :: k, ix, iz, col
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, n-2
             col = ix + (iz-1)*nx
             dst(col, k) = src(k, ix, iz)
          end do
       end do
    end do
  end subroutine hb_to_cl

  subroutine hb_from_cl(n, nx, nz, src, dst)
    integer(ip), intent(in) :: n, nx, nz
    complex(dp), device     :: src(nx*nz, 0:n-2)
    complex(dp), device     :: dst(0:n-2, nx, nz)
    integer(ip) :: k, ix, iz, col
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, n-2
             col = ix + (iz-1)*nx
             dst(k, ix, iz) = src(col, k)
          end do
       end do
    end do
  end subroutine hb_from_cl

  !---------------------------------------------------------------------------
  ! Host launcher. Interface unchanged (y-leading ftilde/uhat; lambda/boff by
  ! value). Transposes the RHS to column-leading, runs the coalesced solve, and
  ! transposes the solution back. The band scratch is the persistent
  ! column-leading module buffers.
  !---------------------------------------------------------------------------
  subroutine helm_band_solve_gpu(n, nx, nz, lambda, kx, kz, sdiag, stail, bdiag, &
                                 boff, ftilde, uhat)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: lambda, boff
    real(dp),    device     :: kx(nx), kz(nz)
    real(dp),    device     :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2)
    complex(dp), device     :: ftilde(0:n-2,nx,nz), uhat(0:n-2,nx,nz)
    type(dim3)  :: grid, block
    integer(cuda_stream_kind) :: strm      ! chevron launches need the stream EXPLICIT
                                           ! (cudaforSetDefaultStream covers CUF kernels/copies;
                                           ! bare <<<>>> is not guaranteed). Serial: legacy stream.
    integer(ip) :: ncol
    ncol = nx*nz
    call ensure_hb(n, nx, nz)
    call hb_to_cl(n, nx, nz, ftilde, hb_fcl)            ! RHS  -> column-leading
    ! Small block (2 warps): this kernel is one thread per column, so at small
    ! grids the few warps are best spread across many SMs (more memory-level
    ! parallelism) rather than packed onto a few; a warp is still 32 consecutive
    ! columns -> coalesced.
    block = dim3(64, 1, 1)
    grid  = dim3((ncol + 63)/64, 1, 1)
    strm  = cudaforGetDefaultStream()
    call helm_band_kernel<<<grid, block, 0, strm>>>(n, nx, nz, lambda, kx, kz, &
                          sdiag, stail, bdiag, boff, hb_fcl, hb_ucl, &
                          hb_dia, hb_up1, hb_up2)
    call hb_from_cl(n, nx, nz, hb_ucl, uhat)            ! solution -> y-leading
  end subroutine helm_band_solve_gpu

end module helm_band_gpu_mod
