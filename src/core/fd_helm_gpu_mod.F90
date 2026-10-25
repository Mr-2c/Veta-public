!===============================================================================
! fd_helm_gpu_mod.F90 -- BATCHED per-(kx,kz) FD Helmholtz solve
! on the GPU: the FD-y twin of helm_band_gpu_mod's solve. For every mode,
!     ( -D2_fd + (kx^2 + kz^2 + lam) I ) u = f ,   u(walls) = 0 ,
! where D2_fd is the fdops band (built ONCE from the ygrid file's nodes and
! shared by all modes) and the wall rows are identity (homogeneous Dirichlet --
! the same convention as the Cheb tau solve rows).
!
! DESIGN (correctness-first):
!   * one THREAD per mode; the mode's (2kl+1) x (0:n) real band is assembled,
!     no-pivot LU-factored, and back-substituted per solve
!     ("build-factors-at-solve-time": storing factored bands per mode would be
!     ~9*ny doubles/mode = tens of GB at production -- regeneration is the
!     memory diet);
!   * band scratch is a GLOBAL chunk buffer (2kl+1, 0:n, chunk), modes processed
!     in host-looped chunks -- NO device automatic/stack arrays (blind-risk) and
!     ~142 MB at chunk=4096 for ny=481. Grow-only module allocatable, passed to
!     the kernel as a PLAIN DUMMY (NVFORTRAN-0155 discipline), device-synced
!     before any regrow;
!   * NO pivoting: -D2+mu with Dirichlet rows and mu >= 0 is nonsingular and
!     benignly conditioned for banded elimination (host twin fd_band_factor
!     carries the zero-pivot guard; this kernel is gated against it bit-tight
!     in test_fd_helm_gpu, and the mu >= 0 contract is the solver's own);
!   * launches on the default stream (explicit chevron stream argument).
!
! Build: nvfortran -cuda -c fd_helm_gpu_mod.F90   (after kinds; no other deps)
!===============================================================================
module fd_helm_gpu_mod
  use kinds, only: dp, ip
  use cudafor
  use iso_c_binding, only: c_intptr_t
  implicit none
  private
  public :: fd_helm_solve_gpu
  public :: fd_wall_derivs_gpu     ! Du(+-1) from point values, per mode
  public :: fd_apply_L_gpu         ! (D2 - k^2) on point values, per mode
  public :: fd_helm_cache_build, fd_helm_cache_clear, fd_helm_cache_stats

  ! grow-only chunk scratch (see header); sized (2*kl+1, 0:n, chunk)
  real(dp), allocatable, device, save :: fh_ab(:,:,:)

  ! TRANSPARENT FACTOR CACHE. Rebuilding the factors at every solve measures
  ! 47.8 ms/step against ~17 for Cheb: each of the ~9
  ! solve calls/step re-assembled + re-LU'd operators that depend only on
  ! (stage lambda, k^2) -- constants for a fixed-dt run. The cache prefactors
  ! the registered lambdas ONCE (rebuilt on dt retune by the same hooks as the
  ! vif precompute) in a MODE-FIRST layout (nmode, 2kl+1, 0:n): warp-adjacent
  ! modes read consecutive addresses -> coalesced substitution (the exact
  ! helm_band_gpu/cheb_ops column-leading recipe). fd_helm_solve_gpu checks the
  ! registry by EXACT fp match (every producer computes lam by the identical
  ! expression chain beta=0.5*C_k; a=beta*dt/Re; lam=1.0/a -- literals shared
  ! by stepper/par_stepper/mean/main hooks) and falls back to the rebuild
  ! kernel otherwise, so arbitrary-lambda callers (tests, mean) are untouched.
  ! Regeneration remains the memory diet at production scale (tens of GB at
  ! ny=481, P<=8): the build REFUSES above the caller's cap and reports.
  ! Factors and substitution reuse the rebuild kernel's arithmetic VERBATIM
  ! (same order) -> results are BIT-IDENTICAL either path (gated).
  ! OPERATOR STAMP: the registry also records the device
  ! ADDRESSES of the d2b/kx/kz arrays the factors were built from, and a solve
  ! call whose arguments live elsewhere falls back to the rebuild path
  ! (shape+lam alone would let a same-shape DIFFERENT operator silently
  ! reuse stale factors -- e.g. a two-grid test or tool). Address
  ! compare costs nothing on the fast path and errs toward rebuild (a copied
  ! array = miss = correct, just uncached; hit-rate regressions are gated by
  ! test_fd_helm_cache leg 3's exact counters). Residual contract, stated
  ! loudly: content behind a STAMPED address must not be rewritten without
  ! re-running fd_helm_cache_build -- in production d2b/kx/kz are uploaded
  ! once and the build hooks (init + dt retune) re-stamp on every rebuild.
  integer(ip), parameter :: FHC_MAX = 8
  integer(ip), save :: fhc_n = -1, fhc_kl = -1, fhc_nx = -1, fhc_nz = -1
  integer(ip), save :: fhc_nlam = 0
  real(dp),    save :: fhc_lam(FHC_MAX)
  integer(ip), save :: fhc_hits = 0, fhc_miss = 0
  integer(c_intptr_t), save :: fhc_pd2b = 0, fhc_pkx = 0, fhc_pkz = 0
  real(dp), allocatable, device, save :: fhc_ab(:,:,:,:)  ! (nmode, 2kl+1, 0:n, nlam)

contains

  !---------------------------------------------------------------------------
  ! fd_helm_solve_gpu -- solve for ALL modes of a slab.
  !   n, kl      : node degree (0:n) and band half-width
  !   nx, nz     : mode counts (kx slab-local, kz full)
  !   lam        : the Helmholtz shift (mu = kx^2 + kz^2 + lam; lam >= 0)
  !   kx, kz     : device wavenumber arrays
  !   d2b        : device (2*kl+1, 0:n) FD second-derivative band (fdops
  !                convention ab(kl+1+i-j, j); build on host, upload once)
  !   f, u       : device complex (0:n, nx, nz); u is overwritten with the
  !                solution (walls forced to 0; f's wall values are ignored)
  !   chunk_in   : optional modes-per-launch override (tests exercise the
  !                multi-chunk path with a tiny value; default 4096)
  !---------------------------------------------------------------------------
  subroutine fd_helm_solve_gpu(n, kl, nx, nz, lam, kx, kz, d2b, f, u, chunk_in)
    integer(ip), intent(in) :: n, kl, nx, nz
    real(dp),    intent(in) :: lam
    real(dp),    device     :: kx(nx), kz(nz), d2b(2*kl+1, 0:n)
    complex(dp), device     :: f(0:n, nx, nz), u(0:n, nx, nz)
    integer(ip), intent(in), optional :: chunk_in
    integer(ip) :: chunk, nmode, c0, nc, istat, il
    integer(cuda_stream_kind) :: strm
    type(dim3) :: grid, block

    chunk = 4096_ip
    if (present(chunk_in)) chunk = max(1_ip, chunk_in)
    nmode = nx*nz
    chunk = min(chunk, nmode)

    ! cached fast path: exact-lambda registry hit -> substitution
    ! only. The address stamps make "same shape, different operator" a MISS
    ! (falls through to the rebuild path below with THIS call's d2b).
    if (fhc_nlam > 0 .and. n == fhc_n .and. kl == fhc_kl .and. &
        nx == fhc_nx .and. nz == fhc_nz .and. &
        transfer(c_devloc(d2b), 0_c_intptr_t) == fhc_pd2b .and. &
        transfer(c_devloc(kx),  0_c_intptr_t) == fhc_pkx  .and. &
        transfer(c_devloc(kz),  0_c_intptr_t) == fhc_pkz) then
       do il = 1, fhc_nlam
          if (lam == fhc_lam(il)) then
             fhc_hits = fhc_hits + 1
             strm  = cudaforGetDefaultStream()
             block = dim3(128, 1, 1)
             grid  = dim3((nmode + 127)/128, 1, 1)
             call fhc_solve_kernel<<<grid, block, 0, strm>>>(n, kl, nx, nz, &
                  nmode, fhc_ab(:,:,:,il), f, u)
             return
          end if
       end do
       fhc_miss = fhc_miss + 1
    end if

    ! grow-only scratch (sync before any dealloc of live device memory)
    if (allocated(fh_ab)) then
       if (size(fh_ab,1) < 2*kl+1 .or. size(fh_ab,2) < n+1 .or. size(fh_ab,3) < chunk) then
          istat = cudaDeviceSynchronize()
          deallocate(fh_ab)
       end if
    end if
    if (.not. allocated(fh_ab)) allocate(fh_ab(2*kl+1, 0:n, chunk))


    strm  = cudaforGetDefaultStream()
    block = dim3(128, 1, 1)
    do c0 = 0, nmode - 1, chunk
       nc   = min(chunk, nmode - c0)
       grid = dim3((nc + 127)/128, 1, 1)
       call fd_helm_kernel<<<grid, block, 0, strm>>>(n, kl, nx, nz, c0, nc, lam, &
                                                     kx, kz, d2b, fh_ab, f, u)
    end do
  end subroutine fd_helm_solve_gpu

  !---------------------------------------------------------------------------
  ! fd_helm_cache_build -- prefactor nlam registered lambdas.
  ! cap_gb: refuse (built=.false., cache cleared) if the footprint exceeds it.
  ! Call from the dt-dependent precompute hooks ONLY (init + retune; the
  ! dealloc drains the device first).
  !---------------------------------------------------------------------------
  subroutine fd_helm_cache_build(n, kl, nx, nz, nlam, lams, kx, kz, d2b, &
                                 cap_gb, built, gb)
    integer(ip), intent(in)  :: n, kl, nx, nz, nlam
    real(dp),    intent(in)  :: lams(nlam), cap_gb
    real(dp),    device      :: kx(nx), kz(nz), d2b(2*kl+1, 0:n)
    logical,     intent(out) :: built
    real(dp),    intent(out) :: gb
    integer(ip) :: nmode, il, istat
    integer(cuda_stream_kind) :: strm
    type(dim3) :: grid, block
    nmode = nx*nz
    gb = real(nmode, dp)*real(2*kl+1, dp)*real(n+1, dp)*8.0_dp &
         *real(nlam, dp)/2.0_dp**30
    call fd_helm_cache_clear()
    built = (gb <= cap_gb .and. nlam >= 1 .and. nlam <= FHC_MAX)
    if (.not. built) return
    allocate(fhc_ab(nmode, 2*kl+1, 0:n, nlam), stat=istat)
    if (istat /= 0) then                 ! cap above free memory: fall back, not abort
       built = .false.
       return
    end if
    fhc_n = n;  fhc_kl = kl;  fhc_nx = nx;  fhc_nz = nz
    fhc_nlam = nlam;  fhc_lam(1:nlam) = lams
    fhc_pd2b = transfer(c_devloc(d2b), 0_c_intptr_t)
    fhc_pkx  = transfer(c_devloc(kx),  0_c_intptr_t)
    fhc_pkz  = transfer(c_devloc(kz),  0_c_intptr_t)
    strm  = cudaforGetDefaultStream()
    block = dim3(128, 1, 1)
    grid  = dim3((nmode + 127)/128, 1, 1)
    do il = 1, nlam
       call fhc_factor_kernel<<<grid, block, 0, strm>>>(n, kl, nx, nz, nmode, &
            lams(il), kx, kz, d2b, fhc_ab(:,:,:,il))
    end do
    istat = cudaDeviceSynchronize()   ! factors complete before first solve
  end subroutine fd_helm_cache_build

  subroutine fd_helm_cache_clear()
    integer(ip) :: istat
    if (allocated(fhc_ab)) then
       istat = cudaDeviceSynchronize()   ! never free live device memory
       deallocate(fhc_ab)
    end if
    fhc_n = -1;  fhc_kl = -1;  fhc_nx = -1;  fhc_nz = -1;  fhc_nlam = 0
    fhc_pd2b = 0;  fhc_pkx = 0;  fhc_pkz = 0
  end subroutine fd_helm_cache_clear

  subroutine fd_helm_cache_stats(hits, miss)
    integer(ip), intent(out) :: hits, miss
    hits = fhc_hits;  miss = fhc_miss
  end subroutine fd_helm_cache_stats

  !-- assemble + no-pivot LU into the MODE-FIRST cache slice (arithmetic is the
  !-- rebuild kernel's VERBATIM: same statements, only the ab indexing layout
  !-- differs -> bit-identical factors) ----------------------------------------
  attributes(global) subroutine fhc_factor_kernel(n, kl, nx, nz, nm, lam, &
                                                  kx, kz, d2b, ab)
    integer(ip), value :: n, kl, nx, nz, nm
    real(dp),    value :: lam
    real(dp)    :: kx(nx), kz(nz), d2b(2*kl+1, 0:n), ab(nm, 2*kl+1, 0:n)
    integer(ip) :: m, ix, iz, i, j, k, b
    real(dp)    :: mu, piv, l
    m = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (m > nm) return
    ix = mod(m - 1, nx) + 1
    iz = (m - 1)/nx + 1
    mu = kx(ix)*kx(ix) + kz(iz)*kz(iz) + lam
    do j = 0, n
       do b = 1, 2*kl + 1
          ab(m, b, j) = -d2b(b, j)
       end do
       ab(m, kl + 1, j) = ab(m, kl + 1, j) + mu
    end do
    do j = 0, min(kl, n)                       ! row 0
       ab(m, kl + 1 - j, j) = 0.0_dp
    end do
    ab(m, kl + 1, 0) = 1.0_dp
    do j = max(0_ip, n - kl), n                ! row n
       ab(m, kl + 1 + n - j, j) = 0.0_dp
    end do
    ab(m, kl + 1, n) = 1.0_dp
    do k = 0, n
       piv = ab(m, kl + 1, k)
       do i = k + 1, min(k + kl, n)
          l = ab(m, kl + 1 + i - k, k)/piv
          ab(m, kl + 1 + i - k, k) = l
          do j = k + 1, min(k + kl, n)
             ab(m, kl + 1 + i - j, j) = ab(m, kl + 1 + i - j, j) - l*ab(m, kl + 1 + k - j, j)
          end do
       end do
    end do
  end subroutine fhc_factor_kernel

  !-- substitution on prefactored MODE-FIRST bands (loops verbatim from the
  !-- rebuild kernel -> bit-identical solutions; band reads are coalesced) -----
  attributes(global) subroutine fhc_solve_kernel(n, kl, nx, nz, nm, ab, f, u)
    integer(ip), value :: n, kl, nx, nz, nm
    real(dp)    :: ab(nm, 2*kl+1, 0:n)
    complex(dp) :: f(0:n, nx, nz), u(0:n, nx, nz)
    integer(ip) :: m, ix, iz, i, j, k
    complex(dp) :: s
    m = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (m > nm) return
    ix = mod(m - 1, nx) + 1
    iz = (m - 1)/nx + 1
    do j = 1, n - 1
       u(j, ix, iz) = f(j, ix, iz)
    end do
    u(0, ix, iz) = (0.0_dp, 0.0_dp)
    u(n, ix, iz) = (0.0_dp, 0.0_dp)
    do k = 0, n
       s = u(k, ix, iz)
       do i = k + 1, min(k + kl, n)
          u(i, ix, iz) = u(i, ix, iz) - ab(m, kl + 1 + i - k, k)*s
       end do
    end do
    do k = n, 0, -1
       s = u(k, ix, iz)
       do j = k + 1, min(k + kl, n)
          s = s - ab(m, kl + 1 + k - j, j)*u(j, ix, iz)
       end do
       u(k, ix, iz) = s/ab(m, kl + 1, k)
    end do
  end subroutine fhc_solve_kernel

  !---------------------------------------------------------------------------
  ! One thread = one mode: assemble -D2+mu*I with identity wall rows into this
  ! thread's scratch slot, no-pivot banded LU in place, then forward/backward
  ! substitution directly on u's global-memory column (complex RHS, real band).
  ! Indexing is the fdops GB convention with 0-based columns:
  ! A(i,j) lives at ab(kl+1+i-j, j) for max(0,j-kl) <= i <= min(n,j+kl).
  !---------------------------------------------------------------------------
  attributes(global) subroutine fd_helm_kernel(n, kl, nx, nz, c0, nc, lam, &
                                               kx, kz, d2b, ab, f, u)
    integer(ip), value :: n, kl, nx, nz, c0, nc
    real(dp),    value :: lam
    real(dp)    :: kx(nx), kz(nz), d2b(2*kl+1, 0:n), ab(2*kl+1, 0:n, nc)
    complex(dp) :: f(0:n, nx, nz), u(0:n, nx, nz)
    integer(ip) :: t, m, ix, iz, i, j, k, b
    real(dp)    :: mu, piv, l
    complex(dp) :: s

    t = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (t > nc) return
    m = c0 + t
    if (m > nx*nz) return
    ix = mod(m - 1, nx) + 1
    iz = (m - 1)/nx + 1
    mu = kx(ix)*kx(ix) + kz(iz)*kz(iz) + lam

    ! --- assemble: A = -D2 + mu I; rows 0 and n -> identity ------------------
    do j = 0, n
       do b = 1, 2*kl + 1
          ab(b, j, t) = -d2b(b, j)
       end do
       ab(kl + 1, j, t) = ab(kl + 1, j, t) + mu
    end do
    do j = 0, min(kl, n)                       ! row 0
       ab(kl + 1 - j, j, t) = 0.0_dp
    end do
    ab(kl + 1, 0, t) = 1.0_dp
    do j = max(0_ip, n - kl), n                ! row n
       ab(kl + 1 + n - j, j, t) = 0.0_dp
    end do
    ab(kl + 1, n, t) = 1.0_dp

    ! --- RHS into u; homogeneous Dirichlet walls ------------------------------
    do j = 1, n - 1
       u(j, ix, iz) = f(j, ix, iz)
    end do
    u(0, ix, iz) = (0.0_dp, 0.0_dp)
    u(n, ix, iz) = (0.0_dp, 0.0_dp)

    ! --- no-pivot banded LU in place ------------------------------------------
    do k = 0, n
       piv = ab(kl + 1, k, t)
       do i = k + 1, min(k + kl, n)
          l = ab(kl + 1 + i - k, k, t)/piv
          ab(kl + 1 + i - k, k, t) = l
          do j = k + 1, min(k + kl, n)
             ab(kl + 1 + i - j, j, t) = ab(kl + 1 + i - j, j, t) - l*ab(kl + 1 + k - j, j, t)
          end do
       end do
    end do

    ! --- forward substitution (L y = b) on the complex column ----------------
    do k = 0, n
       s = u(k, ix, iz)
       do i = k + 1, min(k + kl, n)
          u(i, ix, iz) = u(i, ix, iz) - ab(kl + 1 + i - k, k, t)*s
       end do
    end do
    ! --- backward substitution (U x = y) --------------------------------------
    do k = n, 0, -1
       s = u(k, ix, iz)
       do j = k + 1, min(k + kl, n)
          s = s - ab(kl + 1 + k - j, j, t)*u(j, ix, iz)
       end do
       u(k, ix, iz) = s/ab(kl + 1, k, t)
    end do
  end subroutine fd_helm_kernel

  !---------------------------------------------------------------------------
  ! fd_wall_derivs_gpu -- batched wall first derivatives from POINT-VALUE
  ! columns: dlo = Du(-1), dhi = Du(+1) per mode, using the full-length
  ! one-sided weight rows from fdops fd_wall_weights (host-built, uploaded by
  ! the caller; zeros beyond each wall's stencil, so the uniform full-column
  ! dot is correct and simple). The FD twin of spec_helpers wall_derivs_c.
  !---------------------------------------------------------------------------
  subroutine fd_wall_derivs_gpu(n, nx, nz, wlo, whi, a, dlo, dhi)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    device     :: wlo(0:n), whi(0:n)
    complex(dp), device     :: a(0:n, nx, nz)
    complex(dp), device     :: dlo(nx, nz), dhi(nx, nz)
    integer(cuda_stream_kind) :: strm
    type(dim3) :: grid, block
    strm  = cudaforGetDefaultStream()
    block = dim3(128, 1, 1)
    grid  = dim3((nx*nz + 127)/128, 1, 1)
    call fd_wallder_kernel<<<grid, block, 0, strm>>>(n, nx, nz, wlo, whi, a, dlo, dhi)
  end subroutine fd_wall_derivs_gpu

  attributes(global) subroutine fd_wallder_kernel(n, nx, nz, wlo, whi, a, dlo, dhi)
    integer(ip), value :: n, nx, nz
    real(dp)    :: wlo(0:n), whi(0:n)
    complex(dp) :: a(0:n, nx, nz), dlo(nx, nz), dhi(nx, nz)
    integer(ip) :: t, ix, iz, j
    complex(dp) :: slo, shi
    t = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (t > nx*nz) return
    ix = mod(t - 1, nx) + 1
    iz = (t - 1)/nx + 1
    slo = (0.0_dp, 0.0_dp)
    shi = (0.0_dp, 0.0_dp)
    do j = 0, n
       slo = slo + wlo(j)*a(j, ix, iz)
       shi = shi + whi(j)*a(j, ix, iz)
    end do
    dlo(ix, iz) = slo
    dhi(ix, iz) = shi
  end subroutine fd_wallder_kernel

  !---------------------------------------------------------------------------
  ! fd_apply_L_gpu -- batched L = D2 - k^2 on POINT-VALUE columns: the FD twin
  ! of the Cheb apply_L used in RHS assembly. d2b is the shared fdops band.
  ! u and Lu MUST NOT alias (the banded matvec reads neighbours of rows already
  ! written) -- same contract as the Cheb apply_L's separate in/out arrays.
  !---------------------------------------------------------------------------
  subroutine fd_apply_L_gpu(n, kl, nx, nz, kx, kz, d2b, u, Lu)
    integer(ip), intent(in) :: n, kl, nx, nz
    real(dp),    device     :: kx(nx), kz(nz), d2b(2*kl+1, 0:n)
    complex(dp), device     :: u(0:n, nx, nz), Lu(0:n, nx, nz)
    integer(cuda_stream_kind) :: strm
    type(dim3) :: grid, block
    strm  = cudaforGetDefaultStream()
    block = dim3(128, 1, 1)
    ! (mode, row); int64 count -- nx*nz*(n+1) wraps int32 above
    ! ~4.5M modes/rank at ny=481 (unreachable today, silent-death class if hit)
    grid  = dim3(int((int(nx,8)*int(nz,8)*int(n+1,8) + 127_8)/128_8), 1, 1)
    call fd_applyL_kernel<<<grid, block, 0, strm>>>(n, kl, nx, nz, kx, kz, d2b, u, Lu)
  end subroutine fd_apply_L_gpu

  ! one thread per (mode, ROW): rows are independent, so this is
  ! free parallelism (a per-mode mapping runs ~2 warps/SM at 128^3, x9/step).
  ! The per-row j-ascending sum is UNCHANGED -> bit-identical values.
  attributes(global) subroutine fd_applyL_kernel(n, kl, nx, nz, kx, kz, d2b, u, Lu)
    integer(ip), value :: n, kl, nx, nz
    real(dp)    :: kx(nx), kz(nz), d2b(2*kl+1, 0:n)
    complex(dp) :: u(0:n, nx, nz), Lu(0:n, nx, nz)
    integer(ip) :: m, ix, iz, i, j
    integer(8)  :: t8
    real(dp)    :: k2
    complex(dp) :: s
    t8 = (int(blockIdx%x, 8) - 1)*blockDim%x + threadIdx%x
    if (t8 > int(nx,8)*int(nz,8)*int(n+1,8)) return
    m  = int((t8 - 1)/int(n + 1, 8), ip) + 1   ! mode index (int64-safe)
    i  = int(mod(t8 - 1, int(n + 1, 8)), ip)   ! row 0..n
    ix = mod(m - 1, nx) + 1
    iz = (m - 1)/nx + 1
    k2 = kx(ix)*kx(ix) + kz(iz)*kz(iz)
    s = (0.0_dp, 0.0_dp)
    do j = max(0_ip, i - kl), min(n, i + kl)
       s = s + d2b(kl + 1 + i - j, j)*u(j, ix, iz)
    end do
    Lu(i, ix, iz) = s - k2*u(i, ix, iz)
  end subroutine fd_applyL_kernel

end module fd_helm_gpu_mod
