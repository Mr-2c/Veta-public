!===============================================================================
! cheb_fft_mod.F90  --  cuFFT layer, increment 4b: GPU Chebyshev transform
!                       (Gauss-Lobatto-Chebyshev <-> Chebyshev coefficients)
!                       via the even-extension cosine (DCT-I) trick.
!
! cuFFT has no native DCT, so the Chebyshev transform of N+1 GLC values is done
! through a real FFT of the length-2N even extension. With
!   P(y)_k = real( fft( evenext(y) ) )_k = sum_{j=0}^{N} ctil_j y_j cos(pi j k/N)
!   ctil = (1 at endpoints, 2 interior)        [extension multiplicities]
!   cbar = (2 at endpoints, 1 interior)        [coefficient normalisation]
! and noting cbar_k * ctil_k = 2 for all k, the pair is:
!   FORWARD (values v -> coeffs a):  a_k = P(v)_k / (N * cbar_k)
!   INVERSE (coeffs a -> values v):  v_j = P( a/ctil )_j
! Both reuse the single operator P, so the inverse needs only a pre-scale of
! the coefficients (halve the interior) -- no additive endpoint terms.
! Round-trip is exact because P(P(v)) = 2N v.
!
! Validated against the trusted CPU cheb_forward / cheb_eval (test_cheb_fft).
!
! N = degree (n+1 points). Even-extension length = 2N. R2C of length 2N has
! N+1 complex outputs, whose real parts are the cosine sums. Batched over the
! trailing dimension. Scratch is PERSISTENT module state (grow-only, see the
! declarations below) rather than allocate-per-call.
!
! Requires -cuda -cudalib=cufft on a GPU node.
!===============================================================================

module cheb_fft_mod
  use kinds,          only: tp, ip
  use cudafor
  use transforms_mod, only: tf_r2c, tf_c2c, CUFFT_FORWARD
  implicit none
  private
  public :: cheb_fft_fwd, cheb_fft_inv, cheb_fft_c_fwd, cheb_fft_c_inv
  public :: cheb_fft_scratch_free

  ! Persistent (grow-only) module device scratch, instead of a per-call
  ! allocate/deallocate firing on EVERY transform (dozens of cudaMalloc/cudaFree
  ! per time step).
  ! CONSTRAINT: the leading dimension must EXACTLY match the cuFFT plan layout
  ! (length 2n contiguous, batch distance 2n; R2C output rows n+1), so buffers
  ! are reallocated when n changes and grow-only in the batch dimension. In the
  ! solver a single n is in play, so after the first call there are no
  ! (re)allocations; alternating n values (only the test programs do that)
  ! just realloc per case. Module device arrays are legal inside CUF kernels
  ! (the NVFORTRAN-0155 restriction is derived-type components only).
  real(tp),    device, allocatable, save :: gr_buf(:,:)   ! real even-extension
  complex(tp), device, allocatable, save :: cr_buf(:,:)   ! real-variant R2C out
  ! The complex variant runs its C2C IN PLACE (gc_buf -> gc_buf):
  ! a plain batched cufftMakePlan1d supports in-place execution, so a
  ! second length-2n buffer is unnecessary -- avoiding it saves
  ! 16 B/point of the y-transform tier (~1.26 GB/GPU at the 16-node case).
  ! The R2C real variant stays
  ! out-of-place (in-place R2C needs the padded 2(n/2+1) row layout).
  complex(tp), device, allocatable, save :: gc_buf(:,:)   ! complex even-ext + C2C in-place

contains

  !-- ensure the REAL-variant scratch matches (2n x >=batch) -------------------
  subroutine ensure_r(n, batch)
    integer(ip), intent(in) :: n, batch
    if (allocated(gr_buf)) then
       if (size(gr_buf,1) /= 2*n .or. size(gr_buf,2) < batch) &
            deallocate(gr_buf, cr_buf)
    end if
    if (.not. allocated(gr_buf)) &
         allocate(gr_buf(0:2*n-1, batch), cr_buf(0:n, batch))
  end subroutine ensure_r

  !-- ensure the COMPLEX-variant scratch matches (2n x >=batch) ----------------
  subroutine ensure_c(n, batch)
    integer(ip), intent(in) :: n, batch
    if (allocated(gc_buf)) then
       if (size(gc_buf,1) /= 2*n .or. size(gc_buf,2) < batch) &
            deallocate(gc_buf)
    end if
    if (.not. allocated(gc_buf)) &
         allocate(gc_buf(0:2*n-1, batch))
  end subroutine ensure_c

  !-- optional cleanliness hook (program end frees regardless) -----------------
  subroutine cheb_fft_scratch_free()
    if (allocated(gr_buf)) deallocate(gr_buf, cr_buf)
    if (allocated(gc_buf)) deallocate(gc_buf)
  end subroutine cheb_fft_scratch_free

  !---------------------------------------------------------------------------
  ! Forward: physical GLC values v(0:n,batch) -> Chebyshev coeffs a(0:n,batch).
  ! `plan` must be an R2C plan of length 2*n with `batch` transforms.
  !---------------------------------------------------------------------------
  subroutine cheb_fft_fwd(plan, n, batch, v, a)
    integer,     intent(in) :: plan
    integer(ip), intent(in) :: n, batch
    real(tp),    device     :: v(0:n, batch)
    real(tp),    device     :: a(0:n, batch)
    integer(ip) :: j, k, col
    real(tp)    :: cbar, invn

    call ensure_r(n, batch)                       ! persistent scratch

    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do j = 0, 2*n-1
          if (j <= n) then
             gr_buf(j,col) = v(j,col)
          else
             gr_buf(j,col) = v(2*n-j,col)
          end if
       end do
    end do

    call tf_r2c(plan, gr_buf, cr_buf)

    invn = 1.0_tp / real(n,tp)
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do k = 0, n
          cbar = merge(2.0_tp, 1.0_tp, k == 0 .or. k == n)
          a(k,col) = real(cr_buf(k,col), tp) * invn / cbar
       end do
    end do
  end subroutine cheb_fft_fwd

  !---------------------------------------------------------------------------
  ! Inverse: Chebyshev coeffs a(0:n,batch) -> physical GLC values v(0:n,batch).
  ! Same R2C plan of length 2*n.
  !---------------------------------------------------------------------------
  subroutine cheb_fft_inv(plan, n, batch, a, v)
    integer,     intent(in) :: plan
    integer(ip), intent(in) :: n, batch
    real(tp),    device     :: a(0:n, batch)
    real(tp),    device     :: v(0:n, batch)
    integer(ip) :: j, kk, col
    real(tp)    :: ctil

    call ensure_r(n, batch)                       ! persistent scratch

    ! even extension of the pre-scaled coefficients (interior halved)
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do j = 0, 2*n-1
          if (j <= n) then
             kk = j
          else
             kk = 2*n - j
          end if
          ctil = merge(1.0_tp, 2.0_tp, kk == 0 .or. kk == n)
          gr_buf(j,col) = a(kk,col) / ctil
       end do
    end do

    call tf_r2c(plan, gr_buf, cr_buf)

    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do j = 0, n
          v(j,col) = real(cr_buf(j,col), tp)
       end do
    end do
  end subroutine cheb_fft_inv

  !---------------------------------------------------------------------------
  ! COMPLEX forward: complex GLC values v(0:n,batch) -> complex coeffs a.
  ! Same construction as cheb_fft_fwd but the even extension is complex and the
  ! transform is C2C; keep the full complex output (no real-part extraction).
  ! `plan` must be a C2C plan of length 2*n with `batch` transforms.
  !---------------------------------------------------------------------------
  subroutine cheb_fft_c_fwd(plan, n, batch, v, a)
    integer,     intent(in) :: plan
    integer(ip), intent(in) :: n, batch
    complex(tp), device     :: v(0:n, batch)
    complex(tp), device     :: a(0:n, batch)
    integer(ip) :: j, k, col
    real(tp)    :: cbar, invn

    call ensure_c(n, batch)                       ! persistent scratch

    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do j = 0, 2*n-1
          if (j <= n) then
             gc_buf(j,col) = v(j,col)
          else
             gc_buf(j,col) = v(2*n-j,col)
          end if
       end do
    end do

    call tf_c2c(plan, gc_buf, gc_buf, CUFFT_FORWARD)   ! in place

    invn = 1.0_tp / real(n,tp)
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do k = 0, n
          cbar = merge(2.0_tp, 1.0_tp, k == 0 .or. k == n)
          a(k,col) = gc_buf(k,col) * (invn/cbar)
       end do
    end do
  end subroutine cheb_fft_c_fwd

  !---------------------------------------------------------------------------
  ! COMPLEX inverse: complex coeffs a(0:n,batch) -> complex GLC values v.
  !---------------------------------------------------------------------------
  subroutine cheb_fft_c_inv(plan, n, batch, a, v)
    integer,     intent(in) :: plan
    integer(ip), intent(in) :: n, batch
    complex(tp), device     :: a(0:n, batch)
    complex(tp), device     :: v(0:n, batch)
    integer(ip) :: j, kk, col
    real(tp)    :: ctil

    call ensure_c(n, batch)                       ! persistent scratch

    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do j = 0, 2*n-1
          if (j <= n) then
             kk = j
          else
             kk = 2*n - j
          end if
          ctil = merge(1.0_tp, 2.0_tp, kk == 0 .or. kk == n)
          gc_buf(j,col) = a(kk,col) / ctil
       end do
    end do

    call tf_c2c(plan, gc_buf, gc_buf, CUFFT_FORWARD)   ! in place

    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do j = 0, n
          v(j,col) = gc_buf(j,col)
       end do
    end do
  end subroutine cheb_fft_c_inv

end module cheb_fft_mod
