!===============================================================================
! spectral3d_mod.F90  --  Full 3D spectral <-> physical transform, assembled
!                         from the validated pieces. No new transform code.
!
! Layout: complex array (ny, nx, nz), y LEADING.
!   y : Chebyshev (coeffs <-> GLC values)   -- cheb_fft_c on the leading axis,
!       batch = nx*nz (the 3D array is contiguous, so it reinterprets as
!       (0:ny-1, nx*nz) by sequence association).
!   x,z : Fourier -- ONE rank-2 (x,z)-plane C2C batched over y (fft3d's
!       plan_xz_c2c), executed in place.
!
! SIMPLIFICATION (reference build): x is carried as full C2C rather than R2C,
! so the whole field is complex and no transposes are needed. The physical
! field's imaginary part is ~0 (take the real part where a real field is
! wanted). The optimised real-x path is the _r2c variants below; the
! distributed transposes live in src/parallel.
!
! cuFFT is unnormalised: a full spectral->physical->spectral round-trip scales
! by nx*nz (the two Fourier round-trips; the Chebyshev pair is self-inverse).
! Dealiasing: the _da variants below ARE the 3/2-rule dealiased paths (x/z
! padded to nxd/nzd; y never dealiased); the plain
! variants remain for tests.
!
! Requires -cuda -cudalib=cufft on a GPU node.
!===============================================================================

module spectral3d_mod
  use kinds,          only: tp, ip
  use cudafor
  use transforms_mod, only: tf_c2c_plan, tf_destroy
  use cheb_fft_mod,   only: cheb_fft_c_fwd, cheb_fft_c_inv
  use fft3d_mod,      only: plan_xz_c2c, exec_xz_c2c, CUFFT_FORWARD, CUFFT_INVERSE
  use fft3d_mod,      only: plan_xz_r2c, plan_xz_c2r, exec_xz_r2c, exec_xz_c2r
  implicit none
  private
  ! FD-y IDENTITY LEGS: in FD mode the wall-normal state IS point
  ! values, so the y-transform legs are the identity and are SKIPPED (the x/z
  ! Fourier legs are untouched). Module-save flag, set ONCE at init by the FD
  ! wiring (s3d_set_ybasis_fd); every Chebyshev run leaves it down and executes
  ! the identical statements a Chebyshev-only build would.
  logical, save :: s3d_fd = .false.
  public :: s3d_set_ybasis_fd
  public :: s3d_plans, s3d_setup, s3d_destroy, to_physical, to_spectral
  public :: s3d_setup_da, to_physical_da, to_spectral_da
  ! real-x variants (non-dealiased), validated against C2C:
  public :: s3d_setup_r2c, s3d_destroy_r2c, to_physical_r2c, to_spectral_r2c
  ! real-x DEALIASED variants:
  public :: s3d_setup_da_r2c, to_physical_da_r2c, to_spectral_da_r2c

  ! x and z are ONE rank-2 (x,z)-plane transform batched over y
  ! (plan_xz_c2c), rather than an axis-2 slice loop (nz/nzd separate cuFFT calls
  ! per field per direction) + an axis-3 pass. Executed in place. The `work`/`W`
  ! scratch arguments of to_physical/to_spectral(_da) are RETAINED for interface
  ! compatibility (tests pass them) but are no longer touched.
  type :: s3d_plans
     integer :: cheb = 0, xz = 0
     integer :: xz_r2c = 0, xz_c2r = 0     ! fused real-x forward/inverse
     integer :: cheb_r = 0                 ! Chebyshev on nkx*nz cols
  end type s3d_plans

contains

  subroutine s3d_set_ybasis_fd(fd)
    logical, intent(in) :: fd
    s3d_fd = fd
  end subroutine s3d_set_ybasis_fd

  subroutine s3d_setup(pl, ny, nx, nz)
    type(s3d_plans), intent(out) :: pl
    integer(ip),     intent(in)  :: ny, nx, nz
    call tf_c2c_plan(pl%cheb, 2_ip*(ny-1), nx*nz)   ! Chebyshev even-extension C2C
    call plan_xz_c2c(pl%xz, ny, nx, nz)             ! one (x,z)-plane transform
  end subroutine s3d_setup

  subroutine s3d_destroy(pl)
    type(s3d_plans), intent(inout) :: pl
    call tf_destroy(pl%cheb);  call tf_destroy(pl%xz)
  end subroutine s3d_destroy

  !---------------------------------------------------------------------------
  ! Spectral (y=Chebyshev coeffs, x,z=Fourier) -> physical. Uses `work` as
  ! scratch; result returned in A.
  !---------------------------------------------------------------------------
  subroutine to_physical(pl, ny, nx, nz, A, work)
    type(s3d_plans), intent(in) :: pl
    integer(ip),     intent(in) :: ny, nx, nz
    complex(tp),     device     :: A(ny,nx,nz), work(ny,nx,nz)   ! work unused
    if (.not. s3d_fd) &                                          ! FD: y = identity
       call cheb_fft_c_inv(pl%cheb, ny-1, nx*nz, A, A)            ! y: coeffs -> values
    call exec_xz_c2c(pl%xz, ny, nx, nz, A, A, CUFFT_INVERSE)      ! x,z (one call, in place)
  end subroutine to_physical

  !---------------------------------------------------------------------------
  ! Physical -> spectral. Inverse chain order. Result in A (scaled by nx*nz vs
  ! the original spectral field, per cuFFT's unnormalised convention).
  !---------------------------------------------------------------------------
  subroutine to_spectral(pl, ny, nx, nz, A, work)
    type(s3d_plans), intent(in) :: pl
    integer(ip),     intent(in) :: ny, nx, nz
    complex(tp),     device     :: A(ny,nx,nz), work(ny,nx,nz)   ! work unused
    call exec_xz_c2c(pl%xz, ny, nx, nz, A, A, CUFFT_FORWARD)      ! x,z (one call, in place)
    if (.not. s3d_fd) &                                          ! FD: y = identity
       call cheb_fft_c_fwd(pl%cheb, ny-1, nx*nz, A, A)            ! y: values -> coeffs
  end subroutine to_spectral

  !===========================================================================
  ! [R2C] REAL-x non-dealiased transforms. Spectral is complex
  ! (ny,nkx,nz) with nkx=nx/2+1 (the non-negative kx half; the negative half is
  ! the Hermitian conjugate and is NOT stored). Physical is REAL (ny,nx,nz).
  !
  ! Chain (spectral->physical): cheb_inv on the nkx*nz complex columns, then the
  ! fused C2R does the real-x + complex-z inverse in one launch. Forward reverses.
  !
  ! WHY SEPARATE IN/OUT ARRAYS: R2C/C2R are not in-place (real and complex sizes
  ! differ), and cuFFT's multi-dim C2R may overwrite its complex input. So these
  ! take a complex spectral array S(ny,nkx,nz) and a REAL physical array P(ny,nx,nz)
  ! as distinct arguments. The Chebyshev step runs in place on S (complex->complex
  ! on the leading axis, batch nkx*nz) before the C2R / after the R2C.
  !
  ! Validated by test_spectral3d_r2c: round-trip identity, and the R2C spectral
  ! field equals the C2C field on the retained half kx in [0,nkx-1] to round-off.
  !===========================================================================
  subroutine s3d_setup_r2c(pl, ny, nx, nz)
    type(s3d_plans), intent(out) :: pl
    integer(ip),     intent(in)  :: ny, nx, nz
    integer(ip) :: nkx
    nkx = nx/2_ip + 1_ip
    call tf_c2c_plan(pl%cheb_r, 2_ip*(ny-1), nkx*nz)   ! Chebyshev on the nkx columns
    call plan_xz_r2c(pl%xz_r2c, ny, nx, nz)            ! fused real-x / complex-z fwd
    call plan_xz_c2r(pl%xz_c2r, ny, nx, nz)            ! fused inverse
  end subroutine s3d_setup_r2c

  subroutine s3d_destroy_r2c(pl)
    type(s3d_plans), intent(inout) :: pl
    call tf_destroy(pl%cheb_r);  call tf_destroy(pl%xz_r2c);  call tf_destroy(pl%xz_c2r)
  end subroutine s3d_destroy_r2c

  !-- spectral complex (ny,nkx,nz) -> real physical (ny,nx,nz). Consumes S. ------
  subroutine to_physical_r2c(pl, ny, nx, nz, S, P)
    type(s3d_plans), intent(in) :: pl
    integer(ip),     intent(in) :: ny, nx, nz
    complex(tp),     device     :: S(ny,(nx/2+1),nz)
    real(tp),        device     :: P(ny,nx,nz)
    integer(ip) :: nkx
    nkx = nx/2_ip + 1_ip
    if (.not. s3d_fd) &                                 ! FD: y = identity
       call cheb_fft_c_inv(pl%cheb_r, ny-1, nkx*nz, S, S)  ! y: coeffs -> values (in place on S)
    call exec_xz_c2r(pl%xz_c2r, S, P)                   ! fused C2R (x real, z complex)
  end subroutine to_physical_r2c

  !-- real physical (ny,nx,nz) -> spectral complex (ny,nkx,nz). Scaled by nx*nz. -
  subroutine to_spectral_r2c(pl, ny, nx, nz, P, S)
    type(s3d_plans), intent(in) :: pl
    integer(ip),     intent(in) :: ny, nx, nz
    real(tp),        device     :: P(ny,nx,nz)
    complex(tp),     device     :: S(ny,(nx/2+1),nz)
    integer(ip) :: nkx
    nkx = nx/2_ip + 1_ip
    call exec_xz_r2c(pl%xz_r2c, P, S)                   ! fused R2C (x real, z complex)
    if (.not. s3d_fd) &                                 ! FD: y = identity
       call cheb_fft_c_fwd(pl%cheb_r, ny-1, nkx*nz, S, S)  ! y: values -> coeffs (in place on S)
  end subroutine to_spectral_r2c

  !===========================================================================
  ! DEALIASED variants. The Fourier directions are padded to 3/2 size before
  ! the inverse transforms and truncated after the forward transforms, so
  ! quadratic products formed on the (ny,nxd,nzd) physical grid are alias-free
  ! in the retained (nx,nz) modes. y is not dealiased here.
  !===========================================================================

  subroutine s3d_setup_da(pl, ny, nx, nz)
    type(s3d_plans), intent(out) :: pl
    integer(ip),     intent(in)  :: ny, nx, nz
    integer(ip) :: nxd, nzd
    nxd = 3_ip*nx/2_ip;  nzd = 3_ip*nz/2_ip
    call tf_c2c_plan(pl%cheb, 2_ip*(ny-1), nx*nz)   ! Chebyshev on the coarse modes
    call plan_xz_c2c(pl%xz, ny, nxd, nzd)           ! one (x,z) transform, padded grid
  end subroutine s3d_setup_da

  !-- scatter: coarse modes (ny,nx,nz) -> padded (ny,nxd,nzd), zeros elsewhere -
  subroutine pad_xz(ny, nx, nz, nxd, nzd, src, dst)
    integer(ip), intent(in) :: ny, nx, nz, nxd, nzd
    complex(tp), device     :: src(ny, 0:nx-1, 0:nz-1)
    complex(tp), device     :: dst(ny, 0:nxd-1, 0:nzd-1)
    integer(ip) :: iy, kx, kz, kxd, kzd, nxh, nzh
    nxh = nx/2;  nzh = nz/2
    !$cuf kernel do(3) <<<*,*>>>
    do kz = 0, nzd-1
       do kx = 0, nxd-1
          do iy = 1, ny
             dst(iy,kx,kz) = (0.0_tp, 0.0_tp)
          end do
       end do
    end do
    !$cuf kernel do(3) <<<*,*>>>
    do kz = 0, nz-1
       do kx = 0, nx-1
          do iy = 1, ny
             kxd = merge(kx, nxd-(nx-kx), kx <= nxh)
             kzd = merge(kz, nzd-(nz-kz), kz <= nzh)
             dst(iy,kxd,kzd) = src(iy,kx,kz)
          end do
       end do
    end do
  end subroutine pad_xz

  !-- gather: padded (ny,nxd,nzd) -> coarse modes (ny,nx,nz) -----------------
  subroutine trunc_xz(ny, nxd, nzd, nx, nz, src, dst)
    integer(ip), intent(in) :: ny, nxd, nzd, nx, nz
    complex(tp), device     :: src(ny, 0:nxd-1, 0:nzd-1)
    complex(tp), device     :: dst(ny, 0:nx-1, 0:nz-1)
    integer(ip) :: iy, kx, kz, kxd, kzd, nxh, nzh
    nxh = nx/2;  nzh = nz/2
    !$cuf kernel do(3) <<<*,*>>>
    do kz = 0, nz-1
       do kx = 0, nx-1
          do iy = 1, ny
             kxd = merge(kx, nxd-(nx-kx), kx <= nxh)
             kzd = merge(kz, nzd-(nz-kz), kz <= nzh)
             dst(iy,kx,kz) = src(iy,kxd,kzd)
          end do
       end do
    end do
  end subroutine trunc_xz

  !-- spectral (ny,nx,nz) -> dealiased physical in P (ny,nxd,nzd); W unused
  subroutine to_physical_da(pl, ny, nx, nz, nxd, nzd, A, P, W)
    type(s3d_plans), intent(in) :: pl
    integer(ip),     intent(in) :: ny, nx, nz, nxd, nzd
    complex(tp),     device     :: A(ny,nx,nz), P(ny,nxd,nzd), W(ny,nxd,nzd)
    if (.not. s3d_fd) &                                 ! FD: y = identity
       call cheb_fft_c_inv(pl%cheb, ny-1, nx*nz, A, A)
    call pad_xz(ny, nx, nz, nxd, nzd, A, P)
    call exec_xz_c2c(pl%xz, ny, nxd, nzd, P, P, CUFFT_INVERSE)
  end subroutine to_physical_da

  !-- dealiased physical P (ny,nxd,nzd) -> spectral in A (ny,nx,nz); W unused
  subroutine to_spectral_da(pl, ny, nx, nz, nxd, nzd, P, W, A)
    type(s3d_plans), intent(in) :: pl
    integer(ip),     intent(in) :: ny, nx, nz, nxd, nzd
    complex(tp),     device     :: P(ny,nxd,nzd), W(ny,nxd,nzd), A(ny,nx,nz)
    call exec_xz_c2c(pl%xz, ny, nxd, nzd, P, P, CUFFT_FORWARD)
    call trunc_xz(ny, nxd, nzd, nx, nz, P, A)
    if (.not. s3d_fd) &                                 ! FD: y = identity
       call cheb_fft_c_fwd(pl%cheb, ny-1, nx*nz, A, A)
  end subroutine to_spectral_da

  !===========================================================================
  ! [R2C] DEALIASED real-x transforms. Spectral coarse (ny,nkx,nz),
  ! fine (ny,nkxd,nzd) with nkx=nx/2+1, nkxd=nxd/2+1; physical fine REAL
  ! (ny,nxd,nzd).  The x-axis stores ONLY kx>=0 (no negative half), so its
  ! pad/truncate is a LOW-PASS COPY (kx=0..nx/2 -> same fine index, zero above),
  ! NOT the +-Nyquist wrap.  The z-axis is unchanged from C2C (full +- wrap).
  !
  ! CORRECTNESS RESTS ON THE NYQUIST INVARIANT (project #1 design, section 2.2:
  ! "the Nyquist mode is zeroed and kept zeroed"): with the coarse x-Nyquist
  ! (kx=nx/2, index nkx-1) and z-Nyquist (kz=nz/2) zero, this asymmetric map
  ! reproduces the C2C dealiased retained modes to round-off -- VERIFIED
  ! numerically before coding (a nonzero Nyquist would NOT round-trip, which is
  ! why the solver zeroing it is load-bearing here, not incidental).
  !===========================================================================
  subroutine s3d_setup_da_r2c(pl, ny, nx, nz)
    type(s3d_plans), intent(out) :: pl
    integer(ip),     intent(in)  :: ny, nx, nz
    integer(ip) :: nxd, nzd, nkx
    nxd = 3_ip*nx/2_ip;  nzd = 3_ip*nz/2_ip;  nkx = nx/2_ip + 1_ip
    call tf_c2c_plan(pl%cheb_r, 2_ip*(ny-1), nkx*nz)   ! Chebyshev on coarse nkx cols
    call plan_xz_r2c(pl%xz_r2c, ny, nxd, nzd)          ! fused real-x fwd, padded grid
    call plan_xz_c2r(pl%xz_c2r, ny, nxd, nzd)          ! fused inverse, padded grid
  end subroutine s3d_setup_da_r2c

  !-- scatter coarse (ny,nkx,nz) -> padded (ny,nkxd,nzd): x low-pass, z wrap -----
  subroutine pad_xz_r2c(ny, nx, nz, nxd, nzd, src, dst)
    integer(ip), intent(in) :: ny, nx, nz, nxd, nzd
    complex(tp), device     :: src(ny, 0:(nx/2), 0:nz-1)      ! nkx = nx/2+1 cols
    complex(tp), device     :: dst(ny, 0:(nxd/2), 0:nzd-1)    ! nkxd = nxd/2+1 cols
    integer(ip) :: iy, kx, kz, kzd, nxh, nzh, nkxd
    nxh = nx/2;  nzh = nz/2;  nkxd = nxd/2
    !$cuf kernel do(3) <<<*,*>>>
    do kz = 0, nzd-1
       do kx = 0, nkxd
          do iy = 1, ny
             dst(iy,kx,kz) = (0.0_tp, 0.0_tp)
          end do
       end do
    end do
    !$cuf kernel do(3) <<<*,*>>>
    do kz = 0, nz-1
       do kx = 0, nxh                       ! x: low-pass copy 0..nx/2, NO wrap
          do iy = 1, ny
             kzd = merge(kz, nzd-(nz-kz), kz <= nzh)   ! z: full +- wrap (unchanged)
             dst(iy,kx,kzd) = src(iy,kx,kz)
          end do
       end do
    end do
  end subroutine pad_xz_r2c

  !-- gather padded (ny,nkxd,nzd) -> coarse (ny,nkx,nz): x low-pass, z wrap ------
  subroutine trunc_xz_r2c(ny, nxd, nzd, nx, nz, src, dst)
    integer(ip), intent(in) :: ny, nxd, nzd, nx, nz
    complex(tp), device     :: src(ny, 0:(nxd/2), 0:nzd-1)
    complex(tp), device     :: dst(ny, 0:(nx/2), 0:nz-1)
    integer(ip) :: iy, kx, kz, kzd, nxh, nzh
    nxh = nx/2;  nzh = nz/2
    !$cuf kernel do(3) <<<*,*>>>
    do kz = 0, nz-1
       do kx = 0, nxh                       ! x: low-pass copy 0..nx/2, NO wrap
          do iy = 1, ny
             kzd = merge(kz, nzd-(nz-kz), kz <= nzh)   ! z: full +- wrap (unchanged)
             dst(iy,kx,kz) = src(iy,kx,kzd)
          end do
       end do
    end do
  end subroutine trunc_xz_r2c

  !-- spectral coarse (ny,nkx,nz) -> dealiased REAL physical P (ny,nxd,nzd) ------
  ! Wf is a complex (ny,nkxd,nzd) scratch for the padded fine spectrum (consumed
  ! by the C2R). P is the real output.
  subroutine to_physical_da_r2c(pl, ny, nx, nz, nxd, nzd, A, P, Wf)
    type(s3d_plans), intent(in) :: pl
    integer(ip),     intent(in) :: ny, nx, nz, nxd, nzd
    complex(tp),     device     :: A(ny,(nx/2+1),nz), Wf(ny,(nxd/2+1),nzd)
    real(tp),        device     :: P(ny,nxd,nzd)
    integer(ip) :: nkx
    nkx = nx/2_ip + 1_ip
    if (.not. s3d_fd) &                                  ! FD: y = identity
       call cheb_fft_c_inv(pl%cheb_r, ny-1, nkx*nz, A, A) ! y: coeffs -> values
    call pad_xz_r2c(ny, nx, nz, nxd, nzd, A, Wf)         ! scatter to fine (x low-pass)
    call exec_xz_c2r(pl%xz_c2r, Wf, P)                   ! fused C2R -> real physical
  end subroutine to_physical_da_r2c

  !-- dealiased REAL physical P (ny,nxd,nzd) -> spectral coarse A (ny,nkx,nz) ----
  ! Wf is a complex (ny,nkxd,nzd) scratch for the fine spectrum from the R2C.
  subroutine to_spectral_da_r2c(pl, ny, nx, nz, nxd, nzd, P, Wf, A)
    type(s3d_plans), intent(in) :: pl
    integer(ip),     intent(in) :: ny, nx, nz, nxd, nzd
    real(tp),        device     :: P(ny,nxd,nzd)
    complex(tp),     device     :: Wf(ny,(nxd/2+1),nzd), A(ny,(nx/2+1),nz)
    integer(ip) :: nkx
    nkx = nx/2_ip + 1_ip
    call exec_xz_r2c(pl%xz_r2c, P, Wf)                   ! fused R2C -> fine spectrum
    call trunc_xz_r2c(ny, nxd, nzd, nx, nz, Wf, A)       ! gather to coarse (x low-pass)
    if (.not. s3d_fd) &                                  ! FD: y = identity
       call cheb_fft_c_fwd(pl%cheb_r, ny-1, nkx*nz, A, A) ! y: values -> coeffs
  end subroutine to_spectral_da_r2c

end module spectral3d_mod
