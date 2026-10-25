!===============================================================================
! par_spectral3d_mod.F90  --  multi-GPU DISTRIBUTED dealiased real-x (x,z)
! transform on the 1-D slab decomposition.  The distributed counterpart of
! spectral3d_mod's to_physical_da_r2c / to_spectral_da_r2c, built from pieces
! that are each ALREADY validated:
!   * the 1-D x plans (fft3d_mod)                 -- bit-exact vs the fused plan
!   * the trailing-axis z plan (fft3d_mod)        -- used unchanged
!   * the NCCL A<->B transpose (par_transpose)    -- bit-exact, P-independent
!   * the Chebyshev-y transform (cheb_fft_mod)    -- y is always on-rank
! and an ORDERING that the numpy model (src/models/m2_distributed_transform_model.py)
! proved reproduces the monolithic transform bit-for-bit and is P-independent.
!
! LAYOUTS (y is ALWAYS local -- Chebyshev + the wall-normal solve are per-(kx,kz)
! column / never split; it rides as the leading dim of every array):
!   Layout B (kx-slab):  (ny, b_nkx_loc, *)   this rank's kx block, full z. Home
!                        of the spectral field; where cheb and the z-FFT run.
!   Layout A (z-slab):   (ny, *, ad_nz_loc)   full kx, this rank's dealiased-z
!                        block. Where the x-FFT runs; home of the physical field.
!
! INVERSE  (spectral coarse, Layout B  ->  physical real, Layout A):
!   cheb_inv(y) ; pad-z (nz->nzd) ; ifft-z        [Layout B, coarse kx]
!   --transpose B->A (carries COARSE-kx nkx, FINE-z nzd)-->
!   pad-x (nkx->nkxd) ; irfft-x                    [Layout A, fine-z slab]
! FORWARD  (physical real, Layout A  ->  spectral coarse, Layout B): the reverse,
!   rfft-x ; trunc-x  --transpose A->B--  fft-z ; trunc-z ; cheb_fwd.
! ONE transpose per direction. The dealias SPLITS: x low-pass runs where kx is
! whole (Layout A), z +-wrap runs where kz is whole (Layout B); neither can be
! done in the other layout (proved in the model). Normalisation is unchanged
! from the single-GPU path (cuFFT is unnormalised; the 1/(nxd*nzd) lives in the
! nonlinear product, not here), so a forward(inverse(.)) round-trip = nxd*nzd.
!
! PRECONDITION: P <= min(nkx, nzd) so every rank owns >= 1 kx-column AND >= 1
! z-plane (b_nkx_loc>=1, ad_nz_loc>=1). This is the slab ceiling; every target
! configuration sits far below it. Empty-rank handling (P>min) is
! intentionally NOT added here.
!
! Single-GPU suite is untouched and MPI/NCCL/transpose-independent (this module
! lives in parallel/ and is built only by parallel/build_run_spectral_mp.sh).
!===============================================================================
module par_spectral3d_mod
  use kinds,             only: tp, ip
  use cudafor
  use transforms_mod,    only: tf_c2c_plan, tf_destroy
  use cheb_fft_mod,      only: cheb_fft_c_fwd, cheb_fft_c_inv
  use fft3d_mod,         only: plan_axis_c2c, exec_axis_c2c, &
                               plan_x_r2c, plan_x_c2r, exec_x_r2c, exec_x_c2r, &
                               CUFFT_FORWARD, CUFFT_INVERSE
  use decomp_mod,        only: decomp_t
  use par_transpose_mod, only: par_transpose_A_to_B, par_transpose_B_to_A, &
                               ptp_begin, ptp_pack_field, ptp_a2a_field, &
                               ptp_unpack_field, ptp_end, PTP_A2B, PTP_B2A
  implicit none
  private
  public :: par_s3d_plans, par_s3d_setup_da_r2c, par_s3d_destroy_da_r2c
  public :: par_to_physical_da_r2c, par_to_spectral_da_r2c
  public :: par_to_physical_nf, par_to_spectral_nf      ! nfield-batched (1 transpose)
  public :: par_to_physical_r2c                         ! snapshots: clean nx*nz inverse
  public :: par_s3d_set_overlap                         ! 0 = monolithic, 1 = per-field pipelined A2A
  public :: par_s3d_set_ybasis_fd                       ! FD identity legs
  public :: par_to_physical_nf_begin, par_to_physical_nf_finish
  public :: par_to_spectral_nf_begin, par_to_spectral_nf_finish

  ! comm/compute overlap knob (deck a2a_overlap, default 0 = monolithic path).
  ! Mode 1 pipelines the nf-batched legs PER FIELD: field f's all-to-all (comm
  ! stream) runs while field f+1's local FFT work (compute stream) proceeds, and
  ! the unpack-side tail overlaps the remaining flights. Every per-field FFT call
  ! (plans, order, arguments) is IDENTICAL to the monolithic path and the
  ! transpose is pure data movement, so mode 1 is BIT-EXACT vs mode 0 --
  ! gated in test_a2a_overlap_mp, not just claimed.
  integer(ip), save :: s3d_overlap = 0
  logical, save :: ps3d_fd = .false.   ! FD identity legs

  type :: par_s3d_plans
     integer :: cheb_b = 0     ! Chebyshev over y, batched b_nkx_loc*nz cols (Layout B)
     integer :: z_c2c  = 0     ! 1-D C2C, trailing z, (ny, b_nkx_loc, nzd)
     integer :: x_r2c  = 0     ! 1-D R2C, middle x,   (ny, nxd, ad_nz_loc)
     integer :: x_c2r  = 0     ! 1-D C2R, middle x
     integer :: z_c2c_co = 0   ! [snap] 1-D C2C, trailing z, COARSE (ny, b_nkx_loc, nz)
     integer :: x_c2r_co = 0   ! [snap] 1-D C2R, middle x, CLEAN nx  (ny, nx, a_nz_loc)
  end type par_s3d_plans

contains

  !-----------------------------------------------------------------------------
  ! Plan setup. Sizes the four cuFFT plans for THIS rank's local extents.
  !-----------------------------------------------------------------------------
  subroutine par_s3d_setup_da_r2c(pl, d, ny, nx, nz)
    type(par_s3d_plans), intent(out) :: pl
    type(decomp_t),      intent(in)  :: d
    integer(ip),         intent(in)  :: ny, nx, nz
    integer(ip) :: nxd, nzd
    nxd = 3_ip*nx/2_ip;  nzd = 3_ip*nz/2_ip
    call tf_c2c_plan(pl%cheb_b, 2_ip*(ny-1_ip), d%b_nkx_loc*nz)   ! y-cheb on coarse cols
    call plan_axis_c2c(pl%z_c2c, 3_ip, ny, d%b_nkx_loc, nzd)      ! trailing-axis z, fine
    call plan_x_r2c(pl%x_r2c, ny, nxd, d%ad_nz_loc)               ! per-plane R2C, fine x
    call plan_x_c2r(pl%x_c2r, ny, nxd, d%ad_nz_loc)               ! per-plane C2R, fine x
    ! snapshot: clean nx*nz inverse plans (coarse z, un-dealiased x)
    call plan_axis_c2c(pl%z_c2c_co, 3_ip, ny, d%b_nkx_loc, nz)    ! trailing-axis z, COARSE
    call plan_x_c2r   (pl%x_c2r_co, ny, nx, d%a_nz_loc)           ! per-plane C2R, CLEAN x
  end subroutine par_s3d_setup_da_r2c

  subroutine par_s3d_destroy_da_r2c(pl)
    type(par_s3d_plans), intent(inout) :: pl
    call tf_destroy(pl%cheb_b);  call tf_destroy(pl%z_c2c)
    call tf_destroy(pl%x_r2c);   call tf_destroy(pl%x_c2r)
    call tf_destroy(pl%z_c2c_co); call tf_destroy(pl%x_c2r_co)
  end subroutine par_s3d_destroy_da_r2c

  !=============================================================================
  ! Split dealias kernels.  x = LOW-PASS copy (kx=0..nx/2, no wrap), run where kx
  ! is whole; z = full +-Nyquist WRAP, run where z is whole.  The OTHER Fourier
  ! axis is a passive local batch.  Index math identical to spectral3d_mod's
  ! pad_xz_r2c / trunc_xz_r2c (just one direction each).  CUF kernels touch only
  ! plain device arrays (no decomp_t members) -> no NVFORTRAN-0155.
  !=============================================================================
  subroutine pad_z_loc(ny, nkxl, nz, nzd, src, dst)   ! z coarse->fine, kx passive
    integer(ip), intent(in) :: ny, nkxl, nz, nzd
    complex(tp), device     :: src(ny, nkxl, 0:nz-1)
    complex(tp), device     :: dst(ny, nkxl, 0:nzd-1)
    integer(ip) :: iy, j, kz, kzd, nzh
    nzh = nz/2
    !$cuf kernel do(3) <<<*,*>>>
    do kz = 0, nzd-1
       do j = 1, nkxl
          do iy = 1, ny
             dst(iy,j,kz) = (0.0_tp, 0.0_tp)
          end do
       end do
    end do
    !$cuf kernel do(3) <<<*,*>>>
    do kz = 0, nz-1
       do j = 1, nkxl
          do iy = 1, ny
             kzd = merge(kz, nzd-(nz-kz), kz <= nzh)
             dst(iy,j,kzd) = src(iy,j,kz)
          end do
       end do
    end do
  end subroutine pad_z_loc

  subroutine trunc_z_loc(ny, nkxl, nzd, nz, src, dst)  ! z fine->coarse, kx passive
    integer(ip), intent(in) :: ny, nkxl, nzd, nz
    complex(tp), device     :: src(ny, nkxl, 0:nzd-1)
    complex(tp), device     :: dst(ny, nkxl, 0:nz-1)
    integer(ip) :: iy, j, kz, kzd, nzh
    nzh = nz/2
    !$cuf kernel do(3) <<<*,*>>>
    do kz = 0, nz-1
       do j = 1, nkxl
          do iy = 1, ny
             kzd = merge(kz, nzd-(nz-kz), kz <= nzh)
             dst(iy,j,kz) = src(iy,j,kzd)
          end do
       end do
    end do
  end subroutine trunc_z_loc

  subroutine pad_x_loc(ny, nx, nxd, nzl, src, dst)    ! x coarse->fine, z passive
    integer(ip), intent(in) :: ny, nx, nxd, nzl
    complex(tp), device     :: src(ny, 0:(nx/2),  nzl)
    complex(tp), device     :: dst(ny, 0:(nxd/2), nzl)
    integer(ip) :: iy, kx, kzl, nxh, nkxd
    nxh = nx/2;  nkxd = nxd/2
    !$cuf kernel do(3) <<<*,*>>>
    do kzl = 1, nzl
       do kx = 0, nkxd
          do iy = 1, ny
             dst(iy,kx,kzl) = (0.0_tp, 0.0_tp)
          end do
       end do
    end do
    !$cuf kernel do(3) <<<*,*>>>
    do kzl = 1, nzl
       do kx = 0, nxh
          do iy = 1, ny
             dst(iy,kx,kzl) = src(iy,kx,kzl)
          end do
       end do
    end do
  end subroutine pad_x_loc

  subroutine trunc_x_loc(ny, nxd, nx, nzl, src, dst)  ! x fine->coarse, z passive
    integer(ip), intent(in) :: ny, nxd, nx, nzl
    complex(tp), device     :: src(ny, 0:(nxd/2), nzl)
    complex(tp), device     :: dst(ny, 0:(nx/2),  nzl)
    integer(ip) :: iy, kx, kzl, nxh
    nxh = nx/2
    !$cuf kernel do(3) <<<*,*>>>
    do kzl = 1, nzl
       do kx = 0, nxh
          do iy = 1, ny
             dst(iy,kx,kzl) = src(iy,kx,kzl)
          end do
       end do
    end do
  end subroutine trunc_x_loc

  !=============================================================================
  ! INVERSE: spectral coarse A (Layout B, ny,b_nkx_loc,nz) -> real physical P
  ! (Layout A, ny,nxd,ad_nz_loc).  A is consumed (cheb runs in place).
  ! Scratch (caller-allocated): Bz fine-z Layout B, Az coarse-kx fine-z Layout A
  ! (both carry the transpose's trailing nfield=1 dim), Af fine-kx Layout A.
  !=============================================================================
  subroutine par_to_physical_da_r2c(pl, d, ny, nx, nz, nxd, nzd, &
                                     b_nkx_loc, ad_nz_loc, A, Bz, Az, Af, P)
    type(par_s3d_plans), intent(in) :: pl
    type(decomp_t),      intent(in) :: d
    integer(ip),         intent(in) :: ny, nx, nz, nxd, nzd, b_nkx_loc, ad_nz_loc
    complex(tp), device :: A (ny, b_nkx_loc, nz)                 ! in: coarse spectral
    complex(tp), device :: Bz(ny, b_nkx_loc, nzd, 1)             ! scratch
    complex(tp), device :: Az(ny, (nx/2+1),  ad_nz_loc, 1)       ! scratch
    complex(tp), device :: Af(ny, (nxd/2+1), ad_nz_loc)          ! scratch
    real(tp),    device :: P (ny, nxd,       ad_nz_loc)          ! out: real physical
    if (.not. ps3d_fd) &   ! FD: y = identity
       call cheb_fft_c_inv(pl%cheb_b, ny-1_ip, b_nkx_loc*nz, A, A)              ! y: coeffs->values
    call pad_z_loc(ny, b_nkx_loc, nz, nzd, A, Bz(:,:,:,1))                  ! z: coarse->fine
    call exec_axis_c2c(pl%z_c2c, 3_ip, ny, b_nkx_loc, nzd, Bz(:,:,:,1), Bz(:,:,:,1), CUFFT_INVERSE)
    call par_transpose_B_to_A(d, ny, d%nkx, nzd, ad_nz_loc, b_nkx_loc, 1_ip, Bz, Az)
    call pad_x_loc(ny, nx, nxd, ad_nz_loc, Az(:,:,:,1), Af)                 ! x: coarse->fine
    call exec_x_c2r(pl%x_c2r, ny, nxd, ad_nz_loc, Af, P)                     ! x: spectral->real
  end subroutine par_to_physical_da_r2c

  !=============================================================================
  ! CLEAN INVERSE (snapshots): spectral coarse A (Layout B, ny,b_nkx_loc,nz) ->
  ! real physical P on the UN-DEALIASED nx*nz grid (Layout A, ny,nx,a_nz_loc), for
  ! field OUTPUT. Identical pipeline to par_to_physical_da_r2c but WITHOUT the 3/2
  ! pad steps and with nz/nx-sized FFTs -> the canonical-resolution field, 2.25x
  ! smaller. Matches the single-GPU oracle to_physical_r2c per z-slab (P-independent;
  ! the transpose is a bit-exact permutation). A is consumed (cheb runs in place).
  ! a_nz_loc/a_z0 are the COARSE z-slab (decomp_t %a_*), NOT the dealiased ad_* one.
  ! Scratch (caller-allocated): Bz coarse-z Layout B, Az coarse Layout A (both carry
  ! the transpose's trailing nfield=1 dim).
  !=============================================================================
  subroutine par_to_physical_r2c(pl, d, ny, nx, nz, b_nkx_loc, a_nz_loc, A, Bz, Az, P)
    type(par_s3d_plans), intent(in) :: pl
    type(decomp_t),      intent(in) :: d
    integer(ip),         intent(in) :: ny, nx, nz, b_nkx_loc, a_nz_loc
    complex(tp), device :: A (ny, b_nkx_loc, nz)                 ! in: coarse spectral (consumed)
    complex(tp), device :: Bz(ny, b_nkx_loc, nz, 1)             ! scratch: Layout B, coarse z
    complex(tp), device :: Az(ny, (nx/2+1),  a_nz_loc, 1)       ! scratch: Layout A, coarse
    real(tp),    device :: P (ny, nx,        a_nz_loc)          ! out: real physical, clean grid
    if (.not. ps3d_fd) &   ! FD: y = identity
       call cheb_fft_c_inv(pl%cheb_b, ny-1_ip, b_nkx_loc*nz, A, A)              ! y: coeffs->values
    Bz(:,:,:,1) = A                                                         ! z stays coarse (no pad)
    call exec_axis_c2c(pl%z_c2c_co, 3_ip, ny, b_nkx_loc, nz, Bz(:,:,:,1), Bz(:,:,:,1), CUFFT_INVERSE)
    call par_transpose_B_to_A(d, ny, d%nkx, nz, a_nz_loc, b_nkx_loc, 1_ip, Bz, Az)  ! B->A (coarse z)
    call exec_x_c2r(pl%x_c2r_co, ny, nx, a_nz_loc, Az(:,:,:,1), P)           ! x: spectral->real (nx)
  end subroutine par_to_physical_r2c

  !=============================================================================
  ! FORWARD: real physical P (Layout A) -> spectral coarse A (Layout B).
  ! Scratch (caller-allocated): Af fine-kx Layout A, Az coarse-kx fine-z Layout A,
  ! Bz fine-z Layout B (Az/Bz carry the transpose's trailing nfield=1 dim).
  !=============================================================================
  subroutine par_to_spectral_da_r2c(pl, d, ny, nx, nz, nxd, nzd, &
                                     b_nkx_loc, ad_nz_loc, P, Af, Az, Bz, A)
    type(par_s3d_plans), intent(in) :: pl
    type(decomp_t),      intent(in) :: d
    integer(ip),         intent(in) :: ny, nx, nz, nxd, nzd, b_nkx_loc, ad_nz_loc
    real(tp),    device :: P (ny, nxd,       ad_nz_loc)          ! in: real physical
    complex(tp), device :: Af(ny, (nxd/2+1), ad_nz_loc)          ! scratch
    complex(tp), device :: Az(ny, (nx/2+1),  ad_nz_loc, 1)       ! scratch
    complex(tp), device :: Bz(ny, b_nkx_loc, nzd, 1)             ! scratch
    complex(tp), device :: A (ny, b_nkx_loc, nz)                 ! out: coarse spectral
    call exec_x_r2c(pl%x_r2c, ny, nxd, ad_nz_loc, P, Af)                     ! x: real->spectral
    call trunc_x_loc(ny, nxd, nx, ad_nz_loc, Af, Az(:,:,:,1))               ! x: fine->coarse
    call par_transpose_A_to_B(d, ny, d%nkx, nzd, ad_nz_loc, b_nkx_loc, 1_ip, Az, Bz)
    call exec_axis_c2c(pl%z_c2c, 3_ip, ny, b_nkx_loc, nzd, Bz(:,:,:,1), Bz(:,:,:,1), CUFFT_FORWARD)
    call trunc_z_loc(ny, b_nkx_loc, nzd, nz, Bz(:,:,:,1), A)                ! z: fine->coarse
    if (.not. ps3d_fd) &   ! FD: y = identity
       call cheb_fft_c_fwd(pl%cheb_b, ny-1_ip, b_nkx_loc*nz, A, A)              ! y: values->coeffs
  end subroutine par_to_spectral_da_r2c

  !=============================================================================
  ! BATCHED INVERSE: nf coarse spectral fields A(...,nf) -> nf real physical
  ! P(...,nf), with a SINGLE nfield=nf transpose instead of nf separate ones. The
  ! per-field y-cheb / z-pad / z-ifft and x-pad / x-irfft are LOCAL (no comm), so
  ! they loop over the field index reusing the per-field plans; only the
  ! all-to-all (par_transpose) is batched. Field-for-field identical to calling
  ! par_to_physical_da_r2c nf times (the isolation test asserts exactly that).
  ! A is consumed (cheb runs in place). Scratch Bz/Az/Af carry the nf trailing dim.
  !=============================================================================
  ! The nf transforms SPLIT at the flight boundary: _begin
  ! launches everything up to and including the all-to-all (per-field flights
  ! under overlap; the monolithic stream-ordered transpose otherwise), _finish
  ! consumes the landings. The wrapper below (begin+finish back-to-back) runs
  ! the identical statement sequence, so every caller/test is bit-identical.
  ! The stepper issues its hoisted
  ! L-operator kernels BETWEEN the two halves: they fill par_stream while the
  ! flights run on par_comm_stream.
  subroutine par_to_physical_nf(pl, d, ny, nx, nz, nxd, nzd, &
                                b_nkx_loc, ad_nz_loc, nf, A, Bz, Az, Af, P)
    type(par_s3d_plans), intent(in) :: pl
    type(decomp_t),      intent(in) :: d
    integer(ip),         intent(in) :: ny, nx, nz, nxd, nzd, b_nkx_loc, ad_nz_loc, nf
    complex(tp), device :: A (ny, b_nkx_loc, nz,  nf)
    complex(tp), device :: Bz(ny, b_nkx_loc, nzd, nf)
    complex(tp), device :: Az(ny, (nx/2+1),  ad_nz_loc, nf)
    complex(tp), device :: Af(ny, (nxd/2+1), ad_nz_loc)
    real(tp),    device :: P (ny, nxd,       ad_nz_loc, nf)
    call par_to_physical_nf_begin(pl, d, ny, nx, nz, nxd, nzd, &
                                  b_nkx_loc, ad_nz_loc, nf, A, Bz, Az)
    call par_to_physical_nf_finish(pl, d, ny, nx, nz, nxd, nzd, &
                                   b_nkx_loc, ad_nz_loc, nf, Az, Af, P)
  end subroutine par_to_physical_nf

  subroutine par_to_physical_nf_begin(pl, d, ny, nx, nz, nxd, nzd, &
                                      b_nkx_loc, ad_nz_loc, nf, A, Bz, Az)
    type(par_s3d_plans), intent(in) :: pl
    type(decomp_t),      intent(in) :: d
    integer(ip),         intent(in) :: ny, nx, nz, nxd, nzd, b_nkx_loc, ad_nz_loc, nf
    complex(tp), device :: A (ny, b_nkx_loc, nz,  nf)           ! in: nf coarse spectral (consumed)
    complex(tp), device :: Bz(ny, b_nkx_loc, nzd, nf)           ! scratch
    complex(tp), device :: Az(ny, (nx/2+1),  ad_nz_loc, nf)     ! scratch
    integer(ip) :: f
    if (s3d_overlap > 0 .and. d%nprocs > 1 .and. nf > 1) then
       ! Pipelined: identical per-field FFT calls; only the all-to-all is
       ! chunked per field and moved to the comm stream (bit-exact vs below).
       ! No pack call: the B2A send side is the identity, so the A2A sends
       ! DIRECTLY from Bz(:,:,:,f) (ptp_a2a_field records the pack-position
       ! event itself -- the same stream position a pack call would mark).
       call ptp_begin(PTP_B2A, d, ny, d%nkx, nzd, ad_nz_loc, b_nkx_loc, nf)
       do f = 1, nf
          if (.not. ps3d_fd) &   ! FD: y = identity
             call cheb_fft_c_inv(pl%cheb_b, ny-1_ip, b_nkx_loc*nz, A(:,:,:,f), A(:,:,:,f))
          call pad_z_loc(ny, b_nkx_loc, nz, nzd, A(:,:,:,f), Bz(:,:,:,f))
          call exec_axis_c2c(pl%z_c2c, 3_ip, ny, b_nkx_loc, nzd, Bz(:,:,:,f), Bz(:,:,:,f), CUFFT_INVERSE)
          call ptp_a2a_field(f, Bz(:,:,:,f))                    ! field f flies while f+1 computes
       end do
       return
    end if
    do f = 1, nf                                                ! LOCAL: y-cheb, z coarse->fine, z-ifft
       if (.not. ps3d_fd) &   ! FD: y = identity
          call cheb_fft_c_inv(pl%cheb_b, ny-1_ip, b_nkx_loc*nz, A(:,:,:,f), A(:,:,:,f))
       call pad_z_loc(ny, b_nkx_loc, nz, nzd, A(:,:,:,f), Bz(:,:,:,f))
       call exec_axis_c2c(pl%z_c2c, 3_ip, ny, b_nkx_loc, nzd, Bz(:,:,:,f), Bz(:,:,:,f), CUFFT_INVERSE)
    end do
    call par_transpose_B_to_A(d, ny, d%nkx, nzd, ad_nz_loc, b_nkx_loc, nf, Bz, Az)   ! ONE all-to-all
  end subroutine par_to_physical_nf_begin

  subroutine par_to_physical_nf_finish(pl, d, ny, nx, nz, nxd, nzd, &
                                       b_nkx_loc, ad_nz_loc, nf, Az, Af, P)
    type(par_s3d_plans), intent(in) :: pl
    type(decomp_t),      intent(in) :: d
    integer(ip),         intent(in) :: ny, nx, nz, nxd, nzd, b_nkx_loc, ad_nz_loc, nf
    complex(tp), device :: Az(ny, (nx/2+1),  ad_nz_loc, nf)     ! in (landed)
    complex(tp), device :: Af(ny, (nxd/2+1), ad_nz_loc)         ! scratch
    real(tp),    device :: P (ny, nxd,       ad_nz_loc, nf)     ! out
    integer(ip) :: f
    if (s3d_overlap > 0 .and. d%nprocs > 1 .and. nf > 1) then
       do f = 1, nf
          call ptp_unpack_field(f, Az(:,:,:,f))                 ! waits only on field f's A2A
          call pad_x_loc(ny, nx, nxd, ad_nz_loc, Az(:,:,:,f), Af)
          call exec_x_c2r(pl%x_c2r, ny, nxd, ad_nz_loc, Af, P(:,:,:,f))
       end do
       call ptp_end()
       return
    end if
    do f = 1, nf                                                ! LOCAL: x coarse->fine, x-irfft
       call pad_x_loc(ny, nx, nxd, ad_nz_loc, Az(:,:,:,f), Af)
       call exec_x_c2r(pl%x_c2r, ny, nxd, ad_nz_loc, Af, P(:,:,:,f))
    end do
  end subroutine par_to_physical_nf_finish

  !=============================================================================
  ! BATCHED FORWARD: nf real physical fields P(...,nf) -> nf coarse spectral
  ! A(...,nf), with a SINGLE nfield=nf transpose. Reverse of par_to_physical_nf;
  ! field-for-field identical to nf calls of par_to_spectral_da_r2c.
  !=============================================================================
  subroutine par_to_spectral_nf(pl, d, ny, nx, nz, nxd, nzd, &
                                b_nkx_loc, ad_nz_loc, nf, P, Af, Az, Bz, A)
    type(par_s3d_plans), intent(in) :: pl
    type(decomp_t),      intent(in) :: d
    integer(ip),         intent(in) :: ny, nx, nz, nxd, nzd, b_nkx_loc, ad_nz_loc, nf
    real(tp),    device :: P (ny, nxd,       ad_nz_loc, nf)
    complex(tp), device :: Af(ny, (nxd/2+1), ad_nz_loc)
    complex(tp), device :: Az(ny, (nx/2+1),  ad_nz_loc, nf)
    complex(tp), device :: Bz(ny, b_nkx_loc, nzd, nf)
    complex(tp), device :: A (ny, b_nkx_loc, nz,  nf)
    call par_to_spectral_nf_begin(pl, d, ny, nx, nz, nxd, nzd, &
                                  b_nkx_loc, ad_nz_loc, nf, P, Af, Az, Bz)
    call par_to_spectral_nf_finish(pl, d, ny, nx, nz, nxd, nzd, &
                                   b_nkx_loc, ad_nz_loc, nf, Bz, A)
  end subroutine par_to_spectral_nf

  subroutine par_to_spectral_nf_begin(pl, d, ny, nx, nz, nxd, nzd, &
                                      b_nkx_loc, ad_nz_loc, nf, P, Af, Az, Bz)
    type(par_s3d_plans), intent(in) :: pl
    type(decomp_t),      intent(in) :: d
    integer(ip),         intent(in) :: ny, nx, nz, nxd, nzd, b_nkx_loc, ad_nz_loc, nf
    real(tp),    device :: P (ny, nxd,       ad_nz_loc, nf)     ! in: nf real physical
    complex(tp), device :: Af(ny, (nxd/2+1), ad_nz_loc)         ! scratch (ONE slice, reused per field)
    complex(tp), device :: Az(ny, (nx/2+1),  ad_nz_loc, nf)     ! scratch
    complex(tp), device :: Bz(ny, b_nkx_loc, nzd, nf)           ! scratch
    integer(ip) :: f
    if (s3d_overlap > 0 .and. d%nprocs > 1 .and. nf > 1) then
       ! Pipelined forward (mirror of the inverse; bit-exact vs below).
       ! The A2B recv side is the identity: the A2A receives DIRECTLY into
       ! Bz(:,:,:,f) (the finish loop's ptp_unpack_field is wait-only).
       call ptp_begin(PTP_A2B, d, ny, d%nkx, nzd, ad_nz_loc, b_nkx_loc, nf)
       do f = 1, nf
          call exec_x_r2c(pl%x_r2c, ny, nxd, ad_nz_loc, P(:,:,:,f), Af)
          call trunc_x_loc(ny, nxd, nx, ad_nz_loc, Af, Az(:,:,:,f))
          call ptp_pack_field(f, Az(:,:,:,f))
          call ptp_a2a_field(f, Bz(:,:,:,f))
       end do
       return
    end if
    do f = 1, nf                                                ! LOCAL: x-rfft, x fine->coarse
       call exec_x_r2c(pl%x_r2c, ny, nxd, ad_nz_loc, P(:,:,:,f), Af)
       call trunc_x_loc(ny, nxd, nx, ad_nz_loc, Af, Az(:,:,:,f))
    end do
    call par_transpose_A_to_B(d, ny, d%nkx, nzd, ad_nz_loc, b_nkx_loc, nf, Az, Bz)   ! ONE all-to-all
  end subroutine par_to_spectral_nf_begin

  subroutine par_to_spectral_nf_finish(pl, d, ny, nx, nz, nxd, nzd, &
                                       b_nkx_loc, ad_nz_loc, nf, Bz, A)
    type(par_s3d_plans), intent(in) :: pl
    type(decomp_t),      intent(in) :: d
    integer(ip),         intent(in) :: ny, nx, nz, nxd, nzd, b_nkx_loc, ad_nz_loc, nf
    complex(tp), device :: Bz(ny, b_nkx_loc, nzd, nf)           ! in (landed)
    complex(tp), device :: A (ny, b_nkx_loc, nz,  nf)           ! out
    integer(ip) :: f
    if (s3d_overlap > 0 .and. d%nprocs > 1 .and. nf > 1) then
       do f = 1, nf
          call ptp_unpack_field(f, Bz(:,:,:,f))
          call exec_axis_c2c(pl%z_c2c, 3_ip, ny, b_nkx_loc, nzd, Bz(:,:,:,f), Bz(:,:,:,f), CUFFT_FORWARD)
          call trunc_z_loc(ny, b_nkx_loc, nzd, nz, Bz(:,:,:,f), A(:,:,:,f))
          if (.not. ps3d_fd) &   ! FD: y = identity
             call cheb_fft_c_fwd(pl%cheb_b, ny-1_ip, b_nkx_loc*nz, A(:,:,:,f), A(:,:,:,f))
       end do
       call ptp_end()
       return
    end if
    do f = 1, nf                                                ! LOCAL: z-fft, z fine->coarse, y-cheb
       call exec_axis_c2c(pl%z_c2c, 3_ip, ny, b_nkx_loc, nzd, Bz(:,:,:,f), Bz(:,:,:,f), CUFFT_FORWARD)
       call trunc_z_loc(ny, b_nkx_loc, nzd, nz, Bz(:,:,:,f), A(:,:,:,f))
       if (.not. ps3d_fd) &   ! FD: y = identity
          call cheb_fft_c_fwd(pl%cheb_b, ny-1_ip, b_nkx_loc*nz, A(:,:,:,f), A(:,:,:,f))
    end do
  end subroutine par_to_spectral_nf_finish

  ! Driver opt-in for the pipelined overlap (deck key a2a_overlap).
  subroutine par_s3d_set_overlap(mode)
    integer(ip), intent(in) :: mode
    s3d_overlap = mode
  end subroutine par_s3d_set_overlap

  ! FD-y identity legs (mirror of spectral3d_mod s3d_set_ybasis_fd):
  ! in FD mode the y-state IS point values, the cheb legs are skipped. Set once
  ! at init by the FD wiring; flag down = the identical statements (bit-identity).
  subroutine par_s3d_set_ybasis_fd(fd)
    logical, intent(in) :: fd
    ps3d_fd = fd
  end subroutine par_s3d_set_ybasis_fd

end module par_spectral3d_mod
