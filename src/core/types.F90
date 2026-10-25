!===============================================================================
! types.F90  --  Core data aggregates for the GPU channel-flow DNS solver
!
! NOTE: .F90 (capital F) so the compiler runs the preprocessor (for DEVICE).
!
! DESIGN PHILOSOPHY (agreed)
! --------------------------
!   * Derived types are DUMB RECORDS that aggregate cohesive state. Callers
!     pass a few typed aggregates, not 80-long positional lists; the compiler
!     checks every field by NAME.
!   * Behaviour lives in concern modules that `use types`. Data-only here =>
!     no circular USE dependencies.
!   * Device arrays are `device, allocatable` components. Kernels and cuFFT
!     receive the BARE component (fields%vh), never the aggregate.
!
! PRECISION POLICY (THIS FILE IS AUTHORITATIVE)
! ---------------------------------------------------------
!   The whole source is KIND-GENERIC: no `double precision`, no `real*8`, no
!   `dble`/`dcos`/`dsqrt`, every real literal tagged (`0.5_wp`). So EVERYTHING
!   can compile and run in FP32 -- nothing hard-depends on FP64.
!
!   On top of that we apply a mixed-precision *policy* via THREE kinds -- there
!   are three genuine accuracy tiers, and a two-kind model conflates the
!   transform precision and the field precision, the two you most want to
!   set independently:
!     tp  = TRANSFORM / COMMUNICATION precision. The nl_workspace physical+
!           spectral scratch, the cuFFT plans, and (post-parallelization) the
!           transposed/communicated data. FLIP to real32 (make TP=sp) for the
!           FFT/bandwidth win -- this is most of the payoff.
!     wp  = FIELD / EXPLICIT-ARITHMETIC precision. Persistent fields (eta,v,u0,
!           w0), the RK history, RHS-assembly scratch, and apply_L. FLIP to
!           real32 (make WP=sp) ONLY for memory-bound production,
!           accepting some high-k statistical degradation.
!     dp  = ACCURACY-CRITICAL precision, ALWAYS real64 -- NO build flag. The
!           implicit solves, the solve scratch+factorization (vm_all/vp_all/
!           Cinv_all), the wall-normal Chebyshev operators, the mean (0,0)
!           solve, and time/stat/flux accumulation. Ill-conditioned (~ny^4 ->
!           ~1e12 at ny~1200) or accumulating; FP32 there is silent garbage.
!   Two cast seams: tp<->wp at the nonlinear boundary, wp<->dp at the solve
!   boundary. Rule of thumb: big well-conditioned arrays -> tp/wp; small,
!   ill-conditioned, or accumulating quantities -> dp.
!===============================================================================

#ifdef _CUDA
#  define DEVICE , device
#else
#  define DEVICE
#endif

!-------------------------------------------------------------------------------
module kinds
  use iso_fortran_env, only: real32, real64, int32, int64
  implicit none
  public
  ! --- TRANSFORM / COMMUNICATION precision -------------------------------------
  !     transform/comm scratch, cuFFT plans, (later) transposed data.
  !     `make TP=sp`  ->  -DCLAUSON_TP_SP  ->  tp = real32
#ifdef CLAUSON_TP_SP
  integer, parameter :: tp = real32
#else
  integer, parameter :: tp = real64
#endif
  ! --- FIELD / EXPLICIT-ARITHMETIC precision -----------------------------------
  !     persistent fields (eta,v,u0,w0), RK history, RHS-assembly scratch, apply_L.
  !     `make WP=sp`  ->  -DCLAUSON_WP_SP  ->  wp = real32
#ifdef CLAUSON_WP_SP
  integer, parameter :: wp = real32
#else
  integer, parameter :: wp = real64
#endif
  ! --- ACCURACY-CRITICAL precision: ALWAYS real64, NO build flag ---------------
  !     implicit solves, factorizations (vm_all/vp_all/Cinv_all), wall-normal
  !     Chebyshev ops, mean (0,0) solve, time/stat/flux accumulation.
  integer, parameter :: dp = real64
  ! --- explicit single precision (genuine real32 regardless of the flags) ------
  integer, parameter :: sp = real32
  integer, parameter :: ip = int32
end module kinds

!-------------------------------------------------------------------------------
module types
  use kinds
  implicit none
  public

  integer(ip), parameter :: DRIVE_CONST_FLUX = 1   ! hold bulk mass flux fixed
  integer(ip), parameter :: DRIVE_CONST_DPDX = 2   ! hold mean dp/dx fixed

  !=============================================================================
  ! params_t  --  run control. All scalars in dp (free, and many feed the
  !               FP64 operator build / flux constraint). Resolution is in grid.
  !=============================================================================
  type :: params_t
     ! physical
     real(dp)    :: re_tau   = 180.0_dp
     real(dp)    :: re_bulk  = 0.0_dp
     real(dp)    :: nu       = 0.0_dp                  ! = 1/Re internally
     ! geometry (half-height = 1; walls at y = -1, +1)
     real(dp)    :: lx       = 4.0_dp*acos(-1.0_dp)
     real(dp)    :: lz       = 2.0_dp*acos(-1.0_dp)
     ! time advancement
     real(dp)    :: dt       = 0.0_dp
     real(dp)    :: dt_max   = 1.0e30_dp
     real(dp)    :: cfl_max  = 0.5_dp
     logical     :: variable_dt = .true.
     real(dp)    :: t_end    = 0.0_dp
     integer(ip) :: max_steps = huge(1_ip)
     integer(ip) :: nst      = 3                       ! RK stages (SMR-RK3)
     ! forcing
     integer(ip) :: drive_mode  = DRIVE_CONST_FLUX
     real(dp)    :: flux_target = 1.0_dp
     real(dp)    :: dpdx_target = 0.0_dp
     ! i/o
     integer(ip) :: save_interval = 1000
     logical     :: restart       = .false.
     character(len=256) :: infile  = 'restart.dat'
     character(len=256) :: outfile = 'field.dat'
     integer(ip) :: seed     = 20260602
  end type params_t

  !=============================================================================
  ! grid_t  --  STATIC spectral geometry + Shen basis operators (built once).
  !=============================================================================
  type :: grid_t
     integer(ip) :: nx = 0, ny = 0, nz = 0
     integer(ip) :: nkx = 0      ! = nx/2 + 1  (real-to-complex in x)
     integer(ip) :: nkz = 0      ! = nz        (full complex in z)
     integer(ip) :: nxd = 0, nzd = 0   ! dealiased physical sizes (3/2 rule)

     ! wavenumbers: kx,kz multiply wp fields (hot)  => wp
     real(wp) DEVICE, allocatable :: kx(:)        ! (nkx)
     real(wp) DEVICE, allocatable :: kz(:)        ! (nkz)
     ! k2 feeds the FP64 operator build            => dp
     real(dp) DEVICE, allocatable :: k2(:,:)      ! (nkx,nkz) = kx^2 + kz^2

     ! wall-normal nodes + Shen basis matrices: ill-conditioned linear algebra => dp
     real(dp), allocatable :: yglc(:)              ! (ny) Gauss-Lobatto nodes in [-1,1]
     real(dp), allocatable :: B_dir(:,:)          ! mass  matrix, Dirichlet basis (eta)
     real(dp), allocatable :: A_dir(:,:)          ! stiff matrix, Dirichlet basis (eta)
     real(dp), allocatable :: B_bih(:,:)          ! mass  matrix, biharmonic basis (v)
     real(dp), allocatable :: A_bih(:,:)          ! stiff matrix, biharmonic basis (v)
     real(dp), allocatable :: C_bih(:,:)          ! biharmonic matrix, biharmonic basis (v)

     ! dealias mask multiplies wp fields           => wp
     real(wp) DEVICE, allocatable :: mask(:,:)    ! (nkx,nkz)  1=keep, 0=zero
  end type grid_t

  !=============================================================================
  ! fields_t  --  evolving spectral state + RK history. Big 3D arrays => wp.
  !               Mean profiles are tiny + statistically critical => dp.
  !               Layout (ny,nkx,nkz): y leading, contiguous solve columns.
  !=============================================================================
  type :: fields_t
     complex(wp) DEVICE, allocatable :: vh(:,:,:)    ! v_hat   (wall-normal velocity)
     complex(wp) DEVICE, allocatable :: omh(:,:,:)   ! eta_hat (wall-normal vorticity)
     complex(wp) DEVICE, allocatable :: uh(:,:,:)    ! reconstructed each substep
     complex(wp) DEVICE, allocatable :: wh(:,:,:)
     real(dp)    DEVICE, allocatable :: u0(:)        ! (ny) mean streamwise profile
     real(dp)    DEVICE, allocatable :: w0(:)        ! (ny) mean spanwise   profile
     complex(wp) DEVICE, allocatable :: hv_prev(:,:,:)   ! low-storage RK history
     complex(wp) DEVICE, allocatable :: hom_prev(:,:,:)
  end type fields_t

  !=============================================================================
  ! plans_t  --  cuFFT handles (opaque ints) + field-precision scratch.
  !              The transform layer picks single vs double cuFFT by `wp`.
  !=============================================================================
  type :: plans_t
     integer(ip) :: x_r2c = 0, x_c2r = 0
     integer(ip) :: z_fwd = 0, z_bwd = 0
     integer(ip) :: y_cheb = 0
     complex(wp) DEVICE, allocatable :: cbuf(:,:,:)
     real(wp)    DEVICE, allocatable :: rbuf(:,:,:)
  end type plans_t

  !=============================================================================
  ! solver_t  --  dt-DEPENDENT implicit (Crank-Nicolson) systems, batched over
  !               (kx,kz). ALWAYS dp: this is the conditioning-sensitive part.
  !               RHS arrives in wp, is promoted to dp here, result demoted back.
  !=============================================================================
  type :: solver_t
     real(dp)    :: dt_built = -1.0_dp
     integer(ip) :: stage_built = -1
     real(dp) DEVICE, allocatable :: fac_eta(:,:,:)   ! Helmholtz factors  (Dirichlet)
     real(dp) DEVICE, allocatable :: fac_v(:,:,:)     ! biharmonic factors
     integer(ip) DEVICE, allocatable :: piv_eta(:,:)
     integer(ip) DEVICE, allocatable :: piv_v(:,:)
  end type solver_t

  !=============================================================================
  ! stepper_t  --  RK3/CN integrator state. Coefficients + time in dp.
  !=============================================================================
  type :: stepper_t
     integer(ip) :: nst = 3
     real(dp)    :: anrk(4,2) = 0.0_dp     ! RK weight tables
     real(dp)    :: bnrk(4,2) = 0.0_dp
     real(dp)    :: cnrk(4,2) = 0.0_dp     ! Crank-Nicolson weights (set lambda)
     real(dp)    :: an = 0.0_dp, bn = 0.0_dp, cn = 0.0_dp
     real(dp)    :: anp1 = 0.0_dp, bnp1 = 0.0_dp, cnp1 = 0.0_dp
     integer(ip) :: substep = 0
     real(dp)    :: t = 0.0_dp             ! sim time: dp to avoid step accumulation drift
     integer(ip) :: step = 0
  end type stepper_t

  !=============================================================================
  ! stats_t  --  running statistics. dp accumulation over many samples.
  !=============================================================================
  type :: stats_t
     integer(ip) :: nsamp = 0
     real(dp), allocatable :: umean(:)
     real(dp), allocatable :: uu(:), vv(:), ww(:), uv(:)
  end type stats_t

end module types
