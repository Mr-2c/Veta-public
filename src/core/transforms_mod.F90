!===============================================================================
! transforms_mod.F90  --  cuFFT layer, increment 4a: precision-generic Fourier
!                         transforms (plan management + R2C/C2R + C2C).
!
! Requires nvfortran -cuda -cudalib=cufft, and must run on a GPU compute node.
! The cuFFT Fortran interface details are the delicate part; the round-trip +
! derivative tests in test_transforms.F90 localize any issue.
!
! PRECISION-GENERIC DISPATCH
!   cuFFT has separate single (R2C/C2R/C2C) and double (D2Z/Z2D/Z2Z) entry
!   points. We hide that behind generic interfaces (tf_c2c, tf_r2c, tf_c2r):
!   the compiler selects the specific procedure from the kind of the device
!   arrays passed (which is tp), and the plan type is chosen to match by `tp`
!   at plan-creation time. So switching FP64<->FP32 needs no change here --
!   exactly the precision policy in src/core/types.F90.
!
! SCOPE OF 4a: contiguous 1D (optionally batched) transforms. Strided/3D
!   planMany layouts and the 3/2 dealias padding come in later increments.
!   cuFFT is UNNORMALIZED: a forward then inverse scales data by the transform
!   length N, so callers divide by N (the tests do).
!===============================================================================

module transforms_mod
  use, intrinsic :: iso_fortran_env, only: real32, real64
  use kinds,  only: tp, ip
  use cudafor
  use cufft
  implicit none
  private

  public :: tf_c2c_plan, tf_r2c_plan, tf_c2r_plan, tf_destroy
  public :: tf_c2c, tf_r2c, tf_c2r
  public :: CUFFT_FORWARD, CUFFT_INVERSE      ! re-exported for callers
  public :: tf_arena_attach, tf_arena_bytes   ! shared work-area arena

  ! ONE shared cuFFT work area for ALL plans. Every plan is created with
  ! cufftSetAutoAllocation(0) and pointed at this arena, sized to the LARGEST
  ! single plan's requirement -- instead of every plan owning a private work
  ! area (which at production grids costs O(data) EACH). Safe because all cuFFT
  ! execs in this code run serialized on ONE stream -- the legacy default stream
  ! in serial builds, the process-wide par_stream in parallel builds (pinned
  ! per-plan in tf_arena_attach) -- no two transforms ever in flight.
  ! Any MULTI-stream scheme would have to split this arena per stream.
  ! GROWTH: allocating a bigger arena moves it, so every registered plan is
  ! re-pointed (cufftSetWorkArea is host metadata, cheap). tf_destroy detaches.
  integer, parameter :: TF_ARENA_MAX = 64            ! live plans cap (today ~12)
  integer     :: ar_plans(TF_ARENA_MAX) = 0
  integer     :: ar_n = 0
  integer(8)  :: ar_bytes = 0
  integer(1), device, allocatable :: ar_buf(:)

  ! Precision-generic execution. The transform tier runs at a single precision
  ! (= tp) per build, and the cuFFT plan type is chosen by tp at creation
  ! (tf_c2c_plan etc.). So each generic exposes ONLY the tp-matching specific:
  ! an off-tp transform -- e.g. a real64 array at a TP=sp build -- then has NO
  ! matching specific and FAILS AT COMPILE, instead of resolving to a specific
  ! whose cuFFT exec precision disagrees with the (tp-typed) plan and silently
  ! corrupting at runtime.
  interface tf_c2c
#ifdef CLAUSON_TP_SP
     module procedure tf_c2c_sp
#else
     module procedure tf_c2c_dp
#endif
  end interface
  interface tf_r2c
#ifdef CLAUSON_TP_SP
     module procedure tf_r2c_sp
#else
     module procedure tf_r2c_dp
#endif
  end interface
  interface tf_c2r
#ifdef CLAUSON_TP_SP
     module procedure tf_c2r_sp
#else
     module procedure tf_c2r_dp
#endif
  end interface

contains

  subroutine check(istat, where)
    integer,      intent(in) :: istat
    character(*), intent(in) :: where
    if (istat /= CUFFT_SUCCESS) then
       print '(a,a,a,i0)', 'cuFFT error in ', where, ', istat=', istat
       error stop 1
    end if
  end subroutine check

  !-- arena: register a plan needing `wsz` work bytes; grow + re-point --------
  ! Every plan ALSO gets its execution stream pinned here, to the thread's
  ! CURRENT default stream: the parallel build sets it to the non-blocking
  ! par_stream (cudaforSetDefaultStream in par_init, before any plan exists);
  ! the serial build never sets it, so this is the legacy stream.
  ! ALL creators funnel through this attach, so no plan can be missed.
  subroutine tf_arena_attach(plan, wsz)
    integer,    intent(in) :: plan
    integer(8), intent(in) :: wsz
    integer :: i, istat
    istat = cufftSetStream(plan, cudaforGetDefaultStream())
    call check(istat, 'tf_arena_attach (setstream)')
    if (ar_n >= TF_ARENA_MAX) then
       print '(a,i0)', 'tf_arena_attach: plan registry full at ', TF_ARENA_MAX
       error stop 1
    end if
    ar_n = ar_n + 1;  ar_plans(ar_n) = plan
    if (max(wsz, 1_8) > ar_bytes) then
       if (allocated(ar_buf)) deallocate(ar_buf)
       ar_bytes = max(wsz, 1_8)
       allocate(ar_buf(ar_bytes))
       do i = 1, ar_n                       ! the arena MOVED: re-point every plan
          istat = cufftSetWorkArea(ar_plans(i), ar_buf)
          call check(istat, 'tf_arena_attach (re-point)')
       end do
    else
       istat = cufftSetWorkArea(plan, ar_buf)
       call check(istat, 'tf_arena_attach')
    end if
  end subroutine tf_arena_attach

  subroutine tf_arena_detach(plan)          ! internal: forget a destroyed plan
    integer, intent(in) :: plan
    integer :: i
    do i = 1, ar_n
       if (ar_plans(i) == plan) then
          ar_plans(i) = ar_plans(ar_n);  ar_plans(ar_n) = 0;  ar_n = ar_n - 1
          return
       end if
    end do
  end subroutine tf_arena_detach

  integer(8) function tf_arena_bytes() result(b)   ! for the [init] log line
    b = ar_bytes
  end function tf_arena_bytes

  !-- plan creation (type chosen by tp; work area = the shared arena) --------
  subroutine tf_c2c_plan(plan, n, batch)
    integer,     intent(out) :: plan
    integer(ip), intent(in)  :: n, batch
    integer :: ctype, istat
    integer(8) :: wsz
    ctype = merge(CUFFT_Z2Z, CUFFT_C2C, tp == real64)
    istat = cufftCreate(plan);                          call check(istat, 'tf_c2c_plan (create)')
    istat = cufftSetAutoAllocation(plan, 0);            call check(istat, 'tf_c2c_plan (autoalloc)')
    istat = cufftMakePlan1d(plan, int(n), ctype, int(batch), wsz)
    call check(istat, 'tf_c2c_plan')
    call tf_arena_attach(plan, wsz)
  end subroutine tf_c2c_plan

  subroutine tf_r2c_plan(plan, n, batch)
    integer,     intent(out) :: plan
    integer(ip), intent(in)  :: n, batch
    integer :: ctype, istat
    integer(8) :: wsz
    ctype = merge(CUFFT_D2Z, CUFFT_R2C, tp == real64)
    istat = cufftCreate(plan);                          call check(istat, 'tf_r2c_plan (create)')
    istat = cufftSetAutoAllocation(plan, 0);            call check(istat, 'tf_r2c_plan (autoalloc)')
    istat = cufftMakePlan1d(plan, int(n), ctype, int(batch), wsz)
    call check(istat, 'tf_r2c_plan')
    call tf_arena_attach(plan, wsz)
  end subroutine tf_r2c_plan

  subroutine tf_c2r_plan(plan, n, batch)
    integer,     intent(out) :: plan
    integer(ip), intent(in)  :: n, batch
    integer :: ctype, istat
    integer(8) :: wsz
    ctype = merge(CUFFT_Z2D, CUFFT_C2R, tp == real64)
    istat = cufftCreate(plan);                          call check(istat, 'tf_c2r_plan (create)')
    istat = cufftSetAutoAllocation(plan, 0);            call check(istat, 'tf_c2r_plan (autoalloc)')
    istat = cufftMakePlan1d(plan, int(n), ctype, int(batch), wsz)
    call check(istat, 'tf_c2r_plan')
    call tf_arena_attach(plan, wsz)
  end subroutine tf_c2r_plan

  subroutine tf_destroy(plan)
    integer, intent(inout) :: plan
    integer :: istat
    istat = cufftDestroy(plan)
    call check(istat, 'tf_destroy')
    call tf_arena_detach(plan)
    plan = 0
  end subroutine tf_destroy

  !-- C2C / Z2Z execution ---------------------------------------------------
  subroutine tf_c2c_sp(plan, a, b, dir)
    integer,         intent(in) :: plan, dir
    complex(real32), device     :: a(*), b(*)
    call check(cufftExecC2C(plan, a, b, dir), 'tf_c2c_sp')
  end subroutine tf_c2c_sp

  subroutine tf_c2c_dp(plan, a, b, dir)
    integer,         intent(in) :: plan, dir
    complex(real64), device     :: a(*), b(*)
    call check(cufftExecZ2Z(plan, a, b, dir), 'tf_c2c_dp')
  end subroutine tf_c2c_dp

  !-- R2C / D2Z execution (real -> complex) ---------------------------------
  subroutine tf_r2c_sp(plan, rin, cout)
    integer,         intent(in) :: plan
    real(real32),    device     :: rin(*)
    complex(real32), device     :: cout(*)
    call check(cufftExecR2C(plan, rin, cout), 'tf_r2c_sp')
  end subroutine tf_r2c_sp

  subroutine tf_r2c_dp(plan, rin, cout)
    integer,         intent(in) :: plan
    real(real64),    device     :: rin(*)
    complex(real64), device     :: cout(*)
    call check(cufftExecD2Z(plan, rin, cout), 'tf_r2c_dp')
  end subroutine tf_r2c_dp

  !-- C2R / Z2D execution (complex -> real) ---------------------------------
  subroutine tf_c2r_sp(plan, cin, rout)
    integer,         intent(in) :: plan
    complex(real32), device     :: cin(*)
    real(real32),    device     :: rout(*)
    call check(cufftExecC2R(plan, cin, rout), 'tf_c2r_sp')
  end subroutine tf_c2r_sp

  subroutine tf_c2r_dp(plan, cin, rout)
    integer,         intent(in) :: plan
    complex(real64), device     :: cin(*)
    real(real64),    device     :: rout(*)
    call check(cufftExecZ2D(plan, cin, rout), 'tf_c2r_dp')
  end subroutine tf_c2r_dp

end module transforms_mod
