!===============================================================================
! fft3d_mod.F90  --  cuFFT layer: strided 1D C2C transforms along any axis of a
!                    3D array (n1,n2,n3), column-major (n1 fastest).
!
! Element b,x of a strided batched transform lives at  idata[b*idist + x*istride].
! For an (n1,n2,n3) array:
!   axis 1 (leading) : istride=1,     idist=n1,    batch=n2*n3   (one call)
!   axis 2 (middle)  : istride=n1,    idist=1,     batch=n1      (loop n3 slices)
!   axis 3 (trailing): istride=n1*n2, idist=1,     batch=n1*n2   (one call)
! The middle axis cannot batch over both i1 and i3 with a single distance, so it
! is executed as n3 calls, one per contiguous (n1 x n2) slice.
!
! cuFFT is unnormalised; fwd then inv along an axis scales by that axis length.
!
! The cufftPlanMany inembed/onembed arguments are the most delicate part; the
! per-axis round-trip tests isolate each config.
!===============================================================================

module fft3d_mod
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,          only: tp, ip
  use cudafor
  use cufft
  use transforms_mod, only: tf_c2c, tf_r2c, tf_c2r, CUFFT_FORWARD, CUFFT_INVERSE, &
                            tf_arena_attach          ! shared work-area arena
  implicit none
  private
  public :: plan_axis_c2c, exec_axis_c2c
  public :: plan_xz_c2c, exec_xz_c2c
  public :: plan_xz_r2c, plan_xz_c2r, exec_xz_r2c, exec_xz_c2r
  ! standalone 1-D real-x transforms (the un-fused x pass; the
  ! z pass reuses plan_axis_c2c/exec_axis_c2c on the trailing axis).
  public :: plan_x_r2c, plan_x_c2r, exec_x_r2c, exec_x_c2r
  public :: CUFFT_FORWARD, CUFFT_INVERSE
  public :: xl_release_current    ! drop the just-used xl slot (post-snapshot)

  ! x-leading scratch for the BATCHED x-transform (instead of a per-plane
  ! z-loop -- see the exec_x_* block below). One real (nx,ny,nz) and one complex
  ! (nkx,ny,nz) x-leading buffer.
  ! TWO shape-keyed slots behind pointers: the production driver alternates
  ! between the dealiased step shape and the clean snapshot shape, and a single
  ! exact-shape slot would cudaFree/Malloc multi-GB scratch on EVERY snapshot.
  ! ensure_xl points xl_r/xl_c at the matching slot; on a miss it evicts the slot
  ! that is NOT the most recently used, so the step/snap pair stays resident.
  real(tp),    allocatable, device, target :: xlr_s1(:,:,:), xlr_s2(:,:,:)
  complex(tp), allocatable, device, target :: xlc_s1(:,:,:), xlc_s2(:,:,:)
  real(tp),    device, pointer :: xl_r(:,:,:) => null()
  complex(tp), device, pointer :: xl_c(:,:,:) => null()
  integer :: xl_last = 0                       ! 0 = none, 1/2 = last slot hit

contains

  !-----------------------------------------------------------------------------
  ! [R2C] FUSED rank-2 real x / complex z transforms for a y-leading
  ! array, batched over y -- the R2C counterpart of plan_xz_c2c.  KEEPS the
  ! single-launch fused structure (un-fusing x from z would force a per-z-slab
  ! launch storm on one GPU -- the slab boundary breaks a uniform idist -- so x
  ! and z stay fused here; the split is a PARALLEL-phase change where the
  ! transpose sits between the passes and the storm cost is moot).
  !
  ! R2C transforms the REAL physical x-axis (length nxr) to nkx = nxr/2+1 complex
  ! modes; z stays full C2C (length nz).  Forward = R2C (real -> complex), inverse
  ! = C2R (complex -> real).  Because the real input (ny,nxr,nz) and complex
  ! output (ny,nkx,nz) differ in size, these are NOT in-place: distinct arrays.
  !
  ! Layout (verified numerically against numpy rfft-x / fft-z before coding):
  !   Fortran real  in  P(iy,ix,iz)  at  iy + ix*ny + iz*ny*nxr
  !   Fortran cmplx out S(iy,kx,iz)  at  iy + kx*ny + iz*ny*nkx
  !   cuFFT rank-2 advanced layout, dims n=[nz, nxr] (z outer, x inner):
  !     inembed=[nz, nxr], istride=ny, idist=1   (real)
  !     onembed=[nz, nkx], ostride=ny, odist=1   (complex)
  !     batch = ny
  ! For C2R the embeds swap (complex in, real out); cuFFT requires the SAME embed
  ! convention, so c2r uses inembed=[nz,nkx]/onembed=[nz,nxr].
  !
  ! NOTE (D2Z/Z2D destroy input): cuFFT's multi-dim C2R may overwrite its complex
  ! input. Callers must pass an input they can lose, or a scratch copy (the
  ! transform-layer wiring in increment 2 handles this).
  !-----------------------------------------------------------------------------
  subroutine plan_xz_r2c(plan, ny, nxr, nz)
    integer,     intent(out) :: plan
    integer(ip), intent(in)  :: ny, nxr, nz
    integer :: rtype, istat, nkx
    integer :: nn(2), inembed(2), onembed(2)
    integer(8) :: wsz
    rtype = merge(CUFFT_D2Z, CUFFT_R2C, tp == real64)
    nkx = int(nxr)/2 + 1
    nn(1) = int(nz);  nn(2) = int(nxr)
    inembed(1) = int(nz);  inembed(2) = int(nxr)        ! real input extent
    onembed(1) = int(nz);  onembed(2) = nkx             ! complex output extent
    istat = cufftCreate(plan)
    if (istat == CUFFT_SUCCESS) istat = cufftSetAutoAllocation(plan, 0)   ! arena
    if (istat == CUFFT_SUCCESS) &
       istat = cufftMakePlanMany(plan, 2, nn, inembed, int(ny), 1, onembed, int(ny), 1, &
                                 rtype, int(ny), wsz)
    if (istat /= CUFFT_SUCCESS) then
       print '(a,i0)', 'cufftMakePlanMany (xz R2C) failed, istat=', istat;  error stop 1
    end if
    call tf_arena_attach(plan, wsz)
  end subroutine plan_xz_r2c

  subroutine plan_xz_c2r(plan, ny, nxr, nz)
    integer,     intent(out) :: plan
    integer(ip), intent(in)  :: ny, nxr, nz
    integer :: rtype, istat, nkx
    integer :: nn(2), inembed(2), onembed(2)
    integer(8) :: wsz
    rtype = merge(CUFFT_Z2D, CUFFT_C2R, tp == real64)
    nkx = int(nxr)/2 + 1
    nn(1) = int(nz);  nn(2) = int(nxr)
    inembed(1) = int(nz);  inembed(2) = nkx             ! complex input extent
    onembed(1) = int(nz);  onembed(2) = int(nxr)        ! real output extent
    istat = cufftCreate(plan)
    if (istat == CUFFT_SUCCESS) istat = cufftSetAutoAllocation(plan, 0)   ! arena
    if (istat == CUFFT_SUCCESS) &
       istat = cufftMakePlanMany(plan, 2, nn, inembed, int(ny), 1, onembed, int(ny), 1, &
                                 rtype, int(ny), wsz)
    if (istat /= CUFFT_SUCCESS) then
       print '(a,i0)', 'cufftMakePlanMany (xz C2R) failed, istat=', istat;  error stop 1
    end if
    call tf_arena_attach(plan, wsz)
  end subroutine plan_xz_c2r

  !-- forward: real P(ny,nxr,nz) -> complex S(ny,nkx,nz) ------------------------
  subroutine exec_xz_r2c(plan, P, S)
    integer,     intent(in) :: plan
    real(tp),    device     :: P(*)
    complex(tp), device     :: S(*)
    call tf_r2c(plan, P, S)
  end subroutine exec_xz_r2c

  !-- inverse: complex S(ny,nkx,nz) -> real P(ny,nxr,nz) (may consume S) --------
  subroutine exec_xz_c2r(plan, S, P)
    integer,     intent(in) :: plan
    complex(tp), device     :: S(*)
    real(tp),    device     :: P(*)
    call tf_c2r(plan, S, P)
  end subroutine exec_xz_c2r


  !-----------------------------------------------------------------------------
  ! plan_xz_c2c  --  ONE rank-2 plan doing the full (x,z) Fourier transform of a
  ! (n1,n2,n3) y-leading array, batched over y.
  !
  ! An axis-2-loop (n3 separate cuFFT calls) + axis-3 pass would launch
  ! ~(nzd+1) cuFFT calls per field per direction (thousands of small launches
  ! per step at 64^3); this is a single call.
  !
  ! Layout derivation (0-based): Fortran element (iy,ix,iz) sits at
  !     addr = iy + ix*n1 + iz*n1*n2.
  ! cuFFT rank-2 advanced layout: addr = b*idist + (x0*inembed(2) + x1)*istride,
  ! with x0 the n(1) (outer/slowest) coordinate and x1 the n(2) (inner) one. With
  !     n = [n3, n2]  (z outer, x inner),  inembed = onembed = n,
  !     istride = n1,  idist = 1,  batch = n1,
  ! addr = b + (x0*n2 + x1)*n1  --  exactly the Fortran layout with b=iy, x1=ix,
  ! x0=iz. In-place execution (same array in and out) is supported for C2C.
  !-----------------------------------------------------------------------------
  subroutine plan_xz_c2c(plan, n1, n2, n3)
    integer,     intent(out) :: plan
    integer(ip), intent(in)  :: n1, n2, n3
    integer :: ctype, istat
    integer :: nn(2)
    integer(8) :: wsz
    ctype = merge(CUFFT_Z2Z, CUFFT_C2C, tp == real64)
    nn(1) = int(n3);  nn(2) = int(n2)
    istat = cufftCreate(plan)
    if (istat == CUFFT_SUCCESS) istat = cufftSetAutoAllocation(plan, 0)   ! arena
    if (istat == CUFFT_SUCCESS) &
       istat = cufftMakePlanMany(plan, 2, nn, nn, int(n1), 1, nn, int(n1), 1, &
                                 ctype, int(n1), wsz)
    if (istat /= CUFFT_SUCCESS) then
       print '(a,i0)', 'cufftMakePlanMany (xz, rank-2) failed, istat=', istat
       error stop 1
    end if
    call tf_arena_attach(plan, wsz)
  end subroutine plan_xz_c2c

  !-- transform A -> B over the (x,z) plane (pass A twice for in-place) --------
  subroutine exec_xz_c2c(plan, n1, n2, n3, A, B, dir)
    integer,     intent(in) :: plan, dir
    integer(ip), intent(in) :: n1, n2, n3
    complex(tp), device     :: A(n1,n2,n3), B(n1,n2,n3)
    call tf_c2c(plan, A, B, dir)
  end subroutine exec_xz_c2c

  subroutine plan_axis_c2c(plan, axis, n1, n2, n3)
    integer,     intent(out) :: plan
    integer(ip), intent(in)  :: axis, n1, n2, n3
    integer :: ctype, istat, istride, idist, batch
    integer :: nn(1)
    integer(8) :: wsz
    ctype = merge(CUFFT_Z2Z, CUFFT_C2C, tp == real64)
    select case (axis)
    case (1)
       nn(1) = int(n1);  istride = 1;          idist = int(n1);     batch = int(n2*n3)
    case (2)
       nn(1) = int(n2);  istride = int(n1);    idist = 1;           batch = int(n1)
    case (3)
       nn(1) = int(n3);  istride = int(n1*n2); idist = 1;           batch = int(n1*n2)
    case default
       error stop 'plan_axis_c2c: axis must be 1, 2 or 3'
    end select
    istat = cufftCreate(plan)
    if (istat == CUFFT_SUCCESS) istat = cufftSetAutoAllocation(plan, 0)   ! arena
    if (istat == CUFFT_SUCCESS) &
       istat = cufftMakePlanMany(plan, 1, nn, nn, istride, idist, nn, istride, idist, &
                                 ctype, batch, wsz)
    if (istat /= CUFFT_SUCCESS) then
       print '(a,i0)', 'cufftMakePlanMany failed, istat=', istat
       error stop 1
    end if
    call tf_arena_attach(plan, wsz)
  end subroutine plan_axis_c2c

  !---------------------------------------------------------------------------
  ! Transform A -> B along `axis`. For axis 2, loops over the n3 slices.
  !---------------------------------------------------------------------------
  subroutine exec_axis_c2c(plan, axis, n1, n2, n3, A, B, dir)
    integer,     intent(in) :: plan, dir
    integer(ip), intent(in) :: axis, n1, n2, n3
    complex(tp), device     :: A(n1,n2,n3), B(n1,n2,n3)
    integer(ip) :: k
    if (axis == 2) then
       do k = 1, n3
          call tf_c2c(plan, A(:,:,k), B(:,:,k), dir)
       end do
    else
       call tf_c2c(plan, A, B, dir)
    end if
  end subroutine exec_axis_c2c

  !=============================================================================
  ! STANDALONE 1-D real-x transforms for a y-leading array
  ! (ny, nx, nz)  <->  (ny, nkx, nz),  nkx = nx/2+1.  This is the x pass of the
  ! UN-FUSED transform used by the parallel path: x is FFT'd while it is whole
  ! on-rank (slab layout A), the transpose then moves data, and the z pass runs
  ! separately (plan_axis_c2c on the trailing axis).
  !
  ! x is the MIDDLE axis (addr = iy + ix*ny + iz*ny*nx). cuFFT's advanced layout
  ! cannot batch a middle-axis transform over BOTH the leading (y) AND trailing
  ! (z) index with a single idist (the slab/plane boundary breaks a uniform
  ! distance -- the same reason exec_axis_c2c loops axis 2). A per-(ny,nx)-plane
  ! plan LOOPED over the nz planes would issue nz tiny cuFFT launches per call,
  ! which dominates the step at small grids: ~1440 x-FFT launches/step at
  ! 128^3/P=4 (~26% of GPU kernel time and the bulk of the 120k cuLaunchKernel
  ! calls -- a launch storm, not a bandwidth cost).
  !
  ! We instead REORDER to x-leading once (so x is contiguous), do ONE
  ! contiguous batched cuFFT over all ny*nz columns, and reorder back -- 3 kernel
  ! launches instead of nz. The two transposes are bandwidth-bound full-field
  ! copies, cheap vs the launch tax they remove (and they shrink with 1/P). The
  ! per-column 1-D transform is unchanged (same plan kind, length and precision --
  ! only the batch count and memory order differ), so the result is numerically
  ! identical to the per-plane version up to round-off -- test_fft3d_1d_xz still
  ! asserts equality vs the FUSED single-GPU plan, and main_par --validate (tol
  ! 1e-9; the parallel path agrees with the oracle to ~1e-13) is the end-to-end gate.
  !
  ! Because the x-leading scratch is CONTIGUOUS, the plan is a plain batched
  ! cufftPlan1d (istride=1, idist=nx) -- no advanced layout needed. The C2R path
  ! reads the caller's input only into scratch, so (unlike the old per-plane
  ! version) it does NOT clobber its complex input.
  !=============================================================================
  subroutine plan_x_r2c(plan, ny, nx, nz)
    integer,     intent(out) :: plan
    integer(ip), intent(in)  :: ny, nx, nz        ! batch = ny*nz (x-leading)
    integer :: rtype, istat
    integer(8) :: wsz
    rtype = merge(CUFFT_D2Z, CUFFT_R2C, tp == real64)
    istat = cufftCreate(plan)
    if (istat == CUFFT_SUCCESS) istat = cufftSetAutoAllocation(plan, 0)   ! arena
    if (istat == CUFFT_SUCCESS) istat = cufftMakePlan1d(plan, int(nx), rtype, int(ny*nz), wsz)
    if (istat /= CUFFT_SUCCESS) then
       print '(a,i0)', 'cufftMakePlan1d (x R2C, batched) failed, istat=', istat
       error stop 1
    end if
    call tf_arena_attach(plan, wsz)
  end subroutine plan_x_r2c

  subroutine plan_x_c2r(plan, ny, nx, nz)
    integer,     intent(out) :: plan
    integer(ip), intent(in)  :: ny, nx, nz        ! batch = ny*nz (x-leading)
    integer :: rtype, istat
    integer(8) :: wsz
    rtype = merge(CUFFT_Z2D, CUFFT_C2R, tp == real64)
    istat = cufftCreate(plan)
    if (istat == CUFFT_SUCCESS) istat = cufftSetAutoAllocation(plan, 0)   ! arena
    if (istat == CUFFT_SUCCESS) istat = cufftMakePlan1d(plan, int(nx), rtype, int(ny*nz), wsz)   ! nx = REAL length
    if (istat /= CUFFT_SUCCESS) then
       print '(a,i0)', 'cufftMakePlan1d (x C2R, batched) failed, istat=', istat
       error stop 1
    end if
    call tf_arena_attach(plan, wsz)
  end subroutine plan_x_c2r

  !-- point xl_r/xl_c at a slot of shape real(nx,ny,nz)/complex(nkx,ny,nz) -------
  !-- 2-slot cache: hit = repoint only; miss = realloc the non-MRU slot --------
  subroutine ensure_xl(ny, nx, nz)
    integer(ip), intent(in) :: ny, nx, nz
    integer(ip) :: nkx, victim
    nkx = nx/2 + 1
    if (slot_match(xlr_s1, nx, ny, nz)) then
       xl_r => xlr_s1;  xl_c => xlc_s1;  xl_last = 1;  return
    end if
    if (slot_match(xlr_s2, nx, ny, nz)) then
       xl_r => xlr_s2;  xl_c => xlc_s2;  xl_last = 2;  return
    end if
    victim = merge(2, 1, xl_last == 1)           ! never evict the MRU slot
    if (victim == 1) then
       if (allocated(xlr_s1)) deallocate(xlr_s1, xlc_s1)
       allocate(xlr_s1(nx, ny, nz), xlc_s1(nkx, ny, nz))
       xl_r => xlr_s1;  xl_c => xlc_s1
    else
       if (allocated(xlr_s2)) deallocate(xlr_s2, xlc_s2)
       allocate(xlr_s2(nx, ny, nz), xlc_s2(nkx, ny, nz))
       xl_r => xlr_s2;  xl_c => xlc_s2
    end if
    xl_last = victim
  end subroutine ensure_xl

  ! Free the slot MOST RECENTLY pointed at by ensure_xl. Intended use:
  ! right after a snapshot write (driver knob snap_evict): the clean-grid slot
  ! was just used and is idle until the NEXT snapshot, while holding e.g.
  ! 1.26 GB/GPU at the Re2000 grid on 16 nodes. The every-step dealiased slot
  ! is untouched (it is the OTHER slot at that moment). Costs one realloc at
  ! the next snapshot, so it is deck-selectable. Synchronizes the device BEFORE
  ! deallocating: never free device memory with work in flight.
  subroutine xl_release_current()
    integer :: istat
    if (xl_last == 0) return
    istat = cudaDeviceSynchronize()
    if (xl_last == 1) then
       if (allocated(xlr_s1)) deallocate(xlr_s1, xlc_s1)
    else
       if (allocated(xlr_s2)) deallocate(xlr_s2, xlc_s2)
    end if
    nullify(xl_r, xl_c)
    xl_last = 0
  end subroutine xl_release_current

  logical function slot_match(buf, nx, ny, nz) result(ok)
    real(tp), allocatable, device, intent(in) :: buf(:,:,:)
    integer(ip), intent(in) :: nx, ny, nz
    ok = .false.
    if (.not. allocated(buf)) return
    ok = (size(buf,1) == nx .and. size(buf,2) == ny .and. size(buf,3) == nz)
  end function slot_match

  !-- batched transpose dst(j,i,k) = src(i,j,k), over nz planes (real / complex) -
  subroutine transp_r(n1, n2, nz, src, dst)
    integer(ip), intent(in) :: n1, n2, nz
    real(tp),    device     :: src(n1, n2, nz)
    real(tp),    device     :: dst(n2, n1, nz)
    integer(ip) :: i, j, k
    !$cuf kernel do(3) <<<*,*>>>
    do k = 1, nz
       do j = 1, n2
          do i = 1, n1
             dst(j, i, k) = src(i, j, k)
          end do
       end do
    end do
  end subroutine transp_r

  subroutine transp_c(n1, n2, nz, src, dst)
    integer(ip), intent(in) :: n1, n2, nz
    complex(tp), device     :: src(n1, n2, nz)
    complex(tp), device     :: dst(n2, n1, nz)
    integer(ip) :: i, j, k
    !$cuf kernel do(3) <<<*,*>>>
    do k = 1, nz
       do j = 1, n2
          do i = 1, n1
             dst(j, i, k) = src(i, j, k)
          end do
       end do
    end do
  end subroutine transp_c

  !-- forward x: real P(ny,nx,nz) -> complex S(ny,nkx,nz). ONE batched R2C -------
  subroutine exec_x_r2c(plan, ny, nx, nz, P, S)
    integer,     intent(in) :: plan
    integer(ip), intent(in) :: ny, nx, nz
    real(tp),    device     :: P(ny, nx, nz)
    complex(tp), device     :: S(ny, (nx/2+1), nz)
    call ensure_xl(ny, nx, nz)
    call transp_r(ny, nx, nz, P, xl_r)             ! y-leading -> x-leading (nx,ny,nz)
    call tf_r2c(plan, xl_r, xl_c)                  ! batched over ny*nz columns
    call transp_c(nx/2+1, ny, nz, xl_c, S)         ! x-leading -> y-leading
  end subroutine exec_x_r2c

  !-- inverse x: complex S(ny,nkx,nz) -> real P(ny,nx,nz). ONE batched C2R -------
  subroutine exec_x_c2r(plan, ny, nx, nz, S, P)
    integer,     intent(in) :: plan
    integer(ip), intent(in) :: ny, nx, nz
    complex(tp), device     :: S(ny, (nx/2+1), nz)
    real(tp),    device     :: P(ny, nx, nz)
    call ensure_xl(ny, nx, nz)
    call transp_c(ny, nx/2+1, nz, S, xl_c)         ! y-leading -> x-leading (nkx,ny,nz)
    call tf_c2r(plan, xl_c, xl_r)                  ! batched over ny*nz columns
    call transp_r(nx, ny, nz, xl_r, P)             ! x-leading -> y-leading
  end subroutine exec_x_c2r

end module fft3d_mod
