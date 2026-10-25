!===============================================================================
! test_batched_transform_mp.F90  --  multi-GPU: validate the nfield-BATCHED
! distributed transforms (par_to_physical_nf / par_to_spectral_nf) against N
! calls of the single-field routines (par_to_physical_da_r2c /
! par_to_spectral_da_r2c). The ONLY difference between the two is that the
! batched path does ONE nfield=nf transpose where the single-field path does nf
! separate ones; everything else (the per-field y-cheb, the z/x pads/truncs, the
! z- and x-FFTs) is identical. The transpose is pure data movement, so the two
! must agree to the bit (the tolerance is just a NaN-guarded ceiling).
!
! This isolates the batched transpose, at the field counts the distributed
! nonlinear RHS uses: nf=6 (the 6-field inverse) and nf=3 (the 3-field
! forward). Each rank builds its own deterministic slab (no broadcast); the
! batched-vs-single comparison is per-rank, and par_reduce_and fails the job on
! any rank. Matching at P=1/4/8 establishes the batching is correct and
! P-independent.
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_batched_transform_mp   Run: bash run_all_par_tests.sh run -k test_batched_transform_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_batched_transform_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,              only: tp, ip
  use cudafor
  use decomp_mod,         only: decomp_t, decomp_setup
  use par_spectral3d_mod, only: par_s3d_plans, par_s3d_setup_da_r2c, par_s3d_destroy_da_r2c, &
                                par_to_physical_da_r2c, par_to_spectral_da_r2c, &
                                par_to_physical_nf, par_to_spectral_nf
  use par_mod
  implicit none

  integer(ip), parameter :: nx = 32, ny = 6, nz = 16
  integer(ip), parameter :: nxd = 3*nx/2, nzd = 3*nz/2, nkx = nx/2+1, nkxd = nxd/2+1
  integer(ip), parameter :: nfi = 6, nff = 3        ! inverse batch / forward batch
  real(real64) :: tol
  logical :: all_ok, ok_inv, ok_fwd
  integer  :: istat
  integer(ip) :: iy, jx, kz, f, bnkx, anz

  type(decomp_t)      :: d
  type(par_s3d_plans) :: pl

  ! ---- inverse: nf=6 coarse spectral slabs (Layout B) ----
  complex(tp), allocatable :: A6_h(:,:,:,:)
  real(tp),    allocatable :: P6_b_h(:,:,:,:), P6_s_h(:,:,:,:)
  complex(tp), allocatable, device :: A6_b_d(:,:,:,:), A6_s_d(:,:,:,:)
  real(tp),    allocatable, device :: P6_b_d(:,:,:,:), P1_d(:,:,:)
  complex(tp), allocatable, device :: BzI(:,:,:,:), AzI(:,:,:,:)                 ! batched (nfi) scratch
  complex(tp), allocatable, device :: AfI(:,:,:)   ! x-scratch: ONE slice
  ! ---- forward: nf=3 real physical slabs (Layout A) ----
  real(tp),    allocatable :: P3_h(:,:,:,:)
  complex(tp), allocatable :: A3_b_h(:,:,:,:), A3_s_h(:,:,:,:)
  real(tp),    allocatable, device :: P3_d(:,:,:,:)
  complex(tp), allocatable, device :: A3_b_d(:,:,:,:), A1_d(:,:,:)
  complex(tp), allocatable, device :: BzF(:,:,:,:), AzF(:,:,:,:)                 ! batched (nff) scratch
  complex(tp), allocatable, device :: AfF(:,:,:)   ! x-scratch: ONE slice
  ! ---- single-field scratch (nfield=1) ----
  complex(tp), allocatable, device :: Bz1(:,:,:,:), Az1(:,:,:,:), Af1(:,:,:)

  call par_init()
  tol = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)
  if (par_rank == 0) print '(a,i0,a)', '[batched-mp] world size = ', par_nproc, ' rank(s)'

  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
  bnkx = d%b_nkx_loc;  anz = d%ad_nz_loc
  call par_s3d_setup_da_r2c(pl, d, ny, nx, nz)

  allocate(A6_h(ny,bnkx,nz,nfi), P6_b_h(ny,nxd,anz,nfi), P6_s_h(ny,nxd,anz,nfi))
  allocate(A6_b_d(ny,bnkx,nz,nfi), A6_s_d(ny,bnkx,nz,nfi), P6_b_d(ny,nxd,anz,nfi), P1_d(ny,nxd,anz))
  allocate(BzI(ny,bnkx,nzd,nfi), AzI(ny,nkx,anz,nfi), AfI(ny,nkxd,anz))
  allocate(P3_h(ny,nxd,anz,nff), A3_b_h(ny,bnkx,nz,nff), A3_s_h(ny,bnkx,nz,nff))
  allocate(P3_d(ny,nxd,anz,nff), A3_b_d(ny,bnkx,nz,nff), A1_d(ny,bnkx,nz))
  allocate(BzF(ny,bnkx,nzd,nff), AzF(ny,nkx,anz,nff), AfF(ny,nkxd,anz))
  allocate(Bz1(ny,bnkx,nzd,1), Az1(ny,nkx,anz,1), Af1(ny,nkxd,anz))

  ! ===== INVERSE equivalence (nf=6): batched vs 6x single-field ===============
  do f = 1, nfi; do kz = 1, nz; do jx = 1, bnkx; do iy = 1, ny
     A6_h(iy,jx,kz,f) = cmplx( hr(iy,jx,kz,f), hi(iy,jx,kz,f), tp )
  end do; end do; end do; end do
  A6_b_d = A6_h;  A6_s_d = A6_h                       ! both consumed (cheb in place)

  call par_to_physical_nf(pl, d, ny, nx, nz, nxd, nzd, bnkx, anz, nfi, &
                          A6_b_d, BzI, AzI, AfI, P6_b_d)
  istat = cudaDeviceSynchronize();  P6_b_h = P6_b_d

  do f = 1, nfi
     call par_to_physical_da_r2c(pl, d, ny, nx, nz, nxd, nzd, bnkx, anz, &
                                 A6_s_d(:,:,:,f), Bz1, Az1, Af1, P1_d)
     istat = cudaDeviceSynchronize();  P6_s_h(:,:,:,f) = P1_d
  end do
  ok_inv = close_r(P6_b_h, P6_s_h, tol)

  ! ===== FORWARD equivalence (nf=3): batched vs 3x single-field ===============
  do f = 1, nff; do kz = 1, anz; do jx = 1, nxd; do iy = 1, ny
     P3_h(iy,jx,kz,f) = hr(iy,jx,kz,f)
  end do; end do; end do; end do
  P3_d = P3_h                                          ! read-only (forward does not consume P)

  call par_to_spectral_nf(pl, d, ny, nx, nz, nxd, nzd, bnkx, anz, nff, &
                          P3_d, AfF, AzF, BzF, A3_b_d)
  istat = cudaDeviceSynchronize();  A3_b_h = A3_b_d

  do f = 1, nff
     call par_to_spectral_da_r2c(pl, d, ny, nx, nz, nxd, nzd, bnkx, anz, &
                                 P3_d(:,:,:,f), Af1, Az1, Bz1, A1_d)
     istat = cudaDeviceSynchronize();  A3_s_h(:,:,:,f) = A1_d
  end do
  ok_fwd = close_c(A3_b_h, A3_s_h, tol)

  call par_s3d_destroy_da_r2c(pl)

  call par_reduce_and(ok_inv)
  call par_reduce_and(ok_fwd)
  all_ok = ok_inv .and. ok_fwd
  if (par_rank == 0) then
     call report('inverse: batched nf=6 == 6x single-field', ok_inv)
     call report('forward: batched nf=3 == 3x single-field', ok_fwd)
     if (all_ok) then
        print '(a)', 'ALL TESTS PASSED'
     else
        print '(a)', 'SOME TESTS FAILED'
     end if
  end if

  call par_finalize()
  if (.not. all_ok) error stop 1

contains

  real(tp) function hr(a, b, c, e) result(v)        ! deterministic real in [-0.5,0.5)
    integer(ip), intent(in) :: a, b, c, e
    integer(ip) :: h
    h = modulo( a*1009_ip + b*2027_ip + c*3001_ip + e*4013_ip + b*c*17_ip, 100003_ip )
    v = real(h, tp)/100003.0_tp - 0.5_tp
  end function hr

  real(tp) function hi(a, b, c, e) result(v)        ! a different stream for imag parts
    integer(ip), intent(in) :: a, b, c, e
    integer(ip) :: h
    h = modulo( a*733_ip + b*1597_ip + c*2741_ip + e*5051_ip + a*c*29_ip, 100019_ip )
    v = real(h, tp)/100019.0_tp - 0.5_tp
  end function hi

  logical function close_r(a, b, tol) result(ok)
    real(tp),     intent(in) :: a(:,:,:,:), b(:,:,:,:)
    real(real64), intent(in) :: tol
    ok = all(a == a) .and. all(b == b) .and. (real(maxval(abs(a - b)), real64) <= tol)
  end function close_r

  logical function close_c(a, b, tol) result(ok)
    complex(tp),  intent(in) :: a(:,:,:,:), b(:,:,:,:)
    real(real64), intent(in) :: tol
    ok = all(a == a) .and. all(b == b) .and. (real(maxval(abs(a - b)), real64) <= tol)
  end function close_c

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

end program test_batched_transform_mp
