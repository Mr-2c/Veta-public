!===============================================================================
! test_par_phys_r2c_mp.F90 -- snapshot path: validate the DISTRIBUTED CLEAN-grid
! inverse transform (par_to_physical_r2c, nx*nz un-dealiased) against the single-GPU
! oracle (spectral3d_mod's to_physical_r2c), across whatever world size is launched.
!
! Each rank, on the SAME deterministic field (hash of the GLOBAL physical index ->
! identical on every rank, no broadcast):
!   ORACLE (full, single-GPU, redundant per rank):
!     Pfull --to_spectral_r2c--> Afull ; zero x/z Nyquist (solver invariant) ;
!     Afull --to_physical_r2c--> Por  (reference clean physical, ny,nx,nz)
!   DISTRIBUTED (this rank's kx-slab in, this rank's COARSE z-slab out):
!     A_slab = Afull's kx-slab  --par_to_physical_r2c-->  Pdist (ny,nx,a_nz_loc)
!     compare Pdist to Por's coarse z-slab.
! Because the oracle is identical on every rank, "distributed(P)==oracle" checks
! correctness, P-independence, and (run at P=1 and P=4) P=1<->P=4 identity at once.
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_par_phys_r2c_mp   Run: bash run_all_par_tests.sh run -k test_par_phys_r2c_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_par_phys_r2c_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,              only: tp, ip
  use cudafor
  use decomp_mod,         only: decomp_t, decomp_setup
  use spectral3d_mod,     only: s3d_plans, s3d_setup_r2c, s3d_destroy_r2c, &
                                to_physical_r2c, to_spectral_r2c
  use par_spectral3d_mod, only: par_s3d_plans, par_s3d_setup_da_r2c, &
                                par_s3d_destroy_da_r2c, par_to_physical_r2c
  use fft3d_mod,          only: xl_release_current   ! eviction path
  use par_mod
  implicit none

  integer(ip), parameter :: nx = 32, ny = 6, nz = 16
  integer(ip), parameter :: nkx = nx/2+1
  real(real64) :: tol
  logical :: all_ok, ok1, ok2
  integer :: istat
  integer(ip) :: iy, ix, iz, bnkx, anz, kxlo, kxhi, zlo, zhi

  type(decomp_t)      :: d
  type(s3d_plans)     :: orpl
  type(par_s3d_plans) :: pp

  real(tp),    allocatable :: Pfull_h(:,:,:), Por_h(:,:,:), Pdist_h(:,:,:)
  complex(tp), allocatable :: Afull_h(:,:,:), A_slab_h(:,:,:)
  real(tp),    allocatable, device :: Pfull_d(:,:,:), Por_d(:,:,:), Pdist_d(:,:,:)
  complex(tp), allocatable, device :: Afull_d(:,:,:), A_slab_d(:,:,:)
  complex(tp), allocatable, device :: Bz_d(:,:,:,:), Az_d(:,:,:,:)

  call par_init()
  tol = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)
  if (par_rank == 0) print '(a,i0,a)', '[phys-r2c-mp] world size = ', par_nproc, ' rank(s)'

  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
  bnkx = d%b_nkx_loc;  kxlo = d%b_kx0 + 1;  kxhi = d%b_kx0 + bnkx     ! Layout-B kx-slab
  anz  = d%a_nz_loc;   zlo  = d%a_z0 + 1;   zhi  = d%a_z0 + anz       ! Layout-A COARSE z-slab

  allocate(Pfull_h(ny,nx,nz), Por_h(ny,nx,nz), Pdist_h(ny,nx,anz))
  allocate(Afull_h(ny,nkx,nz), A_slab_h(ny,bnkx,nz))
  allocate(Pfull_d(ny,nx,nz), Por_d(ny,nx,nz), Pdist_d(ny,nx,anz))
  allocate(Afull_d(ny,nkx,nz), A_slab_d(ny,bnkx,nz))
  allocate(Bz_d(ny,bnkx,nz,1), Az_d(ny,nkx,anz,1))

  ! --- deterministic real physical field (identical on every rank) ------------
  do iz = 1, nz; do ix = 1, nx; do iy = 1, ny
     Pfull_h(iy,ix,iz) = pval(iy, ix, iz)
  end do; end do; end do

  ! ===== ORACLE (single-GPU clean transform, full field, redundant per rank) ==
  call s3d_setup_r2c(orpl, ny, nx, nz)
  Pfull_d = Pfull_h
  call to_spectral_r2c(orpl, ny, nx, nz, Pfull_d, Afull_d)
  istat = cudaDeviceSynchronize()
  Afull_h = Afull_d
  Afull_h(:, nx/2+1, :) = (0.0_tp, 0.0_tp)        ! x-Nyquist (solver invariant)
  Afull_h(:, :, nz/2+1) = (0.0_tp, 0.0_tp)        ! z-Nyquist
  Afull_d = Afull_h
  call to_physical_r2c(orpl, ny, nx, nz, Afull_d, Por_d)
  istat = cudaDeviceSynchronize()
  Por_h = Por_d                                    ! reference clean physical

  ! ===== DISTRIBUTED clean inverse: this kx-slab in -> this z-slab out =========
  call par_s3d_setup_da_r2c(pp, d, ny, nx, nz)
  A_slab_h = Afull_h(:, kxlo:kxhi, :)
  A_slab_d = A_slab_h
  call par_to_physical_r2c(pp, d, ny, nx, nz, bnkx, anz, A_slab_d, Bz_d, Az_d, Pdist_d)
  istat = cudaDeviceSynchronize()
  Pdist_h = Pdist_d
  ok1 = close_r(Pdist_h, Por_h(:, :, zlo:zhi), tol) .and. (maxval(abs(A_slab_h)) > 0.0_tp)

  ! release the just-used xl slot, then redo the identical clean inverse:
  ! exercises the post-snapshot eviction path (release -> ensure_xl realloc) and
  ! requires the result to stay bit-comparable to the oracle.
  call xl_release_current()
  Pdist_d = 0.0_tp
  A_slab_d = A_slab_h                              ! transform consumes its input
  call par_to_physical_r2c(pp, d, ny, nx, nz, bnkx, anz, A_slab_d, Bz_d, Az_d, Pdist_d)
  istat = cudaDeviceSynchronize()
  Pdist_h = Pdist_d
  ok2 = close_r(Pdist_h, Por_h(:, :, zlo:zhi), tol)

  call par_s3d_destroy_da_r2c(pp)
  call s3d_destroy_r2c(orpl)

  call par_reduce_and(ok1)
  call par_reduce_and(ok2)
  all_ok = ok1 .and. ok2
  if (par_rank == 0) then
     call report('clean inverse: distributed phys == single-GPU oracle (per z-slab)', ok1)
     call report('xl slot release -> realloc: clean inverse still matches oracle', ok2)
     if (all_ok) then; print '(a)', 'ALL TESTS PASSED'
     else;             print '(a)', 'SOME TESTS FAILED'; end if
  end if

  call par_finalize()
  if (.not. all_ok) error stop 1

contains

  real(tp) function pval(iy, ix, iz) result(v)
    integer(ip), intent(in) :: iy, ix, iz
    integer(ip) :: h
    h = modulo( iy*1009_ip + ix*2027_ip + iz*3001_ip + ix*iz*17_ip, 100003_ip )
    v = real(h, tp)/100003.0_tp - 0.5_tp
  end function pval

  logical function close_r(a, b, tol) result(ok)
    real(tp),     intent(in) :: a(:,:,:), b(:,:,:)
    real(real64), intent(in) :: tol
    ok = all(a == a) .and. all(b == b) .and. (real(maxval(abs(a - b)), real64) <= tol)
  end function close_r

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

end program test_par_phys_r2c_mp
