!===============================================================================
! test_spectral3d_mp.F90  --  multi-GPU: validate the DISTRIBUTED dealiased
! transform (par_spectral3d_mod) against the single-GPU oracle (spectral3d_mod),
! across whatever world size the launcher provides.
!
! Each rank, on the SAME deterministic field (an integer-hash function of the
! GLOBAL physical index -> no rank-0 broadcast, identical on every rank):
!   ORACLE (full, single-GPU, redundant per rank):
!     Pfull --to_spectral--> Afull ; zero x/z Nyquist (solver invariant) ;
!     Afull --to_physical--> Por (reference physical) ;
!     Por   --to_spectral--> Afull2 (reference forward).
!   DISTRIBUTED (this rank's slabs):
!     (1) INVERSE  : A_slab = Afull's kx-slab  --par_to_physical-->  Pdist
!                    compare Pdist to Por's z-slab            [vs oracle]
!     (2) FORWARD  : Por's z-slab  --par_to_spectral-->  Aout_slab
!                    compare Aout_slab to Afull2's kx-slab    [vs oracle, mode-by-mode]
!     (3) ROUND-TRIP: A_slab -> inverse -> forward == (nxd*nzd)*A_slab
! par_reduce_and makes any rank's failure fail the whole job. Because the oracle
! is computed identically on every rank, "distributed(P) == oracle" simultaneously
! checks correctness, P-independence, and (run at P=1 and P=4) P=1<->P=4 identity.
! The 1-D-vs-fused transform is bit-exact, so agreement should be
! at/near 0; the tolerance is the round-off budget (1e-9 dp / 1e-3 sp).
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_spectral3d_mp   Run: bash run_all_par_tests.sh run -k test_spectral3d_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_spectral3d_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,             only: tp, ip
  use cudafor
  use decomp_mod,        only: decomp_t, decomp_setup
  use spectral3d_mod,    only: s3d_plans, s3d_setup_da_r2c, s3d_destroy_r2c, &
                               to_physical_da_r2c, to_spectral_da_r2c
  use par_spectral3d_mod,only: par_s3d_plans, par_s3d_setup_da_r2c, &
                               par_s3d_destroy_da_r2c, &
                               par_to_physical_da_r2c, par_to_spectral_da_r2c
  use par_mod
  implicit none

  integer(ip), parameter :: nx = 32, ny = 6, nz = 16
  integer(ip), parameter :: nxd = 3*nx/2, nzd = 3*nz/2
  integer(ip), parameter :: nkx = nx/2+1, nkxd = nxd/2+1
  real(real64) :: tol
  logical :: all_ok, ok1, ok2, ok3
  integer :: istat
  integer(ip) :: iy, ix, iz
  integer(ip) :: bnkx, anz, kxlo, kxhi, zlo, zhi

  type(decomp_t)      :: d
  type(s3d_plans)     :: orpl       ! single-GPU oracle plans
  type(par_s3d_plans) :: pp         ! distributed plans

  ! oracle (full) buffers
  real(tp),    allocatable :: Pfull_h(:,:,:), Por_h(:,:,:), Preal_h(:,:,:)
  complex(tp), allocatable :: Afull_h(:,:,:), Afull2_h(:,:,:)
  real(tp),    allocatable, device :: Pfull_d(:,:,:), Por_d(:,:,:)
  complex(tp), allocatable, device :: Afull_d(:,:,:), Afull2_d(:,:,:), Wf_or(:,:,:)
  ! distributed (slab) buffers
  complex(tp), allocatable :: A_slab_h(:,:,:), Aout_slab_h(:,:,:), Art_slab_h(:,:,:)
  real(tp),    allocatable :: Pdist_h(:,:,:)
  complex(tp), allocatable, device :: A_slab_d(:,:,:), Aout_slab_d(:,:,:), Art_slab_d(:,:,:)
  real(tp),    allocatable, device :: Pdist_d(:,:,:)
  complex(tp), allocatable, device :: Bz_d(:,:,:,:), Az_d(:,:,:,:), Af_d(:,:,:)

  call par_init()
  tol = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)
  if (par_rank == 0) print '(a,i0,a)', '[spectral-mp] world size = ', par_nproc, ' rank(s)'

  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
  bnkx = d%b_nkx_loc;   kxlo = d%b_kx0 + 1;  kxhi = d%b_kx0 + bnkx   ! Layout-B kx-slab
  anz  = d%ad_nz_loc;   zlo  = d%ad_z0 + 1;  zhi  = d%ad_z0 + anz    ! Layout-A dealiased-z slab

  allocate(Pfull_h(ny,nxd,nzd), Por_h(ny,nxd,nzd), Preal_h(ny,nxd,anz))
  allocate(Afull_h(ny,nkx,nz), Afull2_h(ny,nkx,nz))
  allocate(Pfull_d(ny,nxd,nzd), Por_d(ny,nxd,nzd))
  allocate(Afull_d(ny,nkx,nz), Afull2_d(ny,nkx,nz), Wf_or(ny,nkxd,nzd))
  allocate(A_slab_h(ny,bnkx,nz), Aout_slab_h(ny,bnkx,nz), Art_slab_h(ny,bnkx,nz))
  allocate(Pdist_h(ny,nxd,anz))
  allocate(A_slab_d(ny,bnkx,nz), Aout_slab_d(ny,bnkx,nz), Art_slab_d(ny,bnkx,nz))
  allocate(Pdist_d(ny,nxd,anz))
  allocate(Bz_d(ny,bnkx,nzd,1), Az_d(ny,nkx,anz,1), Af_d(ny,nkxd,anz))

  ! --- deterministic real physical field (identical on every rank) ------------
  do iz = 1, nzd; do ix = 1, nxd; do iy = 1, ny
     Pfull_h(iy,ix,iz) = pval(iy, ix, iz)
  end do; end do; end do

  ! ===== ORACLE (single-GPU, full field, redundant per rank) ==================
  call s3d_setup_da_r2c(orpl, ny, nx, nz)
  Pfull_d = Pfull_h
  call to_spectral_da_r2c(orpl, ny, nx, nz, nxd, nzd, Pfull_d, Wf_or, Afull_d)
  istat = cudaDeviceSynchronize()
  Afull_h = Afull_d
  Afull_h(:, nx/2+1, :) = (0.0_tp, 0.0_tp)        ! x-Nyquist (solver invariant)
  Afull_h(:, :, nz/2+1) = (0.0_tp, 0.0_tp)        ! z-Nyquist
  Afull_d = Afull_h
  call to_physical_da_r2c(orpl, ny, nx, nz, nxd, nzd, Afull_d, Por_d, Wf_or)
  istat = cudaDeviceSynchronize()
  Por_h = Por_d                                    ! reference physical
  Por_d = Por_h                                    ! (restore: cheb consumed Afull_d, not Por_d)
  call to_spectral_da_r2c(orpl, ny, nx, nz, nxd, nzd, Por_d, Wf_or, Afull2_d)
  istat = cudaDeviceSynchronize()
  Afull2_h = Afull2_d                              ! reference forward

  ! ===== DISTRIBUTED ==========================================================
  call par_s3d_setup_da_r2c(pp, d, ny, nx, nz)

  ! (1) INVERSE: distributed physical (this z-slab) == oracle physical (this z-slab)
  A_slab_h = Afull_h(:, kxlo:kxhi, :)
  A_slab_d = A_slab_h
  call par_to_physical_da_r2c(pp, d, ny, nx, nz, nxd, nzd, bnkx, anz, &
                              A_slab_d, Bz_d, Az_d, Af_d, Pdist_d)
  istat = cudaDeviceSynchronize()
  Pdist_h = Pdist_d
  ok1 = close_r(Pdist_h, Por_h(:, :, zlo:zhi), tol) .and. (maxval(abs(A_slab_h)) > 0.0_tp)

  ! (2) FORWARD: distributed spectral (this kx-slab) == oracle forward (this kx-slab)
  Preal_h = Por_h(:, :, zlo:zhi)                   ! oracle physical, this z-slab
  Pdist_d = Preal_h                                ! reuse Pdist_d as the forward input
  call par_to_spectral_da_r2c(pp, d, ny, nx, nz, nxd, nzd, bnkx, anz, &
                              Pdist_d, Af_d, Az_d, Bz_d, Aout_slab_d)
  istat = cudaDeviceSynchronize()
  Aout_slab_h = Aout_slab_d
  ok2 = close_c(Aout_slab_h, Afull2_h(:, kxlo:kxhi, :), tol)

  ! (3) ROUND-TRIP: forward(inverse(A_slab)) == (nxd*nzd) * A_slab
  A_slab_d = A_slab_h                              ! restore (cheb consumes the input)
  call par_to_physical_da_r2c(pp, d, ny, nx, nz, nxd, nzd, bnkx, anz, &
                              A_slab_d, Bz_d, Az_d, Af_d, Pdist_d)
  call par_to_spectral_da_r2c(pp, d, ny, nx, nz, nxd, nzd, bnkx, anz, &
                              Pdist_d, Af_d, Az_d, Bz_d, Art_slab_d)
  istat = cudaDeviceSynchronize()
  Art_slab_h = Art_slab_d
  ok3 = close_c(Art_slab_h, A_slab_h*real(nxd*nzd, tp), tol*real(nxd*nzd, real64))

  call par_s3d_destroy_da_r2c(pp)
  call s3d_destroy_r2c(orpl)

  ! --- global AND of each check; rank 0 reports -------------------------------
  call par_reduce_and(ok1)
  call par_reduce_and(ok2)
  call par_reduce_and(ok3)
  all_ok = ok1 .and. ok2 .and. ok3
  if (par_rank == 0) then
     call report('inverse: distributed phys == single-GPU oracle (per z-slab)', ok1)
     call report('forward: distributed spec == single-GPU oracle (per kx-slab)', ok2)
     call report('round-trip: forward(inverse(A)) == (nxd*nzd)*A', ok3)
     if (all_ok) then
        print '(a)', 'ALL TESTS PASSED'
     else
        print '(a)', 'SOME TESTS FAILED'
     end if
  end if

  call par_finalize()
  if (.not. all_ok) error stop 1

contains

  ! deterministic field value at GLOBAL physical indices (iy, ix, iz) ----------
  real(tp) function pval(iy, ix, iz) result(v)
    integer(ip), intent(in) :: iy, ix, iz
    integer(ip) :: h
    h = modulo( iy*1009_ip + ix*2027_ip + iz*3001_ip + ix*iz*17_ip, 100003_ip )
    v = real(h, tp)/100003.0_tp - 0.5_tp
  end function pval

  logical function close_r(a, b, tol) result(ok)
    real(tp),     intent(in) :: a(:,:,:), b(:,:,:)
    real(real64), intent(in) :: tol
    ok = all(a == a) .and. all(b == b) .and. &
         (real(maxval(abs(a - b)), real64) <= tol)
  end function close_r

  logical function close_c(a, b, tol) result(ok)
    complex(tp),  intent(in) :: a(:,:,:), b(:,:,:)
    real(real64), intent(in) :: tol
    ok = all(a == a) .and. all(b == b) .and. &
         (real(maxval(abs(a - b)), real64) <= tol)
  end function close_c

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

end program test_spectral3d_mp
