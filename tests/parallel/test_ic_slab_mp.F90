!===============================================================================
! test_ic_slab_mp.F90 -- slab-local IC identity gate.
!
! THE CONTRACT UNDER TEST: ic_fill_slab's counter-based RNG makes every spectral
! coefficient a pure function of (seed, GLOBAL kx, folded kz, m, component), so
! filling any kx-slab produces BIT-IDENTICALLY the corresponding slice of a
! full-grid fill -- independent of rank count, loop order, and which rank runs
! it. That is the property that lets the production driver build only its slab
! (the old replicated build OOMed any fresh start >~1 G pts).
!
! CHECKS (host-only; dp path -- the IC is dp regardless of the tp tier):
!   1. slab == full-grid slice, BIT-EXACT (v and eta), at whatever P is launched
!   2. sum over ranks of the slab partial energies == full-grid E2 (rel 1e-12;
!      association order differs -> roundoff, not bit, equality)
!   3. kx=0 plane is EXACTLY z-Hermitian after phase 2 (pairs are conjugate by
!      construction: (a+conj(b))/2 vs (b+conj(a))/2 are IEEE-exact conjugates)
!   4. re-running the slab fill reproduces itself bit-for-bit (purity)
!
! Run at P=1 (slab == full, trivially) and P>1 (the real decomposition test).
!===============================================================================
program test_ic_slab_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,      only: ip, dp
  use par_mod
  use decomp_mod, only: decomp_t, decomp_setup
  use init_mod,   only: ic_fill_slab
  implicit none

  integer(ip), parameter :: n = 32, nx = 32, nz = 24, kcut = 4, seed = 777
  integer(ip), parameter :: nkx = nx/2 + 1
  logical :: all_ok = .true.
  type(decomp_t) :: d
  integer(ip) :: kx0, bnkx, iz, icz
  real(dp) :: kxh(nkx), kzh(nz)
  real(dp) :: E2F, E2S, dmax
  complex(dp), allocatable :: hvF(:,:,:), heF(:,:,:)      ! full-grid fill
  complex(dp), allocatable :: hvS(:,:,:), heS(:,:,:)      ! this rank's slab fill
  complex(dp), allocatable :: hvS2(:,:,:), heS2(:,:,:)    ! purity re-run
  integer(ip) :: ix, mz

  call par_init()
  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, n+1, nz)
  kx0 = d%b_kx0;  bnkx = d%b_nkx_loc
  if (par_rank == 0) print '(a,i0,a)', '[ic-slab] world size = ', par_nproc, ' rank(s)'

  ! wavenumbers for a 2pi x 2pi box: kx = 0..nkx-1, kz folded
  do ix = 1, nkx
     kxh(ix) = real(ix-1, dp)
  end do
  do iz = 1, nz
     mz = iz-1;  if (mz > nz/2) mz = mz - nz
     kzh(iz) = real(mz, dp)
  end do

  ! full-grid fill (reference; every rank computes it -- small grid)
  allocate(hvF(0:n,nkx,nz), heF(0:n,nkx,nz))
  call ic_fill_slab(n, nkx, nz, 0_ip, nkx, kxh, kzh, seed, kcut, hvF, heF, E2F)

  ! this rank's slab fill
  allocate(hvS(0:n,bnkx,nz), heS(0:n,bnkx,nz))
  call ic_fill_slab(n, nkx, nz, kx0, bnkx, kxh, kzh, seed, kcut, hvS, heS, E2S)

  ! 1. slab == full slice, BIT-EXACT
  dmax = max(maxval(abs(hvS - hvF(:, kx0+1:kx0+bnkx, :))), &
             maxval(abs(heS - heF(:, kx0+1:kx0+bnkx, :))))
  call check('slab == full-grid slice (bit-exact)', dmax == 0.0_dp)

  ! 2. global energy: sum of slab partials == full E2 (roundoff-level)
  call par_reduce_sum(E2S)
  call check('sum(E2 partials) == full E2 (rel 1e-12)', &
             abs(E2S - E2F) <= 1.0e-12_dp*max(E2F, 1.0e-300_dp) .and. E2F > 0.0_dp)

  ! 3. kx=0 plane exactly z-Hermitian (checked on the full field, every rank)
  dmax = 0.0_dp
  do iz = 1, nz
     icz = mod(nz-(iz-1), nz) + 1
     dmax = max(dmax, maxval(abs(hvF(:,1,iz) - conjg(hvF(:,1,icz)))), &
                      maxval(abs(heF(:,1,iz) - conjg(heF(:,1,icz)))))
  end do
  call check('kx=0 plane z-Hermitian (exact)', dmax == 0.0_dp)

  ! 4. purity: a second slab fill is bit-identical to the first
  allocate(hvS2(0:n,bnkx,nz), heS2(0:n,bnkx,nz))
  call ic_fill_slab(n, nkx, nz, kx0, bnkx, kxh, kzh, seed, kcut, hvS2, heS2, E2F)
  dmax = max(maxval(abs(hvS2 - hvS)), maxval(abs(heS2 - heS)))
  call check('re-fill reproduces itself (bit-exact)', dmax == 0.0_dp)

  call par_reduce_and(all_ok)
  if (par_rank == 0) then
     if (all_ok) then
        print '(a)', 'ALL TESTS PASSED'
     else
        print '(a)', 'SOME TESTS FAILED'
     end if
  end if

  call par_finalize()
  if (.not. all_ok) error stop 1

contains

  subroutine check(name, ok)
    character(*), intent(in) :: name
    logical,      intent(in) :: ok
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', ok)
    print '(a,i0,a)', '[', par_rank, '] ['//tag//'] '//name
    if (.not. ok) all_ok = .false.
  end subroutine check

end program test_ic_slab_mp
