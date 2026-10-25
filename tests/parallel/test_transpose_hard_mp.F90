!===============================================================================
! test_transpose_hard_mp.F90  --  [test-hardening G1] stress the real NCCL A<->B
! transpose (par_transpose_mod) on a grid + world sizes that exercise the UNEVEN
! dealiased-z slab path the shipped test_par_transpose (nkx=17, nz=16 -> nzd=24,
! divisible by 1/2/4/8) never hits on hardware:
!
!   G1  UNEVEN dealiased-z slab through the transpose. Every shipped _mp test has
!       nzd divisible by P, so the uneven-z count/displacement + pack/unpack path
!       was only ever run in the integer model (test_decomp / numpy), never on
!       device. This test runs it for real.
!
! Grid: nx=24 (nkx=13, prime), ny=8, nz=23 (prime; nzd=3*nz/2=34). nkx AND BOTH
! z-extents (coarse nz, dealiased nzd) are non-divisible by the rank counts, so
! EVERY decomposed axis is split unevenly. (This is a transpose-only test -- pure
! data movement, no FFT/Nyquist -- so nz need not be even; an odd prime nz simply
! maximises the irregularity of the index/count arithmetic under test.) With the
! STANDARD launches:
!   P=4 : nz=23 -> [6,6,6,5]  nzd=34 -> [9,9,8,8]  nkx=13 -> [4,3,3,3]   (all uneven)
!   P=8 : nz=23 -> [3,3,3,3,3,3,3,2]  nzd=34 -> [5,5,4,4,4,4,4,4]  nkx=13 -> [2,2,2,2,2,1,1,1]
!   P=1 : trivial identity.
! Every rank stays NON-EMPTY for P<=min(nkx,nz,nzd)=13, respecting
! par_spectral3d_mod's documented precondition.
! It transposes at BOTH the coarse (nz, a_nz_loc) and the dealiased (nzd,
! ad_nz_loc) z-extents -- the latter is exactly what par_spectral3d_mod uses
! (par_transpose_*(d, ny, nkx, nzd, ad_nz_loc, b_nkx_loc, nf)).
!
! NOTE: EMPTY ranks (P > min(nkx,nzd)) are intentionally NOT exercised here --
! par_spectral3d_mod documents "Empty-rank handling intentionally NOT added" and
! the rank counts in use stay far below the slab ceiling, so empty-rank support
! is not a tested path. The zero-size guards
! below are defensive only.
!
! Correctness criterion: a transpose only MOVES data, so the distributed
! result must equal the serial layout EXACTLY (tol = 0). The field is a
! deterministic function of the GLOBAL index (an integer < 2^24, exact in sp), so
! any mis-mapped element is caught bit-exactly under tp = sp or dp. par_reduce_and
! makes any rank's failure fail the whole job.
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_transpose_hard_mp   Run: bash run_all_par_tests.sh run -k test_transpose_hard_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!   module load nvhpc openmpi nccl
! Build: TP=dp|sp bash scripts/build/build_par.sh test_transpose_hard_mp   Run: bash run_all_par_tests.sh run -k test_transpose_hard_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!   sbatch --nodes=1 --ntasks=1 parallel/run_transpose_hard.sbatch  # P=1
!   sbatch                        parallel/run_transpose_hard.sbatch  # P=4 (uneven)
!   sbatch --nodes=2 --ntasks=8   parallel/run_transpose_hard.sbatch  # P=8 (uneven)
!===============================================================================
program test_transpose_hard_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,             only: tp, ip
  use cudafor
  use decomp_mod,        only: decomp_t, decomp_setup
  use par_transpose_mod, only: par_transpose_A_to_B, par_transpose_B_to_A
  use par_mod
  implicit none

  integer(ip), parameter :: nx = 24, ny = 8, nz = 23      ! nkx = nx/2+1 = 13 (prime); nz prime
  integer(ip), parameter :: nkx_g = nx/2 + 1
  integer(ip), parameter :: nzd = 3*nz/2                   ! dealiased z = 34
  integer(ip), parameter :: nfs(3) = [ 1_ip, 3_ip, 6_ip ]

  type(decomp_t)    :: d
  logical           :: all_ok = .true., ok
  integer           :: jf
  integer(ip)       :: nf
  character(len=80) :: lbl

  call par_init()
  if (par_rank == 0) print '(a,i0,a)', '[transpose-hard] world size = ', par_nproc, ' rank(s)'
  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)

  do jf = 1, size(nfs)
     nf = nfs(jf)
     ! coarse-z transpose (extent nz, Layout-A coarse-z slab)
     ok = run_one(nz,  d%a_z0,  d%a_nz_loc,  nf)
     call par_reduce_and(ok);  if (.not. ok) all_ok = .false.
     if (par_rank == 0) then
        write(lbl,'(a,i0,a,i0)') 'A<->B bit-identity, coarse-z (nz=', nz, '), nfield=', nf
        call report(trim(lbl), ok)
     end if
     ! dealiased-z transpose (extent nzd, Layout-A dealiased-z slab) -- the path
     ! par_spectral3d_mod uses; THIS is where G1's uneven nzd lives.
     ok = run_one(nzd, d%ad_z0, d%ad_nz_loc, nf)
     call par_reduce_and(ok);  if (.not. ok) all_ok = .false.
     if (par_rank == 0) then
        write(lbl,'(a,i0,a,i0)') 'A<->B bit-identity, dealiased-z (nzd=', nzd, '), nfield=', nf
        call report(trim(lbl), ok)
     end if
  end do

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

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

  ! deterministic global field value at GLOBAL indices (iy,kxg,izg,f); zext is the
  ! z-extent (nz or nzd) so the encoding is unique within the chosen layout.
  complex(tp) function gval(iy, kxg, izg, f, zext) result(v)
    integer(ip), intent(in) :: iy, kxg, izg, f, zext
    integer(ip) :: L
    real(tp)    :: x
    L = (iy-1) + ny*((kxg-1) + nkx_g*((izg-1) + zext*(f-1)))
    x = real(L, tp)
    v = cmplx(x, -x, tp)
  end function gval

  ! one case at a given z-extent/slab: distributed transpose vs independent
  ! reference + round trip, bit-exact. Zero-size guards are defensive (no rank is
  ! empty for P<=min(nkx,nzd); empty-rank support is deferred -- see header).
  logical function run_one(zext, z0, znz, nf) result(good)
    integer(ip), intent(in) :: zext, z0, znz, nf
    integer(ip) :: kx0, bnkx
    integer(ip) :: iy, ikx, izl, izg, kxl, f
    integer     :: istat
    complex(tp), allocatable :: A_h(:,:,:,:), B_h(:,:,:,:), R_h(:,:,:,:), B_ref(:,:,:,:)
    complex(tp), allocatable, device :: A_d(:,:,:,:), B_d(:,:,:,:), R_d(:,:,:,:)

    good = .true.
    kx0 = d%b_kx0;  bnkx = d%b_nkx_loc          ! this rank's kx-slab (Layout B)

    allocate(A_h(ny,nkx_g,znz,nf), R_h(ny,nkx_g,znz,nf))
    allocate(B_h(ny,bnkx,zext,nf), B_ref(ny,bnkx,zext,nf))
    allocate(A_d(ny,nkx_g,znz,nf), R_d(ny,nkx_g,znz,nf), B_d(ny,bnkx,zext,nf))

    ! this rank's Layout-A slab from G (full kx, its z range)
    do f = 1, nf
       do izl = 1, znz
          do ikx = 1, nkx_g
             do iy = 1, ny
                A_h(iy,ikx,izl,f) = gval(iy, ikx, z0+izl, f, zext)
             end do
          end do
       end do
    end do
    ! independent reference Layout-B slab from G (its kx range, full z)
    do f = 1, nf
       do izg = 1, zext
          do kxl = 1, bnkx
             do iy = 1, ny
                B_ref(iy,kxl,izg,f) = gval(iy, kx0+kxl, izg, f, zext)
             end do
          end do
       end do
    end do

    A_d = A_h
    call par_transpose_A_to_B(d, ny, nkx_g, zext, znz, bnkx, nf, A_d, B_d)
    call par_transpose_B_to_A(d, ny, nkx_g, zext, znz, bnkx, nf, B_d, R_d)
    istat = cudaDeviceSynchronize()
    B_h = B_d;  R_h = R_d

    ! A->B vs independent reference (only where the B-slab is non-empty)
    if (bnkx > 0) then
       if (.not. all(B_h == B_h)) good = .false.                         ! NaN guard
       if (real(maxval(abs(B_h - B_ref)), real64) /= 0.0_real64) good = .false.
    end if
    ! A->B->A round trip + nonzero guard (only where the A-slab is non-empty)
    if (znz > 0) then
       if (.not. all(A_h == A_h)) good = .false.
       if (.not. all(R_h == R_h)) good = .false.
       if (real(maxval(abs(R_h - A_h)), real64) /= 0.0_real64) good = .false.
       if (.not. (real(maxval(abs(A_h)), real64) > 0.0_real64)) good = .false.
    end if

    deallocate(A_h, B_h, R_h, B_ref, A_d, B_d, R_d)
  end function run_one

end program test_transpose_hard_mp
