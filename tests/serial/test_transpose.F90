!===============================================================================
! test_transpose.F90  --  GPU validation of transpose_mod's single-rank (P=1)
! no-op backend.  At P=1 layout A and layout B are the SAME layout, so both
! transpose_A_to_B and transpose_B_to_A must be the IDENTITY and BIT-EXACT
! (tol = 0) -- this is the property that makes increment (1) byte-identical to
! the current single-GPU code.
!
! Build & run (ON A GPU NODE; canonical: `bash run_all_tests.sh` from repo root):
!   nvfortran -module build -c src/core/types.F90
!   nvfortran -module build -c src/core/decomp_mod.F90
!   nvfortran -cuda -module build -c src/core/transpose_mod.F90
!   nvfortran -cuda -cudalib=cufft -module build tests/serial/test_transpose.F90 \
!             build/transpose_mod.o build/decomp_mod.o build/types.o -o build/test_transpose
!   ./build/test_transpose
!
! Tests (tiny grid nx=16 -> nkx=9, ny=33, nz=8; P=1 so a_nz_loc=8, b_nkx_loc=9):
!   * descriptor P=1 invariant a_nz_loc==nz and b_nkx_loc==nkx (precondition for
!     a bit-exact no-op copy);
!   * A->B is the identity (Bh == Ah, same shape at P=1), BIT-EXACT;
!   * round-trip A->B->A recovers the input, BIT-EXACT (tol = 0);
!   * NaN-hardened (all(R==R)) and nonzero-guarded (max|A|>0, so a vacuous 0==0
!     cannot pass);
!   * nfield = 1 AND nfield > 1 (the batched path).
!
! NOTE: the P>1 branch is a deliberate `error stop` (the real all-to-all lands
! with the MPI scaffold, M1).  It is therefore NOT invoked here -- doing so would
! abort the suite; it will be exercised once the MPI backend exists.
!===============================================================================
program test_transpose
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,         only: tp, ip
  use decomp_mod,    only: decomp_t, decomp_setup
  use transpose_mod, only: transpose_A_to_B, transpose_B_to_A
  use cudafor
  implicit none

  logical :: all_ok = .true.
  logical :: ok

  call run_roundtrip(1_ip, ok);  call check('P=1 no-op A->B->A round-trip, nfield=1', ok)
  call run_roundtrip(4_ip, ok);  call check('P=1 no-op A->B->A round-trip, nfield=4', ok)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine check(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. okk) all_ok = .false.
  end subroutine check

  subroutine run_roundtrip(nf, okout)
    integer(ip), intent(in)  :: nf
    logical,     intent(out) :: okout
    type(decomp_t) :: d
    integer(ip), parameter :: nx = 16, ny = 33, nz = 8
    integer(ip) :: nkx, a_nz_loc, b_nkx_loc
    integer     :: istat
    real(real64) :: maxd, maxa
    complex(tp), allocatable :: Ah(:,:,:,:), Bh(:,:,:,:), Rh(:,:,:,:)
    real(tp),    allocatable :: re(:,:,:,:), im(:,:,:,:)
    complex(tp), allocatable, device :: A_d(:,:,:,:), B_d(:,:,:,:), R_d(:,:,:,:)

    call decomp_setup(d, 0_ip, 1_ip, 0_ip, nx, ny, nz)
    nkx = d%nkx;  a_nz_loc = d%a_nz_loc;  b_nkx_loc = d%b_nkx_loc
    okout = .true.
    ! P=1 precondition for a bit-exact no-op: local slab == global extent
    if (a_nz_loc  /= nz)  okout = .false.
    if (b_nkx_loc /= nkx) okout = .false.

    allocate(Ah(ny, nkx, a_nz_loc, nf), Bh(ny, b_nkx_loc, nz, nf), Rh(ny, nkx, a_nz_loc, nf))
    allocate(re(ny, nkx, a_nz_loc, nf), im(ny, nkx, a_nz_loc, nf))
    allocate(A_d(ny, nkx, a_nz_loc, nf), B_d(ny, b_nkx_loc, nz, nf), R_d(ny, nkx, a_nz_loc, nf))

    call random_number(re);  call random_number(im)
    Ah = cmplx(re, im, tp)

    A_d = Ah
    call transpose_A_to_B(d, ny, nkx, nz, a_nz_loc, b_nkx_loc, nf, A_d, B_d)
    call transpose_B_to_A(d, ny, nkx, nz, a_nz_loc, b_nkx_loc, nf, B_d, R_d)
    istat = cudaDeviceSynchronize()
    Bh = B_d
    Rh = R_d

    ! nonzero guard: input is not all-zero, so bit-exact 0-difference is meaningful
    maxa = real(maxval(abs(Ah)), real64)
    if (.not. (maxa > 0.0_real64)) okout = .false.
    ! NaN-harden: any NaN makes all(X==X) false
    if (.not. all(Rh == Rh)) okout = .false.
    if (.not. all(Bh == Bh)) okout = .false.
    ! A->B is the identity at P=1 (Bh and Ah share shape here): bit-exact
    maxd = real(maxval(abs(Bh - Ah)), real64)
    if (maxd /= 0.0_real64) okout = .false.
    ! round-trip A->B->A: bit-exact recovery
    maxd = real(maxval(abs(Rh - Ah)), real64)
    if (maxd /= 0.0_real64) okout = .false.

    deallocate(Ah, Bh, Rh, re, im, A_d, B_d, R_d)
  end subroutine run_roundtrip

end program test_transpose
