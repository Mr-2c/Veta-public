! test_diag.F90  --  unit test for diag_mod.
! CPU-only. Validates the PURE byte-accounting and prime-factor helpers that
! back the startup memory report + FFT-size guard. No GPU needed.
!
! Pure CPU (no GPU needed). Build:
!   nvfortran -c types.F90          ! provides the `kinds` module (-> kinds.mod, types.o)
!   nvfortran -c diag_mod.F90
!   nvfortran -c test_diag.F90
!   nvfortran -o test_diag test_diag.o diag_mod.o types.o
!
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo root
!  (sources in src/, output to build/). The runner auto-discovers this as
!  tests/test_*.F90 and links it against the module objects; for that to work,
!  diag_mod must appear in the runner's module compile list (it does, right
!  after types) and in the Makefile MODS/CPU_MODS lists. Runs on any node.
!
! Strategy:
!   (A) Hand-computed exact bytes for the default grid (n=32,nkx=9,nz=16,nxd=nzd=24)
!       at all-real64 -- catches any formula/inventory-count error.
!   (B) Tier invariants: fp32 transforms (tp=4) shrink ONLY the transform tier;
!       fp32 fields (tp=wp=4) additionally shrink fields + stepper-assembly; the
!       dp tiers (stepper-solve, factorization, misc) are invariant across all
!       three. This validates the per-array TIER assignment, not just the
!       arithmetic.
!   (C) Totals strictly decrease real64 > fp32 transforms > fp32 fields.
!   (D) largest_prime_factor on known values.
program test_diag
  use kinds,           only: dp, ip
  use iso_fortran_env, only: int64
  use diag_mod,        only: mem_category_bytes, largest_prime_factor
  implicit none
  integer(ip), parameter :: n=32, nkx=9, nz=16, nxd=24, nzd=24
  integer(int64) :: bt,bf,bsw,bsd,bfa,bm                 ! real64 column
  integer(int64) :: bt2,bf2,bsw2,bsd2,bfa2,bm2           ! fp32-transform column
  integer(int64) :: bt3,bf3,bsw3,bsd3,bfa3,bm3           ! fp32-field column
  integer(int64) :: tot1, tot2, tot3
  logical :: ok
  ok = .true.

  write(*,'(a)') '=== diag_mod: device-memory accounting + FFT-size guard ==='

  ! ---- (A) exact bytes at all-real64 for the default grid --------------------
  ! nc=33 ns=31 nkxd=13 n2=64 ; full=33*9*16=4752 shen=31*9*16=4464
  ! phys=33*24*24=19008 fine=33*13*24=10296 plane=144 mean=33.
  ! complex=2*8 bytes, real=8 bytes. transform adds cdtp(2 cfull) +
  ! gc_buf(64*9*16=9216 cplx) -- NO xl terms (parallel-only allocations);
  ! step-asm adds cdwp(2 cfull); step-solve adds hb_fcl/ucl (2 cshen).
  call mem_category_bytes(n,nkx,nz,nxd,nzd, 8,8,8, bt,bf,bsw,bsd,bfa,bm)
  call icheck('transform  (real64)', bt,  2745416_int64, ok)   ! 8*(2*12*4752+9*19008+2*10296 +2*2*4752+2*9216 +25)
  call icheck('fields     (real64)', bf,   459360_int64, ok)   ! 8*(2*6*4752) + 8*(2*6*33)
  call icheck('step-asm   (real64)', bsw,  609312_int64, ok)   ! 8*(2*(6+2)*4752 + 2*2*33)
  call icheck('step-solve (real64)', bsd,  468864_int64, ok)   ! 8*(2*(3+2)*4464+3*4464+2*2*144)
  call icheck('factor     (real64)', bfa,  241920_int64, ok)   ! 8*(2*4752*3 + 4*144*3)
  call icheck('misc       (real64)', bm,     1208_int64, ok)   ! 8*((25)+3*31+33)

  ! ---- (B) fp32 transforms (tp=4): only the transform tier changes -----------
  call mem_category_bytes(n,nkx,nz,nxd,nzd, 4,8,8, bt2,bf2,bsw2,bsd2,bfa2,bm2)
  call icheck('transform halves @ tp=4', bt2, bt/2_int64, ok)  ! transform is pure tp => exact halving
  call icheck('fields unchanged (DT)',   bf2,  bf,  ok)
  call icheck('step-asm unchanged (DT)', bsw2, bsw, ok)
  call icheck('step-solve unchanged(DT)',bsd2, bsd, ok)
  call icheck('factor unchanged (DT)',   bfa2, bfa, ok)
  call icheck('misc unchanged (DT)',     bm2,  bm,  ok)

  ! ---- (B) fp32 fields (tp=wp=4): transform + fields-wp + step-asm shrink ----
  call mem_category_bytes(n,nkx,nz,nxd,nzd, 4,4,8, bt3,bf3,bsw3,bsd3,bfa3,bm3)
  call icheck('transform == DT (RB)',    bt3,  bt2, ok)
  call icheck('fields wp-half (RB)',     bf3,  4_int64*(2_int64*6_int64*4752_int64) &
                                              + 8_int64*(2_int64*6_int64*33_int64), ok)
  call icheck('step-asm halves (RB)',    bsw3, 4_int64*(2_int64*8_int64*4752_int64 &
                                              + 2_int64*2_int64*33_int64), ok)   ! (6 stepper + 2 cdwp) cfull
  call icheck('step-solve unchanged(RB)',bsd3, bsd, ok)        ! pure dp
  call icheck('factor unchanged (RB)',   bfa3, bfa, ok)        ! pure dp
  call icheck('misc unchanged (RB)',     bm3,  bm,  ok)        ! pure dp

  ! ---- (C) totals strictly decrease real64 > fp32 transforms > fp32 fields ---
  tot1 = bt +bf +bsw +bsd +bfa +bm
  tot2 = bt2+bf2+bsw2+bsd2+bfa2+bm2
  tot3 = bt3+bf3+bsw3+bsd3+bfa3+bm3
  if (.not. (tot1 > tot2 .and. tot2 > tot3)) then
     write(*,'(a,3(i0,1x))') '  [FAIL] totals not strictly decreasing: ', tot1, tot2, tot3
     ok = .false.
  else
     write(*,'(a,3(i0,1x),a)') '  [ok] totals real64>DT>RB: ', tot1, tot2, tot3, '(bytes)'
  end if

  ! ---- (D) largest_prime_factor ---------------------------------------------
  call icheck('lpf(1)',   int(largest_prime_factor(1_ip),int64),   1_int64, ok)
  call icheck('lpf(2)',   int(largest_prime_factor(2_ip),int64),   2_int64, ok)
  call icheck('lpf(7)',   int(largest_prime_factor(7_ip),int64),   7_int64, ok)
  call icheck('lpf(8)',   int(largest_prime_factor(8_ip),int64),   2_int64, ok)
  call icheck('lpf(24)',  int(largest_prime_factor(24_ip),int64),  3_int64, ok)   ! 2^3*3
  call icheck('lpf(98)',  int(largest_prime_factor(98_ip),int64),  7_int64, ok)   ! 2*7^2
  call icheck('lpf(100)', int(largest_prime_factor(100_ip),int64), 5_int64, ok)   ! 2^2*5^2
  call icheck('lpf(105)', int(largest_prime_factor(105_ip),int64), 7_int64, ok)   ! 3*5*7
  call icheck('lpf(121)', int(largest_prime_factor(121_ip),int64),11_int64, ok)   ! 11^2 (not 7-smooth)

  if (ok) then
     write(*,'(/,a)') 'ALL TESTS PASSED'
  else
     write(*,'(/,a)') 'SOME TESTS FAILED'
  end if

contains
  subroutine icheck(label, got, want, ok)
    character(*),   intent(in)    :: label
    integer(int64), intent(in)    :: got, want
    logical,        intent(inout) :: ok
    if (got == want) then
       write(*,'(a,a,a,i0)') '  [ok]   ', label, ' = ', got
    else
       write(*,'(a,a,a,i0,a,i0)') '  [FAIL] ', label, ' got ', got, ' want ', want
       ok = .false.
    end if
  end subroutine icheck
end program test_diag
