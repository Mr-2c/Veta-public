!===============================================================================
! test_dealias_c2c.F90  --  GPU validation of C2C split dealiasing (z direction).
!
! Build & run (ON A GPU NODE):
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -cudalib=cufft -c transforms_mod.F90
!   nvfortran -cuda -c dealias_mod.F90
!   nvfortran -cuda -cudalib=cufft test_dealias_c2c.F90 \
!             dealias_mod.o transforms_mod.o types.o -o test_dealias_c2c
!   ./test_dealias_c2c
!
! Coarse nz=8 (cuFFT freqs 0,1,2,3,4,-3,-2,-1), fine nzd=12.
!   u = e^{i 2x} + e^{-i x}   (modes +2 and -1),   v = e^{i 3x}  (mode +3).
!   u v = e^{i 5x} + e^{i 2x}: retained mode +2 (index 2), and +5 which aliases
!   to freq -3 (index 5) on the coarse grid.
! Tests:
!   1. pad placement: + modes at the front, - mode at the tail, zeros between.
!   2. pad -> trunc round-trip = identity.
!   3. product diagnostic |spectrum(5)|/|spectrum(2)|: dealiased ~0 (PASS),
!      non-dealiased ~1 (printed). The -1 mode in u exercises the split.
!===============================================================================

program test_dealias_c2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,          only: tp, ip
  use transforms_mod, only: tf_c2c_plan, tf_destroy, tf_c2c, CUFFT_FORWARD, CUFFT_INVERSE
  use dealias_mod,    only: pad_split, trunc_split
  use cudafor
  implicit none

  integer(ip), parameter :: nz = 8, nzd = 12
  real(real64) :: tol, ratio_clean, ratio_alias
  logical :: all_ok = .true.
  integer :: istat, plan_fc, plan_bc, plan_ff, plan_bf
  integer(ip) :: j

  complex(tp) :: Cu(0:nz-1,1), Cv(0:nz-1,1), Cufh(0:nzd-1,1), Cu2(0:nz-1,1), Wh(0:nz-1,1)
  complex(tp), device :: Cu_d(0:nz-1,1), Cv_d(0:nz-1,1), Cu2_d(0:nz-1,1)
  complex(tp), device :: Cuf_d(0:nzd-1,1), Cvf_d(0:nzd-1,1), Wf_d(0:nzd-1,1)
  complex(tp), device :: Wc_d(0:nz-1,1)
  complex(tp), device :: uf(0:nzd-1), vf(0:nzd-1), wf(0:nzd-1)
  complex(tp), device :: uc(0:nz-1), vc(0:nz-1), wc(0:nz-1)
  real(real64) :: err

  tol = merge(1.0e-10_real64, 1.0e-4_real64, tp == real64)

  Cu = (0.0_tp,0.0_tp);  Cu(2,1) = (1.0_tp,0.0_tp);  Cu(nz-1,1) = (1.0_tp,0.0_tp)  ! +2, -1
  Cv = (0.0_tp,0.0_tp);  Cv(3,1) = (1.0_tp,0.0_tp)                                  ! +3
  Cu_d = Cu;  Cv_d = Cv

  call tf_c2c_plan(plan_fc, nz,  1_ip);  call tf_c2c_plan(plan_bc, nz,  1_ip)
  call tf_c2c_plan(plan_ff, nzd, 1_ip);  call tf_c2c_plan(plan_bf, nzd, 1_ip)

  !-- 1. pad placement ------------------------------------------------------
  call pad_split(nz, nzd, 1_ip, Cu_d, Cuf_d)
  Cufh = Cuf_d
  err = 0.0_real64
  do j = 0, nzd-1
     if (j == 2) then
        err = max(err, abs(cmplx(Cufh(j,1),kind=real64) - (1.0_real64,0.0_real64)))
     else if (j == nzd-1) then                       ! freq -1 lands here
        err = max(err, abs(cmplx(Cufh(j,1),kind=real64) - (1.0_real64,0.0_real64)))
     else
        err = max(err, abs(cmplx(Cufh(j,1),kind=real64)))
     end if
  end do
  call check('pad_split placement', err, tol)

  !-- 2. pad -> trunc round-trip --------------------------------------------
  call trunc_split(nzd, nz, 1_ip, Cuf_d, Cu2_d)
  Cu2 = Cu2_d
  call check('pad_split -> trunc_split = I', real(maxval(abs(Cu2 - Cu)), real64), tol)

  !-- 3. product / aliasing diagnostic --------------------------------------
  ! dealiased
  call pad_split(nz, nzd, 1_ip, Cu_d, Cuf_d)
  call pad_split(nz, nzd, 1_ip, Cv_d, Cvf_d)
  call tf_c2c(plan_bf, Cuf_d, uf, CUFFT_INVERSE)
  call tf_c2c(plan_bf, Cvf_d, vf, CUFFT_INVERSE)
  istat = cudaDeviceSynchronize()
  !$cuf kernel do(1) <<<*,*>>>
  do j = 0, nzd-1
     wf(j) = uf(j)*vf(j)
  end do
  call tf_c2c(plan_ff, wf, Wf_d, CUFFT_FORWARD)
  call trunc_split(nzd, nz, 1_ip, Wf_d, Wc_d)
  istat = cudaDeviceSynchronize()
  Wh = Wc_d
  ratio_clean = real(abs(Wh(5,1)) / abs(Wh(2,1)), real64)

  ! non-dealiased
  call tf_c2c(plan_bc, Cu_d, uc, CUFFT_INVERSE)
  call tf_c2c(plan_bc, Cv_d, vc, CUFFT_INVERSE)
  istat = cudaDeviceSynchronize()
  !$cuf kernel do(1) <<<*,*>>>
  do j = 0, nz-1
     wc(j) = uc(j)*vc(j)
  end do
  call tf_c2c(plan_fc, wc, Wc_d, CUFFT_FORWARD)
  istat = cudaDeviceSynchronize()
  Wh = Wc_d
  ratio_alias = real(abs(Wh(5,1)) / abs(Wh(2,1)), real64)

  call tf_destroy(plan_fc);  call tf_destroy(plan_bc)
  call tf_destroy(plan_ff);  call tf_destroy(plan_bf)

  print '(a,es12.4)', 'non-dealiased |idx5|/|idx2| (expect ~1, the aliasing): ', ratio_alias
  call check('dealiased |idx5|/|idx2| ~ 0', ratio_clean, tol)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine check(name, e, tolerance)
    character(*), intent(in) :: name
    real(real64), intent(in) :: e, tolerance
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', e <= tolerance)
    print '(a,a,a,es12.4,a,es10.2,a)', '[',tag,'] '//name//'  err=', e, '  tol=', tolerance
    if (e > tolerance) all_ok = .false.
  end subroutine check

end program test_dealias_c2c
