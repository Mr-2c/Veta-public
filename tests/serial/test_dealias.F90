!===============================================================================
! test_dealias.F90  --  GPU validation of 3/2 dealiasing.
!
! Build & run (ON A GPU NODE):
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -cudalib=cufft -c transforms_mod.F90
!   nvfortran -cuda -c dealias_mod.F90
!   nvfortran -cuda -cudalib=cufft test_dealias.F90 \
!             dealias_mod.o transforms_mod.o types.o -o test_dealias
!   ./test_dealias
!
! Product cos(2x)*cos(3x) = 1/2 (cos x + cos 5x): true modes 1 and 5.
! Coarse grid N=8 (Nyquist 4): mode 5 aliases onto mode 3.
! Fine grid Nd=12 (3N/2): mode 5 resolved, dropped on truncation, mode 3 clean.
! Diagnostic = |spectrum(3)| / |spectrum(1)|  (scaling-independent):
!   dealiased -> ~0 (PASS),  non-dealiased -> ~1 (printed as the demonstration).
!===============================================================================

program test_dealias
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,          only: tp, ip
  use transforms_mod, only: tf_r2c_plan, tf_c2r_plan, tf_destroy, tf_r2c, tf_c2r
  use dealias_mod,    only: pad_lo, trunc_lo
  use cudafor
  implicit none

  integer(ip), parameter :: n = 8, nd = 12          ! coarse / fine (= 3n/2)
  integer(ip), parameter :: nk = n/2 + 1, nkd = nd/2 + 1   ! 5, 7
  real(real64) :: tol, ratio_clean, ratio_alias
  logical :: all_ok = .true.
  integer :: istat, planf_c, planb_c, planf_f, planb_f
  integer(ip) :: j

  complex(tp) :: Cu(0:nk-1,1), Cv(0:nk-1,1), Wh(0:nk-1,1)
  complex(tp), device :: Cu_d(0:nk-1,1), Cv_d(0:nk-1,1)
  complex(tp), device :: Cuf_d(0:nkd-1,1), Cvf_d(0:nkd-1,1), Wf_d(0:nkd-1,1)
  complex(tp), device :: Wd(0:nk-1,1)
  real(tp), device :: uc(0:n-1), vc(0:n-1), wc(0:n-1)
  real(tp), device :: uf(0:nd-1), vf(0:nd-1), wf(0:nd-1)

  tol = merge(1.0e-10_real64, 1.0e-4_real64, tp == real64)

  ! coarse one-sided spectra: u ~ cos(2x) (mode 2), v ~ cos(3x) (mode 3)
  Cu = (0.0_tp,0.0_tp);  Cu(2,1) = (1.0_tp,0.0_tp)
  Cv = (0.0_tp,0.0_tp);  Cv(3,1) = (1.0_tp,0.0_tp)
  Cu_d = Cu;  Cv_d = Cv

  call tf_r2c_plan(planf_c, n,  1_ip);  call tf_c2r_plan(planb_c, n,  1_ip)
  call tf_r2c_plan(planf_f, nd, 1_ip);  call tf_c2r_plan(planb_f, nd, 1_ip)

  !-- DEALIASED: pad -> fine physical -> product -> fine spectrum -> truncate -
  call pad_lo(nk, nkd, 1_ip, Cu_d, Cuf_d)
  call pad_lo(nk, nkd, 1_ip, Cv_d, Cvf_d)
  call tf_c2r(planb_f, Cuf_d, uf)
  call tf_c2r(planb_f, Cvf_d, vf)
  istat = cudaDeviceSynchronize()
  !$cuf kernel do(1) <<<*,*>>>
  do j = 0, nd-1
     wf(j) = uf(j)*vf(j)
  end do
  call tf_r2c(planf_f, wf, Wf_d)
  call trunc_lo(nkd, nk, 1_ip, Wf_d, Wd)
  istat = cudaDeviceSynchronize()
  Wh = Wd
  ratio_clean = real(abs(Wh(3,1)) / abs(Wh(1,1)), real64)

  !-- NON-DEALIASED: coarse physical -> product -> coarse spectrum -----------
  call tf_c2r(planb_c, Cu_d, uc)
  call tf_c2r(planb_c, Cv_d, vc)
  istat = cudaDeviceSynchronize()
  !$cuf kernel do(1) <<<*,*>>>
  do j = 0, n-1
     wc(j) = uc(j)*vc(j)
  end do
  call tf_r2c(planf_c, wc, Wd)
  istat = cudaDeviceSynchronize()
  Wh = Wd
  ratio_alias = real(abs(Wh(3,1)) / abs(Wh(1,1)), real64)

  call tf_destroy(planf_c);  call tf_destroy(planb_c)
  call tf_destroy(planf_f);  call tf_destroy(planb_f)

  print '(a,es12.4)', 'non-dealiased |m3|/|m1| (expect ~1, the aliasing): ', ratio_alias
  call check('dealiased |m3|/|m1| ~ 0', ratio_clean, tol)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine check(name, err, tolerance)
    character(*), intent(in) :: name
    real(real64), intent(in) :: err, tolerance
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', err <= tolerance)
    print '(a,a,a,es12.4,a,es10.2,a)', '[',tag,'] '//name//'  err=', err, '  tol=', tolerance
    if (err > tolerance) all_ok = .false.
  end subroutine check

end program test_dealias
