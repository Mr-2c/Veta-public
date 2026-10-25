!===============================================================================
! test_vsolve_if_mms.F90  --  manufactured-solution test for the two-Helmholtz +
! influence-matrix v-solve (and the dense biharmonic), ASSERTING.
!
! Build a smooth, clamped exact solution  v_exact = sum_k c_k psi_k  (a degree-10
! polynomial, automatically v=Dv=0 at the walls), form the exact RHS
!     f = (L - alpha L^2) v_exact          (via the validated cheb_d2),
! and for increasing n measure the recovered v against the TRUE v_exact for BOTH
! solvers. Validating against truth (not against another discretization) cleanly
! separates a code bug from a resolution effect.
!
! For a clamped POLYNOMIAL exact solution both solvers are exact at every n >= 10
! (the polynomial is representable and the operators are exact on it); the
! boundary-layer "resolution threshold" only matters for a genuinely
! non-polynomial RHS, so it is NOT asserted here. We
! therefore assert a tight tolerance on BOTH solvers at ALL n; a silent accuracy
! regression fails. NaN/Inf hard-fail.
!
! PRODUCTION alpha row: alv(5) = 1.5e-9 = beta_2*dt/Re at
! Re=43500, dt=1e-3 -- the regime the solver actually runs at. That row gates
! at tol*1e3 (a calibrated gate; see the call-site comment).
!
! Build:
!   nvfortran -c types.F90 cheb_shen.F90 cheb_bih.F90 vsolve_if_mod.F90
!   nvfortran -c test_vsolve_if_mms.F90
!   nvfortran -o test_vsolve_if_mms test_vsolve_if_mms.o vsolve_if_mod.o \
!             cheb_bih.o cheb_shen.o types.o
!   ./test_vsolve_if_mms
!===============================================================================
program test_vsolve_if_mms
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,         only: dp, ip
  use cheb_shen,     only: cheb_d2
  use cheb_bih,      only: bih_basis, biharmonic_solve, shen_bih_to_cheb
  use vsolve_if_mod, only: vif_t, vif_setup, vif_solve
  implicit none

  integer(ip), parameter :: nmax = 128
  real(dp),    parameter :: tol = 1.0e-9_dp      ! generous: observed ~1e-16
  integer(ip) :: nv(7) = [16, 24, 32, 48, 64, 96, 128]
  real(dp)    :: k2v(2) = [0.0_dp, 9.0_dp]
  real(dp)    :: alv(5) = [1.0_dp, 1.0e-2_dp, 1.0e-4_dp, 1.0e-6_dp, 1.5e-9_dp]  ! alv(5) = production alpha
  real(dp)    :: c(0:6) = [0.7_dp,-0.4_dp,0.55_dp,-0.25_dp,0.3_dp,-0.15_dp,0.2_dp]
  real(dp)    :: vex(0:nmax), psi(0:nmax)
  integer(ip) :: k, ik, ia
  logical     :: all_pass

  ! manufactured clamped solution v_exact (degree 10), Chebyshev coeffs
  vex = 0.0_dp
  do k = 0, 6
     call bih_basis(nmax, k, psi)
     vex = vex + c(k)*psi
  end do

  all_pass = .true.
  do ik = 1, size(k2v)
     do ia = 1, size(alv)
        ! per-row tolerance: alv(1:4) keep the historic tol. The production
        ! row alv(5)=1.5e-9 (= beta_2*dt/Re at Re=43500, dt=1e-3) gates at
        ! 5e-14, calibrated from measurement:
        ! max err(cheb_bih)=4.44e-15 / err(vif)=2.46e-15 over both
        ! k2 and all n=16..128 -- the feared 1/alpha amplify-and-cancel never
        ! surfaced above roundoff (small-n conditioning included). Gate =
        ! measured*10.
        if (alv(ia) < 1.0e-8_dp) then
           call run(k2v(ik), alv(ia), vex, 5.0e-14_dp, all_pass)
        else
           call run(k2v(ik), alv(ia), vex, tol, all_pass)
        end if
     end do
  end do

  if (all_pass) then
     write(*,'(/,a)') 'ALL TESTS PASSED'
  else
     write(*,'(/,a)') 'SOME TESTS FAILED'
  end if

contains

  subroutine run(k2, alpha, vex, tolr, all_pass)
    real(dp), intent(in)    :: k2, alpha, vex(0:nmax)
    real(dp), intent(in)    :: tolr           ! per-row gate (tol, or tol*1e3 on the production row)
    logical,  intent(inout) :: all_pass
    real(dp) :: f(0:nmax), Lv(0:nmax), L2v(0:nmax), tmp(0:nmax)
    real(dp) :: vbih(0:nmax), vrefb(0:nmax), vvif(0:nmax)
    type(vif_t) :: d
    real(dp) :: eb, ev, denom
    integer(ip) :: inn, n, k
    logical :: bad

    ! f = (L - alpha L^2) v_exact, exact at nmax for a degree-10 polynomial
    call cheb_d2(nmax, vex, tmp);  Lv  = tmp - k2*vex
    call cheb_d2(nmax, Lv,  tmp);  L2v = tmp - k2*Lv
    f = Lv - alpha*L2v

    write(*,'(/,a,f5.1,a,es9.2,a)') '==== k2=',k2,'  alpha=',alpha,' ===='
    write(*,'(a)') '    n      err(cheb_bih)      err(vif)      result'
    do inn = 1, size(nv)
       n = nv(inn)
       call biharmonic_solve(n, k2, alpha, f(0:n), vbih(0:n-4))
       call shen_bih_to_cheb(n, vbih(0:n-4), vrefb(0:n))
       call vif_setup(n, k2, alpha, d)
       call vif_solve(n, d, f(0:n), vvif(0:n))
       eb = 0.0_dp; ev = 0.0_dp; denom = 0.0_dp
       do k = 0, n
          denom = max(denom, abs(vex(k)))
          eb    = max(eb,    abs(vrefb(k) - vex(k)))
          ev    = max(ev,    abs(vvif(k)  - vex(k)))
       end do
       if (denom == 0.0_dp) denom = 1.0_dp
       eb = eb/denom; ev = ev/denom
       bad = .false.
       if (.not. (eb == eb) .or. .not. (ev == ev)) bad = .true.       ! NaN
       if (eb > huge(1.0_dp) .or. ev > huge(1.0_dp)) bad = .true.     ! Inf
       if (bad .or. eb > tolr .or. ev > tolr) then
          write(*,'(i6,2es19.4,a)') n, eb, ev, '   [FAIL]'
          all_pass = .false.
       else
          write(*,'(i6,2es19.4,a)') n, eb, ev, '   [PASS]'
       end if
    end do
  end subroutine run

end program test_vsolve_if_mms
