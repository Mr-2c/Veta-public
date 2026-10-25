!===============================================================================
! test_helm_band.F90  --  validate the O(n) banded Shen-Helmholtz solve against
! the dense cheb_shen helmholtz_solve, across n and mu.
!
! For each (n, mu): random Chebyshev-coeff RHS fcheb. Reference uhat from the
! dense LU path (cheb_shen helmholtz_solve). Banded uhat from helm_band_setup +
! helmholtz_rhs + helm_band_solve. Both solve the SAME (S+mu B) system, so they
! must agree to round-off (scaled by the system conditioning).
!
! mu sweep spans the realistic regime: in time-stepping mu = k^2 + Re/(beta_k dt)
! is large (~1e3-1e6), where the system is mass-dominated and very well
! conditioned. mu=0 (pure Poisson) is included as a stiff stress test.
!
! Build:
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran -c helm_band_mod.F90
!   nvfortran -c test_helm_band.F90
!   nvfortran -o test_helm_band test_helm_band.o helm_band_mod.o cheb_shen.o types.o
!   ./test_helm_band
!===============================================================================
program test_helm_band
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,        only: dp, ip
  use cheb_shen,    only: helmholtz_solve, helmholtz_rhs
  use helm_band_mod, only: helm_band_setup, helm_band_solve
  implicit none

  ! The n=480 + mu=6.5e8 rows are the production-regime
  ! gate that polices the closed-form mass diagonal. The differencing error is
  ! ~eps*S/B ~ eps*n^2 (3.5e-11 at n=480, only ~1e-13 at n<=32), so a sweep
  ! limited to n<=32 at tol 1e-9 is ~1e4x blind to it at ANY mu -- and mu alone
  ! cannot help: the operator perturbation saturates at |dB|/B once mu*B >> S,
  ! which already holds at mu=1e6 for these n. Gate design + numbers:
  ! src/models/work_a6_mass_diag_model.py. tol=1e-11 for the mass-dominated
  ! mu>=1e8 rows: the MEASURED dense-vs-banded floor at (n=480, mu=6.5e8) is
  ! ~1.1e-12 (worst seed 1.8e-12; numpy emulation of the exact band
  ! algorithm), so
  ! 1e-11 keeps >=5x headroom against compiler/FMA arithmetic wobble while a
  ! regression to the differenced extraction (3.5e-11) still FAILS by 3.5x.
  integer(ip) :: nv(6) = [8, 12, 16, 24, 32, 480]
  real(dp)    :: muv(6) = [0.0_dp, 1.0_dp, 1.0e2_dp, 1.0e4_dp, 1.0e6_dp, 6.5e8_dp]
  integer(ip) :: ic, im
  logical :: all_pass

  all_pass = .true.
  do ic = 1, 6
     call run_n(nv(ic), muv, all_pass)
  end do
  if (all_pass) then
     print *, 'ALL TESTS PASSED'
  else
     print *, 'SOME TESTS FAILED'
  end if

contains

  subroutine run_n(n, muv, all_pass)
    integer(ip), intent(in)    :: n
    real(dp),    intent(in)    :: muv(:)
    logical,     intent(inout) :: all_pass
    real(dp) :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
    real(dp) :: fcheb(0:n), ftilde(0:n-2)
    real(dp) :: uref(0:n-2), uband(0:n-2)
    real(dp) :: mu, err, denom, tol, rr, d
    logical  :: bad
    integer(ip) :: im, k

    call helm_band_setup(n, sdiag, stail, bdiag, boff)

    write(*,'(/,a,i0,a,i0,a)') '--- n=',n,'  (dof=',n-1,') ---'
    do im = 1, size(muv)
       mu = muv(im)
       block   ! [repro] fixed seed, applied once per run -> the suite is a bit-identity oracle
               ! across the precision phases (replaces a bare call random_seed())
         integer              :: rs_n_, rs_j_
         integer, allocatable :: rs_seed_(:)
         logical, save        :: rs_done_ = .false.
         if (.not. rs_done_) then
            call random_seed(size = rs_n_)
            allocate(rs_seed_(rs_n_))
            rs_seed_ = [ (1789 + 101*rs_j_, rs_j_ = 0, rs_n_ - 1) ]
            call random_seed(put = rs_seed_)
            rs_done_ = .true.
         end if
       end block
       do k = 0, n
          call random_number(rr); fcheb(k) = rr - 0.5_dp
       end do

       call helmholtz_solve(n, mu, fcheb, uref)            ! dense reference
       call helmholtz_rhs  (n, fcheb, ftilde)              ! shared projection
       call helm_band_solve(n, mu, sdiag, stail, bdiag, boff, ftilde, uband)

       err = 0.0_dp; denom = 0.0_dp; bad = .false.
       do k = 0, n-2
          d = abs(uband(k) - uref(k))
          if (.not. (d == d))        bad = .true.   ! NaN
          if (abs(uband(k)) > huge(1.0_dp)) bad = .true.   ! Inf
          err   = max(err,   d)
          denom = max(denom, abs(uref(k)))
       end do
       if (denom == 0.0_dp) denom = 1.0_dp
       ! tight where well-conditioned; looser for the stiff mu=0 Poisson case;
       ! TIGHTEST in the mass-dominated production regime (the A6 police row:
       ! must distinguish closed-form bdiag [measured floor ~1.1-1.8e-12 at
       ! n=480] from a regression to the differenced extraction [3.5e-11])
       if (mu == 0.0_dp) then
          tol = 1.0e-6_dp
       else if (mu >= 1.0e8_dp) then
          tol = 1.0e-11_dp
       else
          tol = 1.0e-9_dp
       end if
       if (bad) then
          call report(mu, 1.0e30_dp, tol, all_pass)   ! non-finite -> hard fail
       else
          call report(mu, err/denom, tol, all_pass)
       end if
    end do
  end subroutine run_n

  subroutine report(mu, e, tol, all_pass)
    real(dp), intent(in)    :: mu, e, tol
    logical,  intent(inout) :: all_pass
    character(6) :: tag
    if (e <= tol) then
       tag = '[PASS]'
    else
       tag = '[FAIL]'; all_pass = .false.
    end if
    write(*,'(a,2x,a,es10.2,2x,a,es12.4,2x,a,es10.2)') &
         tag, 'mu=', mu, 'rel err=', e, 'tol=', tol
  end subroutine report

end program test_helm_band
