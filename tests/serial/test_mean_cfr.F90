!===============================================================================
! test_mean_cfr.F90  --  constant-bulk-velocity (CFR) constraint.  CPU, dp.
!
! Drives the mean profile at constant bulk velocity U_b = (1/H) int_{-1}^{1} u0 dy =
! Q_target (H = 2, the channel full height) by solving for the mean pressure
! gradient beta each stage (mean_stage_cfr / step_mean_cfr).
!
!  A) BULK-VELOCITY CONSERVATION (the rigorous check). From rest, relax with the
!     constraint and require U_b equal Q_target to ~machine precision at EVERY
!     step. beta is chosen so int(U_p + c_k dt beta U_h) = H*Q_target exactly,
!     so the bulk velocity is enforced algebraically and holds to round-off
!     regardless of the relaxation transient.
!
!  B) STEADY-STATE beta AND PROFILE. At steady state the profile is the laminar
!     Poiseuille with bulk velocity Q_target (centreline u_c = 1.5 Q_target); the
!     steady mean balance (1/Re) u0'' = -beta with u0'' = -2 u_c gives
!     beta_steady = 2 u_c/Re = 3 Q_target/Re. Require the converged beta to match
!     that, and the profile to match (Re*beta_steady/2)(1-y^2). (Tols
!     here absorb the wall-normal-operator conditioning floor seen across the long
!     relaxation; the rigorous validation is A.)
!
! NaN/Inf hard-fail. Build (CPU only):
!   nvfortran -c types.F90 cheb_shen.F90 helm_band_mod.F90 mean_mod.F90
!   nvfortran -c test_mean_cfr.F90
!   nvfortran -o test_mean_cfr test_mean_cfr.o mean_mod.o helm_band_mod.o cheb_shen.o types.o
!   ./test_mean_cfr
!===============================================================================
program test_mean_cfr
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,         only: dp, ip
  use cheb_shen,     only: glc_nodes, cheb_eval
  use helm_band_mod, only: helm_band_setup
  use mean_mod,      only: mean_flux, bulk_velocity, mean_flux_setup, step_mean_cfr
  implicit none

  integer(ip), parameter :: n = 32, Ns = 300
  real(dp),    parameter :: Re = 1.0_dp, dt = 0.05_dp, Qtar = 1.0_dp
  real(dp) :: sd(0:n-2), st(0:n-2), bd(0:n-2), boff
  real(dp) :: Uh_all(0:n,3), Ih_all(3)
  real(dp) :: q0(0:n), x(0:n)
  real(dp) :: beta, beta_steady, fluxdev_max, dev, emax, ana
  integer(ip) :: it, j
  logical :: all_pass, bad

  all_pass = .true.; bad = .false.
  call helm_band_setup(n, sd, st, bd, boff)
  call glc_nodes(n, x)
  call mean_flux_setup(n, dt, Re, sd, st, bd, boff, Uh_all, Ih_all)

  write(*,'(a)') '=== mean-mode constant-mass-flux constraint (k^2=0, CFR) ==='

  ! relax from rest with the flux constraint; track flux deviation every step
  q0 = 0.0_dp
  fluxdev_max = 0.0_dp; beta = 0.0_dp
  do it = 1, Ns
     call step_mean_cfr(n, dt, Re, Qtar, sd, st, bd, boff, Uh_all, Ih_all, q0, beta)
     dev = abs(bulk_velocity(n, q0) - Qtar)
     if (.not. (dev == dev)) bad = .true.
     fluxdev_max = max(fluxdev_max, dev)
  end do

  !--- A) flux conservation -----------------------------------------------------
  write(*,'(/,a)')        '--- A) bulk velocity held at Q_target every step (Re=1, U_b=1, 300 steps) ---'
  write(*,'(a,es12.4,a)') '   max|U_bulk - Q_target| over all steps = ', fluxdev_max, '   (tol 1e-12)'
  call verdict('bulk velocity conserved to machine precision', (.not. bad) .and. fluxdev_max < 1.0e-12_dp, all_pass)

  !--- B) steady-state beta and profile ----------------------------------------
  beta_steady = 3.0_dp*Qtar/Re                     ! = 3.0 here (U_b convention, u_c=1.5)
  emax = 0.0_dp
  do j = 0, n
     ana = 0.5_dp*Re*beta_steady*(1.0_dp - x(j)**2)
     dev = abs(cheb_eval(n, q0, x(j)) - ana)
     if (.not. (dev == dev)) bad = .true.
     emax = max(emax, dev)
  end do
  write(*,'(/,a)')        '--- B) steady-state pressure gradient and profile ---'
  write(*,'(a,f12.8,a,f12.8,a)') '   beta(steady) = ', beta, '   expected 3Q/Re = ', beta_steady, ''
  write(*,'(a,es12.4,a)') '   max|u0 - (Re*beta/2)(1-y^2)| = ', emax, '   (tol 1e-5)'
  call verdict('steady beta = 3Q/Re', (.not. bad) .and. abs(beta - beta_steady) < 1.0e-4_dp, all_pass)
  call verdict('steady profile = Poiseuille', (.not. bad) .and. emax < 1.0e-5_dp, all_pass)

  if (all_pass) then
     write(*,'(/,a)') 'ALL TESTS PASSED'
  else
     write(*,'(/,a)') 'SOME TESTS FAILED'
  end if

contains

  subroutine verdict(label, ok, all_pass)
    character(*), intent(in)    :: label
    logical,      intent(in)    :: ok
    logical,      intent(inout) :: all_pass
    if (ok) then
       write(*,'(a,a)') '   [PASS] ', label
    else
       write(*,'(a,a)') '   [FAIL] ', label
       all_pass = .false.
    end if
  end subroutine verdict

end program test_mean_cfr
