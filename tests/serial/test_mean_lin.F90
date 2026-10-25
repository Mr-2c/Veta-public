!===============================================================================
! test_mean_lin.F90  --  mean-mode linear stepper.  CPU, dp.
!
! Three checks on mean_mod with N = 0:
!
!  A) TEMPORAL ORDER. Pure viscous decay of a smooth Dirichlet profile (beta=0),
!     mildly damped: self-convergence p = log2(||q(dt)-q(dt/2)|| /
!     ||q(dt/2)-q(dt/4)||) must be ~2 (symmetric CN), the k^2=0 analogue of the
!     eta-stepper order test. Asserted only in the mild-damping / measurable
!     regime.
!
!  B1) POISEUILLE FIXED POINT. The analytic laminar profile u0 = (Re*beta/2)(1-y^2)
!     is the exact steady state of the discrete scheme (the substep factor c_k
!     cancels: D^2 u0 = -Re*beta => the parabola, exact in Chebyshev). Initialise
!     q0 to it, take one full step, and require it is UNCHANGED to ~machine
!     precision. A clean closed-form check of the pressure-gradient + viscous + BC
!     machinery.
!
!  B2) RELAXATION. From rest (q0=0) with constant beta at low Re (fast viscous
!     relaxation), evolve to steady and require convergence to the same analytic
!     profile -- confirms the transient relaxes to the fixed point (right sign /
!     direction), not merely that the fixed point exists.
!
! NaN/Inf hard-fail. Build (CPU only):
!   nvfortran -c types.F90 cheb_shen.F90 helm_band_mod.F90 mean_mod.F90
!   nvfortran -c test_mean_lin.F90
!   nvfortran -o test_mean_lin test_mean_lin.o mean_mod.o helm_band_mod.o cheb_shen.o types.o
!   ./test_mean_lin
!===============================================================================
program test_mean_lin
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,         only: dp, ip
  use cheb_shen,     only: glc_nodes, cheb_eval, shen_to_cheb
  use helm_band_mod, only: helm_band_setup
  use mean_mod,      only: step_mean_lin
  implicit none
  logical :: all_pass
  all_pass = .true.

  write(*,'(a)') '=== mean-mode linear stepper (k^2=0 Dirichlet Helmholtz, SMR/CN) ==='
  call test_order(all_pass)
  call test_poiseuille_fixedpoint(all_pass)
  call test_poiseuille_relax(all_pass)

  if (all_pass) then
     write(*,'(/,a)') 'ALL TESTS PASSED'
  else
     write(*,'(/,a)') 'SOME TESTS FAILED'
  end if

contains

  !--- A) temporal order (viscous decay, beta=0) -------------------------------
  subroutine test_order(all_pass)
    logical, intent(inout) :: all_pass
    integer(ip), parameter :: n = 32, Ns0 = 20
    real(dp),    parameter :: Re = 180.0_dp, dt0 = 0.02_dp
    real(dp) :: sd(0:n-2), st(0:n-2), bd(0:n-2), boff
    real(dp) :: q0(0:n), qa(0:n), qb(0:n), qc(0:n), ush(0:n-2)
    real(dp) :: rr, decay, kc, p, e1, e2, n0, nT, ratio, d
    integer(ip) :: k
    logical :: bad, mild, meas

    call helm_band_setup(n, sd, st, bd, boff)
    kc = real(n,dp)/6.0_dp
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
    do k = 0, n-2
       call random_number(rr); decay = exp(-(real(k,dp)/kc)**2)
       ush(k) = (rr-0.5_dp)*decay
    end do
    call shen_to_cheb(n, ush, q0)           ! smooth Dirichlet IC

    qa = q0; call run(n, Re, dt0,          Ns0,   sd,st,bd,boff, qa)
    qb = q0; call run(n, Re, dt0/2.0_dp, 2*Ns0,   sd,st,bd,boff, qb)
    qc = q0; call run(n, Re, dt0/4.0_dp, 4*Ns0,   sd,st,bd,boff, qc)

    bad = .false.; e1 = 0.0_dp; e2 = 0.0_dp; n0 = 0.0_dp; nT = 0.0_dp
    do k = 0, n
       d = abs(qa(k)-qb(k)); if (.not. (d==d)) bad = .true.; e1 = e1 + d*d
       d = abs(qb(k)-qc(k)); if (.not. (d==d)) bad = .true.; e2 = e2 + d*d
       n0 = n0 + q0(k)**2;   nT = nT + qc(k)**2
       if (abs(qc(k)) > huge(1.0_dp)) bad = .true.
    end do
    e1 = sqrt(e1); e2 = sqrt(e2); ratio = sqrt(nT/max(n0,tiny(1.0_dp)))
    if (e2 > 0.0_dp) then; p = log(e1/e2)/log(2.0_dp); else; p = 0.0_dp; end if

    write(*,'(/,a)')        '--- A) temporal order (viscous decay, beta=0, Re=180) ---'
    write(*,'(a,f6.3,a,es10.2,a,f6.3)') '   p=',p,'   e2=',e2,'   ||q_T||/||q_0||=',ratio
    call verdict('finite', .not. bad, all_pass)
    mild = (ratio > 0.5_dp); meas = (e2 > 1.0e-12_dp)
    if (mild .and. meas) then
       call verdict('order ~ 2 (symmetric CN)', (p > 1.7_dp .and. p < 2.4_dp), all_pass)
    else
       write(*,'(a)') '   [skip] order not asserted (strong damping / round-off)'
    end if
  end subroutine test_order

  !--- B1) analytic Poiseuille is a fixed point --------------------------------
  subroutine test_poiseuille_fixedpoint(all_pass)
    logical, intent(inout) :: all_pass
    integer(ip), parameter :: n = 32
    real(dp),    parameter :: Re = 10.0_dp, beta = 1.0_dp, dt = 0.05_dp
    real(dp) :: sd(0:n-2), st(0:n-2), bd(0:n-2), boff
    real(dp) :: q0(0:n), x(0:n)
    real(dp) :: emax, d, ana
    integer(ip) :: j
    logical :: bad

    call helm_band_setup(n, sd, st, bd, boff)
    call glc_nodes(n, x)
    q0 = 0.0_dp
    q0(0) =  Re*beta/4.0_dp                 ! (Re*beta/2)(1-y^2) = (Re*beta/4)(T_0 - T_2)
    q0(2) = -Re*beta/4.0_dp

    call step_mean_lin(n, dt, Re, beta, sd, st, bd, boff, q0)   ! one full step

    emax = 0.0_dp; bad = .false.
    do j = 0, n
       ana = 0.5_dp*Re*beta*(1.0_dp - x(j)**2)
       d   = abs(cheb_eval(n, q0, x(j)) - ana)
       if (.not. (d==d)) bad = .true.
       emax = max(emax, d)
    end do
    write(*,'(/,a)')        '--- B1) analytic Poiseuille is a fixed point (Re=10, beta=1) ---'
    write(*,'(a,es12.4,a)') '   max|u0 - (Re*beta/2)(1-y^2)| after one step = ', emax, '   (tol 1e-10)'
    call verdict('Poiseuille fixed point', (.not. bad) .and. emax < 1.0e-10_dp, all_pass)
  end subroutine test_poiseuille_fixedpoint

  !--- B2) relaxation from rest to Poiseuille ----------------------------------
  subroutine test_poiseuille_relax(all_pass)
    logical, intent(inout) :: all_pass
    integer(ip), parameter :: n = 32, Ns = 300
    real(dp),    parameter :: Re = 1.0_dp, beta = 1.0_dp, dt = 0.05_dp
    real(dp) :: sd(0:n-2), st(0:n-2), bd(0:n-2), boff
    real(dp) :: q0(0:n), x(0:n)
    real(dp) :: emax, d, ana
    integer(ip) :: it, j
    logical :: bad

    call helm_band_setup(n, sd, st, bd, boff)
    call glc_nodes(n, x)
    q0 = 0.0_dp
    do it = 1, Ns
       call step_mean_lin(n, dt, Re, beta, sd, st, bd, boff, q0)
    end do

    emax = 0.0_dp; bad = .false.
    do j = 0, n
       ana = 0.5_dp*Re*beta*(1.0_dp - x(j)**2)
       d   = abs(cheb_eval(n, q0, x(j)) - ana)
       if (.not. (d==d)) bad = .true.
       emax = max(emax, d)
    end do
    write(*,'(/,a)')        '--- B2) relaxation from rest -> Poiseuille (Re=1, beta=1, 300 steps) ---'
    write(*,'(a,es12.4,a)') '   max|u0 - (Re*beta/2)(1-y^2)| at steady = ', emax, '   (tol 1e-6)'
    call verdict('relaxes to Poiseuille', (.not. bad) .and. emax < 1.0e-6_dp, all_pass)
  end subroutine test_poiseuille_relax

  !--- helpers -----------------------------------------------------------------
  subroutine run(n, Re, dt, Ns, sd, st, bd, boff, q0)
    integer(ip), intent(in)    :: n, Ns
    real(dp),    intent(in)    :: Re, dt, boff
    real(dp),    intent(in)    :: sd(0:n-2), st(0:n-2), bd(0:n-2)
    real(dp),    intent(inout) :: q0(0:n)
    integer(ip) :: it
    do it = 1, Ns
       call step_mean_lin(n, dt, Re, 0.0_dp, sd, st, bd, boff, q0)
    end do
  end subroutine run

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

end program test_mean_lin
