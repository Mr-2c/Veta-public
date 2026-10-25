!===============================================================================
! test_fd_mean.F90 -- gate for the FD-y MEAN machinery
! (fd_mean_mod) against the Chebyshev mean_mod oracle:
!   (1) FLUX: trapz-weight bulk integral exact on linears (any grid), and
!       within the file-weights' 2nd-order envelope on (1-y^2) (analytic 4/3);
!   (2) LIN TRANSIENT: K steps of fd_step_mean_lin vs mean_mod::step_mean_lin
!       from the same IC -- SAME SMR/CN weights (shared constants), so the
!       comparison isolates the spatial backend. TOLERANCES ARE REGIME-SET:
!       the CN stage layers delta = sqrt(bk dt/Re) are
!       2.6-5.2e-3 here, putting the kl=4 kit at a ~1e-6..2e-4 truncation
!       floor at GLC-128 (verified ~8th-order: 9.0e-7 at GLC-256, 3.9e-9 at
!       GLC-512) -- and the n=128 Cheb transient oracle is itself converged
!       only to ~6e-7/1.8e-6 (GLC/natural nodes; converged by n=256);
!   (3) SETUP: the unit-source responses U_h agree the same way;
!   (4) CFR: the flux constraint int q0 = H*Q holds to ROUNDING at every
!       stage (beta is chosen with the same quadrature that measures it --
!       exactness is structural), and the beta trace tracks the Cheb oracle
!       to the quadrature difference (trapz vs exact-Chebyshev integral of
!       the SAME profiles: an O(h^2) definitional gap, NOT an algebra error;
!       an algebra error moves beta O(1)).
! Grids: GLC(128) with test-built trapz weights + the committed natural-481
! file (its own w column). Host-only, dp; run from the repo root.
!===============================================================================
program test_fd_mean
  use kinds,     only: dp, ip
  use cheb_shen, only: glc_nodes, cheb_eval
  use helm_band_mod, only: helm_band_setup
  use mean_mod,  only: mean_flux_setup, mean_stage_cfr, step_mean_lin, H_CH
  use fdops_mod, only: fd_build_bands
  use fd_mean_mod, only: fd_mean_flux, fd_step_mean_lin, fd_mean_flux_setup, &
                         fd_mean_stage_cfr
  use ygrid_mod, only: ygrid_read
  implicit none
  integer(ip), parameter :: n = 128, ny = n + 1, kl = 4, nyg = 481, NCFR = 9
  real(dp), parameter :: DT = 0.01_dp, RE = 100.0_dp, BETA = 2.0_dp, QT = 1.0_dp
  real(dp) :: yA(ny), wA(ny), d1A(2*kl+1,ny), d2A(2*kl+1,ny)
  real(dp) :: yg(nyg), wg(nyg), d1G(2*kl+1,nyg), d2G(2*kl+1,nyg)
  real(dp) :: sdh(0:n-2), sth(0:n-2), bdh(0:n-2), boff
  real(dp) :: qc(0:n), qA(ny), qG(nyg), zN(0:n), zNA(ny)
  real(dp) :: UhC(0:n,3), IhC(3), UhA(ny,3), IhA(3)
  real(dp) :: e, w, ref, betaC, betaF, fx
  logical  :: ok, okf, have_w, allok
  character(len=256) :: msg
  integer(ip) :: j, ks, step
  allok = .true.

  ! ---- grids: GLC(128)+trapz weights; natural-481 file (own w) ---------------
  call glc_nodes(n, yA)                       ! descending +1..-1
  yA(1) = 1.0_dp;  yA(ny) = -1.0_dp
  wA(1) = 0.5_dp*(yA(1) - yA(2))
  do j = 2, ny-1
     wA(j) = 0.5_dp*(yA(j-1) - yA(j+1))
  end do
  wA(ny) = 0.5_dp*(yA(ny-1) - yA(ny))
  call fd_build_bands(ny, kl, yA, d1A, d2A)
  call ygrid_read('examples/for_testing/ygrid_Re2000_natural481.dat', nyg, &
                  yg, wg, have_w, ok, msg)
  if (.not. ok) print '(a)', '   [ygrid] '//trim(msg)
  call report('natural-481 grid file reads (with its w column)', ok .and. have_w)
  if (ok) call fd_build_bands(nyg, kl, yg, d1G, d2G)

  ! ---- (1) flux gates ---------------------------------------------------------
  do j = 1, ny
     qA(j) = 0.3_dp + 1.7_dp*yA(j)            ! linear: trapz EXACT, int = 0.6
  end do
  e = abs(fd_mean_flux(ny, wA, qA) - 0.6_dp)
  call report('flux: linear profile exact under trapz weights', e <= 1.0e-13_dp)
  if (ok) then
     do j = 1, nyg
        qG(j) = 1.0_dp - yg(j)**2             ! int = 4/3
     end do
     e = abs(fd_mean_flux(nyg, wg, qG) - 4.0_dp/3.0_dp)
     call report('flux: (1-y^2) within the file-weights 2nd-order envelope', &
                 e <= 1.0e-4_dp)
  else
     call report('flux: (1-y^2) natural-481 (grid unreadable)', .false.)
  end if

  ! ---- Cheb oracle band structure ---------------------------------------------
  call helm_band_setup(n, sdh, sth, bdh, boff)

  ! ---- (2) linear transient cross-check: GLC(128) ------------------------------
  qc = 0.0_dp;  qA = 0.0_dp
  do step = 1, 20
     call step_mean_lin(n, DT, RE, BETA, sdh, sth, bdh, boff, qc)
     call fd_step_mean_lin(ny, kl, d2A, DT, RE, BETA, qA, okf)
  end do
  e = 0.0_dp;  w = 0.0_dp
  do j = 1, ny
     ref = cheb_eval(n, qc, yA(j))
     w = max(w, abs(ref));  e = max(e, abs(qA(j) - ref))
  end do
  call report('lin transient: FD(GLC128) == Cheb at nodes (rel 1e-5)', &
              okf .and. e <= 1.0e-5_dp*max(1.0e-30_dp, w))

  ! natural-481 (same oracle, evaluated at the file nodes)
  if (ok) then
     qG = 0.0_dp
     do step = 1, 20
        call fd_step_mean_lin(nyg, kl, d2G, DT, RE, BETA, qG, okf)
     end do
     e = 0.0_dp
     do j = 1, nyg
        ref = cheb_eval(n, qc, yg(j))
        e = max(e, abs(qG(j) - ref))
     end do
     ! 2e-5: ~1.8e-6 of this is the n=128 ORACLE's own truncation at these nodes
     call report('lin transient: FD(natural-481) == Cheb at nodes (rel 2e-5)', &
                 okf .and. e <= 2.0e-5_dp*max(1.0e-30_dp, w))
  else
     call report('lin transient natural-481 (grid unreadable)', .false.)
  end if

  ! ---- (3) unit-source responses ----------------------------------------------
  call mean_flux_setup(n, DT, RE, sdh, sth, bdh, boff, UhC, IhC)
  call fd_mean_flux_setup(ny, kl, d2A, wA, DT, RE, UhA, IhA, okf)
  e = 0.0_dp;  w = 0.0_dp
  do ks = 1, 3
     do j = 1, ny
        ref = cheb_eval(n, UhC(:,ks), yA(j))
        w = max(w, abs(ref));  e = max(e, abs(UhA(j,ks) - ref))
     end do
  end do
  ! 1e-3: stage 2's delta=2.58e-3 layer sets a 1.6e-4 FD floor at GLC-128
  call report('setup: U_h responses FD == Cheb at nodes (rel 1e-3)', &
              okf .and. e <= 1.0e-3_dp*max(1.0e-30_dp, w))

  ! ---- (4) CFR: constraint exactness + beta trace ------------------------------
  qc = 0.0_dp;  qA = 0.0_dp;  zN = 0.0_dp;  zNA = 0.0_dp
  ok = .true.;  e = 0.0_dp
  do step = 1, NCFR
     ks = mod(step-1, 3) + 1
     call mean_stage_cfr(n, ks, DT, RE, QT, sdh, sth, bdh, boff, &
                         UhC(:,ks), IhC(ks), qc, zN, zN, betaC)
     call fd_mean_stage_cfr(ny, kl, d2A, wA, ks, DT, RE, QT, UhA(:,ks), IhA(ks), &
                            qA, zNA, zNA, betaF, okf)
     ok = ok .and. okf
     fx = fd_mean_flux(ny, wA, qA)
     if (abs(fx - H_CH*QT) > 1.0e-12_dp*H_CH*QT) ok = .false.   ! constraint to rounding
     e = max(e, abs(betaF - betaC)/max(1.0_dp, abs(betaC)))
  end do
  call report('cfr: flux == H*Q to rounding at every stage', ok)
  call report('cfr: beta trace tracks Cheb (rel 5e-3, quadrature-limited)', &
              e <= 5.0e-3_dp)

  if (allok) then; print '(a)', 'ALL TESTS PASSED'
  else;            print '(a)', 'SOME TESTS FAILED'; end if
  if (.not. allok) error stop 1

contains

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. okk) allok = .false.
  end subroutine report

end program test_fd_mean
