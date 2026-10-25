!===============================================================================
! test_fd_vsolve.F90 -- CROSS-DISCRETIZATION gate for the FD-y
! v-solve (fd_vsolve_mod) against the validated CHEBYSHEV oracle
! (vsolve_if_mod, MMS-gated to ~1e-16). Both machines solve the SAME
! continuous per-mode problem for the same smooth analytic f(y)=exp(y)sin(3y):
!     (-D^2+k2) g = f/alpha            g(+-1) = 0
!     (-D^2+k2+1/alpha) v = -g         v(+-1) = 0,  closure Dv(+-1) = 0
! The oracle solution (Cheb coeffs) is evaluated at the FD nodes by the T_k
! recurrence (cheb_eval), so agreement is measured POINTWISE on each FD grid:
! a truth test independent of either discretization's internals.
!
! ORACLE RESOLUTION (why n = 192 and 256): the stiff case has boundary layers
! of width sqrt(alpha) = 1e-3; Chebyshev error on an entire boundary-layer
! function decays like exp(-n^2 delta/2) once resolved -- ~1e-8 relative at
! n=192, floor at n=256. Rather than trust that estimate blind, the test GATES
! oracle self-agreement (192 vs 256, evaluated at the GLC-257 nodes) at 5e-7,
! two+ orders below every FD tolerance it certifies: a FAIL can never be
! blamed on an unconverged oracle.
!
! PER-CASE FD TOLERANCES (physics, not slop): for (k2,alpha) = (400,1e-6) the
! GLC(129) near-wall spacing (~3e-4..3e-3 over the first nodes) gives only
! ~4 points across the 1e-3 layer, so the FD error there is genuinely ~1e-3
! of max|v| (the layer's own amplitude is only ~sqrt(alpha)*|Dv_outer| --
! small -- which is why the case still gates at 5e-2/1e-3 and not O(1)).
! The moderate cases resolve their layers fully and gate at 1e-5. Convergence
! GLC(129)->GLC(257) must show >= 4x error reduction (wall rows are order-4+,
! interior order-8) unless already at the rounding/oracle floor (1e-10).
! Wall values must be EXACTLY zero (identity Dirichlet rows + homogeneous
! RHS walls admit no rounding), and the closure wall-derivative dots must sit
! under 1e-8*sum|w|*max|v| (eps-scale in practice) -- a sign error in sigma
! leaves them ~5 orders above that gate.
!
! Grids: GLC(129), GLC(257) (descending, walls pinned), and the COMMITTED
! B1.1 natural grid examples/for_testing/ygrid_Re2000_natural481.dat through
! ygrid_read -- the file contract, the operator kit, and the v-solve gate
! each other. kl = 4 throughout (production choice). Host-only, dp.
! Run from the repo root (bash run_all_tests.sh).
!===============================================================================
program test_fd_vsolve
  use kinds,         only: dp, ip
  use cheb_shen,     only: glc_nodes, cheb_forward, cheb_eval
  use vsolve_if_mod, only: vif_t, vif_setup, vif_solve
  use fdops_mod,     only: fd_build_bands, fd_wall_weights
  use fd_vsolve_mod, only: fd_vif_setup, fd_vif_solve
  use ygrid_mod,     only: ygrid_read
  implicit none
  integer(ip), parameter :: kl = 4
  integer(ip), parameter :: n1 = 192, n2 = 256          ! oracle resolutions
  integer(ip), parameter :: nyA = 129, nyB = 257, nyg = 481
  real(dp), parameter :: PI = 3.141592653589793238462643_dp
  real(dp), parameter :: AGR_TOL = 5.0e-7_dp            ! oracle self-agreement
  real(dp), parameter :: FLOOR   = 1.0e-10_dp           ! convergence-gate escape
  real(dp)    :: k2v(3)  = [0.0_dp,    25.0_dp,   400.0_dp]
  real(dp)    :: alv(3)  = [1.0e-2_dp, 1.0e-3_dp, 1.0e-6_dp]
  real(dp)    :: t129(3) = [1.0e-5_dp, 1.0e-5_dp, 5.0e-2_dp]
  real(dp)    :: t257(3) = [1.0e-5_dp, 1.0e-5_dp, 1.0e-3_dp]
  real(dp)    :: tnat(3) = [1.0e-4_dp, 1.0e-4_dp, 2.0e-3_dp]
  character(len=1) :: cid(3) = ['A', 'B', 'C']
  real(dp)    :: yA(nyA), yB(nyB), yg(nyg), wg(nyg)
  real(dp)    :: x1(0:n1), x2(0:n2), fv1(0:n1), fv2(0:n2)
  real(dp)    :: fc1(0:n1), fc2(0:n2), vc1(0:n1), vc2(0:n2)
  real(dp)    :: k2, alpha, agr, vmax, e129, e257, enat, r1, r2
  type(vif_t) :: d1, d2
  logical     :: allok, gridok, have_w, ok, conv
  logical     :: wexA, dokA, wexB, dokB, wexN, dokN
  character(len=256) :: msg
  integer(ip) :: ic, j
  allok = .true.

  ! FD grids: GLC descending with walls pinned; committed natural-481 file ----
  do j = 0, nyA-1
     yA(j+1) = cos(PI*real(j,dp)/real(nyA-1,dp))
  end do
  yA(1) = 1.0_dp;  yA(nyA) = -1.0_dp
  do j = 0, nyB-1
     yB(j+1) = cos(PI*real(j,dp)/real(nyB-1,dp))
  end do
  yB(1) = 1.0_dp;  yB(nyB) = -1.0_dp
  call ygrid_read('examples/for_testing/ygrid_Re2000_natural481.dat', nyg, &
                  yg, wg, have_w, ok, msg)
  gridok = ok
  if (.not. gridok) print '(a)', '   [ygrid] '//trim(msg)
  call report('committed ygrid file reads (Re2000 natural, ny=481)', gridok)

  ! oracle node sets + RHS samples (reused across cases) ----------------------
  call glc_nodes(n1, x1)
  call glc_nodes(n2, x2)
  do j = 0, n1
     fv1(j) = rhsf(x1(j))
  end do
  do j = 0, n2
     fv2(j) = rhsf(x2(j))
  end do
  call cheb_forward(n1, fv1, fc1)
  call cheb_forward(n2, fv2, fc2)

  do ic = 1, 3
     k2 = k2v(ic);  alpha = alv(ic)

     ! Chebyshev oracle at two resolutions --------------------------------------
     call vif_setup(n1, k2, alpha, d1)
     call vif_solve(n1, d1, fc1, vc1)
     call vif_setup(n2, k2, alpha, d2)
     call vif_solve(n2, d2, fc2, vc2)
     agr = 0.0_dp;  vmax = 0.0_dp
     do j = 1, nyB
        r2 = cheb_eval(n2, vc2, yB(j))
        r1 = cheb_eval(n1, vc1, yB(j))
        vmax = max(vmax, abs(r2))
        agr  = max(agr,  abs(r1 - r2))
     end do
     if (vmax == 0.0_dp) vmax = 1.0_dp
     agr = agr/vmax
     call report('case '//cid(ic)//': oracle n=192 vs n=256 self-agreement', &
                 agr <= AGR_TOL)

     ! FD on the three grids -----------------------------------------------------
     call fd_run(nyA, yA, k2, alpha, e129, wexA, dokA)
     call fd_run(nyB, yB, k2, alpha, e257, wexB, dokB)
     if (gridok) then
        call fd_run(nyg, yg, k2, alpha, enat, wexN, dokN)
     else
        enat = huge(1.0_dp);  wexN = .false.;  dokN = .false.
     end if
     print '(a,a,a,4es11.3)', '   [case ', cid(ic), &
          '] agr/129/257/nat481:', agr, e129, e257, enat

     call report('case '//cid(ic)//': FD GLC(129) matches oracle', e129 <= t129(ic))
     call report('case '//cid(ic)//': FD GLC(257) matches oracle', e257 <= t257(ic))
     conv = (4.0_dp*e257 < e129) .or. (e257 < FLOOR)
     call report('case '//cid(ic)//': 129->257 converges (>=4x) or at floor', conv)
     call report('case '//cid(ic)//': FD natural-481 file grid matches oracle', &
                 enat <= tnat(ic))
     call report('case '//cid(ic)//': exact wall zeros + closure dots ~ 0', &
                 wexA .and. wexB .and. wexN .and. dokA .and. dokB .and. dokN)
  end do

  if (allok) then; print '(a)', 'ALL TESTS PASSED'
  else;            print '(a)', 'SOME TESTS FAILED'; end if
  if (.not. allok) error stop 1

contains

  !-- the shared smooth RHS ----------------------------------------------------
  real(dp) function rhsf(x) result(v)
    real(dp), intent(in) :: x
    v = exp(x)*sin(3.0_dp*x)
  end function rhsf

  !---------------------------------------------------------------------------
  ! One FD solve on grid y(1:ny): build operators, setup, solve, compare with
  ! the n2 oracle pointwise. relerr in units of max|v_oracle| on this grid;
  ! wex = walls EXACTLY zero; dok = closure wall-derivative dots at eps level.
  !---------------------------------------------------------------------------
  subroutine fd_run(ny, y, k2, alpha, relerr, wex, dok)
    integer(ip), intent(in)  :: ny
    real(dp),    intent(in)  :: y(ny), k2, alpha
    real(dp),    intent(out) :: relerr
    logical,     intent(out) :: wex, dok
    real(dp) :: d1b(2*kl+1, ny), d2b(2*kl+1, ny), wlo(ny), whi(ny)
    real(dp) :: vm(ny), vp(ny), Cinv(2,2), f(ny), v(ny), vref
    real(dp) :: e, vmx, dlo, dhi, sw
    logical  :: okk
    integer(ip) :: j

    relerr = huge(1.0_dp);  wex = .false.;  dok = .false.
    call fd_build_bands(ny, kl, y, d1b, d2b)     ! d1b unused (builder emits both)
    call fd_wall_weights(ny, kl, y, wlo, whi)
    call fd_vif_setup(ny, kl, y, d2b, wlo, whi, k2, alpha, vm, vp, Cinv, okk)
    if (.not. okk) then
       print '(a)', '   [fd_run] fd_vif_setup refused (factor/det)'
       return
    end if
    do j = 1, ny
       f(j) = rhsf(y(j))
    end do
    call fd_vif_solve(ny, kl, d2b, wlo, whi, k2, alpha, vm, vp, Cinv, f, v, okk)
    if (.not. okk) then
       print '(a)', '   [fd_run] fd_vif_solve refused (factor)'
       return
    end if

    e = 0.0_dp;  vmx = 0.0_dp
    do j = 1, ny
       vref = cheb_eval(n2, vc2, y(j))
       vmx  = max(vmx, abs(vref))
       e    = max(e, abs(v(j) - vref))
    end do
    if (vmx == 0.0_dp) vmx = 1.0_dp
    relerr = e/vmx

    wex = (v(1) == 0.0_dp) .and. (v(ny) == 0.0_dp)
    dlo = dot_product(wlo, v)
    dhi = dot_product(whi, v)
    sw  = sum(abs(wlo)) + sum(abs(whi))
    dok = max(abs(dlo), abs(dhi)) <= 1.0e-8_dp*max(1.0_dp, sw*vmx)
  end subroutine fd_run

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. okk) allok = .false.
  end subroutine report

end program test_fd_vsolve
