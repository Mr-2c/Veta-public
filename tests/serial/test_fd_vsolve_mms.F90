!===============================================================================
! test_fd_vsolve_mms.F90 -- manufactured-solution gate for the FD-y
! two-Helmholtz + influence-matrix v-solve: the FD twin of test_vsolve_if_mms.
!
! v_exact = (1-y^2)^2 * g(y) with g quartic (degree 8). Interior CENTERED rows
! are degree-8 exact; the near-wall TAPER rows (band-limited, 6..8 nodes) leave
! an ~8th-order truncation that DECAYS with n -- worst case
! 1.81e-7 at (k2=0, alpha=1, n=32), 4.2e-13 by n=128; the 2e-6 gate holds 11x
! at the worst point and a silent accuracy regression fails at every n. The
! exact RHS f = (L - alpha L^2) v_exact is built
! in Chebyshev COEFFICIENT space (cheb_forward at n=16 is exact for degree 8;
! cheb_d2 exact) and EVALUATED at the FD nodes by Clenshaw -- truth, not a
! discretization. Gate: rel 2e-6 (= 11x the measured worst truncation; conditioning
! is NOT the binding term -- the cond~7e8 leg measures 1.2e-14).
! Rung 2: NON-polynomial v = (1-y^2)^2 sin(2y) at n=64 vs n=128 -- the error
! must drop by ~2^8 (kl=4 interior order); gate ratio >= 60 (margin for the
! wall-row order taper).
! PRODUCTION alpha row: alv(5) = 1.5e-9 = beta_2*dt/Re at
! Re=43500, dt=1e-3. That row gates at 50*TOL = 1e-4 (a calibrated
! gate; see the loop comment).
! The DEVICE vif is bit-gated against this host path in test_fd_dispatch, so
! this host MMS covers the production device solve by transitivity.
!===============================================================================
program test_fd_vsolve_mms
  use kinds,         only: dp, ip
  use cheb_shen,     only: glc_nodes, cheb_forward, cheb_d2
  use fdops_mod,     only: fd_build_bands, fd_wall_weights
  use fd_vsolve_mod, only: fd_vif_setup, fd_vif_solve
  implicit none
  integer(ip), parameter :: NC = 16, kl = 4
  real(dp),    parameter :: TOL = 2.0e-6_dp
  integer(ip) :: nv(4) = [32, 48, 64, 128]
  real(dp)    :: k2v(2) = [0.0_dp, 9.0_dp]
  real(dp)    :: alv(5) = [1.0_dp, 1.0e-2_dp, 1.0e-4_dp, 1.0e-6_dp, 1.5e-9_dp]  ! alv(5) = production alpha
  real(dp)    :: yc(0:NC), vc(0:NC), ac(0:NC), Lc(0:NC), L2c(0:NC), tmp(0:NC)
  real(dp)    :: e64, e128
  real(dp)    :: tolr             ! per-row gate, set in the alpha loop; run_case host-associates it
  integer(ip) :: j, ik, ia, inn
  logical     :: allok

  allok = .true.

  ! ---- truth in coefficient space: v = (1-y^2)^2 (0.7 - 0.4y + 0.55y^2
  !      - 0.25y^3 + 0.3y^4), degree 8 -> exact at NC=16 -------------------------
  call glc_nodes(NC, yc)
  do j = 0, NC
     vc(j) = poly8(yc(j))
  end do
  call cheb_forward(NC, vc, ac)

  do ik = 1, size(k2v)
     do ia = 1, size(alv)
        ! per-row gate: TOL for the historic rows; the production row
        ! alv(5)=1.5e-9 (= beta_2*dt/Re at Re=43500, dt=1e-3) gates at 5e-8,
        ! calibrated from measurement: truncation-
        ! dominated, max err 4.92e-9 at n=32 decaying 8th-order to
        ! ~1e-13 at n=128 (both k2) -- the 1/alpha amplify-and-cancel stays
        ! below the FD truncation floor, so the production row gates TIGHTER
        ! than the historic TOL. Gate = measured*10.
        tolr = TOL
        if (alv(ia) < 1.0e-8_dp) tolr = 5.0e-8_dp
        write(*,'(/,a,f5.1,a,es9.2,a)') '==== k2=', k2v(ik), '  alpha=', alv(ia), ' ===='
        write(*,'(a)') '    n      err(fd vif)     result'
        do inn = 1, size(nv)
           call run_case(nv(inn), k2v(ik), alv(ia), ac, .true., e64)
        end do
     end do
  end do

  ! ---- rung 2: non-polynomial convergence (k2=9, alpha=1e-4) -----------------
  call run_np(64_ip,  e64)
  call run_np(128_ip, e128)
  write(*,'(/,a,es12.4,a,es12.4,a,f8.1)') ' non-poly err n=64: ', e64, &
       '   n=128: ', e128, '   ratio: ', e64/max(e128, tiny(1.0_dp))
  if (e64/max(e128, tiny(1.0_dp)) >= 60.0_dp) then
     write(*,'(a)') ' [PASS] non-polynomial ~8th-order convergence'
  else
     write(*,'(a)') ' [FAIL] convergence order lost'
     allok = .false.
  end if

  if (allok) then; print '(a)', 'ALL TESTS PASSED'
  else;            print '(a)', 'SOME TESTS FAILED'; end if
  if (.not. allok) error stop 1

contains

  real(dp) function poly8(y) result(p)
    real(dp), intent(in) :: y
    real(dp) :: env, g
    env = (1.0_dp - y*y)**2
    g   = 0.7_dp + y*(-0.4_dp + y*(0.55_dp + y*(-0.25_dp + y*0.3_dp)))
    p   = env*g
  end function poly8

  !-- Clenshaw: sum a(m) T_m(y) ------------------------------------------------
  real(dp) function clen(nm, a, y) result(s)
    integer(ip), intent(in) :: nm
    real(dp),    intent(in) :: a(0:nm), y
    real(dp) :: b1, b2, b
    integer(ip) :: m
    b1 = 0.0_dp;  b2 = 0.0_dp
    do m = nm, 1, -1
       b = a(m) + 2.0_dp*y*b1 - b2
       b2 = b1;  b1 = b
    end do
    s = a(0) + y*b1 - b2
  end function clen

  !-- one (n, k2, alpha) case: FD vif vs truth at the nodes --------------------
  subroutine run_case(n, k2, alpha, acoef, poly, errout)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: k2, alpha, acoef(0:NC)
    logical,     intent(in)  :: poly
    real(dp),    intent(out) :: errout
    real(dp), allocatable :: yA(:), wlo(:), whi(:), d1b(:,:), d2b(:,:)
    real(dp), allocatable :: vm(:), vp(:), f(:), vout(:), vtrue(:)
    real(dp) :: Ci(2,2), den, err
    integer(ip) :: ny, jj
    logical :: ok1, ok2
    ny = n + 1
    allocate(yA(ny), wlo(ny), whi(ny), d1b(2*kl+1,ny), d2b(2*kl+1,ny))
    allocate(vm(ny), vp(ny), f(ny), vout(ny), vtrue(ny))
    call glc_nodes(n, yA)
    yA(1) = 1.0_dp;  yA(ny) = -1.0_dp
    call fd_build_bands(ny, kl, yA, d1b, d2b)
    call fd_wall_weights(ny, kl, yA, wlo, whi)
    ! exact f = (L - alpha L^2) v_exact, from coefficient space, at the nodes
    call cheb_d2(NC, acoef, tmp);  Lc  = tmp - k2*acoef
    call cheb_d2(NC, Lc,    tmp);  L2c = tmp - k2*Lc
    do jj = 1, ny
       f(jj)     = clen(NC, Lc, yA(jj)) - alpha*clen(NC, L2c, yA(jj))
       vtrue(jj) = clen(NC, acoef, yA(jj))
    end do
    call fd_vif_setup(ny, kl, yA, d2b, wlo, whi, k2, alpha, vm, vp, Ci, ok1)
    call fd_vif_solve(ny, kl, d2b, wlo, whi, k2, alpha, vm, vp, Ci, f, vout, ok2)
    err = 0.0_dp;  den = 0.0_dp
    do jj = 1, ny
       den = max(den, abs(vtrue(jj)))
       err = max(err, abs(vout(jj) - vtrue(jj)))
    end do
    err = err/max(den, tiny(1.0_dp))
    errout = err
    if (poly) then
       if (ok1 .and. ok2 .and. err == err .and. err <= tolr) then   ! per-row gate (host-associated)
          write(*,'(i6,es17.4,a)') n, err, '   [PASS]'
       else
          write(*,'(i6,es17.4,a)') n, err, '   [FAIL]'
          allok = .false.
       end if
    end if
    deallocate(yA, wlo, whi, d1b, d2b, vm, vp, f, vout, vtrue)
  end subroutine run_case

  !-- non-polynomial rung: v = (1-y^2)^2 sin(2y), coefficients at NC2=64 -------
  subroutine run_np(n, errout)
    integer(ip), intent(in)  :: n
    real(dp),    intent(out) :: errout
    integer(ip), parameter :: NC2 = 64
    real(dp) :: y2(0:NC2), v2(0:NC2), a2(0:NC2), L2(0:NC2), L22(0:NC2), t2(0:NC2)
    real(dp), allocatable :: yA(:), wlo(:), whi(:), d1b(:,:), d2b(:,:)
    real(dp), allocatable :: vm(:), vp(:), f(:), vout(:), vtrue(:)
    real(dp) :: Ci(2,2), den, err, k2, alpha, env
    integer(ip) :: ny, jj
    logical :: ok1, ok2
    k2 = 9.0_dp;  alpha = 1.0e-4_dp
    call glc_nodes(NC2, y2)
    do jj = 0, NC2
       env = (1.0_dp - y2(jj)*y2(jj))**2
       v2(jj) = env*sin(2.0_dp*y2(jj))
    end do
    call cheb_forward(NC2, v2, a2)                  ! machine-converged at NC2=64
    call cheb_d2(NC2, a2, t2);  L2  = t2 - k2*a2
    call cheb_d2(NC2, L2, t2);  L22 = t2 - k2*L2
    ny = n + 1
    allocate(yA(ny), wlo(ny), whi(ny), d1b(2*kl+1,ny), d2b(2*kl+1,ny))
    allocate(vm(ny), vp(ny), f(ny), vout(ny), vtrue(ny))
    call glc_nodes(n, yA)
    yA(1) = 1.0_dp;  yA(ny) = -1.0_dp
    call fd_build_bands(ny, kl, yA, d1b, d2b)
    call fd_wall_weights(ny, kl, yA, wlo, whi)
    do jj = 1, ny
       f(jj)     = clen(NC2, L2, yA(jj)) - alpha*clen(NC2, L22, yA(jj))
       vtrue(jj) = clen(NC2, a2, yA(jj))
    end do
    call fd_vif_setup(ny, kl, yA, d2b, wlo, whi, k2, alpha, vm, vp, Ci, ok1)
    call fd_vif_solve(ny, kl, d2b, wlo, whi, k2, alpha, vm, vp, Ci, f, vout, ok2)
    err = 0.0_dp;  den = 0.0_dp
    do jj = 1, ny
       den = max(den, abs(vtrue(jj)))
       err = max(err, abs(vout(jj) - vtrue(jj)))
    end do
    errout = err/max(den, tiny(1.0_dp))
    if (.not. (ok1 .and. ok2)) then
       write(*,'(a)') ' [FAIL] fd vif refused (zero pivot)';  allok = .false.
    end if
    deallocate(yA, wlo, whi, d1b, d2b, vm, vp, f, vout, vtrue)
  end subroutine run_np

end program test_fd_vsolve_mms
