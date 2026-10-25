!===============================================================================
! test_mean_gpu.F90  --  validate the ON-DEVICE (0,0)-mode mean stepper
! (mean_gpu_mod) against the trusted CPU mean_mod, for both the constant-mass-flux
! u0 stage (mean_stage_cfr) and the plain w0 stage (mean_stage), over a full
! 3-stage SMR step. Agreement is to round-off (the GPU D^2 = cheb_deriv-twice and
! the GPU banded solve differ from the CPU cheb_d2 / helm_band_solve at ~1e-16),
! NOT bit-exact. NaN/Inf hard-fails. Build & run on a GPU node (run_all_tests.sh).
!===============================================================================
program test_mean_gpu
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,         only: dp, ip
  use helm_band_mod, only: helm_band_setup
  use mean_mod,      only: mean_stage, mean_stage_cfr, mean_flux_setup
  use mean_gpu_mod,  only: mean_gpu_ws, mean_gpu_alloc, mean_gpu_free, &
                           mean_stage_gpu, mean_stage_cfr_gpu
  use cudafor
  implicit none

  integer(ip) :: nv(2) = [ 32, 64 ]
  real(dp), parameter :: dt = 5.0e-3_dp, Re = 2800.0_dp, Qtar = 1.0_dp
  real(real64) :: tol
  logical :: all_ok = .true.
  integer(ip) :: c

  tol = 1.0e-9_real64
  do c = 1, 2
     call run_grid(nv(c), all_ok)
  end do
  call reinit_leg(all_ok)
  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine run_grid(n, all_ok)
    integer(ip), intent(in)    :: n
    logical,     intent(inout) :: all_ok
    ! host structure / CFR precompute
    real(dp) :: sd(0:n-2), st(0:n-2), bd(0:n-2), boff
    real(dp) :: Uh_all(0:n,3), Ih_all(3)
    ! host state + forcing (fixed across the 3 stages -> a clean math check)
    real(dp) :: u0c(0:n), w0c(0:n), Nu(0:n), Nw(0:n), Nup(0:n), Nwp(0:n)
    real(dp) :: betaC, rr
    ! GPU mirrors
    real(dp),    device :: sd_d(0:n-2), st_d(0:n-2), bd_d(0:n-2)
    complex(dp), device :: u0_d(0:n,1,1), w0_d(0:n,1,1)
    complex(dp), device :: Nu_d(0:n,1,1), Nw_d(0:n,1,1), Nup_d(0:n,1,1), Nwp_d(0:n,1,1)
    real(dp),    device :: Uh_d(0:n)                ! CFR unit response is real (rank-1)
    real(dp),    device :: beta_d(1)
    real(dp) :: u0g(0:n), w0g(0:n), betaG
    complex(dp) :: tmp(0:n)
    type(mean_gpu_ws) :: ws
    integer(ip) :: ks, k, istat
    real(real64) :: eu, ew, eb

    call helm_band_setup(n, sd, st, bd, boff)
    call mean_flux_setup(n, dt, Re, sd, st, bd, boff, Uh_all, Ih_all)

    ! deterministic pseudo-random init (no RNG-state dependence across the suite)
    do k = 0, n
       rr = real(modulo(7*k+3, 13), dp)/13.0_dp - 0.5_dp;  u0c(k) = rr
       rr = real(modulo(5*k+1, 11), dp)/11.0_dp - 0.5_dp;  w0c(k) = rr
       rr = real(modulo(3*k+2,  7), dp)/ 7.0_dp - 0.5_dp;  Nu(k)  = rr
       rr = real(modulo(2*k+5,  9), dp)/ 9.0_dp - 0.5_dp;  Nw(k)  = rr
       Nup(k) = 0.7_dp*Nu(k);  Nwp(k) = 0.6_dp*Nw(k)
    end do

    sd_d = sd;  st_d = st;  bd_d = bd
    u0_d(:,1,1) = cmplx(u0c, 0.0_dp, dp);  w0_d(:,1,1) = cmplx(w0c, 0.0_dp, dp)
    Nu_d(:,1,1) = cmplx(Nu, 0.0_dp, dp);   Nw_d(:,1,1) = cmplx(Nw, 0.0_dp, dp)
    Nup_d(:,1,1)= cmplx(Nup,0.0_dp, dp);   Nwp_d(:,1,1)= cmplx(Nwp,0.0_dp, dp)
    call mean_gpu_alloc(ws, n)

    do ks = 1, 3
       ! CPU reference (u0 with CFR, w0 plain)
       call mean_stage_cfr(n, ks, dt, Re, Qtar, sd, st, bd, boff, &
                           Uh_all(:,ks), Ih_all(ks), u0c, Nu, Nup, betaC)
       call mean_stage    (n, ks, dt, Re, 0.0_dp, sd, st, bd, boff, w0c, Nw, Nwp)
       ! GPU
       Uh_d = Uh_all(:,ks)
       call mean_stage_cfr_gpu(ws, n, ks, dt, Re, Qtar, sd_d, st_d, bd_d, boff, &
                               Uh_d, Ih_all(ks), u0_d, Nu_d, Nup_d, beta_d)
       call mean_stage_gpu    (ws, n, ks, dt, Re, 0.0_dp, sd_d, st_d, bd_d, boff, &
                               w0_d, Nw_d, Nwp_d)
    end do
    istat = cudaDeviceSynchronize()

    tmp = u0_d(:,1,1);  u0g = real(tmp, dp)
    tmp = w0_d(:,1,1);  w0g = real(tmp, dp)
    betaG = beta_d(1)

    eu = colmax(n, u0g, u0c);  ew = colmax(n, w0g, w0c)
    eb = abs(real(betaG,real64) - real(betaC,real64))

    write(*,'(a,i0,a)') ' --- n=', n, ' ---'
    call report('u0 (CFR)  GPU vs CPU mean_mod', eu, tol, all_ok)
    call report('w0 (plain) GPU vs CPU mean_mod', ew, tol, all_ok)
    call report('beta (mean dp/dx) GPU vs CPU',   eb, tol, all_ok)
    call mean_gpu_free(ws)
  end subroutine run_grid

  ! a SAME-SIZE second mean_gpu_alloc must deliver
  ! the zeroed history a fresh alloc promises (the old bare early return kept
  ! stale Nu0/Nw0/beta alive into the next run's first step). Poison with
  ! finite sentinels, re-alloc at the same n, and require all-zero readback.
  subroutine reinit_leg(all_ok)
    logical, intent(inout) :: all_ok
    integer(ip), parameter :: nl = 32
    type(mean_gpu_ws) :: wsl
    complex(dp) :: hcol(0:nl)
    real(dp)    :: bh(1)
    logical     :: okz
    character(6):: tag
    call mean_gpu_alloc(wsl, nl)
    wsl%Nu0_d  = (12.5_dp, -3.0_dp)
    wsl%Nw0_d  = (-7.25_dp, 9.5_dp)
    wsl%beta_d = 4.0_dp
    call mean_gpu_alloc(wsl, nl)          ! same n: the early-return path
    hcol = wsl%Nu0_d(:,1,1);  okz = all(hcol == (0.0_dp, 0.0_dp))
    hcol = wsl%Nw0_d(:,1,1);  okz = okz .and. all(hcol == (0.0_dp, 0.0_dp))
    bh   = wsl%beta_d;        okz = okz .and. bh(1) == 0.0_dp
    call mean_gpu_free(wsl)
    if (okz) then
       tag = '[PASS]'
    else
       tag = '[FAIL]';  all_ok = .false.
    end if
    write(*,'(a,2x,a)') tag, 'same-n re-alloc re-zeroes mean history (Nu0/Nw0/beta)'
  end subroutine reinit_leg

  ! max|a-b| with NaN/Inf -> huge (so a non-finite result can never PASS)
  real(real64) function colmax(n, a, b) result(m)
    integer(ip), intent(in) :: n
    real(dp),    intent(in) :: a(0:n), b(0:n)
    real(real64) :: d
    integer(ip)  :: k
    m = 0.0_real64
    do k = 0, n
       d = abs(real(a(k),real64) - real(b(k),real64))
       if (.not. (d == d)) d = huge(1.0_real64)
       if (abs(a(k)) > huge(1.0_dp)) d = huge(1.0_real64)
       m = max(m, d)
    end do
  end function colmax

  subroutine report(name, e, tol, all_ok)
    character(*), intent(in)    :: name
    real(real64), intent(in)    :: e, tol
    logical,      intent(inout) :: all_ok
    character(6) :: tag
    if (e <= tol) then
       tag = '[PASS]'
    else
       tag = '[FAIL]';  all_ok = .false.
    end if
    write(*,'(a,2x,a,2x,a,es12.4,2x,a,es10.2)') tag, name, 'err=', e, 'tol=', tol
  end subroutine report

end program test_mean_gpu
