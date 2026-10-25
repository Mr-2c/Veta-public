!===============================================================================
! main.F90  --  minimal single-GPU driver for the channel-flow DNS solver.
!
! Ties cfl_mod + step_full + io_mod into a launchable run:
!     setup (build C2C wavenumbers, operators, vif/CFR precompute, nl workspace)
!     init  (rest, or restart from a checkpoint)
!     loop  { step_full;  every log_every: CFL monitor + diagnostics;
!                          every chk_every: checkpoint }
!     final (laminar Poiseuille validation against the analytic profile)
!
! FIRST TARGET = laminar Poiseuille: from rest, the constant-bulk-
! velocity drive builds u0(y) toward (Re*beta/2)(1-y^2) with beta = 3Q/Re, where
! Q is the target BULK VELOCITY U_b = (1/H)int u0 dy (H = 2). Centreline u_c =
! 1.5 Q; fluctuations stay ~0. The analytic profile in Chebyshev coefficients is
! a_0 = 3Q/4, a_2 = -3Q/4, rest 0 (since 1-y^2 = (T_0 - T_2)/2); the run compares
! u0 to that.
!
! CONVENTIONS:
!  * The driver builds the C2C wavenumbers kx(nx),kz(nz) in cuFFT order ITSELF
!    (resolving the stale grid_mod R2C mismatch on the compute path).
!  * Time step: BOTH modes are wired. Fixed dt monitors the CFL (CFL_STAB
!    printed at startup, live dt*fmax checked every log_every); variable_dt
!    = .true. drives dt from
!    cfl_target with dt_max/dt_min clamps (rest-start fmax~0 is exactly why
!    dt_max exists), dt_recompute_tol hysteresis, and exact-mark landing on
!    t_end/snap/stats/chk marks -- see the variable-dt block in the time loop.
!  * The CFL monitor reuses the nl_workspace's plans + scratch (idle between steps)
!    rather than allocating its own.
!
! Config via namelist /run/ in ./input.nml (optional; sensible laminar defaults if
! absent). nan_every (default 10, 0 disables): cadence of the NaN/Inf
! watchdog -- on detection writes crash_good.bin (restartable last-good
! snapshot) + crash_nan.bin (poisoned state) and stops with code 2.
! The printed CFL is the CONVENTIONAL dt*sum(|u_i|/dx_i); its stability
! ceiling is sqrt(3)/pi ~ 0.5513 (banner + warn). cfl_target uses the same
! conventional units (sound production value: ~0.32).
! Build (links the full stack):
!   nvfortran -cuda -cudalib=cufft -o dns_main main.F90 \
!     types.o cheb_shen.o cheb_bih.o helm_band_mod.o mean_mod.o vsolve_if_mod.o \
!     wavenum_mod.o cheb_ops_mod.o stats_mod.o init_mod.o cfl_mod.o velocity_mod.o \
!     forcing_mod.o helm_band_gpu_mod.o spec_helpers_mod.o vsolve_if_gpu_mod.o \
!     stepper_lin_mod.o transforms_mod.o cheb_fft_mod.o fft3d_mod.o \
!     spectral3d_mod.o nonlinear_mod.o stepper_mod.o io_mod.o
!===============================================================================
program main
  use kinds,           only: wp, dp, ip
  use cudafor
  use cheb_shen,       only: glc_nodes, cheb_forward, cheb_deriv
  use wavenum_mod,     only: build_wavenumbers, build_wavenumbers_r2c
  use helm_band_mod,   only: helm_band_setup
  use mean_mod,        only: mean_flux, bulk_velocity, mean_flux_setup, MCSUB, &
                             MGAMMA, MZETA                 ! table assert refs
  use mean_gpu_mod,    only: MCSUB_G => MCSUB, MGAMMA_G => MGAMMA, &
                             MZETA_G => MZETA              ! duplicate copy
  use ygrid_mod,       only: ygrid_read                    ! FD wiring
  use fdops_mod,       only: fd_build_bands, fd_wall_weights, fd_apply_band
  use fd_mean_mod,     only: fd_mean_flux_setup
  use fd_helm_gpu_mod, only: fd_helm_cache_build, fd_helm_cache_stats
  use yderiv_gpu_mod,  only: yderiv_set_fd
  use yops_mod,        only: yops_set_fd, yop_backend_tag, yop_ygrid_tag
  use cfl_mod,         only: CFL_STAB, cfl_inv_dy, cfl_freq_max, cfl_dt
  use velocity_mod,    only: recover_uw
  use spectral3d_mod,  only: to_physical_da_r2c, s3d_set_ybasis_fd
  use cheb_fft_mod,    only: cheb_fft_scratch_free
  use nonlinear_mod,   only: nl_workspace, nl_setup, nl_destroy, mask_nyquist, copy_c3
  use stepper_lin_mod, only: stepper_ev_setup, steplin_csub_ok
  use stepper_mod,     only: step_full_ws, stepper_ws, stepper_ws_alloc, stepper_ws_free, &
                             stepper_coeffs_ok
  use solver_ops_mod,  only: band_ops_t, vsolve_ops_t, mean_ops_t, &
                             band_ops_alloc, band_ops_free, vsolve_ops_alloc, &
                             vsolve_ops_free, mean_ops_alloc, mean_ops_free, &
                             band_ops_fd_init, y_vif_setup, &
                             y_apply_L, y_vres_max, &           ! v-solve residual probe
                             y_vres_max_win, y_vres_galerkin    ! galerkin/interior canaries
  use diag_mod,        only: report_device_memory, fft_size_guard
  use io_mod,          only: io_write_state, io_read_state, io_set_ybasis
  use stats_mod,       only: stats_t, stats_init, stats_accumulate, stats_finalize, &
                             stats_destroy, re_tau, divergence_check, &
                             field_sumabs_c
  use init_mod,        only: init_turbulent, init_turbulent_fd
  implicit none
  include 'veta_build.inc'   ! VETA_BUILD_HASH/DATE (generated by the builders)

  real(dp), parameter :: PI = 3.141592653589793238462643_dp

  ! --- config (namelist; defaults = laminar Poiseuille validation) -------------
  integer(ip) :: n, nx, nz, nkx, n_steps, log_every, chk_every
  real(dp)    :: Lx, Lz, Re, Qtar, dt
  logical     :: do_restart, variable_dt
  real(dp)    :: cfl_target, dt_max, dt_min, dt_recompute_tol
  character(len=256) :: restart_file, chk_file
  character(len=256) :: infile               ! input-file name (CLI arg or 'input.nml')
  ! turbulent-run config
  logical            :: turbulent
  real(dp)           :: eps
  integer(ip)        :: pseed, kcut, stats_every, stats_start, div_every, time_every
  character(len=256) :: stats_file
  ! FD-y backend: a nonempty ygrid_file flips the WHOLE y-basis to
  ! banded FD on the file's nodes (state = point values). fd_kl = band half-width.
  character(len=256) :: ygrid_file
  integer(ip)        :: fd_kl
  real(dp)           :: fd_cache_gb   ! helm factor-cache cap (0 = off)
  ! Multi-GPU-only keys, PARSED HERE SO ONE DECK DRIVES BOTH BINARIES. The
  ! shipped decks are shared with src/parallel/main_par.F90, whose &run also
  ! carries the snapshot writer and the --validate horizon. None of that exists
  ! single-GPU; a namelist read is strict, so without these names every shared
  ! deck dies here on the first unknown one. Accepted, never honoured -- the
  ! notice below the read says so out loud.
  integer(ip)        :: snap_every, validate_steps
  real(dp)           :: snap_start
  logical            :: snap_fp32, snap_pressure
  character(len=256) :: snap_file
  namelist /run/ n, nx, nz, Lx, Lz, Re, Qtar, dt, n_steps, log_every, chk_every, &
                 do_restart, restart_file, chk_file, &
                 variable_dt, cfl_target, dt_max, dt_min, dt_recompute_tol, &
                 turbulent, eps, pseed, kcut, stats_every, stats_start, stats_file, &
                 div_every, time_every, nan_every, ygrid_file, fd_kl, fd_cache_gb, &
                 snap_every, snap_start, snap_fp32, snap_file, snap_pressure, &
                 validate_steps

  ! --- setup / state ------------------------------------------------------------
  type(nl_workspace) :: ws
  type(stepper_ws)   :: sws          ! run-lifetime stepper scratch
  real(dp), allocatable :: kxh(:), kzh(:), y(:), inv_dy(:)
  ! FD-y state (allocated only in FD mode)
  logical :: fd_mode, have_w, okg
  character(len=256) :: ymsg
  real(dp), allocatable :: yfile(:), wfile(:), wlof(:), whif(:)
  real(dp), allocatable :: d1f(:,:), d2f(:,:), UhF(:,:), IhF(:)
  complex(dp), device, allocatable :: fvt1(:,:,:), fvt2(:,:,:), fvt3(:,:,:), &
       fvt4(:,:,:), fvt5(:,:,:), fvt6(:,:,:), fvt7(:,:,:)   ! y_vif_setup lends
  type(band_ops_t) :: bops        ! y-operator band structure (host sdh/sth/bdh + device sd/st/bd)
  type(mean_ops_t)   :: mops      ! CFR mean precompute
  type(vsolve_ops_t) :: vops      ! shifted v-solve lift + influence
  real(dp)              :: inv_dx, inv_dz
  real(dp), device, allocatable :: kx(:), kz(:), inv_dy_d(:)
  complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:)
  complex(dp), device, allocatable :: Nv_prev(:,:,:), Neta_prev(:,:,:)
  ! NaN watchdog: last-known-good state (device) + its time/step
  complex(dp), device, allocatable :: v_good(:,:,:), eta_good(:,:,:)
  complex(dp), device, allocatable :: u0_good(:), w0_good(:)
  complex(dp), allocatable :: u0h(:)
  ! statistics + diagnostics
  type(stats_t) :: sacc
  real(dp), allocatable :: mprof(:), mvp(:), mwp(:), uup(:), vvp(:), wwp(:), uvp(:)
  complex(dp), device, allocatable :: dvh_diag(:,:,:)

  real(dp)    :: t, flux, fmax, cfl, beta_an, errpar, cl, aan, dt_new, ret, maxdiv
  real(dp)    :: fdiv                ! divergence-probe private fmax
  real(dp)    :: cfl_max             ! ceiling on the CONVENTIONAL CFL = sqrt(3)/pi
  real(dp)    :: t_good, swd1, swd2  ! watchdog: good-state time, field sums
  integer(ip) :: step_good, nan_every
  integer(ip) :: step, it, istat, nxd, nzd, kk, u, n_recompute
  integer(ip) :: n_clamp_hi, n_clamp_lo     ! [guardrail] steps where dt hit dt_max / dt_min
  real(dp)    :: dt_cfl_raw                  ! [guardrail] unclamped CFL dt, for the warning
  integer(ip) :: rstep, ios
  real(dp)    :: rRe, rt, rdt, rQ
  real(dp)    :: rLx, rLz                    ! checkpoint box identity
  logical     :: okio
  ! --- timing (diagnostic; time_every=0 => no instrumentation/sync overhead) ----
  logical     :: timing
  integer(8)  :: clk_rate, c_a, c_b, c_run0, c_run1
  real(dp)    :: tm_dtmon, tm_rec, tm_step, tm_stats, tm_io
  real(dp)    :: tot_dtmon, tot_rec, tot_step, tot_stats, tot_io, wall_s

  ! ===== defaults (override via the input file) ================================
  n = 32;  nx = 16;  nz = 16
  Lx = 4.0_dp*PI;  Lz = 2.0_dp*PI;  Re = 10.0_dp;  Qtar = 1.0_dp
  dt = 5.0e-3_dp;  n_steps = 4000;  log_every = 200;  chk_every = 1000
  do_restart = .false.;  restart_file = 'restart.bin';  chk_file = 'chk.bin'
  ! cfl_target is in the CONVENTIONAL units (|u|dt/dx), whose stability ceiling
  ! is sqrt(3)/pi ~ 0.5513 -- NOT sqrt(3). A cfl_target of 1.0 implies an
  ! eigenvalue CFL ~ pi > sqrt(3) => UNSTABLE if variable_dt is enabled and
  ! dt_max is raised above the CFL-implied dt. The default is the documented
  ! production value (the shipped vardt decks use 0.30; both sit safely below
  ! 0.5513).
  variable_dt = .false.;  cfl_target = 0.32_dp
  dt_max = 2.0e-2_dp;  dt_min = 1.0e-4_dp;  dt_recompute_tol = 5.0e-2_dp
  turbulent = .false.;  eps = 0.05_dp;  pseed = 12345;  kcut = 8
  stats_every = 50;  stats_start = 0;  stats_file = 'stats.dat';  div_every = 0
  time_every = 0;  nan_every = 10
  ygrid_file = '';  fd_kl = 4;  fd_cache_gb = 4.0_dp
  ! multi-GPU-only (accepted, not honoured); same defaults as main_par
  snap_every = 0;  snap_start = 0.0_dp;  snap_fp32 = .true.;  snap_file = 'snap.bin'
  snap_pressure = .false.;  validate_steps = 5
  ! Input file: first command-line argument if given, else 'input.nml'.
  if (command_argument_count() >= 1) then
     call get_command_argument(1, infile)
  else
     infile = 'input.nml'
  end if
  open(newunit=u, file=trim(infile), status='old', action='read', iostat=ios)
  if (ios == 0) then
     read(u, nml=run, iostat=ios)
     close(u)
     ! present-but-unparseable deck = FATAL (incremental namelist
     ! assignment leaves a defaults/deck mixture otherwise; see main_par).
     if (ios /= 0) then
        write(*,'(a,a,a,i0,a)') '** FATAL: ', trim(infile), &
           ' is present but unparseable (iostat = ', ios, ') -- fix the deck'
        ! Most common cause by far: a deck written for the multi-GPU driver.
        ! Its &run carries run-control this binary does not have, and a
        ! namelist read dies on the first unknown name without naming it.
        write(*,'(a)') '   (multi-GPU decks carry keys this driver lacks -- the' // &
                       ' time-based cadences'
        write(*,'(a)') '    t_end/snap_dt/stats_dt/chk_dt/stats_start_t,' // &
                       ' max_wall_hours, restart_auto,'
        write(*,'(a)') '    u_galilean, io_hints, a2a_overlap, hsym_every.' // &
                       ' Run those with build_par/Veta_dp|sp.)'
        stop 1
     end if
  else
     write(*,'(a,a,a)') '** input file "', trim(infile), '" not found; using built-in defaults'
  end if
  write(*,'(a,a)') ' input file: ', trim(infile)
  ! Say what was dropped. snap_every>0 asks for field dumps; snap_pressure also
  ! adds the <p>, <p'^2> and dp/dx columns to the stats file. Both are
  ! multi-GPU-only, so a shared deck would otherwise hand back a quietly
  ! thinner output set than it asked for.
  if (snap_every > 0 .or. snap_pressure) then
     write(*,'(a)') ' NOTE: snapshots and pressure stats are multi-GPU only' // &
                    ' (build_par/Veta_dp|sp);'
     write(*,'(a)') '       snap_* ignored here -- no field dumps, and stats' // &
                    ' carry no pressure columns.'
  end if

  ! The SMR tables (CSUB/GAMMA/ZETA) live as verbatim copies in FIVE
  ! modules (mean_mod, mean_gpu_mod, stepper_mod, stepper_lin_mod,
  ! par_stepper_mod); the mean solves and the [res] Galerkin canary assume
  ! every copy identical. Checked bitwise, once.
  if (.not. (all(MCSUB == MCSUB_G) .and. all(MGAMMA == MGAMMA_G) .and. &
             all(MZETA == MZETA_G) .and. &
             stepper_coeffs_ok(MCSUB, MGAMMA, MZETA) .and. &
             steplin_csub_ok(MCSUB))) then
     write(*,'(a)') '** FATAL: SMR coefficient tables diverged across modules (12.4)'
     error stop 1
  end if

  nxd = 3_ip*nx/2_ip;  nzd = 3_ip*nz/2_ip
  nkx = nx/2_ip + 1_ip                              ! [R2C] spectral x-extent
  ! (the fft_size_guard call sits below the fd_mode determination: it also
  ! checks the 2n Chebyshev even-extension length, which is skipped under
  ! FD-y where that leg never executes.)

  ! ===== host allocate + wavenumbers (C2C, cuFFT order) ========================
  allocate(kxh(nkx), kzh(nz), y(0:n), inv_dy(0:n))
  call band_ops_alloc(bops, n);  call mean_ops_alloc(mops, n)
  ! PHYSICAL wavenumbers k = 2*pi*m/L via the canonical builder (wavenum_mod).
  ! test_box_scaling pins the scaling against the analytic definition.
  call build_wavenumbers_r2c(nx, Lx, kxh)           ! [R2C] kx half-axis
  call build_wavenumbers(nz, Lz, kzh)
  call glc_nodes(n, y)
  ! FD mode: read + validate the grid file, refuse the paths that
  ! are not FD-ready yet, and swap y so CFL/stats/output follow the file nodes.
  fd_mode = (len_trim(ygrid_file) > 0)
  call fft_size_guard(nx, nz, nxd, nzd, merge(0_ip, 2_ip*n, fd_mode))  ! advisory; incl. the 2n cheb length
  if (fd_mode) then
     allocate(yfile(n+1), wfile(n+1), wlof(n+1), whif(n+1))
     allocate(d1f(2*fd_kl+1, n+1), d2f(2*fd_kl+1, n+1), UhF(n+1,3), IhF(3))
     wfile = 0.0_dp
     call ygrid_read(trim(ygrid_file), n+1, yfile, wfile, have_w, okg, ymsg)
     if (.not. okg) then
        write(*,'(a)') ' ** ygrid REFUSED: '//trim(ymsg);  stop 1
     end if
     if (.not. have_w) then          ! CFR closure needs the weights
        write(*,'(a)') ' ** FD mode REFUSED: grid file has no w column '// &
                       '(the CFR flux closure needs quadrature weights)';  stop 1
     end if
     ! checkpoints/restart are FD-capable: headers stamp
     ! ybasis=fd-point / ygrid=file + a ygrid_moment key (grid identity; the v2
     ! format has no y record), and io_read_state refuses a Chebyshev file or a
     ! different FD grid instead of silently corrupting.
     call io_set_ybasis('fd-point', 'file', yfile)
     y = yfile                       ! (0:n) <- file nodes (CFL + stats output)
     call fd_build_bands(n+1, fd_kl, yfile, d1f, d2f)
     call fd_wall_weights(n+1, fd_kl, yfile, wlof, whif)
  end if
  call cfl_inv_dy(n, y, inv_dy)
  ! CFL convention: the printed CFL is the CONVENTIONAL one everyone expects,
  !     CFL = dt * max( |u|/dx + |v|/dy_loc + |w|/dz ),
  ! and the pi lives in the CEILING instead: RK3's imaginary-axis bound is
  ! |lambda|dt <= sqrt(3) with Fourier |lambda| = |u|*kmax = pi*|u|/dx, so the
  ! ceiling on the conventional number is cfl_max = sqrt(3)/pi ~ 0.5513
  ! (printed in the banner; used by the warn; cfl_target is in the SAME
  ! conventional units). The y term is the local-GLC-spacing heuristic
  ! (subdominant: v ~ y^2 at the walls).
  inv_dx = real(nx, dp)/Lx
  inv_dz = real(nz, dp)/Lz
  cfl_max = CFL_STAB/PI

  ! ===== operators + precompute ================================================
  call helm_band_setup(n, bops%sdh, bops%sth, bops%bdh, bops%boff)
  call mean_flux_setup(n, dt, Re, bops%sdh, bops%sth, bops%bdh, bops%boff, mops%Uhm, mops%Ih)
  call nl_setup(ws, n+1, nx, nz, kxh, kzh)

  ! ===== device allocate + upload ==============================================
  allocate(kx(nkx), kz(nz), inv_dy_d(0:n))
  call vsolve_ops_alloc(vops, n, nkx, nz)
  allocate(eta(0:n,nkx,nz), v(0:n,nkx,nz), u0(0:n), w0(0:n))
  allocate(Nv_prev(0:n,nkx,nz), Neta_prev(0:n,nkx,nz))
  allocate(v_good(0:n,nkx,nz), eta_good(0:n,nkx,nz), u0_good(0:n), w0_good(0:n))
  allocate(u0h(0:n))
  kx = kxh;  kz = kzh;  inv_dy_d = inv_dy
  bops%sd = bops%sdh;  bops%st = bops%sth;  bops%bd = bops%bdh      ! H2D band structure
  if (fd_mode) then
     ! Flip every y seam to FD, repopulate the mean apparatus from
     ! the FD unit-source responses, and build the vif precompute
     ! through the dispatch helper (Cheb runs keep stepper_ev_setup verbatim).
     call band_ops_fd_init(bops, n, fd_kl, yfile, wfile, have_w)
     call yderiv_set_fd(n+1, fd_kl, d1f)
     call yops_set_fd(n, wlof, whif, wfile)
     call s3d_set_ybasis_fd(.true.)
     allocate(fvt1(0:n,nkx,nz), fvt2(0:n-2,nkx,nz), fvt3(0:n-2,nkx,nz), &
              fvt4(0:n,nkx,nz), fvt5(0:n,nkx,nz), fvt6(0:n,nkx,nz), fvt7(0:n,nkx,nz))
     call fd_precompute_dt()
  else
     call stepper_ev_setup(n, nkx, nz, dt, Re, kxh, kzh, vops%vm, vops%vp, vops%Cinv)
  end if
  call stepper_ws_alloc(sws, n, nkx, nz, fd=fd_mode)   ! one alloc for the whole run
                                           ! (FD-y skips the 3 Shen buffers, -48 B/pt)
  call report_device_memory(n, nkx, nz, nxd, nzd)   ! per-tier footprint + precision projections

  ! ===== init: rest, or restart ================================================
  if (do_restart) then
     call io_read_state(restart_file, n, nx, nz, rRe, rt, rdt, rQ, rstep, &
                        v, eta, u0, w0, okio, Lx=rLx, Lz=rLz)
     if (.not. okio) then
        write(*,'(a)') '** restart read FAILED (grid/precision mismatch or missing file)'
        stop 1
     end if
     ! Lx/Lz/Re/Qtar are compared: without that, a checkpoint from a different
     ! box would restart silently reinterpreted onto this deck's wavenumbers.
     ! Box mismatch = fatal; Re/Qtar mismatch = loud note.
     if (rLx > 0.0_dp .and. (abs(rLx-Lx) > 1.0e-9_dp*Lx .or. abs(rLz-Lz) > 1.0e-9_dp*Lz)) then
        write(*,'(a,2es13.5,a,2es13.5)') '** FATAL: checkpoint box (Lx,Lz) = ', rLx, rLz, &
             ' /= deck ', Lx, Lz
        stop 1
     end if
     ! dt included -- parity with the parallel twin's note: under fixed dt a
     ! deck/checkpoint dt mismatch changes the trajectory, and without this it
     ! would leave no trace in the log.
     if (abs(rRe-Re) > 1.0e-9_dp*max(Re,1.0_dp) .or. abs(rQ-Qtar) > 1.0e-12_dp &
         .or. abs(rdt-dt) > 1.0e-12_dp) &
        write(*,'(a,3es12.4,a)') '   [restart] note: checkpoint Re/dt/Qtar (', rRe, rdt, rQ, &
             ') differ from the deck; continuing with the deck values.'
     t = rt;  step = rstep
     write(*,'(a,a,a,i0,a,f10.4)') ' restarted from ', trim(restart_file), &
          '  (step=', step, ', t=', t, ')'
  else if (turbulent) then
     if (fd_mode) then
        call init_turbulent_fd(n, nkx, nz, kxh, kzh, Qtar, eps, pseed, kcut, &
                               yfile, wfile, fd_kl, d1f, v, eta, u0, w0)
     else
        call init_turbulent(n, nkx, nz, kxh, kzh, Qtar, eps, pseed, kcut, v, eta, u0, w0)
     end if
     t = 0.0_dp;  step = 0
  else
     eta = (0.0_dp,0.0_dp);  v = (0.0_dp,0.0_dp)
     u0  = (0.0_dp,0.0_dp);  w0 = (0.0_dp,0.0_dp)
     t = 0.0_dp;  step = 0
  end if
  ! Enforce the Nyquist-zero convention on the starting state. init_mod
  ! already guarantees it; this also sanitises checkpoints whose states may
  ! carry contaminated Nyquist columns. nl_evaluate masks its forcings, so the
  ! columns then stay exactly zero.
  call mask_nyquist(n+1, nx, nz, v)
  call mask_nyquist(n+1, nx, nz, eta)
  ! watchdog: the initial state is the first known-good snapshot
  v_good = v;  eta_good = eta;  u0_good = u0;  w0_good = w0
  t_good = t;  step_good = step
  Nv_prev = (0.0_dp,0.0_dp);  Neta_prev = (0.0_dp,0.0_dp)

  ! ===== statistics + diagnostics setup ========================================
  if (turbulent) then
     call stats_init(sacc, n, nxd, nzd)
     allocate(mprof(0:n), mvp(0:n), mwp(0:n), uup(0:n), vvp(0:n), wwp(0:n), uvp(0:n))
  end if
  if (div_every > 0) allocate(dvh_diag(0:n,nkx,nz))     ! mag fused away

  ! ===== banner ================================================================
  beta_an = 3.0_dp*Qtar/Re
  write(*,'(a)')        '=============================================================='
  write(*,'(a)')        ' VETA SUBLIMESON -- single-GPU driver (channel DNS, KMM lineage;'
  write(*,'(a)')        '                    sister of Pola Sublimeson)'
  write(*,'(4a)')       '   build = ', VETA_BUILD_HASH, ', ', VETA_BUILD_DATE
  write(*,'(a,i0,a,i0,a,i0)')     '   grid (n,nx,nz) = ', n, ',', nx, ',', nz
  write(*,'(a,f8.3,a,f8.3)')      '   box (Lx,Lz)    = ', Lx, ',', Lz
  write(*,'(a,f10.3,a,f8.4)')     '   Re = ', Re, '   U_b(target) = ', Qtar
  write(*,'(a,a,a,a)')            '   y-basis = ', yop_backend_tag(), &
                                  ' / y-grid = ', yop_ygrid_tag()
  if (fd_mode) write(*,'(a,a,a,i0)') '   ygrid file = ', trim(ygrid_file), &
                                  ',  kl = ', fd_kl
  if (variable_dt) then
     write(*,'(a,es10.3,a)')      '   dt = ', dt, ' (initial; VARIABLE -- CFL-driven)'
     write(*,'(a,f6.3,a,es9.2,a,es9.2,a)') '        cfl_target = ', cfl_target, &
                                  ',  dt clamped to [', dt_min, ',', dt_max, ']'
     write(*,'(a,f6.3)')          '        dt_recompute_tol = ', dt_recompute_tol
     ! [guardrail] a tight tolerance retunes the (expensive) vif precompute often.
     ! Each retune rebuilds 3 x nkx*nz banded setups -- it is SLOW to change dt, so
     ! a wide tolerance (>=0.1) is strongly preferred (the input
     ! decks say so too). Warn, do not override the user's choice.
     if (dt_recompute_tol < 0.1_dp) write(*,'(a,f5.3,a)') &
        '   [warn] dt_recompute_tol = ', dt_recompute_tol, ' is TIGHT (<0.10): the vif'// &
        ' precompute will retune often (slow). Prefer >=0.15 unless you need tight dt.'
  else
     write(*,'(a,es10.3,a)')      '   dt = ', dt, ' (fixed)'
  end if
  write(*,'(a,i0)')               '   steps = ', n_steps
  write(*,'(a,f8.4,a)')           '   max stable CFL = ', cfl_max, &
                                  '  (= sqrt(3)/pi, conventional |u|dt/dx units) [warn only]'
  write(*,'(a,f10.6)')            '   analytic laminar beta = 3Q/Re = ', beta_an
  if (turbulent) then
     write(*,'(a,es9.2,a,i0,a,i0)') '   IC = TURBULENT: eps = ', eps, ', kcut = ', kcut, &
                                    ', seed = ', pseed
     write(*,'(a,i0,a,i0,a,a)')     '   stats every ', stats_every, ' steps after step ', &
                                    stats_start, ' -> ', trim(stats_file)
  else
     write(*,'(a)')                 '   IC = laminar (from rest; CFR builds the parabola)'
  end if
  write(*,'(a)')        '--------------------------------------------------------------'
  write(*,'(a)')        '   step        t          dt        CFL       U_bulk    u0(0)    Re_tau'

  ! ===== time loop =============================================================
  n_recompute = 0
  n_clamp_hi = 0;  n_clamp_lo = 0
  timing = (time_every > 0)
  call system_clock(count_rate=clk_rate)
  tot_dtmon = 0.0_dp; tot_rec = 0.0_dp; tot_step = 0.0_dp; tot_stats = 0.0_dp; tot_io = 0.0_dp
  if (timing) write(*,'(a)') '   [timing on] per-step ms: dtmon = variable-dt fmax probe, '// &
                             'rec = stepper_ev_setup+mean_flux_setup, step = step_full, stats, io'
  call system_clock(c_run0)
  do it = 1, n_steps
     tm_dtmon = 0.0_dp; tm_rec = 0.0_dp; tm_step = 0.0_dp; tm_stats = 0.0_dp; tm_io = 0.0_dp
     ! --- variable dt: set this step's dt from the current state, clamp, retune ---
     if (variable_dt) then
        if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_a); end if
        call monitor_fmax(fmax)                            ! fmax of the current state
        if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_b)
                          tm_dtmon = 1.0e3_dp*real(c_b-c_a,dp)/real(clk_rate,dp); end if
        dt_cfl_raw = cfl_dt(fmax, cfl_target)              ! unclamped CFL-implied dt
        dt_new = min(max(dt_cfl_raw, dt_min), dt_max)
        ! [guardrail] detect a BINDING clamp once the flow has developed (skip the
        ! rest-start step 1, where fmax~0 => dt_cfl_raw is huge and clamping to
        ! dt_max is the CORRECT, intended behaviour). A binding dt_max means the
        ! CFL is being held BELOW cfl_target -- adaptation is (partly) defeated and
        ! dt_max is too tight; a binding dt_min means the flow wants a smaller step
        ! than allowed (a stability risk -- the run is riding the floor).
        if (it > 1 .and. fmax > 0.0_dp) then
           if (dt_cfl_raw > dt_max) then
              n_clamp_hi = n_clamp_hi + 1
              if (n_clamp_hi == 1) write(*,'(a,es9.2,a,es9.2,a,f5.3,a)') &
                 '   [warn] dt_max is BINDING: CFL wants dt=', dt_cfl_raw, &
                 ' but dt_max=', dt_max, ' caps it -> CFL held below cfl_target=', &
                 cfl_target, '. Raise dt_max (it should sit ABOVE the operating dt).'
           else if (dt_cfl_raw < dt_min) then
              n_clamp_lo = n_clamp_lo + 1
              if (n_clamp_lo == 1) write(*,'(a,es9.2,a,es9.2,a)') &
                 '   [warn] dt_min is BINDING: CFL wants dt=', dt_cfl_raw, &
                 ' but dt_min=', dt_min, ' floors it -> running ABOVE the stable CFL.'// &
                 ' Lower dt_min or check the flow (possible instability).'
           end if
        end if
        if (it == 1 .or. abs(dt_new - dt) > dt_recompute_tol*dt) then
           dt = dt_new
           ! the dt-dependent precompute must be rebuilt when dt changes; eta is
           ! dt-on-the-fly (step_eta_linear computes lambda_k from dt) so needs none.
           if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_a); end if
           if (fd_mode) then
              call fd_precompute_dt()
           else
              call stepper_ev_setup(n, nkx, nz, dt, Re, kxh, kzh, vops%vm, vops%vp, vops%Cinv)
              call mean_flux_setup(n, dt, Re, bops%sdh, bops%sth, bops%bdh, bops%boff, mops%Uhm, mops%Ih)
           end if
           if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_b)
                             tm_rec = 1.0e3_dp*real(c_b-c_a,dp)/real(clk_rate,dp); end if
           n_recompute = n_recompute + 1
        end if
     end if

     if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_a); end if
     call step_full_ws(ws, sws, n, nkx, nz, dt, Re, kx, kz, bops, vops, mops, &
                       Qtar, u0, w0, eta, v, Nv_prev, Neta_prev)
     if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_b)
                       tm_step = 1.0e3_dp*real(c_b-c_a,dp)/real(clk_rate,dp); end if
     step = step + 1;  t = t + dt

     ! --- NaN/Inf watchdog + last-known-good snapshot -----------------------------
     ! SUM-based probe (field_sumabs_c): a max-reduction silently skips NaNs,
     ! so the [div]/CFL columns can print 0.00 while the state is fully
     ! poisoned. On detection: dump the last GOOD state (crash_good.bin,
     ! restartable) for inspection, plus the poisoned one (crash_nan.bin, to
     ! see WHERE the NaNs live), and stop.
     if (nan_every > 0 .and. mod(it, max(nan_every,1)) == 0) then   ! .and. does NOT short-circuit
        call field_sumabs_c(n+1, nkx, nz, v,   swd1)
        call field_sumabs_c(n+1, nkx, nz, eta, swd2)
        if (.not. (swd1 == swd1 .and. swd2 == swd2 .and. &
                   swd1 < huge(1.0_dp) .and. swd2 < huge(1.0_dp))) then
           write(*,'(a)')        ' !! NON-FINITE STATE DETECTED -- terminating'
           write(*,'(a,i8,a,f10.4)') ' !!   at step ', step, '   t = ', t
           write(*,'(a,i8,a,f10.4)') ' !!   last good state: step ', step_good, '   t = ', t_good
           ! Pass the REAL box: omitting the optionals would stamp the crash
           ! dump with the default 2pi x 2pi header whatever the run's box is.
           call io_write_state('crash_good.bin', n, nx, nz, Re, t_good, dt, Qtar, &
                               step_good, v_good, eta_good, u0_good, w0_good, okio, Lx=Lx, Lz=Lz)
           write(*,'(a,l2)')     ' !!   wrote crash_good.bin (restartable), ok = ', okio
           call io_write_state('crash_nan.bin', n, nx, nz, Re, t, dt, Qtar, &
                               step, v, eta, u0, w0, okio, Lx=Lx, Lz=Lz)
           write(*,'(a,l2)')     ' !!   wrote crash_nan.bin  (poisoned state), ok = ', okio
           stop 2
        end if
        v_good = v;  eta_good = eta;  u0_good = u0;  w0_good = w0
        t_good = t;  step_good = step
     end if

     ! v-solve residual probe, div_every cadence. Runs FIRST among the gated
     ! diagnostics: it reads sws%rhsv (the ks=3 assembled RHS -- y_vif_solve
     ! never modifies it) and lends the dead sws%rhs/Lv/scr. The serial
     ! pressure/snapshot paths do not lend sws, but the ordering + this comment
     ! are kept for twin symmetry with main_par, whose stats-path pressure
     ! diagnostics DO scribble on sws%rhsv/rhs/Lv/scr and would destroy the
     ! probe's inputs. Line 1 (both backends): the STRONG-form recompute
     ! |(L - a3 L^2)v - rhsv|/|rhsv| at the ks=3 a3 = 0.5*MCSUB(3)*dt/Re.
     ! The solve is Shen-GALERKIN, so the strong recompute against the
     ! UNPROJECTED rhsv measures flow/resolution-dependent discretisation
     ! consistency (the marginally-resolved sqrt(a3) wall layer dominates),
     ! NOT solver precision -- it is a resolution/consistency meter, UNGATED:
     ! 0.4-1.4 cheb / ~1e-2 fd in developed turbulence, versus 4-8e-5 /
     ! 4e-4..1e-3 in young flow. Line 2 is the precision canary proper:
     !   cheb: y_vres_galerkin -- per-banded-solve residuals AS SOLVED + the
     !     closure wall-Dv quality. Backward-error quantities, flow- and
     !     deck-INDEPENDENT (cross-checked against a true-basis numpy model).
     !     Measured on a single node: helm 1.8-3.1e-16, closure
     !     5.3e-14..1.9e-13 (the wall-deriv sums carry ~n^2-amplified roundoff
     !     -- a genuinely different floor, hence the split gate). WARN =
     !     measured*10 -> next half-decade: helm 5e-15, closure 2e-12.
     !   fd: y_vres_max_win -- strong residual EXCLUDING 2*kl+1 rows/wall
     !     (BC/closure rows + their banded-apply spread; window also covers
     !     the sqrt(a3) Stokes layer, inside the first 1-2 cells on every
     !     current deck). Interior collocation rows are solver-exact.
     !     Measured on a single node at n=470: 8.9e-13..1.2e-12.
     !     WARN 2e-11 (measured*10 -> next half-decade).
     if (div_every > 0 .and. mod(it, max(div_every,1)) == 0) then
        block
          real(dp) :: aal3_, rres_, rmax_, fmax_
          real(dp) :: r1_, f1_, r2_, f2_, dvv_, dvp_, rg1_, rg2_, rcl_, riw_
          aal3_ = 0.5_dp*MCSUB(3)*dt/Re
          call y_apply_L(bops, n, nkx, nz, kx, kz, v,       sws%rhs, sws%scr)
          call y_apply_L(bops, n, nkx, nz, kx, kz, sws%rhs, sws%Lv,  sws%scr)
          call y_vres_max(n, nkx, nz, aal3_, sws%rhs, sws%Lv, sws%rhsv, rmax_, fmax_)
          rres_ = rmax_/max(fmax_, tiny(1.0_dp))
          write(*,'(a,es10.2,a)') '   [res] v-solve rel residual (ks=3) = ', rres_, &
             '   (strong form; resolution meter, ungated)'
          if (fd_mode) then
             call y_vres_max_win(n, nkx, nz, aal3_, 2*bops%kl+1, n-2*bops%kl-1, &
                                 sws%rhs, sws%Lv, sws%rhsv, rmax_, fmax_)
             riw_ = rmax_/max(fmax_, tiny(1.0_dp))
             write(*,'(a,i0,a,es10.2)') '   [res] fd interior residual (', 2*bops%kl+1, &
                ' wall rows excluded each side) = ', riw_
             if (.not. (riw_ <= 2.0e-11_dp)) write(*,'(a)') &
                '   [res] WARNING: fd interior residual above 2e-11 -- the dp margin is thinning'
          else
             call y_vres_galerkin(bops, n, nkx, nz, aal3_, kx, kz, sws%rhsv, v, &
                                  sws%vgsh, sws%vvsh, sws%rhs, sws%Lv, sws%scr, &
                                  r1_, f1_, r2_, f2_, dvv_, dvp_)
             rg1_ = r1_/max(f1_, tiny(1.0_dp))
             rg2_ = r2_/max(f2_, tiny(1.0_dp))
             rcl_ = dvv_/max(dvp_, tiny(1.0_dp))
             write(*,'(a,es10.2,a,es10.2,a,es10.2)') &
                '   [res] galerkin residual (ks=3): helm1 = ', rg1_, &
                '   helm2 = ', rg2_, '   closure = ', rcl_
             if (.not. (rg1_ <= 5.0e-15_dp .and. rg2_ <= 5.0e-15_dp .and. &
                        rcl_ <= 2.0e-12_dp)) write(*,'(a)') &
                '   [res] WARNING: galerkin residual above gate (helm 5e-15 / closure 2e-12) -- the dp margin is thinning'
          end if
        end block
     end if

     ! --- statistics sampling (turbulent): reuse monitor_fmax's physical velocity --
     if (turbulent .and. stats_every > 0 .and. step >= stats_start .and. &
         mod(it, max(stats_every,1)) == 0) then   ! stats_every=0 would divide unconditionally
        if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_a); end if
        call monitor_fmax(fmax)                ! populates ws%pu, ws%pv, ws%pw at GLC y
        call stats_accumulate(sacc, n, nxd, nzd, ws%pu, ws%pv, ws%pw)
        if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_b)
                          tm_stats = 1.0e3_dp*real(c_b-c_a,dp)/real(clk_rate,dp); end if
     end if
     ! --- divergence + realness diagnostic (probe) ---------------------------------
     ! Also reports max|Im(physical u,v,w)|: a C2C-carried real field should sit
     ! at FFT round-off; growth flags a Hermitian-symmetry leak (the unmasked-Nyquist
     ! failure mode). monitor_fmax populates ws%pu/pv/pw; its fmax goes to a private
     ! fdiv so the CFL/variable-dt fmax semantics are untouched.
     if (div_every > 0 .and. mod(it, max(div_every,1)) == 0) then
        call recover_uw(n+1, nkx, nz, ws%kx, ws%kz, v, eta, u0, w0, ws%uh, ws%wh, ws%dv)
        call divergence_check(n, nkx, nz, kx, kz, ws%uh, v, ws%wh, dvh_diag, maxdiv)  ! mag fused away
        write(*,'(a,i8,a,es10.2)') '   [div] step ', step, &
             '   max|div u| = ', maxdiv
     end if

     if (mod(it, max(log_every,1)) == 0 .or. it == n_steps) then    ! log_every=0 must not SIGFPE
        if (.not. variable_dt) call monitor_fmax(fmax)     ! fixed dt: get fmax for the log
        cfl = dt*fmax                                      ! (variable: fmax is this step's)
        istat = cudaDeviceSynchronize()
        u0h  = u0
        flux = bulk_velocity(n, real(u0h))
        if (fd_mode) then                             ! u0 near y=0: mid NODE value
           cl = real(u0h(n/2))
        else
           cl = 0.0_dp                                ! u0 at centreline: sum a_k T_k(0)
           do kk = 0, n, 2
              cl = cl + real(u0h(kk))*real((-1)**(kk/2), dp)
           end do
        end if
        ret = re_tau(n, Re, real(u0h))                ! Re_tau from the mean wall shear
        write(*,'(i8,1x,f11.4,1x,es11.3,1x,f9.4,1x,f11.6,1x,f9.5,1x,f9.3)') &
             step, t, dt, cfl, flux, cl, ret
        if (cfl > cfl_max) &
             write(*,'(a,f8.4,a,f8.4)') '   ** WARNING: CFL ', cfl, ' exceeds the sqrt(3)/pi ceiling ', cfl_max
     end if

     ! FD checkpoints are stamped -> no fd_mode suppression. max() keeps
     ! mod SIGFPE-safe at chk_every=0 (.and. does NOT short-circuit).
     if ((chk_every > 0 .and. mod(it, max(chk_every,1)) == 0) .or. it == n_steps) then
        if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_a); end if
        call io_write_state(chk_file, n, nx, nz, Re, t, dt, Qtar, step, v, eta, u0, w0, okio, &
                            Lx=Lx, Lz=Lz, flow='channel', ybc='noslip-noslip', drive='CFR')
        if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_b)
                          tm_io = 1.0e3_dp*real(c_b-c_a,dp)/real(clk_rate,dp); end if
        if (.not. okio) write(*,'(a)') '   ** WARNING: checkpoint write failed'
     end if

     if (timing) then
        tot_dtmon = tot_dtmon + tm_dtmon;  tot_rec = tot_rec + tm_rec
        tot_step  = tot_step  + tm_step;   tot_stats = tot_stats + tm_stats
        tot_io    = tot_io    + tm_io
        if (mod(it, time_every) == 0) &
           write(*,'(a,i8,a,f9.3,a,f9.3,a,f9.3,a,f9.3,a,f9.3)') &
              '   [time] step ', step, ':  step=', tm_step, '  dtmon=', tm_dtmon, &
              '  rec=', tm_rec, '  stats=', tm_stats, '  io=', tm_io
     end if
  end do
  call system_clock(c_run1)
  wall_s = real(c_run1-c_run0,dp)/real(clk_rate,dp)

  ! ===== final report =========================================================
  istat = cudaDeviceSynchronize()
  u0h = u0
  flux = bulk_velocity(n, real(u0h))
  ret  = re_tau(n, Re, real(u0h))
  write(*,'(a)')        '--------------------------------------------------------------'
  write(*,'(a,f12.8,a,f12.8)') ' final bulk velocity U_b = ', flux, '   (target ', Qtar, ')'

  if (turbulent) then
     write(*,'(a,f10.4,a,i0,a)') ' final Re_tau = ', ret, '   (stats from ', sacc%nsamp, ' samples)'
     ! step/t-stamped like the parallel writer: each restart
     ! segment leaves a DISTINCT stats file (pool with tools/combine_stats.py)
     ! instead of status='replace' clobbering the previous segment's.
     block
       character(len=256) :: sname
       sname = stats_stamp(stats_file, step, t)
       call write_stats(trim(sname))
       write(*,'(a,a)') ' mean profile + Reynolds stresses written to ', trim(sname)
     end block
  else
     errpar = 0.0_dp
     if (fd_mode) then                 ! FD state = point values
        do kk = 0, n
           aan = 1.5_dp*Qtar*(1.0_dp - y(kk)*y(kk))
           errpar = max(errpar, abs(real(u0h(kk)) - aan))
        end do
        write(*,'(a,es12.4)')  ' max|u0 - analytic Poiseuille (point values)| = ', errpar
     else
        do kk = 0, n
           aan = 0.0_dp
           if (kk == 0) aan =  3.0_dp*Qtar/4.0_dp
           if (kk == 2) aan = -3.0_dp*Qtar/4.0_dp
           errpar = max(errpar, abs(real(u0h(kk)) - aan))
        end do
        write(*,'(a,es12.4)')  ' max|u0 - analytic Poiseuille (Cheb coeffs)| = ', errpar
     end if
     if (errpar < 1.0e-4_dp) then
        write(*,'(a)')  ' [PASS] laminar Poiseuille recovered through the full driver'
     else
        write(*,'(a)')  ' [...] not fully relaxed yet -- increase n_steps (viscous time ~ Re)'
     end if
  end if
  if (variable_dt) then
     write(*,'(a,i0,a,es10.3)') ' variable dt: precompute retuned ', n_recompute, &
                                ' time(s); final dt = ', dt
     ! [guardrail] high retune RATE warning (independent of the tolerance setting):
     ! retunes are the expensive part of variable dt (3 x nkx*nz banded setups each).
     ! >5% of steps retuning means the precompute cost is significant -> widen
     ! dt_recompute_tol. (At tol=0.15 a healthy run retunes a handful of times;
     ! tol=0.05 produces a few hundred retunes in a few thousand steps.)
     if (n_steps > 0 .and. real(n_recompute,dp) > 0.05_dp*real(n_steps,dp)) &
        write(*,'(a,i0,a,i0,a,f5.1,a)') &
        '   [warn] retuned ', n_recompute, ' times in ', n_steps, ' steps (', &
        100.0_dp*real(n_recompute,dp)/real(n_steps,dp), &
        '% of steps) -- the vif precompute is a significant cost. Widen'// &
        ' dt_recompute_tol (>=0.15) to retune less often.'
     ! [guardrail] end-of-run clamp report. A large binding-clamp count means the
     ! [dt_min,dt_max] window is mis-set (see the per-step [warn] above).
     if (n_clamp_hi > 0) write(*,'(a,i0,a,es9.2,a)') &
        '   [warn] dt_max bound the step on ', n_clamp_hi, &
        ' step(s): CFL ran BELOW cfl_target. Raise dt_max (now ', dt_max, ').'
     if (n_clamp_lo > 0) write(*,'(a,i0,a,es9.2,a)') &
        '   [warn] dt_min bound the step on ', n_clamp_lo, &
        ' step(s): the flow wanted a smaller dt. Lower dt_min (now ', dt_min, &
        ') or investigate stability.'
     if (n_clamp_hi == 0 .and. n_clamp_lo == 0) write(*,'(a)') &
        '   (dt clamp never bound after startup -- the [dt_min,dt_max] window is well set)'
  end if
  write(*,'(a,f10.2,a,f10.2,a)') ' wall time = ', wall_s, ' s   (', &
       real(n_steps,dp)/max(wall_s,1.0e-9_dp), ' steps/s)'
  if (timing) then
     write(*,'(a)') ' --- timing breakdown (total ms, and % of wall) ---'
     call tline('step_full        ', tot_step,  wall_s)
     call tline('variable-dt fmax ', tot_dtmon, wall_s)
     call tline('dt recompute     ', tot_rec,   wall_s)
     call tline('stats sampling   ', tot_stats, wall_s)
     call tline('checkpoint io    ', tot_io,    wall_s)
     write(*,'(a,i0,a)') '   (dt recompute fired ', n_recompute, ' times)'
  end if
  write(*,'(a)')        '=============================================================='

  call stepper_ws_free(sws)
  call cheb_fft_scratch_free()
  call nl_destroy(ws)
  if (turbulent) then
     call stats_destroy(sacc)
     deallocate(mprof, mvp, mwp, uup, vvp, wwp, uvp)
  end if
  if (div_every > 0) deallocate(dvh_diag)
  deallocate(kxh, kzh, y, inv_dy)
  deallocate(kx, kz, inv_dy_d)
  if (fd_mode) then
     block
       integer(ip) :: ih2, im2
       call fd_helm_cache_stats(ih2, im2)
       write(*,'(a,i0,a,i0,a)') ' [fd] helm cache: ', ih2, ' hits, ', im2, ' misses'
     end block
  end if
  call band_ops_free(bops);  call mean_ops_free(mops);  call vsolve_ops_free(vops)
  deallocate(eta, v, u0, w0, Nv_prev, Neta_prev, u0h)

contains

  !-- FD dt-dependent precompute: the vif apparatus through the y_vif_setup
  !-- dispatch (per-stage alpha_k) + the FD mean unit responses into mops.
  !-- Called at init and on every variable-dt retune. -------------------------
  subroutine fd_precompute_dt()
    integer(ip) :: ks
    real(dp)    :: aal
    logical     :: okp
    ! Prefactor FIRST, so the vif setup solves below hit the fresh cache --
    ! otherwise every dt retune logs 3 spurious misses; the expression
    ! chain replicates ev_stage/fd_vif VERBATIM so the exact-fp lookup hits.
    block
      real(dp) :: lams(4), bet, aa2, cgb
      logical  :: cb
      integer(ip) :: kss
      lams(1) = 0.0_dp
      do kss = 1, 3
         bet = 0.5_dp*MCSUB(kss)
         aa2 = bet*dt/Re
         lams(1+kss) = 1.0_dp/aa2
      end do
      call fd_helm_cache_build(n, fd_kl, nkx, nz, 4_ip, lams, kx, kz, &
                               bops%d2b, fd_cache_gb, cb, cgb)
      if (cb) then
         write(*,'(a,f7.2,a)') '   [fd] helm factor cache ON  (', cgb, &
              ' GiB, 4 lambdas; bit-identical fast path)'
      else
         write(*,'(a,f7.2,a,f6.2,a)') '   [fd] helm factor cache OFF (needs ', cgb, &
              ' GiB > fd_cache_gb = ', fd_cache_gb, ')'
      end if
    end block
    do ks = 1, 3
       aal = 0.5_dp*MCSUB(ks)*dt/Re
       call y_vif_setup(bops, n, nkx, nz, kx, kz, aal, vops%vm(:,:,:,ks), &
                        vops%vp(:,:,:,ks), vops%Cinv(:,:,:,:,ks), &
                        fvt1, fvt2, fvt3, fvt4, fvt5, fvt6, fvt7)
    end do
    call fd_mean_flux_setup(n+1, fd_kl, d2f, wfile, dt, Re, UhF, IhF, okp)
    if (.not. okp) then
       write(*,'(a)') ' ** FD mean setup FAILED (zero pivot in the banded LU)'
       stop 3
    end if
    mops%Uhm(0:n,:) = UhF(1:n+1,:)
    mops%Ih = IhF
  end subroutine fd_precompute_dt

  !-- print one timing-breakdown line: label, total seconds, % of wall ----------
  subroutine tline(label, tot_ms, wall)
    character(*), intent(in) :: label
    real(dp),     intent(in) :: tot_ms, wall
    write(*,'(a,a,f10.2,a,f6.1,a)') '   ', label, tot_ms/1.0e3_dp, ' s  (', &
         100.0_dp*(tot_ms/1.0e3_dp)/max(wall,1.0e-9_dp), ' %)'
  end subroutine tline

  !-- CFL monitor: physical |u|,|v|,|w| -> max convective frequency. -------------
  ! Reuses the nl_workspace's plans + scratch (idle here, between steps). Note
  ! to_physical_da overwrites its spectral input, so v is copied to ws%dv first.
  subroutine monitor_fmax(fmax)
    real(dp), intent(out) :: fmax
    call recover_uw(n+1, nkx, nz, ws%kx, ws%kz, v, eta, u0, w0, ws%uh, ws%wh, ws%dv)
    call to_physical_da_r2c(ws%plans, n+1, nx, nz, nxd, nzd, ws%uh, ws%pu, ws%Wf)
    call to_physical_da_r2c(ws%plans, n+1, nx, nz, nxd, nzd, ws%wh, ws%pw, ws%Wf)
    call copy_c3(n+1, nkx, nz, v, ws%dv)   ! demote raw v-hat (dp) into the tp scratch
    call to_physical_da_r2c(ws%plans, n+1, nx, nz, nxd, nzd, ws%dv, ws%pv, ws%Wf)
    call cfl_freq_max(n+1, nxd, nzd, ws%pu, ws%pv, ws%pw, inv_dx, inv_dz, inv_dy_d, fmax)
  end subroutine monitor_fmax

  !-- finalize stats, normalise by wall units, write the profile table. ----------
  ! Wall units use the WINDOW-MEAN u_tau: the wall shear of the ACCUMULATED
  ! mean profile (averaging commutes with d/dy), NOT the final instantaneous
  ! Re_tau -- the instantaneous value sits +-2% off the window mean and leaks a
  ! spurious O(2-4%) shift into every + quantity via u_tau^2. The header also carries two convergence diagnostics so no
  ! post-processing is needed to judge a run: the total-stress linearity
  ! residual (momentum balance; shrinks ~1/sqrt(T_avg), ~1% at T~100) and the
  ! uu-peak wall-to-wall asymmetry (the run's internal sampling-error gauge).
  ! y+ = (1-|y|)*Re_tau ; U+ = <u>/u_tau ; stresses/u_tau^2.
  !-- step/t-stamped stats name, mirroring main_par's chk_filename
  !   (minus the precision tag): base 'stats.dat' -> 'stats_step00001000_t12.300.dat'.
  !   The zero-padded step is the sort key; segments never collide.
  function stats_stamp(base, stp, tnow) result(out)
    character(*), intent(in) :: base
    integer(ip),  intent(in) :: stp
    real(dp),     intent(in) :: tnow
    character(len=256) :: out
    character(len=8)   :: sstep
    character(len=24)  :: stime
    integer(ip) :: idot
    write(sstep,'(i8.8)') stp
    write(stime,'(f0.3)') tnow
    idot = index(trim(base), '.', back=.true.)
    if (idot > 0) then
       out = base(1:idot-1)//'_step'//sstep//'_t'//trim(stime)//trim(base(idot:))
    else
       out = trim(base)//'_step'//sstep//'_t'//trim(stime)
    end if
  end function stats_stamp

  subroutine write_stats(fname)
    character(*), intent(in) :: fname
    real(dp) :: utau, ut2, yp, s2, ret_w, acc, Ttot, resid, pk_lo, pk_hi, asym
    real(dp) :: ac(0:n), bc(0:n), dUdy(0:n)
    integer  :: uf, jj, mm
    call stats_finalize(sacc, n, mprof, mvp, mwp, uup, vvp, wwp, uvp)
    if (fd_mode) then
       ! FD: the profile IS point values -- wall shear via the yops
       ! FD branch directly, d<U>/dy via the banded D1 (no cheb_forward).
       ret_w = re_tau(n, Re, mprof)
       call fd_apply_band(n+1, fd_kl, d1f, mprof, dUdy)
    else
       call cheb_forward(n, mprof, ac)           ! accumulated <u> -> Cheb coeffs
       ret_w = re_tau(n, Re, ac)                 ! window-mean Re_tau
       ! (a) total-stress linearity: T = (1/Re) d<U>/dy - <u'v'> = -ut2*y
       call cheb_deriv(n, ac, bc)
       do jj = 0, n
          acc = 0.0_dp
          do mm = 0, n
             acc = acc + bc(mm)*cos(real(mm,dp)*PI*real(jj,dp)/real(n,dp))
          end do
          dUdy(jj) = acc
       end do
    end if
    utau  = ret_w/Re
    ut2   = utau*utau
    resid = 0.0_dp
    do jj = 0, n
       Ttot  = dUdy(jj)/Re - uvp(jj)
       resid = resid + (Ttot + ut2*y(jj))**2
    end do
    resid = sqrt(resid/real(n+1,dp))/max(ut2, tiny(1.0_dp))
    ! (b) uu-peak wall-to-wall asymmetry (internal sampling-error estimate)
    pk_lo = 0.0_dp;  pk_hi = 0.0_dp
    do jj = 0, n
       if (y(jj) < 0.0_dp) pk_lo = max(pk_lo, uup(jj))
       if (y(jj) > 0.0_dp) pk_hi = max(pk_hi, uup(jj))
    end do
    asym = 200.0_dp*abs(pk_lo - pk_hi)/max(pk_lo + pk_hi, tiny(1.0_dp))
    open(newunit=uf, file=trim(fname), status='replace', action='write')
    write(uf,'(a)') '# channel DNS turbulence statistics (x-z-t Reynolds average)'
    write(uf,'(a,f12.5,a,f10.4,a,es12.5,a,i0)') &
         '#  Re = ', Re, '   Re_tau = ', ret_w, '   u_tau = ', utau, '   nsamp = ', sacc%nsamp
    ! y-basis identity stamp (same tags as checkpoints). analyze_stats.py
    ! decides its d/dy branch from the y NODES and uses this line for
    ! provenance/notes.
    if (fd_mode) then
       write(uf,'(a,a,a,a)') '#  ybasis = ', trim(yop_backend_tag()), &
            '   ygrid_file = ', trim(ygrid_file)
    else
       write(uf,'(a,a)') '#  ybasis = ', trim(yop_backend_tag())
    end if
    write(uf,'(a)') '#  normalisation: WINDOW-MEAN u_tau (wall shear of the accumulated <u>)'
    write(uf,'(a,f8.3,a,f8.3,a,f10.4)') '#  convergence: stress-linearity residual = ', &
         100.0_dp*resid, ' %   uu-peak wall asymmetry = ', asym, &
         ' %   instantaneous-final Re_tau = ', ret
    write(uf,'(a)') '#  y            y+           U+           uu+          vv+          ww+          uv+          <u>'
    do jj = 0, n
       yp = (1.0_dp - abs(y(jj)))*ret_w
       if (utau > 0.0_dp) then
          s2 = ut2
          write(uf,'(8(es12.5,1x))') y(jj), yp, mprof(jj)/utau, uup(jj)/s2, &
               vvp(jj)/s2, wwp(jj)/s2, uvp(jj)/s2, mprof(jj)
       else
          write(uf,'(8(es12.5,1x))') y(jj), yp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, mprof(jj)
       end if
    end do
    close(uf)
  end subroutine write_stats

end program main
