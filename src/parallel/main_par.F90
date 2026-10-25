!===============================================================================
! main_par.F90 -- the distributed, NAMELIST-DRIVEN, multi-GPU
! production main. The single-GPU src/serial/main.F90, distributed: same &run namelist,
! same physics, same log/stats output -- but the spectral kx-axis is split across
! ranks (Layout B) and every cross-rank quantity (CFL fmax, divergence, turbulence
! statistics) is an MPI/NCCL reduction.
!
! This is the program that runs examples/input_kmm.nml (128^3, Re=2800) to a
! turbulent Re_tau ~ 178 (MKM 1999). The IC is generated IN-CODE (init_turbulent,
! fixed seed -> identical on every rank, then each rank slices its kx-slab), and
! statistics are written to a file on rank 0 -- so NO distributed checkpoint I/O
! is needed for the headline validation run.
!
! TWO MODES (CLI):
!   main_par [input.nml]              -- PRODUCTION: read the deck, spin up, evolve
!                                        n_steps, log + accumulate stats, write the
!                                        profile table. (default input file: input.nml)
!   main_par --validate [input.nml]   -- VALIDATION RUNG (the cheap regression guard):
!                                        build the SAME deck's IC, evolve a short
!                                        validate_steps (deck-tunable, default 5), and diff
!                                        the distributed slab path against the single-GPU oracle
!                                        (eta, v, mean, Reynolds stresses + Re_tau).
!                                        Exits 0 on PASS, error-stops 1 on FAIL. This
!                                        is test_driver_turb_mp, parameterised
!                                        by the namelist grid so it validates the EXACT
!                                        config you are about to run -- without the
!                                        full ~40-min campaign.
!
! WHAT IS DISTRIBUTED (the only communication, all on the same iterations on every
! rank so the collectives never deadlock):
!   * par_step_full          -- A<->B transpose + nonlinear all-to-alls (per step)
!   * par_to_physical_da_r2c -- transform for the CFL/stats/div monitor (per log/sample)
!   * par_reduce_max         -- global CFL fmax, global max|div u|
!   * par_reduce_sum         -- global watchdog field sum (NaN probe)
!   * par_stats_finalize     -- Allreduce of the 7 partial-sum profiles
! The (0,0) MEAN column (u0,w0 -> U_bulk, Re_tau, the CFR drive) lives ONLY on the
! kx=0 owner, which is ALWAYS rank 0 (block 0 contains global kx=0). So rank 0 holds
! the mean and does ALL logging -- no owner->rank-0 broadcast is needed.
!
! BOTH LONG WIRED:
!   * variable_dt   -- distributed CFL-driven retune (rank-uniform dt via
!                      par_reduce, precompute rebuilds, exact-mark landing).
!   * do_restart / restart_auto -- distributed checkpoint I/O (hardened,
!                      rolling last-K + atomic .latest pointer).
! Both are production-critical: every campaign deck runs variable dt with
! restart_auto chaining.
!
! Build/run: scripts/build/build_par.sh main + scripts/sbatch/run_main_par.sbatch
!   module load nvhpc openmpi nccl/2.28.7-gcc11.5.0-cuda
!   TP=dp bash scripts/build/build_par.sh main
!   TP=sp bash scripts/build/build_par.sh main
!   sbatch                      parallel/run_main_par.sbatch   # P=4, production deck
!   sbatch --export=VAL=1       parallel/run_main_par.sbatch   # P=4, --validate
!===============================================================================
program main_par
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,              only: tp, wp, dp, sp, ip
  use cudafor
  use decomp_mod,         only: decomp_t, decomp_setup
  use wavenum_mod,        only: build_wavenumbers, build_wavenumbers_r2c  ! canonical builders
  use diag_mod,           only: fft_size_guard                           ! advisory FFT-length check
  use cheb_shen,          only: glc_nodes, cheb_forward, cheb_deriv
  use helm_band_mod,      only: helm_band_setup
  use mean_mod,           only: mean_flux_setup, bulk_velocity, MCSUB, &
                                MGAMMA, MZETA               ! table assert refs
  use mean_gpu_mod,       only: MCSUB_G => MCSUB, MGAMMA_G => MGAMMA, &
                                MZETA_G => MZETA            ! duplicate copy
  use ygrid_mod,          only: ygrid_read                    ! 
  use fdops_mod,          only: fd_build_bands, fd_wall_weights, fd_apply_band
  use fd_mean_mod,        only: fd_mean_flux_setup
  use fd_helm_gpu_mod,    only: fd_helm_cache_build, fd_helm_cache_stats  ! 
  use yderiv_gpu_mod,     only: yderiv_set_fd
  use yops_mod,           only: yops_set_fd, yop_backend_tag, yop_ygrid_tag
  use spectral3d_mod,     only: s3d_set_ybasis_fd             ! 
  use par_spectral3d_mod, only: par_s3d_set_ybasis_fd         ! 
  use solver_ops_mod,     only: band_ops_fd_init, y_vif_setup ! 
  use stepper_lin_mod,    only: stepper_ev_setup, stepper_ev_setup_gpu, &
                                stepper_ev_setup_gpu_ws, steplin_csub_ok
  use init_mod,           only: init_turbulent, ic_fill_slab, ic_mean_u0, &
                                ic_fill_slab_fd, ic_mean_u0_fd   ! 
  use velocity_mod,       only: recover_uw
  use cfl_mod,            only: CFL_STAB, cfl_inv_dy, cfl_freq_max, cfl_dt
  use stats_mod,          only: stats_t, stats_init, stats_accumulate, stats_finalize, &
                                stats_destroy, re_tau, divergence_check, field_sumabs_c, &
                                stats_accum_p, hsym_check_c, hsym_fix_c
  use transforms_mod,     only: tf_arena_bytes         ! init log line
  use spectral3d_mod,     only: s3d_plans, s3d_setup_da_r2c, s3d_destroy_r2c, &
                                to_physical_da_r2c, to_spectral_da_r2c
  use par_spectral3d_mod, only: par_to_physical_da_r2c, par_to_physical_nf, &
                                par_to_physical_r2c, par_to_spectral_da_r2c, &
                                par_s3d_set_overlap
  use par_pressure_mod,   only: pres_gpu_setup_host, pres_assemble_solve, &
                                pres_promote_c, pres_half_usq, pres_combine_p, &
                                pres_set_fd                       ! 
  use nonlinear_mod,      only: nl_workspace, nl_setup, nl_destroy, nl_eval_H
  use stepper_mod,        only: step_full, stepper_ws, stepper_ws_alloc, stepper_ws_free, &
                                stepper_coeffs_ok
  use par_nonlinear_mod,  only: par_nl_workspace, par_nl_setup, par_nl_destroy, par_nl_eval_H
  use par_stepper_mod,    only: par_step_full, par_step_full_ws, parstep_coeffs_ok
  use solver_ops_mod,     only: band_ops_t, vsolve_ops_t, mean_ops_t, &
                                band_ops_alloc, band_ops_free, vsolve_ops_alloc, &
                                vsolve_ops_free, mean_ops_alloc, mean_ops_free, &
                                y_apply_L, y_vres_max, &  ! v-solve residual probe
                                y_vres_max_win, y_vres_galerkin  ! galerkin/interior canaries
  use par_stats_mod,      only: par_stats_finalize, par_stats_finalize_p
  use par_transpose_mod,  only: par_transpose_warmup, par_transpose_timing, &
                                par_transpose_set_timing
  use par_io_mod,         only: par_chk_write, par_chk_read, par_snap_write, &
                                par_io_set_hints, par_io_set_ybasis
  use fft3d_mod,          only: xl_release_current
  use par_mod
  implicit none
  include 'veta_build.inc'   ! VETA_BUILD_HASH/DATE (generated by the builders)

  real(dp),    parameter :: PI = 3.141592653589793238462643_dp

  ! --- config (&run namelist; same names/defaults as src/serial/main.F90) -------------
  integer(ip) :: n, nx, nz, n_steps, log_every, chk_every
  real(dp)    :: Lx, Lz, Re, Qtar, dt
  logical     :: do_restart, variable_dt
  logical     :: restart_auto             ! restart iff a checkpoint exists, else fresh IC
  real(dp)    :: cfl_target, dt_max, dt_min, dt_recompute_tol
  real(dp)    :: u_galilean              ! [Galilean] constant streamwise frame speed U_c (0 = lab)
  ! --- variable-dt time-based output (used when variable_dt=.true.) ------------
  real(dp)    :: t_end                   ! run until t >= t_end (variable dt; fixed dt uses n_steps)
  real(dp)    :: snap_dt                 ! snapshot interval in time units
  real(dp)    :: stats_dt                ! stats-sample interval in time units
  real(dp)    :: stats_start_t           ! stats start TIME (variable dt; fixed dt uses stats_start steps)
  real(dp)    :: chk_dt                  ! checkpoint interval in time units
  integer(ip) :: dt_check_every          ! re-evaluate the CFL dt (and the wall clock) every this many steps
  real(dp)    :: max_wall_hours          ! graceful stop: write checkpoint (+stats) and exit after this wall time (0 = off)
  real(dp)    :: wall_margin_min         ! minutes reserved BEFORE max_wall_hours for the final checkpoint
                                         ! write + up to dt_check_every steps of clock blindness + slack.
                                         ! Set max_wall_hours = the Slurm limit and let the margin do the budgeting.
  character(len=256) :: restart_file, chk_file, infile, stats_file
  logical            :: turbulent
  real(dp)           :: eps
  integer(ip)        :: pseed, kcut, stats_every, stats_start, div_every, time_every, nan_every
  integer(ip)        :: hsym_every           ! kx=0 z-Hermitian watchdog cadence (0 = off)
  logical            :: hsym_fix             ! re-symmetrize when checked (default off: keeps bit-exact gates)
  integer(ip)        :: validate_steps        ! --validate horizon (deck-tunable; default 5)
  ! --- physical u,v,w snapshots ------------------------------------------
  integer(ip)        :: snap_every            ! cadence in steps (0 = snapshots off)
  real(dp)           :: snap_start            ! begin snapshots once t >= snap_start (NS-time; skips the transient)
  logical            :: snap_fp32             ! output precision: .true. = fp32 (default), .false. = fp64
  character(len=256) :: snap_file             ! base name -> snap_<dp|sp>_step........_t....bin
  logical            :: snap_pressure         ! also solve + write the diagnostic pressure p (4th field)
  logical            :: chk_fp32              ! checkpoint payload fp32 (half volume; reader promotes)
  character(len=512) :: io_hints              ! MPI-IO write hints 'key=val,key=val'
  logical            :: snap_evict            ! free the clean-grid xl slot after each snapshot
  integer(ip)        :: a2a_overlap           ! 0 = monolithic A2A (default), 1 = per-field pipelined overlap
  ! FD-y backend: nonempty ygrid_file flips the y-basis to banded
  ! FD on the file's nodes (state = point values); fd_kl = band half-width.
  character(len=256) :: ygrid_file
  integer(ip)        :: fd_kl
  real(dp)           :: fd_cache_gb   ! helm factor-cache cap (0 = off)
  logical            :: fd_mode, have_w, fdok
  character(len=256) :: fdmsg
  real(dp), allocatable :: yfile(:), wfile(:), wlof(:), whif(:)
  real(dp), allocatable :: d1f(:,:), d2f(:,:), UhF(:,:), IhF(:)
  namelist /run/ n, nx, nz, Lx, Lz, Re, Qtar, dt, n_steps, log_every, chk_every, &
                 do_restart, restart_file, chk_file, &
                 variable_dt, cfl_target, dt_max, dt_min, dt_recompute_tol, &
                 turbulent, eps, pseed, kcut, stats_every, stats_start, stats_file, &
                 div_every, time_every, nan_every, validate_steps, &
                 snap_every, snap_start, snap_fp32, snap_file, snap_pressure, chk_fp32, &
                 u_galilean, t_end, snap_dt, stats_dt, stats_start_t, chk_dt, &
                 dt_check_every, max_wall_hours, wall_margin_min, hsym_every, hsym_fix, &
                 io_hints, snap_evict, a2a_overlap, ygrid_file, fd_kl, fd_cache_gb, &
                 restart_auto

  ! --- derived extents + decomposition ----------------------------------------
  integer(ip) :: ny, nkx, nxd, nzd, nkxd
  integer(ip) :: bnkx, kxlo, kxhi, anz
  type(decomp_t) :: d
  logical        :: validate
  integer(ip)    :: nargs, ia, ios, u, kk, ix, iz
  integer        :: istat
  character(len=256) :: argv
  real(dp)       :: inv_dx, inv_dz, cfl_max

  ! --- shared host setup (host-associated by the contained routines) ----------
  real(dp), allocatable :: kxh(:), kzh(:), y(:), inv_dy(:)
  type(band_ops_t) :: bops        ! y-operator band structure (host+device)
  type(mean_ops_t) :: mops        ! CFR mean precompute

  ! --- PRODUCTION device state + persistent monitor scratch (program scope so
  !     par_monitor sees them; allocated/freed in run_production) ---------------
  complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:)
  complex(dp), device, allocatable :: Nvp(:,:,:), Nep(:,:,:)
  real(dp),    device, allocatable :: kx(:), kz(:)
  type(vsolve_ops_t) :: vops      ! shifted v-solve lift + influence (slab)
  real(dp),    device, allocatable :: inv_dy_d(:)
  ! ALL gated monitor/stats/div/snap/pressure scratch is aliased onto
  ! pws (tp tier) and sws (dp tier) -- both are step-internal, dead between steps,
  ! and every gated block re-derives its inputs AFTER the last par_nl_eval_H of
  ! that block (eval_H clobbers ALL of S6/P6/PH/S3/Bz/Az/Af). The live slice map:
  !   monitor : S6 1:u^ 2:w^ 3:v^ 4:scratch;  P6 1:pu 2:pw 3:pv
  !   pressure: Hxd/Hyd/Hzd -> sws%Lv/Neta_cur/rhsv, Pi -> sws%scr (dp, promoted;
  !             Neta_cur substitutes for a separate L2v slab: dead between steps,
  !             and the next step's ks=1 WRITES it (WR pair, par_nl_ev_end)
  !             before anything reads it)
  !             qd -> PH(1), qhat -> S3(1), phat -> S3(2), pp_da -> P6(4),
  !             spr(clean p) -> PH(2)
  !   snapshot: u,v,w(clean) -> P6(4:6) (sequence-assoc view of the bigger
  !             dealiased slices), Bz_co/Az_co -> Bz(:,:,:,1:1)/Az(:,:,:,1:1)
  !   div     : dvh scratch -> sws%rhs (the |div| field is a fused
  !             in-kernel reduction, no mag array)
  ! --- diagnostic pressure (snapshots). Persistent setup +
  !     hoisted scratch (allocated only when snap_every>0 .and. snap_pressure).
  !     The path is dp (accuracy-critical solve, like vsolve_if_gpu) -> requires
  !     WP=dp (the only shipped build); ws%S3/P6 (tp) are promoted in.
  real(dp),    device, allocatable :: pp_d(:,:,:), pm_d(:,:,:), Cinv_d(:,:,:,:)  ! influence matrix (setup once)
  real(wp),    device, allocatable :: kxw(:), kzw(:)                   ! wp wavenumbers for apply_L
  real(wp),    device, allocatable :: kxd(:), kzd(:)            ! div probe wants wp
  type(par_nl_workspace) :: pws
  type(stats_t)          :: sacc
  type(stepper_ws)       :: sws            ! hoisted out of the time loop, not allocated per step

  ! ===== defaults (override via the input file -- mirrors src/serial/main.F90) ========
  n = 32;  nx = 16;  nz = 16
  Lx = 4.0_dp*PI;  Lz = 2.0_dp*PI;  Re = 10.0_dp;  Qtar = 1.0_dp
  dt = 5.0e-3_dp;  n_steps = 4000;  log_every = 200;  chk_every = 1000
  do_restart = .false.;  restart_file = 'restart.bin';  chk_file = 'chk.bin'
  restart_auto = .false.                  ! continue-if-checkpoint-exists mode
  variable_dt = .false.;  cfl_target = 0.32_dp
  dt_max = 2.0e-2_dp;  dt_min = 1.0e-4_dp;  dt_recompute_tol = 5.0e-2_dp
  u_galilean = 0.0_dp                     ! [Galilean] lab frame by default
  t_end = 0.0_dp;  snap_dt = 0.0_dp;  stats_dt = 0.0_dp;  stats_start_t = 0.0_dp
  chk_dt = 0.0_dp;  dt_check_every = 20   ! variable-dt time-based output (0 = that stream off)
  max_wall_hours = 0.0_dp                 ! graceful wall-time stop disabled by default
  wall_margin_min = 15.0_dp               ! 15 min covers a multi-TB checkpoint + slack
  turbulent = .false.;  eps = 0.05_dp;  pseed = 12345;  kcut = 8
  stats_every = 50;  stats_start = 0;  stats_file = 'stats.dat';  div_every = 0
  time_every = 0;  nan_every = 10;  validate_steps = 5
  hsym_every = 0;  hsym_fix = .false.
  snap_every = 0;  snap_start = 0.0_dp;  snap_fp32 = .true.;  snap_file = 'snap.bin'
  chk_fp32 = .false.                      ! default: full-precision checkpoints
  io_hints = ''                           ! default: MPI_INFO_NULL
  snap_evict = .false.                    ! default: keep both xl slots (20+ nodes)
  snap_pressure = .true.   ! when snapshots are on, also solve+write p (set .false. to skip)
  a2a_overlap = 0                         ! default: monolithic A2A
  ygrid_file = '';  fd_kl = 4             ! default: Chebyshev backend
  fd_cache_gb = 4.0_dp                    ! cache cap; big-P runs may lower/0

  ! ===== CLI: [--validate] [input.nml] =========================================
  validate = .false.;  infile = 'input.nml'
  nargs = command_argument_count()
  do ia = 1, nargs
     call get_command_argument(ia, argv)
     if (trim(argv) == '--validate') then
        validate = .true.
     else
        infile = argv
     end if
  end do

  ! ===== MPI/NCCL up, then every rank reads the SAME deck (shared FS) ==========
  call par_init()
  open(newunit=u, file=trim(infile), status='old', action='read', iostat=ios)
  if (ios == 0) then
     read(u, nml=run, iostat=ios)
     close(u)
     ! a deck PRESENT but UNPARSEABLE is fatal: namelist assignment is
     ! incremental, so continuing would leave a compiler-dependent mixture of
     ! deck values and the built-in 32x16x16 laminar defaults -- a typo'd
     ! campaign deck would silently run on a toy box. (Absent deck still runs
     ! the defaults: that is the documented test/demo path.)
     if (ios /= 0) then
        if (par_rank == 0) write(*,'(a,a,a,i0,a)') &
           '** FATAL: ', trim(infile), ' is present but unparseable (iostat = ', ios, &
           ') -- fix the deck; refusing to run defaults'
        call par_finalize();  error stop 1
     end if
  else if (par_rank == 0) then
     write(*,'(a,a,a)') '** input file "', trim(infile), '" not found; using built-in defaults'
  end if
  ! the SMR tables (CSUB/GAMMA/ZETA) live as verbatim copies in FIVE
  ! modules; the mean solves and the [res] Galerkin canary assume every copy
  ! is identical, so assert it here. Pure parameter comparison ->
  ! rank-uniform verdict; rank-0 prints, all ranks stop (pattern of the
  ! namelist gate above).
  if (.not. (all(MCSUB == MCSUB_G) .and. all(MGAMMA == MGAMMA_G) .and. &
             all(MZETA == MZETA_G) .and. &
             stepper_coeffs_ok(MCSUB, MGAMMA, MZETA) .and. &
             parstep_coeffs_ok(MCSUB, MGAMMA, MZETA) .and. &
             steplin_csub_ok(MCSUB))) then
     if (par_rank == 0) write(*,'(a)') &
        '** FATAL: SMR coefficient tables diverged across modules (12.4)'
     call par_finalize();  error stop 1
  end if
  call par_io_set_hints(trim(io_hints))   ! '' -> INFO_NULL (no-op)
  call par_s3d_set_overlap(a2a_overlap)   ! 0 = monolithic (default); 1 = pipelined overlap
  if (par_rank == 0 .and. a2a_overlap > 0) &
     print '(a,i0,a)', '[perf] A2A overlap ON (mode ', a2a_overlap, &
           '): per-field pipelined transpose on the comm stream (ulp-level reassociation vs mode 0)'

  ! ===== derived extents + decomposition ======================================
  ny   = n + 1
  nkx  = nx/2 + 1;  nxd = 3*nx/2;  nzd = 3*nz/2;  nkxd = nxd/2 + 1
  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
  bnkx = d%b_nkx_loc;  kxlo = d%b_kx0 + 1;  kxhi = d%b_kx0 + bnkx;  anz = d%ad_nz_loc
  ! the stepper/transform need a non-empty kx- and dealiased-z slab on every rank
  if (nkx < par_nproc .or. nzd < par_nproc) then
     if (par_rank == 0) write(*,'(a,i0,a,i0,a,i0)') &
        '** FATAL: need nkx (', nkx, ') >= P and nzd (', nzd, ') >= P = ', par_nproc
     call par_finalize();  error stop 1
  end if

  ! ===== shared host setup (wavenumbers, GLC nodes, operators, CFR precompute) =
  allocate(kxh(nkx), kzh(nz), y(0:n), inv_dy(0:n))
  call band_ops_alloc(bops, n);  call mean_ops_alloc(mops, n)
  ! THE CANONICAL BUILDERS, not inline loops: an inline pair would produce
  ! bit-identical values but silently skip every guard
  ! the serial driver gets -- no even-nm error stop, no L<=0 error stop. An
  ! odd-nx/nz or nonpositive-Lx/Lz deck dies at startup on every rank
  ! (namelist-uniform inputs -> uniform stop, no collective desync) instead of
  ! running the campaign wrong.
  call build_wavenumbers_r2c(nx, Lx, kxh)                       ! R2C kx half-axis
  call build_wavenumbers    (nz, Lz, kzh)                       ! C2C kz, cuFFT order
  call glc_nodes(n, y)
  ! FD mode: read + validate the grid file, refuse the paths that
  ! are not FD-ready yet, swap y so CFL/banner/stats follow the file nodes.
  fd_mode = (len_trim(ygrid_file) > 0)
  ! Advisory FFT-length check: without it a non-7-smooth length silently
  ! takes cuFFT's Bluestein path, a real cost on an 80-360 GPU campaign.
  ! Rank-0 only (advisory print, no stop);
  ! the 5th executed length -- the 2n Chebyshev even extension -- is checked
  ! too, except under FD-y where that leg never runs.
  if (par_rank == 0) call fft_size_guard(nx, nz, nxd, nzd, merge(0_ip, 2_ip*n, fd_mode))
  if (fd_mode) then
     allocate(yfile(n+1), wfile(n+1), wlof(n+1), whif(n+1))
     allocate(d1f(2*fd_kl+1, n+1), d2f(2*fd_kl+1, n+1), UhF(n+1,3), IhF(3))
     wfile = 0.0_dp
     call ygrid_read(trim(ygrid_file), n+1, yfile, wfile, have_w, fdok, fdmsg)
     if (.not. fdok) then
        if (par_rank == 0) write(*,'(a)') ' ** ygrid REFUSED: '//trim(fdmsg)
        call par_finalize();  error stop 1
     end if
     if (.not. have_w) then          ! [TRAP ii] CFR closure needs the weights
        if (par_rank == 0) write(*,'(a)') ' ** FD mode REFUSED: grid file has '// &
           'no w column (the CFR flux closure needs quadrature weights)'
        call par_finalize();  error stop 1
     end if
     ! checkpoints/restart/snapshots are FD-capable: every header now
     ! stamps ybasis=fd-point / ygrid=file, restart verifies the checkpoint's v3
     ! y record against THIS run's nodes, and a Chebyshev file (or a different
     ! FD grid) refuses cleanly in par_chk_read instead of silently corrupting.
     call par_io_set_ybasis('fd-point', 'file', yfile)
     y = yfile                       ! (0:n) <- file nodes
     call fd_build_bands(n+1, fd_kl, yfile, d1f, d2f)
     call fd_wall_weights(n+1, fd_kl, yfile, wlof, whif)
     ! pressure backend -> FD (same bands + one-sided wall stencils as
     ! the v-solve); snap_pressure works in FD mode too.
     call pres_set_fd(n+1, fd_kl, yfile, d2f, wlof, whif)
  end if
  call cfl_inv_dy(n, y, inv_dy)
  inv_dx  = real(nx, dp)/Lx;  inv_dz = real(nz, dp)/Lz;  cfl_max = CFL_STAB/PI
  call helm_band_setup(n, bops%sdh, bops%sth, bops%bdh, bops%boff)
  call mean_flux_setup(n, dt, Re, bops%sdh, bops%sth, bops%bdh, bops%boff, mops%Uhm, mops%Ih)
  if (fd_mode) then                  ! flip every y seam to FD
     call band_ops_fd_init(bops, n, fd_kl, yfile, wfile, have_w)
     call yderiv_set_fd(n+1, fd_kl, d1f)
     call yops_set_fd(n, wlof, whif, wfile)
     call s3d_set_ybasis_fd(.true.)         ! serial-transform paths (monitor)
     call par_s3d_set_ybasis_fd(.true.)     ! distributed transforms
     call fd_mean_refill()                  ! [TRAP i]
  end if

  if (validate .and. fd_mode) then
     if (par_rank == 0) write(*,'(a)') ' ** --validate is not FD-wired yet '// &
        '(the single-GPU oracle path stays Chebyshev); use the serial FD gates'
     call par_finalize();  error stop 1
  end if
  if (validate) then
     call run_validate()
  else
     call run_production()
  end if

  deallocate(kxh, kzh, y, inv_dy)
  call band_ops_free(bops);  call mean_ops_free(mops)
  call par_finalize()

contains

  !=============================================================================
  ! PRODUCTION -- distributed spin-up + time loop on this rank's kx-slab.
  !=============================================================================
  subroutine run_production()
    complex(dp), allocatable :: he0(:,:,:), hv0(:,:,:)
    complex(dp) :: hu00(0:n), hw00(0:n), u0h(0:n)
    real(dp), allocatable :: mprof(:), mvp(:), mwp(:), uup(:), vvp(:), wwp(:), uvp(:)
    real(dp), allocatable :: mpp(:), ppv(:)            ! [stats] <p>(y), <p'^2>(y)
    real(dp) :: t, fmax, cfl, flux, cl, ret, maxdiv, swd1, swd2, errpar, aan
    real(dp) :: beta_h, beta_sum, dpdx_avg            ! drive force = mean dp/dx (CFR beta)
    real(dp) :: beta_host(1)                          ! host D2H buffer for sws%mg%beta_d (device)
    logical  :: pres_on                               ! pressure machinery needed (snapshot or stats)
    integer(ip) :: it, step, jj
    logical     :: need_phys
    ! --- timing (diagnostic; time_every=0 => no instrumentation/sync overhead) ---
    logical     :: timing
    integer(8)  :: clk_rate, c_a, c_b, c_run0, c_run1, c_init0
    real(dp)    :: tm_step, tm_mon, tm_stats, tm_nan, tm_div
    real(dp)    :: tm_pack, tm_nccl, tm_unpack                 ! transpose phase split (comm probe)
    real(dp)    :: tm_io                                       ! checkpoint + snapshot writes
    real(dp)    :: tot_step, tot_mon, tot_stats, tot_nan, tot_div, wall_s
    real(dp)    :: tot_pack, tot_nccl, tot_unpack, tot_io
    integer(ip) :: n_tr
    ! --- checkpoint I/O: rolling last-2 history + restart temporaries ------
    character(len=256) :: ckname, ck1, ck2, ck_evict
    character(len=256) :: rsname            ! restart name after .latest resolution
    integer     :: rs_unit                  ! pointer-file unit
    integer     :: ios_ptr                  ! pointer-file iostat/exitstat
    real(dp)    :: re_ck, dt_ck, q_ck
    real(dp)    :: lx_ck, lz_ck             ! checkpoint box identity
    logical     :: okck
    logical     :: use_chk                  ! restart_auto found '<chk_file>.latest'
    ! --- physical u,v,w snapshots ------------------------------------------
    integer(ip) :: anz_co, snap_bytes
    character(len=256) :: snapname
    logical     :: oksnap
    ! --- variable dt + time-based output scheduling (exact-time landing) --------
    integer(ip) :: n_steps_cap, mland, step0, isnap, istats, ichk
    real(dp)    :: fmax_cur, dt_cfl, rem, dt_try, t_mark, ratio
    real(dp)    :: snap_next, stats_next, chk_next, elapsed_s
    real(dp), parameter :: TLAND_EPS = 1.0e-9_dp     ! "landed on a mark" tolerance (t.u.)
    logical     :: snap_on, stats_on_t, chk_on_t
    logical     :: fire_stats, fire_snap, fire_chk, fire_log, fire_div, last_step, stop_wall

    ! --- variable-dt config gate: time-based output is required (step-based
    !     *_every cadences are meaningless once dt varies) -----------------------
    if (variable_dt) then
       if (t_end <= 0.0_dp) then
          if (par_rank == 0) write(*,'(a)') &
             '** FATAL: variable_dt requires t_end > 0 (run end time, t.u.).'
          call par_finalize();  error stop 4
       end if
       if (dt_check_every < 1) dt_check_every = 1
       if (par_rank == 0 .and. snap_dt <= 0.0_dp .and. snap_every > 0) write(*,'(a)') &
          '   [vardt] note: snap_every (steps) is ignored under variable_dt; set snap_dt (t.u.).'
    end if

    ! --- banner FIRST, then staged init progress (rank 0; UX timings only) ----
    ! Print the run's identity up front, BEFORE the long precompute below, so a
    ! job never looks hung for seconds before any output. The
    ! [init] lines (Neko-inspired) say what each stage is doing + how long it took.
    call system_clock(count_rate=clk_rate)
    call system_clock(c_init0)
    call print_banner('PRODUCTION', n_steps, .false.)
    if (par_rank == 0) write(*,'(a)') &
         '   initializing (rank-0 timings -- for information, not benchmarking) ...'

    ! --- device state + stepper precompute (this rank's kx-slab) --------------
    call init_stage('per-(kx,kz) v-solve operators: batched on-device band solve + influence matrix')
    call system_clock(c_a)
    allocate(eta(0:n,bnkx,nz), v(0:n,bnkx,nz), u0(0:n), w0(0:n))
    allocate(Nvp(0:n,bnkx,nz), Nep(0:n,bnkx,nz))
    allocate(kx(bnkx), kz(nz))
    call vsolve_ops_alloc(vops, n, bnkx, nz)
    kx = kxh(kxlo:kxhi);  kz = kzh
    bops%sd = bops%sdh;  bops%st = bops%sth;  bops%bd = bops%bdh      ! H2D band structure
    ! the stepper workspace is allocated BEFORE the operator build so the
    ! build can borrow its per-step slab scratch (idle here) -- a self-allocating
    ! build would be a ~9 GB/GPU transient, fatal at a 16-node mid-run rebuild.
    ! Init and every variable-dt rebuild run the SAME zero-alloc path.
    call stepper_ws_alloc(sws, n, bnkx, nz, fd=fd_mode)  ! [perf] one alloc for the whole run (slab
                                               ! extents; FD-y skips the 3 Shen buffers, -48 B/pt)
    call setup_ev_ops_ws()
    call init_sync_check('v-solve operators');  call system_clock(c_b);  call init_done(c_a, c_b, clk_rate)

    ! --- monitor scratch = pws slices; ONE cuFFT plan set = pws%plans ---
    call init_stage('workspace + cuFFT plans (forward/inverse R2C transforms)')
    call system_clock(c_a)
    allocate(inv_dy_d(ny));  inv_dy_d = inv_dy
    call par_nl_setup(pws, d, n+1, nx, nz, kxh, kzh)
    pws%ugal = u_galilean                        ! [Galilean] advect in the frame moving at U_c
    if (par_rank == 0) write(*,'(a,f8.2,a)') '   [init] cuFFT work-area arena: ', &
       real(tf_arena_bytes(),dp)/2.0_dp**30, ' GiB (ONE shared buffer for all plans)'
    if (div_every > 0) then
       allocate(kxd(bnkx), kzd(nz))                          ! dvh -> sws%rhs; mag fused away
       kxd = kxh(kxlo:kxhi);  kzd = kzh
    end if
    if (turbulent) then
       call stats_init(sacc, n, nxd, nzd)                       ! GLOBAL geometry
       allocate(mprof(0:n), mvp(0:n), mwp(0:n), uup(0:n), vvp(0:n), wwp(0:n), uvp(0:n))
       allocate(mpp(0:n), ppv(0:n));  mpp = 0.0_dp;  ppv = 0.0_dp  ! [stats] <p>, <p'^2>
    end if
    anz_co = d%a_nz_loc                                         ! [snap] coarse (clean) z-slab
    ! [output] are snapshots / stats sampling ACTIVE? Under variable_dt the cadence
    ! is time-based (snap_dt/stats_dt); under fixed dt it is step-based (snap_every/
    ! stats_every). These flags drive BOTH the buffer allocation here and the fire
    ! decisions in the loop, so the two can never disagree (a mismatch would crash).
    snap_on    = merge(snap_dt  > 0.0_dp, snap_every > 0,           variable_dt)
    stats_on_t = turbulent .and. merge(stats_dt > 0.0_dp, stats_every > 0, variable_dt)
    chk_on_t   = (variable_dt .and. chk_dt > 0.0_dp)
    ! snapshot buffers need NO allocation: clean-grid u,v,w live in
    ! pws%P6(:,:,:,4:6) and the clean-inverse scratch in pws%Bz/Az slices.
    ! Pressure machinery is needed if EITHER snapshots write p OR the turbulent
    ! stats sample p (the drive force / dp/dx is reported regardless -- it is the
    ! CFR beta, always computed). One per-mode influence-matrix setup serves both.
    pres_on = snap_pressure .and. (snap_on .or. stats_on_t)
    if (pres_on) then                                          ! diagnostic pressure
       allocate(pp_d(0:n,bnkx,nz), pm_d(0:n,bnkx,nz), Cinv_d(2,2,bnkx,nz))
       allocate(kxw(bnkx), kzw(nz))
       ! the per-call pressure scratch (Hxd/Hyd/Hzd/Pi dp; qd/qhat/phat/
       ! pp_da/spr tp) is aliased onto sws/pws -- see the slice map at the top.
       kxw = real(kxh(kxlo:kxhi),wp);  kzw = real(kzh,wp)
       call pres_gpu_setup_host(n, bnkx, nz, kxh(kxlo:kxhi), kzh, pp_d, pm_d, Cinv_d)
    end if
    beta_sum = 0.0_dp;  beta_h = 0.0_dp;  dpdx_avg = 0.0_dp
    call init_sync_check('workspace + cuFFT plans');  call system_clock(c_b);  call init_done(c_a, c_b, clk_rate)

    ! --- initial state: restart from a distributed checkpoint, OR build the IC --
    ! restart_auto: the resubmit-safe campaign mode. do_restart=.true.
    ! stays STRICT (missing checkpoint = FATAL, protecting an intended restart
    ! from a typo'd name); restart_auto continues from a checkpoint when one is
    ! resolvable and otherwise starts fresh -- so the SAME deck serves the first
    ! submission and every resubmission of a wall-limited campaign job.
    if (restart_auto .and. .not. do_restart) then
       ! the exist verdict must be RANK-UNIFORM: per-rank inquire on
       ! a just-written NFS file can diverge (attribute caching), splitting
       ! ranks between the restart path (MPI collectives) and the fresh-IC path
       ! (NCCL warm-up) -- a hang. Rank 0 decides; LAND-reduce broadcasts it
       ! (non-zero ranks contribute .true., so the result IS rank 0's verdict).
       ! the writer maintains '<chk_file>.latest', but
       ! discovery only probed restart_file names -- any deck where the two keys
       ! differ (e.g. restart_file left at its 'restart.bin' default) NEVER found
       ! its own checkpoints and silently replayed the fresh IC. Probe order:
       ! restart_file exact (an explicit seed wins), its .latest, then the chain
       ! THIS run writes; on the last hit retarget restart_file -> chk_file so
       ! the resolver below follows the right pointer. chk_file is rank-uniform
       ! (every rank reads the same deck), so a uniform verdict suffices.
       if (par_rank == 0) then
          use_chk = .false.
          inquire(file=trim(restart_file), exist=okck)
          if (.not. okck) inquire(file=trim(restart_file)//'.latest', exist=okck)
          if (.not. okck .and. trim(chk_file) /= trim(restart_file)) then
             inquire(file=trim(chk_file)//'.latest', exist=use_chk)
             okck = use_chk
          end if
       else
          okck = .true.;  use_chk = .true.
       end if
       call par_reduce_and(okck)
       call par_reduce_and(use_chk)              ! which base name: rank 0's verdict
       if (use_chk) restart_file = chk_file      ! rank-uniform retarget
       do_restart = okck
       if (par_rank == 0) then
          if (okck) then
             write(*,'(a)') '   [restart_auto] checkpoint found -> continuing the previous run'
          else
             write(*,'(a)') '   [restart_auto] no checkpoint found -> fresh initial condition'
          end if
       end if
    end if
    if (do_restart) then
       ! Resolve the restart name. Checkpoints are written with step/time-
       ! stamped names (chk_filename), so the deck's base name usually does not
       ! exist as a file. The writer maintains '<chk base>.latest' (one line: the
       ! newest checkpoint's actual name); if the deck's restart_file is missing,
       ! follow that pointer. Every rank reads the tiny text file itself (shared
       ! FS, static content -> rank-identical resolution, no bcast needed).
       rsname = trim(restart_file)
       inquire(file=trim(rsname), exist=okck)
       if (.not. okck) then
          inquire(file=trim(restart_file)//'.latest', exist=okck)
          if (okck) then
             ! iostat on both: a torn/empty pointer (job killed mid-update)
             ! must NOT hard-crash the
             ! runtime here -- fall through with the unresolved base name and
             ! let par_chk_read's rank-uniform open verdict refuse cleanly.
             open(newunit=rs_unit, file=trim(restart_file)//'.latest', &
                  status='old', action='read', iostat=ios_ptr)
             if (ios_ptr == 0) then
                read(rs_unit, '(a)', iostat=ios_ptr) rsname
                close(rs_unit)          ! unconditional: a failed READ must still release the unit
             end if
             if (ios_ptr /= 0) then
                rsname = trim(restart_file)
                if (par_rank == 0) write(*,'(a)') &
                   ' !! [restart] .latest pointer unreadable (torn write?) -- cannot resolve checkpoint name'
             else if (par_rank == 0) then
                write(*,'(a,a)') '   [restart] resolved via .latest pointer -> ', trim(rsname)
             end if
          end if
       end if
       call init_stage('restart: read distributed checkpoint ('//trim(rsname)//')')
       call system_clock(c_a)
       call par_chk_read(trim(rsname), d, n, nx, nz, re_ck, t, dt_ck, q_ck, step, &
                         v, eta, u0, w0, okck, Lxf=lx_ck, Lzf=lz_ck)
       if (.not. okck) then
          if (par_rank == 0) write(*,'(a,a)') &
             '** FATAL: restart read failed (missing, incompatible, or truncated): ', trim(rsname)
          call par_finalize();  error stop 3
       end if
       ! box identity gate: a checkpoint from a different Lx/Lz would be
       ! silently reinterpreted onto this deck's wavenumbers -- REFUSE. (Gated
       ! only when the header carries the keys; every writer since v2 does.)
       if (lx_ck > 0.0_dp .and. (abs(lx_ck-Lx) > 1.0e-9_dp*Lx .or. abs(lz_ck-Lz) > 1.0e-9_dp*Lz)) then
          if (par_rank == 0) write(*,'(a,2es13.5,a,2es13.5)') &
             '** FATAL: checkpoint box (Lx,Lz) = ', lx_ck, lz_ck, ' /= deck ', Lx, Lz
          call par_finalize();  error stop 3
       end if
       if (par_rank == 0) then
          if (abs(re_ck-Re) > 1.0e-9_dp*max(Re,1.0_dp) .or. abs(dt_ck-dt) > 1.0e-12_dp .or. &
              abs(q_ck-Qtar) > 1.0e-12_dp) &
             write(*,'(a,3es10.3,a)') '   [restart] note: checkpoint Re/dt/Qtar (', &
                re_ck, dt_ck, q_ck, ') differ from the deck; continuing with the deck values.'
          write(*,'(a,i0,a,f0.4)') '   [restart] resumed at step ', step, '   t = ', t
       end if
    else
       if (turbulent) then
          call init_stage('initial condition: turbulent spectrum (in-code, fixed seed)')
       else
          call init_stage('initial condition: laminar rest (CFR builds the parabola)')
       end if
       call system_clock(c_a)
       if (turbulent) then
          call build_ic_slab(hu00, hw00)               ! THIS rank's kx-slab only
       else
          eta = (0.0_dp,0.0_dp);  v = (0.0_dp,0.0_dp)           ! laminar: CFR builds the parabola
          hu00 = (0.0_dp,0.0_dp);  hw00 = (0.0_dp,0.0_dp)
       end if
       if (d%owns_mode00) then        ! mean is rank-0-local
          u0 = hu00;  w0 = hw00
       else
          u0 = (0.0_dp,0.0_dp);  w0 = (0.0_dp,0.0_dp)
       end if
       t = 0.0_dp;  step = 0
    end if
    ! SMR is self-starting (zeta_1=0), so the N^{k-2} history is re-derived each step
    ! and need NOT be restored -- a restart with fresh registers reproduces an
    ! uninterrupted run bit-for-bit (io_mod's restart-continuation contract).
    Nvp = (0.0_dp,0.0_dp);  Nep = (0.0_dp,0.0_dp)
    call init_sync_check('initial condition / restart');  call system_clock(c_b);  call init_done(c_a, c_b, clk_rate)

    ! --- time-loop setup ------------------------------------------------------
    timing = (time_every > 0)
    call par_transpose_set_timing(timing)   ! phase barriers+clocks only when timing
    tot_step = 0.0_dp; tot_mon = 0.0_dp; tot_stats = 0.0_dp; tot_nan = 0.0_dp; tot_div = 0.0_dp
    tot_pack = 0.0_dp; tot_nccl = 0.0_dp; tot_unpack = 0.0_dp; tot_io = 0.0_dp
    ck1 = '';  ck2 = ''                                         ! rolling last-2 checkpoint names
    ! [perf] open NCCL channels + pre-size the persistent transpose buffers OUTSIDE
    ! the timed loop: avoids the one-off lazy-connection multi-ms first all-to-all
    ! and any mid-loop grow-realloc. nfield=6 = the largest (inverse) transpose.
    call init_stage('NCCL transpose warm-up (open channels, pre-size buffers)')
    call system_clock(c_a)
    call par_transpose_warmup(d, ny, nkx, nzd, anz, bnkx, 6_ip)
    call init_sync_check('NCCL transpose warm-up');  call system_clock(c_b);  call init_done(c_a, c_b, clk_rate)
    if (par_rank == 0) write(*,'(a,f8.2,a)') '   [init] all setup complete in ', &
         real(c_b-c_init0,dp)/real(clk_rate,dp), ' s -- starting time loop'

    if (timing .and. par_rank == 0) write(*,'(a)') &
         '   [timing on] per-step ms (rank 0): step_full=RK3 step (= compute + comm)'// &
         '   mon=recover+transform+CFL  stats=accumulate  nan=watchdog  div=divergence'// &
         '  io=checkpoint/snapshot'
    if (par_rank == 0) then
       write(*,'(a)') '   step         t           dt        CFL       U_bulk      u0(0)     Re_tau   drive(dp/dx)'
       write(*,'(a)') '   ----------------------------------------------------------------------------------------'
    end if
    ! --- output schedules (snap_on/stats_on_t/chk_on_t were set before allocation) --
    ! INDEX-based marks: mark = start + k*interval (computed by MULTIPLY, never by a
    ! running += sum) so the mark positions cannot drift over a multi-day run. t may
    ! be > 0 after a restart -> seed each index to the first unfired mark.
    if (variable_dt) then
       isnap  = next_mark_index(t, snap_start,    snap_dt)
       istats = next_mark_index(t, stats_start_t, stats_dt)
       ichk   = next_mark_index(t, 0.0_dp,        chk_dt)         ! checkpoints at k*chk_dt
       snap_next  = merge(snap_start    + real(isnap, dp)*snap_dt,  huge(1.0_dp), snap_on)
       stats_next = merge(stats_start_t + real(istats,dp)*stats_dt, huge(1.0_dp), stats_on_t)
       chk_next   = merge(real(ichk,dp)*chk_dt,                     huge(1.0_dp), chk_on_t)
       call par_monitor(fmax);  fmax_cur = fmax                    ! initial CFL (fresh IC OR restart state)
       n_steps_cap = max(n_steps, int(min(2.0_dp*t_end/max(dt_min,1.0e-12_dp) + 1000.0_dp, &
                                          real(huge(1_ip)-1, dp))))   ! backstop; t_end is the real terminator
       ! completed-run guard: a restart with t >= t_end (a naive resubmit
       ! of a finished campaign job -- .latest points at the final checkpoint)
       ! would otherwise enter the loop with rem = 0 -> dt = 0 -> a forced
       ! operator rebuild at lambda = 1/a = Inf, then CHECKPOINT the poisoned
       ! state over the good one. Zero trips of the loop instead: no step, no output,
       ! no checkpoint, pointers untouched.
       if (t >= t_end - TLAND_EPS) then
          if (par_rank == 0) write(*,'(a,f0.4,a,f0.4,a)') &
             '   [restart] run already COMPLETE (t = ', t, ' >= t_end = ', t_end, &
             ') -- nothing to do; state and checkpoints untouched'
          n_steps_cap = 0
       end if
    else
       n_steps_cap = n_steps
    end if

    step0 = step                              ! steps run THIS invocation (for the steps/s report)
    call system_clock(c_run0)
    do it = 1, n_steps_cap
       tm_step = 0.0_dp; tm_mon = 0.0_dp; tm_stats = 0.0_dp; tm_nan = 0.0_dp; tm_div = 0.0_dp
       tm_pack = 0.0_dp; tm_nccl = 0.0_dp; tm_unpack = 0.0_dp; tm_io = 0.0_dp

       ! --- variable dt: CFL-driven step, subdivided to land EXACTLY on the next
       !     output/end mark (dt = rem/ceil(rem/dt_cfl) <= dt_cfl, evenly spaced so
       !     no degenerate tiny step). Rebuild the dt-dependent operators only when
       !     dt actually drifts past the tolerance (now ~cheap on the GPU). The
       !     global fmax (par_reduce_max in par_monitor) makes every rank pick the
       !     SAME dt -> the collective rebuild can never deadlock. -----------------
       if (variable_dt) then
          dt_cfl = cfl_dt(fmax_cur, cfl_target)
          dt_cfl = min(max(dt_cfl, dt_min), dt_max)
          t_mark = t_end
          if (snap_on)    t_mark = min(t_mark, snap_next)
          if (stats_on_t) t_mark = min(t_mark, stats_next)
          if (chk_on_t)   t_mark = min(t_mark, chk_next)
          rem = t_mark - t                                      ! > 0 always (a fired mark advances past t)
          ! subdivide the time to the mark into ceil(rem/dt_cfl) even steps (each
          ! <= dt_cfl, lands exactly; rem <= dt_cfl => mland=1 => dt_try=rem, one step,
          ! no tiny straggler). Guard mland against int32 overflow when a mark is many
          ! billions of steps away (tiny dt_cfl): take a plain CFL step and subdivide
          ! later, once the mark is within reach.
          ratio = (rem - TLAND_EPS)/dt_cfl
          if (ratio >= real(huge(1_ip)-1, dp)) then
             dt_try = dt_cfl
          else
             mland  = max(1, ceiling(ratio))
             dt_try = rem/real(mland, dp)
          end if
          ! Rebuild the dt-dependent operators (keeping dt == operator dt) on it==1,
          ! when dt drifts past the tolerance, OR when the HELD dt would overshoot the
          ! mark (dt > rem). The overshoot clause is what keeps the landing EXACT: a
          ! held dt slightly larger than rem must be cut to dt_try (<= rem), else the
          ! step would overshoot and fire the output off-mark.
          if (it == 1 .or. abs(dt_try - dt) > dt_recompute_tol*dt .or. dt > rem + TLAND_EPS) then
             dt = dt_try
             call setup_ev_ops_ws()      ! zero-alloc rebuild (borrows sws scratch)
             if (fd_mode) then
                call fd_mean_refill()
             else
                call mean_flux_setup(n, dt, Re, bops%sdh, bops%sth, bops%bdh, bops%boff, mops%Uhm, mops%Ih)
             end if
          end if
       end if

       if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_a)
                         call par_transpose_timing(tm_pack, tm_nccl, tm_unpack, n_tr, .true.); end if
       call par_step_full_ws(pws, sws, n, bnkx, nz, dt, Re, kx, kz, bops, vops, mops, &
                          Qtar, u0, w0, eta, v, Nvp, Nep)
       if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_b)
                         tm_step = 1.0e3_dp*real(c_b-c_a,dp)/real(clk_rate,dp)
                         call par_transpose_timing(tm_pack, tm_nccl, tm_unpack, n_tr, .true.); end if
       step = step + 1;  t = t + dt

       ! --- decide which outputs fire this step (one place; fixed dt = step-based
       !     exactly as before, variable dt = landed-on-a-time-mark) -------------
       if (variable_dt) then
          last_step  = (t >= t_end - TLAND_EPS) .or. (it == n_steps_cap)  ! cap-exit also = last (L2)
          fire_log   = (mod(it, max(log_every,1)) == 0) .or. last_step
          fire_div   = (div_every > 0 .and. mod(it, max(div_every,1)) == 0)
          fire_stats = stats_on_t .and. (t >= stats_next - TLAND_EPS)
          fire_snap  = snap_on   .and. (t >= snap_next - TLAND_EPS)
          fire_chk   = (chk_on_t .and. (t >= chk_next - TLAND_EPS)) .or. last_step   ! always checkpoint at t_end
          need_phys  = fire_log .or. fire_stats .or. fire_div .or. (mod(it-1, dt_check_every) == 0)
       else
          last_step  = (it == n_steps_cap)
          fire_log   = (mod(it, max(log_every,1)) == 0) .or. last_step
          fire_div   = (div_every > 0 .and. mod(it, max(div_every,1)) == 0)
          fire_stats = (turbulent .and. stats_every > 0 .and. step >= stats_start .and. mod(it, max(stats_every,1)) == 0)
          fire_snap  = (snap_every > 0 .and. t >= snap_start .and. (mod(it, max(snap_every,1)) == 0 .or. last_step))
          fire_chk   = (chk_every > 0 .and. (mod(it, max(chk_every,1)) == 0 .or. last_step))
          need_phys  = fire_log .or. fire_stats .or. fire_div
       end if

       ! --- graceful wall-time stop (BOTH fixed and variable dt): once the elapsed
       !     wall time crosses max_wall_hours, write a checkpoint (and the stats file
       !     below) and exit cleanly, so a cluster job always leaves a restart point
       !     before it is killed. The clock is GLOBALLY reduced (max over ranks) so
       !     every rank takes the SAME decision -> the collective write never hangs.
       stop_wall = .false.
       if (max_wall_hours > 0.0_dp .and. mod(it-1, max(dt_check_every,1)) == 0) then
          call system_clock(c_b)
          elapsed_s = real(c_b - c_init0, dp)/real(clk_rate, dp)
          call par_reduce_max(elapsed_s)
          ! stop EARLY by wall_margin_min: a plain '>= limit' would budget ZERO
          ! time for the final checkpoint write and could sail past the Slurm kill.
          stop_wall = (elapsed_s >= max_wall_hours*3600.0_dp - max(wall_margin_min,0.0_dp)*60.0_dp)
       end if
       if (stop_wall) then
          fire_chk = .true.;  fire_log = .true.;  need_phys = .true.
          if (par_rank == 0) write(*,'(a,f7.3,a,i0,a,f0.3,a)') &
             '   [walltime] ', max_wall_hours, ' h reached at step ', step, ', t=', t, &
             ' -- writing checkpoint + stats and exiting'
       end if

       ! NaN/Inf watchdog -- global field sum (SUM probe: a MAX silently skips NaNs).
       ! par_reduce_sum gives every rank the same global sums -> identical decision.
       if (nan_every > 0 .and. mod(it, max(nan_every,1)) == 0) then
          if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_a); end if
          call field_sumabs_c(n+1, bnkx, nz, v,   swd1)
          call field_sumabs_c(n+1, bnkx, nz, eta, swd2)
          call par_reduce_sum(swd1);  call par_reduce_sum(swd2)
          if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_b)
                            tm_nan = 1.0e3_dp*real(c_b-c_a,dp)/real(clk_rate,dp); end if
          if (.not. (swd1 == swd1 .and. swd2 == swd2 .and. &
                     swd1 < huge(1.0_dp) .and. swd2 < huge(1.0_dp))) then
             if (par_rank == 0) then
                write(*,'(a)')            ' !! NON-FINITE STATE DETECTED -- terminating'
                write(*,'(a,i8,a,f10.4)') ' !!   at step ', step, '   t = ', t
             end if
             call par_finalize();  error stop 2
          end if
       end if

       ! kx=0 z-Hermitian watchdog. Real physical fields require the kx=0
       ! spectral columns to be conjugate-symmetric in kz; symmetry is enforced only
       ! at the IC, so a slow leak would grow silently. Runs ONLY on the rank that
       ! owns kx=0 (no communication, no collective -> no deadlock risk from the
       ! rank-local trigger). hsym_fix projects back onto the symmetric subspace --
       ! default OFF so trajectories (and the --validate gate) stay bit-identical.
       if (hsym_every > 0 .and. mod(it, max(hsym_every,1)) == 0 .and. d%owns_mode00) then
          call hsym_check_c(n, bnkx, nz, v,   swd1)
          call hsym_check_c(n, bnkx, nz, eta, swd2)
          write(*,'(a,i8,a,es10.2,a,es10.2)') '   [hsym] step ', step, &
             '   max|v(0,kz)-conj(v(0,-kz))| = ', swd1, '   eta: ', swd2
          if (hsym_fix) then
             call hsym_fix_c(n, bnkx, nz, v)
             call hsym_fix_c(n, bnkx, nz, eta)
          end if
       end if

       ! physical velocities are needed by the log, the stats sample, the div probe,
       ! and (variable dt) the CFL fmax that sets the next step. need_phys is
       ! rank-uniform -> the collectives in par_monitor never deadlock.
       if (need_phys) then
          if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_a); end if
          call par_monitor(fmax)
          fmax_cur = fmax                      ! latest global CFL fmax -> next variable-dt step
          if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_b)
                            tm_mon = 1.0e3_dp*real(c_b-c_a,dp)/real(clk_rate,dp); end if
       end if

       ! v-solve residual probe, div_every cadence. Runs FIRST
       ! among the gated diagnostics: it reads sws%rhsv (the ks=3 assembled RHS --
       ! y_vif_solve never modifies it) and lends the dead sws%rhs/Lv/scr, all of
       ! which the pressure path below scribbles on. Line 1 (both backends):
       ! the STRONG-form recompute |(L - a3 L^2)v - rhsv|/|rhsv| at the ks=3
       ! a3 = 0.5*MCSUB(3)*dt/Re. Measured 0.4-1.4 cheb / ~1e-2 fd in developed
       ! turbulence, vs 4-8e-5 / 4e-4..1e-3 in young flow. Note the
       ! solve is Shen-GALERKIN, so the strong recompute against the
       ! UNPROJECTED rhsv measures flow/resolution-dependent discretisation
       ! consistency (the marginally-resolved sqrt(a3) wall layer dominates),
       ! NOT solver precision -- kept as a resolution/consistency meter,
       ! UNGATED. Line 2 = the backward-error canary proper:
       !   cheb: y_vres_galerkin -- per-banded-solve residuals AS SOLVED + the
       !     closure wall-Dv quality. Backward-error quantities, flow- and
       !     deck-INDEPENDENT (true-basis numpy model, wiring-checked).
       !     CALIBRATED on a single-node cheb run (flagship
       !     conditioning): helm 1.8-3.1e-16, closure 5.3e-14..1.9e-13 (the
       !     wall-deriv sums carry ~n^2-amplified roundoff -- a genuinely
       !     different floor, hence the split gate). WARN = measured*10 ->
       !     next half-decade: helm 5e-15, closure 2e-12.
       !   fd: y_vres_max_win -- strong residual EXCLUDING 2*kl+1 rows/wall
       !     (BC/closure rows + their banded-apply spread; window also covers
       !     the sqrt(a3) Stokes layer, inside the first 1-2 cells on every
       !     current deck). Interior collocation rows are solver-exact.
       !     CALIBRATED on a single-node fd run (flagship n=470):
       !     8.9e-13..1.2e-12. WARN 2e-11 (measured*10 -> next half-decade).
       ! Collectives: all ranks run the same backend branch (fd_mode is
       ! rank-uniform) -> identical par_reduce_max counts, no deadlock.
       if (fire_div) then
          block
            real(dp) :: aal3_, rres_, rmax_, fmax_
            real(dp) :: r1_, f1_, r2_, f2_, dvv_, dvp_, rg1_, rg2_, rcl_, riw_
            aal3_ = 0.5_dp*MCSUB(3)*dt/Re
            call y_apply_L(bops, n, bnkx, nz, kx, kz, v,       sws%rhs, sws%scr)
            call y_apply_L(bops, n, bnkx, nz, kx, kz, sws%rhs, sws%Lv,  sws%scr)
            call y_vres_max(n, bnkx, nz, aal3_, sws%rhs, sws%Lv, sws%rhsv, rmax_, fmax_)
            call par_reduce_max(rmax_);  call par_reduce_max(fmax_)
            rres_ = rmax_/max(fmax_, tiny(1.0_dp))
            if (par_rank == 0) write(*,'(a,es10.2,a)') &
               '   [res] v-solve rel residual (ks=3) = ', rres_, &
               '   (strong form; resolution meter, ungated)'
            if (fd_mode) then
               call y_vres_max_win(n, bnkx, nz, aal3_, 2*bops%kl+1, n-2*bops%kl-1, &
                                   sws%rhs, sws%Lv, sws%rhsv, rmax_, fmax_)
               call par_reduce_max(rmax_);  call par_reduce_max(fmax_)
               riw_ = rmax_/max(fmax_, tiny(1.0_dp))
               if (par_rank == 0) then
                  write(*,'(a,i0,a,es10.2)') '   [res] fd interior residual (', &
                     2*bops%kl+1, ' wall rows excluded each side) = ', riw_
                  if (.not. (riw_ <= 2.0e-11_dp)) write(*,'(a)') &
                     '   [res] WARNING: fd interior residual above 2e-11 -- the dp margin is thinning'
               end if
            else
               call y_vres_galerkin(bops, n, bnkx, nz, aal3_, kx, kz, sws%rhsv, v, &
                                    sws%vgsh, sws%vvsh, sws%rhs, sws%Lv, sws%scr, &
                                    r1_, f1_, r2_, f2_, dvv_, dvp_)
               call par_reduce_max(r1_);   call par_reduce_max(f1_)
               call par_reduce_max(r2_);   call par_reduce_max(f2_)
               call par_reduce_max(dvv_);  call par_reduce_max(dvp_)
               rg1_ = r1_/max(f1_, tiny(1.0_dp))
               rg2_ = r2_/max(f2_, tiny(1.0_dp))
               rcl_ = dvv_/max(dvp_, tiny(1.0_dp))
               if (par_rank == 0) then
                  write(*,'(a,es10.2,a,es10.2,a,es10.2)') &
                     '   [res] galerkin residual (ks=3): helm1 = ', rg1_, &
                     '   helm2 = ', rg2_, '   closure = ', rcl_
                  if (.not. (rg1_ <= 5.0e-15_dp .and. rg2_ <= 5.0e-15_dp .and. &
                             rcl_ <= 2.0e-12_dp)) write(*,'(a)') &
                     '   [res] WARNING: galerkin residual above gate (helm 5e-15 / closure 2e-12) -- the dp margin is thinning'
               end if
            end if
          end block
       end if

       ! statistics sample (turbulent). Time-regular under variable dt (fire_stats
       ! lands on stats marks) -> the sample average is UNBIASED; step-based sampling
       ! would over-weight the small-dt (high-CFL) periods.
       if (fire_stats) then
          if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_a); end if
          call stats_accumulate(sacc, n, nxd, anz, &                  ! LOCAL plane count
               pws%P6(:,:,:,1), pws%P6(:,:,:,3), pws%P6(:,:,:,2))     ! [pu,pv,pw] slice map
          ! drive force (CFR mean dp/dx = beta) -- time-avg accumulated on the (0,0)
          ! owner (rank 0); a global scalar, no reduction needed.
          if (par_rank == 0) then
             beta_host = sws%mg%beta_d                ! D2H whole-array (1 elem); element access on a device component faults on host
             beta_sum  = beta_sum + beta_host(1)
          end if
          ! pressure sample: recover p on the dealiased grid, accumulate <p>,<p'^2>.
          if (snap_pressure) call par_pressure_stats()
          if (variable_dt .and. stats_on_t) then
             istats = istats + 1;  stats_next = stats_start_t + real(istats,dp)*stats_dt
          end if
          if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_b)
                            tm_stats = 1.0e3_dp*real(c_b-c_a,dp)/real(clk_rate,dp); end if
       end if

       ! divergence probe. par_monitor's transforms run IN-PLACE on their spectral
       ! input (cheb_fft_c_inv(A,A)), so they CLOBBER S6(1:3) -- re-recover the
       ! spectral velocities fresh here before reading them, exactly as src/serial/main.F90's
       ! div block does. recover_uw is local (per-column, no comm), so it is safe to
       ! call on this subset of iterations. (v is untouched: par_monitor transforms a
       ! demoted copy.)
       if (fire_div) then
          if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_a); end if
          call recover_uw(ny, bnkx, nz, kx, kz, v, eta, u0, w0, &
                          pws%S6(:,:,:,1), pws%S6(:,:,:,2), pws%S6(:,:,:,4))
          call divergence_check(n, bnkx, nz, kxd, kzd, pws%S6(:,:,:,1), v, pws%S6(:,:,:,2), &
                                sws%rhs, maxdiv)   ! dvh scratch = sws%rhs (dead between steps)
          call par_reduce_max(maxdiv)
          if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_b)
                            tm_div = 1.0e3_dp*real(c_b-c_a,dp)/real(clk_rate,dp); end if
          if (par_rank == 0) write(*,'(a,i8,a,es10.2)') '   [div] step ', step, '   max|div u| = ', maxdiv
       end if

       ! log line (rank 0 holds the mean column)
       if (fire_log) then
          cfl = dt*fmax
          if (par_rank == 0) then
             istat = cudaDeviceSynchronize()
             u0h  = u0
             flux = bulk_velocity(n, real(u0h))
             if (fd_mode) then                     ! u0 near y=0: mid NODE value
                cl = real(u0h(n/2))
             else
                cl = 0.0_dp
                do kk = 0, n, 2
                   cl = cl + real(u0h(kk))*real((-1)**(kk/2), dp)
                end do
             end if
             ret = re_tau(n, Re, real(u0h))
             beta_host = sws%mg%beta_d                      ! D2H (1 elem); see stats block note
             beta_h = beta_host(1)                          ! drive force = mean dp/dx (CFR)
             write(*,'(i8,1x,f11.4,1x,es11.3,1x,f9.4,1x,f11.6,1x,f9.5,1x,f9.3,1x,es11.3)') &
                  step, t, dt, cfl, flux, cl, ret, beta_h
             if (cfl > cfl_max) &
                write(*,'(a,f8.4,a,f8.4)') '   ** WARNING: CFL ', cfl, ' exceeds the sqrt(3)/pi ceiling ', cfl_max
          end if
       end if

       ! --- periodic distributed checkpoint (step-keyed name; rolling last-2) ---
       ! trigger identical on every rank (it/n_steps/config) -> the collective write
       ! never deadlocks. step is the cumulative counter -> names continue across a
       ! restart. Build cost is small + infrequent (NOT inside the [time] step timer).
       ! FD checkpoints are stamped (ybasis=fd-point) -- no suppression.
       if (fire_chk) then
          if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_a); end if
          ckname = chk_filename(chk_file, step, t)
          call par_chk_write(trim(ckname), d, n, nx, nz, Re, t, dt, Qtar, step, &
                             v, eta, u0, w0, y, Lx, Lz, okck, fp32=chk_fp32)
          if (okck) then                     ! roll: keep the 2 newest, delete the 3rd
             ! okck means every MPI write+count verified AND the rename
             ! committed (par_chk_write) -- only then is history rolled.
             ck_evict = ck2;  ck2 = ck1;  ck1 = ckname
             if (par_rank == 0) then
                if (len_trim(ck_evict) > 0) &
                   call execute_command_line('rm -f '//trim(ck_evict), wait=.false.)
                ! +restart pointer '<chk base>.latest', written
                ! ATOMICALLY (.tmp + rename) with iostat -- a job killed here
                ! can never leave a torn pointer for the next restart.
                open(newunit=rs_unit, file=trim(chk_file)//'.latest.tmp', &
                     status='replace', action='write', iostat=ios_ptr)
                if (ios_ptr == 0) then
                   write(rs_unit, '(a)', iostat=ios_ptr) trim(ckname)
                   close(rs_unit)
                end if
                if (ios_ptr == 0) then
                   call execute_command_line('mv -f '//trim(chk_file)//'.latest.tmp '// &
                                             trim(chk_file)//'.latest', wait=.true., exitstat=ios_ptr)
                end if
                if (ios_ptr /= 0) write(*,'(a)') &
                   ' !! [io] .latest pointer update FAILED -- restart will fall back to older names'
             end if
          else
             ! failed checkpoint: NOTHING is evicted and the pointer is
             ! untouched -- the existing history stays the restart chain. Loud
             ! always; FATAL when this was the final/wall-margin write that the
             ! whole restart story depends on.
             if (par_rank == 0) write(*,'(a,i0,a)') &
                ' !! CHECKPOINT WRITE FAILED at step ', step, &
                ' -- previous checkpoints KEPT; check disk quota / filesystem'
             if (last_step .or. stop_wall) then
                if (par_rank == 0) write(*,'(a)') &
                   ' !! this was the FINAL checkpoint of the run -- aborting so the failure is visible'
                call par_finalize();  error stop 4
             end if
          end if
          if (variable_dt .and. chk_on_t .and. t >= chk_next - TLAND_EPS) then   ! real mark, not a forced final
             ichk = ichk + 1;  chk_next = real(ichk,dp)*chk_dt
          end if
          if (timing) then; call system_clock(c_b)
                            tm_io = tm_io + 1.0e3_dp*real(c_b-c_a,dp)/real(clk_rate,dp); end if
       end if

       ! --- physical u,v,w snapshot (clean nx*nz grid; numbered keep-all) --------
       ! starts only once t >= snap_start (skips the transient). recover u,v,w fresh
       ! (par_monitor clobbers S6(1:3)), clean-inverse each, write. Trigger is
       ! rank-uniform (snap_every/snap_start/it/t) -> the collective transform+write
       ! never deadlocks. recover_uw/par_to_physical_r2c consume their inputs (S6
       ! slices: u^, w^, demoted v^) -- v,eta,u0,w0 are untouched.
       if (fire_snap) then
          if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_a); end if
          snap_bytes = merge(4_ip, 8_ip, snap_fp32)
          snapname = chk_filename(snap_file, step, t)    ! generic step/t/tp-tagged namer
          ! ORDER MATTERS: the pressure runs FIRST because its eval_H
          ! clobbers ALL of pws; the u,v,w fields below land in P6(4:6) and must
          ! be produced after the block's last eval_H. p survives in PH(2).
          if (snap_pressure) call par_pressure_snapshot()      ! diagnostic p -> PH(2)
          call recover_uw(ny, bnkx, nz, kx, kz, v, eta, u0, w0, &                 ! S6: u^, w^, scratch
                          pws%S6(:,:,:,1), pws%S6(:,:,:,2), pws%S6(:,:,:,4))
          call demote(n, bnkx, nz, v, pws%S6(:,:,:,3))                              ! S6(3) = v^ (tp copy)
          call par_to_physical_r2c(pws%plans, d, ny, nx, nz, bnkx, anz_co, pws%S6(:,:,:,1), &
                                   pws%Bz(:,:,:,1:1), pws%Az(:,:,:,1:1), pws%P6(:,:,:,4))
          call par_to_physical_r2c(pws%plans, d, ny, nx, nz, bnkx, anz_co, pws%S6(:,:,:,3), &
                                   pws%Bz(:,:,:,1:1), pws%Az(:,:,:,1:1), pws%P6(:,:,:,5))
          call par_to_physical_r2c(pws%plans, d, ny, nx, nz, bnkx, anz_co, pws%S6(:,:,:,2), &
                                   pws%Bz(:,:,:,1:1), pws%Az(:,:,:,1:1), pws%P6(:,:,:,6))
          if (snap_pressure) then
             call par_snap_write(trim(snapname), d, n, nx, nz, Re, t, step, snap_bytes, &
                                 pws%P6(:,:,:,4), pws%P6(:,:,:,5), pws%P6(:,:,:,6), &
                                 y, Lx, Lz, oksnap, p=pws%PH(:,:,:,2))
          else
             call par_snap_write(trim(snapname), d, n, nx, nz, Re, t, step, snap_bytes, &
                                 pws%P6(:,:,:,4), pws%P6(:,:,:,5), pws%P6(:,:,:,6), &
                                 y, Lx, Lz, oksnap)
          end if
          ! test oksnap and warn loudly on failure
          ! (data product lost, run itself unaffected; non-fatal by design).
          if (.not. oksnap .and. par_rank == 0) write(*,'(a,i0,a)') &
             ' !! SNAPSHOT WRITE FAILED at step ', step, ' -- output lost; check disk quota / filesystem'
          if (variable_dt .and. snap_on) then
             isnap = isnap + 1;  snap_next = snap_start + real(isnap,dp)*snap_dt
          end if
          ! optionally drop the clean-grid xl slot the snapshot just used:
          ! reclaims ~1.26 GB/GPU at the Re2000 grid on 16 nodes, at the cost of
          ! one realloc at the next snapshot. Default OFF (20+ nodes keep both
          ! slots resident).
          if (snap_evict) call xl_release_current()
          if (timing) then; istat = cudaDeviceSynchronize(); call system_clock(c_b)
                            tm_io = tm_io + 1.0e3_dp*real(c_b-c_a,dp)/real(clk_rate,dp); end if
       end if

       ! --- per-step timing: accumulate + report AFTER the io blocks so tm_io
       !     lands on the step that paid it. 3-line block:
       !     step and its compute/comm split, the comm phases, everything around
       !     the step (monitor, watchdogs, io). All ms, rank 0. io gets f0.3: a
       !     checkpoint write can run to minutes and must never print as ******.
       if (timing) then
          tot_step  = tot_step  + tm_step;   tot_mon = tot_mon + tm_mon
          tot_stats = tot_stats + tm_stats;  tot_nan = tot_nan + tm_nan
          tot_div   = tot_div   + tm_div;    tot_io  = tot_io  + tm_io
          tot_pack  = tot_pack  + tm_pack;   tot_nccl = tot_nccl + tm_nccl
          tot_unpack = tot_unpack + tm_unpack
          if (par_rank == 0 .and. mod(it, max(time_every,1)) == 0) then
             write(*,'(a,i8,3(a,f9.3))') '   [time] step ', step, &
                ':  step_full=', tm_step, ' ms  =  compute=', &
                tm_step-(tm_pack+tm_nccl+tm_unpack), '  +  comm=', &
                tm_pack+tm_nccl+tm_unpack
             write(*,'(a,3(a,f9.3),a,i0,a)') '   [time]   ', 'comm   :  pack=', tm_pack, &
                '   nccl=', tm_nccl, '   unpack=', tm_unpack, '   (ntr=', n_tr, ')'
             write(*,'(a,4(a,f9.3),a,f0.3)') '   [time]   ', 'around :  mon =', tm_mon, &
                '   stats=', tm_stats, '   nan=', tm_nan, '   div=', tm_div, '   io=', tm_io
          end if
       end if

       ! terminate: variable dt lands exactly on t_end (or hits the backstop cap);
       ! a wall-time stop exits from either fixed or variable dt. The checkpoint above
       ! (fire_chk forced on last_step / stop_wall) is the restart point.
       if (stop_wall .or. (variable_dt .and. last_step)) exit
    end do
    call system_clock(c_run1);  wall_s = real(c_run1-c_run0,dp)/real(clk_rate,dp)
    if (par_rank == 0) then
       write(*,'(a)') '--------------------------------------------------------------'
       write(*,'(a,f10.2,a,f12.2,a)') ' time-loop wall = ', wall_s, ' s   (', &
            real(step-step0,dp)/max(wall_s,1.0e-9_dp), ' steps/s)'
       if (timing) then
          write(*,'(a)') ' --- per-category timing (rank 0; total s, % of loop wall) ---'
          call tline('par_step_full              ', tot_step,  wall_s)
          call tline('  +- transpose comm (total) ', tot_pack+tot_nccl+tot_unpack, wall_s)
          call tline('  +-     NCCL all-to-all     ', tot_nccl,   wall_s)
          call tline('  +-     pack                ', tot_pack,   wall_s)
          call tline('  +-     unpack              ', tot_unpack, wall_s)
          call tline('  +- compute (step - comm)   ', tot_step-(tot_pack+tot_nccl+tot_unpack), wall_s)
          call tline('monitor (recover+xform+CFL)', tot_mon,   wall_s)
          call tline('stats accumulate           ', tot_stats, wall_s)
          call tline('nan watchdog               ', tot_nan,   wall_s)
          call tline('divergence probe           ', tot_div,   wall_s)
          call tline('io (checkpoint + snapshot) ', tot_io,    wall_s)
          ! sum of the top-level rows (sub-rows of par_step_full excluded) -- if
          ! this is well below 100%, a cost is escaping the categories above.
          call tline('accounted (sum of rows)    ', &
               tot_step+tot_mon+tot_stats+tot_nan+tot_div+tot_io, wall_s)
       end if
    end if

    ! --- final report ---------------------------------------------------------
    istat = cudaDeviceSynchronize()
    if (par_rank == 0) then
       u0h = u0;  flux = bulk_velocity(n, real(u0h));  ret = re_tau(n, Re, real(u0h))
       write(*,'(a)')        '--------------------------------------------------------------'
       write(*,'(a,f12.8,a,f12.8,a)') ' final bulk velocity U_b = ', flux, '   (target ', Qtar, ')'
    end if
    if (turbulent) then
       call par_stats_finalize(sacc, n, mprof, mvp, mwp, uup, vvp, wwp, uvp)   ! Allreduce (all ranks)
       if (snap_pressure) call par_stats_finalize_p(sacc, n, mpp, ppv)         ! Allreduce <p>,<p'^2>
       dpdx_avg = beta_sum / real(max(sacc%nsamp,1), dp)                       ! time-avg drive force (rank 0)
       if (par_rank == 0) then
          write(*,'(a,f10.4,a,i0,a)') ' final Re_tau = ', ret, '   (stats from ', sacc%nsamp, ' samples)'
          if (snap_pressure) write(*,'(a,es12.4)') ' time-avg drive force (mean dp/dx) = ', dpdx_avg
          ! write only if samples were actually collected; stamp the name with step/t
          ! so successive runs (restart-continue, or a wall-time stop) each leave a
          ! DISTINCT partial stats file (combine them weighted by their nsamp header).
          if (sacc%nsamp > 0) then
             snapname = chk_filename(stats_file, step, t)
             call write_stats_file(trim(snapname), mprof, mvp, mwp, uup, vvp, wwp, uvp, &
                                   mpp, ppv, dpdx_avg, snap_pressure, sacc%nsamp)
             write(*,'(a,a)') ' mean profile + Reynolds stresses written to ', trim(snapname)
          else
             write(*,'(a)') ' (no stats samples collected -- stats file not written)'
          end if
       end if
    else if (par_rank == 0) then
       errpar = 0.0_dp
       if (fd_mode) then                     ! FD state = point values
          do jj = 0, n
             aan = 1.5_dp*Qtar*(1.0_dp - y(jj)*y(jj))
             errpar = max(errpar, abs(real(u0h(jj)) - aan))
          end do
          write(*,'(a,es12.4)') ' max|u0 - analytic Poiseuille (point values)| = ', errpar
       else
          do jj = 0, n
             aan = 0.0_dp
             if (jj == 0) aan =  3.0_dp*Qtar/4.0_dp
             if (jj == 2) aan = -3.0_dp*Qtar/4.0_dp
             errpar = max(errpar, abs(real(u0h(jj)) - aan))
          end do
          write(*,'(a,es12.4)') ' max|u0 - analytic Poiseuille (Cheb coeffs)| = ', errpar
       end if
       if (errpar < 1.0e-4_dp) then
          write(*,'(a)') ' [PASS] laminar Poiseuille recovered through the distributed driver'
       else
          write(*,'(a)') ' [...] not fully relaxed yet -- increase n_steps (viscous time ~ Re)'
       end if
    end if
    if (par_rank == 0) write(*,'(a)') '=============================================================='

    call par_nl_destroy(pws);  call stepper_ws_free(sws)
    if (turbulent) then
       call stats_destroy(sacc)
       deallocate(mprof, mvp, mwp, uup, vvp, wwp, uvp, mpp, ppv)
    end if
    if (div_every > 0) deallocate(kxd, kzd)
    deallocate(eta, v, u0, w0, Nvp, Nep)
    if (fd_mode .and. par_rank == 0) then
       block
         integer(ip) :: ih_, im_
         call fd_helm_cache_stats(ih_, im_)
         write(*,'(a,i0,a,i0,a)') ' [fd] helm cache: ', ih_, ' hits, ', im_, &
              ' misses (misses>0 with cache ON = an expression drifted; report it)'
       end block
    end if
    deallocate(kx, kz);  call vsolve_ops_free(vops)
    deallocate(inv_dy_d)
    if (pres_on) then
       deallocate(pp_d, pm_d, Cinv_d, kxw, kzw)   ! per-call scratch lives in sws/pws
    end if
  end subroutine run_production

  !-- build/rebuild the per-(kx,kz) v-solve operators for the CURRENT dt,
  !   ZERO slab allocations: the seven slab-sized build temps are borrowed from
  !   the stepper workspace (dead between steps; every temp is fully written
  !   before read, so garbage content is fine). A self-allocating build would be
  !   a ~9 GB/GPU transient -- fatal at the first mid-run variable-dt rebuild on
  !   a 16-node frame, where a 1.29 GB request meets 0.56 GB free. ONE definition
  !   serves init and every rebuild, so the whole suite + any run's first step
  !   gate this exact path. Requires sws already allocated.
  subroutine setup_ev_ops_ws()
    integer(ip) :: ks_
    real(dp)    :: aal_
    if (fd_mode) then
       ! FD build through the dispatch helper, SAME lent temps
       ! (per-stage alpha_k; the Cheb branch below is untouched -> bit-identity).
       ! prefactor FIRST so the vif setup solves below hit the fresh cache
       ! (the other order logs 3 spurious misses per retune); the
       ! expression chain below replicates (par_)ev_stage/fd_vif VERBATIM so
       ! the exact-fp registry lookup hits. Rebuilt here on every dt retune.
       block
         real(dp) :: lams_(4), bet_, aa_, cgb_
         logical  :: cb_
         integer(ip) :: k2_
         lams_(1) = 0.0_dp
         do k2_ = 1, 3
            bet_ = 0.5_dp*MCSUB(k2_)
            aa_  = bet_*dt/Re
            lams_(1+k2_) = 1.0_dp/aa_
         end do
         call fd_helm_cache_build(n, fd_kl, bnkx, nz, 4_ip, lams_, kx, kz, &
                                  bops%d2b, fd_cache_gb, cb_, cgb_)
         if (par_rank == 0) then
            if (cb_) then
               write(*,'(a,f7.2,a)') '   [fd] helm factor cache ON  (', cgb_, &
                    ' GiB, 4 lambdas; bit-identical fast path)'
            else
               write(*,'(a,f7.2,a,f6.2,a)') '   [fd] helm factor cache OFF (needs ', &
                    cgb_, ' GiB > fd_cache_gb = ', fd_cache_gb, &
                    ' -- rebuild-per-solve posture)'
            end if
         end if
       end block
       do ks_ = 1, 3
          aal_ = 0.5_dp*MCSUB(ks_)*dt/Re
          call y_vif_setup(bops, n, bnkx, nz, kx, kz, aal_, &
                           vops%vm(:,:,:,ks_), vops%vp(:,:,:,ks_), &
                           vops%Cinv(:,:,:,:,ks_), &
                           sws%rhs, sws%ftil, sws%vgsh, &
                           sws%Lv, sws%scr, sws%rhsv, sws%Nv_cur)  ! gpc/vpc
       end do
       return
    end if
    call stepper_ev_setup_gpu_ws(n, bnkx, nz, dt, Re, kxh(kxlo:kxhi), kzh, &
                                 vops%vm, vops%vp, vops%Cinv, &
                                 sws%rhs, sws%ftil, sws%vgsh, &     ! rhs / ftil / gsh
                                 sws%Lv, sws%scr, sws%rhsv, &       ! gpc / gmc / vpc
                                 sws%Nv_cur)                        ! vmc
  end subroutine setup_ev_ops_ws

  !-- FD mean apparatus -> mops [TRAP i]; init + every dt retune --
  subroutine fd_mean_refill()
    logical :: okp_
    call fd_mean_flux_setup(n+1, fd_kl, d2f, wfile, dt, Re, UhF, IhF, okp_)
    if (.not. okp_) then
       if (par_rank == 0) write(*,'(a)') ' ** FD mean setup FAILED (zero pivot)'
       call par_finalize();  error stop 1
    end if
    mops%Uhm(0:n,:) = UhF(1:n+1,:)
    mops%Ih = IhF
  end subroutine fd_mean_refill

  !-- distributed CFL/stats monitor: recover u,w; transform u,v,w to physical
  !   (this rank's z-slab); local CFL fmax -> global MAX. Populates pws%P6(:,:,:,1:3)
  !   as [pu,pw,pv] via ONE nfield=3 transpose on pws scratch: S6 ->
  !   1:uhat 2:what 3:vc. The transforms run IN-PLACE on their spectral input, so
  !   this CLOBBERS S6(1:3) -- the divergence probe re-recovers them fresh. --------
  subroutine par_monitor(fmax)
    real(dp), intent(out) :: fmax
    call recover_uw(ny, bnkx, nz, kx, kz, v, eta, u0, w0, &
                    pws%S6(:,:,:,1), pws%S6(:,:,:,2), pws%S6(:,:,:,4))
    call demote(n, bnkx, nz, v, pws%S6(:,:,:,3))
    call par_to_physical_nf(pws%plans, d, ny, nx, nz, nxd, nzd, bnkx, anz, 3_ip, &
                            pws%S6(:,:,:,1:3), pws%Bz(:,:,:,1:3), pws%Az(:,:,:,1:3), &
                            pws%Af, pws%P6(:,:,:,1:3))
    call cfl_freq_max(ny, nxd, anz, pws%P6(:,:,:,1), pws%P6(:,:,:,3), pws%P6(:,:,:,2), &
                      inv_dx, inv_dz, inv_dy_d, fmax, u_galilean)
    call par_reduce_max(fmax)
  end subroutine par_monitor

  !=============================================================================
  ! par_pressure_phat -- recover the diagnostic pressure in COARSE SPECTRAL form
  ! (-> phat, Chebyshev-y coeffs, tp), FRESH from the CURRENT state (v,eta,u0,w0
  ! untouched). Solves the modified pressure Pi = p + |u|^2/2 by the Neumann
  ! influence matrix (par_pressure_mod), then recovers p = Pi - |u|^2/2 with the
  ! |u|^2/2 formed on the 3/2-DEALIASED grid (alias-free --
  ! S6). dp solve tier (promote H_hat tp->dp); demote p_hat->tp. y-local, mode-
  ! local -> no new communication. The caller inverse-transforms phat to whichever
  ! physical grid it needs (clean for snapshots, dealiased for stats). Conventions
  ! validated in src/models/work_pressure_rhs_model.py.
  !=============================================================================
  subroutine par_pressure_phat()
    ! 1. dealiased H_hat -> pws%S3 [Hx,Hy,Hz] (tp), velocities -> pws%P6 [pv,pu,pw]
    call par_nl_eval_H(pws, v, eta, u0, w0)
    ! every buffer below this line is an ALIAS: after eval_H, all of pws
    ! is refillable, and sws (the stepper's dp scratch) is dead between steps.
    ! S3(1:3) are promoted OUT first, so S3(1:2) are free for qhat/phat below.
    call pres_promote_c(n, bnkx, nz, pws%S3(:,:,:,1), sws%Lv)       ! Hx (dp)
    call pres_promote_c(n, bnkx, nz, pws%S3(:,:,:,2), sws%Neta_cur) ! Hy (dp)
    call pres_promote_c(n, bnkx, nz, pws%S3(:,:,:,3), sws%rhsv)     ! Hz (dp)

    ! 2. assemble R = +div.H + Neumann wall data, solve (D^2-k^2)Pi = R -> sws%scr.
    !    The shared transform-free core (same on the oracle + slab --validate paths).
    call pres_assemble_solve(n, bnkx, nz, nx, d%b_kx0, Re, kx, kz, kxw, kzw, &
                             bops%sd, bops%st, bops%bd, bops%boff, &
                             pp_d, pm_d, Cinv_d, sws%Lv, sws%Neta_cur, sws%rhsv, v, sws%scr)

    ! 3. recover p = Pi - |u|^2/2, with |u|^2/2 DEALIASED (3/2 grid -> truncate):
    !    q on nxd grid (PH(1), dead after eval_H's forward transform) -> q_hat
    !    coarse spectral (S3(1)) -> p_hat = Pi - q_hat -> S3(2).
    call pres_half_usq(ny, nxd, anz, 1.0_tp/real(nxd*nzd,tp), &
                       pws%P6(:,:,:,2), pws%P6(:,:,:,1), pws%P6(:,:,:,3), pws%PH(:,:,:,1))
    call par_to_spectral_da_r2c(pws%plans, d, ny, nx, nz, nxd, nzd, bnkx, anz, &
                                pws%PH(:,:,:,1), pws%Af, pws%Az(:,:,:,1:1), &
                                pws%Bz(:,:,:,1:1), pws%S3(:,:,:,1))
    call pres_combine_p(n, bnkx, nz, sws%scr, pws%S3(:,:,:,1), pws%S3(:,:,:,2))
  end subroutine par_pressure_phat                          ! p_hat = S3(2) (tp)

  ! Snapshot: p_hat -> CLEAN nx*nz grid -> PH(2) (the 4th snapshot field; the
  ! clean (ny,nx,a_nz_loc) view is a sequence-association prefix of the bigger
  ! dealiased PH slice -- same trick as the u,v,w fields in P6(4:6)).
  subroutine par_pressure_snapshot()
    integer(ip) :: anz_co
    anz_co = d%a_nz_loc
    call par_pressure_phat()
    call par_to_physical_r2c(pws%plans, d, ny, nx, nz, bnkx, anz_co, pws%S3(:,:,:,2), &
                             pws%Bz(:,:,:,1:1), pws%Az(:,:,:,1:1), pws%PH(:,:,:,2))
  end subroutine par_pressure_snapshot

  ! Stats: p_hat -> DEALIASED grid (-> pp_da), then accumulate <p>,<p'^2> exactly
  ! like the velocity stats (same grid + 1/(nxd*nzd) normalisation -> consistent).
  subroutine par_pressure_stats()
    call par_pressure_phat()
    call par_to_physical_da_r2c(pws%plans, d, ny, nx, nz, nxd, nzd, bnkx, anz, &
                                pws%S3(:,:,:,2), pws%Bz(:,:,:,1:1), pws%Az(:,:,:,1:1), &
                                pws%Af, pws%P6(:,:,:,4))
    call stats_accum_p(sacc, n, nxd, anz, pws%P6(:,:,:,4))    ! pp_da = P6(4)
  end subroutine par_pressure_stats

  !=============================================================================
  ! VALIDATION RUNG -- build the deck's IC, evolve validate_steps, diff the slab
  ! path bit-for-bit vs the single-GPU oracle (== test_driver_turb_mp, but
  ! parameterised by the runtime namelist grid).
  !=============================================================================
  subroutine run_validate()
    real(real64) :: tol, tol_st, tol_p, prof_margin, retau_margin, ediff, vdiff, pdiff
    complex(dp), allocatable :: he0(:,:,:), hv0(:,:,:), eF(:,:,:), vF(:,:,:), eS(:,:,:), vS(:,:,:)
    complex(dp), allocatable :: pF(:,:,:), pS(:,:,:)              ! diagnostic pressure (oracle, slab)
    complex(dp) :: hu00(0:n), hw00(0:n), u0F(0:n), w0F(0:n), u0S(0:n), w0S(0:n)
    real(dp) :: muF(0:n), mvF(0:n), mwF(0:n), uuF(0:n), vvF(0:n), wwF(0:n), uvF(0:n), retF
    real(dp) :: muS(0:n), mvS(0:n), mwS(0:n), uuS(0:n), vvS(0:n), wwS(0:n), uvS(0:n), retS
    logical  :: ok_e, ok_v, ok_m, ok_st, ok_p, all_ok
    integer(ip) :: j

    if (do_restart .and. par_rank == 0) &
       write(*,'(a)') '   (note: --validate ignores do_restart; it builds the in-code IC)'
    ! dist vs single-GPU: with a2a_overlap=0 the paths are BIT-IDENTICAL in tp
    ! (fields/pressure 0.0 observed; only the dp stats accumulation reassociates,
    ! ~1e-14). Bit-identity SURVIVES both the in-place y-DCT and
    ! the fused-vs-split RK RHS kernels: max|eta|=max|v|=max|p| = 0.0E+00
    ! exactly, stats 5.2e-14 (the dp accumulation reassociation).
    ! Neither is GUARANTEED by cuFFT/codegen, so re-verify this line after any
    ! cuFFT/driver/compiler upgrade before citing it. With a2a_overlap=1 the
    ! trajectories pick up tp-ULP-scale drift over the 5 steps: measured at
    ! 128^3, P=4, in sp, fields ~3.2e-10, stats
    ! 1.75e-9, pressure 5.96e-8 (= one fp32 ulp of each quantity's own scale --
    ! eps_sp*|p| ~ 6e-8), and the SAME pattern at dp-ulp scale in dp (8e-18 /
    ! 3e-14 / 6e-16). That is reassociation, not an ordering bug: dp clears the
    ! 1e-9 gate with seven digits to spare. So the gates are per-precision AND
    ! per-quantity, observation-based with >=15x margin -- a dp-calibrated 1e-9
    ! is unmeetable by ANY correct fp32 run on the stats/pressure rows.
    tol    = merge(1.0e-9_real64, 1.0e-8_real64, tp == dp)   ! fields+mean (sp obs 3.2e-10)
    tol_st = merge(1.0e-9_real64, 1.0e-7_real64, tp == dp)   ! stats       (sp obs 1.8e-9)
    tol_p  = merge(1.0e-9_real64, 1.0e-6_real64, tp == dp)   ! pressure    (sp obs 6.0e-8)
    call print_banner('VALIDATION (distributed step vs single-GPU oracle)', validate_steps, .true.)

    allocate(he0(0:n,nkx,nz), hv0(0:n,nkx,nz))
    allocate(eF(0:n,nkx,nz), vF(0:n,nkx,nz), eS(0:n,bnkx,nz), vS(0:n,bnkx,nz))
    allocate(pF(0:n,nkx,nz), pS(0:n,bnkx,nz))
    if (.not. turbulent) then
       he0 = (0.0_dp,0.0_dp);  hv0 = (0.0_dp,0.0_dp);  hu00 = (0.0_dp,0.0_dp);  hw00 = (0.0_dp,0.0_dp)
    else
       call build_ic(he0, hv0, hu00, hw00)
    end if

    call run_full(he0, hv0, hu00, hw00, muF, mvF, mwF, uuF, vvF, wwF, uvF, retF, eF, vF, u0F, w0F, pF)
    ok_e = .true.;  ok_v = .true.;  ok_m = .true.;  ok_p = .true.
    u0S = (0.0_dp,0.0_dp);  w0S = (0.0_dp,0.0_dp)
    ediff = 0.0_real64;  vdiff = 0.0_real64;  pdiff = 0.0_real64
    if (bnkx > 0) then
       call run_slab(he0, hv0, hu00, hw00, muS, mvS, mwS, uuS, vvS, wwS, uvS, retS, eS, vS, u0S, w0S, pS)
       ok_e = close_c(eS, eF(:, kxlo:kxhi, :), tol)
       ok_v = close_c(vS, vF(:, kxlo:kxhi, :), tol)
       ediff = real(maxval(abs(eS - eF(:, kxlo:kxhi, :))), real64)   ! ACTUAL field diffs
       vdiff = real(maxval(abs(vS - vF(:, kxlo:kxhi, :))), real64)   ! (0 = trajectories locked)
       ! diagnostic pressure: distributed-slab p_hat == single-GPU oracle p_hat,
       ! per kx-slab, bit-for-bit (the solve is mode-local -> this re-validates the
       ! par<->src transforms on the pressure path). Only when snap_pressure on.
       if (snap_pressure) then
          ok_p  = close_c(pS, pF(:, kxlo:kxhi, :), tol_p)
          pdiff = real(maxval(abs(pS - pF(:, kxlo:kxhi, :))), real64)
       end if
    end if
    call par_reduce_max(ediff);  call par_reduce_max(vdiff);  call par_reduce_max(pdiff)
    if (d%owns_mode00) ok_m = close_c1(u0S, u0F, tol) .and. close_c1(w0S, w0F, tol)

    prof_margin = 0.0_real64
    do j = 0, n
       prof_margin = max(prof_margin, &
            real(abs(muS(j)-muF(j)),real64), real(abs(mvS(j)-mvF(j)),real64), &
            real(abs(mwS(j)-mwF(j)),real64), real(abs(uuS(j)-uuF(j)),real64), &
            real(abs(vvS(j)-vvF(j)),real64), real(abs(wwS(j)-wwF(j)),real64), &
            real(abs(uvS(j)-uvF(j)),real64))
    end do
    ok_st = (prof_margin <= tol_st) .and. (prof_margin == prof_margin)
    retau_margin = 0.0_real64
    if (d%owns_mode00) then           ! Re_tau is from the (0,0) column -> owner only
       retau_margin = real(abs(retS - retF), real64)
       ok_st = ok_st .and. (retau_margin <= tol_st)
    end if

    call par_reduce_and(ok_e);  call par_reduce_and(ok_v)
    call par_reduce_and(ok_m);  call par_reduce_and(ok_st);  call par_reduce_and(ok_p)
    all_ok = ok_e .and. ok_v .and. ok_m .and. ok_st .and. ok_p

    if (par_rank == 0) then
       print '(a)', '   --------------------------------------------------------------------------'
       print '(a,i0,3(a,es8.1))', '   validate_steps = ', validate_steps, &
            '   tol(field) = ', tol, '  tol(stats) = ', tol_st, '  tol(press) = ', tol_p
       print '(a,es10.2,a,es10.2)', '   max|eta_dist-ref|          = ', ediff, &
            '   max|v_dist-ref| = ', vdiff
       print '(a,es10.2,a,f8.4,a)', '   max|stat_dist - stat_ref|  = ', prof_margin, &
            '   (Re_tau=', retF, ')'
       print '(a,es10.2)', '   |Re_tau_dist - Re_tau_ref| = ', retau_margin
       if (snap_pressure) print '(a,es10.2)', '   max|p_dist-ref|            = ', pdiff
       call report('eta:  distributed step_full == single-GPU oracle (per kx-slab)', ok_e)
       call report('v:    distributed step_full == single-GPU oracle (per kx-slab)', ok_v)
       call report('mean: u0,w0 == single-GPU oracle (on the kx=0 owner)',           ok_m)
       call report('STATS: distributed Reynolds stresses + Re_tau == single-GPU',    ok_st)
       if (snap_pressure) &
          call report('PRESS: distributed pressure p_hat == single-GPU oracle (per kx-slab)', ok_p)
       if (all_ok) then; print '(a)', 'ALL TESTS PASSED'; else; print '(a)', 'SOME TESTS FAILED'; end if
    end if

    deallocate(he0, hv0, eF, vF, eS, vS, pF, pS)
    if (.not. all_ok) then; call par_finalize();  error stop 1; end if
  end subroutine run_validate

  !=============================================================================
  ! Shared helpers (program-contained; used by both modes).
  !=============================================================================

  ! Rank-0 case banner: the run's identity at a glance -- BUILD PRECISION (so a
  ! log can never be mis-attributed), parallel decomposition, grid/box/physics,
  ! resolution, time plan, IC, output. (Neko-inspired; kept ASCII for grep/logs.)
  subroutine print_banner(mode, nsteps_eff, is_val)
    character(*), intent(in) :: mode
    integer(ip),  intent(in) :: nsteps_eff
    logical,      intent(in) :: is_val
    character(len=4) :: stp, swp
    real(dp) :: nu, dx, dz, dymin, dymax, tend
    integer(ip) :: jb
    if (par_rank /= 0) return
    stp = merge('fp32', 'fp64', tp == sp)
    swp = merge('fp32', 'fp64', wp == sp)
    nu  = 1.0_dp/Re
    dx  = Lx/real(nx, dp);  dz = Lz/real(nz, dp)
    dymin = huge(1.0_dp);  dymax = 0.0_dp
    do jb = 0, n-1
       dymin = min(dymin, abs(y(jb+1) - y(jb)))
       dymax = max(dymax, abs(y(jb+1) - y(jb)))
    end do
    tend = real(nsteps_eff, dp)*dt
    ! the family banner (matches sister Pola's style)
    write(*,'(a)') ' ==============================================================================='
    write(*,'(a)') ' ====                                                                       ===='
    write(*,'(a)') ' ====          V  E  T  A      S  U  B  L  I  M  E  S  O  N                 ===='
    write(*,'(a)') ' ====                                                                       ===='
    write(*,'(a)') ' ====      GPU-resident spectral DNS of turbulent channel flow              ===='
    write(*,'(a)') ' ====      Fourier x/z  x  Shen-Cheb / FD-y  |  kx-slabs + NCCL             ===='
    write(*,'(a)') ' ====      (Kim-Moin-Moser lineage; sister of Pola Sublimeson)              ===='
    write(*,'(a)') ' ====                                                                       ===='
    write(*,'(a)') ' ==============================================================================='
    block
      character(len=64) :: jid
      integer :: jst
      call get_environment_variable('SLURM_JOB_ID', jid, status=jst)
      if (jst /= 0) jid = '-'
      write(*,'(6a)') '   build       : ', VETA_BUILD_HASH, '  (', VETA_BUILD_DATE, &
                      ')   job ', trim(jid)
    end block
    write(*,'(a,a)')   '   deck        : ', trim(infile)
    write(*,'(a,a)')   '   run mode    : ', trim(mode)
    write(*,'(6a)')    '   precision   : transform(tp)=', stp, '   field(wp)=', swp, '   solve(dp)=', 'fp64'
    write(*,'(a,i0,a)')'   MPI / GPU   : ', par_nproc, ' rank(s), 1 GPU/rank, 1-D kx-slab (Layout B) + NCCL transpose'
    write(*,'(a,i0,a,i0,a,i0,a,i0)') '   decomp      : kx ', nkx, ' modes (this rank ', bnkx, &
         ', kx0=', kxlo-1, ') ; dealiased-z slab ', anz
    write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0,a,i0,a,i0)') '   grid        : n=', n, '  nx=', nx, '  nz=', nz, &
         '   ny=', ny, '  nkx=', nkx, '  nxd=', nxd, '  nzd=', nzd
    write(*,'(a,f9.4,a,f9.4,a,f7.3,a,f7.3,a)') '   box         : Lx=', Lx, '  Ly=2.0000  Lz=', Lz, &
         '   (', Lx/acos(-1.0_dp), 'pi x 2 x ', Lz/acos(-1.0_dp), 'pi)'   ! label from the deck, not hardcoded
    write(*,'(a,f8.1,a,es11.4,a,f7.4)') '   physics     : Re=', Re, ' (bulk)  nu=', nu, &
         '  drive=CFR  U_b=', Qtar
    write(*,'(a,es10.3,a,es10.3,a,es9.2,a,es9.2,a)') '   resolution  : dx=', dx, '  dz=', dz, &
         '  dy=', dymin, '(wall)..', dymax, '(mid)'
    ! under variable dt the step count is a BACKSTOP and steps*dt is NOT
    ! a time, so it is not printed as a 't_end='. The
    ! real terminator prints on the 'variable dt :' line below.
    if (variable_dt) then
       write(*,'(a,es10.3,a,i0,a,f7.4)') '   time        : dt=', dt, &
            ' (var-dt)  step cap=', nsteps_eff, '  CFLmax=', cfl_max
    else
       write(*,'(a,es10.3,a,i0,a,f9.2,a,f7.4)') '   time        : dt=', dt, &
            ' (fixed) steps=', nsteps_eff, '  t_end=', tend, '  CFLmax=', cfl_max
    end if
    if (variable_dt .and. .not. is_val) then
       write(*,'(a,f5.2,a,es9.2,a,es9.2,a,f9.3,a)') '   variable dt : target CFL=', cfl_target, &
            '  dt in [', dt_min, ',', dt_max, ']  run to t_end=', t_end, &
            '  (exact-time output landing)'
       write(*,'(a,f8.4,a,f8.4,a,f8.4)') '                 time-based output: snap_dt=', snap_dt, &
            '  stats_dt=', stats_dt, '  chk_dt=', chk_dt
    end if
    ! print the EFFECTIVE stop time (limit minus margin): e.g.
    ! max_wall_hours=0.3 against the 15-min default margin gives a 3-minute
    ! run -- correct behavior, but invisible without this line.
    if (max_wall_hours > 0.0_dp .and. .not. is_val) &
       write(*,'(a,f7.3,a,f6.1,a,f7.3,a)') '   wall limit  : ', max_wall_hours, &
            ' h minus ', wall_margin_min, ' min write margin -> graceful stop at ', &
            max(max_wall_hours - wall_margin_min/60.0_dp, 0.0_dp), ' h elapsed'
    if (turbulent) then
       write(*,'(a,f6.3,a,i0,a,i0)') '   IC          : turbulent  eps=', eps, '  kcut=', kcut, '  seed=', pseed
    else
       write(*,'(a)')                '   IC          : laminar (CFR builds the parabola from rest)'
    end if
    if (u_galilean /= 0.0_dp) &
       write(*,'(a,f8.4,a)') '   Galilean    : advect in frame U_c=', u_galilean, &
            '  (CFL uses |u-U_c|; state+output stay lab-frame)'
    ! Banner lines reflect the ACTIVE trigger: variable dt fires on time marks
    ! (snap_dt/stats_dt), fixed dt on step cadences (snap_every/stats_every) --
    ! otherwise the banner could say 'snapshots: OFF' while a time-mark snapshot fires.
    if (is_val) then
       write(*,'(a)') '   validate    : evolve both paths, diff eta/v/mean/stresses/Re_tau (bit-compare)'
    else if (turbulent .and. variable_dt .and. stats_dt > 0.0_dp) then
       if (snap_pressure) then
          write(*,'(a,f0.4,a,f0.3,a,a)') '   stats       : U,Rij,<p>,<p''^2>,dp/dx every stats_dt=', &
               stats_dt, ' t.u. from t=', stats_start_t, '  ->  ', trim(stats_file)
       else
          write(*,'(a,f0.4,a,f0.3,a,a)') '   stats       : U,Rij,dp/dx every stats_dt=', &
               stats_dt, ' t.u. from t=', stats_start_t, '  ->  ', trim(stats_file)
       end if
    else if (turbulent .and. .not. variable_dt .and. stats_every > 0) then
       if (snap_pressure) then
          write(*,'(a,i0,a,i0,a,a)') '   stats       : U,Rij,<p>,<p''^2>,dp/dx every ', &
               stats_every, ' from step ', stats_start, '  ->  ', trim(stats_file)
       else
          write(*,'(a,i0,a,i0,a,a)') '   stats       : U,Rij,dp/dx every ', &
               stats_every, ' from step ', stats_start, '  ->  ', trim(stats_file)
       end if
    else if (turbulent) then
       write(*,'(a)') '   stats       : OFF (stats_every/stats_dt <= 0)'
    end if
    if (.not. is_val) then
       if (variable_dt .and. snap_dt > 0.0_dp) then
          write(*,'(a,a,a,f0.4,a,f0.3,a,a,a,a)') '   snapshots   : ', &
               merge('u,v,w,p', 'u,v,w  ', snap_pressure), ' every snap_dt=', snap_dt, &
               ' t.u. from t=', snap_start, '  (', merge('fp32', 'fp64', snap_fp32), ')  ->  ', &
               trim(snap_file)
          if (snap_pressure) write(*,'(a)') &
               '                 (p = diagnostic Neumann-Poisson pressure; dealiased recovery)'
       else if (.not. variable_dt .and. snap_every > 0) then
          write(*,'(a,a,a,i0,a,f0.3,a,a,a,a)') '   snapshots   : ', &
               merge('u,v,w,p', 'u,v,w  ', snap_pressure), ' every ', snap_every, &
               ' steps from t=', snap_start, '  (', merge('fp32', 'fp64', snap_fp32), ')  ->  ', &
               trim(snap_file)
          if (snap_pressure) write(*,'(a)') &
               '                 (p = diagnostic Neumann-Poisson pressure; dealiased recovery)'
       else
          write(*,'(a)') '   snapshots   : OFF (snap_every/snap_dt <= 0)'
       end if
    end if
    write(*,'(a)') ' ==============================================================================='
    write(*,'(a,a,a,a)') '   y-basis = ', yop_backend_tag(), &
                         ' / y-grid = ', yop_ygrid_tag()
    if (fd_mode) write(*,'(a,a,a,i0)') '   ygrid file = ', trim(ygrid_file), &
                         ',  kl = ', fd_kl
  end subroutine print_banner

  ! Insert the build-precision tag (_dp/_sp) before a filename's extension, so a
  ! deck run at both precisions writes DISTINCT stats files (no overwrite).
  function tp_tagged(fname) result(out)
    character(*), intent(in) :: fname
    character(len=256) :: out
    character(len=3)   :: tag
    integer(ip) :: idot
    tag  = merge('_sp', '_dp', tp == sp)
    idot = index(trim(fname), '.', back=.true.)
    if (idot > 0) then
       out = fname(1:idot-1)//tag//trim(fname(idot:))
    else
       out = trim(fname)//tag
    end if
  end function tp_tagged

  ! Index k of the first output mark (mark = start + k*interval) strictly after the
  ! current time t, for a schedule beginning at `start`, repeating every `interval`
  ! (convective time units). On a fresh start (t < start) k=0 (first mark = start);
  ! after a restart it advances to the next unfired mark. A tiny bias avoids
  ! re-firing when t sits exactly on a mark. The caller forms the mark by MULTIPLY
  ! (start + k*interval), never by accumulation, so mark positions cannot drift.
  pure function next_mark_index(t, start, interval) result(k)
    real(dp), intent(in) :: t, start, interval
    integer(ip) :: k
    if (interval <= 0.0_dp) then
       k = 0                                          ! disabled schedule (gated off by caller)
    else if (t < start - 1.0e-12_dp) then
       k = 0
    else
       k = floor((t - start)/interval + 1.0e-9_dp) + 1
    end if
  end function next_mark_index

  ! Build a step-keyed, time-tagged, precision-tagged checkpoint name from the base
  ! (chk_file): e.g. base "chk.bin" -> "chk_dp_step00001000_t12.300.bin". The
  ! zero-padded step is the SORT KEY (monotonic, restart-stable); the _t tag is
  ! human-friendly decoration. The _dp/_sp tag keeps precisions from colliding.
  function chk_filename(base, step, tnow) result(out)
    character(*), intent(in) :: base
    integer(ip),  intent(in) :: step
    real(dp),     intent(in) :: tnow
    character(len=256) :: out
    character(len=3)   :: tag
    character(len=8)   :: sstep
    character(len=24)  :: stime
    integer(ip) :: idot
    tag = merge('_sp', '_dp', tp == sp)
    write(sstep,'(i8.8)') step
    write(stime,'(f0.3)') tnow                       ! minimal-width, 3 decimals (e.g. 12.300)
    idot = index(trim(base), '.', back=.true.)
    if (idot > 0) then
       out = base(1:idot-1)//tag//'_step'//sstep//'_t'//trim(stime)//trim(base(idot:))
    else
       out = trim(base)//tag//'_step'//sstep//'_t'//trim(stime)
    end if
  end function chk_filename

  ! Rank-0 init progress: announce a setup stage (Neko-inspired UX logging).
  subroutine init_stage(msg)
    character(*), intent(in) :: msg
    if (par_rank == 0) write(*,'(a,a)') '   [init] ', msg
  end subroutine init_stage

  ! Rank-0 init progress: elapsed wall for the stage just announced by init_stage.
  subroutine init_done(c0, c1, rate)
    integer(8), intent(in) :: c0, c1, rate
    if (par_rank == 0) write(*,'(a,f8.2,a)') '   [init]        done in ', &
         real(c1-c0,dp)/real(rate,dp), ' s'
  end subroutine init_done

  ! Checked end-of-stage device sync: EVERY rank verifies its CUDA context here and
  ! names rank + node + stage on failure. A context poisoned during init
  ! ('unspecified launch failure') otherwise
  ! surfaces only at the next NCCL call, which then takes the blame for it.
  subroutine init_sync_check(stage)
    character(*), intent(in) :: stage
    integer :: ierr_
    ierr_ = cudaDeviceSynchronize()
    if (ierr_ /= cudaSuccess) then
       write(*,'(a,i0,a,a,a,a,a,a)') ' ** FATAL: CUDA error on rank ', par_rank, &
            ' @ ', trim(par_hostname), ' after init stage [', trim(stage), ']: ', &
            trim(cudaGetErrorString(ierr_))
       error stop 1
    end if
  end subroutine init_sync_check

  ! one timing-summary line: "label   <s> s  (<pct> %)".
  subroutine tline(label, tot_ms, wall)
    character(*), intent(in) :: label
    real(dp),     intent(in) :: tot_ms, wall
    write(*,'(a,a,f10.2,a,f6.1,a)') '   ', label, tot_ms/1.0e3_dp, ' s  (', &
         100.0_dp*(tot_ms/1.0e3_dp)/max(wall,1.0e-9_dp), ' %)'
  end subroutine tline

  ! Build the full turbulent IC once on host (identical on every rank: fixed seed).
  ! slab-local turbulent IC: each rank fills ONLY its kx-slab on the host
  ! (counter-based RNG -> bit-identical to a full-grid fill of the same seed),
  ! the intensity normalisation is one global scalar sum, and the mean
  ! coefficients go back to the caller's shared owns_mode00 assignment block.
  ! Replaces the replicated full-grid build for PRODUCTION fresh starts (2 host
  ! + 2 device full-grid arrays = OOM for any >~1 G-pt grid).
  subroutine build_ic_slab(hu0, hw0)
    complex(dp), intent(out) :: hu0(0:n), hw0(0:n)
    complex(dp), allocatable :: hv(:,:,:), he(:,:,:)
    real(dp) :: E2, qrms, scale
    allocate(hv(0:n,bnkx,nz), he(0:n,bnkx,nz))
    if (fd_mode) then
       call ic_fill_slab_fd(n, nkx, nz, d%b_kx0, bnkx, kxh, kzh, pseed, kcut, &
                            yfile, wfile, fd_kl, d1f, hv, he, E2)
    else
       call ic_fill_slab(n, nkx, nz, d%b_kx0, bnkx, kxh, kzh, pseed, kcut, hv, he, E2)
    end if
    call par_reduce_sum(E2)                            ! GLOBAL Parseval sum
    qrms = sqrt(max(E2, 0.0_dp)/2.0_dp)
    if (qrms > 0.0_dp) then
       scale = eps/qrms
       if (par_rank == 0) write(*,'(a,es10.3,a,es10.3)') &
          '   [init] raw perturbation q_rms = ', qrms, '  -> normalised to eps = ', eps
    else
       scale = 0.0_dp
       if (par_rank == 0) write(*,'(a)') &
          '   [init] WARNING: empty perturbation band (check kcut); state unperturbed'
    end if
    hv = scale*hv;  he = scale*he
    v = hv;  eta = he                                  ! H2D, slab extents
    if (fd_mode) then
       call ic_mean_u0_fd(n, Qtar, yfile, hu0)
    else
       call ic_mean_u0(n, Qtar, hu0)
    end if
    hw0 = (0.0_dp,0.0_dp)
    deallocate(hv, he)
  end subroutine build_ic_slab

  ! Full-grid IC build -- kept for the --validate rung ONLY (the single-GPU
  ! oracle needs the whole field; validate grids are small by design).
  subroutine build_ic(he, hv, hu0, hw0)
    complex(dp), intent(out) :: he(0:n,nkx,nz), hv(0:n,nkx,nz), hu0(0:n), hw0(0:n)
    complex(wp), device, allocatable :: vd(:,:,:), ed(:,:,:), u0d(:), w0d(:)
    allocate(vd(0:n,nkx,nz), ed(0:n,nkx,nz), u0d(0:n), w0d(0:n))
    call init_turbulent(n, nkx, nz, kxh, kzh, Qtar, eps, pseed, kcut, vd, ed, u0d, w0d)
    istat = cudaDeviceSynchronize()
    hv = vd;  he = ed;  hu0 = u0d;  hw0 = w0d
    deallocate(vd, ed, u0d, w0d)
  end subroutine build_ic

  ! complex(dp) -> complex(tp) elementwise demote (transform input must be tp).
  subroutine demote(m, nxl, nzl, src, dst)
    integer(ip), intent(in) :: m, nxl, nzl
    complex(dp), device :: src(0:m,nxl,nzl)
    complex(tp), device :: dst(0:m,nxl,nzl)
    integer(ip) :: i, jj, kk2
    !$cuf kernel do(3) <<<*,*>>>
    do kk2 = 1, nzl;  do jj = 1, nxl;  do i = 0, m
       dst(i,jj,kk2) = src(i,jj,kk2)
    end do; end do; end do
  end subroutine demote

  ! single-GPU oracle: evolve the full IC validate_steps, then one stats sample.
  subroutine run_full(he0, hv0, hu00, hw00, mu, mv, mw, uu, vv, ww, uv, ret, eout, vout, u0out, w0out, pout)
    complex(dp), intent(in)  :: he0(0:n,nkx,nz), hv0(0:n,nkx,nz), hu00(0:n), hw00(0:n)
    real(dp),    intent(out) :: mu(0:n), mv(0:n), mw(0:n), uu(0:n), vv(0:n), ww(0:n), uv(0:n), ret
    complex(dp), intent(out) :: eout(0:n,nkx,nz), vout(0:n,nkx,nz), u0out(0:n), w0out(0:n)
    complex(dp), intent(out) :: pout(0:n,nkx,nz)              ! diagnostic pressure p_hat (oracle)
    type(nl_workspace) :: ws
    type(s3d_plans)    :: orpl
    type(stats_t)      :: sa
    real(dp),    device, allocatable :: lkx(:), lkz(:)
    type(vsolve_ops_t) :: lvo               ! local v-solve apparatus
    real(tp),    device, allocatable :: lpu(:,:,:), lpv(:,:,:), lpw(:,:,:)
    complex(dp), device, allocatable :: leta(:,:,:), lv(:,:,:), lu0(:), lw0(:), lNvp(:,:,:), lNep(:,:,:)
    complex(tp), device, allocatable :: luh(:,:,:), lwh(:,:,:), ldvt(:,:,:), lvc(:,:,:), lWf(:,:,:)
    ! pressure scratch (oracle, full nkx): influence-matrix setup + recovery buffers
    real(dp),    device, allocatable :: lppd(:,:,:), lpmd(:,:,:), lCid(:,:,:,:)
    real(wp),    device, allocatable :: lkxw(:), lkzw(:)
    complex(dp), device, allocatable :: lHxd(:,:,:), lHyd(:,:,:), lHzd(:,:,:), lPi(:,:,:)
    real(tp),    device, allocatable :: lqd(:,:,:)
    complex(tp), device, allocatable :: lqhat(:,:,:), lphat(:,:,:)
    complex(dp) :: u0h(0:n)
    integer(ip) :: it
    call nl_setup(ws, n+1, nx, nz, kxh, kzh)
    ws%ugal = u_galilean                          ! [Galilean] same frame as the slab path (--validate parity)
    call s3d_setup_da_r2c(orpl, ny, nx, nz)
    call stats_init(sa, n, nxd, nzd)
    allocate(lkx(nkx), lkz(nz))
    call vsolve_ops_alloc(lvo, n, nkx, nz)
    allocate(leta(0:n,nkx,nz), lv(0:n,nkx,nz), lu0(0:n), lw0(0:n), lNvp(0:n,nkx,nz), lNep(0:n,nkx,nz))
    allocate(luh(0:n,nkx,nz), lwh(0:n,nkx,nz), ldvt(0:n,nkx,nz), lvc(0:n,nkx,nz))
    allocate(lWf(ny,nkxd,nzd), lpu(ny,nxd,nzd), lpv(ny,nxd,nzd), lpw(ny,nxd,nzd))
    lkx = kxh;  lkz = kzh
    bops%sd = bops%sdh;  bops%st = bops%sth;  bops%bd = bops%bdh
    call stepper_ev_setup_gpu(n, nkx, nz, dt, Re, kxh, kzh, lvo%vm, lvo%vp, lvo%Cinv)
    leta = he0;  lv = hv0;  lu0 = hu00;  lw0 = hw00
    lNvp = (0.0_dp,0.0_dp);  lNep = (0.0_dp,0.0_dp)
    do it = 1, validate_steps
       call step_full(ws, n, nkx, nz, dt, Re, lkx, lkz, bops, lvo, mops, &
                      Qtar, lu0, lw0, leta, lv, lNvp, lNep)
    end do
    call recover_uw(ny, nkx, nz, lkx, lkz, lv, leta, lu0, lw0, luh, lwh, ldvt)
    call demote(n, nkx, nz, lv, lvc)
    call to_physical_da_r2c(orpl, ny, nx, nz, nxd, nzd, luh, lpu, lWf)
    call to_physical_da_r2c(orpl, ny, nx, nz, nxd, nzd, lvc, lpv, lWf)
    call to_physical_da_r2c(orpl, ny, nx, nz, nxd, nzd, lwh, lpw, lWf)
    call stats_accumulate(sa, n, nxd, nzd, lpu, lpv, lpw)
    call stats_finalize(sa, n, mu, mv, mw, uu, vv, ww, uv)
    istat = cudaDeviceSynchronize()
    u0h = lu0;  ret = re_tau(n, Re, real(u0h, dp))
    eout = leta;  vout = lv;  u0out = lu0;  w0out = lw0

    ! diagnostic pressure on the SINGLE-GPU oracle (full nkx), via the SAME shared
    ! core (pres_assemble_solve) the slab + driver use; only the transforms (src
    ! to_*_da_r2c, serial nl_eval_H) are the single-GPU ones. -> pout = p_hat.
    pout = (0.0_dp, 0.0_dp)
    if (snap_pressure) then
       allocate(lppd(0:n,nkx,nz), lpmd(0:n,nkx,nz), lCid(2,2,nkx,nz), lkxw(nkx), lkzw(nz))
       allocate(lHxd(0:n,nkx,nz), lHyd(0:n,nkx,nz), lHzd(0:n,nkx,nz), lPi(0:n,nkx,nz))
       allocate(lqd(ny,nxd,nzd), lqhat(ny,nkx,nz), lphat(ny,nkx,nz))
       lkxw = real(kxh,wp);  lkzw = real(kzh,wp)
       call pres_gpu_setup_host(n, nkx, nz, kxh, kzh, lppd, lpmd, lCid)
       call nl_eval_H(ws, lv, leta, lu0, lw0)               ! ws%Hx/Hy/Hz, ws%pu/pv/pw
       call pres_promote_c(n, nkx, nz, ws%Hx, lHxd)
       call pres_promote_c(n, nkx, nz, ws%Hy, lHyd)
       call pres_promote_c(n, nkx, nz, ws%Hz, lHzd)
       call pres_assemble_solve(n, nkx, nz, nx, 0_ip, Re, lkx, lkz, lkxw, lkzw, &
                                bops%sd, bops%st, bops%bd, bops%boff, &
                                lppd, lpmd, lCid, lHxd, lHyd, lHzd, lv, lPi)
       call pres_half_usq(ny, nxd, nzd, 1.0_tp/real(nxd*nzd,tp), ws%pu, ws%pv, ws%pw, lqd)
       call to_spectral_da_r2c(ws%plans, ny, nx, nz, nxd, nzd, lqd, lWf, lqhat)
       call pres_combine_p(n, nkx, nz, lPi, lqhat, lphat)   ! p_hat (tp, device)
       call pres_promote_c(n, nkx, nz, lphat, lPi)          ! tp->dp on DEVICE (reuse lPi)
       pout = lPi                                            ! D2H to the host out arg
       deallocate(lppd, lpmd, lCid, lkxw, lkzw, lHxd, lHyd, lHzd, lPi, lqd, lqhat, lphat)
    end if

    call s3d_destroy_r2c(orpl);  call nl_destroy(ws);  call stats_destroy(sa)
    call vsolve_ops_free(lvo)
    deallocate(lkx, lkz, leta, lv, lu0, lw0, &
               lNvp, lNep, luh, lwh, ldvt, lvc, lWf, lpu, lpv, lpw)
  end subroutine run_full

  ! distributed: evolve this rank's kx-slab validate_steps, then one stats sample.
  subroutine run_slab(he0, hv0, hu00, hw00, mu, mv, mw, uu, vv, ww, uv, ret, eout, vout, u0out, w0out, pout)
    complex(dp), intent(in)  :: he0(0:n,nkx,nz), hv0(0:n,nkx,nz), hu00(0:n), hw00(0:n)
    real(dp),    intent(out) :: mu(0:n), mv(0:n), mw(0:n), uu(0:n), vv(0:n), ww(0:n), uv(0:n), ret
    complex(dp), intent(out) :: eout(0:n,bnkx,nz), vout(0:n,bnkx,nz), u0out(0:n), w0out(0:n)
    complex(dp), intent(out) :: pout(0:n,bnkx,nz)            ! diagnostic pressure p_hat (slab)
    type(par_nl_workspace) :: ws
    type(stats_t)          :: sa
    real(dp),    device, allocatable :: lkx(:), lkz(:)
    type(vsolve_ops_t) :: lvo               ! local v-solve apparatus
    real(tp),    device, allocatable :: lpu(:,:,:), lpv(:,:,:), lpw(:,:,:)
    complex(dp), device, allocatable :: leta(:,:,:), lv(:,:,:), lu0(:), lw0(:), lNvp(:,:,:), lNep(:,:,:)
    complex(tp), device, allocatable :: luh(:,:,:), lwh(:,:,:), ldvt(:,:,:), lvc(:,:,:)
    complex(tp), device, allocatable :: lBz(:,:,:,:), lAz(:,:,:,:), lAf(:,:,:)
    ! pressure scratch (slab, bnkx): influence-matrix setup + recovery buffers
    real(dp),    device, allocatable :: lppd(:,:,:), lpmd(:,:,:), lCid(:,:,:,:)
    real(wp),    device, allocatable :: lkxw(:), lkzw(:)
    complex(dp), device, allocatable :: lHxd(:,:,:), lHyd(:,:,:), lHzd(:,:,:), lPi(:,:,:)
    real(tp),    device, allocatable :: lqd(:,:,:)
    complex(tp), device, allocatable :: lqhat(:,:,:), lphat(:,:,:)
    complex(dp) :: u0h(0:n)
    integer(ip) :: it
    call par_nl_setup(ws, d, n+1, nx, nz, kxh, kzh)
    ws%ugal = u_galilean                          ! [Galilean] same frame as the oracle path (--validate parity)
    call stats_init(sa, n, nxd, nzd)
    allocate(lkx(bnkx), lkz(nz))
    call vsolve_ops_alloc(lvo, n, bnkx, nz)
    allocate(leta(0:n,bnkx,nz), lv(0:n,bnkx,nz), lu0(0:n), lw0(0:n), lNvp(0:n,bnkx,nz), lNep(0:n,bnkx,nz))
    allocate(luh(0:n,bnkx,nz), lwh(0:n,bnkx,nz), ldvt(0:n,bnkx,nz), lvc(0:n,bnkx,nz))
    allocate(lpu(ny,nxd,anz), lpv(ny,nxd,anz), lpw(ny,nxd,anz))
    allocate(lBz(ny,bnkx,nzd,1), lAz(ny,nkx,anz,1), lAf(ny,nkxd,anz))
    lkx = kxh(kxlo:kxhi);  lkz = kzh
    bops%sd = bops%sdh;  bops%st = bops%sth;  bops%bd = bops%bdh
    call stepper_ev_setup_gpu(n, bnkx, nz, dt, Re, kxh(kxlo:kxhi), kzh, lvo%vm, lvo%vp, lvo%Cinv)
    leta = he0(:,kxlo:kxhi,:);  lv = hv0(:,kxlo:kxhi,:)
    if (d%owns_mode00) then
       lu0 = hu00;  lw0 = hw00
    else
       lu0 = (0.0_dp,0.0_dp);  lw0 = (0.0_dp,0.0_dp)
    end if
    lNvp = (0.0_dp,0.0_dp);  lNep = (0.0_dp,0.0_dp)
    do it = 1, validate_steps
       call par_step_full(ws, n, bnkx, nz, dt, Re, lkx, lkz, bops, lvo, mops, &
                          Qtar, lu0, lw0, leta, lv, lNvp, lNep)
    end do
    call recover_uw(ny, bnkx, nz, lkx, lkz, lv, leta, lu0, lw0, luh, lwh, ldvt)
    call demote(n, bnkx, nz, lv, lvc)
    call par_to_physical_da_r2c(ws%plans, d, ny, nx, nz, nxd, nzd, bnkx, anz, luh, lBz, lAz, lAf, lpu)
    call par_to_physical_da_r2c(ws%plans, d, ny, nx, nz, nxd, nzd, bnkx, anz, lvc, lBz, lAz, lAf, lpv)
    call par_to_physical_da_r2c(ws%plans, d, ny, nx, nz, nxd, nzd, bnkx, anz, lwh, lBz, lAz, lAf, lpw)
    call stats_accumulate(sa, n, nxd, anz, lpu, lpv, lpw)
    call par_stats_finalize(sa, n, mu, mv, mw, uu, vv, ww, uv)
    istat = cudaDeviceSynchronize()
    u0h = lu0;  ret = re_tau(n, Re, real(u0h, dp))
    eout = leta;  vout = lv;  u0out = lu0;  w0out = lw0

    ! diagnostic pressure on the DISTRIBUTED slab (bnkx), via the SAME shared core
    ! (pres_assemble_solve); only the transforms (par_*_da_r2c, par_nl_eval_H) are
    ! the distributed ones. -> pout = p_hat (compared vs run_full in run_validate).
    pout = (0.0_dp, 0.0_dp)
    if (snap_pressure) then
       allocate(lppd(0:n,bnkx,nz), lpmd(0:n,bnkx,nz), lCid(2,2,bnkx,nz), lkxw(bnkx), lkzw(nz))
       allocate(lHxd(0:n,bnkx,nz), lHyd(0:n,bnkx,nz), lHzd(0:n,bnkx,nz), lPi(0:n,bnkx,nz))
       allocate(lqd(ny,nxd,anz), lqhat(ny,bnkx,nz), lphat(ny,bnkx,nz))
       lkxw = real(kxh(kxlo:kxhi),wp);  lkzw = real(kzh,wp)
       call pres_gpu_setup_host(n, bnkx, nz, kxh(kxlo:kxhi), kzh, lppd, lpmd, lCid)
       call par_nl_eval_H(ws, lv, leta, lu0, lw0)           ! ws%S3, ws%P6
       call pres_promote_c(n, bnkx, nz, ws%S3(:,:,:,1), lHxd)
       call pres_promote_c(n, bnkx, nz, ws%S3(:,:,:,2), lHyd)
       call pres_promote_c(n, bnkx, nz, ws%S3(:,:,:,3), lHzd)
       call pres_assemble_solve(n, bnkx, nz, nx, d%b_kx0, Re, lkx, lkz, lkxw, lkzw, &
                                bops%sd, bops%st, bops%bd, bops%boff, &
                                lppd, lpmd, lCid, lHxd, lHyd, lHzd, lv, lPi)
       call pres_half_usq(ny, nxd, anz, 1.0_tp/real(nxd*nzd,tp), &
                          ws%P6(:,:,:,2), ws%P6(:,:,:,1), ws%P6(:,:,:,3), lqd)
       call par_to_spectral_da_r2c(ws%plans, d, ny, nx, nz, nxd, nzd, bnkx, anz, lqd, lAf, lAz, lBz, lqhat)
       call pres_combine_p(n, bnkx, nz, lPi, lqhat, lphat)  ! p_hat (tp, device)
       call pres_promote_c(n, bnkx, nz, lphat, lPi)         ! tp->dp on DEVICE (reuse lPi)
       pout = lPi                                           ! D2H to the host out arg
       deallocate(lppd, lpmd, lCid, lkxw, lkzw, lHxd, lHyd, lHzd, lPi, lqd, lqhat, lphat)
    end if

    call par_nl_destroy(ws);  call stats_destroy(sa)
    call vsolve_ops_free(lvo)
    deallocate(lkx, lkz, leta, lv, lu0, lw0, &
               lNvp, lNep, luh, lwh, ldvt, lvc, lpu, lpv, lpw, lBz, lAz, lAf)
  end subroutine run_slab

  logical function close_c(a, b, tolc) result(okk)
    complex(dp),  intent(in) :: a(:,:,:), b(:,:,:)
    real(real64), intent(in) :: tolc
    okk = all(a == a) .and. all(b == b) .and. (real(maxval(abs(a - b)), real64) <= tolc)
  end function close_c

  logical function close_c1(a, b, tolc) result(okk)
    complex(dp),  intent(in) :: a(:), b(:)
    real(real64), intent(in) :: tolc
    okk = all(a == a) .and. all(b == b) .and. (real(maxval(abs(a - b)), real64) <= tolc)
  end function close_c1

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

  !-- finalize stats -> wall units -> profile table (rank 0). Ported from
  !   src/serial/main.F90 write_stats, but takes the ALREADY-FINALIZED (par_stats_finalize)
  !   global profiles; the cheb_forward/deriv diagnostics run on rank 0 (host CPU).
  subroutine write_stats_file(fname, mprof, mvp, mwp, uup, vvp, wwp, uvp, mpp, ppv, dpdx, have_p, nsamp)
    character(*), intent(in) :: fname
    real(dp),     intent(in) :: mprof(0:n), mvp(0:n), mwp(0:n)
    real(dp),     intent(in) :: uup(0:n), vvp(0:n), wwp(0:n), uvp(0:n)
    real(dp),     intent(in) :: mpp(0:n), ppv(0:n)   ! <p>(y), <p'^2>(y)
    real(dp),     intent(in) :: dpdx               ! time-avg drive force = mean dp/dx (CFR beta)
    logical,      intent(in) :: have_p             ! pressure stats present (snap_pressure)
    integer(ip),  intent(in) :: nsamp
    real(dp) :: utau, ut2, yp, s2, ret_w, acc, accp, accv, cb, Ttot, resid, pk_lo, pk_hi, asym, prmsp, pvar
    real(dp) :: ac(0:n), bc(0:n), dUdy(0:n)
    real(dp) :: apc(0:n), dpc(0:n), avc(0:n), dvc(0:n), dpdy(0:n), dvvdy(0:n)
    integer  :: uf, jj, mm
    if (fd_mode) then
       ! FD: profiles ARE point values -- wall shear via the yops
       ! FD branch, derivatives via the banded D1 (have_p is forced off in FD).
       ret_w = re_tau(n, Re, mprof);  utau = ret_w/Re;  ut2 = utau*utau
       call fd_apply_band(n+1, fd_kl, d1f, mprof, dUdy)
       dpdy = 0.0_dp;  dvvdy = 0.0_dp
    else
    call cheb_forward(n, mprof, ac)
    ret_w = re_tau(n, Re, ac);  utau = ret_w/Re;  ut2 = utau*utau
    call cheb_deriv(n, ac, bc)
    dpdy = 0.0_dp;  dvvdy = 0.0_dp
    if (have_p) then
       call cheb_forward(n, mpp, apc);  call cheb_deriv(n, apc, dpc)   ! d<p>/dy coeffs
       call cheb_forward(n, vvp, avc);  call cheb_deriv(n, avc, dvc)   ! d<v'v'>/dy coeffs (balance)
    end if
    do jj = 0, n
       acc = 0.0_dp;  accp = 0.0_dp;  accv = 0.0_dp
       do mm = 0, n
          cb  = cos(real(mm,dp)*PI*real(jj,dp)/real(n,dp))
          acc = acc + bc(mm)*cb
          if (have_p) then; accp = accp + dpc(mm)*cb;  accv = accv + dvc(mm)*cb;  end if
       end do
       dUdy(jj) = acc
       if (have_p) then; dpdy(jj) = accp;  dvvdy(jj) = accv;  end if
    end do
    end if
    resid = 0.0_dp
    do jj = 0, n
       Ttot  = dUdy(jj)/Re - uvp(jj)
       resid = resid + (Ttot + ut2*y(jj))**2
    end do
    resid = sqrt(resid/real(n+1,dp))/max(ut2, tiny(1.0_dp))
    pk_lo = 0.0_dp;  pk_hi = 0.0_dp
    do jj = 0, n
       if (y(jj) < 0.0_dp) pk_lo = max(pk_lo, uup(jj))
       if (y(jj) > 0.0_dp) pk_hi = max(pk_hi, uup(jj))
    end do
    asym = 200.0_dp*abs(pk_lo - pk_hi)/max(pk_lo + pk_hi, tiny(1.0_dp))
    open(newunit=uf, file=trim(fname), status='replace', action='write')
    write(uf,'(a)') '# channel DNS turbulence statistics (x-z-t Reynolds average) [distributed]'
    write(uf,'(a,f12.5,a,f10.4,a,es12.5,a,i0)') &
         '#  Re = ', Re, '   Re_tau = ', ret_w, '   u_tau = ', utau, '   nsamp = ', nsamp
    ! y-basis identity stamp (same tags as checkpoints; absent = cheb-T).
    ! analyze_stats.py decides its d/dy branch from the
    ! y NODES and uses this line for provenance/notes.
    if (fd_mode) then
       write(uf,'(a,a,a,a)') '#  ybasis = ', trim(yop_backend_tag()), &
            '   ygrid_file = ', trim(ygrid_file)
    else
       write(uf,'(a,a)') '#  ybasis = ', trim(yop_backend_tag())
    end if
    write(uf,'(a)') '#  normalisation: WINDOW-MEAN u_tau (wall shear of the accumulated <u>) [F6]'
    write(uf,'(a,f8.3,a,f8.3,a)') '#  convergence: stress-linearity residual = ', &
         100.0_dp*resid, ' %   uu-peak wall asymmetry = ', asym, ' %'
    ! drive force = mean dp/dx (CFR beta, time-avg). Momentum balance: -dp/dx = u_tau^2.
    write(uf,'(a,es13.6,a,es13.6,a)') '#  drive force <dp/dx> = ', dpdx, &
         '   (-dp/dx vs u_tau^2 = ', ut2, ')'
    if (have_p) then
       write(uf,'(a)') '#  pressure: prms+ = sqrt(<p''^2>)/u_tau^2 ; <p> is gauge-dependent (k=0 constant free)'
       write(uf,'(a)') '#  channel mean wall-normal momentum balance: d<p>/dy = -d<v''v''>/dy (the last two cols)'
       write(uf,'(a)') '#  y            y+           U+           uu+          vv+          ww+'// &
                       '          uv+          <u>          <p>          prms+        <p''^2>'// &
                       '       d<p>/dy      -d<v''v''>/dy'
    else
       write(uf,'(a)') '#  y            y+           U+           uu+          vv+          ww+          uv+          <u>'
    end if
    do jj = 0, n
       yp = (1.0_dp - abs(y(jj)))*ret_w
       s2 = merge(ut2, 1.0_dp, utau > 0.0_dp)
       pvar  = max(ppv(jj), 0.0_dp)
       prmsp = merge(sqrt(pvar)/s2, 0.0_dp, utau > 0.0_dp .and. have_p)
       if (utau > 0.0_dp) then
          if (have_p) then
             write(uf,'(13(es12.5,1x))') y(jj), yp, mprof(jj)/utau, uup(jj)/s2, &
                  vvp(jj)/s2, wwp(jj)/s2, uvp(jj)/s2, mprof(jj), mpp(jj), prmsp, ppv(jj), &
                  dpdy(jj), -dvvdy(jj)
          else
             write(uf,'(8(es12.5,1x))') y(jj), yp, mprof(jj)/utau, uup(jj)/s2, &
                  vvp(jj)/s2, wwp(jj)/s2, uvp(jj)/s2, mprof(jj)
          end if
       else
          if (have_p) then
             write(uf,'(13(es12.5,1x))') y(jj), yp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, &
                  mprof(jj), mpp(jj), 0.0_dp, ppv(jj), dpdy(jj), -dvvdy(jj)
          else
             write(uf,'(8(es12.5,1x))') y(jj), yp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, mprof(jj)
          end if
       end if
    end do
    close(uf)
  end subroutine write_stats_file

end program main_par
