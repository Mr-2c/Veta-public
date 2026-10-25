!===============================================================================
! solver_ops_mod.F90 -- SOLVER-STATE AGGREGATES.
!
! Gathers the loose per-run solver apparatus -- the Shen/Helmholtz band
! structure, the CN-shifted v-solve lift/influence precompute, and the CFR mean
! precompute -- into derived types, so that:
!   (a) the per-EQUATION precision tier is a property of the AGGREGATE:
!       vsolve_ops_t carries the SHIFTED-solve artifacts (the sp-storage/
!       dp-accumulate adoption target); band_ops_t +
!       mean_ops_t carry structure used by BOTH shifted and bare-k^2 solves and
!       the k^2=0 mean -- their dp members stay dp. Retiering later = change
!       member kinds HERE + the stepper's leaf calls, not 15 scattered files.
!   (b) the FD-y backend swaps in behind the same types (band_ops_t is
!       exactly "the y-operator identity" a backend must provide).
!   (c) ownership is explicit: ONE alloc/free pair per aggregate instead of
!       per-array allocate lists duplicated across drivers and tests.
!
! The LEAF solvers (helm_band_gpu, vsolve_if_gpu, mean_gpu, stepper_lin setup)
! keep their array interfaces -- the steppers unpack members at the call site.
! PURE REPACKAGING: same arrays, same ops, same order -> --validate bit-exact.
!
! Build (after kinds/types, before stepper_mod):  nvfortran -cuda -c ...
!===============================================================================
module solver_ops_mod
  use kinds, only: dp, ip
  use cudafor
  use fdops_mod,         only: fd_build_bands, fd_wall_weights          ! FD builders (host)
  use cheb_ops_mod,      only: apply_L                                  ! dispatch leaves:
  use spec_helpers_mod,  only: project_shen, shen_to_cheb_c             !   Cheb side
  use helm_band_gpu_mod, only: helm_band_solve_gpu
  use vsolve_if_gpu_mod, only: vif_solve_gpu_ws, vif_gpu_setup_dev_ws
  use fd_helm_gpu_mod,   only: fd_helm_solve_gpu, fd_apply_L_gpu        !   FD side
  use fd_vsolve_gpu_mod, only: fd_vif_solve_gpu_ws, fd_vif_setup_gpu_ws
  use mean_gpu_mod,      only: mean_gpu_ws, mean_stage_gpu, mean_stage_cfr_gpu
  use fd_mean_gpu_mod,   only: fd_mean_stage_gpu, fd_mean_stage_cfr_gpu
  implicit none
  private
  public :: band_ops_t, vsolve_ops_t, mean_ops_t
  public :: band_ops_alloc, band_ops_free
  public :: vsolve_ops_alloc, vsolve_ops_free
  public :: mean_ops_alloc, mean_ops_free
  public :: band_ops_fd_init                       ! FD backend population
  public :: y_apply_L, y_helm_solve, y_vif_solve   ! the dispatch seam
  public :: y_vif_setup                            ! per-alpha setup dispatch
  public :: y_mean_stage, y_mean_stage_cfr         ! (0,0) mean-stage dispatch
  public :: y_vres_max                             ! v-solve residual probe
  public :: y_vres_max_win                         ! FD interior window variant
  public :: y_vres_galerkin                        ! Cheb Galerkin-consistent probe
  integer(ip), parameter, public :: YB_CHEB = 1, YB_FD = 2
  real(dp), parameter :: SO_PI = 3.141592653589793238462643383279_dp  ! = spec_helpers PI

  !-- Shen/Helmholtz band structure (the wall-normal operator identity) --------
  ! sd/st: Shen-Dirichlet stiffness diag/tail; bd: mass band; boff: mass offset.
  ! Device copies feed the GPU band solves; host mirrors feed host-side setup
  ! (mean_flux_setup) and the retained-for-ABI host args.
  ! ybasis selects the backend at the THREE per-step primitive sites
  ! (y_apply_L / y_helm_solve / y_vif_solve below). The YB_CHEB helpers hold
  ! the VERBATIM moved statements, so that path stays byte-identical.
  ! YB_FD members: the fdops D1/D2
  ! bands built from the ygrid file's nodes, the one-sided wall rows, and the
  ! nodes themselves (device + host, host quadrature weights for the CFR mean).
  type :: band_ops_t
     real(dp), device, allocatable :: sd(:), st(:), bd(:)     ! (0:n-2)
     real(dp),         allocatable :: sdh(:), sth(:), bdh(:)  ! host mirrors
     real(dp) :: boff = 0.0_dp
     integer(ip) :: ybasis = YB_CHEB                          ! backend tag
     integer(ip) :: kl = 0                                    ! FD band half-width
     real(dp), device, allocatable :: d1b(:,:), d2b(:,:)      ! (2kl+1, ny) FD bands
     real(dp), device, allocatable :: wlo(:), whi(:), yd(:)   ! (ny) wall rows + nodes
     real(dp), device, allocatable :: whd(:)                  ! (ny) quad weights (device; CFR)
     real(dp),         allocatable :: yh(:), wh(:)            ! host nodes + quad weights
     logical :: have_w = .false.
  end type band_ops_t

  !-- CN-shifted v-solve apparatus (per-substage lift + influence inverse) -----
  ! These are the SHIFTED-solve artifacts: the future sp-stored/
  ! dp-accumulated tier lives here (change these kinds, nothing else moves).
  type :: vsolve_ops_t
     real(dp), device, allocatable :: vm(:,:,:,:)      ! (0:n,nx,nz,3) v^- lift
     real(dp), device, allocatable :: vp(:,:,:,:)      ! (0:n,nx,nz,3) v^+ lift
     real(dp), device, allocatable :: Cinv(:,:,:,:,:)  ! (2,2,nx,nz,3) influence inverse
  end type vsolve_ops_t

  !-- CFR mean apparatus (k^2 = 0 -- stays dp under every retiering plan) ------
  type :: mean_ops_t
     real(dp), allocatable :: Uhm(:,:)                  ! (0:n,3) per-substage mean solve
     real(dp), allocatable :: Ih(:)                    ! (3)     flux integrals
  end type mean_ops_t

contains

  subroutine band_ops_alloc(b, n)
    type(band_ops_t), intent(inout) :: b
    integer(ip),      intent(in)    :: n
    allocate(b%sd(0:n-2), b%st(0:n-2), b%bd(0:n-2))
    allocate(b%sdh(0:n-2), b%sth(0:n-2), b%bdh(0:n-2))
  end subroutine band_ops_alloc

  subroutine band_ops_free(b)
    type(band_ops_t), intent(inout) :: b
    if (allocated(b%sd))  deallocate(b%sd, b%st, b%bd)
    if (allocated(b%sdh)) deallocate(b%sdh, b%sth, b%bdh)
    if (allocated(b%d2b)) deallocate(b%d1b, b%d2b, b%wlo, b%whi, b%yd)
    if (allocated(b%whd)) deallocate(b%whd)
    if (allocated(b%yh))  deallocate(b%yh)
    if (allocated(b%wh))  deallocate(b%wh)
    b%ybasis = YB_CHEB;  b%kl = 0;  b%have_w = .false.
  end subroutine band_ops_free

  !---------------------------------------------------------------------------
  ! band_ops_fd_init -- populate the FD backend from the ygrid file's
  ! nodes (caller has already run ygrid_read: yh is DESCENDING +1..-1, ny=n+1
  ! entries; wh are the file's quadrature weights when have_w). Builds the
  ! D1/D2 bands (interior 2kl+1 stencils, within-band wall taper) and the
  ! full-order one-sided wall rows on the HOST (validated fdops kit), uploads,
  ! and flips the tag. The Cheb members (sd/st/bd) are left untouched -- a
  ! band_ops_t may carry both backends; ybasis picks at the call sites.
  !---------------------------------------------------------------------------
  subroutine band_ops_fd_init(b, n, kl, yh, wh, have_w)
    type(band_ops_t), intent(inout) :: b
    integer(ip),      intent(in)    :: n, kl
    real(dp),         intent(in)    :: yh(n+1), wh(n+1)
    logical,          intent(in)    :: have_w
    real(dp) :: d1bh(2*kl+1, n+1), d2bh(2*kl+1, n+1)
    real(dp) :: wloh(n+1), whih(n+1)
    integer(ip) :: ny
    ny = n + 1
    call fd_build_bands(ny, kl, yh, d1bh, d2bh)
    call fd_wall_weights(ny, kl, yh, wloh, whih)
    if (allocated(b%d2b)) deallocate(b%d1b, b%d2b, b%wlo, b%whi, b%yd)
    allocate(b%d1b(2*kl+1, ny), b%d2b(2*kl+1, ny))
    allocate(b%wlo(ny), b%whi(ny), b%yd(ny))
    b%d1b = d1bh;  b%d2b = d2bh
    b%wlo = wloh;  b%whi = whih;  b%yd = yh
    if (allocated(b%yh)) deallocate(b%yh)
    if (allocated(b%wh)) deallocate(b%wh)
    allocate(b%yh(ny), b%wh(ny))
    b%yh = yh
    b%wh = 0.0_dp
    if (have_w) b%wh = wh          ! wh may be undefined when the file has no w column
    if (allocated(b%whd)) deallocate(b%whd)
    allocate(b%whd(ny))
    b%whd = b%wh                   ! device copy of the file's weights (CFR closure)
    b%kl = kl;  b%have_w = have_w
    b%ybasis = YB_FD
  end subroutine band_ops_fd_init

  !=============================================================================
  ! THE DISPATCH SEAM -- the three per-step y-primitives, selected by
  ! bo%ybasis. The YB_CHEB branches are the VERBATIM statements moved out of
  ! ev_stage/par_ev_stage (same leafs, same order, same arguments -> the Cheb
  ! path stays BIT-IDENTICAL). The YB_FD branches
  ! run the batched FD kernels on the SAME (0:n,nx,nz) state
  ! interpreted as point values -- the Shen<->point converters are identity in
  ! point space, so project_shen/shen_to_cheb simply vanish there.
  ! Kinds are complex(dp) = the stepper state tier (wp==dp in every live build;
  ! a WP=sp tier would retier the aggregate, per the header).
  !=============================================================================

  !-- Lq = (D^2 - k^2) q.  scr is Cheb-side derivative scratch (unused by FD). --
  subroutine y_apply_L(bo, n, nx, nz, kx, kz, q, Lq, scr)
    type(band_ops_t), intent(in) :: bo
    integer(ip),      intent(in) :: n, nx, nz
    real(dp),    device :: kx(nx), kz(nz)
    complex(dp), device :: q(0:n,nx,nz), Lq(0:n,nx,nz), scr(0:n,nx,nz)
    if (bo%ybasis == YB_FD) then
       call fd_apply_L_gpu(n, bo%kl, nx, nz, kx, kz, bo%d2b, q, Lq)
    else
       call apply_L(n+1, nx, nz, kx, kz, q, Lq, scr)
    end if
  end subroutine y_apply_L

  !-- Solve (1 - a L) out = rhs  i.e.  (-D^2 + k^2 + lam) out = lam*rhs,
  !   homogeneous Dirichlet walls; lam = 1/a. CONSUMES rhs (both branches may
  !   destroy it). ftil/sh are the Cheb-side Shen scratch (0:n-2; unused by FD).
  ! ftil/sh are ASSUMED-SIZE: the FD branch never touches them, and under
  ! FD-y the steppers pass 1-element stubs (stepper_ws_alloc(fd=.true.)) --
  ! an explicit-shape dummy would make that association illegal. The Cheb
  ! branch forwards them to explicit-shape internals, legal because Cheb-mode
  ! actuals are always full-size.
  subroutine y_helm_solve(bo, n, nx, nz, lam, kx, kz, rhs, out, ftil, sh)
    type(band_ops_t), intent(in) :: bo
    integer(ip),      intent(in) :: n, nx, nz
    real(dp),         intent(in) :: lam
    real(dp),    device :: kx(nx), kz(nz)
    complex(dp), device :: rhs(0:n,nx,nz), out(0:n,nx,nz)
    complex(dp), device :: ftil(*), sh(*)
    if (bo%ybasis == YB_FD) then
       call so_cscale(n, nx, nz, rhs, lam)                    ! f = lam*rhs (consumed)
       call fd_helm_solve_gpu(n, bo%kl, nx, nz, lam, kx, kz, bo%d2b, rhs, out)
    else
       call project_shen(n, nx, nz, rhs, ftil)
       call so_cscale(n-2, nx, nz, ftil, lam)
       call helm_band_solve_gpu(n, nx, nz, lam, kx, kz, bo%sd, bo%st, bo%bd, bo%boff, &
                                ftil, sh)
       call shen_to_cheb_c(n, nx, nz, sh, out)
    end if
  end subroutine y_helm_solve

  !-- The clamped v-solve (two-Helmholtz + influence closure) for one substage.
  !   vm/vp/Cinv are that substage's precomputes (SAME shapes both backends:
  !   point values are (0:n) like Cheb coeffs). Scratch mapping:
  !     Cheb: ftil,gsh (0:n-2), gc,vsh->via vvsh (0:n-2), vpc (0:n), dlo/dhi
  !     FD:   fsc,gc,vpc (0:n) + dlo/dhi   (ftil/gsh/vsh unused)
  !   fsc is a caller-lent (0:n,nx,nz) spare (the steppers lend s%scr, dead at
  !   this point in the stage). rhsv is NOT modified by either branch.
  subroutine y_vif_solve(bo, n, nx, nz, a, kx, kz, vm, vp, Cinv, rhsv, v, &
                         ftil, gsh, gc, vsh, vpc, dlo, dhi, fsc)
    type(band_ops_t), intent(in) :: bo
    integer(ip),      intent(in) :: n, nx, nz
    real(dp),         intent(in) :: a
    real(dp),    device :: kx(nx), kz(nz)
    real(dp),    device :: vm(0:n,nx,nz), vp(0:n,nx,nz), Cinv(2,2,nx,nz)
    complex(dp), device :: rhsv(0:n,nx,nz), v(0:n,nx,nz)
    ! the three Shen buffers are ASSUMED-SIZE: FD-mode steppers pass
    ! 1-element stubs (see stepper_ws_alloc), and the FD branch never touches
    ! them; the Cheb branch forwards to explicit-shape internals (full-size
    ! actuals there). gc/vpc stay explicit -- BOTH branches use them.
    complex(dp), device :: ftil(*), gsh(*)
    complex(dp), device :: gc(0:n,nx,nz), vsh(*), vpc(0:n,nx,nz)
    complex(dp), device :: dlo(nx,nz), dhi(nx,nz), fsc(0:n,nx,nz)
    if (bo%ybasis == YB_FD) then
       call fd_vif_solve_gpu_ws(n, bo%kl, nx, nz, a, kx, kz, bo%d2b, bo%wlo, bo%whi, &
                                vm, vp, Cinv, rhsv, v, fsc, gc, vpc, dlo, dhi)
    else
       call vif_solve_gpu_ws(n, nx, nz, a, kx, kz, bo%sd, bo%st, bo%bd, bo%boff, &
                             vm, vp, Cinv, rhsv, v, ftil, gsh, gc, vsh, vpc, dlo, dhi)
    end if
  end subroutine y_vif_solve

  !-- per-alpha vif setup: vm/vp/Cinv for one substage. The Cheb branch
  !   is the verbatim vif_gpu_setup_dev_ws call (lent scratch: the SEVEN
  !   slab temps); the FD branch is fd_vif_setup_gpu_ws, which needs only THREE
  !   of them (rhs/gpc/vpc slots; ftil/gsh are (0:n-2) and gmc/vmc go unused).
  !   Every temp is fully written before read in both branches (lending-safe).
  subroutine y_vif_setup(bo, n, nx, nz, kx, kz, alpha, vm, vp, Cinv, &
                         rhs, ftil, gsh, gpc, gmc, vpc, vmc)
    type(band_ops_t), intent(in) :: bo
    integer(ip),      intent(in) :: n, nx, nz
    real(dp),         intent(in) :: alpha
    real(dp),    device :: kx(nx), kz(nz)
    real(dp),    device :: vm(0:n,nx,nz), vp(0:n,nx,nz), Cinv(2,2,nx,nz)
    ! ftil/gsh assumed-size: the FD branch ignores them and FD-mode
    ! callers lend 1-element stubs (main_par's setup path under fd_mode)
    complex(dp), device :: rhs(0:n,nx,nz), ftil(*), gsh(*)
    complex(dp), device :: gpc(0:n,nx,nz), gmc(0:n,nx,nz)
    complex(dp), device :: vpc(0:n,nx,nz), vmc(0:n,nx,nz)
    if (bo%ybasis == YB_FD) then
       call fd_vif_setup_gpu_ws(n, bo%kl, nx, nz, kx, kz, alpha, bo%yd, bo%d2b, &
                                bo%wlo, bo%whi, vm, vp, Cinv, rhs, gpc, vpc)
    else
       call vif_gpu_setup_dev_ws(n, nx, nz, kx, kz, alpha, bo%sd, bo%st, bo%bd, &
                                 bo%boff, vm, vp, Cinv, rhs, ftil, gsh, gpc, gmc, &
                                 vpc, vmc)
    end if
  end subroutine y_vif_setup

  !-- (0,0) mean-stage dispatch. Cheb branches = the VERBATIM
  !   mean_stage_gpu / mean_stage_cfr_gpu calls moved out of the steppers'
  !   mean blocks; FD branches = the composed-from-primitives FD device mean
  !   (fd_mean_gpu_mod), CFR closed with the grid file's weights (bo%whd).
  !   q0/Nq/Nq_prev arrive rank-1 from the steppers and sequence-associate to
  !   (0:n,1,1) exactly as they did at the direct call sites.
  subroutine y_mean_stage(bo, mg, n, ks, dt, Re, beta, q0, Nq, Nq_prev)
    type(band_ops_t),  intent(in)    :: bo
    type(mean_gpu_ws), intent(inout) :: mg
    integer(ip), intent(in) :: n, ks
    real(dp),    intent(in) :: dt, Re, beta
    complex(dp), device :: q0(0:n,1,1), Nq(0:n,1,1), Nq_prev(0:n,1,1)
    if (bo%ybasis == YB_FD) then
       call fd_mean_stage_gpu(mg, n, bo%kl, bo%d2b, ks, dt, Re, beta, q0, Nq, Nq_prev)
    else
       call mean_stage_gpu(mg, n, ks, dt, Re, beta, bo%sd, bo%st, bo%bd, bo%boff, &
                           q0, Nq, Nq_prev)
    end if
  end subroutine y_mean_stage

  subroutine y_mean_stage_cfr(bo, mg, n, ks, dt, Re, Qtar, Uh, Ih, q0, Nq, Nq_prev, beta_out)
    type(band_ops_t),  intent(in)    :: bo
    type(mean_gpu_ws), intent(inout) :: mg
    integer(ip), intent(in) :: n, ks
    real(dp),    intent(in) :: dt, Re, Qtar, Ih
    real(dp),    device :: Uh(0:n), beta_out(1)
    complex(dp), device :: q0(0:n,1,1), Nq(0:n,1,1), Nq_prev(0:n,1,1)
    if (bo%ybasis == YB_FD) then
       call fd_mean_stage_cfr_gpu(mg, n, bo%kl, bo%d2b, bo%whd, ks, dt, Re, Qtar, &
                                  Uh, Ih, q0, Nq, Nq_prev, beta_out)
    else
       call mean_stage_cfr_gpu(mg, n, ks, dt, Re, Qtar, bo%sd, bo%st, bo%bd, bo%boff, &
                               Uh, Ih, q0, Nq, Nq_prev, beta_out)
    end if
  end subroutine y_mean_stage_cfr

  !-- v-solve residual probe:
  !   rmax = max|t1 - alpha*t2 - rhs|  and  fmax = max|rhs|  over the slab.
  !   t1 = L v, t2 = L^2 v (caller runs y_apply_L twice into lent scratch);
  !   rhs = the LAST substage's assembled RHS (sws%rhsv -- y_vif_solve never
  !   modifies it). Each scalar gets its OWN single-reduction kernel (the
  !   proven divergence_check/cfl_freq_max idiom -- see the in-body note).
  subroutine y_vres_max(n, nx, nz, alpha, t1, t2, rhs, rmax, fmax)
    integer(ip), intent(in)  :: n, nx, nz
    real(dp),    intent(in)  :: alpha
    complex(dp), device      :: t1(0:n,nx,nz), t2(0:n,nx,nz), rhs(0:n,nx,nz)
    real(dp),    intent(out) :: rmax, fmax
    integer(ip) :: ix, iz, k
    ! TWO kernels, ONE reduction scalar each -- deliberately. Every proven CUF
    ! reduction in this codebase (divergence_check, cfl_freq_max, imag_max_c)
    ! carries exactly one implicit reduction; a dual-reduction region has no
    ! in-repo precedent and its failure mode under a codegen surprise is
    ! SILENT GARBAGE, not a compile error. The second pass costs one slab
    ! read at div_every cadence -- free.
    rmax = 0.0_dp
    fmax = 0.0_dp
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, n
             rmax = max(rmax, abs(t1(k,ix,iz) - alpha*t2(k,ix,iz) - rhs(k,ix,iz)))
          end do
       end do
    end do
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, n
             fmax = max(fmax, abs(rhs(k,ix,iz)))
          end do
       end do
    end do
  end subroutine y_vres_max

  !-- windowed twin of y_vres_max: max over rows jlo..jhi ONLY. FD use:
  !   the solve replaces the wall rows by BCs and the closure acts through them,
  !   so the strong recompute (two banded L applies, half-width kl) is defect-
  !   contaminated up to 2*kl rows in from each wall -- excluding those rows
  !   leaves the collocation rows the banded solve actually satisfied, i.e. a
  !   backward-error-level, flow-independent quantity (numpy model:
  !   4.1e-12/7.5e-12 on the two fd decks vs O(1..1e3) full). The
  !   excluded span also covers the sqrt(alpha) Stokes layer on every current
  !   deck (delta ~ 2e-4 sits inside the first 1-2 cells). fmax is windowed
  !   too (interior residual over interior RHS -- the honest ratio).
  subroutine y_vres_max_win(n, nx, nz, alpha, jlo, jhi, t1, t2, rhs, rmax, fmax)
    integer(ip), intent(in)  :: n, nx, nz, jlo, jhi
    real(dp),    intent(in)  :: alpha
    complex(dp), device      :: t1(0:n,nx,nz), t2(0:n,nx,nz), rhs(0:n,nx,nz)
    real(dp),    intent(out) :: rmax, fmax
    integer(ip) :: ix, iz, k
    rmax = 0.0_dp
    fmax = 0.0_dp
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = jlo, jhi
             rmax = max(rmax, abs(t1(k,ix,iz) - alpha*t2(k,ix,iz) - rhs(k,ix,iz)))
          end do
       end do
    end do
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = jlo, jhi
             fmax = max(fmax, abs(rhs(k,ix,iz)))
          end do
       end do
    end do
  end subroutine y_vres_max_win

  !-- Cheb-only GALERKIN-CONSISTENT v-solve probe.
  !   Re-runs the ks=3 two-Helmholtz chain on the PRESERVED rhsv (verbatim
  !   vif_solve_gpu_ws steps 1-2: same scalings, same banded solves) and
  !   measures the residual of each banded system AS SOLVED, in Shen test
  !   space:  r_i = P(op_i . solution_i) - ftil_i  with the op recomputed as
  !   strong-apply + project (algebraically the assembled band matrix, since
  !   S/B ARE those projections). Unlike the strong-form probe these are
  !   backward-error quantities: flow- and deck-INDEPENDENT floors (numpy
  !   model, true Shen bases: 3.6e-16..8.5e-16 at
  !   production stiffness; runtime slabs -> expect ~1e-14..1e-13). Drift =
  !   real dp-margin loss -- the precision canary. Also returns the closure
  !   quality: max wall |Dv| of the CLOSED state v vs the pre-closure
  !   particular solution (what the influence matrix cancelled).
  !   Lends: sh1/sh2 (0:n-2) + c1/c2/c3 (0:n), all dead post-step; rhsv and v
  !   are NOT modified. CHEB ONLY -- caller guards on the backend (project_shen
  !   on FD point values is meaningless; FD uses y_vres_max_win instead).
  subroutine y_vres_galerkin(bo, n, nx, nz, alpha, kx, kz, rhsv, v, &
                             sh1, sh2, c1, c2, c3, r1, f1, r2, f2, dvv, dvp)
    type(band_ops_t), intent(in) :: bo
    integer(ip),      intent(in) :: n, nx, nz
    real(dp),         intent(in) :: alpha
    real(dp),    device :: kx(nx), kz(nz)
    complex(dp), device :: rhsv(0:n,nx,nz), v(0:n,nx,nz)
    complex(dp), device :: sh1(0:n-2,nx,nz), sh2(0:n-2,nx,nz)
    complex(dp), device :: c1(0:n,nx,nz), c2(0:n,nx,nz), c3(0:n,nx,nz)
    real(dp),    intent(out) :: r1, f1, r2, f2, dvv, dvp
    real(dp) :: inva
    inva = 1.0_dp/alpha
    ! solve 1 re-run:  ftil1 = P(rhsv)/alpha ;  u1 = (S + k2 B)^-1 ftil1
    call project_shen(n, nx, nz, rhsv, sh1)
    call so_cscale(n-2, nx, nz, sh1, inva)
    call helm_band_solve_gpu(n, nx, nz, 0.0_dp, kx, kz, bo%sd, bo%st, bo%bd, &
                             bo%boff, sh1, sh2)
    call shen_to_cheb_c(n, nx, nz, sh2, c1)                    ! g (Cheb coeffs)
    ! r1 = max| P(-L g) - ftil1 |,  f1 = max|ftil1|
    call y_apply_L(bo, n, nx, nz, kx, kz, c1, c2, c3)
    call so_shen_res_max(n, nx, nz, -1.0_dp, 0.0_dp, c2, c2, sh1, r1, f1)
    ! solve 2 re-run:  ftil2 = -P(g) ;  u2 = (S + (k2+1/alpha) B)^-1 ftil2
    call project_shen(n, nx, nz, c1, sh1)
    call so_cscale(n-2, nx, nz, sh1, -1.0_dp)
    call helm_band_solve_gpu(n, nx, nz, inva, kx, kz, bo%sd, bo%st, bo%bd, &
                             bo%boff, sh1, sh2)
    call shen_to_cheb_c(n, nx, nz, sh2, c1)                    ! v_p overwrites g (dead)
    ! r2 = max| P(-L v_p + (1/alpha) v_p) - ftil2 |,  f2 = max|ftil2|
    call y_apply_L(bo, n, nx, nz, kx, kz, c1, c2, c3)
    call so_shen_res_max(n, nx, nz, -1.0_dp, inva, c2, c1, sh1, r2, f2)
    ! closure quality: wall Dv of the closed state v vs the particular v_p
    call so_walldv_max(n, nx, nz, c1, dvp)
    call so_walldv_max(n, nx, nz, v,  dvv)
  end subroutine y_vres_galerkin

  !-- rmax = max_k | (pi/2)(c_k u_k - u_{k+2}) - ft_k |  over k=0..n-2,
  !   with u = ca*qa + cb*qb computed on the fly (qa may alias qb when cb=0),
  !   and fmax = max|ft|. The inline factor IS project_shen's formula (same PI
  !   literal). Two single-reduction kernels (house rule, see y_vres_max).
  subroutine so_shen_res_max(n, nx, nz, ca, cb, qa, qb, ft, rmax, fmax)
    integer(ip), intent(in)  :: n, nx, nz
    real(dp),    intent(in)  :: ca, cb
    complex(dp), device      :: qa(0:n,nx,nz), qb(0:n,nx,nz), ft(0:n-2,nx,nz)
    real(dp),    intent(out) :: rmax, fmax
    integer(ip) :: ix, iz, k
    real(dp)    :: ck
    complex(dp) :: uk, uk2
    rmax = 0.0_dp
    fmax = 0.0_dp
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, n-2
             if (k == 0) then; ck = 2.0_dp; else; ck = 1.0_dp; end if
             uk  = ca*qa(k,ix,iz)   + cb*qb(k,ix,iz)
             uk2 = ca*qa(k+2,ix,iz) + cb*qb(k+2,ix,iz)
             rmax = max(rmax, abs(0.5_dp*SO_PI*(ck*uk - uk2) - ft(k,ix,iz)))
          end do
       end do
    end do
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, n-2
             fmax = max(fmax, abs(ft(k,ix,iz)))
          end do
       end do
    end do
  end subroutine so_shen_res_max

  !-- dmax = max over (kx,kz) of max(|Du(-1)|, |Du(+1)|) from Cheb
  !   coeffs (wall_derivs_c sums, max-reduced instead of stored -- saves the
  !   (nx,nz) wall buffers). Single reduction scalar.
  subroutine so_walldv_max(n, nx, nz, qc, dmax)
    integer(ip), intent(in)  :: n, nx, nz
    complex(dp), device      :: qc(0:n,nx,nz)
    real(dp),    intent(out) :: dmax
    integer(ip) :: ix, iz, j, s
    real(dp)    :: jj
    complex(dp) :: slo, shi
    dmax = 0.0_dp
    !$cuf kernel do(2) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          slo = (0.0_dp,0.0_dp); shi = (0.0_dp,0.0_dp)
          do j = 1, n
             jj  = real(j*j, dp)
             shi = shi + qc(j,ix,iz)*jj
             if (mod(j,2) == 1) then; s = 1; else; s = -1; end if
             slo = slo + real(s,dp)*qc(j,ix,iz)*jj
          end do
          dmax = max(dmax, max(abs(slo), abs(shi)))
       end do
    end do
  end subroutine so_walldv_max

  !-- in-place complex scale: THE production lam-scaler ---------------------
  subroutine so_cscale(m, nx, nz, x, s)
    integer(ip), intent(in) :: m, nx, nz
    real(dp),    intent(in) :: s
    complex(dp), device     :: x(0:m, nx, nz)
    integer(ip) :: ix, iz, k
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz; do ix = 1, nx; do k = 0, m
       x(k,ix,iz) = x(k,ix,iz)*s
    end do; end do; end do
  end subroutine so_cscale

  subroutine vsolve_ops_alloc(vo, n, nx, nz)
    type(vsolve_ops_t), intent(inout) :: vo
    integer(ip),        intent(in)    :: n, nx, nz
    allocate(vo%vm(0:n,nx,nz,3), vo%vp(0:n,nx,nz,3), vo%Cinv(2,2,nx,nz,3))
  end subroutine vsolve_ops_alloc

  subroutine vsolve_ops_free(vo)
    type(vsolve_ops_t), intent(inout) :: vo
    if (allocated(vo%vm)) deallocate(vo%vm, vo%vp, vo%Cinv)
  end subroutine vsolve_ops_free

  subroutine mean_ops_alloc(mo, n)
    type(mean_ops_t), intent(inout) :: mo
    integer(ip),      intent(in)    :: n
    allocate(mo%Uhm(0:n,3), mo%Ih(3))
  end subroutine mean_ops_alloc

  subroutine mean_ops_free(mo)
    type(mean_ops_t), intent(inout) :: mo
    if (allocated(mo%Uhm)) deallocate(mo%Uhm, mo%Ih)
  end subroutine mean_ops_free

end module solver_ops_mod
