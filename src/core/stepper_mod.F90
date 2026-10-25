!===============================================================================
! stepper_mod.F90  --  SMR (RK3) / Crank-Nicolson time step.
!
!   * step_ev_ws / step_full_ws : the production entry points --
!       identical numerics to step_ev/step_full below, but ALL device scratch
!       lives in a caller-owned, run-lifetime stepper_ws, so a step does no
!       cudaMalloc/cudaFree at all.
!   * step_ev / step_full : the ORIGINAL self-allocating interfaces, kept as
!       thin wrappers (allocate a stepper_ws, call the _ws variant, free) so the
!       existing tests compile and behave unchanged. New code should use _ws.
!
! step_ev advances the fluctuations (eta, v) with the nonlinear term ON
! (validated by test_stepper_xcheck etc.). step_full
! runs the same per-stage fluctuation update PLUS the mean profiles u0(y), w0(y)
! advanced ON DEVICE (mean_gpu_mod via y_mean_stage/_cfr into sws%mg; see the
! [mean-on-GPU] notes below) with the constant-mass-flux constraint --
! mean_mod's host stepping routines survive only as test oracles.
!
! Per substage k = 1..3 (symmetric CN, alpha_k = beta_k = c_k/2):
!   eta: [1 - a_k L] eta^k = [1 + a_k L] eta^{k-1}
!                            + gamma_k dt N_eta^{k-1} + zeta_k dt N_eta^{k-2}
!   v  : [L - a_k L^2] v^k = L[1 + a_k L] v^{k-1}
!                            + gamma_k dt N_v^{k-1}   + zeta_k dt N_v^{k-2}
! with a_k = beta_k dt/Re, beta_k = c_k/2, c = gamma + zeta,
!   gamma = (8/15, 5/12, 3/4),  zeta = (0, -17/60, -5/12).
!
! NYQUIST INVARIANT: nl_evaluate masks the kx/kz-Nyquist
! columns of N_v, N_eta, so a state whose Nyquist columns start exactly zero
! (init_mod guarantees this; the driver also sanitises restarts) KEEPS them
! exactly zero: every RHS contribution there is exactly 0 and the banded/vif
! solve of an exactly-zero RHS returns exactly 0. test_nyquist pins this.
!
! MEAN COUPLING (step_full). nl_evaluate produces, alongside the fluctuation
! forcings, the (0,0)-mode forcings hx0/hz0. After each stage's ev_stage, the mean
! advances ON THE DEVICE (mean_gpu_mod, dp), in-stream:
!   - u0 by mean_stage_cfr_gpu (k^2=0 Dirichlet Helmholtz + the constant-mass-flux
!     constraint solving for the mean pressure gradient beta) and w0 by
!     mean_stage_gpu (beta = 0); the mean state lives in sws%mg (no host round-trip);
!   - recover_uw splices u0/w0 at the (0,0) mode for the next stage's nl_evaluate;
!   - the (0,0) Fourier column of (eta, v) is zeroed (v == omega_y == 0 there).
! Keeping the mean on the device removes a host synchronization point that
! blocks comm/compute overlap. It is not a measured speedup: the mean is a 1-D
! problem (~n coeffs, microseconds) and measures ~0.1% of step time, while
! production steps are comm-bound on the all-to-all transpose. See mean_gpu_mod.
!
! INDEX CARE: fields have n+1 Chebyshev coeffs (0:n); apply_L and nl_evaluate get
! ny = n+1, everything else the degree n. ws must be set up with ny = n+1.
!
! Build:  nvfortran -cuda -cudalib=cufft -c stepper_mod.F90   (after mean_mod).
!===============================================================================
module stepper_mod
  use kinds,             only: dp, ip
  use cudafor
  use cheb_ops_mod,      only: apply_L
  use helm_band_gpu_mod, only: helm_band_solve_gpu
  use spec_helpers_mod,  only: project_shen, shen_to_cheb_c
  use vsolve_if_gpu_mod, only: vif_solve_gpu_ws
  use nonlinear_mod,     only: nl_workspace, nl_evaluate
  use mean_gpu_mod,      only: mean_gpu_ws, mean_gpu_alloc, mean_gpu_free, &
                                                              mean_gpu_real, mean_gpu_copy
  use solver_ops_mod, only: band_ops_t, vsolve_ops_t, mean_ops_t, &  ! aggregates
                          y_apply_L, y_helm_solve, y_vif_solve, &  ! dispatch seam
                          y_mean_stage, y_mean_stage_cfr           ! mean dispatch
  implicit none
  private
  public :: stepper_ws, stepper_ws_alloc, stepper_ws_free
  public :: step_ev_ws, step_full_ws
  public :: step_ev, step_full
  public :: stepper_coeffs_ok                      ! cross-module table assert

  real(dp), parameter :: GAMMA(3) = [ 8.0_dp/15.0_dp,  5.0_dp/12.0_dp,  3.0_dp/4.0_dp ]
  real(dp), parameter :: ZETA(3)  = [ 0.0_dp,        -17.0_dp/60.0_dp, -5.0_dp/12.0_dp ]
  real(dp), parameter :: CSUB(3)  = [ 8.0_dp/15.0_dp,  2.0_dp/15.0_dp,  1.0_dp/3.0_dp ]

  ! Run-lifetime device scratch for the fluctuation update. Allocate
  ! ONCE per run (stepper_ws_alloc), reuse every step. SHARED buffer: ftil serves
  ! both the eta Helmholtz solve and the vif sub-solves -- safe because their
  ! uses are strictly sequential within a stage (eta solve completes before the
  ! v solve begins). Members are passed to plain-dummy
  ! helpers only (never touched by an inline CUF kernel here), per the
  ! NVFORTRAN-0155 workaround.
  type :: stepper_ws
     ! nonlinear registers: one PING-PONG slot pair per variable together with
     ! the caller-owned prev arrays -- roles alternate per substage, no
     ! rotation copies. There is no separate L-tier array: the L-tier lands
     ! directly in rhs/rhsv and the fused crhs finishes them in place.
     ! There is no separate Shen/vif scratch either (-48 B/pt): the vif solve runs strictly
     ! AFTER the eta solve consumed rhs and after Lv's last read, so the eta
     ! Helmholtz lends vgsh as its Shen scratch and the vif solve lends the
     ! then-dead rhs (gc slot) and Lv (vpc slot) -- the ftil-sharing argument
     ! extended (uses strictly sequential within the substage, in BOTH twins
     ! including the hoisted parallel stage, and in BOTH y-bases). The
     ! between-steps diagnostic lenders in main_par are unaffected: everything
     ! here is dead there and written before read.
     complex(dp), device, allocatable :: Nv_cur(:,:,:), Neta_cur(:,:,:)
     complex(dp), device, allocatable :: rhs(:,:,:), scr(:,:,:)
     complex(dp), device, allocatable :: ftil(:,:,:)
     ! v-solve scratch
     complex(dp), device, allocatable :: Lv(:,:,:), rhsv(:,:,:)
     ! vif intermediates
     complex(dp), device, allocatable :: vgsh(:,:,:), vvsh(:,:,:)
     complex(dp), device, allocatable :: vdlo(:,:),   vdhi(:,:)
     ! (0,0)-mode forcings (mean handoff)
     complex(dp), device, allocatable :: hx0(:), hz0(:)
     ! on-device (0,0) mean-profile state + scratch (Uh, SMR history, beta, band
     ! scratch) -- no host mean solve, no round-trip (mean_gpu_mod)
     type(mean_gpu_ws) :: mg
  end type stepper_ws

contains

  !-- the SMR tables (CSUB/GAMMA/ZETA) are duplicated across mean_mod,
  !   mean_gpu_mod, stepper_lin_mod, par_stepper_mod and THIS module; the mean
  !   solves and the [res] Galerkin canary silently assume every copy is
  !   identical. The drivers assert it bitwise at startup via this function.
  pure logical function stepper_coeffs_ok(refc, refg, refz) result(okk)
    real(dp), intent(in) :: refc(3), refg(3), refz(3)
    okk = all(refc == CSUB) .and. all(refg == GAMMA) .and. all(refz == ZETA)
  end function stepper_coeffs_ok

  !-- fused SMR substage RHS, IN PLACE:  r := ((x + a*r) + g*p) + z*q.
  !   r enters holding the L-tier term (L eta, or L^2 v) and leaves as the
  !   substage RHS -- a cxpsy + caxpy + caxpy chain fused into ONE kernel with
  !   the IDENTICAL per-element association. In-place is race-free: each
  !   element is read once then written by the same thread. It also removes the
  !   need for a separate L-output array (32 B/pt).
  subroutine crhs(m, nx, nz, x, a, g, z, p, q, r)
    integer(ip), intent(in) :: m, nx, nz
    real(dp),    intent(in) :: a, g, z
    complex(dp), device     :: x(0:m, nx, nz), p(0:m, nx, nz), q(0:m, nx, nz)
    complex(dp), device     :: r(0:m, nx, nz)
    integer(ip) :: ix, iz, k
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz; do ix = 1, nx; do k = 0, m
       r(k,ix,iz) = ((x(k,ix,iz) + a*r(k,ix,iz)) + g*p(k,ix,iz)) + z*q(k,ix,iz)
    end do; end do; end do
  end subroutine crhs

  !-- zero the (0,0) Fourier column (ix=1, iz=1) of a field
  subroutine zero_col11(m, nx, nz, q)
    integer(ip), intent(in) :: m, nx, nz
    complex(dp), device     :: q(0:m, nx, nz)
    integer(ip) :: k
    !$cuf kernel do(1) <<<*,*>>>
    do k = 0, m
       q(k,1,1) = (0.0_dp, 0.0_dp)
    end do
  end subroutine zero_col11

  !-- allocate the run-lifetime workspace (call once; reuse every step) --------
  ! BASIS-AWARE: under FD-y no branch anywhere touches
  ! the three Shen-side buffers (ftil/vgsh/vvsh) -- y_helm_solve's FD leg is
  ! banded-direct, y_vif_solve's FD leg uses fsc/gc/vpc only, y_vif_setup's FD
  ! leg uses rhs/gpc/vpc only -- so allocating them full-size was ~48 B/pt of
  ! dead dp memory in exactly the backend whose big-box runs are
  ! memory-bound (~3.7 GB/GPU on a 16-node frame). fd=.true. allocates
  ! 1-element STUBS instead: the members stay ALLOCATED (a legal actual
  ! argument), and the dispatch seam's Shen-scratch dummies are assumed-size
  ! (solver_ops_mod), so the FD branches never reference beyond element 1.
  ! Callers that don't know the basis (the self-allocating step_ev/step_full
  ! wrappers, tests) omit fd and get the full Cheb-sized workspace -- correct
  ! in both bases, just not minimal.
  subroutine stepper_ws_alloc(s, n, nx, nz, fd)
    type(stepper_ws), intent(inout) :: s
    integer(ip),      intent(in)    :: n, nx, nz
    logical, optional, intent(in)   :: fd
    logical :: fd_
    fd_ = .false.;  if (present(fd)) fd_ = fd
    allocate(s%Nv_cur(0:n,nx,nz), s%Neta_cur(0:n,nx,nz))
    allocate(s%rhs(0:n,nx,nz), s%scr(0:n,nx,nz))
    allocate(s%Lv(0:n,nx,nz), s%rhsv(0:n,nx,nz))
    if (fd_) then
       allocate(s%ftil(0:0,1,1), s%vgsh(0:0,1,1), s%vvsh(0:0,1,1))  ! stubs (never referenced)
    else
       allocate(s%ftil(0:n-2,nx,nz))
       allocate(s%vgsh(0:n-2,nx,nz), s%vvsh(0:n-2,nx,nz))
    end if
    allocate(s%vdlo(nx,nz), s%vdhi(nx,nz))
    allocate(s%hx0(0:n), s%hz0(0:n))
    call mean_gpu_alloc(s%mg, n)     ! on-device (0,0) mean state (history zeroed)
  end subroutine stepper_ws_alloc

  subroutine stepper_ws_free(s)
    type(stepper_ws), intent(inout) :: s
    deallocate(s%Nv_cur, s%Neta_cur, s%rhs, s%scr, s%ftil, &
               s%Lv, s%rhsv, &
               s%vgsh, s%vvsh, s%vdlo, s%vdhi, s%hx0, s%hz0)
    call mean_gpu_free(s%mg)
  end subroutine stepper_ws_free

  !---------------------------------------------------------------------------
  ! ev_stage -- one SMR/CN substage of the FLUCTUATION update (eta, v).
  ! Produces the (0,0)-mode forcings in s%hx0/s%hz0 (for a mean solve by the
  ! caller; step_ev ignores them).
  ! PING-PONG history: the nonlinear forcing N^{k-1} is written into the
  ! WR pair and the ZETA term reads the RD pair (last substage's WR). The step
  ! drivers alternate the two register pairs per substage, avoiding a
  ! ccopy(cur -> prev) rotation (4 full-array passes/substage); the
  ! VALUES read/written each substage are identical to the rotating scheme, so
  ! the trajectory is unchanged. ZETA(1) = 0, so the RD pair's content at step
  ! entry is weightless.
  !---------------------------------------------------------------------------
  subroutine ev_stage(ws, n, nx, nz, ks, dt, Re, kx, kz, bo, vo, &
                      u0, w0, eta, v, Nv_wr, Neta_wr, Nv_rd, Neta_rd, s)
    type(nl_workspace), intent(inout) :: ws
    integer(ip), intent(in) :: n, nx, nz, ks
    real(dp),    intent(in) :: dt, Re
    real(dp),    device :: kx(nx), kz(nz)
    type(band_ops_t),   intent(in) :: bo    ! y-operator band structure
    type(vsolve_ops_t), intent(in) :: vo    ! shifted-solve lift + influence
    complex(dp), device :: u0(0:n), w0(0:n)
    complex(dp), device :: eta(0:n,nx,nz), v(0:n,nx,nz)
    complex(dp), device :: Nv_wr(0:n,nx,nz), Neta_wr(0:n,nx,nz)   ! N^{k-1} target
    complex(dp), device :: Nv_rd(0:n,nx,nz), Neta_rd(0:n,nx,nz)   ! N^{k-2} source
    type(stepper_ws), intent(inout) :: s
    real(dp) :: beta, a, lam, g, z

    beta = 0.5_dp*CSUB(ks)
    a    = beta*dt/Re
    lam  = 1.0_dp/a
    g    = GAMMA(ks)*dt
    z    = ZETA(ks)*dt

    ! explicit nonlinear forcing at the current state q^{k-1} = (v, eta, u0, w0)
    call nl_evaluate(ws, v, eta, u0, w0, Nv_wr, Neta_wr, s%hx0, s%hz0)

    ! --- eta: Helmholtz ---
    ! L eta lands DIRECTLY in rhs; crhs finishes it in place with the same
    ! per-element association the cxpsy+caxpy+caxpy chain had. The Shen
    ! scratch is the lent vgsh (dead until the vif solve below).
    call y_apply_L(bo, n, nx, nz, kx, kz, eta, s%rhs, s%scr)    ! dispatch
    call crhs(n, nx, nz, eta, a, g, z, Neta_wr, Neta_rd, s%rhs) ! rhs = eta + a L eta
                                                                !  + g N^{k-1} + z N^{k-2}
    call y_helm_solve(bo, n, nx, nz, lam, kx, kz, s%rhs, eta, s%ftil, s%vgsh)  ! eta = eta^k

    ! --- v: biharmonic via influence matrix ---
    ! gc/vpc slots are the lent rhs (consumed by the eta solve above) and
    ! Lv (last read by crhs below); next writes to rhs/Lv
    ! are the next substage's y_apply_L outputs, so nothing reads the scribble.
    call y_apply_L(bo, n, nx, nz, kx, kz, v,    s%Lv,   s%scr)
    call y_apply_L(bo, n, nx, nz, kx, kz, s%Lv, s%rhsv, s%scr)  ! L^2 v -> rhsv
    call crhs(n, nx, nz, s%Lv, a, g, z, Nv_wr, Nv_rd, s%rhsv)   ! rhsv = L v + a L^2 v
                                                                !  + g N^{k-1} + z N^{k-2}
    call y_vif_solve(bo, n, nx, nz, a, kx, kz, &
                     vo%vm(:,:,:,ks), vo%vp(:,:,:,ks), vo%Cinv(:,:,:,:,ks), &
                     s%rhsv, v, &
                     s%ftil, s%vgsh, s%rhs, s%vvsh, s%Lv, s%vdlo, s%vdhi, s%scr)  ! v = v^k
  end subroutine ev_stage

  !---------------------------------------------------------------------------
  ! step_ev_ws -- one full (nonlinear + viscous) SMR/CN step advancing (eta, v),
  ! with caller-owned workspace. Mean modes u0,w0 are inputs (held
  ! fixed; caller passes the current mean).
  !---------------------------------------------------------------------------
  subroutine step_ev_ws(ws, sws, n, nx, nz, dt, Re, kx, kz, bo, vo, &
                        u0, w0, eta, v, Nv_prev, Neta_prev)
    type(nl_workspace), intent(inout) :: ws
    type(stepper_ws),   intent(inout) :: sws
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: dt, Re
    real(dp),    device :: kx(nx), kz(nz)
    type(band_ops_t),   intent(in) :: bo    ! y-operator band structure
    type(vsolve_ops_t), intent(in) :: vo    ! shifted-solve lift + influence
    complex(dp), device :: u0(0:n), w0(0:n)
    complex(dp), device :: eta(0:n,nx,nz), v(0:n,nx,nz)
    complex(dp), device :: Nv_prev(0:n,nx,nz), Neta_prev(0:n,nx,nz)
    integer(ip) :: ks
    ! ping-pong the two register pairs (see ev_stage): odd substages write
    ! the ws pair and read the caller pair, even substages the reverse.
    do ks = 1, 3
       if (mod(ks,2) == 1) then
          call ev_stage(ws, n, nx, nz, ks, dt, Re, kx, kz, bo, vo, u0, w0, eta, v, &
                        sws%Nv_cur, sws%Neta_cur, Nv_prev, Neta_prev, sws)
       else
          call ev_stage(ws, n, nx, nz, ks, dt, Re, kx, kz, bo, vo, u0, w0, eta, v, &
                        Nv_prev, Neta_prev, sws%Nv_cur, sws%Neta_cur, sws)
       end if
    end do
  end subroutine step_ev_ws

  !---------------------------------------------------------------------------
  ! step_full_ws -- one full-system SMR/CN step with caller-owned workspace.
  ! [on-device mean]: (eta, v) fluctuations AND the u0,w0 mean profiles (constant
  ! mass flux) all advance on the GPU -- no host round-trip. The mean (0,0) state
  ! (Uh, SMR history, beta) lives in sws%mg (mean_gpu_mod); the only host touch is
  ! a tiny one-way Uh upload per step (tracks dt changes), never a readback.
  !
  ! mo%Uhm/mo%Ih/Qtar drive the CFR mean solve. The band structure rides in bo
  ! (host mirrors bo%sdh for callers that need them); the mean history lives
  ! on-device in sws%mg.
  !---------------------------------------------------------------------------
  subroutine step_full_ws(ws, sws, n, nx, nz, dt, Re, kx, kz, bo, vo, mo, Qtar, &
                          u0, w0, eta, v, Nv_prev, Neta_prev)
    type(nl_workspace), intent(inout) :: ws
    type(stepper_ws),   intent(inout) :: sws
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: dt, Re, Qtar
    real(dp),    device :: kx(nx), kz(nz)
    type(band_ops_t),   intent(in) :: bo    ! y-operator band structure
    type(vsolve_ops_t), intent(in) :: vo    ! shifted-solve lift + influence
    type(mean_ops_t),   intent(in) :: mo    ! CFR mean precompute
    complex(dp), device :: u0(0:n), w0(0:n)
    complex(dp), device :: eta(0:n,nx,nz), v(0:n,nx,nz)
    complex(dp), device :: Nv_prev(0:n,nx,nz), Neta_prev(0:n,nx,nz)
    integer(ip) :: ks

    ! on-device CFR precompute (tiny one-way real H2D; tracks dt-dependent Uh) +
    ! enter the step with u0,w0 exactly real (matches the host's real(u0))
    sws%mg%Uh_d = mo%Uhm
    call mean_gpu_real(n, u0);  call mean_gpu_real(n, w0)

    do ks = 1, 3
       ! fluctuation substage on the GPU (also yields the (0,0)-mode forcings);
       ! ping-pong register pairing, see step_ev_ws
       if (mod(ks,2) == 1) then
          call ev_stage(ws, n, nx, nz, ks, dt, Re, kx, kz, bo, vo, u0, w0, eta, v, &
                        sws%Nv_cur, sws%Neta_cur, Nv_prev, Neta_prev, sws)
       else
          call ev_stage(ws, n, nx, nz, ks, dt, Re, kx, kz, bo, vo, u0, w0, eta, v, &
                        Nv_prev, Neta_prev, sws%Nv_cur, sws%Neta_cur, sws)
       end if

       ! mean solve on the device (dp), in-stream: u0 with constant-mass-flux, w0
       ! unconstrained. Reads sws%hx0/hz0 (forcings) + sws%mg history; no host sync.
       call y_mean_stage_cfr(bo, sws%mg, n, ks, dt, Re, Qtar, &
                             sws%mg%Uh_d(:,ks), mo%Ih(ks), u0, sws%hx0, &
                             sws%mg%Nu0_d, sws%mg%beta_d)   ! dispatch
       call y_mean_stage(bo, sws%mg, n, ks, dt, Re, 0.0_dp, w0, sws%hz0, sws%mg%Nw0_d)

       ! rotate mean history (this stage's forcing -> next stage's N^{k-2})
       call mean_gpu_copy(n, sws%hx0, sws%mg%Nu0_d)
       call mean_gpu_copy(n, sws%hz0, sws%mg%Nw0_d)

       ! the (0,0) Fourier column of the fluctuation fields is identically zero
       call zero_col11(n, nx, nz, v)
       call zero_col11(n, nx, nz, eta)
    end do
  end subroutine step_full_ws

  !---------------------------------------------------------------------------
  ! step_ev -- ORIGINAL self-allocating interface (compat wrapper for the
  ! existing tests). Numerically identical to step_ev_ws.
  !---------------------------------------------------------------------------
  subroutine step_ev(ws, n, nx, nz, dt, Re, kx, kz, bo, vo, &
                     u0, w0, eta, v, Nv_prev, Neta_prev)
    type(nl_workspace), intent(inout) :: ws
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: dt, Re
    real(dp),    device :: kx(nx), kz(nz)
    type(band_ops_t),   intent(in) :: bo    ! y-operator band structure
    type(vsolve_ops_t), intent(in) :: vo    ! shifted-solve lift + influence
    complex(dp), device :: u0(0:n), w0(0:n)
    complex(dp), device :: eta(0:n,nx,nz), v(0:n,nx,nz)
    complex(dp), device :: Nv_prev(0:n,nx,nz), Neta_prev(0:n,nx,nz)
    type(stepper_ws) :: s
    call stepper_ws_alloc(s, n, nx, nz)
    call step_ev_ws(ws, s, n, nx, nz, dt, Re, kx, kz, bo, vo, &
                    u0, w0, eta, v, Nv_prev, Neta_prev)
    call stepper_ws_free(s)
  end subroutine step_ev

  !---------------------------------------------------------------------------
  ! step_full -- ORIGINAL self-allocating interface (compat wrapper for the
  ! existing tests). Numerically identical to step_full_ws.
  !---------------------------------------------------------------------------
  subroutine step_full(ws, n, nx, nz, dt, Re, kx, kz, bo, vo, mo, Qtar, &
                       u0, w0, eta, v, Nv_prev, Neta_prev)
    type(nl_workspace), intent(inout) :: ws
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: dt, Re, Qtar
    real(dp),    device :: kx(nx), kz(nz)
    type(band_ops_t),   intent(in) :: bo
    type(vsolve_ops_t), intent(in) :: vo
    type(mean_ops_t),   intent(in) :: mo
    complex(dp), device :: u0(0:n), w0(0:n)
    complex(dp), device :: eta(0:n,nx,nz), v(0:n,nx,nz)
    complex(dp), device :: Nv_prev(0:n,nx,nz), Neta_prev(0:n,nx,nz)
    type(stepper_ws) :: s
    call stepper_ws_alloc(s, n, nx, nz)
    call step_full_ws(ws, s, n, nx, nz, dt, Re, kx, kz, bo, vo, mo, Qtar, &
                      u0, w0, eta, v, Nv_prev, Neta_prev)
    call stepper_ws_free(s)
  end subroutine step_full

end module stepper_mod
