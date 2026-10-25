!===============================================================================
! par_stepper_mod.F90 -- the multi-GPU DISTRIBUTED stepper.
!
! par_step_ev  advances the FLUCTUATIONS (eta, v) one full SMR/CN step on
!              THIS rank's kx-slab, with the mean (u0, w0) held fixed (an input).
! par_step_full advances the FULL system: the same fluctuation update PLUS
!              the rank-0-local mean (u0, w0) solve (constant mass flux) and the
!              (0,0)-column zeroing -- the mean machinery runs ONLY on the rank
!              owning kx=0 (owns_mode00), with NO communication or reduction.
!
! par_step_ev advances (eta, v) one full SMR/CN step on THIS rank's kx-slab, with
! the mean (u0, w0) held fixed (an input). It is byte-for-byte step_ev with two
! substitutions:
!     nl_workspace  -> par_nl_workspace        (the distributed nl state)
!     nl_evaluate   -> par_nl_evaluate (phased: ev_begin/mid/end)   (the distributed RHS)
! Everything else -- the per-column eta-Helmholtz / v-biharmonic solve stack
! (apply_L, project_shen, helm_band_solve_gpu, shen_to_cheb_c, vif_solve_gpu_ws)
! and the pointwise/AXPY helpers -- is value-based / per-(kx,kz)-column with y as
! the (local) solve direction, so it runs UNCHANGED on the slab (nx = bnkx). The
! ONLY communication is inside par_nl_evaluate (its two batched all-to-alls).
!
! The stepper workspace (stepper_ws) and its alloc/free are REUSED from
! stepper_mod (allocated with nx=bnkx => slab-sized scratch). The SMR-RK3
! coefficients are PRIVATE in stepper_mod, so they are re-declared here
! VERBATIM. The pointwise helpers are NOT verbatim: serial fuses the
! whole RHS into one crhs kernel, while this twin keeps a hoisted cxpsyi
! (inside the flight window) + a fused caxpy2 (post-flight) -- same
! per-element association, different kernel grouping. The helpers
! take PLAIN device dummies, so their CUF kernels never touch a derived-type
! allocatable member -- provided helper INLINING stays suppressed: every shipped
! build uses -O2 -Mnoinline, where -Mnoinline is what carries the guarantee
! (bare -O2 would inline them into par_ev_stage and re-expose the members =>
! NVFORTRAN-S-0155, loud at compile). This module is
! dp on the solve side; the tp tier appears
! only inside par_nl_evaluate (the nonlinear transforms).
!
! Oracle for the fluctuation test: the single-GPU step_ev (stepper_mod). The on-device
! mean solve shares src/mean_gpu_mod with the single-GPU step_full, so both paths
! are bit-identical (main_par --validate).
!===============================================================================
module par_stepper_mod
  use kinds,             only: dp, ip
  use cudafor
  use cheb_ops_mod,      only: apply_L
  use helm_band_gpu_mod, only: helm_band_solve_gpu
  use spec_helpers_mod,  only: project_shen, shen_to_cheb_c
  use vsolve_if_gpu_mod, only: vif_solve_gpu_ws
  use stepper_mod,       only: stepper_ws, stepper_ws_alloc, stepper_ws_free
  use par_nonlinear_mod, only: par_nl_workspace, &
                               par_nl_ev_begin, par_nl_ev_mid, par_nl_ev_end
  use mean_gpu_mod,      only:                                mean_gpu_real, mean_gpu_copy
  use solver_ops_mod, only: band_ops_t, vsolve_ops_t, mean_ops_t, &  ! aggregates
                          y_apply_L, y_helm_solve, y_vif_solve, &  ! dispatch seam
                          y_mean_stage, y_mean_stage_cfr           ! mean dispatch
  implicit none
  private
  public :: par_step_ev, par_step_ev_ws
  public :: par_step_full, par_step_full_ws
  public :: parstep_coeffs_ok                      ! cross-module table assert

  ! SMR-RK3 coefficients -- MUST match stepper_mod's (verbatim copy).
  real(dp), parameter :: GAMMA(3) = [ 8.0_dp/15.0_dp,  5.0_dp/12.0_dp,  3.0_dp/4.0_dp ]
  real(dp), parameter :: ZETA(3)  = [ 0.0_dp,        -17.0_dp/60.0_dp, -5.0_dp/12.0_dp ]
  real(dp), parameter :: CSUB(3)  = [ 8.0_dp/15.0_dp,  2.0_dp/15.0_dp,  1.0_dp/3.0_dp ]

contains

  !-- the drivers check the "MUST match" contract above bitwise at startup
  !   against mean_mod's tables.
  pure logical function parstep_coeffs_ok(refc, refg, refz) result(okk)
    real(dp), intent(in) :: refc(3), refg(3), refz(3)
    okk = all(refc == CSUB) .and. all(refg == GAMMA) .and. all(refz == ZETA)
  end function parstep_coeffs_ok

  !-- IN-PLACE partial RHS: r := x + a*r  (r enters holding the L-tier
  !   term). This piece stays its own
  !   kernel because it is HOISTED into the flight window -- the N terms
  !   it does NOT touch land only after par_nl_ev_end.
  subroutine cxpsyi(m, nx, nz, x, a, r)
    integer(ip), intent(in) :: m, nx, nz
    real(dp),    intent(in) :: a
    complex(dp), device     :: x(0:m, nx, nz), r(0:m, nx, nz)
    integer(ip) :: ix, iz, k
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz; do ix = 1, nx; do k = 0, m
       r(k,ix,iz) = x(k,ix,iz) + a*r(k,ix,iz)
    end do; end do; end do
  end subroutine cxpsyi

  !-- both N terms in ONE kernel: r := (r + g*p) + z*q, with the IDENTICAL
  !   per-element association as a caxpy+caxpy pair.
  subroutine caxpy2(m, nx, nz, g, z, p, q, r)
    integer(ip), intent(in) :: m, nx, nz
    real(dp),    intent(in) :: g, z
    complex(dp), device     :: p(0:m, nx, nz), q(0:m, nx, nz), r(0:m, nx, nz)
    integer(ip) :: ix, iz, k
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz; do ix = 1, nx; do k = 0, m
       r(k,ix,iz) = (r(k,ix,iz) + g*p(k,ix,iz)) + z*q(k,ix,iz)
    end do; end do; end do
  end subroutine caxpy2

  !-- zero the (0,0) Fourier column (ix=1, iz=1) of a field. Verbatim from
  !   stepper_mod (private there). On a kx-slab this is called ONLY by the rank
  !   owning kx=0, where local column (1,1) IS the global (kx=0,kz=0) mean mode
  !   (b_kx0=0 there, and kz=0 is z-index 1 on every rank since z is never split).
  subroutine zero_col11(m, nx, nz, q)
    integer(ip), intent(in) :: m, nx, nz
    complex(dp), device     :: q(0:m, nx, nz)
    integer(ip) :: k
    !$cuf kernel do(1) <<<*,*>>>
    do k = 0, m
       q(k,1,1) = (0.0_dp, 0.0_dp)
    end do
  end subroutine zero_col11

  !---------------------------------------------------------------------------
  ! par_ev_stage -- one SMR/CN substage of the FLUCTUATION update on the slab.
  ! Verbatim ev_stage with nl_workspace->par_nl_workspace, nl_evaluate->
  ! par_nl_evaluate. nx is THIS rank's bnkx; kx is the local-kx slice; vm/vp/Cinv
  ! are the slab precomputes. s%hx0/s%hz0 (mean forcings) are produced for a
  ! caller's mean solve (par_step_ev ignores them -- it holds the mean fixed).
  !---------------------------------------------------------------------------
  subroutine par_ev_stage(ws, n, nx, nz, ks, dt, Re, kx, kz, bo, vo, &
                          u0, w0, eta, v, Nv_wr, Neta_wr, Nv_rd, Neta_rd, s)
    type(par_nl_workspace), intent(inout) :: ws
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

    ! Nonlinear evaluation in PHASES, with the stepper's independent
    ! L-operator/RHS-partial kernels HOISTED into the flight windows: they read
    ! only the SPECTRAL eta/v (which the nl phases only READ) and write only
    ! s%rhs/Lv/rhsv/scr (which the nl phases never touch) -- so they fill
    ! par_stream while the all-to-alls run on par_comm_stream. The L-tier
    ! lands DIRECTLY in rhs/rhsv (in-place cxpsyi); the two N terms land
    ! post-flight in one fused caxpy2 with the association of a caxpy pair.
    ! Ping-pong registers, see par_step_ev_ws.
    call par_nl_ev_begin(ws, v, eta, u0, w0)          ! inverse flights launched
    ! HOIST A (under the 6-field inverse): eta L-chain + rhs partial
    call y_apply_L(bo, n, nx, nz, kx, kz, eta, s%rhs, s%scr)  ! L eta -> rhs
    call cxpsyi(n, nx, nz, eta, a, s%rhs)                     ! rhs = eta + a L eta
    call par_nl_ev_mid(ws)                            ! forward flights launched
    ! HOIST B (under the 3-field forward): v L-chain + rhsv partial
    call y_apply_L(bo, n, nx, nz, kx, kz, v,    s%Lv,   s%scr)
    call y_apply_L(bo, n, nx, nz, kx, kz, s%Lv, s%rhsv, s%scr)  ! L^2 v -> rhsv
    call cxpsyi(n, nx, nz, s%Lv, a, s%rhsv)                   ! rhsv = L v + a L^2 v
    call par_nl_ev_end(ws, Nv_wr, Neta_wr, s%hx0, s%hz0)      ! H lands + projects

    ! --- eta: Helmholtz (N terms + solve; needs Neta_wr from ev_end) ---
    ! Shen scratch = lent vgsh (vif runs strictly after)
    call caxpy2(n, nx, nz, g, z, Neta_wr, Neta_rd, s%rhs)     ! + g N^{k-1} + z N^{k-2}
    call y_helm_solve(bo, n, nx, nz, lam, kx, kz, s%rhs, eta, s%ftil, s%vgsh)  ! eta = eta^k

    ! --- v: biharmonic via influence matrix ---
    ! gc/vpc = lent rhs (consumed by the eta solve above -- NOTE rhs and
    ! rhsv are BOTH live inside the flight windows in this hoisted twin, so
    ! they can never merge; rhs only goes dead here) and lent Lv (last read at
    ! the cxpsyi in HOIST B).
    call caxpy2(n, nx, nz, g, z, Nv_wr, Nv_rd, s%rhsv)        ! + g N^{k-1} + z N^{k-2}
    call y_vif_solve(bo, n, nx, nz, a, kx, kz, &
                     vo%vm(:,:,:,ks), vo%vp(:,:,:,ks), vo%Cinv(:,:,:,:,ks), &
                     s%rhsv, v, &
                     s%ftil, s%vgsh, s%rhs, s%vvsh, s%Lv, s%vdlo, s%vdhi, s%scr)  ! v = v^k
  end subroutine par_ev_stage

  !---------------------------------------------------------------------------
  ! par_step_ev_ws -- one full (nonlinear + viscous) SMR/CN step on the slab,
  ! caller-owned workspace. Mean modes u0,w0 are fixed inputs. No zero_col11.
  !---------------------------------------------------------------------------
  subroutine par_step_ev_ws(ws, sws, n, nx, nz, dt, Re, kx, kz, bo, vo, &
                            u0, w0, eta, v, Nv_prev, Neta_prev)
    type(par_nl_workspace), intent(inout) :: ws
    type(stepper_ws),       intent(inout) :: sws
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: dt, Re
    real(dp),    device :: kx(nx), kz(nz)
    type(band_ops_t),   intent(in) :: bo    ! y-operator band structure
    type(vsolve_ops_t), intent(in) :: vo    ! shifted-solve lift + influence
    complex(dp), device :: u0(0:n), w0(0:n)
    complex(dp), device :: eta(0:n,nx,nz), v(0:n,nx,nz)
    complex(dp), device :: Nv_prev(0:n,nx,nz), Neta_prev(0:n,nx,nz)
    integer(ip) :: ks
    ! ping-pong the two register pairs (see par_ev_stage): odd substages
    ! write the ws pair and read the caller pair, even substages the reverse.
    do ks = 1, 3
       if (mod(ks,2) == 1) then
          call par_ev_stage(ws, n, nx, nz, ks, dt, Re, kx, kz, bo, vo, u0, w0, eta, v, &
                            sws%Nv_cur, sws%Neta_cur, Nv_prev, Neta_prev, sws)
       else
          call par_ev_stage(ws, n, nx, nz, ks, dt, Re, kx, kz, bo, vo, u0, w0, eta, v, &
                            Nv_prev, Neta_prev, sws%Nv_cur, sws%Neta_cur, sws)
       end if
    end do
  end subroutine par_step_ev_ws

  !---------------------------------------------------------------------------
  ! par_step_ev -- self-allocating wrapper (mirrors step_ev). ws is a pre-setup
  ! par_nl_workspace (par_nl_setup); the stepper_ws is allocated here with bnkx.
  !---------------------------------------------------------------------------
  subroutine par_step_ev(ws, n, nx, nz, dt, Re, kx, kz, bo, vo, &
                         u0, w0, eta, v, Nv_prev, Neta_prev)
    type(par_nl_workspace), intent(inout) :: ws
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
    call par_step_ev_ws(ws, s, n, nx, nz, dt, Re, kx, kz, bo, vo, &
                        u0, w0, eta, v, Nv_prev, Neta_prev)
    call stepper_ws_free(s)
  end subroutine par_step_ev

  !---------------------------------------------------------------------------
  ! par_step_full_ws -- one full-system SMR/CN step on the slab: the
  ! fluctuation update (par_ev_stage) PLUS the rank-0-local mean (u0,w0) solve
  ! and the (0,0)-column zeroing. Caller-owned workspace.
  !
  ! Same body as step_full_ws (src/core/stepper_mod.F90) with TWO distribution-aware
  ! changes, both keyed on ws%owns_mode00 (the rank holding global kx=0):
  !   (1) the ON-DEVICE (dp) mean solve (mean_stage_cfr_gpu for u0, mean_stage_gpu
  !       for w0, mean_gpu_mod) runs ONLY on the owner -- the mean (0,0) profile and
  !       its CFR precompute (sws%mg) are entirely local to that rank, with NO
  !       reduction and NO host round-trip (recover_uw reads u0,w0 only at k2=0,
  !       which exists only on the owner, so every other rank ignores them; verified
  !       in src/core/velocity_mod.F90). Keeping this rank-0-only mean on the device avoids
  !       a per-substage host sync (hygiene; it unblocks comm/compute overlap) -- it
  !       is NOT itself a speedup: production is comm-bound on the all-to-all and
  !       the mean solve is microseconds.
  !   (2) zero_col11 runs ONLY on the owner, where local column (1,1) is the global
  !       (kx=0,kz=0) mode. On any other rank local (1,1) is kx=b_kx0>0 -- a real
  !       fluctuation column that must NOT be zeroed.
  ! par_ev_stage itself runs on every rank (the fluctuation update is rank-local);
  ! par_nl_evaluate writes sws%hx0/hz0 only on the owner, which is exactly where the
  ! mean solve consumes them. The CFR precompute Uh_all/Ih_all is decomposition-
  ! independent (per-column at k2=0), so every rank may hold it; only the owner uses
  ! it (uploaded to sws%mg%Uh_d). Band structure rides in bo, and the mean
  ! history lives on-device in sws%mg.
  !---------------------------------------------------------------------------
  subroutine par_step_full_ws(ws, sws, n, nx, nz, dt, Re, kx, kz, bo, vo, mo, Qtar, &
                              u0, w0, eta, v, Nv_prev, Neta_prev)
    type(par_nl_workspace), intent(inout) :: ws
    type(stepper_ws),       intent(inout) :: sws
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

    ! on-device CFR precompute + enter real -- only the kx=0 owner carries the mean
    if (ws%owns_mode00) then
       sws%mg%Uh_d = mo%Uhm
       call mean_gpu_real(n, u0);  call mean_gpu_real(n, w0)
    end if

    do ks = 1, 3
       ! fluctuation substage on this rank's kx-slab (also yields the (0,0)-mode
       ! forcings sws%hx0/hz0 -- written by par_nl_evaluate only on the owner);
       ! ping-pong register pairing, see par_step_ev_ws
       if (mod(ks,2) == 1) then
          call par_ev_stage(ws, n, nx, nz, ks, dt, Re, kx, kz, bo, vo, u0, w0, eta, v, &
                            sws%Nv_cur, sws%Neta_cur, Nv_prev, Neta_prev, sws)
       else
          call par_ev_stage(ws, n, nx, nz, ks, dt, Re, kx, kz, bo, vo, u0, w0, eta, v, &
                            Nv_prev, Neta_prev, sws%Nv_cur, sws%Neta_cur, sws)
       end if

       if (ws%owns_mode00) then
          ! mean solve on the device (dp), IN-STREAM -- replaces the host
          ! D2H/solve/H2D round-trip (a per-substage host sync on rank 0; the mean
          ! itself is microseconds). u0 constant-mass-flux, w0 unconstrained. No sync.
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
       end if
    end do
  end subroutine par_step_full_ws

  !---------------------------------------------------------------------------
  ! par_step_full -- self-allocating wrapper (mirrors step_full). ws is a pre-setup
  ! par_nl_workspace; the stepper_ws is allocated here with nx=bnkx.
  !---------------------------------------------------------------------------
  subroutine par_step_full(ws, n, nx, nz, dt, Re, kx, kz, bo, vo, mo, Qtar, &
                           u0, w0, eta, v, Nv_prev, Neta_prev)
    type(par_nl_workspace), intent(inout) :: ws
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
    call par_step_full_ws(ws, s, n, nx, nz, dt, Re, kx, kz, bo, vo, mo, Qtar, &
                          u0, w0, eta, v, Nv_prev, Neta_prev)
    call stepper_ws_free(s)
  end subroutine par_step_full

end module par_stepper_mod
