!===============================================================================
! mean_gpu_mod.F90  --  ON-DEVICE (0,0)-mode mean-profile stepper.
!
! GPU counterpart of mean_mod (host, dp). It moves the (0,0) mean solve on-device,
! removing the per-substage host D2H/solve/H2D round-trip on rank 0. The mean is one
! (0,0) column -- intrinsically serial and tiny (microseconds) -- so this is NOT a
! standalone speedup: at production scale the step is comm-bound on the all-to-all
! transpose (measured: ncclDevKernel_SendRecv ~70% of GPU time; the mean's
! helm_band ~1.8%), and the step time is unchanged. Its value is that it removes
! a host synchronization point that otherwise blocks comm/compute overlap and
! keeps the mean off the host. Validated bit-exact (--validate P=1/4/8, dp+sp).
! Only the (0,0) owner calls it.
!
! Same SMR/CN stage as mean_mod, ASSEMBLED from already-validated GPU primitives at
! batch=1 (one column): cheb_deriv_c x2 (= D^2 = cheb_d2 to round-off), project_shen
! (= helmholtz_rhs), helm_band_kernel (k^2=0 banded solve), shen_to_cheb_c. RHS
! assembly / CFR flux-beta-update / im=0 forcing are small dp kernels here. dp.
!
! BAND SOLVE: calls helm_band_kernel DIRECTLY at ncol=1 with this module's OWN tiny
! persistent scratch (ws%dia/up1/up2). It does NOT use helm_band_solve_gpu, whose
! column-leading module buffers are exact-resize -- sharing them with the
! fluctuation solve's ncol=bnkx*nz launches would dealloc/realloc multi-MB buffers
! every substage. For ncol=1 the column-leading (1,0:n-2) layout is bit-identical
! memory to y-leading (0:n-2,1,1), so ftil/qsh/dia/up1/up2 (all (0:n-2,1,1)) pass
! straight through with no transpose -- bit-identical to helm_band_solve_gpu@ncol=1.
!
! BIT-EXACTNESS vs the host mean_mod: ~1e-15/step (GPU D^2/band solve vs CPU
! cheb_d2/helm_band_solve), checked by test_mean_gpu. main_par --validate stays
! EXACT because both the distributed and single-GPU paths call THIS routine AND it
! forces u0,w0 to be real each substage (q0 = cmplx(real(.),0), exactly as the host
! did via real(u0)/cmplx(u0h,0)); the band solve's real coefficients decouple the
! re/im channels, so the real result depends only on real(forcing) -- which is
! bit-identical between the two paths -- and any reduction-order difference in
! im(hx0) is discarded.
!
! RANK CONVENTION: the mean is ONE (0,0) column, physically rank-1 (0:n). The
! public q0/Nq/Nq_prev dummies are declared (0:n,1,1) so they feed the batched
! primitives (project_shen/shen_to_cheb_c, shaped (0:n,nx,nz)) with NO rank change.
! The steppers hold the column as rank-1 (u0, sws%hx0, ...) and pass it here via
! F2018 sequence association (15.5.2.11: an explicit-shape dummy may associate with
! a contiguous actual of any rank) -- the SAME mechanism apply_L uses calling
! cheb_deriv_c (rank-3 q -> rank-2 a). Uh is rank-1 (only mg_cfr_kernel consumes it).
!
! Build: nvfortran -cuda -c mean_gpu_mod.F90 (needs kinds, cheb_ops_mod,
! spec_helpers_mod, helm_band_gpu_mod).
!===============================================================================
module mean_gpu_mod
  use kinds,             only: dp, ip
  use cudafor
  use cheb_ops_mod,      only: cheb_deriv_c
  use spec_helpers_mod,  only: project_shen, shen_to_cheb_c
  use helm_band_gpu_mod, only: helm_band_kernel
  implicit none
  private
  public :: mean_gpu_ws, mean_gpu_alloc, mean_gpu_free
  public :: mean_stage_gpu, mean_stage_cfr_gpu
  public :: mean_gpu_real, mean_gpu_copy
  public :: MGAMMA, MZETA, MCSUB, H_CH

  ! SMR RKW3 weights -- identical to mean_mod (kept in sync deliberately).
  real(dp), parameter :: MGAMMA(3) = [ 8.0_dp/15.0_dp,  5.0_dp/12.0_dp, 3.0_dp/4.0_dp ]
  real(dp), parameter :: MZETA(3)  = [ 0.0_dp,        -17.0_dp/60.0_dp, -5.0_dp/12.0_dp ]
  real(dp), parameter :: MCSUB(3)  = [ 8.0_dp/15.0_dp,  2.0_dp/15.0_dp,  1.0_dp/3.0_dp ]
  real(dp), parameter :: H_CH = 2.0_dp    ! channel full height (walls at +/-1)

  ! Per-(0,0)-column scratch + on-device mean state (batch=1). Persistent; this is
  ! a member of stepper_ws, so it is allocated once with the workspace and the mean
  ! state (Uh_d, Nu0_d/Nw0_d history, beta_d) lives entirely on the device.
  type :: mean_gpu_ws
     integer(ip) :: n = -1
     real(dp),    allocatable, device :: kx0(:), kz0(:)            ! (1) = 0  (k^2 = 0)
     complex(dp), allocatable, device :: d1(:,:,:), d2(:,:,:)      ! (0:n,1,1)  Dq, D^2q
     complex(dp), allocatable, device :: rhs(:,:,:)                ! (0:n,1,1)
     complex(dp), allocatable, device :: ftil(:,:,:), qsh(:,:,:)   ! (0:n-2,1,1)
     complex(dp), allocatable, device :: Up(:,:,:)                 ! (0:n,1,1) particular profile
     real(dp),    allocatable, device :: dia(:,:,:), up1(:,:,:), up2(:,:,:) ! band scratch (0:n-2,1,1)
     real(dp),    allocatable, device :: Uh_d(:,:)                 ! (0:n,3) CFR unit-source response
     complex(dp), allocatable, device :: Nu0_d(:,:,:), Nw0_d(:,:,:)! (0:n,1,1) mean SMR history N^{k-2}
     real(dp),    allocatable, device :: beta_d(:)                 ! (1) instantaneous mean dp/dx
  end type mean_gpu_ws

contains

  subroutine mean_gpu_alloc(ws, n)
    type(mean_gpu_ws), intent(inout) :: ws
    integer(ip),       intent(in)    :: n
    ! A same-size re-alloc must deliver the same contract as a fresh one:
    ! ZEROED history. A bare 'return' would keep stale Nu0/Nw0/beta from a
    ! previous in-process run alive into the first new step's z*Nq_prev
    ! (weightless while finite because MZETA(1)=0, but a NaN survivor would
    ! poison substages 2-3). Gated by test_mean_gpu's re-init leg.
    if (ws%n == n) then
       ws%Nu0_d = (0.0_dp, 0.0_dp);  ws%Nw0_d = (0.0_dp, 0.0_dp);  ws%beta_d = 0.0_dp
       return
    end if
    call mean_gpu_free(ws)
    allocate(ws%kx0(1), ws%kz0(1));  ws%kx0 = 0.0_dp;  ws%kz0 = 0.0_dp
    allocate(ws%d1(0:n,1,1), ws%d2(0:n,1,1), ws%rhs(0:n,1,1), ws%Up(0:n,1,1))
    allocate(ws%ftil(0:n-2,1,1), ws%qsh(0:n-2,1,1))
    allocate(ws%dia(0:n-2,1,1), ws%up1(0:n-2,1,1), ws%up2(0:n-2,1,1))
    allocate(ws%Uh_d(0:n,3), ws%Nu0_d(0:n,1,1), ws%Nw0_d(0:n,1,1), ws%beta_d(1))
    ! history starts at zero (and zeta_1 = 0 means it is never used across steps,
    ! so a fresh/restarted run needs no carried mean history)
    ws%Nu0_d = (0.0_dp, 0.0_dp);  ws%Nw0_d = (0.0_dp, 0.0_dp);  ws%beta_d = 0.0_dp
    ws%n = n
  end subroutine mean_gpu_alloc

  subroutine mean_gpu_free(ws)
    type(mean_gpu_ws), intent(inout) :: ws
    if (allocated(ws%kx0)) deallocate(ws%kx0, ws%kz0, ws%d1, ws%d2, ws%rhs, ws%Up, &
                                      ws%ftil, ws%qsh, ws%dia, ws%up1, ws%up2, &
                                      ws%Uh_d, ws%Nu0_d, ws%Nw0_d, ws%beta_d)
    ws%n = -1
  end subroutine mean_gpu_free

  !-- RHS assembly: rhs = q0 + a*D^2 q0 + g*Nq + z*Nq_prev (+ ck*dt*beta on T_0) --
  subroutine mg_rhs(n, a, g, z, cdtbeta, q0, d2, Nq, Nq_prev, rhs)
    integer(ip), intent(in) :: n
    real(dp),    intent(in) :: a, g, z, cdtbeta
    complex(dp), device     :: q0(0:n,1,1), d2(0:n,1,1), Nq(0:n,1,1), Nq_prev(0:n,1,1)
    complex(dp), device     :: rhs(0:n,1,1)
    integer(ip) :: k
    !$cuf kernel do(1) <<<*,*>>>
    do k = 0, n
       rhs(k,1,1) = q0(k,1,1) + a*d2(k,1,1) + g*Nq(k,1,1) + z*Nq_prev(k,1,1)
       if (k == 0) rhs(0,1,1) = rhs(0,1,1) + cdtbeta
    end do
  end subroutine mg_rhs

  !-- scale the Shen-test projection by lam: ftil <- lam*ftil ---------------------
  subroutine mg_scale(n, lam, ftil)
    integer(ip), intent(in) :: n
    real(dp),    intent(in) :: lam
    complex(dp), device     :: ftil(0:n-2,1,1)
    integer(ip) :: k
    !$cuf kernel do(1) <<<*,*>>>
    do k = 0, n-2
       ftil(k,1,1) = lam*ftil(k,1,1)
    end do
  end subroutine mg_scale

  !-- force a mean column to be exactly real (q <- cmplx(real(q),0)). Plain device
  !   dummy so the CUF kernel never touches a derived-type member (the
  !   -O2/0155). PUBLIC: the steppers call it once at step entry (matches the host's
  !   real(u0) before its mean loop).
  subroutine mean_gpu_real(m, q)
    integer(ip), intent(in) :: m
    complex(dp), device     :: q(0:m)
    integer(ip) :: k
    !$cuf kernel do(1) <<<*,*>>>
    do k = 0, m
       q(k) = cmplx(real(q(k)), 0.0_dp, dp)
    end do
  end subroutine mean_gpu_real

  !-- dst <- src (plain-dummy device copy; rotates the mean SMR history register) --
  subroutine mean_gpu_copy(m, src, dst)
    integer(ip), intent(in) :: m
    complex(dp), device     :: src(0:m), dst(0:m)
    integer(ip) :: k
    !$cuf kernel do(1) <<<*,*>>>
    do k = 0, m
       dst(k) = src(k)
    end do
  end subroutine mean_gpu_copy

  !-- CFR close: 1 thread computes qint=flux(Up), beta, then u0 = Up + ck*dt*beta*Uh,
  !   forced exactly real. Uh is the (real) unit-source response. beta -> beta_out.
  !   (local is `qint`, NOT `Ip` -- `Ip` case-folds to the kind `ip`.)
  attributes(global) subroutine mg_cfr_kernel(n, ckdt, Ih, Qtar, Up, Uh, q0, beta_out)
    integer(ip), value :: n
    real(dp),    value :: ckdt, Ih, Qtar
    complex(dp) :: Up(0:n), q0(0:n)
    real(dp)    :: Uh(0:n)
    real(dp)    :: beta_out(1)
    integer(ip) :: k
    real(dp)    :: qint, beta
    if (threadIdx%x /= 1 .or. blockIdx%x /= 1) return
    qint = 0.0_dp
    do k = 0, n, 2                                  ! even-k Chebyshev integral weights
       qint = qint + real(Up(k))*2.0_dp/(1.0_dp - real(k,dp)**2)
    end do
    beta = (H_CH*Qtar - qint)/(ckdt*Ih)
    do k = 0, n
       q0(k) = cmplx(real(Up(k)) + (ckdt*beta)*Uh(k), 0.0_dp, dp)
    end do
    beta_out(1) = beta
  end subroutine mg_cfr_kernel

  !-----------------------------------------------------------------------------
  ! mean_stage_gpu -- one SMR/CN stage for a mean profile q0 at k^2=0, Dirichlet,
  ! constant source beta (0 for w0). q0(0:n) complex(dp), updated in place (forced
  ! real). sd/st/bd are the DEVICE helm_band structure; Nq/Nq_prev the (0,0)
  ! forcings (current / previous-stage).
  !-----------------------------------------------------------------------------
  subroutine mean_stage_gpu(ws, n, ks, dt, Re, beta, sd, st, bd, boff, q0, Nq, Nq_prev)
    type(mean_gpu_ws), intent(inout) :: ws
    integer(ip),       intent(in)    :: n, ks
    real(dp),          intent(in)    :: dt, Re, beta, boff
    real(dp),    device :: sd(0:n-2), st(0:n-2), bd(0:n-2)
    complex(dp), device :: q0(0:n,1,1), Nq(0:n,1,1), Nq_prev(0:n,1,1)
    real(dp)   :: bk, a, lam, g, z, ck
    type(dim3) :: g1, b1
    integer(cuda_stream_kind) :: strm      ! explicit stream for chevron launches
    b1 = dim3(1,1,1);  g1 = dim3(1,1,1);  strm = cudaforGetDefaultStream()
    bk = 0.5_dp*MCSUB(ks);  a = bk*dt/Re;  lam = 1.0_dp/a
    g  = MGAMMA(ks)*dt;     z = MZETA(ks)*dt;  ck = MCSUB(ks)
    call cheb_deriv_c(n+1, 1, q0,    ws%d1)         ! D q0
    call cheb_deriv_c(n+1, 1, ws%d1, ws%d2)         ! D^2 q0
    call mg_rhs(n, a, g, z, ck*dt*beta, q0, ws%d2, Nq, Nq_prev, ws%rhs)
    call project_shen(n, 1, 1, ws%rhs, ws%ftil)     ! Cheb -> Shen-test
    call mg_scale(n, lam, ws%ftil)                  ! ftil <- lam*ftil
    call helm_band_kernel<<<g1,b1,0,strm>>>(n, 1, 1, lam, ws%kx0, ws%kz0, sd, st, bd, boff, &
                                     ws%ftil, ws%qsh, ws%dia, ws%up1, ws%up2)
    call shen_to_cheb_c(n, 1, 1, ws%qsh, q0)        ! Shen -> Cheb (Dirichlet)
    call mean_gpu_real(n, q0)                       ! force exactly real (host parity)
  end subroutine mean_stage_gpu

  !-----------------------------------------------------------------------------
  ! mean_stage_cfr_gpu -- one stage for u0 with the constant-mass-flux constraint.
  ! Solves the beta=0 particular profile Up, then sets beta so int u0 = H*Qtar and
  ! returns it in beta_out (device scalar (1)). Uh/Ih are the stage's precomputed
  ! (real) unit-source response (DEVICE) and its integral (host scalar) from
  ! mean_mod's mean_flux_setup. u0 forced exactly real.
  !-----------------------------------------------------------------------------
  subroutine mean_stage_cfr_gpu(ws, n, ks, dt, Re, Qtar, sd, st, bd, boff, &
                                Uh, Ih, q0, Nq, Nq_prev, beta_out)
    type(mean_gpu_ws), intent(inout) :: ws
    integer(ip),       intent(in)    :: n, ks
    real(dp),          intent(in)    :: dt, Re, Qtar, boff, Ih
    real(dp),    device :: sd(0:n-2), st(0:n-2), bd(0:n-2)
    real(dp),    device :: Uh(0:n)
    complex(dp), device :: q0(0:n,1,1), Nq(0:n,1,1), Nq_prev(0:n,1,1)
    real(dp),    device :: beta_out(1)
    real(dp)   :: bk, a, lam, g, z, ck
    type(dim3) :: g1, b1
    integer(cuda_stream_kind) :: strm      ! explicit stream for chevron launches
    b1 = dim3(1,1,1);  g1 = dim3(1,1,1);  strm = cudaforGetDefaultStream()
    bk = 0.5_dp*MCSUB(ks);  a = bk*dt/Re;  lam = 1.0_dp/a
    g  = MGAMMA(ks)*dt;     z = MZETA(ks)*dt;  ck = MCSUB(ks)
    call cheb_deriv_c(n+1, 1, q0,    ws%d1)
    call cheb_deriv_c(n+1, 1, ws%d1, ws%d2)
    call mg_rhs(n, a, g, z, 0.0_dp, q0, ws%d2, Nq, Nq_prev, ws%rhs)   ! particular: no beta
    call project_shen(n, 1, 1, ws%rhs, ws%ftil)
    call mg_scale(n, lam, ws%ftil)
    call helm_band_kernel<<<g1,b1,0,strm>>>(n, 1, 1, lam, ws%kx0, ws%kz0, sd, st, bd, boff, &
                                     ws%ftil, ws%qsh, ws%dia, ws%up1, ws%up2)
    call shen_to_cheb_c(n, 1, 1, ws%qsh, ws%Up)     ! Up = particular profile (Cheb)
    call mg_cfr_kernel<<<g1,b1,0,strm>>>(n, ck*dt, Ih, Qtar, ws%Up, Uh, q0, beta_out)
  end subroutine mean_stage_cfr_gpu

end module mean_gpu_mod
