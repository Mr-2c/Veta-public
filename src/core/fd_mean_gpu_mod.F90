!===============================================================================
! fd_mean_gpu_mod.F90 -- ON-DEVICE FD-y mean stage: the point-
! value twin of mean_gpu_mod, COMPOSED from the validated batched FD
! primitives at batch = 1 (one (0,0) column), exactly as the Cheb device mean
! is composed from cheb_deriv_c/project_shen/helm_band_kernel/shen_to_cheb_c.
! The FD chain is SHORTER: one fd_apply_L_gpu call (k^2 = 0 -> pure D2)
! replaces the two cheb_derivs, and the Shen projection/conversion vanish in
! point space. Host oracle: fd_mean_mod (statement-mirrored, gated in
! test_fd_mean); device-vs-host gated in test_fd_mean_gpu.
!
! REUSES mean_gpu_ws verbatim (d1 slot idle; d2/rhs/Up + kx0/kz0 + Uh_d/
! Nu0_d/Nw0_d/beta_d carry over; the Shen-side ftil/qsh/dia/up1/up2 sit
! unused in FD mode) -- no new workspace type, no stepper_ws change. The
! q0 <- exactly-real forcing (mean_gpu_real) and the SMR constants are the
! SAME objects the Cheb mean uses (imported -- one source of truth).
! The CFR flux uses the grid FILE's weights (band_ops_t%whd, device).
!
! Build: nvfortran -cuda -c (after kinds, mean_gpu_mod, fd_helm_gpu_mod).
!===============================================================================
module fd_mean_gpu_mod
  use kinds,           only: dp, ip
  use cudafor
  use mean_gpu_mod,    only: mean_gpu_ws, mean_gpu_real, MGAMMA, MZETA, MCSUB, H_CH
  use fd_helm_gpu_mod, only: fd_helm_solve_gpu, fd_apply_L_gpu
  implicit none
  private
  public :: fd_mean_stage_gpu, fd_mean_stage_cfr_gpu

contains

  !-- rhs = lam*(q0 + a*D2q0 + g*Nq + z*Nq_prev + cdtbeta), ALL rows (the
  !   pointwise constant source; walls are overridden by the solve anyway) ------
  subroutine fd_mg_rhs(n, a, g, z, cdtbeta, lam, q0, d2, Nq, Nq_prev, rhs)
    integer(ip), intent(in) :: n
    real(dp),    intent(in) :: a, g, z, cdtbeta, lam
    complex(dp), device     :: q0(0:n,1,1), d2(0:n,1,1), Nq(0:n,1,1), Nq_prev(0:n,1,1)
    complex(dp), device     :: rhs(0:n,1,1)
    integer(ip) :: k
    complex(dp) :: t
    !$cuf kernel do(1) <<<*,*>>>
    do k = 0, n
       t = q0(k,1,1) + a*d2(k,1,1) + g*Nq(k,1,1) + z*Nq_prev(k,1,1) + cdtbeta
       rhs(k,1,1) = lam*t
    end do
  end subroutine fd_mg_rhs

  !-- CFR close (FD): 1 thread; qint = sum w_k * real(Up_k) (the FILE's weights),
  !   beta from the constraint, q0 = Up + ck*dt*beta*Uh forced exactly real. -----
  attributes(global) subroutine fd_mg_cfr_kernel(n, ckdt, Ih, Qtar, w, Up, Uh, q0, beta_out)
    integer(ip), value :: n
    real(dp),    value :: ckdt, Ih, Qtar
    real(dp)    :: w(0:n), Uh(0:n), beta_out(1)
    complex(dp) :: Up(0:n), q0(0:n)
    integer(ip) :: k
    real(dp)    :: qint, beta
    if (threadIdx%x /= 1 .or. blockIdx%x /= 1) return
    qint = 0.0_dp
    do k = 0, n
       qint = qint + w(k)*real(Up(k))
    end do
    beta = (H_CH*Qtar - qint)/(ckdt*Ih)
    do k = 0, n
       q0(k) = cmplx(real(Up(k)) + (ckdt*beta)*Uh(k), 0.0_dp, dp)
    end do
    beta_out(1) = beta
  end subroutine fd_mg_cfr_kernel

  !-----------------------------------------------------------------------------
  ! One SMR/CN FD mean stage at k^2 = 0, Dirichlet walls, constant source beta.
  ! q0 (0:n,1,1) complex point values, updated in place (forced exactly real --
  ! the same host-parity contract as mean_stage_gpu).
  !-----------------------------------------------------------------------------
  subroutine fd_mean_stage_gpu(ws, n, kl, d2b, ks, dt, Re, beta, q0, Nq, Nq_prev)
    type(mean_gpu_ws), intent(inout) :: ws
    integer(ip),       intent(in)    :: n, kl, ks
    real(dp),          intent(in)    :: dt, Re, beta
    real(dp),    device :: d2b(2*kl+1, 0:n)
    complex(dp), device :: q0(0:n,1,1), Nq(0:n,1,1), Nq_prev(0:n,1,1)
    real(dp) :: bk, a, lam, g, z, ck
    bk = 0.5_dp*MCSUB(ks);  a = bk*dt/Re;  lam = 1.0_dp/a
    g  = MGAMMA(ks)*dt;     z = MZETA(ks)*dt;  ck = MCSUB(ks)
    call fd_apply_L_gpu(n, kl, 1_ip, 1_ip, ws%kx0, ws%kz0, d2b, q0, ws%d2)  ! D2 (k2=0)
    call fd_mg_rhs(n, a, g, z, ck*dt*beta, lam, q0, ws%d2, Nq, Nq_prev, ws%rhs)
    call fd_helm_solve_gpu(n, kl, 1_ip, 1_ip, lam, ws%kx0, ws%kz0, d2b, ws%rhs, q0)
    call mean_gpu_real(n, q0)
  end subroutine fd_mean_stage_gpu

  !-----------------------------------------------------------------------------
  ! One CFR FD mean stage: beta=0 particular profile Up, then the flux closure
  ! with the file's weights. Uh is the (real, device) FD unit-source response
  ! (host fd_mean_flux_setup, uploaded by the driver -- same pattern as Cheb);
  ! Ih its FD flux (host scalar). u0 forced exactly real; beta -> beta_out(1).
  !-----------------------------------------------------------------------------
  subroutine fd_mean_stage_cfr_gpu(ws, n, kl, d2b, whd, ks, dt, Re, Qtar, &
                                   Uh, Ih, q0, Nq, Nq_prev, beta_out)
    type(mean_gpu_ws), intent(inout) :: ws
    integer(ip),       intent(in)    :: n, kl, ks
    real(dp),          intent(in)    :: dt, Re, Qtar, Ih
    real(dp),    device :: d2b(2*kl+1, 0:n), whd(0:n), Uh(0:n)
    complex(dp), device :: q0(0:n,1,1), Nq(0:n,1,1), Nq_prev(0:n,1,1)
    real(dp),    device :: beta_out(1)
    real(dp)   :: bk, a, lam, g, z, ck
    type(dim3) :: g1, b1
    integer(cuda_stream_kind) :: strm
    b1 = dim3(1,1,1);  g1 = dim3(1,1,1);  strm = cudaforGetDefaultStream()
    bk = 0.5_dp*MCSUB(ks);  a = bk*dt/Re;  lam = 1.0_dp/a
    g  = MGAMMA(ks)*dt;     z = MZETA(ks)*dt;  ck = MCSUB(ks)
    call fd_apply_L_gpu(n, kl, 1_ip, 1_ip, ws%kx0, ws%kz0, d2b, q0, ws%d2)
    call fd_mg_rhs(n, a, g, z, 0.0_dp, lam, q0, ws%d2, Nq, Nq_prev, ws%rhs)  ! particular
    call fd_helm_solve_gpu(n, kl, 1_ip, 1_ip, lam, ws%kx0, ws%kz0, d2b, ws%rhs, ws%Up)
    call fd_mg_cfr_kernel<<<g1,b1,0,strm>>>(n, ck*dt, Ih, Qtar, whd, ws%Up, Uh, q0, beta_out)
  end subroutine fd_mean_stage_cfr_gpu

end module fd_mean_gpu_mod
