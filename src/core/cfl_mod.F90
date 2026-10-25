!===============================================================================
! cfl_mod.F90  --  convective CFL time-step limit for the SMR/CN integrator.
!
! The viscous term is implicit (Crank-Nicolson, unconditionally stable), so the
! ONLY stability limit on dt is the explicit nonlinear (convective) term. This
! module turns a velocity field into the largest stable dt.
!
! Frequency (standard pointwise channel-DNS form):
!     f(x) = |u|*inv_dx + |v|*inv_dy_local + |w|*inv_dz     (per physical point)
!     dt   = C / max_x f(x)
! C is the run CFL number (the driver's cfl_target). IMPORTANT:
! C is in CONVENTIONAL units |u|dt/dx (the display convention -- see main.F90),
! NOT eigenvalue units. Only the convective (imaginary) eigenvalues matter since
! viscosity is implicit, so the underlying limit is the explicit RK3 (RKW3)
! imaginary-axis bound CFL_STAB = sqrt(3) ~ 1.732 ON THE EIGENVALUE |lambda|dt.
! For a Fourier-resolved field |lambda| = |u|*kmax = pi*|u|/dx, so the ceiling on
! the CONVENTIONAL number C is CFL_STAB/pi ~ 0.5513, NOT CFL_STAB itself.
!   => Run with C <= CFL_STAB/pi ~ 0.5513 (sound production value ~0.30-0.32).
! Passing C ~ 1.0 here is UNSTABLE (it targets eigenvalue CFL ~ pi > sqrt(3));
! the driver default is 0.32. CFL_STAB below is the raw eigenvalue-axis bound;
! the driver forms the displayed/warned ceiling as CFL_STAB/pi
! (main.F90: cfl_max = CFL_STAB/PI).
!
! CONVENTIONS / DECISIONS:
!  * inv_dx/inv_dy/inv_dz enter as PRECOMPUTED RECIPROCALS and are MULTIPLIED,
!    never divided per point -- the whole CFL costs one division (dt = C/fmax).
!  * inv_dy_local is the reciprocal of the local Gauss-Lobatto-Chebyshev node
!    spacing (min adjacent gap), so the clustered near-wall nodes legitimately set
!    the limit (their gap ~ pi^2/2n^2, much smaller than the ~pi/n core gap).
!  * dx,dz are taken on the NOMINAL (nx,nz) grid, NOT the dealiased (nxd,nzd) grid.
!    max|u| is essentially grid-independent for the band-limited field, and the
!    nominal spacing gives the resolved-scale CFL; the dealiased spacing would only
!    make dt needlessly smaller (conservatism here just buys nothing).
!
! THIS MODULE DOES NOT TRANSFORM spectral -> physical. It takes PHYSICAL u,v,w
! and is a reduction only. What the drivers feed it: both monitor paths (serial
! monitor_fmax, parallel par_monitor) do a FRESH recover_uw + three dedicated
! transforms at MONITOR/RETUNE cadence (log_every / dt_check_every -- every
! O(20-250) steps, not every step), and the serial stats sampler consumes the
! same monitor-produced lab-frame fields. The design point: no
! dedicated transform runs EVERY step (that would waste ~half an nl_evaluate
! per step); a fresh transform at monitor cadence is the accepted cost of
! clean lab-frame diagnostics. The one-interval lag (the last firing's
! velocities set the next dt) is standard and negligible. (Those physical
! velocities live on the dealiased grid; pair them with the NOMINAL
! inv_dx = nx/Lx per the decision above.)
!
! NOTE (Galilean convecting frame): a constant convection speed
! U_c is subtracted from u in the advection to shrink max|u| and relax this limit.
! Implemented as a one-element shift of the recovered MEAN streamwise velocity (the
! (0,0) mode's T_0 coeff) inside par_nl_eval_H / nl_eval_H, so H = (u-U_c) x omega;
! the stored state and all output stay LAB-frame (the mean u0 is never modified, the
! CFR drive is untouched), and the CFL uses |u - U_c| via cfl_freq_max's u_offset.
! Exact by Galilean invariance (the moving-frame eqns are standard NS for u-U_c;
! v,eta and the (0,0) mean forcing are frame-invariant, and p = Pi - |u|^2/2 with
! both moving-frame recovers the exact physical pressure). Driver knob: u_galilean.
! Verified by src/models/work_galilean_model.py and the solver's --validate path.
!
! NOTE (for io_mod / driver): print the max stable CFL at startup, and per step
! WARN if the run CFL exceeds it -- but do NOT auto-reduce dt.
! Because the driver displays/compares in CONVENTIONAL units, the ceiling it
! prints and warns against is CFL_STAB/pi ~ 0.5513, NOT CFL_STAB (the
! eigenvalue-units bound, a factor pi too lenient). main.F90 does exactly this
! (cfl_max = CFL_STAB/PI).
!
! Build:  nvfortran -cuda -c cfl_mod.F90
!===============================================================================
module cfl_mod
  use kinds, only: dp, tp, ip
  use cudafor
  implicit none
  private
  public :: CFL_STAB, cfl_inv_dy, cfl_freq_max, cfl_dt

  ! Explicit RK3/RKW3 imaginary-axis stability limit (convective). The exact value
  ! for the specific SMR coefficients should be confirmed against SIMSON (same
  ! caveat as the alpha/beta split in the formulation notes); sqrt(3) is the
  ! standard RK3 bound and a safe reference.
  real(dp), parameter :: CFL_STAB = sqrt(3.0_dp)

contains

  !-- precompute reciprocal local y-spacings from GLC nodes y(0:n) (HOST) --------
  ! inv_dy(j) = 1 / (min adjacent node gap); endpoints use their single gap.
  ! Pass the resulting array to the device for cfl_freq_max.
  subroutine cfl_inv_dy(n, y, inv_dy)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: y(0:n)
    real(dp),    intent(out) :: inv_dy(0:n)
    integer(ip) :: j
    real(dp)    :: gm, gp
    inv_dy(0) = 1.0_dp/abs(y(0) - y(1))
    do j = 1, n-1
       gm = abs(y(j-1) - y(j))
       gp = abs(y(j)   - y(j+1))
       inv_dy(j) = 1.0_dp/min(gm, gp)
    end do
    inv_dy(n) = 1.0_dp/abs(y(n-1) - y(n))
  end subroutine cfl_inv_dy

  !-- fmax = max over the grid of |u|*inv_dx + |v|*inv_dy(j) + |w|*inv_dz ---------
  ! u,v,w are PHYSICAL velocities, shape (ny y-nodes, nx, nz). inv_dy is on device,
  ! aligned with the y index (entry j is the spacing at the j-th node, same index
  ! as u(j,:,:)). Scalar max-reduction into the host variable fmax.
  ! [precision] u,v,w come straight from the transform tier (the dealiased physical
  ! buffers ws%pu/pv/pw), so the velocity dummies are `tp`. The grid metrics
  ! (inv_dx/inv_dz/inv_dy) and the accumulator/result (f, fmax) stay `dp`: the
  ! per-node term `abs(u)*inv_dx + ...` therefore evaluates in `dp` and the
  ! max-reduction is `dp`. At the all-real64 build tp==dp, so this is
  ! bit-identical; under the fp32-transform tier (tp=sp) only the velocity load
  ! narrows.
  ! [Galilean] u_offset (optional, default 0): a constant streamwise convection
  ! speed U_c subtracted from u inside the |u| term so the CFL measures the
  ! MOVING-FRAME advection |u - U_c| (exact by Galilean invariance; see the NOTE
  ! above and the u0 mean-shift in the nonlinear term). Absent/zero => unchanged.
  subroutine cfl_freq_max(ny, nx, nz, u, v, w, inv_dx, inv_dz, inv_dy, fmax, u_offset)
    integer(ip), intent(in)  :: ny, nx, nz
    real(tp),    device      :: u(ny,nx,nz), v(ny,nx,nz), w(ny,nx,nz)
    real(dp),    device      :: inv_dy(ny)
    real(dp),    intent(in)  :: inv_dx, inv_dz
    real(dp),    intent(out) :: fmax
    real(dp),    intent(in), optional :: u_offset
    integer(ip) :: ix, iz, j
    real(dp)    :: f, uoff
    uoff = 0.0_dp;  if (present(u_offset)) uoff = u_offset
    fmax = 0.0_dp
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz; do ix = 1, nx; do j = 1, ny
       f = abs(u(j,ix,iz) - real(uoff,tp))*inv_dx + abs(v(j,ix,iz))*inv_dy(j) + abs(w(j,ix,iz))*inv_dz
       fmax = max(fmax, f)
    end do; end do; end do
  end subroutine cfl_freq_max

  !-- stable dt for run CFL number C (the single division per step) ---------------
  pure real(dp) function cfl_dt(fmax, C) result(dt)
    real(dp), intent(in) :: fmax, C
    dt = C / max(fmax, tiny(1.0_dp))
  end function cfl_dt

end module cfl_mod
