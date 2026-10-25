! stats_mod.F90  --  turbulence statistics for the channel DNS.  GPU; dp accum.
!
! Gathers, over time samples, the x-z-t Reynolds-averaged profiles as functions of
! the wall-normal coordinate at the GLC nodes:
!     <u>(y), <v>(y), <w>(y)  and  <u'u'>, <v'v'>, <w'w'>, <u'v'> (y).
!
! Input is the PHYSICAL velocity at the GLC y-nodes (the to_physical_da output:
! first index = physical y, x,z physical on the dealiased nxd x nzd grid). Per
! sample and y-node j we form the x-z PLANE AVERAGES
!     <u>_xz(j) = mean_{x,z} u ,  <u^2>_xz(j) = mean_{x,z} u^2 ,  (and v,w,uv)
! and add them to running sums. After N samples,
!     mean_u(j)   = (1/N) sum <u>_xz
!     <u'u'>(j)   = (1/N) sum <u^2>_xz - mean_u(j)^2          (and likewise)
! which is the x-z-t Reynolds average. The dealiased plane gives the SAME plane
! statistics as the (nx x nz) grid -- the zero-padded modes contribute nothing
! (Parseval) -- so we average over (nxd x nzd) directly.
!
! Re_tau is read from the MEAN wall shear. With delta=1 and nu=1/Re,
!     u_tau = sqrt((1/Re)|du0/dy|_wall) ,   Re_tau = u_tau*Re = sqrt(Re*|du0/dy|),
! averaged over the two walls; the wall slope comes from the (0,0) Chebyshev
! coeffs via  T_k'(+1) = k^2 ,  T_k'(-1) = (-1)^{k+1} k^2.
!
! divergence_check returns max | i kx u + D_y v + i kz w | over the spectral
! coefficients of every mode (a KMM-recovered field is divergence-free, so this
! is ~round-off; it is a correctness probe, zero iff div u = 0).
!
! Derived-type device-allocatable members are NEVER touched inside a CUF kernel
! (NVFORTRAN-0155): the accumulation kernel takes bare device arrays.
! CPU/GPU build: nvfortran -cuda -c stats_mod.F90  (after kinds, yderiv_gpu_mod,
! yops_mod).

module stats_mod
  use kinds,          only: tp, wp, dp, ip
  use cudafor
  use yderiv_gpu_mod, only: y_deriv_c             ! y-basis dispatch seam
  use yops_mod,       only: yop_wall_deriv        ! the y-op facade
  implicit none
  private
  public :: stats_t, stats_init, stats_reset, stats_accumulate, stats_finalize
  public :: stats_combine, stats_destroy, wall_shear, re_tau, divergence_check
  public :: imag_max_c, field_sumabs_c
  public :: hsym_check_c, hsym_fix_c          ! kx=0 z-Hermitian probes
  ! pressure statistics: same plane-mean idiom, separate routines so the
  ! bit-validated velocity path above is untouched.
  public :: stats_accum_p, stats_finalize_p, stats_combine_p

  ! Running sums of the per-sample x-z plane averages, one entry per GLC y-node.
  type :: stats_t
     integer(ip) :: n = 0, nxd = 0, nzd = 0, nsamp = 0
     real(dp), allocatable, device :: su(:), suu(:)     ! <u>, <u^2>
     real(dp), allocatable, device :: sv(:), svv(:)     ! <v>, <v^2>
     real(dp), allocatable, device :: sw(:), sww(:)     ! <w>, <w^2>
     real(dp), allocatable, device :: suv(:)            ! <u v>
     real(dp), allocatable, device :: sp(:), spp(:)     ! <p>, <p^2>
  end type stats_t

contains

  subroutine stats_init(st, n, nxd, nzd)
    type(stats_t), intent(inout) :: st
    integer(ip),   intent(in)    :: n, nxd, nzd
    st%n = n;  st%nxd = nxd;  st%nzd = nzd
    allocate(st%su(0:n), st%suu(0:n), st%sv(0:n), st%svv(0:n), &
             st%sw(0:n), st%sww(0:n), st%suv(0:n), st%sp(0:n), st%spp(0:n))
    call stats_reset(st)
  end subroutine stats_init

  subroutine stats_destroy(st)
    type(stats_t), intent(inout) :: st
    if (allocated(st%su))  deallocate(st%su, st%suu, st%sv, st%svv, &
                                      st%sw, st%sww, st%suv, st%sp, st%spp)
    st%n = 0; st%nxd = 0; st%nzd = 0; st%nsamp = 0
  end subroutine stats_destroy

  subroutine stats_reset(st)
    type(stats_t), intent(inout) :: st
    st%su  = 0.0_dp;  st%suu = 0.0_dp
    st%sv  = 0.0_dp;  st%svv = 0.0_dp
    st%sw  = 0.0_dp;  st%sww = 0.0_dp
    st%suv = 0.0_dp;  st%nsamp = 0
    st%sp  = 0.0_dp;  st%spp = 0.0_dp
  end subroutine stats_reset

  !---------------------------------------------------------------------------
  ! Add one sample.  pu,pv,pw: physical velocity at (y_GLC, x, z), (0:n,nxd,nzd).
  ! [R2C] pu,pv,pw are REAL physical velocities (R2C: physical fields are real).
  ! The accumulation reads only real parts anyway, so the values are identical to
  ! the C2C complex-input path -- this cut only drops the dead imaginary storage.
  !
  ! [DISTRIBUTED] nzd is the LOCAL number of z-planes this rank holds
  ! (the 3rd extent of pu/pv/pw); the plane-average NORMALISATION uses the GLOBAL
  ! plane size st%nxd*st%nzd stored at stats_init. On one GPU local == global, so
  ! the per-sample contribution su += (plane sum)/(nxd*nzd) is UNCHANGED (and the
  ! single-GPU result is bit-identical).  Under a 1-D z-slab each rank adds the
  ! partial plane-mean of ITS planes; x (nxd) is never split and the normaliser is
  ! global, so summing su.. across ranks (par_stats_finalize) reconstructs the full
  ! x-z plane mean exactly -- to dp sum-reassociation round-off only.
  subroutine stats_accumulate(st, n, nxd, nzd, pu, pv, pw)
    type(stats_t), intent(inout) :: st
    integer(ip),   intent(in)    :: n, nxd, nzd     ! nzd = LOCAL z-plane count
    real(tp),      device        :: pu(0:n,nxd,nzd), pv(0:n,nxd,nzd), pw(0:n,nxd,nzd)
    real(dp) :: fac
    ! int64 product: the int32 form overflows for dealiased planes
    ! above 2^31-1 points (Re_tau ~ 20000-class, e.g. 61440x46080) and silently
    ! mis-normalises EVERY statistic. Value-identical for all in-range sizes.
    fac = 1.0_dp/real(int(st%nxd,8)*int(st%nzd,8), dp)  ! GLOBAL plane size (rank-invariant)
    call accum_kernel(n, nxd, nzd, fac, pu, pv, pw, &
                      st%su, st%suu, st%sv, st%svv, st%sw, st%sww, st%suv)
    st%nsamp = st%nsamp + 1
  end subroutine stats_accumulate

  ! One thread per y-node; serial reduction over the (ix,iz) plane (no race --
  ! each thread owns its node j). Bare device arrays (NVFORTRAN-0155 workaround).
  ! [R2C] pu,pv,pw are REAL(wp); real(...,dp) is a wp->dp promotion (dp
  ! accumulation policy), not a complex->real projection.  fac (the GLOBAL
  ! 1/(nxd*nzd)) is supplied by the caller so the same kernel serves the single-
  ! GPU full plane and a distributed z-slab (which loops only its local planes).
  ! The LOW PARALLELISM (ny threads for the whole GPU) is a
  ! DELIBERATE trade, not an oversight: the fixed per-thread (ix,iz) summation
  ! order is what keeps the serial<->parallel statistics bit-exact (the oracle
  ! gates), and at stats cadence the kernel amortises to 0.01-0.1% of the
  ! step. Do NOT parallelise the plane reduction without re-baselining every
  ! bit gate. (Same trade in accp_kernel below.)
  subroutine accum_kernel(n, nxd, nzd, fac, pu, pv, pw, su, suu, sv, svv, sw, sww, suv)
    integer(ip), intent(in) :: n, nxd, nzd
    real(dp),    intent(in) :: fac
    real(tp),    device     :: pu(0:n,nxd,nzd), pv(0:n,nxd,nzd), pw(0:n,nxd,nzd)
    real(dp),    device     :: su(0:n), suu(0:n), sv(0:n), svv(0:n)
    real(dp),    device     :: sw(0:n), sww(0:n), suv(0:n)
    integer(ip) :: j, ix, iz
    real(dp)    :: u_, v_, w_
    real(dp)    :: mu, muu, mv, mvv, mw, mww, muv
    !$cuf kernel do(1) <<<*,*>>>
    do j = 0, n
       mu=0.0_dp; muu=0.0_dp; mv=0.0_dp; mvv=0.0_dp
       mw=0.0_dp; mww=0.0_dp; muv=0.0_dp
       do iz = 1, nzd
          do ix = 1, nxd
             u_ = real(pu(j,ix,iz), dp)
             v_ = real(pv(j,ix,iz), dp)
             w_ = real(pw(j,ix,iz), dp)
             mu  = mu  + u_;     muu = muu + u_*u_
             mv  = mv  + v_;     mvv = mvv + v_*v_
             mw  = mw  + w_;     mww = mww + w_*w_
             muv = muv + u_*v_
          end do
       end do
       su(j)  = su(j)  + mu*fac
       suu(j) = suu(j) + muu*fac
       sv(j)  = sv(j)  + mv*fac
       svv(j) = svv(j) + mvv*fac
       sw(j)  = sw(j)  + mw*fac
       sww(j) = sww(j) + mww*fac
       suv(j) = suv(j) + muv*fac
    end do
  end subroutine accum_kernel

  !---------------------------------------------------------------------------
  ! Add one PRESSURE sample. pp: physical pressure at (y_GLC,x,z),
  ! (0:n,nxd,nzd) on the dealiased grid -- same grid/normalisation as the velocity
  ! accumulate, so it must be called alongside stats_accumulate (which owns nsamp).
  ! Accumulates the plane-mean sums sp += <p>_xz, spp += <p^2>_xz; nsamp is NOT
  ! touched here. Distributed: each rank adds its local-plane partials; the GLOBAL
  ! 1/(nxd*nzd) normaliser makes the cross-rank sum (par_stats_finalize_p) exact.
  subroutine stats_accum_p(st, n, nxd, nzd, pp)
    type(stats_t), intent(inout) :: st
    integer(ip),   intent(in)    :: n, nxd, nzd     ! nzd = LOCAL z-plane count
    real(tp),      device        :: pp(0:n,nxd,nzd)
    real(dp) :: fac
    fac = 1.0_dp/real(int(st%nxd,8)*int(st%nzd,8), dp)  ! GLOBAL plane size (int64: see stats_accumulate)
    call accp_kernel(n, nxd, nzd, fac, pp, st%sp, st%spp)
  end subroutine stats_accum_p

  ! One thread per y-node; serial reduction over the (ix,iz) plane (no race).
  subroutine accp_kernel(n, nxd, nzd, fac, pp, sp, spp)
    integer(ip), intent(in) :: n, nxd, nzd
    real(dp),    intent(in) :: fac
    real(tp),    device     :: pp(0:n,nxd,nzd)
    real(dp),    device     :: sp(0:n), spp(0:n)
    integer(ip) :: j, ix, iz
    real(dp)    :: p_, mp, mpp
    !$cuf kernel do(1) <<<*,*>>>
    do j = 0, n
       mp = 0.0_dp; mpp = 0.0_dp
       do iz = 1, nzd
          do ix = 1, nxd
             p_  = real(pp(j,ix,iz), dp)
             mp  = mp  + p_
             mpp = mpp + p_*p_
          end do
       end do
       sp(j)  = sp(j)  + mp*fac
       spp(j) = spp(j) + mpp*fac
    end do
  end subroutine accp_kernel

  !---------------------------------------------------------------------------
  ! Reduce to mean profiles + Reynolds stresses (host dp, at the GLC nodes).
  subroutine stats_finalize(st, n, mean_u, mean_v, mean_w, uu, vv, ww, uv)
    type(stats_t), intent(in)  :: st
    integer(ip),   intent(in)  :: n
    real(dp),      intent(out) :: mean_u(0:n), mean_v(0:n), mean_w(0:n)
    real(dp),      intent(out) :: uu(0:n), vv(0:n), ww(0:n), uv(0:n)
    real(dp) :: su(0:n), suu(0:n), sv(0:n), svv(0:n), sw(0:n), sww(0:n), suv(0:n)
    su  = st%su;  suu = st%suu          ! D2H copies
    sv  = st%sv;  svv = st%svv
    sw  = st%sw;  sww = st%sww
    suv = st%suv
    call stats_combine(st%nsamp, n, su, suu, sv, svv, sw, sww, suv, &
                       mean_u, mean_v, mean_w, uu, vv, ww, uv)
  end subroutine stats_finalize

  !---------------------------------------------------------------------------
  ! Host combine of accumulated per-sample plane-mean SUMS (su = sum_s <u>_xz,
  ! etc.) into the time-mean profiles + Reynolds stresses.  Factored out of
  ! stats_finalize so the distributed path (par_stats_mod) reduces su.. across
  ! ranks first and then reuses the IDENTICAL math -- ONE source of truth.  The
  ! single-GPU result is bit-identical to the previous in-lined version (same
  ! operations, same order).
  pure subroutine stats_combine(nsamp, n, su, suu, sv, svv, sw, sww, suv, &
                                mean_u, mean_v, mean_w, uu, vv, ww, uv)
    integer(ip), intent(in)  :: nsamp, n
    real(dp),    intent(in)  :: su(0:n), suu(0:n), sv(0:n), svv(0:n)
    real(dp),    intent(in)  :: sw(0:n), sww(0:n), suv(0:n)
    real(dp),    intent(out) :: mean_u(0:n), mean_v(0:n), mean_w(0:n)
    real(dp),    intent(out) :: uu(0:n), vv(0:n), ww(0:n), uv(0:n)
    real(dp) :: rn
    rn = 1.0_dp/real(max(nsamp,1), dp)
    mean_u = su*rn;  mean_v = sv*rn;  mean_w = sw*rn
    uu = suu*rn - mean_u**2
    vv = svv*rn - mean_v**2
    ww = sww*rn - mean_w**2
    uv = suv*rn - mean_u*mean_v
  end subroutine stats_combine

  !---------------------------------------------------------------------------
  ! Pressure reduce. Single-GPU: D2H sp/spp then combine. The
  ! distributed path (par_stats_finalize_p) reduces sp/spp across ranks first and
  ! calls the IDENTICAL stats_combine_p. mean_p carries the k=0 gauge constant;
  ! p_var = <p^2> - <p>^2 is gauge-independent (the physically meaningful one).
  subroutine stats_finalize_p(st, n, mean_p, p_var)
    type(stats_t), intent(in)  :: st
    integer(ip),   intent(in)  :: n
    real(dp),      intent(out) :: mean_p(0:n), p_var(0:n)
    real(dp) :: sp(0:n), spp(0:n)
    sp = st%sp;  spp = st%spp                    ! D2H copies
    call stats_combine_p(st%nsamp, n, sp, spp, mean_p, p_var)
  end subroutine stats_finalize_p

  pure subroutine stats_combine_p(nsamp, n, sp, spp, mean_p, p_var)
    integer(ip), intent(in)  :: nsamp, n
    real(dp),    intent(in)  :: sp(0:n), spp(0:n)
    real(dp),    intent(out) :: mean_p(0:n), p_var(0:n)
    real(dp) :: rn
    rn = 1.0_dp/real(max(nsamp,1), dp)
    mean_p = sp*rn
    p_var  = spp*rn - mean_p**2
  end subroutine stats_combine_p

  !---------------------------------------------------------------------------
  ! Mean wall shear |du0/dy| averaged over both walls, from the (0,0) mode via
  ! the yops facade (Cheb coeffs: du0/dy(+1) = sum_k a_k k^2, du0/dy(-1) =
  ! sum_k a_k (-1)^{k+1} k^2; FD mode: one-sided stencil dots on point values).
  pure function wall_shear(n, u0) result(g)
    integer(ip), intent(in) :: n
    real(dp),    intent(in) :: u0(0:n)
    real(dp)    :: g, gp, gm
    ! wall sums from the yops facade. gm = Du(-1), gp = Du(+1).
    call yop_wall_deriv(n, u0, gm, gp)
    g = 0.5_dp*(abs(gp) + abs(gm))
  end function wall_shear

  ! Re_tau = u_tau*delta/nu = sqrt(Re*|du0/dy|_wall)  (delta=1, nu=1/Re).
  pure function re_tau(n, Re, u0) result(ret)
    integer(ip), intent(in) :: n
    real(dp),    intent(in) :: Re, u0(0:n)
    real(dp) :: ret
    ret = sqrt(Re*wall_shear(n, u0))
  end function re_tau

  !---------------------------------------------------------------------------
  ! max | i kx u + D_y v + i kz w | over all spectral coefficients (zero iff
  ! divergence-free).  dvh is caller-supplied scratch (0:n,nx,nz).
  ! The |div| field is never materialised: abs+max fuse into a scalar
  ! reduction in the same kernel, which saves a magnitude scratch array
  ! (8 B/pt whenever div_every > 0) and one full-array write+read.
  subroutine divergence_check(n, nx, nz, kx, kz, uhat, vhat, what, dvh, maxdiv)
    integer(ip), intent(in)  :: n, nx, nz
    real(wp),    device      :: kx(nx), kz(nz)
    complex(tp), device      :: uhat(0:n,nx,nz), what(0:n,nx,nz)
    complex(wp), device      :: vhat(0:n,nx,nz)
    complex(wp), device      :: dvh(0:n,nx,nz)            ! scratch: D_y v
    real(dp),    intent(out) :: maxdiv
    integer(ip) :: j, ix, iz
    complex(wp) :: d
    real(wp)    :: mx
    call y_deriv_c(n+1, nx*nz, vhat, dvh)                 ! D_y v (y-basis dispatch)
    mx = 0.0_wp
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz
       do ix = 1, nx
          do j = 0, n
             d = cmplx(0.0_wp, kx(ix), wp)*uhat(j,ix,iz) + dvh(j,ix,iz) &
                 + cmplx(0.0_wp, kz(iz), wp)*what(j,ix,iz)
             mx = max(mx, abs(d))
          end do
       end do
    end do
    maxdiv = real(mx, dp)
  end subroutine divergence_check

  !---------------------------------------------------------------------------
  ! imag_max_c  --  max |Im| over three complex PHYSICAL fields.
  ! A C2C-carried real field should have |Im| at FFT round-off
  ! (~1e-13 relative); growth flags a Hermitian-symmetry leak (the failure mode
  ! of the unmasked-Nyquist bug). Feed it the ws%pu/pv/pw that monitor_fmax /
  ! nl_evaluate already populate -- a reduction only, no extra transform.
  subroutine imag_max_c(ny, nx, nz, u, v, w, vmax)
    integer(ip), intent(in)  :: ny, nx, nz
    complex(wp), device      :: u(ny,nx,nz), v(ny,nx,nz), w(ny,nx,nz)
    real(dp),    intent(out) :: vmax
    integer(ip) :: j, ix, iz
    real(dp)    :: f
    vmax = 0.0_dp
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz; do ix = 1, nx; do j = 1, ny
       f = max(abs(real(aimag(u(j,ix,iz)),dp)), abs(real(aimag(v(j,ix,iz)),dp)), &
               abs(real(aimag(w(j,ix,iz)),dp)))
       vmax = max(vmax, f)
    end do; end do; end do
  end subroutine imag_max_c

  !---------------------------------------------------------------------------
  ! field_sumabs_c  --  SUM of |Re| + |Im| over a complex field (watchdog).
  ! A SUM propagates NaN (NaN + x = NaN) and Inf reliably; a MAX-reduction
  ! silently SKIPS them (NaN comparisons are false), which is why the
  ! [div]/CFL columns can print 0.00 over a fully poisoned state.
  ! Finiteness test on the result: (s == s) .and. (s < huge).
  subroutine field_sumabs_c(ny, nx, nz, q, s)
    integer(ip), intent(in)  :: ny, nx, nz
    complex(wp), device      :: q(ny,nx,nz)
    real(dp),    intent(out) :: s
    integer(ip) :: j, ix, iz
    s = 0.0_dp
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nz; do ix = 1, nx; do j = 1, ny
       s = s + abs(real(q(j,ix,iz), dp)) + abs(real(aimag(q(j,ix,iz)), dp))
    end do; end do; end do
  end subroutine field_sumabs_c

  !---------------------------------------------------------------------------
  ! kx=0 z-Hermitian probes. A real physical field requires the kx=0
  ! spectral columns to satisfy  A(y,0,+kz) = conj(A(y,0,-kz)); the solver
  ! enforces this only at the IC, so a slow leak (roundoff asymmetry feeding
  ! back through the nonlinear terms, or a future transform bug) would grow
  ! silently. hsym_check_c returns the max pair asymmetry; hsym_fix_c projects
  ! back onto the symmetric subspace (conjugate-pair average). Both touch ONLY
  ! local column 1 (the caller guarantees it holds global kx=0) -- O(ny*nz)
  ! work, no communication. Pairs are k=2..nz/2 vs nz+2-k; k=1 (kz=0) and the
  ! z-Nyquist (self-conjugate) are left alone.
  !---------------------------------------------------------------------------
  subroutine hsym_check_c(n, nkxl, nz, A, asym)
    integer(ip), intent(in)  :: n, nkxl, nz
    complex(wp), device      :: A(0:n, nkxl, nz)
    real(dp),    intent(out) :: asym
    integer(ip) :: j, k
    real(dp) :: am
    am = 0.0_dp
    !$cuf kernel do(2) <<<*,*>>>
    do k = 2, nz/2
       do j = 0, n
          am = max(am, abs(cmplx(A(j,1,k) - conjg(A(j,1,nz+2-k)), kind=dp)))
       end do
    end do
    asym = am
  end subroutine hsym_check_c

  subroutine hsym_fix_c(n, nkxl, nz, A)
    integer(ip), intent(in) :: n, nkxl, nz
    complex(wp), device     :: A(0:n, nkxl, nz)
    integer(ip) :: j, k
    complex(wp) :: avg
    !$cuf kernel do(2) <<<*,*>>>
    do k = 2, nz/2
       do j = 0, n
          avg = 0.5_wp*(A(j,1,k) + conjg(A(j,1,nz+2-k)))
          A(j,1,k)      = avg
          A(j,1,nz+2-k) = conjg(avg)
       end do
    end do
  end subroutine hsym_fix_c

end module stats_mod
