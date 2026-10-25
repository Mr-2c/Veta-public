! init_mod.F90  --  initial conditions for the channel DNS.  Host build -> device.
! [R2C path] spectral x-extent is nkx = nx/2+1 (the non-negative kx half; the
! negative half is the Hermitian conjugate and is NOT stored).
!
! init_laminar    : laminar Poiseuille base, zero fluctuations.
! init_turbulent  : laminar base + a divergence-free, BC-satisfying perturbation in
!                   (v, eta) to trip transition.
!
! WHY THIS IS DIVERGENCE-FREE AND BC-CORRECT BY CONSTRUCTION:
!   * The KMM state is (v, eta); u,w are recovered from continuity, so ANY (v,eta)
!     gives a solenoidal velocity -- no projection needed.
!   * v is drawn in the Shen BIHARMONIC basis (psi_m = T_m + a_m T_{m+2} + b_m T_{m+4},
!     each satisfying v = Dv = 0 at the walls); eta in the Shen DIRICHLET basis
!     (phi_m = T_m - T_{m+2}, each eta = 0 at the walls). Converting Shen->Chebyshev
!     (shen_bih_to_cheb / shen_to_cheb) preserves those wall conditions exactly.
!
! [R2C] REALNESS -- the one thing R2C does NOT give for free. A real physical field
! has full spectrum F(kx,kz) = conj(F(-kx,-kz)). In the kx>=0 half storage:
!   * kx>0 columns are FREE: their conjugate partner at -kx is implied, never stored,
!     so the random content stands as-is (it represents itself + its implied mate).
!   * the kx=0 PLANE has both (0,kz) and (0,-kz) stored, so it MUST be made
!     Hermitian-in-z:  q(0,kz) <- 1/2(q(0,kz) + conj(q(0,-kz)))  ("the kx=0 plane
!     stays C2C in z"). This is the ONLY symmetrisation needed (the C2C path
!     symmetrised the full grid; here only the kx=0 plane).
!   * (0,0) and the x/z-Nyquist columns are kept zero.
! The Shen->Cheb shapes are real, so symmetrising a combination of them on the kx=0
! plane keeps the wall BCs. Modelled in numpy before coding (realness round-trip
! 5e-16; intensity weight exact).
!
! The mean (0,0) is carried by u0(y) (laminar parabola for the target bulk velocity
! U_b = Qtar: u_c = 1.5 Qtar) and w0 = 0; v(0,0) = eta(0,0) = 0.
!
! eps  : the EXACT volume-rms perturbation intensity, eps = q_rms =
!        sqrt(<u'^2 + v'^2 + w'^2>) in units of U_b. [R2C] the two-sided Parseval
!        sum over the full grid becomes a HALF sum with the interior kx>0 columns
!        WEIGHTED BY 2 (each stands for itself + its implied -kx conjugate); the
!        kx=0 column and the x-Nyquist count once. Same physical energy => same
!        qrms => same eps normalisation as the C2C build. Per mode,
!        |u|^2+|w|^2 = (|Dv|^2+|eta|^2)/k^2 exactly.
! seed : reproducible RNG seed (per build).
! kcut : integer mode radius; only modes with mx^2 + mz^2 <= kcut^2 are seeded.
! kxh, kzh : PHYSICAL wavenumbers; [R2C] kxh is nkx-long (non-negative half).
!
! SLAB-LOCAL GENERATION. The RNG is COUNTER-BASED (splitmix64 hash of
! (seed, global mx, folded mz, coefficient, component)): every coefficient is a
! pure function of GLOBAL indices, independent of loop order, rank count, and
! which rank evaluates it. ic_fill_slab therefore fills any kx-slab
! [kx0, kx0+nkxl) of the global field bit-identically to a full-grid fill --
! each rank builds ONLY its slab rather than replicating the whole grid on
! every rank (2 full-grid host + 2 device copies = OOM for any >~1 G-pt fresh
! start). Normalisation is split: ic_fill_slab returns the slab's PARTIAL
! energy; the caller sums (par_reduce_sum in parallel) and scales.
! CAVEAT: bit-identity across P holds for the RAW fill
! ONLY. The shipped state is hv*scale, and scale inherits the E2 reduction's
! rank-count-dependent summation order (serial sums sequentially; parallel
! sums per-slab partials then MPI-reduces), so a FRESH-start IC differs
! across P in the last ulps. Never build a cross-P bit gate on a fresh IC --
! seed from a checkpoint (run_validate already feeds both rungs ONE
! full-grid build_ic for exactly this reason).
!
! Build is host/dp (one-time); only the final uploads touch the device. wp=dp now.
! Build (after kinds, cheb_shen, cheb_bih):  nvfortran -cuda -c init_mod.F90

module init_mod
  use kinds,     only: wp, dp, ip
  use cudafor
  use cheb_shen, only: shen_to_cheb, cheb_deriv
  use cheb_bih,  only: shen_bih_to_cheb
  use yops_mod,  only: cheb_l2int => yop_l2int    ! the y-op facade
  use fdops_mod, only: fd_apply_band              ! FD IC energy: D1 matvec
  implicit none
  private
  public :: init_laminar, init_turbulent
  public :: ic_fill_slab, ic_mean_u0       ! slab-local IC building blocks
  public :: ic_fill_slab_fd, ic_mean_u0_fd ! FD point-value IC twins
  public :: init_turbulent_fd              ! serial full-grid FD wrapper

  real(dp), parameter :: QC0 = 0.75_dp     ! laminar: u0 = QC0*Qtar (T0 - T2), u_c = 1.5 Qtar

contains

  !---------------------------------------------------------------------------
  ! init_laminar -- only the (0,0) mean column is nonzero; representation-
  ! independent (kx=0 is column 1 in both C2C and R2C). nx here is the spectral
  ! x-extent (= nkx on the R2C path); the fluctuation arrays are just zeroed.
  !---------------------------------------------------------------------------
  subroutine init_laminar(n, nx, nz, Qtar, v, eta, u0, w0)
    integer(ip), intent(in) :: n, nx, nz
    real(dp),    intent(in) :: Qtar
    complex(wp), device     :: v(0:n,nx,nz), eta(0:n,nx,nz), u0(0:n), w0(0:n)
    complex(wp) :: hu0(0:n)
    v   = (0.0_wp, 0.0_wp);  eta = (0.0_wp, 0.0_wp);  w0 = (0.0_wp, 0.0_wp)
    hu0 = (0.0_wp, 0.0_wp)
    hu0(0) = cmplx( QC0*Qtar, 0.0_dp, wp)
    hu0(2) = cmplx(-QC0*Qtar, 0.0_dp, wp)
    u0 = hu0
  end subroutine init_laminar

  !---------------------------------------------------------------------------
  ! init_turbulent  [R2C]. nkx = nx/2+1 is the spectral x-extent; kxh is nkx-long.
  !---------------------------------------------------------------------------
  subroutine init_turbulent(n, nkx, nz, kxh, kzh, Qtar, eps, seed, kcut, v, eta, u0, w0)
    integer(ip), intent(in) :: n, nkx, nz, seed, kcut
    real(dp),    intent(in) :: kxh(nkx), kzh(nz)      ! [R2C] kxh is nkx-long (kx>=0)
    real(dp),    intent(in) :: Qtar, eps
    complex(wp), device     :: v(0:n,nkx,nz), eta(0:n,nkx,nz), u0(0:n), w0(0:n)
    complex(dp), allocatable :: hv(:,:,:), he(:,:,:)
    complex(wp) :: hu0(0:n)
    real(dp) :: E2, qrms, scale

    ! full-grid fill = the slab fill at (kx0=0, nkxl=nkx); bit-identical
    ! per column to any slab decomposition of the same seed (counter-based RNG).
    allocate(hv(0:n,nkx,nz), he(0:n,nkx,nz))
    call ic_fill_slab(n, nkx, nz, 0_ip, nkx, kxh, kzh, seed, kcut, hv, he, E2)

    qrms = sqrt(max(E2, 0.0_dp)/2.0_dp)
    if (qrms > 0.0_dp) then
       scale = eps/qrms
       write(*,'(a,es10.3,a,es10.3)') '   [init] raw perturbation q_rms = ', qrms, &
            '  -> normalised to eps = ', eps
    else
       scale = 0.0_dp
       write(*,'(a)') '   [init] WARNING: empty perturbation band (check kcut); state unperturbed'
    end if
    hv = scale*hv;  he = scale*he
    v = hv;  eta = he                                   ! H2D (wp = dp)

    call ic_mean_u0(n, Qtar, hu0)                       ! laminar mean base
    u0 = hu0;  w0 = (0.0_wp,0.0_wp)

    deallocate(hv, he)
  end subroutine init_turbulent

  !---------------------------------------------------------------------------
  ! ic_fill_slab -- build the UNNORMALISED perturbation on one kx-slab
  ! [kx0, kx0+nkxl) of the global (nkx,nz) spectral grid, HOST arrays, and
  ! return this slab's contribution to the Parseval intensity sum (E2part).
  ! Rank/order-invariant: every coefficient comes from ic_rand01 keyed on
  ! GLOBAL indices. The kx=0 z-symmetrisation runs only on the slab that holds
  ! kx=0 (that slab holds ALL kz -- y and z are never split). The caller
  ! normalises: qrms = sqrt(sum(E2part)/2), scale = eps/qrms.
  !---------------------------------------------------------------------------
  subroutine ic_fill_slab(n, nkx, nz, kx0, nkxl, kxh, kzh, seed, kcut, hv, he, E2part)
    integer(ip), intent(in)  :: n, nkx, nz, kx0, nkxl, seed, kcut
    real(dp),    intent(in)  :: kxh(nkx), kzh(nz)      ! GLOBAL wavenumbers
    complex(dp), intent(out) :: hv(0:n,nkxl,nz), he(0:n,nkxl,nz)
    real(dp),    intent(out) :: E2part
    real(dp) :: cvr(0:n-4), cvi(0:n-4), cer(0:n-2), cei(0:n-2)
    real(dp) :: avr(0:n), avi(0:n), aer(0:n), aei(0:n)
    real(dp) :: dvr(0:n), dvi(0:n)
    complex(dp) :: cw(0:n)
    complex(dp) :: t0v(0:n,nz), t0e(0:n,nz)            ! kx=0 plane symmetrise scratch
    integer(ip) :: il, iz, mx, mz, icz, m, k2
    real(dp) :: spec, dec, r1, r2, k0
    real(dp) :: k2p, qq, wgt

    hv = (0.0_dp,0.0_dp);  he = (0.0_dp,0.0_dp)
    k0 = max(real(kcut,dp)/2.0_dp, 1.0_dp)

    ! Phase 1: seed this slab's columns. mx GLOBAL (>= 0, no sign fold in x);
    ! mz folds. Skip (0,0), the band cutoff, and the x/z-Nyquist columns.
    do iz = 1, nz
       mz = iz-1;  if (mz > nz/2) mz = mz - nz
       do il = 1, nkxl
          mx = kx0 + il - 1                            ! GLOBAL kx index
          k2 = mx*mx + mz*mz
          if (k2 == 0)                          cycle    ! (0,0): mean only
          if (k2 > kcut*kcut)                   cycle    ! outside the seeded band
          if (mx == nx_nyq(nkx) .or. abs(mz) == nz/2) cycle ! Nyquist: keep zero
          spec = exp(-real(k2,dp)/(2.0_dp*k0*k0))
          ! v: random Shen-biharmonic coeffs (clamped: v = Dv = 0)
          do m = 0, n-4
             dec = exp(-real(m,dp)/3.0_dp)
             r1 = ic_rand01(seed, mx, mz, m, 0)
             r2 = ic_rand01(seed, mx, mz, m, 1)
             cvr(m) = (2.0_dp*r1-1.0_dp)*spec*dec
             cvi(m) = (2.0_dp*r2-1.0_dp)*spec*dec
          end do
          call shen_bih_to_cheb(n, cvr, avr)
          call shen_bih_to_cheb(n, cvi, avi)
          do m = 0, n
             hv(m,il,iz) = cmplx(avr(m), avi(m), dp)
          end do
          ! eta: random Shen-Dirichlet coeffs (eta = 0 at walls)
          do m = 0, n-2
             dec = exp(-real(m,dp)/3.0_dp)
             r1 = ic_rand01(seed, mx, mz, m, 2)
             r2 = ic_rand01(seed, mx, mz, m, 3)
             cer(m) = (2.0_dp*r1-1.0_dp)*spec*dec
             cei(m) = (2.0_dp*r2-1.0_dp)*spec*dec
          end do
          call shen_to_cheb(n, cer, aer)
          call shen_to_cheb(n, cei, aei)
          do m = 0, n
             he(m,il,iz) = cmplx(aer(m), aei(m), dp)
          end do
       end do
    end do

    ! Phase 2 [R2C]: Hermitian-symmetrise ONLY the kx=0 plane in z so the physical
    ! field is real. Runs only on the slab holding global kx=0 (it holds all kz).
    if (kx0 == 0) then
       t0v = hv(:,1,:);  t0e = he(:,1,:)               ! kx=0 plane is local ix=1
       do iz = 1, nz
          icz = mod(nz-(iz-1), nz) + 1                 ! index of -kz
          hv(:,1,iz) = 0.5_dp*(t0v(:,iz) + conjg(t0v(:,icz)))
          he(:,1,iz) = 0.5_dp*(t0e(:,iz) + conjg(t0e(:,icz)))
       end do
    end if

    ! Phase 3 [R2C] intensity PARTIAL sum over THIS slab's columns, with the
    ! GLOBAL half-sum weights: interior kx>0 columns (global 1 <= mx <= nkx-2)
    ! weighted by 2, the kx=0 column by 1; x-Nyquist and (0,0) are zero (moot).
    E2part = 0.0_dp
    do iz = 1, nz
       mz = iz-1;  if (mz > nz/2) mz = mz - nz
       do il = 1, nkxl
          mx = kx0 + il - 1
          k2 = mx*mx + mz*mz
          if (k2 == 0) cycle
          if (k2 > kcut*kcut) cycle                    ! unseeded -> exact 0 contribution
          k2p = kxh(mx+1)**2 + kzh(iz)**2
          if (k2p <= 0.0_dp) cycle
          wgt = 1.0_dp
          if (mx >= 1 .and. mx <= nkx-2) wgt = 2.0_dp  ! interior kx>0 doubled
          avr = real(hv(:,il,iz));  avi = aimag(hv(:,il,iz))
          call cheb_deriv(n, avr, dvr)
          call cheb_deriv(n, avi, dvi)
          cw = cmplx(dvr, dvi, dp)
          qq = wgt*cheb_l2int(n, cw)/k2p               ! |Dv|^2 / k^2
          qq = qq + wgt*cheb_l2int(n, he(:,il,iz))/k2p ! |eta|^2 / k^2
          qq = qq + wgt*cheb_l2int(n, hv(:,il,iz))     ! |v|^2
          E2part = E2part + qq
       end do
    end do
  end subroutine ic_fill_slab

  !---------------------------------------------------------------------------
  ! laminar mean base coefficients: u0 = QC0*Qtar*(T0 - T2), u_c = 1.5 Qtar.
  !---------------------------------------------------------------------------
  subroutine ic_mean_u0(n, Qtar, hu0)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: Qtar
    complex(wp), intent(out) :: hu0(0:n)
    hu0 = (0.0_wp,0.0_wp)
    hu0(0) = cmplx( QC0*Qtar, 0.0_dp, wp)
    hu0(2) = cmplx(-QC0*Qtar, 0.0_dp, wp)
  end subroutine ic_mean_u0

  !---------------------------------------------------------------------------
  ! counter-based RNG: splitmix64 hash chain over the key fields. Pure
  ! function of (seed, GLOBAL mx, folded mz, coefficient m, component 0..3):
  ! rank/order/decomposition-invariant by construction. Uniform in [0,1) with
  ! 53 random bits. (Signed 64-bit multiply wraps two's-complement on every
  ! target nvfortran supports -- the standard splitmix64 idiom in Fortran.)
  !---------------------------------------------------------------------------
  pure function ic_rand01(seed, mx, mz, m, comp) result(r)
    integer(ip), intent(in) :: seed, mx, mz, m, comp
    real(dp) :: r
    integer(8) :: h
    h = ic_mix64(int(seed, 8))
    h = ic_mix64(ieor(h, int(mx, 8)))
    h = ic_mix64(ieor(h, int(mz, 8)))
    h = ic_mix64(ieor(h, int(4*m + comp, 8)))
    r = real(iand(ishft(h, -11), int(Z'1FFFFFFFFFFFFF', 8)), dp) &
        * (1.0_dp/9007199254740992.0_dp)               ! / 2^53
  end function ic_rand01

  pure function ic_mix64(x) result(h)
    integer(8), intent(in) :: x
    integer(8) :: h
    h = x + int(Z'9E3779B97F4A7C15', 8)
    h = ieor(h, ishft(h, -30)) * int(Z'BF58476D1CE4E5B9', 8)
    h = ieor(h, ishft(h, -27)) * int(Z'94D049BB133111EB', 8)
    h = ieor(h, ishft(h, -31))
  end function ic_mix64

  !-- x-Nyquist mode index for an nkx half-axis: mx = nx/2 = nkx-1 --------------
  pure function nx_nyq(nkx) result(m)
    integer(ip), intent(in) :: nkx
    integer(ip) :: m
    m = nkx - 1                  ! nkx = nx/2+1 => nx/2 = nkx-1
  end function nx_nyq

  ! cheb_l2int and wint live in yops_mod (yop_l2int / yop_int_tj); cheb_l2int
  ! is use-renamed above so the call sites here read unchanged.

  !===========================================================================
  ! FD POINT-VALUE IC TWINS. Same counter-based RNG streams (identical
  ! (seed,mx,mz,m,comp) keys and decay/spec envelopes as ic_fill_slab), but the
  ! random coefficients feed a plain CHEBYSHEV-T series evaluated at the ygrid
  ! file's nodes (Clenshaw), with BC-EXACTNESS by ENVELOPE:
  ! v = (1-y^2)^2 * g  (v = Dv = 0 at walls, double root; the file
  ! guarantees walls EXACTLY +-1), eta = (1-y^2) * h. This is a DIFFERENT
  ! (equally random) field than the Cheb IC's Shen-basis draw -- FD runs start
  ! fresh rather than from a Chebyshev checkpoint, so cross-basis IC equality
  ! is a non-goal.
  ! The Parseval energy uses the FD instruments: Dv by the banded D1
  ! (fd_apply_band, the solver's own operator) and integrals by the quadrature
  ! weight dot (the file's w or the caller's trapz fallback).
  !===========================================================================

  !-- laminar mean base at the nodes: u0(y) = QC0*Qtar*(T0 - T2)(y), the same
  !-- FUNCTION as ic_mean_u0 (= 1.5*Qtar*(1 - y^2)), as point values -----------
  subroutine ic_mean_u0_fd(n, Qtar, yv, hu0)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: Qtar, yv(0:n)
    complex(wp), intent(out) :: hu0(0:n)
    integer(ip) :: j
    do j = 0, n
       hu0(j) = cmplx(QC0*Qtar*2.0_dp*(1.0_dp - yv(j)*yv(j)), 0.0_dp, wp)
    end do
  end subroutine ic_mean_u0_fd

  !-- slab-local FD perturbation fill; mirrors ic_fill_slab phase-for-phase ----
  ! yv/wq : nodes + quadrature weights (0:n); klb/d1b : the solver's D1 band.
  subroutine ic_fill_slab_fd(n, nkx, nz, kx0, nkxl, kxh, kzh, seed, kcut, &
                             yv, wq, klb, d1b, hv, he, E2part)
    integer(ip), intent(in)  :: n, nkx, nz, kx0, nkxl, seed, kcut, klb
    real(dp),    intent(in)  :: kxh(nkx), kzh(nz), yv(0:n), wq(0:n)
    real(dp),    intent(in)  :: d1b(2*klb+1, 0:n)
    complex(dp), intent(out) :: hv(0:n,nkxl,nz), he(0:n,nkxl,nz)
    real(dp),    intent(out) :: E2part
    real(dp) :: cvr(0:n-4), cvi(0:n-4), cer(0:n-2), cei(0:n-2)
    real(dp) :: fr(0:n), fi(0:n), dvr(0:n), dvi(0:n), env
    complex(dp) :: t0v(0:n,nz), t0e(0:n,nz)
    integer(ip) :: il, iz, mx, mz, icz, m, k2, j
    real(dp) :: spec, dec, r1, r2, k0, k2p, qq, wgt

    hv = (0.0_dp,0.0_dp);  he = (0.0_dp,0.0_dp)
    k0 = max(real(kcut,dp)/2.0_dp, 1.0_dp)

    ! Phase 1: identical mode/skip logic and RNG keys to ic_fill_slab
    do iz = 1, nz
       mz = iz-1;  if (mz > nz/2) mz = mz - nz
       do il = 1, nkxl
          mx = kx0 + il - 1
          k2 = mx*mx + mz*mz
          if (k2 == 0)                          cycle
          if (k2 > kcut*kcut)                   cycle
          if (mx == nx_nyq(nkx) .or. abs(mz) == nz/2) cycle
          spec = exp(-real(k2,dp)/(2.0_dp*k0*k0))
          do m = 0, n-4
             dec = exp(-real(m,dp)/3.0_dp)
             r1 = ic_rand01(seed, mx, mz, m, 0)
             r2 = ic_rand01(seed, mx, mz, m, 1)
             cvr(m) = (2.0_dp*r1-1.0_dp)*spec*dec
             cvi(m) = (2.0_dp*r2-1.0_dp)*spec*dec
          end do
          do m = 0, n-2
             dec = exp(-real(m,dp)/3.0_dp)
             r1 = ic_rand01(seed, mx, mz, m, 2)
             r2 = ic_rand01(seed, mx, mz, m, 3)
             cer(m) = (2.0_dp*r1-1.0_dp)*spec*dec
             cei(m) = (2.0_dp*r2-1.0_dp)*spec*dec
          end do
          do j = 0, n
             env = (1.0_dp - yv(j)*yv(j))
             hv(j,il,iz) = env*env*cmplx(ic_clenshaw(n-4, cvr, yv(j)), &
                                         ic_clenshaw(n-4, cvi, yv(j)), dp)
             he(j,il,iz) = env*cmplx(ic_clenshaw(n-2, cer, yv(j)), &
                                     ic_clenshaw(n-2, cei, yv(j)), dp)
          end do
       end do
    end do

    ! Phase 2 [R2C]: kx=0 z-Hermitian symmetrisation (identical to ic_fill_slab)
    if (kx0 == 0) then
       t0v = hv(:,1,:);  t0e = he(:,1,:)
       do iz = 1, nz
          icz = mod(nz-(iz-1), nz) + 1
          hv(:,1,iz) = 0.5_dp*(t0v(:,iz) + conjg(t0v(:,icz)))
          he(:,1,iz) = 0.5_dp*(t0e(:,iz) + conjg(t0e(:,icz)))
       end do
    end if

    ! Phase 3: Parseval partial sum with FD instruments (banded D1 + w-dot)
    E2part = 0.0_dp
    do iz = 1, nz
       mz = iz-1;  if (mz > nz/2) mz = mz - nz
       do il = 1, nkxl
          mx = kx0 + il - 1
          k2 = mx*mx + mz*mz
          if (k2 == 0) cycle
          if (k2 > kcut*kcut) cycle
          k2p = kxh(mx+1)**2 + kzh(iz)**2
          if (k2p <= 0.0_dp) cycle
          wgt = 1.0_dp
          if (mx >= 1 .and. mx <= nkx-2) wgt = 2.0_dp
          do j = 0, n
             fr(j) = real(hv(j,il,iz));  fi(j) = aimag(hv(j,il,iz))
          end do
          call fd_apply_band(n+1, klb, d1b, fr, dvr)
          call fd_apply_band(n+1, klb, d1b, fi, dvi)
          qq = wgt*dot_product(wq, dvr*dvr + dvi*dvi)/k2p            ! |Dv|^2 / k^2
          do j = 0, n
             fr(j) = real(he(j,il,iz));  fi(j) = aimag(he(j,il,iz))
          end do
          qq = qq + wgt*dot_product(wq, fr*fr + fi*fi)/k2p           ! |eta|^2 / k^2
          do j = 0, n
             fr(j) = real(hv(j,il,iz));  fi(j) = aimag(hv(j,il,iz))
          end do
          qq = qq + wgt*dot_product(wq, fr*fr + fi*fi)               ! |v|^2
          E2part = E2part + qq
       end do
    end do
  end subroutine ic_fill_slab_fd

  !-- serial full-grid FD wrapper (mirrors init_turbulent; slab = whole grid) --
  subroutine init_turbulent_fd(n, nkx, nz, kxh, kzh, Qtar, eps, seed, kcut, &
                               yv, wq, klb, d1b, v, eta, u0, w0)
    integer(ip), intent(in) :: n, nkx, nz, seed, kcut, klb
    real(dp),    intent(in) :: kxh(nkx), kzh(nz), Qtar, eps
    real(dp),    intent(in) :: yv(0:n), wq(0:n), d1b(2*klb+1, 0:n)
    complex(wp), device     :: v(0:n,nkx,nz), eta(0:n,nkx,nz), u0(0:n), w0(0:n)
    complex(dp), allocatable :: hv(:,:,:), he(:,:,:)
    complex(wp) :: hu0(0:n)
    real(dp) :: E2, qrms, scale
    allocate(hv(0:n,nkx,nz), he(0:n,nkx,nz))
    call ic_fill_slab_fd(n, nkx, nz, 0_ip, nkx, kxh, kzh, seed, kcut, &
                         yv, wq, klb, d1b, hv, he, E2)
    qrms = sqrt(max(E2, 0.0_dp)/2.0_dp)
    if (qrms > 0.0_dp) then
       scale = eps/qrms
       write(*,'(a,es10.3,a,es10.3)') '   [init] raw FD perturbation q_rms = ', &
            qrms, '  -> normalised to eps = ', eps
    else
       scale = 0.0_dp
       write(*,'(a)') '   [init] WARNING: empty perturbation band (check kcut); state unperturbed'
    end if
    hv = scale*hv;  he = scale*he
    v = hv;  eta = he
    call ic_mean_u0_fd(n, Qtar, yv, hu0)
    u0 = hu0;  w0 = (0.0_wp,0.0_wp)
    deallocate(hv, he)
  end subroutine init_turbulent_fd

  !-- Clenshaw: sum_{m=0}^{nm} c(m) T_m(y) -------------------------------------
  pure function ic_clenshaw(nm, c, y) result(s)
    integer(ip), intent(in) :: nm
    real(dp),    intent(in) :: c(0:nm), y
    real(dp) :: s, b1, b2, b
    integer(ip) :: m
    b1 = 0.0_dp;  b2 = 0.0_dp
    do m = nm, 1, -1
       b  = c(m) + 2.0_dp*y*b1 - b2
       b2 = b1;  b1 = b
    end do
    s = c(0) + y*b1 - b2
  end function ic_clenshaw

end module init_mod
