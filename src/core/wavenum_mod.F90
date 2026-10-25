!===============================================================================
! wavenum_mod.F90  --  THE canonical C2C Fourier wavenumber builder.
!
! Physical wavenumbers in cuFFT order for a periodic direction of length L with
! nm (even) points:
!     index i = 1..nm  <->  integer mode  m = i-1            for i-1 <= nm/2,
!                                         m = i-1-nm          otherwise,
!     k(i) = (2*pi/L) * m.
! The Nyquist slot (i = nm/2+1) carries m = +nm/2; by solver convention that
! mode is KEPT IDENTICALLY ZERO in the state (mask_nyquist in nonlinear_mod),
! so its k value only affects the (well-conditioned, unused) solve at that
! column -- keeping the physical +nm/2 value there is deliberate.
!
! WHY THIS MODULE EXISTS: building BARE INTEGER mode indices (no 2*pi/L factor)
! silently hardwires the box to 2*pi x 2*pi while the CFL uses the namelist
! Lx/Lz -- dynamics and CFL then disagree, and every Lx/Lz setting is ignored by
! the spectral operators. A unit suite can be blind to that, because tests tend
! to build the same integer wavenumbers (self-consistent 2*pi boxes) and the
! laminar validation lives in the (0,0) mode (box-independent), so
! test_box_scaling.F90 pins the scaling against the independent analytic
! definition k = 2*pi*m/L at Lx /= 2*pi.
!
! Pure host math, no CUDA. Build:  nvfortran -c wavenum_mod.F90
!===============================================================================
module wavenum_mod
  use kinds, only: dp, ip
  implicit none
  private
  public :: build_wavenumbers, build_wavenumbers_r2c

  real(dp), parameter :: TWOPI = 2.0_dp*acos(-1.0_dp)

contains

  !---------------------------------------------------------------------------
  ! k(1:nm) = (2*pi/L) * (cuFFT-order integer modes). nm must be even.
  !---------------------------------------------------------------------------
  subroutine build_wavenumbers(nm, L, k)
    integer(ip), intent(in)  :: nm
    real(dp),    intent(in)  :: L
    real(dp),    intent(out) :: k(nm)
    integer(ip) :: i, m
    real(dp)    :: scale
    if (mod(nm,2) /= 0) error stop 'build_wavenumbers: nm must be even'
    if (L <= 0.0_dp)    error stop 'build_wavenumbers: L must be positive'
    scale = TWOPI/L
    do i = 1, nm
       m = i - 1
       if (m > nm/2) m = m - nm
       k(i) = scale*real(m, dp)
    end do
  end subroutine build_wavenumbers

  !---------------------------------------------------------------------------
  ! build_wavenumbers_r2c  --  THE canonical R2C x-axis builder. [R2C]
  !
  ! For a REAL transform in x, only the non-negative kx half is stored:
  !     index i = 1..nkx,  nkx = nm/2 + 1,  integer mode m = i-1 (0..nm/2),
  !     k(i) = (2*pi/L) * (i-1).
  ! MONOTONIC, NO cuFFT fold (there are no negative-kx slots -- the negative half
  ! is the Hermitian conjugate and is not stored). nm is the PHYSICAL point count
  ! (even); k is nkx long. This matches build_wavenumbers' non-negative half
  ! exactly (the C2C fold m>nm/2 -> m-nm never triggers for m <= nm/2). The
  ! x-Nyquist mode (m = nm/2) sits at the last slot i = nkx and is kept zero in
  ! the state by convention (mask_nyquist), as in the C2C case.
  !
  ! Use this for the R2C x direction; build_wavenumbers (full, folded) stays the
  ! builder for z (still a full complex axis under R2C).
  !---------------------------------------------------------------------------
  subroutine build_wavenumbers_r2c(nm, L, k)
    integer(ip), intent(in)  :: nm                 ! PHYSICAL x point count (even)
    real(dp),    intent(in)  :: L
    real(dp),    intent(out) :: k(nm/2+1)           ! non-negative half, nkx long
    integer(ip) :: i
    real(dp)    :: scale
    if (mod(nm,2) /= 0) error stop 'build_wavenumbers_r2c: nm must be even'
    if (L <= 0.0_dp)    error stop 'build_wavenumbers_r2c: L must be positive'
    scale = TWOPI/L
    do i = 1, nm/2 + 1
       k(i) = scale*real(i-1, dp)
    end do
  end subroutine build_wavenumbers_r2c

end module wavenum_mod
