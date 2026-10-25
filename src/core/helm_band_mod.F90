!===============================================================================
! helm_band_mod.F90  --  O(n) banded Shen-Dirichlet Helmholtz solver (CPU/dp).
!
! Solves  (S + mu B) uhat = ftilde  for the Shen-Dirichlet Galerkin discretisation
! of  -u'' + mu u = f,  u(+-1)=0,  where S, B are the stiffness/mass matrices
! assembled by cheb_shen. This is the algorithm the batched GPU kernel
! (one mode per thread, helm_band_gpu_mod) implements; it is validated here on
! the CPU against the dense cheb_shen solve.
!
! WHY IT IS O(n).  The diagnostic test_helm_structure confirmed, straight from
! the validated assembly:
!   * B is tridiagonal per parity:  B(k,k)= 3pi/2 (k=0) or pi,  B(k,k+-2)= -pi/2.
!   * S is upper-triangular per parity with a CONSTANT TAIL:
!         S(k,k)   = 2pi(k+1)(k+2)         (diagonal)
!         S(k,j)   = 4pi(k+1)              for all j>k of the same parity.
! The constant tail makes M = S + mu B dense-upper per parity, but differencing
! row k with row k+2 scaled by r_k = S(k,k+2)/S(k+2,k+4) = (k+1)/(k+3) cancels
! the tail (4pi(k+1) - (k+1)/(k+3) 4pi(k+3) = 0), leaving a quasi-pentadiagonal
! band: one subdiagonal (from B), two superdiagonals. An O(n) forward-eliminate
! + back-substitute then solves it. Even/odd parities decouple completely.
!
! VARIABLE dt.  mu = k^2 + Re/(beta_k dt) enters only as a scalar; the band is
! rebuilt from the dt-independent structure arrays (sdiag,stail,bdiag,boff) plus
! mu at solve time. No stored factorisation -> changing dt is free.
!
! Pure host math, dp. NO CUDA. Build:
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran -c helm_band_mod.F90
!===============================================================================
module helm_band_mod
  use kinds,     only: dp, ip
  use cheb_shen, only: assemble_helmholtz
  implicit none
  private
  public :: helm_band_setup, helm_band_solve

contains

  !---------------------------------------------------------------------------
  ! Extract the O(n) structure arrays once per n.
  !   sdiag(k) = S(k,k)             (stiffness diagonal, from the mu=0 assembly)
  !   stail(k) = S(k,k+2)           (constant tail value of row k; 0 if no tail)
  !   bdiag(k) = B(k,k)             (mass diagonal, CLOSED FORM)
  !   boff     = B(k,k+2)           (uniform mass off-diagonal, CLOSED FORM)
  ! The stiffness side stays tied to the validated dense assembly (a SINGLE
  ! assembly -- its entries are exact, no cancellation). The mass side is the
  ! closed form from the header. Extracting it instead as
  ! B = M(mu=1) - M(mu=0) differences entries whose stiffness part is up to
  ! 2pi*(n-1)*n / pi ~ 4.6e5x larger at n=480, losing ~5.7 digits -- a 3.5e-11
  ! RELATIVE error in bdiag (measured; reproduced to the same digit by
  ! src/models/work_a6_mass_diag_model.py), i.e. 3.5e-11 in the DOMINANT term
  ! of every production Helmholtz solve (mu = 1/alpha ~ 6.5e8 makes mu*B the
  ! diagonal's big half at every k). Closed forms, derived independently of
  ! the assembly in the model file (phi_k = T_k - T_{k+2} with
  ! (T_a,T_b)_w = c_a*(pi/2)*delta_ab, c_0 = 2 else 1):
  !   B(k,k) = (c_k + c_{k+2})*pi/2  =  3pi/2 (k=0), pi (k>=1);  B(k,k+2) = -pi/2.
  ! Gate: test_helm_band's (n=480, mu=6.5e8, tol=1e-11) row -- the differenced
  ! extraction FAILS it at 3.5e-11, the closed form passes at the dense-vs-
  ! banded round-off floor (measured ~1.1-1.8e-12 there). Expect results to move at the 1e-11
  ! level vs pre-fix binaries: an accuracy IMPROVEMENT, not a regression --
  ! tolerance gates, not bit-identity, are the right check for this change.
  !---------------------------------------------------------------------------
  subroutine helm_band_setup(n, sdiag, stail, bdiag, boff)
    integer(ip), intent(in)  :: n
    real(dp),    intent(out) :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
    real(dp), parameter :: PI_L = 3.14159265358979323846264338327950288_dp
    real(dp) :: S(0:n-2,0:n-2)
    integer(ip) :: k, m
    ! Minimum-n refusal: below n=4 the Shen system degenerates
    ! (n=2: a single even mode, no odd system; n=3: no k+2 tail) and the solve's
    ! parity loop + the GPU twin's identical loop index out of bounds behind a
    ! DEAD guard (integer division (m-pstart)/2+1 gives L=1 at m=0). Every
    ! in-repo caller uses n >= 8; refuse loudly rather than read garbage.
    if (n < 4) error stop 'helm_band_setup: n < 4 unsupported (Shen system degenerate)'
    m = n-2
    call assemble_helmholtz(n, 0.0_dp, S)          ! stiffness only: exact entries
    do k = 0, m
       sdiag(k) = S(k,k)
       if (k+2 <= m) then
          stail(k) = S(k,k+2)
       else
          stail(k) = 0.0_dp
       end if
       bdiag(k) = merge(1.5_dp*PI_L, PI_L, k == 0)  ! B(k,k) closed form
    end do
    boff = -0.5_dp*PI_L                             ! B(k,k+2) closed form
  end subroutine helm_band_setup

  !---------------------------------------------------------------------------
  ! Solve (S + mu B) uhat = ftilde via parity-split row-differencing.
  ! ftilde is the Shen-projected RHS (f,phi_k)_w; uhat the Shen coefficients.
  !---------------------------------------------------------------------------
  subroutine helm_band_solve(n, mu, sdiag, stail, bdiag, boff, ftilde, uhat)
    integer(ip), intent(in)  :: n
    real(dp),    intent(in)  :: mu
    real(dp),    intent(in)  :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
    real(dp),    intent(in)  :: ftilde(0:n-2)
    real(dp),    intent(out) :: uhat(0:n-2)
    real(dp) :: sub(0:n-2), dia(0:n-2), up1(0:n-2), up2(0:n-2), rhs(0:n-2)
    real(dp) :: uloc(0:n-2)
    real(dp) :: r, fac
    integer(ip) :: m, pstart, L, i, k

    m = n-2
    do pstart = 0, 1
       L = (m - pstart)/2 + 1
       ! (dead guard for n=2 -- (-1)/2 = 0 gives L=1; unreachable since
       !  helm_band_setup refuses n < 4 [r3])
       if (L < 1) cycle

       ! Build the banded system in parity-local index i (k = pstart+2i).
       ! A row needs differencing only if it has a dense tail BEYOND column k+2,
       ! i.e. column k+4 exists (i+2 <= L-1, equivalently i <= L-3). The top two
       ! rows of each parity (i = L-2, L-1) are already in-band: the row at
       ! i = L-2 has its single tail entry at column k+2 (an ordinary up1
       ! super-diagonal), and the top row i = L-1 has no tail at all. Keeping
       ! them undifferenced also avoids stail(k+2)=0 in the ratio.
       do i = 0, L-1
          if (i >= 1) then; sub(i) = mu*boff; else; sub(i) = 0.0_dp; end if
          k = pstart + 2*i
          if (i <= L-3) then                         ! differenced row
             r = stail(k) / stail(k+2)               ! = (k+1)/(k+3), denom /= 0
             dia(i) = sdiag(k) + mu*bdiag(k) - r*(mu*boff)
             up1(i) = stail(k) + mu*boff - r*(sdiag(k+2) + mu*bdiag(k+2))
             up2(i) = -r*(mu*boff)
             rhs(i) = ftilde(k) - r*ftilde(k+2)
          else if (i == L-2) then                    ! in-band: keep the k+2 entry
             dia(i) = sdiag(k) + mu*bdiag(k)
             up1(i) = stail(k) + mu*boff             ! M(k,k+2); k+2 = m exists
             up2(i) = 0.0_dp
             rhs(i) = ftilde(k)
          else                                       ! i == L-1, top row
             dia(i) = sdiag(k) + mu*bdiag(k)
             up1(i) = 0.0_dp
             up2(i) = 0.0_dp
             rhs(i) = ftilde(k)
          end if
       end do

       ! Forward elimination of the single subdiagonal (fill stays within up2).
       do i = 1, L-1
          fac    = sub(i) / dia(i-1)
          dia(i) = dia(i) - fac*up1(i-1)
          up1(i) = up1(i) - fac*up2(i-1)
          rhs(i) = rhs(i) - fac*rhs(i-1)
       end do

       ! Back substitution (upper bandwidth 2).
       uloc(L-1) = rhs(L-1) / dia(L-1)
       if (L >= 2) uloc(L-2) = (rhs(L-2) - up1(L-2)*uloc(L-1)) / dia(L-2)
       do i = L-3, 0, -1
          uloc(i) = (rhs(i) - up1(i)*uloc(i+1) - up2(i)*uloc(i+2)) / dia(i)
       end do

       do i = 0, L-1
          uhat(pstart + 2*i) = uloc(i)
       end do
    end do
  end subroutine helm_band_solve

end module helm_band_mod
