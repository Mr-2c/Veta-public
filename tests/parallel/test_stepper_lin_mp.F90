! =============================================================================
! test_stepper_lin_mp.F90 -- validate that the LINEAR viscous fluctuation
! stepper (stepper_lin_mod: stepper_ev_setup + step_ev_linear) runs correctly on
! a kx-slab. Both routines are value-based / per-(kx,kz)-column (apply_L uses
! k2=kx^2+kz^2; helm_band and the influence solve are per-column kernels keyed on
! kx/kz with per-column precomputes), and y -- the solve direction -- is never
! decomposed. So on the slab each rank calls them UNCHANGED with (bnkx, local-kx,
! local-precomputes); there is NO communication. This test certifies that the
! per-column solve stack + the precompute setup are extent-agnostic, by comparing
! the slab against the single-GPU full field (the established oracle from
! test_stepper_ev) column-for-column.
!
! Two assertions:
!   TEST 1  precompute setup -- stepper_ev_setup on the slab must equal the full
!           setup's columns kxlo:kxhi (checked on its own so that a failure here
!           localizes to the setup rather than to the step).
!   TEST 2  step -- evolve a deterministic (eta,v) Ns steps with step_ev_linear on
!           the full field (oracle) and on the slab; the slab columns must match.
!
! dp-ONLY: the whole solve stack is dp (wp=dp in every shipped config; only the
! transform/comm tier tp is ever sp), so step_ev_linear is precision-invariant --
! there is no separate sp case. Matching must be bit-exact (the tolerance is just
! a NaN-guarded ceiling). The global (0,0) mode is zeroed in the input to match
! the physical DNS (v(0,0)=eta(0,0)=0; the (0,0)-column handling -- zero_col11 --
! is out of scope here); the kx>0 columns distributed across ranks carry the test.
!
! Run at P=1/4/8 (1-2 nodes): matching at every P establishes the solve stack is
! decomposition-independent.
! =============================================================================
program test_stepper_lin_mp
  use, intrinsic :: iso_fortran_env, only: real64, int64
  use kinds,           only: dp, ip
  use cudafor
  use decomp_mod,      only: decomp_t, decomp_setup
  use helm_band_mod,   only: helm_band_setup
  use stepper_lin_mod, only: stepper_ev_setup, step_ev_linear
  use par_mod
  implicit none

  ! --- problem size (small; nx physical -> nkx=nx/2+1 kx-modes to distribute) ---
  integer(ip),  parameter :: nx = 32, ny = 33, nz = 8
  integer(ip),  parameter :: n  = 32                 ! Chebyshev order (y: 0:n)
  real(dp),     parameter :: Re = 180.0_dp, dt = 0.02_dp
  integer(ip),  parameter :: Ns = 10
  real(real64), parameter :: tol = 1.0e-9_real64     ! NaN-guarded ceiling; expect ~0

  type(decomp_t) :: d
  integer(ip)    :: nkx, kxlo, kxhi, bnkx, ix, kk
  logical        :: ok_setup, ok_step, ok

  real(dp), allocatable :: kxh(:), kzh(:)            ! full wavenumbers (host)
  real(dp), allocatable :: sdh(:), sth(:), bdh(:)    ! host Shen band structure
  real(dp)              :: boff

  real(dp),    allocatable :: vmF(:,:,:,:), vpF(:,:,:,:), CinvF(:,:,:,:,:)   ! full setup
  real(dp),    allocatable :: vmS(:,:,:,:), vpS(:,:,:,:), CinvS(:,:,:,:,:)   ! slab setup
  complex(dp), allocatable :: e0F(:,:,:), v0F(:,:,:)  ! deterministic input (full)
  complex(dp), allocatable :: eF(:,:,:),  vF(:,:,:)   ! oracle output (full)
  complex(dp), allocatable :: eS(:,:,:),  vS(:,:,:)   ! slab output

  call par_init()

  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
  nkx  = d%nkx
  kxlo = d%b_kx0 + 1
  kxhi = d%b_kx0 + d%b_nkx_loc
  bnkx = d%b_nkx_loc

  ! --- kx-independent Shen band structure (identical on every rank) ---
  allocate(sdh(0:n-2), sth(0:n-2), bdh(0:n-2))
  call helm_band_setup(n, sdh, sth, bdh, boff)

  ! --- full wavenumbers: r2c non-negative kx (0..nkx-1), FFT-order kz ---
  allocate(kxh(nkx), kzh(nz))
  do ix = 1, nkx
     kxh(ix) = real(ix-1, dp)
  end do
  do ix = 1, nz
     kk = ix-1; if (kk > nz/2) kk = kk - nz
     kzh(ix) = real(kk, dp)
  end do

  ! =========================================================================
  ! TEST 1 -- precompute setup equivalence, isolated.
  ! =========================================================================
  allocate(vmF(0:n,nkx,nz,3), vpF(0:n,nkx,nz,3), CinvF(2,2,nkx,nz,3))
  call setup_to_host(n, nkx, nz, dt, Re, kxh, kzh, vmF, vpF, CinvF)

  ok_setup = .true.
  if (bnkx > 0) then
     allocate(vmS(0:n,bnkx,nz,3), vpS(0:n,bnkx,nz,3), CinvS(2,2,bnkx,nz,3))
     call setup_to_host(n, bnkx, nz, dt, Re, kxh(kxlo:kxhi), kzh, vmS, vpS, CinvS)
     ok_setup =            closer4(vmS,   vmF(:,kxlo:kxhi,:,:),     tol)
     ok_setup = ok_setup .and. closer4(vpS,   vpF(:,kxlo:kxhi,:,:),     tol)
     ok_setup = ok_setup .and. closer5(CinvS, CinvF(:,:,kxlo:kxhi,:,:), tol)
     deallocate(vmS, vpS, CinvS)
  end if

  ! =========================================================================
  ! TEST 2 -- step equivalence: evolve Ns steps, full (oracle) vs slab.
  ! =========================================================================
  allocate(e0F(0:n,nkx,nz), v0F(0:n,nkx,nz))
  call fill_det(n, nkx, nz, e0F, v0F)
  e0F(:,1,1) = (0.0_dp, 0.0_dp)          ! (kx=0,kz=0) mode is identically zero
  v0F(:,1,1) = (0.0_dp, 0.0_dp)

  allocate(eF(0:n,nkx,nz), vF(0:n,nkx,nz))
  call evolve(n, nkx, nz, Re, dt, Ns, kxh, kzh, sdh, sth, bdh, boff, e0F, v0F, eF, vF)

  ok_step = .true.
  if (bnkx > 0) then
     allocate(eS(0:n,bnkx,nz), vS(0:n,bnkx,nz))
     call evolve(n, bnkx, nz, Re, dt, Ns, kxh(kxlo:kxhi), kzh, sdh, sth, bdh, boff, &
                 e0F(:,kxlo:kxhi,:), v0F(:,kxlo:kxhi,:), eS, vS)
     ok_step =           closec3(eS, eF(:,kxlo:kxhi,:), tol)
     ok_step = ok_step .and. closec3(vS, vF(:,kxlo:kxhi,:), tol)
     deallocate(eS, vS)
  end if

  ! --- global verdict ---
  call par_reduce_and(ok_setup)
  call par_reduce_and(ok_step)
  ok = ok_setup .and. ok_step

  if (par_rank == 0) then
     if (ok_setup) then
        write(*,'(a)') '   [PASS] precompute setup (vm/vp/Cinv) slab == full'
     else
        write(*,'(a)') '   [FAIL] precompute setup (vm/vp/Cinv) slab == full'
     end if
     if (ok_step) then
        write(*,'(a)') '   [PASS] step_ev_linear slab == full (eta,v) after Ns steps'
     else
        write(*,'(a)') '   [FAIL] step_ev_linear slab == full (eta,v) after Ns steps'
     end if
     if (ok) then
        write(*,'(a)') 'ALL TESTS PASSED'
     else
        write(*,'(a)') 'TESTS FAILED'
     end if
  end if

  deallocate(kxh, kzh, sdh, sth, bdh, vmF, vpF, CinvF, e0F, v0F, eF, vF)
  call par_finalize()
  if (.not. ok) error stop 1

contains

  ! ---- run stepper_ev_setup, copy the precomputes to host -----------------
  subroutine setup_to_host(n, nx, nz, dt, Re, kxh, kzh, vm_h, vp_h, Cinv_h)
    integer(ip), intent(in)  :: n, nx, nz
    real(dp),    intent(in)  :: dt, Re, kxh(nx), kzh(nz)
    real(dp),    intent(out) :: vm_h(0:n,nx,nz,3), vp_h(0:n,nx,nz,3), Cinv_h(2,2,nx,nz,3)
    real(dp), device, allocatable :: vm(:,:,:,:), vp(:,:,:,:), Cinv(:,:,:,:,:)
    integer(ip) :: istat
    allocate(vm(0:n,nx,nz,3), vp(0:n,nx,nz,3), Cinv(2,2,nx,nz,3))
    call stepper_ev_setup(n, nx, nz, dt, Re, kxh, kzh, vm, vp, Cinv)
    istat = cudaDeviceSynchronize()
    vm_h = vm; vp_h = vp; Cinv_h = Cinv
    deallocate(vm, vp, Cinv)
  end subroutine setup_to_host

  ! ---- evolve (eta,v) Ns steps (mirrors test_stepper_ev's evolve_ev) -------
  subroutine evolve(n, nx, nz, Re, dt, Ns, kxh, kzh, sdiag, stail, bdiag, boff, &
                    e0, v0, eout, vout)
    integer(ip), intent(in)  :: n, nx, nz, Ns
    real(dp),    intent(in)  :: Re, dt, kxh(nx), kzh(nz)
    real(dp),    intent(in)  :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
    complex(dp), intent(in)  :: e0(0:n,nx,nz), v0(0:n,nx,nz)
    complex(dp), intent(out) :: eout(0:n,nx,nz), vout(0:n,nx,nz)
    real(dp),    device, allocatable :: kx(:), kz(:), sd(:), st(:), bd(:)
    real(dp),    device, allocatable :: vm_all(:,:,:,:), vp_all(:,:,:,:), Cinv_all(:,:,:,:,:)
    complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:)
    integer(ip) :: it, istat
    allocate(kx(nx), kz(nz), sd(0:n-2), st(0:n-2), bd(0:n-2))
    allocate(vm_all(0:n,nx,nz,3), vp_all(0:n,nx,nz,3), Cinv_all(2,2,nx,nz,3))
    allocate(eta(0:n,nx,nz), v(0:n,nx,nz))
    kx = kxh; kz = kzh; sd = sdiag; st = stail; bd = bdiag
    call stepper_ev_setup(n, nx, nz, dt, Re, kxh, kzh, vm_all, vp_all, Cinv_all)
    eta = e0; v = v0
    do it = 1, Ns
       call step_ev_linear(n, nx, nz, dt, Re, kx, kz, sd, st, bd, boff, &
                           vm_all, vp_all, Cinv_all, eta, v)
    end do
    istat = cudaDeviceSynchronize()
    eout = eta; vout = v
    deallocate(kx, kz, sd, st, bd, vm_all, vp_all, Cinv_all, eta, v)
  end subroutine evolve

  ! ---- deterministic input: hash of GLOBAL (k,ix,iz) -> [-1,1] (rank-invariant)
  subroutine fill_det(n, nx, nz, e0, v0)
    integer(ip), intent(in)  :: n, nx, nz
    complex(dp), intent(out) :: e0(0:n,nx,nz), v0(0:n,nx,nz)
    integer(ip) :: k, ix, iz
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, n
             e0(k,ix,iz) = cmplx(h(k,ix,iz,1_ip), h(k,ix,iz,2_ip), dp)
             v0(k,ix,iz) = cmplx(h(k,ix,iz,3_ip), h(k,ix,iz,4_ip), dp)
          end do
       end do
    end do
  end subroutine fill_det

  pure real(dp) function h(k, ix, iz, c)
    integer(ip), intent(in) :: k, ix, iz, c
    integer(int64) :: m
    m =   int(k, int64)*2654435761_int64 + int(ix, int64)*40503_int64 &
        + int(iz,int64)*73856093_int64   + int(c, int64)*19349663_int64
    m = iand(m, 1048575_int64)
    h = real(m, dp)/1048575.0_dp * 2.0_dp - 1.0_dp
  end function h

  ! ---- NaN-guarded max-abs comparisons ------------------------------------
  logical function closec3(a, b, tol)
    complex(dp),  intent(in) :: a(:,:,:), b(:,:,:)
    real(real64), intent(in) :: tol
    closec3 = all(a == a) .and. all(b == b) .and. &
              (real(maxval(abs(a-b)), real64) <= tol)
  end function closec3

  logical function closer4(a, b, tol)
    real(dp),     intent(in) :: a(:,:,:,:), b(:,:,:,:)
    real(real64), intent(in) :: tol
    closer4 = all(a == a) .and. all(b == b) .and. &
              (real(maxval(abs(a-b)), real64) <= tol)
  end function closer4

  logical function closer5(a, b, tol)
    real(dp),     intent(in) :: a(:,:,:,:,:), b(:,:,:,:,:)
    real(real64), intent(in) :: tol
    closer5 = all(a == a) .and. all(b == b) .and. &
              (real(maxval(abs(a-b)), real64) <= tol)
  end function closer5

end program test_stepper_lin_mp
