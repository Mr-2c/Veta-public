!===============================================================================
! test_step_full_mp.F90 -- multi-GPU: validate the DISTRIBUTED FULL-system
! stepper (par_stepper_mod: par_step_full) against the single-GPU oracle
! (stepper_mod: step_full), across whatever world size the launcher provides.
!
! par_step_full is step_full with the nonlinear RHS swapped for par_nl_evaluate
! and the per-column solve stack run on the kx-slab, PLUS the rank-0-local mean (u0,w0) solve and the (0,0)-column
! zeroing -- both guarded by owns_mode00 (the rank holding global kx=0). This test
! composes them: evolve a deterministic, Hermitian-consistent (eta, v) with the
! constant-mass-flux MEAN turned ON for Ns SMR/CN steps on the full field (oracle)
! and on this rank's kx-slab, and require the slab to match the oracle:
!   * eta, v  : slab columns == oracle, per kx-slab, on EVERY rank;
!   * u0, w0  : the evolved mean profile == oracle, on the owns_mode00 rank only
!               (the mean lives only there; other ranks keep u0=w0=0, which
!               recover_uw ignores -- so there is nothing to compare off-owner).
!
! Each rank builds the SAME initial (eta, v) via build_init (identical to
! tests/serial/test_stepper_full_xcheck_r2c.F90::build_init and to test_step_ev_mp): random
! Shen coefficients for eta (eta=0 at the walls) + biharmonic-Shen for v (v=Dv=0),
! decaying spectrum, fixed RNG seed (rank-invariant); the (0,0) mean column and
! both Nyquists zeroed. The mean is driven by the CFR constraint (Qtar=1) starting
! from u0=w0=0, so it becomes genuinely non-zero -- exercising the mean->recover_uw
! path that the mean-free test_step_ev_mp (u0=w0=0) cannot.
!
! The fluctuation pieces are bit-exact per kx-slab and the mean solve is host/dp
! fed by the bit-exact (0,0) forcing, so the match should be ~0 (the tolerance
! is a NaN-guarded ceiling: 1e-9 dp / 1e-3 sp). The nonlinear term carries the tp
! tier, so run at BOTH TP=dp and TP=sp. par_reduce_and makes any rank's failure
! fail the job. Matching at P=1/4/8 establishes P-independence of the full stepper.
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_step_full_mp   Run: bash run_all_par_tests.sh run -k test_step_full_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_step_full_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,             only: tp, wp, ip, dp
  use cudafor
  use decomp_mod,        only: decomp_t, decomp_setup
  use cheb_shen,         only: shen_to_cheb
  use cheb_bih,          only: shen_bih_to_cheb
  use helm_band_mod,     only: helm_band_setup
  use mean_mod,          only: mean_flux_setup, bulk_velocity
  use stepper_lin_mod,   only: stepper_ev_setup
  use solver_ops_mod,    only: band_ops_t, vsolve_ops_t, mean_ops_t, &
                               band_ops_alloc, band_ops_free, vsolve_ops_alloc, &
                               vsolve_ops_free, mean_ops_alloc, mean_ops_free
  use nonlinear_mod,     only: nl_workspace, nl_setup, nl_destroy
  use stepper_mod,       only: step_full
  use par_nonlinear_mod, only: par_nl_workspace, par_nl_setup, par_nl_destroy
  use par_stepper_mod,   only: par_step_full
  use par_mod
  implicit none

  integer(ip), parameter :: n = 16, nx = 32, nz = 16, Ns = 5
  integer(ip), parameter :: ny = n + 1
  integer(ip), parameter :: nkx = nx/2+1
  real(dp),    parameter :: Re = 10.0_dp, dt = 5.0e-3_dp, amp = 0.05_dp, Qtar = 1.0_dp
  real(real64) :: tol

  type(decomp_t)  :: d
  integer(ip) :: bnkx, kxlo, kxhi, ix, iz, kk
  integer     :: istat
  logical     :: ok_e, ok_v, ok_m, all_ok

  real(dp) :: kxh(nkx), kzh(nz)
  type(band_ops_t) :: bo          ! y-operator band structure (host+device)
  type(mean_ops_t) :: mo          ! CFR mean precompute

  complex(dp), allocatable :: e0(:,:,:), v0(:,:,:)     ! full input (Hermitian, scaled)
  complex(dp), allocatable :: eF(:,:,:), vF(:,:,:)     ! oracle output (full)
  complex(dp), allocatable :: eS(:,:,:), vS(:,:,:)     ! slab output
  complex(dp) :: u0F(0:n), w0F(0:n), u0S(0:n), w0S(0:n)

  call par_init()
  tol = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)
  if (par_rank == 0) print '(a,i0,a)', '[step_full-mp] world size = ', par_nproc, ' rank(s)'

  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
  bnkx = d%b_nkx_loc;  kxlo = d%b_kx0 + 1;  kxhi = d%b_kx0 + bnkx

  do ix = 1, nkx
     kxh(ix) = real(ix-1, dp)
  end do
  do iz = 1, nz
     kk = iz-1
     if (kk > nz/2) kk = kk - nz
     kzh(iz) = real(kk, dp)
  end do
  call band_ops_alloc(bo, n);  call mean_ops_alloc(mo, n)
  call helm_band_setup(n, bo%sdh, bo%sth, bo%bdh, bo%boff)
  call mean_flux_setup(n, dt, Re, bo%sdh, bo%sth, bo%bdh, bo%boff, mo%Uhm, mo%Ih)

  allocate(e0(0:n,nkx,nz), v0(0:n,nkx,nz), eF(0:n,nkx,nz), vF(0:n,nkx,nz))
  call build_init(e0, v0)

  ! ===== ORACLE (single-GPU step_full, full field, redundant per rank) =========
  call run_full(eF, vF, u0F, w0F)

  ! ===== DISTRIBUTED (par_step_full on this rank's kx-slab) ====================
  ok_e = .true.;  ok_v = .true.;  ok_m = .true.
  u0S = (0.0_dp,0.0_dp);  w0S = (0.0_dp,0.0_dp)
  if (bnkx > 0) then
     allocate(eS(0:n,bnkx,nz), vS(0:n,bnkx,nz))
     call run_slab(eS, vS, u0S, w0S)
     ok_e = close_c(eS, eF(:, kxlo:kxhi, :), tol)
     ok_v = close_c(vS, vF(:, kxlo:kxhi, :), tol)
     deallocate(eS, vS)
  end if
  ! the mean profile lives ONLY on the rank owning kx=0; compare it there.
  if (d%owns_mode00) then
     ok_m = close_c1(u0S, u0F, tol) .and. close_c1(w0S, w0F, tol)
  end if

  call par_reduce_and(ok_e)
  call par_reduce_and(ok_v)
  call par_reduce_and(ok_m)
  all_ok = ok_e .and. ok_v .and. ok_m
  if (par_rank == 0) then
     call report('eta:  distributed step_full == single-GPU oracle (per kx-slab)', ok_e)
     call report('v:    distributed step_full == single-GPU oracle (per kx-slab)', ok_v)
     call report('mean: u0,w0 == single-GPU oracle (on the kx=0 owner)',           ok_m)
     if (all_ok) then
        print '(a)', 'ALL TESTS PASSED'
     else
        print '(a)', 'SOME TESTS FAILED'
     end if
  end if

  call band_ops_free(bo);  call mean_ops_free(mo)
  call par_finalize()
  if (.not. all_ok) error stop 1

contains

  ! Smooth, wall-BC-satisfying initial (eta, v), identical in construction to the
  ! single-GPU oracle tests/serial/test_stepper_full_xcheck_r2c.F90::build_init and to the
  ! test_step_ev_mp. eta in the Shen basis (shen_to_cheb; eta=0 at the walls), v in the
  ! biharmonic-Shen basis (shen_bih_to_cheb; v=Dv=0), each with an exponentially-
  ! decaying spectrum (amp*exp(-(k/kc)^2), kc=n/6). Fixed RNG seed -> rank-invariant.
  ! The (0,0) mean column and both Nyquists are zeroed.
  subroutine build_init(e0o, v0o)
    complex(dp), intent(out) :: e0o(0:n,nkx,nz), v0o(0:n,nkx,nz)
    real(dp) :: ure(0:n-2), uim(0:n-2), are(0:n), aim(0:n)
    real(dp) :: vre(0:n-4), vim(0:n-4), bre(0:n), bim(0:n)
    real(dp) :: rr, ri, decay, kc
    integer(ip) :: jx, jz, k
    integer     :: rs_n, rs_j
    integer, allocatable :: rs_seed(:)
    kc = real(n, dp) / 6.0_dp
    call random_seed(size = rs_n)
    allocate(rs_seed(rs_n))
    do rs_j = 0, rs_n - 1
       rs_seed(rs_j+1) = 1789 + 101*rs_j
    end do
    call random_seed(put = rs_seed)
    deallocate(rs_seed)
    do jz = 1, nz
       do jx = 1, nkx
          do k = 0, n-2
             call random_number(rr);  call random_number(ri)
             decay = amp * exp(-(real(k,dp)/kc)**2)
             ure(k) = (rr - 0.5_dp)*decay;  uim(k) = (ri - 0.5_dp)*decay
          end do
          call shen_to_cheb(n, ure, are)
          call shen_to_cheb(n, uim, aim)
          do k = 0, n
             e0o(k,jx,jz) = cmplx(are(k), aim(k), dp)
          end do
          do k = 0, n-4
             call random_number(rr);  call random_number(ri)
             decay = amp * exp(-(real(k,dp)/kc)**2)
             vre(k) = (rr - 0.5_dp)*decay;  vim(k) = (ri - 0.5_dp)*decay
          end do
          call shen_bih_to_cheb(n, vre, bre)
          call shen_bih_to_cheb(n, vim, bim)
          do k = 0, n
             v0o(k,jx,jz) = cmplx(bre(k), bim(k), dp)
          end do
       end do
    end do
    e0o(:,1,1)      = (0.0_dp,0.0_dp);  v0o(:,1,1)      = (0.0_dp,0.0_dp)
    e0o(:,nkx,:)    = (0.0_dp,0.0_dp);  v0o(:,nkx,:)    = (0.0_dp,0.0_dp)
    e0o(:,:,nz/2+1) = (0.0_dp,0.0_dp);  v0o(:,:,nz/2+1) = (0.0_dp,0.0_dp)
  end subroutine build_init

  ! single-GPU oracle: evolve (eta, v) + mean (u0,w0) Ns full steps on the FULL field.
  subroutine run_full(eout, vout, u0out, w0out)
    complex(dp), intent(out) :: eout(0:n,nkx,nz), vout(0:n,nkx,nz), u0out(0:n), w0out(0:n)
    type(nl_workspace) :: ws
    real(dp),    device, allocatable :: kx(:), kz(:)
    type(vsolve_ops_t) :: vo                ! local v-solve apparatus
    complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:), Nvp(:,:,:), Nep(:,:,:)
    integer(ip) :: it
    call nl_setup(ws, n+1, nx, nz, kxh, kzh)
    allocate(kx(nkx), kz(nz))
    call vsolve_ops_alloc(vo, n, nkx, nz)
    allocate(eta(0:n,nkx,nz), v(0:n,nkx,nz), u0(0:n), w0(0:n), Nvp(0:n,nkx,nz), Nep(0:n,nkx,nz))
    kx = kxh;  kz = kzh
    bo%sd = bo%sdh;  bo%st = bo%sth;  bo%bd = bo%bdh
    call stepper_ev_setup(n, nkx, nz, dt, Re, kxh, kzh, vo%vm, vo%vp, vo%Cinv)
    u0 = (0.0_dp,0.0_dp);  w0 = (0.0_dp,0.0_dp)
    Nvp = (0.0_dp,0.0_dp);  Nep = (0.0_dp,0.0_dp)
    eta = e0;  v = v0
    do it = 1, Ns
       call step_full(ws, n, nkx, nz, dt, Re, kx, kz, bo, vo, mo, &
                      Qtar, u0, w0, eta, v, Nvp, Nep)
    end do
    istat = cudaDeviceSynchronize()
    eout = eta;  vout = v;  u0out = u0;  w0out = w0
    call vsolve_ops_free(vo)
    deallocate(kx, kz, eta, v, u0, w0, Nvp, Nep)
    call nl_destroy(ws)
  end subroutine run_full

  ! distributed: evolve (eta, v) + mean Ns full steps on THIS rank's kx-slab.
  subroutine run_slab(eout, vout, u0out, w0out)
    complex(dp), intent(out) :: eout(0:n,bnkx,nz), vout(0:n,bnkx,nz), u0out(0:n), w0out(0:n)
    type(par_nl_workspace) :: ws
    real(dp),    device, allocatable :: kx(:), kz(:)
    type(vsolve_ops_t) :: vo                ! local v-solve apparatus
    complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:), Nvp(:,:,:), Nep(:,:,:)
    integer(ip) :: it
    call par_nl_setup(ws, d, n+1, nx, nz, kxh, kzh)
    allocate(kx(bnkx), kz(nz))
    call vsolve_ops_alloc(vo, n, bnkx, nz)
    allocate(eta(0:n,bnkx,nz), v(0:n,bnkx,nz), u0(0:n), w0(0:n), Nvp(0:n,bnkx,nz), Nep(0:n,bnkx,nz))
    kx = kxh(kxlo:kxhi);  kz = kzh
    bo%sd = bo%sdh;  bo%st = bo%sth;  bo%bd = bo%bdh
    call stepper_ev_setup(n, bnkx, nz, dt, Re, kxh(kxlo:kxhi), kzh, vo%vm, vo%vp, vo%Cinv)
    u0 = (0.0_dp,0.0_dp);  w0 = (0.0_dp,0.0_dp)
    Nvp = (0.0_dp,0.0_dp);  Nep = (0.0_dp,0.0_dp)
    eta = e0(:, kxlo:kxhi, :);  v = v0(:, kxlo:kxhi, :)
    do it = 1, Ns
       call par_step_full(ws, n, bnkx, nz, dt, Re, kx, kz, bo, vo, mo, &
                          Qtar, u0, w0, eta, v, Nvp, Nep)
    end do
    istat = cudaDeviceSynchronize()
    eout = eta;  vout = v;  u0out = u0;  w0out = w0
    call vsolve_ops_free(vo)
    deallocate(kx, kz, eta, v, u0, w0, Nvp, Nep)
    call par_nl_destroy(ws)
  end subroutine run_slab

  logical function close_c(a, b, tol) result(okk)
    complex(dp),  intent(in) :: a(:,:,:), b(:,:,:)
    real(real64), intent(in) :: tol
    okk = all(a == a) .and. all(b == b) .and. &
          (real(maxval(abs(a - b)), real64) <= tol)
  end function close_c

  logical function close_c1(a, b, tol) result(okk)
    complex(dp),  intent(in) :: a(:), b(:)
    real(real64), intent(in) :: tol
    okk = all(a == a) .and. all(b == b) .and. &
          (real(maxval(abs(a - b)), real64) <= tol)
  end function close_c1

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

end program test_step_full_mp
