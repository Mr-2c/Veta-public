!===============================================================================
! test_step_ev_mp.F90 -- multi-GPU: validate the DISTRIBUTED fluctuation
! stepper (par_stepper_mod: par_step_ev) against the single-GPU oracle
! (stepper_mod: step_ev), across whatever world size the launcher provides.
!
! par_step_ev is step_ev with the nonlinear RHS swapped for par_nl_evaluate
! (validated bit-exact) and the per-column solve stack run on the kx-slab
! (validated bit-exact). This test composes the two: evolve a deterministic,
! Hermitian-consistent (eta, v) for Ns mean-free SMR/CN steps on the full field
! (oracle) and on this rank's kx-slab, and require the slab columns to match.
!
! Each rank builds the SAME state via build_init (identical in construction to the
! single-GPU cross-check oracle tests/serial/test_stepper_xcheck_r2c.F90): random Shen
! coefficients for eta (eta=0 at the walls) + biharmonic-Shen coefficients for v
! (v=Dv=0), each with an exponentially-decaying spectrum and a fixed RNG seed, so
! the field is smooth, wall-BC-satisfying, and rank-invariant. (A white-noise
! physical field pushed through to_spectral violates the wall BCs and has energy
! at every mode, on which step_ev's wall-normal solves diverge to NaN.) Mean
! modes u0=w0=0 (mean-free, matching
! the single-GPU step_ev cross-check); history starts at zero; the (0,0) mean
! column and both Nyquists are zeroed -- the fluctuation stepper must not carry the
! (0,0) mode (its full handling, zero_col11, is test_step_full_mp's concern).
!
! The nonlinear term carries the tp tier, so this is run at
! BOTH TP=dp and TP=sp. The compared (eta, v) are dp; both pieces (par_nl_evaluate
! and the dp solves) are bit-exact per kx-slab, so the match should be ~0 (the
! tolerance is a NaN-guarded ceiling: 1e-9 dp / 1e-3 sp). par_reduce_and makes any
! rank's failure fail the job. Matching at P=1/4/8 establishes P-independence.
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_step_ev_mp   Run: bash run_all_par_tests.sh run -k test_step_ev_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_step_ev_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,             only: tp, wp, ip, dp
  use cudafor
  use decomp_mod,        only: decomp_t, decomp_setup
  use cheb_shen,         only: shen_to_cheb
  use cheb_bih,          only: shen_bih_to_cheb
  use helm_band_mod,     only: helm_band_setup
  use solver_ops_mod,    only: band_ops_t, vsolve_ops_t, &
                               band_ops_alloc, band_ops_free, &
                               vsolve_ops_alloc, vsolve_ops_free
  use stepper_lin_mod,   only: stepper_ev_setup
  use nonlinear_mod,     only: nl_workspace, nl_setup, nl_destroy
  use stepper_mod,       only: step_ev
  use par_nonlinear_mod, only: par_nl_workspace, par_nl_setup, par_nl_destroy
  use par_stepper_mod,   only: par_step_ev
  use par_mod
  implicit none

  integer(ip), parameter :: n = 16, nx = 32, nz = 16, Ns = 5
  integer(ip), parameter :: ny = n + 1
  integer(ip), parameter :: nkx = nx/2+1
  real(dp),    parameter :: Re = 10.0_dp, dt = 5.0e-3_dp, amp = 0.05_dp
  real(real64) :: tol

  type(decomp_t)  :: d
  integer(ip) :: bnkx, kxlo, kxhi, ix, iz, kk
  integer     :: istat
  logical     :: ok_e, ok_v, all_ok

  real(dp) :: kxh(nkx), kzh(nz)
  type(band_ops_t) :: bo          ! y-operator band structure (host+device)

  complex(dp), allocatable :: e0(:,:,:), v0(:,:,:)     ! full input (Hermitian, scaled)
  complex(dp), allocatable :: eF(:,:,:), vF(:,:,:)     ! oracle output (full)
  complex(dp), allocatable :: eS(:,:,:), vS(:,:,:)     ! slab output

  call par_init()
  tol = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)
  if (par_rank == 0) print '(a,i0,a)', '[step_ev-mp] world size = ', par_nproc, ' rank(s)'

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
  call band_ops_alloc(bo, n)
  call helm_band_setup(n, bo%sdh, bo%sth, bo%bdh, bo%boff)

  allocate(e0(0:n,nkx,nz), v0(0:n,nkx,nz), eF(0:n,nkx,nz), vF(0:n,nkx,nz))

  ! --- Smooth, wall-BC-satisfying (eta, v), built exactly like the single-GPU
  !     cross-check oracle (tests/serial/test_stepper_xcheck_r2c.F90::build_init): random
  !     Shen (eta, eta=0) + biharmonic-Shen (v, v=Dv=0) coefficients with a decaying
  !     spectrum and a fixed (rank-invariant) RNG seed; the (0,0) mean column and
  !     both Nyquists are zeroed. This is the field on which step_ev is stable;
  !     a white-noise field through to_spectral (the previous builder) violated the
  !     wall BCs and drove step_ev's wall-normal solves to NaN.
  call build_init(e0, v0)

  ! ===== ORACLE (single-GPU step_ev, full field, redundant per rank) ==========
  call run_full(eF, vF)

  ! ===== DISTRIBUTED (par_step_ev on this rank's kx-slab) =====================
  ok_e = .true.;  ok_v = .true.
  if (bnkx > 0) then
     allocate(eS(0:n,bnkx,nz), vS(0:n,bnkx,nz))
     call run_slab(eS, vS)
     ok_e = close_c(eS, eF(:, kxlo:kxhi, :), tol)
     ok_v = close_c(vS, vF(:, kxlo:kxhi, :), tol)
     deallocate(eS, vS)
  end if

  call par_reduce_and(ok_e)
  call par_reduce_and(ok_v)
  all_ok = ok_e .and. ok_v
  if (par_rank == 0) then
     call report('eta: distributed step_ev == single-GPU oracle (per kx-slab)', ok_e)
     call report('v:   distributed step_ev == single-GPU oracle (per kx-slab)', ok_v)
     if (all_ok) then
        print '(a)', 'ALL TESTS PASSED'
     else
        print '(a)', 'SOME TESTS FAILED'
     end if
  end if

  call band_ops_free(bo)
  call par_finalize()
  if (.not. all_ok) error stop 1

contains

  ! Smooth, wall-BC-satisfying initial (eta, v), identical in construction to the
  ! single-GPU oracle tests/serial/test_stepper_xcheck_r2c.F90::build_init. eta lives in
  ! the Shen basis (shen_to_cheb; eta=0 at the walls), v in the biharmonic-Shen
  ! basis (shen_bih_to_cheb; v=Dv=0), each with an exponentially-decaying spectrum
  ! (amp*exp(-(k/kc)^2), kc=n/6). The RNG seed is fixed so every rank builds the
  ! SAME field. The (0,0) mean column and both Nyquists are zeroed.
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

  ! single-GPU oracle: evolve (eta, v) Ns mean-free steps on the FULL field.
  subroutine run_full(eout, vout)
    complex(dp), intent(out) :: eout(0:n,nkx,nz), vout(0:n,nkx,nz)
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
       call step_ev(ws, n, nkx, nz, dt, Re, kx, kz, bo, vo, &
                    u0, w0, eta, v, Nvp, Nep)
    end do
    istat = cudaDeviceSynchronize()
    eout = eta;  vout = v
    deallocate(kx, kz, eta, v, u0, w0, Nvp, Nep)
    call vsolve_ops_free(vo)
    call nl_destroy(ws)
  end subroutine run_full

  ! distributed: evolve (eta, v) Ns mean-free steps on THIS rank's kx-slab.
  subroutine run_slab(eout, vout)
    complex(dp), intent(out) :: eout(0:n,bnkx,nz), vout(0:n,bnkx,nz)
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
       call par_step_ev(ws, n, bnkx, nz, dt, Re, kx, kz, bo, vo, &
                        u0, w0, eta, v, Nvp, Nep)
    end do
    istat = cudaDeviceSynchronize()
    eout = eta;  vout = v
    deallocate(kx, kz, eta, v, u0, w0, Nvp, Nep)
    call vsolve_ops_free(vo)
    call par_nl_destroy(ws)
  end subroutine run_slab

  logical function close_c(a, b, tol) result(okk)
    complex(dp),  intent(in) :: a(:,:,:), b(:,:,:)
    real(real64), intent(in) :: tol
    okk = all(a == a) .and. all(b == b) .and. &
          (real(maxval(abs(a - b)), real64) <= tol)
  end function close_c

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

end program test_step_ev_mp
