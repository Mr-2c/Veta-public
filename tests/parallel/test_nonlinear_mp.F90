!===============================================================================
! test_nonlinear_mp.F90  --  multi-GPU: validate the DISTRIBUTED nonlinear
! RHS (par_nonlinear_mod) against the single-GPU oracle (nonlinear_mod), across
! whatever world size the launcher provides.
!
! Each rank, on the SAME deterministic Hermitian-consistent state (built like the
! distributed transform test: deterministic real fields -> to_spectral -> zero
! Nyquist, so the transforms stay in their validated regime; identical on every rank, no
! broadcast):
!   ORACLE (full, single-GPU, redundant per rank):
!     nl_evaluate(vhat,ehat,u0,w0) -> hv_or, heta_or, hx0_or, hz0_or
!   DISTRIBUTED (this rank's kx-slab):
!     par_nl_evaluate(vhat_slab,ehat_slab,u0,w0) -> hv_pp, heta_pp, hx0_pp, hz0_pp
!   CHECKS:
!     (1) hv_pp   == hv_or   restricted to this rank's kx-slab
!     (2) heta_pp == heta_or restricted to this rank's kx-slab
!     (3) mean forcings hx0_pp/hz0_pp == hx0_or/hz0_or  (only on the rank owning kx=0)
! par_reduce_and makes any rank's failure fail the job. Oracle computed identically
! per rank => matching at P=1 and P=4 establishes correctness + P-independence.
! Tolerance is the round-off budget (1e-9 dp / 1e-3 sp); the underlying transforms
! are bit-exact per slab, so the slab differences should be at/near 0.
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_nonlinear_mp   Run: bash run_all_par_tests.sh run -k test_nonlinear_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_nonlinear_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,             only: tp, wp, ip
  use cudafor
  use decomp_mod,        only: decomp_t, decomp_setup
  use spectral3d_mod,    only: s3d_plans, s3d_setup_da_r2c, s3d_destroy_r2c, &
                               to_spectral_da_r2c
  use nonlinear_mod,     only: nl_workspace, nl_setup, nl_destroy, nl_evaluate
  use par_nonlinear_mod, only: par_nl_workspace, par_nl_setup, par_nl_destroy, &
                               par_nl_evaluate
  use par_mod
  implicit none

  integer(ip), parameter :: nx = 32, ny = 6, nz = 16
  integer(ip), parameter :: nxd = 3*nx/2, nzd = 3*nz/2, nkx = nx/2+1, nkxd = nxd/2+1
  real(real64) :: tol
  logical :: all_ok, ok1, ok2, ok3
  integer  :: istat
  integer(ip) :: iy, ix, iz, bnkx, kxlo, kxhi

  type(decomp_t)         :: d
  type(s3d_plans)        :: bld         ! to build Hermitian-consistent inputs
  type(nl_workspace)     :: ws_or       ! single-GPU oracle
  type(par_nl_workspace) :: ws_pp       ! distributed

  real(wp)    :: kx_h(nkx), kz_h(nz)
  real(tp),    allocatable :: Preal_h(:,:,:)
  complex(tp), allocatable :: Asp_h(:,:,:)
  complex(wp), allocatable :: vhat_h(:,:,:), ehat_h(:,:,:)
  complex(wp), allocatable :: hv_or_h(:,:,:), heta_or_h(:,:,:)
  complex(wp), allocatable :: hv_pp_h(:,:,:), heta_pp_h(:,:,:)
  complex(wp) :: u0_h(ny), w0_h(ny)
  complex(wp) :: hx0_or_h(ny), hz0_or_h(ny), hx0_pp_h(ny), hz0_pp_h(ny)

  real(tp),    allocatable, device :: Preal_d(:,:,:)
  complex(tp), allocatable, device :: Asp_d(:,:,:), Wfb(:,:,:)
  complex(wp), allocatable, device :: vhat_d(:,:,:), ehat_d(:,:,:)
  complex(wp), allocatable, device :: hv_or_d(:,:,:), heta_or_d(:,:,:)
  complex(wp), allocatable, device :: vslab_d(:,:,:), eslab_d(:,:,:)
  complex(wp), allocatable, device :: hv_pp_d(:,:,:), heta_pp_d(:,:,:)
  complex(wp), device :: u0_d(ny), w0_d(ny)
  complex(wp), device :: hx0_or_d(ny), hz0_or_d(ny), hx0_pp_d(ny), hz0_pp_d(ny)

  call par_init()
  tol = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)
  if (par_rank == 0) print '(a,i0,a)', '[nonlinear-mp] world size = ', par_nproc, ' rank(s)'

  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
  bnkx = d%b_nkx_loc;  kxlo = d%b_kx0 + 1;  kxhi = d%b_kx0 + bnkx

  ! integer wavenumbers: kx=0..nx/2 ; kz in FFT order (kx_h(1)=0, kz_h(1)=0 => the
  ! (0,0) mean mode is detected by k2==0, the value-based path par_nl relies on).
  do ix = 1, nkx;  kx_h(ix) = real(ix-1, wp);  end do
  do iz = 1, nz
     if (iz-1 <= nz/2) then;  kz_h(iz) = real(iz-1, wp)
     else;                    kz_h(iz) = real(iz-1-nz, wp);  end if
  end do

  allocate(Preal_h(ny,nxd,nzd), Asp_h(ny,nkx,nz))
  allocate(vhat_h(ny,nkx,nz), ehat_h(ny,nkx,nz), hv_or_h(ny,nkx,nz), heta_or_h(ny,nkx,nz))
  allocate(hv_pp_h(ny,bnkx,nz), heta_pp_h(ny,bnkx,nz))
  allocate(Preal_d(ny,nxd,nzd), Asp_d(ny,nkx,nz), Wfb(ny,nkxd,nzd))
  allocate(vhat_d(ny,nkx,nz), ehat_d(ny,nkx,nz), hv_or_d(ny,nkx,nz), heta_or_d(ny,nkx,nz))
  allocate(vslab_d(ny,bnkx,nz), eslab_d(ny,bnkx,nz), hv_pp_d(ny,bnkx,nz), heta_pp_d(ny,bnkx,nz))

  ! --- Hermitian-consistent vhat, ehat (real field -> spectral -> zero Nyquist) -
  call s3d_setup_da_r2c(bld, ny, nx, nz)
  call build_spectral(1_ip, vhat_h)
  call build_spectral(2_ip, ehat_h)
  call s3d_destroy_r2c(bld)
  ! mean profiles u0,w0 (deterministic real, imag 0)
  do iy = 1, ny
     u0_h(iy) = cmplx(real(modulo(iy*53_ip, 97_ip), wp)/97.0_wp - 0.5_wp, 0.0_wp, wp)
     w0_h(iy) = cmplx(real(modulo(iy*71_ip, 89_ip), wp)/89.0_wp - 0.5_wp, 0.0_wp, wp)
  end do
  u0_d = u0_h;  w0_d = w0_h

  ! ===== ORACLE (single-GPU, full field, redundant per rank) ==================
  call nl_setup(ws_or, ny, nx, nz, kx_h, kz_h)
  vhat_d = vhat_h;  ehat_d = ehat_h
  call nl_evaluate(ws_or, vhat_d, ehat_d, u0_d, w0_d, hv_or_d, heta_or_d, hx0_or_d, hz0_or_d)
  istat = cudaDeviceSynchronize()
  hv_or_h = hv_or_d;  heta_or_h = heta_or_d;  hx0_or_h = hx0_or_d;  hz0_or_h = hz0_or_d
  call nl_destroy(ws_or)

  ! ===== DISTRIBUTED ==========================================================
  call par_nl_setup(ws_pp, d, ny, nx, nz, kx_h, kz_h)
  vslab_d = vhat_h(:, kxlo:kxhi, :)            ! this rank's kx-slab (host slice -> device)
  eslab_d = ehat_h(:, kxlo:kxhi, :)
  hx0_pp_d = (0.0_wp, 0.0_wp);  hz0_pp_d = (0.0_wp, 0.0_wp)   ! non-owners leave these zero
  call par_nl_evaluate(ws_pp, vslab_d, eslab_d, u0_d, w0_d, hv_pp_d, heta_pp_d, hx0_pp_d, hz0_pp_d)
  istat = cudaDeviceSynchronize()
  hv_pp_h = hv_pp_d;  heta_pp_h = heta_pp_d;  hx0_pp_h = hx0_pp_d;  hz0_pp_h = hz0_pp_d
  call par_nl_destroy(ws_pp)

  ! ===== CHECKS ===============================================================
  ok1 = close_c(hv_pp_h,   hv_or_h(:,   kxlo:kxhi, :), tol)
  ok2 = close_c(heta_pp_h, heta_or_h(:, kxlo:kxhi, :), tol)
  ! mean forcings only meaningful on the kx=0 owner; trivially true elsewhere.
  if (d%owns_mode00) then
     ok3 = close_c1(hx0_pp_h, hx0_or_h, tol) .and. close_c1(hz0_pp_h, hz0_or_h, tol)
  else
     ok3 = .true.
  end if

  call par_reduce_and(ok1)
  call par_reduce_and(ok2)
  call par_reduce_and(ok3)
  all_ok = ok1 .and. ok2 .and. ok3
  if (par_rank == 0) then
     call report('hv:   distributed == single-GPU oracle (per kx-slab)', ok1)
     call report('heta: distributed == single-GPU oracle (per kx-slab)', ok2)
     call report('mean forcings hx0,hz0 == oracle (kx=0 owner)',         ok3)
     if (all_ok) then
        print '(a)', 'ALL TESTS PASSED'
     else
        print '(a)', 'SOME TESTS FAILED'
     end if
  end if

  call par_finalize()
  if (.not. all_ok) error stop 1

contains

  ! deterministic Hermitian-consistent spectral field #f: real physical -> spectral,
  ! then zero the x- and z-Nyquist (the solver invariant).
  subroutine build_spectral(f, out_h)
    integer(ip),  intent(in)  :: f
    complex(wp),  intent(out) :: out_h(ny,nkx,nz)
    integer(ip) :: jy, jx, jz
    do jz = 1, nzd; do jx = 1, nxd; do jy = 1, ny
       Preal_h(jy,jx,jz) = pval(jy, jx, jz, f)
    end do; end do; end do
    Preal_d = Preal_h
    call to_spectral_da_r2c(bld, ny, nx, nz, nxd, nzd, Preal_d, Wfb, Asp_d)
    istat = cudaDeviceSynchronize()
    Asp_h = Asp_d
    Asp_h(:, nx/2+1, :) = (0.0_tp, 0.0_tp)       ! x-Nyquist
    Asp_h(:, :, nz/2+1) = (0.0_tp, 0.0_tp)       ! z-Nyquist
    out_h = Asp_h                                ! host tp -> host wp
  end subroutine build_spectral

  real(tp) function pval(iyy, ixx, izz, f) result(v)
    integer(ip), intent(in) :: iyy, ixx, izz, f
    integer(ip) :: h
    h = modulo( iyy*1009_ip + ixx*2027_ip + izz*3001_ip + f*4013_ip + ixx*izz*17_ip, 100003_ip )
    v = real(h, tp)/100003.0_tp - 0.5_tp
  end function pval

  logical function close_c(a, b, tol) result(ok)
    complex(wp),  intent(in) :: a(:,:,:), b(:,:,:)
    real(real64), intent(in) :: tol
    ok = all(a == a) .and. all(b == b) .and. &
         (real(maxval(abs(a - b)), real64) <= tol)
  end function close_c

  logical function close_c1(a, b, tol) result(ok)
    complex(wp),  intent(in) :: a(:), b(:)
    real(real64), intent(in) :: tol
    ok = all(a == a) .and. all(b == b) .and. &
         (real(maxval(abs(a - b)), real64) <= tol)
  end function close_c1

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

end program test_nonlinear_mp
