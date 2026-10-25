!===============================================================================
! test_driver_state_mp.F90 -- distributed driver: the driver's
! diagnostics on a REAL physically-recovered velocity field (not arbitrary data),
! distributed vs the single-GPU oracle.
!
! Pipeline (the front half of a real time step, then the monitor):
!   build a valid, mean-free (eta, v)  [Shen / biharmonic-Shen, wall-BC-satisfying]
!     -> recover_uw            : (eta,v) -> spectral velocity (uhat, what)   [Layout B]
!     -> divergence_check      : max |i kx u + Dv + i kz w|  -> par_reduce_MAX
!                                (a KMM-recovered field is divergence-free, so this
!                                 is ~round-off -- a physical correctness probe that
!                                 the distributed recovery stays solenoidal)
!     -> par_to_physical x3    : (uhat, v, what) -> physical (u,v,w)          [Layout A]
!     -> cfl_freq_max          : max_grid(|u|inv_dx+|v|inv_dy+|w|inv_dz) -> par_reduce_MAX
!                                (the convective CFL that drives adaptive dt)
!
! This composes the validated distributed pieces (extent-agnostic recover_uw,
! par_to_physical, divergence_check / cfl_freq_max and the collective reductions)
! into the REAL diagnostic path the driver uses, on a genuine recovered and
! transformed state rather than on arbitrary fields.
!
! Both reduced quantities are computed per-slab + reduced and must match the
! single-GPU values. maxdiv is BIT-EXACT: recover_uw is a per-column coefficient
! recurrence (slab columns identical to full) and max is partition-independent, so
! the difference is 0 (tol 1e-12, like the other bit-exact tests). fmax goes through
! the TRANSFORM (distributed 1-D plans + transpose vs the fused oracle), so it uses
! the SAME round-off ceiling as every other tp-carrying parallel test --
! merge(1e-9 dp, 1e-3 sp) (the ceiling is generous, the
! ACTUAL margin is far smaller and is PRINTED below so it never has to be guessed).
! The CFL is the strong reduce_max check (velocity O(1) => fmax > 0); the divergence
! is the solenoidal probe (maxdiv ~ round-off, equal on both paths). par_reduce_and
! fails the job on any rank. dp AND sp. Precondition P <= min(nkx,nzd)=17.
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_driver_state_mp   Run: bash run_all_par_tests.sh run -k test_driver_state_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!   module load nvhpc openmpi nccl
! Build: TP=dp|sp bash scripts/build/build_par.sh test_driver_state_mp   Run: bash run_all_par_tests.sh run -k test_driver_state_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_driver_state_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,              only: tp, wp, dp, ip
  use cudafor
  use decomp_mod,         only: decomp_t, decomp_setup
  use cheb_shen,          only: shen_to_cheb
  use cheb_bih,           only: shen_bih_to_cheb
  use velocity_mod,       only: recover_uw
  use cfl_mod,            only: cfl_inv_dy, cfl_freq_max
  use stats_mod,          only: divergence_check
  use spectral3d_mod,     only: s3d_plans, s3d_setup_da_r2c, s3d_destroy_r2c, &
                                to_physical_da_r2c
  use par_spectral3d_mod, only: par_s3d_plans, par_s3d_setup_da_r2c, &
                                par_s3d_destroy_da_r2c, par_to_physical_da_r2c
  use par_mod
  implicit none

  integer(ip), parameter :: n = 16, nx = 32, nz = 16
  integer(ip), parameter :: ny = n+1, nkx = nx/2+1, nxd = 3*nx/2, nzd = 3*nz/2, nkxd = nxd/2+1
  real(dp),    parameter :: PI = acos(-1.0_dp), amp = 0.05_dp

  type(decomp_t) :: d
  integer(ip) :: bnkx, kxlo, kxhi, anz, zlo, zhi, ix, iz, k
  integer     :: istat
  logical     :: ok_cfl, ok_div, all_ok

  real(dp) :: inv_dx, inv_dz, y(0:n), inv_dy_h(0:n), kxh(nkx), kzh(nz)
  real(dp) :: fmax_ref, maxdiv_ref
  real(real64) :: fmax_g, maxdiv_g, tol_max, tol_cfl

  complex(wp), allocatable :: e0(:,:,:), v0(:,:,:)     ! mean-free state (wp = dp)

  call par_init()
  tol_max = 1.0e-12_real64                                        ! maxdiv: bit-exact difference
  tol_cfl = merge(1.0e-9_real64, 1.0e-3_real64, tp == real64)     ! fmax: standard transform ceiling
  if (par_rank == 0) print '(a,i0,a)', '[driver-state] world size = ', par_nproc, ' rank(s)'

  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
  bnkx = d%b_nkx_loc;  kxlo = d%b_kx0 + 1;  kxhi = d%b_kx0 + bnkx
  anz  = d%ad_nz_loc;  zlo  = d%ad_z0 + 1;  zhi  = d%ad_z0 + anz

  inv_dx = real(nx, dp)/(2.0_dp*PI);  inv_dz = real(nz, dp)/(2.0_dp*PI)
  do k = 0, n
     y(k) = cos(PI*real(k, dp)/real(n, dp))
  end do
  call cfl_inv_dy(n, y, inv_dy_h)
  do ix = 1, nkx
     kxh(ix) = real(ix-1, dp)
  end do
  do iz = 1, nz
     k = iz-1;  if (k > nz/2) k = k - nz;  kzh(iz) = real(k, dp)
  end do

  allocate(e0(0:n,nkx,nz), v0(0:n,nkx,nz))
  call build_init(e0, v0)

  call run_ref(fmax_ref, maxdiv_ref)      ! single-GPU, full field (per rank)
  call run_slab(fmax_g, maxdiv_g)         ! this rank's slabs + reductions

  ok_cfl = (abs(fmax_g   - real(fmax_ref,   real64)) <= tol_cfl) .and. (fmax_ref > 0.0_dp)
  ok_div = (abs(maxdiv_g - real(maxdiv_ref, real64)) <= tol_max)     ! bit-exact (recurrence)

  call par_reduce_and(ok_cfl)
  call par_reduce_and(ok_div)
  all_ok = ok_cfl .and. ok_div
  if (par_rank == 0) then
     ! print the ACTUAL margins so the agreement is visible (not just pass/fail)
     print '(a,es10.2,a,es9.2,a)', '   |fmax_dist - fmax_ref|     = ', &
          abs(fmax_g   - real(fmax_ref,   real64)), '   (fmax=',   real(fmax_ref,   real64), ', tol 1e-9/1e-3)'
     print '(a,es10.2,a,es9.2,a)', '   |maxdiv_dist - maxdiv_ref| = ', &
          abs(maxdiv_g - real(maxdiv_ref, real64)), '   (maxdiv=', real(maxdiv_ref, real64), ', tol 1e-12)'
     call report('CFL fmax (recovered velocity): distributed par_reduce_max == single-GPU', ok_cfl)
     call report('max-divergence (recovered field): distributed par_reduce_max == single-GPU', ok_div)
     if (all_ok) then
        print '(a)', 'ALL TESTS PASSED'
     else
        print '(a)', 'SOME TESTS FAILED'
     end if
  end if

  call par_finalize()
  if (.not. all_ok) error stop 1

contains

  ! Smooth, wall-BC-satisfying mean-free (eta, v): random Shen (eta) + biharmonic-
  ! Shen (v), decaying spectrum, fixed RNG seed (rank-invariant); (0,0)+Nyquists
  ! zeroed. Identical construction to the validated stepper cross-checks.
  subroutine build_init(e0o, v0o)
    complex(wp), intent(out) :: e0o(0:n,nkx,nz), v0o(0:n,nkx,nz)
    real(dp) :: ure(0:n-2), uim(0:n-2), are(0:n), aim(0:n)
    real(dp) :: vre(0:n-4), vim(0:n-4), bre(0:n), bim(0:n)
    real(dp) :: rr, ri, decay, kc
    integer(ip) :: jx, jz, kk
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
          do kk = 0, n-2
             call random_number(rr);  call random_number(ri)
             decay = amp * exp(-(real(kk,dp)/kc)**2)
             ure(kk) = (rr - 0.5_dp)*decay;  uim(kk) = (ri - 0.5_dp)*decay
          end do
          call shen_to_cheb(n, ure, are);  call shen_to_cheb(n, uim, aim)
          do kk = 0, n
             e0o(kk,jx,jz) = cmplx(are(kk), aim(kk), wp)
          end do
          do kk = 0, n-4
             call random_number(rr);  call random_number(ri)
             decay = amp * exp(-(real(kk,dp)/kc)**2)
             vre(kk) = (rr - 0.5_dp)*decay;  vim(kk) = (ri - 0.5_dp)*decay
          end do
          call shen_bih_to_cheb(n, vre, bre);  call shen_bih_to_cheb(n, vim, bim)
          do kk = 0, n
             v0o(kk,jx,jz) = cmplx(bre(kk), bim(kk), wp)
          end do
       end do
    end do
    e0o(:,1,1)      = (0.0_wp,0.0_wp);  v0o(:,1,1)      = (0.0_wp,0.0_wp)
    e0o(:,nkx,:)    = (0.0_wp,0.0_wp);  v0o(:,nkx,:)    = (0.0_wp,0.0_wp)
    e0o(:,:,nz/2+1) = (0.0_wp,0.0_wp);  v0o(:,:,nz/2+1) = (0.0_wp,0.0_wp)
  end subroutine build_init

  ! complex(wp) -> complex(tp) elementwise demote (transform input must be tp)
  subroutine demote(m, nxl, nzl, src, dst)
    integer(ip), intent(in) :: m, nxl, nzl
    complex(wp), device :: src(0:m,nxl,nzl)
    complex(tp), device :: dst(0:m,nxl,nzl)
    integer(ip) :: i, j, kk
    !$cuf kernel do(3) <<<*,*>>>
    do kk = 1, nzl;  do j = 1, nxl;  do i = 0, m
       dst(i,j,kk) = src(i,j,kk)
    end do; end do; end do
  end subroutine demote

  ! single-GPU reference: recover -> divergence -> transform(3) -> CFL, full field.
  subroutine run_ref(fmax, maxdiv)
    real(dp), intent(out) :: fmax, maxdiv
    type(s3d_plans) :: orpl
    real(wp),    device, allocatable :: kxt(:), kzt(:)            ! wp
    real(tp),    device, allocatable :: pu(:,:,:), pv(:,:,:), pw(:,:,:)
    real(wp),    device, allocatable :: kxw(:), kzw(:)
    real(dp),    device, allocatable :: inv_dy_d(:)
    complex(wp), device, allocatable :: vhat(:,:,:), ehat(:,:,:), u0(:), w0(:), dvh(:,:,:)
    complex(tp), device, allocatable :: uhat(:,:,:), what(:,:,:), dv(:,:,:), vc(:,:,:), Wf(:,:,:)
    call s3d_setup_da_r2c(orpl, ny, nx, nz)
    allocate(kxt(nkx), kzt(nz), kxw(nkx), kzw(nz), inv_dy_d(ny))
    allocate(vhat(0:n,nkx,nz), ehat(0:n,nkx,nz), u0(0:n), w0(0:n), dvh(0:n,nkx,nz))
    allocate(uhat(0:n,nkx,nz), what(0:n,nkx,nz), dv(0:n,nkx,nz), vc(0:n,nkx,nz))
    allocate(Wf(ny,nkxd,nzd), pu(ny,nxd,nzd), pv(ny,nxd,nzd), pw(ny,nxd,nzd))
    kxt = kxh;  kzt = kzh;  kxw = real(kxh, wp);  kzw = real(kzh, wp)
    inv_dy_d = inv_dy_h
    vhat = v0;  ehat = e0;  u0 = (0.0_wp,0.0_wp);  w0 = (0.0_wp,0.0_wp)
    call recover_uw(ny, nkx, nz, kxt, kzt, vhat, ehat, u0, w0, uhat, what, dv)
    call divergence_check(n, nkx, nz, kxw, kzw, uhat, vhat, what, dvh, maxdiv)
    call demote(n, nkx, nz, vhat, vc)
    call to_physical_da_r2c(orpl, ny, nx, nz, nxd, nzd, uhat, pu, Wf)   ! consumes uhat
    call to_physical_da_r2c(orpl, ny, nx, nz, nxd, nzd, vc,   pv, Wf)   ! consumes vc
    call to_physical_da_r2c(orpl, ny, nx, nz, nxd, nzd, what, pw, Wf)   ! consumes what
    call cfl_freq_max(ny, nxd, nzd, pu, pv, pw, inv_dx, inv_dz, inv_dy_d, fmax)
    istat = cudaDeviceSynchronize()
    call s3d_destroy_r2c(orpl)
    deallocate(kxt, kzt, kxw, kzw, inv_dy_d, vhat, ehat, u0, w0, dvh, &
               uhat, what, dv, vc, Wf, pu, pv, pw)
  end subroutine run_ref

  ! distributed: recover -> divergence -> par_to_physical(3) -> CFL on this rank's
  ! slabs, then par_reduce_max. (No slab is empty for P<=17.)
  subroutine run_slab(fmax_out, maxdiv_out)
    real(real64), intent(out) :: fmax_out, maxdiv_out
    type(par_s3d_plans) :: pp
    real(wp),    device, allocatable :: kxt(:), kzt(:)            ! wp
    real(tp),    device, allocatable :: pu(:,:,:), pv(:,:,:), pw(:,:,:)
    real(wp),    device, allocatable :: kxw(:), kzw(:)
    real(dp),    device, allocatable :: inv_dy_d(:)
    complex(wp), device, allocatable :: vhat(:,:,:), ehat(:,:,:), u0(:), w0(:), dvh(:,:,:)
    complex(tp), device, allocatable :: uhat(:,:,:), what(:,:,:), dv(:,:,:), vc(:,:,:)
    complex(tp), device, allocatable :: Bz(:,:,:,:), Az(:,:,:,:), Af(:,:,:)
    real(dp) :: fmax_loc, maxdiv_loc
    call par_s3d_setup_da_r2c(pp, d, ny, nx, nz)
    allocate(kxt(bnkx), kzt(nz), kxw(bnkx), kzw(nz), inv_dy_d(ny))
    allocate(vhat(0:n,bnkx,nz), ehat(0:n,bnkx,nz), u0(0:n), w0(0:n), dvh(0:n,bnkx,nz))
    allocate(uhat(0:n,bnkx,nz), what(0:n,bnkx,nz), dv(0:n,bnkx,nz), vc(0:n,bnkx,nz))
    allocate(pu(ny,nxd,anz), pv(ny,nxd,anz), pw(ny,nxd,anz))
    allocate(Bz(ny,bnkx,nzd,1), Az(ny,nkx,anz,1), Af(ny,nkxd,anz))
    kxt = kxh(kxlo:kxhi);  kzt = kzh
    kxw = real(kxh(kxlo:kxhi), wp);  kzw = real(kzh, wp);  inv_dy_d = inv_dy_h
    vhat = v0(:,kxlo:kxhi,:);  ehat = e0(:,kxlo:kxhi,:)
    u0 = (0.0_wp,0.0_wp);  w0 = (0.0_wp,0.0_wp)
    call recover_uw(ny, bnkx, nz, kxt, kzt, vhat, ehat, u0, w0, uhat, what, dv)
    call divergence_check(n, bnkx, nz, kxw, kzw, uhat, vhat, what, dvh, maxdiv_loc)
    call demote(n, bnkx, nz, vhat, vc)
    call par_to_physical_da_r2c(pp, d, ny, nx, nz, nxd, nzd, bnkx, anz, uhat, Bz, Az, Af, pu)
    call par_to_physical_da_r2c(pp, d, ny, nx, nz, nxd, nzd, bnkx, anz, vc,   Bz, Az, Af, pv)
    call par_to_physical_da_r2c(pp, d, ny, nx, nz, nxd, nzd, bnkx, anz, what, Bz, Az, Af, pw)
    call cfl_freq_max(ny, nxd, anz, pu, pv, pw, inv_dx, inv_dz, inv_dy_d, fmax_loc)
    istat = cudaDeviceSynchronize()
    call par_s3d_destroy_da_r2c(pp)
    fmax_out   = real(fmax_loc,   real64);  call par_reduce_max(fmax_out)
    maxdiv_out = real(maxdiv_loc, real64);  call par_reduce_max(maxdiv_out)
    deallocate(kxt, kzt, kxw, kzw, inv_dy_d, vhat, ehat, u0, w0, dvh, &
               uhat, what, dv, vc, pu, pv, pw, Bz, Az, Af)
  end subroutine run_slab

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

end program test_driver_state_mp
