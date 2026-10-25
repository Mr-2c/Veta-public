!===============================================================================
! test_driver_diag_mp.F90 -- distributed driver: validate the
! driver's global-reduction DIAGNOSTICS in a distributed setting: the three
! quantities the time-stepping driver reduces across ranks every step/monitor,
! computed on each rank's slab + combined with the reduction primitives, must equal the
! single-GPU values.
!
!   CFL frequency  : fmax = max_grid(|u|inv_dx + |v|inv_dy + |w|inv_dz)  [cfl_freq_max]
!                    -> per-z-slab local max -> par_reduce_MAX  (drives adaptive dt)
!   max divergence : maxdiv = max_modes |i kx u + Dv + i kz w|          [divergence_check]
!                    -> per-kx-slab local max -> par_reduce_MAX  (health check)
!   watchdog sum   : s = sum |Re|+|Im| of a spectral field             [field_sumabs_c]
!                    -> per-kx-slab local sum -> par_reduce_SUM  (NaN/blow-up monitor)
!
! This wires par_reduce_max / par_reduce_sum (validated in isolation) into the
! ACTUAL driver diagnostic routines (cfl_freq_max, divergence_check, field_sumabs_c
! -- all extent-agnostic, validated single-GPU), on real distributed slabs. It is a
! pure reduction/diagnostic test: the input fields are arbitrary DETERMINISTIC
! functions of the GLOBAL index (bounded in [-1,1], identical on every rank, no
! rank-0 broadcast), NOT a physical state -- which is exactly what makes the
! reduced diagnostics genuinely non-trivial (O(1), not ~0 as a divergence-free
! recovered field would give) and the test a strong check of the reduction wiring.
!
! BIT-EXACT for the MAX reductions (max over identical per-element values is the
! same however the grid is partitioned). ROUND-OFF for the SUM (re-association +
! the nondeterministic on-device sum reduction), so s uses a relative tolerance --
! this also documents that cross-rank sums are round-off, not bit-exact (expected
! and fine for statistics). par_reduce_and fails the job on any rank's failure.
!
! tp tier: cfl velocities and the divergence uhat/what are tp, so run dp AND sp;
! the compared MAX values are still bit-exact within a build (same tp on both
! sides). field_sumabs operates on a wp(=dp) field, unaffected by tp.
! Precondition (as the transform): P <= min(nkx, nzd) so no slab is empty (here
! min(17,24)=17); empty-slab guards below are defensive.
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_driver_diag_mp   Run: bash run_all_par_tests.sh run -k test_driver_diag_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!   module load nvhpc openmpi nccl
! Build: TP=dp|sp bash scripts/build/build_par.sh test_driver_diag_mp   Run: bash run_all_par_tests.sh run -k test_driver_diag_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_driver_diag_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,      only: tp, wp, dp, ip
  use cudafor
  use decomp_mod, only: decomp_t, decomp_setup
  use cfl_mod,    only: cfl_inv_dy, cfl_freq_max
  use stats_mod,  only: divergence_check, field_sumabs_c
  use par_mod
  implicit none

  integer(ip), parameter :: n = 16, nx = 32, nz = 16
  integer(ip), parameter :: ny = n+1, nkx = nx/2+1, nxd = 3*nx/2, nzd = 3*nz/2
  real(dp),    parameter :: PI = acos(-1.0_dp)

  type(decomp_t) :: d
  integer(ip) :: bnkx, kxlo, kxhi, anz, zlo, zhi, ix, iz, k
  integer     :: istat
  logical     :: ok_cfl, ok_div, ok_sum, all_ok

  real(dp) :: inv_dx, inv_dz, y(0:n), inv_dy_h(0:n), kxh(nkx), kzh(nz)
  real(dp) :: fmax_ref, maxdiv_ref, s_ref
  real(real64) :: fmax_g, maxdiv_g, s_g, tol_max, tol_sum

  ! host full fields (built identically on every rank from the global index)
  real(tp),    allocatable :: pu(:,:,:), pv(:,:,:), pw(:,:,:)         ! (ny,nxd,nzd)
  complex(tp), allocatable :: uh(:,:,:), wh(:,:,:)                    ! (0:n,nkx,nz)
  complex(wp), allocatable :: vh(:,:,:)                              ! (0:n,nkx,nz)

  call par_init()
  tol_max = 1.0e-12_real64
  if (par_rank == 0) print '(a,i0,a)', '[driver-diag] world size = ', par_nproc, ' rank(s)'

  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
  bnkx = d%b_nkx_loc;  kxlo = d%b_kx0 + 1;  kxhi = d%b_kx0 + bnkx     ! Layout-B kx-slab
  anz  = d%ad_nz_loc;  zlo  = d%ad_z0 + 1;  zhi  = d%ad_z0 + anz      ! Layout-A dealiased-z slab

  ! grid metrics (nominal grid; values are arbitrary but realistic)
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

  allocate(pu(ny,nxd,nzd), pv(ny,nxd,nzd), pw(ny,nxd,nzd))
  allocate(uh(0:n,nkx,nz), wh(0:n,nkx,nz), vh(0:n,nkx,nz))
  call build_fields()

  call run_ref(fmax_ref, maxdiv_ref, s_ref)     ! single-GPU, full field (per rank)
  call run_slab(fmax_g, maxdiv_g, s_g)          ! this rank's slabs + reductions

  ! relative tolerance for the SUM (re-associated + nondeterministic reduction)
  tol_sum = 1.0e-9_real64 * max(abs(real(s_ref, real64)), 1.0_real64)

  ok_cfl = (abs(fmax_g   - real(fmax_ref,   real64)) <= tol_max) .and. (fmax_ref   > 0.0_dp)
  ok_div = (abs(maxdiv_g - real(maxdiv_ref, real64)) <= tol_max) .and. (maxdiv_ref > 0.0_dp)
  ok_sum = (abs(s_g      - real(s_ref,      real64)) <= tol_sum) .and. (s_ref      > 0.0_dp)

  call par_reduce_and(ok_cfl)
  call par_reduce_and(ok_div)
  call par_reduce_and(ok_sum)
  all_ok = ok_cfl .and. ok_div .and. ok_sum
  if (par_rank == 0) then
     call report('CFL fmax: distributed par_reduce_max == single-GPU', ok_cfl)
     call report('max-divergence: distributed par_reduce_max == single-GPU', ok_div)
     call report('field_sumabs watchdog: distributed par_reduce_sum == single-GPU', ok_sum)
     if (all_ok) then
        print '(a)', 'ALL TESTS PASSED'
     else
        print '(a)', 'SOME TESTS FAILED'
     end if
  end if

  call par_finalize()
  if (.not. all_ok) error stop 1

contains

  ! deterministic real in [-1,1) from a selector + global indices (rank-invariant)
  real(dp) function hr(sel, a, b, c) result(v)
    integer(ip), intent(in) :: sel, a, b, c
    integer(ip) :: h
    h = modulo(sel*7919_ip + a*1009_ip + b*2027_ip + c*3001_ip + a*c*13_ip + b*17_ip, 100003_ip)
    v = real(h, dp)/100003.0_dp*2.0_dp - 1.0_dp
  end function hr

  subroutine build_fields()
    do iz = 1, nzd
       do ix = 1, nxd
          do k = 1, ny
             pu(k,ix,iz) = real(hr(1_ip, k, ix, iz), tp)
             pv(k,ix,iz) = real(hr(2_ip, k, ix, iz), tp)
             pw(k,ix,iz) = real(hr(3_ip, k, ix, iz), tp)
          end do
       end do
    end do
    do iz = 1, nz
       do ix = 1, nkx
          do k = 0, n
             uh(k,ix,iz) = cmplx(hr(4_ip,k,ix,iz), hr(5_ip,k,ix,iz), tp)
             wh(k,ix,iz) = cmplx(hr(6_ip,k,ix,iz), hr(7_ip,k,ix,iz), tp)
             vh(k,ix,iz) = cmplx(hr(8_ip,k,ix,iz), hr(9_ip,k,ix,iz), wp)
          end do
       end do
    end do
  end subroutine build_fields

  ! single-GPU reference: all three diagnostics on the FULL field (redundant per rank)
  subroutine run_ref(fmax, maxdiv, s)
    real(dp), intent(out) :: fmax, maxdiv, s
    real(tp),    device, allocatable :: puf(:,:,:), pvf(:,:,:), pwf(:,:,:)
    real(dp),    device, allocatable :: inv_dy_d(:)
    complex(tp), device, allocatable :: uhf(:,:,:), whf(:,:,:)
    complex(wp), device, allocatable :: vhf(:,:,:), dvh(:,:,:)
    real(wp),    device, allocatable :: kxd(:), kzd(:)
    allocate(puf(ny,nxd,nzd), pvf(ny,nxd,nzd), pwf(ny,nxd,nzd), inv_dy_d(ny))
    allocate(uhf(0:n,nkx,nz), whf(0:n,nkx,nz), vhf(0:n,nkx,nz), dvh(0:n,nkx,nz))
    allocate(kxd(nkx), kzd(nz))
    puf = pu;  pvf = pv;  pwf = pw;  inv_dy_d = inv_dy_h
    uhf = uh;  whf = wh;  vhf = vh;  kxd = real(kxh, wp);  kzd = real(kzh, wp)
    call cfl_freq_max(ny, nxd, nzd, puf, pvf, pwf, inv_dx, inv_dz, inv_dy_d, fmax)
    call divergence_check(n, nkx, nz, kxd, kzd, uhf, vhf, whf, dvh, maxdiv)
    call field_sumabs_c(ny, nkx, nz, vhf, s)
    istat = cudaDeviceSynchronize()
    deallocate(puf, pvf, pwf, inv_dy_d, uhf, whf, vhf, dvh, kxd, kzd)
  end subroutine run_ref

  ! distributed: this rank's slabs -> local diagnostics -> global reductions
  subroutine run_slab(fmax_out, maxdiv_out, s_out)
    real(real64), intent(out) :: fmax_out, maxdiv_out, s_out
    real(tp),    device, allocatable :: pus(:,:,:), pvs(:,:,:), pws(:,:,:)
    real(dp),    device, allocatable :: inv_dy_d(:)
    complex(tp), device, allocatable :: uhs(:,:,:), whs(:,:,:)
    complex(wp), device, allocatable :: vhs(:,:,:), dvh(:,:,:)
    real(wp),    device, allocatable :: kxd(:), kzd(:)
    real(dp) :: fmax_loc, maxdiv_loc, s_loc
    allocate(inv_dy_d(ny), kzd(nz))
    inv_dy_d = inv_dy_h;  kzd = real(kzh, wp)

    ! --- CFL on this rank's dealiased z-slab ---
    fmax_loc = 0.0_dp
    if (anz > 0) then
       allocate(pus(ny,nxd,anz), pvs(ny,nxd,anz), pws(ny,nxd,anz))
       pus = pu(:,:,zlo:zhi);  pvs = pv(:,:,zlo:zhi);  pws = pw(:,:,zlo:zhi)
       call cfl_freq_max(ny, nxd, anz, pus, pvs, pws, inv_dx, inv_dz, inv_dy_d, fmax_loc)
       deallocate(pus, pvs, pws)
    end if
    fmax_out = real(fmax_loc, real64)
    call par_reduce_max(fmax_out)

    ! --- divergence + watchdog sum on this rank's kx-slab ---
    maxdiv_loc = 0.0_dp;  s_loc = 0.0_dp
    if (bnkx > 0) then
       allocate(uhs(0:n,bnkx,nz), whs(0:n,bnkx,nz), vhs(0:n,bnkx,nz))
       allocate(dvh(0:n,bnkx,nz), kxd(bnkx))
       uhs = uh(:,kxlo:kxhi,:);  whs = wh(:,kxlo:kxhi,:);  vhs = vh(:,kxlo:kxhi,:)
       kxd = real(kxh(kxlo:kxhi), wp)
       call divergence_check(n, bnkx, nz, kxd, kzd, uhs, vhs, whs, dvh, maxdiv_loc)
       call field_sumabs_c(ny, bnkx, nz, vhs, s_loc)
       deallocate(uhs, whs, vhs, dvh, kxd)
    end if
    istat = cudaDeviceSynchronize()
    maxdiv_out = real(maxdiv_loc, real64);  call par_reduce_max(maxdiv_out)
    s_out      = real(s_loc, real64);       call par_reduce_sum(s_out)
    deallocate(inv_dy_d, kzd)
  end subroutine run_slab

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

end program test_driver_diag_mp
