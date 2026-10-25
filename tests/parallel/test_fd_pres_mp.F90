!===============================================================================
! test_fd_pres_mp.F90 -- FD-y Neumann pressure solve, manufactured
! solution. Exercises the exact production path: pres_set_fd ->
! pres_gpu_setup_host (FD branch: fd_pres_setup_mode per mode) ->
! pres_solve_gpu (FD branch: fd_helm_solve_gpu particular + one-sided wall
! derivs + the shared closure kernel).
!
! MMS: Pi(y) = (1-y^2)^2 + y^3  (degree 4: the k2=0 column is stencil-EXACT;
! k2>0 columns carry hyperbolic homogeneous corrections, so genuine truncation
! enters there -- the gate is calibrated for both. Nonzero wall derivatives
! exercise the Neumann closure: DPi(+-1) = 3 at both walls).
!   R  = (D^2 - k^2) Pi = (12y^2 - 4) + 6y - k^2 Pi
!   g+- = DPi(+-1) = -4y(1-y^2)*... evaluated analytically below.
! Modes: several k^2 > 0 plus the k^2 = 0 column (pure-Neumann gauge: compare
! after removing each column's mean-node value).
!
! Runs at any P >= 1 (the solve is mode-local; each rank tests its own copy --
! no decomposition needed, P>1 just repeats the identical single-GPU check).
! Grid: GLC nodes AS A FILE-STYLE array (matches the FD KMM smoke posture) --
! proves non-uniform spacing handling without needing an on-disk grid file.
!===============================================================================
program test_fd_pres_mp
  use kinds,      only: dp, ip
  use cudafor
  use par_mod,    only: par_init, par_finalize, par_rank, par_reduce_and
  use fdops_mod,  only: fd_build_bands, fd_wall_weights
  use par_pressure_mod, only: pres_set_fd, pres_gpu_setup_host, pres_solve_gpu
  implicit none

  integer(ip), parameter :: n = 128, kl = 4, nx = 3, nz = 2   ! 6 modes, k2 incl. 0
  integer(ip), parameter :: ny = n + 1
  real(dp),    parameter :: PI = 3.141592653589793238462643_dp
  real(dp),    parameter :: TOL = 2.0e-6_dp    ! conditioning-limited (mirrors fd_vsolve MMS)

  real(dp) :: y(ny), d1b(2*kl+1, ny), d2b(2*kl+1, ny), wlo(ny), whi(ny)
  real(dp) :: kxh(nx), kzh(nz), k2, piex(ny), dmax, dcol, gauge_ex, gauge_nu
  complex(dp), allocatable :: Rh(:,:,:), Pih(:,:,:), gph(:,:), gmh(:,:)
  real(dp),    device, allocatable :: kx_d(:), kz_d(:), sd_d(:), st_d(:), bd_d(:)
  real(dp),    device, allocatable :: pp_d(:,:,:), pm_d(:,:,:), Ci_d(:,:,:,:)
  complex(dp), device, allocatable :: R_d(:,:,:), gp_d(:,:), gm_d(:,:), Pi_d(:,:,:)
  integer(ip) :: j, ix, iz, istat
  logical :: allok

  call par_init()
  do j = 1, ny                                     ! GLC nodes, descending +1..-1
     y(j) = cos(PI*real(j-1,dp)/real(n,dp))
  end do
  call fd_build_bands(ny, kl, y, d1b, d2b)
  call fd_wall_weights(ny, kl, y, wlo, whi)
  call pres_set_fd(ny, kl, y, d2b, wlo, whi)       ! flip the pressure backend

  kxh = (/ 0.0_dp, 1.0_dp, 3.0_dp /)               ! kx=kz=0 column = the gauge mode
  kzh = (/ 0.0_dp, 2.0_dp /)

  allocate(Rh(0:n,nx,nz), Pih(0:n,nx,nz), gph(nx,nz), gmh(nx,nz))
  do j = 1, ny
     piex(j) = (1.0_dp - y(j)**2)**2 + y(j)**3
  end do
  do iz = 1, nz
     do ix = 1, nx
        k2 = kxh(ix)**2 + kzh(iz)**2
        do j = 1, ny                               ! R = Pi'' - k^2 Pi (exact)
           Rh(j-1,ix,iz) = cmplx(12.0_dp*y(j)**2 - 4.0_dp + 6.0_dp*y(j) &
                                 - k2*piex(j), 0.0_dp, dp)
        end do
        ! DPi = -4y(1-y^2) + 3y^2 -> DPi(+1) = 3, DPi(-1) = 3
        gph(ix,iz) = (3.0_dp, 0.0_dp)
        gmh(ix,iz) = (3.0_dp, 0.0_dp)
     end do
  end do

  allocate(kx_d(nx), kz_d(nz), sd_d(0:n-2), st_d(0:n-2), bd_d(0:n-2))
  allocate(pp_d(0:n,nx,nz), pm_d(0:n,nx,nz), Ci_d(2,2,nx,nz))
  allocate(R_d(0:n,nx,nz), gp_d(nx,nz), gm_d(nx,nz), Pi_d(0:n,nx,nz))
  kx_d = kxh;  kz_d = kzh
  sd_d = 0.0_dp;  st_d = 0.0_dp;  bd_d = 0.0_dp    ! cheb bands unused on the FD branch

  call pres_gpu_setup_host(n, nx, nz, kxh, kzh, pp_d, pm_d, Ci_d)
  R_d = Rh;  gp_d = gph;  gm_d = gmh
  call pres_solve_gpu(n, nx, nz, kx_d, kz_d, sd_d, st_d, bd_d, 0.0_dp, &
                      pp_d, pm_d, Ci_d, R_d, gp_d, gm_d, Pi_d)
  Pih = Pi_d;  istat = cudaDeviceSynchronize()

  dmax = 0.0_dp
  do iz = 1, nz
     do ix = 1, nx
        ! gauge: k2=0 fixes Pi only up to a constant -> anchor both at node n/2
        gauge_nu = real(Pih(n/2,ix,iz), dp);  gauge_ex = piex(n/2+1)
        dcol = 0.0_dp
        do j = 1, ny                 ! full complex diff: catches leaked imag too
           dcol = max(dcol, abs(Pih(j-1,ix,iz) - cmplx(gauge_nu, 0.0_dp, dp) &
                                - cmplx(piex(j) - gauge_ex, 0.0_dp, dp)))
        end do
        if (par_rank == 0) write(*,'(a,f6.2,a,es10.2)') &
           '   [fd-pres] k^2 = ', kxh(ix)**2 + kzh(iz)**2, '   max|Pi - Pi_exact| = ', dcol
        dmax = max(dmax, dcol)
     end do
  end do

  allok = (dmax <= TOL) .and. (dmax == dmax)
  call par_reduce_and(allok)
  if (par_rank == 0) then
     write(*,'(a,es10.2,a,es8.1,a)') ' --- test_fd_pres_mp:  worst ', dmax, '   (tol ', TOL, ')'
     if (allok) then; write(*,'(a)') ' ALL TESTS PASSED'
     else;            write(*,'(a)') ' SOME TESTS FAILED'; end if
  end if
  call par_finalize()
  if (.not. allok) error stop 1
end program test_fd_pres_mp
