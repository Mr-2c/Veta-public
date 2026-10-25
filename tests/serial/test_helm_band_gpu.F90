!===============================================================================
! test_helm_band_gpu.F90  --  validate the batched GPU banded Helmholtz solve
! against the CPU helm_band_solve, mode by mode.
!
! For each grid: structure arrays from helm_band_setup (host). Random complex
! ftilde per mode. CPU reference solves each mode's (S+mu B) system (real and
! imaginary parts via helm_band_solve) with mu = kx^2 + kz^2 + lambda. GPU solves
! all modes in one launch. Compare. The comparison HARD-FAILS on any NaN/Inf so a
! non-finite result can never masquerade as a pass (see the helm_band history).
!
! Build:
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran -c helm_band_mod.F90
!   nvfortran -cuda -c helm_band_gpu_mod.F90
!   nvfortran -cuda -c test_helm_band_gpu.F90
!   nvfortran -cuda -o test_helm_band_gpu test_helm_band_gpu.o helm_band_gpu_mod.o \
!             helm_band_mod.o cheb_shen.o types.o
!   ./test_helm_band_gpu
!===============================================================================
program test_helm_band_gpu
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,         only: dp, ip
  use helm_band_mod, only: helm_band_setup, helm_band_solve
  use helm_band_gpu_mod, only: helm_band_solve_gpu
  use cudafor
  implicit none

  ! Grids: 3 small + the PRODUCTION oracle size (n=129, nx=nkx=65, nz=128). The
  ! big one is the real correctness gate for the column-leading helm_band rewrite:
  ! main_par --validate canNOT catch a helm_band-internal bug (oracle AND slab call
  ! the same module, so it cancels in their diff), so this CPU-oracle unit test is
  ! THE independent check. nx=65 is not a multiple of the 64-thread block, so it
  ! also exercises the col->(ix,iz) decode across the warp/nx wrap.
  integer(ip) :: nv(4) = [ 8, 17, 32, 129 ]
  integer(ip) :: xv(4) = [ 8, 12, 16,  65 ]
  integer(ip) :: zv(4) = [ 6,  8, 12, 128 ]
  real(dp), parameter :: lambda = 1.0e3_dp
  integer(ip) :: c
  logical :: all_pass

  all_pass = .true.
  do c = 1, 4
     call run_grid(nv(c), xv(c), zv(c), lambda, all_pass)
  end do
  if (all_pass) then
     print *, 'ALL TESTS PASSED'
  else
     print *, 'SOME TESTS FAILED'
  end if

contains

  subroutine run_grid(n, nx, nz, lambda, all_pass)
    integer(ip), intent(in)    :: n, nx, nz
    real(dp),    intent(in)    :: lambda
    logical,     intent(inout) :: all_pass
    ! host
    real(dp) :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
    real(dp) :: kxh(nx), kzh(nz)
    complex(dp), allocatable :: ftil(:,:,:), ugpu(:,:,:), uref(:,:,:)
    real(dp) :: fre(0:n-2), fim(0:n-2), ure(0:n-2), uim(0:n-2)
    real(dp) :: mu, rr, ri, err, denom, d
    logical  :: bad
    integer(ip) :: ix, iz, k, kk, istat
    ! device
    real(dp),    device, allocatable :: kx_d(:), kz_d(:)
    real(dp),    device, allocatable :: sdiag_d(:), stail_d(:), bdiag_d(:)
    complex(dp), device, allocatable :: ftil_d(:,:,:), uhat_d(:,:,:)

    allocate(ftil(0:n-2,nx,nz), ugpu(0:n-2,nx,nz), uref(0:n-2,nx,nz))

    call helm_band_setup(n, sdiag, stail, bdiag, boff)

    do ix = 1, nx
       kk = ix-1; if (kk > nx/2) kk = kk - nx
       kxh(ix) = real(kk, dp)
    end do
    do iz = 1, nz
       kk = iz-1; if (kk > nz/2) kk = kk - nz
       kzh(iz) = real(kk, dp)
    end do

    block   ! [repro] fixed seed, applied once per run -> the suite is a bit-identity oracle
            ! across the precision phases (replaces a bare call random_seed())
      integer              :: rs_n_, rs_j_
      integer, allocatable :: rs_seed_(:)
      logical, save        :: rs_done_ = .false.
      if (.not. rs_done_) then
         call random_seed(size = rs_n_)
         allocate(rs_seed_(rs_n_))
         rs_seed_ = [ (1789 + 101*rs_j_, rs_j_ = 0, rs_n_ - 1) ]
         call random_seed(put = rs_seed_)
         rs_done_ = .true.
      end if
    end block
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, n-2
             call random_number(rr); call random_number(ri)
             ftil(k,ix,iz) = cmplx(rr-0.5_dp, ri-0.5_dp, dp)
          end do
       end do
    end do

    ! CPU reference: solve each mode (real & imag parts)
    do iz = 1, nz
       do ix = 1, nx
          mu = kxh(ix)**2 + kzh(iz)**2 + lambda
          do k = 0, n-2
             fre(k) = real (ftil(k,ix,iz))
             fim(k) = aimag(ftil(k,ix,iz))
          end do
          call helm_band_solve(n, mu, sdiag, stail, bdiag, boff, fre, ure)
          call helm_band_solve(n, mu, sdiag, stail, bdiag, boff, fim, uim)
          do k = 0, n-2
             uref(k,ix,iz) = cmplx(ure(k), uim(k), dp)
          end do
       end do
    end do

    ! GPU
    allocate(kx_d(nx), kz_d(nz), sdiag_d(0:n-2), stail_d(0:n-2), bdiag_d(0:n-2))
    allocate(ftil_d(0:n-2,nx,nz), uhat_d(0:n-2,nx,nz))
    kx_d = kxh; kz_d = kzh
    sdiag_d = sdiag; stail_d = stail; bdiag_d = bdiag
    ftil_d = ftil
    call helm_band_solve_gpu(n, nx, nz, lambda, kx_d, kz_d, sdiag_d, stail_d, &
                             bdiag_d, boff, ftil_d, uhat_d)
    istat = cudaDeviceSynchronize()
    ugpu = uhat_d

    err = 0.0_dp; denom = 0.0_dp; bad = .false.
    do iz = 1, nz
       do ix = 1, nx
          do k = 0, n-2
             d = abs(ugpu(k,ix,iz) - uref(k,ix,iz))
             if (.not. (d == d)) bad = .true.
             if (abs(ugpu(k,ix,iz)) > huge(1.0_dp)) bad = .true.
             err   = max(err,   d)
             denom = max(denom, abs(uref(k,ix,iz)))
          end do
       end do
    end do
    if (denom == 0.0_dp) denom = 1.0_dp

    write(*,'(a,i0,a,i0,a,i0,a)') ' --- n=',n,'  nx=',nx,'  nz=',nz,' ---'
    if (bad) then
       call report('GPU banded solve vs CPU (rel)', 1.0e30_dp, 1.0e-9_dp, all_pass)
    else
       call report('GPU banded solve vs CPU (rel)', err/denom, 1.0e-9_dp, all_pass)
    end if

    deallocate(ftil, ugpu, uref)
    deallocate(kx_d, kz_d, sdiag_d, stail_d, bdiag_d, ftil_d, uhat_d)
  end subroutine run_grid

  subroutine report(name, e, tol, all_pass)
    character(*), intent(in)    :: name
    real(dp),     intent(in)    :: e, tol
    logical,      intent(inout) :: all_pass
    character(6) :: tag
    if (e <= tol) then
       tag = '[PASS]'
    else
       tag = '[FAIL]'; all_pass = .false.
    end if
    write(*,'(a,2x,a,2x,a,es12.4,2x,a,es10.2)') tag, name, 'err=', e, 'tol=', tol
  end subroutine report

end program test_helm_band_gpu
