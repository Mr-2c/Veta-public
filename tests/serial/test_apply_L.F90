!===============================================================================
! test_apply_L.F90  --  validate the GPU L-operator apply_L = (D^2 - k^2)
!
! Reference: for each (kx,kz) mode, take the column of complex Chebyshev
! coefficients, apply the validated CPU second-derivative recurrence cheb_d2 to
! its real and imaginary parts, and subtract k^2 times the original column.
! Compare against the GPU apply_L, which instead forms D^2 by composing the
! first-derivative recurrence cheb_deriv_c twice. Agreement to round-off both
! confirms the GPU port and cross-checks compose-twice == direct cheb_d2.
!
! Build (GPU node, nvhpc):
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -c cheb_shen.F90
!   nvfortran -cuda -c cheb_ops_mod.F90
!   nvfortran -cuda -c test_apply_L.F90
!   nvfortran -cuda -o test_apply_L test_apply_L.o cheb_ops_mod.o cheb_shen.o types.o
!   ./test_apply_L
!===============================================================================
program test_apply_L
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,        only: wp, dp, ip
  use cheb_shen,    only: cheb_d2
  use cheb_ops_mod, only: apply_L
  use cudafor
  implicit none

  integer(ip) :: cfg
  integer(ip), parameter :: ncfg = 4
  integer(ip) :: nyv(ncfg) = [  9, 17, 16, 33 ]
  integer(ip) :: nxv(ncfg) = [  8, 16, 12, 24 ]
  integer(ip) :: nzv(ncfg) = [  6, 12,  8, 16 ]
  logical :: all_pass

  all_pass = .true.
  do cfg = 1, ncfg
     call run_one(nyv(cfg), nxv(cfg), nzv(cfg))
  end do
  if (all_pass) then
     print *, 'ALL TESTS PASSED'
  else
     print *, 'SOME TESTS FAILED'
  end if

contains

  subroutine run_one(ny, nx, nz)
    integer(ip), intent(in) :: ny, nx, nz
    complex(wp), allocatable :: q(:,:,:), Lq(:,:,:), Lref(:,:,:)
    complex(wp), device, allocatable :: q_d(:,:,:), Lq_d(:,:,:), scr_d(:,:,:)
    real(wp),    allocatable :: kxh(:), kzh(:)
    real(wp), device, allocatable :: kx_d(:), kz_d(:)
    real(dp) :: are(0:ny-1), aim(0:ny-1), cre(0:ny-1), cim(0:ny-1)
    real(dp) :: k2, err, denom
    real(wp) :: rr, ri
    integer(ip) :: ix, iz, iy, kk, istat

    allocate(q(ny,nx,nz), Lq(ny,nx,nz), Lref(ny,nx,nz))
    allocate(kxh(nx), kzh(nz))

    ! cuFFT-order integer wavenumbers (0,1,..,n/2,-(n/2-1),..,-1), as grid_mod
    do ix = 1, nx
       kk = ix-1; if (kk > nx/2) kk = kk - nx
       kxh(ix) = real(kk, wp)
    end do
    do iz = 1, nz
       kk = iz-1; if (kk > nz/2) kk = kk - nz
       kzh(iz) = real(kk, wp)
    end do

    ! random complex Chebyshev coefficients per mode
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
          do iy = 1, ny
             call random_number(rr); call random_number(ri)
             q(iy,ix,iz) = cmplx(rr - 0.5_wp, ri - 0.5_wp, wp)
          end do
       end do
    end do

    ! CPU reference Lref = (D^2 - k^2) q, column by column
    do iz = 1, nz
       do ix = 1, nx
          k2 = real(kxh(ix),dp)**2 + real(kzh(iz),dp)**2
          do iy = 1, ny
             are(iy-1) = real(q(iy,ix,iz), dp)
             aim(iy-1) = real(aimag(q(iy,ix,iz)), dp)
          end do
          call cheb_d2(ny-1, are, cre)
          call cheb_d2(ny-1, aim, cim)
          do iy = 1, ny
             Lref(iy,ix,iz) = cmplx( cre(iy-1) - k2*real(q(iy,ix,iz),dp), &
                                     cim(iy-1) - k2*real(aimag(q(iy,ix,iz)),dp), wp)
          end do
       end do
    end do

    ! GPU apply_L
    allocate(q_d(ny,nx,nz), Lq_d(ny,nx,nz), scr_d(ny,nx,nz))
    allocate(kx_d(nx), kz_d(nz))
    q_d = q;  kx_d = kxh;  kz_d = kzh
    call apply_L(ny, nx, nz, kx_d, kz_d, q_d, Lq_d, scr_d)
    istat = cudaDeviceSynchronize()
    Lq = Lq_d

    err = 0.0_dp; denom = 0.0_dp
    do iz = 1, nz
       do ix = 1, nx
          do iy = 1, ny
             err   = max(err,   abs(Lq(iy,ix,iz) - Lref(iy,ix,iz)))
             denom = max(denom, abs(Lref(iy,ix,iz)))
          end do
       end do
    end do

    write(*,'(a,i0,a,i0,a,i0,a)') ' --- ny=',ny,' nx=',nx,' nz=',nz,' ---'
    call report('apply_L = (D^2 - k^2)q  vs CPU cheb_d2 (rel)', err/denom, 1.0e-12_dp)

    deallocate(q, Lq, Lref, kxh, kzh, q_d, Lq_d, scr_d, kx_d, kz_d)
  end subroutine run_one

  subroutine report(name, e, tol)
    character(*), intent(in) :: name
    real(dp),     intent(in) :: e, tol
    character(6) :: tag
    if (e <= tol) then
       tag = '[PASS]'
    else
       tag = '[FAIL]'
       all_pass = .false.
    end if
    write(*,'(a,2x,a,2x,a,es12.4,2x,a,es10.2)') tag, name, 'err=', e, 'tol=', tol
  end subroutine report

end program test_apply_L
