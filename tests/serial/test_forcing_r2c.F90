!===============================================================================
! test_forcing_r2c.F90  --  [R2C] confirm forcing_mod
! (assemble_hv_heta) is EXTENT-AGNOSTIC: calling it with nkx=nx/2+1 as the
! x-extent (and the nkx-length positive-kx array) reproduces the full-nx result
! on the retained half kx in [0,nkx-1], to machine precision.
!
! assemble_hv_heta is local in (kx,kz) -- every kernel reads kx(ix),kz(iz) and
! the same (iy,ix,iz) element, cheb_deriv_c uses a plain nx*nz batch count, no
! ix+-1 / nx-ix / mirror -- so the R2C path reuses the SAME source with nx->nkx.
! This test proves it, so nonlinear_mod (3a-iii) can pass nkx with no change.
!
! Checks:
!   1. EQUIVALENCE: hv,heta from the nkx call == the full-nx call on kx<nkx.
!   2. NONZERO GUARD: the reference forcings are genuinely nonzero (so an
!      identical-zero output from a bug COMMON to both paths cannot false-pass;
!      correctness of assemble_hv_heta itself is covered by test_forcing.F90).
!
! NaN/Inf-hardened. Build & run ON A GPU NODE:
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -c cheb_ops_mod.F90
!   nvfortran -cuda -c forcing_mod.F90
!   nvfortran -cuda test_forcing_r2c.F90 forcing_mod.o cheb_ops_mod.o types.o -o test_forcing_r2c
!   ./test_forcing_r2c
!===============================================================================
program test_forcing_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,       only: tp, wp, ip
  use forcing_mod, only: assemble_hv_heta
  use cudafor
  implicit none

  integer(ip), parameter :: ny = 9, nx = 8, nz = 6
  integer(ip), parameter :: nkx = nx/2 + 1
  real(real64) :: tol
  logical :: all_ok = .true.
  integer :: istat
  integer(ip) :: iy, ix, iz

  real(wp)    :: kxf_h(nx), kzf_h(nz), kxr_h(nkx)      ! wavenumbers feed wp dummies
  real(wp)    :: re, im
  complex(tp) :: Hx_h(ny,nx,nz), Hy_h(ny,nx,nz), Hz_h(ny,nx,nz)   ! Hx,Hy,Hz tp
  complex(wp) :: hvf_h(ny,nx,nz), hef_h(ny,nx,nz)        ! full-nx results (hv,heta wp)
  complex(wp) :: hvr_h(ny,nkx,nz), her_h(ny,nkx,nz)      ! nkx results (hv,heta wp)

  real(wp),    device :: kxf(nx), kzf(nz), kxr(nkx)
  complex(tp), device :: Hxf(ny,nx,nz), Hyf(ny,nx,nz), Hzf(ny,nx,nz)
  complex(wp), device :: hvf(ny,nx,nz), hef(ny,nx,nz)              ! hv,heta wp
  complex(tp), device :: gf(ny,nx,nz)                              ! g scratch tp
  complex(tp), device :: dgf(ny,nx,nz)                  ! D(g) scratch (tp, full-nx)
  complex(tp), device :: Hxr(ny,nkx,nz), Hyr(ny,nkx,nz), Hzr(ny,nkx,nz)
  complex(wp), device :: hvr(ny,nkx,nz), her(ny,nkx,nz)            ! hv,heta wp
  complex(tp), device :: gr(ny,nkx,nz)                             ! g scratch tp
  complex(tp), device :: dgr(ny,nkx,nz)                 ! D(g) scratch (tp, nkx)

  tol = merge(1.0e-12_real64, 1.0e-4_real64, wp == real64)

  do ix = 1, nx;  kxf_h(ix) = real(ix-1, wp) * 0.7_wp;  end do
  do iz = 1, nz;  kzf_h(iz) = real(iz-1, wp) * 1.3_wp;  end do
  do ix = 1, nkx; kxr_h(ix) = kxf_h(ix);                end do

  do iz = 1, nz; do ix = 1, nx; do iy = 1, ny
     call random_number(re); call random_number(im); Hx_h(iy,ix,iz) = cmplx(re-0.5_wp,im-0.5_wp,wp)
     call random_number(re); call random_number(im); Hy_h(iy,ix,iz) = cmplx(re-0.5_wp,im-0.5_wp,wp)
     call random_number(re); call random_number(im); Hz_h(iy,ix,iz) = cmplx(re-0.5_wp,im-0.5_wp,wp)
  end do; end do; end do

  ! ===== full-nx reference =====================================================
  kxf = kxf_h; kzf = kzf_h; Hxf = Hx_h; Hyf = Hy_h; Hzf = Hz_h
  call assemble_hv_heta(ny, nx, nz, kxf, kzf, Hxf, Hyf, Hzf, hvf, hef, gf, dgf)
  istat = cudaDeviceSynchronize()
  hvf_h = hvf; hef_h = hef

  ! ===== nkx run on the first nkx x-columns ====================================
  kxr = kxr_h
  do iz = 1, nz; do ix = 1, nkx; do iy = 1, ny
     Hxr(iy,ix,iz) = Hx_h(iy,ix,iz); Hyr(iy,ix,iz) = Hy_h(iy,ix,iz); Hzr(iy,ix,iz) = Hz_h(iy,ix,iz)
  end do; end do; end do
  call assemble_hv_heta(ny, nkx, nz, kxr, kzf, Hxr, Hyr, Hzr, hvr, her, gr, dgr)
  istat = cudaDeviceSynchronize()
  hvr_h = hvr; her_h = her

  ! ===== compare on the retained half + nonzero guard =========================
  call cmp('hv   (nkx vs nx)', hvf_h, hvr_h, tol)
  call cmp('heta (nkx vs nx)', hef_h, her_h, tol)
  call assert_nonzero('hv   nonzero', hvf_h)
  call assert_nonzero('heta nonzero', hef_h)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine cmp(label, full, half, tol)
    character(*), intent(in) :: label
    complex(wp),  intent(in) :: full(ny,nx,nz), half(ny,nkx,nz)
    real(real64), intent(in) :: tol
    real(real64) :: dm, d
    integer(ip)  :: i, j, k
    dm = 0.0_real64
    do k=1,nz; do j=1,nkx; do i=1,ny
       d = abs(full(i,j,k) - half(i,j,k))
       if (.not. (d == d)) d = huge(1.0_real64)
       dm = max(dm, d)
    end do; end do; end do
    if (dm /= dm .or. dm > tol) then
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|diff|=', dm;  all_ok = .false.
    else
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|diff|=', dm
    end if
  end subroutine cmp

  subroutine assert_nonzero(label, f)
    character(*), intent(in) :: label
    complex(wp),  intent(in) :: f(ny,nx,nz)
    real(real64) :: mx, a
    integer(ip)  :: i, j, k
    mx = 0.0_real64
    do k=1,nz; do j=1,nkx; do i=1,ny       ! probe the retained half only
       a = abs(f(i,j,k))
       if (.not. (a == a)) a = huge(1.0_real64)
       mx = max(mx, a)
    end do; end do; end do
    if (mx > 0.1_real64 .and. mx < huge(1.0_real64)) then
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|f|=', mx
    else
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|f|=', mx;  all_ok = .false.
    end if
  end subroutine assert_nonzero

end program test_forcing_r2c
