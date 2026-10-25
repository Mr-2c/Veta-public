!===============================================================================
! test_velocity_r2c.F90  --  [R2C] confirm velocity_mod
! (recover_uw, compute_vorticity) is EXTENT-AGNOSTIC: calling it with nkx=nx/2+1
! as the x-extent (and the nkx-length positive-kx array) reproduces the full-nx
! result on the retained half kx in [0,nkx-1], to machine precision.
!
! These routines are local in (kx,kz) -- every kernel reads kx(ix),kz(iz) and the
! same (iy,ix,iz) element, with no ix+-1 / nx-ix / mirror coupling -- so the R2C
! path reuses the SAME source with nx replaced by nkx. This test proves that, so
! nonlinear_mod (3a-iii) can pass nkx with no velocity_mod change.
!
! Check: build random vhat,ehat on the full grid; run recover_uw+compute_vorticity
! at full nx; independently run them at nkx on the first nkx x-columns; require
! uhat,what,ox,oy,oz agree on kx in [0,nkx-1].
!
! NaN/Inf-hardened. Build & run ON A GPU NODE:
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -c cheb_ops_mod.F90
!   nvfortran -cuda -c velocity_mod.F90
!   nvfortran -cuda test_velocity_r2c.F90 velocity_mod.o cheb_ops_mod.o types.o -o test_velocity_r2c
!   ./test_velocity_r2c
!===============================================================================
program test_velocity_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,        only: tp, wp, ip
  use velocity_mod, only: recover_uw, compute_vorticity
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
  complex(wp) :: vh_h(ny,nx,nz), eh_h(ny,nx,nz), u0_h(ny), w0_h(ny)   ! vhat,ehat,u0,w0 wp
  ! full-nx results (uhat,what,vorticity are tp)
  complex(tp) :: uf_h(ny,nx,nz), wf_h(ny,nx,nz), oxf_h(ny,nx,nz), oyf_h(ny,nx,nz), ozf_h(ny,nx,nz)
  ! nkx results (tp)
  complex(tp) :: ur_h(ny,nkx,nz), wr_h(ny,nkx,nz), oxr_h(ny,nkx,nz), oyr_h(ny,nkx,nz), ozr_h(ny,nkx,nz)

  ! device, full nx
  real(wp),    device :: kxf(nx), kzf(nz)
  complex(wp), device :: vhf(ny,nx,nz), ehf(ny,nx,nz), u0d(ny), w0d(ny)   ! wp inputs
  complex(tp), device :: uf(ny,nx,nz), wf(ny,nx,nz), dvf(ny,nx,nz)        ! tp outputs/scratch
  complex(tp), device :: oxf(ny,nx,nz), oyf(ny,nx,nz), ozf(ny,nx,nz), duf(ny,nx,nz), dwf(ny,nx,nz)
  ! device, nkx
  real(wp),    device :: kxr(nkx)
  complex(wp), device :: vhr(ny,nkx,nz), ehr(ny,nkx,nz)                   ! wp inputs
  complex(tp), device :: ur(ny,nkx,nz), wr(ny,nkx,nz), dvr(ny,nkx,nz)     ! tp outputs/scratch
  complex(tp), device :: oxr(ny,nkx,nz), oyr(ny,nkx,nz), ozr(ny,nkx,nz), dur(ny,nkx,nz), dwr(ny,nkx,nz)

  tol = merge(1.0e-12_real64, 1.0e-4_real64, wp == real64)

  ! wavenumbers (arbitrary but distinct); kxr = first nkx of kxf
  do ix = 1, nx;  kxf_h(ix) = real(ix-1, wp) * 0.7_wp;  end do
  do iz = 1, nz;  kzf_h(iz) = real(iz-1, wp) * 1.3_wp;  end do
  do ix = 1, nkx; kxr_h(ix) = kxf_h(ix);                end do

  do iz = 1, nz; do ix = 1, nx; do iy = 1, ny
     call random_number(re); call random_number(im)
     vh_h(iy,ix,iz) = cmplx(re-0.5_wp, im-0.5_wp, wp)
     call random_number(re); call random_number(im)
     eh_h(iy,ix,iz) = cmplx(re-0.5_wp, im-0.5_wp, wp)
  end do; end do; end do
  do iy = 1, ny
     call random_number(re); call random_number(im); u0_h(iy) = cmplx(re,im,wp)
     call random_number(re); call random_number(im); w0_h(iy) = cmplx(re,im,wp)
  end do

  ! ===== full-nx reference =====================================================
  kxf = kxf_h; kzf = kzf_h; vhf = vh_h; ehf = eh_h; u0d = u0_h; w0d = w0_h
  call recover_uw(ny, nx, nz, kxf, kzf, vhf, ehf, u0d, w0d, uf, wf, dvf)
  call compute_vorticity(ny, nx, nz, kxf, kzf, vhf, ehf, uf, wf, oxf, oyf, ozf, duf, dwf)
  istat = cudaDeviceSynchronize()
  uf_h = uf; wf_h = wf; oxf_h = oxf; oyf_h = oyf; ozf_h = ozf

  ! ===== nkx run on the first nkx x-columns ====================================
  kxr = kxr_h
  do iz = 1, nz; do ix = 1, nkx; do iy = 1, ny
     vhr(iy,ix,iz) = vh_h(iy,ix,iz);  ehr(iy,ix,iz) = eh_h(iy,ix,iz)
  end do; end do; end do
  call recover_uw(ny, nkx, nz, kxr, kzf, vhr, ehr, u0d, w0d, ur, wr, dvr)
  call compute_vorticity(ny, nkx, nz, kxr, kzf, vhr, ehr, ur, wr, oxr, oyr, ozr, dur, dwr)
  istat = cudaDeviceSynchronize()
  ur_h = ur; wr_h = wr; oxr_h = oxr; oyr_h = oyr; ozr_h = ozr

  ! ===== compare on the retained half =========================================
  call cmp('uhat   (nkx vs nx)', uf_h, ur_h, tol)
  call cmp('what   (nkx vs nx)', wf_h, wr_h, tol)
  call cmp('omega_x(nkx vs nx)', oxf_h, oxr_h, tol)
  call cmp('omega_y(nkx vs nx)', oyf_h, oyr_h, tol)
  call cmp('omega_z(nkx vs nx)', ozf_h, ozr_h, tol)

  ! nonzero guard: the differential test above cancels any bug COMMON to both
  ! paths (e.g. a silent zeroing). Assert the reference outputs are genuinely
  ! nonzero so identical-zero output cannot false-pass. (recover_uw/vorticity are
  ! independently validated for correctness by test_velocity.F90; this just makes
  ! the equivalence test self-contained.)
  call assert_nonzero('uhat   nonzero', uf_h)
  call assert_nonzero('omega_x nonzero', oxf_h)
  call assert_nonzero('omega_z nonzero', ozf_h)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine cmp(label, full, half, tol)
    character(*), intent(in) :: label
    complex(tp),  intent(in) :: full(ny,nx,nz), half(ny,nkx,nz)
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
    complex(tp),  intent(in) :: f(ny,nx,nz)
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

end program test_velocity_r2c
