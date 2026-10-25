!===============================================================================
! test_fft3d_r2c.F90  --  [R2C] validate the FUSED rank-2 R2C/C2R
! transforms added to fft3d_mod (real x inner, complex z outer, y-leading,
! batched over y).  These are the R2C counterpart of plan_xz_c2c.
!
! This increment ADDS the R2C plans without wiring them into spectral3d yet, so
! the existing C2C path is untouched (test_fft3d still passes unchanged). Here we
! check the new plans in isolation, exactly as test_fft3d first validated C2C.
!
! Tests (n1=ny, nxr, nz deliberately distinct & nxr even so a wrong stride shows):
!   1. Round-trip: R2C then C2R, /(nxr*nz) = identity on a random REAL field.
!   2. Known single x-mode: P(y,x,z)=cos(2*pi*m*x/nxr) -> only |S(:,m+1,:)| nonzero.
!   3. Hermitian/realness sanity: the C2R of an arbitrary-but-consistent spectrum
!      is real (it is by construction of C2R); we check the round-trip residual.
!
! NaN/Inf-hardened. Build/run ON A GPU NODE:
!   nvfortran -cuda -c types.F90
!   nvfortran -cuda -cudalib=cufft -c transforms_mod.F90
!   nvfortran -cuda -cudalib=cufft -c fft3d_mod.F90
!   nvfortran -cuda -cudalib=cufft test_fft3d_r2c.F90 \
!             fft3d_mod.o transforms_mod.o types.o -o test_fft3d_r2c
!   ./test_fft3d_r2c
!===============================================================================
program test_fft3d_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,          only: tp, ip
  use transforms_mod, only: tf_destroy
  use fft3d_mod,      only: plan_xz_r2c, plan_xz_c2r, exec_xz_r2c, exec_xz_c2r
  use cudafor
  implicit none

  integer(ip), parameter :: ny = 4, nxr = 8, nz = 6
  integer(ip), parameter :: nkx = nxr/2 + 1
  real(real64) :: tol
  logical :: all_ok = .true.
  integer :: istat, pf, pi

  real(tp)    :: Ph(ny,nxr,nz), Rh(ny,nxr,nz)
  complex(tp) :: Sh(ny,nkx,nz)
  real(tp),    device :: Pd(ny,nxr,nz), Rd(ny,nxr,nz)
  complex(tp), device :: Sd(ny,nkx,nz)
  integer(ip) :: ix, iy, iz, m
  real(tp)    :: rr

  tol = merge(1.0e-10_real64, 1.0e-4_real64, tp == real64)

  call plan_xz_r2c(pf, ny, nxr, nz)
  call plan_xz_c2r(pi, ny, nxr, nz)

  ! ===== Test 1: random real field, R2C -> C2R -> /(nxr*nz) = identity =========
  do iz = 1, nz; do ix = 1, nxr; do iy = 1, ny
     call random_number(rr);  Ph(iy,ix,iz) = rr - 0.5_tp
  end do; end do; end do
  Pd = Ph
  call exec_xz_r2c(pf, Pd, Sd)
  call exec_xz_c2r(pi, Sd, Rd)          ! note: may consume Sd
  istat = cudaDeviceSynchronize()
  Rh = Rd
  call report_real('R2C->C2R round-trip = identity', Ph, Rh/real(nxr*nz, tp), ny, nxr, nz, tol, all_ok)

  ! ===== Test 2: single x-mode cos(2*pi*m*x/nxr) -> one spectral column ========
  m = 2_ip
  do iz = 1, nz; do ix = 1, nxr; do iy = 1, ny
     Ph(iy,ix,iz) = cos(2.0_tp*acos(-1.0_tp)*real(m,tp)*real(ix-1,tp)/real(nxr,tp))
  end do; end do; end do
  Pd = Ph
  call exec_xz_r2c(pf, Pd, Sd)
  istat = cudaDeviceSynchronize()
  Sh = Sd
  call report_mode('single x-mode lands in column m+1 only', Sh, ny, nkx, nz, m, tol, all_ok)

  call tf_destroy(pf)
  call tf_destroy(pi)
  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine report_real(label, a, b, n1, n2, n3, tol, ok)
    character(*), intent(in)    :: label
    integer(ip),  intent(in)    :: n1, n2, n3
    real(tp),     intent(in)    :: a(n1,n2,n3), b(n1,n2,n3)
    real(real64), intent(in)    :: tol
    logical,      intent(inout) :: ok
    real(real64) :: dm, d
    integer(ip)  :: i, j, k
    dm = 0.0_real64
    do k=1,n3; do j=1,n2; do i=1,n1
       d = abs(real(a(i,j,k),real64) - real(b(i,j,k),real64))
       if (.not. (d == d)) d = huge(1.0_real64)
       dm = max(dm, d)
    end do; end do; end do
    if (dm /= dm .or. dm > tol) then
       print '(a,a,a,es12.4)', '  [FAIL] ', label, '  max|diff|=', dm;  ok = .false.
    else
       print '(a,a,a,es12.4)', '  [ok] ', label, '  max|diff|=', dm
    end if
  end subroutine report_real

  ! check |S(:,k,:)| is ~0 except k=m+1 (1-based), which is ~nxr/2 (cos amplitude)
  subroutine report_mode(label, S, n1, nk, n3, m, tol, ok)
    character(*), intent(in)    :: label
    integer(ip),  intent(in)    :: n1, nk, n3, m
    complex(tp),  intent(in)    :: S(n1,nk,n3)
    real(real64), intent(in)    :: tol
    logical,      intent(inout) :: ok
    real(real64) :: off, on_, a
    integer(ip)  :: i, k, kk
    off = 0.0_real64;  on_ = 0.0_real64
    do k=1,n3; do kk=1,nk; do i=1,n1
       a = abs(S(i,kk,k))
       if (.not. (a == a)) a = huge(1.0_real64)
       if (kk == m+1) then; on_ = max(on_, a); else; off = max(off, a); end if
    end do; end do; end do
    if (off /= off .or. off > tol .or. on_ < 0.1_real64) then
       print '(a,a,a,es12.4,a,es12.4)', '  [FAIL] ', label, '  off-peak=', off, ' on-peak=', on_
       ok = .false.
    else
       print '(a,a,a,es12.4,a,es12.4)', '  [ok] ', label, '  off-peak=', off, ' on-peak=', on_
    end if
  end subroutine report_mode

end program test_fft3d_r2c
