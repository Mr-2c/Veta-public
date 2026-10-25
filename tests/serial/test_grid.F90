!===============================================================================
! test_grid.F90  --  CPU validation of grid_mod.
!
! Build & run:
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran -c grid_mod.F90
!   nvfortran test_grid.F90 grid_mod.o cheb_shen.o types.o -o test_grid
!   ./test_grid
!
! Tests:
!   1. kx values        = i           (Lx = 2 pi).
!   2. kz values & order = j or j-nz  (Lz = 2 pi); checks the negative-freq wrap.
!   3. k2 = kx^2 + kz^2.
!   4. dealias sizes (3 nx/2, 3 nz/2) and Nyquist mask.
!   5. GLC nodes: endpoints +-1, symmetry, formula.
!   6. STRONG TEST: differentiate f(z) = sum c_m e^{i m z} spectrally using
!      grid%kz, via a hand-written reference DFT (no cuFFT), and compare to the
!      analytic derivative. Includes a negative mode (m=-1, landing at j=nz-1)
!      so a wavenumber sign/ordering bug cannot pass.
!===============================================================================

program test_grid
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,    only: wp, dp, ip
  use types,    only: grid_t, params_t
  use grid_mod, only: build_grid, destroy_grid
  implicit none

  real(dp), parameter :: TWOPI = 2.0_dp*acos(-1.0_dp)
  logical :: all_ok = .true.

  type(grid_t)   :: grid
  type(params_t) :: p
  integer(ip), parameter :: nx = 16, ny = 33, nz = 8

  p%lx = TWOPI;  p%lz = TWOPI
  call build_grid(grid, p, nx, ny, nz)

  call test_kx()
  call test_kz()
  call test_k2()
  call test_dealias()
  call test_nodes()
  call test_spectral_derivative()

  call destroy_grid(grid)

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine check(name, err, tol)
    character(*), intent(in) :: name
    real(dp),     intent(in) :: err, tol
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', err <= tol)
    print '(a,a,a,es12.4,a,es10.2,a)', '[',tag,'] '//name//'  err=', err, '  tol=', tol
    if (err > tol) all_ok = .false.
  end subroutine check

  subroutine test_kx()
    real(dp) :: err
    integer(ip) :: i
    err = 0.0_dp
    do i = 0, grid%nkx-1
       err = max(err, abs(real(grid%kx(i),dp) - real(i,dp)))
    end do
    call check('kx(i) = i (Lx=2pi)', err, 1.0e-12_dp)
  end subroutine test_kx

  subroutine test_kz()
    real(dp) :: err, expect
    integer(ip) :: j
    err = 0.0_dp
    do j = 0, grid%nkz-1
       if (j <= nz/2) then
          expect = real(j,dp)
       else
          expect = real(j-nz,dp)
       end if
       err = max(err, abs(real(grid%kz(j),dp) - expect))
    end do
    call check('kz order incl. negative freqs', err, 1.0e-12_dp)
  end subroutine test_kz

  subroutine test_k2()
    real(dp) :: err
    integer(ip) :: i, j
    err = 0.0_dp
    do j = 0, grid%nkz-1
       do i = 0, grid%nkx-1
          err = max(err, abs(grid%k2(i,j) - &
               (real(grid%kx(i),dp)**2 + real(grid%kz(j),dp)**2)))
       end do
    end do
    call check('k2 = kx^2 + kz^2', err, 1.0e-12_dp)
  end subroutine test_k2

  subroutine test_dealias()
    real(dp) :: err
    err = 0.0_dp
    if (grid%nxd /= 3*nx/2) err = 1.0_dp
    if (grid%nzd /= 3*nz/2) err = 1.0_dp
    err = max(err, abs(real(grid%mask(nx/2,0),dp)))   ! x Nyquist zeroed
    err = max(err, abs(real(grid%mask(0,nz/2),dp)))   ! z Nyquist zeroed
    err = max(err, abs(real(grid%mask(1,1),dp) - 1.0_dp))  ! interior kept
    call check('dealias sizes + Nyquist mask', err, 1.0e-12_dp)
  end subroutine test_dealias

  subroutine test_nodes()
    real(dp) :: err
    integer(ip) :: j
    err = max(abs(grid%yglc(0) - 1.0_dp), abs(grid%yglc(ny-1) + 1.0_dp))
    do j = 0, ny-1
       err = max(err, abs(grid%yglc(j) - cos(TWOPI*0.5_dp*real(j,dp)/real(ny-1,dp))))
       err = max(err, abs(grid%yglc(j) + grid%yglc(ny-1-j)))   ! antisymmetry
    end do
    call check('GLC nodes endpoints/symmetry/formula', err, 1.0e-12_dp)
  end subroutine test_nodes

  !-- reference DFT (naive O(N^2), unnormalized forward / 1/N inverse) -------
  subroutine dft_fwd(n, f, fc)
    integer(ip), intent(in)  :: n
    complex(dp), intent(in)  :: f(0:n-1)
    complex(dp), intent(out) :: fc(0:n-1)
    integer(ip) :: j, q
    complex(dp) :: s
    do j = 0, n-1
       s = (0.0_dp, 0.0_dp)
       do q = 0, n-1
          s = s + f(q)*exp(cmplx(0.0_dp, -TWOPI*real(j,dp)*real(q,dp)/real(n,dp), dp))
       end do
       fc(j) = s
    end do
  end subroutine dft_fwd

  subroutine dft_inv(n, fc, f)
    integer(ip), intent(in)  :: n
    complex(dp), intent(in)  :: fc(0:n-1)
    complex(dp), intent(out) :: f(0:n-1)
    integer(ip) :: j, q
    complex(dp) :: s
    do q = 0, n-1
       s = (0.0_dp, 0.0_dp)
       do j = 0, n-1
          s = s + fc(j)*exp(cmplx(0.0_dp, TWOPI*real(j,dp)*real(q,dp)/real(n,dp), dp))
       end do
       f(q) = s/real(n,dp)
    end do
  end subroutine dft_inv

  subroutine test_spectral_derivative()
    integer(ip), parameter :: nm = 3
    integer(ip) :: modes(nm) = [ 1_ip, 2_ip, -1_ip ]
    complex(dp) :: coeff(nm)
    complex(dp) :: f(0:nz-1), fc(0:nz-1), dfc(0:nz-1), df(0:nz-1)
    complex(dp) :: analytic
    real(dp) :: z, err
    integer(ip) :: q, j, m
    coeff = [ cmplx(1.0_dp,0.0_dp,dp), cmplx(0.5_dp,-0.3_dp,dp), cmplx(2.0_dp,0.0_dp,dp) ]
    ! sample f on the physical z grid (Lz = 2 pi)
    do q = 0, nz-1
       z = TWOPI*real(q,dp)/real(nz,dp)
       f(q) = (0.0_dp,0.0_dp)
       do m = 1, nm
          f(q) = f(q) + coeff(m)*exp(cmplx(0.0_dp, real(modes(m),dp)*z, dp))
       end do
    end do
    ! spectral derivative via grid%kz
    call dft_fwd(nz, f, fc)
    do j = 0, nz-1
       dfc(j) = cmplx(0.0_dp, real(grid%kz(j),dp), dp) * fc(j)
    end do
    call dft_inv(nz, dfc, df)
    ! compare to analytic derivative
    err = 0.0_dp
    do q = 0, nz-1
       z = TWOPI*real(q,dp)/real(nz,dp)
       analytic = (0.0_dp,0.0_dp)
       do m = 1, nm
          analytic = analytic + coeff(m)*cmplx(0.0_dp, real(modes(m),dp), dp) &
                     *exp(cmplx(0.0_dp, real(modes(m),dp)*z, dp))
       end do
       err = max(err, abs(df(q) - analytic))
    end do
    call check('spectral d/dz via grid%kz (DFT)', err, 1.0e-12_dp)
  end subroutine test_spectral_derivative

end program test_grid
