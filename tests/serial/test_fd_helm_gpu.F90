!===============================================================================
! test_fd_helm_gpu.F90 -- gate for the batched GPU FD Helmholtz
! solve (fd_helm_gpu_mod) against the HOST fdops oracle, per mode:
!   host:  ab = fd_helm_dirichlet(-D2 + mu) ; fd_band_factor ; fd_band_solve
!          on Re and Im parts separately (real band, complex RHS)
!   GPU :  fd_helm_solve_gpu on all modes at once
! Cases:
!   (1) lam = 3.7, generic wavenumbers, tiny chunk (7) -> exercises the
!       multi-chunk host loop and slot indexing;
!   (2) lam = 0 INCLUDING the (kx,kz) = (0,0) mode -> mu = 0, the pure
!       Dirichlet Laplacian (the vif H1 solve's corner case);
!   (3) default chunk (single launch) must match the tiny-chunk result
!       bit-exactly (chunking is pure batching, same arithmetic per mode).
! Grid: GLC(48) DESCENDING (+1..-1), kl = 4 -- the ygrid file convention.
! Agreement gate: 1e-11 relative (same algorithm, same order, GPU vs host;
! FMA/codegen differences allow last-digits drift, nothing more).
!===============================================================================
program test_fd_helm_gpu
  use kinds,           only: dp, ip
  use cudafor
  use fdops_mod,       only: fd_build_bands, fd_helm_dirichlet, fd_band_factor, &
                             fd_band_solve
  use fd_helm_gpu_mod, only: fd_helm_solve_gpu
  implicit none
  integer(ip), parameter :: n = 48, ny = n + 1, kl = 4, nx = 8, nz = 6
  real(dp), parameter :: PI = 3.141592653589793238462643_dp
  real(dp)    :: y(ny), d1b(2*kl+1, ny), d2b(2*kl+1, ny)
  real(dp)    :: kxh(nx), kzh(nz)
  complex(dp) :: fh(0:n, nx, nz), uh(0:n, nx, nz), u2(0:n, nx, nz)
  real(dp),    device :: kx_d(nx), kz_d(nz), d2b_d(2*kl+1, 0:n)
  complex(dp), device :: f_d(0:n, nx, nz), u_d(0:n, nx, nz)
  integer(ip) :: i, j, ix, iz, kk, istat
  logical     :: allok
  allok = .true.

  do j = 0, n
     y(j+1) = cos(PI*real(j,dp)/real(n,dp))          ! descending +1..-1
  end do
  y(1) = 1.0_dp;  y(ny) = -1.0_dp
  call fd_build_bands(ny, kl, y, d1b, d2b)
  d2b_d = d2b                                        ! host cols 1..ny -> device 0..n

  do i = 1, nx
     kxh(i) = 0.7_dp*real(i-1, dp)                   ! includes kx = 0
  end do
  do i = 1, nz
     kk = i - 1;  if (kk > nz/2) kk = kk - nz
     kzh(i) = 1.3_dp*real(kk, dp)                    ! includes kz = 0
  end do
  kx_d = kxh;  kz_d = kzh

  do iz = 1, nz; do ix = 1, nx; do j = 0, n          ! deterministic complex RHS
     fh(j, ix, iz) = cmplx(fval(j, ix, iz), -0.5_dp*fval(j, iz, ix+1), dp)
  end do; end do; end do

  call one_case(3.7_dp, 'lam = 3.7, chunk = 7 (multi-chunk)', 7_ip)
  call one_case(0.0_dp, 'lam = 0 incl. the mu = 0 (0,0) mode', 7_ip)

  ! (3) chunking is pure batching: default single launch == chunk-7, bit-exact
  f_d = fh
  call fd_helm_solve_gpu(n, kl, nx, nz, 3.7_dp, kx_d, kz_d, d2b_d, f_d, u_d, 7_ip)
  istat = cudaDeviceSynchronize();  uh = u_d
  f_d = fh
  call fd_helm_solve_gpu(n, kl, nx, nz, 3.7_dp, kx_d, kz_d, d2b_d, f_d, u_d)
  istat = cudaDeviceSynchronize();  u2 = u_d
  call report('chunk=7 vs default chunk: BIT-EXACT (pure batching)', &
              all(uh == u2))

  if (allok) then; print '(a)', 'ALL TESTS PASSED'
  else;            print '(a)', 'SOME TESTS FAILED'; end if
  if (.not. allok) error stop 1

contains

  real(dp) function fval(j, a, b) result(v)
    integer(ip), intent(in) :: j, a, b
    integer(ip) :: h
    h = modulo(j*1009_ip + a*2027_ip + b*3001_ip + j*a*17_ip, 100003_ip)
    v = real(h, dp)/100003.0_dp - 0.5_dp
  end function fval

  subroutine one_case(lam, label, chunk)
    real(dp),     intent(in) :: lam
    character(*), intent(in) :: label
    integer(ip),  intent(in) :: chunk
    real(dp) :: ab(2*kl+1, ny), abf(2*kl+1, ny), br(ny), bi(ny)
    real(dp) :: mu, dmax, ref
    integer(ip) :: iix, iiz, jj
    logical :: okf, okall

    f_d = fh
    call fd_helm_solve_gpu(n, kl, nx, nz, lam, kx_d, kz_d, d2b_d, f_d, u_d, chunk)
    istat = cudaDeviceSynchronize()
    uh = u_d

    dmax = 0.0_dp;  ref = 0.0_dp;  okall = .true.
    do iiz = 1, nz
       do iix = 1, nx
          mu = kxh(iix)**2 + kzh(iiz)**2 + lam
          call fd_helm_dirichlet(ny, kl, d2b, mu, ab)
          abf = ab
          call fd_band_factor(ny, kl, abf, okf)
          if (.not. okf) then; okall = .false.; cycle; end if
          do jj = 1, ny
             br(jj) = real(fh(jj-1, iix, iiz))
             bi(jj) = aimag(fh(jj-1, iix, iiz))
          end do
          br(1) = 0.0_dp;  br(ny) = 0.0_dp                ! homogeneous walls
          bi(1) = 0.0_dp;  bi(ny) = 0.0_dp
          call fd_band_solve(ny, kl, abf, br)
          call fd_band_solve(ny, kl, abf, bi)
          do jj = 1, ny
             dmax = max(dmax, abs(real(uh(jj-1, iix, iiz)) - br(jj)) &
                            + abs(aimag(uh(jj-1, iix, iiz)) - bi(jj)))
             ref  = max(ref, abs(br(jj)) + abs(bi(jj)))
          end do
       end do
    end do
    if (.not. okall) then
       call report(label//' -- host factor ok', .false.);  return
    end if
    if (dmax > 1.0e-11_dp*max(1.0_dp, ref)) &
       print '(a,es10.2,a,es10.2)', '   [fd-helm] max diff=', dmax, '  ref=', ref
    call report(label//': GPU == host oracle (rel 1e-11)', &
                dmax <= 1.0e-11_dp*max(1.0_dp, ref))
  end subroutine one_case

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. okk) allok = .false.
  end subroutine report

end program test_fd_helm_gpu
