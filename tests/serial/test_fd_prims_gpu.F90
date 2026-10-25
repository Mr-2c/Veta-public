!===============================================================================
! test_fd_prims_gpu.F90 -- gate for the batched FD device
! primitives that the v-solve swap consumes:
!   (1) HOST fd_wall_weights exactness: the full-order one-sided wall rows must
!       reproduce d/dy(y^p) = p*(+-1)^(p-1) at both walls EXACTLY (p <= 2*kl) --
!       the same complete algebraic characterization as the band gates.
!   (2) fd_wall_derivs_gpu == host dots of the same weight rows on a complex
!       field (rel 1e-12): the batched device twin is pure data plumbing.
!   (3) fd_apply_L_gpu == host banded matvec (Re/Im) minus k^2 u (rel 1e-12).
! Grid: GLC(48) DESCENDING, kl = 4 (the ygrid convention). Host oracles come
! from fdops_mod -- already polynomial/MMS-gated in test_fdops.
!===============================================================================
program test_fd_prims_gpu
  use kinds,           only: dp, ip
  use cudafor
  use fdops_mod,       only: fd_build_bands, fd_apply_band, fd_wall_weights
  use fd_helm_gpu_mod, only: fd_wall_derivs_gpu, fd_apply_L_gpu
  implicit none
  integer(ip), parameter :: n = 48, ny = n + 1, kl = 4, nx = 8, nz = 6
  real(dp), parameter :: PI = 3.141592653589793238462643_dp
  real(dp)    :: y(ny), d1b(2*kl+1, ny), d2b(2*kl+1, ny)
  real(dp)    :: wlo(ny), whi(ny), kxh(nx), kzh(nz)
  real(dp)    :: br(ny), bi(ny), gr(ny), gi(ny)
  complex(dp) :: fh(0:n, nx, nz), dloh(nx, nz), dhih(nx, nz), Lh(0:n, nx, nz)
  real(dp),    device :: kx_d(nx), kz_d(nz), d2b_d(2*kl+1, 0:n), wlo_d(0:n), whi_d(0:n)
  complex(dp), device :: f_d(0:n, nx, nz), Lu_d(0:n, nx, nz)
  complex(dp), device :: dlo_d(nx, nz), dhi_d(nx, nz)
  real(dp)    :: sref, e, sw, k2
  complex(dp) :: s1, s2
  integer(ip) :: i, j, ix, iz, kk, p, istat
  logical     :: ok, allok
  allok = .true.

  do j = 0, n
     y(j+1) = cos(PI*real(j,dp)/real(n,dp))
  end do
  y(1) = 1.0_dp;  y(ny) = -1.0_dp
  call fd_build_bands(ny, kl, y, d1b, d2b)
  call fd_wall_weights(ny, kl, y, wlo, whi)
  d2b_d = d2b;  wlo_d = wlo;  whi_d = whi
  sw = sum(abs(wlo)) + sum(abs(whi))

  ! (1) wall-weight polynomial exactness, p <= 2*kl ----------------------------
  ok = .true.
  do p = 0, 2*kl
     s1 = (0.0_dp, 0.0_dp);  s2 = (0.0_dp, 0.0_dp)
     do j = 1, ny
        s1 = s1 + wlo(j)*y(j)**p
        s2 = s2 + whi(j)*y(j)**p
     end do
     sref = 0.0_dp
     if (p >= 1) sref = real(p, dp)                 ! d/dy y^p at +1
     if (abs(real(s2) - sref) > 1.0e-10_dp*max(1.0_dp, sw)) ok = .false.
     sref = 0.0_dp
     if (p >= 1) sref = real(p, dp)*real((-1)**(p-1), dp)   ! at -1
     if (abs(real(s1) - sref) > 1.0e-10_dp*max(1.0_dp, sw)) ok = .false.
  end do
  call report('wall weights: d/dy(y^p) exact at both walls, p <= 2*kl', ok)

  ! deterministic complex field ------------------------------------------------
  do i = 1, nx
     kxh(i) = 0.7_dp*real(i-1, dp)
  end do
  do i = 1, nz
     kk = i - 1;  if (kk > nz/2) kk = kk - nz
     kzh(i) = 1.3_dp*real(kk, dp)
  end do
  kx_d = kxh;  kz_d = kzh
  do iz = 1, nz; do ix = 1, nx; do j = 0, n
     fh(j, ix, iz) = cmplx(fval(j, ix, iz), -0.5_dp*fval(j, iz, ix+1), dp)
  end do; end do; end do
  f_d = fh

  ! (2) batched wall derivatives vs host dots ----------------------------------
  call fd_wall_derivs_gpu(n, nx, nz, wlo_d, whi_d, f_d, dlo_d, dhi_d)
  istat = cudaDeviceSynchronize()
  dloh = dlo_d;  dhih = dhi_d
  e = 0.0_dp;  sref = 0.0_dp
  do iz = 1, nz
     do ix = 1, nx
        s1 = (0.0_dp, 0.0_dp);  s2 = (0.0_dp, 0.0_dp)
        do j = 1, ny
           s1 = s1 + wlo(j)*fh(j-1, ix, iz)
           s2 = s2 + whi(j)*fh(j-1, ix, iz)
        end do
        e = max(e, abs(dloh(ix,iz) - s1), abs(dhih(ix,iz) - s2))
        sref = max(sref, abs(s1), abs(s2))
     end do
  end do
  call report('fd_wall_derivs_gpu == host dots (rel 1e-12)', &
              e <= 1.0e-12_dp*max(1.0_dp, sref))

  ! (3) batched apply_L vs host banded matvec - k^2 u ---------------------------
  call fd_apply_L_gpu(n, kl, nx, nz, kx_d, kz_d, d2b_d, f_d, Lu_d)
  istat = cudaDeviceSynchronize()
  Lh = Lu_d
  e = 0.0_dp;  sref = 0.0_dp
  do iz = 1, nz
     do ix = 1, nx
        k2 = kxh(ix)**2 + kzh(iz)**2
        do j = 1, ny
           br(j) = real(fh(j-1, ix, iz));  bi(j) = aimag(fh(j-1, ix, iz))
        end do
        call fd_apply_band(ny, kl, d2b, br, gr)
        call fd_apply_band(ny, kl, d2b, bi, gi)
        do j = 1, ny
           s1 = cmplx(gr(j), gi(j), dp) - k2*fh(j-1, ix, iz)
           e = max(e, abs(Lh(j-1, ix, iz) - s1))
           sref = max(sref, abs(s1))
        end do
     end do
  end do
  call report('fd_apply_L_gpu == host banded matvec - k^2 u (rel 1e-12)', &
              e <= 1.0e-12_dp*max(1.0_dp, sref))

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

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. okk) allok = .false.
  end subroutine report

end program test_fd_prims_gpu
