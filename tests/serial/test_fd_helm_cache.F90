!===============================================================================
! test_fd_helm_cache.F90 -- gate for the FD Helmholtz FACTOR CACHE:
!   (1,2) BIT-IDENTITY: the cached fast path (prefactored mode-first bands +
!         substitution) vs the rebuild-per-solve path, from the same RHS, at
!         a production lambda (1/a_1 with the ev_stage expression chain) and
!         at lam = 0 (the vif first leg). The factor/substitution arithmetic
!         is the rebuild kernel's verbatim, so equality is to the BIT.
!   (3)   REGISTRY BOOKKEEPING: hits/misses count as designed (2 hits from
!         rows 1-2; an unregistered lambda takes the rebuild path and counts
!         one miss while still returning the bitwise rebuild answer).
!   (4)   CAP REFUSAL: cap_gb = 0 refuses (built = .false., footprint > 0
!         reported), clears the cache, and solves still work (rebuild path).
!   (5)   GEOMETRY GUARD: a cache built for another (nx,nz) is skipped
!         (neither hit nor miss); the solve equals the no-cache answer.
!   (6)   OPERATOR GUARD: a cache built from d2b_A
!         must NOT serve a same-shape solve called with a DIFFERENT band
!         d2b_B -- the address stamp forces the rebuild path, whose answer is
!         gated bitwise against the no-cache solve of d2b_B. (Pre-fix, the
!         shape+lam key silently returned operator A's solution.)
! Grid GLC(64), kl=4, nx=7, nz=6; Re=2800, dt=5e-3 (the smoke's lambdas).
!===============================================================================
program test_fd_helm_cache
  use kinds,           only: dp, ip
  use cudafor
  use cheb_shen,       only: glc_nodes
  use mean_mod,        only: MCSUB
  use fdops_mod,       only: fd_build_bands
  use fd_helm_gpu_mod, only: fd_helm_solve_gpu, fd_helm_cache_build, &
                             fd_helm_cache_clear, fd_helm_cache_stats
  implicit none
  integer(ip), parameter :: n = 64, ny = n + 1, kl = 4
  integer(ip), parameter :: nx = 7, nz = 6, nx2 = 5
  real(dp),    parameter :: RE = 2800.0_dp, DT = 5.0e-3_dp
  real(dp) :: yA(ny), d1A(2*kl+1,ny), d2A(2*kl+1,ny)
  real(dp) :: yB(ny), dB1(2*kl+1,ny), dB2(2*kl+1,ny)
  real(dp) :: kxh(nx), kzh(nz), lams(4), bet, aa, cgb
  complex(dp) :: fh(0:n,nx,nz), ur(0:n,nx,nz), uc(0:n,nx,nz)
  complex(dp) :: fh2(0:n,nx2,nz), u2a(0:n,nx2,nz), u2b(0:n,nx2,nz)
  real(dp),    device :: kx_d(nx), kz_d(nz), d2b_d(2*kl+1,0:n), kx2_d(nx2)
  real(dp),    device :: d2c_d(2*kl+1,0:n)
  complex(dp), device :: f_d(0:n,nx,nz), u_d(0:n,nx,nz)
  complex(dp), device :: f2_d(0:n,nx2,nz), u2_d(0:n,nx2,nz)
  logical :: allok, cb
  integer(ip) :: j, ix, iz, ks, ih, im, istat
  allok = .true.

  call glc_nodes(n, yA)
  yA(1) = 1.0_dp;  yA(ny) = -1.0_dp
  call fd_build_bands(ny, kl, yA, d1A, d2A)
  d2b_d = d2A
  do ix = 1, nx;  kxh(ix) = real(ix-1, dp);        end do
  do iz = 1, nz;  kzh(iz) = real(iz-1-nz/2, dp);   end do
  kx_d = kxh;  kz_d = kzh;  kx2_d = kxh(1:nx2)

  ! the run's lambdas, ev_stage's exact expression chain
  lams(1) = 0.0_dp
  do ks = 1, 3
     bet = 0.5_dp*MCSUB(ks)
     aa  = bet*DT/RE
     lams(1+ks) = 1.0_dp/aa
  end do

  do iz = 1, nz; do ix = 1, nx; do j = 0, n
     fh(j,ix,iz) = cmplx(hv(j,ix+7*iz), -0.6_dp*hv(j,ix+11*iz), dp)
  end do; end do; end do

  ! ===== reference solves BEFORE any cache exists (rebuild path) ==============
  call fd_helm_cache_clear()
  f_d = fh
  call fd_helm_solve_gpu(n, kl, nx, nz, lams(2), kx_d, kz_d, d2b_d, f_d, u_d)
  istat = cudaDeviceSynchronize();  ur = u_d

  ! ===== (1) cached vs rebuild at the production lambda: BIT =================
  call fd_helm_cache_build(n, kl, nx, nz, 4_ip, lams, kx_d, kz_d, d2b_d, &
                           4.0_dp, cb, cgb)
  call report('cache builds under the cap (footprint reported > 0)', &
              cb .and. cgb > 0.0_dp)
  f_d = fh
  call fd_helm_solve_gpu(n, kl, nx, nz, lams(2), kx_d, kz_d, d2b_d, f_d, u_d)
  istat = cudaDeviceSynchronize();  uc = u_d
  call report('cached fast path BIT-IDENTICAL to rebuild (lam = 1/a_1)', &
              all(uc == ur))

  ! ===== (2) the lam = 0 leg (vif first solve): BIT ===========================
  call fd_helm_cache_clear()
  f_d = fh
  call fd_helm_solve_gpu(n, kl, nx, nz, lams(1), kx_d, kz_d, d2b_d, f_d, u_d)
  istat = cudaDeviceSynchronize();  ur = u_d
  call fd_helm_cache_build(n, kl, nx, nz, 4_ip, lams, kx_d, kz_d, d2b_d, &
                           4.0_dp, cb, cgb)
  f_d = fh
  call fd_helm_solve_gpu(n, kl, nx, nz, lams(1), kx_d, kz_d, d2b_d, f_d, u_d)
  istat = cudaDeviceSynchronize();  uc = u_d
  call report('cached fast path BIT-IDENTICAL to rebuild (lam = 0)', &
              all(uc == ur))

  ! ===== (3) bookkeeping: 2 hits so far; an unregistered lam = 1 miss =========
  f_d = fh
  call fd_helm_solve_gpu(n, kl, nx, nz, 777.0_dp, kx_d, kz_d, d2b_d, f_d, u_d)
  istat = cudaDeviceSynchronize();  uc = u_d
  call fd_helm_cache_clear()
  f_d = fh
  call fd_helm_solve_gpu(n, kl, nx, nz, 777.0_dp, kx_d, kz_d, d2b_d, f_d, u_d)
  istat = cudaDeviceSynchronize();  ur = u_d
  call fd_helm_cache_stats(ih, im)
  call report('unregistered lam falls back BIT-IDENTICAL + counters (2h/1m)', &
              all(uc == ur) .and. ih == 2 .and. im == 1)

  ! ===== (4) cap refusal: built=.false., solve still correct ==================
  call fd_helm_cache_build(n, kl, nx, nz, 4_ip, lams, kx_d, kz_d, d2b_d, &
                           0.0_dp, cb, cgb)
  f_d = fh
  call fd_helm_solve_gpu(n, kl, nx, nz, lams(2), kx_d, kz_d, d2b_d, f_d, u_d)
  istat = cudaDeviceSynchronize();  uc = u_d
  call fd_helm_cache_clear()
  f_d = fh
  call fd_helm_solve_gpu(n, kl, nx, nz, lams(2), kx_d, kz_d, d2b_d, f_d, u_d)
  istat = cudaDeviceSynchronize();  ur = u_d
  call report('cap_gb = 0 refuses (built=F, gb>0) and solves stay exact', &
              (.not. cb) .and. cgb > 0.0_dp .and. all(uc == ur))

  ! ===== (5) geometry guard: cache for (nx,nz) skipped at (nx2,nz) ============
  do iz = 1, nz; do ix = 1, nx2; do j = 0, n
     fh2(j,ix,iz) = cmplx(hv(j,ix+3*iz), 0.4_dp*hv(j,ix+5*iz), dp)
  end do; end do; end do
  call fd_helm_cache_build(n, kl, nx, nz, 4_ip, lams, kx_d, kz_d, d2b_d, &
                           4.0_dp, cb, cgb)
  f2_d = fh2
  call fd_helm_solve_gpu(n, kl, nx2, nz, lams(2), kx2_d, kz_d, d2b_d, f2_d, u2_d)
  istat = cudaDeviceSynchronize();  u2a = u2_d
  call fd_helm_cache_clear()
  f2_d = fh2
  call fd_helm_solve_gpu(n, kl, nx2, nz, lams(2), kx2_d, kz_d, d2b_d, f2_d, u2_d)
  istat = cudaDeviceSynchronize();  u2b = u2_d
  call report('geometry-mismatched cache is skipped (BIT vs no-cache)', &
              all(u2a == u2b))

  ! ===== (6) operator guard: same shape, DIFFERENT d2b must not hit ===========
  ! grid B = a tanh remap of the GLC nodes (endpoints preserved, interior
  ! moved) -> genuinely different bands; non-vacuity asserted below.
  do j = 1, ny
     yB(j) = tanh(1.5_dp*yA(j))/tanh(1.5_dp)
  end do
  yB(1) = 1.0_dp;  yB(ny) = -1.0_dp
  call fd_build_bands(ny, kl, yB, dB1, dB2)
  call report('leg-6 operands differ (non-vacuous)', any(dB2 /= d2A))
  d2c_d = dB2
  call fd_helm_cache_build(n, kl, nx, nz, 4_ip, lams, kx_d, kz_d, d2b_d, &
                           4.0_dp, cb, cgb)                 ! cache from operator A
  f_d = fh
  call fd_helm_solve_gpu(n, kl, nx, nz, lams(2), kx_d, kz_d, d2c_d, f_d, u_d)
  istat = cudaDeviceSynchronize();  uc = u_d               ! called with operator B
  call fd_helm_cache_clear()
  f_d = fh
  call fd_helm_solve_gpu(n, kl, nx, nz, lams(2), kx_d, kz_d, d2c_d, f_d, u_d)
  istat = cudaDeviceSynchronize();  ur = u_d               ! no-cache truth for B
  call report('same-shape different-operator skips the cache (BIT vs no-cache)', &
              all(uc == ur))

  if (allok) then; print '(a)', 'ALL TESTS PASSED'
  else;            print '(a)', 'SOME TESTS FAILED'; end if
  if (.not. allok) error stop 1

contains

  real(dp) function hv(a1, b1) result(vv)
    integer(ip), intent(in) :: a1, b1
    integer(ip) :: h
    h = modulo(a1*1009_ip + b1*2027_ip + a1*b1*17_ip, 100003_ip)
    vv = real(h, dp)/100003.0_dp - 0.5_dp
  end function hv

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. okk) allok = .false.
  end subroutine report

end program test_fd_helm_cache
