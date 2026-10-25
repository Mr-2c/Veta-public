!===============================================================================
! test_fd_dispatch.F90 -- gate for the band_ops_t backend seam
! (solver_ops_mod y_apply_L / y_helm_solve / y_vif_solve):
!   (1) CHEB BIT-IDENTITY: through the helpers with ybasis=YB_CHEB, every
!       result must equal the VERBATIM leaf sequence the steppers ran before
!       this round, to the BIT (the helpers contain the moved statements --
!       any deviation is a transcription error, not noise).
!   (2) FD CORRECTNESS: with ybasis=YB_FD (populated by band_ops_fd_init from
!       a grid file round-tripped through the REAL ygrid_read), each helper
!       must match the HOST fdops/fd_vsolve oracles per mode (Re/Im), at the
!       posture of the earlier FD gates (helm/apply_L rel 1e-11/1e-12; vif
!       rel 1e-8 -- it includes the GPU-vs-host setup difference).
!   (3) the grid FILE path: the test WRITES a valid 17-digit ygrid file and
!       reads it back through ygrid_read -- the B1.1 contract, the fdops kit,
!       and the dispatch seam gate each other end to end.
! Grid GLC(48) DESCENDING, kl=4, nx=8 x nz=6 modes incl. kx=kz=0. Host-only
! oracles; device paths are the cluster-validated batched kernels behind the
! seam. Run from the repo root.
!===============================================================================
program test_fd_dispatch
  use kinds,             only: dp, ip
  use cudafor
  use cheb_shen,         only: glc_nodes
  use helm_band_mod,     only: helm_band_setup
  use cheb_ops_mod,      only: apply_L
  use spec_helpers_mod,  only: project_shen, shen_to_cheb_c
  use helm_band_gpu_mod, only: helm_band_solve_gpu
  use vsolve_if_gpu_mod, only: vif_solve_gpu_ws, vif_gpu_setup_dev
  use fdops_mod,         only: fd_build_bands, fd_wall_weights, fd_apply_band, &
                               fd_helm_dirichlet, fd_band_factor, fd_band_solve
  use fd_vsolve_mod,     only: fd_vif_setup, fd_vif_solve
  use fd_vsolve_gpu_mod, only: fd_vif_setup_gpu
  use ygrid_mod,         only: ygrid_read
  use solver_ops_mod,    only: band_ops_t, band_ops_alloc, band_ops_free, &
                               band_ops_fd_init, y_apply_L, y_helm_solve, &
                               y_vif_solve, y_vif_setup, YB_FD
  implicit none
  integer(ip), parameter :: n = 48, ny = n + 1, kl = 4, nx = 8, nz = 6
  real(dp), parameter :: LAM = 640.0_dp, ALPH = 1.5625e-3_dp   ! lam = 1/alpha
  character(*), parameter :: GFILE = 'build/ygrid_dispatch48.dat'
  real(dp)    :: yg(0:n), yh(ny), wh(ny), kxh(nx), kzh(nz), k2
  real(dp)    :: d1bh(2*kl+1,ny), d2bh(2*kl+1,ny), wloh(ny), whih(ny)
  real(dp)    :: ab(2*kl+1,ny), br(ny), bi(ny), vmh(ny), vph(ny), Cih(2,2)
  real(dp)    :: vre(ny), vim(ny)
  real(dp)    :: e, w
  complex(dp) :: fh(0:n,nx,nz), rh(0:n,nx,nz), oh(0:n,nx,nz), o2h(0:n,nx,nz)
  complex(dp) :: vhost
  real(dp),    device :: kx_d(nx), kz_d(nz)
  complex(dp), device :: f_d(0:n,nx,nz), r_d(0:n,nx,nz), o_d(0:n,nx,nz)
  complex(dp), device :: o2_d(0:n,nx,nz), scr_d(0:n,nx,nz)
  complex(dp), device :: ftil_d(0:n-2,nx,nz), sh_d(0:n-2,nx,nz)
  complex(dp), device :: gsh_d(0:n-2,nx,nz), vsh_d(0:n-2,nx,nz)
  complex(dp), device :: gc_d(0:n,nx,nz), vpc_d(0:n,nx,nz)
  complex(dp), device :: dlo_d(nx,nz), dhi_d(nx,nz)
  real(dp),    device :: vmc_d(0:n,nx,nz), vpc2_d(0:n,nx,nz), Cic_d(2,2,nx,nz)
  real(dp),    device :: vmf_d(0:n,nx,nz), vpf_d(0:n,nx,nz), Cif_d(2,2,nx,nz)
  real(dp),    device :: vm2_d(0:n,nx,nz), vp2_d(0:n,nx,nz), Ci2_d(2,2,nx,nz)
  real(dp)    :: pm(0:n,nx,nz), pp(0:n,nx,nz), pc(2,2,nx,nz)
  real(dp)    :: qm(0:n,nx,nz), qp(0:n,nx,nz), qc2(2,2,nx,nz)
  type(band_ops_t) :: boc, bofd
  logical :: ok, have_w, allok
  character(len=256) :: msg
  integer(ip) :: i, j, ix, iz, kk, u, istat
  allok = .true.

  ! ---- grid file: write (17-digit grammar) then read back via ygrid_read -----
  call glc_nodes(n, yg)
  open(newunit=u, file=GFILE, status='replace', action='write')
  write(u,'(a)') '# GLC(48) test grid for the B1.3 dispatch gate'
  write(u,'(a,i0)') 'ny = ', ny
  write(u,'(a)') 'columns = y'
  do j = 0, n
     if (j == 0) then
        write(u,'(es24.17)')  1.0_dp
     else if (j == n) then
        write(u,'(es24.17)') -1.0_dp
     else
        write(u,'(es24.17)') yg(j)
     end if
  end do
  close(u)
  wh = 0.0_dp                 ! defined even when the file has no w column
  call ygrid_read(GFILE, ny, yh, wh, have_w, ok, msg)
  if (.not. ok) print '(a)', '   [ygrid] '//trim(msg)
  call report('dispatch grid file writes + ygrid_read round-trips', ok)

  ! ---- band_ops: Cheb side (as the mains build it) + FD side ------------------
  call band_ops_alloc(boc, n)
  call helm_band_setup(n, boc%sdh, boc%sth, boc%bdh, boc%boff)
  boc%sd = boc%sdh;  boc%st = boc%sth;  boc%bd = boc%bdh
  call band_ops_fd_init(bofd, n, kl, yh, wh, have_w)
  call report('band_ops_fd_init: tag flips to YB_FD, kl recorded', &
              bofd%ybasis == YB_FD .and. bofd%kl == kl)

  ! host FD operator kit for the oracles (same builders as band_ops_fd_init)
  call fd_build_bands(ny, kl, yh, d1bh, d2bh)
  call fd_wall_weights(ny, kl, yh, wloh, whih)

  do i = 1, nx
     kxh(i) = 0.7_dp*real(i-1, dp)
  end do
  do i = 1, nz
     kk = i - 1;  if (kk > nz/2) kk = kk - nz
     kzh(i) = 1.3_dp*real(kk, dp)
  end do
  kx_d = kxh;  kz_d = kzh
  do iz = 1, nz; do ix = 1, nx; do j = 0, n
     fh(j,ix,iz) = cmplx(hv(j,ix,iz,1), -0.5_dp*hv(j,ix,iz,2), dp)
  end do; end do; end do

  ! ===== (1) CHEB BIT-IDENTITY through the seam ================================
  f_d = fh
  call y_apply_L(boc, n, nx, nz, kx_d, kz_d, f_d, o_d, scr_d)
  f_d = fh
  call apply_L(n+1, nx, nz, kx_d, kz_d, f_d, o2_d, scr_d)
  istat = cudaDeviceSynchronize();  oh = o_d;  o2h = o2_d
  call report('cheb: y_apply_L BIT-IDENTICAL to apply_L', bitsame(oh, o2h))

  r_d = fh
  call y_helm_solve(boc, n, nx, nz, LAM, kx_d, kz_d, r_d, o_d, ftil_d, sh_d)
  r_d = fh                                        ! verbatim pre-B1.3 sequence:
  call project_shen(n, nx, nz, r_d, ftil_d)
  call tscale(n-2, nx, nz, ftil_d, LAM)
  call helm_band_solve_gpu(n, nx, nz, LAM, kx_d, kz_d, boc%sd, boc%st, boc%bd, &
                           boc%boff, ftil_d, sh_d)
  call shen_to_cheb_c(n, nx, nz, sh_d, o2_d)
  istat = cudaDeviceSynchronize();  oh = o_d;  o2h = o2_d
  call report('cheb: y_helm_solve BIT-IDENTICAL to the leaf sequence', bitsame(oh, o2h))

  call vif_gpu_setup_dev(n, nx, nz, kx_d, kz_d, ALPH, boc%sd, boc%st, boc%bd, &
                         boc%boff, vmc_d, vpc2_d, Cic_d)
  r_d = fh
  call y_vif_solve(boc, n, nx, nz, ALPH, kx_d, kz_d, vmc_d, vpc2_d, Cic_d, &
                   r_d, o_d, ftil_d, gsh_d, gc_d, vsh_d, vpc_d, dlo_d, dhi_d, scr_d)
  r_d = fh
  call vif_solve_gpu_ws(n, nx, nz, ALPH, kx_d, kz_d, boc%sd, boc%st, boc%bd, &
                        boc%boff, vmc_d, vpc2_d, Cic_d, r_d, o2_d, &
                        ftil_d, gsh_d, gc_d, vsh_d, vpc_d, dlo_d, dhi_d)
  istat = cudaDeviceSynchronize();  oh = o_d;  o2h = o2_d
  call report('cheb: y_vif_solve BIT-IDENTICAL to vif_solve_gpu_ws', bitsame(oh, o2h))

  ! ===== (2) FD dispatch vs HOST oracles =======================================
  ! y_apply_L: banded matvec - k^2 u
  f_d = fh
  call y_apply_L(bofd, n, nx, nz, kx_d, kz_d, f_d, o_d, scr_d)
  istat = cudaDeviceSynchronize();  oh = o_d
  e = 0.0_dp;  w = 0.0_dp
  do iz = 1, nz; do ix = 1, nx
     k2 = kxh(ix)**2 + kzh(iz)**2
     do j = 1, ny
        br(j) = real(fh(j-1,ix,iz));  bi(j) = aimag(fh(j-1,ix,iz))
     end do
     call fd_apply_band(ny, kl, d2bh, br, vmh)          ! vmh/vph as scratch here
     call fd_apply_band(ny, kl, d2bh, bi, vph)
     do j = 1, ny
        vhost = cmplx(vmh(j), vph(j), dp) - k2*fh(j-1,ix,iz)
        w = max(w, abs(vhost));  e = max(e, abs(oh(j-1,ix,iz) - vhost))
     end do
  end do; end do
  call report('fd: y_apply_L matches host banded oracle (rel 1e-12)', &
              e <= 1.0e-12_dp*max(1.0_dp, w))

  ! y_helm_solve: (-D2 + k2 + lam) out = lam*rhs, Dirichlet walls
  r_d = fh
  call y_helm_solve(bofd, n, nx, nz, LAM, kx_d, kz_d, r_d, o_d, ftil_d, sh_d)
  istat = cudaDeviceSynchronize();  oh = o_d
  e = 0.0_dp;  w = 0.0_dp
  do iz = 1, nz; do ix = 1, nx
     k2 = kxh(ix)**2 + kzh(iz)**2
     call fd_helm_dirichlet(ny, kl, d2bh, k2 + LAM, ab)
     call fd_band_factor(ny, kl, ab, ok)
     if (.not. ok) then
        call report('fd: y_helm_solve host factor refused (unexpected)', .false.)
        exit
     end if
     do j = 1, ny
        br(j) = LAM*real(fh(j-1,ix,iz));  bi(j) = LAM*aimag(fh(j-1,ix,iz))
     end do
     br(1) = 0.0_dp;  br(ny) = 0.0_dp;  bi(1) = 0.0_dp;  bi(ny) = 0.0_dp
     call fd_band_solve(ny, kl, ab, br)
     call fd_band_solve(ny, kl, ab, bi)
     do j = 1, ny
        vhost = cmplx(br(j), bi(j), dp)
        w = max(w, abs(vhost));  e = max(e, abs(oh(j-1,ix,iz) - vhost))
     end do
  end do; end do
  call report('fd: y_helm_solve matches host Dirichlet oracle (rel 1e-11)', &
              e <= 1.0e-11_dp*max(1.0_dp, w))

  ! y_vif_solve: GPU setup products + dispatched solve vs the full host chain
  call fd_vif_setup_gpu(n, kl, nx, nz, kx_d, kz_d, ALPH, bofd%yd, bofd%d2b, &
                        bofd%wlo, bofd%whi, vmf_d, vpf_d, Cif_d)
  r_d = fh
  call y_vif_solve(bofd, n, nx, nz, ALPH, kx_d, kz_d, vmf_d, vpf_d, Cif_d, &
                   r_d, o_d, ftil_d, gsh_d, gc_d, vsh_d, vpc_d, dlo_d, dhi_d, scr_d)
  istat = cudaDeviceSynchronize();  oh = o_d
  e = 0.0_dp;  w = 0.0_dp
  do iz = 1, nz; do ix = 1, nx
     k2 = kxh(ix)**2 + kzh(iz)**2
     call fd_vif_setup(ny, kl, yh, d2bh, wloh, whih, k2, ALPH, vmh, vph, Cih, ok)
     if (.not. ok) then
        call report('fd: y_vif_solve host setup refused (unexpected)', .false.)
        exit
     end if
     do j = 1, ny
        br(j) = real(fh(j-1,ix,iz));  bi(j) = aimag(fh(j-1,ix,iz))
     end do
     call fd_vif_solve(ny, kl, d2bh, wloh, whih, k2, ALPH, vmh, vph, Cih, br, vre, ok)
     if (.not. ok) e = huge(1.0_dp)
     call fd_vif_solve(ny, kl, d2bh, wloh, whih, k2, ALPH, vmh, vph, Cih, bi, vim, ok)
     if (.not. ok) e = huge(1.0_dp)
     do j = 1, ny
        vhost = cmplx(vre(j), vim(j), dp)
        w = max(w, abs(vhost));  e = max(e, abs(oh(j-1,ix,iz) - vhost))
     end do
  end do; end do
  call report('fd: y_vif_solve matches host vif oracle (rel 1e-8)', &
              e <= 1.0e-8_dp*max(1.0_dp, w))

  ! ===== (3) y_vif_setup dispatch: both branches vs the direct calls ===
  ! ws-vs-self-alloc setup is BIT-EXACT for both backends (gated in
  ! test_vif_setup_gpu / test_fd_vsolve_gpu), so bitwise equality is the gate.
  call y_vif_setup(boc, n, nx, nz, kx_d, kz_d, ALPH, vm2_d, vp2_d, Ci2_d, &
                   r_d, ftil_d, gsh_d, gc_d, o2_d, vpc_d, scr_d)
  istat = cudaDeviceSynchronize()
  pm = vm2_d;  pp = vp2_d;  pc = Ci2_d
  qm = vmc_d;  qp = vpc2_d;  qc2 = Cic_d
  call report('cheb: y_vif_setup BIT-IDENTICAL to vif_gpu_setup_dev', &
              all(pm == qm) .and. all(pp == qp) .and. all(pc == qc2))
  call y_vif_setup(bofd, n, nx, nz, kx_d, kz_d, ALPH, vm2_d, vp2_d, Ci2_d, &
                   r_d, ftil_d, gsh_d, gc_d, o2_d, vpc_d, scr_d)
  istat = cudaDeviceSynchronize()
  pm = vm2_d;  pp = vp2_d;  pc = Ci2_d
  qm = vmf_d;  qp = vpf_d;  qc2 = Cif_d
  call report('fd:   y_vif_setup BIT-IDENTICAL to fd_vif_setup_gpu', &
              all(pm == qm) .and. all(pp == qp) .and. all(pc == qc2))

  call band_ops_free(boc);  call band_ops_free(bofd)
  if (allok) then; print '(a)', 'ALL TESTS PASSED'
  else;            print '(a)', 'SOME TESTS FAILED'; end if
  if (.not. allok) error stop 1

contains

  real(dp) function hv(a, b, c, e0) result(v)
    integer(ip), intent(in) :: a, b, c, e0
    integer(ip) :: h
    h = modulo(a*1009_ip + b*2027_ip + c*3001_ip + e0*4013_ip + a*c*17_ip, 100003_ip)
    v = real(h, dp)/100003.0_dp - 0.5_dp
  end function hv

  !-- in-place scale (the steppers' old private cscale, for the verbatim row) --
  subroutine tscale(m, nxx, nzz, x, s)
    integer(ip), intent(in) :: m, nxx, nzz
    real(dp),    intent(in) :: s
    complex(dp), device     :: x(0:m, nxx, nzz)
    integer(ip) :: ix, iz, k
    !$cuf kernel do(3) <<<*,*>>>
    do iz = 1, nzz; do ix = 1, nxx; do k = 0, m
       x(k,ix,iz) = x(k,ix,iz)*s
    end do; end do; end do
  end subroutine tscale

  logical function bitsame(a, b) result(okk)
    complex(dp), intent(in) :: a(:,:,:), b(:,:,:)
    okk = all(a == a) .and. all(b == b) .and. all(a == b)
  end function bitsame

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. okk) allok = .false.
  end subroutine report

end program test_fd_dispatch
