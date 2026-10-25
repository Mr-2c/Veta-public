!===============================================================================
! test_stepper_full_xcheck_r2c.F90  --  [R2C] dynamical-correctness
! check for the R2C step_full (full-system stepper: eta,v fluctuations + u0,w0 mean
! modes). The R2C analogue of test_stepper_full_xcheck.
!
! Evolves the SAME initial state two INDEPENDENT ways and requires agreement:
!   (A) GPU: step_full -- GPU fluctuation substage (nkx-threaded banded/vif) + host
!       (dp) mean solve with the constant-mass-flux constraint, coupled via hx0/hz0.
!   (B) CPU reference: identical per-stage structure, fluctuation solves on the host
!       (dense helmholtz_solve for eta, CPU vif_solve for v -- INDEPENDENT of the GPU
!       banded/vif path), SAME mean_mod solve; N and hx0/hz0 from the validated R2C
!       nl_evaluate at each path's own state.
!
! WHY THIS IS THE RIGHT TEST FOR R2C (the nkx-vs-nx slice trick can't be used here
! because step_full calls nl_evaluate, whose physical-space product couples all
! modes): the fluctuation solves differ in implementation (dense vs banded/vif,
! agree ~1e-13/1e-15, validated 4a/4b/4c-i), the forcing is the shared validated R2C
! nl_evaluate (3a-iii), and the mean solve is mean_mod (validated 6(d)); so a
! discrepancy localizes to the COUPLING GLUE step_full adds -- hx0/hz0 extraction
! (the kx=0 column = column 1, kept under R2C), u0/w0 feedback, mean history
! rotation, and the (0,0) zeroing. CFL-independent (common-mode cancels).
!
! Spectral x-extent is nkx=nx/2+1 (R2C half-axis); the per-mode COLUMN solves are
! representation-independent (a (0:n) column solve is identical C2C or R2C). The
! x-Nyquist is column nkx (last stored); there is NO negative-x half to zero.
!
! Also checks invariants on the GPU result: bulk velocity held at Q_target each
! step; (0,0) Fourier column of eta,v identically zero.
!
! n=16,nx=8,nz=6 (=> nkx=5),Re=10,A=0.05,dt=5e-3,Qtar=1. NaN-hardened.
! Build & run ON A GPU NODE (full stack, like the C2C xcheck):
!   nvfortran -cuda -cudalib=cufft -c types.F90 transforms_mod.F90 cheb_fft_mod.F90 \
!     fft3d_mod.F90 spectral3d_mod.F90 cheb_shen.F90 cheb_bih.F90 helm_band_mod.F90 \
!     vsolve_if_mod.F90 mean_mod.F90 cheb_ops_mod.F90 velocity_mod.F90 forcing_mod.F90 \
!     spec_helpers_mod.F90 helm_band_gpu_mod.F90 vsolve_if_gpu_mod.F90 nonlinear_mod.F90 \
!     stepper_lin_mod.F90 solver_ops_mod.F90 stepper_mod.F90
!   # link ONLY the module objects (NOT *.o -- that would sweep in other test_*.o,
!   # each of which has its own program/MAIN_, giving "multiple definition of MAIN_"):
!   nvfortran -cuda -cudalib=cufft test_stepper_full_xcheck_r2c.F90 \
!     types.o transforms_mod.o cheb_fft_mod.o fft3d_mod.o spectral3d_mod.o cheb_shen.o \
!     cheb_bih.o helm_band_mod.o vsolve_if_mod.o mean_mod.o cheb_ops_mod.o velocity_mod.o \
!     forcing_mod.o spec_helpers_mod.o helm_band_gpu_mod.o vsolve_if_gpu_mod.o \
!     nonlinear_mod.o stepper_lin_mod.o solver_ops_mod.o stepper_mod.o -o test_stepper_full_xcheck_r2c
!   ./test_stepper_full_xcheck_r2c
!===============================================================================
program test_stepper_full_xcheck_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,           only: dp, ip
  use cheb_shen,       only: cheb_d2, helmholtz_solve, shen_to_cheb
  use cheb_bih,        only: shen_bih_to_cheb
  use vsolve_if_mod,   only: vif_t, vif_setup, vif_solve
  use helm_band_mod,   only: helm_band_setup
  use mean_mod,        only: mean_flux, bulk_velocity, mean_flux_setup, mean_stage, mean_stage_cfr
  use nonlinear_mod,   only: nl_workspace, nl_setup, nl_destroy, nl_evaluate
  use stepper_lin_mod, only: stepper_ev_setup
  use stepper_mod,     only: step_full
  use solver_ops_mod,  only: band_ops_t, vsolve_ops_t, mean_ops_t, &
                             band_ops_alloc, band_ops_free, vsolve_ops_alloc, &
                             vsolve_ops_free, mean_ops_alloc, mean_ops_free
  use cudafor
  implicit none

  real(dp), parameter :: GAMMA(3) = [ 8.0_dp/15.0_dp,  5.0_dp/12.0_dp,  3.0_dp/4.0_dp ]
  real(dp), parameter :: ZETA(3)  = [ 0.0_dp,        -17.0_dp/60.0_dp, -5.0_dp/12.0_dp ]
  real(dp), parameter :: CSUB(3)  = [ 8.0_dp/15.0_dp,  2.0_dp/15.0_dp,  1.0_dp/3.0_dp ]

  integer(ip), parameter :: n = 16, nx = 8, nz = 6, Ns = 5
  integer(ip), parameter :: nkx = nx/2 + 1                ! R2C spectral x-extent
  real(dp),    parameter :: Re = 10.0_dp, amp = 0.05_dp, dt = 5.0e-3_dp, Qtar = 1.0_dp
  real(dp),    parameter :: tol = 1.0e-7_dp

  real(dp) :: kxh(nkx), kzh(nz)
  type(band_ops_t) :: bo          ! y-operator band structure (host+device)
  type(mean_ops_t) :: mo          ! CFR mean precompute
  complex(dp) :: e0(0:n,nkx,nz), v0(0:n,nkx,nz)
  complex(dp) :: eg(0:n,nkx,nz), vg(0:n,nkx,nz), ec(0:n,nkx,nz), vc(0:n,nkx,nz)
  complex(dp) :: u0g(0:n), w0g(0:n), u0c(0:n), w0c(0:n)
  real(dp) :: de, dv, du, dw, ne, nv_, nu_, nw, fluxdev, meancol
  integer(ip) :: ix, iz, kk
  logical :: bad, ok

  ! kx = non-negative half [0, nx/2]; kz = full (cuFFT order)
  do ix = 1, nkx; kxh(ix) = real(ix-1, dp); end do
  do iz = 1, nz; kk = iz-1; if (kk > nz/2) kk = kk - nz; kzh(iz) = real(kk, dp); end do
  call band_ops_alloc(bo, n);  call mean_ops_alloc(mo, n)
  call helm_band_setup(n, bo%sdh, bo%sth, bo%bdh, bo%boff)
  call mean_flux_setup(n, dt, Re, bo%sdh, bo%sth, bo%bdh, bo%boff, mo%Uhm, mo%Ih)
  call build_init(e0, v0)

  call gpu_run(e0, v0, eg, vg, u0g, w0g, fluxdev, meancol)
  call cpu_run(e0, v0, ec, vc, u0c, w0c)

  bad = .false.
  de=0;dv=0;du=0;dw=0; ne=0;nv_=0;nu_=0;nw=0
  do iz = 1, nz; do ix = 1, nkx; do kk = 0, n
     de = max(de, abs(eg(kk,ix,iz)-ec(kk,ix,iz)))
     dv = max(dv, abs(vg(kk,ix,iz)-vc(kk,ix,iz)))
     ne = max(ne, abs(eg(kk,ix,iz))); nv_ = max(nv_, abs(vg(kk,ix,iz)))
  end do; end do; end do
  do kk = 0, n
     du = max(du, abs(u0g(kk)-u0c(kk)));  dw = max(dw, abs(w0g(kk)-w0c(kk)))
     nu_ = max(nu_, abs(u0g(kk)));        nw = max(nw, abs(w0g(kk)))
  end do
  if (.not. (de==de) .or. .not. (dv==dv) .or. .not. (du==du) .or. .not. (dw==dw)) bad = .true.

  write(*,'(a)') '=== R2C full-system stepper GPU-vs-CPU cross-check (step_full, eta+v+mean) ==='
  write(*,'(a,i0,a,i0,a,i0,a,i0,a,f6.1,a,es8.1,a,es8.1,a,i0,a)') &
       '  n=',n,' nx=',nx,' (nkx=',nkx,') nz=',nz,'  Re=',Re,'  A=',amp,'  dt=',dt,'  (',Ns,' steps)'
  write(*,'(a,es10.2,a,es10.2,a,es10.2,a,es10.2)') &
       '  GPU magnitude:  |eta|=',ne,' |v|=',nv_,' |u0|=',nu_,' |w0|=',nw
  write(*,'(a,es10.2,a,es10.2,a,es10.2,a,es10.2)') &
       '  abs diff GPU-CPU: eta=',de,' v=',dv,' u0=',du,' w0=',dw
  write(*,'(a,es10.2)') '  rel diff (eta) = ', de/max(ne,tiny(1.0_dp))
  write(*,'(a,es10.2)') '  rel diff (v)   = ', dv/max(nv_,tiny(1.0_dp))
  write(*,'(a,es10.2)') '  rel diff (u0)  = ', du/max(nu_,tiny(1.0_dp))

  ok = (.not. bad)
  ok = ok .and. de/max(ne,tiny(1.0_dp)) < tol
  ok = ok .and. dv/max(nv_,tiny(1.0_dp)) < tol
  ok = ok .and. du/max(nu_,tiny(1.0_dp)) < tol
  ok = ok .and. dw < 1.0e-7_dp
  ! nonzero guard: the fields must be genuinely evolving (not all-zero false-pass)
  ok = ok .and. (ne > 1.0e-9_dp) .and. (nv_ > 1.0e-9_dp) .and. (nu_ > 1.0e-3_dp)
  write(*,'(/,a,es10.2,a)') '  bulk-velocity deviation max|U_bulk - Q_target| over steps = ', fluxdev, '   (tol 1e-10)'
  write(*,'(a,es10.2,a)')   '  max|(0,0) column of eta,v|                        = ', meancol, '   (tol 1e-12)'
  ok = ok .and. (fluxdev < 1.0e-10_dp) .and. (meancol < 1.0e-12_dp)

  call band_ops_free(bo);  call mean_ops_free(mo)

  if (ok) then
     write(*,'(/,a)') '  [PASS] R2C step_full matches the independent reference; flux held; (0,0) clean'
     write(*,'(a)')   '  => R2C mean coupling (hx0/hz0 -> u0/w0, history, constraint) is correct'
     write(*,'(/,a)') 'ALL TESTS PASSED'
  else
     write(*,'(/,a)') '  [FAIL] R2C step_full disagrees with the reference or violates an invariant'
     write(*,'(/,a)') 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  !-- GPU path: R2C step_full for Ns steps; track flux + (0,0) column -----------
  subroutine gpu_run(e0, v0, eout, vout, u0out, w0out, fluxdev, meancol)
    complex(dp), intent(in)  :: e0(0:n,nkx,nz), v0(0:n,nkx,nz)
    complex(dp), intent(out) :: eout(0:n,nkx,nz), vout(0:n,nkx,nz), u0out(0:n), w0out(0:n)
    real(dp),    intent(out) :: fluxdev, meancol
    type(nl_workspace) :: ws
    real(dp),    device, allocatable :: kx(:), kz(:)
    type(vsolve_ops_t) :: vo                ! local v-solve apparatus
    complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:), Nv_prev(:,:,:), Neta_prev(:,:,:)
    complex(dp) :: u0h(0:n), ecol(0:n), vcol(0:n)
    integer(ip) :: it, istat
    call nl_setup(ws, n+1, nx, nz, kxh, kzh)            ! R2C nl_setup: physical nx, kxh nkx-long
    allocate(kx(nkx), kz(nz))
    call vsolve_ops_alloc(vo, n, nkx, nz)
    allocate(eta(0:n,nkx,nz), v(0:n,nkx,nz), u0(0:n), w0(0:n), Nv_prev(0:n,nkx,nz), Neta_prev(0:n,nkx,nz))
    kx = kxh; kz = kzh
    bo%sd = bo%sdh; bo%st = bo%sth; bo%bd = bo%bdh
    call stepper_ev_setup(n, nkx, nz, dt, Re, kxh, kzh, vo%vm, vo%vp, vo%Cinv)
    u0 = (0.0_dp,0.0_dp); w0 = (0.0_dp,0.0_dp)
    Nv_prev = (0.0_dp,0.0_dp); Neta_prev = (0.0_dp,0.0_dp)
    eta = e0; v = v0
    fluxdev = 0.0_dp; meancol = 0.0_dp
    do it = 1, Ns
       call step_full(ws, n, nkx, nz, dt, Re, kx, kz, bo, vo, mo, &
                      Qtar, u0, w0, eta, v, Nv_prev, Neta_prev)
       istat = cudaDeviceSynchronize()
       u0h = u0
       fluxdev = max(fluxdev, abs(bulk_velocity(n, real(u0h)) - Qtar))
       ecol = eta(:,1,1); vcol = v(:,1,1)
       meancol = max(meancol, maxval(abs(ecol)), maxval(abs(vcol)))
    end do
    eout = eta; vout = v; u0out = u0; w0out = w0
    call vsolve_ops_free(vo)
    deallocate(kx, kz, eta, v, u0, w0, Nv_prev, Neta_prev)
    call nl_destroy(ws)
  end subroutine gpu_run

  !-- CPU reference: same per-stage structure, host fluctuation + mean solves ----
  subroutine cpu_run(e0, v0, eout, vout, u0out, w0out)
    complex(dp), intent(in)  :: e0(0:n,nkx,nz), v0(0:n,nkx,nz)
    complex(dp), intent(out) :: eout(0:n,nkx,nz), vout(0:n,nkx,nz), u0out(0:n), w0out(0:n)
    type(nl_workspace) :: ws
    complex(dp), device, allocatable :: eta_d(:,:,:), v_d(:,:,:), Nv_d(:,:,:), Neta_d(:,:,:)
    complex(dp), device, allocatable :: u0_d(:), w0_d(:), hx0_d(:), hz0_d(:)
    complex(dp) :: eta_h(0:n,nkx,nz), v_h(0:n,nkx,nz)
    complex(dp) :: Nv(0:n,nkx,nz), Neta(0:n,nkx,nz), Nv_prev(0:n,nkx,nz), Neta_prev(0:n,nkx,nz)
    complex(dp) :: hx0c(0:n), hz0c(0:n)
    real(dp)    :: u0h(0:n), w0h(0:n), hx0h(0:n), hz0h(0:n), Nu0_prev(0:n), Nw0_prev(0:n)
    real(dp)    :: a, lam, g, z, k2, beta_u
    integer(ip) :: it, ks, ix, iz, istat

    call nl_setup(ws, n+1, nx, nz, kxh, kzh)
    allocate(eta_d(0:n,nkx,nz), v_d(0:n,nkx,nz), Nv_d(0:n,nkx,nz), Neta_d(0:n,nkx,nz))
    allocate(u0_d(0:n), w0_d(0:n), hx0_d(0:n), hz0_d(0:n))

    eta_h = e0; v_h = v0
    u0h = 0.0_dp; w0h = 0.0_dp
    Nv_prev = (0.0_dp,0.0_dp); Neta_prev = (0.0_dp,0.0_dp)
    Nu0_prev = 0.0_dp; Nw0_prev = 0.0_dp

    do it = 1, Ns
       do ks = 1, 3
          a = 0.5_dp*CSUB(ks)*dt/Re; lam = 1.0_dp/a
          g = GAMMA(ks)*dt;          z   = ZETA(ks)*dt
          eta_d = eta_h; v_d = v_h
          u0_d = cmplx(u0h, 0.0_dp, dp); w0_d = cmplx(w0h, 0.0_dp, dp)
          call nl_evaluate(ws, v_d, eta_d, u0_d, w0_d, Nv_d, Neta_d, hx0_d, hz0_d)
          istat = cudaDeviceSynchronize()
          Nv = Nv_d; Neta = Neta_d; hx0c = hx0_d; hz0c = hz0_d
          hx0h = real(hx0c); hz0h = real(hz0c)
          do iz = 1, nz; do ix = 1, nkx           ! per-mode column solves over the nkx half
             k2 = kxh(ix)*kxh(ix) + kzh(iz)*kzh(iz)
             call solve_eta_mode(k2, a, lam, g, z, eta_h(:,ix,iz), Neta(:,ix,iz), Neta_prev(:,ix,iz))
             call solve_v_mode  (k2, a,      g, z,   v_h(:,ix,iz),   Nv(:,ix,iz),   Nv_prev(:,ix,iz))
          end do; end do
          call mean_stage_cfr(n, ks, dt, Re, Qtar, bo%sdh, bo%sth, bo%bdh, bo%boff, &
                              mo%Uhm(:,ks), mo%Ih(ks), u0h, hx0h, Nu0_prev, beta_u)
          call mean_stage    (n, ks, dt, Re, 0.0_dp, bo%sdh, bo%sth, bo%bdh, bo%boff, &
                              w0h, hz0h, Nw0_prev)
          Nv_prev = Nv; Neta_prev = Neta; Nu0_prev = hx0h; Nw0_prev = hz0h
          eta_h(:,1,1) = (0.0_dp,0.0_dp); v_h(:,1,1) = (0.0_dp,0.0_dp)
       end do
    end do

    eout = eta_h; vout = v_h
    u0out = cmplx(u0h, 0.0_dp, dp); w0out = cmplx(w0h, 0.0_dp, dp)
    deallocate(eta_d, v_d, Nv_d, Neta_d, u0_d, w0_d, hx0_d, hz0_d)
    call nl_destroy(ws)
  end subroutine cpu_run

    ! per-mode column solves -- representation-INDEPENDENT (a (0:n) column is the
    ! same C2C or R2C). UNCHANGED from the C2C test (the independence source).
    subroutine solve_eta_mode(k2, a, lam, g, z, col, Ncur, Nprev)
      real(dp),    intent(in)    :: k2, a, lam, g, z
      complex(dp), intent(inout) :: col(0:n)
      complex(dp), intent(in)    :: Ncur(0:n), Nprev(0:n)
      real(dp) :: cr(0:n), ci(0:n), d2r(0:n), d2i(0:n)
      real(dp) :: rr(0:n), ri(0:n), uhr(0:n-2), uhi(0:n-2), outr(0:n), outi(0:n)
      cr = real(col); ci = aimag(col)
      call cheb_d2(n, cr, d2r); call cheb_d2(n, ci, d2i)
      rr = lam*( cr + a*(d2r - k2*cr) + g*real(Ncur)  + z*real(Nprev) )
      ri = lam*( ci + a*(d2i - k2*ci) + g*aimag(Ncur) + z*aimag(Nprev) )
      call helmholtz_solve(n, k2+lam, rr, uhr); call shen_to_cheb(n, uhr, outr)
      call helmholtz_solve(n, k2+lam, ri, uhi); call shen_to_cheb(n, uhi, outi)
      col = cmplx(outr, outi, dp)
    end subroutine solve_eta_mode

    subroutine solve_v_mode(k2, a, g, z, col, Ncur, Nprev)
      real(dp),    intent(in)    :: k2, a, g, z
      complex(dp), intent(inout) :: col(0:n)
      complex(dp), intent(in)    :: Ncur(0:n), Nprev(0:n)
      type(vif_t) :: d
      real(dp) :: cr(0:n), ci(0:n), Lr(0:n), Li(0:n), L2r(0:n), L2i(0:n), tr(0:n), ti(0:n)
      real(dp) :: rr(0:n), ri(0:n), outr(0:n), outi(0:n)
      cr = real(col); ci = aimag(col)
      call cheb_d2(n, cr, tr); Lr = tr - k2*cr
      call cheb_d2(n, ci, ti); Li = ti - k2*ci
      call cheb_d2(n, Lr, tr); L2r = tr - k2*Lr
      call cheb_d2(n, Li, ti); L2i = ti - k2*Li
      rr = Lr + a*L2r + g*real(Ncur)  + z*real(Nprev)
      ri = Li + a*L2i + g*aimag(Ncur) + z*aimag(Nprev)
      call vif_setup(n, k2, a, d)
      call vif_solve(n, d, rr, outr)
      call vif_solve(n, d, ri, outi)
      col = cmplx(outr, outi, dp)
    end subroutine solve_v_mode

  !-- R2C initial state: random BC-satisfying modes on the nkx half-spectrum -----
  subroutine build_init(e0, v0)
    complex(dp), intent(out) :: e0(0:n,nkx,nz), v0(0:n,nkx,nz)
    real(dp) :: ure(0:n-2), uim(0:n-2), are(0:n), aim(0:n)
    real(dp) :: vre(0:n-4), vim(0:n-4), bre(0:n), bim(0:n)
    real(dp) :: rr, ri, decay, kc
    integer(ip) :: ix, iz, k
    kc = real(n, dp)/6.0_dp
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
    do iz = 1, nz; do ix = 1, nkx
       do k = 0, n-2
          call random_number(rr); call random_number(ri)
          decay  = amp*exp(-(real(k,dp)/kc)**2)
          ure(k) = (rr-0.5_dp)*decay; uim(k) = (ri-0.5_dp)*decay
       end do
       call shen_to_cheb(n, ure, are); call shen_to_cheb(n, uim, aim)
       do k = 0, n; e0(k,ix,iz) = cmplx(are(k), aim(k), dp); end do
       do k = 0, n-4
          call random_number(rr); call random_number(ri)
          decay  = amp*exp(-(real(k,dp)/kc)**2)
          vre(k) = (rr-0.5_dp)*decay; vim(k) = (ri-0.5_dp)*decay
       end do
       call shen_bih_to_cheb(n, vre, bre); call shen_bih_to_cheb(n, vim, bim)
       do k = 0, n; v0(k,ix,iz) = cmplx(bre(k), bim(k), dp); end do
    end do; end do
    ! R2C Nyquist zeroing: (0,0) column = column 1; x-Nyquist = LAST column (nkx);
    ! kz-Nyquist row; there is NO negative-x half.
    e0(:,1,1) = (0.0_dp,0.0_dp);   v0(:,1,1) = (0.0_dp,0.0_dp)
    e0(:,nkx,:) = (0.0_dp,0.0_dp); v0(:,nkx,:) = (0.0_dp,0.0_dp)
    e0(:,:,nz/2+1) = (0.0_dp,0.0_dp); v0(:,:,nz/2+1) = (0.0_dp,0.0_dp)
  end subroutine build_init

end program test_stepper_full_xcheck_r2c
