!===============================================================================
! test_stepper_xcheck_r2c.F90  --  [R2C] dynamical-correctness check for the
! FLUCTUATION-ONLY nonlinear stepper step_ev (u0=w0=0, NO mean coupling).
! The R2C analogue of test_stepper_xcheck.
!
! INDEPENDENT VALUE vs test_stepper_full_xcheck_r2c: that test exercises step_full
! (with the mean handoff, which zeroes the (0,0) column every stage). THIS test
! isolates the fluctuation assembly with NO mean activity to hide behind -- a bug
! in ev_stage that the mean path happens to mask would show here. Both pin the
! gamma/zeta coefficients, the a_k=(c_k/2)dt/Re scaling, the +N sign, and the
! register rotation; this one does it mean-free.
!
! Evolve the SAME field two INDEPENDENT ways and require agreement:
!   (A) GPU: step_ev (banded Helmholtz eta, influence-matrix vif v, R2C nl_evaluate).
!   (B) CPU: from-scratch host SMR substage -- L via cheb_d2, eta via DENSE
!       helmholtz_solve, v via CPU vif_solve -- N from the same R2C nl_evaluate.
! Discrepancy localizes to the ASSEMBLY (CFL-independent: common-mode cancels).
! The per-mode COLUMN solves are representation-independent (carried unchanged).
!
! n=16,nx=8,nz=6 (=>nkx=5),Re=10,A=0.05,dt=5e-3. (0,0)&Nyquist zeroed. NaN-hardened.
! Build & run ON A GPU NODE (module objects only, NEVER *.o):
!   nvfortran -cuda -cudalib=cufft -c types.F90 transforms_mod.F90 cheb_fft_mod.F90 \
!     fft3d_mod.F90 spectral3d_mod.F90 cheb_shen.F90 cheb_bih.F90 helm_band_mod.F90 \
!     vsolve_if_mod.F90 mean_mod.F90 cheb_ops_mod.F90 velocity_mod.F90 forcing_mod.F90 \
!     spec_helpers_mod.F90 helm_band_gpu_mod.F90 vsolve_if_gpu_mod.F90 nonlinear_mod.F90 \
!     stepper_lin_mod.F90 solver_ops_mod.F90 stepper_mod.F90
!   nvfortran -cuda -cudalib=cufft test_stepper_xcheck_r2c.F90 \
!     types.o transforms_mod.o cheb_fft_mod.o fft3d_mod.o spectral3d_mod.o cheb_shen.o \
!     cheb_bih.o helm_band_mod.o vsolve_if_mod.o mean_mod.o cheb_ops_mod.o velocity_mod.o \
!     forcing_mod.o spec_helpers_mod.o helm_band_gpu_mod.o vsolve_if_gpu_mod.o \
!     nonlinear_mod.o stepper_lin_mod.o solver_ops_mod.o stepper_mod.o -o test_stepper_xcheck_r2c
!   ./test_stepper_xcheck_r2c
!===============================================================================
program test_stepper_xcheck_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,           only: dp, ip
  use cheb_shen,       only: cheb_d2, helmholtz_solve, shen_to_cheb
  use cheb_bih,        only: shen_bih_to_cheb
  use vsolve_if_mod,   only: vif_t, vif_setup, vif_solve
  use helm_band_mod,   only: helm_band_setup
  use nonlinear_mod,   only: nl_workspace, nl_setup, nl_destroy, nl_evaluate
  use stepper_lin_mod, only: stepper_ev_setup
  use stepper_mod,     only: step_ev
  use solver_ops_mod,  only: band_ops_t, vsolve_ops_t, band_ops_alloc, &
                             band_ops_free, vsolve_ops_alloc, vsolve_ops_free
  use cudafor
  implicit none

  real(dp), parameter :: GAMMA(3) = [ 8.0_dp/15.0_dp,  5.0_dp/12.0_dp,  3.0_dp/4.0_dp ]
  real(dp), parameter :: ZETA(3)  = [ 0.0_dp,        -17.0_dp/60.0_dp, -5.0_dp/12.0_dp ]
  real(dp), parameter :: CSUB(3)  = [ 8.0_dp/15.0_dp,  2.0_dp/15.0_dp,  1.0_dp/3.0_dp ]

  integer(ip), parameter :: n = 16, nx = 8, nz = 6, Ns = 5
  integer(ip), parameter :: nkx = nx/2 + 1
  real(dp),    parameter :: Re = 10.0_dp, amp = 0.05_dp, dt = 5.0e-3_dp
  real(dp),    parameter :: tol = 1.0e-7_dp

  real(dp) :: kxh(nkx), kzh(nz)
  type(band_ops_t) :: bo                 ! y-operator band structure (host+device)
  complex(dp) :: e0(0:n,nkx,nz), v0(0:n,nkx,nz)
  complex(dp) :: eg(0:n,nkx,nz), vg(0:n,nkx,nz)      ! GPU result
  complex(dp) :: ec(0:n,nkx,nz), vc(0:n,nkx,nz)      ! CPU reference result
  real(dp) :: de, dv, ne, nv_, re_, rv
  integer(ip) :: ix, iz, kk
  logical :: bad, ok

  do ix = 1, nkx; kxh(ix) = real(ix-1, dp); end do      ! non-negative half
  do iz = 1, nz; kk = iz-1; if (kk > nz/2) kk = kk - nz; kzh(iz) = real(kk, dp); end do
  call band_ops_alloc(bo, n)
  call helm_band_setup(n, bo%sdh, bo%sth, bo%bdh, bo%boff)
  call build_init(e0, v0)

  call gpu_run (e0, v0, eg, vg)
  call cpu_run (e0, v0, ec, vc)

  bad = .false.
  de = 0.0_dp; dv = 0.0_dp; ne = 0.0_dp; nv_ = 0.0_dp
  do iz = 1, nz; do ix = 1, nkx; do kk = 0, n
     de = max(de, abs(eg(kk,ix,iz) - ec(kk,ix,iz)))
     dv = max(dv, abs(vg(kk,ix,iz) - vc(kk,ix,iz)))
     ne = max(ne, abs(eg(kk,ix,iz)));  nv_ = max(nv_, abs(vg(kk,ix,iz)))
  end do; end do; end do
  if (.not. (de == de) .or. .not. (dv == dv)) bad = .true.
  re_ = de/max(ne, tiny(1.0_dp));  rv = dv/max(nv_, tiny(1.0_dp))

  write(*,'(a)') '=== R2C nonlinear stepper GPU-vs-CPU cross-check (mean-free step_ev) ==='
  write(*,'(a,i0,a,i0,a,i0,a,i0,a,f6.1,a,es8.1,a,es8.1,a,i0,a)') &
       '  n=',n,' nx=',nx,' (nkx=',nkx,') nz=',nz,'  Re=',Re,'  A=',amp,'  dt=',dt,'  (',Ns,' steps)'
  write(*,'(a,es12.4,a,es12.4)') '  GPU field magnitude:  max|eta|=', ne, '   max|v|=', nv_
  write(*,'(a,es12.4,a,es12.4)') '  abs difference GPU-CPU: max|d eta|=', de, '   max|d v|=', dv
  write(*,'(a,es12.4,a,es12.4)') '  rel difference:                  ', re_, '              ', rv
  write(*,'(a,es8.1)')           '  tolerance (relative): ', tol

  ok = (.not. bad) .and. (re_ < tol) .and. (rv < tol) &
       .and. (ne > 1.0e-6_dp) .and. (nv_ > 1.0e-6_dp)       ! nonzero guard
  call band_ops_free(bo)
  if (ok) then
     write(*,'(/,a)') '  [PASS] R2C step_ev matches the independent CPU reference (mean-free)'
     write(*,'(a)')   '  => nonlinear sign/scale/coefficients and history rotation correct on nkx'
     write(*,'(/,a)') 'ALL TESTS PASSED'
  else
     write(*,'(/,a)') '  [FAIL] R2C step_ev disagrees with the CPU reference'
     write(*,'(/,a)') 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine gpu_run(e0, v0, eout, vout)
    complex(dp), intent(in)  :: e0(0:n,nkx,nz), v0(0:n,nkx,nz)
    complex(dp), intent(out) :: eout(0:n,nkx,nz), vout(0:n,nkx,nz)
    type(nl_workspace) :: ws
    real(dp),    device, allocatable :: kx(:), kz(:)
    type(vsolve_ops_t) :: vo               ! local v-solve apparatus
    complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:), Nv_prev(:,:,:), Neta_prev(:,:,:)
    integer(ip) :: it, istat
    call nl_setup(ws, n+1, nx, nz, kxh, kzh)          ! R2C: physical nx, kxh nkx-long
    allocate(kx(nkx), kz(nz))
    call vsolve_ops_alloc(vo, n, nkx, nz)
    allocate(eta(0:n,nkx,nz), v(0:n,nkx,nz), u0(0:n), w0(0:n), Nv_prev(0:n,nkx,nz), Neta_prev(0:n,nkx,nz))
    kx = kxh; kz = kzh
    bo%sd = bo%sdh; bo%st = bo%sth; bo%bd = bo%bdh
    call stepper_ev_setup(n, nkx, nz, dt, Re, kxh, kzh, vo%vm, vo%vp, vo%Cinv)
    u0 = (0.0_dp,0.0_dp); w0 = (0.0_dp,0.0_dp)
    Nv_prev = (0.0_dp,0.0_dp); Neta_prev = (0.0_dp,0.0_dp)
    eta = e0; v = v0
    do it = 1, Ns
       call step_ev(ws, n, nkx, nz, dt, Re, kx, kz, bo, vo, &
                    u0, w0, eta, v, Nv_prev, Neta_prev)
    end do
    istat = cudaDeviceSynchronize()
    eout = eta; vout = v
    call vsolve_ops_free(vo)
    deallocate(kx, kz, eta, v, u0, w0, Nv_prev, Neta_prev)
    call nl_destroy(ws)
  end subroutine gpu_run

  subroutine cpu_run(e0, v0, eout, vout)
    complex(dp), intent(in)  :: e0(0:n,nkx,nz), v0(0:n,nkx,nz)
    complex(dp), intent(out) :: eout(0:n,nkx,nz), vout(0:n,nkx,nz)
    type(nl_workspace) :: ws
    complex(dp), device, allocatable :: eta_d(:,:,:), v_d(:,:,:), Nv_d(:,:,:), Neta_d(:,:,:)
    complex(dp), device, allocatable :: u0_d(:), w0_d(:), hx0_d(:), hz0_d(:)
    complex(dp) :: eta_h(0:n,nkx,nz), v_h(0:n,nkx,nz)
    complex(dp) :: Nv(0:n,nkx,nz), Neta(0:n,nkx,nz), Nv_prev(0:n,nkx,nz), Neta_prev(0:n,nkx,nz)
    real(dp)    :: a, lam, g, z, k2
    integer(ip) :: it, ks, ix, iz, istat

    call nl_setup(ws, n+1, nx, nz, kxh, kzh)
    allocate(eta_d(0:n,nkx,nz), v_d(0:n,nkx,nz), Nv_d(0:n,nkx,nz), Neta_d(0:n,nkx,nz))
    allocate(u0_d(0:n), w0_d(0:n), hx0_d(0:n), hz0_d(0:n))
    u0_d = (0.0_dp,0.0_dp); w0_d = (0.0_dp,0.0_dp)

    eta_h = e0; v_h = v0
    Nv_prev = (0.0_dp,0.0_dp); Neta_prev = (0.0_dp,0.0_dp)

    do it = 1, Ns
       do ks = 1, 3
          a = 0.5_dp*CSUB(ks)*dt/Re;  lam = 1.0_dp/a
          g = GAMMA(ks)*dt;           z   = ZETA(ks)*dt
          eta_d = eta_h;  v_d = v_h
          call nl_evaluate(ws, v_d, eta_d, u0_d, w0_d, Nv_d, Neta_d, hx0_d, hz0_d)
          istat = cudaDeviceSynchronize()
          Nv = Nv_d;  Neta = Neta_d
          do iz = 1, nz; do ix = 1, nkx
             k2 = kxh(ix)*kxh(ix) + kzh(iz)*kzh(iz)
             call solve_eta_mode(k2, a, lam, g, z, eta_h(:,ix,iz), Neta(:,ix,iz), Neta_prev(:,ix,iz))
             call solve_v_mode  (k2, a,      g, z,   v_h(:,ix,iz),   Nv(:,ix,iz),   Nv_prev(:,ix,iz))
          end do; end do
          Nv_prev = Nv;  Neta_prev = Neta
       end do
    end do

    eout = eta_h;  vout = v_h
    deallocate(eta_d, v_d, Nv_d, Neta_d, u0_d, w0_d, hx0_d, hz0_d)
    call nl_destroy(ws)
  end subroutine cpu_run

    subroutine solve_eta_mode(k2, a, lam, g, z, col, Ncur, Nprev)
      real(dp),    intent(in)    :: k2, a, lam, g, z
      complex(dp), intent(inout) :: col(0:n)
      complex(dp), intent(in)    :: Ncur(0:n), Nprev(0:n)
      real(dp) :: cr(0:n), ci(0:n), d2r(0:n), d2i(0:n)
      real(dp) :: rr(0:n), ri(0:n), uhr(0:n-2), uhi(0:n-2), outr(0:n), outi(0:n)
      cr = real(col);  ci = aimag(col)
      call cheb_d2(n, cr, d2r);  call cheb_d2(n, ci, d2i)
      rr = lam*( cr + a*(d2r - k2*cr) + g*real(Ncur)  + z*real(Nprev) )
      ri = lam*( ci + a*(d2i - k2*ci) + g*aimag(Ncur) + z*aimag(Nprev) )
      call helmholtz_solve(n, k2+lam, rr, uhr);  call shen_to_cheb(n, uhr, outr)
      call helmholtz_solve(n, k2+lam, ri, uhi);  call shen_to_cheb(n, uhi, outi)
      col = cmplx(outr, outi, dp)
    end subroutine solve_eta_mode

    subroutine solve_v_mode(k2, a, g, z, col, Ncur, Nprev)
      real(dp),    intent(in)    :: k2, a, g, z
      complex(dp), intent(inout) :: col(0:n)
      complex(dp), intent(in)    :: Ncur(0:n), Nprev(0:n)
      type(vif_t) :: d
      real(dp) :: cr(0:n), ci(0:n), Lr(0:n), Li(0:n), L2r(0:n), L2i(0:n), tr(0:n), ti(0:n)
      real(dp) :: rr(0:n), ri(0:n), outr(0:n), outi(0:n)
      cr = real(col);  ci = aimag(col)
      call cheb_d2(n, cr, tr);  Lr = tr - k2*cr
      call cheb_d2(n, ci, ti);  Li = ti - k2*ci
      call cheb_d2(n, Lr, tr);  L2r = tr - k2*Lr
      call cheb_d2(n, Li, ti);  L2i = ti - k2*Li
      rr = Lr + a*L2r + g*real(Ncur)  + z*real(Nprev)
      ri = Li + a*L2i + g*aimag(Ncur) + z*aimag(Nprev)
      call vif_setup(n, k2, a, d)
      call vif_solve(n, d, rr, outr)
      call vif_solve(n, d, ri, outi)
      col = cmplx(outr, outi, dp)
    end subroutine solve_v_mode

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
    ! R2C Nyquist zeroing: (0,0)=col1; x-Nyquist=LAST col (nkx); kz-Nyquist row; no -x half
    e0(:,1,1) = (0.0_dp,0.0_dp);      v0(:,1,1) = (0.0_dp,0.0_dp)
    e0(:,nkx,:) = (0.0_dp,0.0_dp);    v0(:,nkx,:) = (0.0_dp,0.0_dp)
    e0(:,:,nz/2+1) = (0.0_dp,0.0_dp); v0(:,:,nz/2+1) = (0.0_dp,0.0_dp)
  end subroutine build_init

end program test_stepper_xcheck_r2c
