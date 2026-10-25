!===============================================================================
! test_stepper_order_r2c.F90  --  [R2C] temporal convergence order of the full
! nonlinear stepper step_ev. The R2C analogue of test_stepper_order.
!
! Order is a TEMPORAL property of the SMR/CN scheme and is representation-
! independent (it does not depend on how many x-modes are stored). This test
! pins it on the nkx half-spectrum -- a genuine gap, since the C2C order test is
! skipped (it links the R2C nonlinear_mod) and the cross-checks pin the assembly
! at one dt but NOT the convergence rate.
!
! Self-convergence (Richardson):
!   p = log2( ||q(dt)-q(dt/2)|| / ||q(dt/2)-q(dt/4)|| ),  for eta and v.
!
! EXPECTED ORDER p ~ 2. The viscous term uses symmetric Crank-Nicolson (2nd
! order; the linear stepper showed p=2.000). The nonlinear term uses 3rd-order
! RKW3 weights, but in an IMEX scheme with a 2nd-order implicit part the COMBINED
! order at finite Re is 2 (the CN error dominates); full 3rd order would need the
! asymmetric SMR split. So:
!   * ASSERTED (resolved, mildly damped, nonlinear-active): p ~ 2 for eta & v.
!     n=32 resolves the sqrt(beta*dt/Re) v-solve wall layer to the dt/4 level;
!     small A keeps the explicit term stable.
!   * REPORTED (Re=1e4): order rises toward RKW3's 3 as viscosity vanishes;
!     under-resolved wall layer pollutes the estimate -- qualitative only.
!
! Order is asserted ONLY when the
! field is mildly damped (||q_T||/||q_0|| > 0.5) AND the differences are
! measurable (e2 > 1e-11). Assert window 1.7 < p < 2.4. (0,0)&Nyquist zeroed;
! u0=w0=0. NaN-hardened.
!
! Build & run ON A GPU NODE (module objects only, NEVER *.o):
!   nvfortran -cuda -cudalib=cufft -c types.F90 transforms_mod.F90 cheb_fft_mod.F90 \
!     fft3d_mod.F90 spectral3d_mod.F90 cheb_shen.F90 cheb_bih.F90 helm_band_mod.F90 \
!     vsolve_if_mod.F90 mean_mod.F90 cheb_ops_mod.F90 velocity_mod.F90 forcing_mod.F90 \
!     spec_helpers_mod.F90 helm_band_gpu_mod.F90 vsolve_if_gpu_mod.F90 nonlinear_mod.F90 \
!     stepper_lin_mod.F90 solver_ops_mod.F90 stepper_mod.F90
!   nvfortran -cuda -cudalib=cufft test_stepper_order_r2c.F90 \
!     types.o transforms_mod.o cheb_fft_mod.o fft3d_mod.o spectral3d_mod.o cheb_shen.o \
!     cheb_bih.o helm_band_mod.o vsolve_if_mod.o mean_mod.o cheb_ops_mod.o velocity_mod.o \
!     forcing_mod.o spec_helpers_mod.o helm_band_gpu_mod.o vsolve_if_gpu_mod.o \
!     nonlinear_mod.o stepper_lin_mod.o solver_ops_mod.o stepper_mod.o -o test_stepper_order_r2c
!   ./test_stepper_order_r2c
!===============================================================================
program test_stepper_order_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,           only: dp, ip
  use cheb_shen,       only: shen_to_cheb
  use cheb_bih,        only: shen_bih_to_cheb
  use helm_band_mod,   only: helm_band_setup
  use nonlinear_mod,   only: nl_workspace, nl_setup, nl_destroy
  use stepper_lin_mod, only: stepper_ev_setup
  use stepper_mod,     only: step_ev
  use solver_ops_mod,  only: band_ops_t, vsolve_ops_t, band_ops_alloc, &
                             band_ops_free, vsolve_ops_alloc, vsolve_ops_free
  use cudafor
  implicit none
  logical :: all_pass
  all_pass = .true.

  write(*,'(a)') '=== R2C full nonlinear stepper: temporal order (SMR RKW3 / symmetric CN) ==='
  !        n  nx nz      Re        A        dt0      Ns0  assert(p~2)
  call conv_nl(32, 8, 6,  90.0_dp,  0.01_dp, 1.0e-2_dp, 10, .true.,  all_pass)
  call conv_nl(32, 8, 6, 180.0_dp,  0.01_dp, 1.0e-2_dp, 10, .true.,  all_pass)
  call conv_nl(16, 8, 6, 1.0e4_dp,  0.05_dp, 2.0e-3_dp, 12, .false., all_pass)  ! RK character (report)

  if (all_pass) then
     write(*,'(/,a)') 'ALL TESTS PASSED'
  else
     write(*,'(/,a)') 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine conv_nl(n, nx, nz, Re, A, dt0, Ns0, want_assert, all_pass)
    integer(ip), intent(in)    :: n, nx, nz, Ns0
    real(dp),    intent(in)    :: Re, A, dt0
    logical,     intent(in)    :: want_assert
    logical,     intent(inout) :: all_pass
    integer(ip) :: nkx
    real(dp),    allocatable :: kxh(:), kzh(:)
    type(band_ops_t) :: bo                 ! y-operator band structure (host+device)
    complex(dp), allocatable :: e0(:,:,:), v0(:,:,:)
    complex(dp), allocatable :: ea(:,:,:), eb(:,:,:), ec(:,:,:), va(:,:,:), vb(:,:,:), vc(:,:,:)
    real(dp) :: pe, pv, e2e, e2v, n0e, nTe, ratio
    integer(ip) :: ix, iz, kk
    logical :: bad, mild, meas

    nkx = nx/2 + 1
    allocate(kxh(nkx), kzh(nz))
    do ix = 1, nkx; kxh(ix) = real(ix-1, dp); end do          ! non-negative half
    do iz = 1, nz; kk = iz-1; if (kk > nz/2) kk = kk - nz; kzh(iz) = real(kk, dp); end do
    call band_ops_alloc(bo, n)
    call helm_band_setup(n, bo%sdh, bo%sth, bo%bdh, bo%boff)
    bo%sd = bo%sdh; bo%st = bo%sth; bo%bd = bo%bdh

    allocate(e0(0:n,nkx,nz), v0(0:n,nkx,nz))
    allocate(ea(0:n,nkx,nz), eb(0:n,nkx,nz), ec(0:n,nkx,nz), va(0:n,nkx,nz), vb(0:n,nkx,nz), vc(0:n,nkx,nz))
    call build_init(n, nkx, nz, A, kxh, kzh, e0, v0)

    call evolve(n,nkx,nz, Re, dt0,          Ns0,   kxh,kzh, bo, e0,v0, ea,va)
    call evolve(n,nkx,nz, Re, dt0/2.0_dp, 2*Ns0,   kxh,kzh, bo, e0,v0, eb,vb)
    call evolve(n,nkx,nz, Re, dt0/4.0_dp, 4*Ns0,   kxh,kzh, bo, e0,v0, ec,vc)

    bad = .false.
    call order2(n,nkx,nz, ea,eb,ec, pe, e2e, bad)
    call order2(n,nkx,nz, va,vb,vc, pv, e2v, bad)

    ! mild-damping proxy on eta: ||eta(dt/4)|| / ||eta_0||
    n0e = 0.0_dp; nTe = 0.0_dp
    do iz = 1, nz; do ix = 1, nkx; do kk = 0, n
       n0e = n0e + abs(e0(kk,ix,iz))**2;  nTe = nTe + abs(ec(kk,ix,iz))**2
    end do; end do; end do
    ratio = sqrt(nTe / max(n0e, tiny(1.0_dp)))

    write(*,'(/,a,i0,a,i0,a,i0,a,i0,a,es8.1,a,es8.1,a,es8.1)') &
         '--- n=',n,' nx=',nx,' (nkx=',nkx,') nz=',nz,'  Re=',Re,'  A=',A,'  dt0=',dt0
    write(*,'(a,f6.3,a,es10.2,a,f6.3,a,es10.2)') &
         '   eta: p=',pe,'  e2=',e2e,'      v: p=',pv,'  e2=',e2v
    write(*,'(a,f6.3)') '   ||eta_T||/||eta_0|| = ', ratio

    call verdict('stable / finite (eta & v)', (.not. bad), all_pass)
    if (want_assert) then
       mild = (ratio > 0.5_dp)
       meas = (e2e > 1.0e-11_dp) .and. (e2v > 1.0e-11_dp)
       if (mild .and. meas) then
          call verdict('eta order ~ 2 (symmetric CN)', (pe > 1.7_dp .and. pe < 2.4_dp), all_pass)
          call verdict('v   order ~ 2 (symmetric CN)', (pv > 1.7_dp .and. pv < 2.4_dp), all_pass)
       else
          write(*,'(a)') '   [skip] order not asserted (strong damping or differences at round-off)'
       end if
    else
       write(*,'(a)') '   [note] nonlinear-dominated: order rises toward RKW3 (3); reported, not asserted'
    end if

    call band_ops_free(bo)
    deallocate(kxh, kzh, e0,v0, ea,eb,ec, va,vb,vc)
  end subroutine conv_nl

  subroutine order2(n, nkx, nz, qa, qb, qc, p, e2, bad)
    integer(ip), intent(in)    :: n, nkx, nz
    complex(dp), intent(in)    :: qa(0:n,nkx,nz), qb(0:n,nkx,nz), qc(0:n,nkx,nz)
    real(dp),    intent(out)   :: p, e2
    logical,     intent(inout) :: bad
    real(dp) :: e1, d
    integer(ip) :: ix, iz, k
    e1 = 0.0_dp; e2 = 0.0_dp
    do iz = 1, nz; do ix = 1, nkx; do k = 0, n
       d = abs(qa(k,ix,iz) - qb(k,ix,iz)); if (.not. (d == d)) bad = .true.; e1 = e1 + d*d
       d = abs(qb(k,ix,iz) - qc(k,ix,iz)); if (.not. (d == d)) bad = .true.; e2 = e2 + d*d
       if (abs(qc(k,ix,iz)) > huge(1.0_dp)) bad = .true.
    end do; end do; end do
    e1 = sqrt(e1); e2 = sqrt(e2)
    if (e2 > 0.0_dp) then; p = log(e1/e2)/log(2.0_dp); else; p = 0.0_dp; end if
  end subroutine order2

  subroutine evolve(n, nkx, nz, Re, dt, Ns, kxh, kzh, bo, e0, v0, eout, vout)
    integer(ip), intent(in)  :: n, nkx, nz, Ns
    real(dp),    intent(in)  :: Re, dt, kxh(nkx), kzh(nz)
    type(band_ops_t), intent(in) :: bo    ! band structure (device side already uploaded)
    complex(dp), intent(in)  :: e0(0:n,nkx,nz), v0(0:n,nkx,nz)
    complex(dp), intent(out) :: eout(0:n,nkx,nz), vout(0:n,nkx,nz)
    type(nl_workspace) :: ws
    real(dp),    device, allocatable :: kx(:), kz(:)
    type(vsolve_ops_t) :: vo               ! local v-solve apparatus
    complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:), Nv_prev(:,:,:), Neta_prev(:,:,:)
    integer(ip) :: it, istat, nxp
    nxp = 2*(nkx-1)                                   ! physical nx for nl_setup
    call nl_setup(ws, n+1, nxp, nz, kxh, kzh)
    allocate(kx(nkx), kz(nz))
    call vsolve_ops_alloc(vo, n, nkx, nz)
    allocate(eta(0:n,nkx,nz), v(0:n,nkx,nz), u0(0:n), w0(0:n), Nv_prev(0:n,nkx,nz), Neta_prev(0:n,nkx,nz))
    kx = kxh; kz = kzh
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
  end subroutine evolve

  subroutine build_init(n, nkx, nz, A, kxh, kzh, e0, v0)
    integer(ip), intent(in)  :: n, nkx, nz
    real(dp),    intent(in)  :: A, kxh(nkx), kzh(nz)
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
          decay  = A*exp(-(real(k,dp)/kc)**2)
          ure(k) = (rr-0.5_dp)*decay; uim(k) = (ri-0.5_dp)*decay
       end do
       call shen_to_cheb(n, ure, are); call shen_to_cheb(n, uim, aim)
       do k = 0, n; e0(k,ix,iz) = cmplx(are(k), aim(k), dp); end do
       do k = 0, n-4
          call random_number(rr); call random_number(ri)
          decay  = A*exp(-(real(k,dp)/kc)**2)
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

  subroutine verdict(label, ok, all_pass)
    character(*), intent(in)    :: label
    logical,      intent(in)    :: ok
    logical,      intent(inout) :: all_pass
    if (ok) then
       write(*,'(a,a)') '   [PASS] ', label
    else
       write(*,'(a,a)') '   [FAIL] ', label
       all_pass = .false.
    end if
  end subroutine verdict

end program test_stepper_order_r2c
