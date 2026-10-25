!===============================================================================
! test_stepper_ev.F90  --  validate the combined (eta + v) linear viscous stepper
! step_ev_linear across several Reynolds numbers.
!
! For each Re: evolve smooth clamped initial fields (eta Dirichlet, v clamped) to
! a fixed time with dt, dt/2, dt/4 and measure the temporal self-convergence
! order for BOTH fields,
!   p = log2( ||q(dt)-q(dt/2)|| / ||q(dt/2)-q(dt/4)|| ),
! expecting p ~ 2 (Crank-Nicolson) for eta (Helmholtz) and v (biharmonic).
!
! As established for the eta-only test, the Richardson order is only meaningful
! in the MILD-DAMPING (asymptotic) regime sigma*dt/Re << 1; we therefore assert
! order per field only when that field is mildly damped (||q_T||/||q_0|| > 0.5),
! sweeping Re = 90,180,360 which all sit in that regime here. A strong-damping
! Re=1 case is kept as an A-stability / dissipativity demonstration (no order).
!
! Build:
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran -c cheb_bih.F90
!   nvfortran -c helm_band_mod.F90
!   nvfortran -c vsolve_if_mod.F90
!   nvfortran -cuda -c cheb_ops_mod.F90
!   nvfortran -cuda -c helm_band_gpu_mod.F90
!   nvfortran -cuda -c spec_helpers_mod.F90
!   nvfortran -cuda -c vsolve_if_gpu_mod.F90
!   nvfortran -cuda -c stepper_lin_mod.F90
!   nvfortran -cuda -c test_stepper_ev.F90
!   nvfortran -cuda -o test_stepper_ev test_stepper_ev.o stepper_lin_mod.o \
!             vsolve_if_gpu_mod.o spec_helpers_mod.o helm_band_gpu_mod.o \
!             cheb_ops_mod.o vsolve_if_mod.o helm_band_mod.o cheb_bih.o \
!             cheb_shen.o types.o
!   ./test_stepper_ev
!===============================================================================
program test_stepper_ev
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,           only: dp, ip
  use cheb_shen,       only: shen_to_cheb
  use cheb_bih,        only: shen_bih_to_cheb
  use helm_band_mod,   only: helm_band_setup
  use stepper_lin_mod, only: stepper_ev_setup, step_ev_linear
  use cudafor
  implicit none
  logical :: all_pass
  all_pass = .true.
  call conv_case_ev(32, 8, 6,  90.0_dp, 0.02_dp, 20, .true.,  all_pass)
  call conv_case_ev(32, 8, 6, 180.0_dp, 0.02_dp, 20, .true.,  all_pass)
  call conv_case_ev(32, 8, 6, 360.0_dp, 0.02_dp, 20, .true.,  all_pass)
  call conv_case_ev(24, 8, 6,   1.0_dp, 0.02_dp, 10, .false., all_pass)
  if (all_pass) then
     print *, 'ALL TESTS PASSED'
  else
     print *, 'SOME TESTS FAILED'
  end if

contains

  subroutine conv_case_ev(n, nx, nz, Re, dt0, Ns0, want_order, all_pass)
    integer(ip), intent(in)    :: n, nx, nz, Ns0
    real(dp),    intent(in)    :: Re, dt0
    logical,     intent(in)    :: want_order
    logical,     intent(inout) :: all_pass
    real(dp) :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
    real(dp) :: kxh(nx), kzh(nz)
    complex(dp), allocatable :: e0(:,:,:), v0(:,:,:)
    complex(dp), allocatable :: ea(:,:,:), eb(:,:,:), ec(:,:,:)
    complex(dp), allocatable :: va(:,:,:), vb(:,:,:), vc(:,:,:)
    real(dp) :: ure(0:n-2), uim(0:n-2), are(0:n), aim(0:n)
    real(dp) :: vre(0:n-4), vim(0:n-4), bre(0:n), bim(0:n)
    real(dp) :: rr, ri, decay, kc
    integer(ip) :: ix, iz, k, kk
    real(dp) :: pe, pv, re0, ref, rv0, rvf
    logical  :: bad

    do ix = 1, nx; kk = ix-1; if (kk > nx/2) kk = kk - nx; kxh(ix) = real(kk, dp); end do
    do iz = 1, nz; kk = iz-1; if (kk > nz/2) kk = kk - nz; kzh(iz) = real(kk, dp); end do
    call helm_band_setup(n, sdiag, stail, bdiag, boff)

    allocate(e0(0:n,nx,nz), v0(0:n,nx,nz))
    allocate(ea(0:n,nx,nz), eb(0:n,nx,nz), ec(0:n,nx,nz))
    allocate(va(0:n,nx,nz), vb(0:n,nx,nz), vc(0:n,nx,nz))

    ! smooth initial fields: damp the Shen coeffs with exp(-(k/kc)^2)
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
    do iz = 1, nz; do ix = 1, nx
       do k = 0, n-2
          call random_number(rr); call random_number(ri)
          decay  = exp(-(real(k,dp)/kc)**2)
          ure(k) = (rr-0.5_dp)*decay; uim(k) = (ri-0.5_dp)*decay
       end do
       call shen_to_cheb(n, ure, are); call shen_to_cheb(n, uim, aim)
       do k = 0, n; e0(k,ix,iz) = cmplx(are(k), aim(k), dp); end do
       do k = 0, n-4
          call random_number(rr); call random_number(ri)
          decay  = exp(-(real(k,dp)/kc)**2)
          vre(k) = (rr-0.5_dp)*decay; vim(k) = (ri-0.5_dp)*decay
       end do
       call shen_bih_to_cheb(n, vre, bre); call shen_bih_to_cheb(n, vim, bim)
       do k = 0, n; v0(k,ix,iz) = cmplx(bre(k), bim(k), dp); end do
    end do; end do

    call evolve_ev(n,nx,nz, Re, dt0,          Ns0,   kxh,kzh, sdiag,stail,bdiag,boff, e0,v0, ea,va)
    call evolve_ev(n,nx,nz, Re, dt0/2.0_dp, 2*Ns0,   kxh,kzh, sdiag,stail,bdiag,boff, e0,v0, eb,vb)
    call evolve_ev(n,nx,nz, Re, dt0/4.0_dp, 4*Ns0,   kxh,kzh, sdiag,stail,bdiag,boff, e0,v0, ec,vc)

    ! order_norms ACCUMULATES into bad (intent(inout), eta then v share the
    ! verdict) -- it must start defined. Uninitialized, it read stack garbage
    ! that happened to be .false. in every build until the R4/R5 rebuild
    ! shifted the layout and it came up .true. with bit-identical numerics
    ! (the sibling test_stepper_order_r2c always had this init).
    bad = .false.
    call order_norms(n,nx,nz, ea,eb,ec, e0, pe, re0, ref, bad)
    call order_norms(n,nx,nz, va,vb,vc, v0, pv, rv0, rvf, bad)

    write(*,'(/,a,i0,a,i0,a,i0,a,f7.1)') '--- n=',n,'  nx=',nx,'  nz=',nz,'   Re=',Re
    write(*,'(a,f6.3,a,f8.5,a,f6.3,a,f8.5)') &
         '   eta: order p=',pe,'  ratio=',ref/re0,'      v: order p=',pv,'  ratio=',rvf/rv0

    call verdict('stable / finite (eta & v)', (.not. bad), all_pass)
    call verdict('dissipative eta', (ref < re0), all_pass)
    call verdict('dissipative v',   (rvf < rv0), all_pass)
    if (want_order) then
       if (ref/re0 > 0.5_dp) call verdict('eta temporal order ~ 2', &
            (pe > 1.7_dp .and. pe < 2.3_dp), all_pass)
       if (rvf/rv0 > 0.5_dp) call verdict('v   temporal order ~ 2', &
            (pv > 1.7_dp .and. pv < 2.3_dp), all_pass)
    else
       write(*,'(a)') '   [note] strong-damping case: order reported only, not asserted'
    end if

    deallocate(e0,v0, ea,eb,ec, va,vb,vc)
  end subroutine conv_case_ev

  subroutine order_norms(n, nx, nz, qa, qb, qc, q0, p, nrm0, nrmf, bad)
    integer(ip), intent(in)    :: n, nx, nz
    complex(dp), intent(in)    :: qa(0:n,nx,nz), qb(0:n,nx,nz), qc(0:n,nx,nz), q0(0:n,nx,nz)
    real(dp),    intent(out)   :: p, nrm0, nrmf
    logical,     intent(inout) :: bad
    real(dp) :: e1, e2, d
    integer(ip) :: ix, iz, k
    e1 = 0.0_dp; e2 = 0.0_dp; nrm0 = 0.0_dp; nrmf = 0.0_dp
    do iz = 1, nz; do ix = 1, nx; do k = 0, n
       d = abs(qa(k,ix,iz) - qb(k,ix,iz)); if (.not. (d == d)) bad = .true.; e1 = e1 + d*d
       d = abs(qb(k,ix,iz) - qc(k,ix,iz)); if (.not. (d == d)) bad = .true.; e2 = e2 + d*d
       nrm0 = nrm0 + abs(q0(k,ix,iz))**2
       nrmf = nrmf + abs(qc(k,ix,iz))**2
       if (abs(qc(k,ix,iz)) > huge(1.0_dp)) bad = .true.
    end do; end do; end do
    e1 = sqrt(e1); e2 = sqrt(e2); nrm0 = sqrt(nrm0); nrmf = sqrt(nrmf)
    if (e2 > 0.0_dp) then; p = log(e1/e2)/log(2.0_dp); else; p = 0.0_dp; end if
  end subroutine order_norms

  subroutine evolve_ev(n, nx, nz, Re, dt, Ns, kxh, kzh, sdiag, stail, bdiag, boff, &
                       e0, v0, eout, vout)
    integer(ip), intent(in)  :: n, nx, nz, Ns
    real(dp),    intent(in)  :: Re, dt, kxh(nx), kzh(nz)
    real(dp),    intent(in)  :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
    complex(dp), intent(in)  :: e0(0:n,nx,nz), v0(0:n,nx,nz)
    complex(dp), intent(out) :: eout(0:n,nx,nz), vout(0:n,nx,nz)
    integer(ip) :: it, istat
    real(dp),    device, allocatable :: kx(:), kz(:), sd(:), st(:), bd(:)
    real(dp),    device, allocatable :: vm_all(:,:,:,:), vp_all(:,:,:,:), Cinv_all(:,:,:,:,:)
    complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:)

    allocate(kx(nx), kz(nz), sd(0:n-2), st(0:n-2), bd(0:n-2))
    allocate(vm_all(0:n,nx,nz,3), vp_all(0:n,nx,nz,3), Cinv_all(2,2,nx,nz,3))
    allocate(eta(0:n,nx,nz), v(0:n,nx,nz))

    kx = kxh; kz = kzh; sd = sdiag; st = stail; bd = bdiag
    ! per-stage vif precompute for THIS dt (variable-dt hook)
    call stepper_ev_setup(n, nx, nz, dt, Re, kxh, kzh, vm_all, vp_all, Cinv_all)

    eta = e0; v = v0
    do it = 1, Ns
       call step_ev_linear(n, nx, nz, dt, Re, kx, kz, sd, st, bd, boff, &
                           vm_all, vp_all, Cinv_all, eta, v)
    end do
    istat = cudaDeviceSynchronize()
    eout = eta; vout = v

    deallocate(kx, kz, sd, st, bd, vm_all, vp_all, Cinv_all, eta, v)
  end subroutine evolve_ev

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

end program test_stepper_ev
