!===============================================================================
! test_stepper_eta.F90  --  validate the linear (N=0) eta viscous stepper.
!
! Strategy: temporal self-convergence (Richardson). Evolve the SAME clamped-
! Dirichlet initial field to a fixed time T with steps dt, dt/2, dt/4 and form
!   e1 = ||eta(dt) - eta(dt/2)||,  e2 = ||eta(dt/2) - eta(dt/4)||,
!   observed order p = log2(e1/e2).
! The pure-viscous part is Crank-Nicolson, so p ~ 2 is the expected order (the
! 3rd-order RK acts only on the nonlinear term, which is off here).
!
! VALIDITY OF THE ORDER ESTIMATE -- IMPORTANT:
! CN's truncation error per step is ~(sigma*dt/Re)^3 ONLY while sigma*dt/Re << 1,
! where sigma is a wall-normal (Helmholtz) eigenvalue. The energetic mid/high
! modes have sigma of order hundreds, so the Richardson order is only meaningful
! when sigma_max*dt/Re is small -- i.e. in the MILD-DAMPING (asymptotic) regime.
! In a strongly damped run the stiff modes sit in the non-asymptotic CN regime
! (amplification -> -1, unrelated to (sigma*dt/Re)^3) and the observed order is
! garbage (can read 5+). So we ASSERT the order only when the run is mildly
! damped (||eta_T||/||eta_0|| > 0.5), and otherwise report it for information.
!
! Case A (Re=180, want_order): mild damping, the valid order measurement.
! Case B (Re=1): strong damping -- a CN A-stability / dissipativity demo only.
!
! The initial field is smooth (spectral coeffs damped with k) but still spans
! enough modes that Case B genuinely exercises the stiff regime.
!
! Build:
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran -c helm_band_mod.F90
!   nvfortran -cuda -c cheb_ops_mod.F90
!   nvfortran -cuda -c helm_band_gpu_mod.F90
!   nvfortran -cuda -c spec_helpers_mod.F90
!   nvfortran -cuda -c stepper_lin_mod.F90
!   nvfortran -cuda -c test_stepper_eta.F90
!   nvfortran -cuda -o test_stepper_eta test_stepper_eta.o stepper_lin_mod.o \
!             spec_helpers_mod.o helm_band_gpu_mod.o cheb_ops_mod.o \
!             helm_band_mod.o cheb_shen.o types.o
!   ./test_stepper_eta
!===============================================================================
program test_stepper_eta
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,           only: dp, ip
  use cheb_shen,       only: shen_to_cheb
  use helm_band_mod,   only: helm_band_setup
  use stepper_lin_mod, only: step_eta_linear
  use cudafor
  implicit none
  logical :: all_pass
  all_pass = .true.
  call conv_case(32,  8, 6, 180.0_dp, 0.02_dp, 20, .true.,  all_pass)  ! mild damping: order check
  call conv_case(24,  8, 6,   1.0_dp, 0.02_dp, 10, .false., all_pass)  ! strong damping: stability only
  if (all_pass) then
     print *, 'ALL TESTS PASSED'
  else
     print *, 'SOME TESTS FAILED'
  end if

contains

  subroutine conv_case(n, nx, nz, Re, dt0, Ns0, want_order, all_pass)
    integer(ip), intent(in)    :: n, nx, nz, Ns0
    real(dp),    intent(in)    :: Re, dt0
    logical,     intent(in)    :: want_order
    logical,     intent(inout) :: all_pass
    real(dp) :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
    real(dp) :: kxh(nx), kzh(nz)
    complex(dp), allocatable :: eta0(:,:,:), ea(:,:,:), eb(:,:,:), ec(:,:,:)
    real(dp) :: ure(0:n-2), uim(0:n-2), are(0:n), aim(0:n), rr, ri, decay, kc
    integer(ip) :: ix, iz, k, kk
    real(dp) :: e1, e2, p, nrm0, nrmf, d, ratio
    logical  :: bad, mild

    do ix = 1, nx; kk = ix-1; if (kk > nx/2) kk = kk - nx; kxh(ix) = real(kk, dp); end do
    do iz = 1, nz; kk = iz-1; if (kk > nz/2) kk = kk - nz; kzh(iz) = real(kk, dp); end do
    call helm_band_setup(n, sdiag, stail, bdiag, boff)

    allocate(eta0(0:n,nx,nz), ea(0:n,nx,nz), eb(0:n,nx,nz), ec(0:n,nx,nz))

    ! smooth clamped-Dirichlet random init: random Shen coeffs damped by exp(-(k/kc)^2)
    kc = 0.25_dp*real(n, dp)
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
          ure(k) = (rr-0.5_dp)*decay
          uim(k) = (ri-0.5_dp)*decay
       end do
       call shen_to_cheb(n, ure, are)
       call shen_to_cheb(n, uim, aim)
       do k = 0, n; eta0(k,ix,iz) = cmplx(are(k), aim(k), dp); end do
    end do; end do

    call evolve(n,nx,nz, Re, dt0,          Ns0,   kxh,kzh, sdiag,stail,bdiag,boff, eta0, ea)
    call evolve(n,nx,nz, Re, dt0/2.0_dp, 2*Ns0,   kxh,kzh, sdiag,stail,bdiag,boff, eta0, eb)
    call evolve(n,nx,nz, Re, dt0/4.0_dp, 4*Ns0,   kxh,kzh, sdiag,stail,bdiag,boff, eta0, ec)

    e1 = 0.0_dp; e2 = 0.0_dp; nrm0 = 0.0_dp; nrmf = 0.0_dp; bad = .false.
    do iz = 1, nz; do ix = 1, nx; do k = 0, n
       d = abs(ea(k,ix,iz) - eb(k,ix,iz)); if (.not. (d == d)) bad = .true.; e1 = e1 + d*d
       d = abs(eb(k,ix,iz) - ec(k,ix,iz)); if (.not. (d == d)) bad = .true.; e2 = e2 + d*d
       nrm0 = nrm0 + abs(eta0(k,ix,iz))**2
       nrmf = nrmf + abs(ec (k,ix,iz))**2
       if (abs(ec(k,ix,iz)) > huge(1.0_dp)) bad = .true.
    end do; end do; end do
    e1 = sqrt(e1); e2 = sqrt(e2); nrm0 = sqrt(nrm0); nrmf = sqrt(nrmf)
    if (e2 > 0.0_dp) then; p = log(e1/e2)/log(2.0_dp); else; p = 0.0_dp; end if
    ratio = nrmf/nrm0
    mild  = (ratio > 0.5_dp)

    write(*,'(/,a,i0,a,i0,a,i0,a,f7.1)') '--- n=',n,'  nx=',nx,'  nz=',nz,'   Re=',Re
    write(*,'(a,es12.4,a,es12.4,a,f6.3)') '   e(dt,dt/2)=',e1,'   e(dt/2,dt/4)=',e2,'   order p=',p
    write(*,'(a,es12.4,a,es12.4,a,f8.5)') '   |eta0|   =',nrm0,'   |eta_T|     =',nrmf,'   ratio =',ratio

    call verdict('stable / finite', (.not. bad), all_pass)
    call verdict('dissipative (|eta_T| < |eta0|)', (nrmf < nrm0), all_pass)
    if (want_order) then
       if (mild) then
          call verdict('temporal order ~ 2 (mild-damping/asymptotic)', &
                       (p > 1.7_dp .and. p < 2.3_dp), all_pass)
       else
          write(*,'(a)') '   [note] damping too strong to assert order (Richardson invalid here)'
       end if
    else
       write(*,'(a)') '   [note] strong-damping case: stiff modes non-asymptotic, order not asserted'
    end if

    deallocate(eta0, ea, eb, ec)
  end subroutine conv_case

  subroutine evolve(n, nx, nz, Re, dt, Ns, kxh, kzh, sdiag, stail, bdiag, boff, eta0, out)
    integer(ip), intent(in)  :: n, nx, nz, Ns
    real(dp),    intent(in)  :: Re, dt, kxh(nx), kzh(nz)
    real(dp),    intent(in)  :: sdiag(0:n-2), stail(0:n-2), bdiag(0:n-2), boff
    complex(dp), intent(in)  :: eta0(0:n,nx,nz)
    complex(dp), intent(out) :: out(0:n,nx,nz)
    integer(ip) :: it, istat
    real(dp),    device, allocatable :: kx(:), kz(:), sd(:), st(:), bd(:)
    complex(dp), device, allocatable :: eta(:,:,:), Leta(:,:,:), rhs(:,:,:), scr(:,:,:)
    complex(dp), device, allocatable :: ftil(:,:,:), etash(:,:,:)

    allocate(kx(nx), kz(nz), sd(0:n-2), st(0:n-2), bd(0:n-2))
    allocate(eta(0:n,nx,nz), Leta(0:n,nx,nz), rhs(0:n,nx,nz), scr(0:n,nx,nz))
    allocate(ftil(0:n-2,nx,nz), etash(0:n-2,nx,nz))

    kx = kxh; kz = kzh; sd = sdiag; st = stail; bd = bdiag
    eta = eta0
    do it = 1, Ns
       call step_eta_linear(n, nx, nz, dt, Re, kx, kz, sd, st, bd, boff, eta, &
                            Leta, rhs, ftil, etash, scr)
    end do
    istat = cudaDeviceSynchronize()
    out = eta

    deallocate(kx, kz, sd, st, bd, eta, Leta, rhs, scr, ftil, etash)
  end subroutine evolve

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

end program test_stepper_eta
