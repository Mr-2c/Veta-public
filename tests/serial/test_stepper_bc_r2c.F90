!===============================================================================
! test_stepper_bc_r2c.F90  --  [R2C] wall boundary conditions after stepping.
! The R2C analogue of test_stepper_bc.
!
! The Shen-Dirichlet basis enforces eta(+/-1)=0 and the influence-matrix v-solve
! enforces v(+/-1)=Dv(+/-1)=0 BY CONSTRUCTION -- a cheap, direct guard against the
! Lesson-8 Cinv-corruption / skipped-shen_to_cheb class (a wiring error can break
! the walls while the bulk still evolves plausibly). Step a smooth field with the
! FULL nonlinear stepper (step_ev) on the nkx half-spectrum, then assert the wall
! values/derivatives vanish for EVERY mode.
!
! The per-mode COLUMN wall formulas (degree n) are representation-independent; the
! solves are extent-agnostic (proven 4a/4b/4c) -- so the BCs should hold on the
! half-spectrum, and this verifies it rather than assuming.
!   f(+1)=sum a_k ; f(-1)=sum a_k(-1)^k ; f'(+1)=sum a_k k^2 ; f'(-1)=sum a_k(-1)^{k+1}k^2
!
! n=16,nx=8,nz=6 (=>nkx=5),A=0.05. (0,0) & Nyquist zeroed; u0=w0=0.
! TWO (Re,dt) rows: row 1 = a gentle regime
! (Re=10, dt=5e-3); row 2 = the PRODUCTION regime (Re=43500, dt=1e-3), where
! 1/alpha_2 = Re/(beta_2 dt) ~ 6.5e8, so the wall conditions are exercised at
! production conditioning. eta/v wall gates stay 1e-8 on both
! rows; the Dv gate on row 2 is 1e-14, calibrated from measurement (see below).
! NaN-hardened. Build & run ON A GPU NODE (module objects only, NEVER *.o):
!   nvfortran -cuda -cudalib=cufft -c types.F90 transforms_mod.F90 cheb_fft_mod.F90 \
!     fft3d_mod.F90 spectral3d_mod.F90 cheb_shen.F90 cheb_bih.F90 helm_band_mod.F90 \
!     vsolve_if_mod.F90 mean_mod.F90 cheb_ops_mod.F90 velocity_mod.F90 forcing_mod.F90 \
!     spec_helpers_mod.F90 helm_band_gpu_mod.F90 vsolve_if_gpu_mod.F90 nonlinear_mod.F90 \
!     stepper_lin_mod.F90 solver_ops_mod.F90 stepper_mod.F90
!   nvfortran -cuda -cudalib=cufft test_stepper_bc_r2c.F90 \
!     types.o transforms_mod.o cheb_fft_mod.o fft3d_mod.o spectral3d_mod.o cheb_shen.o \
!     cheb_bih.o helm_band_mod.o vsolve_if_mod.o mean_mod.o cheb_ops_mod.o velocity_mod.o \
!     forcing_mod.o spec_helpers_mod.o helm_band_gpu_mod.o vsolve_if_gpu_mod.o \
!     nonlinear_mod.o stepper_lin_mod.o solver_ops_mod.o stepper_mod.o -o test_stepper_bc_r2c
!   ./test_stepper_bc_r2c
!===============================================================================
program test_stepper_bc_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,           only: dp, ip
  use cheb_shen,       only: shen_to_cheb
  use cheb_bih,        only: shen_bih_to_cheb
  use helm_band_mod,   only: helm_band_setup
  use mean_mod,        only: MCSUB              ! 1/alpha_2 banner (beta_2 = MCSUB(2)/2)
  use nonlinear_mod,   only: nl_workspace, nl_setup, nl_destroy
  use stepper_lin_mod, only: stepper_ev_setup
  use stepper_mod,     only: step_ev
  use solver_ops_mod,  only: band_ops_t, vsolve_ops_t, band_ops_alloc, &
                             band_ops_free, vsolve_ops_alloc, vsolve_ops_free
  use cudafor
  implicit none

  integer(ip), parameter :: n = 16, nx = 8, nz = 6, Ns = 5
  integer(ip), parameter :: nkx = nx/2 + 1
  real(dp),    parameter :: A = 0.05_dp
  ! 2-row (Re,dt) sweep. Row 1 = a gentle regime;
  ! row 2 = the PRODUCTION regime: 1/alpha_2 = Re/(beta_2 dt) ~ 6.5e8 (beta_2 =
  ! MCSUB(2)/2), 2e4 stiffer than row 1.
  real(dp),    parameter :: Rev(2) = [10.0_dp, 43500.0_dp]
  real(dp),    parameter :: dtv(2) = [5.0e-3_dp, 1.0e-3_dp]
  real(dp),    parameter :: tol = 1.0e-8_dp

  real(dp) :: kxh(nkx), kzh(nz)
  type(band_ops_t) :: bo                 ! y-operator band structure (host+device)
  complex(dp) :: e0(0:n,nkx,nz), v0(0:n,nkx,nz), eta(0:n,nkx,nz), v(0:n,nkx,nz)
  complex(dp) :: vlo, vhi, dlo, dhi
  real(dp) :: meta, mv, mdv, nmag, tol_dv, inva
  integer(ip) :: ix, iz, kk, ir
  logical :: ok, allok

  ! kx = non-negative half [0, nx/2]; kz = full (cuFFT order)
  do ix = 1, nkx; kxh(ix) = real(ix-1, dp); end do
  do iz = 1, nz; kk = iz-1; if (kk > nz/2) kk = kk - nz; kzh(iz) = real(kk, dp); end do
  call band_ops_alloc(bo, n)
  call helm_band_setup(n, bo%sdh, bo%sth, bo%bdh, bo%boff)
  call build_init(e0, v0)

  write(*,'(a)') '=== R2C wall boundary conditions after stepping (full nonlinear step_ev) ==='
  write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0,a)') '  n=',n,' nx=',nx,' (nkx=',nkx,') nz=',nz,'  (',Ns,' steps per row)'

  allok = .true.
  do ir = 1, 2
     call evolve(Rev(ir), dtv(ir), e0, v0, eta, v)

     meta = 0.0_dp; mv = 0.0_dp; mdv = 0.0_dp; nmag = 0.0_dp
     do iz = 1, nz; do ix = 1, nkx
        call wall_vals(eta(:,ix,iz), vlo, vhi, dlo, dhi)
        meta = max(meta, abs(vlo), abs(vhi))
        call wall_vals(v(:,ix,iz), vlo, vhi, dlo, dhi)
        mv  = max(mv,  abs(vlo), abs(vhi))
        mdv = max(mdv, abs(dlo), abs(dhi))
        nmag = max(nmag, maxval(abs(eta(:,ix,iz))), maxval(abs(v(:,ix,iz))))  ! evolved, nonzero
     end do; end do

     ! Dv wall gate: 1e-8 on row 1. Row 2 calibrated from measurement:
     ! max|Dv| =
     ! 1.9432e-16 at 1/alpha_2 ~ 6.5e8 -- the closure enforces discrete Dv=0
     ! BY CONSTRUCTION and the margin holds at production conditioning.
     ! Gate = 1e-14 (~50x measured; headroom for compiler/FMA wobble).
     tol_dv = tol
     if (ir == 2) tol_dv = 1.0e-14_dp
     inva = Rev(ir)/(0.5_dp*MCSUB(2)*dtv(ir))
     write(*,'(/,a,i0,a,f8.1,a,es9.2,a,es9.2)') '--- row ', ir, ':  Re=', Rev(ir), &
          '  dt=', dtv(ir), '  1/alpha_2=', inva
     write(*,'(a,es12.4,a,es8.1,a)') '  max|eta(+/-1)|     = ', meta, '   (Dirichlet, tol ', tol, ')'
     write(*,'(a,es12.4,a,es8.1,a)') '  max|v(+/-1)|       = ', mv,   '   (clamped,   tol ', tol, ')'
     write(*,'(a,es12.4,a,es8.1,a)') '  max|Dv(+/-1)|      = ', mdv,  '   (clamped,   tol ', tol_dv, ')'
     write(*,'(a,es12.4)')           '  evolved field mag  = ', nmag

     ok = (meta == meta) .and. (mv == mv) .and. (mdv == mdv) &
          .and. meta < tol .and. mv < tol .and. mdv < tol_dv &
          .and. nmag > 1.0e-6_dp .and. nmag < huge(1.0_dp)     ! nonzero guard (not vacuous)
     if (ok) then
        write(*,'(a)') '  [PASS] eta=0 and v=Dv=0 at both walls for every mode (R2C half-spectrum)'
     else
        write(*,'(a)') '  [FAIL] wall BCs violated after stepping'
     end if
     allok = allok .and. ok
  end do

  call band_ops_free(bo)
  if (allok) then
     write(*,'(/,a)') 'ALL TESTS PASSED'
  else
     write(*,'(/,a)') 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  ! wall values and derivatives of a Chebyshev-coefficient column (degree n)
  subroutine wall_vals(col, vlo, vhi, dlo, dhi)
    complex(dp), intent(in)  :: col(0:n)
    complex(dp), intent(out) :: vlo, vhi, dlo, dhi   ! f(-1), f(+1), f'(-1), f'(+1)
    integer(ip) :: k, s
    real(dp)    :: kk2
    vlo = (0.0_dp,0.0_dp); vhi = (0.0_dp,0.0_dp)
    dlo = (0.0_dp,0.0_dp); dhi = (0.0_dp,0.0_dp)
    do k = 0, n
       s   = 1 - 2*mod(k,2)            ! (-1)^k
       kk2 = real(k*k, dp)
       vhi = vhi + col(k)
       vlo = vlo + col(k)*real(s, dp)
       dhi = dhi + col(k)*kk2
       dlo = dlo - col(k)*real(s, dp)*kk2
    end do
  end subroutine wall_vals

  ! (Re_r, dt_r) are per-row ARGUMENTS (were module parameters): the same
  ! IC evolves once per sweep row; stepper_ev_setup below already takes dt, Re.
  subroutine evolve(Re_r, dt_r, e0, v0, eout, vout)
    real(dp),    intent(in)  :: Re_r, dt_r
    complex(dp), intent(in)  :: e0(0:n,nkx,nz), v0(0:n,nkx,nz)
    complex(dp), intent(out) :: eout(0:n,nkx,nz), vout(0:n,nkx,nz)
    type(nl_workspace) :: ws
    real(dp),    device, allocatable :: kx(:), kz(:)
    type(vsolve_ops_t) :: vo               ! local v-solve apparatus
    complex(dp), device, allocatable :: etad(:,:,:), vd(:,:,:), u0(:), w0(:), Nv_prev(:,:,:), Neta_prev(:,:,:)
    integer(ip) :: it, istat
    call nl_setup(ws, n+1, nx, nz, kxh, kzh)          ! R2C: physical nx, kxh nkx-long
    allocate(kx(nkx), kz(nz))
    call vsolve_ops_alloc(vo, n, nkx, nz)
    allocate(etad(0:n,nkx,nz), vd(0:n,nkx,nz), u0(0:n), w0(0:n), Nv_prev(0:n,nkx,nz), Neta_prev(0:n,nkx,nz))
    kx = kxh; kz = kzh
    bo%sd = bo%sdh; bo%st = bo%sth; bo%bd = bo%bdh
    call stepper_ev_setup(n, nkx, nz, dt_r, Re_r, kxh, kzh, vo%vm, vo%vp, vo%Cinv)
    u0 = (0.0_dp,0.0_dp); w0 = (0.0_dp,0.0_dp)
    Nv_prev = (0.0_dp,0.0_dp); Neta_prev = (0.0_dp,0.0_dp)
    etad = e0; vd = v0
    do it = 1, Ns
       call step_ev(ws, n, nkx, nz, dt_r, Re_r, kx, kz, bo, vo, &
                    u0, w0, etad, vd, Nv_prev, Neta_prev)
    end do
    istat = cudaDeviceSynchronize()
    eout = etad; vout = vd
    call vsolve_ops_free(vo)
    deallocate(kx, kz, etad, vd, u0, w0, Nv_prev, Neta_prev)
    call nl_destroy(ws)
  end subroutine evolve

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
    e0(:,1,1) = (0.0_dp,0.0_dp);   v0(:,1,1) = (0.0_dp,0.0_dp)
    e0(:,nkx,:) = (0.0_dp,0.0_dp); v0(:,nkx,:) = (0.0_dp,0.0_dp)
    e0(:,:,nz/2+1) = (0.0_dp,0.0_dp); v0(:,:,nz/2+1) = (0.0_dp,0.0_dp)
  end subroutine build_init

end program test_stepper_bc_r2c
