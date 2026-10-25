!===============================================================================
! test_stepper_full_r2c.F90  --  [R2C] stability probe / NaN watchdog for the
! full nonlinear stepper step_ev. The R2C analogue of test_stepper_full.
!
! A cheap structural-sanity guard on the nkx half-spectrum: a tiny-amplitude /
! tiny-dt run MUST stay finite if the wiring is sound (the bug-vs-CFL
! discriminator -- the ONLY asserted case). The escalating cases are
! INFORMATIONAL: they print FINITE / NON-FINITE to map where the EXPLICIT scheme
! loses stability (advective CFL on the clustered Chebyshev grid + y-aliasing,
! since 3/2 dealiasing is x,z only) -- not a code bug, the physics of the
! explicit term, representation-independent.
!
! u0=w0=0; (0,0) & Nyquist columns zeroed. NaN-hardened (sum/maxval finiteness).
! Build & run ON A GPU NODE (module objects only, NEVER *.o):
!   nvfortran -cuda -cudalib=cufft -c types.F90 transforms_mod.F90 cheb_fft_mod.F90 \
!     fft3d_mod.F90 spectral3d_mod.F90 cheb_shen.F90 cheb_bih.F90 helm_band_mod.F90 \
!     vsolve_if_mod.F90 mean_mod.F90 cheb_ops_mod.F90 velocity_mod.F90 forcing_mod.F90 \
!     spec_helpers_mod.F90 helm_band_gpu_mod.F90 vsolve_if_gpu_mod.F90 nonlinear_mod.F90 \
!     stepper_lin_mod.F90 solver_ops_mod.F90 stepper_mod.F90
!   nvfortran -cuda -cudalib=cufft test_stepper_full_r2c.F90 \
!     types.o transforms_mod.o cheb_fft_mod.o fft3d_mod.o spectral3d_mod.o cheb_shen.o \
!     cheb_bih.o helm_band_mod.o vsolve_if_mod.o mean_mod.o cheb_ops_mod.o velocity_mod.o \
!     forcing_mod.o spec_helpers_mod.o helm_band_gpu_mod.o vsolve_if_gpu_mod.o \
!     nonlinear_mod.o stepper_lin_mod.o solver_ops_mod.o stepper_mod.o -o test_stepper_full_r2c
!   ./test_stepper_full_r2c
!===============================================================================
program test_stepper_full_r2c
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

  write(*,'(a)') '=== R2C nonlinear stepper stability probe (per-step max-norms) ==='
  !        n  nx nz      Re         A        dt      Ns   assert_finite
  call probe(16, 8, 6, 1.0e3_dp, 1.0e-3_dp, 1.0e-3_dp, 12, .true.,  all_pass)  ! structural sanity
  call probe(16, 8, 6, 1.0e3_dp, 1.0e-2_dp, 1.0e-3_dp, 12, .false., all_pass)  ! small
  call probe(16, 8, 6, 1.0e4_dp, 5.0e-2_dp, 1.0e-3_dp, 12, .false., all_pass)  ! high-Re, larger A
  call probe(16, 8, 6, 1.0e4_dp, 5.0e-2_dp, 5.0e-3_dp, 12, .false., all_pass)  ! + larger dt
  call probe(24, 8, 6, 1.0e4_dp, 2.0e-1_dp, 1.0e-2_dp, 12, .false., all_pass)  ! ~ original (expected blow-up)

  if (all_pass) then
     write(*,'(/,a)') 'ALL TESTS PASSED'
  else
     write(*,'(/,a)') 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine probe(n, nx, nz, Re, A, dt, Ns, assert_finite, all_pass)
    integer(ip), intent(in)    :: n, nx, nz, Ns
    real(dp),    intent(in)    :: Re, A, dt
    logical,     intent(in)    :: assert_finite
    logical,     intent(inout) :: all_pass
    integer(ip) :: nkx
    type(band_ops_t) :: bo                 ! y-operator band structure (host+device)
    real(dp), allocatable :: kxh(:), kzh(:)
    complex(dp), allocatable :: e0(:,:,:), v0(:,:,:), eh(:,:,:), vh(:,:,:)
    type(nl_workspace) :: ws
    real(dp),    device, allocatable :: kx(:), kz(:)
    type(vsolve_ops_t) :: vo               ! local v-solve apparatus
    complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:), Nv_prev(:,:,:), Neta_prev(:,:,:)
    integer(ip) :: it, istat, kk, ix, iz
    real(dp)    :: en, vn, en0, vn0
    logical     :: bad

    nkx = nx/2 + 1
    allocate(kxh(nkx), kzh(nz))
    do ix = 1, nkx; kxh(ix) = real(ix-1, dp); end do          ! non-negative half
    do iz = 1, nz; kk = iz-1; if (kk > nz/2) kk = kk - nz; kzh(iz) = real(kk, dp); end do
    call band_ops_alloc(bo, n)
    call helm_band_setup(n, bo%sdh, bo%sth, bo%bdh, bo%boff)

    allocate(e0(0:n,nkx,nz), v0(0:n,nkx,nz), eh(0:n,nkx,nz), vh(0:n,nkx,nz))
    call build_init(n, nkx, nz, A, e0, v0)

    call nl_setup(ws, n+1, nx, nz, kxh, kzh)          ! physical nx, kxh nkx-long
    allocate(kx(nkx), kz(nz))
    call vsolve_ops_alloc(vo, n, nkx, nz)
    allocate(eta(0:n,nkx,nz), v(0:n,nkx,nz), u0(0:n), w0(0:n), Nv_prev(0:n,nkx,nz), Neta_prev(0:n,nkx,nz))
    kx = kxh; kz = kzh
    bo%sd = bo%sdh; bo%st = bo%sth; bo%bd = bo%bdh
    call stepper_ev_setup(n, nkx, nz, dt, Re, kxh, kzh, vo%vm, vo%vp, vo%Cinv)

    u0 = (0.0_dp,0.0_dp); w0 = (0.0_dp,0.0_dp)
    Nv_prev = (0.0_dp,0.0_dp); Neta_prev = (0.0_dp,0.0_dp)
    eta = e0; v = v0

    eh = e0; vh = v0
    en0 = maxval(abs(eh)); vn0 = maxval(abs(vh))
    write(*,'(/,a,i0,a,i0,a,i0,a,i0,a,es8.1,a,es8.1,a,es8.1)') &
         '--- n=',n,' nx=',nx,' (nkx=',nkx,') nz=',nz,'  Re=',Re,'  A=',A,'  dt=',dt
    write(*,'(a,es12.4,a,es12.4)') '   step  0:  max|eta|=', en0, '   max|v|=', vn0

    bad = .false.
    do it = 1, Ns
       call step_ev(ws, n, nkx, nz, dt, Re, kx, kz, bo, vo, &
                    u0, w0, eta, v, Nv_prev, Neta_prev)
       istat = cudaDeviceSynchronize()
       eh = eta; vh = v
       en = maxval(abs(eh)); vn = maxval(abs(vh))
       if (.not. (en == en) .or. .not. (vn == vn)) bad = .true.
       if (en > huge(1.0_dp) .or. vn > huge(1.0_dp)) bad = .true.
       if (bad) then
          write(*,'(a,i3,a)') '   step ', it, ':  (non-finite)'
          exit
       end if
       write(*,'(a,i3,a,es12.4,a,es12.4)') '   step ', it, ':  max|eta|=', en, '   max|v|=', vn
    end do

    if (bad) then
       write(*,'(a)') '   --> NON-FINITE (blew up)'
       if (assert_finite) then
          write(*,'(a)') '   [FAIL] structural sanity: tiny-amplitude run must stay finite'
          all_pass = .false.
       else
          write(*,'(a)') '   [note] informational: explicit instability at these parameters'
       end if
    else
       write(*,'(a)') '   --> FINITE'
       if (assert_finite) write(*,'(a)') '   [PASS] structural sanity: wiring is sound (finite evolution)'
    end if

    call vsolve_ops_free(vo);  call band_ops_free(bo)
    deallocate(kx, kz, eta, v, u0, w0, Nv_prev, Neta_prev)
    call nl_destroy(ws)
    deallocate(kxh, kzh, e0, v0, eh, vh)
  end subroutine probe

  subroutine build_init(n, nkx, nz, A, e0, v0)
    integer(ip), intent(in)  :: n, nkx, nz
    real(dp),    intent(in)  :: A
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
    e0(:,1,1) = (0.0_dp,0.0_dp); v0(:,1,1) = (0.0_dp,0.0_dp)
    e0(:,nkx,:) = (0.0_dp,0.0_dp); v0(:,nkx,:) = (0.0_dp,0.0_dp)
    e0(:,:,nz/2+1) = (0.0_dp,0.0_dp); v0(:,:,nz/2+1) = (0.0_dp,0.0_dp)
  end subroutine build_init

end program test_stepper_full_r2c
