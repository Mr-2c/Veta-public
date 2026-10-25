!===============================================================================
! test_fd_mean_gpu.F90 -- gate for the ON-DEVICE FD mean
! (fd_mean_gpu_mod) and the y_mean_stage / y_mean_stage_cfr dispatch seam:
!   (1) CHEB BIT-IDENTITY: the helpers with ybasis=YB_CHEB vs the direct
!       mean_stage_gpu / mean_stage_cfr_gpu calls (the verbatim moved
!       statements), from identical device state -- equal to the BIT.
!   (2) FD DEVICE vs FD HOST: the composed-from-primitives device stages vs
!       the fd_mean_mod host oracle (real channel; the device runs complex
!       with a nonzero imaginary forcing DELIBERATELY -- the real-coefficient
!       solve decouples channels and q0 is forced exactly real, so the real
!       result must match the host given real(N); rel 1e-12).
!   (3) FD CFR on device: flux constraint == H*Q to rounding on the
!       downloaded profile; beta matches the host oracle.
! Grid GLC(128) DESCENDING + trapz weights (the test_fd_mean regime), kl=4.
!===============================================================================
program test_fd_mean_gpu
  use kinds,         only: dp, ip
  use cudafor
  use cheb_shen,     only: glc_nodes
  use helm_band_mod, only: helm_band_setup
  use mean_mod,      only: mean_flux_setup, H_CH
  use mean_gpu_mod,  only: mean_gpu_ws, mean_gpu_alloc, mean_gpu_free, &
                           mean_stage_gpu, mean_stage_cfr_gpu
  use fdops_mod,     only: fd_build_bands
  use fd_mean_mod,   only: fd_mean_flux, fd_step_mean_lin, fd_mean_flux_setup, &
                           fd_mean_stage_cfr
  use solver_ops_mod, only: band_ops_t, band_ops_alloc, band_ops_free, &
                            band_ops_fd_init, y_mean_stage, y_mean_stage_cfr
  implicit none
  integer(ip), parameter :: n = 128, ny = n + 1, kl = 4, NCFR = 9
  real(dp), parameter :: DT = 0.01_dp, RE = 100.0_dp, BETA = 2.0_dp, QT = 1.0_dp
  real(dp) :: yA(ny), wA(ny), d1A(2*kl+1,ny), d2A(2*kl+1,ny)
  real(dp) :: UhC(0:n,3), IhC(3), UhA(ny,3), IhA(3)
  real(dp) :: qh(ny), Nqr(ny), Nqpr(ny), betaH, betaD(1), fx, e, w, eb
  complex(dp) :: q0h(0:n), Nqh(0:n), Nqph(0:n), qtmp(0:n)
  ! device state
  complex(dp), device :: q0_d(0:n), q02_d(0:n), Nq_d(0:n), Nqp_d(0:n)
  real(dp),    device :: beta_d(1), beta2_d(1)
  type(mean_gpu_ws) :: mgc, mgc2, mgf
  type(band_ops_t)  :: boc, bofd
  logical :: ok, okf, allok
  integer(ip) :: j, ks, step, istat
  allok = .true.

  call glc_nodes(n, yA)
  yA(1) = 1.0_dp;  yA(ny) = -1.0_dp
  wA(1) = 0.5_dp*(yA(1) - yA(2))
  do j = 2, ny-1
     wA(j) = 0.5_dp*(yA(j-1) - yA(j+1))
  end do
  wA(ny) = 0.5_dp*(yA(ny-1) - yA(ny))
  call fd_build_bands(ny, kl, yA, d1A, d2A)

  call band_ops_alloc(boc, n)
  call helm_band_setup(n, boc%sdh, boc%sth, boc%bdh, boc%boff)
  boc%sd = boc%sdh;  boc%st = boc%sth;  boc%bd = boc%bdh
  call band_ops_fd_init(bofd, n, kl, yA, wA, .true.)

  call mean_gpu_alloc(mgc, n);  call mean_gpu_alloc(mgc2, n);  call mean_gpu_alloc(mgf, n)
  call mean_flux_setup(n, DT, RE, boc%sdh, boc%sth, boc%bdh, boc%boff, UhC, IhC)
  call fd_mean_flux_setup(ny, kl, d2A, wA, DT, RE, UhA, IhA, okf)
  mgc%Uh_d = UhC;  mgc2%Uh_d = UhC;  mgf%Uh_d(0:n,:) = UhA(1:ny,:)

  ! deterministic complex forcings (im nonzero DELIBERATELY, discarded by design)
  do j = 0, n
     Nqh(j)  = cmplx(hv(j,1), 0.3_dp*hv(j,2), dp)
     Nqph(j) = cmplx(hv(j,3), -0.2_dp*hv(j,4), dp)
  end do
  Nq_d = Nqh;  Nqp_d = Nqph
  do j = 1, ny
     Nqr(j) = real(Nqh(j-1));  Nqpr(j) = real(Nqph(j-1))
  end do

  ! ===== (1) CHEB BIT-IDENTITY of the mean dispatch ===========================
  do j = 0, n
     q0h(j) = cmplx(hv(j,5), 0.1_dp*hv(j,6), dp)
  end do
  q0_d = q0h;  q02_d = q0h
  do ks = 1, 3
     call y_mean_stage(boc, mgc, n, ks, DT, RE, BETA, q0_d, Nq_d, Nqp_d)
     call mean_stage_gpu(mgc2, n, ks, DT, RE, BETA, boc%sd, boc%st, boc%bd, &
                         boc%boff, q02_d, Nq_d, Nqp_d)
  end do
  istat = cudaDeviceSynchronize()
  qtmp = q0_d;  ok = all(qtmp == qtmp)
  q0h  = q02_d
  call report('cheb: y_mean_stage BIT-IDENTICAL to mean_stage_gpu', &
              ok .and. all(qtmp == q0h))

  q0_d = (0.0_dp, 0.0_dp);  q02_d = (0.0_dp, 0.0_dp)
  beta_d = 0.0_dp;  beta2_d = 0.0_dp
  do ks = 1, 3
     call y_mean_stage_cfr(boc, mgc, n, ks, DT, RE, QT, mgc%Uh_d(:,ks), IhC(ks), &
                           q0_d, Nq_d, Nqp_d, beta_d)
     call mean_stage_cfr_gpu(mgc2, n, ks, DT, RE, QT, boc%sd, boc%st, boc%bd, &
                             boc%boff, mgc2%Uh_d(:,ks), IhC(ks), q02_d, Nq_d, &
                             Nqp_d, beta2_d)
  end do
  istat = cudaDeviceSynchronize()
  qtmp = q0_d;  q0h = q02_d
  betaD = beta_d;  betaH = 0.0_dp
  fx = betaD(1);  betaD = beta2_d
  call report('cheb: y_mean_stage_cfr BIT-IDENTICAL to mean_stage_cfr_gpu', &
              all(qtmp == q0h) .and. fx == betaD(1))

  ! ===== (2) FD device lin stages vs the host oracle ==========================
  q0_d = (0.0_dp, 0.0_dp);  qh = 0.0_dp
  do step = 1, 5
     do ks = 1, 3
        call y_mean_stage(bofd, mgf, n, ks, DT, RE, BETA, q0_d, Nq_d, Nqp_d)
     end do
  end do
  ! host: same stages with the REAL parts of the forcings
  do step = 1, 5
     do ks = 1, 3
        call fd_stage_host(ks, BETA, qh, Nqr, Nqpr)
     end do
  end do
  istat = cudaDeviceSynchronize();  qtmp = q0_d
  e = 0.0_dp;  w = 0.0_dp
  do j = 1, ny
     w = max(w, abs(qh(j)))
     e = max(e, abs(real(qtmp(j-1)) - qh(j)))
     if (aimag(qtmp(j-1)) /= 0.0_dp) e = huge(1.0_dp)   ! forced-real contract
  end do
  call report('fd: device lin stages == host oracle, exactly-real (rel 1e-12)', &
              e <= 1.0e-12_dp*max(1.0_dp, w))

  ! ===== (3) FD device CFR vs host + flux constraint ==========================
  q0_d = (0.0_dp, 0.0_dp);  qh = 0.0_dp;  ok = .true.;  e = 0.0_dp;  eb = 0.0_dp
  do step = 1, NCFR
     ks = mod(step-1, 3) + 1
     call y_mean_stage_cfr(bofd, mgf, n, ks, DT, RE, QT, mgf%Uh_d(:,ks), IhA(ks), &
                           q0_d, Nq_d, Nqp_d, beta_d)
     call fd_mean_stage_cfr(ny, kl, d2A, wA, ks, DT, RE, QT, UhA(:,ks), IhA(ks), &
                            qh, Nqr, Nqpr, betaH, okf)
     ok = ok .and. okf
     istat = cudaDeviceSynchronize();  qtmp = q0_d;  betaD = beta_d
     do j = 1, ny
        e = max(e, abs(real(qtmp(j-1)) - qh(j)))
     end do
     eb = max(eb, abs(betaD(1) - betaH)/max(1.0_dp, abs(betaH)))
     ! flux constraint on the DOWNLOADED device profile
     fx = 0.0_dp
     do j = 1, ny
        fx = fx + wA(j)*real(qtmp(j-1))
     end do
     if (abs(fx - H_CH*QT) > 1.0e-12_dp*H_CH*QT) ok = .false.
  end do
  w = maxval(abs(qh))
  ! beta gated at the DEVICE posture 1e-11 (min|beta|=0.32 with a 376x
  ! 1/(ck dt Ih) amplification makes 1e-12 the file's thinnest margin, ~1.7x)
  call report('fd: device CFR profile == host oracle (rel 1e-12)', &
              ok .and. e <= 1.0e-12_dp*max(1.0_dp, w))
  call report('fd: device CFR beta == host oracle (rel 1e-11)', eb <= 1.0e-11_dp)
  call report('fd: device CFR flux == H*Q to rounding at every stage', ok)

  call mean_gpu_free(mgc);  call mean_gpu_free(mgc2);  call mean_gpu_free(mgf)
  call band_ops_free(boc);  call band_ops_free(bofd)
  if (allok) then; print '(a)', 'ALL TESTS PASSED'
  else;            print '(a)', 'SOME TESTS FAILED'; end if
  if (.not. allok) error stop 1

contains

  !-- host FD mean stage with explicit forcings (mirrors fd_mean_mod's stage) --
  subroutine fd_stage_host(ks, bet, q, Nr, Npr)
    use fd_mean_mod, only: fd_mean_stage
    integer(ip), intent(in)    :: ks
    real(dp),    intent(in)    :: bet
    real(dp),    intent(inout) :: q(ny)
    real(dp),    intent(in)    :: Nr(ny), Npr(ny)
    logical :: okk
    call fd_mean_stage(ny, kl, d2A, ks, DT, RE, bet, q, Nr, Npr, okk)
    if (.not. okk) allok = .false.
  end subroutine fd_stage_host

  real(dp) function hv(a, b) result(v)
    integer(ip), intent(in) :: a, b
    integer(ip) :: h
    h = modulo(a*1009_ip + b*2027_ip + a*b*17_ip, 100003_ip)
    v = real(h, dp)/100003.0_dp - 0.5_dp
  end function hv

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. okk) allok = .false.
  end subroutine report

end program test_fd_mean_gpu
