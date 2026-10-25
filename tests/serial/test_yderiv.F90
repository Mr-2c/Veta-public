!===============================================================================
! test_yderiv.F90 -- gate for the transform-tier D_y dispatch
! seam (yderiv_gpu_mod) and the yops facade's FD branches:
!   (1,2) CHEB BIT-IDENTITY: y_deriv_tp / y_deriv_c with the flag down vs the
!         direct cheb_deriv_c_tp / cheb_deriv_c calls -- equal to the BIT
!         (same routine, same arguments; this is what keeps every Chebyshev
!         run's trajectory untouched by this round).
!   (3,4) FD vs HOST ORACLE: the device banded D1 matvec vs fd_apply_band on
!         the SAME band, from the SAME kind-quantized inputs. Both accumulate
!         in dp with the identical fixed j-ascending order, so the only
!         difference is the device's single final store rounding to the field
!         kind: gate at 50*eps(kind) max-norm relative (~1.1e-14 for real64
!         kinds, ~6e-6 at TP=sp). This is a SEMANTICS gate (band indexing,
!         layout, batching, re/im packing -- any such bug lands orders of
!         magnitude above it); it can NOT resolve dp-vs-tp accumulation on
!         rough data (a simulated tp-native accumulation errs
!         4.4e-5 abs, under this gate's 7.0e-3, because the hash columns'
!         D1 magnitudes ~1e3 dwarf the 9-term rounding). Stencil ACCURACY is
!         test_fdops' job.
!   (4b)  LINEAR ACCUMULATION PROBE (the dp-accumulation POLICY gate): D1 of
!         f(y) = y is exactly 1 at every row, but the wall rows build that 1
!         from +-1.5e3 weights with ||row||_1 ~ 8e3 -- a cancellation ratio
!         the hash rows lack. Against the quantized-input oracle the correct
!         dp-accumulate/tp-store path differs by <= 0.5 ulp_tp (measured
!         6e-8), while a tp-native accumulation errs ~1.3e-5 (~100 ulp):
!         gate 10*eps(tp), FLOORED at 1e-11 abs (at tp==dp the probe has
!         nothing to discriminate and NOFMA=0 FMA contraction wobbles
!         device-vs-host ~2e-12 on these cancellative rows), catches the
!         policy regression with ~10x on each side.
!   (5)   FD EXACTNESS through the device path: fd_build_bands rows are
!         degree-(#stencil-1) exact (wall row 1 has kl+1=5 nodes), so a
!         degree-4 polynomial differentiates EXACTLY at every row; only
!         rounding remains (wall weights ~1/h_wall ~ 8e2 at GLC-64 amplify
!         eps_dp to ~1e-12 abs) -- rel 1e-10.
!   (6)   RESET HYGIENE: yderiv_set_cheb restores the bit-identical Cheb path
!         (guards test-order coupling; production never resets).
!   (7,8) YOPS FD BRANCHES: yop_wall_deriv vs the analytic derivative (the
!         full-order 9-node one-sided rows are degree-8 exact: error ~ 1e-12
!         for a smooth transcendental at GLC-64 -- gate 1e-9) and
!         yop_mean_flux vs the analytic integral with trapezoid weights
!         (nonuniform-trapz error ~ sum h^3 |f''|/12 ~ 5e-5 abs at GLC-64 for
!         the f below -- gate rel 1e-3).
!   (9)   IDENTITY TAGS flip with the backend and reset.
!   (10) GALILEAN MEAN SHIFT is basis-aware: Cheb moves exactly the T_0
!         coeff (rows 2..ny bit-untouched); FD moves EVERY point value by -U_c
!         (bit-exact single subtraction vs the host). The pre-X1 bug shifted
!         row 1 in both bases = wall-node-only in FD: the advecting frame was
!         u - U_c at y=+1 but lab-u at all interior nodes.
! Grid GLC(64) DESCENDING (+1..-1, exact endpoints), kl=4, batch=5x3 modes.
!===============================================================================
program test_yderiv
  use kinds,          only: tp, wp, dp, ip
  use cudafor
  use cheb_shen,      only: glc_nodes
  use cheb_ops_mod,   only: cheb_deriv_c, cheb_deriv_c_tp
  use fdops_mod,      only: fd_build_bands, fd_apply_band, fd_wall_weights
  use yderiv_gpu_mod, only: yderiv_set_fd, yderiv_set_cheb, y_deriv_tp, y_deriv_c, &
                            gal_shift_mean
  use yops_mod,       only: yop_wall_deriv, yop_mean_flux, yops_set_fd, &
                            yops_set_cheb, yop_backend_tag, yop_ygrid_tag
  implicit none
  integer(ip), parameter :: n = 64, ny = n + 1, kl = 4, nx = 5, nz = 3, nb = nx*nz
  real(dp) :: yA(ny), wA(ny), d1A(2*kl+1,ny), d2A(2*kl+1,ny)
  real(dp) :: wloA(ny), whiA(ny)
  ! host mirrors (the tp/wp host arrays are the KIND-QUANTIZED oracle inputs)
  complex(tp) :: ah_tp(ny,nb), b1h_tp(ny,nb), b2h_tp(ny,nb)
  complex(wp) :: ah_wp(ny,nb), b1h_wp(ny,nb), b2h_wp(ny,nb)
  real(dp)    :: fr(ny), fi(ny), gr(ny), gi(ny), qv(ny)
  real(dp)    :: e, den, dlo, dhi, s, c1, c2
  ! device state
  complex(tp), device :: atp_d(ny,nb), btp1_d(ny,nb), btp2_d(ny,nb)
  complex(wp), device :: awp_d(ny,nb), bwp1_d(ny,nb), bwp2_d(ny,nb)
  logical :: allok, oktag
  integer(ip) :: j, c, istat
  allok = .true.

  call glc_nodes(n, yA)
  yA(1) = 1.0_dp;  yA(ny) = -1.0_dp
  wA(1) = 0.5_dp*(yA(1) - yA(2))
  do j = 2, ny-1
     wA(j) = 0.5_dp*(yA(j-1) - yA(j+1))
  end do
  wA(ny) = 0.5_dp*(yA(ny-1) - yA(ny))
  call fd_build_bands(ny, kl, yA, d1A, d2A)
  call fd_wall_weights(ny, kl, yA, wloA, whiA)

  ! deterministic inputs, quantized at the host assignment to each kind
  do c = 1, nb
     do j = 1, ny
        ah_tp(j,c) = cmplx(hv(j,c),  -0.7_dp*hv(j,c+3), tp)
        ah_wp(j,c) = cmplx(hv(j,c+5), 0.4_dp*hv(j,c+9), wp)
     end do
  end do
  atp_d = ah_tp;  awp_d = ah_wp

  ! ===== (1,2) CHEB BIT-IDENTITY of the dispatch (flag down = default) ========
  call y_deriv_tp(ny, nb, atp_d, btp1_d)
  call cheb_deriv_c_tp(ny, nb, atp_d, btp2_d)
  call y_deriv_c(ny, nb, awp_d, bwp1_d)
  call cheb_deriv_c(ny, nb, awp_d, bwp2_d)
  istat = cudaDeviceSynchronize()
  b1h_tp = btp1_d;  b2h_tp = btp2_d
  b1h_wp = bwp1_d;  b2h_wp = bwp2_d
  call report('cheb: y_deriv_tp BIT-IDENTICAL to cheb_deriv_c_tp', &
              all(b1h_tp == b2h_tp))
  call report('cheb: y_deriv_c  BIT-IDENTICAL to cheb_deriv_c', &
              all(b1h_wp == b2h_wp))

  ! ===== (3,4) FD device matvec vs fd_apply_band host oracle ==================
  call yderiv_set_fd(ny, kl, d1A)
  call y_deriv_tp(ny, nb, atp_d, btp1_d)
  call y_deriv_c(ny, nb, awp_d, bwp1_d)
  istat = cudaDeviceSynchronize()
  b1h_tp = btp1_d;  b1h_wp = bwp1_d

  e = 0.0_dp;  den = 0.0_dp
  do c = 1, nb
     do j = 1, ny
        fr(j) = real(ah_tp(j,c), dp);  fi(j) = real(aimag(ah_tp(j,c)), dp)
     end do
     call fd_apply_band(ny, kl, d1A, fr, gr)
     call fd_apply_band(ny, kl, d1A, fi, gi)
     do j = 1, ny
        den = max(den, abs(gr(j)), abs(gi(j)))
        e   = max(e, abs(real(b1h_tp(j,c), dp) - gr(j)), &
                     abs(real(aimag(b1h_tp(j,c)), dp) - gi(j)))
     end do
  end do
  call report('fd: y_deriv_tp == host banded oracle (50 ulp_tp rel)', &
              e <= 50.0_dp*real(epsilon(1.0_tp), dp)*max(1.0_dp, den))

  e = 0.0_dp;  den = 0.0_dp
  do c = 1, nb
     do j = 1, ny
        fr(j) = real(ah_wp(j,c), dp);  fi(j) = real(aimag(ah_wp(j,c)), dp)
     end do
     call fd_apply_band(ny, kl, d1A, fr, gr)
     call fd_apply_band(ny, kl, d1A, fi, gi)
     do j = 1, ny
        den = max(den, abs(gr(j)), abs(gi(j)))
        e   = max(e, abs(real(b1h_wp(j,c), dp) - gr(j)), &
                     abs(real(aimag(b1h_wp(j,c)), dp) - gi(j)))
     end do
  end do
  call report('fd: y_deriv_c == host banded oracle (50 ulp_wp rel)', &
              e <= 50.0_dp*real(epsilon(1.0_wp), dp)*max(1.0_dp, den))

  ! ===== (5) FD quartic exactness through the device path (wp) ================
  ! f = s*(y^4 - 0.3y^3 + 0.2y^2 - y + 0.7), g = s*(0.5y^4 + y^3 - 0.4y^2 + 0.1y)
  ! per column (s = 1 + 0.1c keeps columns distinct); every band row is at least
  ! degree-4 exact, so D1 reproduces f',g' to rounding.
  do c = 1, nb
     s = 1.0_dp + 0.1_dp*real(c, dp)
     do j = 1, ny
        ah_wp(j,c) = cmplx(s*(((yA(j) - 0.3_dp)*yA(j) + 0.2_dp)*yA(j)*yA(j) &
                              - yA(j) + 0.7_dp), &
                           s*((((0.5_dp*yA(j) + 1.0_dp)*yA(j) - 0.4_dp)*yA(j) &
                              + 0.1_dp)*yA(j)), wp)
     end do
  end do
  awp_d = ah_wp
  call y_deriv_c(ny, nb, awp_d, bwp1_d)
  istat = cudaDeviceSynchronize()
  b1h_wp = bwp1_d
  e = 0.0_dp;  den = 0.0_dp
  do c = 1, nb
     s = 1.0_dp + 0.1_dp*real(c, dp)
     do j = 1, ny
        c1 = s*(((4.0_dp*yA(j) - 0.9_dp)*yA(j) + 0.4_dp)*yA(j) - 1.0_dp)
        c2 = s*(((2.0_dp*yA(j) + 3.0_dp)*yA(j) - 0.8_dp)*yA(j) + 0.1_dp)
        den = max(den, abs(c1), abs(c2))
        e   = max(e, abs(real(b1h_wp(j,c), dp) - c1), &
                     abs(real(aimag(b1h_wp(j,c)), dp) - c2))
     end do
  end do
  call report('fd: quartic differentiates exactly on device (rel 1e-10)', &
              e <= 1.0e-10_dp*max(1.0_dp, den))

  ! ===== (4b) linear accumulation probe (see header) ===========================
  do c = 1, nb
     do j = 1, ny
        ah_tp(j,c) = cmplx(yA(j), -0.5_dp*yA(j), tp)
     end do
  end do
  atp_d = ah_tp
  call y_deriv_tp(ny, nb, atp_d, btp1_d)
  istat = cudaDeviceSynchronize()
  b1h_tp = btp1_d
  e = 0.0_dp
  do c = 1, nb
     do j = 1, ny
        fr(j) = real(ah_tp(j,c), dp);  fi(j) = real(aimag(ah_tp(j,c)), dp)
     end do
     call fd_apply_band(ny, kl, d1A, fr, gr)
     call fd_apply_band(ny, kl, d1A, fi, gi)
     do j = 1, ny
        e = max(e, abs(real(b1h_tp(j,c), dp) - gr(j)), &
                   abs(real(aimag(b1h_tp(j,c)), dp) - gi(j)))
     end do
  end do
  ! gate floored at 1e-11 abs: at TP=real64 (tp==dp) the probe has nothing to
  ! discriminate (tp-native IS dp) and a NOFMA=0 build's FMA contraction on the
  ! cancellative wall rows wobbles device-vs-host by ~2e-12; the 10 ulp_tp term
  ! (1.2e-6) does the real work at TP=sp, far above the floor.
  call report('fd: linear probe -- dp accumulation policy (10 ulp_tp)', &
              e <= max(10.0_dp*real(epsilon(1.0_tp), dp), 1.0e-11_dp))

  ! ===== (6) reset hygiene: back to the bit-identical Cheb branch =============
  call yderiv_set_cheb()
  atp_d = ah_tp
  call y_deriv_tp(ny, nb, atp_d, btp1_d)
  call cheb_deriv_c_tp(ny, nb, atp_d, btp2_d)
  istat = cudaDeviceSynchronize()
  b1h_tp = btp1_d;  b2h_tp = btp2_d
  call report('cheb: yderiv_set_cheb restores the BIT-IDENTICAL path', &
              all(b1h_tp == b2h_tp))

  ! ===== (7,8) yops facade FD branches =========================================
  call yops_set_fd(n, wloA, whiA, wA)
  do j = 1, ny
     qv(j) = sin(2.0_dp*yA(j)) + 0.3_dp*cos(3.0_dp*yA(j))
  end do
  call yop_wall_deriv(n, qv, dlo, dhi)
  c1 = 2.0_dp*cos(-2.0_dp) - 0.9_dp*sin(-3.0_dp)     ! f'(-1)
  c2 = 2.0_dp*cos( 2.0_dp) - 0.9_dp*sin( 3.0_dp)     ! f'(+1)
  call report('yops fd: yop_wall_deriv == analytic walls (1e-9)', &
              abs(dlo - c1) <= 1.0e-9_dp .and. abs(dhi - c2) <= 1.0e-9_dp)

  do j = 1, ny
     qv(j) = cos(yA(j)) + 0.5_dp*yA(j)*yA(j)
  end do
  s  = yop_mean_flux(n, qv)
  c1 = 2.0_dp*sin(1.0_dp) + 1.0_dp/3.0_dp            ! int_{-1}^{1} f dy
  call report('yops fd: yop_mean_flux == analytic integral (trapz, rel 1e-3)', &
              abs(s - c1) <= 1.0e-3_dp*abs(c1))

  ! ===== (9) identity tags flip and reset ======================================
  oktag = (yop_backend_tag() == 'fd-point') .and. (yop_ygrid_tag() == 'file')
  call yops_set_cheb()
  oktag = oktag .and. (yop_backend_tag() == 'cheb-T') &
                .and. (yop_ygrid_tag()   == 'glc')
  call report('yops: tags fd-point/file under FD, cheb-T/glc after reset', oktag)

  ! ===== (10) Galilean mean shift is basis-aware ==========================
  ! One subtraction per touched entry, tp on both sides -> the comparison is to
  ! the BIT (identical IEEE op host and device; imag parts subtract 0 exactly).
  do c = 1, nb
     do j = 1, ny
        ah_tp(j,c) = cmplx(hv(j,c+2), 0.3_dp*hv(j,c+7), tp)
     end do
  end do
  ! Cheb (flag is down after (6); (9) reset yops only): exactly row 1 moves
  atp_d = ah_tp
  call gal_shift_mean(ny, 0.8_wp, atp_d(:,1))
  istat = cudaDeviceSynchronize()
  b1h_tp = atp_d
  oktag = (b1h_tp(1,1) == ah_tp(1,1) - cmplx(0.8_wp, 0.0_wp, tp)) &
          .and. all(b1h_tp(2:ny,1) == ah_tp(2:ny,1))
  call report('gal: cheb shift moves the T_0 coeff ONLY (bit)', oktag)

  ! FD: every node moves by -U_c (the pre-X1 bug left rows 2..ny untouched)
  call yderiv_set_fd(ny, kl, d1A)
  atp_d = ah_tp
  call gal_shift_mean(ny, 0.8_wp, atp_d(:,1))
  istat = cudaDeviceSynchronize()
  b1h_tp = atp_d
  oktag = .true.
  do j = 1, ny
     oktag = oktag .and. (b1h_tp(j,1) == ah_tp(j,1) - cmplx(0.8_wp, 0.0_wp, tp))
  end do
  call report('gal: fd shift moves EVERY point value by -U_c (bit)', oktag)
  call yderiv_set_cheb()                        ! leave the module in its default

  if (allok) then; print '(a)', 'ALL TESTS PASSED'
  else;            print '(a)', 'SOME TESTS FAILED'; end if
  if (.not. allok) error stop 1

contains

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

end program test_yderiv
