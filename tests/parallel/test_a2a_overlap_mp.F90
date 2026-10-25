!===============================================================================
! test_a2a_overlap_mp.F90 -- gate for the per-field PIPELINED overlap
! path of the nf-batched distributed transforms: par_to_physical_nf /
! par_to_spectral_nf with a2a_overlap=1 (comm-stream chunked all-to-all,
! event-ordered against the compute stream) must be BIT-EXACT against the
! monolithic a2a_overlap=0 path. The claim is structural: the pipelined branch
! issues the IDENTICAL per-field FFT/pad/trunc calls (same plans, same order,
! same arguments) and only the transpose -- pure data movement -- is chunked
! per field and moved to par_comm_stream; cuFFT with the same plan + same
! input + same GPU is run-to-run deterministic, so any nonzero difference is
! an ordering/race bug in the event pipeline, not "noise". Tolerance is
! therefore EXACTLY zero (plus NaN guards), the strongest gate we can state.
!
! Exercised at the production batch sizes (nf=6 inverse, nf=3 forward), at
! every harness rank count (P=1 collapses both paths to the same monolithic
! code -- trivially equal -- while P=4/8 exercise the real pipeline), dp+sp.
! Three overlapped cases: (a) each direction alone vs monolithic; (b) a
! REPEATED overlapped leg (module-state/event reset across legs); (c) the
! PRODUCTION PATTERN -- inverse(nf=6) immediately followed by forward(nf=3)
! with NO intervening sync, so two legs with DIFFERENT slot geometries are
! genuinely in flight together: the real cross-leg WAR gate for the
! event-transitivity argument in par_transpose_mod's header.
!===============================================================================
program test_a2a_overlap_mp
  use kinds,              only: tp, ip
  use cudafor
  use decomp_mod,         only: decomp_t, decomp_setup
  use par_spectral3d_mod, only: par_s3d_plans, par_s3d_setup_da_r2c, par_s3d_destroy_da_r2c, &
                                par_to_physical_nf, par_to_spectral_nf, par_s3d_set_overlap
  use par_mod
  implicit none

  integer(ip), parameter :: nx = 32, ny = 6, nz = 16
  integer(ip), parameter :: nxd = 3*nx/2, nzd = 3*nz/2, nkx = nx/2+1, nkxd = nxd/2+1
  integer(ip), parameter :: nfi = 6, nff = 3        ! production batch sizes
  logical :: all_ok, ok_inv, ok_fwd, ok_rep, ok_seq
  integer  :: istat
  integer(ip) :: iy, jx, kz, f, bnkx, anz

  type(decomp_t)      :: d
  type(par_s3d_plans) :: pl

  complex(tp), allocatable :: A6_h(:,:,:,:)
  real(tp),    allocatable :: P6_m_h(:,:,:,:), P6_o_h(:,:,:,:), P6_r_h(:,:,:,:)
  complex(tp), allocatable, device :: A6_d(:,:,:,:)
  real(tp),    allocatable, device :: P6_d(:,:,:,:)
  real(tp),    allocatable :: P3_h(:,:,:,:)
  complex(tp), allocatable :: A3_m_h(:,:,:,:), A3_o_h(:,:,:,:)
  real(tp),    allocatable, device :: P3_d(:,:,:,:)
  complex(tp), allocatable, device :: A3_d(:,:,:,:)
  complex(tp), allocatable, device :: BzI(:,:,:,:), AzI(:,:,:,:), AfI(:,:,:)
  complex(tp), allocatable, device :: BzF(:,:,:,:), AzF(:,:,:,:), AfF(:,:,:)

  call par_init()
  if (par_rank == 0) print '(a,i0,a)', '[a2a-overlap-mp] world size = ', par_nproc, ' rank(s)'

  call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
  bnkx = d%b_nkx_loc;  anz = d%ad_nz_loc
  call par_s3d_setup_da_r2c(pl, d, ny, nx, nz)

  allocate(A6_h(ny,bnkx,nz,nfi))
  allocate(P6_m_h(ny,nxd,anz,nfi), P6_o_h(ny,nxd,anz,nfi), P6_r_h(ny,nxd,anz,nfi))
  allocate(A6_d(ny,bnkx,nz,nfi), P6_d(ny,nxd,anz,nfi))
  allocate(P3_h(ny,nxd,anz,nff), A3_m_h(ny,bnkx,nz,nff), A3_o_h(ny,bnkx,nz,nff))
  allocate(P3_d(ny,nxd,anz,nff), A3_d(ny,bnkx,nz,nff))
  allocate(BzI(ny,bnkx,nzd,nfi), AzI(ny,nkx,anz,nfi), AfI(ny,nkxd,anz))
  allocate(BzF(ny,bnkx,nzd,nff), AzF(ny,nkx,anz,nff), AfF(ny,nkxd,anz))

  ! deterministic per-rank spectral fields (inverse input; consumed by cheb)
  do f = 1, nfi; do kz = 1, nz; do jx = 1, bnkx; do iy = 1, ny
     A6_h(iy,jx,kz,f) = cmplx( hr(iy,jx,kz,f), hi(iy,jx,kz,f), tp )
  end do; end do; end do; end do

  ! ===== INVERSE nf=6: monolithic vs pipelined, BIT-EXACT =====================
  call par_s3d_set_overlap(0_ip)
  A6_d = A6_h
  call par_to_physical_nf(pl, d, ny, nx, nz, nxd, nzd, bnkx, anz, nfi, &
                          A6_d, BzI, AzI, AfI, P6_d)
  istat = cudaDeviceSynchronize();  P6_m_h = P6_d

  call par_s3d_set_overlap(1_ip)
  A6_d = A6_h
  call par_to_physical_nf(pl, d, ny, nx, nz, nxd, nzd, bnkx, anz, nfi, &
                          A6_d, BzI, AzI, AfI, P6_d)
  istat = cudaDeviceSynchronize();  P6_o_h = P6_d
  ok_inv = bitsame_r(P6_o_h, P6_m_h)

  ! back-to-back overlapped leg: exercises slot reuse across legs (WAR gate)
  A6_d = A6_h
  call par_to_physical_nf(pl, d, ny, nx, nz, nxd, nzd, bnkx, anz, nfi, &
                          A6_d, BzI, AzI, AfI, P6_d)
  istat = cudaDeviceSynchronize();  P6_r_h = P6_d
  ok_rep = bitsame_r(P6_r_h, P6_m_h)

  ! ===== FORWARD nf=3: monolithic vs pipelined, BIT-EXACT =====================
  do f = 1, nff; do kz = 1, anz; do jx = 1, nxd; do iy = 1, ny
     P3_h(iy,jx,kz,f) = hr(iy,jx,kz,f)
  end do; end do; end do; end do
  P3_d = P3_h                                        ! forward does not consume P

  call par_s3d_set_overlap(0_ip)
  call par_to_spectral_nf(pl, d, ny, nx, nz, nxd, nzd, bnkx, anz, nff, &
                          P3_d, AfF, AzF, BzF, A3_d)
  istat = cudaDeviceSynchronize();  A3_m_h = A3_d

  call par_s3d_set_overlap(1_ip)
  call par_to_spectral_nf(pl, d, ny, nx, nz, nxd, nzd, bnkx, anz, nff, &
                          P3_d, AfF, AzF, BzF, A3_d)
  istat = cudaDeviceSynchronize();  A3_o_h = A3_d
  ok_fwd = bitsame_c(A3_o_h, A3_m_h)

  ! ===== PRODUCTION PATTERN: inverse(nf=6) -> forward(nf=3), overlap ON, =====
  ! NO sync between the legs -- two pipelined legs with DIFFERENT slot
  ! geometries genuinely in flight together (the real cross-leg WAR case;
  ! mirrors the nonlinear eval's transpose sequence). Both outputs must
  ! still be bit-exact vs the monolithic references.
  A6_d = A6_h
  call par_to_physical_nf(pl, d, ny, nx, nz, nxd, nzd, bnkx, anz, nfi, &
                          A6_d, BzI, AzI, AfI, P6_d)
  call par_to_spectral_nf(pl, d, ny, nx, nz, nxd, nzd, bnkx, anz, nff, &
                          P3_d, AfF, AzF, BzF, A3_d)
  istat = cudaDeviceSynchronize()
  P6_r_h = P6_d;  A3_o_h = A3_d
  ok_seq = bitsame_r(P6_r_h, P6_m_h) .and. bitsame_c(A3_o_h, A3_m_h)

  call par_s3d_set_overlap(0_ip)                     ! module-state hygiene
  call par_s3d_destroy_da_r2c(pl)

  call par_reduce_and(ok_inv)
  call par_reduce_and(ok_fwd)
  call par_reduce_and(ok_rep)
  call par_reduce_and(ok_seq)
  all_ok = ok_inv .and. ok_fwd .and. ok_rep .and. ok_seq
  if (par_rank == 0) then
     call report('inverse nf=6: pipelined overlap BIT-EXACT vs monolithic', ok_inv)
     call report('forward nf=3: pipelined overlap BIT-EXACT vs monolithic', ok_fwd)
     call report('repeated overlapped leg BIT-EXACT (state/event reset)',   ok_rep)
     call report('inverse->forward back-to-back, no sync (cross-leg WAR)',  ok_seq)
     if (all_ok) then
        print '(a)', 'ALL TESTS PASSED'
     else
        print '(a)', 'SOME TESTS FAILED'
     end if
  end if

  call par_finalize()
  if (.not. all_ok) error stop 1

contains

  real(tp) function hr(a, b, c, e) result(v)        ! deterministic real in [-0.5,0.5)
    integer(ip), intent(in) :: a, b, c, e
    integer(ip) :: h
    h = modulo( a*1009_ip + b*2027_ip + c*3001_ip + e*4013_ip + b*c*17_ip, 100003_ip )
    v = real(h, tp)/100003.0_tp - 0.5_tp
  end function hr

  real(tp) function hi(a, b, c, e) result(v)        ! a different stream for imag parts
    integer(ip), intent(in) :: a, b, c, e
    integer(ip) :: h
    h = modulo( a*733_ip + b*1597_ip + c*2741_ip + e*5051_ip + a*c*29_ip, 100019_ip )
    v = real(h, tp)/100019.0_tp - 0.5_tp
  end function hi

  logical function bitsame_r(a, b) result(ok)        ! NaN-guarded EXACT equality
    real(tp), intent(in) :: a(:,:,:,:), b(:,:,:,:)
    ok = all(a == a) .and. all(b == b) .and. all(a == b)
  end function bitsame_r

  logical function bitsame_c(a, b) result(ok)
    complex(tp), intent(in) :: a(:,:,:,:), b(:,:,:,:)
    ok = all(a == a) .and. all(b == b) .and. all(a == b)
  end function bitsame_c

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

end program test_a2a_overlap_mp
