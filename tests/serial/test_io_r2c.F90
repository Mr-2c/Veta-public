!===============================================================================
! test_io_r2c.F90  --  [R2C] unit test for io_mod (checkpoint / restart, nkx payload).
! The R2C analogue of test_io. In R2C the spectral state is (0:n, nkx, nz) with
! nkx = nx/2+1; the checkpoint payload is nkx-wide and the v2 header carries
! `xtype = r2c` so an R2C build refuses a C2C file (and vice versa).
!
! Checks:
!  (A) Round-trip: write a known (0:n,nkx,nz) state, read into cleared arrays,
!      require BIT-IDENTICAL fields + exact scalars. io round-trips raw bytes, so
!      the kx=0 Hermitian structure is irrelevant here -- a deterministic fill is
!      enough (and reproducible).
!  (B) Grid mismatch refused: reading with the wrong nx (hence wrong nkx) / n / nz
!      must return ok=.false.
!  (C) v2 header self-documents R2C: the fixed ASCII header carries `xtype = r2c`
!      and the shape manifest `v.shape = ny,nkx,nz` (NOT ny,nx,nz). We read the
!      header bytes and assert both -- this is the C2C/R2C cross-read guard.
!
! NaN/Inf-hardened. Writes a scratch file ./chk_test_r2c.bin.
! Build/link (only needs io_mod):
!   nvfortran -c types.F90
!   nvfortran -cuda -c io_mod.F90
!   nvfortran -cuda -o test_io_r2c test_io_r2c.F90 types.o io_mod.o
!===============================================================================
program test_io_r2c
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,      only: wp, dp, ip
  use io_mod,     only: io_write_state, io_read_state, io_set_ybasis
  use cudafor
  implicit none

  integer(ip), parameter :: n = 16, nx = 8, nz = 6
  integer(ip), parameter :: nkx = nx/2 + 1                 ! = 5
  real(dp),    parameter :: Re = 10.0_dp, dt = 5.0e-3_dp, Qtar = 1.0_dp
  real(dp),    parameter :: tt = 1.25_dp
  integer(ip), parameter :: stp = 7
  integer(ip), parameter :: IO_HDR_BYTES = 1024            ! must match io_mod

  ! host known state + readback (nkx-wide)
  complex(dp) :: v0(0:n,nkx,nz), e0(0:n,nkx,nz), u00(0:n), w00(0:n)
  complex(dp) :: cv(0:n,nkx,nz), ce(0:n,nkx,nz), cu(0:n), cw(0:n)
  ! device state
  complex(dp), device :: dv(0:n,nkx,nz), de(0:n,nkx,nz), du0(0:n), dw0(0:n)
  real(dp)    :: rRe, rt, rdt, rQ
  integer(ip) :: rstep, ix, iz, kk, istat
  real(dp)    :: d, dmax
  real(dp)    :: yfd(0:n), yfd2(0:n), dmaxf       ! FD stamp legs
  logical     :: okio, bad, all_pass
  logical     :: okfd, refA, refB, refC
  character(len=IO_HDR_BYTES) :: hdr
  integer(ip) :: u, ios

  all_pass = .true.

  ! ===== build a deterministic known state (0:n, nkx, nz) =====================
  do iz = 1, nz
     do ix = 1, nkx
        do kk = 0, n
           v0(kk,ix,iz) = cmplx( real(kk + 2*ix + 3*iz, dp), &
                                 real(kk - ix + 2*iz,   dp), dp)
           e0(kk,ix,iz) = cmplx( real(5*kk - ix + iz,   dp), &
                                 real(kk + ix - 2*iz,   dp), dp)
        end do
     end do
  end do
  do kk = 0, n
     u00(kk) = cmplx(real(kk,dp),       real(-kk,dp),     dp)
     w00(kk) = cmplx(real(2*kk - 1,dp), real(kk + 3,dp),  dp)
  end do

  ! ===== (A) round-trip ========================================================
  dv = v0;  de = e0;  du0 = u00;  dw0 = w00
  call io_write_state('chk_test_r2c.bin', n, nx, nz, Re, tt, dt, Qtar, stp, &
                      dv, de, du0, dw0, okio, &
                      Lx=12.566370614359172_dp, Lz=6.283185307179586_dp, &
                      flow='channel', ybc='noslip-noslip', drive='CFR')
  if (.not. okio) then
     write(*,'(a)') ' [FAIL] (A) write failed'
     all_pass = .false.
  end if
  ! clear, then read back
  cv = (0.0_dp,0.0_dp); ce = (0.0_dp,0.0_dp); cu = (0.0_dp,0.0_dp); cw = (0.0_dp,0.0_dp)
  dv = (0.0_dp,0.0_dp); de = (0.0_dp,0.0_dp); du0 = (0.0_dp,0.0_dp); dw0 = (0.0_dp,0.0_dp)
  call io_read_state('chk_test_r2c.bin', n, nx, nz, rRe, rt, rdt, rQ, rstep, &
                     dv, de, du0, dw0, okio)
  if (.not. okio) then
     write(*,'(a)') ' [FAIL] (A) read failed (xtype/grid gate?)'
     all_pass = .false.
  end if
  cv = dv;  ce = de;  cu = du0;  cw = dw0
  istat = cudaDeviceSynchronize()

  dmax = 0.0_dp;  bad = .false.
  do iz = 1, nz
     do ix = 1, nkx
        do kk = 0, n
           d = abs(cv(kk,ix,iz) - v0(kk,ix,iz));  dmax = max(dmax, d)
           if (.not. (d == d)) bad = .true.
           d = abs(ce(kk,ix,iz) - e0(kk,ix,iz));  dmax = max(dmax, d)
           if (.not. (d == d)) bad = .true.
        end do
     end do
  end do
  do kk = 0, n
     dmax = max(dmax, abs(cu(kk) - u00(kk)), abs(cw(kk) - w00(kk)))
  end do
  write(*,'(a)')        '=== R2C io_mod: checkpoint round-trip (nkx payload) ==='
  write(*,'(a,es10.2)') ' (A) max|read - written| (fields)   = ', dmax
  write(*,'(a,4(es12.4,1x))') '     scalars Re,t,dt,Q back     = ', rRe, rt, rdt, rQ
  if (bad .or. dmax /= 0.0_dp) then
     write(*,'(a)') ' [FAIL] (A) round-trip not bit-identical'
     all_pass = .false.
  else if (rRe /= Re .or. rt /= tt .or. rdt /= dt .or. rQ /= Qtar .or. rstep /= stp) then
     write(*,'(a)') ' [FAIL] (A) scalars not exact'
     all_pass = .false.
  else
     write(*,'(a)') ' [ok]   (A) round-trip bit-identical, scalars exact'
  end if

  ! ===== (B) grid mismatch refused ============================================
  call io_read_state('chk_test_r2c.bin', n, nx*2, nz, rRe, rt, rdt, rQ, rstep, &
                     dv, de, du0, dw0, okio)
  ! NOTE: nx*2 changes nkx, so the read arrays would be the wrong size; but the
  ! grid gate (fnx /= nx) fires BEFORE any payload read, so okio must be .false.
  if (okio) then
     write(*,'(a)') ' [FAIL] (B) grid mismatch (wrong nx) NOT refused'
     all_pass = .false.
  else
     write(*,'(a)') ' [ok]   (B) grid mismatch refused'
  end if

  ! ===== (C) header self-documents R2C: xtype=r2c + v.shape = ny,nkx,nz ========
  open(newunit=u, file='chk_test_r2c.bin', access='stream', form='unformatted', &
       status='old', action='read', iostat=ios)
  if (ios /= 0) then
     write(*,'(a)') ' [FAIL] (C) cannot reopen file to read header'
     all_pass = .false.
  else
     read(u, pos=1, iostat=ios) hdr
     close(u)
     if (ios /= 0) then
        write(*,'(a)') ' [FAIL] (C) cannot read header bytes'
        all_pass = .false.
     else
        if (index(hdr, 'xtype = r2c') == 0) then
           write(*,'(a)') ' [FAIL] (C) header missing "xtype = r2c"'
           all_pass = .false.
        else if (index(hdr, 'v.shape = 17,5,6') == 0) then   ! ny,nkx,nz = 17,5,6
           write(*,'(a)') ' [FAIL] (C) header v.shape is not ny,nkx,nz (17,5,6)'
           all_pass = .false.
        else
           write(*,'(a)') ' [ok]   (C) header carries xtype=r2c and v.shape=ny,nkx,nz'
        end if
     end if
  end if

  ! ===== (D) FD ybasis stamp: an fd-point/file checkpoint round-trips
  !       under its own stamp (ygrid_moment identity passes on the same nodes),
  !       and every cross-read direction refuses:
  !         refA: fd run, same file, DIFFERENT grid (perturbed nodes)
  !         refB: cheb run reading the fd file
  !         refC: fd run reading the cheb file from (A)
  do kk = 0, n
     yfd(kk) = 1.0_dp - 2.0_dp*real(kk,dp)/real(n,dp)
  end do
  call io_set_ybasis('fd-point', 'file', yfd)
  dv = v0;  de = e0;  du0 = u00;  dw0 = w00
  call io_write_state('chk_test_r2c_fd.bin', n, nx, nz, Re, tt, dt, Qtar, stp, &
                      dv, de, du0, dw0, okio, &
                      Lx=12.566370614359172_dp, Lz=6.283185307179586_dp, &
                      flow='channel', ybc='noslip-noslip', drive='CFR')
  okfd = okio
  dmaxf = -1.0_dp
  if (okfd) then
     dv = (0.0_dp,0.0_dp);  de = (0.0_dp,0.0_dp)
     call io_read_state('chk_test_r2c_fd.bin', n, nx, nz, rRe, rt, rdt, rQ, rstep, &
                        dv, de, du0, dw0, okio)
     okfd = okio
     if (okfd) then
        cv = dv;  istat = cudaDeviceSynchronize()
        dmaxf = 0.0_dp
        do iz = 1, nz
           do ix = 1, nkx
              do kk = 0, n
                 dmaxf = max(dmaxf, abs(cv(kk,ix,iz) - v0(kk,ix,iz)))
              end do
           end do
        end do
        okfd = (dmaxf == 0.0_dp) .and. (rstep == stp)
     end if
  end if
  yfd2 = yfd;  yfd2(n/2) = yfd2(n/2) + 1.0e-6_dp
  call io_set_ybasis('fd-point', 'file', yfd2)
  call io_read_state('chk_test_r2c_fd.bin', n, nx, nz, rRe, rt, rdt, rQ, rstep, &
                     dv, de, du0, dw0, refA)
  call io_set_ybasis('cheb-T', 'glc')
  call io_read_state('chk_test_r2c_fd.bin', n, nx, nz, rRe, rt, rdt, rQ, rstep, &
                     dv, de, du0, dw0, refB)
  call io_set_ybasis('fd-point', 'file', yfd)
  call io_read_state('chk_test_r2c.bin', n, nx, nz, rRe, rt, rdt, rQ, rstep, &
                     dv, de, du0, dw0, refC)
  call io_set_ybasis('cheb-T', 'glc')              ! restore the default stamp
  write(*,'(a,es10.2,a,l1,a,l1,a,l1,a)') ' (D) fd round-trip max|d| = ', dmaxf, &
       '   refusals(grid,cheb<-fd,fd<-cheb) = ', .not.refA, ',', .not.refB, ',', .not.refC, &
       '  (want T,T,T)'
  if (okfd .and. .not.refA .and. .not.refB .and. .not.refC) then
     write(*,'(a)') ' [ok]   (D) fd-point stamp: round-trip exact, all cross-reads refused'
  else
     write(*,'(a)') ' [FAIL] (D) fd-point stamp legs'
     all_pass = .false.
  end if

  write(*,'(a)') '=============================================================='
  if (all_pass) then
     write(*,'(a)') ' ALL R2C IO TESTS PASSED'
  else
     write(*,'(a)') ' SOME R2C IO TESTS FAILED'
     error stop 1
  end if

end program test_io_r2c
