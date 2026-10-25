!===============================================================================
! test_decomp.F90  --  CPU validation of decomp_mod (the multi-GPU 1D-slab
! descriptor + index/count arithmetic).  All-integer; mirrors the numpy
! reference src/models/work_decomp_model.py exactly (the asserts here are the
! Fortran image of that model).
!
! Build & run (canonical: `bash run_all_tests.sh` from repo root):
!   nvfortran -module build -c src/core/types.F90
!   nvfortran -module build -c src/core/decomp_mod.F90
!   nvfortran -cuda -cudalib=cufft -module build \
!             tests/serial/test_decomp.F90 build/decomp_mod.o build/types.o -o build/test_decomp
!   ./build/test_decomp
!
! Tests:
!   1. block_range tiles [0,n-1] exactly (contiguous, sums to n, imbalance<=1),
!      over the same (n,P) sweep as the model (incl. P>n empty-block cases).
!   2. g2l/l2g round-trip + EXACTLY-ONE-OWNER for every global index.
!   3. A->B all-to-all counts: exclusive-prefix displacements, per-rank A/B slab
!      volume, cross-rank consistency send_p(q)==recv_q(p), global conservation.
!   4. P=1 reduces to local==global (the no-op transpose's bit-exact precondition).
!   5. owns_mode00 / kx-index-0 owned by exactly rank 0 in layout B.
!   6. decomp_setup derives the SAME extents as grid_mod (nkx,nxd,nzd,nkxd) and
!      the P=1 slab bounds (a_*_loc==global, ranges 0..n-1).
!===============================================================================
program test_decomp
  use kinds,      only: ip
  use decomp_mod, only: decomp_t, decomp_setup, block_range, &
                        alltoall_counts_A_to_B, g2l, l2g, owner
  implicit none

  logical :: all_ok = .true.

  call test_block_range_tiling()
  call test_g2l_roundtrip()
  call test_alltoall_counts()
  call test_p1_local_eq_global()
  call test_owns_mode00()
  call test_setup_extents()

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine check(name, ok)
    character(*), intent(in) :: name
    logical,      intent(in) :: ok
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', ok)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. ok) all_ok = .false.
  end subroutine check

  !-- (n,P) sweep identical to the numpy model main() --------------------------
  subroutine test_block_range_tiling()
    integer(ip) :: Ns(13), Ps(9)
    integer(ip) :: ki, kj, n, P, r, lo, hi, cnt, total, expect_lo, mn, mx
    logical :: ok
    Ns = [1, 2, 3, 4, 5, 7, 8, 16, 17, 65, 128, 129, 192]
    Ps = [1, 2, 3, 4, 5, 7, 8, 16, 32]
    ok = .true.
    do ki = 1, size(Ns)
       n = Ns(ki)
       do kj = 1, size(Ps)
          P = Ps(kj)
          total = 0_ip;  expect_lo = 0_ip
          mn = huge(0_ip);  mx = -1_ip
          do r = 0, P - 1
             call block_range(n, P, r, lo, hi, cnt)
             total = total + cnt
             if (cnt > 0_ip) then
                if (lo /= expect_lo) ok = .false.       ! contiguous, no gap/overlap
                expect_lo = hi + 1_ip
             else
                if (.not. (lo == expect_lo .or. lo == n)) ok = .false.
             end if
             if (cnt < mn) mn = cnt
             if (cnt > mx) mx = cnt
          end do
          if (total /= n) ok = .false.                  ! counts sum to n
          if (expect_lo /= n) ok = .false.              ! union is exactly [0,n-1]
          if (mx - mn > 1_ip) ok = .false.              ! balanced: imbalance <= 1
       end do
    end do
    call check('block_range tiles [0,n-1], sums to n, imbalance<=1', ok)
  end subroutine test_block_range_tiling

  subroutine test_g2l_roundtrip()
    integer(ip) :: Ns(13), Ps(9)
    integer(ip) :: ki, kj, n, P, g, r, rr, il, nown, the_owner
    logical :: ok
    Ns = [1, 2, 3, 4, 5, 7, 8, 16, 17, 65, 128, 129, 192]
    Ps = [1, 2, 3, 4, 5, 7, 8, 16, 32]
    ok = .true.
    do ki = 1, size(Ns)
       n = Ns(ki)
       do kj = 1, size(Ps)
          P = Ps(kj)
          do g = 0, n - 1
             r = owner(n, P, g)
             if (r < 0) ok = .false.
             il = g2l(n, P, r, g)
             if (il < 0) ok = .false.
             if (l2g(n, P, r, il) /= g) ok = .false.    ! round-trip
             ! exactly one rank claims g
             nown = 0_ip;  the_owner = -1_ip
             do rr = 0, P - 1
                if (g2l(n, P, rr, g) >= 0) then
                   nown = nown + 1_ip
                   the_owner = rr
                end if
             end do
             if (nown /= 1_ip) ok = .false.
             if (the_owner /= r) ok = .false.
          end do
       end do
    end do
    call check('g2l/l2g round-trip + exactly-one-owner', ok)
  end subroutine test_g2l_roundtrip

  subroutine test_alltoall_counts()
    integer(ip) :: nyc(4), nkxc(4), nzc(4), Ps(6), nfs(3)
    integer(ip) :: kc, kj, kf, P, nfield, ny, nkx, nz_e, pr, q
    integer(ip) :: a_nz_p, b_nkx_p, lo, hi, ssum, rsum, tots, totr, acc
    integer(ip), allocatable :: sc(:), sd(:), rc(:), rd(:)
    integer(ip), allocatable :: send_all(:,:), recv_all(:,:)
    logical :: ok
    ! the same representative grids as the model (tiny + MKM 128^3, dealiased &
    ! truncated z): (ny, nkx, nz_e)
    nyc  = [33,  33,  129, 129]
    nkxc = [ 9,   9,   65,  65]
    nzc  = [12,   8,  192, 128]
    Ps   = [1, 2, 3, 4, 8, 16]
    nfs  = [1, 3, 6]
    ok = .true.
    do kc = 1, 4
       ny = nyc(kc);  nkx = nkxc(kc);  nz_e = nzc(kc)
       do kj = 1, size(Ps)
          P = Ps(kj)
          do kf = 1, size(nfs)
             nfield = nfs(kf)
             allocate(sc(0:P-1), sd(0:P-1), rc(0:P-1), rd(0:P-1))
             allocate(send_all(0:P-1, 0:P-1), recv_all(0:P-1, 0:P-1))
             do pr = 0, P - 1
                call alltoall_counts_A_to_B(ny, nkx, nz_e, P, pr, nfield, sc, sd, rc, rd)
                ! displacement == exclusive prefix sum of counts
                acc = 0_ip
                do q = 0, P - 1
                   if (sd(q) /= acc) ok = .false.
                   acc = acc + sc(q)
                end do
                acc = 0_ip
                do q = 0, P - 1
                   if (rd(q) /= acc) ok = .false.
                   acc = acc + rc(q)
                end do
                ! per-rank slab volume: send the whole layout-A slab, recv the whole layout-B slab
                call block_range(nz_e, P, pr, lo, hi, a_nz_p)
                call block_range(nkx,  P, pr, lo, hi, b_nkx_p)
                ssum = sum(sc);  rsum = sum(rc)
                if (ssum /= ny * nkx * a_nz_p * nfield) ok = .false.
                if (rsum /= ny * b_nkx_p * nz_e * nfield) ok = .false.
                do q = 0, P - 1
                   send_all(pr, q) = sc(q)
                   recv_all(pr, q) = rc(q)
                end do
             end do
             ! cross-rank consistency: what p sends q == what q receives from p
             do pr = 0, P - 1
                do q = 0, P - 1
                   if (send_all(pr, q) /= recv_all(q, pr)) ok = .false.
                end do
             end do
             ! global conservation
             tots = sum(send_all);  totr = sum(recv_all)
             if (tots /= totr) ok = .false.
             if (tots /= ny * nkx * nz_e * nfield) ok = .false.
             deallocate(sc, sd, rc, rd, send_all, recv_all)
          end do
       end do
    end do
    call check('A->B all-to-all counts/displ/consistency/conservation', ok)
  end subroutine test_alltoall_counts

  subroutine test_p1_local_eq_global()
    integer(ip) :: nyc(4), nkxc(4), nzc(4)
    integer(ip) :: kc, ny, nkx, nz_e, lo, hi, cnt
    integer(ip) :: sc1(0:0), sd1(0:0), rc1(0:0), rd1(0:0)
    logical :: ok
    nyc  = [33,  33,  129, 129]
    nkxc = [ 9,   9,   65,  65]
    nzc  = [12,   8,  192, 128]
    ok = .true.
    do kc = 1, 4
       ny = nyc(kc);  nkx = nkxc(kc);  nz_e = nzc(kc)
       call block_range(nkx, 1_ip, 0_ip, lo, hi, cnt)       ! (0, nkx-1, nkx)
       if (lo /= 0 .or. hi /= nkx - 1 .or. cnt /= nkx) ok = .false.
       call block_range(nz_e, 1_ip, 0_ip, lo, hi, cnt)      ! (0, nz_e-1, nz_e)
       if (lo /= 0 .or. hi /= nz_e - 1 .or. cnt /= nz_e) ok = .false.
       call alltoall_counts_A_to_B(ny, nkx, nz_e, 1_ip, 0_ip, 6_ip, sc1, sd1, rc1, rd1)
       if (sc1(0) /= ny * nkx * nz_e * 6 .or. sd1(0) /= 0) ok = .false.
       if (rc1(0) /= ny * nkx * nz_e * 6 .or. rd1(0) /= 0) ok = .false.
    end do
    call check('P=1: local==global (no-op transpose = bit-exact copy)', ok)
  end subroutine test_p1_local_eq_global

  subroutine test_owns_mode00()
    integer(ip) :: nkxs(2), Ps(4), ki, kj, P, nkx, r
    type(decomp_t) :: d
    logical :: ok
    nkxs = [9, 65];  Ps = [1, 2, 4, 8]
    ok = .true.
    ! pure ownership: kx index 0 lives on rank 0 in layout B
    do ki = 1, 2
       nkx = nkxs(ki)
       do kj = 1, 4
          P = Ps(kj)
          if (owner(nkx, P, 0_ip) /= 0) ok = .false.
       end do
    end do
    ! descriptor flag on a concrete grid (nx=16 -> nkx=9): only rank 0 owns mode 0
    do kj = 1, 4
       P = Ps(kj)
       do r = 0, P - 1
          call decomp_setup(d, 0_ip, P, r, 16_ip, 33_ip, 8_ip)
          if (r == 0) then
             if (.not. d%owns_mode00) ok = .false.
          else
             if (d%owns_mode00) ok = .false.
          end if
       end do
    end do
    call check('owns_mode00 / kx-index-0 owned by rank 0', ok)
  end subroutine test_owns_mode00

  subroutine test_setup_extents()
    type(decomp_t) :: d
    logical :: ok
    integer(ip) :: nx, ny, nz
    ok = .true.
    ! MKM 128^3-ish: nx=128 -> nkx=65, nxd=192, nkxd=97 ; nz=128 -> nzd=192
    nx = 128;  ny = 129;  nz = 128
    call decomp_setup(d, 0_ip, 1_ip, 0_ip, nx, ny, nz)
    if (d%nkx  /= nx/2 + 1)        ok = .false.    ! 65
    if (d%nxd  /= 3*nx/2)          ok = .false.    ! 192
    if (d%nzd  /= 3*nz/2)          ok = .false.    ! 192
    if (d%nkxd /= (3*nx/2)/2 + 1)  ok = .false.    ! 97
    if (d%ny   /= ny)              ok = .false.
    ! P=1 slab invariant: local == global on every split axis (the no-op
    ! transpose's bit-exact precondition)
    if (d%a_nz_loc   /= d%nz)      ok = .false.
    if (d%ad_nz_loc  /= d%nzd)     ok = .false.
    if (d%b_nkx_loc  /= d%nkx)     ok = .false.
    if (d%bd_nkx_loc /= d%nkxd)    ok = .false.
    if (d%a_z0  /= 0 .or. d%a_z1  /= d%nz  - 1)  ok = .false.
    if (d%b_kx0 /= 0 .or. d%b_kx1 /= d%nkx - 1)  ok = .false.
    ! tiny grid nx=16 -> nkx=9, nxd=24, nkxd=13 ; nz=8 -> nzd=12
    nx = 16;  ny = 33;  nz = 8
    call decomp_setup(d, 0_ip, 1_ip, 0_ip, nx, ny, nz)
    if (d%nkx  /= 9)   ok = .false.
    if (d%nxd  /= 24)  ok = .false.
    if (d%nzd  /= 12)  ok = .false.
    if (d%nkxd /= 13)  ok = .false.
    call check('decomp_setup extents match grid_mod + P=1 local==global', ok)
  end subroutine test_setup_extents

end program test_decomp
