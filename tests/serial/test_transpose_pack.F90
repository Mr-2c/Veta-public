!===============================================================================
! test_transpose_pack.F90  --  SINGLE-GPU validation of transpose_mod's device
! pack/unpack kernels (increment 2b-2a), WITHOUT NCCL.
!
! Strategy (mirrors src/models/work_decomp_model.py:check_transpose_bit_identity):
! emulate a P-way decomposition inside ONE process.  Build a distinct-valued
! global field G; for each emulated rank pack its layout-A slab on the GPU into a
! per-destination send buffer; perform the all-to-all as plain HOST-SIDE segment
! copies (exactly what grouped ncclSend/Recv will do in 2b-2b -- a verbatim
! per-(src,dst) segment move); unpack on the GPU into layout B; and compare to a
! reference layout B built INDEPENDENTLY from G by direct indexing.  Then do the
! reverse (B->A) and check the A->B->A round trip recovers A.
!
! The transpose only MOVES data (no arithmetic), so every check is BIT-EXACT
! (tol = 0).  NaN-hardened (all(X==X)) and nonzero-guarded (max|G|>0).  Covers
! nfield = 1 and > 1, several P, two grids with nz /= nkx; all extents >= P so no
! empty blocks (that edge is covered in the numpy model; the kernels handle it
! via zero-trip loops regardless).
!
! Build & run (ON A GPU NODE; canonical: `bash run_all_tests.sh` from repo root):
!   nvfortran -module build -c src/core/types.F90
!   nvfortran -module build -c src/core/decomp_mod.F90
!   nvfortran -cuda -module build -c src/core/transpose_mod.F90
!   nvfortran -cuda -cudalib=cufft -module build tests/serial/test_transpose_pack.F90 \
!             build/transpose_mod.o build/decomp_mod.o build/types.o -o build/test_transpose_pack
!   ./build/test_transpose_pack
!===============================================================================
program test_transpose_pack
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,         only: tp, ip
  use decomp_mod,    only: block_range
  use transpose_mod, only: pack_A_to_B, unpack_A_to_B, pack_B_to_A, unpack_B_to_A
  use cudafor
  implicit none

  type :: bufh
     complex(tp), allocatable :: v(:)
  end type bufh

  integer(ip), parameter :: nxs(2) = [ 16_ip, 10_ip ]
  integer(ip), parameter :: nys(2) = [ 33_ip, 17_ip ]
  integer(ip), parameter :: nzs(2) = [  8_ip, 12_ip ]
  integer(ip), parameter :: nfs(3) = [  1_ip,  3_ip, 6_ip ]
  integer(ip), parameter :: Ps(4)  = [  1_ip,  2_ip,  3_ip, 4_ip ]

  logical           :: all_ok = .true., ok
  integer(ip)       :: ig, jf, kp
  character(len=80) :: lbl

  do ig = 1, size(nxs)
     do jf = 1, size(nfs)
        do kp = 1, size(Ps)
           ok = emul(nxs(ig), nys(ig), nzs(ig), nfs(jf), Ps(kp))
           write(lbl,'(a,i0,a,i0,a,i0,a,i0,a,i0)') &
                'pack/unpack emul nx=', nxs(ig), ' ny=', nys(ig), &
                ' nz=', nzs(ig), ' nf=', nfs(jf), ' P=', Ps(kp)
           call check(trim(lbl), ok)
        end do
     end do
  end do

  if (all_ok) then
     print '(a)', 'ALL TESTS PASSED'
  else
     print '(a)', 'SOME TESTS FAILED'
     error stop 1
  end if

contains

  subroutine check(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
    if (.not. okk) all_ok = .false.
  end subroutine check

  integer(ip) function cntb(n, P, r) result(c)        ! block count on a split axis
    integer(ip), intent(in) :: n, P, r
    integer(ip) :: lo, hi
    call block_range(n, P, r, lo, hi, c)
  end function cntb

  integer(ip) function lob(n, P, r) result(lo)        ! 0-based block start
    integer(ip), intent(in) :: n, P, r
    integer(ip) :: hi, c
    call block_range(n, P, r, lo, hi, c)
  end function lob

  ! full emulation + bit-identity checks for one (nx,ny,nz,nfield,P) ----------
  logical function emul(nx, ny, nz, nfield, P) result(good)
    integer(ip), intent(in) :: nx, ny, nz, nfield, P
    integer(ip) :: nkx, pr, q, iy, ikx, iz, izl, kxl, f
    integer(ip) :: anz_pr, bnkx_q, az0, kx0, slen, rlen, cnt, sdisp, recvbase
    real(real64) :: maxd, maxa
    integer      :: istat
    real(tp),    allocatable :: gre(:,:,:,:), gim(:,:,:,:)
    complex(tp), allocatable :: G(:,:,:,:)
    complex(tp), allocatable :: loc(:,:,:,:), ref(:,:,:,:)
    complex(tp), allocatable, device :: A_d(:,:,:,:), B_d(:,:,:,:), buf_d(:)
    type(bufh), allocatable :: sb(:), rb(:)

    good = .true.
    nkx  = nx/2 + 1

    ! distinct-valued global field G (ny, nkx, nz, nfield) -----------------------
    allocate(G(ny,nkx,nz,nfield), gre(ny,nkx,nz,nfield), gim(ny,nkx,nz,nfield))
    call random_number(gre);  call random_number(gim)
    G = cmplx(gre, gim, tp)
    maxa = real(maxval(abs(G)), real64)
    if (.not. (maxa > 0.0_real64)) good = .false.      ! nonzero guard

    allocate(sb(0:P-1), rb(0:P-1))

    !===========================================================================
    ! A -> B : pack each rank (GPU) -> host segment-exchange -> unpack (GPU)
    !===========================================================================
    do pr = 0, P-1
       anz_pr = cntb(nz, P, pr);   az0 = lob(nz, P, pr)
       slen   = ny*nkx*anz_pr*nfield
       allocate(loc(ny, nkx, anz_pr, nfield))                 ! layout A on rank pr
       loc = G(:, :, az0+1 : az0+anz_pr, :)
       allocate(A_d(ny, nkx, anz_pr, nfield), buf_d(slen))
       A_d = loc
       call pack_A_to_B(ny, nkx, nz, anz_pr, nfield, P, pr, A_d, slen, buf_d)
       istat = cudaDeviceSynchronize()
       allocate(sb(pr)%v(slen));  sb(pr)%v = buf_d            ! device -> host
       deallocate(loc, A_d, buf_d)
    end do

    ! emulated all-to-all (verbatim per-(src,dst) segment copies):
    !   senddisp(pr->q) = ny*nfield*anz_pr*kx0(q) ; recvbase accumulates on q
    do q = 0, P-1
       bnkx_q = cntb(nkx, P, q)
       rlen   = ny*bnkx_q*nz*nfield
       allocate(rb(q)%v(rlen))
       recvbase = 0_ip
       do pr = 0, P-1
          anz_pr = cntb(nz, P, pr)
          cnt    = ny*bnkx_q*anz_pr*nfield
          sdisp  = ny*nfield*anz_pr*lob(nkx, P, q)
          rb(q)%v(recvbase+1 : recvbase+cnt) = sb(pr)%v(sdisp+1 : sdisp+cnt)
          recvbase = recvbase + cnt
       end do
    end do

    do q = 0, P-1
       bnkx_q = cntb(nkx, P, q);   kx0 = lob(nkx, P, q)
       rlen   = ny*bnkx_q*nz*nfield
       allocate(B_d(ny, bnkx_q, nz, nfield), buf_d(rlen))
       allocate(loc(ny, bnkx_q, nz, nfield), ref(ny, bnkx_q, nz, nfield))
       buf_d = rb(q)%v
       call unpack_A_to_B(ny, nkx, nz, bnkx_q, nfield, P, q, rlen, buf_d, B_d)
       istat = cudaDeviceSynchronize()
       loc = B_d                                              ! device -> host
       ref = G(:, kx0+1 : kx0+bnkx_q, :, :)                   ! independent reference B_q
       if (.not. all(loc == loc)) good = .false.             ! NaN guard
       maxd = real(maxval(abs(loc - ref)), real64)
       if (maxd /= 0.0_real64) good = .false.                ! bit-exact
       deallocate(B_d, buf_d, loc, ref)
    end do

    !===========================================================================
    ! B -> A round trip : pack each q's layout B -> exchange -> unpack to A,
    ! compare to the original layout A.
    !===========================================================================
    do q = 0, P-1
       if (allocated(sb(q)%v)) deallocate(sb(q)%v)
       if (allocated(rb(q)%v)) deallocate(rb(q)%v)
    end do

    do q = 0, P-1
       bnkx_q = cntb(nkx, P, q);   kx0 = lob(nkx, P, q)
       slen   = ny*bnkx_q*nz*nfield
       allocate(loc(ny, bnkx_q, nz, nfield))                  ! layout B on rank q
       loc = G(:, kx0+1 : kx0+bnkx_q, :, :)
       allocate(B_d(ny, bnkx_q, nz, nfield), buf_d(slen))
       B_d = loc
       call pack_B_to_A(ny, nkx, nz, bnkx_q, nfield, P, q, B_d, slen, buf_d)
       istat = cudaDeviceSynchronize()
       allocate(sb(q)%v(slen));  sb(q)%v = buf_d
       deallocate(loc, B_d, buf_d)
    end do

    ! exchange for B->A:  senddisp(q->pr) = ny*nfield*bnkx_q*az0(pr)
    do pr = 0, P-1
       anz_pr = cntb(nz, P, pr)
       rlen   = ny*nkx*anz_pr*nfield
       allocate(rb(pr)%v(rlen))
       recvbase = 0_ip
       do q = 0, P-1
          bnkx_q = cntb(nkx, P, q)
          cnt    = ny*bnkx_q*anz_pr*nfield
          sdisp  = ny*nfield*bnkx_q*lob(nz, P, pr)
          rb(pr)%v(recvbase+1 : recvbase+cnt) = sb(q)%v(sdisp+1 : sdisp+cnt)
          recvbase = recvbase + cnt
       end do
    end do

    do pr = 0, P-1
       anz_pr = cntb(nz, P, pr);   az0 = lob(nz, P, pr)
       rlen   = ny*nkx*anz_pr*nfield
       allocate(A_d(ny, nkx, anz_pr, nfield), buf_d(rlen))
       allocate(loc(ny, nkx, anz_pr, nfield), ref(ny, nkx, anz_pr, nfield))
       buf_d = rb(pr)%v
       call unpack_B_to_A(ny, nkx, nz, anz_pr, nfield, P, pr, rlen, buf_d, A_d)
       istat = cudaDeviceSynchronize()
       loc = A_d
       ref = G(:, :, az0+1 : az0+anz_pr, :)                   ! original layout A on pr
       if (.not. all(loc == loc)) good = .false.
       maxd = real(maxval(abs(loc - ref)), real64)
       if (maxd /= 0.0_real64) good = .false.
       deallocate(A_d, buf_d, loc, ref)
    end do

    do q = 0, P-1
       if (allocated(sb(q)%v)) deallocate(sb(q)%v)
       if (allocated(rb(q)%v)) deallocate(rb(q)%v)
    end do
    deallocate(sb, rb, G, gre, gim)
  end function emul

end program test_transpose_pack
