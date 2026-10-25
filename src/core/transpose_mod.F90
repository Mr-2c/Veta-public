!===============================================================================
! transpose_mod.F90  --  A<->B global transpose interface for the multi-GPU
!                        transform path.
!
!   This is the transpose ABSTRACTION the transform path calls, with the
!   batch dimension `nfield` in the signature (a real run fuses
!   several fields into one all-to-all).  The backend HERE is the single-rank
!   case; the P > 1 backend exists and is production
!   (src/parallel/par_transpose_mod.F90, NCCL A2A) -- this module remains the
!   serial suite's MPI/NCCL-independent half (identity backend + the pack
!   EMULATION kernels the P>1 side is bit-gated against):
!     * P == 1 : layout A and layout B are the SAME layout (full kx, full z), so
!                the A<->B transpose is the IDENTITY.  The body is a whole-array
!                device copy out = in -- same kind (tp), conformant at P == 1
!                (a_nz_loc == nz and b_nkx_loc == nkx there), hence BIT-EXACT.
!     * P >  1 : THIS module fails loudly -- much better to fail at a
!                standstill than to compile and fail silently later;
!                multi-rank traffic belongs to
!                par_transpose_mod, never here.
!
! LAYOUTS, y always leading (length ny), batched nfield:
!   Layout A (z-slab):  in/out shaped (ny, nkx,       a_nz_loc, nfield)  -- full
!                       kx, this rank's z-slab.
!   Layout B (kx-slab): in/out shaped (ny, b_nkx_loc, nz,       nfield)  -- this
!                       rank's kx-slab, full z.
!   nkx / nz are the EFFECTIVE (full) extents on each axis; for the dealiased
!   nonlinear path the caller passes the dealiased extents (nkxd as `nkx`, nzd as
!   `nz`) and the matching local slab sizes -- the routine is extent-agnostic.
!   The all-to-all (P>1) swaps which of {kx, z} is split; y rides untouched.
!
! Explicit-shape device dummy arrays (project style; cf. fft3d_mod exec_*): the
! shapes are passed in, so the no-op copy is a plain conformant assignment with
! no derived-type device-member reference (no NVFORTRAN-0155) and no cross-kind
! whole-array copy (no NVFORTRAN-S-0519): in and out are both complex(tp).
!
! Device module: depends on `kinds` + `decomp_t` (host scalar `nprocs` only,
! read in host code -- not inside any kernel).  Build: nvfortran -cuda -c
! transpose_mod.F90  (no cuFFT).  Validated by tests/serial/test_transpose.F90.
!===============================================================================
module transpose_mod
  use kinds,      only: tp, ip
  use decomp_mod, only: decomp_t, block_range
  use cudafor
  implicit none
  private
  public :: transpose_A_to_B, transpose_B_to_A
  ! Device pack/unpack kernels (the per-rank gather/scatter for the all-to-all).
  ! NCCL-FREE on purpose: the grouped ncclSend/Recv that composes them lives in
  ! the parallel/ backend (increment 2b-2b), so the single-GPU test suite stays
  ! MPI/NCCL-independent.  Index mapping validated in src/models/work_decomp_model.py
  ! (pack_A_to_B/unpack_A_to_B/...): pack & unpack share the per-message buffer
  ! ordering [iy, local-kx, local-z, field], so the exchange is a verbatim
  ! per-(src,dst) segment copy and the round trip is bit-exact.
  ! Production stages ONLY the strided Layout-A side (pack_A_to_B /
  ! unpack_B_to_A): the Layout-B side's two kernels are per-(peer,field)
  ! IDENTITY copies (src/models/work_m1_direct_a2a_model.py), so par_transpose
  ! sends/receives the field arrays directly. pack_B_to_A / unpack_A_to_B stay
  ! for the single-GPU pack emulation test (tests/serial/test_transpose_pack).
  public :: pack_A_to_B, unpack_A_to_B, pack_B_to_A, unpack_B_to_A

contains

  !-----------------------------------------------------------------------------
  ! transpose_A_to_B -- z-slab (layout A) -> kx-slab (layout B), nfield batched.
  !   in_A : (ny, nkx,       a_nz_loc, nfield)   full kx, local z-slab
  !   out_B: (ny, b_nkx_loc, nz,       nfield)   local kx-slab, full z
  ! P == 1: identity (out_B = in_A, bit-exact).  P > 1: not yet implemented.
  !-----------------------------------------------------------------------------
  subroutine transpose_A_to_B(d, ny, nkx, nz, a_nz_loc, b_nkx_loc, nfield, in_A, out_B)
    type(decomp_t), intent(in) :: d
    integer(ip),    intent(in) :: ny, nkx, nz, a_nz_loc, b_nkx_loc, nfield
    complex(tp), device :: in_A (ny, nkx,       a_nz_loc, nfield)
    complex(tp), device :: out_B(ny, b_nkx_loc, nz,       nfield)
    if (d%nprocs == 1) then
       ! single rank: layout A == layout B; the transpose is the identity.
       out_B = in_A
    else
       error stop 'transpose_mod: multi-rank A->B all-to-all not yet implemented (M1)'
    end if
  end subroutine transpose_A_to_B

  !-----------------------------------------------------------------------------
  ! transpose_B_to_A -- kx-slab (layout B) -> z-slab (layout A), nfield batched.
  !   in_B : (ny, b_nkx_loc, nz,       nfield)   local kx-slab, full z
  !   out_A: (ny, nkx,       a_nz_loc, nfield)   full kx, local z-slab
  ! P == 1: identity (out_A = in_B, bit-exact).  P > 1: not yet implemented.
  !-----------------------------------------------------------------------------
  subroutine transpose_B_to_A(d, ny, nkx, nz, a_nz_loc, b_nkx_loc, nfield, in_B, out_A)
    type(decomp_t), intent(in) :: d
    integer(ip),    intent(in) :: ny, nkx, nz, a_nz_loc, b_nkx_loc, nfield
    complex(tp), device :: in_B (ny, b_nkx_loc, nz,       nfield)
    complex(tp), device :: out_A(ny, nkx,       a_nz_loc, nfield)
    if (d%nprocs == 1) then
       ! single rank: layout B == layout A; the transpose is the identity.
       out_A = in_B
    else
       error stop 'transpose_mod: multi-rank B->A all-to-all not yet implemented (M1)'
    end if
  end subroutine transpose_B_to_A

  !=============================================================================
  ! Device pack/unpack kernels.  All indices follow src/models/work_decomp_model.py
  ! exactly, translated from 0-based (model) to 1-based (Fortran) arrays:
  !   * global kx g (0-based)  -> Fortran kx index g+1; a kx-slab owned by rank r
  !     starts at 0-based kx0=block_range(nkx,P,r)%lo, so its local kxl=1..bnkx
  !     maps to in/out kx index  kx0 + kxl.
  !   * global z  g (0-based)  -> Fortran z  index g+1; a z-slab owned by rank r
  !     starts at 0-based az0, so local izl=1..anz maps to z index  az0 + izl.
  !   * per-(src,dst) message buffer offset (0-based) is
  !       (iy-1) + ny*((kxl-1) + bnkx*((izl-1) + anz*(f-1)))
  !     so the Fortran (1-based) buffer index is  base + iy + ny*(...)  with base
  !     the running per-destination displacement (exclusive prefix of chunk
  !     sizes), matching alltoall_counts_A_to_B's senddsp/recvdsp.
  ! Empty blocks (P > extent) give zero-trip loops and a zero-size contribution,
  ! exactly as the model's empty ranges do.  No derived-type device member is
  ! referenced and in/out are both complex(tp) (no NVFORTRAN-0155 / -S-0519).
  !=============================================================================

  !-- A->B PACK: layout A (full kx, z-slab pr) -> per-destination send buffer. ---
  subroutine pack_A_to_B(ny, nkx, nz, anz_pr, nfield, P, pr, in_A, slen, sendbuf)
    integer(ip), intent(in) :: ny, nkx, nz, anz_pr, nfield, P, pr, slen
    complex(tp), device     :: in_A(ny, nkx, anz_pr, nfield)
    complex(tp), device     :: sendbuf(slen)
    integer(ip) :: q, kx0, kx1, bnkx_q, base, iy, kxl, izl, f
    base = 0_ip
    do q = 0, P - 1
       call block_range(nkx, P, q, kx0, kx1, bnkx_q)
       !$cuf kernel do(4) <<<*,*>>>
       do f = 1, nfield
          do izl = 1, anz_pr
             do kxl = 1, bnkx_q
                do iy = 1, ny
                   sendbuf(base + iy + ny*((kxl-1) + bnkx_q*((izl-1) + anz_pr*(f-1)))) = &
                        in_A(iy, kx0 + kxl, izl, f)
                end do
             end do
          end do
       end do
       base = base + ny*bnkx_q*anz_pr*nfield
    end do
  end subroutine pack_A_to_B

  !-- A->B UNPACK: received buffer (concat over sources) -> layout B (kx-slab q). -
  subroutine unpack_A_to_B(ny, nkx, nz, bnkx_q, nfield, P, q, rlen, recvbuf, out_B)
    integer(ip), intent(in) :: ny, nkx, nz, bnkx_q, nfield, P, q, rlen
    complex(tp), device     :: recvbuf(rlen)
    complex(tp), device     :: out_B(ny, bnkx_q, nz, nfield)
    integer(ip) :: pr, az0, az1, anz_pr, base, iy, kxl, izl, f
    base = 0_ip
    do pr = 0, P - 1
       call block_range(nz, P, pr, az0, az1, anz_pr)
       !$cuf kernel do(4) <<<*,*>>>
       do f = 1, nfield
          do izl = 1, anz_pr
             do kxl = 1, bnkx_q
                do iy = 1, ny
                   out_B(iy, kxl, az0 + izl, f) = &
                        recvbuf(base + iy + ny*((kxl-1) + bnkx_q*((izl-1) + anz_pr*(f-1))))
                end do
             end do
          end do
       end do
       base = base + ny*bnkx_q*anz_pr*nfield
    end do
  end subroutine unpack_A_to_B

  !-- B->A PACK: layout B (kx-slab q, full z) -> per-destination send buffer. -----
  subroutine pack_B_to_A(ny, nkx, nz, bnkx_q, nfield, P, q, in_B, slen, sendbuf)
    integer(ip), intent(in) :: ny, nkx, nz, bnkx_q, nfield, P, q, slen
    complex(tp), device     :: in_B(ny, bnkx_q, nz, nfield)
    complex(tp), device     :: sendbuf(slen)
    integer(ip) :: pr, az0, az1, anz_pr, base, iy, kxl, izl, f
    base = 0_ip
    do pr = 0, P - 1
       call block_range(nz, P, pr, az0, az1, anz_pr)
       !$cuf kernel do(4) <<<*,*>>>
       do f = 1, nfield
          do izl = 1, anz_pr
             do kxl = 1, bnkx_q
                do iy = 1, ny
                   sendbuf(base + iy + ny*((kxl-1) + bnkx_q*((izl-1) + anz_pr*(f-1)))) = &
                        in_B(iy, kxl, az0 + izl, f)
                end do
             end do
          end do
       end do
       base = base + ny*bnkx_q*anz_pr*nfield
    end do
  end subroutine pack_B_to_A

  !-- B->A UNPACK: received buffer -> layout A (full kx, z-slab pr). --------------
  subroutine unpack_B_to_A(ny, nkx, nz, anz_pr, nfield, P, pr, rlen, recvbuf, out_A)
    integer(ip), intent(in) :: ny, nkx, nz, anz_pr, nfield, P, pr, rlen
    complex(tp), device     :: recvbuf(rlen)
    complex(tp), device     :: out_A(ny, nkx, anz_pr, nfield)
    integer(ip) :: q, kx0, kx1, bnkx_q, base, iy, kxl, izl, f
    base = 0_ip
    do q = 0, P - 1
       call block_range(nkx, P, q, kx0, kx1, bnkx_q)
       !$cuf kernel do(4) <<<*,*>>>
       do f = 1, nfield
          do izl = 1, anz_pr
             do kxl = 1, bnkx_q
                do iy = 1, ny
                   out_A(iy, kx0 + kxl, izl, f) = &
                        recvbuf(base + iy + ny*((kxl-1) + bnkx_q*((izl-1) + anz_pr*(f-1))))
                end do
             end do
          end do
       end do
       base = base + ny*bnkx_q*anz_pr*nfield
    end do
  end subroutine unpack_B_to_A

end module transpose_mod
