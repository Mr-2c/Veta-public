#!/usr/bin/env python3
"""
work_m1_direct_a2a_model.py -- numpy decision/spec model for the "pack
identity": the transpose's Layout-B side never needs a staging buffer.

Index/layout logic is modelled in numpy before it is written in Fortran.  This
file is the EXECUTABLE SPEC for the par_transpose_mod arrangement that (a) has
no identity-side pack/unpack passes and (b) uses ONE persistent staging buffer
rather than two.

THE OBSERVATION
  The per-(src,dst) wire order [iy, local-kx, local-z, field] (validated in
  work_decomp_model.py) makes the Layout-B side's staging a PURE IDENTITY per
  (peer, field):
    * A->B: the message from src pr, field f, lands in out_B(:, :, az0(pr)+1 :
      az0(pr)+anz_pr, f) -- one CONTIGUOUS column-major run, and the exclusive-
      prefix displacement equals the run's array offset (block_range tiles z
      ascending and contiguous, so sum of prior anz == az0).
    * B->A: the message to dst pr, field f, is read from in_B(:, :, az0(pr)+1 :
      az0(pr)+anz_pr, f) -- the same contiguous run.
  The kx side (Layout A) is genuinely strided (kx is dim 2 under full nkx) and
  keeps its gather/scatter kernels + the single staging buffer.

THE CONSEQUENCE (what the Fortran must implement)
  * A->B: pack (unchanged, gather) -> per-(q,f) ncclSend from the packed buffer
    at senddsp(q) + (f-1)*sc1(q) -> per-(q,f) ncclRecv DIRECTLY into
    out_B(1, 1, az0(q)+1, f).  NO unpack kernel, NO recv buffer.
  * B->A: per-(q,f) ncclSend DIRECTLY from in_B(1, 1, az0(q)+1, f) -> per-(q,f)
    ncclRecv into the staging buffer at a2b_sd(q) + (f-1)*(a2b_sc(q)/nf), which
    is BYTE-IDENTICAL to the buffer pack_B_to_A used to build -> unpack
    (unchanged, scatter).  NO pack kernel, NO send buffer.
  * NCCL p2p matching is one-to-one per (src,dst) posting order with equal
    counts, so BOTH ends split per (peer, field); both iterate f ascending.
  * nfield == 1 (every pipelined ptp leg + the r2c single-field transposes)
    degenerates to one message per peer -- today's message count exactly.

Asserted here, for uneven P/extents/nfield sweeps, all BIT-EXACT:
  1. the identity claims themselves (buffer == array slice / whole array);
  2. direct A->B == reference pack/exchange/unpack (final out_B equal);
  3. direct B->A produces a staging buffer byte-identical to the reference
     pack_B_to_A buffer (so the unchanged unpack sees identical input), and
     the final out_A is equal;
  4. per-(q,f) send/recv counts match pairwise in posting order;
  5. the ONE-buffer footprint claim: max over directions of the staged side
     never exceeds the current per-buffer size mx = max(slen, rlen).
"""

import numpy as np

from work_decomp_model import (
    block_range, alltoall_counts_A_to_B, _cmaj, _make_global, _scatter_to_A,
    _ref_B, pack_A_to_B, unpack_A_to_B, pack_B_to_A, unpack_B_to_A,
)


# ---------------------------------------------------------------------------
# helper: the Layout-B array's column-major offset of element (0, 0, az0, f).
# This is THE displacement the direct side hands to NCCL (via 4-D subscripts
# in Fortran -- 64-bit address arithmetic, no int32 linearisation).
# ---------------------------------------------------------------------------
def _bside_off(ny, bnkx, nz, az0, f):
    return _cmaj(0, 0, az0, f, ny, bnkx, nz)


# ---------------------------------------------------------------------------
# 1. identity claims
# ---------------------------------------------------------------------------
def check_identity_claims(ny, nkx, nz, nfield, P, seed):
    G = _make_global(ny, nkx, nz, nfield, seed)
    for q in range(P):
        B_q = _ref_B(G, ny, nkx, nz, nfield, P, q)
        bkx0, _, bnkx_q = block_range(nkx, P, q)
        # B->A send side: reference-packed buffer vs direct slices of in_B
        sendbuf, sc, sd = pack_B_to_A(B_q, ny, nkx, nz, nfield, P, q)
        for pr in range(P):
            az0, _, anz_pr = block_range(nz, P, pr)
            c1 = sc[pr] // nfield if nfield else 0
            assert c1 * nfield == sc[pr]                    # exact split
            for f in range(nfield):
                off = _bside_off(ny, bnkx_q, nz, az0, f)
                msg_ref = sendbuf[sd[pr] + f * c1: sd[pr] + (f + 1) * c1]
                msg_direct = B_q[off: off + c1]
                assert np.array_equal(msg_ref, msg_direct), \
                    f"B->A identity broken pr={pr} f={f}"
        if nfield == 1:
            # the whole-buffer identity
            assert np.array_equal(sendbuf, B_q), "nf=1 whole-buffer identity"


# ---------------------------------------------------------------------------
# 2+4. direct A->B: pack unchanged; per-(q,f) messages land straight in out_B.
# ---------------------------------------------------------------------------
def direct_A_to_B(G, ny, nkx, nz, nfield, P):
    sendbufs = {pr: pack_A_to_B(_scatter_to_A(G, ny, nkx, nz, nfield, P, pr),
                                ny, nkx, nz, nfield, P, pr) for pr in range(P)}
    Bdist = {}
    for q in range(P):
        bkx0, _, bnkx_q = block_range(nkx, P, q)
        _sc, _sd, rc_q, _rd = alltoall_counts_A_to_B(ny, nkx, nz, P, q, nfield)
        B = np.zeros(ny * bnkx_q * nz * nfield, dtype=complex)
        for pr in range(P):
            sb, sc_pr, sd_pr = sendbufs[pr]
            az0, _, anz_pr = block_range(nz, P, pr)
            s1 = sc_pr[q] // nfield          # per-field send chunk on pr
            r1 = rc_q[pr] // nfield          # per-field recv chunk on q
            assert s1 == r1, "posting-order pairwise count match"
            for f in range(nfield):          # both ends iterate f ascending
                if s1 == 0:
                    continue
                msg = sb[sd_pr[q] + f * s1: sd_pr[q] + (f + 1) * s1]
                off = _bside_off(ny, bnkx_q, nz, az0, f)
                B[off: off + r1] = msg       # DIRECT landing -- no unpack
        Bdist[q] = B
    return Bdist


# ---------------------------------------------------------------------------
# 3+4. direct B->A: per-(q,f) messages read straight from in_B; staging buffer
#      byte-identical to the reference pack; unpack unchanged.
# ---------------------------------------------------------------------------
def direct_B_to_A(Bdist, ny, nkx, nz, nfield, P):
    Adist = {}
    for pr in range(P):
        # B->A recv counts/disps on pr == A->B SEND counts/disps on pr
        sc_pr, sd_pr, _rc, _rd = alltoall_counts_A_to_B(ny, nkx, nz, P, pr, nfield)
        az0_pr, _, anz_pr = block_range(nz, P, pr)
        rb = np.zeros(sum(sc_pr), dtype=complex)
        for q in range(P):
            bkx0, _, bnkx_q = block_range(nkx, P, q)
            # B->A send counts on q == A->B RECV counts on q; per-field chunk:
            _s, _d, rc_q, _rd_q = alltoall_counts_A_to_B(ny, nkx, nz, P, q, nfield)
            s1 = rc_q[pr] // nfield          # per-field send chunk on q
            r1 = sc_pr[q] // nfield          # per-field recv chunk on pr
            assert s1 == r1, "posting-order pairwise count match"
            for f in range(nfield):
                if s1 == 0:
                    continue
                off = _bside_off(ny, bnkx_q, nz, az0_pr, f)
                msg = Bdist[q][off: off + s1]            # DIRECT read -- no pack
                rb[sd_pr[q] + f * r1: sd_pr[q] + (f + 1) * r1] = msg
        # staging buffer must be byte-identical to the reference-packed one
        ref_rb = np.zeros_like(rb)
        for q in range(P):
            sb_q, sc_q, sd_q = pack_B_to_A(Bdist[q], ny, nkx, nz, nfield, P, q)
            ref_rb[sd_pr[q]: sd_pr[q] + sc_pr[q]] = sb_q[sd_q[pr]: sd_q[pr] + sc_q[pr]]
        assert np.array_equal(rb, ref_rb), f"B->A staging buffer differs pr={pr}"
        Adist[pr] = unpack_B_to_A(rb, ny, nkx, nz, nfield, P, pr)   # unchanged
    return Adist


# ---------------------------------------------------------------------------
# 5. one-buffer footprint: the staged side (A->B send stage / B->A recv stage)
#    never exceeds today's single-buffer size mx = max(slen, rlen).
# ---------------------------------------------------------------------------
def check_one_buffer(ny, nkx, nz, nfield, P):
    for pr in range(P):
        sc, _sd, rc, _rd = alltoall_counts_A_to_B(ny, nkx, nz, P, pr, nfield)
        mx = max(sum(sc), sum(rc))
        assert sum(sc) <= mx and sum(rc) <= mx   # A->B stages slen; B->A stages rlen


def check_direct_exchange(ny, nkx, nz, nfield, P, seed):
    G = _make_global(ny, nkx, nz, nfield, seed)
    Bdist = direct_A_to_B(G, ny, nkx, nz, nfield, P)
    for q in range(P):
        assert np.array_equal(Bdist[q], _ref_B(G, ny, nkx, nz, nfield, P, q)), \
            f"direct A->B mismatch ny={ny} nkx={nkx} nz={nz} nf={nfield} P={P} q={q}"
    Adist = direct_B_to_A(Bdist, ny, nkx, nz, nfield, P)
    for pr in range(P):
        assert np.array_equal(Adist[pr],
                              _scatter_to_A(G, ny, nkx, nz, nfield, P, pr)), \
            f"direct B->A mismatch ny={ny} nkx={nkx} nz={nz} nf={nfield} P={P} pr={pr}"
    check_one_buffer(ny, nkx, nz, nfield, P)


def main():
    # same small-grid philosophy as work_decomp_model: the index logic is
    # size-independent; hit ny=1, prime-ish extents, nkx!=nz, uneven slabs,
    # and the production nfields {1,3,6} (+ odd 5).  P <= min(nkx, nz)
    # (pt_guard's slab precondition) PLUS the P>extent empty-block edge.
    cases = [(33, 9, 8), (33, 9, 12), (16, 9, 8), (1, 5, 7), (33, 13, 13),
             (8, 2, 3), (5, 17, 24)]     # (5,17,24) = the hard_mp uneven combo
    Ps = [1, 2, 3, 4, 5, 8]
    nfs = [1, 2, 3, 5, 6]
    ncomb = 0
    for (ny, nkx, nz) in cases:
        for P in Ps:
            for nf in nfs:
                seed = ny + nkx + nz + nf + P
                check_identity_claims(ny, nkx, nz, nf, P, seed)
                check_direct_exchange(ny, nkx, nz, nf, P, seed)
                ncomb += 1
    print(f"[ok] M1 identity + direct-exchange bit-identity over {ncomb} "
          f"(grid, P, nfield) combinations")
    print("[ok] B->A staging buffers byte-identical to the reference pack "
          "(unpack input unchanged)")
    print("[ok] per-(peer,field) send/recv counts pairwise matched in posting order")
    print("[ok] one staging buffer (mx = max(slen, rlen)) covers both directions")
    print("\nALL M1 MODEL CHECKS PASSED")


if __name__ == "__main__":
    main()
