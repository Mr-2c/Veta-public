#!/usr/bin/env python3
"""
work_decomp_model.py -- numpy/python reference model for CLAUSON decomp_mod.

Index/layout logic is modelled in Python/numpy before it is written in Fortran:
the count/displacement arithmetic is the #1 bug source.

This models the 1D-slab decomposition arithmetic that decomp_mod.F90 implements,
and the A<->B all-to-all count/displacement arithmetic that transpose_mod will
consume (M1).  The Fortran test_decomp.F90 mirrors the asserts below.

Decomposition (the multi-GPU plan):
  - Layout B (kx-slab):  each rank owns a contiguous block of kx in [0,nkx),
    all z.   Solve- and z-FFT-friendly.
  - Layout A (z-slab):   each rank owns a contiguous block of z  in [0,nz_e),
    all kx.  Physical- and x-FFT-friendly.   (z extent is the *dealiased* nzd
    for the nonlinear-path transpose; the truncated nz for a pure-spectral one.)
  - The A<->B transpose is a single all-to-all that swaps which of {kx, z} is
    split.  y (leading, length ny) rides along inside every message untouched.

Convention: 0-based global indices [0, n-1] (the Fortran code uses the same
0-based kx/kz indexing the solver already uses, e.g. grid%kx(0:nkx-1)).
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Block decomposition of n items [0, n-1] across P ranks (the standard
#    "first `rem` ranks get one extra" balanced block).  Returns (lo, hi, cnt)
#    for rank r; an empty block (when P > n) has cnt==0 and lo == n, hi == n-1.
# ---------------------------------------------------------------------------
def block_range(n, P, r):
    assert 0 <= r < P and P >= 1 and n >= 0
    base = n // P
    rem = n % P
    if r < rem:
        cnt = base + 1
        lo = r * (base + 1)
    else:
        cnt = base
        lo = rem * (base + 1) + (r - rem) * base
    hi = lo + cnt - 1
    return lo, hi, cnt


# ---------------------------------------------------------------------------
# 2. All-to-all counts for an A->B transpose (z-slab -> kx-slab), per field,
#    in units of complex elements.  Rank p (layout A) holds (ny, all-kx,
#    z in A-slab_p).  Rank q (layout B) will hold (ny, kx in B-slab_q, all-z).
#    So p sends q the block (ny, B-slab_q kx, A-slab_p z):
#        sendcnt_p[q] = ny * b_nkx_loc(q) * a_nz_loc(p)
#    and p receives from q (ny, B-slab_p kx, A-slab_q z):
#        recvcnt_p[q] = ny * b_nkx_loc(p) * a_nz_loc(q)
#    Displacements are exclusive prefix sums (contiguous per-destination bufs).
#    nfield batches: multiply every count by nfield (one fused all-to-all).
# ---------------------------------------------------------------------------
def alltoall_counts_A_to_B(ny, nkx, nz_e, P, p, nfield=1):
    b_nkx = [block_range(nkx, P, q)[2] for q in range(P)]   # layout B kx counts
    a_nz = [block_range(nz_e, P, q)[2] for q in range(P)]   # layout A z  counts
    sendcnt = [ny * b_nkx[q] * a_nz[p] * nfield for q in range(P)]
    recvcnt = [ny * b_nkx[p] * a_nz[q] * nfield for q in range(P)]
    senddsp = np.concatenate([[0], np.cumsum(sendcnt)[:-1]]).astype(int).tolist()
    recvdsp = np.concatenate([[0], np.cumsum(recvcnt)[:-1]]).astype(int).tolist()
    return sendcnt, senddsp, recvcnt, recvdsp


# ---------------------------------------------------------------------------
# 3. global <-> local index maps for a block-split axis.
# ---------------------------------------------------------------------------
def g2l(n, P, r, g):
    lo, hi, cnt = block_range(n, P, r)
    return (g - lo) if (lo <= g <= hi) else -1   # -1 => not owned by rank r


def l2g(n, P, r, il):
    lo, hi, cnt = block_range(n, P, r)
    assert 0 <= il < cnt
    return lo + il


def owner(n, P, g):
    for r in range(P):
        lo, hi, cnt = block_range(n, P, r)
        if lo <= g <= hi:
            return r
    return -1


# ===========================================================================
# 4. Data-movement model: pack -> all-to-all -> unpack, verified bit-identical
#    to the serial reshape.  This is the EXECUTABLE SPEC for transpose_mod's
#    device pack/unpack kernels + grouped NCCL send/recv (increment 2b).
#    Column-major linear indices (ny fastest) mirror the Fortran arrays
#    A:(ny,nkx,a_nz,nfield), B:(ny,b_nkx,nz,nfield), and the per-pair message
#    buffer ordering [iy, local-kx, local-z, field] used by BOTH pack & unpack.
# ===========================================================================
def _cmaj(iy, i1, i2, f, n0, n1, n2):
    return iy + n0 * (i1 + n1 * (i2 + n2 * f))


def _chunk_lin(iy, kxl, izl, f, ny, bnkx, anz):
    # per-(src,dst) message-buffer offset; identical on pack and unpack sides.
    return iy + ny * (kxl + bnkx * (izl + anz * f))


def _make_global(ny, nkx, nz, nfield, seed):
    rng = np.random.default_rng(seed)
    n = ny * nkx * nz * nfield
    return rng.standard_normal(n) + 1j * rng.standard_normal(n)


def _scatter_to_A(G, ny, nkx, nz, nfield, P, pr):
    """global field -> layout A (z-slab) on rank pr: (ny, all-kx, z-slab_pr)."""
    az0, _, anz = block_range(nz, P, pr)
    A = np.zeros(ny * nkx * anz * nfield, dtype=complex)
    for f in range(nfield):
        for izl in range(anz):
            for kx in range(nkx):
                for iy in range(ny):
                    A[_cmaj(iy, kx, izl, f, ny, nkx, anz)] = \
                        G[_cmaj(iy, kx, az0 + izl, f, ny, nkx, nz)]
    return A


def _ref_B(G, ny, nkx, nz, nfield, P, q):
    """serial reference of layout B (kx-slab) on rank q: (ny, kx-slab_q, all-z)."""
    bkx0, _, bnkx = block_range(nkx, P, q)
    B = np.zeros(ny * bnkx * nz * nfield, dtype=complex)
    for f in range(nfield):
        for iz in range(nz):
            for kxl in range(bnkx):
                for iy in range(ny):
                    B[_cmaj(iy, kxl, iz, f, ny, bnkx, nz)] = \
                        G[_cmaj(iy, bkx0 + kxl, iz, f, ny, nkx, nz)]
    return B


def pack_A_to_B(A_pr, ny, nkx, nz, nfield, P, pr):
    """A (z-slab) on rank pr -> per-destination contiguous send buffer."""
    az0, _, anz_pr = block_range(nz, P, pr)
    sc, sd, rc, rd = alltoall_counts_A_to_B(ny, nkx, nz, P, pr, nfield)
    sendbuf = np.zeros(sum(sc), dtype=complex)
    for q in range(P):
        bkx0, _, bnkx_q = block_range(nkx, P, q)
        base = sd[q]
        for f in range(nfield):
            for izl in range(anz_pr):
                for kxl in range(bnkx_q):
                    for iy in range(ny):
                        sendbuf[base + _chunk_lin(iy, kxl, izl, f, ny, bnkx_q, anz_pr)] = \
                            A_pr[_cmaj(iy, bkx0 + kxl, izl, f, ny, nkx, anz_pr)]
    return sendbuf, sc, sd


def unpack_A_to_B(recvbuf, ny, nkx, nz, nfield, P, q):
    """received buffer (concat over sources) -> layout B (kx-slab) on rank q."""
    bkx0, _, bnkx_q = block_range(nkx, P, q)
    sc, sd, rc, rd = alltoall_counts_A_to_B(ny, nkx, nz, P, q, nfield)
    B = np.zeros(ny * bnkx_q * nz * nfield, dtype=complex)
    for pr in range(P):
        az0, _, anz_pr = block_range(nz, P, pr)
        base = rd[pr]
        for f in range(nfield):
            for izl in range(anz_pr):
                for kxl in range(bnkx_q):
                    for iy in range(ny):
                        B[_cmaj(iy, kxl, az0 + izl, f, ny, bnkx_q, nz)] = \
                            recvbuf[base + _chunk_lin(iy, kxl, izl, f, ny, bnkx_q, anz_pr)]
    return B


def pack_B_to_A(B_q, ny, nkx, nz, nfield, P, q):
    """B (kx-slab) on rank q -> per-destination send buffer.  B->A send
    counts/disps on q EQUAL the A->B RECV counts/disps on q (send<->recv swap)."""
    bkx0, _, bnkx_q = block_range(nkx, P, q)
    _sc, _sd, sc, sd = alltoall_counts_A_to_B(ny, nkx, nz, P, q, nfield)
    sendbuf = np.zeros(sum(sc), dtype=complex)
    for pr in range(P):
        az0, _, anz_pr = block_range(nz, P, pr)
        base = sd[pr]
        for f in range(nfield):
            for izl in range(anz_pr):
                for kxl in range(bnkx_q):
                    for iy in range(ny):
                        sendbuf[base + _chunk_lin(iy, kxl, izl, f, ny, bnkx_q, anz_pr)] = \
                            B_q[_cmaj(iy, kxl, az0 + izl, f, ny, bnkx_q, nz)]
    return sendbuf, sc, sd


def unpack_B_to_A(recvbuf, ny, nkx, nz, nfield, P, pr):
    """received buffer -> layout A (z-slab) on rank pr.  B->A recv counts/disps
    on pr EQUAL the A->B SEND counts/disps on pr."""
    az0, _, anz_pr = block_range(nz, P, pr)
    sc, sd, _rc, _rd = alltoall_counts_A_to_B(ny, nkx, nz, P, pr, nfield)
    A = np.zeros(ny * nkx * anz_pr * nfield, dtype=complex)
    for q in range(P):
        bkx0, _, bnkx_q = block_range(nkx, P, q)
        base = sd[q]
        for f in range(nfield):
            for izl in range(anz_pr):
                for kxl in range(bnkx_q):
                    for iy in range(ny):
                        A[_cmaj(iy, bkx0 + kxl, izl, f, ny, nkx, anz_pr)] = \
                            recvbuf[base + _chunk_lin(iy, kxl, izl, f, ny, bnkx_q, anz_pr)]
    return A


def check_transpose_bit_identity(ny, nkx, nz, nfield, P, seed):
    """pack -> all-to-all -> unpack must equal the serial reshape (A->B), and
    A->B->A must return the original (round trip).  Data is MOVED, never
    combined, so equality is BIT-EXACT (no float reassociation)."""
    G = _make_global(ny, nkx, nz, nfield, seed)
    # forward A->B
    sendbufs = {pr: pack_A_to_B(_scatter_to_A(G, ny, nkx, nz, nfield, P, pr),
                                ny, nkx, nz, nfield, P, pr) for pr in range(P)}
    Bdist = {}
    for q in range(P):
        sc_q, sd_q, rc_q, rd_q = alltoall_counts_A_to_B(ny, nkx, nz, P, q, nfield)
        rb = np.zeros(sum(rc_q), dtype=complex)
        for pr in range(P):
            sb, sc_pr, sd_pr = sendbufs[pr]
            rb[rd_q[pr]:rd_q[pr] + rc_q[pr]] = sb[sd_pr[q]:sd_pr[q] + sc_pr[q]]
            assert sc_pr[q] == rc_q[pr]
        Bdist[q] = unpack_A_to_B(rb, ny, nkx, nz, nfield, P, q)
        assert np.array_equal(Bdist[q], _ref_B(G, ny, nkx, nz, nfield, P, q)), \
            f"A->B mismatch ny={ny} nkx={nkx} nz={nz} nf={nfield} P={P} q={q}"
    # backward B->A round trip
    sendbufs2 = {q: pack_B_to_A(Bdist[q], ny, nkx, nz, nfield, P, q) for q in range(P)}
    for pr in range(P):
        sc_pr, sd_pr, _rc, _rd = alltoall_counts_A_to_B(ny, nkx, nz, P, pr, nfield)
        rb = np.zeros(sum(sc_pr), dtype=complex)
        for q in range(P):
            sb, sc_q, sd_q = sendbufs2[q]
            rb[sd_pr[q]:sd_pr[q] + sc_pr[q]] = sb[sd_q[pr]:sd_q[pr] + sc_q[pr]]
        A = unpack_B_to_A(rb, ny, nkx, nz, nfield, P, pr)
        assert np.array_equal(A, _scatter_to_A(G, ny, nkx, nz, nfield, P, pr)), \
            f"A->B->A round-trip mismatch ny={ny} nkx={nkx} nz={nz} nf={nfield} P={P} pr={pr}"


# ===========================================================================
# ASSERTS (mirrored by tests/serial/test_decomp.F90)
# ===========================================================================
def check_tiling(n, P):
    """Ranges tile [0,n-1] exactly: contiguous, no gaps/overlaps, counts sum n."""
    covered = []
    total = 0
    expect_lo = 0
    for r in range(P):
        lo, hi, cnt = block_range(n, P, r)
        total += cnt
        if cnt > 0:
            assert lo == expect_lo, (n, P, r, lo, expect_lo)
            covered.extend(range(lo, hi + 1))
            expect_lo = hi + 1
        else:
            # empty block: contributes nothing, leaves expect_lo unchanged
            assert lo == expect_lo or lo == n, (n, P, r, lo, expect_lo)
    assert total == n, (n, P, total)
    assert covered == list(range(n)), (n, P)
    # max imbalance is at most 1 element (balanced block property)
    counts = [block_range(n, P, r)[2] for r in range(P)]
    assert max(counts) - min(counts) <= 1, (n, P, counts)


def check_g2l_roundtrip(n, P):
    for g in range(n):
        r = owner(n, P, g)
        assert r != -1
        il = g2l(n, P, r, g)
        assert il >= 0
        assert l2g(n, P, r, il) == g
        # exactly one owner
        owners = [rr for rr in range(P) if g2l(n, P, rr, g) >= 0]
        assert owners == [r], (n, P, g, owners)


def check_alltoall(ny, nkx, nz_e, P, nfield=1):
    # build every rank's send/recv arrays
    send = {}
    recv = {}
    for p in range(P):
        s, sd, rc, rd = alltoall_counts_A_to_B(ny, nkx, nz_e, P, p, nfield)
        send[p] = (s, sd)
        recv[p] = (rc, rd)
        # displacement = exclusive prefix sum of counts
        assert sd == np.concatenate([[0], np.cumsum(s)[:-1]]).astype(int).tolist()
        assert rd == np.concatenate([[0], np.cumsum(rc)[:-1]]).astype(int).tolist()
        # local volume bookkeeping
        a_nz_p = block_range(nz_e, P, p)[2]
        b_nkx_p = block_range(nkx, P, p)[2]
        assert sum(s) == ny * nkx * a_nz_p * nfield, "A-volume"   # send whole layout-A slab
        assert sum(rc) == ny * b_nkx_p * nz_e * nfield, "B-volume"  # recv whole layout-B slab
    # cross-rank consistency: what p sends q == what q receives from p
    for p in range(P):
        for q in range(P):
            assert send[p][0][q] == recv[q][0][p], (p, q, send[p][0][q], recv[q][0][p])
    # global conservation: total sent == total received
    tot_send = sum(sum(send[p][0]) for p in range(P))
    tot_recv = sum(sum(recv[q][0]) for q in range(P))
    assert tot_send == tot_recv == ny * nkx * nz_e * nfield


def main():
    # MKM-ish and tiny grids; P sweeps include P > extent (empty-block edge case)
    Ns = [1, 2, 3, 4, 5, 7, 8, 16, 17, 65, 128, 129, 192]
    Ps = [1, 2, 3, 4, 5, 7, 8, 16, 32]
    for n in Ns:
        for P in Ps:
            check_tiling(n, P)
            check_g2l_roundtrip(n, P)
    print("[ok] block_range tiling + g2l round-trip over",
          len(Ns) * len(Ps), "(n,P) combinations")

    # all-to-all counts for representative (ny, nkx, nz_e) at several P, nfield
    cases = [
        (33, 9, 12),     # tiny test grid: ny=33, nx=16 -> nkx=9, nz=8 -> nzd=12
        (33, 9, 8),      # same grid, truncated z (pure-spectral transpose)
        (129, 65, 192),  # MKM 128^3: ny=129, nkx=65, nzd=192
        (129, 65, 128),  # MKM truncated z
    ]
    for (ny, nkx, nz_e) in cases:
        for P in [1, 2, 3, 4, 8, 16]:
            for nfield in [1, 3, 6]:
                check_alltoall(ny, nkx, nz_e, P, nfield)
    print("[ok] all-to-all count/displacement arithmetic (counts, displacements,")
    print("     cross-rank consistency, A/B volume, global conservation)")

    # P=1 special case: local == global, no-op transpose is a pure copy
    for (ny, nkx, nz_e) in cases:
        assert block_range(nkx, 1, 0) == (0, nkx - 1, nkx)
        assert block_range(nz_e, 1, 0) == (0, nz_e - 1, nz_e)
        s, sd, rc, rd = alltoall_counts_A_to_B(ny, nkx, nz_e, 1, 0, nfield=6)
        assert s == [ny * nkx * nz_e * 6] and sd == [0]
        assert rc == [ny * nkx * nz_e * 6] and rd == [0]
    print("[ok] P=1 reduces to local==global (no-op transpose = bit-exact copy)")

    # mode (0,0) ownership: kx=0 lives in layout B rank owning kx-index 0 == rank 0
    for nkx in [9, 65]:
        for P in [1, 2, 4, 8]:
            assert owner(nkx, P, 0) == 0
    print("[ok] (0,0) mode (kx index 0) owned by exactly rank 0 in layout B")

    # data-movement bit-identity: pack -> all-to-all -> unpack == serial reshape,
    # plus A->B->A round trip.  Small grids (the index logic is size-independent;
    # these still hit ny=1, nz!=nkx, prime-ish extents, and P>extent empty blocks).
    dm_cases = [(33, 9, 8, 1), (33, 9, 8, 6), (33, 9, 12, 3), (16, 9, 8, 2),
                (1, 5, 7, 4), (33, 13, 13, 5), (8, 2, 3, 2)]
    for (ny, nkx, nz, nfield) in dm_cases:
        for P in [1, 2, 3, 4, 5, 8]:
            check_transpose_bit_identity(ny, nkx, nz, nfield, P,
                                         seed=ny + nkx + nz + nfield + P)
    print("[ok] transpose data-movement: pack/all-to-all/unpack bit-identical to")
    print("     serial reshape, and A->B->A round-trip exact (small-grid x P sweep)")

    print("\nALL MODEL CHECKS PASSED")


if __name__ == "__main__":
    main()
