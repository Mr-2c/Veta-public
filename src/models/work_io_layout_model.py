#!/usr/bin/env python3
"""
work_io_layout_model.py -- numpy reference model for CLAUSON distributed-I/O layout.

Index/layout arithmetic is modelled in numpy before it is written in Fortran.
This validates the "single-shift" file-layout scheme of the
distributed checkpoint/snapshot writer so the Fortran
par_io module + its test can mirror these asserts exactly.

THE SCHEME
  Choose the on-disk canonical order so the SPLIT axis is the OUTERMOST (slowest,
  last-in-Fortran-order) dimension. Then each rank's data is ONE contiguous block:
      offset(r) = elems_per_split_index * slab_start(r)
      length(r) = elems_per_split_index * slab_count(r)
  block_range tiles the split axis exactly, so the blocks partition the file with no
  gaps/overlap, and a restart at a DIFFERENT rank count reads its own band's block.

  - PHYSICAL snapshot: per-rank array (ny, nx, a_nz_loc), split axis = z = already the
    outermost dim. Disk order (ny, nx, nz). NO rotation; write the slab as-is.
  - SPECTRAL checkpoint: per-rank array (ny, bnkx, nz), split axis = kx = the MIDDLE
    dim. Disk order (ny, nz, nkx) [kx outermost]. Needs a LOCAL rotation
    (ny, bnkx, nz) -> (ny, nz, bnkx) on write; the inverse on read.

Everything is Fortran (column-major) order: dim-1 fastest, last dim slowest.
0-based global indices, matching decomp_mod / work_decomp_model.
"""

import numpy as np


def block_range(n, P, r):
    """Balanced block of [0,n): first `rem` ranks get one extra. (lo, cnt)."""
    base, rem = n // P, n % P
    if r < rem:
        return r * (base + 1), base + 1
    return rem * (base + 1) + (r - rem) * base, base


def tiles_exactly(n, P):
    """The P blocks partition [0,n) with no gap/overlap."""
    covered, nxt = [], 0
    for r in range(P):
        lo, cnt = block_range(n, P, r)
        assert lo == nxt, f"gap/overlap at rank {r}: lo={lo} expected {nxt}"
        covered += list(range(lo, lo + cnt))
        nxt = lo + cnt
    assert covered == list(range(n)) and nxt == n
    return True


# ---------------------------------------------------------------------------
# Disk canonical images (independent of P) -- the single source of truth a
# single-GPU oracle / Python reader would load.
# ---------------------------------------------------------------------------
def disk_physical(G):
    """G: global physical (ny, nx, nz). Disk = (ny, nx, nz) Fortran-flat (z slowest)."""
    return G.flatten(order="F")


def disk_spectral(G):
    """G: global spectral (ny, nkx, nz). Disk = (ny, nz, nkx) Fortran-flat (kx slowest)."""
    return G.transpose(0, 2, 1).flatten(order="F")          # rotate kx to outermost


# ---------------------------------------------------------------------------
# Per-rank WRITE: produce (offset, contiguous buffer) for rank r at count P.
# ---------------------------------------------------------------------------
def write_block_physical(G, P, r):
    ny, nx, nz = G.shape
    z0, cnt = block_range(nz, P, r)
    slab = G[:, :, z0:z0 + cnt]                              # (ny, nx, cnt) -- as-is
    return ny * nx * z0, slab.flatten(order="F")


def write_block_spectral(G, P, r):
    ny, nkx, nz = G.shape
    k0, cnt = block_range(nkx, P, r)
    slab = G[:, k0:k0 + cnt, :]                              # (ny, cnt, nz) in memory
    rot = slab.transpose(0, 2, 1)                            # -> (ny, nz, cnt): the rotation
    return ny * nz * k0, rot.flatten(order="F")


# ---------------------------------------------------------------------------
# Per-rank READ: pull rank r's slab back out of the disk image at count P,
# un-rotating spectral back to the (ny, bnkx, nz) in-memory layout.
# ---------------------------------------------------------------------------
def read_slab_physical(disk, ny, nx, nz, P, r):
    z0, cnt = block_range(nz, P, r)
    off, length = ny * nx * z0, ny * nx * block_range(nz, P, r)[1]
    return disk[off:off + length].reshape((ny, nx, cnt), order="F")


def read_slab_spectral(disk, ny, nkx, nz, P, r):
    k0, cnt = block_range(nkx, P, r)
    off, length = ny * nz * k0, ny * nz * cnt
    rot = disk[off:off + length].reshape((ny, nz, cnt), order="F")  # (ny, nz, cnt)
    return rot.transpose(0, 2, 1)                            # un-rotate -> (ny, cnt, nz)


# ===========================================================================
# Validation
# ===========================================================================
def check(ny, nx, nz, Pset):
    nkx = nx // 2 + 1
    Gp = (np.arange(ny * nx * nz).reshape((ny, nx, nz), order="F")).astype(np.float64)
    Gs = (np.arange(ny * nkx * nz).reshape((ny, nkx, nz), order="F")
          + 1j * np.arange(ny * nkx * nz).reshape((ny, nkx, nz), order="F"))

    dp, ds = disk_physical(Gp), disk_spectral(Gs)

    for n, P in (("nz", nz), ("nkx", nkx)):
        for pp in Pset:
            tiles_exactly(nz if n == "nz" else nkx, pp)

    # (1) WRITE: blocks from all ranks, concatenated in offset order, == the disk image.
    for P in Pset:
        buf = np.zeros_like(dp)
        for r in range(P):
            off, blk = write_block_physical(Gp, P, r)
            buf[off:off + blk.size] = blk
        assert np.array_equal(buf, dp), f"physical write mismatch P={P}"

        buf = np.zeros_like(ds)
        for r in range(P):
            off, blk = write_block_spectral(Gs, P, r)
            buf[off:off + blk.size] = blk
        assert np.array_equal(buf, ds), f"spectral write mismatch P={P}"

    # (2) READ: each rank reads its slab back from the disk image and it matches the
    #     in-memory slab it would have held -- at ANY P (this is cross-P restart).
    for P in Pset:
        for r in range(P):
            z0, cnt = block_range(nz, P, r)
            assert np.array_equal(read_slab_physical(dp, ny, nx, nz, P, r),
                                  Gp[:, :, z0:z0 + cnt]), f"physical read P={P} r={r}"
            k0, cnt = block_range(nkx, P, r)
            assert np.array_equal(read_slab_spectral(ds, ny, nkx, nz, P, r),
                                  Gs[:, k0:k0 + cnt, :]), f"spectral read P={P} r={r}"

    # (3) CROSS-P end-to-end: write at P1, read back at P2, reassemble == original.
    for P1 in Pset:
        for P2 in Pset:
            buf = np.zeros_like(ds)
            for r in range(P1):                              # write at P1
                off, blk = write_block_spectral(Gs, P1, r)
                buf[off:off + blk.size] = blk
            G2 = np.zeros_like(Gs)                           # read at P2
            for r in range(P2):
                k0, cnt = block_range(nkx, P2, r)
                G2[:, k0:k0 + cnt, :] = read_slab_spectral(buf, ny, nkx, nz, P2, r)
            assert np.array_equal(G2, Gs), f"cross-P spectral write@{P1} read@{P2}"
    return True


if __name__ == "__main__":
    # a few shapes incl. ones where nkx/nz are NOT divisible by P (unequal blocks)
    cases = [(5, 8, 6), (3, 12, 7), (4, 10, 9), (6, 16, 12)]
    Pset = [1, 2, 3, 4, 5]
    for (ny, nx, nz) in cases:
        ok = check(ny, nx, nz, Pset)
        print(f"  ny={ny} nx={nx} nz={nz} nkx={nx//2+1}  P in {Pset}:  ALL CHECKS PASS")
    print("work_io_layout_model: single-shift layout VALIDATED "
          "(physical z-slab no-rotation; spectral kx-slab with rotation; cross-P).")
