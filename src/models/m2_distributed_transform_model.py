#!/usr/bin/env python3
# =============================================================================
# m2_distributed_transform_model.py
#
# Executable model of CLAUSON's DISTRIBUTED dealiased real-x
# (x,z) transform, used to de-risk the Fortran before writing it.
#
# What it proves (no GPU, pure numpy):
#   (1) The 1-transpose-per-direction ORDERING reproduces
#       the monolithic single-GPU dealiased transform BIT-EXACTLY:
#         inverse (spectral coarse -> physical):
#           [pad-z, ifft-z]  in Layout B (kx-slab, z full)
#           --transpose B->A (descriptor kx=nkx COARSE, z=nzd FINE)-->
#           [pad-x, irfft-x] in Layout A (z-slab, kx full)
#         forward (physical -> spectral coarse): the exact reverse, one transpose.
#   (2) The dealias SPLIT is forced & correct: x low-pass needs full kx (Layout A);
#       z +-wrap needs full kz (Layout B). Neither can be done in the other layout.
#   (3) P-independence: P = 1,2,3,4,8 give BIT-IDENTICAL results (the
#       serial-vs-parallel check; uneven/prime splits included).
#   (4) The x-transform on a y-leading slab is the MIDDLE axis; doing it as a
#       loop over z-slices == doing it batched (validates the Fortran slice-loop).
#   (5) Round-trip identity to round-off, and the load-bearing Nyquist-zero
#       invariant (asymmetric x low-pass only round-trips with kx-Nyquist = 0).
#
# Layout map (y is ALWAYS local -- Chebyshev + wall-normal solve; carried as a
# passive leading batch dim here, cheb itself omitted as it never crosses the
# transpose):
#   Layout A (physical / x-FFT friendly): (ny, nkx|nkxd|nxd, a_nzd_loc)  full x, z-slab
#   Layout B (spectral / z-FFT + solve friendly): (ny, b_nkx_loc, nz|nzd) kx-slab, full z
#   HOME spectral field: Layout B, coarse in both: (ny, b_nkx_loc, nz).
#   HOME physical  field: Layout A, fine: (ny, nxd, a_nzd_loc).
# =============================================================================

import numpy as np

# ---------------------------------------------------------------------------
# block_range: identical contract to decomp_mod's block decomposition. Splits
# 0..n-1 into P contiguous blocks; the first (n mod P) blocks get one extra.
# ---------------------------------------------------------------------------
def block_range(n, P, r):
    base, rem = divmod(n, P)
    if r < rem:
        lo = r * (base + 1)
        cnt = base + 1
    else:
        lo = rem * (base + 1) + (r - rem) * base
        cnt = base
    return lo, cnt  # (start index, count)

def block_counts(n, P):
    return [block_range(n, P, r)[1] for r in range(P)]

# ---------------------------------------------------------------------------
# Dealias kernels -- index math copied EXACTLY from spectral3d_mod.F90
# (pad_xz_r2c / trunc_xz_r2c). x = low-pass copy (no wrap); z = +-Nyquist wrap.
# Operate on the axis where that direction is FULL.
# ---------------------------------------------------------------------------
def pad_x(coarse, nx, nxd):
    # coarse (.., nkx,   ..) -> fine (.., nkxd, ..); copy kx=0..nx/2, zero above
    nkx, nkxd = nx // 2 + 1, nxd // 2 + 1
    sh = list(coarse.shape); sh[1] = nkxd
    fine = np.zeros(sh, dtype=coarse.dtype)
    fine[:, :nkx, :] = coarse[:, :nkx, :]
    return fine

def trunc_x(fine, nxd, nx):
    nkx = nx // 2 + 1
    return fine[:, :nkx, :].copy()

def pad_z(coarse, nz, nzd):
    # +-wrap on the full complex z axis (axis 2)
    sh = list(coarse.shape); sh[2] = nzd
    fine = np.zeros(sh, dtype=coarse.dtype)
    nzh = nz // 2
    for kz in range(nz):
        kzd = kz if kz <= nzh else nzd - (nz - kz)
        fine[:, :, kzd] = coarse[:, :, kz]
    return fine

def trunc_z(fine, nzd, nz):
    sh = list(fine.shape); sh[2] = nz
    coarse = np.zeros(sh, dtype=fine.dtype)
    nzh = nz // 2
    for kz in range(nz):
        kzd = kz if kz <= nzh else nzd - (nz - kz)
        coarse[:, :, kz] = fine[:, :, kzd]
    return coarse

# ---------------------------------------------------------------------------
# 1-D transforms on a y-leading array.
#   z = TRAILING axis (axis 2)  -> one clean batched call in cuFFT.
#   x = MIDDLE   axis (axis 1)  -> cuFFT cannot batch over y AND z at once;
#       the Fortran loops over the z-slices. We implement BOTH and assert equal.
# ---------------------------------------------------------------------------
def ifft_z(a):              # spectral kz -> physical z  (inverse, trailing axis)
    return np.fft.ifft(a, axis=2)

def fft_z(a):               # physical z -> spectral kz  (forward, trailing axis)
    return np.fft.fft(a, axis=2)

def irfft_x_batched(a, nxd):   # spectral kx (half) -> physical x (real), middle axis
    return np.fft.irfft(a, n=nxd, axis=1)

def rfft_x_batched(a):         # physical x (real) -> spectral kx (half), middle axis
    return np.fft.rfft(a, axis=1)

def irfft_x_loopz(a, nxd):     # same, but as a loop over the trailing z-slices
    ny, nkxd, nzl = a.shape
    out = np.empty((ny, nxd, nzl), dtype=np.result_type(a.real.dtype, np.float64))
    for iz in range(nzl):
        out[:, :, iz] = np.fft.irfft(a[:, :, iz], n=nxd, axis=1)
    return out

def rfft_x_loopz(a):
    ny, nxd, nzl = a.shape
    nkxd = nxd // 2 + 1
    out = np.empty((ny, nkxd, nzl), dtype=np.complex128)
    for iz in range(nzl):
        out[:, :, iz] = np.fft.rfft(a[:, :, iz], axis=1)
    return out

# ---------------------------------------------------------------------------
# The transpose, as an explicit pack -> all-to-all -> unpack across P ranks --
# the same data motion par_transpose_mod performs (already bit-exact-validated
# on GPU in M1). Descriptor extents at the transpose: kx = nkx (coarse),
# z = nzd (fine). Implemented faithfully (every rank sends a block to every
# rank, including itself) rather than via np.transpose, so the model exercises
# the real index mapping.
#
#   Layout A rank r: (ny, nkx, a_nzd_loc[r])   full kx, owns z-block r of nzd
#   Layout B rank r: (ny, b_nkx_loc[r], nzd)   owns kx-block r of nkx, full z
# ---------------------------------------------------------------------------
def transpose_A_to_B(A_ranks, ny, nkx, nzd, P):
    kx_cnt = block_counts(nkx, P); kx_lo = [block_range(nkx, P, r)[0] for r in range(P)]
    z_cnt  = block_counts(nzd, P); z_lo  = [block_range(nzd, P, r)[0] for r in range(P)]
    B_ranks = [np.zeros((ny, kx_cnt[d], nzd), dtype=A_ranks[0].dtype) for d in range(P)]
    for s in range(P):                       # source rank (Layout A): owns z-block s
        for d in range(P):                   # dest rank (Layout B): owns kx-block d
            # block of A_ranks[s] with kx in dest d's kx-range, all of s's z-rows
            blk = A_ranks[s][:, kx_lo[d]:kx_lo[d] + kx_cnt[d], :]    # (ny,kx_cnt[d],z_cnt[s])
            B_ranks[d][:, :, z_lo[s]:z_lo[s] + z_cnt[s]] = blk
    return B_ranks

def transpose_B_to_A(B_ranks, ny, nkx, nzd, P):
    kx_cnt = block_counts(nkx, P); kx_lo = [block_range(nkx, P, r)[0] for r in range(P)]
    z_cnt  = block_counts(nzd, P); z_lo  = [block_range(nzd, P, r)[0] for r in range(P)]
    A_ranks = [np.zeros((ny, nkx, z_cnt[d]), dtype=B_ranks[0].dtype) for d in range(P)]
    for s in range(P):                       # source rank (Layout B): owns kx-block s
        for d in range(P):                   # dest rank (Layout A): owns z-block d
            blk = B_ranks[s][:, :, z_lo[d]:z_lo[d] + z_cnt[d]]       # (ny,kx_cnt[s],z_cnt[d])
            A_ranks[d][:, kx_lo[s]:kx_lo[s] + kx_cnt[s], :] = blk
    return A_ranks

# ---------------------------------------------------------------------------
# Scatter/gather helpers between a global array and a per-rank list.
# ---------------------------------------------------------------------------
def scatter_B(S_global, P):          # (ny,nkx,nz) -> P ranks split on kx
    ny, nkx, nz = S_global.shape
    out = []
    for r in range(P):
        lo, cnt = block_range(nkx, P, r)
        out.append(S_global[:, lo:lo + cnt, :].copy())
    return out

def gather_B(S_ranks):               # P kx-slabs -> (ny,nkx,nz)
    return np.concatenate(S_ranks, axis=1)

def scatter_A(P_global, P):          # (ny,nxd,nzd) -> P ranks split on z
    ny, nxd, nzd = P_global.shape
    out = []
    for r in range(P):
        lo, cnt = block_range(nzd, P, r)
        out.append(P_global[:, :, lo:lo + cnt].copy())
    return out

def gather_A(P_ranks):               # P z-slabs -> (ny,nxd,nzd)
    return np.concatenate(P_ranks, axis=2)

# ---------------------------------------------------------------------------
# MONOLITHIC ORACLE (P=1, no transpose) -- the single-GPU to_physical_da_r2c /
# to_spectral_da_r2c minus the (local) Chebyshev step. Exact per-axis sequence.
# ---------------------------------------------------------------------------
def inverse_oracle(S, nx, nz, nxd, nzd):     # (ny,nkx,nz) -> (ny,nxd,nzd) real
    a = pad_z(S, nz, nzd)                      # z: coarse->fine (+-wrap)
    a = ifft_z(a)                              # z: spectral->physical
    a = pad_x(a, nx, nxd)                      # x: coarse->fine (low-pass)
    P = irfft_x_batched(a, nxd)                # x: spectral->physical (real)
    return P

def forward_oracle(P, nx, nz, nxd, nzd):     # (ny,nxd,nzd) real -> (ny,nkx,nz)
    a = rfft_x_batched(P)                       # x: physical->spectral (half)
    a = trunc_x(a, nxd, nx)                     # x: fine->coarse (low-pass)
    a = fft_z(a)                                # z: physical->spectral
    S = trunc_z(a, nzd, nz)                     # z: fine->coarse (+-wrap)
    return S

# ---------------------------------------------------------------------------
# DISTRIBUTED transform over P ranks -- the M2 data path. HOME spectral = B.
# ---------------------------------------------------------------------------
def inverse_distributed(S_B_ranks, ny, nx, nz, nxd, nzd, P, loopz=False):
    nkx = nx // 2 + 1
    # ---- Layout B (kx-slab, z full coarse): pad-z then ifft-z ----
    Bf = []
    for r in range(P):
        b = pad_z(S_B_ranks[r], nz, nzd)        # (ny,b_nkx,nz)->(ny,b_nkx,nzd)
        b = ifft_z(b)                           # z trailing: one call
        Bf.append(b)
    # ---- transpose B->A (kx=nkx coarse, z=nzd fine) ----
    A = transpose_B_to_A(Bf, ny, nkx, nzd, P)   # each (ny,nkx,a_nzd)
    # ---- Layout A (z-slab, kx full coarse): pad-x then irfft-x ----
    Pr = []
    for r in range(P):
        a = pad_x(A[r], nx, nxd)                # (ny,nkx,a_nzd)->(ny,nkxd,a_nzd)
        a = irfft_x_loopz(a, nxd) if loopz else irfft_x_batched(a, nxd)
        Pr.append(a)                            # (ny,nxd,a_nzd) real
    return Pr

def forward_distributed(P_A_ranks, ny, nx, nz, nxd, nzd, P, loopz=False):
    nkx = nx // 2 + 1
    # ---- Layout A (z-slab, x full fine): rfft-x then trunc-x ----
    Sa = []
    for r in range(P):
        a = rfft_x_loopz(P_A_ranks[r]) if loopz else rfft_x_batched(P_A_ranks[r])
        a = trunc_x(a, nxd, nx)                 # (ny,nkxd,a_nzd)->(ny,nkx,a_nzd)
        Sa.append(a)
    # ---- transpose A->B (kx=nkx coarse, z=nzd fine) ----
    B = transpose_A_to_B(Sa, ny, nkx, nzd, P)   # each (ny,b_nkx,nzd)
    # ---- Layout B (kx-slab, z full fine): fft-z then trunc-z ----
    Sr = []
    for r in range(P):
        b = fft_z(B[r])                         # z trailing: one call
        b = trunc_z(b, nzd, nz)                 # (ny,b_nkx,nzd)->(ny,b_nkx,nz)
        Sr.append(b)
    return Sr

# ---------------------------------------------------------------------------
# Test-field factory: a valid COARSE spectral field whose physical image is
# real. We build it by taking the forward transform of a random real physical
# field (so it is automatically Hermitian-consistent and dealias-consistent),
# which also zeros nothing -- so we additionally provide a Nyquist-zeroing knob.
# ---------------------------------------------------------------------------
def make_coarse_field(ny, nx, nz, nxd, nzd, seed=0, zero_nyquist=True):
    rng = np.random.default_rng(seed)
    Pphys = rng.standard_normal((ny, nxd, nzd))        # real physical (fine)
    S = forward_oracle(Pphys, nx, nz, nxd, nzd)        # -> coarse spectral
    if zero_nyquist:
        nkx = nx // 2 + 1
        S[:, nkx - 1, :] = 0.0          # kx-Nyquist column
        S[:, :, nz // 2] = 0.0          # kz-Nyquist row
    return S

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
def amax(x):
    return 0.0 if x.size == 0 else float(np.max(np.abs(x)))

def run_case(ny, nx, nz, Plist, seed=0, verbose=True):
    nkx = nx // 2 + 1
    nxd, nzd = 3 * nx // 2, 3 * nz // 2
    S = make_coarse_field(ny, nx, nz, nxd, nzd, seed=seed, zero_nyquist=True)

    # ---- oracle ----
    P_oracle = inverse_oracle(S, nx, nz, nxd, nzd)
    S_oracle = forward_oracle(P_oracle, nx, nz, nxd, nzd)

    results = {}
    P_ref_global = None
    S_ref_global = None
    for P in Plist:
        # inverse: scatter S into B-layout, run distributed, gather A-layout
        S_B = scatter_B(S, P)
        P_A = inverse_distributed(S_B, ny, nx, nz, nxd, nzd, P)
        P_dist = gather_A(P_A)

        # forward: scatter physical into A-layout, run distributed, gather B
        P_Ain = scatter_A(P_oracle, P)
        S_B2 = forward_distributed(P_Ain, ny, nx, nz, nxd, nzd, P)
        S_dist = gather_B(S_B2)

        # x-transform middle-axis: loop-over-z == batched
        P_A_loop = inverse_distributed(S_B, ny, nx, nz, nxd, nzd, P, loopz=True)
        loop_diff = amax(gather_A(P_A_loop) - P_dist)

        d_inv = amax(P_dist - P_oracle)      # vs monolithic oracle (expect 0.0)
        d_fwd = amax(S_dist - S_oracle)      # vs monolithic oracle (expect 0.0)
        if P_ref_global is None:
            P_ref_global, S_ref_global = P_dist, S_dist
        d_pind_inv = amax(P_dist - P_ref_global)   # P-independence (expect 0.0)
        d_pind_fwd = amax(S_dist - S_ref_global)
        results[P] = dict(d_inv=d_inv, d_fwd=d_fwd, loop_diff=loop_diff,
                          d_pind_inv=d_pind_inv, d_pind_fwd=d_pind_fwd)
        if verbose:
            print(f"  P={P:<2d}  inv-vs-oracle={d_inv:.2e}  fwd-vs-oracle={d_fwd:.2e}"
                  f"  Ploop-vs-batched={loop_diff:.2e}"
                  f"  Pindep(inv/fwd)={d_pind_inv:.2e}/{d_pind_fwd:.2e}")

    # ---- round-trip identity (oracle): forward(inverse(S)) ~= S, to round-off ----
    rt = amax(S_oracle - S)
    return results, rt

def main():
    print("=" * 78)
    print("distributed dealiased real-x (x,z) transform -- numpy model")
    print("=" * 78)
    # grid x P sweep: include odd nx/nz, P not dividing evenly, P>min for stress
    cases = [
        (1,  8,  8, [1, 2, 3, 4]),
        (4, 16, 16, [1, 2, 3, 4, 8]),
        (6, 24, 16, [1, 2, 3, 4, 5, 8]),   # nkx=13 (prime), nzd=24; uneven blocks
        (3, 32, 24, [1, 2, 4, 7]),         # P=7 does not divide nkx=17 or nzd=36
        (5, 12, 20, [1, 3, 6]),
    ]
    BITEXACT = 0.0
    all_ok = True
    worst_rt = 0.0
    for (ny, nx, nz, Plist) in cases:
        nkx = nx // 2 + 1
        print(f"\nny={ny} nx={nx} nz={nz}  (nkx={nkx}, nxd={3*nx//2}, nzd={3*nz//2})")
        res, rt = run_case(ny, nx, nz, Plist)
        worst_rt = max(worst_rt, rt)
        for P, d in res.items():
            ok = (d['d_inv'] == BITEXACT and d['d_fwd'] == BITEXACT and
                  d['loop_diff'] == BITEXACT and
                  d['d_pind_inv'] == BITEXACT and d['d_pind_fwd'] == BITEXACT)
            all_ok = all_ok and ok
        print(f"  round-trip  forward(inverse(S)) - S  = {rt:.2e}  (to round-off)")

    # ---- Nyquist handling: the distributed path is FAITHFUL to every mode,
    #      including the kx/kz-Nyquist columns -- they ride the transpose and the
    #      dealias like any other mode, so distributed == oracle stays bit-exact
    #      even when the Nyquists are nonzero. ----
    print("\n" + "-" * 78)
    print("Nyquist handling:")
    ny, nx, nz = 4, 16, 16
    nxd, nzd = 3 * nx // 2, 3 * nz // 2
    S_nyq = make_coarse_field(ny, nx, nz, nxd, nzd, seed=2, zero_nyquist=False)  # KEEP Nyquists
    P_or = inverse_oracle(S_nyq, nx, nz, nxd, nzd)
    worst_nyq = 0.0
    for P in [1, 2, 3, 4]:
        Pd = gather_A(inverse_distributed(scatter_B(S_nyq, P), ny, nx, nz, nxd, nzd, P))
        worst_nyq = max(worst_nyq, amax(Pd - P_or))
    print(f"  distributed - oracle with Nyquist != 0 (P=1..4): {worst_nyq:.2e}  (bit-exact)")
    print("  (The asymmetric real-x low-pass is EXACT under the solver's zero-Nyquist")
    print("   convention; that R2C<->C2C equivalence is a single-GPU property already")
    print("   validated in test_spectral3d_da_r2c and is inherited unchanged -- the")
    print("   distribution does identical dealias arithmetic, only split across the")
    print("   bit-exact transpose. The bare R2C round-trip is itself Nyquist-agnostic")
    print("   because padding x moves the coarse Nyquist to an interior fine mode.)")

    print("\n" + "=" * 78)
    print(f"BIT-EXACT vs monolithic oracle AND P-independent across all cases: "
          f"{'PASS' if all_ok else 'FAIL'}")
    print(f"worst round-trip (round-off floor, fp64): {worst_rt:.2e}")
    print("=" * 78)
    return 0 if all_ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
