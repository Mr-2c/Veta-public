#!/usr/bin/env python3
"""resample_checkpoint.py -- offline cross-resolution checkpoint converter.

Resamples a ClOp DNS v2 checkpoint (v, eta + the mean (0,0) profiles u0, w0)
from its stored grid to a DIFFERENT (n, nx, nz), writing a new, standard
checkpoint that the *unchanged* solver reads at any GPU count.

WHY THIS IS A SPECTRAL OPERATION, NOT INTERPOLATION
    The checkpoint stores the full restart state as SPECTRAL COEFFICIENTS
    (io_mod.F90 / par_io_mod.F90): v, eta as raw Chebyshev T_k coefficients in
    y x Fourier in x,z, plus the mean profiles u0, w0 (Chebyshev in y). So a
    resolution change is a per-axis truncation (downscale) or zero-padding
    (upscale) of coefficients -- spectrally exact, no physical-space interp.

    * y  (Chebyshev T_k coeffs, 0:n)     : slice / zero-pad the leading coeffs.
    * kx (R2C half-spectrum, 0..nx/2)    : low-pass copy (index == wavenumber).
    * kz (full complex, FFT-ordered)     : signed-wavenumber +/- wrap remap.

    This is operationally IDENTICAL to the solver's own 3/2 dealiasing pad/unpad
    (spectral3d_mod.F90 pad_xz_r2c / trunc_xz_r2c) -- same kx low-pass copy, same
    kz `merge(kz, nzd-(nz-kz), kz<=nzh)` wrap -- generalised to an arbitrary grid
    ratio instead of the fixed 3/2.

NYQUIST INVARIANT (load-bearing; spectral3d_mod.F90:234-239)
    The solver keeps the x- and z-Nyquist modes zeroed. This converter ENFORCES
    that on output: the smaller grid's Nyquist mode is dropped (left zero). For a
    valid solver checkpoint (Nyquist already zero) an upscale is therefore exact
    and a same-grid resample is bit-identical; a downscale additionally discards
    the modes above the new Nyquist (the expected, lossy truncation).

EXACTNESS
    * Upscale (zero-pad): EXACT -- the represented field is unchanged, only the
      resolution increases. Ideal for cheap-spin-up-then-refine (converge
      turbulence at low res, prolong to seed the high-res run).
    * Downscale (truncate): spectrally accurate but lossy by construction (high
      modes are discarded). The first solve re-imposes wall BCs and the v/eta
      formulation re-enforces div-free, so the restart is clean in practice.

CONSUMPTION (no Fortran change needed)
    Set the namelist to the TARGET grid, point restart_file at the resampled
    file, do_restart=.true.  The reader gates the file against the deck's grid
    (main_par.F90:337); since the converter already produced it at that grid, the
    read is accepted and t/step/Re/dt are restored (deck Re/dt take precedence,
    with a [restart] note). Works at any P (the slab-kx layout is P-independent).

Usage
    python3 resample_checkpoint.py SRC DST --n 96 --nx 96 --nz 96 [--reset-time]
    python3 resample_checkpoint.py --selftest

    --n/--nx/--nz use NAMELIST semantics (n = Chebyshev parameter; ny = n+1).
    --reset-time  zero t and step in the output header (fresh clock for the
                  refined run) instead of carrying them over.

Both serial-yxz (single-GPU io_mod) and slab-kx (distributed par_io_mod) layouts
are handled; the output keeps the source layout.
"""
import sys
import os
import numpy as np

# read_checkpoint lives alongside this file (the single format-driven reader).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_checkpoint import read_checkpoint, _parse_header   # noqa: E402

HDR_BYTES = 1024


# ---------------------------------------------------------------------------
# Per-axis spectral resize primitives. Each operates on `axis` of `a`; the
# active axis is moved to the front, resized, and moved back.
# ---------------------------------------------------------------------------
def _resize_cheb_y(a, ny_new, axis=0):
    """Chebyshev T_k coefficient axis: truncate or zero-pad the leading coeffs."""
    a = np.moveaxis(a, axis, 0)
    ny_old = a.shape[0]
    out = np.zeros((ny_new,) + a.shape[1:], dtype=a.dtype)
    m = min(ny_old, ny_new)
    out[:m] = a[:m]
    return np.moveaxis(out, 0, axis)


def _resize_kx(a, nx_old, nx_new, axis=0):
    """R2C half-spectrum (0..nx/2): low-pass copy; smaller-grid Nyquist -> 0.

    Index == wavenumber, so this is the `pad_xz_r2c` x low-pass copy generalised:
    keep kx = 0..h-1 with h = min(nx_old,nx_new)//2 (skip the smaller Nyquist).
    """
    a = np.moveaxis(a, axis, 0)
    nkx_new = nx_new // 2 + 1
    out = np.zeros((nkx_new,) + a.shape[1:], dtype=a.dtype)
    h = min(nx_old, nx_new) // 2
    out[:h] = a[:h]
    return np.moveaxis(out, 0, axis)


def _resize_kz(a, nz_old, nz_new, axis=0):
    """Full complex FFT-ordered axis (0,1,..,nz/2,-nz/2+1,..,-1): +/- wrap remap.

    Mirrors the solver's `merge(kz, nzd-(nz-kz), kz<=nzh)` wrap: keep wavenumbers
    |m| <= h-1 (h = min(nz_old,nz_new)//2) from BOTH the positive head and the
    negative tail; the smaller-grid Nyquist (+/-h) is dropped (left zero).
    """
    a = np.moveaxis(a, axis, 0)
    out = np.zeros((nz_new,) + a.shape[1:], dtype=a.dtype)
    h = min(nz_old, nz_new) // 2
    out[:h] = a[:h]                       # positive incl DC: wavenumbers 0..h-1
    if h >= 2:                            # negative tail: wavenumbers -(h-1)..-1
        out[nz_new - (h - 1):] = a[nz_old - (h - 1):]
    return np.moveaxis(out, 0, axis)


# ---------------------------------------------------------------------------
# Field resampling on the CANONICAL (ny, nkx, nz) axis order.
# ---------------------------------------------------------------------------
def _to_canonical(arr, layout):
    """Bring a field to (ny, nkx, nz). slab-kx stores (ny, nz, nkx) on disk."""
    if layout == 'slab-kx':
        return np.ascontiguousarray(arr.transpose(0, 2, 1))
    return np.asarray(arr)                         # serial-yxz is already (ny,nkx,nz)


def _from_canonical(arr, layout):
    """Inverse of _to_canonical: (ny,nkx,nz) -> the on-disk axis order."""
    if layout == 'slab-kx':
        return np.ascontiguousarray(arr.transpose(0, 2, 1))   # -> (ny,nz,nkx)
    return arr


def resample_volume(arr_canon, n_old, nx_old, nz_old, n_new, nx_new, nz_new):
    """Resize a canonical (ny,nkx,nz) spectral volume to the target grid."""
    out = _resize_cheb_y(arr_canon, n_new + 1, axis=0)
    out = _resize_kx(out, nx_old, nx_new, axis=1)
    out = _resize_kz(out, nz_old, nz_new, axis=2)
    return out


# ---------------------------------------------------------------------------
# Header patching: take the SOURCE header text verbatim and override only the
# grid / per-field shape / (optionally) time keys. Preserving the producing
# code's exact key set guarantees the output stays read-compatible.
# ---------------------------------------------------------------------------
def _patch_header(src_text, overrides, hdr_bytes=HDR_BYTES, drop=()):
    out_lines = []
    for ln in src_text.split('\n'):
        if '=' in ln:
            k = ln.split('=', 1)[0].strip()
            if k in drop:
                continue
            if k in overrides:
                out_lines.append(f"{k} = {overrides[k]}")
                continue
        out_lines.append(ln)
    text = '\n'.join(out_lines) + '\n'
    b = text.encode('ascii')
    if len(b) > hdr_bytes:
        raise ValueError(f"patched header is {len(b)} > {hdr_bytes} bytes")
    return b + b' ' * (hdr_bytes - len(b))


def _ftoa(x):
    """Full-precision, round-trippable float text (mirrors io_mod ftoa)."""
    return f"{x:.16E}"


def _shape_str(layout, n_new, nx_new, nz_new):
    ny, nkx = n_new + 1, nx_new // 2 + 1
    if layout == 'slab-kx':
        return f"{ny},{nz_new},{nkx}"          # on-disk (ny,nz,nkx)
    return f"{ny},{nkx},{nz_new}"              # serial-yxz (ny,nkx,nz)


# ---------------------------------------------------------------------------
# Low-level writer: header bytes + fields in manifest order, Fortran layout.
# ---------------------------------------------------------------------------
def _write_file(path, header_bytes, fields_on_disk):
    """fields_on_disk: ordered list of arrays already in on-disk axis order."""
    with open(path, 'wb') as f:
        f.write(header_bytes)
        for arr in fields_on_disk:
            f.write(np.asarray(arr).flatten(order='F').tobytes())


# ---------------------------------------------------------------------------
# Top-level converter.
# ---------------------------------------------------------------------------
def resample_checkpoint(src, dst, n_new, nx_new, nz_new, reset_time=False,
                        verbose=True):
    """Read SRC, resample to (n_new,nx_new,nz_new), write DST. Returns meta dict."""
    data = read_checkpoint(src)
    meta = data['meta']
    layout = meta.get('layout', 'serial-yxz')
    src_text = data['_header_text']

    ny_old = meta['ny']
    n_old, nx_old, nz_old = ny_old - 1, meta['nx'], meta['nz']

    # y-resampling here IS Chebyshev coefficient pad/truncate
    # (_resize_cheb_y / resample_volume). An fd-point checkpoint holds POINT
    # VALUES on a file grid -- padding those "coefficients" would silently
    # produce garbage that reads back cleanly. Refuse.
    hdr_kv = _parse_header(src_text)
    ybasis = hdr_kv.get('ybasis', 'cheb-T')
    if ybasis != 'cheb-T':
        raise SystemExit(
            f"{src}: REFUSED -- ybasis = {ybasis}. This tool resamples y by "
            "Chebyshev coefficient pad/truncate, which is meaningless for "
            "fd-point states (point values on a file grid). Regrid FD "
            "checkpoints with a dedicated interpolator, or rerun from an IC.")

    fields = [s.strip() for s in hdr_kv['fields'].split(',')
              if s.strip()]
    if fields != ['v', 'eta', 'u0', 'w0']:
        raise ValueError(f"{src}: unexpected fields manifest {fields!r} "
                         f"(this converter handles the v,eta,u0,w0 spectral state)")

    # resample the two spectral volumes (canonical) and the two mean profiles (y).
    v = resample_volume(_to_canonical(data['v'], layout),
                        n_old, nx_old, nz_old, n_new, nx_new, nz_new)
    eta = resample_volume(_to_canonical(data['eta'], layout),
                          n_old, nx_old, nz_old, n_new, nx_new, nz_new)
    u0 = _resize_cheb_y(np.asarray(data['u0']), n_new + 1, axis=0)
    w0 = _resize_cheb_y(np.asarray(data['w0']), n_new + 1, axis=0)

    overrides = {
        'nx': str(nx_new), 'ny': str(n_new + 1), 'nz': str(nz_new),
        'nkx': str(nx_new // 2 + 1),
        'v.shape': _shape_str(layout, n_new, nx_new, nz_new),
        'eta.shape': _shape_str(layout, n_new, nx_new, nz_new),
        'u0.shape': str(n_new + 1), 'w0.shape': str(n_new + 1),
    }
    if reset_time:
        overrides['t'] = _ftoa(0.0)
        overrides['step'] = '0'
    # The output of this tool is ALWAYS v2 (header + fields, no coordinate
    # records): _write_file writes no coord block, so a copied-through v3 header
    # ('coords = y') would misplace every field by ny*coord_bytes on read. Strip
    # the v3 keys and stamp version = 2 -- v2 is read by every consumer.
    overrides['version'] = '2'
    header_bytes = _patch_header(src_text, overrides, drop=('coords', 'coord_bytes'))

    on_disk = [_from_canonical(v, layout), _from_canonical(eta, layout), u0, w0]
    _write_file(dst, header_bytes, on_disk)

    if verbose:
        def tag(o, nw):
            return 'same' if nw == o else ('prolong' if nw > o else 'restrict')
        print(f"[resample] {src} -> {dst}   layout={layout}")
        print(f"[resample]   n  : {n_old:4d} -> {n_new:4d}  ({tag(n_old, n_new)})")
        print(f"[resample]   nx : {nx_old:4d} -> {nx_new:4d}  ({tag(nx_old, nx_new)})")
        print(f"[resample]   nz : {nz_old:4d} -> {nz_new:4d}  ({tag(nz_old, nz_new)})")
        if nx_new < nx_old or nz_new < nz_old or n_new < n_old:
            print("[resample]   note: a restrict drops high modes (lossy by design).")
        if reset_time:
            print("[resample]   t and step reset to 0 (fresh clock).")
        else:
            print(f"[resample]   carried over: t={meta.get('t')}  step={meta.get('step')}")
    return overrides


# ===========================================================================
# Self-test: index correctness, physical round-trips vs numpy's own FFTs (the
# exact ordering the solver uses), and full-file plumbing for both layouts.
# ===========================================================================
def _synth_header_text(layout, n, nx, nz, Re=2800.0, Qtar=1.0, t=1.5, dt=5e-3,
                       step=42, Lx=12.566370614359172, Lz=4.188790204786391,
                       real_bytes=8):
    """A minimal-but-complete v2 header text (mirrors the Fortran writers)."""
    ny, nkx = n + 1, nx // 2 + 1
    L = [
        'magic = ClOpDNS', 'version = 2', 'endian = little',
        f'real_bytes = {real_bytes}', 'int_bytes = 4',
        f'layout = {layout}', 'xtype = r2c', 'space = spectral',
        f'hdr_bytes = {HDR_BYTES}', 'flow = channel', 'ybc = noslip-noslip',
        f'nx = {nx}', f'ny = {ny}', f'nz = {nz}', f'nkx = {nkx}',
        f'Lx = {_ftoa(Lx)}', f'Lz = {_ftoa(Lz)}', f'Re = {_ftoa(Re)}',
        f'Qtar = {_ftoa(Qtar)}', f't = {_ftoa(t)}', f'dt = {_ftoa(dt)}',
        f'step = {step}', 'drive = CFR', 'nfields = 4', 'fields = v,eta,u0,w0',
    ]
    shp = _shape_str(layout, n, nx, nz)
    L += [f'v.shape = {shp}', 'v.kind = complex',
          f'eta.shape = {shp}', 'eta.kind = complex',
          f'u0.shape = {ny}', 'u0.kind = complex',
          f'w0.shape = {ny}', 'w0.kind = complex', 'end_header']
    return '\n'.join(L)


def _rand_valid_canonical(rng, n, nx, nz):
    """Random canonical (ny,nkx,nz) spectral volume with both Nyquists zeroed."""
    ny, nkx = n + 1, nx // 2 + 1
    a = (rng.standard_normal((ny, nkx, nz))
         + 1j * rng.standard_normal((ny, nkx, nz))).astype(np.complex128)
    a[:, nkx - 1, :] = 0.0                 # x-Nyquist (kx = nx/2)
    a[:, :, nz // 2] = 0.0                 # z-Nyquist (kz = nz/2)
    return a


def _selftest():
    rng = np.random.default_rng(20260626)
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        status = 'PASS' if cond else 'FAIL'
        ok = ok and cond
        print(f"  [{status}] {name}{('  ' + detail) if detail else ''}")

    print("== 1. coefficient-level index / round-trip ==")
    a = _rand_valid_canonical(rng, n=8, nx=16, nz=16)
    same = resample_volume(a, 8, 16, 16, 8, 16, 16)
    check("same-grid resample is bit-identical", np.array_equal(same, a))
    up = resample_volume(a, 8, 16, 16, 14, 32, 24)
    back = resample_volume(up, 14, 32, 24, 8, 16, 16)
    check("prolong then restrict recovers the original", np.array_equal(back, a),
          f"max|diff|={np.abs(back - a).max():.2e}")
    check("upscaled x-Nyquist stays zero",
          np.all(up[:, 32 // 2, :] == 0.0))
    check("upscaled z-Nyquist stays zero",
          np.all(up[:, :, 24 // 2] == 0.0))

    print("== 2. kz +/- wrap vs numpy fft (the tricky axis) ==")
    nz0, nz1 = 16, 32
    z0 = 2 * np.pi * np.arange(nz0) / nz0
    z1 = 2 * np.pi * np.arange(nz1) / nz1
    # band-limited real signal, |m| <= 4 < min(nz0,nz1)/2
    def gz(z):
        return (1.3 + 0.7 * np.cos(z) - 0.4 * np.sin(2 * z)
                + 0.9 * np.cos(3 * z + 0.3) + 0.2 * np.sin(4 * z))
    G0 = np.fft.fft(gz(z0))
    G1 = _resize_kz(G0, nz0, nz1, axis=0)
    g1 = np.fft.ifft(G1).real * (nz1 / nz0)        # ifft renormalises by length
    check("kz prolong reproduces the band-limited signal on the fine grid",
          np.allclose(g1, gz(z1), atol=1e-12),
          f"max|diff|={np.abs(g1 - gz(z1)).max():.2e}")
    # restrict nz1 -> nz0: forward-transform on the FINE grid, then coarsen.
    # (physical recon factor is N_recon/N_forward = nz0/nz1 here.)
    G1f = np.fft.fft(gz(z1))
    G0b = _resize_kz(G1f, nz1, nz0, axis=0)
    g0b = np.fft.ifft(G0b).real * (nz0 / nz1)
    check("kz restrict recovers the coarse signal",
          np.allclose(g0b, gz(z0), atol=1e-12),
          f"max|diff|={np.abs(g0b - gz(z0)).max():.2e}")

    print("== 3. kx R2C low-pass copy vs numpy rfft ==")
    nx0, nx1 = 16, 32
    x0 = 2 * np.pi * np.arange(nx0) / nx0
    x1 = 2 * np.pi * np.arange(nx1) / nx1
    def fx(x):
        return (0.5 + 1.1 * np.cos(x + 0.2) - 0.6 * np.sin(2 * x)
                + 0.3 * np.cos(3 * x))           # |kx| <= 3 < min/2
    F0 = np.fft.rfft(fx(x0))
    F1 = _resize_kx(F0, nx0, nx1, axis=0)
    f1 = np.fft.irfft(F1, n=nx1) * (nx1 / nx0)
    check("kx prolong reproduces the band-limited signal on the fine grid",
          np.allclose(f1, fx(x1), atol=1e-12),
          f"max|diff|={np.abs(f1 - fx(x1)).max():.2e}")

    print("== 4. Chebyshev y truncate/pad (polynomial round-trip) ==")
    # GLC nodes x_j = cos(pi j / n); discrete Chebyshev (DCT-I) forward; eval.
    def glc(n):
        return np.cos(np.pi * np.arange(n + 1) / n)

    def cheb_forward(u):                          # u on GLC nodes -> a_k
        n = len(u) - 1
        k = np.arange(n + 1)
        c = np.ones(n + 1); c[0] = c[n] = 2.0
        # a_k = 2/(n c_k) sum_j (1/c_j) u_j cos(pi j k / n)
        M = np.cos(np.pi * np.outer(k, np.arange(n + 1)) / n)
        return (2.0 / (n * c)) * (M @ (u / c))

    def cheb_eval(a, x):                          # sum a_k T_k(x)
        th = np.arccos(np.clip(x, -1.0, 1.0))
        k = np.arange(len(a))
        return np.cos(np.outer(th, k)) @ a

    n0 = 8
    poly = lambda x: 1.0 - 2.0 * x + 0.5 * x**2 - 0.3 * x**3 + 0.1 * x**5
    a0 = cheb_forward(poly(glc(n0)))
    a1 = _resize_cheb_y(a0, 16, axis=0)           # zero-pad to ny=16
    xt = np.linspace(-1, 1, 37)
    check("y zero-pad preserves the represented polynomial",
          np.allclose(cheb_eval(a1, xt), poly(xt), atol=1e-11),
          f"max|diff|={np.abs(cheb_eval(a1, xt) - poly(xt)).max():.2e}")

    print("== 5. full-file round-trip, both layouts ==")
    import tempfile
    for layout in ('serial-yxz', 'slab-kx'):
        with tempfile.TemporaryDirectory() as tmp:
            n, nx, nz = 6, 12, 12
            src = os.path.join(tmp, 'src.bin')
            mid = os.path.join(tmp, 'mid.bin')
            dst = os.path.join(tmp, 'dst.bin')
            vC = _rand_valid_canonical(rng, n, nx, nz)
            eC = _rand_valid_canonical(rng, n, nx, nz)
            u0 = (rng.standard_normal(n + 1) + 1j * rng.standard_normal(n + 1))
            w0 = (rng.standard_normal(n + 1) + 1j * rng.standard_normal(n + 1))
            hdr = _patch_header(_synth_header_text(layout, n, nx, nz), {})
            _write_file(src, hdr, [_from_canonical(vC, layout),
                                   _from_canonical(eC, layout), u0, w0])
            # same-grid resample must reproduce the arrays bit-for-bit
            resample_checkpoint(src, mid, n, nx, nz, verbose=False)
            d = read_checkpoint(mid)
            check(f"[{layout}] same-grid file round-trip (v)",
                  np.array_equal(_to_canonical(d['v'], layout), vC))
            check(f"[{layout}] same-grid file round-trip (eta)",
                  np.array_equal(_to_canonical(d['eta'], layout), eC))
            check(f"[{layout}] same-grid file round-trip (u0,w0)",
                  np.array_equal(d['u0'], u0) and np.array_equal(d['w0'], w0))
            check(f"[{layout}] header grid unchanged",
                  d['meta']['nx'] == nx and d['meta']['ny'] == n + 1
                  and d['meta']['nz'] == nz)
            # prolong then restrict through real files recovers the original
            resample_checkpoint(src, mid, 12, 24, 18, verbose=False)
            resample_checkpoint(mid, dst, n, nx, nz, verbose=False)
            d2 = read_checkpoint(dst)
            check(f"[{layout}] file prolong->restrict recovers v",
                  np.array_equal(_to_canonical(d2['v'], layout), vC))
            check(f"[{layout}] file prolong->restrict recovers eta",
                  np.array_equal(_to_canonical(d2['eta'], layout), eC))

    print()
    print("SELFTEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


def _main(argv):
    import argparse
    if '--selftest' in argv:
        return _selftest()
    p = argparse.ArgumentParser(
        description='Offline cross-resolution checkpoint converter '
                    '(see module docstring for the full theory).',
        epilog='Example: resample_checkpoint.py chk_lo.bin chk_hi.bin '
               '--n 128 --nx 128 --nz 128 --reset-time   '
               '|   self-check: resample_checkpoint.py --selftest')
    p.add_argument('src', help='source checkpoint (v2, serial-yxz or slab-kx)')
    p.add_argument('dst', help='destination checkpoint to write')
    p.add_argument('--n', type=int, required=True,
                   help='target Chebyshev parameter (namelist n; ny = n+1)')
    p.add_argument('--nx', type=int, required=True, help='target physical nx')
    p.add_argument('--nz', type=int, required=True, help='target physical nz')
    p.add_argument('--reset-time', action='store_true',
                   help='zero t and step in the output (fresh clock)')
    p.add_argument('--selftest', action='store_true',
                   help='run the built-in validation suite and exit')
    a = p.parse_args(argv[1:])
    resample_checkpoint(a.src, a.dst, a.n, a.nx, a.nz, reset_time=a.reset_time)
    return 0


if __name__ == '__main__':
    sys.exit(_main(sys.argv))
