#!/usr/bin/env python3
"""read_checkpoint.py -- the single, format-driven reader for ClOp DNS checkpoints.

FORMAT v2 (see io_mod.F90): a fixed-size ASCII header (1024 bytes, `head -c1024`
readable) of `key = value` lines, then a binary payload of the fields named in
the `fields =` manifest, each in the precision given by `real_bytes`, written
Fortran column-major (order='F').

Because the reader is driven ENTIRELY by the header (the field manifest, the
per-field `.shape`/`.kind`, the precision, the grid, the flow/BC tags), it needs
NO hard-coded knowledge of which fields a file contains. Add a temperature field
or physical u,v,w dumps later -> they appear in `fields=` and this same reader
returns them by name, no code change. One reader for every file the code writes.

Usage:
    from read_checkpoint import read_checkpoint
    data = read_checkpoint('chk_kmm.bin')
    v   = data['v']        # complex ndarray, shape (ny, nx, nz), Fortran order
    eta = data['eta']
    meta = data['meta']    # dict: flow, ybc, drive, Re, Qtar, t, dt, step, Lx, Lz, ...

    # command line: print the header + a summary
    python3 read_checkpoint.py chk_kmm.bin
"""
import sys
import numpy as np


def _parse_header(raw):
    """raw: the decoded ASCII header string. Returns {key: value_string}."""
    h = {}
    for line in raw.split('\n'):
        if '=' in line:
            k, v = line.split('=', 1)
            k = k.strip()
            if k:
                h[k] = v.strip()
    return h


def read_checkpoint(path, mmap=True):
    """Read a v2 checkpoint. Returns {field_name: ndarray, ..., 'meta': {...}}."""
    with open(path, 'rb') as f:
        # The header size is itself in the header; 1024 is the v2 default, but
        # read hdr_bytes robustly: peek a generous chunk, parse, then re-seek.
        head = f.read(4096)
        # locate hdr_bytes from the first parse of up to 1024 bytes
        prelim = _parse_header(head[:1024].decode('ascii', errors='replace'))
        if prelim.get('magic') != 'ClOpDNS':
            raise ValueError(f"{path}: not a ClOpDNS checkpoint "
                             f"(magic={prelim.get('magic')!r}); v1 files are unsupported")
        ver = int(prelim.get('version', '0'))
        if ver not in (2, 3):
            raise ValueError(f"{path}: unsupported format version {ver} (this reader handles v2/v3)")
        hdr_bytes = int(prelim.get('hdr_bytes', '1024'))
        raw = head[:hdr_bytes].decode('ascii', errors='replace')
        h = _parse_header(raw)

        # Known y-discretisations: cheb-T/glc (y-records are
        # Chebyshev-T COEFFICIENT columns) and fd-point/file (y-records are point
        # VALUES on the nodes stored in the v3 y coordinate record). Anything else
        # is refused. Absent keys = legacy file (all legacy = cheb-T/glc). The
        # caller distinguishes the two via header['ybasis'] in the return value.
        if h.get('ybasis', 'cheb-T') not in ('cheb-T', 'fd-point'):
            raise ValueError(f"{path}: unsupported ybasis={h['ybasis']!r} "
                             "(this reader understands cheb-T and fd-point)")
        if h.get('ygrid', 'glc') not in ('glc', 'file'):
            raise ValueError(f"{path}: unsupported ygrid={h['ygrid']!r} "
                             "(this reader understands glc and file grids)")

        real_bytes = int(h['real_bytes'])
        cdtype = np.complex128 if real_bytes == 8 else np.complex64
        rdtype = np.float64 if real_bytes == 8 else np.float32

        endian = h.get('endian', 'little')
        if endian not in ('little', 'big'):
            raise ValueError(f"{path}: bad endian tag {endian!r}")
        # numpy byte-order prefix
        bo = '<' if endian == 'little' else '>'
        cdtype = np.dtype(cdtype).newbyteorder(bo)
        rdtype = np.dtype(rdtype).newbyteorder(bo)

        fields = [s.strip() for s in h['fields'].split(',') if s.strip()]
        out = {}
        # v3 coordinate records between the header and the fields, declared
        # by 'coords' (comma list; x->nx, y->ny, z->nz points each) + 'coord_bytes'
        # (8 = float64, independent of the payload precision). Checkpoints store
        # 'coords = y' -- the wall-normal NODES, which for an FD-era file are the
        # only trustworthy grid definition. v2 files have no 'coords' key: zero
        # shift, and y is RECONSTRUCTED from 'ygrid = glc' below.
        axis_len = {'x': int(h.get('nx', 0)), 'y': int(h.get('ny', 0)), 'z': int(h.get('nz', 0))}
        cnames = [s.strip() for s in h.get('coords', '').split(',') if s.strip()]
        cbyt = int(h.get('coord_bytes', '8'))
        cdt = np.dtype(np.float64 if cbyt == 8 else np.float32).newbyteorder(bo)
        coords = {}
        off = hdr_bytes
        for nm in cnames:
            f.seek(off)
            buf = f.read(axis_len[nm]*cbyt)
            if len(buf) != axis_len[nm]*cbyt:
                raise ValueError(f"{path}: short read for coord {nm} "
                                 f"({len(buf)} of {axis_len[nm]*cbyt} bytes)")
            coords[nm] = np.frombuffer(buf, dtype=cdt).astype(np.float64)
            off += axis_len[nm]*cbyt
        if 'y' not in coords and h.get('ygrid', 'glc') == 'glc' and axis_len['y'] > 1:
            coords['y'] = np.cos(np.pi*np.arange(axis_len['y'])/(axis_len['y'] - 1))
        if 'y' not in coords and h.get('ygrid', 'glc') == 'file':
            # an fd-point file without a y record has no reconstructible
            # grid (the nodes came from an external ygrid file) -> refuse.
            raise ValueError(f"{path}: ygrid=file but no y coordinate record (v2?)")
        # Fields are LAZY read-only memmaps by default: at 10 G-pt-class
        # checkpoints the eager read needs 100s of GB of RAM; a memmap loads only
        # the blocks you touch (slice per kx-band to stream). Pass mmap=False for
        # eager in-RAM arrays (writable copies).
        for nm in fields:
            shape = tuple(int(x) for x in h[f'{nm}.shape'].split(','))
            kind = h.get(f'{nm}.kind', 'complex')
            dt = cdtype if kind == 'complex' else rdtype
            count = int(np.prod(shape))
            nbytes = count * np.dtype(dt).itemsize
            if mmap:
                out[nm] = np.memmap(path, mode='r', dtype=dt, offset=off,
                                    shape=shape, order='F')
            else:
                f.seek(off)
                out[nm] = np.frombuffer(f.read(nbytes), dtype=dt).reshape(shape, order='F').copy()
            off += nbytes

    meta = {}
    for k in ('flow', 'ybc', 'drive', 'space', 'layout', 'endian', 'xtype', 'ybasis', 'ygrid'):
        if k in h:
            meta[k] = h[k]
    for k in ('Re', 'Qtar', 't', 'dt', 'Lx', 'Lz'):
        if k in h:
            meta[k] = float(h[k])
    for k in ('nx', 'ny', 'nz', 'nkx', 'step', 'nfields'):
        if k in h:
            meta[k] = int(h[k])
    # Precision provenance (v2): real_bytes = wp = the payload precision AND the
    # reader's gate (it drives the dtype above); tp_bytes/dp_bytes are informational
    # tier sizes (tp=4 => fp32 transforms). Checkpoints written
    # before these keys existed simply omit them -- the `in h` guard keeps old files
    # readable unchanged.
    for k in ('real_bytes', 'tp_bytes', 'dp_bytes', 'int_bytes'):
        if k in h:
            meta[k] = int(h[k])
    out['meta'] = meta
    out['coords'] = coords          # stored (v3) or reconstructed-GLC (v2) nodes
    out['_header_text'] = raw.rstrip()
    return out


def _main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    for path in argv[1:]:
        d = read_checkpoint(path)
        print(f"=== {path} ===")
        print(d['_header_text'])
        m = d['meta']
        if 'real_bytes' in m:
            tiers = f"wp(field/payload)={m['real_bytes']}B"
            if 'tp_bytes' in m: tiers += f"  tp(transform)={m['tp_bytes']}B"
            if 'dp_bytes' in m: tiers += f"  dp(solve)={m['dp_bytes']}B"
            print(f"--- precision tiers: {tiers} ---")
        if d['coords']:
            src = 'stored (v3)' if 'coords' in d['_header_text'] else 'reconstructed (v2)'
            for nm, c in d['coords'].items():
                print(f"--- coord {nm}: n={c.size}  [{c.min():.6f}, {c.max():.6f}]  ({src}) ---")
        print("--- arrays ---")
        for k, v in d.items():
            if k in ('meta', '_header_text', 'coords'):
                continue
            print(f"  {k:8s} shape={v.shape}  dtype={v.dtype}  "
                  f"|max|={np.abs(v).max():.6e}")
    return 0


if __name__ == '__main__':
    sys.exit(_main(sys.argv))
