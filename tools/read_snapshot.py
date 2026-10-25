#!/usr/bin/env python3
"""
read_snapshot.py -- read a CLAUSON physical field snapshot (u,v,w and optional p).

Snapshots are written by par_snap_write (layout=slab-z, space=physical): a fixed
1024-byte ASCII "KEY = VALUE" header, then the real fields listed in the header's
'fields' manifest (u,v,w, and p when snap_pressure was on), each on the clean
nx*ny*nz grid in Fortran (column-major) order -- y fastest, x, z slowest -- at the
precision in the header (real_bytes = 4 fp32 / 8 fp64). No inverse FFT needed.

Usage:
    python3 tools/read_snapshot.py snap_dp_step00000100_t0.500.bin
    # or, to find the newest:  python3 tools/read_snapshot.py "$(cat latest_snapshot.txt)"

As a library:
    from read_snapshot import read_snapshot, read_snapshot_fields
    hdr, u, v, w = read_snapshot(path)          # u,v,w are (ny, nx, nz) numpy arrays
    hdr, flds = read_snapshot_fields(path)      # dict of ALL fields (incl. p if present)
    hdr, coords, flds = read_snapshot_fields(path, return_coords=True)
        # coords = {'x': (nx,), 'y': (ny,), 'z': (nz,)} float64. Format v3 stores
        # them in the file (header 'coords' manifest); for v2 files they are
        # RECONSTRUCTED (uniform x,z from Lx,Lz; GLC y from 'ygrid = glc') --
        # same call, either vintage.
"""
import sys
import numpy as np


AXIS_LEN = {"x": "nx", "y": "ny", "z": "nz"}    # coord name -> axis length key


def _coord_block(hdr):
    """(names, total bytes) of the v3 coordinate records; ([], 0) for v2."""
    names = [s.strip() for s in hdr.get("coords", "").split(",") if s.strip()]
    cbyt = int(hdr.get("coord_bytes", "8"))
    nbytes = sum(int(hdr[AXIS_LEN[nm]]) for nm in names) * cbyt
    return names, nbytes


def read_snapshot_fields(path, return_coords=False):
    """Return (header dict, {name: (ny,nx,nz) array}) for every field in the
    'fields' manifest (u,v,w and optionally p). With return_coords=True the
    return is (hdr, coords, fields) -- stored coordinates for v3 files,
    reconstructed ones for v2."""
    with open(path, "rb") as f:
        raw = f.read(1024)
        hdr = {}
        for line in raw.split(b"\n"):
            if b"=" in line:
                k, _, val = line.partition(b"=")
                hdr[k.strip().decode(errors="replace")] = val.strip().decode(errors="replace")
        if hdr.get("magic") != "ClOpDNS":
            raise ValueError(f"{path}: not a CLAUSON file (magic={hdr.get('magic')!r})")
        ver = int(hdr.get("version", "2"))
        if ver not in (2, 3):
            raise ValueError(f"{path}: unsupported format version {ver} (this reader handles v2/v3)")
        if hdr.get("space") != "physical":
            raise ValueError(f"{path}: not a physical snapshot (space={hdr.get('space')!r})")
        names = [s for s in hdr.get("fields", "u,v,w").split(",") if s]
        if names[:3] != ["u", "v", "w"]:
            raise ValueError(f"{path}: unexpected fields manifest {hdr.get('fields')!r}")
        nx, ny, nz = int(hdr["nx"]), int(hdr["ny"]), int(hdr["nz"])
        rb = int(hdr["real_bytes"])
        dtype = np.float32 if rb == 4 else np.float64
        count = ny * nx * nz

        # v3: coordinate records between header and fields (always float64);
        # v2: nothing stored -> reconstruct below. Either way, fields start at
        # 1024 + coord block size.
        cnames, cbytes = _coord_block(hdr)
        coords = {}
        f.seek(1024)
        cdt = np.dtype(np.float64) if int(hdr.get("coord_bytes", "8")) == 8 else np.dtype(np.float32)
        for nm in cnames:
            npts = int(hdr[AXIS_LEN[nm]])
            a = np.fromfile(f, dtype=cdt, count=npts)
            if a.size != npts:
                raise ValueError(f"{path}: short read for coord {nm} ({a.size} of {npts})")
            coords[nm] = a.astype(np.float64)
        if return_coords:                       # v2 (or partial manifest): reconstruct
            if "x" not in coords:
                coords["x"] = np.arange(nx, dtype=np.float64) * float(hdr["Lx"]) / nx
            if "z" not in coords:
                coords["z"] = np.arange(nz, dtype=np.float64) * float(hdr["Lz"]) / nz
            if "y" not in coords:
                if hdr.get("ygrid", "glc") != "glc":
                    raise ValueError(f"{path}: no stored y coords and ygrid={hdr.get('ygrid')!r}"
                                     " is not reconstructible (need a v3 file)")
                coords["y"] = np.cos(np.pi * np.arange(ny, dtype=np.float64) / (ny - 1))

        f.seek(1024 + cbytes)
        out = {}
        for name in names:
            a = np.fromfile(f, dtype=dtype, count=count)
            if a.size != count:
                raise ValueError(f"{path}: short read for field {name} ({a.size} of {count})")
            out[name] = a.reshape((ny, nx, nz), order="F")   # on-disk: y fastest, z slowest
        if return_coords:
            return hdr, coords, out
        return hdr, out


def read_snapshot(path):
    """Return (header dict, u, v, w). Each field is (ny, nx, nz), Fortran-ordered.
    (Back-compat helper; use read_snapshot_fields to also get p when present.)"""
    hdr, flds = read_snapshot_fields(path)
    return hdr, flds["u"], flds["v"], flds["w"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    hdr, coords, flds = read_snapshot_fields(path, return_coords=True)
    print(f"snapshot: {path}")
    for k in ("version", "space", "layout", "real_bytes", "coords", "nfields", "fields",
              "nx", "ny", "nz", "Lx", "Lz", "Re", "t", "step"):
        if k in hdr:
            print(f"   {k:10s} = {hdr[k]}")
    src = "stored (v3)" if "coords" in hdr else "reconstructed (v2)"
    for nm in ("x", "y", "z"):
        c = coords[nm]
        print(f"   coord {nm}: n={c.size}  [{c.min(): .6f}, {c.max(): .6f}]  ({src})")
    for name, a in flds.items():
        print(f"   {name}: shape={a.shape}  min={a.min(): .4e}  max={a.max(): .4e}  "
              f"mean={a.mean(): .4e}  finite={bool(np.isfinite(a).all())}")
