#!/usr/bin/env python3
"""work_galilean_model.py -- verify the Galilean-frame algebra the solver relies on.

The CLAUSON Galilean feature subtracts a constant streamwise speed U_c from the
MEAN streamwise velocity (the (0,0) Fourier mode) before forming the rotational
nonlinear term H = u x omega, so advection and the CFL see u - U_c. This numpy
model confirms, on a divergence-free field, the three facts that make that a
correct, frame-invariant change (so it is safe to apply in the channel solver):

  1. omega = curl u is UNCHANGED by the constant mean shift (curl of a constant
     is zero) -- so vorticity may be formed from the lab u, as the code does.
  2. The MEAN forcing of the (v, eta) mean equations is UNCHANGED:
       - Hx does not contain u_x at all  ->  <Hx> identical;
       - <Hz' - Hz> = -U_c <omega_y> = 0  (the (0,0) mode of omega_y vanishes).
     So the lab-frame mean u0/w0 (and hence the CFR drive and Re_tau) evolve
     identically -- the shift only rotates the fluctuation phases (Galilean).
  3. The recovered PRESSURE is frame-invariant (up to an additive constant):
     solving (Pi = p + |u|^2/2) by Poisson with the moving-frame H and moving-
     frame |u|^2 returns the same physical p. p' - p is spatially constant.

A tri-periodic box is used because the Galilean algebra is geometry-independent;
the channel wall BCs are exercised end-to-end (bit-exact) by the solver's own
--validate (distributed vs single-GPU oracle, with u_galilean set in both).
"""
import numpy as np


def _build(N=16, seed=7):
    """A real, divergence-free velocity u plus a y-dependent streamwise mean."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(N, d=1.0 / N)                  # integer wavenumbers
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0

    def rfield():
        a = rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N))
        a[K2 > (N // 3)**2] = 0.0                      # band-limit (smooth)
        return a

    a, b, c = rfield(), rfield(), rfield()
    # project to divergence-free: u <- u - k (k.u)/|k|^2
    kdu = (KX * a + KY * b + KZ * c) / K2
    uh = a - KX * kdu
    vh = b - KY * kdu
    wh = c - KZ * kdu
    uh[0, 0, 0] = vh[0, 0, 0] = wh[0, 0, 0] = 0.0
    u = np.fft.ifftn(uh).real
    v = np.fft.ifftn(vh).real
    w = np.fft.ifftn(wh).real
    # add a y-dependent streamwise MEAN U(y) (the channel's (kx=0,kz=0) profile)
    yy = np.arange(N) * (2 * np.pi / N)
    U = 1.0 + 0.4 * np.cos(yy)                          # varies in y only
    u = u + U[None, :, None]
    return u, v, w, (KX, KY, KZ, K2)


def _curl(u, v, w, K):
    KX, KY, KZ, _ = K
    uh, vh, wh = np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)
    ox = np.fft.ifftn(1j * (KY * wh - KZ * vh)).real
    oy = np.fft.ifftn(1j * (KZ * uh - KX * wh)).real
    oz = np.fft.ifftn(1j * (KX * vh - KY * uh)).real
    return ox, oy, oz


def _cross(u, v, w, ox, oy, oz):
    return (v * oz - w * oy,          # Hx
            w * ox - u * oz,          # Hy
            u * oy - v * ox)          # Hz


def _pressure(Hx, Hy, Hz, u, v, w, K):
    """p = Pi - |u|^2/2, with Pi from  lap(Pi) = div(H)  (zero-mean gauge)."""
    KX, KY, KZ, K2 = K
    div = 1j * (KX * np.fft.fftn(Hx) + KY * np.fft.fftn(Hy) + KZ * np.fft.fftn(Hz))
    Pih = -div / K2                    # lap = -k^2 ; (0,0,0) gauge below
    Pih[0, 0, 0] = 0.0
    Pi = np.fft.ifftn(Pih).real
    return Pi - 0.5 * (u**2 + v**2 + w**2)


def _xzmean(f):
    """the (kx=0,kz=0) profile in y -- the channel mean operator's analogue."""
    return f.mean(axis=(0, 2))


def main():
    Uc = 0.55
    u, v, w, K = _build()
    up = u - Uc                         # Galilean: subtract U_c from streamwise u

    ox, oy, oz = _curl(u, v, w, K)
    oxp, oyp, ozp = _curl(up, v, w, K)
    Hx, Hy, Hz = _cross(u, v, w, ox, oy, oz)
    Hxp, Hyp, Hzp = _cross(up, v, w, oxp, oyp, ozp)

    ok = True
    def chk(name, val, tol=1e-11):
        nonlocal ok
        p = val < tol
        ok = ok and p
        print(f"  [{'PASS' if p else 'FAIL'}] {name:54s} {val:.2e}")

    print("== 1. vorticity unchanged by the constant mean shift ==")
    chk("max|omega' - omega|", max(np.abs(oxp - ox).max(),
                                   np.abs(oyp - oy).max(), np.abs(ozp - oz).max()))

    print("== 2. mean (v,eta) forcing unchanged (lab u0/w0 evolve identically) ==")
    chk("max|Hx' - Hx|  (Hx has no u_x)", np.abs(Hxp - Hx).max())
    chk("max|<Hx'> - <Hx>|_(0,0) profile", np.abs(_xzmean(Hxp) - _xzmean(Hx)).max())
    chk("max|<Hz'> - <Hz>|_(0,0) profile", np.abs(_xzmean(Hzp) - _xzmean(Hz)).max())
    chk("max|<omega_y>|_(0,0)  (=0 => the Hz shift averages out)",
        np.abs(_xzmean(oy)).max())

    print("== 3. recovered pressure is frame-invariant (up to a constant) ==")
    p = _pressure(Hx, Hy, Hz, u, v, w, K)
    pp = _pressure(Hxp, Hyp, Hzp, up, v, w, K)
    diff = pp - p
    chk("std(p' - p)  (difference is a pure gauge constant)", diff.std())
    # fluctuating pressure (mean-removed) is exactly equal -> stats frame-invariant
    chk("max|(p'-<p'>) - (p-<p>)|", np.abs((pp - pp.mean()) - (p - p.mean())).max())

    print("\nGALILEAN MODEL:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
