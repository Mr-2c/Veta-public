#!/usr/bin/env python3
"""
analyze_stats.py -- channel-DNS statistics post-processing & convergence audit.

Usage:  python3 analyze_stats.py run1.dat [run2.dat ...] [--mkm [reystress]] [--png out.png]

  --mkm [path]  overlay the Moser-Kim-Mansour (1999) chan180 Re_tau=178.12
                reference. The mean profile U+(y+) is built in; for the
                Reynolds-stress overlay, pass the path to chan180.reystress
                (downloadable from turbulence.oden.utexas.edu/data/MKM/chan180/
                profiles/) -- e.g. --mkm chan180.reystress. With --mkm the
                figure gains a U+(y+) log-law panel.

For each stats file (format written by stats_mod):
  * Re-derives u_tau from the TOTAL-STRESS LINEAR FIT  T(y) = (1/Re)dU/dy - <u'v'>
    = -u_tau^2 * y  (least squares through the origin). This is the
    self-consistent normalisation: the file header's Re_tau is the FINAL
    INSTANTANEOUS value (can sit +-2% off the window mean), so renormalising
    by the stress fit removes a spurious O(2-4%) shift in the + quantities.
  * Reports the linearity RESIDUAL (rms, % of u_tau^2) -- the momentum-balance
    convergence metric. Healthy runs at T_avg ~ 100 sit near ~1%; it shrinks
    ~ 1/sqrt(T_avg).
  * Reports the WALL-TO-WALL ASYMMETRY of the uu peak -- the two walls are
    statistically identical, so this is a direct INTERNAL estimate of the
    run's sampling error. Compare cross-run differences against it: if they
    are the same size, the runs agree to within their own noise.
  * Builds TWO-WALL-AVERAGED profiles (even quantities mirrored, uv mirrored
    with sign), halving the variance for free.

With 2+ files it prints a cross-run comparison (renormalised peaks, U+ at the
centreline, Re_tau) and writes a figure: total-stress check + renormalised
two-wall-averaged profiles.

y-derivative: exact Chebyshev WHEN THE NODES ARE GAUSS-LOBATTO -- which is
decided from the file's own y column, not assumed. Stats files store point
values in BOTH y-bases, so a fd-point run on a GLC-as-file grid still gets
the exact transform derivative; only a genuinely non-GLC grid falls back to
2nd-order nonuniform FD (np.gradient), with a printed [fd] note -- the wall
gradient is steep, so treat that branch's stress fit as approximate. The
basis is read from the '#  ybasis =' header stamp (absent = pre-stamp era,
all Chebyshev).
"""
import sys
import numpy as np

def load(fn):
    with open(fn) as f:
        lines = f.readlines()
    hdr = lines[1]
    Re      = float(hdr.split('Re =')[1].split('Re_tau')[0])
    Rt_hdr  = float(hdr.split('Re_tau =')[1].split('u_tau')[0])
    ut_hdr  = float(hdr.split('u_tau =')[1].split('nsamp')[0])
    nsamp   = int(hdr.split('nsamp =')[1])
    ybasis  = 'cheb-T'                    # pre-stamp files were all Chebyshev runs
    for ln in lines[:10]:
        if ln.startswith('#') and 'ybasis =' in ln:
            ybasis = ln.split('ybasis =')[1].split()[0].strip()
            break
    d = np.loadtxt(fn)
    return dict(fn=fn, Re=Re, Rt_hdr=Rt_hdr, ut_hdr=ut_hdr, nsamp=nsamp,
                ybasis=ybasis,
                y=d[:,0], yp=d[:,1], Up=d[:,2], uu=d[:,3], vv=d[:,4],
                ww=d[:,5], uv=d[:,6], Uraw=d[:,7])

# ---------------------------------------------------------------------------
# MKM (Moser-Kim-Mansour 1999) Re_tau = 178.12 reference, chan180.
# Source: turbulence.oden.utexas.edu/data/MKM/chan180/profiles/  (means embedded
# below; Reynolds stresses optionally loaded from chan180.reystress -- see
# mkm_reference()).  Mean profile is (y, y+, U+), 65 points, half-channel.
_MKM_MEANS = """
0.0 0.0 0.0
0.00030118 0.053648 0.053639
0.0012045 0.21456 0.21443
0.0027095 0.48263 0.48197
0.0048153 0.85771 0.85555
0.0075205 1.3396 1.3339
0.010823 1.9279 1.9148
0.014722 2.6224 2.5939
0.019215 3.4226 3.3632
0.024298 4.3280 4.2095
0.029969 5.3381 5.1133
0.036224 6.4523 6.0493
0.043060 7.6700 6.9892
0.050472 8.9902 7.9052
0.058456 10.412 8.7741
0.067007 11.936 9.5790
0.076120 13.559 10.311
0.085790 15.281 10.967
0.096011 17.102 11.550
0.10678 19.019 12.066
0.11808 21.033 12.520
0.12991 23.141 12.921
0.14227 25.342 13.276
0.15515 27.635 13.590
0.16853 30.019 13.870
0.18242 32.492 14.121
0.19679 35.053 14.349
0.21165 37.701 14.557
0.22699 40.432 14.750
0.24279 43.247 14.931
0.25905 46.143 15.101
0.27575 49.118 15.264
0.29289 52.171 15.419
0.31046 55.300 15.569
0.32844 58.503 15.714
0.34683 61.778 15.855
0.36561 65.123 15.993
0.38477 68.536 16.128
0.40430 72.016 16.260
0.42419 75.559 16.389
0.44443 79.164 16.515
0.46500 82.828 16.637
0.48590 86.550 16.756
0.50710 90.327 16.872
0.52860 94.157 16.985
0.55039 98.037 17.094
0.57244 101.97 17.200
0.59476 105.94 17.302
0.61732 109.96 17.400
0.64010 114.02 17.494
0.66311 118.12 17.585
0.68632 122.25 17.672
0.70972 126.42 17.756
0.73329 130.62 17.835
0.75702 134.84 17.911
0.78090 139.10 17.981
0.80491 143.37 18.045
0.82904 147.67 18.103
0.85327 151.99 18.154
0.87759 156.32 18.198
0.90198 160.66 18.235
0.92644 165.02 18.264
0.95093 169.38 18.285
0.97546 173.75 18.297
1.0 178.12 18.301
"""

def mkm_reference(reystress_path=None):
    """MKM chan180 reference. Returns dict with y+, U+ (always, embedded) and,
    if chan180.reystress is provided/found, uu+ vv+ ww+ -uv+ (two-wall folded,
    half-channel). The .reystress columns are: y y+ R_uu R_vv R_ww R_uv R_uw R_vw
    in U_tau^2 units -- i.e. already the + stresses."""
    m = np.array([[float(x) for x in ln.split()]
                  for ln in _MKM_MEANS.strip().splitlines()])
    ref = dict(yp=m[:, 1], Up=m[:, 2])
    import os
    cand = reystress_path or 'chan180.reystress'
    if os.path.exists(cand):
        d = np.loadtxt(cand, comments='#')
        ref['st_yp'] = d[:, 1]
        ref['uu'] = d[:, 2]; ref['vv'] = d[:, 3]
        ref['ww'] = d[:, 4]; ref['muv'] = -d[:, 5]   # plot -uv+ like the run
    return ref

def is_glc(y):
    """True when the file's y column IS the GLC node set cos(pi j/n) to
    rounding -- the exactness condition for cheb_deriv_vals, independent of
    which y-BASIS the run used (stats store point values in both)."""
    n = len(y) - 1
    return bool(np.allclose(y, np.cos(np.pi*np.arange(n+1)/n), atol=1e-9))

def cheb_deriv_vals(f):
    """df/dy on GLC nodes y_j = cos(pi j / n), j = 0..n (descending y)."""
    n = len(f) - 1
    F = np.concatenate([f, f[-2:0:-1]])              # even extension
    a = np.real(np.fft.fft(F))[:n+1] / n
    a[0] /= 2.0; a[n] /= 2.0
    b = np.zeros(n+1)
    for k in range(n-1, -1, -1):
        b[k] = (b[k+2] if k+2 <= n else 0.0) + 2.0*(k+1)*a[k+1]
    b[0] /= 2.0
    th = np.arange(n+1)*np.pi/n
    return np.cos(np.outer(th, np.arange(n+1))) @ b

def analyze(r):
    ut2_h = r['ut_hdr']**2
    uv_raw = r['uv']*ut2_h
    y      = r['y']
    # d<U>/dy branch decided by the NODES (see module docstring):
    # GLC nodes (either basis) -> exact transform derivative; non-GLC file
    # grid -> 2nd-order nonuniform FD, flagged so the choice is visible
    # (applying cheb_deriv_vals unconditionally would silently mangle such
    # files).
    if is_glc(y):
        dUdy = cheb_deriv_vals(r['Uraw'])
    else:
        dUdy = np.gradient(r['Uraw'], y, edge_order=2)
        print(f"  [fd] {r['fn']}: non-GLC y-grid (ybasis = {r['ybasis']}) -> "
              "nonuniform-FD d<U>/dy; stress fit is approximate near the wall")
    if not np.allclose(y, -y[::-1], atol=1e-9):
        print(f"  [warn] {r['fn']}: y-grid is not index-symmetric -- the "
              "two-wall averages below fold mismatched stations")
    T      = dUdy/r['Re'] - uv_raw                   # total stress, raw units
    ut2    = -np.sum(T*y)/np.sum(y*y)                # fit T = -ut2*y
    resid  = np.sqrt(np.mean((T + ut2*y)**2))/ut2
    r['ut2_fit'] = ut2
    r['Rt_fit']  = r['Re']*np.sqrt(ut2)
    r['res_pct'] = 100.0*resid
    r['Ttot']    = T/ut2
    # renormalised + quantities (consistent: stress-fit u_tau)
    s = ut2_h/ut2
    for k in ('uu','vv','ww','uv'):
        r[k+'_f'] = r[k]*s
    r['Up_f'] = r['Uraw']/np.sqrt(ut2)
    # two-wall average: GLC grid is y-symmetric by index
    n = len(y) - 1
    mir = lambda f, sgn: 0.5*(f + sgn*f[::-1])
    half = slice(n//2, n+1)                          # y in [-1, 0] ... index n/2..n
    r['ya']  = -y[half]                              # 0 -> 1 wall distance side
    r['ypa'] = (1.0 - np.abs(y[half]))*r['Rt_fit']
    r['uu_a'] = mir(r['uu_f'], +1)[half]
    r['vv_a'] = mir(r['vv_f'], +1)[half]
    r['ww_a'] = mir(r['ww_f'], +1)[half]
    r['uv_a'] = mir(r['uv_f'], -1)[half]             # odd in y
    r['Up_a'] = mir(r['Up_f'], +1)[half]
    # wall-to-wall asymmetry of the uu peak (internal sampling-error estimate)
    lo = r['uu_f'][y < 0].max();  hi = r['uu_f'][y > 0].max()
    r['asym_pct'] = 100.0*abs(lo-hi)/(0.5*(lo+hi))
    r['pk_lo'], r['pk_hi'] = lo, hi
    return r

def main(argv):
    png = None
    mkm = None            # --mkm [path]: overlay MKM chan180. Optional arg =
    if '--mkm' in argv:   # path to chan180.reystress (else means-only overlay).
        i = argv.index('--mkm')
        if i+1 < len(argv) and not argv[i+1].endswith('.dat') and argv[i+1] != '--png':
            mkm = argv[i+1]; argv = argv[:i]+argv[i+2:]
        else:
            mkm = ''; argv = argv[:i]+argv[i+1:]
    if '--png' in argv:
        i = argv.index('--png'); png = argv[i+1]; argv = argv[:i]+argv[i+2:]
    runs = [analyze(load(fn)) for fn in argv]
    print(f"{'file':<28}{'nsamp':>6}{'Rt(hdr)':>9}{'Rt(fit)':>9}"
          f"{'stress res%':>12}{'wall asym%':>11}{'uu+ pk':>8}")
    for r in runs:
        pk = 0.5*(r['pk_lo']+r['pk_hi'])
        print(f"{r['fn'].split('/')[-1]:<28}{r['nsamp']:>6}{r['Rt_hdr']:>9.3f}"
              f"{r['Rt_fit']:>9.3f}{r['res_pct']:>12.2f}{r['asym_pct']:>11.2f}{pk:>8.3f}")
    if len(runs) >= 2:
        a, b = runs[0], runs[1]
        d = lambda x, y: 100.0*abs(x-y)/(0.5*(x+y))
        print("\ncross-run (stress-fit renormalised, two-wall-averaged):")
        print(f"  uu+ peak     : {a['uu_a'].max():.3f} vs {b['uu_a'].max():.3f}"
              f"   diff {d(a['uu_a'].max(), b['uu_a'].max()):.2f}%"
              f"   (internal noise {a['asym_pct']:.1f}% / {b['asym_pct']:.1f}%)")
        print(f"  U+ centreline: {a['Up_a'][0]:.3f} vs {b['Up_a'][0]:.3f}"
              f"   diff {d(a['Up_a'][0], b['Up_a'][0]):.2f}%")
        print(f"  Re_tau (fit) : {a['Rt_fit']:.2f} vs {b['Rt_fit']:.2f}"
              f"   diff {d(a['Rt_fit'], b['Rt_fit']):.2f}%")
        print("  VERDICT: if cross-run diffs <~ internal noise, the runs agree"
              " to within sampling error.")
    if png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ref = mkm_reference(mkm if mkm else None) if mkm is not None else None
        ncol = 3 if ref is not None else 2
        fig, ax = plt.subplots(1, ncol, figsize=(5.5*ncol, 4.2))
        sty = ['-', ':', '--', '-.']
        # --- panel 0: total-stress momentum balance --------------------------
        for i, r in enumerate(runs):
            lab = r['fn'].split('/')[-1].replace('stats_turb_','').replace('.dat','')
            ax[0].plot(r['y'], r['Ttot'], sty[i % 4], color='k', label=lab)
        ax[0].plot([-1, 1], [1, -1], 'r-', lw=0.8, label='exact -y')
        ax[0].set_xlabel('y'); ax[0].set_ylabel('total stress / u_tau^2')
        ax[0].set_title('momentum-balance check'); ax[0].legend(fontsize=8)
        # --- panel 1: mean velocity U+(y+) (log law) -------------------------
        if ref is not None:
            for i, r in enumerate(runs):
                lab = r['fn'].split('/')[-1].replace('stats_turb_','').replace('.dat','')
                ax[1].semilogx(r['ypa'][1:], r['Up_a'][1:], sty[i % 4], color='k',
                               label='U+ '+lab)
            ax[1].semilogx(ref['yp'][1:], ref['Up'][1:], 'o', ms=3.5,
                           mfc='none', color='r', label='MKM 1999')
            yv = np.logspace(0, np.log10(11), 20); ax[1].semilogx(yv, yv, '0.7', lw=0.7)
            yl = np.logspace(np.log10(11), np.log10(180), 30)
            ax[1].semilogx(yl, np.log(yl)/0.41 + 5.2, '0.7', lw=0.7,
                           label='u+=y+, (1/.41)ln y+ +5.2')
            ax[1].set_xlabel('y+'); ax[1].set_ylabel('U+')
            ax[1].set_title('mean velocity vs MKM'); ax[1].legend(fontsize=8)
            sp = 2
        else:
            sp = 1
        # --- last panel: Reynolds stresses (+ MKM if present) ----------------
        for i, r in enumerate(runs):
            lab = r['fn'].split('/')[-1].replace('stats_turb_','').replace('.dat','')
            ax[sp].semilogx(r['ypa'][1:], r['uu_a'][1:], sty[i % 4], color='k', label='uu+ '+lab)
            ax[sp].semilogx(r['ypa'][1:], r['vv_a'][1:], sty[i % 4], color='0.45')
            ax[sp].semilogx(r['ypa'][1:], r['ww_a'][1:], sty[i % 4], color='0.45')
            ax[sp].semilogx(r['ypa'][1:], -r['uv_a'][1:], sty[i % 4], color='0.45')
        if ref is not None and 'uu' in ref:
            ax[sp].semilogx(ref['st_yp'], ref['uu'], 'o', ms=3, mfc='none', color='r', label='MKM')
            for key in ('vv', 'ww', 'muv'):
                ax[sp].semilogx(ref['st_yp'], ref[key], 'o', ms=3, mfc='none', color='r')
        elif ref is not None:
            ax[sp].text(0.5, 0.95, '(stresses: add chan180.reystress for overlay)',
                        transform=ax[sp].transAxes, ha='center', va='top', fontsize=7, color='r')
        ax[sp].set_xlabel('y+'); ax[sp].set_ylabel('(.)+  [stress-fit u_tau, two-wall avg]')
        ax[sp].set_title('uu+ (black), vv+ ww+ -uv+ (grey)'); ax[sp].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(png, dpi=140)
        print(f"\nfigure -> {png}")
        if mkm is not None and ref is not None and 'uu' not in ref:
            print("  (MKM means overlaid; for the Reynolds-stress overlay, download")
            print("   chan180.reystress from the Oden site and pass --mkm <path>)")

if __name__ == '__main__':
    main(sys.argv[1:])
