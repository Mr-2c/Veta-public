#!/usr/bin/env python3
"""
work_fd_ygrid_model.py -- two-zone stretched wall-normal grid design for
the FD-y backend, + the proof that a single-tanh map cannot hit the targets.

DESIGN TARGETS (Re_tau = 20000 production):
    dy+_wall = 0.5   at the first off-wall spacing
    dy+_core = 10    uniform across the channel core
    ny       ~ 4400  total points across [-1, 1]

TWO-ZONE MAP (per half-channel, mirrored):
    wall zone : geometric growth  dy+_j = dy+_wall * r^j,  capped at dy+_core
    core zone : uniform dy+_core to the centerline
    The final core spacing is scaled by a factor ~1 so the half exactly spans
    retau plus units (C0 landing; a production map should smooth the zone
    junction to C1 -- see the blend note in the report).

Also reports:  (a) the single-tanh scan -- for each N, pick the stretch that
just achieves dy+_wall = 0.5, then read off the resulting dy+_CL; (b) the
Chebyshev(GLC) comparison at the same ny; (c) the Re_tau=2000 mid-term case.

Run:  python3 src/models/work_fd_ygrid_model.py
"""
import numpy as np


def two_zone_spacings(retau, dyp_wall=0.5, dyp_core=10.0, r=1.03):
    """Per-half spacing list (plus units), wall -> centerline. Returns
    (dyp array, n_wall_cells). Exact landing: core spacings scaled ~1."""
    dyp = []
    d = dyp_wall
    h = 0.0
    while d < dyp_core and h + d < retau:
        dyp.append(d)
        h += d
        d *= r
    n_wall = len(dyp)
    rem = retau - h
    n_core = max(int(np.ceil(rem / dyp_core)), 1)
    core = np.full(n_core, rem / n_core)          # uniform, exact landing
    return np.concatenate([np.array(dyp), core]), n_wall


def two_zone_grid(retau, dyp_wall=0.5, dyp_core=10.0, r=1.03):
    """Full-channel grid y in [-1, 1] (outer units), wall-clustered both sides.
    Returns y (ny,), built by mirroring the half-spacing list."""
    dyp, _ = two_zone_spacings(retau, dyp_wall, dyp_core, r)
    dy = dyp / retau                               # outer units
    half = np.concatenate([[0.0], np.cumsum(dy)])  # 0 .. 1 (wall -> CL)
    y = np.concatenate([half - 1.0, (1.0 - half[::-1])[1:]])
    return y


def natural_spacings(retau, alpha=0.375, dwall=0.6, c=0.6, cap=22.0, q=2.0):
    """[P&O-natural, alpha-corrected] spacing law in plus units:
         law(y+)   = ( dwall^q + (c * y+**alpha)^q )^(1/q)
         delta(y+) = cap * tanh(law / cap)
    Pirozzoli & Orlandi (JCP 2021) 'natural' stretching follows the local
    Kolmogorov scale: flat ~dwall below the buffer layer, then power-law
    growth (their log-layer exponent 1/4; here alpha = 0.35-0.40 per the
    design correction). The q-norm is a SMOOTH-MAX blend of the
    two regimes (q=2: C-inf, crossover where c*y^alpha = dwall, i.e. y+ ~ 1);
    the tanh is a C-inf cap at 'cap' plus units. Marched from the wall with
    midpoint evaluation; the whole set is rescaled by ~0.999 so the half
    lands exactly on retau (no clamped last cell). Returns dy+ (wall->CL)."""
    def delta(yy):
        law = (dwall**q + (c * yy**alpha)**q) ** (1.0 / q)
        return cap * np.tanh(law / cap)
    dyp = []
    y = 0.0
    while y < retau:
        d = delta(y + 0.5 * delta(y))            # midpoint evaluation
        dyp.append(d)
        y += d
    dyp = np.array(dyp)
    return dyp * (retau / dyp.sum())             # exact landing (factor ~1)


def natural_c_for_ny(retau, ny_half_target, alpha, dwall, cap):
    """Bisect the coefficient c so the half-channel point count hits target."""
    lo, hi = 0.01, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        n = len(natural_spacings(retau, alpha, dwall, mid, cap))
        if n > ny_half_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def natural_grid(retau, alpha=0.375, dwall=0.6, c=0.6, cap=22.0):
    """Full-channel y in [-1,1] from the natural-stretching spacing law."""
    dy = natural_spacings(retau, alpha, dwall, c, cap) / retau
    half = np.concatenate([[0.0], np.cumsum(dy)])
    half[-1] = 1.0
    y = np.concatenate([half - 1.0, (1.0 - half[::-1])[1:]])
    return y


def eta_plus(yp, kappa=0.41):
    """log-layer Kolmogorov estimate eta+ = (kappa y+)^(1/4) (outer: lower
    bound -- eps drops below 1/(kappa y) toward the CL, so true eta is a bit
    larger there; Delta/eta printed with this is thus slightly PESSIMISTIC)."""
    return (kappa * np.maximum(yp, 1.0)) ** 0.25


def natural_report(retau, alphas=(0.35, 0.375, 0.40), dwall=0.6, cap=22.0,
                   ny_targets=(1250, 1375, 1500)):
    print(f"\n=== NATURAL stretching (P&O 2021, alpha-corrected), "
          f"Re_tau={retau:g}, dy+_wall={dwall}, cap={cap}+ ===")
    print(f"{'alpha':>6} {'c':>7} {'ny_tot':>7} {'dy+_1':>6} {'dy+_CL':>7} "
          f"{'cap@y+':>8} {'maxgrow@y+':>12} {'D/eta@100':>9} {'D/eta@1k':>9} "
          f"{'D/eta@CL':>9}")
    picks = {}
    for a in alphas:
        for nt in ny_targets:
            c = natural_c_for_ny(retau, nt, a, dwall, cap)
            dyp = natural_spacings(retau, a, dwall, c, cap)
            yp = np.concatenate([[0.0], np.cumsum(dyp)])[:-1] + dyp / 2
            ratios = dyp[1:] / dyp[:-1]
            grow = ratios.max()
            ygrow = yp[1 + int(np.argmax(ratios))]
            icap = np.nonzero(dyp > 0.95 * cap)[0]
            cap_str = f"{yp[icap[0]]:8.0f}" if len(icap) else "      --"
            de = dyp / eta_plus(yp)
            i100 = np.searchsorted(yp, 100.0)
            i1k = np.searchsorted(yp, 1000.0)
            print(f"{a:6.3f} {c:7.3f} {2*len(dyp)+1:7d} {dyp[0]:6.2f} "
                  f"{dyp[-1]:7.2f} {cap_str} {grow:6.3f}@{ygrow:5.1f} "
                  f"{de[i100]:9.2f} {de[i1k]:9.2f} {de[-1]:9.2f}")
            picks[(a, nt)] = c
    return picks


def tanh_scan(retau, dyp_wall=0.5, N_list=(2201, 4401, 8801, 17601, 35201)):
    """Single-tanh map y = tanh(a x)/tanh(a), x uniform on [-1,1], N points.
    For each N: bisect the stretch a so dy+_wall = dyp_wall, report dy+_CL."""
    rows = []
    for N in N_list:
        x = np.linspace(-1.0, 1.0, N)

        def dwall(a):
            y = np.tanh(a * x) / np.tanh(a)
            return (y[1] - y[0]) * retau

        lo, hi = 0.5, 25.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if dwall(mid) > dyp_wall:
                lo = mid
            else:
                hi = mid
        a = 0.5 * (lo + hi)
        y = np.tanh(a * x) / np.tanh(a)
        dyp = np.diff(y) * retau
        rows.append((N, a, dyp[0], dyp.max()))
    return rows


def cheb_spacings(retau, ny):
    """GLC nodes y_j = cos(pi j/(ny-1)): wall + centerline spacings in plus."""
    j = np.arange(ny)
    y = np.cos(np.pi * j / (ny - 1))
    dyp = np.abs(np.diff(y)) * retau
    return dyp[0], dyp.max()


def report_case(retau, target_ny=None, dyp_wall=0.5, dyp_core=10.0):
    print(f"\n=== Re_tau = {retau:g}  (targets: dy+_wall={dyp_wall}, "
          f"dy+_core={dyp_core}) ===")
    print(f"{'r':>6} {'n_wall':>7} {'h_wall+':>8} {'ny_half':>8} "
          f"{'ny_total':>9} {'maxgrow':>8}")
    for r in (1.01, 1.02, 1.03, 1.05):
        dyp, n_w = two_zone_spacings(retau, dyp_wall, dyp_core, r)
        grow = (dyp[1:] / dyp[:-1]).max()
        ny_total = 2 * len(dyp) + 1
        print(f"{r:6.2f} {n_w:7d} {dyp[:n_w].sum():8.1f} {len(dyp):8d} "
              f"{ny_total:9d} {grow:8.3f}")
    if target_ny:
        print(f"  (assessment sizing: ny ~ {target_ny})")


def main():
    print("wall-normal grid design -- NATURAL stretching is the")
    print("production choice; two-zone/tanh/GLC")
    print("kept below as comparisons.")

    natural_report(20000.0, dwall=0.6)
    natural_report(20000.0, dwall=0.5)
    natural_report(20000.0, dwall=0.5, cap=25.0, alphas=(0.40,),
                   ny_targets=(1250, 1500))

    report_case(20000.0, target_ny=4400)
    report_case(2000.0)

    print("\n=== single-tanh CANNOT hit both targets (Re_tau=20000) ===")
    print("stretch a chosen per N so dy+_wall = 0.5 exactly; then dy+_CL is:")
    print(f"{'N':>7} {'a':>7} {'dy+_wall':>9} {'dy+_CL':>8}")
    for N, a, dw, dc in tanh_scan(20000.0):
        flag = "  <-- violates dy+_CL <= 10" if dc > 10.0 else "  OK"
        print(f"{N:7d} {a:7.3f} {dw:9.3f} {dc:8.1f}{flag}")

    print("\n=== Chebyshev(GLC) at the same point counts (Re_tau=20000) ===")
    for ny in (4401, 6145, 8801):
        dw, dc = cheb_spacings(20000.0, ny)
        print(f"  ny={ny:5d}:  dy+_wall={dw:7.4f}   dy+_CL={dc:6.1f}"
              + ("   <-- CL under-resolved" if dc > 10 else ""))

    # the designed production map (r=1.03)
    y = two_zone_grid(20000.0, r=1.03)
    dyp = np.diff(y) * 20000.0
    print(f"\nDESIGNED MAP (r=1.03): ny={len(y)}, dy+_wall={dyp[0]:.3f}, "
          f"dy+_max={dyp.max():.3f}, wall-zone cells/side="
          f"{two_zone_spacings(20000.0, r=1.03)[1]}")
    print("NOTE: production map should C1-blend the zone junction (the FD")
    print("metric sees the spacing-ratio jump r->1; a 5-10 cell geometric")
    print("taper removes it at ~zero point cost).")


if __name__ == "__main__":
    main()
