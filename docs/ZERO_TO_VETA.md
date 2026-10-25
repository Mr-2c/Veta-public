# Zero to Veta — the complete human's tour of Veta Sublimeson

You know how to code.  You may not know spectral methods, CUDA Fortran
semantics, or why the wall-normal direction is the hard one.  This
document takes you from zero to a full working understanding of this
code — its physics, its numerics, its data layouts, its GPU machinery,
and every optimization it carries — in one continuous read.  At the end
you should be able to open any file, know why it exists, and start
changing it.

Read it top to bottom once; then use §17 (the guided reading order) with
the source open.  The two derivations it leans on sit next to this file:
`kmm_formulation.pdf` (the v–η formulation) and `influence_matrix.pdf`
(the clamped v-solve).

Veta has a sister code, Pola Sublimeson, which solves a neighbouring
problem with deliberately parallel conventions; where Veta differs from
Pola in an interesting way, this document says so.  Pola is not part of
this release.

---

## 0. What this program is, in three sentences

Veta Sublimeson simulates turbulent flow between two parallel walls
(plane channel flow) by solving the incompressible Navier–Stokes
equations with **no turbulence model** — every eddy down to the
dissipative scale is computed (Direct Numerical Simulation, DNS).  It is
GPU-resident and multi-node, written in **CUDA Fortran** (nvfortran),
using Fourier × Fourier discretization in the two wall-parallel
directions and a choice of two wall-normal discretizations behind one
seam.  One binary per precision tier (`Veta_dp` / `Veta_sp`), driven by
a Fortran namelist deck, runs everything from a 64³ smoke test to
Re_τ = 2000 campaigns on tens of GPUs.

Why DNS is expensive, in one line: the ratio of largest to smallest eddy
scales like Re_τ, and cost scales roughly like Re_τ⁴ in 3D+time.
Everything in this code exists to make that feasible.

**The lay of the land** (every top-level directory, one line):

    src/core/          the per-mode numerics + the single-GPU solver modules
    src/parallel/      the multi-GPU driver and its par_* twins (production)
    src/serial/        the single-GPU driver (the validation oracle)
    src/models/        python design models -- executable specs, not built
    tests/serial/      single-GPU unit tests (73 files)
    tests/parallel/    multi-GPU / MPI+NCCL tests (23 files)
    tools/             python post-processing: stats, checkpoints, grids
    scripts/           build + sbatch helpers
    examples/          campaign folders (deck + sbatch per Re_tau) + test decks
    docs/              this document, the two derivations, the development history

## 1. The physics and the terms you need

- **Incompressible Navier–Stokes**: ∂u/∂t + (u·∇)u = −∇p + ν∇²u, with
  ∇·u = 0.  The pressure p is not an independent unknown — it is
  whatever field makes the velocity divergence-free.  That constraint
  is THE structural difficulty of incompressible flow solvers.
- **The channel**: two solid walls at y = ±1 with no-slip (u = 0),
  periodic in the streamwise direction x with length L_x and in the
  spanwise direction z with length L_z.  Wall-normal y is the
  non-periodic, non-homogeneous direction and carries all the
  difficulty.
- **Bulk velocity**: this code drives at **constant flux** (CFR — the
  deck's `Qtar` is the target), so the mean pressure gradient is an
  OUTPUT, and the friction (hence Re_τ) emerges from the run rather
  than being imposed.
- **Friction velocity and wall units**: the wall drag defines
  u_τ = sqrt(τ_wall/ρ); lengths scaled by ν/u_τ are "wall units",
  written y⁺, dx⁺ etc.  **Re_τ = u_τ h/ν** (h = half-channel height =
  1) is the honest measure of how turbulent a case is: the campaigns
  step through Re_τ = 180, 550, 1000, 2000.
- **What comes out**: statistics profiles (mean velocity, Reynolds
  stresses, pressure statistics) in a KMM-style table, checkpoints (the
  full spectral state), and optional physical-space snapshots.
  Validation is against **Lee & Moser (2015)** at Re_τ 550 and 1000, and
  against **Moser–Kim–Mansour (1999)** at Re_τ 180.  Both are published
  third-party datasets and are not redistributed with this release;
  fetch them from the authors' data pages.

## 2. The discretization: Fourier × Fourier × (Chebyshev | FD)

x and z are periodic ⇒ Fourier series ⇒ derivatives become
multiplications by ik ⇒ **spectral accuracy**.  So:

    u(x, y, z, t) = Σ_kx Σ_kz  û_{kx,kz}(y, t) · e^{i(kx·x + kz·z)}

Because u is real, only kx ≥ 0 is stored (an **R2C** transform); the
current production path is R2C throughout.

The wall-normal direction is where the two backends live, **behind one
seam**, selected at runtime by the deck:

- **Chebyshev (Shen)** — the default.  Expand in Chebyshev polynomials
  on Gauss–Lobatto (GLC) nodes and use Shen's basis recombination so
  the Dirichlet boundary conditions are built into the basis, giving
  banded (rather than dense) Helmholtz operators.  Spectral accuracy in
  y.  `src/core/cheb_shen.F90`, `cheb_ops_mod.F90`, `cheb_fft_mod.F90`.
- **FD-y** (`ygrid_file` non-empty) — 8th-order banded finite
  differences on an **arbitrary node set read from a file**.  Weights
  come from Fornberg (1988); storage is banded with a no-pivot banded
  LU.  `src/core/fdops_mod.F90` is the host oracle,
  `fd_helm_gpu_mod.F90` / `fd_vsolve_gpu_mod.F90` the device twins.
  This buys wall-clustered grids that are not tied to the Chebyshev
  node distribution, at the cost of algebraic (not spectral) accuracy.

The two are triple-validated against each other and the literature:
Orr–Sommerfeld eigenvalues, a manufactured solution, and a t = 650 KMM
twin agreeing within 0.09%.

Numbers you will see everywhere:

    n              wall-normal degree; ny = n+1 points/coefficients
    nx, nz         streamwise / spanwise mode counts from the deck
    nkx = nx/2+1   stored streamwise modes (R2C)
    nxd = 3nx/2,   the DEALIASED physical grid (the 3/2 rule, §7)
    nzd = 3nz/2
    Lx, Lz         box lengths (decks use 8pi x 3pi for statistics)

**Dealiasing (3/2 rule)**: a product of two fields has twice the modes;
sampling on the original grid folds ("aliases") them back onto resolved
ones.  Cure: evaluate products on a 3/2-finer grid, then truncate.

## 3. The KMM formulation: why (v, η) and not (u, v, w, p)

This is the structural heart of the code, and Veta's biggest departure
from its pipe sister.

Rather than march the three velocity components and pressure, the
Kim–Moin–Moser (KMM) formulation eliminates pressure **exactly** by
taking two scalar combinations of the momentum equations:

- **v** — the wall-normal velocity.  Taking the double curl of the
  momentum equation kills the pressure gradient and leaves a
  **fourth-order** equation for v alone (the Orr–Sommerfeld operator).
- **η = ∂u/∂z − ∂w/∂x** — the wall-normal vorticity.  It satisfies a
  **second-order** advection–diffusion equation (Squire).

Given (v, η) at a mode, continuity plus the definition of η recover u
and w algebraically — that is `recover_uw` in `velocity_mod.F90`.  The
mean mode (kx, kz) = (0, 0) is singular for that recovery (both v and η
carry no information about the mean profile) and is evolved separately
as two 1-D problems for the mean u and w — that is the **CFR mean**
(`mean_mod.F90` / `mean_gpu_mod.F90`), where the constant-flux
constraint is enforced.

What this buys: **no pressure solve in the time loop at all.**  Where
Pola runs a Kleiser–Schumann influence matrix every substage to project
onto divergence-free fields, Veta gets divergence-free-ness for free
from the formulation, and pays instead with a fourth-order operator.
Pressure is recovered only when statistics need it
(`par_pressure_mod.F90`, a diagnostic Neumann–Poisson solve).

The cost: boundary conditions.  v needs **four** conditions
(v = ∂v/∂y = 0 at both walls) for a fourth-order operator whose natural
factorization only supplies two.  §6 is how that is closed.

## 4. How a field lives in memory

Spectral fields are `complex(wp)` arrays dimensioned

    A(0:n, nkx, nz)        -- y is the LEADING dimension

**y is always local and always leading.**  The wall-normal solves
recurse over y and cannot be parallelized across it, so y stays whole on
every rank and contiguous in memory; the parallel unit is the (kx, kz)
mode.  This single decision shapes the decomposition (§10), the kernels
(§8), and why the transpose exists.

Two layouts alternate through the transform chain:

    Layout B (kx-slab):  (ny, b_nkx_loc, *)   this rank's kx block, FULL z
                         home of the spectral field; where the y-transform
                         and the z-FFT run
    Layout A (z-slab):   (ny, *, ad_nz_loc)   FULL kx, this rank's z block
                         where the x-FFT runs; home of the physical field

The precision of each array is governed by the **tier** (§12): `wp` is
the working precision of the spectral state, `tp` the transform
precision.  `TP=sp` runs the transform trip in fp32 while the solves
stay fp64.

## 5. The time step: SMR-RK3 with Crank–Nicolson viscous

The viscous term is stiff ⇒ implicit.  The nonlinear term is explicit.
Veta uses the **Spalart–Moser–Rogers (SMR) three-substage Runge–Kutta**
scheme with Crank–Nicolson viscous treatment — the standard KMM-family
stepper, and unlike Pola there is no corrector iteration: the cost of a
step is fixed at three substages.

The tableau (`par_stepper_mod.F90`):

    GAMMA = [ 8/15,   5/12,  3/4  ]      explicit weight, this substage
    ZETA  = [ 0,    -17/60, -5/12 ]      explicit weight, previous substage
    CSUB  = [ 8/15,   2/15,  1/3  ]      implicit (viscous) substage size

`ZETA(1) = 0` matters: nothing crosses a step boundary, so the scheme is
self-starting and checkpoints are single-level.

Per substage, per mode, with β = ½·CSUB(k), a = β·dt/Re:

    eta:  (1 + a L) eta^k = eta + a L eta + gamma·dt·N^{k-1} + zeta·dt·N^{k-2}
          -> ONE Helmholtz solve
    v:    the fourth-order operator, factored -> TWO Helmholtz solves
          + the influence correction (§6)

**Variable dt with exact-time landing.**  `variable_dt` grows dt toward
the CFL bound (`cfl_target`, ceiling sqrt(3)/π ≈ 0.5513), re-evaluated
every `dt_check_every` steps (default 20), and lands exactly on output
times.  Operators are only rebuilt when dt drifts past
`dt_recompute_tol` — rebuilding is the expensive part, so a tolerance
band avoids refactoring every step.

## 6. Closing the fourth-order v-solve: the influence matrix

The v equation `(L − α L²) v = f` with `L = D² − k²` and four clamped
conditions `v = Dv = 0` at `y = ±1` is factored into two second-order
Helmholtz problems:

    L − α L²  =  −α L (L − 1/α)
       H1 = D² − k²            H2 = D² − (k² + 1/α)

Each is solved with the validated Shen-Dirichlet Helmholtz solve, which
supplies `v = 0` at both walls.  The two MISSING conditions,
`Dv(±1) = 0`, are recovered per mode with a **2×2 influence
(capacitance) matrix**: precompute the two homogeneous solutions
`v₋, v₊` and `C⁻¹` once per operator build, then each solve does two
Helmholtz solves, a 2×2 apply, and a combine.  Source:
`vsolve_if_mod.F90` (host oracle) and `vsolve_if_gpu_mod.F90` (device);
the derivation with sign conventions is `docs/influence_matrix.pdf`.

The FD-y backend has the same structure with banded operators:
`fd_vsolve_mod.F90` / `fd_vsolve_gpu_mod.F90`.

Note the contrast with Pola: both codes use an influence matrix, but for
different reasons.  Pola's couples pressure to no-slip (a 4×4 per mode);
Veta's recovers the two clamped derivative conditions the factorization
drops (a 2×2 per mode).

## 7. The nonlinear term: rotational form, transforms, dealiasing

N = u × ω (rotational form: cheaper, and better energy conservation).
Products are pointwise in physical space and convolutions in spectral
space — so transform, multiply, transform back:

    spectral (v, eta) --recover_uw--> (u,v,w) --compute_vorticity--> omega
    6 fields  --y-transform, pad-z, ifft-z, TRANSPOSE, pad-x, irfft-x-->
    physical:  H = u x omega  (cross_prod, pointwise, /(nxd*nzd))
    3 fields  --rfft-x, trunc-x, TRANSPOSE, fft-z, trunc-z, y-transform-->
    spectral  --assemble_hv_heta--> the (v, eta) projections

Six fields make the spectral→physical trip and three come back.  Both
directions are batched into ONE transpose each (`nfield = 6` inverse,
`nfield = 3` forward) rather than one per field.

**The dealias split.**  x low-pass runs where kx is whole (Layout A); z
±Nyquist wrap runs where z is whole (Layout B).  Neither can be done in
the other layout — proved in the numpy model
(`src/models/m2_distributed_transform_model.py`), which also proves the
distributed transform reproduces the monolithic one bit-for-bit.

After every nonlinear evaluation the **Hermitian symmetry** of the R2C
representation must be re-enforced on the kx = 0 plane and the Nyquist
rows (`mask_nyquist_slab`, `hsym_every`/`hsym_fix` in the deck).

## 8. One GPU: CUDA Fortran, and the two kinds of kernel

Everything lives permanently on the GPU; the host orchestrates and does
file I/O.  Veta is written in **CUDA Fortran** (nvfortran), which offers
two ways to get code onto the device — and the balance between them is
the single most distinctive thing about this codebase:

- **`!$cuf kernel do` directives — 116 of them.**  The dominant idiom.
  You write an ordinary Fortran loop nest over device arrays, mark it,
  and nvfortran generates the kernel.  The loop stays readable as
  numerics; there is no separate kernel body to keep in sync.
- **Hand-written kernels — 11 of them** (`attributes(global)`).  Used
  only where a directive cannot express the algorithm: the banded
  factor/solve chains and the mean-flow CFR closure.  They live in
  `fd_helm_gpu_mod.F90` (5), `helm_band_gpu_mod.F90`,
  `fd_vsolve_gpu_mod.F90`, `vsolve_if_gpu_mod.F90`, `mean_gpu_mod.F90`,
  `fd_mean_gpu_mod.F90`, and `par_pressure_mod.F90` (1 each).

Everything else is library calls: **cuFFT** for the transforms (90 call
sites) and **NCCL** for the transpose (114).  There is **no OpenACC and
no OpenMP anywhere** — `!$cuf` is an NVIDIA-proprietary CUDA Fortran
extension, so the code is nvfortran-only (§19).

The pragmatic consequence: most of Veta's "GPU code" reads as plain
Fortran array arithmetic over `device` arrays.  When you go looking for
where the work happens, follow the `, device ::` declarations (556 lines
of them in the solver core) rather than hunting for kernel bodies.

## 9. The time loop, mapped end to end (the full call map)

Files: `main_par.F90` (driver, 2056 lines), `par_stepper_mod.F90`
(the substage), `par_nonlinear_mod.F90` (the RHS),
`par_spectral3d_mod.F90` (the distributed transform),
`par_transpose_mod.F90` (the NCCL exchange).  Indentation = calls.

    main_par                                    src/parallel/main_par.F90
    ├── STARTUP (once per segment)
    │   ├── MPI_Init + NCCL communicator             par_mod.F90, nccl_mod.F90
    │   ├── read namelist /run/ from DECK            (~45 keys)
    │   ├── decomp_init: the kx-slab / z-slab block ranges   decomp_mod.F90
    │   ├── y-basis select: Chebyshev (default) or FD-y if ygrid_file  [§2]
    │   ├── operator build (GPU-resident):
    │   │     band_ops_t   Shen/Helmholtz band structure
    │   │     vsolve_ops_t CN-shifted lift + 2x2 influence precompute  [§6]
    │   │     mean_ops_t   CFR mean precompute                  solver_ops_mod.F90
    │   ├── cuFFT plans (x r2c/c2r, z c2c, y cheb) per local extent
    │   └── initial condition: restart (basis-stamped) or seeded turbulent IC
    │
    └── TIME LOOP:  for step = 1 .. n_steps   (or until t_end / wall budget)
        │
        ├── par_step_full_ws                        par_stepper_mod.F90
        │   └── for ks = 1, 2, 3            SMR-RK3 substages          [§5]
        │       ├── par_ev_stage(ks)  -- the fluctuation update
        │       │   ├── par_nl_ev_begin        6-field INVERSE legs + a2a launch
        │       │   │     recover_uw, compute_vorticity           [§3]
        │       │   │     per field: cheb_inv(y), pad-z, ifft-z, a2a(f)  [§11]
        │       │   ├── HOIST A: y_apply_L(eta) + cxpsyi -> rhs   [§11]
        │       │   ├── par_nl_ev_mid         landings + x legs, cross_prod,
        │       │   │     forward legs + a2a launches
        │       │   ├── HOIST B: y_apply_L(v), L^2 v -> rhsv      [§11]
        │       │   ├── par_nl_ev_end         forward landings + project
        │       │   ├── caxpy2 + y_helm_solve       eta^k          [§5]
        │       │   └── caxpy2 + y_vif_solve        v^k (influence) [§6]
        │       └── mean solve on the kx=0 owner rank, IN-STREAM
        │             y_mean_stage_cfr: constant-flux u0, free w0   [§3]
        │
        ├── diagnostics on cadence
        │   ├── div_every    divergence check
        │   ├── nan_every    non-finite tripwire (default every 10 steps)
        │   ├── hsym_every   Hermitian-symmetry check (+ hsym_fix)
        │   └── dt_check_every (20): CFL re-evaluation + wall-clock check
        │
        ├── outputs on exact-landing times
        │   ├── stats_dt     profile accumulators
        │   ├── snap_dt      physical-space snapshot (optionally fp32)
        │   └── chk_dt       checkpoint (basis-stamped, atomic)  par_io_mod.F90
        │
        └── wall budget: max_wall_hours - wall_margin_min
              -> clean checkpoint + exit before the SLURM kill

**A property worth internalizing: the production step has NO host
sync.**  Every `cudaDeviceSynchronize` in the loop is guarded by
`if (timing)`.  An untimed production run issues the whole step
asynchronously; the CFL reduction (the only collective) runs every 20
steps, not every step.  That is why the overlap machinery in §11 has
room to work — and it is a real difference from Pola, whose PC2
corrector blocks on an `MPI_Allreduce` every iteration.

## 10. Multi-GPU: kx-slabs, z-slabs, and the NCCL transpose

The decomposition is 1-D.  Each of P ranks owns a contiguous block of
kx in Layout B and a contiguous block of dealiased z-planes in Layout A;
**y is whole on every rank in both** (§4).  One NCCL all-to-all converts
between them, once per direction per transform.

    Layout B (kx-slab, z whole)   -- y-transform, z-FFT run here
              |  NCCL all-to-all  |
    Layout A (z-slab, kx whole)   -- x-FFT runs here; physical space

Mechanics (`par_transpose_mod.F90`, 723 lines):

- The payload is sent as **raw bytes** (`ncclChar`): a transpose only
  MOVES data, so it is exact and kind-agnostic — bit-identical for
  `tp = sp` or `dp` alike.
- Counts and displacements come from `decomp_mod`'s pure, separately
  tested `alltoall_counts_A_to_B`.  B→A is the same permutation with
  send/recv swapped.
- **The Layout-B side is never staged.**  The wire order
  `[iy, local-kx, local-z, field]` makes the B-side placement a pure
  identity per (peer, field), so NCCL sends directly from — and
  receives directly into — the field arrays.  Only the genuinely
  strided Layout-A side keeps a gather/scatter kernel and ONE
  persistent staging buffer.  Carrying only that one buffer is worth
  ~5.7 GB/GPU at the 16-node frame.
- The staging buffer is **grow-only and hoisted** out of the per-call
  path: at large grids it is multi-GB, and allocating it per call would
  mean 12 `cudaMalloc`/`cudaFree` pairs per step — `cudaFree`
  synchronizes, which serializes the step and adds jitter.

Precondition: `P ≤ min(nkx, nzd)` — the slab ceiling.  Empty-rank
handling is deliberately NOT implemented; the rank counts in use sit far
below the ceiling.

## 11. Comm/compute overlap: two layers (`a2a_overlap = 1`)

The all-to-all is the scaling ceiling, so the game is hiding compute
under it.  Veta does this in **two layers**, and the second is the more
unusual idea.

**Layer 1 — per-field pipelining.**  Instead of one monolithic
`nfield`-wide exchange, each field's all-to-all is launched the moment
that field's local FFT chain finishes.  Field *f* flies on
`par_comm_stream` while field *f+1* computes on `par_stream`.  Nine
flights per substage (6 inverse + 3 forward).  Ordering is entirely
device-side via **per-field event pairs** — `ev_pk(f)` records the
pack-position on the compute stream, the comm stream waits it, then
records `ev_cm(f)`; a consumer waits only on ITS OWN field, never on
the whole group.  No host syncs in the leg.

**Layer 2 — hoisting stepper work into the flight window.**  Once all
the flights are launched, the compute stream would idle waiting.  So
the stepper's linear-operator chains — which belong to the *time
stepper*, not the nonlinear term — are moved inside the window:

    call par_nl_ev_begin(...)          ! 6 inverse flights launched
    call y_apply_L(bo, ..., eta, s%rhs, s%scr)   ! HOIST A
    call cxpsyi(n, nx, nz, eta, a, s%rhs)
    call par_nl_ev_mid(...)            ! 3 forward flights launched
    call y_apply_L(...v...); call y_apply_L(...) ! HOIST B
    call par_nl_ev_end(...)

The nonlinear evaluation was deliberately split into
`begin`/`mid`/`end` phases so this interleaving is possible.  The safety
argument is disjointness, and it is written at the call site: the
hoisted kernels only READ `eta`/`v` and only WRITE `rhs`/`Lv`/`rhsv`,
which the nonlinear phases never touch.

Measured **1.38× at 8 nodes**; the assessment puts the ceiling at
1.5–2.0× and calls it the only remaining lever.  The default in code is
`a2a_overlap = 0` (bit-identical to the pre-overlap path), but **every
campaign deck sets 1** — under overlap the trajectory picks up
tp-ULP-scale reassociation drift, which is gated per precision rather
than pretended away.

Under `time_every > 0` the four per-transpose barriers come back, so a
timed run has the exact pre-overlap barrier structure and the
phase-split timings mean something; an untimed run is barrier-free.
The A2A occupancy is event-timed so overlap is preserved even in timed
runs.

## 12. Precision tiers and memory

- **`wp` (working precision)** — the spectral state and all wall-normal
  solves.  `dp` in production.
- **`tp` (transform precision)** — the transform trip and the
  all-to-all payload.  **`TP=sp` is the shipped tier**:
  transforms-only fp32, solves still fp64.  The `TP=sp` suite is
  full-pass.  Halving the wire payload is most of the win.
- Putting the whole state in fp32 (`wp = sp`) is **de-prioritized**:
  ~1.8% memory saving, and not wired into the parallel path at all —
  there the fields are literal `complex(dp)`.

Memory levers in place: the direct all-to-all (a single staging buffer,
§10), the run-lifetime `stepper_ws` (no per-step
`cudaMalloc`/`cudaFree` anywhere in the stepper or the v-solve), and
the FD-y factor cache (`fd_cache_gb`, a deck budget).

## 13. The Galilean frame and the constant-flux drive

- **Constant flux (CFR)**: the mean mode is advanced so the bulk flux
  hits `Qtar` exactly every substage.  The required mean pressure
  gradient is reported — that is how Re_τ emerges as an output.
- **Galilean frame** (`u_galilean`, campaigns use 0.6): the state is
  the lab velocity seen from a frame translating at U_g ≈ u_cl/2.  An
  exact reformulation — statistics, checkpoints and boundary conditions
  are all shift-invariant — but the x-CFL now measures |u − U_g|, about
  half the binding speed, so dt grows at identical ms/step.  Free speed
  from physics.

## 14. I/O and the campaign machinery

- **The deck**: ONE Fortran namelist (`&run ... /`) passed as
  `DECK=examples/...nml`.  ~45 keys.  **Nothing is read from
  environment variables** — env vars do not reach a namelist, and
  SLURM's `--export` mangles comma-containing values (this cost a
  cluster round-trip once; the decks now carry a warning comment).
  Missing file or key ⇒ built-in defaults plus a loud line.
- **Checkpoints** (`par_io_mod.F90`, 995 lines): collective MPI-IO,
  **basis-stamped** (a Chebyshev checkpoint cannot be silently restarted
  on an FD grid), atomic writes, `restart_auto` chains segments.
  `tools/resample_checkpoint.py` does offline cross-resolution restart.
- **Statistics**: accumulated on the fly, pooled across segments by
  `tools/combine_stats.py` (sample-weighted; the header lists the
  segments).  `tools/analyze_stats.py` prints the physics verdicts and
  carries the y-derivative operator (exact Chebyshev on GLC nodes, with
  an FD fallback that announces itself).
- **Campaigns**: `examples/campaigns/<case>/` holds a tuned `input.nml`
  and a sized `veta.sbatch` at the minimum node count, Chebyshev and
  FD twins at Re_τ 550/1000/2000.  The README there is the
  clone-compile-copy-run recipe.
- **A cluster trap worth reading before your first run**: on any
  cluster, confirm NCCL is using InfiniBand and not TCP.  A silent
  socket fallback is ~5–10× slower.  The driver prints the warm
  all-to-all bandwidth and a loud warning if it detects a socket
  fallback; `CLAUSON_REQUIRE_RDMA=1` turns that warning into a hard
  abort.  The sbatch files in `scripts/sbatch/` show the environment
  that made RDMA work on the machine this code was developed on.

## 15. Testing: what it covers, and what it does not

- **96 test programs** — `tests/serial/` (73) and `tests/parallel/`
  (23), ~12,500 lines of Fortran.  Coverage is genuinely broad: the
  y-operators, both backends, the transforms, Orr–Sommerfeld
  eigenvalues, manufactured solutions, the influence solve, the
  distributed transpose, the parallel drivers, I/O contracts.
- **`--validate`**: the distributed driver can cross-check itself
  against the single-GPU oracle.  With `a2a_overlap = 0` this is
  bit-exact; with overlap on it is tp-ULP, gated per precision.
- **Executable design models**: `src/models/*.py` are numpy models that
  PROVE properties before the Fortran is written — the distributed
  transform's bit-identity, the decomposition index arithmetic, the
  direct-a2a identity.  They are specs you can run, not documentation.

Two gaps you should know about, because they change how you work:

- **No test runs without a GPU.**  Every test is compiled
  `-cuda -gpu=cc90 -cudalib=cufft` and linked against the full device
  stack — including the tests of the plain-Fortran "CPU reference
  group" modules.  There is no host tier.  (Pola's equivalent suite has
  4,126 lines that run GPU-free in ~3 minutes.)
- **CI never compiles the solver.**  GitHub runners have no nvfortran
  and no NVIDIA GPU, so `.github/workflows` runs three *static* Python
  gates: structural brace/block balance, a CUF-0155 device-dereference
  audit (a real nvfortran trap), and an advisory case-fold check.  They
  are well-judged for what they can see, but a type error, a broken
  interface, or a wrong result all pass CI and must reach the cluster
  to be found.

The runner (`run_all_tests.sh`) is disciplined about what counts as a
pass: it fails on explicit markers, a crash, a stray non-finite VALUE
matched as a standalone token, OR the **absence of a positive pass
marker** — a test that silently skips all its checks is a failure, not
a pass.  Two assertion-free inspection tools are whitelisted by name.

## 16. The file-by-file map

**`src/core/` — 9,551 lines, 40 modules.**  The single-GPU solver and
the per-mode numerics.

| file | lines | what |
|---|---:|---|
| `solver_ops_mod.F90` | 515 | the solver-state aggregates: band/vsolve/mean ops |
| `io_mod.F90` | 457 | single-GPU checkpoint + snapshot I/O |
| `init_mod.F90` | 453 | initial conditions (turbulent seed, laminar, restart) |
| `fd_helm_gpu_mod.F90` | 442 | FD-y Helmholtz on device — 5 of the 11 hand kernels |
| `fft3d_mod.F90` | 404 | the 1-D axis plans (x r2c/c2r, z c2c) |
| `stats_mod.F90` | 395 | statistics accumulators |
| `stepper_mod.F90` | 394 | SMR-RK3 + CN, the `_ws` (workspace-owning) entry points |
| `vsolve_if_gpu_mod.F90` | 359 | the clamped v-solve + 2×2 influence, device |
| `spectral3d_mod.F90` | 347 | the single-GPU dealiased transform |
| `nonlinear_mod.F90` | 276 | u×ω and the (v, η) projection |
| `cheb_shen.F90` | 274 | the Shen-basis Helmholtz solve |
| `fdops_mod.F90` | — | Fornberg weights, banded storage, no-pivot banded LU |
| `velocity_mod.F90`, `mean_mod.F90` | — | `recover_uw`, the CFR mean |
| `decomp_mod.F90` | — | the pure block/count arithmetic the transpose uses |

**`src/parallel/` — 5,943 lines, 10 modules.**  The production path.

| file | lines | what |
|---|---:|---|
| `main_par.F90` | 2056 | the driver: deck, setup, time loop, outputs, guards |
| `par_io_mod.F90` | 995 | collective MPI-IO, basis-stamped, atomic |
| `par_transpose_mod.F90` | 723 | the NCCL transpose + the `ptp_*` pipelined primitives |
| `par_pressure_mod.F90` | 613 | diagnostic pressure (Neumann–Poisson), 1 hand kernel |
| `par_spectral3d_mod.F90` | 457 | the distributed transform + the overlap phases |
| `par_stepper_mod.F90` | 339 | the SMR substage with the hoists (§11) |
| `par_nonlinear_mod.F90` | 321 | the distributed RHS, phased begin/mid/end |
| `par_mod.F90`, `nccl_mod.F90` | 189 / 161 | ranks, streams, the NCCL bindings |
| `par_stats_mod.F90` | 89 | distributed statistics reduction |

**`src/serial/main.F90` (894 lines)** — the single-GPU driver, kept as
the validation oracle for `--validate`.

**`tests/` — 96 programs**, `tools/` — 9 python post-processors,
`src/models/` — 12 executable design models.

## 17. Guided reading order (open the source now)

1. `src/core/types.F90`, `decomp_mod.F90` — kinds and the block
   arithmetic; small, and everything else assumes them.
2. `src/core/cheb_shen.F90` — the Shen Helmholtz solve, the primitive
   the whole y-direction rests on.
3. `src/core/vsolve_if_mod.F90` — the clamped v-solve; read it with §6
   and `docs/influence_matrix.pdf` open.
4. `src/core/stepper_mod.F90` — the SMR substage on one GPU.
5. `src/core/spectral3d_mod.F90` — the dealiased transform, monolithic.
6. `src/parallel/par_transpose_mod.F90` — THE file for the parallel
   story: the wire format, the direct-a2a identity, and the `ptp_*`
   per-field pipeline.
7. `src/parallel/par_spectral3d_mod.F90` — how the transform splits
   around the transpose, and the overlap phases.
8. `src/parallel/par_stepper_mod.F90` — the hoists (§11) in context.
9. `src/parallel/main_par.F90` — the driver end to end.
10. `src/models/m2_distributed_transform_model.py` next to §7 — the
    executable proof that the distributed transform is exact.

Build and run:

    bash scripts/build/build_par.sh main       # -> build_par/Veta_dp
    TP=sp bash scripts/build/build_par.sh main # -> build_par/Veta_sp
    bash run_all_tests.sh                      # single-GPU suite (needs a GPU)
    bash run_all_par_tests.sh                  # multi-GPU suite

The rest of `docs/`:

| doc | what it holds |
|---|---|
| kmm_formulation.pdf | the (v, η) formulation and the SMR/CN time discretization, derived |
| influence_matrix.pdf | the two-Helmholtz factorization and the 2×2 clamped v-solve, derived |
| PULL_REQUEST_HISTORY.md | what each of the 102 development rounds delivered |

## 18. Where the next performance lives

Numbers to carry: comm/compute overlap is measured **1.38× at 8 nodes**
with a **1.5–2.0×** assessed ceiling; the `TP=sp` transform tier is
shipped; post-overlap the step is **comm-bound at every scale ≥ 16
nodes**.

That last point is the hierarchy, and it has a sharp corollary: once
overlap lands, the H200's ~1.28×
compute advantage nearly vanishes, so H100 queues become equally good —
queue-time arbitrage beats hardware shopping.

1. **Finish exploiting overlap** — it is the single biggest lever and the
   campaign decks already enable it; the remaining gap between 1.38×
   and the 1.5–2.0× ceiling is measurement and tuning, not new
   machinery.
2. **The 2-D pencil question** — the 1-D slab has a `P ≤ min(nkx, nzd)`
   ceiling (§10) and the transpose volume is the binding cost.  It was
   assessed and deliberately not adopted at the scales the campaigns run
   at: within one non-blocking network island, slab and pencil move the
   same inter-node volume.  This is the one BYTE lever if the target
   grows.
3. **Stay inside one network island.**  The measured plateau across
   4–16 nodes is a blocked-fabric effect, not a code gap: both Veta and
   Pola converge to the same rate when crossing islands.  Request
   `--switches=1` where the scheduler allows it.
4. **Retire the frozen prototype** (§19) — not performance, but it is
   22% of the code and every structural tool has to work around it.

Anything you attempt: re-derive the premise first, decide the
certification BEFORE coding (bit-identity? a gated tolerance?), and put
a leg in the test runner.

## 19. Known gaps and structural debt

Tracked here rather than rediscovered.  None of these is a correctness
bug; all of them cost somebody time.

- **No test runs without a GPU, and CI never compiles** (§15).  The
  cheapest fix is a host tier: the "CPU reference group" modules
  (`cheb_shen`, `cheb_bih`, `helm_band_mod`, `yops_mod`, `ygrid_mod`,
  `fdops_mod`, `fd_vsolve_mod`, `mean_mod`, `fd_mean_mod`,
  `vsolve_if_mod`) already compile with plain `nvfortran -c`; only
  their test drivers force `-cuda`.
- **There is no deck reference document.**  A user learns the namelist
  from comments inside the example decks — 288 comment lines against
  472 lines of settings, well written but spread across 12 files.  A
  single reference listing every key, its default and its units is the
  obvious thing to write next.
- **The code is nvfortran-only.**  Zero OpenACC, zero OpenMP, zero
  portable pragmas; `!$cuf` is a CUDA Fortran extension no other
  compiler implements.  If non-NVIDIA hardware ever matters, the
  directive layer is a rewrite — whereas Pola's `__host__ __device__`
  cores are already compiler-neutral C.
- **Licensing is settled: MIT** (`LICENSE`, held by the authors named
  there).  Individual source files carry no licence header, so a file
  copied out of the tree does not carry the notice with it; adding an
  `SPDX-License-Identifier: MIT` line per file is the obvious follow-up.

On the credit side, and worth preserving: comment density in the solver
core runs **0.47 comment lines per line of code**, and the comments are
load-bearing — they cite equations, name the bug that motivated a
guard, and quote the measured numbers behind a gate.  Across the whole
repository there are **zero** `TODO`/`FIXME`/`HACK` markers.

## 20. Conventions and things to know

Five things that are not obvious from reading the code, each of which
has cost real debugging time.

**The two "n"s — the single biggest source of off-by-one bugs.**  A
wall-normal field of polynomial *degree* `n` has `n+1` Chebyshev
coefficients, stored `0:n`.  Both numbers are in play and they are not
interchangeable:

- **`n` = degree** is what `cheb_shen`, `helm_band`, `vif`,
  `project_shen` and `shen_to_cheb_c` take.  Shen (Dirichlet) degrees of
  freedom are `0:n-2`; biharmonic degrees of freedom are `0:n-4`.
- **`ny = n+1` = the number of coefficients** is what `apply_L` and
  `cheb_deriv_c` take — they loop over coefficient *count*.

So in the integrator `apply_L` is called with `n+1` and everything else
with `n`.  Field arrays are `(0:n, nkx, nz)`; Shen-space intermediates
are `(0:n-2, nkx, nz)`.  Passing a `(0:n, …)` array to `apply_L`'s
1-based `(ny, …)` dummy works by sequence association (contiguous, same
size) — which is why getting it wrong is silent rather than fatal.

**Fortran is case-insensitive, and it has bitten this code repeatedly.**
`dp`-the-kind versus `dp`-a-variable, `pw` versus `pW`, `p` versus `P`
are all the same symbol.  Hence the standing naming rules: `pl` for cuFFT
plans, `pscr`/`scr` for scratch, and wall-derivative outputs named
`dlo`/`dhi` rather than `dl`/`dh`.  `tools/gate2_clean.py` scans for
case-fold collisions.

**Three static gates run on every Fortran file**, in CI and before any
commit.  They are pure Python, need no GPU and no compiler, and each has
caught a silent bug at least once:

1. `tools/gate_general.py` — structural balance.  Every
   `subroutine`/`end subroutine`, `module`, `do`, `if-then` and derived
   type opens and closes, and there is exactly one `contains` per module.
   A mismatch produces a cascade of misleading declaration errors that
   mask the real cause.
2. `tools/gate23.py` — the device/host audit.  Every CUDA kernel and
   `device` dummy must receive a `device` array, and a CUF directive may
   not reference a derived-type device-allocatable member (NVFORTRAN-0155
   — a real nvfortran trap, which is why those are wrapped in
   plain-array helper subroutines).
3. `tools/gate2_clean.py` — the case-fold scan above.  Advisory: a hit is
   a review prompt, not an automatic reject.

**The scheme is second order in time, not third.**  It is an RK3, but the
symmetric Crank–Nicolson viscous weights dominate the RK3 nonlinear side;
`test_stepper_order_r2c` measures p ≈ 2 directly.  Do not assume third
order when choosing a time step or interpreting a convergence study.

**Two limits on the validation.**  `--validate` runs the Chebyshev path
(the oracle path), so the FD-y distributed path is covered by its own
unit rungs and a single-GPU KMM twin rather than by an end-to-end
distributed digit gate.  And on the influence-matrix path, fp64 at
`ny ≈ 1200` (condition number ~1e12) leaves roughly four accurate
digits — add a residual check at the high end.  The *dense* `cheb_bih`
reference path scales like `ny^8` and is a small-`n` CPU oracle only;
never run it at production sizes.

## 21. Glossary

| term | meaning |
|---|---|
| DNS | direct numerical simulation — all scales resolved, no model |
| Re_τ | friction Reynolds number u_τh/ν — the turbulence "level" |
| wall units (y⁺) | lengths scaled by ν/u_τ — resolution is judged here |
| KMM | Kim–Moin–Moser: the (v, η) formulation that eliminates pressure |
| v, η | wall-normal velocity and wall-normal vorticity — the state |
| `recover_uw` | algebraic reconstruction of u, w from (v, η) per mode |
| CFR | constant flux rate — the drive; the pressure gradient is output |
| Shen basis | Chebyshev recombination giving banded Dirichlet operators |
| FD-y | the 8th-order banded finite-difference y-backend on a file grid |
| influence matrix | the 2×2 per-mode recovery of the clamped Dv(±1)=0 conditions |
| SMR-RK3 | the Spalart–Moser–Rogers 3-substage stepper (γ, ζ, C tableau) |
| CN | Crank–Nicolson — the implicit viscous treatment |
| Layout A / B | z-slab (kx whole) / kx-slab (z whole) — §4 |
| a2a / transpose | the NCCL all-to-all converting between the layouts |
| `a2a_overlap` | 0 = monolithic exchange, 1 = per-field pipelined + hoists |
| HOIST A / B | stepper operator work moved inside the flight window (§11) |
| `ev_pk` / `ev_cm` | the per-field CUDA events ordering pack → comm → consume |
| wp / tp | working precision (state, solves) / transform precision |
| `TP=sp` | the shipped fp32 transform tier; solves stay fp64 |
| R2C | real-to-complex transform — only kx ≥ 0 stored |
| Hermitian fix | re-enforcing conjugate symmetry after a nonlinear evaluation |
| dealias (3/2) | products evaluated on a 1.5× grid, then truncated |
| `!$cuf kernel do` | the CUDA Fortran directive that generates a kernel (116 uses) |
| basis stamp | the checkpoint field recording which y-backend wrote it |
| GLC | Gauss–Lobatto–Chebyshev nodes |
| `stepper_ws` | the run-lifetime device scratch (no per-step malloc/free) |

