# Development history — what each pull request delivered

This release is the tree as it stood when the code was frozen for the write-up.
The work arrived as **102 pull requests**, each one turn of a single loop:
design (in numpy, wherever the change was index, layout, precision or
coefficient arithmetic), implement, run the static gates, build and run both
test suites on real GPUs, then merge. This document indexes them, so a reader
can see how the code reached its present state.

The phases are those of the accompanying paper. **Phase F2 does not exist for
this code.** Phase F1 produced no pull requests at all: the entire single-GPU
solver — formulation, module-by-module construction with a validated unit test
for each, the Re_τ ≈ 180 turbulence validation, the real-to-complex conversion
in *x*, and the three-tier precision refactor — was built in chat sessions
before the repository existed.

Categories, where useful: **feature**, **fix**, **hardening** (guards, gates,
memory, correctness margins), **perf**, **validation**, **docs**, **ops** (decks,
launchers, campaign plumbing), **process**.

---

## Phase F1 — The single-GPU solver (2026-06-02 → 2026-06-20, 0 PRs)

Built entirely before the repository: the whole serial GPU solver, one validated
unit test per module, the Re_τ ≈ 178 turbulence reproduction, the real-to-complex
transform conversion, and the three-precision architecture. It enters the
repository as the first commit, and serves from then on as the oracle every
distributed result is checked against.

---

## Phase F3 — Multi-GPU port and production driver (2026-06-20 → 2026-06-27, PRs 1–38)

*Eight days.* The single-GPU solver becomes a distributed one — a one-dimensional
slab decomposition with all-to-all transposes over NCCL — climbs a validation
sequence from laminar flow through linear instability to turbulence, acquires a
namelist-driven production program that can bit-compare itself against the
single-GPU oracle, and then grows the production tooling on top: distributed
I/O, a diagnostic pressure solve, statistics, a moving reference frame and an
adaptive time step.

| # | merged | title | what it changed |
|---|---|---|---|
| 1 | 2026-06-20 | First full time step running across GPUs, plus the cross-GPU reductions | the whole stepper and the sum/max primitives the diagnostics need went distributed |
| 2 | 2026-06-20 | Make the single-precision build compile, and stress the transpose on uneven slabs | five build failures fixed; new stress tests for lopsided and empty slabs |
| 3 | 2026-06-20 | More transpose stress: prime grid sizes, where the slab split is worst | worst-case index arithmetic pinned by test |
| 4 | 2026-06-20 | Distributed diagnostics 1: time-step, divergence and watchdog vs one GPU | the monitoring path checked digit-for-digit against the oracle |
| 5 | 2026-06-20 | Distributed diagnostics 2: the same checks on a real reconstructed velocity | moved off synthetic fields onto a velocity rebuilt from the solver state |
| 6 | 2026-06-20 | Distributed diagnostics 3: turbulence statistics gathered across GPUs | Reynolds stresses and friction Reynolds number, accumulated in parallel |
| 7 | 2026-06-20 | Validation rung A: recover the exact laminar parabolic profile, distributed | the first physics result off the distributed code |
| 8 | 2026-06-20 | Validation rung B: instability growth rate against Orszag's 1971 eigenvalue | the linear-physics rung against a published number |
| 9 | 2026-06-20 | Validation rung C: turbulent initial condition and statistics, distributed | the turbulence rung |
| 10 | 2026-06-21 | The production program: one input file, and a built-in check against one GPU | `main_par` becomes a real program with `--validate` |
| 11 | 2026-06-21 | Chase an apparent single-precision error near the wall — it was run-to-run scatter | diagnosis plus the observability that proved it |
| 12 | 2026-06-22 | Close the single-precision question with a multi-seed campaign and its data | different random seeds span the apparent deficit |
| 13 | 2026-06-22 | First profiler-led speed-up: reorder the x-transforms, coalesce the y solve | plus the timer and profiling tooling |
| 14 | 2026-06-23 | Make multi-node communication use InfiniBand, not TCP — 5.9× faster step | the library had silently fallen back to sockets; also added CI |
| 15 | 2026-06-24 | Solve the mean flow on the GPU instead of shipping it to one host rank | removes a communication desync (architecture, not speed) |
| 16 | 2026-06-24 | Instrument the transpose and sweep node counts: communication is bandwidth-bound | established the ~3.4× cost of crossing nodes |
| 17 | 2026-06-24 | Make a long start-up legible: banner first, then staged progress lines | plus a documentation refresh |
| 18 | 2026-06-25 | Cut start-up from ~270 s to under a second by skipping unused initial modes | the intensity sum was scanning out-of-band columns |
| 19 | 2026-06-25 | Re-plan the I/O, statistics and pressure work before building any of it | the design the three features were then built from |
| 20 | 2026-06-25 | Model the on-disk field ordering in numpy first, then start the parallel writer | the "single-shift" layout, modelled in numpy first |
| 21 | 2026-06-25 | The distributed checkpoint writer and reader, validated at 1, 2 and 4 GPUs | state can now be saved from many ranks |
| 22 | 2026-06-25 | Wire checkpoint and restart into the production program | a run can be resumed |
| 23–24 | 2026-06-25 | The clean physical-space inverse transform that snapshots need | #24 re-issues #23, which was lost in a merge race |
| 25 | 2026-06-25 | Write full physical velocity-field snapshots to disk | writer plus its test |
| 26 | 2026-06-25 | Wire snapshots into the driver, with a start time and a Python reader | skip the transient, read the files back |
| 27 | 2026-06-25 | Model the pressure solve in numpy and check it before writing any Fortran | the Neumann–Poisson solve, proved on paper first |
| 28 | 2026-06-26 | The diagnostic pressure solver, checked against a manufactured exact solution | pressure as a post-processed diagnostic |
| 29 | 2026-06-26 | Wire pressure into the driver as a fourth snapshot field | with alias-free recovery |
| 30 | 2026-06-26 | Fix a compiler error in the pressure kernel by nesting its loops tightly | a real compiler restriction, not a workaround |
| 31 | 2026-06-26 | Pressure statistics and drive force, validated against the single-GPU reference | mean pressure, its variance and its wall-normal gradient |
| 32 | 2026-06-26 | Fix compile errors and a crash on the GPU/host boundary of the pressure path | production enablement |
| 33 | 2026-06-26 | Fix a crash at the first log step: copy the flow-rate control value back properly | a device-to-host transfer through a proper buffer |
| 34 | 2026-06-26 | An offline tool that moves a saved state onto a different grid resolution | spectrally exact, no interpolation |
| 35 | 2026-06-26 | Advect in a moving frame so the mean flow no longer sets the time-step limit | exact by Galilean invariance |
| 36 | 2026-06-27 | Build the wall-normal solve operators on the GPU in one batch: ~100 s → under 1 s | which is what made an adaptive time step affordable at scale |
| 37 | 2026-06-27 | Adaptive time step across GPUs, with outputs landing on exactly the requested times | not on the nearest step |
| 38 | 2026-06-27 | Harden adaptive stepping and exit cleanly before a job hits its wall-clock limit | checkpoint, timestamped statistics, graceful stop |

## Phase F4 — Performance, memory and a second wall-normal discretization (2026-06-28 → 2026-07-12, PRs 39–86)

*Fifteen days.* The tree is reorganised and the two recurring architectural
questions — two-dimensional decomposition, and single-precision fields — are
closed with measurements rather than opinion. The resulting findings are worked
off in batches that cut hundreds of bytes per grid point. Then the big
capability step: a complete second wall-normal discretization
(eighth-order banded finite differences on a grid supplied as a file) is built
behind a single dispatch seam and validated rung by rung against the Chebyshev
path, while communication and computation are made to overlap.

| # | merged | title | what it changed |
|---|---|---|---|
| 39 | 2026-06-28 | Reorganise the tree, unify the parallel build, add the multi-GPU test suite | and answer the two-dimensional-decomposition and single-precision-field questions with numbers |
| 40 | 2026-06-28 | Work out on paper what overlapping communication with compute can actually buy | also: the transpose is already at its data-volume floor |
| 41 | 2026-06-28 | Add large Re_τ ≈ 2000 decks for profiling the communication/compute split | profiling vehicles, not science runs |
| 42 | 2026-06-28 | A switch that makes the communication library report which network path it took | rail and routing visibility on demand |
| 43 | 2026-07-01 | Fix the profiling decks: the turbulent initial condition runs out of memory there | switch them to a laminar start |
| 44 | 2026-07-01 | Cost out a Re_τ = 20 000 run and turn the findings into a work plan | the plan behind the next nine pull requests |
| 45 | 2026-07-02 | Memory batch 1: delete dead solver scratch, and add overflow and sanity guards | −57 bytes per grid point |
| 46 | 2026-07-02 | Memory batch 2: one shared transform plan set, reused monitor scratch | −100 to −150 bytes per grid point; plus a wall-normal-basis gate |
| 47 | 2026-07-02 | Record the full-suite result behind batch 2, and a deck memory note | the full-suite result behind that batch |
| 48 | 2026-07-02 | Memory batches 3–4: one shared transform workspace, two numerics fixes | and an initial condition built per-GPU instead of replicated |
| 49 | 2026-07-02 | Executable numpy studies for the next two features: stretched grid, cheaper solve | designs proved before any Fortran |
| 50 | 2026-07-02 | Settle the wall-normal grid law, and test the cheaper solve at production size | natural stretching wins; the solve idea survives at realistic conditioning |
| 51 | 2026-07-02 | Memory batch 5: overlay scratch buffers that are never live at the same time | −~80 bytes per grid point in production |
| 52 | 2026-07-02 | Halve checkpoint size with an optional single-precision payload | recorded in the file header, refused on mismatch |
| 53 | 2026-07-04 | Gather the band, v-solve and mean operators into named types | the seam a second wall-normal backend will plug into |
| 54 | 2026-07-04 | Trial decks at Re_τ ≈ 2000, and the plan for the whole-machine scaling slot | campaign preparation |
| 55 | 2026-07-04 | Make start-up GPU failures name the step that failed; plan the single-stream move | failures stop being anonymous |
| 56 | 2026-07-04 | Put all GPU work on one stream — the prerequisite for overlapping comms with compute | includes the follow-up fix it needed |
| 57 | 2026-07-04 | A pointer file that lets a job resume without editing its deck | plus the trial-run verdicts and capability decks |
| 58 | 2026-07-04 | Record that a restart reproduces the trajectory exactly, even at a different GPU count | restart validated cross-rank-count |
| 59 | 2026-07-04 | Two default-off switches: parallel-I/O tuning hints, and freeing memory after snapshots | both opt-in |
| 60 | 2026-07-04 | Rebuild the solve operators without allocating — fixes an out-of-memory crash | when the time step changes mid-run |
| 61 | 2026-07-05 | Files now carry their own x/y/z coordinates | older readers still work unchanged |
| 62 | 2026-07-05 | Finite-difference backend, step 1: the wall-normal grid becomes an input file | a generator plus a reader; the solver stops computing the grid |
| 63 | 2026-07-05 | Finite-difference backend, step 2: difference weights, banded derivatives, factoriser | the operator kit |
| 64 | 2026-07-05 | Finite-difference backend, step 3: the batched implicit wall-normal solve, on GPU | the finite-difference counterpart of the Chebyshev solve |
| 65 | 2026-07-05 | Finite-difference backend, step 4: wall derivatives and the fourth-order operator | the primitive set the v-solve needs, host and device |
| 66 | 2026-07-05 | Finite-difference backend, step 5: the influence-matrix v-solve, gated against Chebyshev | two different discretizations forced to agree |
| 67 | 2026-07-05 | Finite-difference backend, step 6: the same v-solve, batched on the GPU | device implementation |
| 68 | 2026-07-08 | Write down how to finish the finite-difference backend | the plan for the remaining steps |
| 69 | 2026-07-08 | Fix a real hang at 128 GPUs and beyond during the first collective | root-caused to a connection-setup storm; pinned by configuration |
| 70 | 2026-07-09 | Write up the whole-machine scaling run against what the network topology allows | measured versus achievable |
| 71 | 2026-07-09 | Correct the campaign figures, and reject an alternative time integrator on cost | it loses at this stability limit |
| 72 | 2026-07-10 | Overlap the all-to-all with computation: pipeline the transpose field by field | the first working overlap |
| 73 | 2026-07-10 | One seam, two wall-normal backends: the finite-difference path enters the stepper | a backend tag dispatches every wall-normal primitive |
| 74 | 2026-07-10 | Record what is validated at the backend seam, and the precedent behind it | validation stamps |
| 75 | 2026-07-10 | Finite-difference backend, step 7: the mean-flow and constant-flow-rate machinery | the zero-wavenumber solve |
| 76 | 2026-07-10 | Finite-difference backend, step 8: the mean-flow solve on the GPU | dispatched from the stepper |
| 77 | 2026-07-10 | Finite-difference backend, step 9: wall-normal derivatives dispatch per backend too | the transform-tier seam |
| 78 | 2026-07-11 | Finite-difference backend, step 10: the first end-to-end finite-difference time steps | identity legs, initial condition, and the gate that proves the step |
| 79 | 2026-07-11 | Finite-difference backend, step 11: one deck key turns it on in both programs | plus a smoke-test deck |
| 80 | 2026-07-11 | Cache the finite-difference factorisations: 47.8 → ~17 ms per step | the finite-difference path becomes as cheap as Chebyshev |
| 81 | 2026-07-11 | Turbulence validation: finite differences match Chebyshev and the 1999 reference | at Re_τ ≈ 178, over 130 000 coupled steps |
| 82 | 2026-07-11 | The analytic checks for the finite-difference backend — and the code gets its name | instability growth rate and a manufactured solution |
| 83 | 2026-07-11 | More overlap: hoist the implicit solver's linear-operator work under the transfers | more compute hidden behind communication |
| 84 | 2026-07-12 | Files record which wall-normal basis wrote them; finite-difference campaign twins | a Chebyshev checkpoint can no longer be restarted on a difference grid |
| 85 | 2026-07-12 | The pressure solve gets its finite-difference backend too | influence matrix on point values |
| 86 | 2026-07-12 | Refresh the state-of-the-code notes at the end of the phase | documentation refresh |

## Phase F5 — Hardening, memory and correctness (2026-07-13 → 2026-08-04, PRs 87–95)

*Twenty-three days, of which the first eighteen produced no pull requests at
all.* Production campaigns ran throughout. The PR-free stretch is not idle time
— it is queue and allocation time, and it is the clearest sign that the pace of
this project was set by machine access, not by how fast code could be written.
The nine pull requests that follow harden the code tier by tier, ending with the
campaigns re-sized to the machine actually available.

| # | merged | title | what it changed |
|---|---|---|---|
| 87 | 2026-08-01 | A checkpoint can no longer report success when it failed | every parallel write and read is now verified and agreed across ranks; plus campaign operating traps |
| 88 | 2026-08-02 | Make the moving frame backend-aware, and the tools honest | the frame shift was wrong at interior points on the difference grid; tools stop reporting what they cannot know |
| 89 | 2026-08-02 | Run the wall-normal transform in place and delete its buffer | ~1.26 GB per GPU returned at 16 nodes |
| 90 | 2026-08-02 | Make optimiser and GPU-target flags identical across all three build paths | the serial build had been unoptimised and untargeted |
| 91 | 2026-08-02 | Fuse the time-stepper's vector kernels and stop copying the history | −12 memory passes per substage, −40 bytes per grid point |
| 92 | 2026-08-02 | Send one side of the transpose straight from its buffer; shrink the stepper workspace | the two largest remaining memory levers |
| 93 | 2026-08-03 | Close the remaining confirmed correctness findings across the codebase | guards, gates, and the finite-difference memory sequel |
| 94 | 2026-08-03 | Match the statistics quadrature to the scheme; add a residual probe | a probe that can catch a silently wrong solve before a campaign burns on it |
| 95 | 2026-08-04 | Re-size the production campaigns to the machine actually available | plus a minutes-long single-node calibration case to run first |

## Phase F6 — Final campaigns, validation, reporting and closure (2026-08-04 → 2026-08-16, PRs 96–102)

*Thirteen days.* The production campaigns run at Re_τ 550, 1000 and 2000 with
Chebyshev and finite-difference twins; the results are checked against Lee &
Moser (2015); the scaling campaign is written up for the machine's operators;
and every remaining open item is closed — fixed, folded into another change, or
accepted with a stated reason.

| # | merged | title | what it changed |
|---|---|---|---|
| 96 | 2026-08-08 | Residual probes in the solver's own basis, plus a tool to pool statistics across jobs | a probe consistent with the discretization it measures |
| 97 | 2026-08-11 | Write up the scaling campaign and current validation for the machine's operators | the benchmark report |
| 98 | 2026-08-14 | Rework that report and add its scaling figure | restructure and revisions |
| 99 | 2026-08-15 | Production statistics at Re_τ 550 and 1000 against Lee & Moser, both backends | validation at campaign scale |
| 100 | 2026-08-16 | Accuracy work: close the outstanding items; the integrator asserts its own coefficients | plus timestamped statistics from the single-GPU program |
| 101 | 2026-08-16 | Low-priority items: cache stamping, a mean-solver re-zero, a documentation sweep | the sweep brought stale claims across the docs up to date |
| 102 | 2026-08-16 | More low-priority items: I/O gates, 64-bit statistics counters, a Nyquist-mask fix | the last hardenings before the freeze |

---

*This release is the tree at the end of PR 102. The next pull request, #103,
carried only the shared data schemas used to write the paper, and is not part
of the code.*
