# Veta — a GPU channel-flow DNS solver (single- and multi-GPU)

Direct Numerical Simulation of turbulent plane channel flow in CUDA Fortran
(nvfortran), GPU-resident and multi-node. Kim–Moin–Moser velocity /
wall-normal-vorticity formulation; Fourier in *x*, *z* (R2C in *x*); Chebyshev
(Shen–Galerkin) or 8th-order banded finite differences in *y*, behind one
dispatch seam; SMR-RK3 with Crank–Nicolson viscous treatment. Validated against
Moser–Kim–Mansour (1999) at Re_τ ≈ 180 and Lee & Moser (2015) at Re_τ 550 and
1000; the multi-GPU driver (1-D kx-slab + NCCL all-to-all) is checked against
the single-GPU oracle with `--validate`, dp and sp, at P = 1..96+.

**Read `docs/ZERO_TO_VETA.md` first** — the complete tour of the code, from the
physics to a file-by-file map and a guided reading order. This file covers only
how to build and run.

| also in `docs/` | |
|---|---|
| `kmm_formulation.pdf` | the (v, η) formulation and time discretization, derived |
| `influence_matrix.pdf` | the two-Helmholtz factorization and clamped v-solve, derived |
| `PULL_REQUEST_HISTORY.md` | what each of the 102 development rounds delivered |

## Build

The single-GPU program needs only nvfortran (NVHPC, which ships cuFFT); the
multi-GPU program additionally needs a CUDA-aware MPI and NCCL. Developed and run
with NVHPC 24.11, Open MPI 5.0.x and NCCL 2.28 on H100/H200 nodes; the GPU target
defaults to `cc90` (`GPUCC=cc80` for A100). A GPU is required for everything,
including the tests.

```bash
make                                          # -> build/Veta_serial      (single-GPU, fp64 transforms)
make TP=sp                                    # -> build/Veta_serial      (single-GPU, fp32 transforms)
TP=dp bash scripts/build/build_par.sh main    # -> build_par/Veta_dp      (multi-GPU)
TP=sp bash scripts/build/build_par.sh main    # -> build_par/Veta_sp      (multi-GPU, the production tier)
make help                                     # every target, including the two test suites
```

`TP` sets the transform precision tier (the transforms and the all-to-all payload);
the wall-normal solves stay fp64 either way. The two build paths spell the fp64 tier
differently: the Makefile takes `TP=real64` (its default) or `TP=sp`, the parallel
build takes `TP=dp` or `TP=sp` and rejects anything else.

### Consumer GPUs (tests underway)

Everything above was built and run on data-centre GPUs. Nothing in the single-GPU
program is specific to them: it needs only nvfortran with CUDA Fortran support and
cuFFT, both part of the NVIDIA HPC SDK, and a CUDA-capable NVIDIA GPU, so it should
build and run on a GeForce or workstation card. **We are trying this ourselves right now**;
the notes below follow from the code and from NVIDIA's published specifications.

For an RTX5060, use `cc120`.

- *Software.* Linux (x86-64 or arm64; the HPC SDK does not target Windows or macOS),
  the NVIDIA HPC SDK (free download; developed with 24.11), an NVIDIA driver at least
  as new as the CUDA release bundled with that SDK, GNU make, and, for the scripts in
  `tools/`, Python 3 with numpy. No MPI and no NCCL are needed for `make`.
- *GPU target.* Pass the card's compute capability: `make GPUCC=cc75` for the RTX 20
  series, `cc86` for RTX 30, `cc89` for RTX 40, `cc120` for RTX 50 (Blackwell needs
  HPC SDK 25.x with CUDA 12.8 or newer); the same variable drives the test suite,
  `GPUCC=cc89 bash run_all_tests.sh`.
- *Memory.* `examples/input_kmm.nml` (128 × 129 × 128) fits any current card. The
  decks under `examples/campaigns/` were sized for 80–96 GB cards and will not fit.
  The program prints its device-memory footprint per precision tier at startup, so
  the first screen tells you whether a deck fits before it runs.
- *fp64 throughput.* Parts of the solver run in fp64 by design and cannot be switched
  off: the implicit wall-normal solves and their factorizations, the wall-normal
  operators, the mean-flow solve, and the time and statistics accumulations (the `dp`
  kind in `src/core/types.F90`). The default `make` build is fp64 throughout; `make
  TP=sp` moves the transforms, the nonlinear products and the transform scratch to fp32
  and is the configuration to use on a consumer card. GeForce cards execute fp64 at
  1/64 of their fp32 peak (RTX 30, 40 and 50 series; 1/32 on the RTX 20 and GTX 10
  series), against 1/2 on A100 and H100, so relative to fp32 the fp64 parts are up to
  32× slower on such a card than they are on the hardware we ran on. How much of that
  reaches the time step depends on how far those kernels are memory-bound rather than
  arithmetic-bound, which we have not measured on consumer hardware; expect the fp64
  parts to take a much larger share of the step than they do in the numbers quoted in
  the documentation. `WP=sp` (fp32 fields) compiles but is not one of the validated
  configurations.

## Run

A run takes one Fortran namelist deck, documented by comment inside the example
decks; nothing is read from the environment.

```bash
./build/Veta_serial examples/input_kmm.nml    # the Re_tau ~ 180 validation case

sbatch --export=ALL,PREC=sp,DECK=examples/input_kmm.nml scripts/sbatch/run_main_par.sbatch

cd examples/campaigns/Re550 && cp ../../../build_par/Veta_sp . && sbatch veta.sbatch
```

`examples/campaigns/` holds ready-to-run folders, Chebyshev and FD twins at
Re_τ 550/1000/2000.

**Confirm NCCL is using InfiniBand and not TCP** — a silent socket fallback is
5–10× slower and nothing else warns you. The driver prints the warm all-to-all
bandwidth at startup; `CLAUSON_REQUIRE_RDMA=1` turns a fallback into a hard
abort. Keep a job on one node if the grid fits: crossing nodes costs ~3.4×,
because the all-to-all's inter-node half rides InfiniBand rather than NVLink.

## Tests

```bash
bash run_all_tests.sh                            # 73 single-GPU tests
sbatch scripts/sbatch/run_all_par_tests.sbatch   # 23 multi-GPU (MPI+NCCL) tests
```

`-k <substring>` selects tests by name. CI runs only the static Python gates in
`tools/` and a syntax check of the shell scripts — GitHub runners have no
nvfortran and no GPU, so the build and the suites run on the cluster.

## Notes

Site-specific details — module names and versions, partitions, HCA names, the
`libibverbs` preload — live in `scripts/build/_mpi_env.sh` and
`scripts/sbatch/`, and need adapting to your machine.

The reference profiles the solver is validated against (Moser–Kim–Mansour 1999;
Lee & Moser 2015) are published third-party datasets, **not redistributed
here**. `tools/analyze_stats.py --mkm` overlays the 1999 profiles on this code's
statistics output; the Lee & Moser comparison at Re_τ 550 and 1000 is reported in
the accompanying paper.

The RDMA hard-abort (`CLAUSON_REQUIRE_RDMA=1`) is set by the reference launchers
in `scripts/sbatch/`; the campaign launchers under `examples/campaigns/` print the
warning only, so add the variable there if you want the abort.

The programs introduce themselves as *Veta Sublimeson*, the code's full name, and
mention *Pola Sublimeson*: the sister pipe-flow code developed alongside it, which is
not part of this release.

`CLAUSON` (the working name) and `ClOpDNS` (the checkpoint sentinel) predate the
name *Veta* and are load-bearing in file formats and environment variables, so
they are left alone.

## License and citation

Veta is released under the MIT License (see `LICENSE`). If you use it, please
cite the accompanying paper, which describes both this code and its sister pipe
code Pola; `CITATION.cff` carries the reference (GitHub shows it under "Cite this
repository") and will be updated with the arXiv identifier and, later, the
journal reference.

## Authorship

The source code and documentation in this repository were written by Claude
(Anthropic; [claude.ai](https://claude.ai) chat sessions and Claude Code),
working under the direction of Siavash Toosi and Philipp Schlatter (Institute of
Fluid Mechanics, FAU Erlangen-Nürnberg), who set the physical and numerical
requirements, made the design decisions, reviewed the work, and built, tested
and ran every version on the target hardware. Every compiled binary, every test
run and every number in the documentation came off real hardware and passed
through human hands; nothing was accepted on the model's word alone. The
development process is described in the accompanying paper (see
`CITATION.cff`).

The authors, as the parties who specified, directed, verified and assembled the
work, hold the copyright in this repository and release it under the MIT License
(`LICENSE`), so that its reuse is governed by a standard, well-understood
licence rather than by any uncertainty about the status of machine-written code.

## Acknowledgements

Development was supported by the Bavarian Competence Network for Scientific
High Performance Computing (KONWIHR).
