#==============================================================================
# Makefile -- build the SERIAL (single-GPU) Veta_serial driver. The parallel
# multi-GPU build (the main production code) is scripts/build/build_par.sh; this
# Makefile carries a `make parallel` target that delegates to it.
#
#   make            # build build/Veta_serial   (single-GPU)
#   make parallel   # delegate to scripts/build/build_par.sh main (-> Veta_dp/sp)
#   make run        # build, then run on examples/for_testing/input_laminar_fixeddt.nml
#   make tests      # delegate to run_all_tests.sh (build + run the unit suite)
#   make clean      # remove build/ objects, modules, and Veta_serial
#   make help
#
# LAYOUT: shared modules in src/core/, the serial program in src/serial/main.F90,
# ALL serial build output (.o, .mod, Veta_serial) in build/.
# Fortran module ordering is explicit (OBJ list below in dependency order) --
# nvfortran needs each module's .mod built before its dependents compile, so we
# do NOT rely on make's file-time inference for module deps; the ordered list is
# the source of truth (same order as run_all_tests.sh).
#
# Requirements: nvfortran (NVHPC), cuFFT. Override the compiler with `make FC=...`.
#==============================================================================
# GNU make PREDEFINES FC (to f77), so `FC ?= nvfortran` never fires and a plain
# `make` runs f77. Override make's own default only -- a command-line or
# environment FC still wins.
ifeq ($(origin FC),default)
  FC := nvfortran
endif
SRC     := src/core
SERIAL  := src/serial
OBJ     := build
# Pin the device target: without -gpu=ccXX nvfortran targets the GPU
# present on the BUILD machine -- a login-node build carries PTX only and every
# process JIT-compiles it at startup (360 simultaneous JITs on a 90-node job).
# cc90 = H100 and GH200 alike. Override: make GPUCC=cc80 ...
GPUCC   ?= cc90
CUDA    := -cuda -gpu=$(GPUCC)
CUFFT   := -cuda -gpu=$(GPUCC) -cudalib=cufft
# ---- precision / codegen flags ----------------------------------------------
# Three-precision architecture (see src/core/types.F90):
#   TP = transform/comm kind,  WP = field kind   (dp is ALWAYS real64, no flag).
# Default = all-real64 (the validated reference build). Flip a kind to FP32:
#   make TP=sp          # FP32 transforms (the fp32-transform tier)
#   make WP=sp          # FP32 fields     (memory-bound)
#   make TP=sp WP=sp    # both FP32
# NOFMA: NOFMA=1 adds -Mnofma, so that an FP32 build differs from the FP64
# baseline ONLY by the intended kind change (FMA contraction otherwise perturbs
# even unchanged FP64 codegen and defeats the per-step bit-identity check).
# The default is the stock (FMA-contracted) codegen, matching the parallel
# build, which never carries -Mnofma. fp32smoke pins NOFMA=1.
TP    ?= real64
WP    ?= real64
NOFMA ?= 0

PRECFLAGS :=
ifeq ($(TP),sp)
  PRECFLAGS += -DCLAUSON_TP_SP
endif
ifeq ($(WP),sp)
  PRECFLAGS += -DCLAUSON_WP_SP
endif
ifeq ($(NOFMA),1)
  PRECFLAGS += -Mnofma
endif

# -module $(OBJ): write .mod here AND search here for already-built deps.
# -O2 -Mnoinline: bare -O2 inlines the plain-dummy CUF helpers into callers that
# pass derived-type device-allocatable members -> NVFORTRAN-S-0155 (loud, at
# compile); -Mnoinline disables exactly that inlining. Under -O2 the
# non-short-circuit `.and.` in the output-cadence checks would evaluate
# mod(it,0); explicit divisor guards keep that from happening.
FFLAGS  := -module $(OBJ) -O2 -Mnoinline $(PRECFLAGS)

# ---- modules in dependency order (mirrors run_all_tests.sh) -----------------
# (main.o is built and linked separately; it is the program entry point.)
MODS := \
  types decomp_mod diag_mod wavenum_mod cheb_shen cheb_bih helm_band_mod yops_mod ygrid_mod fdops_mod fd_vsolve_mod mean_mod fd_mean_mod vsolve_if_mod \
  grid_mod transforms_mod cheb_fft_mod dealias_mod fft3d_mod transpose_mod spectral3d_mod \
  cheb_ops_mod yderiv_gpu_mod stats_mod init_mod cfl_mod io_mod velocity_mod forcing_mod \
  nonlinear_mod helm_band_gpu_mod fd_helm_gpu_mod fd_vsolve_gpu_mod spec_helpers_mod mean_gpu_mod fd_mean_gpu_mod vsolve_if_gpu_mod \
  solver_ops_mod stepper_lin_mod stepper_mod

OBJS := $(addprefix $(OBJ)/,$(addsuffix .o,$(MODS)))
BIN  := $(OBJ)/Veta_serial

# ---- per-module compile flags (most are -cuda; a few need cuFFT; CPU group plain)
CUFFT_MODS := transforms_mod cheb_fft_mod fft3d_mod spectral3d_mod nonlinear_mod stepper_mod
CPU_MODS   := types decomp_mod diag_mod wavenum_mod cheb_shen cheb_bih helm_band_mod yops_mod ygrid_mod fdops_mod fd_vsolve_mod mean_mod fd_mean_mod vsolve_if_mod

# flags(name) = plain for CPU group, -cuda -cudalib=cufft for the cuFFT group, else -cuda
flagof = $(if $(filter $(1),$(CPU_MODS)),,$(if $(filter $(1),$(CUFFT_MODS)),$(CUFFT),$(CUDA)))

.PHONY: all parallel run tests partests clean help fp32smoke
all: $(BIN)

# ---- parallel (multi-GPU) build: delegate to the unified builder -------------
# Needs the BUILD modules loaded (nvhpc + nvhpc-built Open MPI 5.0.8 + NCCL).
# Produces build_par/Veta_dp and build_par/Veta_sp.
parallel:
	TP=dp bash scripts/build/build_par.sh main
	TP=sp bash scripts/build/build_par.sh main

# ---- link the driver --------------------------------------------------------
$(BIN): $(OBJ)/main.o $(OBJS)
	@echo "  LD   $@"
	@$(FC) $(CUFFT) $(FFLAGS) -o $@ $(OBJ)/main.o $(OBJS)
	@echo "  built $@"

# ---- compile main (program; lives in src/serial/) ---------------------------
$(OBJ)/main.o: $(SERIAL)/main.F90 $(OBJS) | $(OBJ)
	@echo "  CC   main.F90  [$(CUFFT)]"
	@BH=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown); \
	 BD=$$(date '+%b %e %Y %H:%M:%S'); \
	 printf "      character(*), parameter :: VETA_BUILD_HASH = '%s'\n      character(*), parameter :: VETA_BUILD_DATE = '%s'\n" "$$BH" "$$BD" > $(OBJ)/veta_build.inc
	@$(FC) $(CUFFT) $(FFLAGS) -I $(OBJ) -c $< -o $@

# ---- compile each module with its appropriate flags -------------------------
# Static pattern rule: one recipe, per-target flags via $(call flagof,stem).
$(OBJS): $(OBJ)/%.o: $(SRC)/%.F90 | $(OBJ)
	@echo "  CC   $*.F90  [$(if $(strip $(call flagof,$*)),$(call flagof,$*),plain)]"
	@$(FC) $(call flagof,$*) $(FFLAGS) -c $< -o $@

# NOTE: the ordered prerequisite trick -- because module M's .mod must exist
# before a dependent compiles, and make may build prerequisites in parallel,
# build SERIALLY (default) or use `make -j1`. For correctness with -j, the
# OBJS list order + the link deps suffice for a clean (serial) build; a parallel
# build would need explicit inter-module .mod deps (not generated here).

$(OBJ):
	@mkdir -p $(OBJ)

# ---- convenience ------------------------------------------------------------
run: $(BIN)
	./$(BIN) examples/for_testing/input_laminar_fixeddt.nml

tests:
	FC=$(FC) TP=$(TP) WP=$(WP) NOFMA=$(NOFMA) GPUCC=$(GPUCC) bash run_all_tests.sh

# ---- parallel unit suite: build then run (see scripts/sbatch for the cluster) -
# NOTE: run_all_par_tests.sh delegates to scripts/build/build_par.sh, which has
# its own TP=dp|sp convention and doesn't read FC/WP/NOFMA/GPUCC -- nothing to
# forward here (unlike `tests`, see above).
partests:
	bash run_all_par_tests.sh build
	@echo "built; run inside an allocation: sbatch scripts/sbatch/run_all_par_tests.sbatch"

# ---- fp32 crossing smoke test ----------------------------------------------
# Compile-only build with BOTH kinds forced to FP32. This is EXPECTED TO FAIL
# until the cast seams land: every error should be one of the documented wp/dp
# (or tp/wp) kind-mismatch crossings. An error that is NOT on that list means
# the crossing inventory missed a site -- which is the point.
# Cleans first (so the FP32 flags actually rebuild every .mod) and cleans again
# afterward (so the tree is left pristine for a normal `make`).
fp32smoke:
	@echo "=================================================================="
	@echo " fp32smoke: clean + compile-only build, TP=sp WP=sp (-Mnofma)."
	@echo " EXPECTED TO FAIL at the documented kind-mismatch crossings"
	@echo " (these close only when the stepper state tier is re-kinded to fp32 -- the"
	@echo " wp<->dp seam covers the solve boundary, not the stepper state;"
	@echo " see src/core/types.F90). Any crossing off that list = inventory gap."
	@echo "=================================================================="
	@$(MAKE) --no-print-directory clean
	-@$(MAKE) --no-print-directory -k TP=sp WP=sp NOFMA=1 $(OBJS) $(OBJ)/main.o
	@echo "------------------------------------------------------------------"
	@echo " (The failure above is EXPECTED.) Send me the kind-mismatch lines."
	@echo " Cleaning the FP32 partial objects so a plain \`make\` rebuilds clean."
	@$(MAKE) --no-print-directory clean

clean:
	rm -f $(OBJ)/*.o $(OBJ)/*.mod $(BIN)
	@echo "  cleaned $(OBJ)/"

help:
	@echo "CLAUSON Makefile targets:"
	@echo "  make            build build/Veta_serial   (single-GPU)"
	@echo "  make parallel   build build_par/Veta_dp + Veta_sp (multi-GPU; needs MPI modules)"
	@echo "  make run        build + run examples/for_testing/input_laminar_fixeddt.nml"
	@echo "  make tests      build + run the serial unit suite (run_all_tests.sh)"
	@echo "  make partests   build the parallel unit suite (run on the cluster via sbatch)"
	@echo "  make fp32smoke  compile-only TP=sp WP=sp (expected to fail at crossings)"
	@echo "  make clean      remove build/ objects, modules, and Veta_serial"
	@echo "  make FC=ftn ... override the compiler"
	@echo ""
	@echo "Precision knobs (defaults in [brackets]):"
	@echo "  TP=real64|sp    transform/comm kind   [real64]   (TP=sp => fp32 transforms)"
	@echo "  WP=real64|sp    field kind            [real64]   (WP=sp => fp32 fields; not validated)"
	@echo "  NOFMA=1|0       add -Mnofma           [0]        (1 => disable FMA contraction)"
	@echo "  GPUCC=ccXX      device target         [cc90]     (H100/GH200; cc80 A100; cc86/cc89/cc120 RTX 30/40/50)"
	@echo "  NOTE: dp is ALWAYS real64 (no flag). Default build = all-real64."
	@echo "  NOTE: build serially (module .mod deps); avoid -j unless you add"
	@echo "        explicit inter-module dependencies."
