#!/usr/bin/env bash
# =============================================================================
# build_par.sh -- the UNIFIED multi-GPU builder. Compiles the full module chain
# once into build_par/, then links the requested target(s). The toolchain
# boilerplate lives in scripts/build/_mpi_env.sh.
#
# Targets:
#   all              modules + main (Veta_$TP) + every tests/parallel/test_*
#   modules          just compile the module objects
#   main             modules + main_par -> build_par/Veta_$TP   (production binary)
#   <test_stem>      modules + that one test (e.g. test_step_full_mp) -> _$TP binary
#
# Usage (BUILD modules loaded; openmpi must be the nvhpc-built 5.0.8):
#   module load nvhpc/24.11 openmpi/5.0.8-nvhpc24.11-cuda_old nccl/2.28.7-gcc11.5.0-cuda
#   TP=dp bash scripts/build/build_par.sh all       # everything, dp
#   TP=sp bash scripts/build/build_par.sh all       # everything, sp
#   TP=dp bash scripts/build/build_par.sh main      # just the production Veta_dp
#   TP=sp bash scripts/build/build_par.sh test_step_full_mp
#
# Binaries: build_par/Veta_<tp> (production) and build_par/<test>_<tp> (tests).
# Run them via the scripts/sbatch/*.sbatch files (gcc-built Open MPI 5.0.5 launcher).
# =============================================================================
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

. scripts/build/_mpi_env.sh        # -> TP CPPFLAGS FC CU CUF IFLAGS LFLAGS MPI_LL NCCL_LL OBJ

C=src/core
P=src/parallel
T=tests/parallel

# cc <CU|CUF|CPP> <src.F90> [mpi]   -- compile one module; 'mpi' adds the mpif.h -I path
cc() {
  local kind="$1" src="$2" mpi="${3:-}"
  local name; name="$(basename "${src%.F90}")"
  local fl; case "$kind" in CU) fl="$CU";; CUF) fl="$CUF";; CPP) fl="$CPPFLAGS";; esac
  local extra=""; [ "$mpi" = "mpi" ] && extra="$IFLAGS"
  # shellcheck disable=SC2086
  $FC $fl $extra -module "$OBJ" -c "$src" -o "$OBJ/$name.o"
}

build_par_modules() {
  echo "[build_par] compiling module chain (TP=$TP) -> $OBJ/"
  cc CU  $C/types.F90
  cc CPP $C/wavenum_mod.F90        # main_par uses the canonical builders
  cc CPP $C/diag_mod.F90           # main_par calls fft_size_guard (CPU-only module)
  cc CPP $C/cheb_shen.F90
  cc CPP $C/cheb_bih.F90
  cc CPP $C/yops_mod.F90
  cc CPP $C/ygrid_mod.F90
  cc CPP $C/fdops_mod.F90
  cc CPP $C/fd_vsolve_mod.F90
  cc CU  $C/init_mod.F90
  cc CPP $C/helm_band_mod.F90
  cc CPP $C/mean_mod.F90
  cc CPP $C/fd_mean_mod.F90
  cc CPP $C/vsolve_if_mod.F90
  cc CUF $C/transforms_mod.F90
  cc CUF $C/cheb_fft_mod.F90
  cc CUF $C/fft3d_mod.F90
  cc CUF $C/spectral3d_mod.F90
  cc CU  $C/cheb_ops_mod.F90
  cc CU  $C/yderiv_gpu_mod.F90
  cc CU  $C/stats_mod.F90
  cc CU  $C/cfl_mod.F90
  cc CU  $C/velocity_mod.F90
  cc CU  $C/forcing_mod.F90
  cc CUF $C/nonlinear_mod.F90
  cc CU  $C/helm_band_gpu_mod.F90
  cc CU  $C/fd_helm_gpu_mod.F90
  cc CU  $C/fd_vsolve_gpu_mod.F90
  cc CU  $C/spec_helpers_mod.F90
  cc CU  $C/mean_gpu_mod.F90
  cc CU  $C/fd_mean_gpu_mod.F90
  cc CU  $P/par_pressure_mod.F90
  cc CU  $C/vsolve_if_gpu_mod.F90
  cc CU  $C/solver_ops_mod.F90
  cc CU  $C/stepper_lin_mod.F90
  cc CUF $C/stepper_mod.F90
  cc CPP $C/decomp_mod.F90
  cc CU  $C/transpose_mod.F90
  cc CU  $P/nccl_mod.F90
  cc CU  $P/par_mod.F90            mpi
  cc CU  $P/par_io_mod.F90         mpi
  cc CU  $P/par_stats_mod.F90
  cc CU  $P/par_transpose_mod.F90
  cc CUF $P/par_spectral3d_mod.F90
  cc CUF $P/par_nonlinear_mod.F90
  cc CU  $P/par_stepper_mod.F90
  echo "[build_par] modules OK"
}

# module objects to link = everything except program objects (main_par) and tests
modobjs() { ls "$OBJ"/*.o 2>/dev/null | grep -vE "/(main_par|test_)[^/]*\.o$"; }

veta_build_inc() {
  BH=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
  BD=$(date '+%b %e %Y %H:%M:%S')
  {
    printf "      character(*), parameter :: VETA_BUILD_HASH = '%s'\n" "$BH"
    printf "      character(*), parameter :: VETA_BUILD_DATE = '%s'\n" "$BD"
  } > "$OBJ/veta_build.inc"
}

link_main() {
  echo "[build_par] link main_par -> $OBJ/Veta_$TP"
  veta_build_inc
  # shellcheck disable=SC2046,SC2086
  $FC $CUF $IFLAGS -I "$OBJ" -module "$OBJ" -c "$P/main_par.F90" -o "$OBJ/main_par.o"
  $FC $CUF $IFLAGS -module "$OBJ" "$OBJ/main_par.o" $(modobjs) $LFLAGS $MPI_LL $NCCL_LL -o "$OBJ/Veta_$TP"
  echo "   OK -> $OBJ/Veta_$TP"
}

link_test() {
  local t="$1"
  [ -f "$T/$t.F90" ] || { echo "ERROR: no such test $T/$t.F90" >&2; exit 2; }
  echo "[build_par] link $t -> $OBJ/${t}_$TP"
  # shellcheck disable=SC2046,SC2086
  $FC $CUF $IFLAGS -module "$OBJ" -c "$T/$t.F90" -o "$OBJ/$t.o"
  $FC $CUF $IFLAGS -module "$OBJ" "$OBJ/$t.o" $(modobjs) $LFLAGS $MPI_LL $NCCL_LL -o "$OBJ/${t}_$TP"
  echo "   OK -> $OBJ/${t}_$TP"
}

TARGETS="${*:-all}"
build_par_modules
for tg in $TARGETS; do
  case "$tg" in
    modules) : ;;
    all)
      link_main
      for f in "$T"/test_*.F90; do link_test "$(basename "${f%.F90}")"; done ;;
    main|main_par) link_main ;;   # main_par accepted as an alias
    test_*)  link_test "$tg" ;;
    *) echo "ERROR: unknown target '$tg' (use: all | modules | main | main_par | <test_stem>)" >&2; exit 2 ;;
  esac
done
echo "[build_par] done ($TARGETS, TP=$TP)"
