#!/usr/bin/env bash
# =============================================================================
# _mpi_env.sh -- SOURCED by the parallel build (build_par.sh). Resolves the
# nvfortran + CUDA-aware Open MPI + NCCL toolchain ONCE and exports the
# compile/link variables. It is the single home of that toolchain resolution,
# shared by every parallel build.
#
# Exports: TP CPPFLAGS FC CU CUF IFLAGS LFLAGS MPI_LL NCCL_LL OBJ
# Requires the BUILD modules (the 24.11-paired Open MPI 5.0.8; the bare 'openmpi'
# default is 25.7-paired and warn-spams):
#     module unload nvhpc openmpi
#     module load nvhpc/24.11 openmpi/5.0.8-nvhpc24.11-cuda_old nccl/2.28.7-gcc11.5.0-cuda
# RUN later against the gcc-built 5.0.5 (the sbatch files).
#
# Codegen: -O2 -Mnoinline. -O2 auto-inlines the plain-dummy CUF helpers back into
# callers that pass derived-type device-allocatable members -> NVFORTRAN-S-0155;
# -Mnoinline disables JUST that inlining.
# =============================================================================
OBJ="${OBJ:-build_par}"
mkdir -p "$OBJ"

TP="${TP:-dp}"
CPPFLAGS=""
case "$TP" in
  sp) CPPFLAGS="-DCLAUSON_TP_SP" ;;
  dp) : ;;
  *)  echo "ERROR: TP must be sp or dp (got '$TP')" >&2; exit 1 ;;
esac

have() { command -v "$1" >/dev/null 2>&1; }
have nvfortran || { echo "ERROR: nvfortran not in PATH (load the nvhpc module)" >&2; exit 1; }

OMPI_INC=""; OMPI_LIB=""; FLIBS="mpi_mpifh mpi"
if have mpicc; then
  OMPI_INC=$(mpicc --showme:incdirs 2>/dev/null)
  OMPI_LIB=$(mpicc --showme:libdirs 2>/dev/null)
fi
if have mpifort; then
  flibs=$(mpifort --showme:libs 2>/dev/null); [ -n "$flibs" ] && FLIBS="$flibs"
fi
if [ -z "$OMPI_INC" ] && have mpicc; then
  OMPI_INC=$(mpicc -show 2>/dev/null | tr ' ' '\n' | sed -n 's/^-I//p' | head -1)
  OMPI_LIB=$(mpicc -show 2>/dev/null | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)
fi
NCCL="${NCCL_HOME:-${NCCL_ROOT:-}}"

IFLAGS=""; for dir in $OMPI_INC; do IFLAGS="$IFLAGS -I$dir"; done
LFLAGS=""; for dir in $OMPI_LIB; do LFLAGS="$LFLAGS -L$dir"; done
MPI_LL="";  for l in $FLIBS;    do MPI_LL="$MPI_LL -l$l"; done
NCCL_LL=""; [ -n "$NCCL" ] && NCCL_LL="-L$NCCL/lib -lnccl"

# Loud guard: building against the gcc-built 5.0.5 dies on its gfortran mpif.h.
OMPI_PREFIX="$(dirname "$(dirname "$(command -v mpirun 2>/dev/null || command -v mpicc 2>/dev/null || echo /x/x)")")"
case "$OMPI_PREFIX" in
  *openmpi-5.0.5*|*gcc11.4.1*)
     echo "FATAL: building against the gcc-built Open MPI ($OMPI_PREFIX)." >&2
     echo "       nvfortran cannot parse its gfortran mpif.h (NVFORTRAN-S-0047)." >&2
     echo "       module load nvhpc/24.11 openmpi/5.0.8-nvhpc24.11-cuda_old nccl/2.28.7-gcc11.5.0-cuda" >&2
     exit 1 ;;
esac

FC="nvfortran -O2 -Mnoinline"
# -gpu=cc90 pins the device target: without it nvfortran targets the
# GPU on the BUILD machine -- a login-node build ships PTX only and every rank
# JIT-compiles at startup (360 simultaneous JITs on a 90-node job). cc90 covers
# H100 and GH200 alike; override with GPUCC=ccXX for anything else.
GPUCC="${GPUCC:-cc90}"
CU="-cuda -gpu=$GPUCC $CPPFLAGS"
CUF="-cuda -gpu=$GPUCC -cudalib=cufft $CPPFLAGS"

echo "[mpi_env] nvfortran=$(command -v nvfortran)  TP=$TP  GPUCC=$GPUCC"
echo "[mpi_env] OMPI=$OMPI_PREFIX  NCCL=${NCCL:-<unset! load the nccl module>}"

# Soft pin notice: the proven stack is nvhpc/24.11 + the 24.11-paired Open MPI
# 5.0.8 (module openmpi/5.0.8-nvhpc24.11-cuda_old). The cluster preloads 25.7,
# and the bare 'openmpi' default is the 25.7-paired 5.0.8 whose mpif-sizeof.h
# spams NVFORTRAN-W-0031 at every compile. Both drifts still build -- warn
# loudly so the log shows the truth.
case "$(command -v nvfortran)" in
  *24.11*) : ;;
  *) echo "[mpi_env] WARNING: nvfortran is NOT the proven 24.11 (module unload nvhpc; module load nvhpc/24.11)" >&2 ;;
esac
# BLACKLIST, not whitelist: the proven pin
# 'openmpi/5.0.8-nvhpc24.11-cuda_old' resolves to a spack prefix named
# .../gcc-11.5.0/openmpi-5.0.8-<hash>, so a path-token whitelist would
# false-positive on every CORRECT build. Only the 25.7-paired default is the
# known W-0031 spammer; the gcc-5.0.5 case is already FATAL above.
case "$OMPI_PREFIX" in
  *nvhpc-25.7*)
     echo "[mpi_env] WARNING: Open MPI is the 25.7-paired 5.0.8 -- W-0031 spam ahead" >&2
     echo "[mpi_env]          (module unload openmpi; module load openmpi/5.0.8-nvhpc24.11-cuda_old)" >&2 ;;
esac
