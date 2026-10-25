#!/bin/bash -l
# =============================================================================
# run_all_par_tests.sbatch -- request resources, then BUILD and RUN the whole
# parallel unit suite (+ the production --validate oracle diff) in one allocation
# on Helma (NHR@FAU). One-shot: sbatch it and walk away.
#
#   sbatch scripts/sbatch/run_all_par_tests.sbatch                       # 1 node, P<=4
#   sbatch --nodes=2 --ntasks=8 scripts/sbatch/run_all_par_tests.sbatch  # +the P=8 cases
#   sbatch --export=ALL,BUILD=0 scripts/sbatch/run_all_par_tests.sbatch  # skip build (pre-built)
#
# It BUILDS under the nvhpc-built Open MPI 5.0.8, then swaps to the gcc-built 5.0.5
# launcher to RUN (binaries are ABI-stable across 5.0.x). Each test runs at every
# rank count in its set that is <= --ntasks, in dp and sp.
# =============================================================================
#SBATCH --job-name=veta_par_suite
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=par_suite_%j.out

cd "${SLURM_SUBMIT_DIR:-.}"
BUILD="${BUILD:-1}"

# ---- BUILD phase (nvhpc-built Open MPI 5.0.8) --------------------------------
if [ "$BUILD" = "1" ]; then
  # The cluster PRELOADS nvhpc/25.7 -- a bare load of 24.11 conflicts, fails,
  # and the job silently proceeds on 25.7 (whose paired OpenMPI mpif-sizeof.h
  # also spams W-0031 warnings). Unload first, then pin.
  module unload nvhpc 2>/dev/null
  module load nvhpc/24.11
  # The default 'openmpi' module resolves to the 25.7-paired 5.0.8 whose
  # mpif-sizeof.h spams NVFORTRAN-W-0031 at every compile. The clean
  # 24.11-paired build is carried as '*_old' -- pin it; fall back to the
  # default only if the pin fails (a _mpi_env.sh WARNING then fires). The
  # '_old' module resolves to a spack prefix NAMED
  # .../gcc-11.5.0/openmpi-5.0.8-<hash>, so do not test it by path token.
  # The loud fallback stays for the day the module really vanishes.
  module unload openmpi 2>/dev/null
  if ! module load openmpi/5.0.8-nvhpc24.11-cuda_old; then
     echo "[suite] WARNING: openmpi/5.0.8-nvhpc24.11-cuda_old did not load (Lmod's reason above) -> falling back to default openmpi"
     module load openmpi
  fi
  module load nccl/2.28.7-gcc11.5.0-cuda
  echo "[suite] BUILD phase: $(command -v mpirun)"
  bash run_all_par_tests.sh build || { echo "BUILD phase FAILED" >&2; exit 1; }
fi

# ---- RUN phase (gcc-built Open MPI 5.0.5 launcher) ---------------------------
module unload nvhpc 2>/dev/null   # same 25.7-preload conflict as above
module load nvhpc/24.11
module unload openmpi 2>/dev/null
module load openmpi/5.0.5-gcc11.4.1-cuda
module load nccl/2.28.7-gcc11.5.0-cuda

export OMP_NUM_THREADS=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export GPUS_PER_NODE=4

# InfiniBand RDMA for NCCL (else ~1 GB/s IPoIB sockets). Site-specific.
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3,mlx5_4,mlx5_5}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export LD_LIBRARY_PATH=/usr/lib64/libibverbs:$LD_LIBRARY_PATH
export LD_PRELOAD="/lib64/libibverbs.so.1${LD_PRELOAD:+:$LD_PRELOAD}"   # ABI-57 system libibverbs
# NCCL large-scale connection-setup hardening + benign-knem silence -- kept in sync
# with run_main_par.sbatch (the multi-node warm-up hang). OMPI_MCA_smsc auto-forwards
# to ranks (no -x). Avoids the large-node NCCL warm-up hang.
export NCCL_NCHANNELS_PER_NET_PEER="${NCCL_NCHANNELS_PER_NET_PEER:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker,lo,virbr}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-10}"
export OMPI_MCA_smsc="${OMPI_MCA_smsc:-^knem}"

MPI_PREFIX="$(dirname "$(dirname "$(command -v mpirun)")")"
echo "[suite] RUN phase: Open MPI prefix = $MPI_PREFIX ; SLURM_NTASKS=$SLURM_NTASKS ; nodes=$SLURM_NNODES"
# record placement: which nodes/switches (Helma island forensics -- 24 nodes
# = one non-blocking island; 25+ crosses 3:1 blocking; never fit one curve through 25).
echo "[topo] nodelist: $SLURM_JOB_NODELIST"
scontrol show topology 2>/dev/null | head -50 || true
case "$MPI_PREFIX" in
  *openmpi-5.0.5*) : ;;
  *) echo "FATAL: active Open MPI is not the proven gcc-built 5.0.5 -> $MPI_PREFIX" >&2; exit 1 ;;
esac

bash run_all_par_tests.sh run
