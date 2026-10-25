#!/bin/bash -l
# =============================================================================
# veta.sbatch -- MINIMAL production launcher for Veta Sublimeson on Helma.
#
# Designed to run from a campaign working directory that contains the binary:
#
#   cp <repo>/build_par/Veta_sp .          # (and/or Veta_dp)
#   sbatch veta.sbatch                     # sp, the single *.nml in this dir
#   sbatch veta.sbatch my_deck.nml         # sp, explicit deck
#   sbatch veta.sbatch my_deck.nml dp      # dp instead
#   sbatch -N 24 -n 96 veta.sbatch         # CLI overrides the #SBATCH sizing
#
# No --export needed. Precision defaults to sp. GPU binding is inline (one GPU
# per rank via CUDA_VISIBLE_DEVICES). For validate/nsys/sweeps use the full
# scripts/sbatch/run_main_par.sbatch instead -- this one only launches.
# =============================================================================
#SBATCH --job-name=veta_re550fd
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=veta_%j.out

module load nvhpc/24.11
module unload openmpi 2>/dev/null
module load openmpi/5.0.5-gcc11.4.1-cuda
module load nccl/2.28.7-gcc11.5.0-cuda

# --- proven Helma NCCL/IB environment (see run_main_par.sbatch for the full
#     forensics; comma-valued vars MUST live in-script, not in --export) -------
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3,mlx5_4,mlx5_5}"      # IB rails (mlx5_0/1 are Ethernet)
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker,lo,virbr}"   # bootstrap: exclude virtual NICs
export NCCL_NCHANNELS_PER_NET_PEER="${NCCL_NCHANNELS_PER_NET_PEER:-1}" # >=128-rank connect-storm fix
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-10}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export LD_LIBRARY_PATH=/usr/lib64/libibverbs:$LD_LIBRARY_PATH          # mlx5 provider dir
export LD_PRELOAD="/lib64/libibverbs.so.1${LD_PRELOAD:+:$LD_PRELOAD}"  # force system ABI-57 libibverbs
export OMP_NUM_THREADS=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK   # Slurm >=22.05: srun no longer inherits -c

# --- arguments: any *.nml is the deck, dp|sp picks the binary (default sp) ----
DECK=""; PREC="sp"
for a in "$@"; do
  case "$a" in
    dp|sp)  PREC="$a" ;;
    *.nml)  DECK="$a" ;;
    *) echo "FATAL: unrecognised argument '$a' (expected a .nml deck and/or dp|sp)" >&2; exit 1 ;;
  esac
done
if [ -z "$DECK" ]; then
  shopt -s nullglob; set -- *.nml; shopt -u nullglob
  if [ "$#" -eq 1 ]; then DECK="$1"
  else
    echo "FATAL: $# .nml files here -- name one: sbatch veta.sbatch <deck.nml> [dp]" >&2; exit 1
  fi
fi
BIN="./Veta_$PREC"
[ -x "$BIN" ] || { echo "FATAL: $BIN not found here -- cp <repo>/build_par/Veta_$PREC ." >&2; exit 1; }
[ -f "$DECK" ] || { echo "FATAL: deck '$DECK' not found" >&2; exit 1; }

# --- placement: ranks/node from the allocation; one GPU per rank --------------
NNODES="${SLURM_NNODES:-1}"
PPN=$(( SLURM_NTASKS / NNODES )); [ "$PPN" -lt 1 ] && PPN=1
export GPUS_PER_NODE="$PPN"

MPI_PREFIX="$(dirname "$(dirname "$(command -v mpirun)")")"
case "$MPI_PREFIX" in
  *openmpi-5.0.5*) : ;;
  *) echo "FATAL: active Open MPI is not the proven gcc-built 5.0.5 -> $MPI_PREFIX (fix modules)" >&2
     exit 1 ;;
esac

echo "[run]  $BIN $DECK   P=$SLURM_NTASKS  ppr:${PPN}:node  bind-to=numa"
echo "[topo] nodelist: $SLURM_JOB_NODELIST"

# inline per-rank GPU binding (the gpu_bind.sh logic, self-contained)
mpirun --prefix "$MPI_PREFIX" --mca smsc '^knem' \
       -x PATH -x LD_LIBRARY_PATH -x LD_PRELOAD -x GPUS_PER_NODE \
       -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX -x NCCL_NCHANNELS_PER_NET_PEER \
       -x NCCL_SOCKET_IFNAME -x NCCL_IB_TIMEOUT -x NCCL_IB_RETRY_CNT \
       -np "$SLURM_NTASKS" --map-by ppr:${PPN}:node --bind-to numa \
       bash -c 'export CUDA_VISIBLE_DEVICES=$(( ${OMPI_COMM_WORLD_LOCAL_RANK:-0} % GPUS_PER_NODE ))
                echo "[bind] global=${OMPI_COMM_WORLD_RANK:-?} local=${OMPI_COMM_WORLD_LOCAL_RANK:-0} host=$(hostname -s) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
                exec "$@"' _bind "$BIN" "$DECK"
