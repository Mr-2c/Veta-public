#!/bin/bash -l
# =============================================================================
# run_main_par.sbatch -- launch the distributed PRODUCTION main (main_par) on
# Helma. Same proven mpirun recipe as the parallel tests; the target is the
# namelist-driven main.
#
#   DECK : the input file (default examples/input_kmm.nml -- the 128^3 MKM box).
#          Override with:  sbatch --export=ALL,DECK=examples/input_kmm.nml ...
#   VAL  : if set (VAL=1), run --validate (distributed vs single-GPU oracle bit-compare,
#          validate_steps from the deck). Unset -> the full production run.
#   PREC : which precision binary/binaries to run -- "dp", "sp", or "dp sp" (default
#          "dp sp", i.e. BOTH back-to-back; each banner prints its own BUILD PRECISION).
#          For long production submit one precision per job: --export=ALL,PREC=sp.
#   NSYS : if set (NSYS=1), wrap each rank in `nsys profile` -> one .nsys-rep per
#          rank (nsys_<prec>_p<P>_r<rank>.nsys-rep). Use a SHORT deck (input_kmm_nsys.nml)
#          so traces stay small. Read with: nsys stats nsys_sp_p4_r0.nsys-rep
#          (the kernel summary splits NCCL vs cuFFT vs helm_band solve time).
#
# Rank placement is P-FLEXIBLE: ranks/node = SLURM_NTASKS / SLURM_NNODES, so a
# single-node strong-scaling sweep (P=1,2,4) works by varying --ntasks (see below).
#
# Build BOTH first (separate binaries, one per precision):
#   module load nvhpc/24.11 openmpi/5.0.8-nvhpc24.11-cuda_old nccl/2.28.7-gcc11.5.0-cuda ; TP=dp bash scripts/build/build_par.sh main
#                                    TP=sp bash scripts/build/build_par.sh main
#
# Usage:
#   sbatch --export=ALL,VAL=1 scripts/sbatch/run_main_par.sbatch                  # P=4, validate dp+sp
#   sbatch --export=ALL,PREC=dp scripts/sbatch/run_main_par.sbatch               # P=4, production dp only
#   sbatch --export=ALL,PREC=sp scripts/sbatch/run_main_par.sbatch               # P=4, production sp only
#   sbatch --export=ALL,VAL=1 --nodes=2 --ntasks=8 scripts/sbatch/run_main_par.sbatch  # P=8, validate
#
# Strong-scaling sweep (one node, fixed grid, vary P) -- submit three jobs:
#   sbatch --nodes=1 --ntasks=1 --gres=gpu:h100:1 --export=ALL,PREC=sp,DECK=examples/input_kmm.nml scripts/sbatch/run_main_par.sbatch
#   sbatch --nodes=1 --ntasks=2 --gres=gpu:h100:2 --export=ALL,PREC=sp,DECK=examples/input_kmm.nml scripts/sbatch/run_main_par.sbatch
#   sbatch --nodes=1 --ntasks=4 --gres=gpu:h100:4 --export=ALL,PREC=sp,DECK=examples/input_kmm.nml scripts/sbatch/run_main_par.sbatch
#   # compare the final "steps/s" line across the three .out files -> efficiency curve.
#
# nsys profile (short deck, per-rank trace):
#   sbatch --nodes=1 --ntasks=4 --gres=gpu:h100:4 --export=ALL,NSYS=1,PREC=sp,DECK=examples/input_kmm.nml scripts/sbatch/run_main_par.sbatch
# =============================================================================
#SBATCH --job-name=veta_main_par
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=main_par_%j.out

# The cluster PRELOADS nvhpc/25.7 -- a bare load of 24.11 conflicts, fails, and
# the job silently proceeds on 25.7 (whose paired OpenMPI mpif-sizeof.h also
# spams W-0031 warnings). Unload first, then pin.
module unload nvhpc 2>/dev/null
module load nvhpc/24.11
module unload openmpi 2>/dev/null
module load openmpi/5.0.5-gcc11.4.1-cuda
module load nccl/2.28.7-gcc11.5.0-cuda

# --- InfiniBand RDMA for NCCL (Helma) ------------------------------------------
# ibv_devinfo: the IB FABRIC is mlx5_2..5 (link_layer InfiniBand, PORT_ACTIVE,
# mtu 4096); mlx5_0/1 are Ethernet (do NOT use them). Without this, NCCL picked no
# IB device and fell back to TCP sockets (~1 GB/s) -> ~1 s/all-to-all. List all 4
# rails for multi-rail bandwidth. (Commas force this to live in-script, not in
# --export.) Override by exporting NCCL_IB_HCA before sbatch if the topology changes.
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3,mlx5_4,mlx5_5}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"   # harmless for IB; only used if it ever falls to RoCE
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"              # set NCCL_DEBUG=INFO via --export to see the transport
# The mlx5 verbs provider lives in this SUBdir, which is NOT in ldconfig or the
# default dlopen path. Keep it on the path so the provider is found by bare name.
export LD_LIBRARY_PATH=/usr/lib64/libibverbs:$LD_LIBRARY_PATH
# ABI FIX: a module pulls in SPACK's rdma-core-52 libibverbs.so.1 (ABI 52), but the
# system mlx5 provider is ABI 57 -> "version IBVERBS_PRIVATE_57 not found" -> NCCL
# fell back to TCP sockets. Force the SYSTEM v57 libibverbs (a superset: it has both
# the v52 and v57 symbol sets, so it satisfies the SPACK libs AND the v57 provider).
# Surgical (only libibverbs), propagated to ranks by `mpirun -x LD_PRELOAD` below.
export LD_PRELOAD="/lib64/libibverbs.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
# ------------------------------------------------------------------------------

# --- NCCL LARGE-SCALE connection-setup hardening (>=~128 ranks) ----------------
# SYMPTOM this fixes: the job reaches "[init] NCCL transpose warm-up ..." and then
# HANGS forever (or dies with NCCL "bootstrap.cc: Message truncated: received 1024
# bytes instead of 512"). Note ncclCommInitRank has ALREADY succeeded ("[par] NCCL
# communicator initialised") -- the stall is the FIRST NCCL collective: a flat
# all-to-all that lazily opens EVERY peer channel in ONE group (par_transpose_warmup
# -> exchange_bytes). At P>=128 that simultaneous connection-setup storm stalls; at
# P=360 it is near-certain. This is NCCL-only and is NOT the /dev/knem MPI warning
# (that is cosmetic -- see the `--mca smsc` on the mpirun line below). Two independent
# NCCL planes are stressed at scale; pin BOTH. Everything here is overridable.
#
# (1) DATA plane. NCCL opens 2 logical channels (= 2 independent IB QP bundles) per
#     net peer by default -> ~2x359 QP setups fired at once per rank at 360 ranks.
#     NVIDIA's DOCUMENTED workaround for "a hang [that] may occur during alltoall
#     connection establishment between multiple nodes (more likely at scale)" is
#     NCCL_NCHANNELS_PER_NET_PEER=1 (halves the storm). This is the primary fix.
#     (Costs some A2A bandwidth; A/B =1 vs =2 for throughput only AFTER it runs.)
export NCCL_NCHANNELS_PER_NET_PEER="${NCCL_NCHANNELS_PER_NET_PEER:-1}"
#
# (2) CONTROL / OOB plane. NCCL's bootstrap ring (the out-of-band TCP path that
#     exchanges the lazy connect-info) picks an interface AUTOMATICALLY when
#     NCCL_SOCKET_IFNAME is unset -- and it WAS unset. These nodes are multi-homed
#     (Ethernet mlx5_0/1, IPoIB, docker0, lo); docker0 typically carries the SAME
#     172.17.0.1 on every node, so ranks mis-peer -> the "1024 vs 512" truncation
#     and/or the warm-up hang. The default below EXCLUDES the virtual/loopback NICs
#     (leading '^' = exclude; tokens are prefixes), which is SAFE -- NCCL still
#     auto-picks a real routable NIC even if this guess is loose. Commas would be
#     split by Slurm --export, so (like NCCL_IB_HCA) this MUST live in-script.
#     If NCCL_DEBUG=INFO shows nodes still disagree on the bootstrap interface, pin
#     the exact IPoIB name instead:  sbatch --export=ALL,NCCL_SOCKET_IFNAME=ib0 ...
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker,lo,virbr}"
#
# (3) IB retransmit headroom for the 3:1-blocked spine beyond one 24-node island.
#     NOT a hang fix (there is no NCCL bootstrap timeout; this only lengthens the
#     wait before a DATA-path error). Kept modest (22, not the max 31) so a genuine
#     fabric fault still surfaces rather than being masked.
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-10}"
# ------------------------------------------------------------------------------

# --- optional NCCL introspection (NCCLDIAG=1) ---------------------------------
# Answers "is the all-to-all using PXN (the intra-node-NVLink-aggregate-then-inter-
# node, rail-optimized 'pencil-like' path) + all 4 IB rails, or arbitrary comms?".
# Set INSIDE the script because NCCL_DEBUG_SUBSYS is comma-valued (Slurm --export
# splits on commas). NCCL logs this at init only -> use a SHORT deck (..._nsys.nml).
# After the run, grep the .out (rank 0):
#   grep -E 'PXN|NET/IB|Channel|GPU Direct|Ring|Trees|Using network' main_par_<jobid>.out
if [ -n "${NCCLDIAG:-}" ]; then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING
fi

# --- process binding (measured A/B result) ------------------------------------
# BIND default is 'numa': measured 1.54x (16n) / 1.56x (32n) on the WHOLE STEP
# vs 'core' (NCCL all-to-all 23 -> 48 GB/s/rank -- 'core' starves the NCCL proxy
# threads, exactly the pipe code's finding), physics digit-identical. Revert with
# BIND=core via --export. (The 128-rank NCCL warm-up flake -- hang / 'Message
# truncated ... 1024 instead of 512' -- is addressed by the NCCL LARGE-SCALE
# block above; BIND=core is only a last-resort stopgap if that ever recurs, at the
# ~1.5x step cost.) QPS/NCHANNELS knobs kept for A/B; measured no gain on top of numa.
BIND="${BIND:-numa}"
[ -n "${NCCL_IB_QPS_PER_CONNECTION:-}" ]   && export NCCL_IB_QPS_PER_CONNECTION
[ -n "${NCCL_NCHANNELS_PER_NET_PEER:-}" ]  && export NCCL_NCHANNELS_PER_NET_PEER

export OMP_NUM_THREADS=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
# arm the RDMA guard (abort on NCCL TCP fallback; CLAUSON_REQUIRE_RDMA=0 to waive)
export CLAUSON_REQUIRE_RDMA=${CLAUSON_REQUIRE_RDMA:-1}
# GPUS_PER_NODE is set below from the SLURM allocation (ranks/node), not hardcoded,
# so the single-node strong-scaling sweep (P=1,2,4) pins GPUs correctly.

DECK="${DECK:-examples/input_kmm.nml}"
PREC="${PREC:-dp sp}"                    # which precision binaries to run (default both)
ARGS="$DECK"
if [ -n "${VAL:-}" ]; then ARGS="--validate $DECK"; fi

# Ranks-per-node: derive from the SLURM allocation so a single-node P-sweep
# (P=1,2,4 via --ntasks) and multi-node runs both place correctly. Falls back to
# all ranks on one node if SLURM_NNODES is unset.
NNODES="${SLURM_NNODES:-1}"
PPN=$(( SLURM_NTASKS / NNODES ))
if [ "$PPN" -lt 1 ]; then PPN=1; fi
export GPUS_PER_NODE="$PPN"             # gpu_bind.sh maps local rank -> GPU modulo this

echo "[run] DECK = $DECK ; ARGS = $ARGS ; PREC = $PREC ; SLURM_NTASKS=$SLURM_NTASKS ; NNODES=$NNODES ; PPN=$PPN ; NSYS=${NSYS:-0}"

if [ -n "${NSYS:-}" ] && ! command -v nsys >/dev/null 2>&1; then
  echo "FATAL: NSYS=1 but 'nsys' is not on PATH -- load nvhpc (Nsight Systems ships with it)." >&2
  exit 1
fi

MPI_PREFIX="$(dirname "$(dirname "$(command -v mpirun)")")"
echo "[run] Open MPI prefix = $MPI_PREFIX"
# record placement: which nodes/switches (Helma island forensics -- 24 nodes
# = one non-blocking island; 25+ crosses 3:1 blocking; never fit one curve through 25).
echo "[topo] nodelist: $SLURM_JOB_NODELIST"
scontrol show topology 2>/dev/null | head -50 || true

case "$MPI_PREFIX" in
  *openmpi-5.0.5*) : ;;
  *) echo "FATAL: active Open MPI is not the proven gcc-built 5.0.5 -> $MPI_PREFIX" >&2
     echo "       Fix modules (unload openmpi/5.0.8 first), rebuild with" >&2
     echo "       scripts/build/build_par.sh main, then resubmit." >&2
     exit 1 ;;
esac

run_one() {
  local prec="$1"
  local bin="./build_par/Veta_$prec"
  if [ ! -x "$bin" ]; then
    echo "   [skip] $bin not built (TP=$prec bash scripts/build/build_par.sh main)"
    return
  fi

  # nsys wrapper (per-rank trace). %q{VAR} is nsys's env-var substitution, expanded
  # by each rank's nsys at launch -> nsys_<prec>_p<P>_r<rank>.nsys-rep. Sits AFTER
  # gpu_bind.sh (which execs "$@"), so it profiles the binary on the pinned GPU.
  local nsys_pre=""
  if [ -n "${NSYS:-}" ]; then
    nsys_pre="nsys profile -t cuda,nvtx,mpi --force-overwrite=true -o nsys_${prec}_p${SLURM_NTASKS}_r%q{OMPI_COMM_WORLD_RANK}"
  fi

  echo
  echo "################################################################################"
  echo "## RUN $bin   ($ARGS)   [P=$SLURM_NTASKS, ppr:${PPN}:node, bind-to=$BIND]${nsys_pre:+   [nsys]}"
  echo "################################################################################"
  # `--mca smsc '^knem'` silences the benign "/dev/knem cannot be opened" warning
  # (Helma replaced the knem kernel module with xpmem). EXCLUDE-mode ('^knem') is
  # safe: it drops only knem and lets xpmem/cma/two-copy still load, so it can never
  # abort the launch -- unlike `--mca smsc xpmem` (INCLUDE-mode), which ABORTS if
  # xpmem happens not to be built into this Open MPI. smsc is intra-node MPI shared-
  # memory only; it is IRRELEVANT to NCCL and to the warm-up hang -- purely cosmetic.
  # The new NCCL_* vars are exported in-script (they reach ranks like NCCL_IB_HCA
  # does), and forwarded with -x too as belt-and-suspenders (-x passes by NAME, so
  # the comma value of NCCL_SOCKET_IFNAME is preserved).
  mpirun --prefix "$MPI_PREFIX" --mca smsc '^knem' \
         -x PATH -x LD_LIBRARY_PATH -x LD_PRELOAD \
         -x NCCL_NCHANNELS_PER_NET_PEER -x NCCL_SOCKET_IFNAME \
         -x CLAUSON_REQUIRE_RDMA \
         -x NCCL_IB_TIMEOUT -x NCCL_IB_RETRY_CNT \
         -np "$SLURM_NTASKS" --map-by ppr:${PPN}:node --bind-to "$BIND" \
         ./scripts/gpu_bind.sh $nsys_pre "$bin" $ARGS
}

for p in $PREC; do run_one "$p"; done
