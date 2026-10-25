#!/usr/bin/env bash
# =============================================================================
# gpu_bind.sh -- per-rank GPU pinning for Helma multi-GPU runs.
# (Adapted from the site-proven wrapper_Helma.sh.)
#
# mpirun execs ONE copy of this per rank. It pins the rank to a single GPU via
# CUDA_VISIBLE_DEVICES, then hands off to the real program with `exec "$@"`.
#
#   OMPI_COMM_WORLD_LOCAL_RANK : Open MPI's index for this rank AMONG the ranks
#                                on the SAME node (0..GPUS_PER_NODE-1).
#   => local ranks 0..3 map to GPUs 0..3 on each node.
#
# This is an alternative to in-code cudaSetDevice; par_mod also self-binds, so
# the two are compatible (with CUDA_VISIBLE_DEVICES set, the app sees its one
# GPU as device 0). It has nothing to do with MPI bootstrap / daemon launch --
# it is purely GPU assignment, and its echo line makes the mapping visible.
# =============================================================================
set -u
LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export CUDA_VISIBLE_DEVICES=$(( LOCAL_RANK % GPUS_PER_NODE ))

echo "[bind] global=${OMPI_COMM_WORLD_RANK:-?} local=${LOCAL_RANK} host=$(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
exec "$@"
