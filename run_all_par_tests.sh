#!/usr/bin/env bash
#==============================================================================
# run_all_par_tests.sh -- the MULTI-GPU analogue of run_all_tests.sh. Builds the
# whole parallel suite (via the unified scripts/build/build_par.sh) and runs every
# tests/parallel/test_*_mp at a range of rank counts and both precisions, PLUS the
# production binary's --validate oracle diff, reporting one PASS/FAIL per case.
#
# WHY TWO PHASES. nvfortran can only COMPILE against the nvhpc-built Open MPI 5.0.8,
# but the working multi-node launcher is the gcc-built 5.0.5 (ABI-stable within
# 5.0.x). So BUILD under 5.0.8, RUN under 5.0.5 -- different module sets, two phases.
#
# Modes:
#   build   scripts/build/build_par.sh all, for TP=dp and TP=sp. Login node:
#           module load nvhpc/24.11 openmpi/5.0.8-nvhpc24.11-cuda_old nccl/2.28.7-gcc11.5.0-cuda
#   run     launch every built binary at its rank set. Inside an allocation, RUN
#           modules (5.0.5):  sbatch scripts/sbatch/run_all_par_tests.sbatch
#   all     build then run (the sbatch wrapper uses this, swapping modules between).
#
# Usage:
#   bash run_all_par_tests.sh build                # compile everything (dp+sp)
#   bash run_all_par_tests.sh run                  # run every built binary
#   bash run_all_par_tests.sh run -k step          # only names matching 'step'
#   TP=dp bash run_all_par_tests.sh build          # restrict to one precision
#   RANKS="1 4" bash run_all_par_tests.sh run      # override the rank set
#   SLOW=1 bash run_all_par_tests.sh run           # include any 'slow'-flagged tests (none by default)
#   DECK=examples/input_kmm.nml ...                # deck for the --validate case
#
# Binaries: build_par/<test>_<tp> (tests) and build_par/Veta_<tp> (production).
# PASS iff rc=0, a positive marker ('ALL TESTS PASSED' / '[PASS]' / '[ok]'), and no
# failure marker ('[FAIL]','FAIL:','TESTS FAILED','ERROR STOP','NaN','Infinity').
# Tests flagged 'diag' print a diagnostic table with NO marker by design: those
# PASS on clean completion (rc=0, no failure marker, no stray non-finite value).
# Per-case stdout: build_par/run_<name>_<tp>_p<P>.out.
#==============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OBJ=build_par
mkdir -p "$OBJ"

MODE="${1:-all}"; shift || true
KEY=""
[ "${1:-}" = "-k" ] && { KEY="${2:-}"; shift 2 || true; }
case "$MODE" in build|run|all) : ;; *) echo "usage: $0 {build|run|all} [-k <key>]" >&2; exit 2 ;; esac

TP_FILTER="${TP:-}"
NP_MAX="${SLURM_NTASKS:-4}"
SLOW="${SLOW:-0}"
DECK="${DECK:-examples/input_kmm.nml}"

#--- the test table:  name | precisions | rank_set | flags ----------------------
# name = the tests/parallel/<name>.F90 stem (binary = build_par/<name>_<tp>), or
# 'validate' = the production Veta_<tp> run with --validate (the oracle diff).
TESTS=(
  "test_par_nccl|dp|1 4 8|-"
  "test_par_reduce_mp|dp|1 4 8|-"
  "test_par_transpose|dp sp|1 4 8|-"
  "test_transpose_hard_mp|dp sp|4 8|-"
  "test_batched_transform_mp|dp sp|1 4 8|-"
  "test_a2a_overlap_mp|dp sp|1 4 8|-"
  "test_spectral3d_mp|dp sp|1 4 8|-"
  "test_par_phys_r2c_mp|dp sp|1 2 4|-"
  "test_nonlinear_mp|dp sp|1 4 8|-"
  "test_stepper_lin_mp|dp sp|1 4 8|-"
  "test_step_ev_mp|dp sp|1 4 8|-"
  "test_step_full_mp|dp sp|1 4 8|-"
  "test_par_io_mp|dp sp|1 2 4|-"
  "test_fd_pres_mp|dp sp|1 4|-"
  "test_par_snap_mp|dp sp|1 2 4|-"
  "test_ic_slab_mp|dp sp|1 4 8|-"
  "test_driver_diag_mp|dp sp|1 4|-"
  "test_driver_state_mp|dp sp|1 4|-"
  "test_driver_stats_mp|dp sp|1 4|-"
  "test_driver_nldiss_mp|dp sp|4 8|diag"
  "test_driver_laminar_mp|dp sp|1 4|-"
  "test_driver_turb_mp|dp sp|1 4|-"
  "test_driver_orr_mp|dp sp|1 4|-"
  "validate|dp sp|1 4|-"
)

say(){ printf '%s\n' "$*"; }
hr(){  say "------------------------------------------------------------------"; }
precs_for(){ local sup="$1" out=""; for p in $sup; do { [ -z "$TP_FILTER" ] || [ "$TP_FILTER" = "$p" ]; } && out="$out $p"; done; printf '%s' "$out"; }

#=============================================================================
do_build(){
  command -v nvfortran >/dev/null 2>&1 || { say "ERROR: nvfortran not in PATH (load BUILD modules: nvhpc/24.11 openmpi/5.0.8-nvhpc24.11-cuda_old nccl)"; exit 1; }
  local rc=0
  for tp in $(precs_for "dp sp"); do
    say "== BUILD all (TP=$tp) =="
    TP="$tp" bash scripts/build/build_par.sh all || rc=1
  done
  return $rc
}

#=============================================================================
launch(){  # launch <bin> <np> <outfile> <flags> [extra args...]
  local bin="$1" np="$2" outf="$3" tflags="$4"; shift 4
  local pfx; pfx="$(dirname "$(dirname "$(command -v mpirun)")")"
  local xpre=""; [ -n "${LD_PRELOAD:-}" ] && xpre="-x LD_PRELOAD"
  mpirun --prefix "$pfx" -x PATH -x LD_LIBRARY_PATH $xpre \
         -np "$np" --map-by ppr:4:node --bind-to core \
         ./scripts/gpu_bind.sh "./$bin" "$@" >"$outf" 2>&1
  local rc=$?
  # 1. an explicit failure marker wins.
  if grep -qE '\[FAIL\]|FAIL:|TESTS FAILED|ERROR STOP|[Ee]rror stop' "$outf"; then echo "FAIL"; return; fi
  # 2. a nonzero exit code.
  [ $rc -ne 0 ] && { echo "FAIL(rc=$rc)"; return; }
  # 3. a positive marker is AUTHORITATIVE -- the test's own asserts (including its own
  #    NaN/Inf watchdog) passed. Checked BEFORE the non-finite scan so descriptive prose
  #    like a "[PASS] watchdog ... (NaN/Inf guard)" line cannot false-fail a passing test.
  if grep -qE 'ALL TESTS PASSED|\[PASS\]|\[ok\]' "$outf"; then echo "PASS"; return; fi
  # 4. fallback (marker-less output): a stray non-finite VALUE -- match a standalone token
  #    (space/line-delimited), NOT the word inside prose ('NaN/Inf guard', '(finite)').
  if grep -qiE '(^|[[:space:]])[-+]?(nan|inf|infinity)([[:space:]]|$)' "$outf"; then echo "FAIL(non-finite)"; return; fi
  # 5. 'diag' tests emit a table, not a verdict: clean completion IS the pass.
  case " $tflags " in *" diag "*) echo "PASS(diag)"; return ;; esac
  echo "FAIL(no-pass-marker)"
}

do_run(){
  command -v mpirun >/dev/null 2>&1 || { say "ERROR: mpirun not found (load RUN modules + get an allocation)"; exit 1; }
  local pfx; pfx="$(dirname "$(dirname "$(command -v mpirun)")")"
  case "$pfx" in *openmpi-5.0.5*) : ;; *) say "WARNING: active Open MPI is not the proven gcc-built 5.0.5 ($pfx)" ;; esac
  say "== RUN (NP_MAX=$NP_MAX, modules: $pfx) =="
  local np_pass=0 np_fail=0 np_skip=0 fails=""
  for row in "${TESTS[@]}"; do
    IFS='|' read -r name precs ranks flags <<<"$row"
    [ -n "$KEY" ] && [[ "$name" != *"$KEY"* ]] && continue
    if [ "$flags" = "slow" ] && [ "$SLOW" != "1" ]; then
      printf '  -- %-26s SKIP (slow; SLOW=1)\n' "$name"; continue
    fi
    local rset="${RANKS:-$ranks}"
    for tp in $(precs_for "$precs"); do
      # binary + extra args: 'validate' -> Veta_<tp> --validate <deck>; else <name>_<tp>
      local bin args label
      if [ "$name" = "validate" ]; then bin="$OBJ/Veta_${tp}"; args="--validate $DECK"; label="validate"
      else bin="$OBJ/${name}_${tp}"; args=""; label="$name"; fi
      if [ ! -x "$bin" ]; then
        printf '  -- %-26s [TP=%s] SKIP (not built: %s)\n' "$label" "$tp" "$bin"; np_skip=$((np_skip+1)); continue
      fi
      for np in $rset; do
        [ "$np" -gt "$NP_MAX" ] && { np_skip=$((np_skip+1)); continue; }
        printf '  -- %-26s [TP=%s P=%s] ... ' "$label" "$tp" "$np"
        local outf="$OBJ/run_${label}_${tp}_p${np}.out"
        local res; res="$(launch "$bin" "$np" "$outf" "$flags" $args)"
        say "$res"
        case "$res" in PASS*) np_pass=$((np_pass+1)) ;;
          *) np_fail=$((np_fail+1)); fails="$fails\n    $label [$tp P=$np] $res (see $outf)" ;;
        esac
      done
    done
  done
  hr; say "  RUN: $np_pass passed, $np_fail failed, $np_skip skipped"
  [ "$np_fail" -eq 0 ] || { printf '  failures:%b\n' "$fails"; return 1; }
  return 0
}

rc=0
case "$MODE" in
  build) do_build || rc=$? ;;
  run)   do_run   || rc=$? ;;
  all)   do_build && do_run || rc=$? ;;
esac
hr
[ "$rc" -eq 0 ] && say "  ALL PARALLEL TESTS PASSED ($MODE)" || say "  PARALLEL TESTS: FAILURES ($MODE) -- see above"
exit "$rc"
