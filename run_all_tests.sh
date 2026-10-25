#!/usr/bin/env bash
#==============================================================================
# run_all_tests.sh -- compile every module (dependency order) and then build,
# link, and run every unit test for the GPU channel-flow DNS solver (CLAUSON).
#
# LAYOUT (post directory restructure):
#   src/core/      shared solver modules (.F90)
#   src/serial/    the single-GPU program (main.F90)
#   src/parallel/  the multi-GPU driver + par_* modules (built by build_par.sh)
#   tests/serial/  the single-GPU unit tests that RUN (this runner)
#   tests/parallel/ the multi-GPU (MPI+NCCL) tests (run_all_par_tests.sh)
#   build/         ALL serial compiler output (.o, .mod) -- keeps the tree clean
#
# This runner builds the serial R2C tree in src/core/ + tests/serial/.
#
# Requirements: nvfortran (NVHPC, e.g. 24.11), cuFFT, and a GPU-equipped node.
#   CPU reference group : plain  nvfortran -c
#   device group        : nvfortran -cuda [-cudalib=cufft]
# Each test is compiled -cuda -cudalib=cufft and linked against the full module
# set in build/. main.F90 is a PROGRAM and is NOT in the test-link objects (its
# main symbol would collide with every test at link).
#
# Usage:
#   bash run_all_tests.sh            # build modules + run all tests
#   bash run_all_tests.sh -k <key>   # only tests whose name contains <key>
#
# Output: per-test PASS/FAIL on stdout; full compiler output in build/build.log;
# each test's stdout in build/run_<test>.out.
#==============================================================================
set -u
FC=${FC:-nvfortran}
SRC=src/core
TST=tests/serial
OBJ=build
LOG="$OBJ/build.log"
KEY=""
[ "${1:-}" = "-k" ] && KEY="${2:-}"

# ---- precision / codegen flags (mirror the Makefile) -------------------------
# TP/WP/NOFMA come from the environment so the suite can be built in any
# precision configuration; default is all-real64, stock FMA codegen (NOFMA=1
# adds -Mnofma for a bit-identity-locked build).
# dp is ALWAYS real64 (no flag). Examples:
#   TP=sp bash run_all_tests.sh        # transforms FP32
#   TP=sp WP=sp bash run_all_tests.sh  # both FP32 (won't link until the cast seams land)
#   NOFMA=0 bash run_all_tests.sh      # stock (FMA-contracted) codegen
TP=${TP:-real64}
WP=${WP:-real64}
NOFMA=${NOFMA:-0}    # NOFMA=1 disables FMA contraction (-Mnofma)
GPUCC=${GPUCC:-cc90} # device target (H100/GH200); see Makefile
PRECFLAGS=""
if [ "$TP" = "sp" ];   then PRECFLAGS="$PRECFLAGS -DCLAUSON_TP_SP"; fi
if [ "$WP" = "sp" ];   then PRECFLAGS="$PRECFLAGS -DCLAUSON_WP_SP"; fi
if [ "$NOFMA" = "1" ]; then PRECFLAGS="$PRECFLAGS -Mnofma"; fi
# Same optimizer pair as the parallel build (-Mnoinline carries the
# 0155 guarantee; the Makefile comment has the full story).
PRECFLAGS="$PRECFLAGS -O2 -Mnoinline"

mkdir -p "$OBJ"
: > "$LOG"
say(){ printf '%s\n' "$*"; }
die(){ printf '\n!! %s\n' "$*" >&2; exit 1; }

command -v "$FC" >/dev/null 2>&1 || die "$FC not found in PATH (load the NVHPC module first)"

#--- fresh build: clear stale objects/modules/executables ---------------------
say "== cleaning stale build artifacts (build/) =="
rm -f "$OBJ"/*.o "$OBJ"/*.mod
for t in $(ls "$TST"/test_*.F90 2>/dev/null | xargs -n1 basename | sed 's/\.F90$//'); do rm -f "$OBJ/$t"; done

#--- compile a single source with the given flags, abort on failure -----------
# objects + modules land in build/ (-module sets BOTH the .mod output dir and
# the search path for already-built .mod dependencies).
compile(){
  local src=$1; shift
  local xtra="$*"
  # -gpu only rides -cuda compiles (bare -gpu would just draw a warning)
  case " $xtra " in *" -cuda "*) xtra="$xtra -gpu=$GPUCC" ;; esac
  printf '  CC  %-22s [%s]\n' "$(basename "$src")" "${xtra:-plain}"
  $FC $PRECFLAGS $xtra -module "$OBJ" -c "$src" -o "$OBJ/$(basename "${src%.F90}").o" >>"$LOG" 2>&1 \
     || die "compile FAILED: $src  (see $LOG)"
}

#--- modules, in dependency order --------------------------------------------
say ""; say "== precision: TP=$TP WP=$WP NOFMA=$NOFMA  (extra flags:${PRECFLAGS:- none}) =="
say "== compiling modules (-> $OBJ/) =="
# CPU reference group (plain)
compile $SRC/types.F90
compile $SRC/decomp_mod.F90
compile $SRC/diag_mod.F90
compile $SRC/wavenum_mod.F90
compile $SRC/cheb_shen.F90
compile $SRC/cheb_bih.F90
compile $SRC/helm_band_mod.F90
compile $SRC/yops_mod.F90
compile $SRC/ygrid_mod.F90
compile $SRC/fdops_mod.F90
compile $SRC/fd_vsolve_mod.F90
compile $SRC/mean_mod.F90
compile $SRC/fd_mean_mod.F90
compile $SRC/vsolve_if_mod.F90
# device + cuFFT transform stack
compile $SRC/grid_mod.F90          -cuda
compile $SRC/transforms_mod.F90    -cuda -cudalib=cufft
compile $SRC/cheb_fft_mod.F90      -cuda -cudalib=cufft
compile $SRC/dealias_mod.F90       -cuda
compile $SRC/fft3d_mod.F90         -cuda -cudalib=cufft
compile $SRC/transpose_mod.F90     -cuda
compile $SRC/spectral3d_mod.F90    -cuda -cudalib=cufft
# device wall-normal ops / physics
compile $SRC/cheb_ops_mod.F90      -cuda
compile $SRC/yderiv_gpu_mod.F90    -cuda
compile $SRC/stats_mod.F90         -cuda
compile $SRC/init_mod.F90          -cuda
compile $SRC/cfl_mod.F90           -cuda
compile $SRC/io_mod.F90            -cuda
compile $SRC/velocity_mod.F90      -cuda
compile $SRC/forcing_mod.F90       -cuda
compile $SRC/nonlinear_mod.F90     -cuda -cudalib=cufft
compile $SRC/helm_band_gpu_mod.F90 -cuda
compile $SRC/fd_helm_gpu_mod.F90 -cuda
compile $SRC/fd_vsolve_gpu_mod.F90 -cuda
compile $SRC/spec_helpers_mod.F90  -cuda
compile $SRC/mean_gpu_mod.F90      -cuda
compile $SRC/fd_mean_gpu_mod.F90   -cuda
compile $SRC/vsolve_if_gpu_mod.F90 -cuda
# diagnostic pressure solve -- per-rank Neumann-Poisson influence
# matrix; no MPI deps (cudafor + helm_band_gpu/spec_helpers/helm_band/cheb_shen),
# so it builds + links into the serial GPU test harness like vsolve_if_gpu_mod.
compile src/parallel/par_pressure_mod.F90 -cuda
# time steppers
compile $SRC/stepper_lin_mod.F90   -cuda
compile $SRC/solver_ops_mod.F90    -cuda
compile $SRC/stepper_mod.F90       -cuda -cudalib=cufft

#--- guard: every non-test module source must have been compiled above --------
# Catches the "new module added but not listed" failure mode. main.F90 is a
# PROGRAM and is deliberately not compiled here (its main symbol would collide
# with every test's at link time).
for src in "$SRC"/*.F90; do
  b=$(basename "$src")
  case "$b" in test_*|main.F90) continue ;; esac
  [ -f "$OBJ/${b%.F90}.o" ] || die "module source $b was not compiled -- add it to the module list above"
done

# all module objects (everything in build/ that is not a test object)
OBJS=$(ls "$OBJ"/*.o | grep -v '/test_' | tr '\n' ' ')

#--- build + run each test ----------------------------------------------------
say ""; say "== building & running tests (tests/ -> $OBJ/) =="
RESULTS=""
npass=0; nfail=0
# Superseded / non-gating tests (defensive; not present in tests/serial):
#   test_vsolve_if -- early random-RHS comparison of two DIFFERENT discretizations
#   (vif vs cheb_bih), replaced by the manufactured-solution test_vsolve_if_mms.
SKIP="test_vsolve_if"

for t in $(ls "$TST"/test_*.F90 | xargs -n1 basename | sed 's/\.F90$//'); do
  if [ -n "$KEY" ] && [[ "$t" != *"$KEY"* ]]; then continue; fi
  case " $SKIP " in *" $t "*)
    printf '  -- %-30s SKIP (superseded)\n' "$t"; continue ;;
  esac
  printf '  -- %-30s ' "$t"
  rm -f "$OBJ/$t"
  if ! $FC $PRECFLAGS -cuda -gpu=$GPUCC -cudalib=cufft -module "$OBJ" -c "$TST/$t.F90" -o "$OBJ/$t.o" >>"$LOG" 2>&1; then
    printf 'BUILD-FAIL\n'; RESULTS="$RESULTS
  $t  BUILD-FAIL"; nfail=$((nfail+1)); continue
  fi
  if ! $FC $PRECFLAGS -cuda -gpu=$GPUCC -cudalib=cufft "$OBJ/$t.o" $OBJS -o "$OBJ/$t" >>"$LOG" 2>&1; then
    printf 'LINK-FAIL\n'; RESULTS="$RESULTS
  $t  LINK-FAIL"; nfail=$((nfail+1)); continue
  fi
  out=$("$OBJ/$t" 2>&1); rc=$?
  printf '%s\n' "$out" > "$OBJ/run_$t.out"
  # FAIL only on: an explicit failure MARKER, a crash, or a stray non-finite VALUE.
  # The marker must be unambiguous -- tests print '[FAIL]' for a real failure and
  # 'TESTS FAILED' / 'ERROR STOP' on abort. We deliberately do NOT grep bare words
  # like 'wrong' or 'mismatch': those appear in PASSING descriptive lines (e.g.
  # test_io_r2c prints '[ok] (B) grid mismatch refused'), which the old broad grep
  # mis-flagged as a failure (a false-positive harness bug seen in practice).
  if printf '%s' "$out" | grep -qE '\[FAIL\]|FAIL:|TESTS FAILED|ERROR STOP|[Ee]rror stop'; then
    printf 'FAIL\n'; RESULTS="$RESULTS
  $t  FAIL"; nfail=$((nfail+1)); continue
  fi
  if [ $rc -ne 0 ]; then
    printf 'FAIL (rc=%d)\n' "$rc"; RESULTS="$RESULTS
  $t  FAIL (rc=$rc)"; nfail=$((nfail+1)); continue
  fi
  # Non-finite scan AFTER the explicit checks, and matched as a standalone numeric
  # TOKEN (space/line-delimited) -- so descriptive prose like '(NaN/Inf guard)' or
  # '(finite)' in a passing test does not false-fail it (same class of harness bug
  # as the 'mismatch' one above).
  if printf '%s' "$out" | grep -qiE '(^|[[:space:]])[-+]?(nan|inf|infinity)([[:space:]]|$)'; then
    printf 'FAIL (non-finite)\n'; RESULTS="$RESULTS
  $t  FAIL (non-finite)"; nfail=$((nfail+1)); continue
  fi
  # POSITIVE-marker requirement (parity with the parallel
  # harness): exit-0 alone must not count as PASS -- a test that silently
  # skips all its checks (early return, dead config loop) prints nothing.
  # Recognized markers cover every assertion-bearing test's convention;
  # the two assertion-FREE inspection tools (matrix/sequence dumps with
  # nothing to assert) are whitelisted by name.
  case " $t " in
    *" test_bih_structure "*|*" test_bih_tail "*) : ;;  # inspection-only: no assertions exist
    *)
      if ! printf '%s' "$out" | grep -qE 'ALL TESTS PASSED|\[PASS\]|PASSED|\[ok\]|\[ OK \]'; then
        printf 'FAIL (no-pass-marker)\n'; RESULTS="$RESULTS
  $t  FAIL (no-pass-marker)"; nfail=$((nfail+1)); continue
      fi ;;
  esac
  printf 'PASS\n'; npass=$((npass+1))
done

#--- summary ------------------------------------------------------------------
say ""
say "=============================================================="
say "  PASS: $npass    FAIL: $nfail"
if [ -n "$RESULTS" ]; then say "  failures:$RESULTS"; fi
say "  (compiler output: $LOG ; per-test stdout: $OBJ/run_<test>.out)"
say "=============================================================="
# A zero-test run (e.g. -k with no match) is an ERROR, not a
# green summary -- 'PASS: 0 FAIL: 0' exit 0 read as success.
if [ $((npass + nfail)) -eq 0 ]; then
  say "  ERROR: no tests matched${KEY:+ -k '$KEY'} -- nothing was run."
  exit 1
fi
[ "$nfail" -eq 0 ] || exit 1
