#!/usr/bin/env bash
# B8 Clairvoyance pipeline for one kernel (called by benchmarks/Makefile).
#
#   cv_compile.sh <kernel.c> <output-binary>
#
# 1. mark every for-loop for Clairvoyance (#pragma clang loop vectorize_width(1337),
#    the artifact's marker) in a temporary copy of the source;
# 2. clang 3.8 -O3 -emit-llvm (x86_64 front end: clang 3.8 has no RISC-V target);
# 3. Clairvoyance passes with opt 3.8 (mark-loops, annotate-cfg-indir,
#    single-loop-unroll, second-loop-extract, dae-swoop/<type>, -O3), exactly
#    the artifact's Makefile.defaults recipe;
# 4. write LLVM 3.8 *bitcode*; LLVM 21 reads it (bitcode backward compat);
# 5. retarget (riscv64: triple, datalayout, drop x86 cpu/features attributes);
# 6. optional WinHint pass on the transformed IR (PASS_FLAGS, winhint_clairvoyance);
# 7. lower with modern clang at the requested -O level, codegen only (the
#    Clairvoyance schedule is not re-optimized), link.
#
# Env (from the Makefile): CV_ARCH=riscv64|x86_64, CLANG (modern clang; NOT $CC,
# which the winhint env sets to GCC), CFLAGS, LDFLAGS, LDLIBS,
# PASS_FLAGS; CV_KEEP=1 (keep temps); LLVM38_PREFIX (default
# $MAMBA_ROOT_PREFIX/envs/winhint-llvm38), CV_LIB (default
# $WINHINT_BUILD/clairvoyance/lib, written by build.sh).
#
# Knobs (the artifact's own, experiments/swoop/sources/common/SWOOP/Makefile.{defaults,targets};
# tunable ranges in knobs.json):
#   CV_TYPE        consv (default) | specsafe | spec | multispecsafe | multispec
#                  consv -dae-swoop, specsafe -aggressive-swoop, spec -speculative-swoop;
#                  multi* = the same plus -multi-access (several access phases)
#   CV_UNROLL      unroll count of the marked loop (-single-loop-unroll / -unroll), default 2;
#                  artifact grid 1 2 4
#   CV_INDIR       max indirections hoisted into the access phase (-indir-thresh), default 1;
#                  artifact grid 0 1 2 3
#   CV_BRANCH_PROB branch-merge threshold (-branch-prob-threshold, >= 0.5), default 0.9
#                  (fixed in the artifact recipe)
#   CV_STRICT=1    fail if the passes crash instead of excluding the crashing function
set -euo pipefail
SRC=$(realpath "$1"); OUT=$(realpath -m "$2")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WINHINT_ROOT=${WINHINT_ROOT:-$(cd "$HERE/../../.." && pwd)}
WINHINT_BUILD=${WINHINT_BUILD:-$WINHINT_ROOT/build}
LLVM38_PREFIX=${LLVM38_PREFIX:-${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}/envs/winhint-llvm38}
B38=$LLVM38_PREFIX/bin
LIB=${CV_LIB:-$WINHINT_BUILD/clairvoyance/lib}
CV_ARCH=${CV_ARCH:-riscv64}
CV_TYPE=${CV_TYPE:-consv}
CV_UNROLL=${CV_UNROLL:-2}
CV_INDIR=${CV_INDIR:-1}
CV_BRANCH_PROB=${CV_BRANCH_PROB:-0.9}
die() { echo "[ERROR] cv_compile.sh: $*" >&2; exit 2; }
case "$CV_TYPE" in consv|specsafe|spec|multispecsafe|multispec) ;;
  *) die "CV_TYPE=$CV_TYPE (expected consv|specsafe|spec|multispecsafe|multispec)" ;; esac
[[ "$CV_UNROLL" =~ ^[0-9]+$ ]] && [ "$CV_UNROLL" -ge 1 ] && [ "$CV_UNROLL" -le 16 ] \
  || die "CV_UNROLL=$CV_UNROLL (expected an integer in 1..16; artifact grid 1 2 4)"
[[ "$CV_INDIR" =~ ^[0-9]+$ ]] && [ "$CV_INDIR" -le 8 ] \
  || die "CV_INDIR=$CV_INDIR (expected an integer in 0..8; artifact grid 0 1 2 3)"
awk -v p="$CV_BRANCH_PROB" 'BEGIN { exit !(p ~ /^[0-9]*\.?[0-9]+$/ && p >= 0.5 && p <= 1.0) }' \
  || die "CV_BRANCH_PROB=$CV_BRANCH_PROB (expected a number in [0.5, 1.0])"
CLANG=${CLANG:-clang}
K=$(basename "$SRC" .c)
[ -x "$B38/clang" ] && [ -f "$LIB/libOptimisticSwoop.so" ] || {
  echo "[ERROR] Clairvoyance toolchain not built: compiler/baselines/clairvoyance/build.sh (needs env winhint-llvm38)" >&2; exit 1; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/cv_${K}_XXXX")
[ "${CV_KEEP:-0}" = 1 ] || trap 'rm -rf "$TMP"' EXIT

# 1. mark loops
sed -E 's/^([[:space:]]*)for[[:space:]]*\(/\1_Pragma("clang loop vectorize_width(1337)") for (/' "$SRC" > "$TMP/$K.c"

# 2. LLVM 3.8 front end
"$B38/clang" -target x86_64-linux-gnu -O3 -std=c11 -S -emit-llvm $(echo "${CFLAGS:-}" | grep -o -- '-D[^ ]*' || true) \
  "$TMP/$K.c" -o "$TMP/$K.ll"

# 3. Clairvoyance passes (artifact recipe, experiments/swoop/sources/common/SWOOP/Makefile.defaults)
case "$CV_TYPE" in
  consv)    SWOOP_OPTS="-dae-swoop -hoist-delinquent=false" ;;
  specsafe) SWOOP_OPTS="-aggressive-swoop -hoist-delinquent=false" ;;
  spec)     SWOOP_OPTS="-speculative-swoop -hoist-delinquent=false" ;;
  multispecsafe) SWOOP_OPTS="-aggressive-swoop -hoist-delinquent=false -multi-access" ;;
  multispec)     SWOOP_OPTS="-speculative-swoop -hoist-delinquent=false -multi-access" ;;
esac
cd "$TMP"
# The artifact's passes are research code. Known crashes are fixed by
# patches/*.patch (0002: SwoopDAE null latch predecessor on contrast_mobilenet's
# conv3x3_stem); none of the 15 kernels needs an exclusion with the default knobs.
# As a safety net for other knob settings, on a crash the function being transformed
# (printed by the pass as "<fn>___kernel__<loop>:") is excluded -- its loop
# markers are stripped from the IR -- and the passes are re-run, so the kernel
# keeps Clairvoyance code in every other function. Excluded functions are listed
# in <out>.cv.json (they run as plain code).
strip_markers() {  # $1 = in.ll, $2 = out.ll, rest = excluded functions
  local in=$1 out=$2; shift 2
  awk -v ex=" $* " '
    /^define / { f=$0; sub(/^[^@]*@/, "", f); sub(/\(.*/, "", f); gsub(/"/, "", f);
                 skip = index(ex, " " f " ") > 0 }
    skip { gsub(/, !llvm\.loop ![0-9]+/, "") }
    /^}/ { skip = 0 }
    { print }' "$in" > "$out"
}
run_passes() {
  "$B38/opt" -S -load "$LIB/libMarkLoopsToSwoopify.so" -mark-loops -require-delinquent=false \
    -bench-name "$K" "$K.in.ll" -o "$K.marked.ll" &&
  "$B38/opt" -S -load "$LIB/libCFGIndirectionCount.so" -annotate-cfg-indir -loop-name __kernel__ \
    "$K.marked.ll" -o "$K.annotated.ll" &&
  "$B38/opt" -S -loop-unswitch -instcombine -loops -lcssa -loop-simplify -loop-rotate -indvars \
    -scalar-evolution -licm -lcssa -load "$LIB/libUtilLoops.so" -single-loop-unroll \
    -loop-name __kernel__ -unroll "$CV_UNROLL" "$K.annotated.ll" -o "$K.unroll.ll" &&
  "$B38/opt" -S -load "$LIB/libLoopExtract.so" -aggregate-extracted-args -second-loop-extract \
    -bench-name "$K" -mergereturn -load "$LIB/libBranchAnnotate.so" -branchannotate \
    "$K.unroll.ll" -o "$K.extract.ll" &&
  "$B38/opt" -S -tbaa -basicaa -globals-aa -scev-aa -load "$LIB/libOptimisticSwoop.so" $SWOOP_OPTS \
    -merge-branches -branch-prob-threshold "$CV_BRANCH_PROB" -indir-thresh "$CV_INDIR" -unroll "$CV_UNROLL" -mem2reg \
    "$K.extract.ll" -o "$K.swoop.ll"
}
EXCLUDED=()
for attempt in $(seq 1 ${CV_MAX_RETRIES:-12}); do
  strip_markers "$K.ll" "$K.in.ll" "${EXCLUDED[@]}"
  if run_passes > "$K.passes.log" 2>&1; then break; fi
  bad=$(grep -oE '^[A-Za-z_][A-Za-z0-9_.]*___kernel__' "$K.passes.log" | tail -1 | sed 's/___kernel__$//')
  if [ "${CV_STRICT:-0}" = 1 ] || [ -z "$bad" ] || printf '%s\n' "${EXCLUDED[@]}" | grep -qx "$bad"; then
    tail -30 "$K.passes.log" >&2
    echo "[ERROR] Clairvoyance passes failed on $K (no function to exclude)" >&2; exit 1
  fi
  echo "[CV] $K: passes crashed in $bad; excluding it and retrying" >&2
  EXCLUDED+=("$bad")
done
[ -f "$K.swoop.ll" ] || { echo "[ERROR] Clairvoyance: too many retries on $K" >&2; exit 1; }
NCV=$(grep -c '^define .*__kernel__' "$K.swoop.ll" || true)
printf '{"kernel": "%s", "cv_type": "%s", "unroll": %s, "indir": %s, "branch_prob": %s, "extracted_functions": %s, "excluded_functions": [%s]}\n' \
  "$K" "$CV_TYPE" "$CV_UNROLL" "$CV_INDIR" "$CV_BRANCH_PROB" "${NCV:-0}" \
  "$(for f in "${EXCLUDED[@]}"; do printf '"%s",' "$f"; done | sed 's/,$//')" > "$OUT.cv.json"
# 4. optimize with 3.8 (as the artifact does) and emit bitcode
"$B38/opt" -O3 "$K.swoop.ll" -o "$K.cv.bc"

# 5. modern LLVM reads 3.8 bitcode; retarget
llvm-dis "$K.cv.bc" -o "$K.cv.ll"
opt -passes=verify -disable-output "$K.cv.ll"
if [ "$CV_ARCH" = riscv64 ]; then
  sed -i -E \
    -e 's/^target triple = .*/target triple = "riscv64-unknown-linux-gnu"/' \
    -e 's/^target datalayout = .*/target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"/' \
    -e 's/"target-cpu"="[^"]*"//g' -e 's/"target-features"="[^"]*"//g' -e 's/"tune-cpu"="[^"]*"//g' \
    -e 's/~\{dirflag\},~\{fpsr\},~\{flags\},?//g' \
    "$K.cv.ll"
  # Retargeting x86_64 front-end IR is sound only for target-neutral ABI lowering.
  if grep -nE 'x86_fp80|va_arg| byval|@llvm\.x86\.' "$K.cv.ll" >&2; then
    echo "[ERROR] $K: x86-specific IR (above) cannot be retargeted to riscv64" >&2; exit 1
  fi
fi

# 6. optional WinHint pass (clang-style flags -> opt flags)
IN="$K.cv.ll"
if [ -n "${PASS_FLAGS:-}" ]; then
  OPTARGS=(); PLUGIN=""
  set -- ${PASS_FLAGS}
  while [ $# -gt 0 ]; do
    case "$1" in
      -fpass-plugin=*) PLUGIN=${1#-fpass-plugin=} ;;
      -fplugin=*) ;;
      -mllvm) shift; OPTARGS+=("$1") ;;
      -D*) ;;
      *) OPTARGS+=("$1") ;;
    esac
    shift
  done
  opt -load-pass-plugin="$PLUGIN" -passes=winhint "${OPTARGS[@]}" "$IN" -S -o "$K.hinted.ll"
  IN="$K.hinted.ll"
fi

# 7. lower (codegen only) and link
"$CLANG" ${CFLAGS:-} -Wno-override-module -Xclang -disable-llvm-optzns -c "$IN" -o "$K.o"
"$CLANG" ${CFLAGS:-} "$K.o" -o "$OUT" ${LDFLAGS:-} ${LDLIBS:-}
echo "[CV] $OUT ($CV_TYPE, unroll $CV_UNROLL, indir $CV_INDIR, branch-prob $CV_BRANCH_PROB)"
