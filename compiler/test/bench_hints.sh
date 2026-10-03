#!/usr/bin/env bash
# Run the WinHint plugin over benchmarks/*.c and report, per kernel and arch:
# static setwin count, regions, W* range, compile time of plain vs. winhint
# (median of REPS full clang -O2 -c runs; wall_% is noisy on a shared machine)
# and the pass's own time (pass_ms, from the stats JSON) as a share of the
# plain compile (pass_%), which is the robust overhead number.
#   compiler/test/bench_hints.sh                  # both arches, -O2
#   ARCHS=x86 REPS=5 OPT=-O3 compiler/test/bench_hints.sh
# Output: table on stdout; per-kernel JSON in $WINHINT_BUILD/compiler-tests/bench/.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/env.sh"
OUT=$WINHINT_BUILD/compiler-tests/bench
REPS=${REPS:-5}
OPT=${OPT:--O2 -gline-tables-only}
ARCHS=${ARCHS:-riscv x86}
mkdir -p "$OUT"

median() { sort -n | awk '{a[NR]=$1} END {print (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'; }
timeit() { # cmd... -> ms
  local s e; s=$(date +%s%N); "$@" >/dev/null 2>&1 || return 1; e=$(date +%s%N)
  echo $(( (e - s) / 1000000 ))
}

printf '%-30s %-5s %6s %7s %6s %-14s %8s %8s %7s %7s %6s\n' kernel arch setwin regions loops "W* min/med/max" plain_ms wh_ms wall_% pass_ms pass_%
for a in $ARCHS; do
  if [ $a = riscv ]; then FL="$RV_CFLAGS"; else FL="$X86_CFLAGS"; fi
  for src in "$WINHINT_ROOT"/benchmarks/*.c; do
    k=$(basename "$src" .c)
    d=$OUT/$a; mkdir -p "$d"
    inc="-I$WINHINT_ROOT/benchmarks"
    if ! clang -Wno-unused-command-line-argument $FL $OPT $inc $WHF -mllvm -winhint-out-dir="$d" -mllvm -winhint-kernel="$k" \
         -c "$src" -o "$d/$k.wh.o" 2> "$d/$k.err"; then
      printf '%-30s %-5s COMPILE FAILED (see %s)\n' "$k" "$a" "$d/$k.err"; continue
    fi
    tp=(); tw=()
    for r in $(seq "$REPS"); do
      tp+=("$(timeit clang -Wno-unused-command-line-argument $FL $OPT $inc -c "$src" -o "$d/$k.plain.o")")
      tw+=("$(timeit clang -Wno-unused-command-line-argument $FL $OPT $inc $WHF -c "$src" -o "$d/$k.wh2.o")")
    done
    mp=$(printf '%s\n' "${tp[@]}" | median); mw=$(printf '%s\n' "${tw[@]}" | median)
    python3 - "$d/$k.winhint.json" "$k" "$a" "$mp" "$mw" <<'EOF'
import json, sys, statistics as st
d = json.load(open(sys.argv[1])); k, a, mp, mw = sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
ws = [r["w_star"] for r in d["regions"]] or [0]
ovh = 100.0 * (mw - mp) / mp if mp > 0 else 0.0
print("%-30s %-5s %6d %7d %6d %-14s %8.0f %8.0f %7.1f %7.2f %6.2f" % (
    k, a, d["hints_setwin"], d["num_regions"], len(d["loops"]),
    "%d/%d/%d" % (min(ws), st.median(ws), max(ws)), mp, mw, ovh, d["compile_time_ms"],
    100.0 * d["compile_time_ms"] / mp if mp > 0 else 0.0))
EOF
  done
done
