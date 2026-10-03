#!/usr/bin/env bash
# B9 (LTP) end-to-end check: window_policy=ltp vs static on RISCV_winhint.
#   - no deadlock (the run finishes), identical program output (squash /
#     LSQ-order correctness) against qemu-riscv64, and parking happens
#     (system.cpu.ltp.* stats).
# Usage: run_ltp_tests.sh [gem5.opt] [iters]
# Every gem5 run takes the heavy lock (docs/interfaces.md §1).
set -euo pipefail
: "${WINHINT_ROOT:?activate the winhint env}" "${WINHINT_BUILD:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEM5="${1:-$WINHINT_BUILD/gem5/src/build/RISCV_winhint/gem5.opt}"
ITERS="${2:-20000}"
OUT="${LTP_TEST_OUT:-$WINHINT_BUILD/ltp_tests}"
MACHINE="$WINHINT_ROOT/sim/machines/riscv_ooo.json"
mkdir -p "$OUT"

BIN="$OUT/ltp_test"
"${WINHINT_RISCV_CC:-riscv64-conda-linux-gnu-gcc}" -static -O2 -Wall \
    -o "$BIN" "$HERE/ltp_test.c"

fail=0
for mode in mem compute mixed; do
    ref="$(qemu-riscv64 "$BIN" "$mode" "$ITERS")"
    for cfg in "static 3" "static 1" "ltp 1"; do
        set -- $cfg; pol=$1; init=$2
        d="$OUT/$mode.$pol.$init"
        rm -rf "$d"
        flock "$WINHINT_BUILD/.heavy.lock" timeout 3600 "$GEM5" --outdir="$d" \
            "$WINHINT_ROOT/sim/se.py" --machine "$MACHINE" \
            --cmd "$BIN" --options "$mode $ITERS" \
            --window-policy "$pol" --window-initial "$init" \
            --output prog.out > "$d.log" 2>&1 || { echo "FAIL $mode $pol/$init: gem5 exited ($d.log)"; fail=1; continue; }
        got="$(cat "$d/prog.out")"
        st="$d/stats.txt"
        ipc=$(awk '/^system.cpu.ipc /{print $2; exit}' "$st")
        parked=$(awk '/^system.cpu.ltp.parked /{print $2; exit}' "$st")
        rel=$(awk '/^system.cpu.ltp.released /{print $2; exit}' "$st")
        sq=$(awk '/^system.cpu.ltp.squashed /{print $2; exit}' "$st")
        occ=$(awk '/^system.cpu.ltp.occMean /{print $2; exit}' "$st")
        if [[ "$got" != "$ref" ]]; then
            echo "FAIL $mode $pol/$init: output differs ('$got' vs '$ref')"; fail=1
        else
            printf "ok   %-8s %-6s init=%s ipc=%-8s parked=%-9s released=%-9s squashed=%-7s occMean=%s\n" \
                "$mode" "$pol" "$init" "$ipc" "${parked:--}" "${rel:--}" "${sq:--}" "${occ:--}"
        fi
        if [[ $pol == ltp && ( -z "$parked" || "$parked" == 0 ) && $mode != compute ]]; then
            echo "FAIL $mode ltp: nothing was parked"; fail=1
        fi
    done
done
exit $fail
