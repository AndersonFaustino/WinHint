#!/usr/bin/env bash
# =============================================================================
# sim/tests/run_tests.sh — WinHint gem5 mechanism / policy checks (host, conda
# env `winhint`; PROPOSAL §7 "Mechanism" and "Correctness").
#
#   (a) mechanism: a forced setwin(W) caps ROB/IQ/LQ/SQ occupancy
#       (stats.txt per-config maxima, occupancy histograms, full-event
#       counters), and a larger W really allows more occupancy;
#   (b) every registered window_policy runs the phases program to completion
#       (no deadlock / assert) with the expected checksum;
#   (c) the hinted binary prints identical output on RISCV_clean and
#       RISCV_winhint, under qemu-riscv64, and equals the no-hints build.
#
# Usage:   sim/tests/run_tests.sh [a|b|c|all]          (default all)
# Env:     ITERS (default 20000), WINHINT_GEM5 / CLEAN_GEM5 (gem5.opt paths),
#          POLICIES (default: every policy known to se.py),
#          NOLOCK=1 to skip the heavy lock (only if you already hold it).
# Results: $WINHINT_ROOT/results/sim_tests/<name>/ ; summary on stdout.
# All simulations run serially under flock "$WINHINT_BUILD/.heavy.lock".
# =============================================================================
set -uo pipefail
export LC_ALL=C   # printf %f needs a "." decimal separator

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${WINHINT_ROOT:-$(cd "${HERE}/../.." && pwd)}"
BUILD="${WINHINT_BUILD:-${ROOT}/build}"
GEM5_BUILD="${BUILD}/gem5/src/build"
WINHINT="${WINHINT_GEM5:-${GEM5_BUILD}/RISCV_winhint/gem5.opt}"
CLEAN="${CLEAN_GEM5:-${GEM5_BUILD}/RISCV_clean/gem5.opt}"
SE="${ROOT}/sim/se.py"
MACHINE="${MACHINE:-${ROOT}/sim/machines/riscv_ooo.json}"
BIN="${BUILD}/sim_tests"
OUT="${ROOT}/results/sim_tests"
ITERS="${ITERS:-20000}"
WHAT="${1:-all}"
POLICIES="${POLICIES:-static hint hybrid occupancy mlp bbv lut ltp}"

if [[ -z "${NOLOCK:-}" ]]; then
    exec env NOLOCK=1 flock "${BUILD}/.heavy.lock" "$0" "$@"
fi

make -s -C "${HERE}" OUT="${BIN}" >/dev/null || { echo "FAIL: building test programs"; exit 1; }
mkdir -p "${OUT}"
fail=0

run() {  # run <gem5> <name> <binary-args> [se.py args...]
    local gem5="$1" name="$2" opts="$3"; shift 3
    local d="${OUT}/${name}"
    rm -rf "${d}"; mkdir -p "${d}"
    local t0=${SECONDS}
    if ! timeout 900 "${gem5}" --outdir="${d}" "${SE}" --machine "${MACHINE}" \
            --cmd "${BIN}/${BINARY:-window_test}" --options "${opts}" \
            --output program.out "$@" > "${d}/gem5.log" 2>&1; then
        echo "FAIL ${name}: gem5 exited non-zero or timed out (see ${d}/gem5.log)"
        fail=1; return 1
    fi
    if ! grep -q "checksum=" "${d}/program.out" 2>/dev/null; then
        echo "FAIL ${name}: program did not finish"; fail=1; return 1
    fi
    echo "${name}: $((SECONDS - t0))s" >> "${OUT}/timing.txt"
}

st() {  # st <run> <stat> : value of system.cpu.window.<stat> (or full name)
    local key="$2"
    [[ "${key}" == system.* ]] || key="system.cpu.window.${key}"
    awk -v k="${key}" '$1==k {print $2; exit}' "${OUT}/$1/stats.txt"
}

gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !((a+0) > (b+0))}'; }

cap() {  # cap <struct> <config index>
    python3 -c "import json,sys; print(json.load(open('${MACHINE}'))['window']['$1'][$2])"
}

# ---------------------------------------------------------------------------
if [[ "${WHAT}" == a || "${WHAT}" == all ]]; then
    echo "== (a) mechanism: forced setwin(W), window_policy=hint =="
    printf "%-5s %-4s %-14s %-14s %-12s %-12s %-9s %-9s %-8s %-9s\n" \
        W cfg robMax/cap iqMax/cap lqMax/cap sqMax/cap robMean robFull ipc cycles
    prev_max=0
    for w in 64 128 192 256; do
        run "${WINHINT}" "a_force${w}" "force ${w} ${ITERS}" \
            --window-policy hint --window-trace || continue
        cfg=$(awk -F, 'NR>1 && $1==1 {print $2; exit}' "${OUT}/a_force${w}/region_stats.csv")
        if [[ -z "${cfg}" ]]; then
            echo "FAIL a_force${w}: no region 1 row in region_stats.csv"; fail=1; continue
        fi
        line=""
        for s in rob iq lq sq; do
            m=$(st "a_force${w}" "${s}OccMax::${cfg}"); c=$(cap "${s}" "${cfg}")
            line+=$(printf "%-14s " "${m}/${c}")
            if gt "${m}" "${c}"; then
                echo "FAIL a_force${w}: ${s} occupancy ${m} exceeds cap ${c}"; fail=1
            fi
            # The max above is over settled cycles only; the mean covers
            # every cycle in the config (drain included) and must also
            # respect the cap.
            mn=$(st "a_force${w}" "${s}OccMean::${cfg}")
            if gt "${mn}" "${c}"; then
                echo "FAIL a_force${w}: ${s} mean occupancy ${mn} exceeds cap ${c}"; fail=1
            fi
        done
        robmax=$(st "a_force${w}" "robOccMax::${cfg}")
        robmean=$(st "a_force${w}" "robOccMean::${cfg}")
        robfull=$(st "a_force${w}" "robFullCycles::${cfg}")
        ipc=$(st "a_force${w}" system.cpu.ipc)
        cyc=$(st "a_force${w}" system.cpu.numCycles)
        printf "%-5s %-4s %s%-9.1f %-9s %-8.4f %-9s\n" "${w}" "${cfg}" "${line}" \
            "${robmean}" "${robfull}" "${ipc}" "${cyc}"
        exp_cfg=$(python3 -c "
import json; r=json.load(open('${MACHINE}'))['window']['rob']
print(next((i for i,x in enumerate(r) if x>=${w}), len(r)-1))")
        if [[ "${cfg}" != "${exp_cfg}" ]]; then
            echo "FAIL a_force${w}: config ${cfg}, expected ${exp_cfg}"; fail=1
        fi
        if ! gt "${robmax}" "${prev_max}"; then
            echo "FAIL a_force${w}: ROB max ${robmax} not above the smaller window's ${prev_max}"
            fail=1
        fi
        prev_max="${robmax}"
    done
    # The histogram of the W=64 run must have no settled mass above 64 beyond
    # the transient drain; the full-event counter must show the cap binding.
    if [[ -f "${OUT}/a_force64/stats.txt" ]]; then
        rf=$(st a_force64 "robFullCycles::0")
        gt "${rf}" 0 || { echo "FAIL a_force64: ROB cap never binding (robFullCycles=0)"; fail=1; }
        dc=$(st a_force64 drainCycles); cyc=$(st a_force64 system.cpu.numCycles)
        echo "  a_force64: drainCycles=${dc} switches=$(st a_force64 switches) cycles=${cyc}"
        # Above-cap cycles only come from the drain after the one shrink.
        gt "$(awk -v a="${dc}" -v b="${cyc}" 'BEGIN{print a/b}')" 0.01 &&
            { echo "FAIL a_force64: drain cycles ${dc} > 1% of ${cyc}"; fail=1; }
    fi
fi

# ---------------------------------------------------------------------------
if [[ "${WHAT}" == b || "${WHAT}" == all ]]; then
    echo "== (b) policies on the phases program (no deadlock, same output) =="
    ref=$(qemu-riscv64 "${BIN}/window_test" phases "${ITERS}")
    for p in ${POLICIES}; do
        extra=()
        [[ "${p}" == lut ]] && extra=(--window-lut "${HERE}/test.lut")
        [[ "${p}" == static ]] && extra=(--window-initial 1)
        f0=${fail}
        if ! run "${WINHINT}" "b_${p}" "phases ${ITERS}" --window-policy "${p}" \
                --window-period 500 --window-trace "${extra[@]}"; then
            if grep -q "is unknown (built-in" "${OUT}/b_${p}/gem5.log"; then
                echo "  ${p}: SKIP (not compiled into this gem5)"; fail=${f0}
            else
                grep -m1 -E "fatal|panic|Assertion" "${OUT}/b_${p}/gem5.log" | sed 's/^/    /'
            fi
            continue
        fi
        out=$(cat "${OUT}/b_${p}/program.out")
        [[ "${out}" == "${ref}" ]] || { echo "FAIL b_${p}: output differs from qemu"; fail=1; }
        regions=$(($(wc -l < "${OUT}/b_${p}/region_stats.csv") - 1))
        cpc=""
        for i in 0 1 2 3; do cpc+="$(st "b_${p}" "cyclesInConfig::${i}") "; done
        echo "  ${p}: ok  switches=$(st "b_${p}" switches) hints=$(st "b_${p}" hints)" \
             "regions=${regions} ipc=$(st "b_${p}" system.cpu.ipc) cycles_per_config=[${cpc% }]"
        [[ ${regions} -eq 4 ]] || { echo "FAIL b_${p}: expected 4 region rows"; fail=1; }
    done
    if [[ -f "${OUT}/b_hint/region_stats.csv" ]]; then
        got=$(awk -F, 'NR>1 {printf "%s ", $2}' "${OUT}/b_hint/region_stats.csv")
        # phases: setwin 256, 64, 128, 0 -> configs 3 0 1 3
        [[ "${got}" == "3 0 1 3 " ]] || { echo "FAIL b_hint: region configs '${got}', expected '3 0 1 3'"; fail=1; }
        echo "  hint region configs: ${got}"
    fi
fi

# ---------------------------------------------------------------------------
if [[ "${WHAT}" == c || "${WHAT}" == all ]]; then
    echo "== (c) hinted binary: RISCV_clean vs RISCV_winhint vs qemu vs no-hints =="
    run "${CLEAN}" c_clean "phases ${ITERS}"
    run "${WINHINT}" c_winhint "phases ${ITERS}" --window-policy hint
    BINARY=window_test_nohints run "${CLEAN}" c_nohints "phases ${ITERS}"
    q=$(qemu-riscv64 "${BIN}/window_test" phases "${ITERS}")
    for n in c_clean c_winhint c_nohints; do
        echo "  ${n}: $(cat "${OUT}/${n}/program.out" 2>/dev/null)"
    done
    echo "  qemu: ${q}"
    if cmp -s "${OUT}/c_clean/program.out" "${OUT}/c_winhint/program.out" &&
       cmp -s "${OUT}/c_clean/program.out" "${OUT}/c_nohints/program.out" &&
       [[ "$(cat "${OUT}/c_clean/program.out")" == "${q}" ]]; then
        echo "  identical: yes"
    else
        echo "FAIL (c): outputs differ"; fail=1
    fi
    if [[ -f "${OUT}/c_clean/stats.txt" ]] && grep -q "^system.cpu.window" "${OUT}/c_clean/stats.txt"; then
        echo "FAIL (c): RISCV_clean has window stats (wrong binary?)"; fail=1
    fi
fi

[[ ${fail} -eq 0 ]] && echo "ALL OK" || echo "SOME TESTS FAILED"
exit ${fail}
