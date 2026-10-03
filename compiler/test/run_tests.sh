#!/usr/bin/env bash
# WinHint compiler tests (host, `winhint` micromamba env; see build/ENV.md).
#   compiler/build.sh && compiler/test/run_tests.sh
# Scratch outputs: $WINHINT_BUILD/compiler-tests (override with TEST_OUT).
#
#  1. model     print<winhint-demand> on small C loops:
#                 stream  (independent misses) -> largest window, MLP-driven
#                 chase   (dependent misses)   -> smallest window
#                 compute (L1-resident FP)     -> W* = W_cp (critical path)
#                 libm    (expf in the loop)   -> call summarized in body / CP
#  2. placement phases.c -> one setwin per phase, hoisted out of inner loops;
#               switch cost / P-E model reduce the hint count; stats JSON schema
#  3. regions   deterministic region ids (riscv == x86, sorted by function name)
#  4. encoding  llvm-objdump of riscv64 / x86-64 objects: setwin, region,
#               from-json, call mode (docs/interfaces.md §2)
#  5. identity  hinted vs plain program output: riscv64 under qemu-riscv64,
#               x86-64 native (setwin, regions, from-json, call, none)
#  6. lit       compiler/test/lit: model-demand, placement, encodings,
#               model portability, B6 jones-iq, qemu identity, regressions,
#               target-model (machine JSON loader), placement-options,
#               demand-paths (hand-written IR), jones-paths
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/env.sh"
IN=$HERE/inputs
W=${TEST_OUT:-$WINHINT_BUILD/compiler-tests}
rm -rf "$W"; mkdir -p "$W"
PASS=0; FAIL=0
ok()   { echo "PASS: $*"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $*"; FAIL=$((FAIL+1)); }
check(){ local d=$1; shift; if "$@"; then ok "$d"; else bad "$d"; fi; }
CC="clang -Wno-unused-command-line-argument"

# field <line> <key>  -> value of key=value in a WINHINT line
field() { echo "$1" | tr ' ' '\n' | sed -n "s/^$2=//p" | head -1; }
# The innermost (last printed) loop line of function $2 in printer output $1.
winline() { grep "WINHINT fn=$2 " "$1" | tail -1; }
num()   { printf '%.0f' "$1"; }

echo "== 0. versions"
echo "  $(clang --version | head -1)"
echo "  plugin: $WH   machine: $MACHINE"

echo "== 1. window-demand model"
for k in stream chase compute phases libm; do
  $CC $RV_CFLAGS -O2 -gline-tables-only -S -emit-llvm "$IN/$k.c" -o "$W/$k.ll" || bad "clang $k"
  opt -load-pass-plugin="$WH" -passes='print<winhint-demand>' -disable-output \
      -winhint-target="$MACHINE" "$W/$k.ll" 2> "$W/$k.demand.txt" || bad "opt print $k"
done
WMAX=$(python3 -c "import json;print(max(json.load(open('$MACHINE'))['window']['rob']))")
WMIN=$(python3 -c "import json;print(min(json.load(open('$MACHINE'))['window']['rob']))")
L=$(winline "$W/stream.demand.txt" triad); echo "  stream : $L"
check "stream: W* = W_max ($WMAX), MLP-driven (W_mlp > W_cp)" \
  test "$(field "$L" 'W\*')" = "$WMAX" -a "$(field "$L" W_mlp)" -gt "$(field "$L" W_cp)"
LC=$(winline "$W/chase.demand.txt" walk); echo "  chase  : $LC"
check "chase: smallest config (0), W* <= $WMIN" \
  test "$(field "$LC" config)" = 0 -a "$(field "$LC" 'W\*')" -le "$WMIN"
L=$(winline "$W/compute.demand.txt" poly); echo "  compute: $L"
check "compute: no long-latency loads, W* = W_cp" \
  test "$(field "$L" W_mlp)" = 0 -a "$(field "$L" 'W\*')" = "$(field "$L" W_cp)"
check "compute: W*(chase) < W*(compute) < W_max" \
  test "$(field "$L" 'W\*')" -gt "$(field "$LC" 'W\*')" -a "$(field "$L" 'W\*')" -lt "$WMAX"
L=$(winline "$W/libm.demand.txt" softmax_num); echo "  libm   : $L"
B=$(grep -B20 "WINHINT fn=softmax_num .*depth=2" "$W/libm.demand.txt" | grep -o 'body=[0-9.]*' | tail -1 | cut -d= -f2)
check "libm: expf summarized (body $B >= 25 insts, CP >= 30)" \
  test "$(num "${B:-0}")" -ge 25 -a "$(num "$(field "$L" CP)")" -ge 30
check "target JSON parsed (L_mem of a DRAM miss = 20 + 200)" grep -q "L_mem=220.0" "$W/stream.demand.txt"

echo "== 2. placement"
$CC $RV_CFLAGS -O2 -gline-tables-only $WHF -mllvm -winhint-out-dir="$W/place" -c "$IN/phases.c" -o "$W/phases.o" \
  || bad "compile phases"
J=$W/place/phases.winhint.json
python3 - "$J" "$WMAX" <<'EOF' && ok "phases: setwin per phase in step(), hoisted to depth<=2, no redundant hint" || bad "phases placement"
import json, sys
d = json.load(open(sys.argv[1])); wmax = int(sys.argv[2])
h = [x for x in d["hints"] if x["function"] == "step" and x["kind"] == "setwin"]
print("  hints in step() (line, depth, W):", [(x["line"], x["depth"], x["value"]) for x in h])
assert len(h) >= 2, "expected a hint per phase"
assert all(x["depth"] <= 2 for x in h), "hint inside an inner loop"
vals = [x["value"] for x in sorted(h, key=lambda x: x["line"])]
assert all(a != b for a, b in zip(vals, vals[1:])), "redundant consecutive hint"
assert wmax in vals, "streaming phase must get the largest window"
assert min(vals) < wmax, "compute phase must get a smaller window"
EOF
python3 - "$J" <<'EOF' && ok "stats JSON: schema winhint-stats/1 keys present" || bad "stats JSON schema"
import json, sys
d = json.load(open(sys.argv[1]))
for k in ["schema", "kernel", "llvm_version", "mode", "emit", "triple", "target", "switch_cost_cycles",
          "hysteresis", "hints_setwin", "hints_region", "compile_time_ms", "analysis_time_ms",
          "num_regions", "window", "regions", "hints", "loops"]:
    assert k in d, k
assert d["schema"] == "winhint-stats/1"
assert d["hints_setwin"] == sum(1 for h in d["hints"] if h["kind"] == "setwin")
for r in d["regions"]:
    for k in ["id", "function", "line", "w_star", "nest_w_star", "config", "entry_setwin", "setwin"]:
        assert k in r, k
print("  compile_time_ms=%.2f analysis_time_ms=%.2f regions=%d setwin=%d" % (
    d["compile_time_ms"], d["analysis_time_ms"], d["num_regions"], d["hints_setwin"]))
EOF
nset() { python3 -c "import json;print(json.load(open('$1'))['hints_setwin'])"; }
N0=$(nset "$J")
$CC $RV_CFLAGS -O2 $WHF -mllvm -winhint-switch-cost=1e12 -mllvm -winhint-out-dir="$W/place_hi" -c "$IN/phases.c" -o "$W/hi.o"
N1=$(nset "$W/place_hi/phases.winhint.json")
check "switch cost 1e12 cycles: fewer hints ($N1 < $N0)" test "$N1" -lt "$N0"
$CC $RV_CFLAGS -O2 $WHF -mllvm -winhint-switch-model=pe -mllvm -winhint-migration-us=50 -mllvm -winhint-out-dir="$W/place_pe" -c "$IN/phases.c" -o "$W/pe.o"
N2=$(nset "$W/place_pe/phases.winhint.json")
check "P/E migration model (50us): no more hints than gem5 model ($N2 <= $N0)" test "$N2" -le "$N0"
$CC $RV_CFLAGS -O2 $WHF -mllvm -winhint-hysteresis=0 -mllvm -winhint-out-dir="$W/place_h0" -c "$IN/phases.c" -o "$W/h0.o"
N3=$(nset "$W/place_h0/phases.winhint.json")
check "hysteresis 0: at least as many hints as default ($N3 >= $N0)" test "$N3" -ge "$N0"

echo "== 3. region ids"
for a in riscv x86; do
  FL=$RV_CFLAGS; [ $a = x86 ] && FL=$X86_CFLAGS
  for k in phases stream; do
    $CC $FL -O2 -gline-tables-only $WHF -mllvm -winhint-mode=regions -mllvm -winhint-out-dir="$W/reg_$a" \
      -c "$IN/$k.c" -o "$W/reg_$a.$k.o" || bad "regions $a $k"
  done
done
for k in phases stream; do
python3 - "$W/reg_riscv/$k.regions.json" "$W/reg_x86/$k.regions.json" <<'EOF' && ok "$k: region ids identical on riscv/x86, ordered by function name" || bad "$k region ids"
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ka = [(i, r["function"], r["line"]) for i, r in sorted(a["regions"].items(), key=lambda x: int(x[0]))]
kb = [(i, r["function"], r["line"]) for i, r in sorted(b["regions"].items(), key=lambda x: int(x[0]))]
print("  ", ka)
assert ka == kb
fn = [f for _, f, _ in ka]
assert fn == sorted(fn), "ids not in function-name order"
assert [int(i) for i, _, _ in ka] == list(range(len(ka)))
EOF
done

echo "== 4. encodings"
# setwin W -> payload ceil(W/8); riscv word 0x00006013 | ((payload<<5|tag) << 20)
rvword() { printf '%08x' $(( 0x6013 | (( ($2<<5) | $1 ) << 20) )); }
x86disp() { printf '%02x %02x %02x %02x' $(( $2 & 0xff )) $(( ($1<<4) | ($2>>8) )) 0x48 0x57; }
llvm-objdump -d "$W/phases.o" > "$W/phases.rv.dis"
for v in $(python3 -c "import json;print(' '.join(str(x['value']) for x in json.load(open('$J'))['hints']))"); do
  p=$(( (v + 7) / 8 )); wd=$(rvword 21 $p)
  check "riscv setwin($v) word $wd = ori zero,zero,$(( (p<<5)|21 ))" \
    grep -qE "$wd[[:space:]]+ori[[:space:]]+zero, zero, ($(( (p<<5)|21 ))|0x$(printf '%x' $(( (p<<5)|21 ))))\$" "$W/phases.rv.dis"
done
NRV=$(grep -cE "ori[[:space:]]+zero, zero, " "$W/phases.rv.dis")
check "riscv: #hint instructions in object ($NRV) = hints_setwin ($N0)" test "$NRV" = "$N0"
grep -m3 -E "ori[[:space:]]+zero, zero" "$W/phases.rv.dis" | sed 's/^/    /'
$CC $X86_CFLAGS -O2 $WHF -mllvm -winhint-out-dir="$W/place86" -c "$IN/phases.c" -o "$W/phases.x86.o" || bad "compile x86"
llvm-objdump -d "$W/phases.x86.o" > "$W/phases.x86.dis"
J86=$W/place86/phases.winhint.json
for v in $(python3 -c "import json;print(' '.join(str(x['value']) for x in json.load(open('$J86'))['hints']))"); do
  p=$(( (v + 7) / 8 )); bytes="0f 1f 80 $(x86disp 1 $p)"
  check "x86 setwin($v) bytes '$bytes' = nopl 0x$(printf '%x' $(( 0x57481000 | p )))(%rax)" \
    grep -qE "$bytes[[:space:]]+nopl[[:space:]]+0x$(printf '%x' $(( 0x57481000 | p )))\(%rax\)" "$W/phases.x86.dis"
done
grep -m2 "nopl" "$W/phases.x86.dis" | sed 's/^/    /'
# region mode
llvm-objdump -d "$W/reg_riscv.phases.o" > "$W/reg.dis"
llvm-objdump -d "$W/reg_x86.phases.o" > "$W/reg86.dis"
NR=$(python3 -c "import json;print(json.load(open('$W/reg_riscv/phases.regions.json'))['num_regions'])")
for id in $(seq 0 $((NR-1))); do
  check "riscv region($id) word $(rvword 23 $id)" grep -q "$(rvword 23 $id)" "$W/reg.dis"
  check "x86 region($id) bytes 0f 1f 80 $(x86disp 2 $id)" grep -q "0f 1f 80 $(x86disp 2 $id)" "$W/reg86.dis"
done
rvtags() { grep -oE 'ori[[:space:]]+zero, zero, (0x)?[0-9a-f]+' "$1" | awk '{print $NF}' | python3 -c 'import sys; print(" ".join(sorted({str(int(x, 0) & 31) for x in sys.stdin.read().split()})) + " ")'; }
check "regions mode: only region tags (23) present [$(rvtags "$W/reg.dis")]" test "$(rvtags "$W/reg.dis")" = "23 "
# from-json mode (oracle_hinted / pgo maps: flat {id: config} and {regions: {id: {W|config}}})
echo '{"regions": {"0": {"config": 0}, "1": 2}}' > "$W/map.json"
echo '{"0": 1, "1": 3}' > "$W/map_flat.json"
$CC $RV_CFLAGS -O2 $WHF -mllvm -winhint-mode=from-json="$W/map.json" -mllvm -winhint-out-dir="$W/fj" -c "$IN/phases.c" -o "$W/fj.o"
llvm-objdump -d "$W/fj.o" > "$W/fj.dis"
check "from-json: setwin(64) for region 0" grep -q "$(rvword 21 8)" "$W/fj.dis"
check "from-json: setwin(192) for region 1" grep -q "$(rvword 21 24)" "$W/fj.dis"
check "from-json: region markers emitted too" grep -q "$(rvword 23 0)" "$W/fj.dis"
$CC $X86_CFLAGS -O2 $WHF -mllvm -winhint-mode=from-json="$W/map_flat.json" -c "$IN/phases.c" -o "$W/fj86.o"
llvm-objdump -d "$W/fj86.o" > "$W/fj86.dis"
check "from-json (flat map, x86): setwin(128) region 0, setwin(256) region 1" \
  bash -c "grep -q '0f 1f 80 $(x86disp 1 16)' $W/fj86.dis && grep -q '0f 1f 80 $(x86disp 1 32)' $W/fj86.dis"
# call mode
$CC $RV_CFLAGS -O2 $WHF -mllvm -winhint-emit=call -c "$IN/phases.c" -o "$W/call.o"
check "call mode references __winhint_setwin" bash -c "llvm-nm $W/call.o | grep -q 'U __winhint_setwin'"
$CC $X86_CFLAGS -O2 $WHF -mllvm -winhint-emit=call -mllvm -winhint-mode=regions -c "$IN/phases.c" -o "$W/callr.o"
check "call mode (regions) references __winhint_region" bash -c "llvm-nm $W/callr.o | grep -q 'U __winhint_region'"
$CC $RV_CFLAGS -O2 $WHF -mllvm -winhint-emit=none -c "$IN/phases.c" -o "$W/none.o"
check "emit=none: no hint instructions" bash -c "! llvm-objdump -d $W/none.o | grep -qE 'ori[[:space:]]+zero, zero, '"

echo "== 5. bit-identical outputs"
build() { # arch variant src out
  local a=$1 v=$2 s=$3 o=$4 f="" extra=""
  case $v in
    plain)    f="" ;;
    winhint)  f="$WHF" ;;
    regions)  f="$WHF -mllvm -winhint-mode=regions" ;;
    fromjson) f="$WHF -mllvm -winhint-mode=from-json=$W/map.json" ;;
    call)     f="$WHF -mllvm -winhint-emit=call"; extra="$IN/winhint_stub.c" ;;
    none)     f="$WHF -mllvm -winhint-emit=none" ;;
  esac
  if [ $a = riscv ]; then $CC $RV_CFLAGS -O2 $f "$s" $extra -o "$o" -static -lm
  else $CC $X86_CFLAGS -O2 $f "$s" $extra -o "$o" -lm; fi
}
for a in riscv x86; do
  run=""; [ $a = riscv ] && run=qemu-riscv64
  for k in stream chase compute phases libm; do
    build $a plain "$IN/$k.c" "$W/$k.$a.plain" || { bad "build $a plain $k"; continue; }
    ref=$($run "$W/$k.$a.plain")
    for v in winhint regions fromjson call none; do
      build $a $v "$IN/$k.c" "$W/$k.$a.$v" || { bad "build $a $v $k"; continue; }
      out=$($run "$W/$k.$a.$v")
      check "$a $k $v output identical ($out)" test "$ref" = "$out"
    done
    if [ $a = riscv ]; then n=$(llvm-objdump -d "$W/$k.$a.winhint" | grep -cE 'ori[[:space:]]+zero, zero, ')
    else n=$(llvm-objdump -d "$W/$k.$a.winhint" | grep -c 'nopl[[:space:]]*0x5748'); fi
    echo "    ($a $k: $n hint instructions in the winhint binary)"
  done
done

echo "== 6. lit suite (compiler/test/lit)"
if command -v lit >/dev/null 2>&1; then
  check "lit compiler/test/lit" lit -sv "$HERE/lit"
else
  bad "lit not found in the env"
fi

echo "== $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
