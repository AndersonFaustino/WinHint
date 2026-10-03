# Running WinHint

How to run every experiment and produce every figure: the gem5 study, the real-hardware
campaign, and the full reproduction in one list. This is the runbook: it says *what to run, in
which order*. *Why* each experiment exists — the claims, the baselines, equal-effort tuning,
fidelity checks — is in [Evaluation methodology](../concepts/methodology.md); terms are in
the [Glossary](../reference/glossary.md). Setup is in the [README](../../README.md#quickstart);
the binding formats are in [Interfaces](../interfaces.md).

- [Before you start](#before-you-start)
- [Part 1: gem5 evaluation](#part-1-gem5-evaluation)
- [Part 2: Real hardware](#part-2-real-hardware)
- [Part 3: Reproduce everything](#part-3-reproduce-everything)

## Before you start

Run every command from the repository root, inside the `winhint` env:

```bash
eval "$(~/.local/bin/micromamba shell hook -s bash)"
micromamba activate winhint
```

- **Heavy lock.** Every gem5 run, McPAT call and gem5/McPAT build takes
  `flock "$WINHINT_BUILD/.heavy.lock"` by itself, so heavy jobs queue one after the other.
  Do not wrap the scripts in another `flock` on that file: they would wait for themselves.
- **Parallelism.** Benchmark builds run at `-j1`. The experiment scripts default to `--jobs 1`
  and refuse more than 2 (6 GB of RAM).
- **Resuming.** Every script skips finished runs; `--force` reruns them.
- **Outputs.** Builds go to `build/`, results to `results/` (both gitignored).

---

## Part 1: gem5 evaluation

### Baselines

| ID | Method | `run_experiments.py` variant(s) |
|----|--------|---------------------------------|
| B0 | static window, one run per configuration | `static_c0` … `static_c3` |
| B1 | per-region oracle (upper bound) | `oracle_hinted` |
| B2 | occupancy-driven resizing (Ponomarev, MICRO'01) | `occupancy` |
| B3 | MLP-aware resizing (Kora, MICRO'13) | `mlp` |
| B4 | BBV phase prediction (Sherwood, ISCA'03) | `bbv` |
| B5 | learned counter LUT (Dubach style) | `lut`, `lut_xfer` |
| B6 | compiler IQ resizing (Jones, HPCA'05) | `jones`, `jones_full` |
| B7 | profile-guided positional adaptation (Huang, ISCA'03) | `pgo` |
| B8 | Clairvoyance (Tran, CGO'17), alone and with WinHint | `clairvoyance`, `winhint_clairvoyance` |
| B9 | Long-Term Parking (Sembrant, MICRO'15) | `ltp`, `ltp_c0` … `ltp_c2` |
| — | WinHint; WinHint + hardware override | `winhint`, `winhint_hw` |
| — | hint overhead (hints decoded, ignored) | `winhint_nop` |

Variant notes:

- `static_c0` … `static_c3` run window configurations c0 (smallest) … c3 (largest).
- `lut` uses the LUT trained on the same machine. `lut_xfer` reuses the reference machine's
  LUT on the other machines (portability, "no retraining").
- `jones` resizes the IQ only, as published; `jones_full` extends it to ROB/LSQ.
- `ltp` runs at the largest window; `ltp_c<i>` matches the resources of `static_c<i>`.
- `winhint_nop` is compared with `static_c3` to measure the dynamic cost of the hints.

### How each variant runs

| Variant | Binary (`VARIANT=`) | Per machine | `window_policy` | Inputs from step |
|---------|---------------------|:-----------:|-----------------|------------------|
| `static_c<i>` | `plain` | | `static`, initial = i | 2 |
| `occupancy`, `mlp`, `bbv` | `plain` | | same as the name | 2, 6 |
| `lut`, `lut_xfer` | `plain` | | `lut` | 3, 4, 6 |
| `ltp`, `ltp_c<i>` | `plain` | | `ltp` | 2, 6 |
| `oracle_hinted` | `oracle_hinted` | ✓ | `hint` | 3, 7 |
| `jones` | `jones` | ✓ | `hint`, `structs=iq` | 7 |
| `jones_full` | `jones_full` | ✓ | `hint` | 7 |
| `pgo` | `pgo` | ✓ | `hint` | 3, 7 |
| `clairvoyance` | `clairvoyance` | | `static`, largest | 6, 7 |
| `winhint_clairvoyance` | `winhint_clairvoyance` | ✓ | `hint` | 6, 7 |
| `winhint` | `winhint` | ✓ | `hint` | 6, 7 |
| `winhint_hw` | `winhint` | ✓ | `hybrid` | 6, 7 |
| `winhint_nop` | `winhint` | ✓ | `static`, largest | 7 |

**Machines:** `riscv_ooo` (default, B5 reference), `riscv_ooo_small`, `riscv_ooo_big`
(`sim/machines/*.json`).

**Per-machine binaries** are compiled for one machine JSON (`MACHINE=`):

- the default machine's build goes to `build/benchmarks/riscv/<variant>/`;
- another machine's build goes to `build/benchmarks/riscv/<variant>/<machine>/`.

`run_experiments.py` looks for `<variant>/<machine>/<kernel>` first. It falls back to
`<variant>/<kernel>` only if that binary's sidecar names the same machine; otherwise the run is
reported as missing.

### Step 1: Prerequisites

Once. The gem5 and McPAT builds take the heavy lock and several hours at `-j2`.

```bash
# B8 source (git submodule)
git submodule update --init third_party/clairvoyance

# gem5: RISCV_clean, then RISCV_winhint (used by every script)
tooling/winhint.sh gem5:clone
tooling/winhint.sh gem5:build all

# McPAT → build/mcpat/mcpat (without it, energy uses an analytic proxy)
tooling/winhint.sh mcpat:clone
tooling/winhint.sh mcpat:build

# LLVM plugins → build/compiler/{WinHint,JonesIQ}.so
tooling/winhint.sh compiler:build

# B8 passes (env winhint-llvm38) → build/clairvoyance/lib
tooling/winhint.sh clairvoyance:build

# optional: mechanism + every policy in gem5 (heavy lock)
tooling/winhint.sh test sim
```

### Step 2: Machine-independent binaries

`plain` runs B0, B2–B5 and B9. `oracle` (region markers only) is the binary of the oracle
sweep; its region profile also places the samples of every `large` run
(`results/runlen/oracle/`).

```bash
make -C benchmarks ARCH=riscv VARIANT=plain     # → build/benchmarks/riscv/plain/
make -C benchmarks ARCH=riscv VARIANT=oracle    # → build/benchmarks/riscv/oracle/
```

### Step 3: Oracle sweeps

Each sweep runs the `oracle` build under every static configuration of one machine
(4 configurations × 15 kernels). B1 needs the `large` sweep of every machine; B7 and B5 need
the `small` sweep of every machine.

```bash
for m in riscv_ooo riscv_ooo_small riscv_ooo_big; do
  for i in small large; do
    python3 sim/baselines/oracle/oracle_sweep.py \
        --kernels all --machine sim/machines/$m.json --input $i
  done
done
```

| Output (`results/`) | Content |
|---------------------|---------|
| `oracle/runs/<m>/<k>/<input>/c<i>/` | gem5 runs; `window_trace.csv` is the B5 training data |
| `oracle/<m>/<input>/<k>.json` | best configuration per region (+ `.ipc`, `.ed2p`, `.table`, `.summary`) |
| `oracle/<k>.json` | `riscv_ooo` + `large` copy, read by `oracle_hinted` |
| `pgo/<k>.json` | `riscv_ooo` + `small` copy, read by `pgo` |

Other machines' builds read `results/oracle/<m>/large/` and `results/oracle/<m>/small/`
directly.

Useful flags: `--dry-run` (print the runs), `--analyze-only` (recompute the maps),
`--energy proxy|mcpat` (ED²P maps), `--jobs` (≤ 2).

### Step 4: B5 learned LUT

`build_lut.py` runs the three B5 steps on every machine's `small` sweep. Labels are
oracle-best configurations; features are per-window values from `window_trace.csv`. PyTorch
training takes the heavy lock.

```bash
# every sim/machines/*.json, model mlp → results/b5/<m>/
python3 sim/baselines/lut/build_lut.py
```

It writes `results/b5/<m>/`: `dataset.csv`, `model.pkl`, `train_report.json`, `lut.txt` (the
runtime LUT of `lut` and `lut_xfer`) and `lut.check.json`. Options: `--model
tree|knn|bins|transformer|all` (with `--select`), `--edges`, `--machines`.

The same pipeline step by step, for one machine:

```bash
python3 sim/baselines/lut/label_phases.py \
    --machine riscv_ooo --input small \
    --out results/b5/riscv_ooo/dataset.csv

python3 sim/baselines/lut/train_phase_classifier.py \
    --dataset results/b5/riscv_ooo/dataset.csv \
    --out-dir results/b5/riscv_ooo --model mlp --final-fit

python3 sim/baselines/lut/export_lookup_table.py \
    --model results/b5/riscv_ooo/model.pkl \
    --dataset results/b5/riscv_ooo/dataset.csv \
    --runtime-lut results/b5/riscv_ooo/lut.txt --check
```

`label_phases.py --threshold-baseline` adds the legacy threshold-rule column.

### Step 5: Baseline fidelity

Each reimplemented baseline (B2–B7, B9) must reproduce its paper's qualitative trend on a
microbenchmark before it is used (PROPOSAL §7). This step gives verdicts, not figures: about
30 short gem5 runs under the lock.

```bash
# build the microbenchmarks; every variant must equal plain under qemu
make -C benchmarks micro-check

# plan, run, re-evaluate → results/fidelity/<baseline>.json, summary.json
python3 sim/fidelity/fidelity.py --dry-run
python3 sim/fidelity/fidelity.py
python3 sim/fidelity/fidelity.py --evaluate-only
```

Verdicts (`PASS`, `FAIL`, `INVALID`, `INCOMPLETE`) are defined in
[docs/guide/fidelity.md](fidelity.md). `--machine sim/machines/<m>.json` checks
another machine; `--baselines B3 B9` restricts the run.

### Step 6: Baseline tuning

Equal tuning effort (PROPOSAL §8). Each tunable method — B2, B3, B4, B5, B8, B8+WinHint, B9,
WinHint+HW and WinHint's own compiler knobs — tries at most `--budget` points (default 16,
always including its published default) on the `small` input. The point with the best
geomean ED²P wins, per machine. This is the heaviest step after the matrix.

```bash
M="riscv_ooo riscv_ooo_small riscv_ooo_big"
python3 sim/baselines/tune/tune_baselines.py --machines $M --dry-run
python3 sim/baselines/tune/tune_baselines.py --machines $M
python3 sim/baselines/tune/tune_baselines.py --machines $M --select-only
```

| Output | Content |
|--------|---------|
| `results/tune/tuned.json` | chosen parameters, read by `run_experiments.py` by default |
| `results/tune/runs/` | tuning runs |
| `results/tune/b5/<point>/<m>/` | tuned B5 LUTs |
| `build/tune/bins/` | WinHint knob builds |

The first machine is the reference. **Tune every machine**: an untuned machine falls back to
the reference's B5 LUT root, which has no LUT for it, so `lut` is reported missing. To use
the untuned defaults instead, pass `--tuned none` in steps 8–9.

### Step 7: Hinted and per-machine binaries

The Makefile does not read `tuned.json`, but `run_experiments.py` refuses `winhint` and
`winhint_clairvoyance` binaries built without the tuned knobs, and looks for B8 in the tuned
`-cv_*` directory. First turn `tuned.json` into make arguments (the arrays stay empty without
`tuned.json` or when a method kept its default):

```bash
tuned_args() {  # tuned_args WinHint|B8|B8+WinHint → one VAR=value per line
  python3 -c 'import json, sys
try:
    t = json.load(open("results/tune/tuned.json"))
except OSError:
    sys.exit()
m = sys.argv[1]
if m == "WinHint":
    p = t.get("compiler", {}).get("winhint", {}).get("make", {})
else:
    ref = t["machines"][0]
    p = t.get("points", {}).get(m, {}).get(t.get("chosen", {}).get(m, {}).get(ref)) or {}
for k, v in p.items():
    print(f"{k}={v}")' "$1"
}
mapfile -t WH_ARGS  < <(tuned_args WinHint)
mapfile -t CV_ARGS  < <(tuned_args B8)
mapfile -t WCV_ARGS < <(tuned_args B8+WinHint)
```

Then build every variant; `machines` builds it once per `sim/machines/*.json`:

```bash
B="make -C benchmarks ARCH=riscv"

# WinHint, WinHint+HW, overhead
$B VARIANT=winhint "${WH_ARGS[@]}" machines

# B1 (reads results/oracle/…/large) and B7 (reads results/…/small)
$B VARIANT=oracle_hinted machines
$B VARIANT=pgo machines

# B6, IQ only and extended
$B VARIANT=jones machines
$B VARIANT=jones_full machines

# B8 alone (machine-independent) and with WinHint
$B VARIANT=clairvoyance "${CV_ARGS[@]}"
$B VARIANT=winhint_clairvoyance "${WH_ARGS[@]}" "${WCV_ARGS[@]}" machines

# optional: bit-identical vs plain under qemu
$B VARIANT=winhint check
```

Next to each binary the compiler writes `<kernel>.regions.json` (W\* per region, used by the
figures) and `<kernel>.winhint.json` / `<kernel>.jones*.json` (hint counts, target, knobs).
`tooling/winhint.sh verify` checks every hinted variant against `plain`
(→ `results/correctness.csv`).

### Step 8: Evaluation matrix

Every variant × 15 kernels × 3 machines, `large` input (about 21–22 variants). `large` runs
use region-aligned sampling (`sim/run_lengths.json`): one profile pass per kernel, one
checkpoint pass per binary (`build/ckpt/`), then up to 24 samples per run (36 for decoders).
`small` runs are simulated in full.

```bash
# the plan, and every MISSING input (binary, LUT, policy)
python3 sim/run_experiments.py --dry-run

# everything (resumable)
python3 sim/run_experiments.py

# sensitivity points (L2 size, memory latency)
python3 sim/run_experiments.py --machines riscv_ooo \
    --policies static winhint mlp \
    --sweep l2_size=256kB,512kB,2MB \
    --sweep mem_extra_latency_ns=20,40,80
```

Each run writes `results/gem5/<machine>/<kernel>/<variant>/`:

- the gem5 outdir: `stats.txt`, `config.ini`, `simout`, `simerr`, `region_stats.csv`,
  `window_trace.csv` (with `--trace`); sampled runs add `sNN/`, `stats.measured.txt` and
  `runlength.json`;
- `run.json` (command, status, wall time) and `gem5.log`.

Inputs other than `large` go to `<variant>.<input>/`. A sensitivity point is the derived machine
`<m>+<param>=<v>`, which reuses `<m>`'s binaries and LUTs.

**Restricting the matrix:**

| Flag | Effect |
|------|--------|
| `--kernels K …` | kernel subset |
| `--machines M …` | machine subset |
| `--input small` | input (default `large`) |
| `--policies P …` | variants, baseline IDs (`B0`–`B9`, `WinHint`, …) or tags (below) |
| `--jobs N` | parallel gem5 runs (≤ 2) |
| `--dry-run` | print the plan only |
| `--force` | rerun finished runs |
| `--bin-suffix -O3` | another build configuration (needs `oracle-O3` too) |
| `--window-args P:k=v,…` | override a policy's tunables |
| `--tuned FILE\|none` | tuned parameters (default `results/tune/tuned.json`) |
| `--lut-root DIR` | B5 LUT root (default `results/b5`) |
| `--lut-ref-machine M` | `lut_xfer` source (default `riscv_ooo`) |
| `--run-length full` | whole programs instead of sampled budgets (`--max-insts N`) |

All tags: `static`, `b0`–`b9`, `oracle`, `hw`, `winhint`, `hybrid`, `xfer`, `overhead`.

### Step 9: Collection and energy

`run_experiments.py` writes the summary after the runs (`--no-collect` skips it). To redo it:

```bash
# → results/gem5/summary.csv (IPC, window counters, energy, EDP, ED²P)
python3 sim/run_experiments.py --collect-only

# → <run>/energy.json, results/gem5/energy.csv
python3 sim/estimate_energy.py --results-root results/gem5
```

Energy splits each run's time by its residency in each window configuration. It uses McPAT
when built (`--model auto`, under the heavy lock), otherwise an analytic proxy that is only
meaningful for relative comparisons on one machine (`--model proxy`).

### Step 10: Figures

One call renders every figure (all inputs shown with their defaults). The real-hardware
figures are included when `results/hw/summary.csv` exists ([Part 2](#part-2-real-hardware)).

```bash
python3 analysis/plot_results.py \
    --summary results/gem5/summary.csv \
    --oracle-root results/oracle \
    --compiler-dir build/benchmarks/riscv/winhint \
    --hw-summary results/hw/summary.csv \
    --machines-dir sim/machines \
    --bin-root build/benchmarks/riscv \
    --machine riscv_ooo \
    --oracle-metric ipc \
    --out-dir results/figures \
    --format pdf
```

| Figure (`results/figures/`) | Shows | Needs steps |
|-----------------------------|-------|-------------|
| `wstar_vs_oracle.pdf` | predicted W\* vs oracle window, Spearman ρ | 3, 7 |
| `ipc_bars.pdf`, `ed2p_bars.pdf` | IPC / ED²P per kernel and baseline | 8, 9 |
| `switch_frequency.pdf` | switches per million instructions | 8, 9 |
| `hint_overhead.pdf` | hint density, static count, `winhint_nop`, code size | 7, 8, 9 |
| `sensitivity.pdf` | speedup vs L2 size and memory latency | 8 |
| `portability.pdf` | `lut` vs `lut_xfer` vs `winhint` per machine | 4, 8 |
| `hw_edp.pdf` | EDP vs R1 on silicon, 95 % CI | Part 2 |
| `hw_nop_overhead.pdf` | x86 NOP-hint overhead | Part 2 |
| `metrics.json` | every number behind the figures | — |

Only `large` rows are plotted. A figure whose inputs are missing is skipped with `[SKIP]`.
`--synthetic` renders every figure from generated data (self-test).

### Quick check

One kernel, one machine, `small` input (full runs of 40–550M instructions). It checks the
pipeline end to end, not the paper's numbers.

```bash
K=encoder_bert_tiny_infer
B="make -C benchmarks ARCH=riscv KERNELS=$K"

$B VARIANT=plain
$B VARIANT=oracle
$B VARIANT=winhint

# → results/oracle/riscv_ooo/small/, results/pgo/$K.json
python3 sim/baselines/oracle/oracle_sweep.py --kernels $K --input small

$B VARIANT=pgo
$B VARIANT=oracle_hinted ORACLE_DIR="$PWD/results/oracle/riscv_ooo/small"

python3 sim/baselines/lut/build_lut.py \
    --machines riscv_ooo --kernels $K --model tree --split none

P="static B1 B2 B3 B5 B7 WinHint WinHint+HW overhead"
python3 sim/run_experiments.py --machines riscv_ooo --kernels $K \
    --input small --tuned none --policies $P --dry-run
python3 sim/run_experiments.py --machines riscv_ooo --kernels $K \
    --input small --tuned none --policies $P

# figures plot `large` rows only: check the pipeline with synthetic data
python3 analysis/plot_results.py --synthetic --out-dir results/figures-selftest
```

- The rows land in `results/gem5/summary.csv` with `input = small`.
- B5 uses `--split none`: the leave-kernels-out split needs at least two kernels.
- `ORACLE_DIR` maps B1 to the `small` oracle, so no `large` sweep is needed. It must be an
  absolute path (make runs in `benchmarks/`).

### gem5 builds and statistics

- `RISCV_clean`: hints are plain no-ops; used for bit-identical checks.
- `RISCV_winhint`: window resizing plus every `window_policy`.

New statistics live under `system.cpu.window.*`: `switches`, `periods`, `hints`,
`setwinHints`, `regionHints`, `drainCycles`, `cyclesInConfig::<i>`, per-structure occupancy
(`robOccMean`, `robOccMax`, `robOccDist`, … for ROB/IQ/LQ/SQ) and `robFullCycles`. B9 adds an
`ltp` group with the parking-queue counters. The authoritative list is in
`sim/gem5/src/cpu/o3/window/controller.cc` and `ltp/ltp.hh`.

---

## Part 2: Real hardware

On Intel hybrid silicon the window hint becomes a P-core/E-core placement decision.
[docs/guide/hardware/index.md](hardware/index.md) has the deepest details (per-host status, R4/R5 deviations,
every libwinhint variable).

### Platform

Intel Core 5 120U (Raptor Lake-U refresh, family 6 model 186):

- CPUs 0–3: P-cores (2 cores × SMT, PMU `cpu_core`);
- CPUs 4–11: E-cores (PMU `cpu_atom`);
- RAPL domains `package-0`, `core`, `uncore`, `psys`; `energy_uj` is root-only (mode 0400);
- `intel_pstate` (HWP), kernel 7.0, `perf_event_paranoid=1`.

### libwinhint

The runtime of the call-mode builds (`hw/libwinhint/` → `build/libwinhint/`).
`__winhint_setwin(W)` moves the calling thread:

| W | Thread goes to |
|---|----------------|
| ≥ `WINHINT_THRESHOLD` (default 192) | P-core set |
| 0 < W < threshold | E-core set |
| 0 | `WINHINT_RELEASE` (default: original affinity) |

Redundant requests cost no syscall; every real migration is timed and checked with
`sched_getcpu()`. `__winhint_region(id)` marks regions for logging and for R5.

The x86 NOP hints of the `winhint` (asm) build are no-ops that libwinhint never sees; they
only measure hint overhead. Migration uses call mode (`-winhint-emit=call`), which places the
calls at the same boundaries with the same W.

| Variable | Default | Meaning |
|----------|---------|---------|
| `WINHINT_MODE` | `migrate` | `off`, `log`, `migrate`, `sondag` (R5) |
| `WINHINT_THRESHOLD` | 192 | W from which the P-cores are requested |
| `WINHINT_PCPUS`, `WINHINT_ECPUS` | from sysfs | P/E core sets (cpulists) |
| `WINHINT_PIN` | `set` | `single`: one CPU per side |
| `WINHINT_HYST` | 1 | requests needed before migrating |
| `WINHINT_MIN_DWELL_US` | 0 | minimum time between migrations |
| `WINHINT_RELEASE`, `WINHINT_INITIAL` | `orig` | side for W = 0 / at start |
| `WINHINT_LOG`, `WINHINT_TRACE` | — | per-region CSV + JSON; migration trace |
| `WINHINT_PERF`, `WINHINT_RAPL` | — | per-region counters; per-region energy |
| `WINHINT_SONDAG_*` | 2, 1.4 | R5 sampling (`_K`), threshold, `_TYPES` |

Modes: `off` returns at once (call overhead only), `log` decides and counts but never
migrates. Without a hybrid topology, `migrate` and `sondag` fall back to `log`.

### Configurations

Every run is measured by `build/hw/wh_measure`: wall time, RAPL package and core energy, EDP,
and user-mode cycles/instructions per core type.

| Config | Baseline | Binary | Root |
|--------|----------|--------|:----:|
| `R0-P`, `R0-E` | pinned to all P / all E CPUs | `plain` | |
| `R1` | stock Linux (EEVDF + ITMT/HFI), the reference | `plain` | |
| `R2-<preset>` | sched_ext, power-saving presets | `plain` | ✓ |
| `R3-lpmd` | intel-lpmd, E-cores as low-power CPUs | `plain` | ✓ |
| `R4-PIE` | PIE-style daemon (Van Craeynest, ISCA'12) | `plain` | |
| `R5-Sondag` | Sondag & Rajan (CGO'11) | `oracle_call` | |
| `WH` | WinHint | `winhint_call` | |
| `WH-off` | WinHint call overhead | `winhint_call` | |
| `NOP-*-{P,E}` | x86 NOP-hint overhead | `plain` / `winhint` | |

Binaries are in `build/benchmarks/x86/<variant>/`. Details per configuration:

- **R1** runs unpinned; `record_system.sh` records the scheduler state.
- **R2** presets: `bpfland_powersave`, `cosmos_powersave`, `lavd_powersave`
  (`hw/baselines/r2_sched_ext/run_scx.sh presets`); the scheduler is attached per group
  (`--enable-scx`).
- **R3** starts the daemon per group (`--enable-lpmd`; mode `--lpmd-mode`, default `AUTO`).
- **R4** runs `pie_daemon -b perf` (`--pie-interval-ms`, `--pie-slack`, `--pie-hyst`). The
  PMCTrack backend needs a kernel module that does not build on kernel 7.0.
- **R5** runs `WINHINT_MODE=sondag` (`--sondag-k`, `--sondag-threshold`, `--sondag-types-dir`).
- **WH** runs `WINHINT_MODE=migrate` (`--threshold`, `--hyst`, `--min-dwell-us`).
- **NOP** rows pin to one P (E) CPU and use `perf stat` when available.

`--configs core` (default) is every row except R2 and R3; `--configs all` adds them. Energy
needs RAPL access in every row (root, or the read grant below).

### Step 1: Build

No root, `-j1`.

```bash
# plugins (if not built yet), libwinhint and the hw tools
tooling/winhint.sh compiler:build
tooling/winhint.sh hw:build                    # → build/libwinhint/, build/hw/
python3 -m pytest -q hw/tests                  # driver unit tests

# P/E switch cost on an idle machine (read by the call-mode builds)
build/libwinhint/bench_migration -n 200 -s \
    -o results/hw/system/migration_cost_smt-on.json

# x86 binaries → build/benchmarks/x86/<variant>/
make -C benchmarks ARCH=x86 plain winhint winhint_call oracle_call
make -C benchmarks ARCH=x86 VARIANT=winhint_call check
```

The call-mode builds use `switch_cost_us` from `migration_cost_smt-on.json` as the compiler's
P/E switch cost (`MIGRATION_US=<µs>` overrides it), so measure it first.

Optional R5 static region typing (default: one type per region):

```bash
mkdir -p results/hw/sondag_types
for j in build/benchmarks/x86/oracle_call/*.regions.json; do
  k=$(basename "$j" .regions.json)
  python3 hw/baselines/r5_sondag/region_types.py "$j" \
      -o results/hw/sondag_types/$k.types
done
```

Optional third-party baselines (network, no root, heavy lock):

```bash
tooling/winhint.sh hw-baselines:build r2       # sched_ext → build/hw-baselines/scx
tooling/winhint.sh hw-baselines:build r3       # intel-lpmd → build/hw-baselines/lpmd/prefix
tooling/winhint.sh hw-baselines:build r4       # PMCTrack CLI (R4 uses perf anyway)
```

### Step 2: One-time host setup

Root, opt-in, run by you. The scripts never call `sudo`. System-wide changes need
`WINHINT_ALLOW_SYSTEM_CHANGES=1` (or the driver's `--allow-system-changes`, `--enable-scx`,
`--enable-lpmd`), and the saved state is restored at the end, also on errors.

| Need | For | How |
|------|-----|-----|
| nothing | own-thread affinity (libwinhint, R0, R4, R5), own counters | — |
| RAPL read | energy in every measured run | driver as root, or the read grant below |
| root | fixed frequency, SMT on/off | driver as root + `--allow-system-changes` |
| root | R2 sched_ext | driver as root + `--enable-scx` |
| root + D-Bus | R3 intel-lpmd | install the policy below, then `--enable-lpmd` |

```bash
# RAPL read grant (temporary; revoke when done)
sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r1_stock/rapl_access.sh grant
hw/baselines/r1_stock/rapl_access.sh status
sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r1_stock/rapl_access.sh revoke

# R3 D-Bus policy (once)
sudo install -m 644 \
    build/hw-baselines/lpmd/prefix/etc/dbus-1/system.d/org.freedesktop.intel_lpmd.conf \
    /etc/dbus-1/system.d/
sudo systemctl reload dbus
```

Measure on AC power on an idle machine, never while gem5 or compiler builds are running.

### Step 3: Campaign

`hw/run_hw_experiments.py`. Defaults: every kernel, `--input large`, `--reps 10`,
`--cooldown 2`, `--out results/hw`. As root, run it through
`sudo -E env "PATH=$PATH"` (results are chowned back).

```bash
# a. plan + privilege check (no measurement)
python3 hw/run_hw_experiments.py --dry-run

# b. smoke run on the synthetic phase workload (no root, no energy)
python3 hw/run_hw_experiments.py --synthetic --reps 1 \
    --allow-no-rapl --cooldown 0.2 --out results/hw/smoke

# c. µarch probes and R4/R5 fidelity (no root; idle machine)
python3 hw/run_hw_experiments.py --uarch       # → results/hw/uarch/
python3 hw/run_hw_experiments.py --fidelity    # → results/hw/fidelity/

# d. R0, R1, R4, R5, WH, WH-off, NOP; SMT on and off; fixed frequency
sudo -E env "PATH=$PATH" python3 hw/run_hw_experiments.py \
    --reps 10 --smt on,off \
    --governor performance --epp performance --no-turbo 1 \
    --sondag-types-dir results/hw/sondag_types \
    --allow-system-changes

# e. R2 and R3 groups (after hw-baselines:build r2/r3 and the D-Bus policy)
sudo -E env "PATH=$PATH" python3 hw/run_hw_experiments.py \
    --configs R1,R2-bpfland_powersave,R2-cosmos_powersave,R2-lavd_powersave,R3-lpmd \
    --enable-scx --enable-lpmd --reps 10 --smt on,off \
    --governor performance --epp performance --no-turbo 1 \
    --allow-system-changes

# f. re-summarize, then the hardware figures
python3 hw/run_hw_experiments.py --summarize-only
python3 analysis/plot_results.py \
    --hw-summary results/hw/summary.csv --out-dir results/figures
```

How the driver behaves:

- **Order.** Runs are shuffled per repetition with a seed (`--seed`), R1 first. R2/R3 runs
  are grouped so each daemon starts once per group.
- **Migration cost** is measured once per SMT state into `results/hw/system/`
  (`--no-bench-migration` skips it).
- **Refusals.** It refuses to measure unless the governor is `performance` or the campaign
  fixes it (`--allow-unfixed-governor` only records the policy), and without RAPL access
  unless `--allow-no-rapl`. Fewer than 10 repetitions are flagged as a smoke run.
- **Correctness.** Each run's stdout SHA-256 is compared with R1's (`output_ok`).
- **Restricting.** `--kernels k1,k2`, `--configs …`, `--smt on`; `--retry-failed` reruns
  failed rows. Fidelity takes `--fidelity-secs`, `--fidelity-reps` and the campaign's
  `--pie-slack` / `--pie-hyst`. Tuning R4 on the `small` inputs is manual
  ([docs/guide/hardware/index.md](hardware/index.md) §3).

### llama.cpp (optional)

The kernels are replaced by llama.cpp's `llama-simple`, built vanilla and with ggml hooks that
call libwinhint. Single-threaded (`OMP_THREAD_LIMIT=1`); the NOP rows are dropped.

```bash
# fetch + build (heavy lock) + model + functional check → build/integrations/llamacpp/
bash hw/integrations/llamacpp/build_llamacpp.sh all

sudo -E env "PATH=$PATH" python3 hw/run_hw_experiments.py --llamacpp \
    --configs R0-P,R0-E,R1,WH,WH-off,R5-Sondag --reps 10 \
    --governor performance --epp performance --no-turbo 1 \
    --allow-system-changes --out results/hw/llamacpp
```

### Outputs

| Path (`results/hw/`) | Content |
|----------------------|---------|
| `raw.csv` | one row per run; append-only, so the campaign resumes |
| `summary.csv` | mean and 95 % CI per kernel, config and SMT state, ratios to R1 |
| `runs/<run_id>/` | `wh_measure` JSON, libwinhint CSV/JSON, PIE log |
| `system/` | system snapshots, migration cost per SMT state |
| `uarch/`, `fidelity/` | µarch probe knees; R4/R5 trend checks (must PASS first) |

`summary.csv` feeds `analysis/plot_results.py --hw-summary` → `hw_edp.pdf`,
`hw_nop_overhead.pdf`.

---

## Part 3: Reproduce everything

The artifact is this repository. Everything is built under `build/` and written under
`results/`; nothing needs root except the measured real-hardware runs. No results are
shipped: every number in the paper is regenerated by the commands below.

**Resources:** 6 GB of RAM; about 13 GB of disk for the envs and package cache, plus 5 GB or
more for `build/`. The two gem5 builds take several hours at `-j2`; the full matrix takes far
longer than the builds.

| # | What | Where |
|---|------|-------|
| 0 | environment and sources | [README → Quickstart](../../README.md#quickstart) |
| 1 | tests and correctness | below |
| 2 | gem5 study, steps 1–10 | [Part 1](#part-1-gem5-evaluation) |
| 3 | real hardware (optional; Intel hybrid CPU, root) | [Part 2](#part-2-real-hardware) |

Tests and correctness (PROPOSAL §7, "Correctness" and "Mechanism"), once the gem5 and
benchmark builds exist:

```bash
make validate                          # host tests, coverage gate, docs
tooling/winhint.sh test sim            # gem5 mechanism + every policy (heavy lock)
tooling/winhint.sh verify              # every hinted variant == plain → results/correctness.csv
```

For a fast end-to-end check, run the [quick check](#quick-check); `--kernels`, `--machines` and
`--input small` restrict any step.

## Next

- [Evaluation methodology](../concepts/methodology.md): what each experiment tests.
- [Deviations](../deviations.md): where the baselines differ from their papers.
- [Testing and verification](testing.md): the checks behind the numbers.
