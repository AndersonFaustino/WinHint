# Compiler statistics

Besides the hinted binary, the WinHint compiler pass writes two JSON files per kernel: the
region map and the statistics file. They are how the rest of the evaluation sees the
compiler's decisions: the B1 oracle and B7 profile builds reuse the region ids, and the
figures compare the predicted window W\* with the oracle and count the hints
([Evaluation methodology](../concepts/methodology.md)). This page is the schema of both files.

**Terms.**

- *Region*: a top-level loop nest, numbered as below.
- `W*`: the window (ROB entries) the cost model predicts a loop needs; how it is computed is
  in [Compiler → Cost model](../guide/compiler.md#cost-model).
- *Configuration*: a row of the machine's window table
  ([interfaces.md §3](../interfaces.md#3-window-configuration-table));
  `configForW(W)` is the smallest configuration whose ROB is at least W.
- *`setwin(W)` / `region(id)`*: the two hint kinds
  ([interfaces.md §2](../interfaces.md#2-the-hint-isa-contract)).

Other terms are in the [Glossary](glossary.md).

## Files

The `HintPlacement` pass (`WinHint.so`) writes two JSON files per module when
`-mllvm -winhint-out-dir=<dir>` is given (the benchmarks Makefile always passes it):

| File | Written in modes | Consumers |
|------|------------------|-----------|
| `<kernel>.regions.json` | all (`setwin`, `regions`, `from-json=`) | B1 `sim/baselines/oracle/`, B7 `compiler/baselines/pgo/`, `analysis/plot_results.py` |
| `<kernel>.winhint.json` | all | `analysis/plot_results.py`, compile-time / hint-count tables |

`<kernel>` is `-winhint-kernel=<name>`, or the stem of the source file up to the first dot.
`-winhint-stats-file=<path>` writes the stats file to an explicit path instead.
Doubles are plain JSON numbers; `null` stands for "unbounded / not applicable".

## Region ids

The numbering is part of the contract in
[interfaces.md §5](../interfaces.md#5-benchmarks-and-compiler-flags).

Regions are the **top-level loop nests** of every defined function. Functions are sorted
by name; within a function the nests are numbered in program order (reverse post-order of
their headers). The id is the index in that sequence, starting at 0. Ids are deterministic
for a given IR, so the `oracle` (markers), `oracle_hinted`/`pgo` (`from-json=`) and
`winhint` builds of the same source at the same `-O` level agree. The hint payload holds
6 bits, so ids ≥ 63 share the marker `region(63)` (`encoded_id`; `overflow: true`).

## `<kernel>.regions.json`

```json
{
  "kernel": "encoder_bert_tiny_infer",
  "target": "riscv_ooo",              // "name" of the machine JSON
  "num_regions": 12,
  "overflow": false,
  "regions": {
    "0": {
      "function": "attention",        // enclosing function
      "header": "for.cond1.preheader",// loop header block name (or "loopN")
      "line": 57,                     // source line of the loop (needs -g / -gline-tables-only; else 0)
      "encoded_id": 0,                // id carried by region(id) = min(id, 63)
      "w_star": 256,                  // W* of the outermost loop of the nest
      "nest_w_star": 143.2,           // W* weighted by dynamic instructions over the nest  <- predicted window
      "config": 3,                    // configForW(w_star)
      "nest_config": 2,               // best single config for the whole nest (cost model)
      "L_mem": 220.0, "D_indep": 53.3, "CP": 14.0,   // of the outermost loop
      "footprint_bytes": 25165824,    // null = unbounded
      "dyn_insts_est": 1.2e7,
      "conservative": false           // an unknown trip count or stride was assumed
    }
  }
}
```

The predicted window of a region for the Spearman-vs-oracle figure is `nest_w_star`
(fall back to `w_star`).

## `<kernel>.winhint.json` (stats, `"schema": "winhint-stats/1"`)

| Key | Type | Meaning |
|-----|------|---------|
| `schema` | str | `winhint-stats/1` |
| `kernel`, `module` | str | kernel name, LLVM module id |
| `llvm_version` | str | LLVM the plugin was built against |
| `mode` | str | `-winhint-mode` (`setwin`, `model`, `regions`, `from-json=<f>`) |
| `emit` | str | `asm`, `call`, `none` |
| `triple`, `target` | str | target triple, machine name |
| `switch_cost_cycles`, `hysteresis`, `min_region_insts` | num | placement parameters actually used |
| `cp_model` | str | `-winhint-cp-model` (`ii` or `width`) |
| `hints_setwin`, `hints_region` | int | **static** hint counts actually emitted into this module's IR (0 with `emit: none`, or in `asm` mode on a triple without a hint encoding; the hints are still placed and listed in `hints`) |
| `compile_time_ms` | num | wall time of the WinHint pass (analysis + placement + emission) |
| `analysis_time_ms` | num | part of it spent in `WindowDemandAnalysis` (incl. SCEV/DA/LoopInfo) |
| `num_regions` | int | number of regions |
| `window` | array | the configuration table used: `[{rob, iq, lq, sq}, ...]` |
| `regions` | array | one object per region: `id` plus every key of the `regions.json` entry, plus `entry_setwin` (W of the hint at the nest entry, `null` = inherits the window in force), `entry_config` (`configForW(entry_setwin)` or `null`), `setwin` (all setwin W values placed inside the nest, program order) |
| `hints` | array | one object per placed hint: `function`, `region`, `line`, `loop`, `depth` (1 = top-level loop), `kind` (`setwin`/`region`), `value` (W or id), `emitted` (`false` when the hint was not inserted, see `hints_setwin`), `encoding` (human-readable instruction) |
| `loops` | array | every loop (pre-order per function): `function`, `header`, `line`, `depth`, `region` (top-level loops only), `trip`, `trip_known` (exact SCEV count; `false` when `trip` is only an upper bound from call-site constants or SCEV, or the assumed `-winhint-unknown-trip`), `body_insts`, `L_mem`, `misses_per_iter`, `D_indep` (`null` = no long-latency loads), `MLP`, `CP`, `RecMII`, `W_mlp`, `W_cp`, `w_star`, `config`, `conservative` |

A `setwin` W value always satisfies `W = 8·payload`; it is the ROB size of the chosen
configuration ([interfaces.md §3](../interfaces.md#3-window-configuration-table): the hardware picks the smallest config with ROB ≥ W).

## Model quantities (per loop, `print<winhint-demand>` shows the same values)

- `L_mem`: mean service latency (cycles) of the loop's independent long-latency load
  classes, from the level that serves them (footprint / reuse distance vs. cache capacity
  × `effective_fraction`).
- `D_indep`: dynamic instructions per independent miss = instructions of one iteration
  (own body plus the inner loops' dynamic instructions) / misses per iteration of the own
  blocks (unrolled copies are grouped; pointer-chase and DA loop-carried loads are excluded).
- `MLP` = min(L_mem · rate / D_indep, L1D MSHRs), rate = min(issue_width, body/II): the
  number of independent misses issued while one is outstanding. It is fractional (< 1 when
  misses are rarer than one per `L_mem · rate` instructions); a pointer chase has `MLP` = 1,
  a loop without long-latency loads `MLP` = 0. A `winhint.mlp_target` in the machine JSON
  overrides it.
- `CP`: critical path of one iteration's dependence DAG; `RecMII`: longest register recurrence;
  `II` = max(RecMII, body/issue_width, divider occupancy).
- `W_mlp` = ceil(MLP · D_indep) (= L_mem · rate unless the MSHRs bound it; one body for a
  pointer chase); `W_cp` = ceil(CP · rate) (= CP · issue_width when the loop is
  issue-bound); with `-winhint-cp-model=width`, `W_cp` = ceil(CP · issue_width), the
  closed form of PROPOSAL §3.1 (ablation).
- `W*` = min(W_max, max(W_mlp, W_cp)); `config` = smallest config with ROB ≥ W*.

## Next

- [Compiler](../guide/compiler.md): the pass, its options and the cost model behind these
  quantities.
- [Interfaces](../interfaces.md): the hint encoding and the region-id contract.
