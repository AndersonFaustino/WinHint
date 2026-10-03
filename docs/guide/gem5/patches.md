# gem5 patches

WinHint's window resizing is not in stock gem5. This page documents the build that adds it,
`RISCV_winhint`: which existing gem5 files the patches change, the files the overlay adds, how
a resize behaves, how a hint is recognised, and what each run writes. Every gem5 baseline and
WinHint run on this one mechanism, so that only the policy and the binary differ between them
([Evaluation methodology](../../concepts/methodology.md)). The flags that drive it are in
the [gem5 model](index.md) reference.

**Terms.** The *window* is the ROB, IQ, LQ and SQ; a *configuration* is one row of the window
table ([interfaces.md §3](../../interfaces.md#3-window-configuration-table)); a *cap* is
the size a configuration allows a structure. Other terms are in the
[Glossary](../../reference/glossary.md).

## Build composition

`RISCV_winhint` is

1. unmodified gem5 v25.1.0.0,
2. plus the overlay `sim/gem5/src/` (new files only, copied to `<gem5>/src/`),
3. plus the patches in this directory, applied in name order:

| Patch | Owner | Content |
|-------|-------|---------|
| `gem5_v25.1.0.0_winhint.patch` | sim (mechanism) | edits to **existing** gem5 files only: window caps in ROB/IQ/LSQ, CPU hooks, `window_*` params, cache counters |
| `gem5_v25.1.0.0_zz_ltp.patch` | B9 | Long-Term Parking hooks (applied after the winhint patch; see [`ltp/`](../baselines/ltp.md)) |

`RISCV_clean` is built from the untouched tree. `tooling/winhint.sh gem5:build
winhint` assembles overlay + patches; nothing is ever left applied to the
clean tree. The hint instruction (`ori x0, x0, IMM`) is already decoded by
stock gem5 as the no-op `ori_hint`, so the RISC-V decoder is **not** patched:
the hint stays a no-op on `RISCV_clean` and is recognised only at commit by
the window controller on `RISCV_winhint`.

## What the winhint patch changes (existing files)

| File(s) | Change |
|---------|--------|
| `cpu/o3/rob.{hh,cc}` | `setCap/getCap/physEntries/occupancy`; `numFreeEntries()` (both overloads) = `min(phys_free, cap − occupancy)`; `isFull()` also true at the cap |
| `cpu/o3/inst_queue.{hh,cc}` | v25.1 `IQUnit`: per-unit cap, `numFreeEntries()` capped the same way. `InstructionQueue::setCap(total)` splits a total cap over the IQ units in proportion to their physical size (one unit: the cap itself); `physEntries/occupancy/getCap` |
| `cpu/o3/lsq_unit.{hh,cc}`, `cpu/o3/lsq.{hh,cc}` | `setCaps(lq, sq)`; `numFreeLoad/StoreEntries()` capped; `lqFull()/sqFull()` also true at the cap (IEW dispatch blocks) |
| `cpu/o3/commit.hh`, `cpu/o3/iew.hh` | `windowCapsChanged()`: re-broadcast free entries to rename on the next cycle after a cap change |
| `cpu/o3/cpu.{hh,cc}` | `std::unique_ptr<WindowController> window`, created at the end of the constructor; hooks in `startup()`, end of `tick()`, `instDone()` (commit) |
| `cpu/o3/BaseO3CPU.py` | params of [interfaces.md §4](../../interfaces.md#4-gem5-interface-riscv_winhint-build) |
| `mem/cache/base.hh`, `mem/cache/queue.hh` | read-only accessors: `windowDemandMisses()` (cumulative demand misses; the controller takes deltas) and `windowOutstandingMisses()` (allocated MSHRs) |

### Resize semantics

- Physical structures are sized for the largest configuration (`se.py` does
  this). A configuration only sets caps.
- free = max(0, min(phys_free, cap − occupancy)) — no unsigned underflow, cap
  clamped to [1, physical], so the pipeline can always make progress.
- Growing takes effect at once. Shrinking flushes nothing: rename/IEW see 0
  free entries and stall dispatch until occupancy drains to the new cap
  (`system.cpu.window.drainCycles` counts the cycles in which some structure
  is still above its cap).
- Rename learns free counts through the time buffers; instructions already in
  flight between rename and the ROB when a shrink happens can overshoot the
  cap transiently (never the physical size). Occupancy maxima per
  configuration are therefore recorded only in "settled" cycles (every
  structure at or below its cap).

## Overlay (`sim/gem5/src/cpu/o3/window/`)

| File | Role |
|------|------|
| `controller.{hh,cc}` | mechanism + instrumentation (caps, sampling, hint decode, region/trace CSVs, stats) |
| `policy.hh` | policy interface v1, `WindowArgs` (`k=v`), `WindowPolicyRegistry` + `WINHINT_REGISTER_POLICY` |
| `static_policy.cc` | B0 |
| `hint_policy.{hh,cc}` | WinHint, B1 (`oracle_hinted`), B6 (`structs=iq` for IQ-only), B7 |
| `hybrid_policy.cc` | WinHint+HW |
| `occupancy/mlp/bbv/lut_policy.*`, `docs/guide/gem5/policies.md` | B2–B5 (documented in [`docs/guide/gem5/policies.md`](policies.md)) |
| `ltp/` | B9 ([README](../baselines/ltp.md)) |
| `SConscript` | compiles `controller.cc` and every `*_policy.cc`; debug flag `WinHint` |

## Hint decoding

The encoding is the contract of [interfaces.md §2](../../interfaces.md#2-the-hint-isa-contract).

At commit, a 32-bit `ori x0, x0, IMM` (low 20 bits `0x06013`) with
`IMM[11] = 0` and `IMM[4:0] = 21` is `setwin(W = IMM[10:5] · 8)`;
`IMM[4:0] = 23` is `region(IMM[10:5])`. Other immediates (including the
Zicbop prefetch tags 0/1/3) are ignored. Detection is non-speculative.

## Policies (`window_policy`, `window_args`)

Unknown policy names are rejected by `se.py` and by `fatal()` in the
registry; `window_args` keys not read by the selected policy are `fatal()`.

| Policy | Baseline | Documented in |
|--------|----------|---------------|
| `static` | B0 | below |
| `hint` | WinHint, B1, B6, B7 | below |
| `hybrid` | WinHint+HW | below |
| `occupancy`, `mlp`, `bbv`, `lut` | B2–B5 | [`docs/guide/gem5/policies.md`](policies.md) |
| `ltp` | B9 | [`ltp/`](../baselines/ltp.md) |

**`static`** keeps the fixed `window_initial`. Hints are ignored (regions are still recorded).
It reads no `window_args`.

**`hint`** maps `setwin(W)` to the smallest configuration with ROB ≥ W; W = 0 or a W that is
too large selects the largest. It starts in `window_initial` (se.py: largest).

| `window_args` key | Default | Meaning |
|-------------------|---------|---------|
| `structs` | `all` | structures that follow the hint: `iq` = B6 as published (IQ only), or any of `rob+iq+lq+sq`, `lsq`; the other structures stay at the largest config |

**`hybrid`** (WinHint+HW) treats the hint as a ceiling and jumps to it. Before any hint the
ceiling is the largest configuration. Per period, within [`floor`, ceiling]:

- if L2 MPKI ≥ `miss_mpki`: MLP ≥ `mlp_thr` → ceiling, else shrink one step;
- otherwise grow one step if some structure was full in more than `up_frac` of the cycles, and
  shrink one step if every mean occupancy is below `down_margin` × the next smaller
  configuration. The full cycles of the period right after a shrink are its drain and do not
  trigger growth.
- If no L2 miss was ever seen (no L2 attached), L1D MPKI is used instead.

| `window_args` key | Default |
|-------------------|---------|
| `miss_mpki` | 1.0 |
| `mlp_thr` | 1.5 |
| `up_frac` | 0.05 |
| `down_margin` | 0.9 |
| `floor` | 0 |

MLP (`WindowSample::mlp`, trace column `mlp`): mean number of outstanding L1D
misses (allocated L1D MSHRs) over the cycles with at least one outstanding
miss (Chou et al.); `WindowSample::mlpAll` is the mean over all cycles.
L1D/L2 misses are per-period deltas of the caches' demand-miss counters
(`window_l1d`, `window_l2`, set by `se.py`).

## Outputs (gem5 outdir)

- `stats.txt`, group `system.cpu.window`: `switches`, `periods`, `hints`,
  `setwinHints`, `regionHints`, `drainCycles`; per configuration `<i>`:
  `cyclesInConfig`, `{rob,iq,lq,sq}OccSum`, `{rob,iq,lq,sq}OccMean`,
  `{rob,iq,lq,sq}OccMax` (settled cycles), `{rob,iq,lq,sq}FullCycles`
  (no free entry under the cap); histograms `{rob,iq,lq,sq}OccDist` and
  `l1dOutstandingDist` (all cycle-weighted).
- `window_trace.csv` (`--window-trace`), one row per period:
  `cycle,insts,ipc,rob_occ_mean,iq_occ_mean,lq_occ_mean,l1d_mpki,l2_mpki,mlp,branch_mpki,config,region`
  (`config` = configuration in effect at the end of the period, before the
  policy's decision).
- `region_stats.csv` (whenever region markers commit, under every policy),
  one row per visit: `region,config,enter_cycle,cycles,insts`; `config` is
  the configuration in effect for most of the visit's cycles.

## Development

```bash
# worktree of the clean clone, overlay + patch, build under the heavy lock
git -C $WINHINT_BUILD/gem5/src worktree add $WINHINT_BUILD/gem5/dev-B HEAD
cp -r $WINHINT_ROOT/sim/gem5/src/. $WINHINT_BUILD/gem5/dev-B/src/
git -C $WINHINT_BUILD/gem5/dev-B apply $WINHINT_ROOT/sim/patches/gem5_v25.1.0.0_winhint.patch

# edit existing gem5 files in dev-B, then regenerate the patch from the
# tracked files only (the overlay is untracked there and stays out)
git -C $WINHINT_BUILD/gem5/dev-B diff > $WINHINT_ROOT/sim/patches/gem5_v25.1.0.0_winhint.patch
```

Tests: [`sim/tests/run_tests.sh`](../../../sim/tests/run_tests.sh) (see the header of that script, and
[Testing and verification](../testing.md)).

## Next

- [Window policies](policies.md): the reactive B2–B5 policies.
- [Long-Term Parking](../baselines/ltp.md): the B9 policy and the
  `zz_ltp` patch.
- [Deviations](../../deviations.md#infrastructure): where the shared mechanism departs from the
  proposal.
