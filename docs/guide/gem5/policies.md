# Reactive window policies (B2–B5)

This page documents the four reactive hardware baselines of the gem5 study: `occupancy` (B2),
`mlp` (B3), `bbv` (B4) and `lut` (B5). Each is a published (or, for B5, published-style) way
for the core to resize its window from run-time counters, without compiler help, and each is
reimplemented here because no artifact exists. Why these baselines are in the comparison and
how they are tuned is in [Evaluation methodology](../../concepts/methodology.md);
the gem5 setup they run in is the [gem5 model](index.md).

**Terms.** The *window* is the ROB, IQ, LQ and SQ, resized together. A *configuration* is one
row of the machine's window table (`window_rob/iq/lq/sq`,
[interfaces.md §3](../../interfaces.md#3-window-configuration-table)); index 0
is the smallest. A *period* is `window_period` cycles. A *cap* is the size a configuration
allows a structure. Other terms are in the [Glossary](../../reference/glossary.md).

## Shared mechanism

All four policies implement [`policy.hh`](../../../sim/gem5/src/cpu/o3/window/policy.hh) (interface v1) and use the same resize
mechanism, the `WindowController` ([`controller.cc`](../../../sim/gem5/src/cpu/o3/window/controller.cc)):

- Growing takes effect immediately.
- Shrinking gates dispatch until each structure's occupancy is at or below its new cap. Nothing is flushed.

Only the decisions differ between policies. Every policy decides at the end of each `window_period` (default 1000 cycles), using the per-period deltas in `WindowSample`. BBV also observes every committed control instruction.

Tunables are passed as `window_args` (`se.py --window-args k=v,k=v`). A key that the selected policy does not read is rejected with `fatal()`.

Unit tests are in [`tests/`](../../../sim/gem5/src/cpu/o3/window/tests). They need no gem5, because gem5's logging and types headers are stubbed:

```bash
# build and run the policy unit tests on the host
make -C sim/gem5/src/cpu/o3/window/tests
```

---

## B2 `occupancy`: Ponomarev, Kucuk, Ghose, MICRO-34 2001

**Paper.** IQ, ROB and LSQ are split into partitions, and each resource is resized on its own:

- **Down.** Occupancy is averaged over an *update period* (2048 cycles). At the end of the period, if `size - avg_occupancy >= partition`, the resource gives up one partition.
- **Up.** An *overflow counter* counts the cycles in which dispatch blocks because the resource is full. As soon as the counter passes a threshold within the update period, the resource gains one partition.

**Here.**

- A partition is one step of the configuration table, and ROB, IQ, LQ and SQ move together.
- The window grows by one step as soon as **any** structure's accumulated full cycles (`robFullCycles` … `sqFullCycles`) pass `up_frac × update × window_period`. This is checked at every period.
- At the end of each update period (`update` periods), the window shrinks by one step only if **every** structure passes `cap[c] − mean_occ ≥ down_factor × (cap[c] − cap[c−1])`.
- The counters restart after every resize.

| key | default | meaning |
|-----|---------|---------|
| `update` | 2 | sample periods per update period (2 × 1000 cycles ≈ the paper's 2048) |
| `up_frac` | 0.05 | overflow threshold, as a fraction of the update-period cycles |
| `down_factor` | 1.0 | shrink when this many steps of slack are free (1.0 is the paper's rule, `avg_occ ≤ next smaller size`) |

**Deviations.**

1. All structures resize together (whole-window table), whereas the paper resizes each one independently. The rule is conservative: grow if any structure overflows, shrink only if all structures fit.
2. Overflow is tested at `window_period` granularity, not every cycle.
3. The paper samples occupancy every *k* cycles. We use the controller's exact per-cycle mean.
4. The paper's threshold values were tuned for its 4-way SimpleScalar machine. `up_frac` is ours.
5. The full cycles of the first period after a downsize are not counted as overflow. In our mechanism, the entries above the new cap gate dispatch while they drain. In the paper they do not, because a partition is switched off only once it is empty. Without this rule, every shrink could be undone at once.

---

## B3 `mlp`: Kora, Yamaguchi, Ando, MICRO-46 2013

**Paper.**

- The core normally runs with a small window ("ILP mode").
- When last-level-cache misses occur and memory-level parallelism can be exploited, the window is enlarged level by level ("MLP mode").
- When the memory-intensive phase ends, the window returns to the ILP level.
- A larger level is useful only if it exposes more independent misses.

**Here.** The rules are applied once per period, in this order:

- **Memory-intensive period.** A period counts as memory-intensive if it has ≥ `miss_min` long-latency misses (`l2Misses`, or `l1dMisses` with `miss_level=1`).
- **Verification.** After every enlargement, the next period checks whether the MLP (`WindowSample::mlp`, the mean number of outstanding L1D misses over the cycles that have one) rose by at least `gain` (relative).
  - If it did not, the window reverts to the previous level. It is then capped at that level for a back-off time, which starts at `backoff` periods and doubles after each failed attempt up to `backoff_max`. The back-off resets after a successful enlargement.
- **Growth.** A memory-intensive period with `mlp ≥ mlp_thr` grows the window by one level, subject to the back-off cap.
- **No MLP.** A memory-intensive period with `mlp < mlp_thr` (isolated or dependent misses) shrinks the window by one level, down to `ilp`.
- **Return to ILP mode.** After `shrink_delay` consecutive periods without misses, the window goes back to `ilp`.

| key | default | meaning |
|-----|---------|---------|
| `miss_level` | 2 | 2: L2 misses are long-latency; 1: L1D misses (use when no L2 is attached as `window_l2`) |
| `miss_min` | 1 | long-latency misses per period that mark a memory-intensive period |
| `mlp_thr` | 1.5 | MLP above which the misses are considered parallel |
| `gain` | 0.1 | relative MLP gain needed to keep an enlarged level |
| `ilp` | 0 | ILP-mode configuration index |
| `shrink_delay` | 2 | miss-free periods before returning to ILP mode |
| `backoff`, `backoff_max` | 8, 128 | back-off (periods) after an unproductive enlargement |

**Deviations.**

1. The paper reacts to individual misses, cycle by cycle. We decide once per period from counts and the mean MLP.
2. The paper checks for misses in the newly added window portion. We use the measured change in MLP after the enlargement as a proxy, because the sample has no per-ROB-position miss information.
3. MLP is measured on L1D MSHRs, which is what the controller provides. It is not measured on LLC misses.
4. In the paper, the ILP-mode benefit includes a shorter wakeup/select latency for the small window. That effect is not modeled here: smaller configurations save energy but never get a faster clock. Keep this in mind when comparing IPC.
5. The policy starts at `window_initial`, not necessarily in ILP mode.

---

## B4 `bbv`: Sherwood, Sair, Calder, ISCA 2003

**Phase tracker (paper hardware).**

- **Accumulator.** It has `buckets` counters. Each committed control instruction (`onBranchCommit`) adds the length of the basic block it ends to the counter chosen by a hash of its PC.
- **End of interval.** At the end of each interval, the vector is normalized and quantized to `sig_bits` bits per counter; the result is the signature.
- **Matching.** The signature is compared by Manhattan distance with each entry of the past-footprint table (`max_phases` entries, LRU replacement).
  - If the closest entry is nearer than `thr`, the interval joins that phase.
  - Otherwise a new phase id is allocated.
- **Distance scale.** `thr` uses the scale of normalized vectors, whose maximum distance is 2. The default of 0.25 is 12.5% of that maximum.

**Next-phase predictor.**

- It is a run-length-encoded Markov table, indexed by (last phase, min(run length, `run_max`)).
- Each entry holds a predicted next phase and a saturating confidence counter (`conf_max`). An entry is replaced only when its confidence is 0.
- If there is no entry, the predictor falls back to last value (same phase).

**Per-phase best configuration, learned online.**

- **Exploration.** For each phase, every configuration is tried for `explore` intervals, largest first.
- **Exploitation.** After that, the policy uses the smallest configuration whose mean IPC is within `tol` of the phase's best mean IPC. This is energy-aware.
- **Updates.** The mean IPC of the configuration in effect keeps updating at every interval, so learning never stops. An interval in which the configuration changed is not credited.
- **What is applied.** At each interval boundary, the policy applies the configuration of the **predicted** next phase.
- **Dump.** At exit, `bbv_phases.csv` is written to the outdir. It lists each phase's visits, learned configuration and per-configuration IPC, plus a header with the interval count and prediction accuracy.

| key | default | meaning |
|-----|---------|---------|
| `interval_insts` | 100000 | interval length in committed instructions (checked at period boundaries) |
| `buckets` | 32 | accumulator counters (paper: 32) |
| `sig_bits` | 6 | bits per signature bucket |
| `thr` | 0.25 | phase-match distance threshold |
| `max_phases` | 32 | past-footprint table entries |
| `run_max` | 15 | run-length cap in the Markov index |
| `conf_max` | 3 | Markov confidence saturation |
| `explore` | 1 | intervals per configuration during exploration |
| `tol` | 0.02 | IPC tolerance when picking the smallest near-best configuration |

**Deviations.**

1. The interval is 100k instructions instead of 10M, so that phases appear within gem5-sized runs. It is tunable.
2. Intervals end at the first `window_period` boundary after `interval_insts`. Their lengths therefore vary slightly, and the signature is normalized to compensate.
3. The per-phase configuration search is our addition. The paper evaluates phase-based adaptation of other structures (for example caches and widths) with an analogous explore/exploit scheme. The objective here is IPC within `tol`, then the smallest window.
4. A phase evicted from the table forgets what it learned.

---

## B5 `lut`: learned counter-based LUT (Dubach et al. style, our pipeline)

- The policy loads `window_lut_file` once, through `window_lut.hh`, using the format in [interfaces.md §6](../../interfaces.md#6-b5-lut-file-format-window_lut_file). The file is produced by [`sim/baselines/lut/export_lookup_table.py`](../../../sim/baselines/lut/export_lookup_table.py) `--runtime-lut`.
- **One lookup per period.**
  - Each feature is taken from the period's `WindowSample`.
  - It is binned with `numpy.searchsorted(edges, x, side="left")` semantics: the bin is the number of edges strictly below x, so a value equal to an edge falls in the lower bin.
  - The cells are indexed row-major, with the first feature varying slowest.
  - The resulting configuration is applied as is, with no hysteresis.
- **Accepted feature names.** These are the `window_trace.csv` column names or their short aliases, matching `whdata.FEATURE_SOURCE`: `ipc`, `rob_occ`/`rob_occ_mean`, `iq_occ`/`iq_occ_mean`, `lq_occ`/`lq_occ_mean`, `sq_occ`/`sq_occ_mean`, `l1d_mpki`, `l2_mpki`, `mlp`, `branch_mpki`, `insts`, `config`.
- **Errors.** Any malformed file, unknown feature or out-of-range configuration index is fatal.
- **Tunables.** This policy reads no `window_args` keys.

**Deviations.** Dubach et al. predict a full configuration from counters with a trained model. Our model is distilled offline into a quantized LUT, which is what a hardware implementation would store. The labels are the oracle-best configuration for each region (problem 1 in [design proposal](../../reference/proposal.md) §2).

---

## Other policies (pointers and deviations)

| Policy | Baseline | Where documented |
|--------|----------|------------------|
| `static` | B0 | fixed `window_initial`; hints ignored, regions still recorded |
| `hint` | WinHint, B1 (`oracle_hinted` binaries), B6 (`structs=iq`), B7 (`pgo` binaries) | [gem5 patches](patches.md) |
| `hybrid` | WinHint+HW | [`hybrid_policy.cc`](../../../sim/gem5/src/cpu/o3/window/hybrid_policy.cc) header, [gem5 patches](patches.md) |
| `ltp` | B9 (Sembrant et al., MICRO-48 2015) | [Long-Term Parking](../baselines/ltp.md) |

**Hint decoding (all hint-driven policies).** The design proposal's Phase D planned to
decode the hint in `decoder.isa`. Stock gem5 v25.1 already decodes
`ori x0, x0, IMM` (tags 21/23, not the Zicbop tags 0/1/3) as the no-op
`ori_hint`, so the decoder is left untouched. The controller recognises the
hint from the committed instruction word (non-speculative), on
`RISCV_winhint` only. A hint therefore takes effect when it commits, not when
it is decoded: the instructions already in flight behind it were dispatched
under the previous configuration. With hints placed outside hot loops, this
delay is at most one window's worth of instructions.

**B6 (Jones et al., HPCA 2005).** The paper resizes the IQ only, in banks,
and turns off empty banks. Here `structs=iq` caps only the IQ, using the
same cap mechanism as every other policy (dispatch is gated until the
occupancy is under the new size); ROB, LQ and SQ stay at the largest
configuration. `structs=all` is the extension to ROB/LSQ.

**B1/B7.** No run-time logic of their own: they are `hint` runs of binaries
whose `setwin` values come from the oracle sweep (B1) or a profile of the
`small` input (B7).

**B9 (LTP).** The ROB stays at the largest configuration and the IQ/LQ/SQ
follow `window_initial` (`structs`, default `iq+lq+sq`), matching the
paper's premise of a large ROB with small scheduling structures. Physical
registers are allocated at rename, not at release (see [docs/guide/baselines/ltp.md](../baselines/ltp.md)).
The decision rules are in [`ltp/ltp_core.hh`](../../../sim/gem5/src/cpu/o3/window/ltp/ltp_core.hh) and unit-tested on the host
(`make -C sim/gem5/src/cpu/o3/window/ltp/tests`).

Each policy's deviations are also summarized, per baseline, in
[Deviations](../../deviations.md).

## Next

- [Long-Term Parking](../baselines/ltp.md): the B9 policy and its pipeline changes.
- [gem5 patches](patches.md): the `hint` and `hybrid` policies and the
  resize mechanism in existing gem5 files.
- [Simulator fidelity](../fidelity.md): the check each policy passes
  against its paper's trend.
