# Baseline fidelity

Most baselines in this study are reimplementations of papers that ship no code, so a weak
reimplementation could make WinHint look better than it is. The proposal's guard against that
([design proposal](../reference/proposal.md) §7) is a fidelity check:

> "Each reimplemented baseline reproduces the *qualitative* trend reported in its
> paper on a microbenchmark before it is used."

This page defines how that check is run and graded in gem5. It is one of the evaluation's
validity checks ([Evaluation methodology](../concepts/methodology.md)); the real-hardware
counterpart for R4/R5 is in [docs/guide/hardware/index.md](hardware/index.md).

## What it runs

[`fidelity.py`](../../sim/fidelity/fidelity.py) runs, for each gem5-side baseline (B2, B3, B4, B5, B6, B7, B9),
the baseline and its static reference windows on the microbenchmarks in
`benchmarks/micro/`. It then evaluates a machine-checkable predicate and writes
`results/fidelity/<baseline>.json` (and `summary.json`). A machine other than `riscv_ooo`
(`--machine sim/machines/<m>.json`) writes to `results/fidelity/<machine>/`.

```bash
eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate winhint

# build + qemu: every variant == plain, both inputs
make -C benchmarks micro-check

# 30 gem5 runs + derive steps
python sim/fidelity/fidelity.py --dry-run

# the study (resumable; gem5 under the heavy lock)
python sim/fidelity/fidelity.py

# restrict to some baselines
python sim/fidelity/fidelity.py --baselines B3 B9

# re-grade existing runs without simulating
python sim/fidelity/fidelity.py --evaluate-only

# harness tests (fake gem5)
python -m pytest sim/fidelity/tests
```

In the runbook this is [Baseline fidelity](usage.md#step-5-baseline-fidelity).

## Verdicts

Every check is either a **precondition** or a **trend** check.

- A **precondition** says that the microbenchmark really shows the condition the paper targets on this machine. For example, "the gather benchmark is faster with the large window".
- A **trend** check is the paper's qualitative claim.

| status | meaning |
|---|---|
| `PASS` | all preconditions and all trend checks hold |
| `FAIL` | the preconditions hold, but a trend check does not: the reimplementation does not reproduce the paper's trend |
| `INVALID` | a precondition fails. The microbenchmark does not exhibit the condition, so the test says nothing about the baseline. Fix the microbenchmark, not the baseline |
| `INCOMPLETE` | some needed runs are missing or failed. The checks that can be computed are still reported |

Each JSON file holds:

- the paper and the trend;
- the thresholds;
- every check (`value`, `op`, `threshold`, `ok`);
- the metrics of every run it used: IPC, configuration residency, mean allocated
  ROB/IQ/LQ/SQ (`*_cap`, `*_cap_frac`) and mean occupancies.

## Definitions

- **IPC ratio**: whole-run IPC of run A divided by that of run B. Every run of
  the same benchmark executes the same instructions, apart from the hint NOPs of
  hinted builds.
- **Mean allocated size (`cap`)**: the residency-weighted size of each structure,
  as a fraction of its largest size. Residency comes from
  `system.cpu.window.cyclesInConfig::<i>`.
  - A structure that a policy does not resize counts at its largest size. This
    applies to the ROB under `hint structs=iq` (B6), and to the ROB under `ltp`
    and under the `smalliq` reference (B9).
  - This is the paper-independent stand-in for "power saved by turning
    partitions off" (B2, B6). Energy proper is left to `sim/estimate_energy.py`.
- **Gap recovered**: `(IPC_policy − IPC_small) / (IPC_large − IPC_small)`.
- **Per-region oracle** (micro_phased, B4, B5, B7):
  - It comes from the `region_stats.csv` of the four static runs of the `oracle`
    build (`region(id)` markers).
  - A region's **best** configuration is the smallest one within 2 % of its best
    IPC. This is the same tie rule as `oracle_sweep.analyze(tie_tol=0.02)` and the
    same objective as the B4 policy (`tol=0.02`).
  - A configuration is **near-best** for a region if it reaches ≥ 97 % of the
    region's best static IPC.
  - The **phase regions** are those with ≥ 5 % of the cycles.
- **Near-best share**: the cycle-weighted share of `window_trace.csv` periods
  whose (region, configuration) is near-best. Periods outside a known region are
  ignored.

## Microbenchmarks (`benchmarks/micro/`, built by `benchmarks/micro/micro.mk`)

All of them are integer-only and deterministic, and print one checksum line. `argv[1]` is
`small|large`. Instruction counts are QEMU counts at -O2:

| kernel | condition | small / large |
|---|---|---|
| `micro_gather` | independent L2 misses: 8 MiB table, one load per line, per-page permutation; 3-op address slice, 6-op dependent chain per load (≈17 insts/load) | 3.6 M / 8.1 M |
| `micro_chase` | dependent misses: random single-cycle pointer chase over 8 MiB, the same per-step chain | 5.6 M / 6.5 M |
| `micro_compute` | cache-resident high-ILP integer matrix-vector product, no misses | 1.4 M / 4.0 M |
| `micro_lowilp` | serial 64-bit LCG + ~50 % mispredicted data-dependent branches, no memory traffic | 2.5 M / 7.6 M |
| `micro_phased` | rounds of `gather → compute → lowilp`, each phase a separate function (its own region id), ≈0.5 M insts per visit; large = 6 rounds, small = 3 shorter rounds with different data | 4.8 M / 10.8 M |

Variants (`make -C benchmarks micro`): `plain`, `oracle`, `winhint`, `jones`, `jones_full`
in `$WINHINT_BUILD/benchmarks/micro/riscv/<variant>/`. `pgo` is built by
`fidelity.py` (`make -C benchmarks micro-pgo`) from the small-input
per-region map. `make -C benchmarks micro-check` runs every variant and `plain` on both
inputs under qemu-riscv64 and requires bit-identical output.

## Per baseline: paper trend, predicate, rationale

The thresholds are in `THRESHOLDS` in `fidelity.py`.

- They are **ours**. The papers report results on SPEC-era machines and simulators,
  not on these microbenchmarks.
- They encode the *direction* and a clear *margin* of each claimed effect.
- Margins are wide enough to be robust to gem5 noise. Simulation is deterministic,
  but small changes to the code layout move IPC by about 1 %.

Common precondition: `micro_gather` IPC(static large) / IPC(static small) ≥ 1.20. MLP must
actually be available to a larger window on this machine.

### B2 occupancy — Ponomarev, Kucuk, Ghose, MICRO-34 2001

**Trend.**

- The IQ, ROB and LSQ are often under-used, so occupancy-driven downsizing
  (partitions switched off) saves a large share of their power with a small
  performance loss.
- Resources grow back when dispatch stalls on a full structure.

**Runs.**

- `micro_lowilp`: `static c3`, plus `occupancy` starting at c3.
- `micro_gather`: `static c0`, `static c3`, plus `occupancy` starting at c0.

**Predicate.**

| kind | check | threshold |
|---|---|---|
| pre | lowilp mean ROB occupancy / 256 at static c3 | ≤ 0.50 (the window really is under-used) |
| pre | gather IPC c3 / c0 | ≥ 1.20 |
| trend | lowilp mean allocated ROB/IQ/LQ/SQ (largest fraction of the four) | ≤ 0.60 (≥ 40 % fewer active entries on average, in every structure) |
| trend | lowilp IPC(occupancy) / IPC(c3) | ≥ 0.95 (small loss) |
| trend | gather, starting small: mean allocated window | ≥ 0.60 (it grows on dispatch stalls) |
| trend | gather, starting small: gap recovered | ≥ 0.50 |

**Rationale.**

- The savings case is low-occupancy code (squash-bound), where the paper's down rule
  (`size − avg_occ ≥ partition`) fires. A code whose window is full, such as
  `micro_compute` (the ROB fills behind its dependence chains), would correctly
  *not* shrink, so it is not used as the savings case.
- 5 % is the "small loss" bound: the paper reports a small average slowdown.

### B3 mlp — Kora, Yamaguchi, Ando, MICRO-46 2013

**Trend.**

- The window stays small (ILP mode) unless last-level misses with exploitable MLP occur.
- It is then enlarged, which gives a large speedup on MLP-rich code.
- Isolated or dependent misses (pointer chasing) and compute code get no
  enlargement and no gain.

**Runs.** On gather, chase and compute: `static c0`, `static c3`, and `mlp` starting at c0
(`ilp=0`).

**Predicate.**

| kind | check | threshold |
|---|---|---|
| pre | gather IPC c3 / c0 | ≥ 1.20 |
| pre | chase IPC c3 / c0 | ≤ 1.05 (chasing really has no MLP) |
| trend | gather IPC(mlp) / IPC(c0) | ≥ 1.10 |
| trend | gather gap recovered | ≥ 0.50 |
| trend | chase IPC(mlp) / IPC(c0) | ≤ 1.03 and ≥ 0.97 |
| trend | chase residency in c0 (ILP mode) | ≥ 0.60 |
| trend | compute residency in c0 | ≥ 0.90 |

**Rationale.** The chase residency bound is looser than the compute one because the policy
legitimately probes one level up and backs off (verification plus back-off,
[docs/guide/gem5/policies.md](gem5/policies.md)). Compute has no misses at all, so it should essentially never leave ILP
mode.

### B4 bbv — Sherwood, Sair, Calder, ISCA 2003

**Trend.**

- BBV signatures identify the program's recurring phases.
- The run-length Markov predictor predicts the next phase with high accuracy.
- Once learned (after warm-up), each recurring phase runs with its best
  configuration.

**Runs.** `micro_phased` large: `static c0..c3` (`oracle` build) and `bbv` starting at c3.

**Predicate.**

| kind | check | threshold |
|---|---|---|
| pre | distinct oracle-best configs among the phase regions | ≥ 2 (phases matter) |
| trend | share of intervals in the 3 most visited BBV phases | ≥ 0.80 (3 code phases ⇒ 3 dominant BBV phases; transition intervals may add a few small ones) |
| trend | phase ids allocated | ≤ 12 |
| trend | next-phase prediction accuracy (`bbv_phases.csv`) | ≥ 0.80 |
| trend | after 25 % of the instructions: near-best share | ≥ 0.75 |
| trend | IPC(bbv) / best static IPC | ≥ 0.93 |
| trend | mean allocated window | ≤ 0.85 |

**Rationale.**

- The first round (≈17 % of the instructions) contains the exploration
  (4 configurations × 1 interval per phase). Skipping 25 % leaves only learned
  behaviour.
- The 0.93 IPC bound allows for the exploration and for the boundary intervals
  that mix two phases.
- The cap bound shows that the policy does use the smaller windows in the
  compute and lowilp phases.

### B5 lut — Dubach, Jones, Bonilla, TACO 2013 (our LUT pipeline)

**Trend.** A model trained offline on hardware counters predicts a good configuration for
each window of a phased program.

**Runs.**

- `micro_phased` **small**: `static c0..c3`. These are the training traces, labelled
  with the small-input per-region oracle.
- `sim/baselines/lut`: `label_phases.label_trace`, then
  `train_phase_classifier.py --model tree --split none`, then
  `export_lookup_table.py --runtime-lut`. Output:
  `results/fidelity/b5/<machine>/lut.txt`.
- `micro_phased` **large**: `static c0..c3`, and `lut` with that LUT.

**Predicate.**

| kind | check | threshold |
|---|---|---|
| pre | distinct oracle-best configs among phase regions | ≥ 2 |
| trend | near-best share over the whole lut run | ≥ 0.75 |
| trend | IPC(lut) / best static IPC | ≥ 0.95 |
| trend | mean allocated window | ≤ 0.85 |

**Rationale.** Testing on the other input avoids grading the model on its training data.
The LUT has no warm-up, hence the whole run and a 0.95 IPC bound. The exact-match share is
also reported (`exact_frac`), but it is not graded: two configurations within 3 % are
equally right.

### B6 jones — Jones, O'Boyle, Abella, González, HPCA 2005

**Trend.** Compiler-directed IQ resizing, based on the loop DAG's issue-queue demand,
reduces the IQ size (energy) with negligible performance loss.

**Runs.** On compute, lowilp and gather: `static c3`, and `hint structs=iq` on the `jones`
build, starting at c3. This is the as-published IQ-only form.

**Predicate.**

| kind | check | threshold |
|---|---|---|
| trend | IPC(jones) / IPC(c3), each of compute, lowilp, gather | ≥ 0.97 |
| trend | compute and lowilp mean allocated IQ | ≤ 0.75 |

**Rationale.**

- "Negligible loss" means within 3 %. Gather is included because there the IQ must
  *not* be starved.
- The IQ must shrink by ≥ 25 % on average where the demand is low.

### B7 pgo — Huang, Renau, Torrellas, ISCA 2003; Lau, Perelman, Calder, CGO 2006

**Trend.**

- Configurations chosen per code region (position) from a profiling run carry
  over to another input.
- They come close to the per-region oracle while saving resources.

**Runs.**

- `micro_phased` small `static c0..c3` gives the per-region map,
  `results/fidelity/oracle/<machine>/small/micro_phased.json`.
- `make -C benchmarks micro-pgo` builds the `pgo` binary from that map.
- On large: `static c0..c3`, plus `hint` with the `pgo` binary.

**Predicate.**

| kind | check | threshold |
|---|---|---|
| pre | distinct oracle-best configs among phase regions | ≥ 2 |
| trend | phase regions whose small-input config is not near-best on large | ≤ 0 |
| trend | IPC(pgo) / per-region-oracle IPC on large | ≥ 0.95 |
| trend | IPC(pgo) / IPC(static c3) | ≥ 0.97 |
| trend | mean allocated window | ≤ 0.85 |

**Rationale.**

- The per-region oracle IPC is the sum of instructions over the sum of per-region
  best cycles. It ignores the switch cost, which explains the 5 % slack.
- The transfer check is the positional-adaptation claim itself.

### B9 ltp — Sembrant et al., MICRO-48 2015

**Trend.**

- With small IQ/LSQ, Long-Term Parking keeps non-urgent instructions out of them.
  This recovers much of the performance of a large IQ/LSQ on MLP-rich code.
- Code without long-latency loads is not hurt.

**Runs.**

- On gather and compute: `static c3`, plus `smalliq`. `smalliq` is `hint
  structs=iq+lq+sq` starting at c0 on the `plain` build, which has no hints: the
  ROB stays at 256 and IQ/LQ/SQ are 32/16/16. These are exactly LTP's resources, but
  without parking.
- `ltp` starting at c0.

**Predicate.**

| kind | check | threshold |
|---|---|---|
| pre | gather IPC(c3) / IPC(smalliq) | ≥ 1.15 (the small IQ/LSQ really hurts) |
| trend | gather gap recovered by LTP | ≥ 0.40 |
| trend | gather IPC(ltp) / IPC(smalliq) | ≥ 1.10 |
| trend | compute IPC(ltp) / IPC(smalliq) | ≥ 0.95 |

**Rationale.**

- "Much of" is read as at least 40 % of the gap. Our LTP cannot also enlarge the
  register file ([docs/guide/baselines/ltp.md](baselines/ltp.md)).
- The equal-resource reference isolates the parking mechanism from the
  ROB size.

## Not covered here

- **B1** (the oracle) is the reference and not a reimplementation of a paper.
- **B8** (Clairvoyance) reuses the authors' artifact, so its fidelity is a matter
  of bitcode compatibility (see [compiler/baselines/clairvoyance](baselines/clairvoyance.md)).
- The real-hardware baselines R4/R5 are handled in `hw/`
  ([docs/guide/hardware/index.md §5.2](hardware/index.md)).

## Next

- [Baselines](baselines/index.md): what each baseline is and where it is implemented.
- [Window policies](gem5/policies.md): the B2–B5 algorithms these checks
  exercise.
- [Deviations](../deviations.md): every departure from the papers, per baseline.
