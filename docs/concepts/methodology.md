# Evaluation methodology

This page explains how WinHint's claims are tested: which experiment addresses each claim,
on which platforms and machines, with which workloads, against which baselines, how the
baselines are made fair, and what is measured. It is the bridge between the concepts
([Background](background.md), [How WinHint works](../guide/architecture.md)) and the
runbook ([Running experiments](../guide/usage.md)). Each section links to the page that owns
the details.

**Before you read:** [Background](background.md) introduces the window, MLP and the two
mechanisms.

No results are shipped with the repository. This page describes what is measured and how;
the numbers come from running the pipeline.

## The claims and the experiments

The design proposal ([§1](../reference/proposal.md#1-title-short-name-and-thesis)) states
four claims about compiler-emitted window hints. Each one maps to a part of the evaluation:

| Claim | Tested by |
|-------|-----------|
| (a) matches or beats reactive hardware | gem5 matrix: WinHint against B2–B5 and B9 (IPC, ED²P), and against the B1 oracle; on silicon, against R0–R5 (EDP) |
| (b) needs no training data | WinHint is compiled once, with no profile, and evaluated on the `large` input; B5 (trained LUT) and B7 (profile of the `small` input) need data |
| (c) switches exactly at region boundaries | switch frequency and hint overhead figures; region-aligned statistics (`region_stats.csv`) |
| (d) carries over to a new core | portability study: the same kernels on three machines, WinHint recompiled per machine versus B5 trained on one machine (`lut_xfer`) |

Two further experiments support the claims:

- **Model accuracy.** The predicted window W\* of each region is compared with the window
  the oracle found best, by rank correlation (figure `wstar_vs_oracle.pdf`).
- **Sensitivity.** WinHint and selected baselines are rerun with other L2 sizes and memory
  latencies (figure `sensitivity.pdf`).

The list of figures and the steps that produce them is in
[Running experiments, Step 10](../guide/usage.md#step-10-figures).

## Two platforms, and why both

- **gem5 (RISC-V O3, syscall-emulation mode).** The only place where the whole window
  (ROB, IQ, LQ, SQ) can be resized: no real CPU exposes that control. Simulation is
  deterministic, every baseline shares the same resize mechanism, and every internal counter
  is visible. It is also slow, so long inputs are sampled (below), and it is a model of a core,
  not a core. Details: [gem5 model](../guide/gem5/index.md).
- **Intel hybrid silicon (Core 5 120U).** The window cannot be resized, but a thread can move
  between P-cores (large window) and E-cores (small window). This tests the same hints on a
  real machine, with real energy (RAPL), real migration costs and real hint overhead, against
  production schedulers. Details: [Real hardware](../guide/hardware/index.md).

The gem5 study answers "is the static model right about the window?"; the silicon study
answers "does it still pay when the mechanism is coarse and the costs are real?".

## Simulated machines

Three machine descriptions in `sim/machines/*.json` are read by both gem5 and the
compiler's cost model, so the compiler and the simulator always agree on the target. They
differ in every parameter the model uses:

| | `riscv_ooo_small` | `riscv_ooo` (default) | `riscv_ooo_big` |
|---|---|---|---|
| Width | 2 | 4 | 6 |
| ROB configurations | 32–128 | 64–256 | 96–384 |
| L1D (MSHRs) / L2 | 16 kB (8) / 256 kB | 32 kB (16) / 1 MB | 48 kB (24) / 2 MB |
| Memory latency (model estimate) | 190 cycles | 200 cycles | 280 cycles |

Each machine has a four-entry [window table](../reference/glossary.md#window-table) that
scales ROB, IQ, LQ and SQ together. `riscv_ooo` is the reference machine: B5's LUT is trained
on it and transferred to the others. The full table is in
[gem5 model, Machine descriptions](../guide/gem5/index.md#machine-descriptions).

The real-hardware platform is a single machine: CPUs 0–3 are P-cores (two cores with SMT),
CPUs 4–11 are E-cores. Its micro-architectural sizes are documented and checked by probes
([Microarchitecture parameters](../guide/hardware/uarch-params.md)).

## Workloads and inputs

Fifteen deterministic C11 inference kernels: encoders, decoders, CNN/MLP contrast models,
image generators, and a regression control group with no transformer structure. Each phase
is a separate function, so each phase is its own region. See [Workloads](../guide/workloads.md).

Every kernel takes one of two inputs with the **same model shapes**:

- **`small`** shrinks only the input (sequence, prompt, batch or image). A run is about
  40–550 M RISC-V instructions and is simulated in full. It is the **training** input: the
  oracle sweep that labels B5's data, B7's profile, and the equal-effort tuning all use it.
- **`large`** is the **evaluation** input (0.2–13.6 G instructions). It is too long to
  simulate in full, so gem5 runs use region-aligned sampling
  ([gem5 model, Run lengths](../guide/gem5/index.md#run-lengths)). The figures plot only
  `large` runs.

Keeping the shapes fixed means both inputs execute the same code regions, so a choice learned
on `small` can be applied to `large`. Testing on an input other than the training input is
what keeps B5 and B7 from being graded on their own training data.

## Baselines

The baselines fall into five families. The full table with citations is in
[Baselines](../guide/baselines/index.md); the variant each one runs as is in
[Running experiments](../guide/usage.md#baselines).

| Family | IDs | Role |
|--------|-----|------|
| Static and oracle | B0, B1 | B0 fixes one window for the whole run (four runs, one per configuration). B1 picks the best configuration per region from exhaustive per-region simulation: the reference WinHint should approach |
| Reactive hardware | B2, B3, B4, B9 | resize from run-time counters: occupancy (B2), MLP (B3), basic-block-vector phase prediction (B4), and Long-Term Parking (B9), which keeps the window but parks non-urgent instructions |
| Learned | B5 | a model trained offline on counter traces, exported as a lookup table (LUT) that gem5 consults at run time |
| Compiler | B6, B7, B8 | compiler-directed IQ resizing (B6), profile-guided per-region configuration (B7), and Clairvoyance (B8), which changes the code instead of the window and is also combined with WinHint |
| OS and scheduler | R0–R5 | on silicon: pinning, the stock Linux scheduler, sched_ext, intel-lpmd, a PIE-style reactive migrator, and Sondag & Rajan's phase-based tuning |

WinHint itself runs with the `hint` policy, and a hybrid variant (WinHint+HW) lets the
hardware move the window below the hinted ceiling.

B1 is not a strict upper bound: it chooses each region independently and ignores switch and
drain costs, so WinHint can in principle beat it by a small margin
([Deviations §3](../deviations.md#3-b1-oracle)).

### Why most baselines are reimplemented

The rule ([proposal §4](../reference/proposal.md#4-baselines)) is: use the authors' code when
a public artifact exists, otherwise reimplement the paper inside this infrastructure and
document every deviation. The artifact search (September 2026) found **no public artifact
for any hardware window-resizing paper**. Only Clairvoyance (B8), PMCTrack (used by R4) and
the production schedulers (R2 sched_ext, R3 intel-lpmd) are public, and they are reused.

Reimplementing has an advantage: every gem5 baseline is a mode of one parameter,
`window_policy`, over the same resize mechanism and the same window table. Only the decision
policy differs, so the comparison isolates the decision. See
[Window policies](../guide/gem5/policies.md).

## Equal-effort tuning

A reimplemented baseline with poorly chosen parameters would make WinHint look better than
it is ([proposal §8](../reference/proposal.md#8-main-risks)). So every tunable method gets
the same tuning budget as WinHint:

- each method (B2, B3, B4, B5, B8, B8+WinHint, B9, WinHint+HW, and WinHint's own compiler
  knobs) evaluates at most the same number of parameter points (16 by default), always
  including its published default;
- every point runs on the same kernels, machines and the `small` input, through the same
  runner;
- per machine, the point with the best geometric-mean ED²P (relative to the default) wins;
  ties keep the default.

The final campaign reads the chosen parameters. How to run it:
[Running experiments, Step 6](../guide/usage.md#step-6-baseline-tuning); the method is
documented in [`tune_baselines.py`](../api/python/sim/baselines/tune/tune_baselines.md).

## Fidelity checks before use

A reimplementation could also be wrong. Before a reimplemented baseline is used, it must
reproduce the **qualitative trend** its paper reports, on a microbenchmark built to show the
condition the paper targets ([proposal §7](../reference/proposal.md#7-verification-and-success-criteria)).
For example, B3 must enlarge the window and gain on independent misses, and must not on a
pointer chase.

Each check has preconditions (the microbenchmark really shows the condition on this machine)
and trend predicates (the paper's claim), with thresholds chosen by this project. The verdict
is `PASS`, `FAIL`, `INVALID` (the precondition failed: fix the microbenchmark) or
`INCOMPLETE`.

- gem5 baselines B2–B7 and B9: [Simulator fidelity](../guide/fidelity.md).
- Real-hardware R4 and R5, plus probes of the P/E micro-architecture:
  [Real hardware](../guide/hardware/index.md) (`hw/fidelity.py`).

B1 is the reference, not a reimplementation, and B8 reuses the authors' artifact, so neither
has a trend check.

## Deviations

A **deviation** is any place where a baseline differs from its paper, or WinHint differs from
the proposal: a structure resized in whole-window steps instead of independently, a counter
measured differently, a tool substituted. Each one is recorded with what the paper says, what
the code does, why, and its expected impact on the comparison. They are recorded so that a
reader can judge whether a baseline was weakened, and so that the artifact can be audited.

The consolidated list is [Deviations](../deviations.md); the detailed descriptions live next
to each implementation.

## Metrics

| Metric | Meaning | Where |
|--------|---------|-------|
| IPC | instructions per cycle (performance) | gem5 `stats.txt` |
| Energy | core and cache energy of a run (gem5), or RAPL package/core energy (silicon) | below |
| EDP | energy × delay | both platforms |
| ED²P | energy × delay²; weights performance more than EDP | gem5 |
| Switch frequency | window switches per million instructions | gem5 |
| Hint overhead | hint density and static count, code size, and the run time of a hinted binary whose hints are ignored (`winhint_nop` against the largest static window; NOP-hint runs on silicon) | both |

On silicon every configuration is measured with a fixed frequency policy, repeated (at least
10 times per the proposal), reported as mean with a 95 % confidence interval and as a ratio
to the stock scheduler R1, with SMT on and off.

### How energy is estimated in gem5

gem5 itself does not report energy. `sim/estimate_energy.py` splits each run's time by its
*residency* in each window configuration and sums the energy of each configuration:

- **McPAT** when it is built: one power model per window configuration, built from the gem5
  statistics and the machine JSON;
- otherwise an **analytic proxy**: static power that scales with the ROB size, plus fixed
  energies per instruction and per cache miss. Its absolute values mean nothing; relative
  comparisons on one machine do.

Either way, energy is compared across methods on the same machine, never as absolute numbers
([Deviations §1.8](../deviations.md#18-energy-mcpat-per-configuration-residency-weighted-or-a-proxy),
[API](../api/python/sim/estimate_energy.md)).

## Correctness

Hints must not change what a program computes. Every kernel prints a checksum and an FNV-1a
hash of its output, and every hinted variant must print exactly the same line as the `plain`
build under QEMU, on unmodified gem5 (`RISCV_clean`) and natively on x86. A second check
confirms the mechanism: forcing `setwin` values must change the occupancy caps seen in gem5's
statistics. See [Testing and verification](../guide/testing.md).

## What counts as success

The proposal fixes the targets before any experiment
([§7](../reference/proposal.md#7-verification-and-success-criteria)):

- **Model:** Spearman ρ of about 0.8 or more between the predicted W\* and the oracle-best
  window per region.
- **gem5 end-to-end:** WinHint within about 5 % of the oracle (B1) IPC; ED²P better than
  B0-large, B2, B3, B4, B5 and B9, better than B6 (IQ only), and competitive with B7 on an
  unseen input.
- **Real hardware:** EDP better than R0-P, R1 and R4 on memory-bound layers, with run-time
  overhead within noise.
- **Portability:** on unseen core configurations, B5's drop without retraining is compared
  with WinHint's after recompilation only.

Where a criterion is not met, it is reported as such.

## Next

- [Installation](../getting-started/installation.md), then
  [Running experiments](../guide/usage.md) to reproduce the evaluation.
- [Baselines](../guide/baselines/index.md) for each method in detail.
- [Deviations](../deviations.md) for every departure from the papers.
