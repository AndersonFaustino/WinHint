# Deviations

A **deviation** is a place where this repository does something different from what a
baseline's paper describes, or, for WinHint itself, from what the
[design proposal](reference/proposal.md) describes. This page lists every
deviation in one place, grouped by component, so that a reader can judge whether a baseline was
weakened or a result depends on a substitution. It is the appendix that PROPOSAL §7 asks for and
that the paper's artifact appendix points to. The evaluation as a whole, and why the baselines
are reimplemented, is explained in [Evaluation methodology](concepts/methodology.md#deviations).

Before you read: [Evaluation methodology](concepts/methodology.md) introduces the baselines
(B0–B9 in gem5, R0–R5 on real hardware). Terms such as ROB, IQ, LSQ, MLP, MSHR, ED²P,
`setwin` and region are defined in the [Glossary](reference/glossary.md).

## Why deviations exist

- **No artifact to reuse.** None of the hardware window-resizing papers has a public artifact
  (PROPOSAL §4). B2–B7, B9, R4 and R5 are therefore reimplemented from their papers, inside
  this infrastructure: gem5 v25.1 with a RISC-V out-of-order (O3) core, ML inference kernels,
  and one shared resize mechanism. The papers used other simulators (SimpleScalar, Wattch, gem5
  x86 full-system), other ISAs (Alpha, x86) and other workloads, so a faithful port still
  differs in places. Only B8 (Clairvoyance) and the real-hardware schedulers R2/R3 reuse
  published code.
- **Equal resources.** PROPOSAL §4.1 requires every gem5 policy to use the same mechanism and
  the same configuration table, so that only the decision differs. A paper that resized one
  structure independently has to be fitted to that table.
- **Unavailable resources.** Every tool must come from a conda package (no apt, no pip wheels
  outside conda, no toolchain built from source), and the host is a 12-thread, 6 GB laptop with
  an Intel Core 5 120U running Linux 7.0. Some packages do not exist, one kernel module does not
  build, and simulation time limits the input sizes.

## Why they are recorded

A reimplemented baseline can be accused of being weak. Listing each difference, with its reason
and its expected effect, lets a reviewer check that the comparison is fair, and lets the
artifact be audited. PROPOSAL §7 requires the list; the paper cites it.

## How to read an entry

Each entry has an identifier (for example `B3-2`) and four fields:

Paper / Proposal
:   What the original paper (or the design proposal, for WinHint and the shared infrastructure) says.

Here
:   What this repository does.

Why
:   The reason for the difference.

Impact
:   The expected effect on the comparison. This is a **prediction, not a measured result**:
    the page was written on 2026-10-01, before any experiment campaign, and no entry quotes a
    measured number.

Some entries add **Source**, the document next to the code that describes the deviation in
detail, and **Status**, when a later change made the entry obsolete. A field marked
*(reconstructed)* was not written down by the implementer; it is inferred from the code and
the surrounding documents and should be confirmed. The source document and
the code are authoritative: when they disagree with this page, fix this page.

## Summary

| Component | Entries | Gist |
|-----------|--------:|------|
| [WinHint](#winhint) | 9 | analytic proxies for cache, trip counts, libm and placement cost; x86 migration through call mode (2 entries resolved) |
| [B0 and B1](#b0-b1) | 2 | registers not scaled; oracle is the best per-region independent choice |
| [B2 occupancy](#b2) | 5 | whole-window steps, period granularity, retuned thresholds |
| [B3 MLP-aware](#b3) | 5 | per-period decisions, MLP on L1D MSHRs, no faster scheduler |
| [B4 BBV phases](#b4) | 4 | 100 k-instruction intervals, our per-phase search |
| [B5 learned LUT](#b5) | 4 | oracle labels, per-window MLP distilled into a LUT |
| [B6 Jones IQ](#b6) | 6 | RISC-V, `setwin` encoding, IR-level DAG, replicated loop bodies |
| [B7 positional PGO](#b7) | 2 | WinHint's regions as positions |
| [B8 Clairvoyance](#b8) | 5 | artifact built with conda LLVM 3.8, two crash fixes, retargeted to RISC-V |
| [B9 Long-Term Parking](#b9) | 6 | registers at rename, LSQ order kept, extra releases |
| [Real hardware R0–R5](#real-hardware) | 12 | single-job PIE on perf, coarse Sondag sections, lpmd on AC, platform facts |
| [Infrastructure and toolchain](#infrastructure) | 11 | hint seen at commit, whole-window caps, energy proxy, reduced inputs, tool substitutions |

---

## WinHint itself (deviations from PROPOSAL §3) { #winhint }

WinHint is the compiler pass that estimates each loop nest's window demand `W*` and places
`setwin` hints ([How WinHint works](guide/architecture.md), [Compiler](guide/compiler.md)).
Sources: [WindowDemandAnalysis.cpp](../compiler/winhint/WindowDemandAnalysis.cpp),
[HintPlacement.cpp](../compiler/winhint/HintPlacement.cpp),
[Options.cpp](../compiler/winhint/Options.cpp).

### W1 · Critical-path term uses the sustained issue rate { #w1 }

Proposal
:   Originally `W* = min(W_max, max(⌈MLP_target·D_indep⌉, CP·issue_width))`.

Here
:   Default `-winhint-cp-model=ii`: `W_cp = ⌈CP·Rate⌉` with `Rate = min(issue_width, B/II)`,
    where `B` is the body's instruction count and
    `II = max(RecMII, B/issue_width, FpDivOcc/FpDivUnits, IntDivOcc/IntDivUnits, 1)`.
    `-winhint-cp-model=width` gives `CP·issue_width` exactly.

Why
:   `CP·issue_width` overestimates the window of recurrence- or divider-bound loops, which
    cannot issue at full width (Little's law with the real throughput).

Impact
:   Smaller `W*` for recurrence-bound loops (reductions, softmax/layernorm row sums).

Status
:   **Resolved.** PROPOSAL §3 now states this formula (`ceil(CP · rate)`, with
    `CP · issue_width` for issue-bound loops), so this is no longer a deviation. Kept for the
    record.

### W2 · MLP_target derived, fractional, MSHR-bounded { #w2 }

Proposal
:   Originally left `MLP_target` unspecified.

Here
:   `MLP_target = min(N_MSHR(L1D), L_mem·Rate/D_indep)` (fractional);
    `W_mlp = ⌈MLP_target·D_indep⌉`. Unless the MSHRs bind, this equals `⌈L_mem·Rate⌉`: the
    instructions issued during one miss latency. A per-machine override exists
    (`"winhint": {"mlp_target": …}` in the machine JSON).

Why
:   Rounding MLP up to an integer would force a whole `D_indep` into the window when misses are
    rare.

Impact
:   A smaller memory term for loops with rare misses than an integer `MLP_target` would give.

Status
:   **Resolved.** PROPOSAL §3 now defines `MLP_target = min(MSHR_L1D, L_mem · rate / D_indep)`.
    Kept for the record.

### W3 · Cache model: reuse distance by loop level, not a stack-distance simulation { #w3 }

Proposal
:   A footprint/reuse-distance cache model driven by the machine JSON.

Here
:   Accesses with the same base and strides (offsets within 4 lines) form one group. The reuse
    distance of a group is the per-iteration footprint of the innermost enclosing loop along
    which its address is invariant. The serving level is the smallest cache whose
    capacity × 0.75 holds that distance (or, without reuse, the larger of the nest footprint
    and the object size). Misses per iteration: 0 if invariant or L1; `members` for indirect
    accesses and pointer chases; `stride/line` for sub-line strides; otherwise
    `min(members, max(1, span/line))`. Memory latency = L2 hit latency +
    `memory.latency_cycles` (20 + 200 = 220 cycles on `riscv_ooo`). DependenceAnalysis is used
    only to drop loads on memory-carried recurrences (innermost loops); pointer chases and
    those loads add no MLP.

Why
:   *(reconstructed)* A static pass has no address trace; grouping accesses per loop level is the compile-time
    approximation of reuse distance.

Impact
:   No conflict misses, no cross-nest reuse, and no prefetch modelling beyond the
    streaming→L2 rule ([I7](#i7)). Errors show up as `W*` versus oracle disagreement (the
    Spearman ρ criterion of PROPOSAL §7).

### W4 · Trip counts from call-site constants; unknown trips assumed 1000 { #w4 }

<span id="134-trip-counts-from-call-site-constants-unknown-trips-assumed-1000"></span>

Proposal
:   Trip counts from ScalarEvolution; the treatment of unknown bounds is not specified.

Here
:   Symbolic trip counts are bounded by the maximum constant that reaches each argument over
    all call sites (`argumentBounds`). Unknown trips use `-winhint-unknown-trip=1000` and mark
    the loop `conservative`. An unknown extent along a varying dimension makes the footprint
    unbounded.

Why
:   The kernels pass their shapes as arguments; the call sites carry the constants.

Impact
:   Loops whose bounds come only from run-time input are sized from a guess and flagged; an
    unbounded footprint pushes their accesses to memory in the cache model.

### W5 · Library calls summarised by a fixed table { #w5 }

Proposal
:   "Summarizes library calls such as `expf` and `tanhf`."

Here
:   A fixed table of (instruction count, latency) per libm routine (for example `expf` 22/30,
    `tanhf` 34/44) inside the dependence DAG.

Why
:   The pass sees only a call, not the library's code.

Impact
:   The values are estimates, not measurements of the conda glibc; they affect `CP` and `B` of
    softmax and GELU loops.

### W6 · Placement cost model is a proxy { #w6 }

Proposal
:   A dynamic program places hints so that each change's benefit outweighs its switch cost,
    with hysteresis.

Here
:   The DP minimises `cost(n, W*, c) = n / (IPC0·min(1, ROB_c/W*)) · (1 + λ·ROB_c/ROB_max)`
    with `IPC0 = max(1, issue_width/2)` and `λ = 0.15` (`-winhint-energy-weight`). Hysteresis
    is a static margin on the switch cost, `S_eff = S·(1 + 0.25)` (`-winhint-hysteresis`), not
    a run-time mechanism. Switch cost `S = 10 + ROB_max/(2·commit_width)` cycles for gem5 (42 on
    `riscv_ooo`), or `t_mig·f` for P/E migration. Hint cost is 1 cycle (asm) or 8 (call). Loops
    with fewer than `2·W_max` estimated instructions per entry are never hinted.

Why
:   *(reconstructed)* Measured costs are not available at compile time; the proxy needs only machine-JSON
    parameters.

Impact
:   The DP decides where switches pay off under this proxy; the oracle comparison measures how
    good the proxy is.

### W7 · Hint granularity and encoding { #w7 }

Proposal
:   `setwin(W)` "maximum window W, advisory"; `region(id)` markers for the oracle.

Here
:   The `setwin` payload is 6 bits × 8 entries (`W ≤ 504`); `region` ids are 0–63, with 63 as
    the overflow id. Hints go in loop preheaders only (no mid-body hints). The entry state of a
    function is "unknown" and charged the worst case.

Why
:   The payload must fit the 12-bit immediate of `ori x0, x0, imm` next to the tag
    ([Interfaces](interfaces.md)).

Impact
:   Window demands are rounded to multiples of 8 entries; functions with more than 63 regions
    share one region id ([B1-1](#b1-1)).

### W8 · Real hardware: x86 NOP hints are not observed; call mode drives migration { #w8 }

Proposal
:   The x86 twin is a unique multi-byte NOP; `libwinhint` turns hints into P/E migration.

Here
:   The NOP (`nopl DISP32(%rax)`, `DISP32 = 0x5748kppp`) is an architectural no-op that no CPU
    reports; it is used only to measure hint overhead. Migration uses the compiler's call mode
    (`__winhint_setwin(W)` at the same positions with the same values): `W ≥ 192`
    (`WINHINT_THRESHOLD`) → P-cores, smaller → E-cores, `W = 0` → original affinity. There is
    no x86 machine JSON: the x86 call-mode builds use the default `riscv_ooo.json` window table
    and cache model, with the measured migration cost as switch cost. The threshold 192
    corresponds to configuration index ≥ 2 of that table.

Why
:   *(reconstructed)* No CPU reports the NOP, so nothing can act on it; a call runs `libwinhint`.

Impact
:   The P/E decision is the RISC-V model's demand, binarised. An x86 target description
    (Raptor Cove / Gracemont, values in [µarch parameters](guide/hardware/uarch-params.md))
    would be the faithful version.

### W9 · WinHint+HW (hybrid) policy is our design { #w9 }

<span id="139-winhinthw-hybrid-policy-is-our-design"></span>

Proposal
:   "`hybrid` (a hint plus a hardware override)", without a rule.

Here
:   The hint is a ceiling; below it, an MLP/occupancy rule after Kora et al. (B3) and Ponomarev
    et al. (B2): `miss_mpki=1.0`, `mlp_thr=1.5`, `up_frac=0.05`, `down_margin=0.9`
    ([hybrid_policy.cc](../sim/gem5/src/cpu/o3/window/hybrid_policy.cc)).

Why
:   No paper defines a hint-plus-hardware policy.

Impact
:   WinHint+HW results measure this particular combination, not a published one.

---

## B0 and B1 — static windows and the oracle { #b0-b1 }

B0 runs each fixed window configuration; B1, the oracle, picks the best configuration per
region from a sweep and is the reference that WinHint is compared against
([Baselines](guide/baselines/index.md), [gem5 model](guide/gem5/index.md)).

### B0-1 · Registers are not scaled with the window { #b0-1 }

Proposal
:   Small/medium/large windows (for example 64, 128, 192, 256) with IQ, LSQ and registers
    scaled.

Here
:   Four configurations per machine (`static_c0` … `static_c3`); the registers are fixed at the
    never-binding size ([I5](#i5)).

Why
:   The window, not the register file, must be the binding limit (PROPOSAL §2, problem 6).

Impact
:   B0-small is limited only by ROB, IQ and LSQ, as intended.

### B1-1 · Best per-region independent choice, not an exhaustive search { #b1-1 }

<span id="3-b1-oracle"></span>

Proposal
:   The best configuration per region from exhaustive per-region simulation; the upper bound.

Here
:   The `oracle` binary (region markers at top-level loop nests) runs once under each
    *uniform* static configuration. Per region, the configuration with the best IPC (default
    `--primary ipc`) or ED²P is chosen from `region_stats.csv`, and the `oracle_hinted` build
    emits one `setwin` per region.

Why
:   An exhaustive search over per-region *combinations* needs K^R simulations (K
    configurations, R regions).

Impact
:   B1 is the best *per-region independent* choice, not a strict upper bound: it ignores
    switch and drain costs and cross-region cache-state interaction, and regions are top-level
    loop nests (inner-loop changes are invisible). WinHint can in principle beat B1 by a small
    margin; report that honestly if it happens. Region ids ≥ 63 share the overflow marker 63.

Source
:   [oracle_sweep.py](../sim/baselines/oracle/oracle_sweep.py),
    [HintPlacement.cpp](../compiler/winhint/HintPlacement.cpp) (region numbering).

---

## B2 — occupancy-driven resizing (Ponomarev et al., MICRO-34 2001) { #b2 }

B2 shrinks under-used structures and grows them on dispatch stalls, from occupancy counters
(`window_policy=occupancy`, [Window policies](guide/gem5/policies.md)). Source: the B2 section of
[Window policies](guide/gem5/policies.md).

### B2-1 · Whole-window steps { #b2-1 }

Paper
:   IQ, ROB and LSQ are resized independently, in partitions.

Here
:   Whole-window steps ([I2](#i2)); grow if *any* structure overflows, shrink only if *all*
    fit.

Why
:   One configuration table for every policy (PROPOSAL §4.1).

Impact
:   More conservative shrinking than the paper: less energy saved, fewer IPC losses.

### B2-2 · Overflow checked per period { #b2-2 }

Paper
:   Overflow checked every cycle.

Here
:   Checked at `window_period` (1000-cycle) granularity.

Why
:   The controller makes one decision per sampling period for every policy.

Impact
:   Up to one period of reaction lag.

### B2-3 · Exact mean occupancy { #b2-3 }

Paper
:   Occupancy sampled every *k* cycles.

Here
:   Exact per-cycle mean.

Why
:   The controller accumulates occupancy every cycle.

Impact
:   Slightly better information than the paper.

### B2-4 · Thresholds retuned { #b2-4 }

Paper
:   Thresholds tuned for a 4-way SimpleScalar machine.

Here
:   `up_frac=0.05`, `update=2` (≈ 2048 cycles, the paper's update period), `down_factor=1.0`;
    tuned on `small` inputs with the same budget as WinHint. `up_frac` is ours.

Why
:   Different machine and workloads.

Impact
:   Depends on tuning.

### B2-5 · Drain cycles after a downsize not counted as overflow { #b2-5 }

Paper
:   A partition is switched off only once it is empty.

Here
:   The full cycles of the period after a downsize are not counted as overflow.

Why
:   Otherwise the drain after a shrink would undo every shrink.

Impact
:   Prevents an artificial oscillation.

---

## B3 — MLP-aware window resizing (Kora et al., MICRO-46 2013) { #b3 }

B3 enlarges the window when long-latency misses overlap (MLP) and returns to a small "ILP mode"
otherwise; it is the closest hardware competitor (`window_policy=mlp`,
[Window policies](guide/gem5/policies.md)).

### B3-1 · One decision per period { #b3-1 }

Paper
:   Reacts to individual LLC misses, cycle by cycle.

Here
:   One decision per period, from miss counts and the mean MLP.

Why
:   The controller samples per period.

Impact
:   Slower reaction; MLP bursts within a period are lost.

### B3-2 · Verification by MLP gain { #b3-2 }

Paper
:   Checks for misses in the newly added window portion.

Here
:   Uses the change in measured MLP after an enlargement (≥ `gain`, default 0.1 relative),
    with exponential back-off.

Why
:   The sample has no per-ROB-position miss information.

Impact
:   Noisier verification.

### B3-3 · MLP on L1D MSHRs { #b3-3 }

Paper
:   MLP of LLC misses.

Here
:   MLP on L1D MSHRs ([I6](#i6)).

Why
:   The controller reads only the L1D MSHR queue.

Impact
:   L2 hits count as parallel misses; `mlp_thr` is tuned for this definition.

### B3-4 · No faster scheduler in ILP mode { #b3-4 }

Paper
:   The small ILP-mode window gets a faster wake-up/select.

Here
:   Not modelled ([I4](#i4)).

Why
:   gem5 O3 has no size-dependent scheduler latency.

Impact
:   B3's ILP-mode IPC benefit is absent; its energy benefit remains.

### B3-5 · Initial configuration { #b3-5 }

Paper
:   Starts in ILP mode.

Here
:   Starts at `window_initial`.

Why
:   Every policy starts from the same configuration.

Impact
:   A different transient at the start of a run.

---

## B4 — BBV phase tracking and prediction (Sherwood et al., ISCA 2003) { #b4 }

B4 classifies execution intervals into phases by their basic-block vectors, predicts the next
phase and applies a configuration learned per phase (`window_policy=bbv`,
[Window policies](guide/gem5/policies.md)).

### B4-1 · 100 k-instruction intervals { #b4-1 }

Paper
:   10 M-instruction intervals.

Here
:   100 k instructions (`interval_insts`, tunable).

Why
:   Phases must appear within gem5-sized runs.

Impact
:   More, shorter phases; more pressure on the predictor.

### B4-2 · Intervals end at a period boundary { #b4-2 }

Paper
:   Fixed-length intervals.

Here
:   An interval ends at the first period boundary after `interval_insts`; signatures are
    normalised to compensate.

Why
:   The controller sees the policy only at period boundaries.

Impact
:   Interval lengths vary slightly; normalisation removes the effect on signatures.

### B4-3 · Per-phase configuration search is ours { #b4-3 }

Paper
:   Evaluates phase-based adaptation of other structures (caches, widths) with an analogous
    explore/exploit scheme.

Here
:   Our own per-phase explore/exploit search over window configurations: IPC within `tol`
    (0.02), then the smallest window.

Why
:   The paper does not resize the window.

Impact
:   Exploration spends intervals in poor configurations; this is inherent to any per-phase
    configuration learned online.

### B4-4 · Evicted phases forget { #b4-4 }

Paper
:   A table of past phase footprints.

Here
:   The table has 32 entries (`max_phases`, LRU); a phase evicted from it forgets what it
    learned.

Why
:   A finite hardware table.

Impact
:   A phase that returns after eviction is explored again.

---

## B5 — learned counter-based LUT (Dubach et al. style; our earlier prototype, fixed) { #b5 }

B5 maps per-period counters to a window configuration through a lookup table distilled from an
offline-trained model; it represents ML-based adaptation (`window_policy=lut`,
[Window policies](guide/gem5/policies.md)). Sources: the B5 section of
[Window policies](guide/gem5/policies.md) and [sim/baselines/lut/](../sim/baselines/lut)
(`label_phases.py`, `train_phase_classifier.py`, `wh_models.py`, `export_lookup_table.py`,
`build_lut.py`).

### B5-1 · Oracle labels, per-window features, a LUT at run time { #b5-1 }

Original prototype
:   Threshold-rule labels (ATTENTION/FFN/OTHER), a sliding-window tiny Transformer, cumulative
    counters, a fake "miss rate", a compiled-in `phase_lookup.h`.

Here
:   Labels are the oracle-best configuration of each window's region (PROPOSAL §2, problem 1).
    Features are per-window deltas from `window_trace.csv` (problem 3): `ipc`, `rob_occ`,
    `l1d_mpki`, `mlp`. The default model is a small PyTorch MLP (4→32→32→K), with a tree, kNN,
    LUT-native bins and a *per-window* Transformer as alternatives. The model is distilled into
    a quantised 4-D LUT (`--runtime-lut`, [Interfaces](interfaces.md) §6) that gem5 loads at
    run time.

Why
:   The sliding-window Transformer cannot be exported faithfully to a per-window LUT; a hardware
    implementation would store the LUT.

Impact
:   Quantisation loss, reported by `export_lookup_table.py --check`.

### B5-2 · Classifier over window configurations only { #b5-2 }

Paper
:   Dubach et al. predict the full configuration from counters with an offline-trained model
    (soft-max/regression family).

Here
:   A classifier over the K window configurations only.

Why
:   The window table is the only thing any policy may change (PROPOSAL §4.1).

Impact
:   A smaller prediction space than the paper's.

### B5-3 · Trained on oracle labels of the `small` input { #b5-3 }

Paper
:   Trained offline on profiled configurations.

Here
:   Training data is the oracle sweep of the `small` input, evaluated leave-kernels-out. The LUT
    is retrained per machine; `lut_xfer` runs the reference machine's LUT on other machines
    (portability study).

Why
:   Oracle labels are the best available ground truth.

Impact
:   B5 is given oracle labels, which is generous to B5 relative to a real deployment.

### B5-4 · One lookup per period, no hysteresis { #b5-4 }

Paper
:   —

Here
:   One lookup per period, applied with no hysteresis.

Why
:   This is the policy as designed.

Impact
:   Possible thrashing at phase edges.

---

## B6 — compiler-directed IQ resizing (Jones et al., HPCA 2005; Trans. HiPEAC 2009) { #b6 }

B6 is the closest compiler prior work: an LLVM pass that computes the issue-queue demand of each
block from its dependence DAG and emits a hint per region
([B6 Jones IQ](guide/baselines/jones-iq.md)). Run as published (IQ only, variant `jones`) and
extended to ROB and LSQ (`jones_full`).

### B6-1 · Target ISA and simulator { #b6-1 }

Paper
:   Alpha on SimpleScalar/Wattch.

Here
:   RISC-V rv64gc on gem5 O3 (and x86-64).

Why
:   The evaluation platforms of this project.

Impact
:   Code generation and latencies are LLVM's for RISC-V, not the paper's.

### B6-2 · Hint semantics through `setwin` { #b6-2 }

Paper
:   A special NOOP sets the IQ size.

Here
:   The contract has only `setwin(W)`. The IQ demand `Q` (banks of 8) selects the smallest
    configuration with `IQ_c ≥ Q`, emitted as `setwin(ROB_c)`. In gem5 the `jones` variant runs
    with `structs=iq`, so only the IQ is capped and ROB/LSQ stay at the largest configuration
    ("IQ only, as published"). `jones_full` is the extension (ROB, IQ, LQ, SQ demand jointly).
    A dedicated `setiq` hint (tag 25 / x86 kind 3) exists in the pass
    (`-jones-iq-encoding=setiq`) but not in the contract or in gem5.

Why
:   One hint instruction and one mechanism for every variant.

Impact
:   The IQ cap is rounded to a configuration of the shared table.

### B6-3 · IR-level DAG { #b6-3 }

Paper
:   Analysis of machine code.

Here
:   Analysis of optimised LLVM IR at the optimizer-last extension point; IR-level instruction
    counts and latencies.

Why
:   An LLVM pass plugin runs on IR.

Impact
:   Instruction counts differ from the final RISC-V code.

### B6-4 · Loops by body replication { #b6-4 }

Paper
:   A closed-form steady-state analysis of loops.

Here
:   The loop body is replicated 2–16 times (≥ 96 instructions); the scheduler is bounded to the
    first 384 instructions of a region.

Why
:   *(reconstructed)* Replication approximates the steady state without a closed-form analysis; the bound limits
    the O(n²) scheduler on huge blocks.

Impact
:   An approximation of the steady state for long or very short bodies.

### B6-5 · Loads assumed to hit L1 { #b6-5 }

Paper
:   Loads are assumed L1 hits.

Here
:   The same.

Why
:   Faithful to the paper.

Impact
:   None relative to the paper; this is the cache-oblivious demand that contrasts with WinHint.

### B6-6 · Hint placement { #b6-6 }

Paper
:   Redundant NOOPs are removed.

Here
:   Loop hints go in the preheader (one per loop entry); redundant hints are removed by a
    forward must-dataflow.

Why
:   The same placement positions as WinHint.

Impact
:   B6 is faithful in its *analysis* (an ILP-only, cache-oblivious demand), which is exactly the
    contrast with WinHint's MLP-aware demand.

---

## B7 — profile-guided positional adaptation (Huang et al., ISCA 2003; Lau et al., CGO 2006) { #b7 }

B7 picks the best configuration per code position from a profiling run on the `small` input and
tests it on the `large` input; it asks "static model versus profiling"
([Baselines](guide/baselines/index.md)). Source:
[pgo_flow.py](../compiler/baselines/pgo/pgo_flow.py).

### B7-1 · WinHint's regions as positions { #b7-1 }

Paper
:   Huang et al. choose subroutine/loop "positions" from a profile with their own selection
    algorithm; Lau et al. choose phase markers from the loop/procedure hierarchy graph.

Here
:   The positions are WinHint's top-level loop-nest regions (`region(id)` markers). The
    per-region configuration is the best one measured on the `small` input (the same sweep as
    B1, input `small`), by ED²P (default) or IPC.

Why
:   Isolates the question "static model versus profiling" on identical positions.

Impact
:   B7 cannot pick finer positions than WinHint's regions.

### B7-2 · Energy per region { #b7-2 }

Paper
:   —

Here
:   From the `energy` column of `region_stats.csv` if present, else the proxy
    `P(c) = 1 + 0.15·ROB_c/ROB_max` (the WinHint cost model's), or McPAT watts per
    configuration (`--power-json`).

Why
:   McPAT has no per-region statistics ([I8](#i8)).

Impact
:   With the proxy, B7's ED²P choice shares WinHint's energy assumption.

---

## B8 — Clairvoyance (Tran et al., CGO 2017), public artifact reused { #b8 }

B8 is a compile-time transformation that reorders loads to expose MLP without hardware changes;
the authors' LLVM 3.8 artifact is reused, alone and combined with WinHint
([B8 Clairvoyance](guide/baselines/clairvoyance.md)).

### B8-1 · Artifact built from conda LLVM 3.8, two patches { #b8-1 }

Paper
:   The artifact's Makefile clones LLVM `release_38` and builds it.

Here
:   The clone URL is dead; LLVM/Clang 3.8.1 come from conda (env `winhint-llvm38`) and only the
    passes are built. `patches/0001-missing-returns.patch` fixes missing `return`s (undefined
    behaviour that GCC 13 turns into crashes); `patches/0002-swoopdae-null-latch.patch` fixes a
    SwoopDAE segfault ([B8-4](#b8-4)). The submodule stays pristine.

Why
:   Conda-only toolchain; the artifact does not build as published.

Impact
:   None on the transformation.

### B8-2 · Loop marking { #b8-2 }

Paper
:   The artifact transforms only loops with `#pragma clang loop vectorize_width(1337)`.

Here
:   A temporary copy of each kernel gets the pragma before every line-initial `for (`.

Why
:   The kernels are not annotated.

Impact
:   Every loop is offered to the transformation.

### B8-3 · x86-64 front end, retargeted to RISC-V { #b8-3 }

Paper
:   Evaluated on x86/ARM-like cores in the authors' simulators.

Here
:   Clang 3.8 has no RISC-V target. The 3.8 bitcode is read by LLVM 23, its triple and
    datalayout rewritten, x86 attributes and inline-asm clobbers stripped; IR with `x86_fp80`,
    `va_arg`, `byval` or `llvm.x86.*` is refused (none occur). Lowering uses
    `-Xclang -disable-llvm-optzns` so that the schedule is not undone.

Why
:   The evaluation ISA is RISC-V.

Impact
:   B8 is the artifact's transformation on a different ISA than the authors evaluated; the
    RISC-V code generation is LLVM 23's.

### B8-4 · SwoopDAE crash fixed { #b8-4 }

Paper
:   `SwoopDAE` assumes that an unconditional loop latch has a single predecessor holding the
    exit branch.

Here
:   After unrolling `contrast_mobilenet_infer`'s `conv3x3_stem` (guarded 3×3×3 taps) the latch
    has three, and the pass segfaulted. Patch 0002 falls back to the loop's unique exiting
    block (the choice the artifact's `BranchMerge` already makes). Every kernel is now
    transformed (`"excluded_functions": []`); the marker-stripping fallback in `cv_compile.sh`
    remains as a safety net (`CV_STRICT=1` makes it an error).

Why
:   A crash in the artifact.

Impact
:   A behaviour-preserving fix of a crash, not a change of the transformation.

### B8-5 · Knobs tuned per machine, not per kernel { #b8-5 }

Paper
:   —

Here
:   `CV_TYPE` / `CV_UNROLL` / `CV_INDIR` defaults (`consv`, 2, 1). `tune_baselines.py` samples
    at most `--budget` (16) of the 60 grid points in `knobs.json` on `small` and keeps the best
    geomean-ED²P point **per machine**, the same budget and rule as every other method. All 60
    grid points compile and match `plain` on every kernel.

Why
:   Equal-effort tuning ([Evaluation methodology](concepts/methodology.md#equal-effort-tuning)).

Impact
:   One knob setting per machine can be worse for an individual kernel than per-kernel tuning
    would be; every method is tuned the same way, so the comparison stays equal-effort.

---

## B9 — Long-Term Parking (Sembrant et al., MICRO-48 2015) { #b9 }

B9 parks non-urgent instructions in a queue between rename and the IQ/LSQ, so that the
scheduling structures fill only with critical ones (`window_policy=ltp`,
[B9 Long-Term Parking](guide/baselines/ltp.md)).

### B9-1 · Physical registers allocated at rename { #b9-1 }

Paper
:   Registers are allocated when an instruction leaves the parking queue.

Here
:   Allocated at rename.

Why
:   Deferring them needs virtual-physical registers (a rewrite of gem5's rename).

Impact
:   None on performance here (the register file never binds, [I5](#i5)); LTP's register-file
    savings are not credited.

### B9-2 · Memory operations keep LSQ program order { #b9-2 }

Paper
:   Parking is decided per instruction, by urgency.

Here
:   Memory operations keep LSQ program order; an urgent load behind a parked store pulls that
    store out (`memOrder`).

Why
:   gem5's LSQ requires program order.

Impact
:   Some non-urgent stores leave the parking queue early.

### B9-3 · Extra releases { #b9-3 }

Paper
:   Strict parking.

Here
:   Instructions are also released when the IQ has room (`room`, default 0.5), and non-urgent
    instructions bypass an empty queue. `room=1` gives strict paper-style parking.

Why
:   *(reconstructed)* Avoids idling the IQ when it has free entries.

Impact
:   *(reconstructed)* Less parking than the paper by default; `room=1` restores it.

### B9-4 · Urgency seeds { #b9-4 }

Paper
:   Seeds are long-latency (LLC-miss) loads and branches.

Here
:   Loads whose measured load-to-use latency at commit is ≥ `lll` (30 cycles), not an LLC-miss
    flag; mispredicted branches by default.

Why
:   *(reconstructed)* The latency is what gem5 exposes at commit.

Impact
:   *(reconstructed)* A load counts as long-latency by its measured latency (30 cycles is above an L2 hit),
    whatever level served it.

### B9-5 · Wake-up distance in sequence numbers { #b9-5 }

Paper
:   —

Here
:   The wake-up distance from the ROB head (`wake`, 16) is measured in sequence numbers.

Why
:   gem5 orders instructions by sequence number.

Impact
:   Over-counts right after a squash.

### B9-6 · Equal resources { #b9-6 }

Paper
:   A large ROB with small scheduling structures.

Here
:   The ROB stays at the largest configuration; IQ, LQ and SQ follow `window_initial` (`structs`,
    default IQ+LQ+SQ). `ltp_c<i>` is compared with `static_c<i>`.

Why
:   Comparison on the shared configuration table.

Impact
:   B9 gets a larger ROB than the static run it is compared with.

---

## Real-hardware baselines R0–R5 (Intel Core 5 120U) { #real-hardware }

On real hardware the "window" is the choice between P-cores and E-cores: WinHint's hints become
P/E migrations, compared with the stock scheduler, published schedulers and two reimplemented
policies ([Real hardware](guide/hardware/index.md)). Source: [docs/guide/hardware/index.md](guide/hardware/index.md).

**R0 — `taskset` on P or E.** No deviation.

### R1-1 · Stock Linux: EAS unavailable { #r1-1 }

Proposal
:   Stock EEVDF with ITMT and HFI/Thread Director hints; report SMT on and off.

Here
:   EAS is disabled by `intel_pstate` on SMT hybrids; the campaign runs SMT on and off
    (switching needs root). Kernel 7.0 (Ubuntu); the configuration is recorded by
    `record_system.sh`.

Why
:   The kernel's behaviour on this chip.

Impact
:   R1 is the kernel as shipped, without EAS.

### R2-1 · sched_ext built from conda tools { #r2-1 }

Proposal
:   Reuse `scx_bpfland` or `scx_cosmos` in power-save mode, `scx_lavd` as an alternative.

Here
:   Reused unchanged (scx 1.1.3), built with the env's cargo/clang; libbpf is vendored by
    `libbpf-sys` ([I10](#i10)). Power-save presets; flags checked against the built `--help`.

Why
:   No conda libbpf package.

Impact
:   None expected on the schedulers' behaviour.

### R3-1 · intel-lpmd: generic configuration { #r3-1 }

Proposal
:   Reuse intel-lpmd if the platform supports it.

Here
:   This CPU (family 6 model 186) has no model-specific lpmd configuration; the generic one is
    used with `<lp_mode_cpus>` = the E-cores (4–11).

Why
:   No upstream configuration for this model.

Impact
:   lpmd runs with a configuration not tuned by Intel for this chip.

### R3-2 · intel-lpmd: `upower-glib` stub { #r3-2 }

Proposal
:   Build intel-lpmd with its dependencies.

Here
:   `upower-glib` is replaced by a stub that always reports AC power (no battery/AC tracking).
    This is the only remaining build deviation: glib 2.90 (with `glib-compile-resources`) comes
    from the `winhint` env. Recorded in `WINHINT_LPMD_DEVIATIONS`.

Why
:   No conda package for `upower-glib`.

Impact
:   lpmd behaves as on AC power; the campaign must run on AC.

### R4-1 · PIE: one job, energy objective { #r4-1 }

Paper
:   Van Craeynest et al. (ISCA 2012) schedule several threads on a simulated big/small CMP to
    maximise throughput.

Here
:   One job; the objective is energy: run on E whenever the predicted E/P time ratio is
    ≤ 1 + slack (`-s 0.15`), with hysteresis (`-H 2`), every 10 ms.

Why
:   The evaluation runs one inference job at a time.

Impact
:   R4 is a single-job adaptation of PIE.

### R4-2 · PIE model inputs { #r4-2 }

Paper
:   CPI split into base and memory parts; MLP from MSHR occupancy.

Here
:   The model is kept: the base part scales with width/ILP, the memory part with the MLP ratio.
    MLP is measured on P-cores with `L1D_PEND_MISS.PENDING / PENDING_CYCLES`; otherwise it is
    estimated as LLC-misses/instruction × ROB size (512 Raptor Cove, 256 Gracemont, capped at
    16).

Why
:   The counters this CPU exposes.

Impact
:   MLP on E-cores is an estimate.

### R4-3 · perf_event_open backend; PMCTrack module does not build { #r4-3 }

Proposal
:   PMCTrack (reused) or `perf_event_open`.

Here
:   PMCTrack v4.0's kernel module does **not** compile against kernel 7.0 (about 25 call sites
    of removed kernel APIs across 8 files: a port, not a build fix), so R4 uses its
    `perf_event_open` backend. The PMCTrack backend of `pie_daemon` (driving the PMCTrack CLI)
    is implemented but untested.

Why
:   Kernel 7.0 removed APIs PMCTrack uses; the artifact is reused unchanged.

Impact
:   None on the policy: both backends implement the same one.

### R4-4 · Tuning and fidelity before the campaign { #r4-4 }

Paper
:   Parameters chosen by the authors.

Here
:   The slack (`-s`) and hysteresis (`-H`) must be tuned on the `small` inputs before the
    campaign (the smoke run flipped inside phases with the defaults). The qualitative-trend
    predicates for R4 (and R5) are in [hw/fidelity.py](../hw/fidelity.py).

Why
:   Equal-effort tuning; different platform.

Impact
:   Untuned defaults would migrate too often.

### R5-1 · Sondag & Rajan: WinHint's regions as sections { #r5-1 }

Paper
:   Sondag & Rajan (CGO 2011) type basic-block sections by static instruction-mix similarity,
    insert phase marks, sample each type on every core type, then assign; throughput objective.

Here
:   The marks are WinHint's `region(id)` markers (top-level loop nests, call mode). Static
    typing by `region_types.py` (k-means on per-region features of `<kernel>.regions.json`,
    else one type per function or region). The first `K=2` visits of each type are sampled on P
    then E; the type is assigned P iff the E-core slowdown in time per instruction is ≥ 1.4.

Why
:   Reuses the compiler's region markers; one job at a time.

Impact
:   Coarser sections than the paper; the slowdown threshold is a single-job
    energy/performance knob.

### R-P1 · Platform micro-architecture { #r-p1 }

Proposal
:   "Intel Core 5 120U: P-cores with SMT, E-cores"; µarch parameters from Intel's optimisation
    manual, confirmed with microbenchmarks (PROPOSAL §8).

Here
:   The chip is family 6 model 186 (0xBA), **Raptor Lake-U refresh**: Raptor Cove P-cores and
    Gracemont E-cores (not Meteor Lake's Redwood Cove/Crestmont). Documented values (ROB 512 /
    256, and others) are in [µarch parameters](guide/hardware/uarch-params.md); several Gracemont
    sizes (load/store buffers, physical registers, fill buffers) are still "to confirm" by
    `hw/tools/uarch_probe.c` (`hw/fidelity.py --uarch`).

Why
:   Intel does not document every Gracemont size.

Impact
:   R4's PIE model and the libwinhint threshold use the documented ROB sizes; an unconfirmed
    value must not be quoted in the paper until the probe settles it.

### R-P2 · llama.cpp integration (Phase E2, optional) { #r-p2 }

Proposal
:   Optional operator-level hooks in llama.cpp or ONNX Runtime.

Here
:   [llama.cpp integration](guide/hardware/llamacpp.md) patches the ggml CPU backend (tag
    b11327) to call `__winhint_region` / `__winhint_setwin` per graph node. Only ggml thread 0
    calls libwinhint, so the hardware driver (`hw/run_hw_experiments.py`) forces one thread
    (`OMP_THREAD_LIMIT=1`). The per-op-class window table (for example MUL_MAT 256, norms 64) is
    a **static placeholder**, not produced by the WinHint analysis (ggml's kernels are not
    compiled with the plugin).

Why
:   ggml's kernels are C/C++ with SIMD intrinsics compiled by GCC, outside the plugin flow.

Impact
:   llama.cpp runs demonstrate the runtime path only; they are not evidence for the static
    model.

### R-P3 · Measurement method { #r-p3 }

Proposal
:   RAPL energy, fixed governor.

Here
:   RAPL is root-only on this host (mode 0400); energy needs the driver run as root or a
    temporary read grant. A fixed frequency policy is opt-in (root).

Why
:   Host permissions.

Impact
:   Without root or the read grant, runs have no energy numbers.

---

## Infrastructure and toolchain (affects every gem5 policy) { #infrastructure }

The shared resize mechanism in gem5, the energy model, the workloads and the toolchain, used by
WinHint and every baseline alike ([gem5 model](guide/gem5/index.md),
[gem5 patches](guide/gem5/patches.md), [Toolchain](reference/toolchain.md)).

### I1 · Hint recognised at commit, not decoded in `decoder.isa` { #i1 }

Proposal
:   Phase D: decode the HINT in gem5's `src/arch/riscv/isa/decoder.isa` through the patch,
    signal the O3 CPU, and stay a no-op on the clean build.

Here
:   The RISC-V decoder is not patched. Stock gem5 v25.1 already decodes `ori x0, x0, IMM` as
    the no-op `ori_hint`. The `WindowController` inspects the committed 32-bit instruction word
    (low 20 bits `0x06013`, `IMM[11]=0`, `IMM[4:0]` = 21 `setwin` / 23 `region`) and acts on it
    at commit, only on `RISCV_winhint`.

Why
:   A smaller patch; identical decoding on `RISCV_clean` and `RISCV_winhint` (the bit-identical
    check runs the same decoder); non-speculative detection: a hint on a squashed path never
    resizes the window.

Impact
:   A hint takes effect when it commits; the instructions already in flight behind it were
    dispatched under the old configuration. With hints in loop preheaders outside hot loops,
    the lag is at most one window of instructions per switch, which is negligible against
    region lengths of at least `2·W_max` instructions ([W6](#w6)). It favours no variant: all
    hint-driven variants (WinHint, B1, B6, B7, B8+WinHint, WinHint+HW) share it.

Source
:   [gem5 patches](guide/gem5/patches.md) "Hint decoding";
    [Window policies](guide/gem5/policies.md) "Hint decoding".

### I2 · Whole-window configuration table instead of per-structure resizing { #i2 }

Papers
:   Several baselines (B2, B6, Petoumenos et al.) resize one structure, or each structure
    independently.

Here
:   Every policy picks an index into one table (`sim/machines/*.json`, `"window"`), which moves
    ROB, IQ, LQ and SQ together (`riscv_ooo`: 64/32/16/16, 128/64/32/32, 192/96/48/48,
    256/128/64/64). B6 "IQ only" and B9 use `structs=` to cap a subset.

Why
:   PROPOSAL §4.1 requires the same mechanism and table for every policy, so that only the
    decision differs.

Impact
:   Removes per-structure freedom from B2 ([B2-1](#b2-1)); equal resources for all.

### I3 · Resize = caps on physically maximal structures; shrink gates dispatch { #i3 }

Proposal
:   Phase A: set `maxEntries`, gate dispatch until occupancy is at or below the target.

Here
:   As proposed. Structures are built at the largest configuration; a configuration sets caps;
    `free = max(0, min(phys_free, cap − occupancy))`; shrinking flushes nothing. Instructions
    already between rename and the ROB can overshoot a new cap transiently (never the physical
    size); occupancy maxima are recorded only in settled cycles.

Why
:   gem5 rename learns free counts through time buffers; a flush would cost more than the
    hardware the papers model (a partition switched off once empty).

Impact
:   A shrink costs a drain (counted in `system.cpu.window.drainCycles`); growth is immediate.
    The same for every policy.

Source
:   [gem5 patches](guide/gem5/patches.md) "Resize semantics".

### I4 · No clock or wake-up/select latency benefit for small windows { #i4 }

Papers
:   B3 (Kora et al.) and B2 (Ponomarev et al.) credit smaller structures with shorter
    wake-up/select latency or lower energy per access.

Here
:   All configurations run at the same clock and pipeline depth; smaller configurations save
    energy only in the energy model.

Why
:   gem5 O3 has no size-dependent scheduler latency; adding one would be a new model.

Impact
:   The IPC of small-window configurations is not inflated by a faster scheduler. This
    penalises policies that live in small windows (B3's ILP mode) in IPC, not in energy.

Source
:   [Window policies](guide/gem5/policies.md), B3 deviation 4.

### I5 · Register file sized to never bind { #i5 }

<span id="15-register-file-sized-to-never-bind"></span>

Proposal
:   Phase A: "180+ physical registers".

Here
:   `num_int_regs = num_fp_regs = 32 + largest ROB` (288 on `riscv_ooo`, 416 on
    `riscv_ooo_big`, 160 on `riscv_ooo_small`).

Why
:   PROPOSAL §2, problem 6: the window, not the register file, must be the binding limit.

Impact
:   Removes the confound; B9's register-file savings cannot show ([B9-1](#b9-1)).

### I6 · MLP measured on L1D MSHRs { #i6 }

Papers
:   MLP in Kora et al. and PIE refers to outstanding LLC misses / MSHR occupancy.

Here
:   `WindowSample::mlp` is the mean number of allocated L1D MSHRs over the cycles with at least
    one (the Chou et al. definition); L1D/L2 misses are per-period deltas of the caches'
    demand-miss counters.

Why
:   The controller reads only the L1D MSHR queue (a read-only accessor in the patch).

Impact
:   L2 hits count as "parallel misses" as well as memory misses; thresholds (`mlp_thr`) are
    tuned for this definition.

Source
:   [gem5 patches](guide/gem5/patches.md) "MLP".

### I7 · No prefetcher by default { #i7 }

Proposal
:   Phase A: drop the fake `setPrefetchDistance` call, or attach a real `StridePrefetcher`.

Here
:   Dropped: `l1d_prefetcher: "none"` in every machine JSON. `se.py --l1d-prefetcher stride`
    attaches `StridePrefetcher(degree=4)`; the cost model then maps streaming groups to L2
    (`TargetModel::StridePrefetcher`).

Why
:   *(reconstructed)* The proposal allows dropping the call; a real prefetcher stays available as an option.

Impact
:   Streaming phases are more memory-bound than on a real core with prefetchers, so the window
    matters more. The prefetcher run is a sensitivity point, not the default.

### I8 · Energy: McPAT per configuration, residency-weighted, or a proxy { #i8 }

<span id="18-energy-mcpat-per-configuration-residency-weighted-or-a-proxy"></span>

Proposal
:   Reuse McPAT.

Here
:   `sim/estimate_energy.py` splits a run by residency in each configuration and uses McPAT if
    built, else an analytic proxy
    (`P_static(i) = P_CORE·(1 − 0.15 + 0.15·rob_i/rob_max)` plus per-instruction and per-miss
    energies). McPAT has no per-region statistics: the oracle's ED²P map uses whole-run power
    per configuration times region time. McPAT has no LTP component (B9's parking queue should
    be modelled as a 128-entry RAM FIFO plus a 256-entry 4-way UIT).

Why
:   McPAT models whole runs of one fixed configuration.

Impact
:   Energy numbers are relative, single-machine comparisons; absolute values are not claimed.

### I9 · Workloads: reduced inputs, reduced vocabularies, one layer { #i9 }

Proposal
:   Phase B: BERT-base / GPT-2-small shapes, sequence 128–512, working sets beyond L2; two
    input sizes per kernel.

Here
:   Model shapes are full (H 768, FFN 3072, 12 heads; 28.3 MB of weights per transformer
    layer); vocabularies are reduced to 8192; one layer by default (`argv[2]`). The `large`
    sequence is 128–256 (encoders), 96+32 / 64+64 / 64+16 (decoders). The `small` inputs are
    tiny (for example sequence 4 for BERT, prompt 4 + 2 generated tokens for GPT-2), so that a
    `small` run is about 40–550 M RISC-V instructions. Shapes are documented in each
    `benchmarks/*.c` header and in [Workloads](guide/workloads.md).

Why
:   gem5 O3 simulation time on a 12-thread, 6 GB host.

Impact
:   B7 (trains on `small`, tests on `large`) and B5 (trains on `small` traces) see a much
    shorter training input than the test input. Testing on an unseen input is the point of B7,
    but the small inputs shift the balance of phases (for example prefill versus decode).
    Sequence 512 is not run by default (`-DSEQ_LARGE=` overrides).

Status
:   An earlier note said the README workload table listed older `small` sizes. It no longer
    does: [Workloads](guide/workloads.md) matches the headers.

### I10 · Toolchain substitutions { #i10 }

Every tool must come from conda (PROPOSAL §5, Phase 0). Where a proposed package does not
exist, the substitution is recorded in [tooling/versions.lock](../tooling/versions.lock) and
summarised here ([Toolchain](reference/toolchain.md)).

| Item: proposal → here | Why | Impact |
|---|---|---|
| RISC-V cross compiler: `gcc_linux-riscv64` (GCC 13) → `gcc_impl_linux-riscv64` 15.3, or clang 23 `--target=riscv64-conda-linux-gnu` | no conda RISC-V GCC 13; the wrapper pins host GCC | none on hints; benchmarks are built with clang 23 and the plugin |
| QEMU: `qemu` → `qemu-execve-riscv64` 11.0.3, exposed as `qemu-riscv64` | no `qemu` package on conda-forge | none |
| libc++: `libcxx-devel` if needed → not installed; the plugin uses libstdc++ | conflicts with GCC 13 | none |
| libbpf / bpftool: conda packages → none; `libbpf-sys` vendors libbpf, BPF objects built with clang 23 | no package on any channel | R2 builds; bpftool unused |
| `upower-glib` → stub (`hw/baselines/r3_lpmd/upower_stub/`) | no conda package | R3 always sees "AC power, upowerd unreachable"; run the campaign on AC ([R3-2](#r3-2)). glib itself comes from the env (conda-forge `libglib`/`glib`/`glib-tools` 2.90) |
| LLVM 3.8: `llvmdev=3.8`, `clang=3.8` → `llvmdev`/`clangdev` 3.8.1 (dependencies from the `numba` channel) | no `clang` 3.8 package | none |
| PMCTrack module compiler: env GCC 13 → separate env `winhint-kmod` with GCC 15.2 | the host kernel is built with GCC 15.2 (`-fmin-function-alignment=16`) | the compiler is fixed, but the module still does not build on kernel 7.0 ([R4-3](#r4-3)) |
| TVM/IREE-generated C (Phase B, optional) → **not done** | no conda package for the TVM compiler or IREE (only `apache-tvm-ffi`) | no compiler-generated kernels in the build matrix; see [Generated kernels](guide/workloads-generated.md) |
| `perf` → conda-forge `linux-perf` | package name | none |

### I11 · Python, LLVM and gem5 versions { #i11 }

Pinned as proposed (Python 3.14, LLVM 23.1.2, gem5 v25.1.0.0). No deviation; listed so that the
check is visible.

---

## Adding a deviation

1. Describe it next to the code first: in the component's README or `docs/guide/gem5/policies.md`, in a
   "Deviations" section, or in `tooling/versions.lock` for a tool substitution.
2. Add an entry here under its component, with the next free identifier (`B3-6`, `I12`, …) and
   the four fields Paper/Proposal, Here, Why and Impact. State the impact as an expected
   direction, never as a measured number. Link the source document.
3. Update the entry count in the [summary](#summary).
4. When a later change removes a deviation, keep the entry and add a **Status** field saying
   what changed, so that references from the paper and other pages stay valid.

## Next

- [Evaluation methodology](concepts/methodology.md): how the baselines are tuned and checked
  for fidelity.
- [Baselines](guide/baselines/index.md): each baseline, its variant name and how to run it.
- [Simulator fidelity](guide/fidelity.md): the qualitative-trend checks each reimplementation
  must pass.
