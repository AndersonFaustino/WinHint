# B6 — Compiler-directed issue-queue resizing (Jones et al., HPCA 2005)

B6 is the closest prior work to WinHint: a compiler pass that analyses each region's
issue-queue demand and emits a hint that resizes the queue. It is in the comparison to show
what a compiler-directed, IQ-only scheme achieves against WinHint's whole-window demand model.
No artifact exists, so it is reimplemented here as an LLVM pass that emits the same hint
instruction as WinHint and runs on the same gem5 mechanism
([Evaluation methodology](../../concepts/methodology.md),
[Baselines](index.md)).

Reimplementation of

> T. M. Jones, M. F. P. O'Boyle, J. Abella, A. González. *Software Directed Issue
> Queue Power Reduction.* HPCA 2005; extended as *Compiler Directed Issue Queue
> Energy Reduction*, Trans. HiPEAC 4(1), 2009.

**Terms.** The *IQ* (issue queue) holds dispatched instructions until their operands are
ready. A *region* here is an innermost loop or a large basic block (below). `setwin(W)` is the
WinHint hint that selects the smallest window configuration whose ROB is at least W
([interfaces.md §2](../../interfaces.md#2-the-hint-isa-contract)); the configurations are the
rows of the window table ([interfaces.md §3](../../interfaces.md#3-window-configuration-table)).
Other terms are in the [Glossary](../../reference/glossary.md).

## Build

The original work used Wattch/SimpleScalar on Alpha. This is an LLVM (23, conda env
`winhint`) new-pass-manager plugin, `JonesIQ.so`, built with the WinHint plugin into
`$WINHINT_BUILD/compiler/` ([`compiler/build.sh`](../../../compiler/build.sh)). The target is defined in
[`CMakeLists.txt`](../../../compiler/baselines/jones_iq/CMakeLists.txt), which `compiler/CMakeLists.txt` includes with
`add_subdirectory`, and which can also be built standalone. It emits the hint through
`compiler/common`'s `HintEmitter`.

## What it does

1. **Regions.** Every innermost loop, and every other basic block with at least
   `-jones-min-block` (8) queue-occupying instructions.
2. **DAG analysis.** For each region the data-dependence DAG of its instructions in
   program order is scheduled on an idealized out-of-order core taken from the
   machine JSON: in-order dispatch of `dispatch_width` per cycle, issue of up to
   `issue_width` ready instructions per cycle, per-operation latencies, in-order commit
   of `commit_width`. Loads hit in the L1 (as in the paper). Loop bodies are
   replicated (≥ `-jones-loop-insts`, 96, instructions; 2–16 copies) with header PHIs
   linked to the previous copy, so cross-iteration overlap is visible.
3. **Demand.** The smallest IQ size, in banks of 8 entries, whose schedule is not
   longer than the schedule with the largest IQ (`-jones-tolerance`, default 0).
4. **Hints** at the region entry (loop preheader, or block start). Hints whose value
   is already in force on every incoming path are removed (forward must-dataflow over
   the CFG), as the paper removes redundant special NOOPs.

## Variants (`benchmarks/Makefile`)

| Variant | Flags | Meaning |
|---------|-------|---------|
| `jones` | `-jones-mode=iq` | As published: IQ demand only |
| `jones_full` | `-jones-mode=full` | Extension: ROB, IQ, LQ and SQ demand jointly; emitted as `setwin(ROB of that config)` |

`jones_full` picks the smallest window configuration (interfaces.md §3 table) whose schedule
is not longer than with the largest one.

### How the IQ-only demand maps to `W` (deviation note)

The contract ([interfaces.md §2](../../interfaces.md#2-the-hint-isa-contract)) only has `setwin(W)`, which resizes ROB, IQ, LQ
and SQ **together**. An IQ-only hint therefore cannot be expressed exactly. Two
encodings are implemented (`-jones-iq-encoding`, Makefile `JONES_ENC`):

* `setwin` (**default**, inside the contract): the IQ demand `Q` (entries, banks of
  8) selects the smallest configuration `c` of the §3 table with `IQ_c ≥ Q`, and the
  hint is `setwin(W = ROB_c)`. Because the core picks the smallest configuration with
  `ROB ≥ W`, this hint selects exactly `c`. With `riscv_ooo.json`:
  `Q ≤ 32 → W = 64`, `Q ≤ 64 → 128`, `Q ≤ 96 → 192`, otherwise `256`.
  The whole window (ROB, LQ, SQ) follows the IQ demand: the ROB/LSQ are resized as a
  side effect of the IQ decision, not from their own demand (that is `jones_full`). The stats file
  (`<kernel>.jones.json`) records `"iq_only": true` so the analysis can label the
  results as "B6 (IQ demand, whole-window mechanism)".
* `setiq` (**proposed, not in the contract**): a third hint kind that resizes only
  the IQ. RISC-V: `ori x0, x0, IMM`, `IMM = (payload << 5) | 0b11001` (tag 25, also
  outside the Zicbop `prefetch.*` encodings); x86: `nopl DISP32(%rax)` with kind 3,
  `DISP32 = 0x57483000 | payload`; payload = IQ entries / 8 (banks of 8, 0..63).
  On the clean gem5 build and on any unmodified core it is a no-op. It needs gem5
  support (decode tag 25 → cap only the IQ) before it has any effect; in
  `-jones-emit=call` mode no runtime entry point exists, so `setiq` hints are dropped.

Recommendation: keep `setwin` as the default for the paper's B6 numbers, and add
`setiq` to [interfaces.md](../../interfaces.md) (kind 3 / tag 25) only if the gem5 side implements
IQ-only capping, which is what "IQ only, as published" really means.

## Deviations from the paper

* **Target and simulator:** RISC-V rv64gc on gem5 O3 (and x86-64), not Alpha on
  SimpleScalar/Wattch.
* **Mechanism granularity:** with the default encoding the IQ demand drives the
  whole window (see above); the paper resized only the IQ.
* **Hint semantics:** the paper's special NOOP set the IQ size directly; here the
  hint is the contract's `setwin(W)` (see the mapping above), so "B6 IQ-only" means
  *IQ-demand analysis, whole-window mechanism*. `jones_full` uses the same
  instruction and the same placement/redundancy removal; only the demand analysis
  differs (ROB, IQ, LQ and SQ checked jointly).
* **IR-level DAG:** the analysis runs on optimized LLVM IR at the optimizer-last
  extension point, not on the final machine code; instruction counts and latencies
  are IR-level approximations (constant-index GEPs, PHIs and free casts occupy no
  entry; non-constant GEPs and calls count as one).
* **Loop handling:** the paper analysed loops with their cross-iteration overlap;
  we replicate the body a bounded number of times (see above) instead of a
  closed-form steady-state analysis.
* **Banking:** 8-entry banks, and the IQ sizes are those of the configuration table.
* **Huge blocks:** the scheduler is bounded to the first 384 instructions of a region.
* **Placement:** loop hints go to the preheader (one per loop entry) rather than
  inside the loop header.

## Usage

The two variants build through the benchmarks Makefile (one directory per variant):

```sh
# B6 as published -> $WINHINT_BUILD/benchmarks/riscv/jones/
make -C benchmarks ARCH=riscv VARIANT=jones

# extension to ROB/LSQ -> $WINHINT_BUILD/benchmarks/riscv/jones_full/
make -C benchmarks ARCH=riscv VARIANT=jones_full
```

The pass can also be run by hand. With clang (add `-mllvm -jones-emit=call` to emit
`__winhint_setwin(W)` calls instead, and link libwinhint):

```sh
clang -O2 -fplugin=$WINHINT_BUILD/compiler/JonesIQ.so \
      -fpass-plugin=$WINHINT_BUILD/compiler/JonesIQ.so \
      -mllvm -jones-target=sim/machines/riscv_ooo.json -mllvm -jones-mode=iq \
      -mllvm -jones-out-dir=jones-out -mllvm -jones-verbose \
      -c benchmarks/encoder_bert_tiny_infer.c -o encoder_bert_tiny_infer.o
```

With opt, on existing IR:

```sh
opt -load-pass-plugin=$WINHINT_BUILD/compiler/JonesIQ.so \
    -passes=jones-iq -jones-mode=full x.ll
```

Output: `<out-dir>/<kernel>.jones.json` (or `.jones_full.json`) with one entry per
region (IQ demand, chosen config, hint value, whether it was removed as redundant).
The runbook builds both variants for every machine
([Hinted and per-machine binaries](../usage.md#step-7-hinted-and-per-machine-binaries)).

## Next

- [Clairvoyance](clairvoyance.md): B8, the other compiler baseline.
- [Simulator fidelity](../fidelity.md):
  the B6 trend check.
- [Deviations](../../deviations.md#b6): this baseline's entries, next to every other baseline's.
