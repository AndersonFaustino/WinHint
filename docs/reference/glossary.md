# Glossary

Every project term and abbreviation used on this site, in alphabetical order, with a short
definition and a link to the page that owns the details. Other pages link here on first use
of a term; each entry's anchor is the term's slug (for example `glossary.md#rob`,
`glossary.md#w` for W\*). For the ideas behind the terms, read
[Background](../concepts/background.md) first.

## B

### B0 {#b0}

Static windows: the whole run uses one [window configuration](#window-configuration). Each
machine has four, so B0 is four runs (`static_c0` … `static_c3`, policy `static`); "B0-large"
is the largest. See [Baselines](../guide/baselines/index.md).

### B1 {#b1}

The per-region [oracle](#oracle): the best configuration of each region from exhaustive
per-region simulation, compiled back into the `oracle_hinted` binary as `setwin` hints. The
reference WinHint should approach; not a strict upper bound because it ignores switch costs
([Deviations §3](../deviations.md#3-b1-oracle)).

### B2 {#b2}

Occupancy-driven resizing (Ponomarev, Kucuk and Ghose, MICRO-34 2001), reimplemented as the
gem5 policy `occupancy`: shrink when the window is under-used, grow when dispatch stalls on a
full structure. See [Window policies](../guide/gem5/policies.md).

### B3 {#b3}

MLP-aware window resizing (Kora, Yamaguchi and Ando, MICRO-46 2013), reimplemented as the
policy `mlp`: keep a small window unless last-level misses with exploitable [MLP](#mlp)
appear, then enlarge it. The closest hardware competitor. See
[Window policies](../guide/gem5/policies.md).

### B4 {#b4}

Basic-block-vector phase tracking and prediction (Sherwood, Sair and Calder, ISCA 2003),
reimplemented as the policy `bbv`: a [BBV](#bbv) signature per interval, a phase table, a
next-phase predictor and a per-phase best configuration learned online. See
[Window policies](../guide/gem5/policies.md).

### B5 {#b5}

A learned counter-based predictor in the style of Dubach et al. (TACO 2013), built from the
project's earlier reactive prototype after fixing it: per-window counters labeled with the
oracle's best configuration, a trained model, exported as a [LUT](#lut) that gem5 reads at
run time (policy `lut`). `lut_xfer` reuses the reference machine's LUT on other machines. See
[Baselines](../guide/baselines/index.md).

### B6 {#b6}

Compiler-directed issue-queue resizing (Jones, O'Boyle, Abella and González, HPCA 2005),
reimplemented as the LLVM pass `JonesIQ`. Variant `jones` resizes the IQ only, as published;
`jones_full` extends it to ROB and LSQ. See [Jones IQ](../guide/baselines/jones-iq.md).

### B7 {#b7}

Profile-guided positional adaptation (Huang, Renau and Torrellas, ISCA 2003; Lau, Perelman
and Calder, CGO 2006): the best configuration per region from a profile of the `small` input,
compiled into hints and evaluated on the `large` input (variant `pgo`). Tests a static model
against profiling. See [Baselines](../guide/baselines/index.md).

### B8 {#b8}

Clairvoyance (Tran et al., CGO 2017): look-ahead compile-time scheduling that raises MLP with
no hardware change. The authors' public LLVM 3.8 artifact is reused, alone (`clairvoyance`)
and combined with WinHint (`winhint_clairvoyance`). See
[Clairvoyance](../guide/baselines/clairvoyance.md).

### B9 {#b9}

[Long-Term Parking](#ltp) (Sembrant et al., MICRO-48 2015), reimplemented as the policy `ltp`.
See [Long-Term Parking](../guide/baselines/ltp.md).

### BBV {#bbv}

Basic-block vector: a per-interval count of executed basic blocks, used as a signature of the
program phase. B4 compares BBVs to recognize recurring phases. See [B4](#b4).

## C

### Call mode {#call-mode}

The compiler option `-winhint-emit=call`: hints become calls to `__winhint_setwin` /
`__winhint_region` instead of NOP instructions, so [libwinhint](#libwinhint) can act on them
on real hardware (variants `winhint_call`, `oracle_call`). See
[Interfaces §2](../interfaces.md#runtime-call-mode-real-hardware).

### Commit gate {#commit-gate}

`make validate`, run by a git pre-commit hook on every commit: environment check, every host
test suite with more than 90 % line coverage per language, and the documentation checks. It
needs no gem5 build and no root. See [Commit gate and coverage](../contributing/quality.md).

### CP {#cp}

Critical path: the length in cycles of the longest dependence chain in one iteration of a
loop body, using the target's operation latencies. One of the three inputs of the cost model.
See [Compiler, Cost model](../guide/compiler.md#cost-model).

## D

### D_indep {#d-indep}

$D_{indep}$: the distance, in dynamic instructions, between independent long-latency loads of
a loop (instructions per iteration divided by independent misses per iteration). Loads on a
loop-carried recurrence, such as a pointer chase, are excluded. See
[Compiler, Cost model](../guide/compiler.md#cost-model).

### Deviation {#deviation}

A recorded difference between a baseline and its paper, or between WinHint and the design
proposal, with the reason and the expected impact on the comparison. See
[Deviations](../deviations.md) and
[Evaluation methodology](../concepts/methodology.md#deviations).

## E

### E-core {#e-core}

Efficiency core of an Intel hybrid CPU: a smaller out-of-order core (Gracemont, 256-entry
ROB on the test machine, CPUs 4–11). WinHint sends a thread there for a small `setwin(W)`.
See [Microarchitecture parameters](../guide/hardware/uarch-params.md).

### ED²P {#ed2p}

Energy–delay-squared product, energy × time². It weights performance more than [EDP](#edp)
and is the main energy-efficiency metric of the gem5 study and the objective of
[equal-effort tuning](#equal-effort-tuning). See
[Evaluation methodology](../concepts/methodology.md#metrics).

### EDP {#edp}

Energy–delay product, energy × time. The main metric on real hardware. See
[Evaluation methodology](../concepts/methodology.md#metrics).

### EEVDF {#eevdf}

Earliest Eligible Virtual Deadline First, the default Linux CPU scheduler. Part of the stock
scheduler baseline [R1](#r1).

### Equal-effort tuning {#equal-effort-tuning}

Every tunable method, WinHint included, gets the same budget of parameter points on the
`small` input, always including its published default; the best geometric-mean ED²P per
machine is kept. It answers the risk that reimplemented baselines are weak. See
[Evaluation methodology](../concepts/methodology.md#equal-effort-tuning).

## F

### Fidelity check {#fidelity-check}

A test, run before a reimplemented baseline is used, that it reproduces its paper's
qualitative trend on a microbenchmark. Verdicts are `PASS`, `FAIL`, `INVALID` or
`INCOMPLETE`. See [Simulator fidelity](../guide/fidelity.md).

### FNV-1a {#fnv-1a}

A simple 32-bit hash. Each kernel prints the FNV-1a hash of its output bytes, so a hinted
build can be checked bit-identical against `plain`. See
[Workloads](../guide/workloads.md#running-a-kernel).

## G

### gem5 builds {#gem5-builds}

`RISCV_clean` is unmodified gem5, where hints are no-ops (used for the bit-identical checks);
`RISCV_winhint` adds the window controller and policies. See
[gem5 model](../guide/gem5/index.md#the-two-builds).

### gem5 O3 {#gem5-o3}

gem5's detailed out-of-order CPU model (`DerivO3CPU`), here version 25.1 on RISC-V. WinHint's
window controller is added to it. See [gem5 model](../guide/gem5/index.md).

## H

### Heavy lock {#heavy-lock}

The file lock `$WINHINT_BUILD/.heavy.lock`. Every gem5 build, long gem5 run and McPAT call
takes it, so heavy jobs run one at a time on the 6 GB host. See
[Running experiments](../guide/usage.md#before-you-start).

### HFI {#hfi}

Hardware Feedback Interface (Intel Thread Director): the processor reports to the OS the
relative performance and efficiency of each CPU. Used by the stock Linux scheduler, [R1](#r1).

### Hint {#hint}

An advisory instruction the compiler inserts to tell the hardware about the code that
follows: [setwin](#setwin) (window size) or a [region marker](#region-marker). Both are
architectural no-ops on unmodified cores; the hardware may ignore them. See
[Interfaces §2](../interfaces.md#2-the-hint-isa-contract).

## I

### II {#ii}

Initiation interval: the cycles between the starts of consecutive loop iterations, bounded by
recurrences, issue width and divider occupancy. It sets the loop's sustainable issue rate in
the cost model. See [Compiler, Cost model](../guide/compiler.md#cost-model).

### IPC {#ipc}

Instructions per cycle, the performance metric of the gem5 study. See
[Evaluation methodology](../concepts/methodology.md#metrics).

### IQ {#iq}

Issue queue: holds dispatched instructions until their operands are ready and they issue to a
functional unit. One of the four structures a [window configuration](#window-configuration)
sizes. See [Background](../concepts/background.md#the-out-of-order-window).

### ITMT {#itmt}

Intel Turbo Boost Max Technology 3.0 support in the Linux scheduler: it prefers the
higher-performance cores. Part of the stock scheduler baseline [R1](#r1).

## L

### L_mem {#l-mem}

$L_{mem}$: the expected service latency, in cycles, of a loop's independent long-latency
loads. Each group of accesses is assigned the first cache level that holds its footprint or
reuse distance; memory costs the last cache latency plus the memory latency. See
[Compiler, Cost model](../guide/compiler.md#cost-model).

### libwinhint {#libwinhint}

The real-hardware runtime: it implements `__winhint_setwin(W)` by moving the calling thread to
the P-cores (W at or above a threshold, default 192) or the E-cores, and also implements
[R5](#r5). See [Real hardware](../guide/hardware/index.md).

### lpmd {#lpmd}

Intel Low Power Mode Daemon (`intel-lpmd`), a public daemon that confines work to low-power
CPUs when the load allows. Reused as baseline [R3](#r3).

### LQ {#lq}

Load queue: holds in-flight loads. With the [SQ](#sq) it forms the [LSQ](#lsq). See
[Background](../concepts/background.md#the-out-of-order-window).

### LSQ {#lsq}

Load/store queue: the [LQ](#lq) and the [SQ](#sq) together.

### LTP {#ltp}

Long-Term Parking (Sembrant et al., MICRO-48 2015): a parking queue between rename and
dispatch holds non-urgent instructions, so the IQ and LSQ fill only with instructions on the
path to long-latency loads and branches. Baseline [B9](#b9).

### LUT {#lut}

Lookup table. In [B5](#b5), a table indexed by binned counter values (IPC, ROB occupancy, L1D
misses per kilo-instruction, MLP) whose entries are window configurations. Format:
[Interfaces §6](../interfaces.md#6-b5-lut-file-format-window_lut_file).

## M

### McPAT {#mcpat}

A power, area and timing model for processors. Built from source here and used by
`sim/estimate_energy.py` to estimate the energy of each window configuration; an analytic
proxy replaces it when it is not built. See
[Evaluation methodology](../concepts/methodology.md#how-energy-is-estimated-in-gem5).

### MLP {#mlp}

Memory-level parallelism: the number of long-latency loads in flight at the same time. A
larger window exposes more of it when the misses are independent. In gem5 it is measured as
the mean number of allocated L1D MSHRs over cycles with at least one. See
[Background](../concepts/background.md#when-a-larger-window-helps).

### MLP_target {#mlp-target}

$MLP_{target} = \min(MSHR_{L1D},\ L_{mem} \cdot rate / D_{indep})$: the number of misses the
cost model wants in flight, bounded by the L1D [MSHRs](#mshr). Kept fractional. See
[Compiler, Cost model](../guide/compiler.md#cost-model).

### MSHR {#mshr}

Miss status holding register: a cache entry that tracks one outstanding miss. The L1D MSHR
count bounds how many misses can be in flight. See [MLP_target](#mlp-target).

## O

### Oracle {#oracle}

The best window configuration per region, found by running the `oracle` build (region markers
only) under every static configuration and picking, per region, the configuration with the
best IPC (or ED²P) from `region_stats.csv`. Baseline [B1](#b1); the `small`-input sweep also
labels B5's data and gives B7's map. See
[Running experiments, Step 3](../guide/usage.md#step-3-oracle-sweeps).

### Out-of-order window {#out-of-order-window}

See [Window](#window).

## P

### P-core {#p-core}

Performance core of an Intel hybrid CPU: a large out-of-order core (Raptor Cove, 512-entry ROB
on the test machine, CPUs 0–3 with SMT). WinHint sends a thread there for a large
`setwin(W)`. See [Microarchitecture parameters](../guide/hardware/uarch-params.md).

### PGO {#pgo}

Profile-guided optimization: compiling with information from a profiling run. The `pgo`
variant is baseline [B7](#b7), which profiles the `small` input.

### PIE {#pie}

Performance Impact Estimation (Van Craeynest et al., ISCA 2012): estimate from counters how
much a thread would gain on the big core, and schedule accordingly. Reimplemented as the
reactive migration daemon of [R4](#r4).

## R

### R0 {#r0}

Pinning with `taskset`: always on the P-cores (R0-P) or always on the E-cores (R0-E). See
[Baselines](../guide/baselines/index.md#real-hardware-baselines).

### R1 {#r1}

The stock Linux scheduler ([EEVDF](#eevdf) with [ITMT](#itmt) and [HFI](#hfi) hints), run
unpinned. The reference that real-hardware results are reported against. See
[Baselines](../guide/baselines/index.md#real-hardware-baselines).

### R2 {#r2}

Hybrid-aware [sched_ext](#sched_ext) schedulers (`scx_bpfland`, `scx_cosmos`, `scx_lavd`) in
power-saving presets. Public code, reused; needs root. See
[Baselines](../guide/baselines/index.md#real-hardware-baselines).

### R3 {#r3}

Intel's low-power mode daemon ([lpmd](#lpmd)). Public code, reused; needs root. See
[Baselines](../guide/baselines/index.md#real-hardware-baselines).

### R4 {#r4}

Reactive counter-driven migration in the style of [PIE](#pie): a user-space daemon estimates
the P/E benefit from counters each interval and migrates the thread. Reimplemented. See
[Real hardware](../guide/hardware/index.md).

### R5 {#r5}

Phase-based tuning for asymmetric multicores (Sondag and Rajan, CGO 2011): static typing of
code regions, run-time sampling of each type on both core types, then assignment.
Reimplemented as a [libwinhint](#libwinhint) mode. See [Real hardware](../guide/hardware/index.md).

### RAPL {#rapl}

Running Average Power Limit: Intel's energy counters (package, core and other domains). The
source of energy and EDP on real hardware; reading them needs root or a temporary grant. See
[Running experiments, Part 2](../guide/usage.md#part-2-real-hardware).

### Region {#region}

A top-level loop nest of a defined function: the unit WinHint analyzes and hints. Ids are
assigned deterministically, so every build of the same source agrees on them. See
[Compiler, Outputs](../guide/compiler.md#outputs).

### Region marker {#region-marker}

The hint `region(id)`: a no-op that tells gem5 (or libwinhint) that region `id` starts. Used
by the oracle build, B7, R5 and the per-region statistics. See
[Interfaces §2](../interfaces.md#2-the-hint-isa-contract).

### Residency {#residency}

The fraction of a run's cycles spent in each window configuration. The energy model weights
each configuration's energy by it. See
[Evaluation methodology](../concepts/methodology.md#how-energy-is-estimated-in-gem5).

### ROB {#rob}

Reorder buffer: holds every in-flight instruction in program order until it commits. Its size
is the window size W. See [Background](../concepts/background.md#the-out-of-order-window).

## S

### sched_ext {#sched_ext}

The Linux extensible scheduler class: schedulers written as BPF programs and loaded at run
time. Baseline [R2](#r2) uses schedulers from the public `scx` project.

### SE mode {#se-mode}

Syscall-emulation mode of gem5: it runs a user-space binary without booting an operating
system and emulates its system calls. All gem5 runs here use it. See
[gem5 model](../guide/gem5/index.md).

### setwin {#setwin}

The hint `setwin(W)`: "use at most a W-entry window from here on", advisory. A RISC-V
`ori x0, x0, imm` or a unique x86 multi-byte NOP, both no-ops on unmodified cores; `W = 0`
releases the limit. gem5 selects the smallest configuration with ROB ≥ W. See
[Interfaces §2](../interfaces.md#2-the-hint-isa-contract).

### Sidecar {#sidecar}

A JSON file the compiler writes next to a binary, such as `<kernel>.winhint.json` or
`<kernel>.jones.json`. It records the target machine and the knobs used, so the experiment
runner can refuse a binary built for another machine or with untuned knobs. See
[Compiler statistics](stats-schema.md).

### Small and large inputs {#small-large-input}

The two inputs of every kernel, with the same model shapes. `small` shrinks only the input,
is simulated in full and is used for training and tuning; `large` is the evaluation input,
simulated with region-aligned sampling. See
[Evaluation methodology](../concepts/methodology.md#workloads-and-inputs).

### SMT {#smt}

Simultaneous multithreading: two hardware threads per core. The test machine's P-cores have
it; real-hardware runs are reported with SMT on and off.

### SQ {#sq}

Store queue: holds in-flight stores until they write to the cache. With the [LQ](#lq) it
forms the [LSQ](#lsq).

### Switch cost {#switch-cost}

The cost of changing the window, used by WinHint's placement to decide whether a change pays:
a few cycles plus a drain in gem5, tens of µs for a P/E migration. See
[Compiler, Placement](../guide/compiler.md#placement).

## V

### Variant {#variant}

A build flavor of the benchmarks Makefile (`VARIANT=plain`, `winhint`, `oracle`, `pgo`, …),
and by extension a row of the evaluation matrix (`static_c0`, `mlp`, `winhint_hw`, …), which
pairs a binary with a window policy. See [Workloads](../guide/workloads.md#variants) and
[Running experiments](../guide/usage.md#how-each-variant-runs).

## W

### W* {#w}

$W^*$: the window demand the cost model predicts for a region,
$\min(W_{max}, \max(\lceil MLP_{target} \cdot D_{indep}\rceil, \lceil CP \cdot rate\rceil))$.
It maps to the smallest window configuration with ROB ≥ W\*. See
[Compiler, Cost model](../guide/compiler.md#cost-model).

### Window {#window}

The out-of-order window: the instructions a core holds between dispatch and commit, bounded
by the ROB, IQ, LQ and SQ. "Window size W" means the ROB size, with the other structures
scaled by the [window configuration](#window-configuration). See
[Background](../concepts/background.md#the-out-of-order-window).

### Window configuration {#window-configuration}

One set of ROB, IQ, LQ and SQ sizes that are resized together; `c0` is the smallest, `c3` the
largest. On `riscv_ooo`, c0 is 64/32/16/16 and c3 is 256/128/64/64. See
[Interfaces §3](../interfaces.md#3-window-configuration-table).

### Window policy {#window-policy}

The gem5 parameter `window_policy` that decides when to change the window configuration:
`static`, `occupancy`, `mlp`, `bbv`, `lut`, `hint`, `hybrid` or `ltp`. All share one resize
mechanism. See [Window policies](../guide/gem5/policies.md).

### Window table {#window-table}

The four [window configurations](#window-configuration) of a machine, in the `"window"`
section of `sim/machines/*.json`. Read by both gem5 and the compiler. See
[Interfaces §3](../interfaces.md#3-window-configuration-table).

### WinHint+HW {#winhint-hw}

The hybrid policy (`hybrid`, variant `winhint_hw`): the hint sets a ceiling, and an MLP and
occupancy rule may move the window below it. This project's design, not from a paper
([Deviations §13.9](../deviations.md#139-winhinthw-hybrid-policy-is-our-design)).

## Next

- [Background](../concepts/background.md) and [How WinHint works](../guide/architecture.md)
  for the concepts behind these terms.
- [Interfaces](../interfaces.md) for the exact encodings and formats.
