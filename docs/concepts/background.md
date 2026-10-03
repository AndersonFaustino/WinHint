# Background

This page explains the problem WinHint addresses, for a reader who knows computer
architecture basics but not this project. It covers what the out-of-order window is, when a
larger window pays off, why ML inference changes its needs from phase to phase, why hardware
that reacts to those changes lags behind them, and what a compiler can know ahead of time.
[How WinHint works](../guide/architecture.md) builds on it.

## The out-of-order window

An out-of-order (OoO) core fetches and decodes instructions in program order, executes them
as soon as their operands are ready, and retires (commits) them in program order again. The
instructions it holds between dispatch and commit form its *window*. Four structures bound
the window:

| Structure | Holds |
|-----------|-------|
| [ROB](../reference/glossary.md#rob) (reorder buffer) | every in-flight instruction, in program order, until it commits |
| [IQ](../reference/glossary.md#iq) (issue queue) | dispatched instructions waiting for their operands or a functional unit |
| [LQ](../reference/glossary.md#lq) (load queue) | in-flight loads |
| [SQ](../reference/glossary.md#sq) (store queue) | in-flight stores until they write to the cache |

LQ and SQ together are the load/store queue ([LSQ](../reference/glossary.md#lsq)). The
physical register file is a fifth limit: every instruction that writes a register needs a
free physical register to rename into. When any of these structures is full, dispatch stops.
In this project the register file is sized so that it never fills before the window does
([Deviations §1.5](../deviations.md#15-register-file-sized-to-never-bind)), so the
ROB/IQ/LQ/SQ sizes are what bind.

In the rest of the site, "window size W" means the ROB size; the IQ, LQ and SQ scale with it
(see [window configuration](../reference/glossary.md#window-configuration)).

## When a larger window helps

A load that misses in the last-level cache takes hundreds of cycles; the default simulated
machine estimates an L2 miss at 200 cycles (`memory.latency_cycles` in
`sim/machines/riscv_ooo.json`). Because commit is in order, the missing load stays at the
head of the ROB, and the window fills up behind it. The core keeps working only on the
instructions that already fit in the window. If another, *independent* long-latency load is
among them, its miss overlaps with the first one. The number of misses in flight at once is
the **memory-level parallelism** ([MLP](../reference/glossary.md#mlp)).

The window is what exposes MLP. If independent misses are $D_{indep}$ instructions apart, a
window of $W$ instructions can hold about $W / D_{indep}$ of them. Two other limits apply:
the number of outstanding misses the L1 data cache can track (its
[MSHRs](../reference/glossary.md#mshr)), and the number of misses worth overlapping at all,
which depends on the miss latency $L_{mem}$ and on how fast the core issues instructions.
Streaming over a large array, gathering rows of a weight matrix, or reading a big KV cache
are typical MLP-rich patterns.

A larger window also helps compute-bound code up to a point: it must be long enough to cover
the dependence critical path ([CP](../reference/glossary.md#cp)) of the loop body at the
core's issue rate.

## When a larger window only costs energy

A larger window does not help when nothing in it can overlap:

- **Dependent misses.** In a pointer chase each load's address comes from the previous load,
  so only one miss is ever in flight, whatever the window size.
- **Cache-resident code.** Without long-latency loads, a window that already covers the
  critical path at full issue width adds nothing.
- **Low-ILP code.** Serial dependence chains or frequent branch mispredictions keep the
  window from filling usefully.

In these cases the extra ROB, IQ and LSQ entries still cost power: the IQ in particular is a
content-addressable structure searched every cycle. A core that could shrink its window in
such phases, and grow it only where MLP exists, would save energy at little or no performance
loss. That trade-off is what the evaluation measures as energy, [EDP](../reference/glossary.md#edp)
and [ED²P](../reference/glossary.md#ed2p).

## Why ML inference phases differ

An inference layer is a sequence of distinct kernels, and they stress the core differently:

- **FFN and other linear layers** multiply activations by weight matrices. A transformer
  layer in this project carries 24–28 MB of weights, far beyond a 256 kB–1 MB L2, so these
  phases stream weights from memory: many independent misses, the case where a large window
  pays.
- **Attention** computes query–key scores and the weighted sum over values. Its working set
  grows with the sequence length (and with the KV cache in decoders), so whether it misses
  depends on the input shape.
- **Layer norm and softmax** are row-wise reductions and element-wise maps (with library
  calls such as `expf`) over small working sets. They are bound by dependence chains and
  latency, not by memory, so a small window is usually enough.

These phases alternate within every layer, many times per inference. A single fixed window
size is therefore either too large for the normalization phases or too small for the
memory-bound ones. The kernels used here are described in [Workloads](../guide/workloads.md).

## Why reactive hardware lags

The classic answer is to let the hardware resize the window at run time. Such policies
sample counters (occupancy, cache misses, MLP) over a period, then decide. In the gem5 model
the sampling period is 1000 cycles (`window.period` in the machine JSON). Whatever the rule,
a reactive policy:

- **detects a phase change only after it happened**, at least one sampling period late, and
  runs that period with the previous phase's window;
- **may oscillate** near its thresholds, and pays a switch cost every time it does;
- **needs warm-up or training** when it learns per-phase configurations (phase predictors,
  learned lookup tables), and the learned state is tied to one machine.

The shorter the phases are compared with the sampling and learning time, the more this lag
costs. The reactive designs WinHint is compared with are the baselines B2–B5 and B9
([Evaluation methodology](methodology.md#baselines)).

## What a compiler can know

For ML inference the phases are not hidden: each one is a separate function with its own
loop nest, and the compiler sees them all. For such code a static analysis can determine,
before the program runs:

- **affine accesses**: array indices that are linear in the loop counters, so the stride and
  the set of touched cache lines are known;
- **trip counts**: from the loop bounds, often compile-time constants or call-site
  constants (model dimensions);
- **working sets**: the footprint and reuse distance of each group of accesses, which,
  compared with the cache capacities, say which loads will miss and where;
- **dependences**: which loads depend on earlier loads (a chase) and the critical path of
  the loop body.

From these, WinHint's cost model estimates each region's window demand $W^*$ and places a
hint at the region's entry, so the switch happens exactly at the boundary. The analysis is
not perfect: it does not see cache interference between phases or data-dependent behavior,
and it assumes a default when a trip count is unknown
([Deviations §13.4](../deviations.md#134-trip-counts-from-call-site-constants-unknown-trips-assumed-1000)).
That is why the hint is *advisory*: the hardware may ignore it, and an unmodified core treats
it as a no-op. The cost model itself is in [Compiler](../guide/compiler.md#cost-model).

## The two mechanisms evaluated

The same compiler output drives two different mechanisms:

1. **Window resizing in gem5.** A modified gem5 O3 core (RISC-V, syscall-emulation mode)
   resizes ROB, IQ, LQ and SQ together between the four configurations of a
   [window table](../reference/glossary.md#window-table). A `setwin(W)` hint selects the
   smallest configuration with ROB ≥ W. Shrinking stops dispatch until occupancy falls below
   the new size; growing is immediate. A switch costs a few cycles plus that drain. No real
   CPU exposes this control, which is why it is evaluated in simulation
   ([gem5 model](../guide/gem5/index.md)).
2. **P-core/E-core migration on a hybrid CPU.** An Intel hybrid processor has large
   performance cores ([P-cores](../reference/glossary.md#p-core), a 512-entry ROB on the
   test machine) and small efficiency cores ([E-cores](../reference/glossary.md#e-core), a
   256-entry ROB). Moving a thread between them is a coarse way to change its window. The
   runtime library `libwinhint` turns a large `setwin(W)` into a move to the P-cores and a
   small one into a move to the E-cores. A migration costs tens of µs, so the compiler's
   placement is told a much larger switch cost
   ([Real hardware](../guide/hardware/index.md)).

## Next

- [How WinHint works](../guide/architecture.md): the components and how data flows between
  them.
- [Evaluation methodology](methodology.md): how the claims are tested.
- [Glossary](../reference/glossary.md): every term and abbreviation.
