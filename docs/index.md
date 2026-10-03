# WinHint — Know Your Window

**Static analysis of memory-level parallelism to drive microarchitectural reconfiguration.**

WinHint is a compiler-directed approach to sizing the out-of-order window (ROB, issue queue,
load/store queues) for ML inference. For these workloads the compiler already knows the
phases: attention, FFN, layer norm and softmax are separate loop nests with affine accesses,
known trip counts and known working sets. A static, loop-level cost model estimates, for each
region,

- $L_{mem}$, the expected miss latency per load class (footprint versus cache capacity);
- $D_{indep}$, the distance between independent long-latency loads;
- $CP$, the critical-path length of the loop body,

and derives a window demand

$$
W^* = \min\bigl(W_{max},\ \max(\lceil MLP_{target} \cdot D_{indep} \rceil,\ \lceil CP \cdot rate \rceil)\bigr),
\qquad rate = \min(issue\_width,\ body / II),
$$

where $II$ is the loop's initiation interval and
$MLP_{target} = \min(MSHR_{L1D},\ L_{mem} \cdot rate / D_{indep})$. For issue-bound loops the
critical-path term is $CP \cdot issue\_width$ ([Compiler](guide/compiler.md#cost-model)).

A placement pass then emits advisory `setwin(W)` hints: a RISC-V HINT-space
`ori x0, x0, imm`, or a unique x86 multi-byte NOP. Both are no-ops on unmodified cores.

The claims under evaluation ([design proposal](reference/proposal.md)) are that the hints

- match or beat reactive hardware window resizing;
- need no training data;
- switch exactly at region boundaries;
- carry over to a new core by recompiling with that core's parameters.

They are evaluated in gem5 (RISC-V O3 with window resizing) and on Intel hybrid silicon
(P-core/E-core migration). How each claim is tested is in
[Evaluation methodology](concepts/methodology.md).

## How to read this site

The site is organized as Concepts → Getting started → User guide → Reference → API →
Contributing. Pick the path that matches what you want to do.

**New to the project.** Read the Concepts in order: they assume computer architecture
basics and nothing about WinHint.

1. [Background](concepts/background.md): the out-of-order window, memory-level parallelism,
   why inference phases need different windows, and what a compiler can know.
2. [How WinHint works](guide/architecture.md): the components and how data flows between
   them.
3. [Evaluation methodology](concepts/methodology.md): the claims, platforms, baselines,
   metrics and success criteria.

Unfamiliar terms are defined in the [Glossary](reference/glossary.md).

**Reproducing the results.** [Installation](getting-started/installation.md) →
[Quickstart](getting-started/quickstart.md) → [Running experiments](guide/usage.md), which
lists every step of the gem5 study, the real-hardware campaign and the full reproduction.
Read [Evaluation methodology](concepts/methodology.md) first if you want to know why each
step exists.

**Extending the code.** [How WinHint works](guide/architecture.md) → the guide for the
component you change ([Compiler](guide/compiler.md), [gem5 model](guide/gem5/index.md),
[Baselines](guide/baselines/index.md), [Real hardware](guide/hardware/index.md),
[Workloads](guide/workloads.md)) → [Interfaces](interfaces.md) and the
[API reference](api/index.md) → [Commit gate and coverage](contributing/quality.md) and
[Writing documentation](contributing/documentation.md).

<div class="grid cards" markdown>

-   **Concepts**

    ---

    The problem, the design and how it is evaluated, for a first-time reader.

    [Background](concepts/background.md) ·
    [How WinHint works](guide/architecture.md) ·
    [Methodology](concepts/methodology.md)

-   **Getting started**

    ---

    Create the host environment (micromamba, conda-forge only, no Docker), check it and
    run a first kernel.

    [Installation](getting-started/installation.md) ·
    [Quickstart](getting-started/quickstart.md)

-   **User guide**

    ---

    How to build and run each part: the experiment runbook, workloads, compiler plugins,
    gem5 model, baselines, simulator fidelity, real hardware, testing.

    [Running experiments](guide/usage.md) ·
    [gem5 model](guide/gem5/index.md)

-   **Reference**

    ---

    The cross-component contract (hint ISA, window table, file formats), statistics
    schema, toolchain pins, documented deviations, glossary and the design proposal.

    [Interfaces](interfaces.md) · [Deviations](deviations.md) ·
    [Glossary](reference/glossary.md)

-   **API**

    ---

    Generated from the code: every Python module (Google-style docstrings) and every
    C/C++ file (Doxygen).

    [Python](api/python/index.md) · [C/C++](api/cpp/index.md)

-   **Contributing**

    ---

    The commit gate, coverage rules, documentation conventions and implementation status.

    [Commit gate](contributing/quality.md) ·
    [Writing documentation](contributing/documentation.md) ·
    [Project status](checklist.md)

</div>

!!! warning "Read the contract first"
    The cross-component contract (hint encoding, window table, gem5 flags, file formats) is
    [Interfaces](interfaces.md). Change it only together with every consumer.

## Citation

```bibtex
@inproceedings{winhint,
  title     = {Know Your Window: Static Analysis of Memory-Level Parallelism to Drive
               Microarchitectural Reconfiguration},
  author    = {[Authors]},
  booktitle = {[Venue]},
  year      = {[Year]}
}
```

## License

WinHint is licensed under the Apache License, Version 2.0 ([`LICENSE`](../LICENSE)). The
Clairvoyance submodule keeps its own license (`third_party/clairvoyance/LICENSE`).
