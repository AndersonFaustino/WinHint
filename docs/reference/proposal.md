# Design proposal — Know Your Window

!!! note "About this document"
    The research proposal that started the project: the problem, the claims, the baselines,
    the roadmap (§5) and the evaluation plan (§7–§8). It is kept as written; where the
    implementation departs from it, [Deviations](../deviations.md) records the difference and
    why. The current state is in [Project status](../checklist.md); the methodology as
    implemented is in [Evaluation methodology](../concepts/methodology.md).

**Target venue:** ACM SIGPLAN International Conference on Compiler Construction (CC). About 10 pages plus references, ACM `acmsigplan` format, artifact evaluation encouraged.
**Fallbacks:** CGO, LCTES; TACO for the extended version.
**Deadline:** check the CC 2027 CFP (usually early November).

---

## 1. Title, short name and thesis

**Title:** *Know Your Window: Static Analysis of Memory-Level Parallelism to Drive Microarchitectural Reconfiguration*

**Project name:** **WinHint**. The same name is used for the repository, the micromamba environment, the compiler pass, the ISA hint and the runtime.

**Thesis.** For ML inference, the compiler already knows the phase. Attention, FFN, layernorm and softmax are separate loop nests with affine accesses, known trip counts and known working sets. A static, loop-level cost model can predict how much each region benefits from a given out-of-order window (ROB, IQ, LSQ and physical registers). The model uses:
- memory-level parallelism (MLP), meaning how many long-latency loads can be in flight at once;
- the length of the dependence critical path;
- the cache footprint.

Emitting advisory reconfiguration hints from that model:
- **(a)** matches or beats reactive hardware predictors;
- **(b)** needs no training data;
- **(c)** switches exactly at region boundaries, with no detection lag;
- **(d)** carries over to another microarchitecture by changing its parameters, with no retraining.

**Novelty claim.** WinHint is the first *static MLP-window cost model* for ML inference loop nests that drives *resizing of the whole OoO window* through an advisory ISA hint. We evaluate it against:
- reactive hardware resizing;
- hardware phase prediction;
- learned predictors;
- earlier compiler-directed issue-queue resizing.

We also show it on real hybrid (P-core/E-core) silicon.

---

## 2. Starting point: the current codebase and its problems

The WinHint repo starts from an earlier adaptive-ROB prototype. It has a gem5 v25.1 RISC-V O3 setup, 15 C inference kernels, and a reactive pipeline. The pipeline runs in five steps:
1. It samples 4 hardware counters every 500 cycles.
2. It labels phases with threshold rules.
3. It trains a tiny Transformer classifier on those labels.
4. It exports the classifier as a 4-D lookup table (LUT).
5. At run time it resizes the ROB to 64, 128 or 256 entries.

The infrastructure is reused: the gem5 patch flow, the experiment runners, McPAT and the plotting scripts. The Docker setup is dropped in favor of a host micromamba environment (§5, Phase 0). The reactive LUT becomes **one of the baselines**. The analysis found these problems, which must be fixed before any result is credible:

| # | Problem | Where |
|---|---------|-------|
| 1 | The labels are circular: threshold rules produce them and the Transformer just re-learns the thresholds. | `sim/baselines/lut/label_phases.py` → `train_phase_classifier.py` |
| 2 | The "D-cache miss rate" is really squashed loads and stores divided by all accesses, which is not a miss rate. | `sim/patches/o3_cpu_adaptive_rob_cc.patch` |
| 3 | The counters are cumulative, not per window. IPC keeps its state in `static` locals, and `predict()` is called three times per update. | same |
| 4 | Resizing only writes `rob.numEntries`. It ignores the per-thread `maxEntries` and never drains the ROB when shrinking. | same |
| 5 | `prefetcher->setPrefetchDistance()` is not a gem5 API, and no prefetcher is attached. | same; `sim/se.py` |
| 6 | The ROB is not the real limit: 96 physical registers (about 64 for renaming), IQ=64 and LQ/SQ=32 mean a 256-entry ROB can never fill, so the sweep is confounded. | `sim/se.py:94-100` |
| 7 | "Attention → small ROB" goes against theory: MLP-rich, memory-bound phases usually gain the most from a large window. | README |
| 8 | The build breaks on a filename mismatch (`gem_v25…` vs `gem5_v25…`), and `phase_lookup.h` is not committed. | `sim/patches/`, `tooling/winhint.sh` |
| 9 | The kernels are toy-sized (seq=32), built only at `-O2`, and 6 of the 15 are not transformers. | `benchmarks/` |

---

## 3. Contributions

1. **Window-demand analysis.** An LLVM analysis pass over loop nests that uses ScalarEvolution, DependenceAnalysis, LoopInfo and a footprint/reuse-distance cache model. The cache model is driven by a JSON target description (an extended `sim/machines/riscv_ooo.json`). For each region it computes three quantities:
   - `L_mem`: the expected miss latency per load class, from footprint versus cache capacity;
   - `D_indep`: the distance in dynamic instructions between independent long-latency loads, from the loop body size, the unroll factor and loop-carried dependences;
   - `CP`: the critical-path length of the loop body's dependence DAG.

   From these it derives `W* = min(W_max, max(ceil(MLP_target · D_indep), ceil(CP · rate)))` and maps `W*` to a discrete window configuration. Here `rate = min(issue_width, body / II)` is the loop's sustainable issue rate (II from recurrences, issue width and dividers) and `MLP_target = min(MSHR_L1D, L_mem · rate / D_indep)`; for issue-bound loops the critical-path term is `CP · issue_width`.
2. **Hint placement.** A dynamic program over the region tree (the loop nest plus call-graph summaries). It places `setwin` hints so that the benefit of each change outweighs its switch cost. It adds hysteresis to avoid thrashing, hoists hints out of hot loops, and summarizes library calls such as `expf` and `tanhf`. The switch cost is a parameter, so the same algorithm serves both mechanisms:
   - gem5 window resizing, which costs a few cycles plus a drain;
   - P/E-core migration on real hardware, which costs tens of µs.
3. **ISA and µarch contract.**
   - **Encoding:** a RISC-V HINT-space instruction, for example `ori x0, x0, imm`, which is a no-op on unmodified cores.
   - **Semantics:** "maximum window W, advisory".
   - **gem5 O3 implementation:** ROB, IQ and LSQ are resized together. Shrinking is safe: dispatch stops until occupancy falls to the new size.
   - **x86 twin:** a unique multi-byte NOP.
4. **Evaluation against published baselines** (§4). It is done in gem5 and on real Intel hybrid silicon, and reports performance, energy/ED²P, hint count and dynamic overhead, code size, compile time and switch frequency.
5. **Portability study.** Only the target parameters are recompiled, for 3 core configurations. The learned LUT has to be retrained for each one; WinHint does not.

---

## 4. Baselines

**Rule:** if a public artifact exists, we use its code. If none exists, we reimplement the baseline from the paper, inside our infrastructure, and document every deviation.

The artifact check was done in September 2026. **None of the hardware window-resizing papers has a public artifact.** The only research code found is Clairvoyance/SWOOP and PMCTrack, plus production schedulers (sched_ext, intel-lpmd).

### 4.1 gem5 baselines (RISC-V O3)

All O3-side baselines are modes of a single new SimObject parameter, `window_policy` (a string, since `static` is a C++ keyword), which replaces the boolean `adaptive_rob`. Each mode uses the **same resize mechanism** (Phase A) and the **same configuration table**, so only the *decision policy* differs.

| ID | Baseline | Citation | Artifact | Action |
|----|----------|----------|----------|--------|
| B0 | Static windows: small, medium and large (for example 64, 128, 192 and 256, with IQ/LSQ/registers scaled) | — | — | Config only: the window tables in `sim/machines/*.json` and `window_policy=static` (`sim/gem5/.../static_policy.*`) |
| B1 | **Oracle**: the best configuration per region, from exhaustive per-region simulation. This is the upper bound. | — | — | **Implement** in `sim/baselines/oracle/`: the compiler emits `region(id)` markers (`oracle` build variant); `oracle_sweep.py` runs every static configuration, takes the best per region from `region_stats.csv`, and writes the per-region map that the `oracle_hinted` build turns into `setwin` hints (`window_policy=hint`) |
| B2 | **Occupancy-driven resizing of IQ, ROB and LSQ** | Ponomarev, Kucuk, Ghose, "Reducing Power Requirements of Instruction Scheduling Through Dynamic Allocation of Multiple Datapath Resources", MICRO-34, 2001 | none found (originally AccuPower/SimpleScalar) | **Implement** as `window_policy=occupancy` (`sim/gem5/.../occupancy_policy.*`): sample occupancy per period, shrink when under-used, grow on dispatch stalls |
| B3 | **MLP-aware window resizing**, the closest hardware competitor | Kora, Yamaguchi, Ando, "MLP-Aware Dynamic Instruction Window Resizing for Adaptively Exploiting Both ILP and MLP", MICRO-46, 2013 | none found (in-house simulator on SimpleScalar) | **Implement** as `window_policy=mlp` (`sim/gem5/.../mlp_policy.*`): enlarge on an L2 miss when MLP is detected, shrink otherwise |
| B4 | **Basic-block-vector (BBV) phase tracking and prediction** | Sherwood, Sair, Calder, "Phase Tracking and Prediction", ISCA 2003 | none found (SimpleScalar/Wattch) | **Implement** as `window_policy=bbv` (`sim/gem5/.../bbv_policy.*`): BBV signature, phase table, next-phase predictor, and a per-phase best configuration learned online |
| B5 | **Learned counter-based predictor** (the current reactive LUT, after fixes), representing ML-based adaptation in the style of Dubach et al. | Dubach, Jones, Bonilla, "Dynamic Microarchitectural Adaptation Using Machine Learning", TACO 10(4), 2013 (no artifact found); our own code | **ours** | **Reuse** `sim/baselines/lut/*` (offline labeling, training, LUT export) with a new run-time `window_policy=lut` (`sim/gem5/.../lut_policy.*`), after fixing problems 1–5 and labeling with the oracle's best configuration rather than threshold rules. gem5 loads the exported LUT at run time, so the compiled-in `phase_lookup.h` and `phase_predictor.cc` go away |
| B6 | **Compiler-directed issue-queue resizing**, the closest compiler prior work | Jones, O'Boyle, Abella, González, "Software Directed Issue Queue Power Reduction", HPCA 2005; extended as "Compiler Directed Issue Queue Energy Reduction", Trans. HiPEAC 4(1), 2009 | none found (Wattch/SimpleScalar) | **Implement** as a second LLVM pass, `compiler/baselines/jones_iq/`, following the paper's DAG-based IQ-demand analysis. Run it (i) as published, IQ only, and (ii) extended to ROB and LSQ. Both use the same hint instruction |
| B7 | **Profile-guided positional adaptation**: pick the best configuration per code region from a profiling run, and test on a different input | Huang, Renau, Torrellas, "Positional Adaptation of Processors", ISCA 2003; Lau, Perelman, Calder, "Selecting Software Phase Markers with Code Structure Analysis", CGO 2006 | none found | **Implement** as a PGO flow: profile on the small input, emit hints, evaluate on the large input. Tests *static model versus profiling* |
| B8 | **Compile-time MLP transformation** with no hardware change (complementary; also combined as WinHint+Clairvoyance) | Tran, Carlson, Koukos, Själander, Spiliopoulos, Kaxiras, Jimborean, "Clairvoyance: Look-ahead Compile-time Scheduling", CGO 2017 | **[github.com/ktran/clairvoyance](https://github.com/ktran/clairvoyance)** (LLVM 3.8, artifact-evaluated) | **Reuse the artifact.** Run its passes with LLVM 3.8 to produce transformed IR, then lower with LLVM 23 to RISC-V and x86. First check bitcode compatibility; if it fails, port the pass to LLVM 23 and document the port |
| B9 | Long-Term Parking: criticality-aware allocation | Sembrant, Carlson, Hagersten, Black-Schaffer, Perais, Seznec, Michaud, MICRO-48, 2015 | none found (originally gem5 x86 FS) | **Implement** as `window_policy=ltp` (`sim/gem5/.../ltp/`): a parking queue between rename and dispatch holds non-urgent instructions, so the IQ and LSQ fill only with critical ones. Urgency is marked by a backward-slice table from long-latency loads and branches, as in the paper. It shares the window tables so it can be compared on equal resources |

**Related work that is cited but not used as a baseline:**
- Folegnani & González, ISCA 2001 (IQ-only, subsumed by B2);
- Dhodapkar & Smith, ISCA 2002 (an alternative phase detector to B4);
- Petoumenos et al., ARCS 2010 (MLP-aware IQ resizing);
- Hsu & Kremer, PLDI 2003 (compiler-directed DVFS);
- SWOOP (Tran et al., **PLDI 2018**, Sniper-based);
- Stretch (Margaritov et al., HPCA 2019);
- Composite Cores (Lukefahr et al., MICRO 2012).

### 4.2 Real-hardware baselines (Intel Core 5 120U: CPUs 0–3 are P-cores with SMT, CPUs 4–11 are E-cores; RAPL available)

| ID | Baseline | Citation / source | Artifact | Action |
|----|----------|-------------------|----------|--------|
| R0 | Always on a P-core; always on an E-core | — | — | `taskset` |
| R1 | **Stock Linux scheduler** (EEVDF with ITMT and HFI/Thread Director hints) | mainline Linux. Note: EAS in `intel_pstate` (6.16) is **disabled on SMT hybrids** such as this chip. Report both SMT on and SMT off. | kernel | Use as-is; record the kernel version and config |
| R2 | **Hybrid-aware sched_ext scheduler** | [github.com/sched-ext/scx](https://github.com/sched-ext/scx) (`scx_bpfland` or `scx_cosmos` in power-save mode; `scx_lavd` as an alternative) | **public** | **Reuse** |
| R3 | **Intel Low Power Mode Daemon** | [github.com/intel/intel-lpmd](https://github.com/intel/intel-lpmd) | **public** | **Reuse** if the platform supports it; otherwise explain why it was dropped |
| R4 | **Reactive counter-driven migration** (PIE-style) | Van Craeynest, Jaleel, Eeckhout, Narvaez, Emer, "Scheduling Heterogeneous Multi-Cores through Performance Impact Estimation (PIE)", ISCA 2012; Isci, Contreras, Martonosi, MICRO-39, 2006 | none found | **Implement** a userspace policy on **PMCTrack** ([github.com/jcsaezal/pmctrack](https://github.com/jcsaezal/pmctrack), reused) or `perf_event_open`: estimate the P/E benefit from MLP/CPI counters each interval and migrate with `sched_setaffinity` |
| R5 | **Phase-based tuning for asymmetric multicores**, the closest code-region prior | Sondag & Rajan, "Phase-based Tuning for Better Utilization of Performance-Asymmetric Multicore Processors", CGO 2011 | none found | **Implement** the published flow: static code-region typing, then runtime sampling of each type on both core types, then assignment |

---

## 5. Implementation roadmap

**Phase 0: Rename, restructure and set up the environment.** The prototype still uses its legacy name (`transphase`) throughout; every occurrence becomes `winhint`. The tree is reorganized by component (§6): there is no `src/` directory, and `scripts/` becomes `tooling/`.
- **Environment:** drop Docker (`Dockerfile`, `docker-compose.yml`, `.env`, `docker_manager.sh`). Everything runs on the host from one micromamba environment, `environment.yml` (env `winhint`). **Every tool comes from conda packages** (conda-forge first, other public conda channels only when conda-forge lacks a package). Nothing comes from apt, pip wheels outside conda, or a local build of a toolchain. The environment contains:
  - Python 3.14 with numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, PyTorch (CPU) and pytest;
  - LLVM/Clang 23, as prebuilt conda packages: `clang`, `clangxx`, `clang-tools`, `llvmdev` (headers, CMake config and static libraries for the plugin), `llvm-tools`, `lld`, `compiler-rt` and `libcxx-devel` if needed, all pinned to the same 23.x version;
  - the gem5 and McPAT build dependencies (`scons`, `m4`, GCC 13 via `gxx_linux-64`, zlib, protobuf, gperftools, libpng, boost, hdf5, `pkg-config`, `ccache`, `make`, `cmake`, `ninja`, `git`);
  - the RISC-V toolchain: the conda cross compiler and sysroot (`gcc_linux-riscv64`/`gxx_linux-riscv64` with `sysroot_linux-riscv64`), or, if no cross GCC is packaged, LLVM 23's `clang --target=riscv64-unknown-linux-gnu` with the conda RISC-V sysroot and `compiler-rt` builtins;
  - `qemu` (user mode, for `qemu-riscv64`);
  - the real-hardware tools: `rust` (for building sched_ext schedulers), `libbpf`, `bpftool` and `perf` where conda packages exist; `libwinhint` itself uses `perf_event_open` directly and does not need `perf`.

  Only the **research artifacts** are compiled from source, with the environment's compilers: gem5 (patched), McPAT, the Clairvoyance passes, and the R2/R3/R4 baselines (sched_ext schedulers, intel-lpmd, PMCTrack). The PMCTrack kernel module is built against the host kernel's headers (`/lib/modules/$(uname -r)/build`) and loading it is opt-in. No tool or toolchain is compiled.
- **Environment script:** `tooling/create_conda_env.sh` creates the environment in one step. It:
  - installs micromamba into `~/.local/bin` if it is missing, with no sudo and no changes to shell rc files unless `--init-shell` is given;
  - creates the `winhint` env from its toolchain table (`environment.yml` is the declarative equivalent) and `requirements*.txt` under `MAMBA_ROOT_PREFIX` (default `~/.local/share/mamba`), or updates it if it already exists (`--update`), and can delete and recreate it (`--force`);
  - installs every tool from conda and compiles nothing. If a pinned package is missing from the channels, it stops and names the package;
  - creates a second, separate env, `winhint-llvm38`, with the conda LLVM 3.8 packages (`llvmdev=3.8`, `clang=3.8`) needed by the Clairvoyance artifact (B8);
  - writes an activation hook that sets `WINHINT_ROOT`, `WINHINT_BUILD` and `PATH`;
  - checks the result, by running `clang`, `llvm-config`, `scons`, the RISC-V compiler, `qemu-riscv64` and `python -c "import torch, pandas"` and compiling a RISC-V hello-world under QEMU, then prints how to activate the env.

  It is idempotent and stops at the first failing step with a clear message.
- **Pinned versions:** `tooling/versions.lock` pins Python 3.14, LLVM 23 and gem5 v25.1.0.0, and `create_conda_env.sh` fails if the installed versions differ.
- **Manager script:** `tooling/winhint.sh` replaces `transphase.sh` with commands `env:create` and `env:update` (both call `create_conda_env.sh`), `gem5:clone`, `gem5:build clean|winhint`, `mcpat:clone|build`, `benchmarks:build`, `compiler:build` and `status`. Pinned versions move from `pkg_version.lock` to `tooling/versions.lock`.
- **Build outputs:** everything fetched or generated goes under `build/` (gitignored, overridable with `WINHINT_BUILD`). That covers the gem5 clone and its builds, McPAT, the compiler plugins, benchmark binaries, the Clairvoyance passes and the R2–R4 baseline builds. Experiment data goes under `results/` (gitignored).
- **Moves** (`git mv`, then update every import, path and doc reference):

  | Current | New |
  |---------|-----|
  | `transphase.sh` | `tooling/winhint.sh` |
  | `scripts/apply_patches.sh` | `tooling/apply_patches.sh` |
  | `pkg_version.lock` | `tooling/versions.lock` |
  | `src/benchmarks/` | `benchmarks/` |
  | `src/transphase/{label_phases,train_phase_classifier,export_lookup_table}.py` | `sim/baselines/lut/` |
  | `src/transphase/plot_results.py` | `analysis/plot_results.py` |
  | `src/transphase/phase_predictor.cc` | removed (gem5 loads the LUT at run time) |
  | `src/benchmarks/bin/` (committed binaries) | removed (rebuilt into `build/`) |
  | `hw/baselines/pie_policy/`, `hw/baselines/sondag/` | `hw/baselines/r4_pie/`, `hw/baselines/r5_sondag/` |
- **Check:**
  - `git grep -i transphase` returns nothing, `src/` and `scripts/` no longer exist, and no file references them;
  - `tooling/create_conda_env.sh` succeeds on a clean machine, and a second run changes nothing;
  - `tooling/winhint.sh status` reports every tool.

**Phase A: Fix the simulation base (1–2 weeks).**
- Fix the patch filename mismatch and commit the `phase_lookup.h` generation flow.
- Rewrite counter sampling:
  - use real L1D/L2 miss deltas per window;
  - move state into CPU members;
  - call `predict()` once;
  - drop the fake prefetcher call, or attach a real `StridePrefetcher`.
- Implement a safe resize of ROB, IQ and LSQ (`rob.cc/hh`, `inst_queue`, `lsq`): set `maxEntries` and gate dispatch until occupancy is at or below the target.
- Rebalance the `sim/se.py` defaults (180+ physical registers, IQ/LSQ scaled) so the window configuration is actually binding.
- Replace `adaptive_rob` (bool) with `window_policy` (string) in `BaseO3CPU.py`.

**Phase B: Workloads (1 week).**
- Scale the kernels to BERT-base and GPT-2-small layer shapes (seq 128–512, working sets beyond L2).
- Build matrix: `-O2`/`-O3`, unrolling on/off, and tiled GEMM. Optionally add TVM- or IREE-generated C.
- Keep `regression_*` as a non-transformer control group.
- Provide two input sizes per kernel so that B7 can train on one and test on the other.

**Phase C: WinHint compiler (4–6 weeks), the core of the paper.**
- `compiler/winhint/`: an out-of-tree LLVM 23 new-pass-manager plugin, loaded with `clang -fpass-plugin`:
  - the `WindowDemandAnalysis` analysis pass;
  - the `HintPlacement` transform pass;
  - RISC-V HINT and x86 NOP emission.
- `compiler/baselines/jones_iq/` (B6) and a PGO flow, `compiler/baselines/pgo/` (B7).
- Clairvoyance (B8) pipeline integration, `compiler/baselines/clairvoyance/`.
- `benchmarks/Makefile` targets: `winhint`, `oracle`, `jones`, `pgo`, `clairvoyance`.

**Phase D: gem5 hint and policy support (3–4 weeks, including B9).**
- Decode the HINT in gem5's `src/arch/riscv/isa/decoder.isa` (through the `sim/patches/` patch). It signals the O3 CPU and stays a nop on the clean build.
- Add policy modes, one source pair each under `sim/gem5/src/cpu/o3/window/` behind a common policy interface: `static` (B0), `occupancy` (B2), `mlp` (B3), `bbv` (B4), `lut` (B5), `hint` (WinHint, B1 via `oracle_hinted`, B6, B7), `hybrid` (a hint plus a hardware override), and `ltp` (B9, which also needs the parking-queue stage in `ltp/`).
- Unit-check each policy with forced traces before the full runs.

**Phase E: gem5 evaluation (3 weeks).**
- Extend `sim/run_experiments.py` to cover every B-baseline, WinHint and WinHint+HW on ≥ 3 core configurations.
- Reuse `sim/estimate_energy.py` (McPAT) and `analysis/plot_results.py`. New figures:
  - predicted W* versus the oracle-best window;
  - switch frequency;
  - hint overhead;
  - sensitivity to cache size and memory latency.

**Phase E2: Real-silicon evaluation on the Core 5 120U (2–3 weeks).**
- **Hint overhead:** the x86 NOP-hint overhead, measured with `perf stat`, should be within noise. RISC-V hint binaries must produce bit-identical output under `qemu-riscv64` and on clean gem5. No RISC-V board is needed.
- **`libwinhint` runtime:** it turns hints at layer or operator boundaries into P/E migration. The DP placement uses the measured migration cost.
- **Comparison:** against R0–R5.
- **Energy and EDP:** from RAPL (package and core).
- **Method:** fixed governor, ≥ 10 repetitions, 95% confidence intervals; SMT on and off.
- **Optional:** operator-level hooks in llama.cpp or ONNX Runtime.

**Phase F: Writing and artifact (2 weeks).**
- Motivation (problems 6 and 7), model, placement, ISA/µarch, gem5 evaluation, real hardware, related work.
- Artifact built on `tooling/create_conda_env.sh`, `environment.yml` and `tooling/winhint.sh`.

**Timeline:** about 17–21 weeks.

---

## 6. Repository layout and critical files

The tree is organized by component. There is no `src/` directory. Everything that builds, installs or runs the toolchain is under `tooling/`.

```
WinHint/
├── PROPOSAL.md, README.md
├── docs/                         design notes; interfaces.md (hint ISA, gem5 params, file formats)
├── tooling/                      create_conda_env.sh, environment.yml, versions.lock, winhint.sh, apply_patches.sh
├── compiler/
│   ├── winhint/                  WinHint LLVM plugin: WindowDemandAnalysis, HintPlacement, hint emission
│   │                             (also emits the region(id) markers used by B1 and B7)
│   ├── baselines/
│   │   ├── jones_iq/             B6  compiler-directed IQ resizing (IQ-only and ROB/LSQ-extended)
│   │   ├── pgo/                  B7  positional adaptation: profile small input → hints for large input
│   │   └── clairvoyance/         B8  build of the LLVM 3.8 artifact and its glue to the modern pipeline
│   ├── common/                   shared target model (JSON) and hint emitter, used by winhint/ and jones_iq/
│   └── test/                     lit-style tests for the cost model and the hint encodings
├── benchmarks/                   inference kernels (*.c) and the Makefile with its build variants
├── sim/
│   ├── se.py                     gem5 SE configuration
│   ├── machines/                 B0  window tables and target descriptions (gem5 and the cost model)
│   ├── gem5/src/cpu/o3/window/   new gem5 sources, copied into the gem5 tree at build time
│   │   ├── controller.{hh,cc}        mechanism: ROB/IQ/LSQ caps, counters, traces, hint decode
│   │   ├── policy.hh                 common policy interface
│   │   ├── static_policy.*       B0
│   │   ├── occupancy_policy.*    B2  occupancy-driven resizing (Ponomarev et al.)
│   │   ├── mlp_policy.*          B3  MLP-aware resizing (Kora et al.)
│   │   ├── bbv_policy.*          B4  BBV phase tracking and prediction (Sherwood et al.)
│   │   ├── lut_policy.*          B5  run-time LUT lookup
│   │   ├── hint_policy.*             WinHint; also B1 (oracle_hinted), B6, B7
│   │   ├── hybrid_policy.*           WinHint+HW
│   │   └── ltp/                  B9  Long-Term Parking queue and urgency table (Sembrant et al.)
│   ├── patches/                  edits to existing gem5 files: CPU hooks, params, RISC-V decoder
│   ├── baselines/
│   │   ├── oracle/               B1  oracle_sweep.py: per-region best configuration
│   │   └── lut/                  B5  labeling, training, LUT export
│   ├── tests/                    mechanism and per-policy checks
│   ├── run_experiments.py, estimate_energy.py
├── hw/
│   ├── libwinhint/               runtime: hints → P/E migration (also the R5 mode)
│   ├── baselines/
│   │   ├── r0_taskset/           R0  always P / always E
│   │   ├── r1_stock/             R1  stock Linux scheduler
│   │   ├── r2_sched_ext/         R2  scx_bpfland / scx_cosmos / scx_lavd
│   │   ├── r3_lpmd/              R3  intel-lpmd
│   │   ├── r4_pie/               R4  PIE-style reactive migration (PMCTrack / perf)
│   │   └── r5_sondag/            R5  phase-based tuning (Sondag & Rajan)
│   ├── tools/                    measurement helpers
│   └── run_hw_experiments.py
├── analysis/                     plot_results.py and the figure scripts
├── third_party/clairvoyance      B8  submodule (artifact source)
├── build/                        (gitignored) gem5, McPAT, plugins, binaries
└── results/                      (gitignored) experiment data and figures
```

- **Modify:**
  - `sim/se.py`
  - `sim/run_experiments.py`
  - `sim/estimate_energy.py`
  - `benchmarks/Makefile`
  - `benchmarks/*.c`
  - `analysis/plot_results.py`
  - every file that still carries the legacy project name or an old path (Phase 0)
- **Replace:** the gem5 patches in `sim/patches/` with a single `gem5_v25.1.0.0_winhint.patch` that only edits existing gem5 files (cap hooks in ROB/IQ/LSQ, CPU and commit hooks, `window_policy` params, RISC-V hint decode); the new sources live in `sim/gem5/`
- **Reuse as baseline B5:** `sim/baselines/lut/{label_phases,train_phase_classifier,export_lookup_table}.py`
- **New:**
  - `tooling/create_conda_env.sh`, `environment.yml`, `tooling/winhint.sh`
  - `compiler/winhint/`
  - `compiler/baselines/{jones_iq,pgo,clairvoyance}/`
  - `third_party/clairvoyance` (submodule)
  - `sim/gem5/src/cpu/o3/window/` (controller, all policies, and B9's `ltp/`)
  - `sim/baselines/oracle/` (B1), `sim/tests/`
  - `hw/libwinhint/`, `hw/baselines/r0_taskset … r5_sondag/`, `hw/run_hw_experiments.py`

---

## 7. Verification and success criteria

- **Correctness:** hinted binaries give bit-identical output on clean gem5, QEMU and native x86.
- **Mechanism:** forced `setwin` values change the effective occupancy caps (checked in the `stats.txt` occupancy histograms and full-event counters).
- **Baseline fidelity:** each reimplemented baseline reproduces the *qualitative* trend reported in its paper on a microbenchmark before it is used. Deviations are listed in an appendix or in the artifact README.
- **Model:** Spearman ρ ≥ ~0.8 between the predicted W* and the oracle-best window per region.
- **gem5 end-to-end:** WinHint within ~5% of oracle (B1) IPC; ED²P better than B0-large, B2, B3, B4, B5 and B9; better than B6 (IQ only); competitive with B7 on an unseen input.
- **Real hardware:** WinHint EDP better than R0-P, R1 and R4 on memory-bound layers, with runtime overhead within noise.
- **Portability:** on unseen core configurations, the drop in B5 (retraining needed) is compared with WinHint (recompilation only).
- Report honestly wherever a criterion is not met.

---

## 8. Main risks

| Risk | Mitigation |
|------|------------|
| The Clairvoyance artifact (LLVM 3.8) does not interoperate with LLVM 23, or no conda channel ships LLVM 3.8. | Build the passes against the conda LLVM 3.8 in the separate `winhint-llvm38` env. If no LLVM 3.8 package exists, go straight to the port. If bitcode compatibility fails, port the pass and document it as a reimplementation. |
| Reimplemented baselines could be accused of being weak. | Tune each baseline's parameters on the training inputs with the same effort as WinHint, and publish all baseline code in the artifact. |
| The window gains are small once the register file and IQ are rebalanced. | Report the honest sensitivity. Also emphasize energy and ED²P, where a smaller window saves the most. |
| P/E migration costs outweigh the gains at operator granularity. | Move to layer granularity. The DP placement already takes this into account. |
| gem5 v25.1's SCons build or a Python package (e.g. PyTorch) does not support Python 3.14 yet. | `create_conda_env.sh` checks this first. Patch gem5's build scripts if the fix is small and record it in `sim/patches/`. Otherwise report the blocker and pin the closest working version in `tooling/versions.lock`. |
| A required tool has no conda package (for example the RISC-V cross GCC, `qemu`, `bpftool` or `perf`). | Do not build it. Use the conda alternative: clang 23 with the conda RISC-V sysroot instead of cross GCC; gem5's functional `AtomicSimpleCPU` as the correctness reference if `qemu` is missing; `libwinhint`'s own `perf_event_open` counters instead of `perf`. Record the substitution in `tooling/versions.lock`. |
| conda-forge has no LLVM 23 package, or only some of the components (`llvmdev`, `clang`, `lld`, `compiler-rt`). | Do not build from source. Pin the newest LLVM version for which conda-forge ships every component, record it in `tooling/versions.lock`, and move to 23 when its packages are published. The plugin uses only the new pass manager API, so moving between versions should take little or no code change. |
| Uncertainty about µarch parameters (P/E ROB sizes, etc.). | Cite Intel's optimization manual and confirm with microbenchmarks. |
