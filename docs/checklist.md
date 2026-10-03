# Project status

This page tracks the implementation against the roadmap of the
[design proposal](reference/proposal.md) §5, phase by phase, and against the verification
criteria of §7. It is a development artifact: it says what exists in the repository today, not
what the experiments found. The roadmap's phases build on each other, from the environment
(Phase 0) to the simulator, workloads, compiler, gem5 policies, evaluations and the paper. What
each experiment measures and why is in [Evaluation methodology](concepts/methodology.md); how to
run it is in [Running experiments](guide/usage.md).

How to read the checkboxes:

- `[x]` the item is implemented and present in the repository (code, scripts or build
  outputs);
- `[ ]` the item is not done yet. Most open items are experiment campaigns: the tooling exists,
  but the runs that produce results have not been made;
- notes in parentheses say where the item lives or why it differs from the proposal. Every
  difference from the proposal is listed in [Deviations](deviations.md).

Status as of 2026-10-01, before any experiment campaign.

## Phase 0 — Rename, restructure, environment

Purpose: one reproducible host environment from conda packages, and the tree reorganised by
component ([Installation](getting-started/installation.md), [Toolchain](reference/toolchain.md)).

- [x] Docker dropped (`Dockerfile`, `docker-compose.yml`, `.env`, `docker_manager.sh`)
- [x] Moves: `tooling/` (ex `scripts/`, the legacy manager script, `pkg_version.lock`),
      `benchmarks/`, `sim/baselines/lut/`, `analysis/`, `hw/baselines/r4_pie`,
      `hw/baselines/r5_sondag`; `phase_predictor.cc` and the committed binaries removed
- [x] `environment.yml`: Python 3.14, LLVM/Clang 23.1.2, GCC 13, RISC-V GCC 15.3 +
      sysroot 2.39, qemu-riscv64 11.0.3, gem5/McPAT dependencies, PyTorch CPU, Rust, perf
- [x] `tooling/create_conda_env.sh` (idempotent; `--update`, `--recreate`, `--init-shell`,
      `--verify-only`): env `winhint`, activation hook, version check against
      `tooling/versions.lock`, RISC-V hello-world under QEMU (GCC and clang)
- [x] Env `winhint-llvm38` (conda-forge `llvmdev`/`clangdev` 3.8.1) for B8
- [x] Env `winhint-kmod` (GCC 15.2, the host kernel's compiler) for the PMCTrack module (R4)
- [x] `tooling/versions.lock` with pins and the recorded substitutions (PROPOSAL §8;
      [Deviations I10](deviations.md#i10))
- [x] `tooling/winhint.sh`: `env:create|update`, `gem5:clone`, `gem5:build clean|winhint|all`,
      `mcpat:clone|build`, `benchmarks:build`, `compiler:build`, `status`
- [x] `tooling/apply_patches.sh`: overlay `sim/gem5/` + `sim/patches/gem5_*_*.patch` in name
      order, `--check`, `--revert`, `--status`
- [x] gem5 v25.1.0.0 cloned into `build/gem5/src` (`tooling/winhint.sh gem5:clone`)
- [x] McPAT cloned and built (`build/mcpat/mcpat`)
- [x] Clean gem5 built (`build/gem5/src/build/RISCV_clean/gem5.opt`)
- [x] Check: `git grep -i` for the legacy name is empty (the design proposal mentions it historically);
      no file references `src/`, `scripts/`, Docker, `/home/winhint` or `/workspace`
- [x] Check: a second `tooling/create_conda_env.sh` changes nothing; `tooling/winhint.sh status`
      reports every tool

## Phase A — Simulation base

Purpose: a gem5 O3 core whose window can be resized safely, with real counters, so that every
policy shares one mechanism ([gem5 model](guide/gem5/index.md),
[gem5 patches](guide/gem5/patches.md)).

- [x] Window controller (`sim/gem5/src/cpu/o3/window/controller.*`): real L1D/L2 miss deltas
      per window, state in CPU members, one `predict()` per period, no fake prefetcher call
- [x] Safe resize of ROB, IQ and LSQ (caps + dispatch gating until occupancy ≤ target)
- [x] `window_policy` (string) replaces `adaptive_rob` in `BaseO3CPU.py`
- [x] `sim/se.py` defaults rebalanced (288 physical registers on `riscv_ooo`, IQ/LSQ scaled) so
      that the window binds
- [x] Compiled-in `phase_lookup.h` replaced by a LUT file loaded at run time (`window_lut.hh`)
- [x] `RISCV_winhint` gem5 built (`tooling/winhint.sh gem5:build winhint`)

## Phase B — Workloads

Purpose: ML inference kernels at real model shapes, with two inputs per kernel
([Workloads](guide/workloads.md)).

- [x] Kernels at BERT-base / GPT-2-small layer shapes, working sets beyond L2 (inputs reduced
      for simulation time: [Deviations I9](deviations.md#i9))
- [x] Build matrix: `-O2`/`-O3`, unrolling on/off, tiled GEMM
- [x] `regression_*` control group kept
- [x] Two input sizes (`small`, `large`) per kernel
- [ ] Optional TVM/IREE-generated C kernels: not feasible under the conda-only rule
      ([Generated kernels](guide/workloads-generated.md))

## Phase C — WinHint compiler

Purpose: the window-demand analysis and hint placement, plus the compiler baselines
([Compiler](guide/compiler.md)).

- [x] `compiler/winhint/`: `WindowDemandAnalysis`, `HintPlacement`, RISC-V HINT and x86 NOP
      emission
- [x] `compiler/baselines/jones_iq/` (B6), `compiler/baselines/pgo/` (B7)
- [x] `compiler/baselines/clairvoyance/` (B8) built in `winhint-llvm38`
- [x] `benchmarks/Makefile` variants: `winhint`, `oracle`, `oracle_hinted`, `jones`, `pgo`,
      `clairvoyance`
- [x] `compiler/test/run_tests.sh` passes

## Phase D — gem5 hint and policy support

Purpose: gem5 reacts to the hints, and every hardware baseline exists as a window policy
([Window policies](guide/gem5/policies.md)).

- [x] Hint recognition: stock gem5 already decodes `ori x0,x0,imm` as a no-op; the controller
      recognises it at commit on `RISCV_winhint` only ([Deviations I1](deviations.md#i1))
- [x] Policies: `static` (B0), `occupancy` (B2), `mlp` (B3), `bbv` (B4), `lut` (B5), `hint`,
      `hybrid`, `ltp` (B9)
- [x] Per-policy unit checks with forced traces (`sim/tests/run_tests.sh`)

## Phase E — gem5 evaluation

Purpose: run WinHint and B0–B9 on at least three simulated machines and produce the figures
([gem5 model](guide/gem5/index.md)). Run each pytest suite separately
(`tooling/winhint.sh test python` does this): their `conftest.py` modules clash in a single
pytest call.

- [x] Evaluation driver `sim/run_experiments.py`: every baseline variant, WinHint and
      WinHint+HW, on every machine in `sim/machines/` (three), plus sensitivity sweeps
- [x] Energy estimator `sim/estimate_energy.py` (McPAT or proxy) and figure script
      `analysis/plot_results.py` (W* vs oracle, switch frequency, hint overhead, sensitivity,
      portability)
- [ ] B1 oracle sweep (`sim/baselines/oracle/oracle_sweep.py`) and B5 LUT per machine
      (`sim/baselines/lut/`): not run (no `results/oracle/`)
- [ ] Baseline tuning run (`sim/baselines/tune/tune_baselines.py` → `results/tune/tuned.json`)
- [ ] Evaluation campaign over every baseline, WinHint and WinHint+HW on ≥ 3 machines
- [ ] Energy with McPAT and the final figures

## Phase E2 — Real silicon (Core 5 120U)

Purpose: the same hints drive P/E-core migration on an Intel hybrid CPU, compared with R0–R5
([Real hardware](guide/hardware/index.md)).

- [x] `libwinhint` runtime (P/E migration, DP placement with measured migration cost)
- [x] Hardware driver `hw/run_hw_experiments.py` and smoke runs (`results/hw/smoke*`)
- [x] R2 (sched_ext) builds (libbpf vendored by `libbpf-sys`, no bpftool needed); R3
      (intel-lpmd) builds with a `upower` stub and the env's conda `glib` (see `docs/guide/hardware/index.md`)
- [x] llama.cpp operator hooks (`hw/integrations/llamacpp/`, optional; the window table is a
      placeholder: [Deviations R-P2](deviations.md#r-p2))
- [ ] Hint overhead within noise (`perf stat`); RISC-V hint binaries bit-identical under
      `qemu-riscv64` and on `RISCV_clean`
- [ ] R0–R5 comparison; RAPL energy/EDP; fixed governor, ≥ 10 repetitions, 95% CI; SMT on/off
- [ ] Root steps (RAPL, governor, SMT, sched_ext, lpmd D-Bus) and the idle-machine campaign:
      commands in `docs/guide/hardware/index.md` §4
- [ ] PMCTrack module (R4, optional): does not build on kernel 7.0; a port is not planned, R4
      runs on `perf_event_open` ([Deviations R4-3](deviations.md#r4-3))

## Phase F — Writing and artifact

Purpose: the paper and an artifact that reproduces it from the environment scripts.

- [x] Paper draft (results as `\TODO` placeholders), kept outside this repository in the sibling directory `../WinHint-paper`
- [x] Artifact instructions built on `tooling/create_conda_env.sh`, `environment.yml` and
      `tooling/winhint.sh` ([Running experiments](guide/usage.md))
- [x] Documentation site (MkDocs, env `winhint-docs`) and the consolidated
      [Deviations](deviations.md) appendix

## Added beyond the roadmap

Support the proposal implies but does not list as a roadmap item.

- [x] Run-length control for `large` inputs: region-aligned checkpoint sampling
      (`sim/run_lengths.{json,py}`)
- [x] Baseline tuning harness (`sim/baselines/tune/tune_baselines.py`)
- [x] Fidelity harnesses: gem5 (`benchmarks/micro/`, `sim/fidelity/`) and hardware
      (`hw/fidelity.py`)
- [x] µarch probe (`hw/tools/uarch_probe.c`, `docs/guide/hardware/uarch-params.md`)
- [x] B8 SwoopDAE crash fixed (patch 0002), knobs exposed (`knobs.json`)
- [x] Correctness checker `tooling/verify_correctness.py` (smoke subset in
      `results/correctness.csv`)

## Verification (PROPOSAL §7)

Purpose: the success criteria the paper reports against, met or not.

- [ ] Correctness: hinted binaries bit-identical on clean gem5, QEMU and native x86 (so far a
      smoke subset of two kernels under QEMU and native x86 passes)
- [x] Mechanism: forced `setwin` changes the occupancy caps (`sim/tests/run_tests.sh`)
- [ ] Baseline fidelity: each baseline reproduces its paper's qualitative trend
      ([Simulator fidelity](guide/fidelity.md))
- [ ] Model: Spearman ρ ≥ ~0.8 between predicted W* and the oracle-best window
- [ ] gem5 end-to-end, real hardware and portability criteria; report honestly where unmet

## Next

- [Deviations](deviations.md): every difference from the proposal and the papers.
- [Running experiments](guide/usage.md): the commands behind the open items.
- [Commit gate and coverage](contributing/quality.md): the checks every change must pass.
