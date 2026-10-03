# WinHint — Know Your Window

[![Python](https://img.shields.io/badge/python-3.14-blue)](https://www.python.org/)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](docs)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](LICENSE)
[![Commit gate](https://img.shields.io/badge/commit_gate-make_validate-blue)](#development)

**Static analysis of memory-level parallelism to drive microarchitectural reconfiguration.**

WinHint is a compiler-directed approach to sizing the out-of-order window (ROB, issue queue,
load/store queues) for ML inference. For these workloads the compiler already knows the
phases: attention, FFN, layer norm and softmax are separate loop nests with affine accesses,
known trip counts and known working sets. A static, loop-level cost model estimates, for each
region:

- `L_mem`, the expected miss latency per load class (footprint versus cache capacity);
- `D_indep`, the distance between independent long-latency loads;
- `CP`, the critical-path length of the loop body.

From these it derives a window demand

```
W*         = min(W_max, max(ceil(MLP_target · D_indep), ceil(CP · rate)))
rate       = min(issue_width, body / II)          # sustainable issue rate
MLP_target = min(MSHR_L1D, L_mem · rate / D_indep)
```

For issue-bound loops the critical-path term is `CP · issue_width`. A placement pass then
emits advisory `setwin(W)` hints — a RISC-V HINT-space `ori x0, x0, imm` or a unique x86
multi-byte NOP — which are no-ops on unmodified cores.

The claims under evaluation (see the [design proposal](docs/reference/proposal.md)) are that the hints:

- match or beat reactive hardware window resizing;
- need no training data;
- switch exactly at region boundaries;
- carry over to a new core by recompiling with that core's parameters.

They are evaluated in gem5 (RISC-V O3, window resizing) and on Intel hybrid silicon (P-core/E-core
migration).

> The cross-component contract (hint encoding, window table, gem5 flags, file formats) is
> [docs/interfaces.md](docs/interfaces.md). Read it before changing any interface.

## Contents

- [Quickstart](#quickstart) — environment and a first hinted kernel
- [Repository layout](#repository-layout)
- [Baselines](#baselines) and [Workloads](#workloads)
- **[Running experiments](docs/guide/usage.md)** — running the gem5 study and the real-hardware campaign, step by step
- [Development](#development) — the commit gate (`make validate`), tests, documentation
- Documentation site: `make docs-serve`, then <http://127.0.0.1:8000>

## Quickstart

Everything runs on the host from a conda env (conda-forge for the toolchain, pip for the Python
packages; no Docker, no `sudo`).

```bash
git clone --recursive git@github.com:AndersonFaustino/WinHint.git
cd WinHint

# envs winhint (+ winhint-llvm38, winhint-kmod), then checks
tooling/create_conda_env.sh
eval "$(~/.local/bin/micromamba shell hook -s bash)"
micromamba activate winhint

# tools, builds, heavy lock
tooling/winhint.sh status
```

A first hinted kernel, checked bit-identical against the plain build under QEMU:

```bash
tooling/winhint.sh compiler:build
make -C benchmarks ARCH=riscv VARIANT=plain
make -C benchmarks ARCH=riscv VARIANT=winhint check
qemu-riscv64 build/benchmarks/riscv/winhint/encoder_bert_tiny_infer small
```

Next: [Running experiments](docs/guide/usage.md); `tooling/winhint.sh help` for every command.

**Toolchain.** Python 3.14, LLVM/Clang 23.1.2, GCC 13 (gem5, McPAT), RISC-V GCC 15.3 with a
glibc 2.39 sysroot, `qemu-riscv64` 11.0.3. Exact pins and every substitution:
`tooling/versions.lock`; `create_conda_env.sh` fails when an installed version differs. Only gem5
v25.1.0.0, McPAT, the Clairvoyance passes and the R2–R4 baselines are built from source.

**Resources.** Developed on 12 threads and 6 GB of RAM. Heavy jobs (gem5/McPAT builds, long
simulations) take `flock "$WINHINT_BUILD/.heavy.lock"` themselves and run one at a time at
`-j2` at most; everything else uses `-j1`.

**Paths.** Builds go to `build/` (`WINHINT_BUILD`), results to `results/`; both are
gitignored. Activating the env sets `WINHINT_ROOT`, `WINHINT_BUILD` and the RISC-V helpers
(`WINHINT_RISCV_CC`, `WINHINT_RISCV_CLANG_FLAGS`, `QEMU_LD_PREFIX`).

## Repository layout

Organized by component (details: [docs/estrutura.md](docs/estrutura.md)).

```
compiler/      LLVM 23 plugins: WinHint (analysis + placement), B6/B7/B8 baselines, tests
benchmarks/    15 C inference kernels, micro-benchmarks, Makefile (build variants)
sim/           gem5 config (se.py), machines/*.json, window controller sources, patches,
               B1/B5 pipelines, experiment matrix, energy model, fidelity study
hw/            libwinhint (hint → P/E migration), measurement tools, baselines R0–R5
analysis/      paper figures (plot_results.py)
tooling/       create_conda_env.sh, versions.lock, winhint.sh, coverage and docstring gates
environment.yml, requirements*.txt   the conda env and its Python packages
docs/          documentation site (mkdocs.yml); interfaces.md is the binding contract
third_party/   Clairvoyance artifact (B8, git submodule)
```

## Baselines

Hardware window-resizing papers have no public artifacts, so B2–B5 and B9 are reimplemented
here; deviations are documented next to each implementation and in
[docs/deviations.md](docs/deviations.md). Citations and the artifact search:
[design proposal](docs/reference/proposal.md) §4.

| ID | Baseline | Kind |
|----|----------|------|
| B0 | static windows (4 configurations per machine) | gem5 |
| B1 | per-region oracle (upper bound) | gem5 |
| B2 | occupancy-driven resizing (Ponomarev, MICRO'01) | gem5 |
| B3 | MLP-aware resizing (Kora, MICRO'13) | gem5 |
| B4 | BBV phase prediction (Sherwood, ISCA'03) | gem5 |
| B5 | learned counter LUT (Dubach style) | gem5 |
| B6 | compiler IQ resizing (Jones, HPCA'05) | compiler |
| B7 | profile-guided positional adaptation (Huang, ISCA'03) | compiler |
| B8 | Clairvoyance (Tran, CGO'17), public artifact | compiler |
| B9 | Long-Term Parking (Sembrant, MICRO'15) | gem5 |
| R0–R5 | taskset, stock Linux, sched_ext, intel-lpmd, PIE, Sondag & Rajan | real hardware |

How each one is built and run: [Running experiments](docs/guide/usage.md).

## Workloads

Fifteen portable, deterministic C11 kernels (`benchmarks/`). Each takes `small|large`
(default `large`) and prints a checksum and an FNV-1a hash for bit-identical checks. Phases
are separate non-inlined functions with their own loop nests.

| Group | Kernels | Models |
|-------|---------|--------|
| encoder | `encoder_{bert_tiny,roberta,vit_micro}_infer` | BERT-base, RoBERTa-base, ViT-B/16 layers |
| decoder | `decoder_{gpt2,llama_mini,beam_search}_infer` | GPT-2-small, LLaMA-style, 4-beam GPT-2 |
| contrast | `contrast_{mlp,mobilenet,siamese}_infer` | MLP, MobileNetV1, Siamese MLP towers |
| imggen | `imggen_{conv_autoenc,unet,vae_decode}_infer` | conv autoencoder, UNet decoder, VAE decoder |
| control | `regression_{linear,ridge,svr}_infer` | non-transformer control group |

Transformer layers carry 24–28 MB of weights, far beyond a 256 kB–1 MB L2. `small` keeps the
model shapes and shrinks only the input, so a run fits a gem5 O3 simulation (about 40–550M
instructions). Shapes, phases and knobs (`-DTILED`, `-DSEQ_LARGE=…`) are in each file's header
and in the [workloads guide](docs/guide/workloads.md).

```bash
# variables: ARCH VARIANT OPT UNROLL TILED MACHINE …
make -C benchmarks list
make -C benchmarks ARCH=riscv VARIANT=plain OPT=O3 TILED=on
```

Build variants: `plain`, `winhint`, `oracle`, `oracle_hinted`, `jones`, `jones_full`, `pgo`,
`clairvoyance`, `winhint_clairvoyance`, and the x86 call-mode `winhint_call` / `oracle_call`
([interfaces.md](docs/interfaces.md) §5).

## Development

Every commit goes through one gate, `make validate`. There is no hosted CI: a git pre-commit
hook runs the gate and refuses the commit when it fails.

```bash
# once per clone: every `git commit` now runs `make validate` first
make install-hooks
```

| Command | What it does |
|---------|--------------|
| `make validate` | the commit gate (1–2 minutes; no gem5, no root) |
| `make coverage` | the coverage gate alone → `build/coverage/` |
| `make docs` | strict documentation build → `build/site/` |
| `make docs-serve` | live preview at <http://127.0.0.1:8000> |
| `make docs-check` | docstring coverage and syntax |

`make validate` runs three steps and stops at the first failure:

1. **Environment** — tool versions against `versions.lock`; RISC-V hello-world under QEMU.
2. **Tests and coverage** — every host test suite on instrumented builds. Python, C++ and C
   must each cover **more than 90 %** of their lines; a file no test reaches fails the gate.
3. **Documentation** — docstrings, then a strict MkDocs build.

Code that only runs inside gem5 or needs root/hybrid hardware is excluded by a commented list.
pre-commit validates exactly what is staged; `git commit --no-verify` skips the gate (only for
work in progress). Details: [docs/contributing/quality.md](docs/contributing/quality.md).

**Other tests** (not in the gate; they need the gem5 builds):

```bash
tooling/winhint.sh test sim            # gem5 mechanism + every policy (heavy lock)
tooling/winhint.sh verify              # every hinted variant == plain → results/correctness.csv
```

**Documentation.** The site combines the guides in `docs/`, the component READMEs, this
the runbook, and an API reference generated from the code (Google-style
docstrings, Doxygen). Conventions: [docs/contributing/documentation.md](docs/contributing/documentation.md).

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

WinHint is licensed under the Apache License, Version 2.0 (see [`LICENSE`](LICENSE)).
The Clairvoyance submodule keeps its own license
(`third_party/clairvoyance/LICENSE`).
