# Repository layout

The repository is organized by component, following the pipeline described in
[How WinHint works](guide/architecture.md): workloads are compiled by the compiler plugins,
run in the simulator or on real hardware, and the results are analyzed. Each top-level
directory is one of those components. This page is a map for a newcomer: the tree first, then
what each part is for and why it is where it is. The formats that cross component boundaries
are fixed in [Interfaces](interfaces.md).

## The tree

```text
WinHint/
├── README.md, LICENSE
├── Makefile                  shortcuts: validate, install-hooks, coverage, docs
├── .pre-commit-config.yaml   pre-commit hook: runs make validate
├── .coveragerc               Python coverage scope and exclusions
├── mkdocs.yml                this documentation site
├── benchmarks/               15 C inference kernels, micro/, generated/, Makefile
├── compiler/
│   ├── winhint/              the WinHint pass: cost model, placement
│   ├── common/               shared target model and hint emitter
│   ├── baselines/            jones_iq/ (B6), pgo/ (B7), clairvoyance/ (B8)
│   └── test/                 lit tests and the compiler test driver
├── sim/
│   ├── se.py                 gem5 configuration script
│   ├── machines/             riscv_ooo{,_small,_big}.json
│   ├── gem5/                 new gem5 sources (window controller, policies, LTP)
│   ├── patches/              edits to existing gem5 files
│   ├── baselines/            oracle/ (B1), lut/ (B5), tune/
│   ├── fidelity/             baseline-fidelity study
│   ├── tests/                mechanism and policy checks
│   └── run_experiments.py, run_baseline.py, run_lengths.py, estimate_energy.py
├── hw/
│   ├── libwinhint/           runtime: hints → P/E-core migration
│   ├── baselines/            r0_taskset … r5_sondag
│   ├── tools/                measurement helpers and µarch probes
│   ├── integrations/         llamacpp/
│   ├── tests/
│   └── run_hw_experiments.py, fidelity.py, Makefile
├── analysis/                 plot_results.py (figures) and tests
├── tooling/                  envs, versions.lock, winhint.sh, gates
├── docs/                     site pages (this page included)
│   ├── concepts/             background, evaluation methodology
│   ├── guide/                user guide; usage.md is the runbook
│   └── reference/            glossary, toolchain; proposal.md is the design proposal
├── third_party/clairvoyance/ B8 artifact source (git submodule)
├── build/                    (gitignored) everything fetched or built
└── results/                  (gitignored) experiment data and figures
```

There is no `src/` or `scripts/` directory: each script lives with the component it drives.

## What lives where, and why

### Workloads: `benchmarks/`

The 15 ML-inference kernels in portable C11 and the Makefile that builds each of them in every
[build variant](reference/glossary.md#variant) (plain, hinted, oracle, baselines).
`micro/` holds five microbenchmarks for the [fidelity study](guide/fidelity.md). `generated/`
documents the optional TVM/IREE kernels, which are not implemented. Details:
[Workloads](guide/workloads.md).

### Compiler: `compiler/`

Out-of-tree LLVM 23 plugins. `winhint/` is the contribution itself
(`WindowDemandAnalysis` and `HintPlacementPass`); `common/` is the machine model and the hint
encoder that the WinHint and B6 plugins share. The compiler-side baselines sit under
`baselines/` because they are built and invoked the same way. `build.sh` produces
`build/compiler/WinHint.so` and `JonesIQ.so`. Details: [Compiler](guide/compiler.md).

### Simulator: `sim/`

Everything for the gem5 study.

- `se.py` configures a RISC-V O3 core from a machine JSON in `machines/`.
- `gem5/src/cpu/o3/window/` mirrors the gem5 source tree: it is copied into gem5 at build
  time, and `patches/` holds the edits to existing gem5 files. Keeping both here, instead of
  in a gem5 fork, lets one pristine gem5 checkout serve two builds.
- `baselines/` holds the offline pipelines: the B1 oracle sweep, the B5 learned lookup table
  and the baseline tuning (`tune/`).
- `run_experiments.py` runs the evaluation matrix; `estimate_energy.py` adds McPAT energy.

Details: [gem5 model](guide/gem5/index.md).

### Real hardware: `hw/`

The runtime that turns hints into P-core/E-core migrations on Intel hybrid CPUs
(`libwinhint/`), the R0–R5 baselines, measurement tools and the campaign driver
`run_hw_experiments.py`. Details: [Real hardware](guide/hardware/index.md).

### Analysis: `analysis/`

`plot_results.py` reads `results/` and writes every figure and `metrics.json`.

### Tooling: `tooling/`

Everything that installs, builds or checks the toolchain:

| File | Role |
|------|------|
| `create_conda_env.sh` | creates, updates and verifies the conda envs (`environment.yml` mirrors its toolchain table) |
| `bootstrap_venv.sh` | creates/updates the env on demand; `make validate` runs it first |
| `versions.lock` | pinned versions and substitutions |
| `winhint.sh` | the command-line manager (builds, tests, status) |
| `apply_patches.sh` | overlays `sim/gem5/` and `sim/patches/` on gem5 |
| `verify_correctness.py` | bit-identical output check (`winhint.sh verify`) |
| `coverage_gate.py` | coverage gate (`winhint.sh coverage`) |
| `check_docstrings.py` | docstring check (`winhint.sh docs:check`) |

`winhint.sh help` lists every command. Versions: [Toolchain](reference/toolchain.md).

### Documentation: `docs/`

Every document lives in `docs/`, with lowercase file names, and is a page of this site:
`concepts/`, `getting-started/`, `guide/` (including the runbook `guide/usage.md` and one page
per component), `reference/` (including the design proposal) and `contributing/`.
`docs/_tools/` builds the site. The repository's `README.md` is the only Markdown file outside
`docs/`. See [Writing documentation](contributing/documentation.md).

### Generated: `build/` and `results/`

Both are gitignored. `results/` holds data only (CSV, JSON, figures); every binary is rebuilt
into `build/`.

## Build outputs

`build/` is `$WINHINT_BUILD`. Each entry is produced by one command:

| Path | Contents | Produced by |
|------|----------|-------------|
| `.env-ready`, `ENV.md` | env marker, tool summary | `create_conda_env.sh` |
| `.heavy.lock` | the [heavy lock](reference/glossary.md#heavy-lock) | every heavy job |
| `gem5/src/` | gem5 source (tag in `versions.lock`) | `gem5:clone` |
| `gem5/src/build/RISCV_clean/` | unmodified gem5 | `gem5:build clean` |
| `gem5/src/build/RISCV_winhint/` | gem5 with window resizing | `gem5:build winhint` |
| `mcpat/` | McPAT source and binary | `mcpat:clone`, `mcpat:build` |
| `compiler/` | `WinHint.so`, `JonesIQ.so` | `compiler:build` |
| `benchmarks/<arch>/<variant>/` | kernels, `*.regions.json` | `benchmarks:build` |
| `libwinhint/`, `hw/` | libwinhint, hw tools | `hw:build` |
| `clairvoyance/` | B8 passes (LLVM 3.8) | `clairvoyance:build` |
| `hw-baselines/` | R2–R4 builds | `hw-baselines:build r2\|r3\|r4` |
| `ccache/`, `ccache-bin/` | compiler cache | gem5 and McPAT builds |

Commands in the last column are `tooling/winhint.sh` subcommands.

## Design decisions

1. **Host only, conda only.** Every tool comes from a conda package; nothing is installed with
   apt/sudo and no toolchain is compiled. Missing packages are substituted as described in
   PROPOSAL §8 and recorded in `tooling/versions.lock`. This keeps the setup reproducible
   without root ([Installation](getting-started/installation.md)).
2. **One gem5 tree, two builds.** `RISCV_clean` is built from the pristine tree.
   `RISCV_winhint` is built after `tooling/apply_patches.sh` copies the overlay and applies the
   patches; the tree is restored right after the build, even if it fails. Both builds run
   under the heavy lock, so they never race.
3. **Shared ccache.** `build/ccache` is shared by the gem5 builds, so switching between the
   clean and the winhint build recompiles only the files that differ.
4. **Results are data only.** `results/` never holds binaries; every binary is rebuilt into
   `build/`.

## Next

- [Running experiments](guide/usage.md): the runbook for the gem5 study and the hardware
  campaign.
- [Workloads](guide/workloads.md): the kernels and how they are built.
