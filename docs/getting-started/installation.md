# Installation

This page sets up the toolchain that every other page assumes: the compilers that place the
hints, the RISC-V cross toolchain and QEMU that run the kernels, and the build dependencies of
gem5 and McPAT. Everything runs on the host. There is no Docker, nothing is installed with
`apt` or `sudo`, and no toolchain is compiled: **every tool comes from conda-forge packages**
installed with [micromamba](https://mamba.readthedocs.io/) into isolated environments. Only
the research artifacts (gem5, McPAT, the Clairvoyance passes and the R2–R4 hardware
baselines) are built from source, under `build/`.

**Before you read:** [How WinHint works](../guide/architecture.md) explains what the pieces
installed here are for.

## Requirements

| Resource | Needed |
|----------|--------|
| OS | Linux x86-64 |
| RAM | 6 GB (heavy jobs are serialized, see [Resource limits](#resource-limits)) |
| Disk | about 13 GB for the envs and the package cache, plus 5 GB or more for `build/` |
| Network | conda-forge and GitHub (gem5, McPAT, the Clairvoyance submodule) |
| Root | only for the optional measured runs on [real hardware](../guide/hardware/index.md) |

## Create the environments

```bash
git clone --recursive https://github.com/AndersonFaustino/WinHint.git
cd WinHint
# only if the clone was made without --recursive
git submodule update --init third_party/clairvoyance
tooling/create_conda_env.sh
```

[`tooling/create_conda_env.sh`](../../tooling/create_conda_env.sh) does six things:

1. uses `mamba`, `conda` or `micromamba` from `PATH`, or downloads micromamba into
   `$MAMBA_ROOT_PREFIX/bin` when there is none;
2. creates the environments below: the conda toolchain from its table (which
   [`environment.yml`](../../environment.yml) mirrors), then the Python packages with pip
   from [`requirements.txt`](../../requirements.txt) (PyTorch's CPU build first) and the
   `requirements-dev.txt` / `requirements-docs.txt` groups chosen with `--extras`;
3. installs an activation hook (the variables in [Activate](#activate));
4. verifies every tool against [`tooling/versions.lock`](../reference/toolchain.md);
5. writes a one-screen summary to `build/ENV.md`;
6. refuses to replace an environment that exists unless `--force` is given (`--update` syncs it
   instead).

`make validate` runs `tooling/bootstrap_venv.sh` first, which creates (or updates) the
`winhint` env with `--extras dev,docs` when needed, so the first commit needs no manual setup.

| Env | Defined by | Used for |
|-----|------------|----------|
| `winhint` | [`environment.yml`](../../environment.yml) + `requirements*.txt` | everything below unless stated, including this site |
| `winhint-llvm38` | `create_conda_env.sh` | the Clairvoyance artifact (baseline B8) |
| `winhint-kmod` | `create_conda_env.sh` | the PMCTrack kernel module (baseline R4) |
| `winhint-tex` | by hand (`build.sh` of the paper, `../WinHint-paper`) | the paper (tectonic) |

The `winhint` env holds the Python 3.14 analysis stack, LLVM/Clang 23.1.2, GCC 13, the
RISC-V cross GCC 15.3 with its sysroot, `qemu-riscv64`, the gem5 and McPAT build
dependencies, Rust and `perf`. `winhint-llvm38` holds LLVM 3.8.1 (the version the
Clairvoyance passes are written for) and numba; `winhint-kmod` holds GCC 15.2, the compiler
of the host kernel, which a kernel module must be built with. Exact versions are on the
[Toolchain](../reference/toolchain.md) page.

### Options

| Option | Effect |
|--------|--------|
| *(none)* | create the `winhint` env (and `winhint-llvm38`, `winhint-kmod`); fails if it exists |
| `-e, --extras LIST` | Python groups: `dev`, `docs`, `optional`, `all`, `none` (default `dev,docs`) |
| `--verify-only` | run the checks only (versions, RISC-V hello-world under QEMU) |
| `--update` | sync the env with the toolchain table and `requirements*.txt` (`winhint.sh env:update`) |
| `-f, --force` | delete and recreate an existing env |
| `-n, --name`, `-p, --python` | env name and Python version |
| `--accel MODE` | PyTorch build: `cpu` (default), `auto`, `cuda`, `rocm` |
| `--dry-run` | print the commands only |
| `--skip-llvm38` | do not create `winhint-llvm38` (same for `--skip-kmod`) |

The verification builds a RISC-V hello-world with both GCC and clang and runs it under QEMU.
The variables `MAMBA_ROOT_PREFIX` (default `~/.local/share/mamba`) and `WINHINT_BUILD`
(default `<repo>/build`) override the defaults.

## Activate

```bash
eval "$(~/.local/bin/micromamba shell hook -s bash)"
micromamba activate winhint
```

Alternatively, prefix single commands with `~/.local/bin/micromamba run -n winhint`.
[`tooling/winhint.sh`](../../tooling/winhint.sh), the command-line manager used throughout
this site, re-executes itself inside the env when it is not active. Activation sets:

| Variable | Value |
|----------|-------|
| `WINHINT_ROOT` | the repository |
| `WINHINT_BUILD` | `$WINHINT_ROOT/build` |
| `PATH` | adds `tooling/`, `build/bin/` and LLVM's `libexec/llvm` (`FileCheck`) |
| `CC`, `CXX` | the conda host GCC 13 |
| `WINHINT_RISCV_CC`, `WINHINT_RISCV_CXX` | `riscv64-conda-linux-gnu-gcc` / `-g++` |
| `WINHINT_RISCV_TRIPLE` | the conda RISC-V triple |
| `WINHINT_RISCV_SYSROOT` | its sysroot |
| `WINHINT_RISCV_CLANG_FLAGS` | flags for `clang` cross builds (target, sysroot, `-fuse-ld=lld`) |
| `QEMU_LD_PREFIX` | the RISC-V sysroot, so dynamic binaries run under `qemu-riscv64` |

!!! note "Use the conda RISC-V triple"
    `clang --target=riscv64-unknown-linux-gnu` does **not** work with this sysroot (clang
    cannot find `crt*`/`libgcc` under the conda triple). Use `$WINHINT_RISCV_CLANG_FLAGS`,
    which expands to `--target=riscv64-conda-linux-gnu`, the sysroot and `-fuse-ld=lld`.

## Check

```bash
# every tool and build, and whether the heavy lock is held
tooling/winhint.sh status
# create_conda_env.sh --verify-only
tooling/winhint.sh test env
```

## Where things go

- **Build outputs** go under `build/` (overridable with `WINHINT_BUILD`): gem5, McPAT, the
  compiler plugins (`build/compiler/`), the benchmark binaries
  (`build/benchmarks/<arch>/<variant>/`), libwinhint and the hw tools, the Clairvoyance
  passes, the R2–R4 baseline builds (`build/hw-baselines/`) and this site (`build/site/`).
- **Results** (CSV, JSON and figures) go to `results/`.

Both are gitignored. The full path table is in [Interfaces §1](../interfaces.md#1-environment)
and a per-directory description in [Repository layout](../estrutura.md#build-outputs).

## Resource limits

The development machine has 12 threads and 6 GB of RAM. Heavy jobs (gem5 and McPAT builds,
gem5 simulations longer than about a minute, env solves) take the
[heavy lock](../reference/glossary.md#heavy-lock), `flock "$WINHINT_BUILD/.heavy.lock"`, so
they run one at a time and with at most `-j2`. `winhint.sh` and the experiment scripts take
the lock themselves; do not wrap them in another `flock` on that file. Every other compile
uses `-j1`. The experiment scripts default to `--jobs 1` and refuse more than 2.

## Next

- [Quickstart](quickstart.md): a first hinted kernel, built and checked under QEMU.
- [Repository layout](../estrutura.md): what lives where.
