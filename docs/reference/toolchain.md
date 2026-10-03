# Toolchain and versions

Results from a compiler study depend on the exact compiler, simulator and libraries that
produced them, so every version used by WinHint is pinned in one file and checked on every
environment setup. This page lists the environments, which pins each one depends on, and the
pin file itself. How to create the environments is in
[Installation](../getting-started/installation.md).

Every tool comes from conda-forge packages installed by
[`tooling/create_conda_env.sh`](../getting-started/installation.md) into micromamba environments.
The pinned versions, and every place where a pin had to be substituted, are recorded in
[`tooling/versions.lock`](../../tooling/versions.lock). `create_conda_env.sh` fails when an
installed version differs from it (`tooling/winhint.sh test env` runs that check alone).

## Environments

| Env | Specification | Key pins (`versions.lock`) |
|-----|---------------|----------------------------|
| `winhint` | [`environment.yml`](../../environment.yml), `requirements*.txt` | `PYTHON`, `LLVM`, `HOST_GCC`, `RISCV_GCC`, `RISCV_SYSROOT`, `QEMU`, `GLIB` |
| `winhint-llvm38` | `create_conda_env.sh` | `LLVM38` |
| `winhint-kmod` | `create_conda_env.sh` | `KMOD_GCC` |
| `winhint-tex` | by hand (`build.sh` of the paper, `../WinHint-paper`) | `TECTONIC` |

Research artifacts built from source (gem5, McPAT, the llama.cpp integration) are pinned in
the same file by tag or commit.

## `tooling/versions.lock`

The file below is included verbatim at build time.

```properties
--8<-- "tooling/versions.lock"
```

## Next

- [Deviations](../deviations.md): where the implementation departs from the proposal and the
  baseline papers.
- [Glossary](glossary.md): the terms used across the site.
