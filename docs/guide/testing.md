# Testing and verification

WinHint makes two promises that tests must keep: a hint never changes what a program computes
(hints are architectural no-ops, [Interfaces §2](../interfaces.md#2-the-hint-isa-contract)),
and each component does what its documentation says (the cost model, the placement, the gem5
window mechanism and policies, the analysis pipeline). This page lists every test entry point,
what it checks and what it costs. All of them go through
[`tooling/winhint.sh`](../../tooling/winhint.sh). The host suites also form the commit gate,
`make validate`, described in [Commit gate and coverage](../contributing/quality.md).

**Before you read:** [Installation](../getting-started/installation.md); the gem5 suite also
needs both gem5 builds ([Running experiments, step 1](usage.md#step-1-prerequisites)).

## Test suites

| Command | What it checks | Cost |
|---------|----------------|------|
| `test env` | tool versions, RISC-V hello-world under QEMU | seconds |
| `test python` | every pytest directory (below) | about a minute |
| `test compiler` | cost model, placement, region ids, encodings, lit tests | minutes |
| `test sim` | gem5 mechanism and every policy (heavy lock) | long |
| `test docs` | docstrings, then a strict build of this site | about a minute |
| `test all` | `env` + `python` + `compiler` + `docs` (no gem5) | minutes |

Each command is a `tooling/winhint.sh` subcommand.

- **`test env`** runs `create_conda_env.sh --verify-only`: every tool version against
  `versions.lock`, and a RISC-V hello-world under `qemu-riscv64` built with GCC and clang,
  static and dynamic.
- **`test python`** runs pytest in `sim/baselines/{lut,oracle,tune}/tests`,
  `sim/fidelity/tests`, `sim/tests`, `analysis/tests`, `hw/tests`, `tooling/tests` and
  `compiler/baselines/pgo/tests`, one process per directory: several of these directories
  have their own `conftest.py` and import helpers from it, which collides in a single
  session. Extra arguments go to pytest, for example `tooling/winhint.sh test python -k lut`.
- **`test compiler`** runs [`compiler/test/run_tests.sh`](../../compiler/test/run_tests.sh):
  the per-loop model on small C loops, placement, deterministic region ids, hint encodings in
  the object code, hinted-vs-plain output identity, and the lit suite in
  [`compiler/test/lit/`](../../compiler/test/lit).
- **`test sim`** runs [`sim/tests/run_tests.sh`](../../sim/tests/run_tests.sh) on both gem5
  builds.
- **`test docs`** runs [`tooling/check_docstrings.py`](../api/python/tooling/check_docstrings.md)
  (docstring coverage and syntax), then `mkdocs build --strict`.

## Correctness of hinted binaries

Every hinted [build variant](../reference/glossary.md#variant) must print exactly the
same checksum and hash as `plain`.

```bash
# every variant: qemu-riscv64, native x86 and gem5 RISCV_clean if built → results/correctness.csv
tooling/winhint.sh verify
# options: --arch, --variants, --kernels, --gem5 auto|on|off, --dry-run
tooling/winhint.sh verify --help
# one variant under qemu
make -C benchmarks ARCH=riscv VARIANT=winhint check
```

See [`tooling/verify_correctness.py`](../api/python/tooling/verify_correctness.md).

## Unit tests of the gem5 code

The window controller and the policies have host-side unit tests that build without gem5
(stubbed gem5 headers), under `sim/gem5/src/cpu/o3/window/tests/` and `.../ltp/tests/`, and
`sim/tests/unit/` (`make -C sim/tests unit`). The coverage gate runs them.

## Documentation checks

```bash
# all Python files
tooling/winhint.sh docs:check
# selected files
tooling/winhint.sh docs:check sim/se.py
# strict site build → build/site
tooling/winhint.sh docs:build
```

See [Writing documentation](../contributing/documentation.md).

## Next

- [Interfaces](../interfaces.md): the contract the tests hold every component to.
- [Commit gate and coverage](../contributing/quality.md): which of these suites gate a commit.
