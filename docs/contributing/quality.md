# Commit gate and coverage

This page is for anyone changing the code. WinHint's results are only as trustworthy as the
tools that produce them, so every commit must pass the same checks: the toolchain matches its
pins, every host test suite passes with more than 90 % line coverage in each language, and the
documentation builds without warnings. This page explains that gate, how coverage is measured,
what is excluded and why, and what to do when it fails. The test suites themselves are
described in [Testing and verification](../guide/testing.md).

Every commit must pass `make validate`. There is no hosted CI: a git pre-commit hook is the
gate, and it refuses the commit when validation fails.

## Install the hook

```bash
make install-hooks
```

This runs `pre-commit install` from the `winhint` env (pre-commit is in
[`requirements-dev.txt`](../../requirements-dev.txt)). The hook is defined in
[`.pre-commit-config.yaml`](../../.pre-commit-config.yaml): one local hook whose entry is
`make validate`. pre-commit stashes unstaged changes before running it, so what is validated
is exactly what is being committed.

`git commit --no-verify` skips the hook. Use it only on work-in-progress branches; the
commit that lands must pass the gate.

## What `make validate` checks

`make validate` runs `tooling/winhint.sh validate`, in three steps. It needs no gem5 build and
no root.

| Step | Command | Fails when |
|------|---------|------------|
| 1. Environment | `tooling/winhint.sh test env` | a tool version differs from [`tooling/versions.lock`](../reference/toolchain.md), or the RISC-V toolchain cannot build and run hello-world under QEMU |
| 2. Tests + coverage | `tooling/winhint.sh coverage` ([`tooling/coverage_gate.py`](../api/python/tooling/coverage_gate.md)) | any test fails, any language is at or below **90 %** line coverage, or a file in scope has no coverage data |
| 3. Documentation | `tooling/winhint.sh test docs` | an undocumented Python definition, a malformed docstring, or any warning in the strict site build |

The gem5 simulation tests (`tooling/winhint.sh test sim`) and the correctness sweep
(`tooling/winhint.sh verify`) are not part of the gate: they need the gem5 builds and take
far longer. Run them before a release or when you change the simulator.

## The coverage gate

`tooling/coverage_gate.py` builds instrumented copies of the code under `build/coverage/`, runs
every host test suite against them and computes line coverage per language. **Each language
must be strictly above 90 %.**

| Language | Code measured | Driven by |
|----------|---------------|-----------|
| Python | every module under `sim/`, `analysis/`, `hw/`, `tooling/`, `compiler/baselines/` (`.coveragerc`), imported by a test or not | pytest, one process per `tests/` directory |
| C++ | LLVM plugins: `compiler/winhint/`, `compiler/common/`, `compiler/baselines/jones_iq/` | `compiler/test/run_tests.sh` (incl. the lit suite) on plugins rebuilt with `--coverage -O0` |
| C++ | gem5 window policies, LUT, LTP core: `sim/gem5/src/cpu/o3/window/` | the host unit tests (gem5 stubs, no gem5 build) |
| C | the 15 kernels and the micro-benchmarks: `benchmarks/` | native x86 builds with `--coverage`, untiled and `-DTILED`, each run on `small` |

Gcov data comes from conda's GCC `gcov` (the plugins and unit tests are built with the env's
GCC) and `llvm-cov gcov` (the kernels are built with clang); `gcovr` reads both, and the gate
merges the builds per file and line.

A file in scope that no test compiles or imports would be invisible to gcov and inflate the
percentage, so the gate lists every C/C++ source in scope and fails when one is missing from
the report. For Python, `.coveragerc`'s `source` setting does the same: an unimported module
counts as 0 %.

### What is excluded, and why

Exclusions are explicit and commented where they are configured. There are two kinds:

| Excluded | Why | Tested by |
|----------|-----|-----------|
| `sim/gem5/src/cpu/o3/window/controller.{cc,hh}`, `ltp/ltp.{cc,hh}`, `ltp/ltp_iew.cc`, `ltp/ltp_hooks.hh`, `ltp/ltp_policy.cc` (`CXX_EXCLUDED` in `tooling/coverage_gate.py`) | gem5-internal glue: compiles only inside the gem5 tree against the O3 CPU | `tooling/winhint.sh test sim` on the `RISCV_winhint` build |
| `sim/se.py` (`.coveragerc`) | runs only inside gem5's embedded Python (`import m5`) | `tooling/winhint.sh test sim` |
| `hw/run_hw_experiments.py` (`.coveragerc`); the C code under `hw/` (not in the C/C++ scope) | needs an Intel hybrid CPU, RAPL, perf counters and, for most configurations, root | the measurement campaign ([Real hardware](../guide/hardware/index.md)) |

Tests and test inputs (`*/tests/*`, `compiler/test/`) are not measured.

### Reading the reports

```bash
# all three languages
make coverage
# one language: python, cxx or c
tooling/winhint.sh coverage python
# report only, never fails
tooling/winhint.sh coverage --no-gate cxx
```

The gate prints every file with its covered/total lines and writes:

| File | Content |
|------|---------|
| `build/coverage/python.json`, `cxx.json`, `c.json` | per-file and total line counts, missing files |
| `build/coverage/python-html/index.html` | annotated Python sources |
| `build/coverage/*.gcovr.json` | raw gcovr output per C/C++ build |

### When the gate fails

- **Below 90 %**: add tests for the code you changed; the per-file table shows where. Tests
  must check behavior, not only execute lines.
- **MISSING file**: a new C/C++ source in scope has no test that compiles it. Add it to a host
  test; if it can only run inside gem5 or on real hardware, add it to the exclusion list with
  the reason, in the same commit.
- An exclusion pragma (`# pragma: no cover`, `LCOV_EXCL_LINE`) is acceptable only for code that
  cannot run on the host by construction, with a one-line reason next to it.

## Next

- [Writing documentation](documentation.md): the docstring and Doxygen rules that step 3
  enforces.
- [Implementation checklist](../checklist.md): what is implemented, per proposal phase.
