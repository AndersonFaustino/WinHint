# Quickstart

A first pass through the system, from an empty checkout to a hinted kernel running under
QEMU. It exercises the core idea of WinHint on one
workload: the compiler plugin estimates how large an out-of-order
[window](../reference/glossary.md#window) each loop needs and inserts
[`setwin`](../reference/glossary.md#setwin) [hints](../reference/glossary.md#hint) that leave
the program's output unchanged. No simulator is involved yet; gem5 and real hardware come
later in [Running experiments](../guide/usage.md).

**Before you read:** [Installation](installation.md) is done and the `winhint` env is active.

## 1. Look around

```bash
# every command of the manager
tooling/winhint.sh help
# tools, builds, heavy lock
tooling/winhint.sh status
```

## 2. Build the compiler plugins

```bash
# → build/compiler/WinHint.so, build/compiler/JonesIQ.so
tooling/winhint.sh compiler:build
```

`WinHint.so` is the WinHint pass; `JonesIQ.so` is the compiler baseline B6. Both are LLVM
plugins loaded by `clang` ([Compiler](../guide/compiler.md)).

## 3. Build and run a kernel

```bash
# → build/benchmarks/riscv/plain/
make -C benchmarks ARCH=riscv VARIANT=plain
qemu-riscv64 build/benchmarks/riscv/plain/encoder_bert_tiny_infer small
```

`plain` is the unhinted [build variant](../reference/glossary.md#variant). The kernel
is one BERT encoder layer; `small` is its short input. Every kernel prints a float checksum
and an FNV-1a hash of its output ([Workloads](../guide/workloads.md#running-a-kernel)).

## 4. Build the hinted variant and check it is bit-identical

```bash
# setwin hints placed by the plugin
make -C benchmarks ARCH=riscv VARIANT=winhint
# output identical to plain under qemu
make -C benchmarks ARCH=riscv VARIANT=winhint check
```

The hints are architectural no-ops (an `ori x0,x0,imm` on RISC-V,
[Interfaces §2](../interfaces.md#2-the-hint-isa-contract)), so the hinted binary must produce
exactly the same output as the plain one; `check` prints `[SAME]` per kernel. Next to each
binary the plugin writes `<kernel>.regions.json` (the [regions](../reference/glossary.md#region)
and their predicted windows) and `<kernel>.winhint.json` (per-kernel statistics, schema in
[Compiler statistics](../reference/stats-schema.md)).

## 5. Run the tests

```bash
# env + python + compiler + docs (no gem5)
tooling/winhint.sh test all
```

What each suite covers is in [Testing and verification](../guide/testing.md).

## 6. Go further

| Goal | Read |
|------|------|
| Understand the moving parts | [How WinHint works](../guide/architecture.md) |
| Run the gem5 study or the hardware campaign | [Running experiments](../guide/usage.md) |
| Reproduce the paper | [Reproduce everything](../guide/usage.md#part-3-reproduce-everything) |
| Browse this site locally | `tooling/winhint.sh docs:serve`, then <http://127.0.0.1:8000> |

## Next

- [Repository layout](../estrutura.md): where each component lives.
- [Running experiments](../guide/usage.md): the full runbook.
