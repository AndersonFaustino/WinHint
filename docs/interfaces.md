# Interfaces

WinHint is several programs written separately: a compiler pass that emits hints, a gem5 model
and a hardware runtime that act on them, and scripts that turn their output into results
([How WinHint works](guide/architecture.md)). This page is the contract that lets them agree:
the hint encodings, the [window configuration](reference/glossary.md#window-configuration)
table, the gem5 parameters, the build variants and the file formats that cross a component
boundary. Every other page that mentions one of these formats links here instead of
restating it.

This file is the contract between the compiler (`compiler/`), the simulator
(`sim/`, gem5 patches), the workloads (`benchmarks/`), the analysis pipeline (`sim/baselines/`, `analysis/`) and the
real-hardware runtime (`hw/`). Change it only together with every consumer.

**Before you read:** the terms [`setwin`](reference/glossary.md#setwin),
[region](reference/glossary.md#region) and [ROB](reference/glossary.md#rob) are defined in the
[Glossary](reference/glossary.md). How each side uses the contract:
[Compiler](guide/compiler.md), [gem5 model](guide/gem5/index.md),
[Real hardware](guide/hardware/index.md), [Workloads](guide/workloads.md).

## 1. Environment

- **Host only, no Docker.** Every tool comes from conda packages in the micromamba
  env `winhint` (created by `tooling/create_conda_env.sh`; activate with
  `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate winhint`,
  or prefix commands with `~/.local/bin/micromamba run -n winhint`).
  Clairvoyance uses the separate env `winhint-llvm38`.
- `WINHINT_ROOT` = repository root; `WINHINT_BUILD` = `$WINHINT_ROOT/build` (gitignored).
  Everything fetched or generated goes under `build/`:

| What | Path |
|------|------|
| gem5 source | `build/gem5/src` |
| gem5 builds | `build/gem5/src/build/RISCV_clean/gem5.opt`, `build/gem5/src/build/RISCV_winhint/gem5.opt` |
| McPAT | `build/mcpat/mcpat` |
| LLVM plugin build | `build/compiler/` (→ `WinHint.so`, `JonesIQ.so`) |
| Benchmark binaries | `build/benchmarks/<arch>/<variant>/<kernel>` (`arch` = `riscv` or `x86`) |
| Fidelity microbenchmarks (PROPOSAL §7; `make -C benchmarks micro`, `sim/fidelity/`) | `build/benchmarks/micro/<arch>/<variant>/<micro kernel>` |
| libwinhint and hw tools | `build/libwinhint/`, `build/hw/` |
| Clairvoyance passes | `build/clairvoyance/` |
| R2–R4 baseline builds | `build/hw-baselines/` |

- Experiment results (CSV/JSON/figures) go to `results/` (gitignored).
- **Resource limit:** 12 threads, 6 GB of RAM. Heavy jobs (gem5 builds, gem5 simulations
  longer than about a minute, McPAT/scx/Clairvoyance/PMCTrack builds) must run under
  `flock "$WINHINT_BUILD/.heavy.lock" <cmd>`, so only one runs at a time, with at most `-j2`
  inside it. Every other compile uses `-j1`.

## 2. The hint ISA contract

Two hint kinds. Both are architectural no-ops on an unmodified core.

| Kind | Meaning | `kind` code |
|------|---------|-------------|
| `setwin(W)` | Advisory: "use at most a W-entry window from here on". `W = 0` means "release" (back to the default/max). | 1 |
| `region(id)` | Marks entry into static region `id` (oracle, PGO and per-region statistics). It does not change the window. | 2 |

### RISC-V encoding (HINT space, `ORI rd=x0, rs1=x0`)

```
ori x0, x0, IMM        IMM = (payload << 5) | tag      (12-bit signed imm; payload 0..63)
tag = 0b10101 (21) → setwin,  W = payload * 8   (W ∈ {0, 8, ..., 504})
tag = 0b10111 (23) → region,  id = payload       (id ∈ 0..63)
```

The tags avoid Zicbop `prefetch.{i,r,w}` (imm[4:0] = 0, 1, 3). Raw word:
`0x00006013 | (IMM << 20)`. In asm: `.insn i 0x13, 6, x0, x0, IMM`.

### x86-64 encoding (unique multi-byte NOP)

```
nopl DISP32(%rax)     bytes: 0F 1F 80 <disp32 little-endian>
DISP32 = 0x57480000 | (kind << 12) | payload        ('W','H' magic in the top 16 bits)
```

`payload` has the same meaning on both ISAs: `W / 8` for setwin and `id` for region.

### Runtime-call mode (real hardware)

With `-mllvm -winhint-emit=call`, the pass emits calls instead of NOPs:

```c
// hw/libwinhint/winhint.h
void __winhint_setwin(unsigned w);   // W in entries, 0 = release
void __winhint_region(unsigned id);
```

`-winhint-emit=` takes `asm` (default; ISA hint chosen from the target triple), `call` or `none`.

## 3. Window configuration table

A window configuration scales the ROB, the IQ, the LQ and the SQ together. The table is defined in
`sim/machines/*.json` under `"window"` and passed to gem5 as parallel vector
params. The default machine (`riscv_ooo.json`) uses:

| index | ROB | IQ | LQ | SQ |
|-------|-----|----|----|----|
| 0 | 64  | 32 | 16 | 16 |
| 1 | 128 | 64 | 32 | 32 |
| 2 | 192 | 96 | 48 | 48 |
| 3 | 256 | 128| 64 | 64 |

The physical structures are sized for the largest configuration, and the register file is
rebalanced so that it is not the binding limit. A `setwin(W)` selects the
**smallest configuration with ROB ≥ W**. If no configuration is large enough, or W = 0, it selects the largest.

## 4. gem5 interface (`RISCV_winhint` build)

SimObject params on `BaseO3CPU`:

| Param | Type | Meaning |
|-------|------|---------|
| `window_policy` | `String`: one of `static`, `occupancy`, `mlp`, `bbv`, `lut`, `hint`, `hybrid`, `ltp` (B9; a string, not an enum: `static` is a C++ keyword); invalid values are rejected in `se.py` and with `fatal()` | decision policy (default `static`) |
| `window_args` | `String`, `k=v,...` | per-policy tunables for the baselines (`se.py --window-args`) |
| `window_rob` / `window_iq` / `window_lq` / `window_sq` | `VectorParam.Unsigned` | the config table |
| `window_initial` | `Unsigned` | starting config index (static: the fixed one) |
| `window_period` | `Cycles` (default 1000) | sampling period for the reactive policies |
| `window_lut_file` | `String` | B5 LUT file (see §6) |
| `window_trace` | `Bool` | write `window_trace.csv` to the outdir |
| `window_l1d` / `window_l2` | `Param.BaseCache(NULL)` | caches whose miss counters feed the policies |

`sim/se.py` flags: `--machine <json>`, `--window-policy`, `--window-initial`,
`--window-period`, `--window-lut`, `--window-trace`, `--window-args`, `--no-window` (do not
pass the window table to gem5 at all; used with the `RISCV_clean` build).
Run length (method and per-kernel budgets: `sim/run_lengths.json`, versioned): `--maxinsts`
(measured instructions), `--warmup-insts` (detailed warm-up, stats reset after it),
`--fast-forward N` (in-process AtomicSimpleCPU, then switch to the O3 `system.cpu`),
`--take-checkpoints N1,..` + `--checkpoint-dir` and `--profile-regions FILE` (functional passes;
`--profile-cap N`, default 2048, bounds the visits recorded per region-marker PC),
`--restore-checkpoint DIR` + `--window-seed auto|off` (hint/hybrid start in the config of the
last setwin before the checkpoint). Such runs also write `runlength.json` to the outdir.
Sampled `large` runs of `sim/run_experiments.py` keep one sub-outdir per sample (`sNN/`) and
merge them into the outdir files below (`stats.txt` = stratified whole-program estimate).
gem5 v25.1 sizes the IQ through `cpu.instQueues = [IQUnit(...)]` instead of `numIQEntries`; `se.py` hides this.

Outputs in the gem5 outdir:
- `stats.txt` (standard), with new stats under `system.cpu.window.*`;
- `window_trace.csv`, one row per period:
  `cycle,insts,ipc,rob_occ_mean,iq_occ_mean,lq_occ_mean,l1d_mpki,l2_mpki,mlp,branch_mpki,config,region`;
- `region_stats.csv`, one row per region visit:
  `region,config,enter_cycle,cycles,insts`. It is written for every policy whenever region markers are present.

## 5. Benchmarks and compiler flags

- Kernels: `benchmarks/<kernel>.c`. Each kernel accepts an input-size argument:
  `argv[1] = small|large` (default `large`). B7 trains on `small` and tests on `large`.
  Same model shapes for both; `small` shrinks only the input (≈40–550 M RISC-V instructions at -O2, so
  it is simulable in gem5 O3); each kernel prints one checksum/hash line, bit-identical across platforms.
- Build variants (Makefile targets, `ARCH=riscv|x86`):

| Variant | What it emits |
|---------|---------------|
| `plain` | no hints |
| `winhint` | WinHint `setwin` hints (static model) |
| `oracle` | `region(id)` markers only (the pass is in region-marker mode, `-winhint-mode=regions`; `-DWINHINT_ORACLE` is also defined, but no kernel reads it) |
| `oracle_hinted` | `setwin` from a per-region best-config JSON (`results/oracle/<kernel>.json`; machine M ≠ `riscv_ooo`: `results/oracle/M/large/<kernel>.json`) |
| `jones` | B6 IQ-only hints (and `jones_full`: extended to ROB/LSQ) |
| `pgo` | B7 hints from a profile of the `small` input (`results/pgo/<kernel>.json`; machine M: `results/oracle/M/small/<kernel>.json`) |
| `clairvoyance` | B8-transformed code, no hints (`winhint_clairvoyance`: plus WinHint hints) |
| `winhint_call` / `oracle_call` | x86 runtime-call mode for real hardware (`-winhint-emit=call`, linked with libwinhint): `setwin` hints / `region` markers as calls to `__winhint_setwin` / `__winhint_region` (§2) |

- Region ids are assigned by the pass deterministically. The module's defined functions (as
  seen by the pass, after inlining) are sorted by symbol name (byte-wise); within each function,
  each top-level (outermost) loop nest is one region, in program order (reverse-post-order
  position of its header block). Ids are consecutive from 0 across the module; inner loops get
  no id. The pass writes `<kernel>.regions.json`
  mapping each id to `{function, loop header, source line}`, so the oracle and B7 can map stats back.
- Target description for the cost model: `sim/machines/*.json` (`"cpu"`, `"cache"`, `"memory"`, `"window"` sections), passed with `-mllvm -winhint-target=<json>`.

## 6. B5 LUT file format (`window_lut_file`)

Plain text, produced by `sim/baselines/lut/export_lookup_table.py --runtime-lut`:

```
WINHINT_LUT 1
features 4 ipc rob_occ l1d_mpki mlp
edges ipc <n> e1 ... en          # n ascending upper bin edges; values above en go in bin n
edges rob_occ <n> ...
edges l1d_mpki <n> ...
edges mlp <n> ...
table <N>                        # N = Π(n_i+1)
c0 c1 ... c(N-1)                 # config indices, row-major (first feature slowest)
```

Labels are the **oracle-best configuration** of the region that each window belongs to (this fixes
problem 1). Features are per-window deltas from `window_trace.csv` (this fixes problem 3).

## Next

- [Compiler statistics](reference/stats-schema.md): the schema of `<kernel>.winhint.json`.
- [Toolchain](reference/toolchain.md): the pinned tool versions.
