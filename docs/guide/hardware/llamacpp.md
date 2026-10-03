# llama.cpp integration

This optional workload checks WinHint's real-hardware mechanism on a real application rather
than on the benchmark kernels: llama.cpp's `llama-simple` text generation, with hooks in the
ggml CPU backend that request a P-core or an E-core per operator or per layer through
libwinhint (PROPOSAL Phase E2, optional). It is part of the real-hardware campaign
([Real hardware](index.md), [Evaluation methodology](../../concepts/methodology.md)).

**Terms.** *ggml* is llama.cpp's tensor library; a *graph node* is one operator of the model's
compute graph; *libwinhint* is the runtime that turns `setwin(W)` into a P/E-core migration
([Real hardware §2](index.md#2-libwinhint)). Other terms are in the
[Glossary](../../reference/glossary.md).

## Hooks

The hooks call the libwinhint runtime-call API from inside the ggml CPU backend
([`hw/libwinhint/winhint.h`](../../../hw/libwinhint/winhint.h), [interfaces.md §2](../../interfaces.md#runtime-call-mode-real-hardware)). For every graph node that thread 0
computes, they call `__winhint_region(id)` when the region changes and `__winhint_setwin(W)`
when the advisory window changes. At the end of each graph they call `__winhint_setwin(0)`.
llama.cpp is never vendored. It is cloned at a pinned tag into `build/` and patched there.

```
hw/integrations/llamacpp/
├── winhint-ggml-hooks.patch   against llama.cpp tag b11327 (552f18f9, 2026-10-01)
│     ggml/CMakeLists.txt              option GGML_WINHINT, GGML_WINHINT_INCLUDE, GGML_WINHINT_LIB
│     ggml/src/ggml-cpu/CMakeLists.txt -DGGML_WINHINT, include dir, link libwinhint
│     ggml/src/ggml-cpu/ggml-cpu.c     hook call before each node, release at graph end (#ifdef GGML_WINHINT)
│     ggml/src/ggml-cpu/winhint-hooks.h  op-class table, region/setwin logic, env parsing
└── build_llamacpp.sh          fetch | build | model | check | all
```

## Build and functional check (host, conda env `winhint`)

```sh
eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate winhint

# libwinhint.a, if not built yet
make -C hw -j1

# fetch + build + model + check
bash hw/integrations/llamacpp/build_llamacpp.sh all
```

- `build` makes two static builds of `llama-simple` (greedy decoding). It runs
  under `flock build/.heavy.lock` with `-j2` (`JOBS=`) and uses only conda cmake, ninja,
  GCC 13 and OpenMP:
  - `build/integrations/llamacpp/build-vanilla/bin/llama-simple`, with `GGML_WINHINT=OFF`;
  - `build/integrations/llamacpp/build-winhint/bin/llama-simple`, linked with
    `build/libwinhint/libwinhint.a`.
- `model` downloads `stories15M-q4_0.gguf` (19 MB, the model llama.cpp's own CI uses) from
  `huggingface.co/ggml-org/models` and checks its SHA-256.
- `check` is a functional check, not a measurement. The vanilla run and the hooked runs with
  `WINHINT_MODE=off`, `WINHINT_MODE=log` (op regions), `log` + `GGML_WINHINT_REGIONS=layer` and
  `log` + `GGML_WINHINT_SETWIN=none` must produce byte-identical text. libwinhint's
  per-region CSV (`WINHINT_LOG`) must list the matmul (1) and norm (2) regions in op mode and
  layer 0 (region 16) in layer mode. The outputs go to `build/integrations/llamacpp/check/`.

## Runtime knobs

These are the patch side's variables; libwinhint's own `WINHINT_*` variables are in
[Real hardware §2](index.md#2-libwinhint).

| Variable | Values | Meaning |
|---|---|---|
| `GGML_WINHINT_REGIONS` | `op` (default) / `layer` / `off` | op: region = operator class 0..9. layer: region = 16 + (layer % 48), parsed from llama.cpp node names `name-<il>`; nodes without a layer use their op class. off: no calls |
| `GGML_WINHINT_SETWIN` | `c:W,...` / `none` | override the per-class window, or emit region markers only (oracle/R5-style accounting) |

Op classes and default windows (ROB entries): 0 other (release), 1 MUL_MAT/MUL_MAT_ID/OUT_PROD
256, 2 norms 64, 3 SOFT_MAX 64, 4 ROPE 64, 5 FLASH_ATTN_EXT 256, 6 binary elementwise 128,
7 unary/GLU 64, 8 data movement (CPY/CONT/GET_ROWS/...) 128, 9 reductions 64. With the default
`WINHINT_THRESHOLD=192`, matmul and attention go to P-cores and everything else goes to E-cores.
This table is a static placeholder. The WinHint pass does not analyse ggml's kernels: they are
C/C++ with SIMD intrinsics and are compiled by GCC here. Replacing the table with
compiler-derived windows is future work.

Caveats: only thread 0 (the caller, which is the OpenMP master) calls libwinhint, and libwinhint
migrates only the calling thread. `llama-simple` has no `-t` option and, run by hand, uses
llama.cpp's default thread count. The campaign driver therefore limits the threads itself (see
below). Another route to a single-thread placement experiment is a llama.cpp tool that accepts
`-t 1` (e.g. `llama-completion`; add its target in `build_one`). Fused ops
(`ggml_cpu_try_fuse_ops`) take the region of their first node.

## Measurement with the campaign driver

`hw/run_hw_experiments.py --llamacpp` runs llama.cpp in place of the benchmark kernels
([Real hardware §5.3](index.md)):

- **Binaries.** `plain` → `build-vanilla/bin/llama-simple`, and the `call`/`regions` roles
  → `build-winhint/bin/llama-simple`, under `--llamacpp-dir` (default
  `build/integrations/llamacpp`).
- **Arguments.** `--llamacpp-args`, default
  `-m <llamacpp-dir>/models/stories15M-q4_0.gguf -n 256 'Once upon a time'`.
- **Configurations.** `WH` runs `WINHINT_MODE=migrate`, `WH-off` runs `off`, and `R5-Sondag`
  adds `GGML_WINHINT_SETWIN=none` (region markers only) to `WINHINT_MODE=sondag`.
- **One thread.** Every configuration runs with `OMP_THREAD_LIMIT=1 OMP_NUM_THREADS=1`, which
  caps ggml's OpenMP team at one thread, so all of them do the same single-threaded work.
- **No NOP rows.** There is no NOP-hint build, so the `NOP-*` rows are dropped (named
  explicitly in `--configs`, they are an error).

```sh
# root for RAPL energy and the fixed frequency policy (results are chowned back)
sudo -E env "PATH=$PATH" python hw/run_hw_experiments.py --llamacpp \
    --llamacpp-dir build/integrations/llamacpp \
    --configs R0-P,R0-E,R1,WH,WH-off,R5-Sondag --reps 10 \
    --governor performance --epp performance --no-turbo 1 \
    --allow-system-changes --out results/hw/llamacpp
```

The runbook entry is [llama.cpp (optional)](../usage.md#llamacpp-optional). A larger
model, still tiny, would be better for timing, e.g. stories110M or TinyLlama-1.1B Q4.

## Next

- [Real hardware](index.md): the campaign, its configurations and its method.
- [Deviations](../../deviations.md#real-hardware): the llama.cpp integration's departures from the plan.
