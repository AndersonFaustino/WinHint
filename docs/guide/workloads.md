# Workloads

The workloads are what WinHint is evaluated on: 15 ML-inference kernels in portable C11 under
`benchmarks/`, and the Makefile that builds each of them in every
[build variant](../reference/glossary.md#variant) (plain, WinHint-hinted, oracle, and
the compiler baselines). ML inference is the target because its phases (matrix products,
attention, normalization) differ sharply in how much memory-level parallelism they expose,
which is what the window size trades off ([Background](../concepts/background.md)). The
same binaries run under QEMU, in gem5 and natively on x86, and must print bit-identical
output. A separate set of baseline-fidelity microbenchmarks lives in `benchmarks/micro/`.

**Before you read:** [How WinHint works](architecture.md) for the role of
[regions](../reference/glossary.md#region) and hints. API reference:
[api/cpp/benchmarks](../api/cpp/benchmarks/files.md).

## Kernels

Each kernel keeps its phases in separate non-inlined functions with separate loop nests, so
each phase is its own WinHint region. Model shapes are the same for both inputs; `small`
shrinks only the input (sequence, prompt, batch, image), so both inputs execute the same code
regions. This is what B7 needs: it trains on `small` and tests on `large`. Every file header
documents its shapes, phases and working sets.

| Group | Kernel | Model | `large` / `small` |
|-------|--------|-------|-------------------|
| encoder | `encoder_bert_tiny_infer` | BERT-base layer: H 768, 12 heads, FFN 3072, post-LN, GELU | seq 128 / 4 |
| encoder | `encoder_roberta_infer` | RoBERTa-base layer with padding mask | seq 256 / 4 |
| encoder | `encoder_vit_micro_infer` | ViT-B/16 layer, pre-LN | image 224² (197 tokens) / 32² (5) |
| decoder | `decoder_gpt2_infer` | GPT-2-small layer, KV cache, vocab 8192 | prompt 96 + 32 generated / 4 + 2 |
| decoder | `decoder_llama_mini_infer` | LLaMA-style: RoPE, RMSNorm, SwiGLU 2048, GQA 12/4 | 64 + 64 / 2 + 2 |
| decoder | `decoder_beam_search_infer` | GPT-2-small, 4 beams, KV-cache reorder | 64 + 16 / 2 + 2 |
| contrast | `contrast_mlp_infer` | MLP 1024-1536-1536-1536-256 | batch 32 / 4 |
| contrast | `contrast_mobilenet_infer` | MobileNetV1, 6 depthwise-separable blocks | 128² / 64² |
| contrast | `contrast_siamese_infer` | shared-weight MLP towers 512-1024-512-128 | 128 / 8 pairs |
| imggen | `imggen_conv_autoenc_infer` | conv autoencoder 1-16-32-64 | 128² / 64² |
| imggen | `imggen_unet_infer` | UNet decoder 128-64-32-16 | latent 8² / 4² |
| imggen | `imggen_vae_decode_infer` | VAE decoder (16 MB FC) | 8 / 2 latents |
| control | `regression_linear_infer` | linear regression | batch 2048 / 512 |
| control | `regression_ridge_infer` | ridge regression | batch 4096 / 1024 |
| control | `regression_svr_infer` | RBF support-vector regression | 1024 / 256 test points |

| Kernel | Phase functions |
|--------|-----------------|
| `encoder_bert_tiny_infer` | linear, attention_scores, softmax_rows, attention_context, residual_add, layer_norm, gelu_inplace |
| `encoder_roberta_infer` | linear, attention_scores, softmax_rows (masked), attention_context, residual_add, layer_norm, gelu_inplace |
| `encoder_vit_micro_infer` | patchify (im2col), linear, add_position, attention_scores, softmax_rows, attention_context, residual_add, layer_norm, gelu_inplace |
| `decoder_gpt2_infer` | embed, layer_norm, linear, kv_append, attention_scores, softmax_causal, attention_context, residual_add, gelu_inplace, argmax |
| `decoder_llama_mini_infer` | embed, rms_norm, linear, rope, kv_append, attention_scores, softmax_causal, attention_context, swiglu, residual_add, argmax |
| `decoder_beam_search_infer` | embed, layer_norm, linear, kv_append, attention_scores, softmax_causal, attention_context, residual_add, gelu_inplace, log_softmax_rows, beam_select, kv_reorder |
| `contrast_mlp_infer` | linear, relu_inplace |
| `contrast_mobilenet_infer` | conv3x3_stem, depthwise_conv3x3, pointwise_conv, bn_relu6, global_avg_pool, fc |
| `contrast_siamese_infer` | linear, relu_inplace, l2_normalize, cosine_scores |
| `imggen_conv_autoenc_infer` | conv3x3, relu_inplace, maxpool2, upsample2, sigmoid_inplace |
| `imggen_unet_infer` | upsample2, conv3x3, skip_add_relu, conv1x1 |
| `imggen_vae_decode_infer` | linear, relu_inplace, upsample2, conv3x3, sigmoid_inplace |
| `regression_linear_infer` | predict, mse, gradient_step (2 kB column stride) |
| `regression_ridge_infer` | predict, residuals, gradient, update_weights, ridge_loss |
| `regression_svr_infer` | sq_distances, rbf_inplace (`expf`), weighted_sum |

Transformer layer weights are 24–28 MB per layer, far beyond a 256 kB–1 MB L2. A `small` run
is about 40–550 M RISC-V instructions at `-O2` (QEMU count), and a `large` run about
0.2–13.6 G. The regression kernels are the non-transformer control group.

## Running a kernel

```text
<kernel> [small|large] [N]
```

- `argv[1]`: input, default `large`.
- `argv[2]`: number of layers for the encoders and decoders, number of repetitions for the
  other kernels (default 1, must be ≥ 1).
- A bad argument exits with status 2.

```bash
qemu-riscv64 build/benchmarks/riscv/plain/encoder_bert_tiny_infer small
```

Each kernel first prints a configuration line (shapes, layers/reps and `gemm=tiled|naive`),
then one result line:

```text
[encoder_bert_tiny] input=small checksum=<%.9e> hash=0x<8 hex digits>
```

- `checksum`: double-precision sum of the outputs.
- `hash`: 32-bit FNV-1a over the bytes of the output floats, mixed with an extra word (for
  example, generated token ids).

Kernels are deterministic (xorshift32 PRNG, fixed summation order) and compiled with
`#pragma STDC FP_CONTRACT OFF`, so the line is bit-identical across QEMU, gem5 and native
x86 for the same source. That is what `make check` and `tooling/winhint.sh verify` compare.

## Compile-time knobs

| Macro | Kernels | Effect |
|-------|---------|--------|
| `-DTILED` (Makefile `TILED=on`) | kernels with a GEMM `linear` | Tiled GEMM (below). |
| `-DHIDDEN`, `-DNUM_HEADS`, `-DFFN_DIM` | encoders, decoders | Model shape. |
| `-DNUM_KV_HEADS` | `decoder_llama_mini` | GQA key/value heads (default 4). |
| `-DVOCAB` | decoders | Vocabulary (default 8192). |
| `-DSEQ_LARGE`, `-DSEQ_SMALL` | BERT, RoBERTa | Sequence length. |
| `-DPROMPT_{LARGE,SMALL}`, `-DGEN_{LARGE,SMALL}` | decoders | Prompt and generated tokens. |
| `-DIMG_{LARGE,SMALL}` | ViT, MobileNet, conv autoencoder | Image side. |
| `-DLAT_{LARGE,SMALL}` | UNet | Latent side. |
| `-DBATCH_{LARGE,SMALL}` | `contrast_mlp`, `imggen_vae_decode`, `regression_linear`, `regression_ridge` | Batch. |
| `-DPAIRS_{LARGE,SMALL}` | `contrast_siamese` | Pairs. |
| `-DTEST_{LARGE,SMALL}` | `regression_svr` | Test points. |
| `-DWIDTH` | `contrast_mlp` | Hidden width (default 1536). |

`-DTILED` applies to every kernel with a GEMM `linear` (encoders, decoders, `contrast_mlp`,
`contrast_siamese`, `imggen_vae_decode`). It selects a 4×4 register-blocked, N-outer
cache-tiled GEMM. Each output keeps its ascending-k summation order, so the output is
bit-identical to the untiled build.

Pass shape overrides through the Makefile with `CFLAGS_EXTRA=-D...`, for example for a very
short simulation:

```bash
make -C benchmarks ARCH=riscv VARIANT=plain KERNELS=encoder_bert_tiny_infer \
    CFLAGS_EXTRA="-DSEQ_LARGE=16"
```

!!! warning
    `CFLAGS_EXTRA` does not change the output directory. Builds with different
    `CFLAGS_EXTRA` overwrite each other; set `OUT_ROOT` to keep them apart.

## `benchmarks/Makefile`

Run inside the `winhint` env. Benchmarks are compiled with `clang` (variable `CLANG`), not
`$CC`:

```bash
# → build/benchmarks/riscv/plain/
make -C benchmarks ARCH=riscv VARIANT=plain
# → build/benchmarks/riscv/plain-O3-tiled/
make -C benchmarks ARCH=riscv VARIANT=plain OPT=O3 TILED=on
# bit-identical vs plain under qemu
make -C benchmarks ARCH=riscv VARIANT=winhint check
# cost model of another machine → build/benchmarks/riscv/winhint/riscv_ooo_big/
make -C benchmarks ARCH=riscv VARIANT=winhint MACHINE=sim/machines/riscv_ooo_big.json
# kernels, variants, output directory
make -C benchmarks list
```

### Variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `ARCH` | `riscv` | `riscv`: static rv64gc (`--target=riscv64-conda-linux-gnu -march=rv64gc -mabi=lp64d`, `-static -fuse-ld=lld`) for gem5/QEMU. `x86`: native x86-64 (`-march=$(X86_MARCH)`, default `x86-64-v3`). |
| `VARIANT` | `plain` | See [Variants](#variants). Unknown names are an error. |
| `OPT` | `O2` | `O2` or `O3` (`-O2`/`-O3` also accepted). |
| `UNROLL` | `on` | `off` adds `-fno-unroll-loops`. |
| `TILED` | `off` | `on` adds `-DTILED`. |
| `KERNELS` | every `benchmarks/*.c` with `int main` | Subset to build. |
| `EMIT` | `asm` | `-winhint-emit` / `-jones-emit`: `asm`, `call`, `none`. Forced to `call` for the `*_call` variants. |
| `MACHINE` | `sim/machines/riscv_ooo.json` | Machine JSON for the cost model (`-winhint-target`, `-jones-target`). |
| `SWITCH_COST` | empty (model default) | `-winhint-switch-cost` in cycles. |
| `MIGRATION_US` | `switch_cost_us` from `results/hw/system/migration_cost_smt-on.json` (`MIGRATION_JSON`), if present | P/E switch cost for `EMIT=call` builds (`-winhint-switch-model=pe -winhint-migration-us=`). |
| `ORACLE_DIR` | `results/oracle` (default machine), else `results/oracle/<machine>/large` | Region maps for `oracle_hinted`. |
| `PGO_DIR` | `results/pgo` (default machine), else `results/oracle/<machine>/small` | Region maps for `pgo`. |
| `JONES_ENC` | `setwin` | `-jones-iq-encoding` for `jones`: `setwin` or `setiq` (proposed, not in the contract). |
| `CV_TYPE` | `consv` | B8 Clairvoyance type: `consv`, `specsafe`, `spec`, `multispecsafe`, `multispec`. |
| `CV_UNROLL`, `CV_INDIR`, `CV_BRANCH_PROB` | `2`, `1`, `0.9` | B8 knobs; ranges in `compiler/baselines/clairvoyance/knobs.json`. |
| `CFLAGS_EXTRA` | empty | Extra compiler flags (e.g. `-D` shape overrides). |
| `CHECK_INPUT` | `small` | Input used by `check`. |
| `MATRIX_VARIANTS` | `plain winhint oracle jones jones_full` | Variants built by `matrix`. |
| `MATRIX_CONFIGS` | all 8 `OPT:UNROLL:TILED` combinations | Configurations of `matrix-all`. |
| `MACHINES` | `sim/machines/*.json` | Machines of the `machines` target. |
| `CLANG` | `clang` | Compiler. |
| `RUN` | `qemu-riscv64` (riscv), empty (x86) | Runner used by `check`. |
| `RISCV_TRIPLE`, `RISCV_SYSROOT`, `X86_TRIPLE`, `X86_SYSROOT`, `X86_MARCH` | conda triples and sysroots, `x86-64-v3` | Cross-toolchain settings. |
| `WINHINT_ROOT`, `WINHINT_BUILD` | repository, `$(WINHINT_ROOT)/build` | Roots. |
| `OUT_ROOT` | `$(WINHINT_BUILD)/benchmarks` | Output root; must not be under `benchmarks/`. |
| `COMPILER_BUILD` | `$(WINHINT_BUILD)/compiler` | Location of `WinHint.so` / `JonesIQ.so`. |
| `LIBWINHINT_DIR` | `$(WINHINT_BUILD)/libwinhint` | `libwinhint.a` for call-mode builds. |
| `RESULTS_DIR` | `$(WINHINT_ROOT)/results` | Root of the region maps and the migration-cost JSON. |
| `CV_SCRIPT` | `compiler/baselines/clairvoyance/cv_compile.sh` | B8 build driver. |

Common flags: `-$(OPT) -std=c11 -Wall -gline-tables-only $(CFLAGS_EXTRA)`, plus `-lm`.

### Variants

All WinHint variants load the plugin with
`-fplugin=WinHint.so -fpass-plugin=WinHint.so` and pass `-winhint-target=$(MACHINE)`,
`-winhint-emit=$(EMIT)`, `-winhint-out-dir=<out dir>` and `-winhint-kernel=<kernel>`
([compiler options](compiler.md#command-line-options)).

| Variant | Pass flags | Emits | gem5 use |
|---------|-----------|-------|----------|
| `plain` | none | no hints | B0, B2–B5, B9 |
| `winhint` | `-winhint-mode=setwin` | `setwin` from the static model | WinHint, WinHint+HW |
| `oracle` | `-DWINHINT_ORACLE -winhint-mode=regions` | `region(id)` markers only | B1 sweep, run-length profile |
| `oracle_hinted` | `-winhint-mode=from-json=$(ORACLE_DIR)/<kernel>.json` | `setwin` per region from the B1 map | B1 |
| `pgo` | `-winhint-mode=from-json=$(PGO_DIR)/<kernel>.json` | `setwin` per region from the `small`-input profile | B7 |
| `jones` | `JonesIQ.so`, `-jones-mode=iq -jones-iq-encoding=$(JONES_ENC)` | IQ-only demand | B6 |
| `jones_full` | `JonesIQ.so`, `-jones-mode=full` | ROB/IQ/LQ/SQ demand | B6 extended |
| `clairvoyance` | Clairvoyance passes (LLVM 3.8) via `cv_compile.sh`, then LLVM 23 | transformed code, no hints | B8 |
| `winhint_clairvoyance` | Clairvoyance + `-winhint-mode=setwin` | transformed code + `setwin` | B8 + WinHint |
| `winhint_call` | `winhint` + `EMIT=call`, `-winhint-switch-model=pe`, linked with `libwinhint.a` | `__winhint_setwin` calls | real hardware (`ARCH=x86` only) |
| `oracle_call` | `oracle` + `EMIT=call`, `libwinhint.a` | `__winhint_region` calls | real hardware (`ARCH=x86` only) |

The variant table of the contract is [interfaces.md §5](../interfaces.md#5-benchmarks-and-compiler-flags).

- `oracle_hinted` and `pgo` fail with a message if their map is missing. The maps come from
  the oracle sweep ([Running experiments, step 3](usage.md#step-3-oracle-sweeps)) or from
  `compiler/baselines/pgo/pgo_flow.py all --kernel <k>`.
- The plugin-based variants fail if `WinHint.so` / `JonesIQ.so` are missing; build them first
  (`make plugins`).
- `EMIT=call` with `ARCH=riscv` is an error: `libwinhint` is x86-only.

### Targets

| Target | Effect |
|--------|--------|
| `all` (default) | Build `VARIANT` for every kernel in `KERNELS`. |
| `<variant>` | Shorthand for `VARIANT=<variant> all`; several names build one after the other (`make ARCH=x86 winhint_call oracle_call`). |
| `matrix` | Build every `MATRIX_VARIANTS` variant for the current `ARCH/OPT/UNROLL/TILED`. |
| `matrix-all` | `matrix` for every `MATRIX_CONFIGS` entry (O2/O3 × unroll on/off × tiled off/on). |
| `machines` | Build `VARIANT` once per `MACHINES` JSON (portability study). |
| `check` | Build `VARIANT` and `plain`, run both on `CHECK_INPUT` (`$(RUN)`; call-mode binaries with `WINHINT_MODE=log`) and compare stdout: `[SAME]`, `[DIFF]` or `[FAIL]` per kernel, non-zero exit on any mismatch. |
| `plugins` | `tooling/winhint.sh compiler:build`. |
| `clairvoyance-passes` | `tooling/winhint.sh clairvoyance:build` (B8 passes, env `winhint-llvm38`). |
| `list` | Print kernels, variants and the output directory. |
| `outdir` | Print the output directory. |
| `clean` | Remove the current output directory. |
| `micro`, `micro-check`, `micro-pgo`, `micro-list` | Microbenchmarks, see [below](#microbenchmarks). |

### Output paths

```text
$(WINHINT_BUILD)/benchmarks/<arch>/<variant><cfg>[/<machine>]/<kernel>
```

- `<cfg>` is empty for the default build (O2, unroll on, untiled, default B8 knobs).
  Otherwise it concatenates `-O3`, `-nounroll`, `-tiled` and, for B8 with non-default
  knobs, `-cv_<type>-u<unroll>-i<indir>[-bp<p>]`. Example: `plain-O3-nounroll-tiled`.
- `/<machine>` (stem of `MACHINE`) is added for `ARCH=riscv` and the model-dependent variants
  (`winhint`, `winhint_clairvoyance`, `jones`, `jones_full`, `oracle_hinted`, `pgo`) when
  `MACHINE` is not the default `riscv_ooo.json`. `sim/run_experiments.py` looks there first
  (`--bin-suffix` selects `<cfg>`).
- The compiler side files (`<kernel>.regions.json`, `<kernel>.winhint.json`,
  `<kernel>.jones.json` / `.jones_full.json`) are written next to the binaries. Nothing is
  written under `benchmarks/`.

## Microbenchmarks

`benchmarks/micro/` holds five integer-only kernels for the baseline-fidelity study
(PROPOSAL §7, [Simulator fidelity](fidelity.md); run in
[step 5](usage.md#step-5-baseline-fidelity) of the runbook). Each targets one condition that a hardware baseline
must react to:

| Kernel | Condition | `large` / `small` |
|--------|-----------|-------------------|
| `micro_gather` | independent long-latency misses over an 8 MiB table (MLP-rich) | 3 sweeps / 1 sweep |
| `micro_chase` | dependent misses: pointer chase over one random cycle of 64-byte nodes (MLP = 1) | 96 Ki / 32 Ki steps |
| `micro_compute` | cache-resident, high-ILP integer work | 160 / 48 matrix-vector products |
| `micro_lowilp` | serial multiply chain + ~50% mispredicted branches, low occupancy | 400 Ki / 128 Ki iterations |
| `micro_phased` | recurring `[gather → compute → lowilp]` phases, one region each | 6 / 3 rounds |

They take `argv[1] = small|large` (default `large`; `small` is a different, shorter input) and
print one line, bit-identical on QEMU, gem5 and every variant:

```text
[<name>] input=<small|large> checksum=0x<16 hex digits>
```

[`micro.mk`](../../benchmarks/micro/micro.mk) re-invokes the main Makefile with
`BENCH_DIR=benchmarks/micro` and a separate output root, so the toolchain and pass flags are
identical:

```bash
# MICRO_VARIANTS → build/benchmarks/micro/<arch>/<variant><cfg>/
make -C benchmarks micro
# each variant vs plain on small and large, under qemu
make -C benchmarks micro-check
# hinted builds from the <kernel>.json maps in <map dir> (written by sim/fidelity/fidelity.py)
make -C benchmarks micro-pgo MICRO_HINTED=pgo MICRO_MAP_DIR=<map dir>
make -C benchmarks micro-list
```

| Variable | Default |
|----------|---------|
| `MICRO_KERNELS` | every `benchmarks/micro/*.c` with `int main` |
| `MICRO_VARIANTS` | `plain oracle winhint jones jones_full` |
| `MICRO_INPUTS` | `small large` (for `micro-check`) |
| `MICRO_HINTED` | `pgo` (or `oracle_hinted`) |
| `MICRO_MAP_DIR` | empty (required by `micro-pgo`; maps written by `sim/fidelity/fidelity.py`) |
| `MICRO_OUT_ROOT` | `$(WINHINT_BUILD)/benchmarks/micro` |

## Generated kernels

TVM/IREE-generated C kernels (PROPOSAL Phase B, optional) are not implemented; the reason and
the integration plan are in [Generated kernels](workloads-generated.md).

## Next

- [Generated kernels](workloads-generated.md): the optional TVM/IREE kernels.
- [Compiler](compiler.md): how the WinHint pass analyzes these kernels and places hints.
