# Generated kernels (TVM/IREE): not implemented

The evaluation's workloads are hand-written C inference kernels
([Workloads](workloads.md)), built in several shapes (the build matrix:
`-O2`/`-O3`, unrolling on/off, tiled GEMM). The proposal (Phase B) also lists an optional
extra shape: kernels emitted by an ML compiler such as TVM or IREE instead of written by hand. This page records why that item is not implemented, and how it
would be added. No result of the evaluation depends on it
([Evaluation methodology](../concepts/methodology.md)).

**Terms.** *TVM* (Apache TVM) and *IREE* are ML compilers that lower a model's operators to
code; TVM's *C codegen* (`target="c"`) emits plain C. *conda-forge* is the package channel
every project tool comes from.

## Why it is not done

Phase B lists "Optionally add TVM- or IREE-generated C" for the build matrix. **This item is
not implemented.** It cannot be done under the project's toolchain rule ([interfaces.md §1](../interfaces.md#1-environment),
`build/ENV.md`): every tool comes from conda packages. No pip wheels outside conda, no apt, no
toolchain built from source.

Availability check (2026-10-01, `micromamba search -c conda-forge`, linux-64 and noarch):

| Wanted | conda-forge | Result |
|---|---|---|
| Apache TVM compiler (`tvm`, `apache-tvm`, `tlcpack*`) | no package | only `apache-tvm-ffi` 0.1.x, which is the FFI/ABI runtime library. It has no TIR/Relax compiler and no `target="c"` codegen |
| IREE (`iree*`, `iree-base-compiler`, `iree-compiler`) | no package | none |

The upstream distributions are PyPI wheels (`apache-tvm`, `mlc-ai` nightlies,
`iree-base-compiler`) or source builds of the compiler. The rule excludes both: TVM and IREE are
toolchains, not research artifacts. Building TVM with its LLVM backend would also need several
CPU-hours at `-j2` under the 6 GB limit.

## If the rule is relaxed later

The integration point already exists, so the work would be small:

1. Generator script `benchmarks/generated/gen_tvm.py`. In a separate env (e.g. `winhint-tvm`),
   build 2–3 BERT-base operators with TVM's C codegen (`tvm.build(..., target="c")`): attention
   block, FFN (768→3072→768, GELU) and LayerNorm. Commit the emitted `.c` sources, so the
   benchmark build never needs TVM.
2. One small C driver per operator. It uses fixed-seed inputs and prints one checksum line and
   takes `argv[1] = small|large`, like the other kernels ([interfaces.md §5](../interfaces.md#5-benchmarks-and-compiler-flags)).
3. `benchmarks/generated/generated.mk`. It adds the drivers to `KERNELS` with their own source
   directory, and `benchmarks/Makefile` takes one line `-include generated/generated.mk`. The
   plain and winhint variants then build through the same clang/RISC-V/x86 toolchain and the
   WinHint plugin. Check: `make -C benchmarks check VARIANT=winhint KERNELS="tvm_..."` (qemu)
   and `ARCH=x86`.

None of these files exists now, and `benchmarks/Makefile` is unchanged.

## Next

- [Workloads](workloads.md): the kernels the evaluation does use.
- [Toolchain and versions](../reference/toolchain.md): the conda-only toolchain rule.
