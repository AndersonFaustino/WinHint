# B8 — Clairvoyance (Tran et al., CGO 2017)

Clairvoyance ("look-ahead compile-time scheduling") is a compile-time transformation that
raises memory-level parallelism with no hardware change: for marked loops it creates an
*access phase*, a copy of the loop's loads (and prefetches) that runs ahead of the original
computation. It attacks the same problem as WinHint from the other side, by reshaping the code
instead of resizing the window, and the two are complementary. B8 is the only gem5-side
baseline with a public artifact, so it is **reused, not
reimplemented** ([Evaluation methodology](../../concepts/methodology.md),
[Baselines](index.md)). It runs alone (`clairvoyance`, static largest
window) and combined with WinHint (`winhint_clairvoyance`, `hint` policy).

**Terms.** *Bitcode* is LLVM's binary IR; the artifact's LLVM 3.8 writes it and this
project's LLVM 23 reads it. *Retargeting* rewrites the IR's target triple and data layout from
x86-64 to RISC-V. *Knobs* are the artifact's tunable options. Other terms are in the
[Glossary](../../reference/glossary.md).

## Artifact

Artifact: `third_party/clairvoyance` (git submodule of
<https://github.com/ktran/clairvoyance>, commit `becde64`, LLVM 3.8 passes:
`MarkLoopsToSwoopify`, `CFGIndirectionCount`, `UtilLoops` (single-loop unroll),
`LoopExtract`, `BranchAnnotate`, `OptimisticSwoop`/`SwoopDAE`).

The artifact's own `compiler/Makefile` clones LLVM/Clang `release_38` from
`llvm.org/git`, which no longer exists; LLVM/Clang 3.8.1 instead comes from conda-forge
(`llvmdev`/`clangdev` in env `winhint-llvm38`), and `build.sh` builds only the passes.

### Files

| File | Purpose |
|------|---------|
| `build.sh` | Builds the Clairvoyance passes against the conda LLVM/Clang 3.8.1 of env `winhint-llvm38` into `build/clairvoyance/lib/` (host only; `JOBS=1` by default). |
| `cv_compile.sh` | Per-kernel pipeline used by `benchmarks/Makefile` for `VARIANT=clairvoyance` and `winhint_clairvoyance`. |
| `knobs.json` | The artifact's knobs, defaults and tuning grid (read by the tuning harness). |
| `patches/` | Fixes to the artifact applied by `build.sh` to a copy of its sources (submodule untouched). |

## Pipeline (`cv_compile.sh <kernel.c> <out>`)

1. **Marking.** The artifact transforms only loops carrying the marker
   `#pragma clang loop vectorize_width(1337)` (`loopToBeDAE` in
   `DAE/Utils/SkelUtils/Utils.cpp`; `toBeDAE` accepts every function in the artifact
   build). A temporary copy of the kernel gets the pragma before every line-initial
   `for (`; kernel sources are not modified.
2. **Front end:** clang 3.8 `-O3 -S -emit-llvm`, `x86_64-linux-gnu` (clang 3.8 has no
   RISC-V target).
3. **Passes:** exactly the artifact recipe (`experiments/swoop/sources/common/SWOOP/
   Makefile.defaults`): `-mark-loops -require-delinquent=false`,
   `-annotate-cfg-indir`, the unroll prelude + `-single-loop-unroll -unroll N`,
   `-second-loop-extract ... -branchannotate`, then the SWOOP/Clairvoyance pass
   (`CV_TYPE=consv` → `-dae-swoop`, `specsafe` → `-aggressive-swoop`, `spec` →
   `-speculative-swoop`, `multispecsafe`/`multispec` → the same plus `-multi-access`)
   with `-merge-branches -branch-prob-threshold P -indir-thresh K -unroll N -mem2reg`,
   then `opt -O3` (3.8), as the artifact does.

4. **Bitcode hand-off:** opt 3.8 writes bitcode; LLVM 23 reads it (LLVM keeps bitcode
   backward compatibility to 3.0), `opt -passes=verify` checks it.
5. **Retarget (RISC-V):** triple and datalayout are rewritten to riscv64 and the x86
   `target-cpu`/`target-features` attributes dropped.
6. `winhint_clairvoyance`: the WinHint pass runs on the transformed IR (`opt
   -passes=winhint` with the same `-winhint-*` options).
7. **Lowering:** modern clang at the requested `-O` level, codegen only
   (`-Xclang -disable-llvm-optzns`), so the Clairvoyance schedule is not undone by a
   second middle-end run; then link (static for RISC-V).

## Knobs

All are the artifact's own (`experiments/swoop/sources/common/SWOOP/Makefile.targets`
enumerates the grid, `Makefile.defaults` the options). Set them on the `make` command
line (or in the environment of `cv_compile.sh`); both validate them. Non-default values
build into `<variant>[<cfg>]-cv_<type>-u<N>-i<K>[-bp<P>]`, so binaries for different
settings never overwrite each other.

| Knob | Default | Artifact grid | Accepted | Pass option |
|------|---------|---------------|----------|-------------|
| `CV_TYPE` | `consv` | consv specsafe spec multispecsafe multispec | same | see above |
| `CV_UNROLL` | 2 | 1 2 4 | 1..16 | `-unroll N` |
| `CV_INDIR` | 1 | 0 1 2 3 | 0..8 | `-indir-thresh K` |
| `CV_BRANCH_PROB` | 0.9 | 0.9 (fixed) | 0.5..1.0 | `-branch-prob-threshold P` |
| `CV_STRICT` | 0 | – | 0/1 | 1 = fail on a pass crash instead of excluding the function |

PROPOSAL §8 asks for the three tunable knobs (60 grid points) to be tuned on the `small`
input with the same budget as WinHint; `knobs.json` holds the grid for the harness. The
harness, [`sim/baselines/tune/tune_baselines.py`](../../../sim/baselines/tune/tune_baselines.py),
evaluates at most `--budget` points of the grid (default 16, always including the default
point) and keeps, per machine, the point with the best geometric mean over the kernels
([Baseline tuning](../usage.md#step-6-baseline-tuning)).

## Bitcode compatibility check plan (PROPOSAL §8)

1. `llvm-dis` (LLVM 23) of every `*.cv.bc` succeeds and `opt -passes=verify` passes
   (automated in `cv_compile.sh`).
2. Retargeting x86_64 → riscv64 IR is sound only for code whose front-end ABI
   lowering is target-neutral. Check per kernel: no `x86_fp80`, no `byval`/`sret`
   struct coercions in calls to externally defined functions, no `va_arg`, no
   `char` arithmetic depending on signedness (x86 `char` is signed, RISC-V's is
   unsigned), no inline asm:
   `grep -E 'x86_fp80|va_arg|byval|asm ' <kernel>.cv.ll` must be empty.
3. Functional check: `make ARCH=riscv VARIANT=clairvoyance check` (QEMU) and
   `ARCH=x86` natively must print bit-identical results to `plain`.
4. Transformation check: the extracted `__kernel__*` functions and the duplicated
   (access-phase) loads must be present in the IR of each kernel that has marked
   loops; otherwise the variant degenerates to `plain` and is reported as such.

If step 1 or 3 fails, or the passes cannot be built against the conda LLVM 3.8, fall back to the
**port plan** below.

## Port plan (fallback: reimplementation on LLVM 23)

~6.1 kLOC of pass code. Required changes:

* Legacy `RegisterPass`/`FunctionPass`/`LoopPass` → new-PM `PassInfoMixin` passes
  registered from one plugin (`llvmGetPassPluginInfo`), loop passes as function
  passes walking `LoopInfo` (they rewrite the CFG and extract functions).
* Typed pointers → opaque pointers: every `getPointerElementType()`/
  `getElementType()` on pointer types must take the type from the load/store/GEP.
* `CallSite` → `CallBase`; `TerminatorInst` → `Instruction::isTerminator()`;
  `getGlobalContext()` removed; `Function::getArgumentList` → `args()`;
  `BasicBlock::getInstList` mutators → `insertInto`/iterators.
* Alias analysis: legacy `AliasAnalysis` wrapper passes (`-basicaa -globals-aa
  -scev-aa -tbaa`) → `AAManager` via `FAM.getResult<AAManager>`.
* `LoopVectorizeHints` copy used for marking → read `llvm.loop.vectorize.width`
  metadata directly (or a dedicated `winhint.clairvoyance` loop attribute).
* Loop extraction via `CodeExtractor` (API changed: `CodeExtractorAnalysisCache`).
* `LoopInfo`/`ScalarEvolution` APIs are largely source compatible.

The port would be validated against the 3.8 artifact on its own
`experiments/swoop/sources/myBenchmark` (same transformed loop structure) before use.

## Status (2026-10-01)

**The artifact route works; no port is needed.** Verified end to end:

* `tooling/winhint.sh clairvoyance:build` (= `build.sh`, `-j1`) builds the seven pass
  libraries against conda LLVM/Clang 3.8.1 (env `winhint-llvm38`) in a few minutes; every
  library loads into `opt` 3.8. It is a light build, so it was run with
  `WINHINT_NO_LOCK=1` while gem5 held the heavy lock. `patches/0001-missing-returns.patch`
  fixes missing `return`s (UB that GCC 13 turns into crashes); `0002-swoopdae-null-latch.patch`
  fixes the SwoopDAE crash below; the submodule stays pristine.
* **Bitcode interop:** LLVM 23 (`llvm-dis`, `opt -passes=verify`) reads the 3.8 bitcode
  of all 15 kernels. The RISC-V retarget (step 5) also strips the x86 clobbers
  (`~{dirflag},~{fpsr},~{flags}`) from the artifact's inline-asm phase markers and
  refuses IR with `x86_fp80`, `va_arg`, `byval` or `llvm.x86.*` (none occur).
* **Transformation present:** the extracted `__kernel__*` loops are re-inlined by the
  artifact's final `opt -O3`, but the access phases remain (e.g. 76 `llvm.prefetch` in
  `encoder_bert_tiny`, 100 in `decoder_gpt2`). `<binary>.cv.json` records per kernel
  the number of extracted loop functions and any excluded function.
* **SwoopDAE crash fixed (was: `contrast_mobilenet_infer`'s `conv3x3_stem` excluded).**
  Root cause: `SwoopDAE::CreatePhaseWithCFGLoads` and `PhaseStitching.cpp:getExitingBlock`
  assume that a loop latch ending in an unconditional branch has a *single predecessor*
  holding the exit branch. In `conv3x3_stem` the innermost (x) loop body is the fully
  unrolled 3×3×3 window, with the padding guard `if (iy >= 0 && iy < S && ix >= 0 && ix < S)`
  per tap; after `-single-loop-unroll -unroll 2` the latch is the join block of the last
  guarded tap of the second copy (3 predecessors), so
  `getSinglePredecessor()` is null and `->getTerminator()` segfaults. Patch 0002 falls
  back to the loop's unique exiting block (the block with the exit branch;
  `BranchMerge`'s `minimizeFunctionFromBranchPred` makes the same choice, so the loop-exit
  branch is never "reduced") and, for the phase-end marker, to the latch itself.
  `conv3x3_stem` is now transformed (53 loads in its access phase); every `<binary>.cv.json`
  lists `"excluded_functions": []`.
* **Safety net kept:** if the passes still crash on some knob setting, `cv_compile.sh`
  strips the loop markers of the crashing function, re-runs, and lists it under
  `excluded_functions` (`CV_STRICT=1` turns this into an error). With the fix, all
  60 grid points × 15 kernels compile with `CV_STRICT=1` and match `plain` bit for bit
  (native x86-64, `small` input).
* **Correctness:** `make -C benchmarks ARCH={riscv,x86} VARIANT={clairvoyance,winhint_clairvoyance} check`
  is bit-identical to `plain` for all 15 kernels on the `small` input (qemu-riscv64 and
  native x86-64).

## Usage

```sh
# B8 alone (machine-independent)
make -C benchmarks ARCH=riscv VARIANT=clairvoyance

# B8 combined with WinHint
make -C benchmarks ARCH=riscv VARIANT=winhint_clairvoyance
```

OPT/UNROLL apply only to the LLVM 23 code generation (the 3.8 middle end always runs the
artifact's `-O3` recipe); `-D` flags such as `TILED=on` reach the 3.8 front end. Tuning
`CV_TYPE`/`CV_UNROLL`/`CV_INDIR` in gem5 is part of the baseline-tuning step (see
[Knobs](#knobs)); the runbook builds the tuned variants in
[Hinted and per-machine binaries](../usage.md#step-7-hinted-and-per-machine-binaries).

## Next

- [Jones IQ](jones-iq.md): B6, the other compiler baseline.
- [Compiler](../compiler.md): the WinHint pass that `winhint_clairvoyance`
  runs on the transformed IR.
- [Deviations](../../deviations.md#b8): this baseline's entries, next to every other baseline's.
