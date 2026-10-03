# C/C++ API

The C/C++ code of WinHint is in four components: the LLVM plugins that place hints
([Compiler](../../guide/compiler.md)), the gem5 window controller that acts on them
([gem5 model](../../guide/gem5/index.md)), the real-hardware runtime
([Real hardware](../../guide/hardware/index.md)) and the workloads
([Workloads](../../guide/workloads.md)). Each is a separate Doxygen project, generated on
every build.

Four Doxygen projects, one per component. Each has a file list, a class list and indexes of
functions, variables and macros (see the links at the top of each project page).

| Project | Sources | Start at |
|---------|---------|----------|
| Compiler (LLVM 23 plugins) | `compiler/winhint/`, `common/`, `baselines/` | [Files](compiler/files.md) · [Classes](compiler/annotated.md) |
| gem5 window controller | `sim/gem5/src/cpu/o3/window/` | [Classes](gem5/annotated.md) · [Files](gem5/files.md) |
| Real-hardware runtime | `hw/` | [Files](hw/files.md) · [Structs](hw/annotated.md) |
| Workloads | `benchmarks/` | [Files](benchmarks/files.md) |

The gem5 project holds the controller, the policies and LTP (B9); the hardware project holds
libwinhint, the tools and the R3/R4 baselines; the workloads project holds the 15 kernels and
the microbenchmarks.

The gem5 sources are compiled inside the gem5 tree (they include gem5 headers), so Doxygen
sees them without gem5's own declarations: references to gem5 types are plain text.

Unit tests (`*/tests/`) are not part of these projects.

## Next

- [Commit gate and coverage](../../contributing/quality.md): how the code is tested before
  it is committed.
- [Writing documentation](../../contributing/documentation.md): how to write the comments these
  pages are generated from.
