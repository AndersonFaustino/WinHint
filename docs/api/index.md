# API reference

These pages document the code itself: every Python module and every C/C++ file of the
compiler plugins, the gem5 window controller, the real-hardware runtime and the workloads.
They are generated from the code on every build, so they never drift from it. Use them when
the guides are not detailed enough, for example to read a function's exact arguments; for
how the pieces fit together, start from [How WinHint works](../guide/architecture.md).

| Part | Generated from | Covers |
|------|----------------|--------|
| [Python](python/index.md) | Google-style docstrings | experiment drivers, baseline pipelines, energy model, fidelity, analysis, hardware driver, tooling |
| [C/C++](cpp/index.md) | Doxygen comments | LLVM plugins, gem5 window controller and policies, hardware runtime and tools, workloads |

The Python pages are read statically by [mkdocstrings](https://mkdocstrings.github.io/) and
griffe (nothing is imported) and cover every module except tests. The C/C++ pages are
rendered by [mkdoxy](https://github.com/JakubAndrysek/MkDoxy).

Most Python modules are command-line scripts: their module docstring describes the command
line, and `--help` prints the same text. Private helpers (`_name`) are documented in the code
but hidden from these pages.

Both parts are checked: `tooling/winhint.sh docs:check` fails on any undocumented Python
definition or malformed docstring, and `tooling/winhint.sh docs:build` builds the site in
strict mode. See [Writing documentation](../contributing/documentation.md).

## Next

- [Python API](python/index.md)
- [C/C++ API](cpp/index.md)
