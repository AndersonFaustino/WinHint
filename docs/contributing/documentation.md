# Writing documentation

This page is for anyone adding or changing documentation, including the comments in the code.
The documentation is part of the commit gate ([Commit gate and coverage](quality.md)), so it
follows rules that a strict build can check. The page covers how the site is built, how it
is assembled from hand-written pages and generated API pages, and the conventions for Python docstrings and C/C++ comments.

The documentation has three layers, all in this repository and all checked by
`tooling/winhint.sh test docs`:

1. **Guides and reference pages** — Markdown under `docs/`. All prose documentation lives
   there, with lowercase file names; component directories hold code, not documents. The
   only Markdown file outside `docs/` is the repository's `README.md`.
2. **Python API** — Google-style docstrings in the code.
3. **C/C++ API** — Doxygen comments in the code.

## Build and preview

```bash
# live preview at http://127.0.0.1:8000 (shortcut: make docs-serve)
tooling/winhint.sh docs:serve
# strict build → build/site/index.html (shortcut: make docs)
tooling/winhint.sh docs:build
# docstring coverage + syntax (shortcut: make docs-check)
tooling/winhint.sh docs:check
# docs:check, then docs:build
tooling/winhint.sh test docs
```

The site is built in the `winhint` env ([Toolchain](../reference/toolchain.md)), created
by `tooling/create_conda_env.sh` (the tools are in `requirements-docs.txt`). The build is **strict**: a broken link or anchor, a malformed
docstring, a page missing from the navigation or a missing snippet fails it.

## How the site is assembled

| Piece | File |
|-------|------|
| Site configuration and navigation | [`mkdocs.yml`](../../mkdocs.yml) |
| Hand-written pages | `docs/**/*.md` |
| Python API page generator (one page per module) | [`docs/_tools/gen_pages.py`](../_tools/gen_pages.py) |
| Shared tables and the link rewriter | [`docs/_tools/sitemap.py`](../_tools/sitemap.py) |
| Link rewriting, mkdoxy workarounds | [`docs/_tools/hooks.py`](../_tools/hooks.py) |
| C/C++ projects (sources, Doxygen settings) | `mkdoxy` section of `mkdocs.yml` |

**Links.** Write links relative to the file you are editing, exactly as they work on GitHub.
At build time a hook rewrites them: a link to another `docs/` page stays a site link; a link to any other file or directory of the
repository points to its GitHub source view. A link to a path that does not exist is left
alone, so the strict build reports it.

**Adding a page.** Put it under `docs/` (lowercase, words separated by `-`) and add it to
`nav` in `mkdocs.yml`. Code comments refer to it by its repository path, for example
`docs/guide/hardware/index.md`.

**Page pattern.** Every page follows the same shape
so that the site reads top to bottom:

1. An H1 title, then a lead of two to four sentences: what the page covers, why it matters,
   and where it sits in the system (link the concept page it builds on).
2. Optionally, a "Before you read" line naming the page to read first.
3. The body, in the order a reader needs it. Keep tables narrow; put command comments on
   their own line; never leave placeholders such as `...` in a command.
4. A closing "Next" section with one to three links to the logical following pages.

Define a term at first use or link it to its [Glossary](../reference/glossary.md) entry.
Do not copy long content between pages: link to the page that owns it (the runbook is
[Running experiments](../guide/usage.md); file formats are in
[Interfaces](../interfaces.md)).

**Snippets.** `--8<-- "path/from/repo/root"` includes a repository file verbatim (used for
`tooling/versions.lock`); a missing file fails the build.

## Python docstrings

Every module, class, function and method has a docstring, including private helpers and
tests (nested functions are exempt). The style is
[Google](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings), as
parsed by griffe:

```python
"""One-line summary of the module.

What the module does, what it reads and writes, and how to run it:

    python3 path/to/script.py --input small --out results/example/
"""


def fractions(counts: dict[str, int], keys: list[str]) -> list[float]:
    """Return the share of the total count that belongs to each key.

    Args:
        counts: Count per key.
        keys: Keys to report, in output order.

    Returns:
        One fraction per key, in the order of ``keys``.

    Raises:
        ZeroDivisionError: If every count is zero.
    """


def load(path):
    """Load a JSON file.

    Args:
        path (str | pathlib.Path): File to read.

    Returns:
        data (dict): The parsed JSON object.
    """
```

- Summary line: imperative mood, one line, ends with a period.
- Sections in this order: `Args:`, `Returns:` / `Yields:`, `Raises:`, `Attributes:`
  (classes and dataclasses), `Example:`.
- Annotated parameters: `name: description`. **Unannotated** parameters and return values
  need a type: `name (type): description`.
- Docstrings render as Markdown. Put command lines, usage synopses and tables in an indented
  or fenced code block: text like `[--input small] [--jobs N]` outside a code block is read
  as a Markdown link and fails the strict build.
- Document important module constants with a `#:` comment on the line above.
- Many module docstrings double as `argparse` descriptions (`--help`); use a raw string
  (`r"""`) when they contain backslashes.

`tooling/check_docstrings.py` enforces coverage and parses every docstring with griffe's
Google parser; any griffe warning is an error.

## C/C++ comments

Doxygen, Javadoc style. An excerpt from
[`hw/libwinhint/whutil.h`](../../hw/libwinhint/whutil.h):

```c
/**
 * @file whutil.h
 * @brief Shared helpers for the WinHint real-hardware runtime (hw/).
 *
 * Hybrid topology detection (P/E cpusets), hybrid-aware perf_event_open
 * counters (one event per core-type PMU: cpu_core / cpu_atom), RAPL energy
 * (powercap sysfs, with a perf "power" PMU fallback) and timing.
 */

/** @brief Description of one perf event to open with wh_perf_open(). */
typedef struct {
    uint64_t type;         /**< PERF_TYPE_HARDWARE / HW_CACHE / RAW */
    uint64_t config;       /**< without the hybrid PMU-type bits */
    int pmu_only;          /**< -1: all PMUs; 0: P PMU only; 1: E PMU only */
    const char *name;      ///< Label used in error messages (may be NULL).
} wh_evspec;

/**
 * @brief Read all open events.
 * @param[in]  p   Events from wh_perf_open().
 * @param[out] out Zeroed, then filled; unopened or failed reads stay 0.
 * @return Always 0.
 */
int  wh_perf_read(const wh_perf *p, wh_perf_vals *out);
```

- Every file starts with an `@file` / `@brief` block (after any license header).
- Every function, type, enum, macro and global that matters has a block. Fields and
  enumerators use trailing `///<`.
- Full documentation goes on the declaration in the header; the definition gets at most a
  one-line pointer (`// See whutil.h.`).
- gem5 code follows gem5's conventions and keeps its license headers.
- Preformatted text (tables, examples) goes in `@code` / `@endcode` or `@verbatim` /
  `@endverbatim`.

## Changing documentation without changing code

Documentation-only changes must not change behavior. Two quick proofs:

```bash
# Python: AST identical once docstrings are removed
python3 - sim/run_lengths.py <<'EOF'
import ast, subprocess, sys
def strip(src):
    t = ast.parse(src)
    for n in ast.walk(t):
        body = getattr(n, "body", None)
        if isinstance(body, list):
            n.body = [s for s in body if not (isinstance(s, ast.Expr)
                      and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))] or [ast.Pass()]
    return ast.dump(t)
for f in sys.argv[1:]:
    old = subprocess.run(["git", "show", "HEAD:" + f], capture_output=True, text=True).stdout
    print("OK  " if strip(old) == strip(open(f).read()) else "DIFF", f)
EOF

# C/C++: token stream identical once comments are removed
f=hw/libwinhint/whutil.c
diff <(git show HEAD:$f | gcc -fpreprocessed -dD -E -P -x c - | tr -s ' \n\t' '\n') \
     <(gcc -fpreprocessed -dD -E -P -x c $f | tr -s ' \n\t' '\n') && echo OK
```

## Next

- [Implementation checklist](../checklist.md): what is implemented, per proposal phase.
