#!/usr/bin/env python3
"""Check that the Python code is fully documented with valid Google-style docstrings.

Two checks over every Python file of the repository (tests included, ``third_party/``
and ``docs/_tools/`` excluded), or over the files given on the command line:

1. **Coverage** — every module, class, function and method (including private helpers;
   nested functions are exempt) has a docstring.
2. **Syntax** — every docstring parses as a Google-style docstring without griffe
   warnings (unknown parameters, missing types for unannotated parameters or return
   values, malformed sections). These are the warnings that would make
   ``mkdocs build --strict`` fail on the API reference.

The coverage check needs only the standard library. The syntax check needs ``griffe``,
which is installed in the ``winhint`` env (requirements-docs.txt); without it the check is skipped with a
note (``--require-griffe`` turns that into a failure).

Usage::

    tooling/check_docstrings.py                      # whole repository
    tooling/check_docstrings.py sim/run_lengths.py   # selected files
    tooling/winhint.sh test docs                     # this check + strict site build

Exit status: 0 when both checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import logging
import subprocess
import sys
from pathlib import Path

#: Repository root (this file lives in ``tooling/``).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Path prefixes that are never checked.
EXCLUDED_PREFIXES = ("third_party/", "docs/_tools/")


def repo_python_files() -> list[str]:
    """Return every tracked or untracked-but-not-ignored Python file to check.

    Returns:
        Sorted repository-relative POSIX paths.
    """
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "--", "*.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return sorted(
        p for p in out.splitlines()
        if not p.startswith(EXCLUDED_PREFIXES) and (REPO_ROOT / p).is_file()
    )


def missing_docstrings(path: str) -> list[str]:
    """List the definitions of one file that have no docstring.

    Args:
        path: Repository-relative path of a Python file.

    Returns:
        One ``"path:line: kind name"`` entry per undocumented module, class, function or
        method. Functions nested inside other functions are not required to have one.
    """
    tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"), filename=path)
    missing = [] if ast.get_docstring(tree) else [f"{path}:1: module"]

    def visit(node: ast.AST, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not in_function and not ast.get_docstring(child):
                    kind = "class" if isinstance(child, ast.ClassDef) else "def"
                    missing.append(f"{path}:{child.lineno}: {kind} {child.name}")
                visit(child, in_function or not isinstance(child, ast.ClassDef))
            else:
                visit(child, in_function)

    visit(tree, False)
    return missing


class _Collector(logging.Handler):
    """Logging handler that keeps every griffe warning it receives.

    Attributes:
        records: The formatted warning messages, in emission order.
    """

    def __init__(self) -> None:
        """Create an empty collector at WARNING level."""
        super().__init__(logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store one log record.

        Args:
            record: The record emitted by griffe.
        """
        self.records.append(record.getMessage())


def griffe_warnings(paths: list[str]) -> list[str] | None:
    """Parse every docstring of the given files with griffe's Google parser.

    Args:
        paths: Repository-relative Python file paths.

    Returns:
        The griffe warnings (``"file:line: message"``), or ``None`` when griffe is not
        installed.
    """
    try:
        import griffe
    except ImportError:
        return None
    collector = _Collector()
    logger = logging.getLogger("griffe")
    logger.addHandler(collector)
    logger.setLevel(logging.WARNING)
    logger.propagate = False

    def walk(obj) -> None:
        if obj.docstring is not None:
            obj.docstring.parse("google")
        for member in obj.members.values():
            if not member.is_alias:
                walk(member)

    for path in paths:
        ident = path[: -len(".py")].replace("/", ".")
        if "." in Path(path).stem:          # e.g. lit.cfg.py: not importable by name
            continue
        try:
            module = griffe.load(ident, search_paths=[str(REPO_ROOT)], docstring_parser="google",
                                 allow_inspection=False)
        except Exception as exc:            # noqa: BLE001 — report and keep going
            collector.records.append(f"{path}: griffe could not load the module: {exc}")
            continue
        walk(module)
    return collector.records


def main(argv: list[str] | None = None) -> int:
    """Run the coverage and syntax checks and print a report.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        Process exit status: 0 when every check passed, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", help="files to check (default: the whole repository)")
    ap.add_argument("--require-griffe", action="store_true",
                    help="fail instead of skipping the syntax check when griffe is missing")
    args = ap.parse_args(argv)

    files = [str(Path(f).resolve().relative_to(REPO_ROOT)) for f in args.files] or repo_python_files()
    ok = True

    missing = [m for f in files for m in missing_docstrings(f)]
    print(f"coverage: {len(files)} files, {len(missing)} undocumented definitions")
    for m in missing:
        print(f"  {m}")
    ok &= not missing

    warnings = griffe_warnings(files)
    if warnings is None:
        print("syntax:   skipped (griffe not installed; it is in requirements-docs.txt)")
        ok &= not args.require_griffe
    else:
        print(f"syntax:   {len(warnings)} griffe warnings")
        for w in warnings:
            print(f"  {w}")
        ok &= not warnings
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
