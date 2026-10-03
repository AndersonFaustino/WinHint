#!/usr/bin/env python3
"""Coverage gate: run every host test suite on instrumented builds, fail at <= 90 %.

Part of ``make validate`` (the commit gate). Three languages are measured and each must
be **strictly above** :data:`THRESHOLD` percent of lines:

* **Python** — pytest on every ``tests/`` directory under ``coverage.py``
  (configuration: ``.coveragerc``).
* **C++** — the LLVM plugins (``compiler/``), rebuilt with ``--coverage`` and driven by
  ``compiler/test/run_tests.sh``; the gem5 window policies and the LTP core, driven by
  their host unit tests (gem5 stubs, no gem5 build).
* **C** — the 15 inference kernels and the micro-benchmarks, built natively with
  ``--coverage`` (untiled and ``-DTILED``) and run on their ``small`` input.

What is *not* measured is listed, with the reason, in :data:`CXX_EXCLUDED` and in
``.coveragerc``: code that only runs inside a full gem5 build, and code that needs root
or specific hardware (Intel hybrid CPU, RAPL, perf counters). Tests are not measured.

Every source file in scope must show up in the report (:data:`C_SCOPE`,
:data:`CXX_SCOPE`); a file that no test compiles or imports would otherwise be invisible
and inflate the percentage.

Usage::

    tooling/coverage_gate.py                 # all three languages (what make validate runs)
    tooling/coverage_gate.py python          # one language: python | cxx | c
    tooling/coverage_gate.py --no-gate c     # report only

Outputs go to ``$WINHINT_BUILD/coverage/`` (default ``build/coverage/``): instrumented
builds, ``python.json``, ``cxx.json``, ``c.json`` (per-file line counts) and
``python-html/``. Run it inside the ``winhint`` env (``tooling/winhint.sh coverage``
does that).

Exit status: 0 when every selected language is above the threshold and complete,
1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

#: Repository root (this file lives in ``tooling/``).
REPO = Path(__file__).resolve().parents[1]

#: Minimum line coverage, in percent; the measured value must be strictly greater.
THRESHOLD = 90.0

#: pytest directories, one pytest process each (same list as ``winhint.sh test python``).
PYTEST_DIRS = (
    "sim/baselines/lut/tests", "sim/baselines/oracle/tests", "sim/baselines/tune/tests",
    "sim/fidelity/tests", "sim/tests", "analysis/tests", "hw/tests", "tooling/tests",
    "compiler/baselines/pgo/tests",
)

#: gem5 window sources measured through the host unit tests.
GEM5_WINDOW = "sim/gem5/src/cpu/o3/window"

#: C++ sources in scope (regular expressions on repository paths).
CXX_SCOPE = (
    r"compiler/(winhint|common|baselines/jones_iq)/[^/]+\.(cpp|h)$",
    GEM5_WINDOW + r"/[^/]+\.(cc|hh)$",
    GEM5_WINDOW + r"/ltp/[^/]+\.(cc|hh)$",
)

#: C++ files excluded from the gate, with the reason.
CXX_EXCLUDED = {
    # gem5-internal glue: compiles only inside the gem5 tree against the O3 CPU
    # (cpu/o3/cpu.hh, dyn_inst.hh, iew.hh, rob.hh, gem5 stats). Tested by
    # `tooling/winhint.sh test sim` on the RISCV_winhint build, not on the host.
    GEM5_WINDOW + "/controller.cc": "gem5-internal (O3 CPU glue)",
    GEM5_WINDOW + "/controller.hh": "gem5-internal (O3 CPU glue)",
    GEM5_WINDOW + "/ltp/ltp.cc": "gem5-internal (O3 CPU glue)",
    GEM5_WINDOW + "/ltp/ltp.hh": "gem5-internal (O3 CPU glue)",
    GEM5_WINDOW + "/ltp/ltp_iew.cc": "gem5-internal (IEW stage hooks)",
    GEM5_WINDOW + "/ltp/ltp_hooks.hh": "gem5-internal (IEW stage hooks)",
    # Includes ltp.hh (gem5 stats, RegId, DynInstPtr) and calls LtpParams::fromArgs(),
    # defined in ltp.cc.
    GEM5_WINDOW + "/ltp/ltp_policy.cc": "gem5-internal (needs ltp.hh / LtpParams from ltp.cc)",
}

#: C sources in scope (regular expressions on repository paths).
C_SCOPE = (r"benchmarks/[^/]+_infer\.c$", r"benchmarks/micro/[^/]+\.(c|h)$")

#: Lines a gcov report treats as code; header files that hold only declarations
#: produce no executable lines and are not required to appear.
_HEADER = re.compile(r"\.(h|hh)$")


@dataclass
class FileCov:
    """Line coverage of one source file.

    Attributes:
        lines: Executable line number → hit count (merged over every build: a line
            counts as covered when any build executed it).
    """

    lines: dict[int, int] = field(default_factory=dict)

    def merge(self, line: int, count: int) -> None:
        """Record one executable line.

        Args:
            line: Line number.
            count: Times it was executed in one build.
        """
        self.lines[line] = max(self.lines.get(line, 0), count)

    @property
    def total(self) -> int:
        """Number of executable lines."""
        return len(self.lines)

    @property
    def covered(self) -> int:
        """Number of executed lines."""
        return sum(1 for c in self.lines.values() if c > 0)


def percent(covered: int, total: int) -> float:
    """Return covered/total in percent (100 for an empty total).

    Args:
        covered: Executed lines.
        total: Executable lines.

    Returns:
        The percentage.
    """
    return 100.0 * covered / total if total else 100.0


def run(cmd: list[str], *, cwd: Path = REPO, env: dict[str, str] | None = None,
        quiet: bool = True) -> None:
    """Run a command, raising with its output when it fails.

    Args:
        cmd: Command and arguments.
        cwd: Working directory.
        env: Extra environment variables.
        quiet: Capture the output and print it only on failure.

    Raises:
        SystemExit: If the command fails (the suite under coverage failed).
    """
    full_env = {**os.environ, **(env or {})}
    res = subprocess.run(cmd, cwd=cwd, env=full_env, text=True,
                         capture_output=quiet, check=False)
    if res.returncode != 0:
        if quiet:
            sys.stdout.write(res.stdout[-6000:])
            sys.stderr.write(res.stderr[-6000:])
        raise SystemExit(f"[coverage] command failed ({res.returncode}): {' '.join(cmd)}")


# ── Python ───────────────────────────────────────────────────────────────────

def python_coverage(out: Path) -> dict[str, FileCov]:
    """Run every pytest directory under coverage.py and collect per-file lines.

    Args:
        out: Output directory (``build/coverage``).

    Returns:
        Repository path → line coverage, for every file ``.coveragerc`` puts in scope
        (``source`` includes files no test imports, which then count as 0 %).
    """
    data = out / ".coverage.python"
    data.unlink(missing_ok=True)
    env = {"COVERAGE_FILE": str(data)}
    for d in PYTEST_DIRS:
        if not (REPO / d).is_dir():
            raise SystemExit(f"[coverage] missing test directory {d}")
        print(f"  pytest {d}", flush=True)
        run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             f"--cov-config={REPO / '.coveragerc'}", "--cov", "--cov-append",
             "--cov-report=", d], env=env)
    report = out / "python.coverage.json"
    run([sys.executable, "-m", "coverage", "json", f"--rcfile={REPO / '.coveragerc'}",
         "-o", str(report)], env=env)
    run([sys.executable, "-m", "coverage", "html", f"--rcfile={REPO / '.coveragerc'}",
         "-d", str(out / "python-html")], env=env)
    files: dict[str, FileCov] = {}
    for name, info in json.loads(report.read_text())["files"].items():
        fc = FileCov()
        for ln in info["executed_lines"]:
            fc.merge(ln, 1)
        for ln in info["missing_lines"]:
            fc.merge(ln, 0)
        files[Path(name).resolve().relative_to(REPO).as_posix()] = fc
    return files


# ── gcov (C and C++) ─────────────────────────────────────────────────────────

def gcov_tool(compiler: str) -> str:
    """Return the gcov command matching a compiler.

    Args:
        compiler: Compiler path or name (``$CXX``, ``clang``…).

    Returns:
        ``llvm-cov gcov`` for clang, otherwise the gcov next to the GCC driver
        (conda's ``x86_64-conda-linux-gnu-gcov``), falling back to ``gcov``.
    """
    if "clang" in Path(compiler).name:
        return "llvm-cov gcov"
    stem = re.sub(r"(c\+\+|g\+\+|gcc|cc)$", "gcov", compiler)
    return stem if stem != compiler and shutil.which(stem) else "gcov"


def gcovr_files(search: Path, root: Path, gcov: str, out_json: Path) -> dict[str, FileCov]:
    """Run gcovr over one build tree and return per-file line coverage.

    Args:
        search: Directory holding the ``.gcno``/``.gcda`` files.
        root: Directory the recorded (possibly relative) source paths resolve against;
            gcovr runs there.
        gcov: gcov command (:func:`gcov_tool`).
        out_json: Where to write gcovr's JSON report.

    Returns:
        Repository path → line coverage, for files under the repository.
    """
    run(["gcovr", "--gcov-executable", gcov, "--gcov-ignore-errors=no_working_dir_found",
         "-r", str(root), "--json", str(out_json), str(search)], cwd=root)
    report = json.loads(out_json.read_text())["files"]
    check_units(search, {Path(f["file"]).name for f in report})
    files: dict[str, FileCov] = {}
    for f in report:
        path = (root / f["file"]).resolve()
        try:
            rel = path.relative_to(REPO).as_posix()
        except ValueError:
            continue                           # system headers
        fc = files.setdefault(rel, FileCov())
        for ln in f["lines"]:
            if not ln.get("gcovr/excluded", False):
                fc.merge(ln["line_number"], ln["count"])
    return files


def unit_candidates(gcda: Path) -> set[str]:
    """Return the source file names a ``.gcda`` file can belong to.

    Object names differ by build system: CMake writes ``<src>.cpp.gcda``, while a combined
    compile-and-link writes ``<target>-<src stem>.gcda``.

    Args:
        gcda: A ``.gcda`` file.

    Returns:
        File names (with or without extension) one of which must be in the report.
    """
    stem = gcda.name[: -len(".gcda")]
    cands = {stem, stem.split("-", 1)[-1]}
    return cands | {Path(c).stem for c in cands}


def check_units(search: Path, reported: set[str]) -> None:
    """Fail when a measured translation unit's own source is missing from the report.

    gcovr has to run with ``--gcov-ignore-errors=no_working_dir_found`` (system headers
    such as ``<new>`` cannot be resolved), which also silently drops a unit compiled from
    a path it cannot resolve, together with the header lines that unit executed.

    Args:
        search: Build directory with the ``.gcda`` files.
        reported: File names present in gcovr's report.

    Raises:
        SystemExit: If a unit with run-time data does not appear in the report.
    """
    names = reported | {Path(n).stem for n in reported}
    lost = sorted(str(g.relative_to(search)) for g in search.rglob("*.gcda")
                  if not unit_candidates(g) & names)
    if lost:
        raise SystemExit("[coverage] gcovr could not resolve the sources of: " + ", ".join(lost)
                         + " (compile them with an absolute or root-relative path)")


def merge(into: dict[str, FileCov], new: dict[str, FileCov]) -> None:
    """Merge per-file coverage from another build (union of executed lines).

    Args:
        into: Accumulated coverage, updated in place.
        new: Coverage of one more build.
    """
    for path, fc in new.items():
        acc = into.setdefault(path, FileCov())
        for ln, c in fc.lines.items():
            acc.merge(ln, c)


def cxx_coverage(out: Path) -> dict[str, FileCov]:
    """Build the C++ code with ``--coverage``, run its host tests, collect coverage.

    Args:
        out: Output directory (``build/coverage``).

    Returns:
        Repository path → line coverage of every C++ file the tests compiled.
    """
    cxx = os.environ.get("CXX") or "g++"
    gcov = gcov_tool(cxx)
    cov = {"COVERAGE_FLAGS": "--coverage -O0"}

    comp = out / "compiler"
    print("  compiler plugins (instrumented build + compiler/test/run_tests.sh)", flush=True)
    run(["compiler/build.sh"], env={"WINHINT_COVERAGE": "1", "COMPILER_BUILD": str(comp),
                                    "JOBS": "1"})
    for g in comp.rglob("*.gcda"):
        g.unlink()
    run(["compiler/test/run_tests.sh"], env={"COMPILER_BUILD": str(comp),
                                             "TEST_OUT": str(out / "compiler-tests")})
    files = gcovr_files(comp, REPO, gcov, out / "compiler.gcovr.json")

    g5 = out / "gem5"
    shutil.rmtree(g5, ignore_errors=True)
    print("  gem5 window policies + LTP core (host unit tests)", flush=True)
    make = ["make", "-s", "-j1"]
    run(make + ["-C", f"{GEM5_WINDOW}/tests", f"WINHINT_BUILD={g5}", f"CXX={cxx}"], env=cov)
    run(make + ["-C", f"{GEM5_WINDOW}/ltp/tests", f"WINHINT_BUILD={g5}", f"CXX={cxx}"], env=cov)
    run(make + ["-C", "sim/tests", "unit", f"OUT={g5}/simtests", f"HOSTCXX={cxx}"], env=cov)
    merge(files, gcovr_files(g5, REPO, gcov, out / "gem5.gcovr.json"))
    return files


def c_coverage(out: Path) -> dict[str, FileCov]:
    """Build the kernels natively with ``--coverage``, run them, collect coverage.

    Both the untiled and the ``-DTILED`` builds run, so both GEMM paths are measured.

    Args:
        out: Output directory (``build/coverage``).

    Returns:
        Repository path → line coverage of every benchmark source.
    """
    bench = out / "bench"
    shutil.rmtree(bench, ignore_errors=True)
    files: dict[str, FileCov] = {}
    builds = (("kernels", "", REPO / "benchmarks"),
              ("kernels-tiled", "on", REPO / "benchmarks"),
              ("micro", "", REPO / "benchmarks" / "micro"))
    for name, tiled, src in builds:
        root = bench / name
        print(f"  benchmarks: {name}", flush=True)
        args = ["make", "-s", "-j1", "-C", "benchmarks", "ARCH=x86", "VARIANT=plain", "OPT=O0",
                f"OUT_ROOT={root}", "CFLAGS_EXTRA=--coverage"]
        if tiled:
            args.append("TILED=on")
        if src != REPO / "benchmarks":
            args.append(f"BENCH_DIR={src}")
        run(args)
        bins = sorted(p for p in root.rglob("*") if p.is_file() and os.access(p, os.X_OK)
                      and p.suffix == "")
        if not bins:
            raise SystemExit(f"[coverage] no binaries built in {root}")
        for b in bins:
            run([str(b), "small"], cwd=root)
        merge(files, gcovr_files(root, REPO / "benchmarks", gcov_tool("clang"),
                                 out / f"{name}.gcovr.json"))
    return files


# ── gate ─────────────────────────────────────────────────────────────────────

def in_scope(path: str, patterns: Iterable[str]) -> bool:
    """Tell whether a repository path matches one of the scope patterns.

    Args:
        path: Repository-relative POSIX path.
        patterns: Regular expressions (matched from the start of the path).

    Returns:
        True when one pattern matches.
    """
    return any(re.match(p, path) for p in patterns)


def expected_files(patterns: Iterable[str], excluded: Iterable[str] = ()) -> set[str]:
    """List the tracked source files a language's report must contain.

    Args:
        patterns: Scope patterns (:data:`CXX_SCOPE`, :data:`C_SCOPE`).
        excluded: Paths left out on purpose.

    Returns:
        Repository paths of every non-header file in scope (headers can legitimately
        have no executable line).
    """
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout.split()
    pats = list(patterns)
    skip = set(excluded)
    return {p for p in out if in_scope(p, pats) and p not in skip and not _HEADER.search(p)}


def evaluate(lang: str, files: dict[str, FileCov], expected: set[str],
             out: Path) -> bool:
    """Print the per-file table for one language and apply the gate.

    Args:
        lang: Language label.
        files: Per-file coverage restricted to the language's scope.
        expected: Files that must appear in ``files``.
        out: Output directory; ``<lang>.json`` is written there.

    Returns:
        True when the total is above :data:`THRESHOLD` and no expected file is missing.
    """
    total = sum(f.total for f in files.values())
    covered = sum(f.covered for f in files.values())
    pct = percent(covered, total)
    print(f"\n{lang}: {covered}/{total} lines = {pct:.2f} % (gate: > {THRESHOLD:g} %)")
    for path in sorted(files):
        f = files[path]
        print(f"  {percent(f.covered, f.total):6.1f} %  {f.covered:5d}/{f.total:<5d} {path}")
    missing = sorted(expected - set(files))
    for m in missing:
        print(f"  MISSING  {m}: in scope but no test compiles or runs it")
    (out / f"{lang.lower().replace('+', 'x')}.json").write_text(json.dumps({
        "percent": pct, "covered": covered, "total": total, "missing": missing,
        "files": {p: {"covered": f.covered, "total": f.total} for p, f in files.items()},
    }, indent=1))
    ok = pct > THRESHOLD and not missing
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    """Run the selected coverage suites and apply the gate.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        Process exit status: 0 when every selected language passes, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("languages", nargs="*", metavar="{python,cxx,c}",
                    help="languages to measure (default: all three)")
    ap.add_argument("--no-gate", action="store_true", help="report only; always exit 0")
    args = ap.parse_args(argv)
    langs = args.languages or ["python", "cxx", "c"]
    unknown = sorted(set(langs) - {"python", "cxx", "c"})
    if unknown:
        ap.error(f"unknown language(s): {', '.join(unknown)} (choose from python, cxx, c)")

    out = Path(os.environ.get("WINHINT_BUILD", REPO / "build")) / "coverage"
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    if "python" in langs:
        print("== Python", flush=True)
        ok &= evaluate("Python", python_coverage(out), set(), out)
    if "cxx" in langs:
        print("== C++", flush=True)
        files = {p: f for p, f in cxx_coverage(out).items()
                 if in_scope(p, CXX_SCOPE) and p not in CXX_EXCLUDED}
        ok &= evaluate("C++", files, expected_files(CXX_SCOPE, CXX_EXCLUDED), out)
    if "c" in langs:
        print("== C", flush=True)
        files = {p: f for p, f in c_coverage(out).items() if in_scope(p, C_SCOPE)}
        ok &= evaluate("C", files, expected_files(C_SCOPE), out)
    return 0 if ok or args.no_gate else 1


if __name__ == "__main__":
    sys.exit(main())
