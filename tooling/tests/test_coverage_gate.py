"""Tests of the coverage gate (tooling/coverage_gate.py) with faked builds, pytest runs and gcovr.

No instrumented build or real test suite runs here: ``run`` and the per-language
collectors are monkeypatched, and gcovr/coverage.py reports are synthetic JSON.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOLING = Path(__file__).resolve().parents[1]
# Loaded under another name: ``coverage`` is already the coverage.py package (pytest-cov).
_SPEC = importlib.util.spec_from_file_location("winhint_coverage_gate", TOOLING / "coverage_gate.py")
cov = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cov          # dataclasses resolve annotations through sys.modules
_SPEC.loader.exec_module(cov)


def _fc(lines):
    """Build a FileCov from a ``{line: count}`` dict.

    Args:
        lines (dict[int, int]): Line hit counts.

    Returns:
        (cov.FileCov): The file coverage.
    """
    f = cov.FileCov()
    for ln, c in lines.items():
        f.merge(ln, c)
    return f


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Point ``REPO`` at a temporary directory.

    Args:
        tmp_path (Path): pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): pytest monkeypatch fixture.

    Returns:
        (Path): The fake repository root.
    """
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(cov, "REPO", root)
    return root


@pytest.fixture
def calls(monkeypatch):
    """Replace ``run`` with a recorder.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest monkeypatch fixture.

    Returns:
        (list[tuple]): Receives ``(cmd, cwd, env)`` per call.
    """
    rec = []
    monkeypatch.setattr(cov, "run", lambda cmd, cwd=None, env=None, quiet=True: rec.append((cmd, cwd, env)))
    return rec


def test_module_is_the_gate():
    """Make sure the import resolved to tooling/coverage_gate.py, not the coverage package."""
    assert Path(cov.__file__).resolve() == TOOLING / "coverage_gate.py"


def test_filecov_and_percent():
    """Check FileCov merging (max over builds), totals and percent()."""
    f = _fc({1: 0, 2: 3, 3: 0})
    f.merge(1, 2)
    f.merge(2, 0)
    assert f.lines == {1: 2, 2: 3, 3: 0}
    assert (f.total, f.covered) == (3, 2)
    assert cov.percent(1, 4) == 25.0
    assert cov.percent(0, 0) == 100.0


def test_merge():
    """Check that merge() unions files and lines, keeping the larger count."""
    into = {"a.c": _fc({1: 0, 2: 1})}
    cov.merge(into, {"a.c": _fc({1: 4, 5: 0}), "b.c": _fc({7: 1})})
    assert into["a.c"].lines == {1: 4, 2: 1, 5: 0}
    assert into["b.c"].lines == {7: 1}


def test_gcov_tool(monkeypatch):
    """Check gcov selection for clang, a conda GCC driver, and a plain fallback."""
    assert cov.gcov_tool("/opt/bin/clang++") == "llvm-cov gcov"
    monkeypatch.setattr(cov.shutil, "which", lambda p: "/x/" + p if p.startswith("x86_64") else None)
    assert cov.gcov_tool("x86_64-conda-linux-gnu-c++") == "x86_64-conda-linux-gnu-gcov"
    assert cov.gcov_tool("g++") == "gcov"           # derived name not on PATH
    assert cov.gcov_tool("icx") == "gcov"           # nothing to derive


def test_run_success_and_failure(tmp_path, capsys):
    """Check that run() passes env/cwd and raises with the captured output on failure."""
    cov.run([sys.executable, "-c", "import os,sys; sys.exit(os.environ['X'] != 'y')"],
            cwd=tmp_path, env={"X": "y"})
    with pytest.raises(SystemExit, match=r"command failed \(3\)"):
        cov.run([sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"])
    cap = capsys.readouterr()
    assert "out" in cap.out and "err" in cap.err


def test_in_scope_and_expected_files(repo, monkeypatch):
    """Check scope matching and expected_files() filtering of headers and exclusions."""
    assert cov.in_scope("benchmarks/micro/micro.h", cov.C_SCOPE)
    assert not cov.in_scope("x/benchmarks/a_infer.c", cov.C_SCOPE)
    listing = "benchmarks/a_infer.c\nbenchmarks/micro/m.c\nbenchmarks/micro/m.h\nbenchmarks/b_infer.c\nREADME.md\n"

    def fake_run(cmd, cwd, capture_output, text, check):
        """Fake ``git ls-files``."""
        assert cmd[:2] == ["git", "ls-files"] and cwd == repo
        return subprocess.CompletedProcess(cmd, 0, stdout=listing)

    monkeypatch.setattr(cov.subprocess, "run", fake_run)
    assert cov.expected_files(cov.C_SCOPE, ["benchmarks/b_infer.c"]) == {
        "benchmarks/a_infer.c", "benchmarks/micro/m.c"}


def test_evaluate(tmp_path, capsys):
    """Check the gate decision, the per-file table and the JSON summary."""
    files = {"a.py": _fc({i: 1 for i in range(1, 92)} | {i: 0 for i in range(92, 101)})}
    assert cov.evaluate("Python", files, set(), tmp_path) is True
    out = capsys.readouterr().out
    assert "Python: 91/100 lines = 91.00 %" in out and "PASS" in out
    doc = json.loads((tmp_path / "python.json").read_text())
    assert doc["covered"] == 91 and doc["files"]["a.py"] == {"covered": 91, "total": 100}
    exactly = {"b.cc": _fc({i: int(i <= 90) for i in range(1, 101)})}
    assert cov.evaluate("C++", exactly, set(), tmp_path) is False      # strictly above 90 %
    assert json.loads((tmp_path / "cxx.json").read_text())["percent"] == 90.0
    assert cov.evaluate("C", {"a.c": _fc({1: 1})}, {"a.c", "z.c"}, tmp_path) is False
    out = capsys.readouterr().out
    assert "MISSING  z.c" in out and "FAIL" in out
    assert json.loads((tmp_path / "c.json").read_text())["missing"] == ["z.c"]


def test_gcovr_files(repo, tmp_path, monkeypatch):
    """Check gcovr JSON parsing: relative paths, system headers, excluded lines."""
    rep = {"files": [
        {"file": "src/a.cpp", "lines": [{"line_number": 1, "count": 2},
                                        {"line_number": 2, "count": 0},
                                        {"line_number": 3, "count": 0, "gcovr/excluded": True}]},
        {"file": "/usr/include/vector", "lines": [{"line_number": 9, "count": 1}]},
        {"file": "src/../src/a.cpp", "lines": [{"line_number": 2, "count": 5}]},
    ]}
    seen = []

    def fake_run(cmd, cwd=None, env=None, quiet=True):
        """Fake gcovr: write the report to the ``--json`` path."""
        seen.append((cmd, cwd))
        Path(cmd[cmd.index("--json") + 1]).write_text(json.dumps(rep))

    monkeypatch.setattr(cov, "run", fake_run)
    out_json = tmp_path / "g.json"
    files = cov.gcovr_files(repo / "build", repo, "gcov-x", out_json)
    assert list(files) == ["src/a.cpp"]
    assert files["src/a.cpp"].lines == {1: 2, 2: 5}
    cmd, cwd = seen[0]
    assert cmd[:3] == ["gcovr", "--gcov-executable", "gcov-x"] and cwd == repo
    assert cmd[-1] == str(repo / "build")


def test_python_coverage(repo, tmp_path, monkeypatch, calls):
    """Check one pytest run per directory and parsing of the coverage.py JSON report."""
    for d in cov.PYTEST_DIRS:
        (repo / d).mkdir(parents=True)
    out = tmp_path / "out"
    out.mkdir()
    (out / ".coverage.python").write_text("stale")
    (out / "python.coverage.json").write_text(json.dumps({"files": {
        str(repo / "sim" / "a.py"): {"executed_lines": [1, 2], "missing_lines": [3]},
        "tooling/b.py": {"executed_lines": [], "missing_lines": [4]}}}))
    monkeypatch.chdir(repo)
    files = cov.python_coverage(out)
    assert not (out / ".coverage.python").exists()
    pytest_cmds = [c for c, _, _ in calls if "pytest" in c]
    assert [c[-1] for c in pytest_cmds] == list(cov.PYTEST_DIRS)
    assert all("--cov-append" in c for c in pytest_cmds)
    assert all(env == {"COVERAGE_FILE": str(out / ".coverage.python")} for _, _, env in calls)
    assert any("json" in c for c, _, _ in calls) and any("html" in c for c, _, _ in calls)
    assert files["sim/a.py"].lines == {1: 1, 2: 1, 3: 0}
    assert files["tooling/b.py"].covered == 0


def test_python_coverage_missing_dir(repo, tmp_path, calls):
    """Check that a missing test directory aborts the gate."""
    with pytest.raises(SystemExit, match="missing test directory"):
        cov.python_coverage(tmp_path)
    assert calls == []


def test_cxx_coverage(repo, tmp_path, monkeypatch, calls):
    """Check the C++ build/test sequence and the merge of the two gcovr reports."""
    out = tmp_path / "out"
    gcda = out / "compiler" / "x" / "a.gcda"
    gcda.parent.mkdir(parents=True)
    gcda.write_text("")
    (out / "gem5").mkdir()
    (out / "gem5" / "old").write_text("")
    monkeypatch.setenv("CXX", "clang++")
    reports = iter([{"compiler/winhint/A.cpp": _fc({1: 1, 2: 0})},
                    {"compiler/winhint/A.cpp": _fc({2: 1}), "sim/x.cc": _fc({1: 0})}])
    gc = []

    def fake_gcovr(search, root, gcov, out_json):
        """Return the next synthetic gcovr result."""
        gc.append((search, gcov))
        return next(reports)

    monkeypatch.setattr(cov, "gcovr_files", fake_gcovr)
    files = cov.cxx_coverage(out)
    assert not gcda.exists() and not (out / "gem5" / "old").exists()
    assert files["compiler/winhint/A.cpp"].lines == {1: 1, 2: 1} and "sim/x.cc" in files
    assert gc == [(out / "compiler", "llvm-cov gcov"), (out / "gem5", "llvm-cov gcov")]
    cmds = [c for c, _, _ in calls]
    assert cmds[0] == ["compiler/build.sh"] and calls[0][2]["WINHINT_COVERAGE"] == "1"
    assert cmds[1] == ["compiler/test/run_tests.sh"]
    assert all("CXX=clang++" in c or "HOSTCXX=clang++" in c for c in cmds[2:])
    assert all(env == {"COVERAGE_FLAGS": "--coverage -O0"} for _, _, env in calls[2:])


def test_c_coverage(repo, tmp_path, monkeypatch):
    """Check the three benchmark builds, one run per built binary and the gcovr merge."""
    out = tmp_path / "out"
    ran = []

    def fake_run(cmd, cwd=None, env=None, quiet=True):
        """Fake make (creates one executable kernel and a side file) and kernel runs."""
        if cmd[0] == "make":
            root = Path(next(a for a in cmd if a.startswith("OUT_ROOT="))[9:]) / "plain"
            root.mkdir(parents=True)
            b = root / "k_infer"
            b.write_text("")
            b.chmod(0o755)
            (root / "k_infer.json").write_text("")
            ran.append(("make", "TILED=on" in cmd, any(a.startswith("BENCH_DIR=") for a in cmd)))
        else:
            ran.append((Path(cmd[0]).name, cmd[1], cwd.name))

    monkeypatch.setattr(cov, "run", fake_run)
    monkeypatch.setattr(cov, "gcovr_files", lambda search, root, gcov, out_json: {
        "benchmarks/k_infer.c": _fc({1: int(search.name == "kernels"), 2: int(search.name != "kernels")})})
    files = cov.c_coverage(out)
    assert ran == [("make", False, False), ("k_infer", "small", "kernels"),
                   ("make", True, False), ("k_infer", "small", "kernels-tiled"),
                   ("make", False, True), ("k_infer", "small", "micro")]
    assert files["benchmarks/k_infer.c"].lines == {1: 1, 2: 1}


def test_c_coverage_no_binaries(repo, tmp_path, calls):
    """Check that a build producing no binary aborts."""
    with pytest.raises(SystemExit, match="no binaries built"):
        cov.c_coverage(tmp_path / "out")


@pytest.fixture
def fake_suites(tmp_path, monkeypatch):
    """Replace the three collectors and expected_files(); send outputs to tmp_path.

    Args:
        tmp_path (Path): pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): pytest monkeypatch fixture.

    Returns:
        (dict): Mutable per-language results returned by the fakes.
    """
    res = {"python": {"a.py": _fc({1: 1})},
           "cxx": {"compiler/winhint/A.cpp": _fc({1: 1}), "elsewhere/B.cpp": _fc({1: 0}),
                   cov.GEM5_WINDOW + "/controller.cc": _fc({1: 0})},
           "c": {"benchmarks/a_infer.c": _fc({1: 1}), "benchmarks/other.c": _fc({1: 0})}}
    monkeypatch.setenv("WINHINT_BUILD", str(tmp_path / "b"))
    monkeypatch.setattr(cov, "python_coverage", lambda out: res["python"])
    monkeypatch.setattr(cov, "cxx_coverage", lambda out: res["cxx"])
    monkeypatch.setattr(cov, "c_coverage", lambda out: res["c"])
    monkeypatch.setattr(cov, "expected_files", lambda pats, excl=(): set())
    return res


def test_main_all_languages(fake_suites, tmp_path):
    """Check main() over all languages: scope filtering and per-language JSON."""
    assert cov.main([]) == 0
    d = tmp_path / "b" / "coverage"
    assert set(json.loads((d / "cxx.json").read_text())["files"]) == {"compiler/winhint/A.cpp"}
    assert set(json.loads((d / "c.json").read_text())["files"]) == {"benchmarks/a_infer.c"}
    assert (d / "python.json").exists()


def test_main_gate_and_no_gate(fake_suites):
    """Check that a failing language fails the gate unless --no-gate."""
    fake_suites["c"] = {"benchmarks/a_infer.c": _fc({1: 0})}
    assert cov.main(["c"]) == 1
    assert cov.main(["--no-gate", "c"]) == 0
    assert cov.main(["python"]) == 0


def test_main_unknown_language(fake_suites, capsys):
    """Check that an unknown language is a usage error."""
    with pytest.raises(SystemExit) as e:
        cov.main(["rust"])
    assert e.value.code == 2 and "unknown language(s): rust" in capsys.readouterr().err


def test_unit_candidates_cover_cmake_and_combined_link_names(tmp_path):
    """Map CMake (``x.cpp.gcda``) and compile-and-link (``target-x.gcda``) names to sources."""
    assert {"HintPlacement.cpp", "HintPlacement"} <= cov.unit_candidates(
        tmp_path / "HintPlacement.cpp.gcda")
    assert "test_policies_b" in cov.unit_candidates(tmp_path / "test_policies_b-test_policies_b.gcda")
    assert "bbv_policy" in cov.unit_candidates(tmp_path / "test_window_policies-bbv_policy.gcda")


def test_check_units_fails_on_a_unit_gcovr_dropped(tmp_path):
    """A .gcda whose source is not in gcovr's report must fail the gate, naming the unit."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "t-kept.gcda").write_bytes(b"")
    (tmp_path / "sub" / "t-lost.gcda").write_bytes(b"")
    with pytest.raises(SystemExit, match="sub/t-lost.gcda") as err:
        cov.check_units(tmp_path, {"kept.cc", "other.hh"})
    assert "t-kept" not in str(err.value)


def test_check_units_accepts_every_reported_unit(tmp_path):
    """No error when every unit's source appears, with or without its extension."""
    (tmp_path / "a.cpp.gcda").write_bytes(b"")
    (tmp_path / "b-b.gcda").write_bytes(b"")
    cov.check_units(tmp_path, {"a.cpp", "b.c"})
