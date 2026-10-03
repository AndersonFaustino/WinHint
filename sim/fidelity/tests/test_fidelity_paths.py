"""sim/fidelity/fidelity.py: metric fallbacks, run execution, derive-step errors and CLI paths.

No gem5: runs are synthetic directories or tiny Python commands.
"""
import json
import runpy
import shutil
import sys
import types
from pathlib import Path

import pytest

import fidelity as fd

N = 4
TABLE = [{"rob": r, "iq": q, "lq": l, "sq": l} for r, q, l in
         zip([64, 128, 192, 256], [32, 64, 96, 128], [16, 32, 48, 64])]
MACHINE = {"name": "riscv_ooo", "window_table": TABLE}
TRACE_HEADER = ("cycle,insts,ipc,rob_occ_mean,iq_occ_mean,lq_occ_mean,l1d_mpki,l2_mpki,mlp,"
                "branch_mpki,config,region\n")


def _done(d: Path, cyc=None, extra=(), spec=None) -> Path:
    """Write a finished run (stats.txt + ok run.json) into ``d`` and return it.

    Args:
        d: Run directory (created).
        cyc (list[int] | None): Per-config ``cyclesInConfig`` values, or None to omit them.
        extra (Sequence[str]): Additional ``stats.txt`` lines.
        spec (dict | None): ``spec`` stored in run.json.
    """
    d.mkdir(parents=True, exist_ok=True)
    lines = ["simInsts 2000", "system.cpu.numCycles 1000", *extra]
    lines += [f"system.cpu.window.cyclesInConfig::{i} {c}" for i, c in enumerate(cyc or [])]
    (d / "stats.txt").write_text("\n".join(lines) + "\n")
    (d / "run.json").write_text(json.dumps({"status": "ok", "spec": spec or {}}))
    return d


def _spec(policy="static", initial=3, window_args=""):
    """Return a micro_gather RunSpec with the given policy, initial config and window args."""
    return fd.RunSpec(fd.GATHER, "t", "plain", policy, initial, window_args, "large")


def _phased_small(ctx: fd.Ctx, traces=True) -> None:
    """Create done small-input static micro_phased runs (region stats, optional traces)."""
    for c in range(N):
        d = _done(ctx.outdir(fd.PHASED, f"c{c}", "small"), [1000 if i == c else 0 for i in range(N)])
        (d / "region_stats.csv").write_text("region,config,enter_cycle,cycles,insts\n"
                                            f"1,{c},0,{1000 - 100 * c},700\n"
                                            f"2,{c},1000,{100 + c},300\n")
        if traces and c == 0:
            (d / "window_trace.csv").write_text(TRACE_HEADER + "1000,700,0.7,1,1,1,0,0,0,0,0,1\n"
                                                "2000,300,0.3,1,1,1,0,0,0,0,0,2\n")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_run_metrics_residency_from_trace(tmp_path):
    """Without cyclesInConfig stats the residency comes from window_trace.csv (no occupancy)."""
    d = _done(tmp_path / "r", extra=["system.cpu.window.robOccSum::0 5000",
                                     "system.cpu.window.l1dOutstandingDist::mean 2.5"])
    (d / "window_trace.csv").write_text(TRACE_HEADER + "1000,700,0.7,1,1,1,0,0,0,0,0,1\n"
                                        "2000,700,0.7,1,1,1,0,0,0,0,0,1\n"
                                        "3000,700,0.7,1,1,1,0,0,0,0,2,1\n"
                                        "4000,700,0.7,1,1,1,0,0,0,0,2,1\n")
    m = fd.run_metrics(d, _spec("mlp"), TABLE)
    assert m["residency_source"] == "trace" and m["residency"] == [0.5, 0, 0.5, 0]
    assert m["rob_cap"] == pytest.approx(0.5 * 64 + 0.5 * 192)
    assert "rob_occ" not in m                        # occupancy only with stats residency
    assert m["l1d_outstanding_mean"] == 2.5


@pytest.mark.parametrize("initial, res", [(1, [0, 1, 0, 0]), ("max", [0, 0, 0, 1])])
def test_run_metrics_static_fallback(tmp_path, initial, res):
    """Without stats or trace the run counts 100 % in its initial config (symbolic -> largest)."""
    m = fd.run_metrics(_done(tmp_path / "r"), _spec(initial=initial), TABLE)
    assert m["residency_source"] == "static" and m["residency"] == res
    assert "l1d_outstanding_mean" not in m


def test_region_table_skips_dirs_and_empty_phase_regions(tmp_path):
    """Static runs without region_stats.csv are skipped; an empty table has no phases."""
    (tmp_path / "c0").mkdir()
    assert fd.region_table({0: tmp_path / "c0"}) == {}
    assert fd.phase_regions({}, 0.05) == []


# ---------------------------------------------------------------------------
# Commands and execution
# ---------------------------------------------------------------------------

def test_binary_path_per_machine_hint_builds(tmp_path):
    """jones/winhint use a per-machine build on other machines only when it exists."""
    s = fd.resolve(fd.policy_run(fd.COMPUTE, "jones_i3", "hint", "max", variant="jones"), N)
    assert fd.binary_path(tmp_path, s, "big") == tmp_path / "jones" / fd.COMPUTE
    p = tmp_path / "jones" / "big" / fd.COMPUTE
    p.parent.mkdir(parents=True)
    p.write_text("")
    assert fd.binary_path(tmp_path, s, "big") == p
    assert fd.binary_path(tmp_path, s, fd.DEFAULT_MACHINE) == tmp_path / "jones" / fd.COMPUTE


def test_gem5_cmd_period_and_window_args(tmp_path):
    """``--window-period`` and ``--window-args`` are passed when given; no LUT without one."""
    cmd = fd.gem5_cmd("g", tmp_path, Path("m.json"), Path("b"), _spec("hint", 3, "structs=iq"),
                      None, 500)
    assert cmd[cmd.index("--window-period") + 1] == "500"
    assert cmd[cmd.index("--window-args") + 1] == "structs=iq"
    assert "--window-lut" not in cmd


def test_is_done_requires_ok_json(tmp_path):
    """A run is done only with stats.txt and a parseable ``ok`` run.json."""
    d = _done(tmp_path / "r")
    assert fd.is_done(d)
    (d / "run.json").write_text("{broken")
    assert not fd.is_done(d)
    (d / "run.json").write_text(json.dumps({"status": "failed"}))
    assert not fd.is_done(d)


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock not available")
def test_execute_under_lock_without_stats_fails(tmp_path):
    """A locked command that writes no stats.txt is recorded as failed (rc 0)."""
    lock = tmp_path / "l" / ".heavy.lock"
    meta = fd.execute([sys.executable, "-c", "print('hello')"], tmp_path / "o", _spec(), lock,
                      None)
    assert meta["lock"] == str(lock) and lock.exists()
    assert meta["returncode"] == 0 and meta["status"] == "failed"
    assert (tmp_path / "o" / "gem5.log").read_text() == "hello\n"
    assert json.loads((tmp_path / "o" / "run.json").read_text())["spec"]["policy"] == "static"


def test_execute_timeout(tmp_path):
    """A timed-out command gets return code -9 and the timeout is recorded."""
    out = tmp_path / "o"
    meta = fd.execute([sys.executable, "-c", "import time; time.sleep(30)"], out, _spec(), None, 1)
    assert meta["returncode"] == -9 and meta["timeout"] == 1 and meta["status"] == "failed"
    assert (out / "gem5.log").read_text() == ""


# ---------------------------------------------------------------------------
# Context and derive steps
# ---------------------------------------------------------------------------

def test_ctx_table_missing_runs_and_region_stats(tmp_path):
    """The region table needs every static run done and at least one region_stats.csv."""
    ctx = fd.Ctx(tmp_path, MACHINE, N)
    with pytest.raises(fd.Missing, match="c0"):
        ctx.table("small")
    for c in range(N):
        _done(ctx.outdir(fd.PHASED, f"c{c}", "small"))
    with pytest.raises(fd.Missing, match="no region_stats.csv"):
        ctx.table("small")


def test_missing_trace_and_bbv_phases_are_reported(tmp_path):
    """A done run without window_trace.csv / bbv_phases.csv makes the dependent checks missing."""
    ctx = fd.Ctx(tmp_path, MACHINE, N)
    _done(ctx.outdir(fd.PHASED, "bbv_i3"), [0, 0, 0, 100])
    with pytest.raises(fd.Missing, match="window_trace.csv"):
        fd._trace_near(ctx, "bbv_i3", 0.0)
    missing = {c.missing for c in fd.eval_B4(ctx) if c.missing}
    assert any(m.endswith("bbv_i3/bbv_phases.csv") for m in missing)


def test_derive_lut_missing_trace_and_failing_step(tmp_path):
    """derive_lut needs every small static trace; a failing pipeline step raises RuntimeError."""
    ctx = fd.Ctx(tmp_path, MACHINE, N)
    _phased_small(ctx)
    with pytest.raises(fd.Missing, match="c1/window_trace.csv"):
        fd.derive_lut(ctx)
    for c in range(1, N):
        shutil.copy(ctx.outdir(fd.PHASED, "c0", "small") / "window_trace.csv",
                    ctx.outdir(fd.PHASED, f"c{c}", "small") / "window_trace.csv")
    with pytest.raises(RuntimeError, match=r"B5 pipeline step failed \(1\)"):
        fd.derive_lut(ctx, python="false")
    out = tmp_path / "b5" / "riscv_ooo"
    assert "train_phase_classifier.py" in (out / "pipeline.log").read_text()
    assert len((out / "dataset.csv").read_text().splitlines()) == 1 + 2 * N
    assert json.loads((tmp_path / "oracle" / "riscv_ooo" / "small" / f"{fd.PHASED}.json")
                      .read_text()) == {"1": 3, "2": 0}


def test_derive_pgo_build_failure(tmp_path, monkeypatch):
    """A failing pgo build raises RuntimeError with the build output."""
    ctx = fd.Ctx(tmp_path, MACHINE, N)
    _phased_small(ctx, traces=False)
    monkeypatch.setattr(fd, "pgo_make_cmd", lambda d, m: [
        sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"])
    with pytest.raises(RuntimeError, match="pgo build failed(.|\n)*boom"):
        fd.derive_pgo(ctx, tmp_path / "m.json")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def test_collect_runs_conflicting_specs(monkeypatch):
    """Two baselines defining different specs for one run key are rejected."""
    def mk(policy):
        """Return a fake baseline whose only run is gather/large/x with ``policy``."""
        return types.SimpleNamespace(runs=lambda n: [fd.policy_run(fd.GATHER, "x", policy, 0)])
    monkeypatch.setattr(fd, "BASELINES", {"A": mk("mlp"), "B": mk("occupancy")})
    assert len(fd.collect_runs(["A", "A"], N)) == 1
    with pytest.raises(ValueError, match="conflicting run specs for micro_gather/large/x"):
        fd.collect_runs(["A", "B"], N)


def test_unknown_baseline_is_a_usage_error(capsys):
    """An unknown ``--baselines`` entry is an argparse error (exit 2)."""
    with pytest.raises(SystemExit) as e:
        fd.parse_args(["--baselines", "B3", "B42"])
    assert e.value.code == 2 and "unknown baseline(s) ['B42']" in capsys.readouterr().err


def _args(tmp_path, machines_dir, *extra):
    """Return CLI arguments pointing fidelity.py at temp results and a missing gem5."""
    return ["--machines-dir", str(machines_dir), "--results", str(tmp_path / "res"),
            "--bin-root", str(tmp_path / "bin"), "--gem5", str(tmp_path / "no-gem5"),
            "--no-lock", *extra]


def test_only_selecting_nothing_and_build(tmp_path, machines_dir, monkeypatch, capsys):
    """``--only`` matching no run warns; ``--build`` runs make; verdicts merge into summary."""
    calls = []
    monkeypatch.setattr(fd, "subprocess", types.SimpleNamespace(
        run=lambda cmd, **kw: calls.append((cmd, kw))))
    (tmp_path / "res").mkdir()
    (tmp_path / "res" / "summary.json").write_text(json.dumps({"B9": "PASS"}))
    assert fd.main(_args(tmp_path, machines_dir, "--baselines", "B6", "--only", "nope/*",
                         "--evaluate-only", "--build")) == 0
    out = capsys.readouterr()
    assert "selects no run" in out.err and "missing: " in out.out
    assert calls[0][0][:5] == ["make", "-j1", "-C", str(fd.REPO / "benchmarks"), "micro"]
    assert calls[0][1] == {"check": True}
    assert json.loads((tmp_path / "res" / "summary.json").read_text()) == {
        "B9": "PASS", "B6": "INCOMPLETE"}


def test_derive_error_is_a_failure(tmp_path, machines_dir, capsys):
    """A derive step lacking its static runs is reported and makes the exit code 1."""
    assert fd.main(_args(tmp_path, machines_dir, "--baselines", "B5", "--only",
                         f"{fd.PHASED}/large/lut_i3")) == 1
    err = capsys.readouterr().err
    assert "[ERROR] derive step:" in err and "c0" in err
    assert json.loads((tmp_path / "res" / "B5.json").read_text())["status"] == "INCOMPLETE"


def test_script_entry_point(monkeypatch):
    """Run as a script, fidelity.py exits 2 on an unknown baseline."""
    monkeypatch.setattr(sys, "argv", [fd.__file__, "--baselines", "nope"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(fd.__file__, run_name="__main__")
    assert e.value.code == 2
