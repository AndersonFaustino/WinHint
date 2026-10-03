"""In-process tests of the --uarch and --fidelity drivers with faked tools (no hardware needed).

``run_hw_experiments.topology`` and ``subprocess.run`` are monkeypatched, so the drivers
see a hybrid CPU and fake probe/workload/PIE/libwinhint outputs.
"""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW))
import fidelity as fid  # noqa: E402
import run_hw_experiments as rhe  # noqa: E402

#: Fake hybrid topology.
TOPO = {"pcpus": "0-3", "ecpus": "4-11", "online": "0-11", "hybrid": True, "model": "FakeCPU",
        "smt": "off", "source": "env"}


@pytest.fixture
def hybrid(monkeypatch):
    """Make ``run_hw_experiments.topology()`` report a hybrid P/E CPU.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest monkeypatch fixture.
    """
    monkeypatch.setattr(rhe, "topology", lambda: dict(TOPO))


# ----------------------------------------------------------------------------- small helpers
def test_confirm_uarch_mlp_without_effective_value():
    """Check that an MLP measurement without ``mlp_effective`` is reported missing."""
    assert fid.confirm_uarch("mlp", 16, {"x": [1, 2]}) == ("missing", "no mlp_effective")


def test_parse_workload_skips_non_numeric():
    """Check that non-numeric ``key=value`` tokens and bare words are dropped."""
    assert fid.parse_workload("mode=ilp n=3 junk ns=1e3") == {"n": 3.0, "ns": 1000.0}


def test_workload_and_pie_argv(tmp_path):
    """Check the phase_workload and pie_daemon command lines."""
    assert fid.workload_argv(tmp_path, "memory", 2) == [
        str(tmp_path / "phase_workload"), "2", str(fid.FID_PHASE_MS), str(fid.FID_MB), "memory"]
    assert fid.workload_argv(tmp_path, "alt-ilp", 1, hinted=True)[0].endswith("phase_workload_hinted")
    a = SimpleNamespace(tool_dir=str(tmp_path), pie_interval_ms=10, pie_slack=0.15, pie_hyst=2)
    argv = fid.pie_argv(a, tmp_path / "r", ["prog", "x"])
    assert argv[0] == str(tmp_path / "pie_daemon")
    assert argv[argv.index("-o") + 1] == str(tmp_path / "r" / "pie.csv")
    assert argv[argv.index("-I") + 1] == "P" and argv[-3:] == ["--", "prog", "x"]


def test_pie_phase_tracking_constant_mpki():
    """Check that tracking is not evaluable when every interval has the same MPKI."""
    rows = [{"side": "P", "decision": "E", "mpki": 5.0, "ratio": 1.0}] * 4
    assert fid.pie_phase_tracking(rows) == (None, None)


# ----------------------------------------------------------------------------- run_uarch
def _uarch_args(tmp_path, dry=False):
    """Return a run_uarch namespace whose tool dir holds a (fake) uarch_probe.

    Args:
        tmp_path (Path): Test temporary directory.
        dry (bool): Value of ``dry_run``.

    Returns:
        (SimpleNamespace): The namespace.
    """
    (tmp_path / "tools").mkdir(exist_ok=True)
    (tmp_path / "tools" / "uarch_probe").write_text("")
    return SimpleNamespace(tool_dir=str(tmp_path / "tools"), out=str(tmp_path / "out"),
                           uarch_smoke=True, dry_run=dry)


def test_run_uarch_dry_run(tmp_path, hybrid, capsys):
    """Check that --dry-run prints both probe commands and writes nothing."""
    fid.run_uarch(_uarch_args(tmp_path, dry=True))
    out = capsys.readouterr().out
    assert "[dry-run] uarch P:" in out and "[dry-run] uarch E:" in out
    assert not (tmp_path / "out").exists()


def test_run_uarch_missing_probe(tmp_path, hybrid):
    """Check that a missing uarch_probe binary aborts with a build hint."""
    a = _uarch_args(tmp_path)
    (tmp_path / "tools" / "uarch_probe").unlink()
    with pytest.raises(SystemExit, match="uarch_probe missing"):
        fid.run_uarch(a)


def test_run_uarch_not_hybrid(tmp_path, monkeypatch):
    """Check that a non-hybrid topology is rejected before anything runs."""
    monkeypatch.setattr(rhe, "topology", lambda: dict(TOPO, hybrid=False, ecpus=""))
    with pytest.raises(SystemExit, match="not a hybrid"):
        fid.run_uarch(_uarch_args(tmp_path))


def test_run_uarch_writes_check(tmp_path, hybrid, monkeypatch, capsys):
    """Run run_uarch with a fake probe and check the per-side report and statuses."""
    calls = []

    def fake_run(cmd, stdout=None, check=False):
        """Fake ``subprocess.run``: write the probe JSON named by the last argument."""
        calls.append(cmd)
        cpu = int(cmd[cmd.index("-c") + 1])
        stdout.write("probe,cpu\n")
        probes = {"rob": {"x": [0, 504, 512], "size_lo": 505, "size_hi": 513},
                  "mlp": {"x": list(range(1, 13)), "mlp_effective": 11.5}}
        Path(cmd[-1]).write_text(json.dumps({"cpu": cpu, "probes": probes}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(fid.subprocess, "run", fake_run)
    fid.run_uarch(_uarch_args(tmp_path))
    out = tmp_path / "out" / "uarch"
    assert [c[2] for c in calls] == ["0", "4"]
    assert (out / "uarch_P.csv").read_text() == "probe,cpu\n"
    rep = json.loads((out / "uarch_check.json").read_text())
    assert rep["model"] == "FakeCPU" and rep["smoke"] is True
    p, e = rep["sides"]["P"], rep["sides"]["E"]
    assert p["cpu"] == 0 and e["cpu"] == 4
    assert p["probes"]["rob"]["status"] == "confirmed"
    assert p["probes"]["mlp"]["status"] == "consistent_bound"
    assert p["probes"]["lb"]["status"] == "missing"
    assert e["probes"]["rob"]["status"] == "differs"           # 512 measured vs 256 documented
    assert e["probes"]["mlp"]["status"] == "undocumented"
    assert "functional check only" in capsys.readouterr().out


# ----------------------------------------------------------------------------- run_fidelity
def _fid_args(tmp_path, dry=False, secs=1.0):
    """Return a run_fidelity namespace with fake tool binaries.

    Args:
        tmp_path (Path): Test temporary directory.
        dry (bool): Value of ``dry_run``.
        secs (float): ``fidelity_secs``.

    Returns:
        (SimpleNamespace): The namespace.
    """
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    for b in ("phase_workload", "phase_workload_hinted", "pie_daemon"):
        (tools / b).write_text("")
    return SimpleNamespace(tool_dir=str(tools), out=str(tmp_path / "out"), fidelity_secs=secs,
                           fidelity_reps=2, sondag_k=2, sondag_threshold=1.4, pie_interval_ms=10,
                           pie_slack=0.15, pie_hyst=2, dry_run=dry)


#: Fake P/E work rate (steps per ns) per class: ilp gains most from P, memory least.
RATE = {("ilp", "P"): 2.0, ("ilp", "E"): 1.0, ("compute", "P"): 1.5, ("compute", "E"): 1.0,
        ("memory", "P"): 1.1, ("memory", "E"): 1.0}

PIE_HDR = "t_ms,side,instructions,cpi,mpki,mlp_cur,mlp_oth,freq_ghz,ratio_e_over_p,decision,migrated\n"


def _pie_rows(cls):
    """Return fake pie.csv rows that make every PIE predicate hold.

    Args:
        cls (str): ``ilp``, ``memory`` or ``alt``.

    Returns:
        (str): CSV body.
    """
    ilp = "10,P,1,1,0.1,1,1,4,1.6,P,0\n"
    mem = "10,E,1,1,30,1,1,3,1.05,E,1\n"
    return {"ilp": ilp * 4, "memory": mem * 4, "alt": (ilp + mem) * 2}[cls]


def _fake_fidelity_run(calls, fail=(), summary=True):
    """Return a fake ``subprocess.run`` for run_fidelity.

    Args:
        calls (list): Receives ``(argv, env)`` per call.
        fail (tuple[str]): Workload classes whose ground-truth run on E exits 1.
        summary (bool): Whether the sondag run writes its summary JSON.

    Returns:
        (callable): The fake.
    """
    def fake(argv, env=None, stdout=None, stderr=None, text=None, preexec_fn=None, timeout=None):
        """Emulate phase_workload, pie_daemon and the sondag run."""
        calls.append((argv, env))
        if Path(argv[0]).name == "pie_daemon":
            cls = argv[-1]
            Path(argv[argv.index("-o") + 1]).write_text(PIE_HDR + _pie_rows(cls))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        cls = argv[-1]
        if cls == "alt-ilp":
            if summary:
                Path(env["WINHINT_LOG"] + ".summary.json").write_text(json.dumps(
                    {"sondag": [{"type": 2, "e_over_p": 1.9, "assigned": "P"},
                                {"type": 1, "e_over_p": 1.12, "assigned": "E"}]}))
            return SimpleNamespace(returncode=0, stdout="sondag", stderr="boom\n")
        side = "P" if preexec_fn is not None and calls_side(preexec_fn) == "P" else "E"
        if cls in fail and side == "E":
            return SimpleNamespace(returncode=1, stdout="", stderr="crashed\n")
        steps = RATE[(cls, side)] * 1000
        return SimpleNamespace(returncode=0, stderr="",
                               stdout=f"{fid.STEPS_KEY[cls]}={steps} {fid.NS_KEY[cls]}=1000")
    return fake


def calls_side(pre):
    """Return the core side a preexec_fn would pin to (CPU 0 = P, otherwise E).

    Args:
        pre (callable): The ``preexec_fn`` built by run_fidelity (binds ``c``).

    Returns:
        (str): ``"P"`` or ``"E"``.
    """
    return "P" if pre.__defaults__[0] == 0 else "E"


def test_run_fidelity_dry_run(tmp_path, hybrid, capsys):
    """Check the --dry-run plan: pinned ground truth, PIE runs, sondag line, low-secs warning."""
    fid.run_fidelity(_fid_args(tmp_path, dry=True, secs=0.1))
    out = capsys.readouterr().out
    assert "warning: --fidelity-secs 0.1" in out
    assert "[dry-run] fidelity gt_ilp_P_0: [cpu 0]" in out and "gt_memory_E_1: [cpu 4]" in out
    assert "fidelity pie_alt: " in out and "pie_daemon -b perf" in out
    assert "WINHINT_MODE=sondag" in out and "alt-ilp" in out
    assert not (tmp_path / "out").exists()


def test_run_fidelity_missing_tool(tmp_path, hybrid):
    """Check that a missing tool binary aborts with a build hint."""
    a = _fid_args(tmp_path)
    (Path(a.tool_dir) / "pie_daemon").unlink()
    with pytest.raises(SystemExit, match="pie_daemon missing"):
        fid.run_fidelity(a)


def test_run_fidelity_full(tmp_path, hybrid, monkeypatch, capsys):
    """Run run_fidelity end to end with fake tools and check the JSON/CSV report."""
    calls = []
    monkeypatch.setattr(fid.subprocess, "run", _fake_fidelity_run(calls))
    fid.run_fidelity(_fid_args(tmp_path, secs=5))
    out = tmp_path / "out" / "fidelity"
    assert len(calls) == 3 * 2 * 2 + 3 + 1
    rep = json.loads((out / "fidelity.json").read_text())
    assert rep["model"] == "FakeCPU" and rep["reps"] == 2
    assert rep["speedup"] == pytest.approx({"ilp": 2.0, "compute": 1.5, "memory": 1.1})
    assert rep["threshold_separating"] is True
    assert rep["sondag_threshold"] == pytest.approx((2.0 * 1.1) ** 0.5)
    preds = {p["id"]: p for p in rep["predicates"]}
    assert all(p["pass"] for p in preds.values() if p["kind"] == "trend")
    sondag_env = calls[-1][1]
    assert sondag_env["WINHINT_MODE"] == "sondag"
    assert sondag_env["WINHINT_SONDAG_THRESHOLD"] == f"{rep['sondag_threshold']:.4f}"
    assert (out / "sondag" / "stderr.txt").read_text() == "boom\n"
    assert (out / "gt_ilp_P_0.stdout").read_text().startswith("ilp_steps=")
    with open(out / "fidelity.csv") as f:
        rows = list(csv.DictReader(f))
    assert {r["id"] for r in rows} == set(preds)
    printed = capsys.readouterr().out
    assert "ok   pie_alt" in printed and "trend predicates failing" not in printed


def test_run_fidelity_failures(tmp_path, hybrid, monkeypatch, capsys):
    """Check failed ground-truth runs and a missing sondag summary are reported, not fatal."""
    calls = []
    monkeypatch.setattr(fid.subprocess, "run", _fake_fidelity_run(calls, fail=("ilp",), summary=False))
    fid.run_fidelity(_fid_args(tmp_path, secs=5))
    printed = capsys.readouterr().out
    assert "FAIL gt_ilp_E_0 (1): crashed" in printed
    assert "FAIL sondag (0): boom" in printed
    rep = json.loads((tmp_path / "out" / "fidelity" / "fidelity.json").read_text())
    assert rep["speedup"]["ilp"] is None and rep["threshold_separating"] is False
    preds = {p["id"]: p for p in rep["predicates"]}
    assert preds["GT-mem-gains-less"]["pass"] is None and preds["R5-order"]["pass"] is None


def test_run_fidelity_reports_failing_trends(tmp_path, hybrid, monkeypatch, capsys):
    """Check that failing trend predicates are listed in the final line."""
    calls = []
    monkeypatch.setattr(fid.subprocess, "run", _fake_fidelity_run(calls))
    monkeypatch.setitem(RATE, ("memory", "P"), 3.0)        # memory gains more than ilp: GT fails
    fid.run_fidelity(_fid_args(tmp_path, secs=5))
    assert "trend predicates failing: GT-mem-gains-less" in capsys.readouterr().out
