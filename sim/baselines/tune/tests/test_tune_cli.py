"""tune_baselines.py: grids, build products, run costs, tuned.json writing and CLI paths."""
import contextlib
import io
import json
import runpy
import sys
import types

import pytest

import run_experiments as rx
import tune_baselines as tb

STATS = ("---------- Begin Simulation Statistics ----------\n"
         "simSeconds 0.000500000\nsimInsts 1000000\nsystem.cpu.numCycles 1000000\n"
         "system.cpu.window.cyclesInConfig::3 1000000\n"
         "---------- End Simulation Statistics   ----------\n")


def _run_dir(d, status="ok", run_json=None):
    """Write a finished run directory (stats.txt and run.json) and return it."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "stats.txt").write_text(STATS)
    (d / "run.json").write_text(run_json if run_json is not None
                                else json.dumps({"status": status}))
    return d


def test_cv_grid_from_knobs_file_and_fallback(tmp_path, monkeypatch):
    """The B8 grid comes from knobs.json (non-tunable knobs dropped), else a built-in grid."""
    k = tmp_path / "knobs.json"
    k.write_text(json.dumps({"knobs": {
        "CV_TYPE": {"values": ["consv", "spec"], "default": "consv"},
        "CV_UNROLL": {"values": [1, 2], "default": 2, "tunable": False},
        "CV_INDIR": {"values": [0, 1], "default": 1}}}))
    monkeypatch.setattr(tb, "KNOBS_B8", k)
    assert tb._cv_grid() == ({"CV_TYPE": ["consv", "spec"], "CV_INDIR": [0, 1]},
                             {"CV_TYPE": "consv", "CV_INDIR": 1})
    for bad in (tmp_path / "missing.json", tmp_path / "bad.json"):
        if bad.name == "bad.json":
            bad.write_text("{not json")
        monkeypatch.setattr(tb, "KNOBS_B8", bad)
        grid, default = tb._cv_grid()
        assert len(tb.Method("B8", "clairvoyance", "cv", grid, default).points()) == 60
        assert default == {"CV_TYPE": "consv", "CV_UNROLL": 2, "CV_INDIR": 1}


def test_choose_points_adds_off_grid_default():
    """A default outside the grid is evaluated first, on top of the grid points."""
    m = tb.Method("X", "mlp", "window_args", {"a": [1, 2]}, {"a": 9})
    assert tb.choose_points(m, 5, 0) == [{"a": 9}, {"a": 1}, {"a": 2}]
    pts = tb.choose_points(m, 2, 0)
    assert pts[0] == {"a": 9} and len(pts) == 2


def test_built_products(tmp_path):
    """No product means built; a LUT point is built iff its lut.txt exists; else binaries."""
    assert tb.built(tb.Prep(tmp_path, "plain", None, None, None), "riscv_ooo", ["ka"])
    lut = tmp_path / "b5" / "riscv_ooo" / "lut.txt"
    prep = tb.Prep(tmp_path, "plain", tmp_path / "b5", ["build"], lut)
    assert not tb.built(prep, "riscv_ooo", ["ka"])
    lut.parent.mkdir(parents=True)
    lut.write_text("WINHINT_LUT 1\n")
    assert tb.built(prep, "riscv_ooo", ["ka"])
    prep = tb.Prep(tmp_path / "bin", "winhint", None, ["make"], tmp_path / "bin" / "winhint")
    assert not tb.built(prep, "riscv_ooo", ["ka"])
    (tmp_path / "bin" / "winhint").mkdir(parents=True)
    (tmp_path / "bin" / "winhint" / "ka").write_text("")
    assert tb.built(prep, "riscv_ooo", ["ka"])


def test_run_cost_status_and_objectives(tmp_path, machines_dir):
    """Only ``ok`` runs have a cost: CPI for ``ipc``, ED²P for ``ed2p``."""
    from whdata import load_machine
    m = load_machine(machines_dir / "riscv_ooo.json")
    assert tb.run_cost(tmp_path / "none", m, "ipc", "proxy") is None
    assert tb.run_cost(_run_dir(tmp_path / "f", "failed"), m, "ipc", "proxy") is None
    assert tb.run_cost(_run_dir(tmp_path / "j", run_json="{oops"), m, "ipc", "proxy") is None
    ok = _run_dir(tmp_path / "ok")
    assert tb.run_cost(ok, m, "ipc", "proxy") == 1.0
    e = tb.run_cost(ok, m, "ed2p", "proxy")
    assert e is not None and e > 0


def test_write_tuned_lut_and_non_reference_machine(tmp_path):
    """B5 picks become LUT roots; non-reference machines get entries but no ``*`` fallback."""
    a = tb.parse_args(["--machines", "riscv_ooo", "m2", "--kernels", "ka",
                       "--results-root", str(tmp_path / "tune"), "--methods", "B5", "B3"])
    b5 = {"model": "tree", "edges": 5}
    b3 = {"mlp_thr": 2.0, "gain": 0.1, "miss_min": 1, "shrink_delay": 2}
    p5, p3 = tb.point_id(b5), tb.point_id(b3)
    chosen = {"B5": {"riscv_ooo": (p5, {p5: 0.9}), "m2": (p5, {p5: 0.8})},
              "B3": {"riscv_ooo": (None, {}), "m2": (p3, {p3: 0.95})}}
    t = tb.write_tuned(a, {"B5": [b5], "B3": [b3]}, chosen)
    root = str(tmp_path / "tune" / "b5" / p5)
    assert t["lut_root"] == {"riscv_ooo": {"root": root}, "*": {"root": root},
                             "m2": {"root": root}}
    assert t["window_args"] == {"m2": {"mlp": "mlp_thr=2.0,gain=0.1,miss_min=1,shrink_delay=2"}}
    assert t["chosen"] == {"B5": {"riscv_ooo": p5, "m2": p5}, "B3": {"riscv_ooo": None, "m2": p3}}
    assert json.loads(a.out.read_text()) == t


def test_unknown_method_is_a_usage_error(capsys):
    """An unknown ``--methods`` entry is an argparse error (exit 2)."""
    with pytest.raises(SystemExit) as e:
        tb.parse_args(["--methods", "B3", "B42", "--kernels", "ka"])
    assert e.value.code == 2 and "unknown methods ['B42']" in capsys.readouterr().err


def test_dry_run_flags_unsupported_policy_and_missing_lut(tmp_path, machines_dir, fake_gem5,
                                                          monkeypatch):
    """Runs whose se.py policy is unknown or whose LUT is missing are listed as MISSING."""
    monkeypatch.setattr(rx, "MACHINES_DIR", machines_dir)
    monkeypatch.setattr(rx, "se_policies", lambda *a: {"static", "lut"})
    bins = tmp_path / "bin"
    (bins / "plain").mkdir(parents=True)
    (bins / "plain" / "ka").write_text("")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert tb.main(["--dry-run", "--methods", "B3", "B5", "--budget", "2", "--kernels", "ka",
                        "--bin-root", str(bins), "--gem5", str(fake_gem5), "--no-lock",
                        "--results-root", str(tmp_path / "tune")]) == 0
    out = buf.getvalue()
    assert out.count("policy 'mlp' not accepted by sim/se.py yet") == 2
    assert "not accepted" not in "".join(l for l in out.splitlines(True) if " B5 " in l)
    assert out.count(" MISSING ") == 4 and "LUT " in out and "/lut.txt" in out
    assert "# build step: 2 commands" in out        # one build_lut.py per B5 point


def test_build_step_failure_warns_and_selects_nothing(tmp_path, machines_dir, fake_gem5,
                                                      monkeypatch, capsys):
    """A failing build only warns; its runs are skipped and no point gets chosen."""
    monkeypatch.setattr(rx, "MACHINES_DIR", machines_dir)
    calls = []

    def fake_run(cmd, **kw):
        """Record a build command and report failure."""
        calls.append(cmd)
        return types.SimpleNamespace(returncode=2)
    monkeypatch.setattr(tb, "subprocess", types.SimpleNamespace(run=fake_run))
    assert tb.main(["--methods", "WinHint", "--budget", "1", "--kernels", "ka",
                    "--bin-root", str(tmp_path / "bin"), "--gem5", str(fake_gem5), "--no-lock",
                    "--results-root", str(tmp_path / "tune"),
                    "--tune-bins", str(tmp_path / "tbins"), "--objective", "ipc"]) == 0
    out = capsys.readouterr().out
    assert len(calls) == 1 and calls[0][:2] == ["make", "-C"] and "VARIANT=winhint" in calls[0]
    assert "[WARN] build failed: WinHint switch_cost=0-hysteresis=0.25" in out
    assert "[SKIP] riscv_ooo/ka/WinHint/" in out
    t = json.loads((tmp_path / "tune" / "tuned.json").read_text())
    assert t["chosen"] == {"WinHint": {"riscv_ooo": None}} and t["compiler"] == {}


def test_script_entry_point(monkeypatch, capsys):
    """Run as a script, tune_baselines exits 2 on an unknown method."""
    monkeypatch.setattr(sys, "argv", [tb.__file__, "--methods", "nope", "--kernels", "ka"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(tb.__file__, run_name="__main__")
    assert e.value.code == 2
