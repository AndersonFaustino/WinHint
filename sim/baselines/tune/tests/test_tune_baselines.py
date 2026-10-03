"""Baseline tuning harness (sim/baselines/tune/tune_baselines.py) with a fake gem5."""
import contextlib
import io
import json

import run_experiments as rx
import tune_baselines as tb


def _bins(root, dirs, kernels=("ka", "kb")):
    """Create empty kernel binaries for every variant directory in ``dirs``; return ``root``."""
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
        for k in kernels:
            (root / d / k).write_text("")
    return root


def _common(tmp_path, fake_gem5, bins):
    """Return the CLI arguments shared by the tuning tests (kernels, roots, fake gem5)."""
    return ["--kernels", "ka", "kb", "--bin-root", str(bins), "--gem5", str(fake_gem5),
            "--no-lock", "--results-root", str(tmp_path / "tune"),
            "--tune-bins", str(tmp_path / "tbins"), "--lut-root", str(tmp_path / "b5"),
            "--energy-model", "proxy"]


def test_equal_budget_default_point_and_shared_b8_points():
    """Every method gets the same budget, starts at its default and B8 variants share points."""
    ms = tb.methods()
    assert {"B2", "B3", "B4", "B5", "B8", "B8+WinHint", "B9", "hybrid", "WinHint"} <= set(ms)
    budget = len(ms["WinHint"].points())
    assert budget == 16
    for name, m in ms.items():
        pts = tb.choose_points(m, budget, seed=0)
        assert len(pts) == min(budget, len(m.points())), name
        assert pts[0] == m.default and len({tb.point_id(p) for p in pts}) == len(pts)
        assert all(p in m.points() for p in pts), name
        assert set(m.default) == set(m.grid)
    # the two Clairvoyance variants are tuned over the same subset of the 60-point grid
    assert len(ms["B8"].points()) == 60
    assert tb.choose_points(ms["B8"], budget, 0) == tb.choose_points(ms["B8+WinHint"], budget, 0)
    assert tb.choose_points(ms["B3"], budget, 1) != tb.choose_points(ms["B3"], budget, 0)


def test_dry_run_lists_points_builds_and_jobs(tmp_path, machines_dir, fake_gem5, monkeypatch):
    """``--dry-run`` lists points, build commands and gem5 jobs without running anything."""
    monkeypatch.setattr(rx, "MACHINES_DIR", machines_dir)
    bins = _bins(tmp_path / "bin", ["plain", "winhint", "clairvoyance", "winhint_clairvoyance"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert tb.main(["--dry-run"] + _common(tmp_path, fake_gem5, bins)) == 0
    out = buf.getvalue()
    assert "# budget 16 points per method" in out
    assert "# B8 (clairvoyance, cv): 16 of 60 grid points" in out
    assert "# WinHint (winhint, winhint): 16 of 16 grid points" in out
    assert "VARIANT=winhint " in out and "-winhint-hysteresis=0.1" in out   # compiler-knob builds
    assert "CV_TYPE=" in out and "build_lut.py" in out
    assert "--window-policy mlp" in out and "--options small" in out
    assert not (tmp_path / "tune").exists()                                # nothing ran


def test_tune_window_args_compiler_and_cv_knobs(tmp_path, machines_dir, fake_gem5, monkeypatch):
    """Tuning finds the fake gem5's optima and the campaign picks up the tuned parameters."""
    monkeypatch.setattr(rx, "MACHINES_DIR", machines_dir)
    ms = tb.methods()
    wh = [tb.point_id(p) for p in tb.choose_points(ms["WinHint"], 16, 0)]
    cv = tb.choose_points(ms["B8"], 16, 0)
    dirs = ["plain", "winhint"] + ["clairvoyance" + tb.cv_suffix(p, ms["B8"].default) for p in cv]
    if {"CV_TYPE": "spec", "CV_UNROLL": 4, "CV_INDIR": 2} not in cv:
        cv_best = None
    else:
        cv_best = "-cv_spec-u4-i2"
    bins = _bins(tmp_path / "bin", dirs)
    # WinHint knob builds (normally made by the build step) with their sidecars
    for pid in wh:
        d = tmp_path / "tbins" / "WinHint" / pid / "riscv" / "winhint"
        _bins(d.parent, ["winhint"])
        sc, hy = (500, 0.1) if pid == "switch_cost=500-hysteresis=0.1" else (42, 0.25)
        for k in ("ka", "kb"):
            (d / f"{k}.winhint.json").write_text(json.dumps(
                {"target": "riscv_ooo", "switch_cost_cycles": sc, "hysteresis": hy}))
    args = (["--methods", "B3", "hybrid", "WinHint", "B8", "--no-build", "--objective", "ipc"]
            + _common(tmp_path, fake_gem5, bins))
    assert tb.main(args) == 0
    t = json.loads((tmp_path / "tune" / "tuned.json").read_text())
    assert t["budget"] == 16 and t["input"] == "small" and t["objective"] == "ipc"
    mlp = dict(kv.split("=") for kv in t["window_args"]["riscv_ooo"]["mlp"].split(","))
    pts_b3 = tb.choose_points(ms["B3"], 16, 0)
    best_thr = min({p["mlp_thr"] for p in pts_b3}, key=lambda x: abs(x - 2.0))
    assert float(mlp["mlp_thr"]) == best_thr
    assert t["window_args"]["*"] == t["window_args"]["riscv_ooo"]
    assert "miss_mpki=0.5" in t["window_args"]["riscv_ooo"]["hybrid"]
    assert t["compiler"]["winhint"]["make"] == {
        "SWITCH_COST": "500", "CFLAGS_EXTRA": "-mllvm -winhint-hysteresis=0.1"}
    assert t["compiler"]["winhint"]["sidecar"] == {"switch_cost_cycles": 500, "hysteresis": 0.1}
    if cv_best:
        assert t["binary_suffix"]["clairvoyance"] == cv_best
    # every point of every method ran on every kernel (equal effort)
    for meth in ("B3", "hybrid", "WinHint", "B8"):
        assert len(t["scores"][meth]["riscv_ooo"]) == 16
    # the campaign picks the tuned parameters up
    runs = {r.variant.name: r for r in rx.build_runs(rx.parse_args(
        ["--kernels", "ka", "--bin-root", str(bins), "--run-length", "full",
         "--tuned", str(tmp_path / "tune" / "tuned.json")]))}
    assert f"mlp_thr={best_thr}" in " ".join(runs["mlp"].cmd)
    # default winhint build (sidecar-less here) is flagged: it lacks the tuned knobs
    assert any("tuned WinHint knobs" in m for m in runs["winhint"].missing)
    # --select-only reproduces the same choice from the existing runs
    assert tb.main(args + ["--select-only"]) == 0
    t2 = json.loads((tmp_path / "tune" / "tuned.json").read_text())
    assert t2["chosen"] == t["chosen"]
