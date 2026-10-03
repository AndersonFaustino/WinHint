"""Edge cases and CLIs of the host-side runners (no gem5 needed).

Covers sim/run_baseline.py (forwarding to run_experiments with
``--policies static``), the ``show`` CLI and error paths of sim/run_lengths.py,
winhint_elf's rejection paths, small helpers and failure paths of
sim/run_experiments.py, and the dry-run, McPAT-energy and error paths of
sim/baselines/oracle/oracle_sweep.py.
"""
import json
import math
import runpy
import sys

import pandas as pd
import pytest

import oracle_sweep
import run_baseline
import run_experiments as rx
import run_lengths as rl
import winhint_elf
from conftest import make_region_stats

# ---------------------------------------------------------------------------
# run_baseline.py
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_main(monkeypatch):
    """Replace ``run_experiments.main`` with a recorder; return the list of argv it got."""
    calls = []

    def fake(argv):
        """Record ``argv`` and return 7."""
        calls.append(argv)
        return 7

    monkeypatch.setattr(rx, "main", fake)
    return calls


def test_run_baseline_forces_static_policies(captured_main, monkeypatch):
    """Any --policies (and its values) is replaced by ``--policies static``; status forwarded."""
    assert run_baseline.main(["--kernels", "k", "--policies", "B1", "B5", "--dry-run"]) == 7
    assert run_baseline.main(["--policies", "winhint"]) == 7
    monkeypatch.setattr(sys, "argv", ["run_baseline.py", "--jobs", "2"])
    assert run_baseline.main() == 7
    assert captured_main == [["--policies", "static", "--kernels", "k", "--dry-run"],
                             ["--policies", "static"],
                             ["--policies", "static", "--jobs", "2"]]


def test_run_baseline_script_entry_point(captured_main, monkeypatch):
    """Running run_baseline.py as a script exits with run_experiments' status."""
    monkeypatch.setattr(sys, "argv", ["run_baseline.py", "--policies", "x"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(run_baseline.__file__, run_name="__main__")
    assert e.value.code == 7 and captured_main == [["--policies", "static"]]


def test_run_baseline_dry_run_matrix_is_static_only(tmp_path, machine_json, monkeypatch, capsys):
    """Through the real run_experiments, only the static_c<i> variants are listed."""
    monkeypatch.setattr(rx, "MACHINES_DIR", machine_json.parent)
    (tmp_path / "bin" / "plain").mkdir(parents=True)
    (tmp_path / "bin" / "plain" / "k").write_text("")
    assert run_baseline.main(["--dry-run", "--policies", "winhint", "--kernels", "k",
                              "--run-length", "full", "--bin-root", str(tmp_path / "bin"),
                              "--results-root", str(tmp_path / "res")]) == 0
    out = capsys.readouterr().out
    assert all(f"static_c{i}" in out for i in range(4))
    assert "winhint_hw" not in out and "--window-policy hint" not in out


# ---------------------------------------------------------------------------
# run_lengths.py
# ---------------------------------------------------------------------------

#: region profile: 10000 insts; region 0 has two visits, -1 and 2 have 10 % each
PROFILE = {"total_insts": 10000, "truncated": True,
           "visits": [[0, 1, 1001], [1, 1, 4001], [0, 2, 6001], [2, 1, 9001]]}


@pytest.fixture
def rl_config(tmp_path):
    """Write a run-length config (large: regions, small: fixed) and return its path."""
    p = tmp_path / "rl.json"
    p.write_text(json.dumps({
        "version": 3,
        "defaults": {"large": {"mode": "regions", "measure_insts": 800, "warmup_insts": 0,
                               "min_region_frac": 0.15, "per_region": 1},
                     "small": {"mode": "fixed", "starts": [5000, 1000], "warmup_insts": 100,
                               "measure_insts": 500}},
        "kernels": {"odd": {"large": {"mode": "bogus"}}}}))
    return p


def test_run_lengths_config_and_spec_errors(tmp_path, rl_config):
    """A config without defaults and an unknown mode are rejected."""
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(ValueError, match="no 'defaults'"):
        rl.load_config(tmp_path / "bad.json")
    cfg = rl.load_config(rl_config)
    assert cfg["_path"] == str(rl_config)
    with pytest.raises(ValueError, match="unknown mode 'bogus'"):
        rl.spec_for(cfg, "odd", "large")
    assert rl.spec_for(cfg, "k", "tiny") == {"mode": "full", "version": 3}


def test_run_lengths_plan_modes_and_errors(rl_config):
    """full/fixed plans, region filtering by share, one visit per region and plan errors."""
    cfg = rl.load_config(rl_config)
    assert rl.plan({"mode": "full", "version": 2}) == rl.Plan("full", 2)
    fixed = rl.plan(rl.spec_for(cfg, "k", "small"), {"total_insts": 7000})
    assert [(s.start, s.warmup, s.measure) for s in fixed.samples] == [(900, 100, 500),
                                                                       (4900, 100, 500)]
    assert fixed.total_insts == 7000 and fixed.starts == [900, 4900]
    pl = rl.plan(rl.spec_for(cfg, "k", "large"), PROFILE)
    # regions -1 and 2 (10 % each) are below min_region_frac; region 0: first visit only
    assert [(s.stratum, s.visit, s.start) for s in pl.samples] == [(0, 1, 999), (1, 1, 3999)]
    assert pl.samples[0].stratum_insts == 6000 and pl.unsampled_frac == pytest.approx(0.2)
    assert pl.approximate
    assert rl._evenly(5, 3) == [0, 2, 4] and rl._evenly(4, 1) == [0]
    with pytest.raises(ValueError, match="measure_insts"):
        rl.plan({"mode": "fixed", "starts": [1]})
    with pytest.raises(ValueError, match="'starts'"):
        rl.plan({"mode": "fixed", "measure_insts": 5})
    with pytest.raises(ValueError, match="checkpoint directory"):
        rl.sample_args(rl.Sample(0, 10, 5, 100, 0, 100), None)


def test_run_lengths_stats_merge_edge_cases(tmp_path):
    """Non-numeric/non-finite values, -inf maxima and the recomputed CPI."""
    (tmp_path / "s.txt").write_text("simInsts 100\nfoo bar\nx inf\n")
    assert rl.read_stats(tmp_path / "s.txt") == [("simInsts", 100.0), ("x", 0.0)]
    m = rl.merge_stats([[("simInsts", 100.0), ("system.cpu.numCycles", 200.0),
                         ("system.cpu.cpi", 9.0), ("q.max", -math.inf)]], [1.0])
    assert m["system.cpu.cpi"] == 2.0 and m["q.max"] == 0.0


def test_merge_samples_fixed_mode_unscaled(tmp_path):
    """Fixed mode sums samples unscaled; an empty or missing window_trace.csv is skipped."""
    pl = rl.Plan("fixed", 1, [rl.Sample(0, 0, 0, 50, -2, 50), rl.Sample(1, 90, 10, 50, -2, 50)])
    dirs = [tmp_path / "s00", tmp_path / "s01"]
    for i, d in enumerate(dirs):
        d.mkdir()
        (d / "stats.txt").write_text(f"simInsts {50 + i}\nsystem.cpu.numCycles 100\n")
    (dirs[0] / "window_trace.csv").write_text("")
    summ = rl.merge_samples(tmp_path, pl, dirs)
    assert summ["scales"] == [1.0, 1.0] and summ["measured_insts"] == [50.0, 51.0]
    assert dict(rl.read_stats(tmp_path / "stats.txt"))["simInsts"] == 101
    assert not (tmp_path / "window_trace.csv").exists()
    assert not (tmp_path / "region_stats.csv").exists()


def test_run_lengths_show_cli(tmp_path, rl_config, capsys, monkeypatch):
    """``show`` prints the spec, and the samples when a plan can be made."""
    assert rl.main(["show", "--kernel", "k", "--input", "small", "--config", str(rl_config)]) == 0
    out = capsys.readouterr().out
    assert '"mode": "fixed"' in out and "mode fixed: 2 samples, 1200 detailed insts" in out
    assert "s01 region  -2" in out
    assert rl.main(["show", "--kernel", "k", "--config", str(rl_config)]) == 0
    assert "pass --profile" in capsys.readouterr().out
    prof = tmp_path / "prof.json"
    prof.write_text(json.dumps(PROFILE))
    monkeypatch.setattr(sys, "argv", ["run_lengths.py", "show", "--kernel", "k", "--config",
                                      str(rl_config), "--profile", str(prof)])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(rl.__file__, run_name="__main__")
    out = capsys.readouterr().out
    assert e.value.code == 0
    assert "mode regions: 2 samples" in out and "of 10000 total" in out
    assert "unsampled 20.000%" in out and "(approximate: truncated profile)" in out


# ---------------------------------------------------------------------------
# winhint_elf.py
# ---------------------------------------------------------------------------

def test_winhint_elf_rejections(tmp_path):
    """Non-ORI-x0 words are not hints; ELF32 or big-endian files are rejected."""
    assert winhint_elf.decode_hint(0x00000013) is None          # addi x0, x0, 0
    assert winhint_elf.decode_hint(0x00006013 | (0x15 << 20)) == (1, 0)
    for cls, data in ((1, 1), (2, 2)):
        (tmp_path / "e").write_bytes(b"\x7fELF" + bytes([cls, data]) + bytes(58))
        with pytest.raises(ValueError, match="little-endian ELF64"):
            winhint_elf.scan_hints(tmp_path / "e")


# ---------------------------------------------------------------------------
# run_experiments.py
# ---------------------------------------------------------------------------

def test_run_experiments_helpers(tmp_path):
    """Fallbacks of se_policies, clock/memory parsing, window args and done checks."""
    assert rx.se_policies(tmp_path / "missing_se.py") is None
    assert rx._clock_ghz({"cpu": {"clock": "fast"}}) == 2.0
    assert rx._clock_ghz({"cpu": {"clock": "500MHz"}}) == 0.5
    with pytest.raises(SystemExit, match="POLICY:k=v"):
        rx.parse_window_args(["static"])
    assert rx._mem_size(tmp_path / "missing.json") == "512MB"
    (tmp_path / "m.json").write_text(json.dumps({"memory": {"size": "2GB"}}))
    assert rx._mem_size(tmp_path / "m.json") == "2GB"
    (tmp_path / "stats.txt").write_text("")
    (tmp_path / "run.json").write_text("{not json")
    assert not rx.is_done(tmp_path)
    ck = tmp_path / "ck"
    ck.mkdir()
    job = rx.AuxJob("checkpoint", ck, tmp_path, [], starts=[5, 1])
    assert not job.done()
    (ck / "checkpoints.json").write_text("{bad")
    assert not job.done()
    (ck / "checkpoints.json").write_text(json.dumps({"requested": [1, 5]}))
    assert job.done()


def test_binary_sidecars(tmp_path):
    """binary_target skips *.regions.json; a missing WinHint sidecar is reported."""
    b = tmp_path / "k"
    b.write_text("")
    (tmp_path / "k.a.regions.json").write_text(json.dumps({"target": "wrong"}))
    (tmp_path / "k.b.json").write_text("[]")
    (tmp_path / "k.winhint.json").write_text(json.dumps({"target": "riscv_big"}))
    assert rx.binary_target(b) == "riscv_big"
    tuned = {"compiler": {"winhint": {"sidecar": {"alpha": 0.5}}}}
    assert rx.tuned_compiler_mismatch(tuned, tmp_path / "other", "winhint").startswith(
        "no sidecar")


def test_collect_tolerates_bad_runs(tmp_path, machine_json, monkeypatch):
    """Corrupt run.json is skipped; unknown machines and bad energy caches still give rows."""
    monkeypatch.setattr(rx, "MACHINES_DIR", machine_json.parent)
    root = tmp_path / "res"
    for m, v, meta in (("riscv_ooo", "bad", None),
                       ("ghost", "plain", {"machine": "ghost", "status": "ok"}),
                       ("riscv_ooo", "plain", {"machine": "riscv_ooo", "status": "ok"})):
        d = root / m / "k" / v
        d.mkdir(parents=True)
        (d / "run.json").write_text("{" if meta is None else json.dumps(meta))
        (d / "stats.txt").write_text("simInsts 10\nsystem.cpu.numCycles 20\n")
    (root / "riscv_ooo" / "k" / "plain" / "energy.json").write_text("{corrupt")
    assert rx.collect(root, "proxy", root / "summary.csv") == 2
    s = pd.read_csv(root / "summary.csv").set_index("machine")
    assert s.loc["riscv_ooo", "energy_model"].startswith("error:")
    assert s.loc["ghost", "ipc"] == 0.5 and pd.isna(s.loc["ghost", "energy_model"])


def test_main_clamps_jobs_and_collect_only(tmp_path, capsys):
    """--jobs above MAX_JOBS is clamped; --collect-only only writes summary.csv."""
    res = tmp_path / "res"
    assert rx.main(["--jobs", str(rx.MAX_JOBS + 3), "--collect-only",
                    "--results-root", str(res)]) == 0
    assert "clamped" in capsys.readouterr().out
    assert (res / "summary.csv").exists()


def test_main_without_gem5_fails(tmp_path, machine_json, monkeypatch, capsys):
    """Pending runs with no gem5 binary are counted as failures."""
    monkeypatch.setattr(rx, "MACHINES_DIR", machine_json.parent)
    (tmp_path / "bin" / "plain").mkdir(parents=True)
    (tmp_path / "bin" / "plain" / "k").write_text("")
    rc = rx.main(["--kernels", "k", "--policies", "static_c0", "--run-length", "full",
                  "--bin-root", str(tmp_path / "bin"), "--results-root", str(tmp_path / "res"),
                  "--gem5", str(tmp_path / "no-gem5"), "--no-collect"])
    assert rc == 1
    assert "gem5 binary not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# oracle_sweep.py
# ---------------------------------------------------------------------------

def test_oracle_analyze_mcpat_energy_and_missing_dirs(tmp_path, machine_json, monkeypatch):
    """McPAT energy uses each config's run power; configs without region_stats are skipped."""
    from whdata import load_machine
    m = load_machine(machine_json)
    dirs = [tmp_path / f"c{i}" for i in range(4)]
    for i, d in enumerate(dirs[:3]):
        make_region_stats(d / "region_stats.csv", i)
    calls = []

    def fake_estimate(d, machine, model, static_config):
        """Return a run power of (config + 1) W."""
        calls.append((model, static_config))
        return {"power_w": static_config + 1.0}

    monkeypatch.setattr(oracle_sweep, "estimate_run", fake_estimate)
    df, bi, _, _ = oracle_sweep.analyze("k", m, dirs, "mcpat")
    assert calls == [("mcpat", 0), ("mcpat", 1), ("mcpat", 2)]
    assert sorted(df.config.unique()) == [0, 1, 2]
    row = df[(df.config == 2) & (df.region == 0)].iloc[0]
    assert row.energy == pytest.approx(3.0 * row.cycles / 2e9)
    assert bi == {0: 2, 1: 0, 2: 1}            # c3 (region 0's optimum) was not run
    with pytest.raises(RuntimeError, match="no region_stats.csv"):
        oracle_sweep.analyze("k", m, [tmp_path / "none"])


def test_oracle_main_dry_run_and_analysis_error(tmp_path, machine_json, capsys, monkeypatch):
    """--dry-run lists every config run; --analyze-only with no runs fails per kernel."""
    (tmp_path / "bin" / "oracle").mkdir(parents=True)
    (tmp_path / "bin" / "oracle" / "k").write_text("")
    common = ["--kernels", "k", "--machine", str(machine_json), "--input", "large",
              "--run-length", "full", "--bin-root", str(tmp_path / "bin"),
              "--out-root", str(tmp_path / "oracle"), "--no-lock"]
    assert oracle_sweep.main(common + ["--dry-run", "--configs", "1,3"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and out[0].split()[:4] == ["k", "c1", "todo", "full"]
    assert "--window-initial 3" in out[1]
    monkeypatch.setattr(sys, "argv", ["oracle_sweep.py", *common, "--analyze-only"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(oracle_sweep.__file__, run_name="__main__")
    assert e.value.code == 1
    assert "[ERROR] k: k: no region_stats.csv" in capsys.readouterr().out
