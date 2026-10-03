"""Tests of analysis/plot_results.py (statistics, input parsing and figure generation)."""

import inspect
import json

import numpy as np
import pandas as pd
import pytest

import plot_results


def test_spearman_matches_definition():
    """Check spearman() on perfect, tied and random data (against scipy when available)."""
    x = np.array([1, 2, 3, 4, 5.0])
    assert plot_results.spearman(x, x * 3) == pytest.approx(1.0)
    assert plot_results.spearman(x, -x) == pytest.approx(-1.0)
    assert abs(plot_results.spearman([1, 2, 2, 3], [1, 3, 2, 4]) - 0.9486832980505138) < 1e-9
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 5, 40), rng.integers(0, 5, 40)   # with ties
    assert plot_results.spearman(a, b) == pytest.approx(stats.spearmanr(a, b)[0])


def test_read_regions_json_variants(tmp_path):
    """Check that read_regions_json() prefers nest_w_star and accepts a region list."""
    p = tmp_path / "k.regions.json"
    p.write_text(json.dumps({"regions": {"0": {"w_star": 100, "nest_w_star": 120.5, "config": 1},
                                         "1": {"w_star": 64}}}))
    r = plot_results.read_regions_json(p)
    assert r[0]["w_pred"] == 120.5 and r[1]["w_pred"] == 64
    p.write_text(json.dumps([{"id": 3, "w_star": 7}]))
    assert plot_results.read_regions_json(p)[3]["w_pred"] == 7


def test_all_figures_from_synthetic_data(tmp_path):
    """Check that --synthetic writes every gem5 figure and plausible metrics."""
    assert plot_results.main(["--synthetic", "--out-dir", str(tmp_path), "--format", "png"]) == 0
    for f in ("wstar_vs_oracle", "ipc_bars", "ed2p_bars", "switch_frequency", "hint_overhead",
              "sensitivity", "portability"):
        assert (tmp_path / f"{f}.png").stat().st_size > 1000, f
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["wstar"]["spearman_rho"] > 0.8
    assert m["ipc_bars"]["geomean"]["B0-large"] == 1.0
    assert {r["v"] for r in m["portability"]} >= {"lut", "lut_xfer", "winhint"}


def test_fig_bars_plots_raw_ratios_for_both_metric_directions(tmp_path):
    """IPC (higher better) and ED2P (lower better) bars are both plain ratios to B0-large;
    fig_bars takes no higher_better flag (the direction is stated in the y label)."""
    assert "higher_better" not in inspect.signature(plot_results.fig_bars).parameters
    rows = [dict(machine="m", kernel=k, v=v, ipc=i, ed2p=e)
            for k in ("a", "b")
            for v, i, e in (("static_max", 2.0, 4.0), ("winhint", 3.0, 2.0))]
    df = pd.DataFrame(rows)
    metrics = {}
    plot_results.fig_bars(df, tmp_path, "png", "ipc", "IPC / B0-large", "ipc_bars", "m", metrics)
    plot_results.fig_bars(df, tmp_path, "png", "ed2p", "ED2P / B0-large (lower is better)",
                          "ed2p_bars", "m", metrics)
    assert metrics["ipc_bars"]["geomean"]["WinHint"] == pytest.approx(1.5)
    assert metrics["ed2p_bars"]["geomean"]["WinHint"] == pytest.approx(0.5)
    assert (tmp_path / "ed2p_bars.png").is_file()

def test_hw_and_b9_figures(tmp_path):
    """Check the B9 label, switch counter detection and the real-silicon figures/metrics."""
    assert plot_results.main(["--synthetic", "--out-dir", str(tmp_path), "--format", "png"]) == 0
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert "B9 LTP" in m["ipc_bars"]["geomean"]
    assert m["switch_frequency"]["counter"] == "win.switches"
    assert set(m["hw_edp"]) >= {"metric", "smt-on", "smt-off"}
    assert m["hw_edp"]["metric"] == "edp_pkg_js"
    assert m["hw_edp"]["smt-on"]["R1 stock"] == 1.0
    assert m["hw_edp"]["smt-on"]["WinHint"] < 1.0
    for f in ("hw_edp", "hw_nop_overhead"):
        assert (tmp_path / f"{f}.png").stat().st_size > 1000, f
    assert abs(m["hw_nop_overhead"]["P"]["decoder_gpt2_infer"] - 0.25) < 1e-6


def test_hw_edp_falls_back_to_time_without_energy(tmp_path):
    """Check that fig_hw_edp() uses wall time when no energy column has data."""
    import pandas as pd
    rows = [{"kernel": "phase_workload", "config": c, "smt": "on", "n": 1,
             "wall_s_mean": t, "wall_s_ci_lo": float("nan"), "wall_s_ci_hi": float("nan"),
             "edp_pkg_js_mean": float("nan")}
            for c, t in (("R1", 1.0), ("R0-P", 0.8), ("R0-E", 1.4), ("WH", 0.95))]
    p = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    metrics = {}
    plot_results.fig_hw_edp(plot_results.load_hw_summary(p), tmp_path, "png", metrics)
    assert metrics["hw_edp"]["metric"] == "wall_s"
    assert list(metrics["hw_edp"]["smt-on"]) == ["R0 P-only", "R0 E-only", "R1 stock", "WinHint"]


def test_regions_json_machine_selection(tmp_path):
    """Check find_regions_json() machine-directory precedence and target filtering."""
    d = tmp_path / "winhint"
    (d / "m2").mkdir(parents=True)
    (d / "k.regions.json").write_text(json.dumps({"target": "m1", "regions": {"0": {"w_star": 1}}}))
    (d / "m2" / "k.regions.json").write_text(json.dumps({"target": "m2", "regions": {}}))
    assert plot_results.find_regions_json([d], "k", "m1") == d / "k.regions.json"
    assert plot_results.find_regions_json([d], "k", "m2") == d / "m2" / "k.regions.json"
    assert plot_results.find_regions_json([d], "k", "m3") is None


def test_compiler_schema_winhint_stats_1(tmp_path):
    """Files as written by compiler/winhint (docs/reference/stats-schema.md)."""
    d = tmp_path / "x86" / "winhint"
    d.mkdir(parents=True)
    (d / "k.regions.json").write_text(json.dumps({
        "kernel": "k", "target": "riscv_ooo", "num_regions": 2, "overflow": False,
        "regions": {"0": {"function": "f", "header": "for.cond", "line": 3, "encoded_id": 0,
                          "w_star": 256, "nest_w_star": 143.2, "config": 3, "nest_config": 2,
                          "L_mem": 220.0, "D_indep": 53.3, "CP": 14.0, "footprint_bytes": None,
                          "dyn_insts_est": 1.2e7, "conservative": False},
                    "1": {"function": "g", "line": 9, "w_star": 64, "nest_w_star": None,
                          "config": 0, "nest_config": 0}}}))
    (d / "k.winhint.json").write_text(json.dumps({
        "schema": "winhint-stats/1", "kernel": "k", "hints_setwin": 4, "hints_region": 0,
        "compile_time_ms": 1.5, "analysis_time_ms": 1.0, "regions": [], "hints": [], "loops": []}))
    r = plot_results.read_regions_json(d / "k.regions.json")
    assert r[0]["w_pred"] == 143.2 and r[0]["config_pred"] == 2
    assert r[1]["w_pred"] == 64          # null nest_w_star falls back to w_star
    assert plot_results.find_regions_json([d], "k", "riscv_ooo") == d / "k.regions.json"
    cs = plot_results.compiler_stats([d])
    assert cs["k"]["hints_setwin"] == 4 and cs["k"]["compile_time_ms"] == 1.5
    assert plot_results.static_hint_counts([d]) == {"k": 4}


def test_code_size_from_real_binaries(tmp_path):
    """Executable-section bytes of hinted builds vs plain (real benchmark ELFs
    when they exist; RISC-V hints are 4 bytes each)."""
    root = plot_results.BUILD / "benchmarks" / "riscv"
    plain = sorted(p for p in (root / "plain").glob("*_infer") if p.is_file()) \
        if (root / "plain").is_dir() else []
    if not plain or not (root / "winhint" / plain[0].name).is_file():
        pytest.skip("no plain + winhint benchmark builds")
    n = plot_results.elf_exec_bytes(plain[0])
    assert n and n > 0
    t = plot_results.code_size_table(root)
    w = t[t.v == "winhint"]
    assert len(w) and (w.plain_bytes > 0).all()
    assert plot_results.elf_exec_bytes(tmp_path) is None
    (tmp_path / "x").write_text("not an elf")
    assert plot_results.elf_exec_bytes(tmp_path / "x") is None
