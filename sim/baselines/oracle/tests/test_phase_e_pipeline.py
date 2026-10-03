"""Phase E end to end with a fake gem5.

Covers B1 oracle sweep -> B5 LUT per machine -> experiment matrix (incl.
sensitivity points) -> summary/energy -> figures, plus the trace-format and
se.py command-line contracts.
"""
import json

import numpy as np
import pandas as pd
import pytest

import build_lut
import estimate_energy as ee
import oracle_sweep
import plot_results
import run_experiments as rx
from conftest import BEST
from whdata import load_window_trace, read_lut


def _bins(root, kernels=("ka", "kb"), variants=("plain", "oracle", "winhint")):
    """Create empty kernel binaries (and winhint sidecars targeting riscv_ooo) under ``root``."""
    for v in variants:
        (root / v).mkdir(parents=True, exist_ok=True)
        for k in kernels:
            (root / v / k).write_text("")
            if v == "winhint":
                (root / v / f"{k}.winhint.json").write_text(json.dumps({"target": "riscv_ooo"}))
    return root


@pytest.fixture
def env(tmp_path, machines_dir, fake_gem5, monkeypatch):
    """Return paths of a hermetic environment: fake binaries, gem5, machines and output roots."""
    monkeypatch.setattr(rx, "MACHINES_DIR", machines_dir)
    return {"bin": _bins(tmp_path / "bin"), "oracle": tmp_path / "oracle", "b5": tmp_path / "b5",
            "res": tmp_path / "res", "gem5": fake_gem5, "machines": machines_dir}


def _sweep(env, machine, inp):
    """Run the oracle sweep of kernels ka/kb on ``machine``/``inp``; return the exit code."""
    return oracle_sweep.main(["--kernels", "ka", "kb", "--machine", str(env["machines"] / f"{machine}.json"),
                              "--input", inp, "--run-length", "full",
                              "--bin-root", str(env["bin"]), "--gem5", str(env["gem5"]),
                              "--out-root", str(env["oracle"]), "--no-lock"])


def test_oracle_then_b5_lut_per_machine(env):
    """The oracle recovers ``BEST`` on both machines and each machine's B5 LUT reproduces it."""
    for m in ("riscv_ooo", "riscv_ooo_small"):
        assert _sweep(env, m, "small") == 0
        flat = json.loads((env["oracle"] / m / "small" / "ka.json").read_text())
        assert {int(k): v for k, v in flat.items()} == BEST
    # per-machine LUT, trained on the oracle-labelled windows of the small input
    assert build_lut.main(["--machines-dir", str(env["machines"]), "--runs-root",
                           str(env["oracle"] / "runs"), "--oracle-root", str(env["oracle"]),
                           "--out-root", str(env["b5"]), "--model", "tree", "--split", "random",
                           "--no-lock"]) == 0
    for m in ("riscv_ooo", "riscv_ooo_small"):
        lut = read_lut(env["b5"] / m / "lut.txt")
        assert lut.features == ["ipc", "rob_occ", "l1d_mpki", "mlp"]
        # region feature centres (fake gem5) map to the oracle configuration
        for r, x in {0: (0.6, 200, 40, 6), 1: (2.8, 40, 0.5, 1), 2: (1.6, 90, 8, 2)}.items():
            assert lut.lookup(np.array([x]))[0] == BEST[r]
        chk = json.loads((env["b5"] / m / "lut.check.json").read_text())
        assert chk["lut_vs_oracle"] == 1.0


def test_real_trace_format_per_period_insts(env):
    """Real-format traces (cumulative cycle, per-period insts) are not differenced in insts."""
    assert _sweep(env, "riscv_ooo", "small") == 0
    tr = env["oracle"] / "runs" / "riscv_ooo" / "ka" / "small" / "c0" / "window_trace.csv"
    df = load_window_trace(tr)
    raw = pd.read_csv(tr)
    assert np.allclose(df["d_cycles"], 1000)
    assert (df["d_insts"] == raw["insts"]).all()      # per-period, not differenced


def test_trace_short_increasing_per_period_insts_not_differenced(tmp_path):
    """Short per-period insts are kept when ipc confirms them; cumulative ones are differenced."""
    p = tmp_path / "t.csv"
    p.write_text("cycle,insts,ipc,config,region\n1000,1000,1.0,0,0\n2000,1500,1.5,0,0\n"
                 "3000,2000,2.0,0,0\n")
    df = load_window_trace(p)
    assert df["d_insts"].tolist() == [1000, 1500, 2000]
    p.write_text("cycle,insts,config,region\n1000,1000,0,0\n2000,2500,0,0\n3000,4500,0,0\n")
    assert load_window_trace(p)["d_insts"].tolist() == [1000, 1500, 2000]   # cumulative


def test_machine_dependent_binary_lookup(tmp_path):
    """Machine-dependent variants need a per-machine build; others fall back to the default."""
    b = _bins(tmp_path / "bin", kernels=("k",))
    # default-machine build (sidecar target riscv_ooo) is not used for another machine
    assert rx.find_binary(b, "winhint", "riscv_ooo", "k") == b / "winhint" / "k"
    assert rx.find_binary(b, "winhint", "riscv_ooo_small", "k") is None
    (b / "winhint" / "riscv_ooo_small").mkdir()
    (b / "winhint" / "riscv_ooo_small" / "k").write_text("")
    assert rx.find_binary(b, "winhint", "riscv_ooo_small", "k") == b / "winhint" / "riscv_ooo_small" / "k"
    # machine-independent variants fall back to the default build
    assert rx.find_binary(b, "plain", "riscv_ooo_small", "k") == b / "plain" / "k"


def test_derived_machine_sweep_point():
    """Derived sensitivity machines update latency/L2 fields without mutating the input."""
    raw = {"name": "m", "cpu": {"clock": "2GHz"}, "cache": {"l2": {"size": "1MB"}, "l2_size": "1MB"},
           "memory": {"latency_cycles": 200, "extra_latency_ns": 0}}
    d = rx.derive_machine(raw, "mem_extra_latency_ns", "40")
    assert d["memory"]["extra_latency_ns"] == 40 and d["memory"]["latency_cycles"] == 280
    assert d["name"] == "m+mem_extra_latency_ns=40" and raw["memory"]["latency_cycles"] == 200
    d = rx.derive_machine(raw, "l2_size", "256kB")
    assert d["cache"]["l2"]["size"] == "256kB" and d["cache"]["l2_size"] == "256kB"
    assert rx.machine_point_params(d)["l2_kb"] == 256
    with pytest.raises(SystemExit):
        rx.parse_sweeps(["rob=1,2"])


def test_residency_ignores_other_per_config_vectors(tmp_path):
    """Window residency reads only ``cyclesInConfig`` vectors from stats."""
    st = {"system.cpu.window.cyclesInConfig::1": 300.0, "system.cpu.window.cyclesInConfig::3": 100.0,
          "system.cpu.window.robFullCycles::0": 9e9, "system.cpu.window.iqFullCycles::2": 9e9}
    res, src = ee.window_residency(tmp_path, st, 4)
    assert src == "stats" and res == [0.0, 0.75, 0.0, 0.25]


def test_matrix_sweep_collect_and_figures(env, tmp_path):
    """Full matrix: sweep, LUTs, experiments with a sensitivity point, summary and figures."""
    for m in ("riscv_ooo", "riscv_ooo_small"):
        assert _sweep(env, m, "small") == 0
    assert build_lut.main(["--machines-dir", str(env["machines"]), "--runs-root",
                           str(env["oracle"] / "runs"), "--oracle-root", str(env["oracle"]),
                           "--out-root", str(env["b5"]), "--model", "bins", "--split", "none",
                           "--no-lock"]) == 0
    args = ["--kernels", "ka", "kb", "--policies", "static", "winhint", "winhint_nop", "B5",
            "--bin-root", str(env["bin"]), "--results-root", str(env["res"]),
            "--lut-root", str(env["b5"]), "--gem5", str(env["gem5"]), "--energy-model", "proxy",
            "--sweep", "l2_size=256kB", "--no-lock", "--run-length", "full"]
    assert rx.main(args + ["--dry-run"]) == 0
    assert not env["res"].exists()                       # dry run has no side effects
    assert rx.main(args) == 0
    s = pd.read_csv(env["res"] / "summary.csv")
    ok = s[s.status == "ok"]
    # every B-variant ran on the reference machine; hinted ones only where a build exists
    ref = set(ok[ok.machine == "riscv_ooo"].variant)
    assert {"static_c0", "static_c3", "winhint", "winhint_hw", "winhint_nop", "lut"} <= ref
    small = set(ok[ok.machine == "riscv_ooo_small"].variant)
    assert {"lut", "lut_xfer"} <= small and "winhint" not in small
    pt = ok[ok.machine == "riscv_ooo+l2_size=256kB"]
    assert len(pt) and (pt.sens_param == "l2_size").all() and (pt.l2_kb == 256).all()
    assert (pt.base_machine == "riscv_ooo").all()
    assert (env["res"] / "riscv_ooo+l2_size=256kB" / "machine.json").is_file()
    assert (ok.ed2p > 0).all() and (ok.energy_model == "proxy").all()
    # the winhint binary under static policy pays exactly the 6 fake hint cycles
    r = ok[(ok.machine == "riscv_ooo") & (ok.kernel == "ka")].set_index("variant")
    assert r.loc["winhint_nop", "cycles"] - r.loc["static_c3", "cycles"] == 6
    assert r.loc["winhint", "ipc"] > r.loc["static_c3", "ipc"]
    # figures from the real summary + oracle maps
    comp = tmp_path / "comp"
    comp.mkdir()
    for k in ("ka", "kb"):
        (comp / f"{k}.regions.json").write_text(json.dumps({"target": "riscv_ooo", "regions": {
            "0": {"nest_w_star": 240.0}, "1": {"nest_w_star": 40.0}, "2": {"nest_w_star": 100.0}}}))
        (comp / f"{k}.winhint.json").write_text(json.dumps({"hints_setwin": 3}))
    # B1 oracle on the large input too (W* vs oracle uses large)
    assert _sweep(env, "riscv_ooo", "large") == 0
    figs = tmp_path / "figs"
    assert plot_results.main(["--summary", str(env["res"] / "summary.csv"), "--oracle-root",
                              str(env["oracle"]), "--compiler-dir", str(comp), "--machines-dir",
                              str(env["machines"]), "--hw-summary", str(tmp_path / "none.csv"),
                              "--out-dir", str(figs), "--format", "png"]) == 0
    met = json.loads((figs / "metrics.json").read_text())
    assert met["wstar"]["spearman_rho"] == pytest.approx(1.0)
    assert met["wstar"]["n_regions"] == 6
    assert met["sensitivity"]["source"] == "sweep"
    assert met["hint_overhead"]["dynamic_overhead_pct_geomean"] > 0
    assert "WinHint (hints ignored)" not in met["ipc_bars"]["geomean"]
    for f in ("wstar_vs_oracle", "ipc_bars", "ed2p_bars", "switch_frequency", "hint_overhead",
              "sensitivity", "portability"):
        assert (figs / f"{f}.png").stat().st_size > 1000, f


def _import_real_se(monkeypatch):
    """Import sim/se.py with a stub ``m5`` package, so its real argparse can be used."""
    import importlib.util
    import types
    m5 = types.ModuleType("m5")
    objs = types.ModuleType("m5.objects")
    objs.__getattr__ = lambda name: type(name, (), {"_params": {"window_policy": None}})
    m5.objects = objs
    monkeypatch.setitem(__import__("sys").modules, "m5", m5)
    monkeypatch.setitem(__import__("sys").modules, "m5.objects", objs)
    spec = importlib.util.spec_from_file_location("se_real", rx.SE_PY)
    se = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(se)
    return se


def test_every_matrix_command_parses_with_real_se_py(tmp_path, monkeypatch):
    """Every se.py argument list run_experiments builds is accepted by se.py.

    Covers all variants, the 3 real machines and sensitivity points; each list
    parses with sim/se.py's own parser and maps to the intended
    policy/config/LUT, and every variant's policy is a se.py policy.
    """
    se = _import_real_se(monkeypatch)
    lut_root = tmp_path / "b5"
    for m in ("riscv_ooo", "riscv_ooo_small", "riscv_ooo_big"):
        (lut_root / m).mkdir(parents=True)
        (lut_root / m / "lut.txt").write_text("")
    res = tmp_path / "res"
    a = rx.parse_args(["--kernels", "k", "--run-length", "full",
                       "--results-root", str(res), "--lut-root", str(lut_root),
                       "--bin-root", str(tmp_path / "nobin"), "--period", "500",
                       "--sweep", "l2_size=256kB", "--sweep", "mem_extra_latency_ns=40",
                       "--window-args", "mlp:mlp_thr=2.0"])
    runs = rx.build_runs(a)
    names = {r.variant.name for r in runs}
    assert {"static_c0", "static_c3", "oracle_hinted", "occupancy", "mlp", "bbv", "lut",
            "lut_xfer", "jones", "jones_full", "pgo", "clairvoyance", "winhint_clairvoyance",
            "ltp", "winhint", "winhint_hw", "winhint_nop"} <= names
    assert {r.machine for r in runs} >= {"riscv_ooo", "riscv_ooo_small", "riscv_ooo_big",
                                         "riscv_ooo+l2_size=256kB",
                                         "riscv_ooo+mem_extra_latency_ns=40"}
    for r in runs:
        assert r.variant.policy in se.WINDOW_POLICIES
        i = r.cmd.index(str(rx.SE_PY))
        if r.machine_data is not None:      # derived machine JSON is written at run time
            r.machine_json.parent.mkdir(parents=True, exist_ok=True)
            r.machine_json.write_text(json.dumps(r.machine_data))
        args = se.parse_args(r.cmd[i + 1:])
        assert args.window_policy == r.variant.policy
        n = len(args.machine_cfg["window"]["rob"])
        want = r.variant.initial if r.variant.initial is not None else n - 1
        assert args.window_initial == want and args.window_period == 500
        assert args.options == "large"
        if r.variant.lut:
            assert args.window_lut.endswith("lut.txt")
        if r.variant.name == "jones":
            assert args.window_args == "structs=iq"
        if r.variant.policy == "mlp":
            assert args.window_args == "mlp_thr=2.0"
        if r.machine.endswith("l2_size=256kB"):
            assert args.l2_size == "256kB"
        if r.machine.endswith("mem_extra_latency_ns=40"):
            assert args.mem_extra_latency_ns == 40
