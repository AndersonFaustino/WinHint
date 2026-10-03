"""Tests of oracle_sweep analysis, energy estimation and the run matrix (no gem5 needed).

Covers sim/baselines/oracle/oracle_sweep.py (analysis and output layout),
sim/estimate_energy.py (stats parsing, residency, proxy energy, McPAT XML)
and sim/run_experiments.py (dry run, collection, execution with a fake
gem5.opt and the heavy lock). ``STATS`` is a minimal stats.txt;
``FAKE_GEM5`` is a stand-in gem5 binary that writes deterministic outputs.
"""
import json

import pandas as pd

import estimate_energy as ee
import oracle_sweep
import run_experiments
from conftest import REGIONS, make_region_stats, make_trace

STATS = """
---------- Begin Simulation Statistics ----------
simSeconds                                   0.001000                       # Number of seconds simulated (Second)
simInsts                                      3000000                       # Number of instructions simulated (Count)
system.cpu.numCycles                          2000000                       # Number of cpu cycles simulated (Cycle)
system.cpu.ipc                                1.500000                       # IPC
system.cpu.dcache.overallMisses::total          20000                       # misses
system.cpu.dcache.overallAccesses::total      1000000                       # accesses
system.l2.overallMisses::total                   5000                       # misses
system.cpu.window.numSwitches                      42                       # switches
system.cpu.window.cyclesInConfig::0            500000                       # residency
system.cpu.window.cyclesInConfig::3           1500000                       # residency
---------- End Simulation Statistics   ----------
"""


def test_analyze_picks_region_optimum(tmp_path, machine_json):
    """Check analyze(): per-region IPC optimum; ED²P never prefers a larger window."""
    from whdata import load_machine
    m = load_machine(machine_json)
    dirs = [tmp_path / f"c{i}" for i in range(4)]
    for i, d in enumerate(dirs):
        make_region_stats(d / "region_stats.csv", i)
    df, bi, be, tot = oracle_sweep.analyze("k", m, dirs, "proxy")
    assert bi == {r: c for r, (c, _) in REGIONS.items()}
    # ED2P can only prefer an equal-or-smaller window than IPC here
    assert all(be[r] <= bi[r] for r in bi)
    assert tot["oracle_ipc_cycles"] <= min(v["cycles"] for v in tot["static"].values())
    assert df["best_ipc"].sum() == len(REGIONS)


def test_best_static_needs_every_region(tmp_path, machine_json):
    """A config whose region_stats.csv lacks a region never becomes best_static_config."""
    from whdata import load_machine
    m = load_machine(machine_json)
    dirs = [tmp_path / f"c{i}" for i in range(4)]
    for i, d in enumerate(dirs):
        df = make_region_stats(d / "region_stats.csv", i)
        if i == 3:                                   # c3 misses region 1: fewer cycles
            df[df.region != 1].to_csv(d / "region_stats.csv", index=False)
    _, _, _, tot = oracle_sweep.analyze("k", m, dirs)
    assert tot["complete_configs"] == [0, 1, 2]
    assert tot["static"][3]["cycles"] < min(tot["static"][c]["cycles"] for c in (0, 1, 2))
    assert tot["best_static_config"] == min((0, 1, 2), key=lambda c: tot["static"][c]["cycles"])
    # no config covers every region: no best static config, no speedup
    for i, d in enumerate(dirs):
        df = make_region_stats(d / "region_stats.csv", i)
        df[df.region == i % 3].to_csv(d / "region_stats.csv", index=False)
    _, _, _, tot = oracle_sweep.analyze("k", m, dirs)
    assert tot["complete_configs"] == []
    assert "best_static_config" not in tot and "oracle_speedup_vs_best_static" not in tot


def test_write_outputs_layout(tmp_path, machine_json):
    """Check the oracle output files (interfaces.md §5 path, summary, PGO copy for ``small``)."""
    from whdata import load_machine, load_oracle
    m = load_machine(machine_json)
    dirs = [tmp_path / "r" / f"c{i}" for i in range(4)]
    for i, d in enumerate(dirs):
        make_region_stats(d / "region_stats.csv", i)
    df, bi, be, tot = oracle_sweep.analyze("k", m, dirs)
    root = tmp_path / "oracle"
    p = oracle_sweep.write_outputs(root, "k", "riscv_ooo", "large", df, bi, be, tot, "ipc", {})
    assert json.loads(p.read_text()) == {str(r): c for r, c in bi.items()}
    assert (root / "k.json").exists()          # interfaces.md §5 path
    assert load_oracle(root / "riscv_ooo" / "large" / "k.summary.json", "ed2p") == be
    oracle_sweep.write_outputs(root, "k", "riscv_ooo", "small", df, bi, be, tot, "ipc", {})
    assert (tmp_path / "pgo" / "k.json").exists()          # sibling of the oracle root
    nested = tmp_path / "alt" / "oracle"
    oracle_sweep.write_outputs(nested, "k", "riscv_ooo", "small", df, bi, be, tot, "ipc", {})
    assert (tmp_path / "alt" / "pgo" / "k.json").exists()  # follows --out-root
    assert oracle_sweep.pgo_root(oracle_sweep.ORACLE_ROOT) == oracle_sweep.REPO / "results" / "pgo"


def test_parse_stats_and_energy(tmp_path, machine_json):
    """Check key_stats(), stats-based residency and that the proxy energy grows with the window."""
    (tmp_path / "stats.txt").write_text(STATS)
    ks = ee.key_stats(ee.parse_stats(tmp_path / "stats.txt"))
    assert ks["cycles"] == 2e6 and ks["insts"] == 3e6 and ks["win.numSwitches"] == 42
    r = ee.estimate_run(tmp_path, machine_json, model="proxy")
    assert r["residency_source"] == "stats"
    assert r["residency"] == [0.25, 0.0, 0.0, 0.75]
    assert r["energy_j"] > 0 and abs(r["ed2p"] - r["energy_j"] * 1e-6) < 1e-15
    # smaller window -> less proxy energy at equal activity
    full = ee.proxy_energy(ks, ee.load_machine(machine_json)["window_table"], [0, 0, 0, 1])
    small = ee.proxy_energy(ks, ee.load_machine(machine_json)["window_table"], [1, 0, 0, 0])
    assert small < full


def test_residency_from_trace(tmp_path, machine_json):
    """Check the window_trace.csv residency fallback when stats have no per-config cycles."""
    (tmp_path / "stats.txt").write_text(STATS.replace("cyclesInConfig", "other"))
    make_trace(tmp_path / "window_trace.csv", 2, seed=0)
    r = ee.estimate_run(tmp_path, machine_json, model="proxy")
    assert r["residency_source"] == "trace" and r["residency"][2] == 1.0


def test_mcpat_xml_has_window_sizes(tmp_path, machine_json):
    """Check that the McPAT XML carries the configuration's window sizes and no placeholders."""
    m = ee.load_machine(machine_json)
    (tmp_path / "s.txt").write_text(STATS)
    ks = ee.key_stats(ee.parse_stats(tmp_path / "s.txt"))
    xml = ee.build_mcpat_xml(ks, m, m["window_table"][1])
    assert '<param name="ROB_size" value="128"/>' in xml
    assert '<param name="load_buffer_size" value="32"/>' in xml
    assert "${" not in xml


def test_run_matrix_dry_run(tmp_path, capsys, monkeypatch, machine_json):
    """Check the --dry-run matrix (variants, MISSING, no lut_xfer) and static_c<i> commands."""
    monkeypatch.setattr(run_experiments, "MACHINES_DIR", machine_json.parent)
    b = tmp_path / "bin"
    for v in ("plain", "winhint"):
        (b / v).mkdir(parents=True)
        (b / v / "encoder_bert_tiny_infer").write_text("")
    rc = run_experiments.main(["--dry-run", "--run-length", "full", "--kernels", "encoder_bert_tiny_infer",
                               "--bin-root", str(b), "--results-root", str(tmp_path / "res"),
                               "--lut-root", str(tmp_path / "b5")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "static_c3" in out and "winhint_hw" in out and "--window-policy hybrid" in out
    assert "MISSING" in out            # e.g. jones binary / LUT absent
    assert "lut_xfer" not in out       # only on non-reference machines
    runs = run_experiments.build_runs(run_experiments.parse_args(
        ["--kernels", "encoder_bert_tiny_infer", "--policies", "static", "--run-length", "full",
         "--bin-root", str(b),
         "--results-root", str(tmp_path / "res")]))
    assert [r.variant.name for r in runs] == [f"static_c{i}" for i in range(4)]
    assert runs[2].cmd[runs[2].cmd.index("--window-initial") + 1] == "2"
    assert runs[0].cmd[runs[0].cmd.index("--options") + 1] == "large"


def test_collect_summary(tmp_path, machine_json, monkeypatch):
    """Check that collect() writes IPC, window counters and proxy energy to summary.csv."""
    monkeypatch.setattr(run_experiments, "MACHINES_DIR", machine_json.parent)
    d = tmp_path / "res" / "riscv_ooo" / "k" / "winhint"
    d.mkdir(parents=True)
    (d / "stats.txt").write_text(STATS)
    (d / "run.json").write_text(json.dumps({"machine": "riscv_ooo", "kernel": "k",
                                            "variant": "winhint", "policy": "hint",
                                            "status": "ok", "input": "large"}))
    assert run_experiments.is_done(d)
    run_experiments.collect(tmp_path / "res", "proxy", tmp_path / "res" / "summary.csv")
    s = pd.read_csv(tmp_path / "res" / "summary.csv")
    assert s.loc[0, "ipc"] == 1.5 and s.loc[0, "win.numSwitches"] == 42
    assert s.loc[0, "energy_model"] == "proxy" and s.loc[0, "ed2p"] > 0


# ---------------------------------------------------------------------------
# Runner plumbing with a fake gem5 (no simulator needed)
# ---------------------------------------------------------------------------

FAKE_GEM5 = r'''#!/usr/bin/env python3
"""Fake gem5.opt: parses the se.py flags run_experiments emits and writes
stats.txt / region_stats.csv / window_trace.csv into --outdir."""
import sys, pathlib
argv = sys.argv[1:]
out = pathlib.Path([a for a in argv if a.startswith("--outdir=")][0].split("=", 1)[1])
cfg = int(argv[argv.index("--window-initial") + 1])
out.mkdir(parents=True, exist_ok=True)
best = {0: 3, 1: 0, 2: 1}
rows, t, cyc_total = [], 0, 0
for visit in range(2):
    for r, b in best.items():
        insts = 100000
        cyc = int(insts * (1.0 + 0.05 * abs(cfg - b)))
        rows.append(f"{r},{cfg},{t},{cyc},{insts}")
        t += cyc
        cyc_total += cyc
(out / "region_stats.csv").write_text("region,config,enter_cycle,cycles,insts\n" + "\n".join(rows) + "\n")
if "--window-trace" in argv:
    tr = ["cycle,insts,ipc,rob_occ_mean,iq_occ_mean,lq_occ_mean,l1d_mpki,l2_mpki,mlp,branch_mpki,config,region"]
    c = 0
    for r in (0, 1, 2) * 10:
        c += 1000
        tr.append(f"{c},1000,1.0,50,20,10,{5 * r},1,{1 + r},0.5,{cfg},{r}")
    (out / "window_trace.csv").write_text("\n".join(tr) + "\n")
(out / "stats.txt").write_text(
    "---------- Begin Simulation Statistics ----------\n"
    f"simSeconds {cyc_total / 2e9:.9f}\nsimInsts 600000\n"
    f"system.cpu.numCycles {cyc_total}\n"
    f"system.cpu.window.cyclesInConfig::{cfg} {cyc_total}\n"
    "---------- End Simulation Statistics   ----------\n")
'''


def _fake_gem5(tmp_path):
    """Write the ``FAKE_GEM5`` script as an executable gem5.opt in ``tmp_path`` and return it."""
    g = tmp_path / "gem5.opt"
    g.write_text(FAKE_GEM5)
    g.chmod(0o755)
    return g


def test_se_policies_and_unsupported_policy(tmp_path, machine_json, monkeypatch):
    """Check that policies se.py does not accept make their variants MISSING."""
    pols = run_experiments.se_policies()
    assert pols is None or {"static", "hint", "hybrid"} <= pols
    fake_se = tmp_path / "se.py"
    fake_se.write_text('WINDOW_POLICIES = ["static", "hint"]\n')
    monkeypatch.setattr(run_experiments, "SE_PY", fake_se)
    monkeypatch.setattr(run_experiments, "MACHINES_DIR", machine_json.parent)
    b = tmp_path / "bin" / "plain"
    b.mkdir(parents=True)
    (b / "k").write_text("")
    runs = run_experiments.build_runs(run_experiments.parse_args(
        ["--kernels", "k", "--policies", "B9", "B0", "--run-length", "full",
         "--bin-root", str(tmp_path / "bin"),
         "--results-root", str(tmp_path / "res")]))
    ltp = [r for r in runs if r.variant.name == "ltp"]
    assert ltp and any("ltp" in m for m in ltp[0].missing)
    assert all(not r.missing for r in runs if r.variant.baseline == "B0")


def test_execute_with_fake_gem5_and_lock(tmp_path, machine_json, monkeypatch):
    """Run one variant end to end with the fake gem5 (flock if available); check resume."""
    import shutil
    monkeypatch.setattr(run_experiments, "MACHINES_DIR", machine_json.parent)
    monkeypatch.setattr(run_experiments, "HEAVY_LOCK", tmp_path / ".heavy.lock")
    b = tmp_path / "bin" / "plain"
    b.mkdir(parents=True)
    (b / "k").write_text("")
    args = ["--kernels", "k", "--policies", "static_c1", "--run-length", "full",
            "--bin-root", str(tmp_path / "bin"),
            "--results-root", str(tmp_path / "res"), "--gem5", str(_fake_gem5(tmp_path)),
            "--energy-model", "proxy", "--window-args", "static:foo=1"]
    if shutil.which("flock") is None:
        args.append("--no-lock")
    assert run_experiments.main(args) == 0
    d = tmp_path / "res" / "riscv_ooo" / "k" / "static_c1"
    meta = json.loads((d / "run.json").read_text())
    assert meta["status"] == "ok"
    assert "--window-args foo=1" in meta["command"]
    s = pd.read_csv(tmp_path / "res" / "summary.csv")
    assert s.loc[0, "variant"] == "static_c1" and s.loc[0, "ed2p"] > 0
    # resumable: a second invocation does nothing
    assert run_experiments.main(args) == 0
    assert json.loads((d / "run.json").read_text())["started"] == meta["started"]


def test_oracle_sweep_end_to_end_fake_gem5(tmp_path, machine_json):
    """Run oracle_sweep.main with the fake gem5 and check the oracle map and outputs."""
    b = tmp_path / "bin" / "oracle"
    b.mkdir(parents=True)
    (b / "k").write_text("")
    out = tmp_path / "oracle"
    rc = oracle_sweep.main(["--kernels", "k", "--machine", str(machine_json), "--input", "large",
                            "--run-length", "full",
                            "--bin-root", str(tmp_path / "bin"), "--gem5", str(_fake_gem5(tmp_path)),
                            "--out-root", str(out), "--no-lock"])
    assert rc == 0
    flat = json.loads((out / "k.json").read_text())       # interfaces.md §5 path
    assert flat == {"0": 3, "1": 0, "2": 1}
    summ = json.loads((out / "riscv_ooo" / "large" / "k.summary.json").read_text())
    assert set(summ["best"]) == {"ipc", "ed2p"}
    assert (out / "runs" / "riscv_ooo" / "k" / "large" / "c2" / "window_trace.csv").exists()
