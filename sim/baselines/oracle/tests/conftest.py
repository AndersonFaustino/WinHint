"""Phase E pipeline tests (no gem5): a fake gem5.opt writes the interfaces.md §4 outputs."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for p in (REPO / "sim", REPO / "sim" / "baselines" / "lut", REPO / "sim" / "baselines" / "oracle",
          REPO / "analysis"):
    sys.path.insert(0, str(p))

#: Window table (parallel lists) of the synthetic machines.
TABLE = {"rob": [64, 128, 192, 256], "iq": [32, 64, 96, 128], "lq": [16, 32, 48, 64],
         "sq": [16, 32, 48, 64]}

#: region -> oracle-best config; the fake gem5 makes cycles minimal there
BEST = {0: 3, 1: 0, 2: 1}

#: Source of the fake gem5.opt: writes stats.txt, region_stats.csv and window_trace.csv
#: whose cycles are minimal at ``BEST`` (hint/hybrid/lut policies switch to it per region).
FAKE_GEM5 = r'''#!/usr/bin/env python3
"""Fake gem5.opt: parses the se.py flags the runners emit and writes stats.txt,
region_stats.csv and (with --window-trace) window_trace.csv into --outdir."""
import json, pathlib, sys
argv = sys.argv[1:]
out = pathlib.Path([a for a in argv if a.startswith("--outdir=")][0].split("=", 1)[1])
cfg = int(argv[argv.index("--window-initial") + 1])
pol = argv[argv.index("--window-policy") + 1]
machine = json.loads(pathlib.Path(argv[argv.index("--machine") + 1]).read_text())
l2 = machine["cache"]["l2"]["size"]
slow = 1.0 + (0.2 if l2 == "256kB" else 0.0)
out.mkdir(parents=True, exist_ok=True)
best = {0: 3, 1: 0, 2: 1}
feat = {0: (0.6, 200, 40, 6), 1: (2.8, 40, 0.5, 1), 2: (1.6, 90, 8, 2)}
rows, tr, t, cyc_total, switches = [], [], 0, 0, 0
c = cfg
for visit in range(2):
    for r, b in best.items():
        if pol in ("hint", "hybrid", "lut"):
            switches += c != b
            c = b
        insts = 100000
        # whole periods: the real controller's trace is contiguous (cumulative
        # end-of-period cycle, per-period insts)
        cyc = 1000 * round(slow * insts * (1.0 + 0.05 * abs(c - b)) / 1000)
        rows.append(f"{r},{c},{t},{cyc},{insts}")
        ipc, rob, mpki, mlp = feat[r]
        for _ in range(cyc // 1000):
            tr.append(f"{t + 1000},{int(ipc * 1000)},{ipc},{rob},{rob / 2},{rob / 4},{mpki},{mpki / 4},{mlp},0.5,{c},{r}")
            t += 1000
        cyc_total += cyc
(out / "region_stats.csv").write_text("region,config,enter_cycle,cycles,insts\n" + "\n".join(rows) + "\n")
if "--window-trace" in argv:
    (out / "window_trace.csv").write_text("cycle,insts,ipc,rob_occ_mean,iq_occ_mean,lq_occ_mean,"
                                          "l1d_mpki,l2_mpki,mlp,branch_mpki,config,region\n"
                                          + "\n".join(tr) + "\n")
hints = 6 if "winhint" in argv[argv.index("--cmd") + 1] else 0
(out / "stats.txt").write_text(
    "---------- Begin Simulation Statistics ----------\n"
    f"simSeconds {cyc_total / 2e9:.9f}\nsimInsts 600000\n"
    f"system.cpu.numCycles {cyc_total + hints}\n"
    f"system.cpu.window.switches {switches}\n"
    f"system.cpu.window.setwinHints {hints}\n"
    f"system.cpu.window.cyclesInConfig::{cfg} {cyc_total}\n"
    f"system.cpu.window.robFullCycles::0 {10 * cyc_total}\n"
    "---------- End Simulation Statistics   ----------\n")
'''


@pytest.fixture
def machines_dir(tmp_path):
    """Write riscv_ooo (1MB L2) and riscv_ooo_small (256kB L2) JSONs; return their directory."""
    d = tmp_path / "machines"
    d.mkdir()
    for name, l2 in (("riscv_ooo", "1MB"), ("riscv_ooo_small", "256kB")):
        (d / f"{name}.json").write_text(json.dumps({
            "name": name, "cpu": {"clock": "2GHz", "issue_width": 4},
            "cache": {"l2": {"size": l2, "assoc": 16}, "l2_size": l2},
            "memory": {"latency_cycles": 200, "extra_latency_ns": 0}, "window": TABLE}))
    return d


@pytest.fixture
def fake_gem5(tmp_path):
    """Install the ``FAKE_GEM5`` script as an executable ``gem5.opt`` and return its path."""
    g = tmp_path / "gem5.opt"
    g.write_text(FAKE_GEM5)
    g.chmod(0o755)
    return g


@pytest.fixture(autouse=True)
def _hermetic_run_experiments(tmp_path, monkeypatch):
    """No tuned.json, region profile or checkpoint of the real tree leaks in."""
    import run_experiments
    monkeypatch.setattr(run_experiments, "TUNED_DEFAULT", tmp_path / "no-tuned.json")
    monkeypatch.setattr(run_experiments, "RUNLEN_ROOT", tmp_path / "runlen")
    monkeypatch.setattr(run_experiments, "CKPT_ROOT", tmp_path / "ckpt")
