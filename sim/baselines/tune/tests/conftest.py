"""Fixtures for the baseline-tuning tests: a fake gem5 and a machine dir."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for p in (REPO / "sim", REPO / "sim" / "baselines" / "lut", REPO / "sim" / "baselines" / "tune"):
    sys.path.insert(0, str(p))

#: Window table (parallel lists) of the synthetic machine.
TABLE = {"rob": [64, 128, 192, 256], "iq": [32, 64, 96, 128], "lq": [16, 32, 48, 64],
         "sq": [16, 32, 48, 64]}

#: cost model of the fake gem5: the best point of each knob is the one listed here
FAKE_GEM5 = r'''#!/usr/bin/env python3
"""Fake gem5.opt: cycles depend on --window-args and on the binary path."""
import pathlib, sys
argv = sys.argv[1:]
out = pathlib.Path([a for a in argv if a.startswith("--outdir=")][0].split("=", 1)[1])
out.mkdir(parents=True, exist_ok=True)
wa = argv[argv.index("--window-args") + 1] if "--window-args" in argv else ""
kv = dict(x.split("=", 1) for x in wa.split(",") if "=" in x)
binary = argv[argv.index("--cmd") + 1]
f = 1.0
f += 0.1 * abs(float(kv.get("mlp_thr", 2.0)) - 2.0)                 # B3 best: mlp_thr=2.0
f += 0.1 * abs(float(kv.get("miss_mpki", 0.5)) - 0.5)               # hybrid best: miss_mpki=0.5
if "switch_cost=500-hysteresis=0.1" in binary: f -= 0.2             # WinHint best point
if "-cv_spec-u4-i2" in binary: f -= 0.3                             # B8 best point
cyc = int(1e6 * f)
(out / "stats.txt").write_text(
    "---------- Begin Simulation Statistics ----------\n"
    f"simSeconds {cyc / 2e9:.9f}\nsimInsts 1000000\nsystem.cpu.numCycles {cyc}\n"
    f"system.cpu.window.cyclesInConfig::3 {cyc}\n"
    "---------- End Simulation Statistics   ----------\n")
'''


@pytest.fixture
def machines_dir(tmp_path):
    """Write a riscv_ooo machine JSON with the 4-config ``TABLE`` and return its directory."""
    d = tmp_path / "machines"
    d.mkdir()
    (d / "riscv_ooo.json").write_text(json.dumps({
        "name": "riscv_ooo", "cpu": {"clock": "2GHz", "issue_width": 4},
        "cache": {"l2": {"size": "1MB", "assoc": 16}}, "memory": {"latency_cycles": 200},
        "window": TABLE}))
    return d


@pytest.fixture
def fake_gem5(tmp_path):
    """Install the ``FAKE_GEM5`` script as an executable ``gem5.opt`` and return its path."""
    g = tmp_path / "gem5.opt"
    g.write_text(FAKE_GEM5)
    g.chmod(0o755)
    return g


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    """Keep the real tuned.json, run-length profiles and checkpoints out of the tests."""
    import run_experiments
    monkeypatch.setattr(run_experiments, "TUNED_DEFAULT", tmp_path / "no-tuned.json")
    monkeypatch.setattr(run_experiments, "RUNLEN_ROOT", tmp_path / "runlen")
    monkeypatch.setattr(run_experiments, "CKPT_ROOT", tmp_path / "ckpt")
