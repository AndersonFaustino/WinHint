"""Fidelity-harness test fixtures (no gem5).

A fake gem5.opt writes the interfaces.md §4 outputs (stats.txt,
window_trace.csv, region_stats.csv, bbv_phases.csv) with behaviour that
reproduces every paper trend, so the whole pipeline can PASS.
``FAKE_MODE=bad_mlp`` makes the mlp policy enlarge on micro_chase.
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO / "sim", REPO / "sim" / "baselines" / "lut", REPO / "sim" / "fidelity"):
    sys.path.insert(0, str(p))

#: Window table (parallel lists) of the synthetic machine.
TABLE = {"rob": [64, 128, 192, 256], "iq": [32, 64, 96, 128], "lq": [16, 32, 48, 64],
         "sq": [16, 32, 48, 64]}

#: Source of the fake gem5.opt (behaviour per kernel/policy, see the module docstring).
FAKE_GEM5 = r'''#!/usr/bin/env python3
"""Fake gem5.opt for sim/fidelity: parses the se.py flags and writes the outputs."""
import json, os, pathlib, sys
argv = sys.argv[1:]
def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default
out = pathlib.Path([a for a in argv if a.startswith("--outdir=")][0].split("=", 1)[1])
out.mkdir(parents=True, exist_ok=True)
binary = pathlib.Path(opt("--cmd"))
kernel, variant = binary.name, binary.parent.name
inp = opt("--options", "large")
pol = opt("--window-policy")
init = int(opt("--window-initial"))
wargs = opt("--window-args", "")
if pol == "lut":
    assert pathlib.Path(opt("--window-lut")).read_text().startswith("WINHINT_LUT 1")
mode = os.environ.get("FAKE_MODE", "good")
N = 4
# static IPC per config
IPC = {"micro_gather": [0.40, 0.70, 0.90, 1.00], "micro_chase": [0.10, 0.10, 0.10, 0.10],
       "micro_compute": [2.84, 2.85, 2.86, 2.86], "micro_lowilp": [0.90, 0.90, 0.90, 0.90]}
OCC = {"micro_gather": 0.9, "micro_chase": 0.9, "micro_compute": 0.5, "micro_lowilp": 0.12}
# phased: region -> (per-config IPC, features ipc-independent: rob_occ, l1d_mpki, mlp)
REG = {3: ([0.40, 0.70, 0.90, 1.00], 200, 40.0, 8.0),
       4: ([2.84, 2.85, 2.86, 2.86], 60, 0.5, 1.0),
       5: ([0.90, 0.90, 0.90, 0.90], 20, 0.1, 1.0)}
BEST = {3: 3, 4: 0, 5: 0}
lines, regions, tr = [], [], []
res = [0.0] * N
cycles_total, insts_total, occ_sum = 0.0, 0.0, [0.0] * N

def account(c, insts, ipc, occ, region=-1, feat=(0, 0, 0)):
    global cycles_total, insts_total
    cyc = insts / ipc
    res[c] += cyc
    occ_sum[c] += cyc * occ
    cycles_total += cyc
    insts_total += insts
    periods = max(1, int(round(cyc / 1000)))
    for _ in range(periods):
        tr.append((1000.0, insts / periods, ipc, feat[0], feat[1], feat[2], c, region))
    return cyc

if kernel != "micro_phased":
    ipc_tab = IPC[kernel]
    plan = [(init, 1.0)]  # (config, fraction of insts)
    ipc_over = None
    if pol == "mlp":
        plan = [(3, 0.9), (0, 0.1)] if kernel == "micro_gather" else [(0, 1.0)]
        if mode == "bad_mlp" and kernel == "micro_chase":
            plan = [(3, 1.0)]
    elif pol == "occupancy":
        plan = [(0, 0.9), (3, 0.1)] if kernel == "micro_lowilp" else [(3, 0.85), (0, 0.15)]
    elif pol == "hint" and variant == "jones":
        plan = [(0, 1.0)] if kernel != "micro_gather" else [(3, 1.0)]
        ipc_over = IPC[kernel][3] * (0.99 if kernel != "micro_gather" else 1.0)
    elif pol == "hint" and "structs=iq+lq+sq" in wargs:
        ipc_over = {"micro_gather": 0.5, "micro_compute": 2.80}.get(kernel, ipc_tab[0])
    elif pol == "ltp":
        ipc_over = {"micro_gather": 0.85, "micro_compute": 2.79}.get(kernel, ipc_tab[0])
    for c, f in plan:
        account(c, 1e6 * f, ipc_over or ipc_tab[c], OCC[kernel] * [64, 128, 192, 256][c])
else:
    rounds = 6 if inp == "large" else 3
    for r in range(rounds):
        for reg, (ipcs, rob, mpki, mlp) in REG.items():
            if pol == "static":
                c = init
            elif pol == "bbv":
                c = 3 if r == 0 else BEST[reg]
            else:  # lut / pgo hints: the region's best
                c = BEST[reg]
            start, insts = cycles_total, 5e5 * (1 if inp == "large" else 0.8)
            cyc = account(c, insts, ipcs[c], rob, reg,
                          (rob * [64, 128, 192, 256][c] / 256, mpki, mlp))
            regions.append((reg, c, int(start), int(cyc), int(insts)))
    (out / "region_stats.csv").write_text("region,config,enter_cycle,cycles,insts\n" + "".join(
        f"{a},{b},{s},{cy},{i}\n" for a, b, s, cy, i in regions))
    if pol == "bbv":
        n_int = int(insts_total / 1e5)
        (out / "bbv_phases.csv").write_text(
            f"# intervals={n_int} predicted_correct={int(0.9 * n_int)} phases_allocated=4\n"
            "phase,visits,learned_config,ipc_c0,n_c0,ipc_c1,n_c1,ipc_c2,n_c2,ipc_c3,n_c3\n"
            f"0,{int(0.33 * n_int)},3,0.4,1,0.7,1,0.9,1,1.0,10\n"
            f"1,{int(0.33 * n_int)},0,2.8,10,2.85,1,2.86,1,2.86,1\n"
            f"2,{int(0.32 * n_int)},0,0.9,10,0.9,1,0.9,1,0.9,1\n"
            f"3,{max(1, int(0.02 * n_int))},3,0,0,0,0,0,0,1.0,1\n")
if "--window-trace" in argv:
    cyc, rows = 0.0, []
    for d, ins, ipc, rob, mpki, mlp, c, reg in tr:
        cyc += d
        rows.append(f"{int(cyc)},{int(ins)},{ipc:.4f},{rob},{rob / 2},{rob / 4},{mpki},{mpki / 4},{mlp},0.5,{c},{reg}")
    (out / "window_trace.csv").write_text(
        "cycle,insts,ipc,rob_occ_mean,iq_occ_mean,lq_occ_mean,l1d_mpki,l2_mpki,mlp,branch_mpki,config,region\n"
        + "\n".join(rows) + "\n")
st = ["---------- Begin Simulation Statistics ----------",
      f"simSeconds {cycles_total / 2e9:.9f}", f"simInsts {int(insts_total)}",
      f"system.cpu.numCycles {int(cycles_total)}",
      f"system.cpu.ipc {insts_total / cycles_total:.6f}"]
for i in range(N):
    st.append(f"system.cpu.window.cyclesInConfig::{i} {int(res[i])}")
    st.append(f"system.cpu.window.robOccSum::{i} {int(occ_sum[i])}")
st.append("---------- End Simulation Statistics   ----------")
(out / "stats.txt").write_text("\n".join(st) + "\n")
'''

#: Microbenchmarks for which fake binaries are created.
KERNELS = ["micro_gather", "micro_chase", "micro_compute", "micro_lowilp", "micro_phased"]


@pytest.fixture
def machines_dir(tmp_path):
    """Write a riscv_ooo JSON (4 configs, initial 3, period 1000); return its directory."""
    d = tmp_path / "machines"
    d.mkdir()
    (d / "riscv_ooo.json").write_text(json.dumps({
        "name": "riscv_ooo", "cpu": {"clock": "2GHz"}, "cache": {"l2": {"size": "1MB"}},
        "memory": {}, "window": {**TABLE, "initial": 3, "period": 1000}}))
    return d


@pytest.fixture
def fake_gem5(tmp_path):
    """Install the ``FAKE_GEM5`` script as an executable ``gem5.opt`` and return its path."""
    g = tmp_path / "gem5.opt"
    g.write_text(FAKE_GEM5)
    g.chmod(0o755)
    return g


@pytest.fixture
def bin_root(tmp_path):
    """Create fake plain/oracle/jones binaries of every microbenchmark and return their root."""
    root = tmp_path / "bin"
    for v in ("plain", "oracle", "jones"):
        (root / v).mkdir(parents=True)
        for k in KERNELS:
            (root / v / k).write_text("fake\n")
    return root
