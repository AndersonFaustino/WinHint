"""Shared fixtures: synthetic gem5 outputs (window_trace.csv, region_stats.csv).

Puts sim/, sim/baselines/lut and sim/baselines/oracle on ``sys.path`` so the
tests can import the host-side scripts directly. ``REGIONS`` defines three
synthetic regions with a known oracle configuration; ``TABLE`` is the
4-config window table of the test machine.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "sim"))
sys.path.insert(0, str(REPO / "sim" / "baselines" / "lut"))
sys.path.insert(0, str(REPO / "sim" / "baselines" / "oracle"))

# region id -> (oracle config, feature centre: ipc, rob_occ, l1d_mpki, mlp)
REGIONS = {
    0: (3, (0.6, 200.0, 40.0, 6.0)),   # memory bound, MLP rich -> large window
    1: (0, (2.8, 40.0, 0.5, 1.0)),     # compute bound -> small window
    2: (1, (1.6, 90.0, 8.0, 2.0)),     # in between
}
#: window table of the test machine (4 configs, ascending)
TABLE = [{"rob": 64, "iq": 32, "lq": 16, "sq": 16}, {"rob": 128, "iq": 64, "lq": 32, "sq": 32},
         {"rob": 192, "iq": 96, "lq": 48, "sq": 48}, {"rob": 256, "iq": 128, "lq": 64, "sq": 64}]


def make_trace(path: Path, config: int, seed: int, rows_per_region: int = 60,
               cumulative: bool = True) -> pd.DataFrame:
    """Write a synthetic window_trace.csv whose rows cluster around ``REGIONS``' features.

    Two passes over the regions, ``rows_per_region`` 1000-cycle periods each,
    with Gaussian noise around each region's IPC/occupancy/MPKI/MLP centre.

    Args:
        path: Output CSV (parent directories are created).
        config: Value of the ``config`` column.
        seed: RNG seed.
        rows_per_region: Periods per region visit.
        cumulative: Write ``insts`` as a cumulative counter (else per period).

    Returns:
        The written DataFrame.
    """
    rng = np.random.default_rng(seed)
    recs, cyc, ins = [], 0, 0
    for rep in range(2):
        for rid, (_, (ipc, rob, mpki, mlp)) in REGIONS.items():
            for _ in range(rows_per_region):
                i = max(0.05, rng.normal(ipc, 0.1 * ipc))
                cyc += 1000
                ins += int(i * 1000)
                recs.append({"cycle": cyc, "insts": ins if cumulative else int(i * 1000),
                             "ipc": i,
                             "rob_occ_mean": max(1.0, rng.normal(rob, 0.1 * rob)),
                             "iq_occ_mean": rob / 2, "lq_occ_mean": rob / 4,
                             "l1d_mpki": max(0.0, rng.normal(mpki, 0.1 * mpki + 0.05)),
                             "l2_mpki": mpki / 4, "mlp": max(1.0, rng.normal(mlp, 0.1 * mlp)),
                             "branch_mpki": 1.0, "config": config, "region": rid})
    df = pd.DataFrame(recs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def make_region_stats(path: Path, config: int) -> pd.DataFrame:
    """Write region_stats.csv where REGIONS' oracle config is each region's IPC optimum.

    Three visits per region; cycles grow by 5 % per config step away from the
    region's best config.

    Args:
        path: Output CSV (parent directories are created).
        config: Configuration the synthetic run used.

    Returns:
        The written DataFrame.
    """
    recs, t = [], 0
    for visit in range(3):
        for rid, (best, _) in REGIONS.items():
            insts = 100000 + 1000 * rid
            # cycles minimal at `best`, +5 % per config step away
            cycles = int(insts * (1.0 + 0.05 * abs(config - best)))
            recs.append({"region": rid, "config": config, "enter_cycle": t,
                         "cycles": cycles, "insts": insts})
            t += cycles
    df = pd.DataFrame(recs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


@pytest.fixture
def machine_json(tmp_path):
    """Write a 4-config test machine JSON (``riscv_ooo``) and return its path."""
    import json
    p = tmp_path / "machines" / "riscv_ooo.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({
        "name": "riscv_ooo", "cpu": {"clock": "2GHz", "issue_width": 4,
                                    "num_int_regs": 288, "num_fp_regs": 288},
        "cache": {"l2": {"size": "1MB", "assoc": 16}}, "memory": {"latency_cycles": 200},
        "window": {k: [c[k] for c in TABLE] for k in ("rob", "iq", "lq", "sq")}}))
    return p


@pytest.fixture
def sweep_tree(tmp_path):
    """Build a synthetic oracle sweep tree and return its root.

    Layout: results/oracle/runs/<m>/<k>/<input>/c<i>/{window_trace,region_stats}.csv
    for three kernels and four configs on ``riscv_ooo``/``small``.
    """
    root = tmp_path / "oracle"
    for kn, k in enumerate(("kern_a_infer", "kern_b_infer", "kern_c_infer")):
        for c in range(4):
            d = root / "runs" / "riscv_ooo" / k / "small" / f"c{c}"
            make_trace(d / "window_trace.csv", c, seed=100 * kn + c)
            make_region_stats(d / "region_stats.csv", c)
    return root


@pytest.fixture(autouse=True)
def _hermetic_run_experiments(tmp_path, monkeypatch):
    """Keep the real tree's tuned.json, region profiles and checkpoints out of every test."""
    import run_experiments
    monkeypatch.setattr(run_experiments, "TUNED_DEFAULT", tmp_path / "no-tuned.json")
    monkeypatch.setattr(run_experiments, "RUNLEN_ROOT", tmp_path / "runlen")
    monkeypatch.setattr(run_experiments, "CKPT_ROOT", tmp_path / "ckpt")
