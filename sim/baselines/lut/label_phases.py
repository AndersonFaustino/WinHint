#!/usr/bin/env python3
"""B5 step 1: build the labelled per-window dataset (label_phases.py).

Each row of a gem5 ``window_trace.csv`` (docs/interfaces.md §4) becomes one
training sample:

    features  per-window values from the trace (ipc, rob_occ_mean, l1d_mpki,
              mlp, ...); counters that the trace stores cumulatively are
              differenced first (fixes problem 3: no cumulative counters)
    label     the ORACLE-best window configuration of the region the window
              belongs to (``region`` column), read from the oracle map that
              sim/baselines/oracle/oracle_sweep.py wrote for the same kernel/machine/input
              (fixes problem 1: no circular threshold labels)

The traces normally come from the oracle sweep itself
(``results/oracle/runs/<machine>/<kernel>/<input>/c<i>/window_trace.csv``), i.e.
the same binary under every static configuration.

The old threshold rules (ATTENTION / FFN / OTHER) are kept only as an
optional, clearly separate column (``--threshold-baseline``) for plots and
for showing how poorly they agree with the oracle.

Usage:

    python sim/baselines/lut/label_phases.py --machine riscv_ooo --input small \
        --runs-root results/oracle/runs --oracle-root results/oracle \
        --out results/b5/riscv_ooo/dataset.csv

    # explicit trace / oracle pair
    python sim/baselines/lut/label_phases.py --trace path/window_trace.csv --oracle path/kernel.json \
        --kernel encoder_bert_tiny_infer --out dataset.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[3]
from whdata import FEATURE_SOURCE, load_oracle, load_window_trace  # noqa: E402

#: Legacy threshold rules - BASELINE / PLOT ONLY, never a training label.
THRESHOLD_RULES = (
    ("ATTENTION", lambda d: (d["l1d_mpki"] > 20.0) & (d["ipc"] < 1.5)),
    ("FFN", lambda d: (d["l1d_mpki"] < 5.0) & (d["ipc"] > 1.8)),
)

#: Output columns of the dataset CSV, in order (those absent from a trace are skipped).
DATASET_COLUMNS = ["kernel", "machine", "input", "run_config", "window", "cycle", "d_cycles",
                   "region", *FEATURE_SOURCE.values(), "iq_occ_mean", "lq_occ_mean",
                   "l2_mpki", "branch_mpki", "config", "label"]


def threshold_phase(df: pd.DataFrame) -> pd.Series:
    """Classify windows with the legacy threshold rules (baseline/plot only).

    Rules in ``THRESHOLD_RULES`` are tried in order; the first match wins.

    Args:
        df: Trace with ``l1d_mpki`` and ``ipc`` columns.

    Returns:
        ``"ATTENTION"``, ``"FFN"`` or ``"OTHER"`` per row, indexed like ``df``.
    """
    out = pd.Series("OTHER", index=df.index)
    for name, rule in THRESHOLD_RULES:
        out[(out == "OTHER") & rule(df)] = name
    return out


def label_trace(trace: Path, oracle: dict[int, int], kernel: str, machine: str = "",
                input_size: str = "", run_config: int | None = None,
                unlabeled: str = "drop", default_config: int | None = None,
                threshold_baseline: bool = False) -> pd.DataFrame:
    """Label one trace with the oracle map.

    Every window gets the oracle-best config of its ``region`` as ``label``.
    ``unlabeled`` controls windows whose region is unknown to the oracle
    (outside any region, or never visited in the sweep): ``drop`` them or label
    them ``default`` (= default_config, or the most common oracle label when
    None, or 0 for an empty oracle).

    Args:
        trace: ``window_trace.csv`` path.
        oracle: ``{region_id: config_index}`` map (see ``whdata.load_oracle``).
        kernel: Kernel name stored in the ``kernel`` column.
        machine: Machine name stored in the ``machine`` column.
        input_size: Input size stored in the ``input`` column.
        run_config: Static config the trace was recorded under; ``None`` copies
            the trace's per-window ``config`` column.
        unlabeled: ``"drop"`` or ``"default"`` (any value other than ``"drop"``
            fills).
        default_config: Fill label for ``unlabeled="default"``.
        threshold_baseline: Also add the legacy ``threshold_phase`` column.

    Returns:
        One row per kept window with the ``DATASET_COLUMNS`` present in the
        trace (plus ``threshold_phase`` if requested).
    """
    df = load_window_trace(trace)
    df.insert(0, "window", np.arange(len(df)))
    df["label"] = df["region"].map(lambda r: oracle.get(int(r), -1)).astype(int)
    miss = df["label"] < 0
    if miss.any():
        if unlabeled == "drop":
            df = df[~miss].copy()
        else:
            fill = default_config
            if fill is None:
                vals = list(oracle.values())
                fill = int(np.bincount(vals).argmax()) if vals else 0
            df.loc[miss, "label"] = int(fill)
    df["kernel"] = kernel
    df["machine"] = machine
    df["input"] = input_size
    df["run_config"] = int(run_config) if run_config is not None else df["config"]
    cols = [c for c in DATASET_COLUMNS if c in df.columns]
    out = df[cols].copy()
    if threshold_baseline:
        out["threshold_phase"] = threshold_phase(df)
    return out


def discover(runs_root: Path, oracle_root: Path, machine: str, input_size: str,
             kernels: list[str] | None = None,
             metric: str | None = None) -> list[tuple[Path, Path, str, int]]:
    """Find labelling jobs in the layout written by oracle_sweep.py.

    Traces are ``<runs_root>/<machine>/<kernel>/<input>/c<i>/window_trace.csv``;
    oracle maps are ``<oracle_root>/<machine>/<input>/<kernel>.json``. With
    ``metric`` the per-metric map ``<kernel>.<metric>.json`` is preferred over the
    primary ``<kernel>.json``. Kernels without an oracle map are skipped with a
    message on stderr.

    Args:
        runs_root: Root of the oracle sweep run directories.
        oracle_root: Root of the oracle maps.
        machine: Machine name.
        input_size: Input size (``small``/``large``).
        kernels: Restrict to these kernel names; ``None``/empty means all.
        metric: Optional oracle metric (``ipc``/``ed2p``).

    Returns:
        Sorted ``(trace, oracle_json, kernel, run_config)`` tuples; empty if
        ``<runs_root>/<machine>`` does not exist.
    """
    found = []
    base = runs_root / machine
    if not base.is_dir():
        return found
    for kdir in sorted(p for p in base.iterdir() if p.is_dir()):
        if kernels and kdir.name not in kernels:
            continue
        oracle = oracle_root / machine / input_size / f"{kdir.name}.json"
        if metric and (oracle.parent / f"{kdir.name}.{metric}.json").exists():
            oracle = oracle.parent / f"{kdir.name}.{metric}.json"
        if not oracle.exists():
            print(f"[SKIP] {kdir.name}: no oracle map {oracle}", file=sys.stderr)
            continue
        for cdir in sorted((kdir / input_size).glob("c*")):
            trace = cdir / "window_trace.csv"
            if trace.exists() and cdir.name[1:].isdigit():
                found.append((trace, oracle, kdir.name, int(cdir.name[1:])))
    return found


def parse_args(argv=None):
    """Parse the command line.

    Args:
        argv (list[str] | None): Argument list; ``None`` reads ``sys.argv``.

    Returns:
        args (argparse.Namespace): The parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", default="riscv_ooo", help="machine name (sim/machines/<name>.json)")
    p.add_argument("--input", default="small", choices=["small", "large"],
                   help="input size whose traces/oracle are used (B5 trains on small)")
    p.add_argument("--runs-root", type=Path, default=REPO / "results" / "oracle" / "runs")
    p.add_argument("--oracle-root", type=Path, default=REPO / "results" / "oracle")
    p.add_argument("--kernels", nargs="*", help="restrict to these kernels")
    p.add_argument("--trace", type=Path, help="explicit window_trace.csv (with --oracle)")
    p.add_argument("--oracle", type=Path, help="oracle map JSON for --trace")
    p.add_argument("--kernel", default="kernel", help="kernel name for --trace")
    p.add_argument("--metric", default="ipc", choices=["ipc", "ed2p"],
                   help="which oracle optimum labels the windows (<kernel>.<metric>.json)")
    p.add_argument("--unlabeled", default="drop", choices=["drop", "default"])
    p.add_argument("--threshold-baseline", action="store_true",
                   help="add the legacy threshold phase as an extra (non-label) column")
    p.add_argument("--out", type=Path, required=True, help="output dataset CSV")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Build the labelled dataset CSV and print the label distribution.

    Args:
        argv (list[str] | None): Argument list; ``None`` reads ``sys.argv``.

    Returns:
        Exit code: 0 on success, 1 if no traces were found, 2 if ``--trace`` is
        given without ``--oracle``.
    """
    a = parse_args(argv)
    if a.trace:
        if not a.oracle:
            print("--trace needs --oracle", file=sys.stderr)
            return 2
        jobs = [(a.trace, a.oracle, a.kernel, None)]
    else:
        jobs = discover(a.runs_root, a.oracle_root, a.machine, a.input, a.kernels, a.metric)
    if not jobs:
        print("[ERROR] no traces found", file=sys.stderr)
        return 1
    frames = []
    for trace, oracle_path, kernel, cfg in jobs:
        oracle = load_oracle(oracle_path, a.metric)
        df = label_trace(trace, oracle, kernel, a.machine, a.input, cfg,
                         a.unlabeled, threshold_baseline=a.threshold_baseline)
        print(f"[OK] {kernel} c{cfg}: {len(df)} windows")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(a.out, index=False)
    dist = data["label"].value_counts().sort_index()
    print(f"[OK] {len(data)} labelled windows -> {a.out}")
    print("     label distribution: " + ", ".join(f"c{k}={v}" for k, v in dist.items()))
    if a.threshold_baseline:
        agree = pd.crosstab(data["threshold_phase"], data["label"])
        print("     threshold phase vs oracle label:\n" + agree.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
