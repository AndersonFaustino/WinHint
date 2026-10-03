#!/usr/bin/env python3
"""B5 per-machine driver: oracle-labelled traces -> model -> runtime LUT (build_lut.py).

For every machine (``sim/machines/<name>.json``) this chains the three B5 steps
on the oracle sweep of that machine (sim/baselines/oracle/oracle_sweep.py,
which records window_trace.csv in every static run):

    1. label_phases.py            results/b5/<m>/dataset.csv
                                  (windows of results/oracle/runs/<m>/*/<input>/c*/,
                                   label = oracle-best config of the window's region)
    2. train_phase_classifier.py  results/b5/<m>/model.pkl, train_report.json
                                  (leave-kernels-out evaluation, then --final-fit)
    3. export_lookup_table.py     results/b5/<m>/lut.txt (+ lut.check.json)
                                  = the window_lut_file of interfaces.md §6, read by
                                  gem5's `lut` policy at run time and used by
                                  sim/run_experiments.py (variants lut / lut_xfer)

The LUT is retrained per machine (that is the point of the portability study:
lut_xfer runs the reference machine's LUT on the other machines).
B5 trains on the `small` input by default (the oracle sweep of `small`), so
evaluation on `large` is on an unseen input; when a `large` sweep exists its
labelled windows are reported as an extra test set.

Training a PyTorch model (``--model mlp|transformer``) is a heavy job: it runs under
``flock $WINHINT_BUILD/.heavy.lock`` (taken here; do not wrap this script in an
outer flock on the same file; --no-lock disables it).

Usage (host, `winhint` env active):

    python sim/baselines/lut/build_lut.py                          # every machine
    python sim/baselines/lut/build_lut.py --machines riscv_ooo --model tree
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import export_lookup_table  # noqa: E402
import label_phases  # noqa: E402
import train_phase_classifier  # noqa: E402
from wh_models import MODEL_TYPES  # noqa: E402
from whdata import load_machine  # noqa: E402

#: Build directory holding the heavy-job lock (``$WINHINT_BUILD``, default ``<repo>/build``).
BUILD = Path(os.environ.get("WINHINT_BUILD", str(REPO / "build")))
#: Lock file serialising heavy jobs (PyTorch training) across the host.
HEAVY_LOCK = BUILD / ".heavy.lock"


@contextlib.contextmanager
def heavy_lock(path: Path | None):
    """Hold an exclusive ``flock`` on ``path`` for the duration of the block.

    Creates the lock file (and its directory) if needed and prints a message
    before waiting.

    Args:
        path: Lock file; ``None`` makes the context manager a no-op.

    Yields:
        held (None): Nothing; the lock is held while the block runs.
    """
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        print(f"[LOCK] waiting for {path}", flush=True)
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def build_one(machine: Path, a) -> int:
    """Run the three B5 steps for one machine.

    Labels the ``--input`` sweep into ``<out-root>/<m>/dataset.csv`` (and the
    ``--test-input`` sweep, if it labels successfully, into
    ``dataset.<test-input>.csv`` as an extra test set), trains with
    ``--final-fit`` and ``--n-configs`` = size of the machine's window table
    (under ``HEAVY_LOCK`` for PyTorch models unless ``--no-lock``), then exports
    ``lut.txt`` with ``--check``.

    Args:
        machine: Machine JSON path; its file stem is the machine name.
        a (argparse.Namespace): Parsed CLI namespace.

    Returns:
        0 on success, otherwise the non-zero exit code of the failing step.
    """
    m = load_machine(machine)
    name = machine.stem
    n_cfg = len(m["window_table"])
    out = a.out_root / name
    out.mkdir(parents=True, exist_ok=True)
    ds = out / "dataset.csv"
    common = ["--machine", name, "--runs-root", str(a.runs_root),
              "--oracle-root", str(a.oracle_root), "--metric", a.metric,
              "--unlabeled", a.unlabeled]
    if a.kernels:
        common += ["--kernels", *a.kernels]
    rc = label_phases.main(common + ["--input", a.input, "--out", str(ds)])
    if rc != 0:
        print(f"[ERROR] {name}: no labelled windows (run the oracle sweep with --input "
              f"{a.input} on this machine first)", file=sys.stderr)
        return rc
    extra = []
    if a.test_input and a.test_input != a.input:
        tds = out / f"dataset.{a.test_input}.csv"
        if label_phases.main(common + ["--input", a.test_input, "--out", str(tds)]) == 0:
            extra = ["--test", str(tds)]
    targs = ["--dataset", str(ds), "--out-dir", str(out), "--model", a.model,
             "--split", a.split, "--n-configs", str(n_cfg), "--seed", str(a.seed),
             "--final-fit", *extra]
    if a.model == "all":
        targs += ["--select", a.select]
    lock = HEAVY_LOCK if (not a.no_lock and a.model in ("mlp", "transformer", "all")) else None
    with heavy_lock(lock):
        rc = train_phase_classifier.main(targs)
    if rc != 0:
        return rc
    rc = export_lookup_table.main(["--model", str(out / "model.pkl"), "--dataset", str(ds),
                                   "--runtime-lut", str(out / "lut.txt"),
                                   "--edges", str(a.edges), "--check"])
    if rc == 0:
        print(f"[OK] {name}: B5 LUT -> {out / 'lut.txt'}")
    return rc


def parse_args(argv=None):
    """Parse the command line.

    Args:
        argv (list[str] | None): Argument list; ``None`` reads ``sys.argv``.

    Returns:
        args (argparse.Namespace): The parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machines", nargs="*", help="machine names (default: all sim/machines/*.json)")
    p.add_argument("--machines-dir", type=Path, default=REPO / "sim" / "machines")
    p.add_argument("--kernels", nargs="*", help="restrict the training kernels")
    p.add_argument("--input", default="small", choices=["small", "large"],
                   help="input whose oracle sweep is the training set")
    p.add_argument("--test-input", default="large", choices=["small", "large", ""],
                   help="also label this input's sweep (if present) as an extra test set")
    p.add_argument("--runs-root", type=Path, default=REPO / "results" / "oracle" / "runs")
    p.add_argument("--oracle-root", type=Path, default=REPO / "results" / "oracle")
    p.add_argument("--out-root", type=Path, default=REPO / "results" / "b5")
    p.add_argument("--metric", default="ipc", choices=["ipc", "ed2p"])
    p.add_argument("--unlabeled", default="drop", choices=["drop", "default"])
    p.add_argument("--model", default="mlp", choices=[*MODEL_TYPES, "all"])
    p.add_argument("--select", default="mlp", choices=MODEL_TYPES)
    p.add_argument("--split", default="kernel", choices=["kernel", "random", "none"])
    p.add_argument("--edges", type=int, default=7, help="LUT edges per feature")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-lock", action="store_true",
                   help=f"do not take {HEAVY_LOCK} around PyTorch training")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Build the B5 LUT for every selected machine.

    Args:
        argv (list[str] | None): Argument list; ``None`` reads ``sys.argv``.

    Returns:
        2 if no machine JSON is selected, otherwise the bitwise OR of the
        per-machine return codes.
    """
    a = parse_args(argv)
    files = sorted(a.machines_dir.glob("*.json"))
    if a.machines:
        files = [f for f in files if f.stem in a.machines]
    if not files:
        print("[ERROR] no machine JSON selected", file=sys.stderr)
        return 2
    rc = 0
    for f in files:
        rc |= build_one(f, a)
    return rc


if __name__ == "__main__":
    sys.exit(main())
