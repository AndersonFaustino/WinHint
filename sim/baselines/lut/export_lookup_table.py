#!/usr/bin/env python3
"""B5 step 3: export the trained predictor as a LUT (export_lookup_table.py).

The gem5 ``lut`` window policy (docs/interfaces.md §4, ``window_lut_file``)
reads a plain-text LUT (§6):

    WINHINT_LUT 1
    features 4 ipc rob_occ l1d_mpki mlp
    edges ipc <n> e1 ... en        # ascending upper bin edges; > en -> bin n
    ...
    table <N>                      # N = prod(n_i + 1)
    c0 c1 ... c(N-1)               # config indices, row-major, first feature slowest

Edges are per-feature quantiles of the training dataset (``--edges`` per
feature), or the model's own edges for the ``bins`` model. Each cell is
filled with the model's prediction at a representative point of the cell
(per-feature median of the training samples falling in that bin; bin
midpoint when the bin is empty).

``--c-header`` additionally writes the same table as a C header (cheap; kept
for tools that want to compile the LUT in). ``--check`` re-reads the text LUT
and reports how often the LUT agrees with the model and with the oracle
labels on the dataset (the quantisation loss).

Usage:

    python sim/baselines/lut/export_lookup_table.py --model results/b5/riscv_ooo/model.pkl \
        --dataset results/b5/riscv_ooo/dataset.csv \
        --runtime-lut results/b5/riscv_ooo/lut.txt --check
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wh_models  # noqa: E402,F401  (needed to unpickle models)
from whdata import Lut, quantile_edges, read_lut, write_lut  # noqa: E402
from train_phase_classifier import load_dataset  # noqa: E402


def load_model(path: Path) -> dict:
    """Load a pickled model bundle written by train_phase_classifier.py.

    Args:
        path: ``model.pkl`` path.

    Returns:
        The bundle dict; it has at least ``model`` (and, as written by the
        trainer, ``type``, ``features`` and ``n_configs``).

    Raises:
        ValueError: If the pickle is not a dict with a ``model`` key.
    """
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, dict) or "model" not in obj:
        raise ValueError(f"{path}: not a model.pkl written by train_phase_classifier.py")
    return obj


def representative_points(X: np.ndarray, edges: list[np.ndarray]) -> list[np.ndarray]:
    """Return one representative value per bin of each feature.

    The representative of a bin is the median of the samples falling in it; an
    empty bin uses the midpoint of its edges, where the outer bins are bounded by
    the sample min/max (or 0.0/1.0 when ``X`` is empty).

    Args:
        X: ``(n, n_features)`` training features.
        edges: Ascending upper bin edges per feature.

    Returns:
        One array of ``len(edges[i]) + 1`` values per feature.
    """
    reps = []
    for i, e in enumerate(edges):
        x = X[:, i]
        b = np.searchsorted(e, x, side="left")
        lo, hi = (float(np.min(x)), float(np.max(x))) if len(x) else (0.0, 1.0)
        pts = []
        for k in range(len(e) + 1):
            sel = x[b == k]
            if sel.size:
                pts.append(float(np.median(sel)))
            else:
                a = e[k - 1] if k > 0 else min(lo, e[0] if len(e) else lo)
                z = e[k] if k < len(e) else max(hi, e[-1] if len(e) else hi)
                pts.append(0.5 * (a + z))
        reps.append(np.array(pts))
    return reps


def build_lut(model, features: list[str], X: np.ndarray, n_edges: int) -> Lut:
    """Build the runtime LUT for a trained model.

    A ``bins`` model (anything with ``edges`` and ``table`` attributes) is copied
    exactly. Otherwise each feature gets ``n_edges`` quantile edges from ``X``
    and every cell is filled with ``model.predict`` at the cell's representative
    point (see ``representative_points``).

    Args:
        model (object): Trained model with a ``predict(X)`` method.
        features: Feature names, in column order of ``X``.
        X: ``(n, n_features)`` training features.
        n_edges: Requested edges per feature (ignored for ``bins`` models).

    Returns:
        The filled ``Lut``.
    """
    if hasattr(model, "edges") and hasattr(model, "table"):  # bins model: exact
        return Lut(list(features), [np.asarray(e, float) for e in model.edges],
                   np.asarray(model.table, dtype=np.int64))
    edges = [quantile_edges(X[:, i], n_edges) for i in range(X.shape[1])]
    reps = representative_points(X, edges)
    grid = np.stack(np.meshgrid(*reps, indexing="ij"), axis=-1).reshape(-1, len(reps))
    table = np.asarray(model.predict(grid), dtype=np.int64)
    return Lut(list(features), edges, table)


def write_c_header(path: Path, lut: Lut) -> None:
    """Write the LUT as a self-contained C header (``WINHINT_LUT_H``).

    Emits the feature names, per-feature edge counts and edge arrays, and the
    flat table as ``unsigned char``. Parent directories are created.

    Args:
        path: Output header path.
        lut: Table to write.

    Raises:
        ValueError: If a table entry does not fit ``unsigned char`` (0..255).
    """
    if lut.table.size and (int(lut.table.min()) < 0 or int(lut.table.max()) > 255):
        raise ValueError(f"LUT config indices must be in 0..255 for the unsigned char "
                         f"table, got {int(lut.table.min())}..{int(lut.table.max())}")
    lines = ["/* Generated by sim/baselines/lut/export_lookup_table.py - do not edit. */",
             "#ifndef WINHINT_LUT_H", "#define WINHINT_LUT_H", "",
             f"#define WINHINT_LUT_NFEATURES {len(lut.features)}",
             f"#define WINHINT_LUT_NCELLS {lut.table.size}", ""]
    lines.append("static const char *const winhint_lut_features[] = {"
                 + ", ".join(f'"{f}"' for f in lut.features) + "};")
    for f, e in zip(lut.features, lut.edges):
        body = ", ".join(f"{v:.9g}" for v in e) or "0"
        lines.append(f"static const unsigned winhint_lut_nedges_{f} = {len(e)};")
        lines.append(f"static const double winhint_lut_edges_{f}[] = {{{body}}};")
    lines.append("static const unsigned char winhint_lut_table[WINHINT_LUT_NCELLS] = {")
    vals = [str(int(v)) for v in lut.table]
    for i in range(0, len(vals), 32):
        lines.append("    " + ", ".join(vals[i:i + 32]) + ",")
    lines += ["};", "", "#endif", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_args(argv=None):
    """Parse the command line.

    Args:
        argv (list[str] | None): Argument list; ``None`` reads ``sys.argv``.

    Returns:
        args (argparse.Namespace): The parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="model.pkl from train_phase_classifier.py")
    p.add_argument("--dataset", type=Path, required=True,
                   help="training dataset (edges / representative points)")
    p.add_argument("--runtime-lut", type=Path, help="write the §6 text LUT here")
    p.add_argument("--c-header", type=Path, help="also write the table as a C header")
    p.add_argument("--edges", type=int, default=7, help="edges per feature (bins = edges + 1)")
    p.add_argument("--check", action="store_true", help="re-read the LUT and report agreement")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Export the model as a text LUT and/or C header, optionally checking it.

    With ``--check`` the agreement of LUT, model and oracle labels on the
    dataset is printed and, when a text LUT was written, saved next to it as
    ``<lut>.check.json``.

    Args:
        argv (list[str] | None): Argument list; ``None`` reads ``sys.argv``.

    Returns:
        Exit code: 0 on success, 2 if neither ``--runtime-lut`` nor
        ``--c-header`` is given.
    """
    a = parse_args(argv)
    if not a.runtime_lut and not a.c_header:
        print("nothing to do: pass --runtime-lut and/or --c-header", file=sys.stderr)
        return 2
    obj = load_model(a.model)
    features = obj["features"]
    _, X, y = load_dataset(a.dataset, features)
    lut = build_lut(obj["model"], features, X, a.edges)
    counts = np.bincount(lut.table, minlength=obj.get("n_configs", 1)).tolist()
    comment = (f"model={obj['type']} n_configs={obj.get('n_configs')} "
               f"dataset={a.dataset.name} cells={lut.table.size} config_counts={counts}")
    if a.runtime_lut:
        write_lut(a.runtime_lut, lut, comment)
        print(f"[OK] runtime LUT ({'x'.join(map(str, lut.shape))} = {lut.table.size} cells) "
              f"-> {a.runtime_lut}")
    if a.c_header:
        write_c_header(a.c_header, lut)
        print(f"[OK] C header -> {a.c_header}")
    if a.check:
        chk = read_lut(a.runtime_lut) if a.runtime_lut else lut
        lp = chk.lookup(X)
        mp = obj["model"].predict(X)
        rep = {"lut_vs_model": round(float((lp == mp).mean()), 4),
               "lut_vs_oracle": round(float((lp == y).mean()), 4),
               "model_vs_oracle": round(float((mp == y).mean()), 4),
               "config_counts": counts}
        print("[CHECK] " + json.dumps(rep))
        if a.runtime_lut:
            (a.runtime_lut.with_suffix(".check.json")).write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
