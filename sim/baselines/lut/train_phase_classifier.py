#!/usr/bin/env python3
"""B5 step 2: learn per-window features -> window config (train_phase_classifier.py).

Input is the dataset written by label_phases.py: one row per gem5 sampling
window, features taken from window_trace.csv and the label = ORACLE-best
configuration index of the window's region. The classifier therefore
predicts a configuration index directly (no ATTENTION/FFN/OTHER phases).

Models (``--model``, see wh_models.py):

    mlp   small PyTorch MLP (default; the "learned predictor" of B5)
    tree  CART decision tree
    knn   k-nearest neighbours
    bins  majority label per LUT cell (LUT-native)
    transformer  tiny per-window Transformer (features as tokens; closest to the
          original B5 classifier, but exportable to the 4-D LUT)

``--model all`` trains every type and reports a comparison table; the model
named by ``--select`` (default mlp) is saved as model.pkl.

Evaluation split (``--split``):

    kernel  leave-kernels-out (every ``--folds``-th kernel held out; default,
            the honest setting: B5 must generalise to unseen code); needs
            2 <= ``--folds`` <= number of kernels
    random  random 20 % of windows
    none    train on everything, report training accuracy only

``--test`` optionally evaluates the saved model on a different dataset, e.g.
the large input or another machine (portability study).

Besides plain accuracy, the report gives the mean absolute config-index
error and, on the test split, the accuracy weighted by cycles-in-window
(``d_cycles``). No cycle regret against the oracle is computed: the dataset
holds only the cycles of the run that produced each window, not the window's
cycles under every configuration.

Usage:

    python sim/baselines/lut/train_phase_classifier.py --dataset results/b5/riscv_ooo/dataset.csv \
        --out-dir results/b5/riscv_ooo --model mlp
    python sim/baselines/lut/train_phase_classifier.py --dataset ... --model all --select tree
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whdata import FEATURES, FEATURE_SOURCE  # noqa: E402
from wh_models import MODEL_TYPES, make_model  # noqa: E402


def load_dataset(path: Path, features=FEATURES) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load a labelled dataset CSV written by label_phases.py.

    Feature columns are looked up through ``FEATURE_SOURCE``; non-numeric values
    and NaN become 0.

    Args:
        path: Dataset CSV.
        features (Sequence[str]): Feature names (default ``FEATURES``).

    Returns:
        ``(df, X, y)``: the raw table, the ``(n, len(features))`` float feature
        matrix and the integer ``label`` vector.

    Raises:
        ValueError: If a feature column or ``label`` is missing.
    """
    df = pd.read_csv(path)
    cols = [FEATURE_SOURCE.get(f, f) for f in features]
    missing = [c for c in cols + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    y = df["label"].astype(int).to_numpy()
    return df, X, y


def split_indices(df: pd.DataFrame, how: str, seed: int, folds: int):
    """Split dataset rows into train and test indices.

    ``"none"`` puts every row in train. ``"kernel"`` (with a ``kernel`` column
    and more than one kernel) holds out every ``folds``-th kernel in sorted
    order, starting at ``seed % folds``. Otherwise (``"random"`` or a single
    kernel) a seeded random 80/20 split is used.

    Args:
        df: Dataset table.
        how: ``"kernel"``, ``"random"`` or ``"none"``.
        seed: RNG seed / fold offset.
        folds: Kernel-split stride.

    Returns:
        split (tuple[numpy.ndarray, numpy.ndarray]): ``(train_idx, test_idx)`` integer arrays.

    Raises:
        ValueError: For a kernel split with ``folds < 2`` (every kernel would be held
            out). More folds than kernels is fine: the stride then holds out one kernel.
    """
    n = len(df)
    rng = np.random.default_rng(seed)
    if how == "none":
        idx = np.arange(n)
        return idx, np.array([], dtype=int)
    if how == "kernel" and "kernel" in df.columns and df["kernel"].nunique() > 1:
        ks = sorted(df["kernel"].unique())
        if folds < 2:
            raise ValueError(f"kernel split needs folds >= 2, got folds={folds}")
        held = set(ks[(seed % folds)::folds]) or {ks[-1]}
        m = df["kernel"].isin(held).to_numpy()
        return np.where(~m)[0], np.where(m)[0]
    perm = rng.permutation(n)
    cut = int(round(0.8 * n))
    return perm[:cut], perm[cut:]


def metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
            weights: np.ndarray | None = None) -> dict:
    """Compute classification metrics for predicted config indices.

    Args:
        y_true: Oracle labels.
        y_pred: Predicted labels.
        n_classes: Number of configs (confusion matrix size).
        weights: Optional per-sample weights (e.g. ``d_cycles``) for the
            cycle-weighted accuracy.

    Returns:
        ``{"n": 0}`` for empty input; otherwise ``n``, ``accuracy``,
        ``mean_abs_config_error``, ``confusion`` (rows = true, columns =
        predicted), ``per_class`` precision/recall/support and, when weights with
        a positive sum are given, ``cycle_weighted_accuracy``.
    """
    if len(y_true) == 0:
        return {"n": 0}
    ok = y_true == y_pred
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    per_class = {}
    for c in range(n_classes):
        tp = cm[c, c]
        prec = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        rec = tp / cm[c, :].sum() if cm[c, :].sum() else 0.0
        per_class[str(c)] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                             "support": int(cm[c, :].sum())}
    out = {"n": int(len(y_true)), "accuracy": round(float(ok.mean()), 4),
           "mean_abs_config_error": round(float(np.abs(y_true - y_pred).mean()), 4),
           "confusion": cm.tolist(), "per_class": per_class}
    if weights is not None and weights.sum() > 0:
        out["cycle_weighted_accuracy"] = round(float((ok * weights).sum() / weights.sum()), 4)
    return out


def cap_samples(idx: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Uniformly subsample ``idx`` to at most ``cap`` rows (0 = no cap).

    Full traces have ~10^6 windows per machine, too many for 6 GB of RAM.

    Args:
        idx: Candidate row indices.
        cap: Maximum number of rows; <= 0 disables the cap.
        seed: RNG seed.

    Returns:
        ``idx`` unchanged when no capping is needed, otherwise a sorted random
        subset without replacement.
    """
    if cap <= 0 or len(idx) <= cap:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))


def train_one(kind: str, X, y, n_classes: int, args) -> object:
    """Build and fit one model.

    Args:
        kind: Model type (see ``wh_models.MODEL_TYPES``).
        X (numpy.ndarray): ``(n, n_features)`` training features.
        y (numpy.ndarray): Training labels.
        n_classes: Number of configs.
        args (argparse.Namespace): Parsed CLI namespace supplying the hyper-parameters.

    Returns:
        The fitted model.
    """
    return make_model(kind, n_classes, hidden=args.hidden, epochs=args.epochs, lr=args.lr,
                      seed=args.seed, max_depth=args.max_depth, min_leaf=args.min_leaf,
                      k=args.k, n_edges=args.n_edges,
                      t_epochs=getattr(args, "t_epochs", None)).fit(X, y)


def parse_args(argv=None):
    """Parse the command line.

    Args:
        argv (list[str] | None): Argument list; ``None`` reads ``sys.argv``.

    Returns:
        args (argparse.Namespace): The parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True, help="labelled CSV from label_phases.py")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model", default="mlp", choices=[*MODEL_TYPES, "all"])
    p.add_argument("--select", default="mlp", choices=MODEL_TYPES,
                   help="with --model all: which model to save as model.pkl")
    p.add_argument("--split", default="kernel", choices=["kernel", "random", "none"])
    p.add_argument("--folds", type=int, default=4, help="kernel split: hold out every n-th kernel")
    p.add_argument("--final-fit", action="store_true",
                   help="after evaluation, refit the saved model on all data")
    p.add_argument("--n-configs", type=int, default=None,
                   help="number of window configs (default: max label + 1)")
    p.add_argument("--test", type=Path, nargs="*", default=[],
                   help="extra labelled datasets to evaluate on (e.g. large input, other machine)")
    p.add_argument("--seed", type=int, default=0)
    # model hyper-parameters
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--min-leaf", type=int, default=5)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-edges", type=int, default=7, help="bins model: edges per feature")
    p.add_argument("--t-epochs", type=int, default=None,
                   help="transformer model: epochs (default 60, mini-batch)")
    p.add_argument("--max-train-samples", type=int, default=200000,
                   help="random subsample of the training windows (0 = all)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Train/evaluate the requested model(s) and save ``model.pkl``.

    Writes ``<out-dir>/model.pkl`` (dict with ``type``, ``features``,
    ``n_configs``, ``model``) and ``<out-dir>/train_report.json``.

    Args:
        argv (list[str] | None): Argument list; ``None`` reads ``sys.argv``.

    Returns:
        Exit code: 0 on success, 1 for an empty dataset.
    """
    a = parse_args(argv)
    df, X, y = load_dataset(a.dataset)
    if len(y) == 0:
        print("[ERROR] empty dataset", file=sys.stderr)
        return 1
    n_classes = a.n_configs or int(y.max()) + 1
    w = df["d_cycles"].to_numpy(float) if "d_cycles" in df.columns else None
    tr, te = split_indices(df, a.split, a.seed, a.folds)
    kinds = list(MODEL_TYPES) if a.model == "all" else [a.model]
    save_kind = a.select if a.model == "all" else a.model

    a.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"dataset": str(a.dataset), "features": list(FEATURES), "n_configs": n_classes,
              "split": a.split, "n_train": int(len(tr)), "n_test": int(len(te)),
              "label_distribution": np.bincount(y, minlength=n_classes).tolist(),
              "models": {}}
    if a.split == "kernel" and "kernel" in df.columns:
        report["held_out_kernels"] = sorted(df.iloc[te]["kernel"].unique().tolist())
    saved = None
    trs = cap_samples(tr, a.max_train_samples, a.seed)
    report["n_train_used"] = int(len(trs))
    for kind in kinds:
        m = train_one(kind, X[trs], y[trs], n_classes, a)
        r = {"train": metrics(y[trs], m.predict(X[trs]), n_classes)}
        if len(te):
            r["test"] = metrics(y[te], m.predict(X[te]), n_classes,
                                None if w is None else w[te])
        for extra in a.test:
            _, Xe, ye = load_dataset(extra)
            r[f"extra:{extra}"] = metrics(ye, m.predict(Xe), n_classes)
        report["models"][kind] = r
        acc_te = r.get("test", {}).get("accuracy", float("nan"))
        print(f"[{kind:>4}] train acc {r['train']['accuracy']:.3f}   test acc {acc_te:.3f}")
        if kind == save_kind:
            saved = m
    # majority-class reference
    maj = int(np.bincount(y[tr], minlength=n_classes).argmax())
    report["majority_baseline"] = {"config": maj, "test_accuracy":
                                   round(float((y[te] == maj).mean()), 4) if len(te) else None}

    if a.final_fit and len(te):
        al = cap_samples(np.arange(len(y)), a.max_train_samples, a.seed)
        saved = train_one(save_kind, X[al], y[al], n_classes, a)
        report["final_fit"] = True
    with open(a.out_dir / "model.pkl", "wb") as fh:
        pickle.dump({"type": save_kind, "features": list(FEATURES), "n_configs": n_classes,
                     "model": saved}, fh)
    with open(a.out_dir / "train_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[OK] saved {save_kind} model -> {a.out_dir / 'model.pkl'}")
    print(f"[OK] report -> {a.out_dir / 'train_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
