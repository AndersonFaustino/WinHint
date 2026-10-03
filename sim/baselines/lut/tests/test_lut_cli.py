"""B5 CLI drivers, error paths and less-used options (build_lut, label/train/export, whdata).

Everything runs on small synthetic traces (conftest.py) with the fast ``tree``
model; PyTorch training is never started (the heavy-lock path is checked with
a stubbed trainer).
"""
import json
import pickle
import runpy
import sys

import numpy as np
import pandas as pd
import pytest

import build_lut
import export_lookup_table
import label_phases
import train_phase_classifier
import wh_models
import whdata
from conftest import REGIONS, make_region_stats, make_trace
from whdata import Lut, write_flat_oracle, write_lut


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

def _oracle_maps(sweep_tree, kernels=("kern_a_infer", "kern_b_infer", "kern_c_infer"),
                 input_size="small"):
    """Write the flat oracle map of every kernel of the sweep tree for ``input_size``."""
    for k in kernels:
        write_flat_oracle(sweep_tree / "riscv_ooo" / input_size / f"{k}.json",
                          {r: c for r, (c, _) in REGIONS.items()})


@pytest.fixture
def small_dataset(tmp_path):
    """Label a tiny 2-kernel sweep (20 rows per region visit) and return the dataset CSV."""
    root = tmp_path / "mini"
    for k in ("ka", "kb"):
        for c in (0, 3):
            d = root / "runs" / "riscv_ooo" / k / "small" / f"c{c}"
            make_trace(d / "window_trace.csv", c, seed=c + (k == "kb"), rows_per_region=20)
        write_flat_oracle(root / "riscv_ooo" / "small" / f"{k}.json",
                          {r: c for r, (c, _) in REGIONS.items()})
    out = tmp_path / "ds.csv"
    assert label_phases.main(["--runs-root", str(root / "runs"), "--oracle-root", str(root),
                              "--out", str(out)]) == 0
    return out


def _tree(tmp_path, dataset, *extra):
    """Train a tree model on ``dataset`` into ``tmp_path/tree``; return the out dir."""
    out = tmp_path / "tree"
    assert train_phase_classifier.main(["--dataset", str(dataset), "--out-dir", str(out),
                                        "--model", "tree", *extra]) == 0
    return out


def _run_main(module, argv, monkeypatch):
    """Run ``module`` as ``__main__`` with ``argv``; return its ``SystemExit`` code."""
    monkeypatch.setattr(sys, "argv", [module.__file__, *argv])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(module.__file__, run_name="__main__")
    return e.value.code


# ---------------------------------------------------------------------------
# whdata
# ---------------------------------------------------------------------------

def test_window_table_layouts():
    """List-of-dicts, ``configs`` and missing/unknown window sections are normalised."""
    lst = [{"rob": "64", "iq": 32, "lq": 16, "sq": 16, "x": 1}, {"rob": 128}]
    assert whdata.window_table({"window": lst}) == [
        {"rob": 64, "iq": 32, "lq": 16, "sq": 16}, {"rob": 128}]
    assert whdata.window_table({"window": {"configs": lst}}) == whdata.window_table(
        {"window": lst})
    assert whdata.window_table({}) == whdata.DEFAULT_WINDOW
    assert whdata.window_table({"window": {"initial": 3}}) == whdata.DEFAULT_WINDOW
    whdata.window_table({})[0]["rob"] = 1          # a copy: the default stays intact
    assert whdata.DEFAULT_WINDOW[0]["rob"] == 64


def test_trace_and_region_stats_missing_columns(tmp_path):
    """Traces and region stats without the required columns are rejected."""
    (tmp_path / "t.csv").write_text("cycle,ipc\n1000,1.0\n")
    with pytest.raises(ValueError, match=r"lacks columns \['insts', 'config'\]"):
        whdata.load_window_trace(tmp_path / "t.csv")
    (tmp_path / "r.csv").write_text("region,config,cycles\n1,0,10\n")
    with pytest.raises(ValueError, match="enter_cycle"):
        whdata.load_region_stats(tmp_path / "r.csv")


def test_region_stats_aggregate(tmp_path):
    """Region visits are summed per region; IPC is insts / cycles (0 without cycles)."""
    (tmp_path / "r.csv").write_text(" region,config,enter_cycle,cycles,insts\n"
                                    "1,0,0,100,50\n1,0,100,300,150\n2,0,400,x,10\n")
    df = whdata.load_region_stats(tmp_path / "r.csv")
    assert df["cycles"].tolist() == [100, 300, 0]
    g = whdata.aggregate_regions(df).set_index("region")
    assert g.loc[1, "cycles"] == 400 and g.loc[1, "visits"] == 2
    assert g.loc[1, "ipc"] == 0.5 and g.loc[2, "ipc"] == 0.0


def test_trace_increasing_per_period_insts_follow_ipc(tmp_path):
    """Per-period ``insts`` that increase by chance are kept raw when ``ipc`` says so."""
    ins = [100, 200, 300, 400]
    pd.DataFrame({"cycle": [1000, 2000, 3000, 4000], "insts": ins,
                  "ipc": [i / 1000 for i in ins], "config": 0}).to_csv(tmp_path / "t.csv",
                                                                       index=False)
    df = whdata.load_window_trace(tmp_path / "t.csv")
    assert df["d_insts"].tolist() == ins
    assert (df["region"] == whdata.NO_REGION).all()      # missing region column


def test_first_start_and_window_features_errors():
    """``_first_start`` handles short traces; an unknown feature column raises KeyError."""
    assert whdata._first_start(np.array([500.0])) == 0.0
    assert whdata._first_start(np.array([1000.0, 1500.0])) == 500.0
    assert whdata._first_start(np.array([100.0, 1500.0])) == 0.0     # clamped
    with pytest.raises(KeyError, match="needs trace column"):
        whdata.window_features(pd.DataFrame({"ipc": [1.0]}), ["ipc", "nope"])
    assert whdata.window_features(pd.DataFrame({"ipc": [1.0, 2.0]}), []).shape == (2, 0)


def test_load_oracle_summary_and_bad_entries(tmp_path):
    """Full oracle summaries pick the metric (fallback ipc); bad entries are skipped."""
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"best": {"ipc": {"1": 2, "2": {"config": 3}, "x": 1, "3": None},
                                      "ed2p": {"1": 0}}}))
    assert whdata.load_oracle(p) == {1: 2, 2: 3}
    assert whdata.load_oracle(p, "ed2p") == {1: 0}
    p.write_text(json.dumps({"best": {"ipc": {"5": 1}}}))
    assert whdata.load_oracle(p, "ed2p") == {5: 1}


def test_write_lut_rejects_wrong_table_size(tmp_path):
    """``write_lut`` refuses a table whose size does not match the bins."""
    with pytest.raises(ValueError, match="expected 4"):
        write_lut(tmp_path / "l.txt", Lut(["a", "b"], [np.array([1.0]), np.array([2.0])],
                                          np.zeros(3, dtype=int)))


@pytest.mark.parametrize("text, msg", [
    ("WINHINT_LUT 2\n", "not a WINHINT_LUT 1"),
    ("WINHINT_LUT 1\nfeat 1 ipc\n", "expected 'features'"),
    ("WINHINT_LUT 1\nfeatures 1 ipc\nedges mlp 0\n", "expected 'edges ipc'"),
    ("WINHINT_LUT 1\nfeatures 1 ipc\nedges ipc 2 2.0 1.0\n", "not ascending"),
    ("WINHINT_LUT 1\nfeatures 1 ipc\nedges ipc 1 1.0\ntab 2 0 1\n", "expected 'table'"),
    ("WINHINT_LUT 1\nfeatures 1 ipc\nedges ipc 1 1.0\ntable 3 0 1 2\n", "table size 3"),
    ("WINHINT_LUT 1\nfeatures 1 ipc\nedges ipc 1 1.0\ntable 2 0\n", "truncated"),
    ("", "truncated"),
])
def test_read_lut_errors(tmp_path, text, msg):
    """Malformed LUT files raise ``ValueError`` naming the problem."""
    p = tmp_path / "bad.txt"
    p.write_text(text)
    with pytest.raises(ValueError, match=msg):
        whdata.read_lut(p)


def test_quantile_edges_empty_and_iter_trace_dirs(tmp_path):
    """No finite values or no edges give an empty edge list; trace discovery finds files."""
    assert whdata.quantile_edges(np.array([np.nan, np.inf]), 3).size == 0
    assert whdata.quantile_edges(np.array([1.0, 2.0]), 0).size == 0
    a = tmp_path / "x" / "b" / "window_trace.csv"
    b = tmp_path / "x" / "a" / "window_trace.csv"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_text("")
    assert list(whdata.iter_trace_dirs(tmp_path / "x")) == [b, a]
    assert list(whdata.iter_trace_dirs(a)) == [a]


# ---------------------------------------------------------------------------
# wh_models
# ---------------------------------------------------------------------------

def test_tree_without_useful_split_is_a_leaf():
    """A constant feature and a split that does not lower Gini leave a single leaf."""
    X = np.array([[5.0, 0.0], [5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    y = np.array([0, 1, 0, 1])
    m = wh_models.TreeModel(min_leaf=2).fit(X, y)
    assert m.feat.tolist() == [-1]                 # root is a leaf
    assert m.predict(X).tolist() == [0, 0, 0, 0]   # majority, ties -> smallest index


def test_make_model_unknown_type():
    """``make_model`` rejects an unknown model type."""
    with pytest.raises(ValueError, match="unknown model type 'svm'"):
        wh_models.make_model("svm", 4)


# ---------------------------------------------------------------------------
# label_phases
# ---------------------------------------------------------------------------

def test_label_trace_default_fill_uses_majority(tmp_path, sweep_tree):
    """``unlabeled=default`` without a default config fills with the majority oracle label."""
    trace = sweep_tree / "runs" / "riscv_ooo" / "kern_a_infer" / "small" / "c0" / "window_trace.csv"
    d = label_phases.label_trace(trace, {0: 3, 1: 3, 5: 1}, "k", unlabeled="default")
    assert (d[d["region"] == 2]["label"] == 3).all()
    d = label_phases.label_trace(trace, {}, "k", unlabeled="default", run_config=2)
    assert (d["label"] == 0).all() and (d["run_config"] == 2).all()


def test_discover_filters_metric_and_missing_oracle(tmp_path, sweep_tree, capsys):
    """``discover`` honours --kernels, prefers the per-metric map and skips map-less kernels."""
    runs = sweep_tree / "runs"
    assert label_phases.discover(runs, sweep_tree, "other_machine", "small") == []
    _oracle_maps(sweep_tree, kernels=("kern_a_infer", "kern_b_infer"))
    ed2p = sweep_tree / "riscv_ooo" / "small" / "kern_a_infer.ed2p.json"
    write_flat_oracle(ed2p, {0: 0, 1: 0, 2: 0})
    (runs / "riscv_ooo" / "kern_a_infer" / "small" / "cx").mkdir()   # not a config dir
    found = label_phases.discover(runs, sweep_tree, "riscv_ooo", "small", metric="ed2p")
    assert {f[2] for f in found} == {"kern_a_infer", "kern_b_infer"}
    assert "[SKIP] kern_c_infer" in capsys.readouterr().err
    assert all(f[1] == ed2p for f in found if f[2] == "kern_a_infer")
    assert sorted(f[3] for f in found if f[2] == "kern_a_infer") == [0, 1, 2, 3]
    only = label_phases.discover(runs, sweep_tree, "riscv_ooo", "small", ["kern_b_infer"])
    assert {f[2] for f in only} == {"kern_b_infer"} and len(only) == 4


def test_label_main_explicit_trace_and_errors(tmp_path, sweep_tree, capsys):
    """``--trace`` needs ``--oracle``; with both one trace is labelled; no jobs is exit 1."""
    trace = sweep_tree / "runs" / "riscv_ooo" / "kern_a_infer" / "small" / "c2" / "window_trace.csv"
    out = tmp_path / "one.csv"
    assert label_phases.main(["--trace", str(trace), "--out", str(out)]) == 2
    assert "--trace needs --oracle" in capsys.readouterr().err
    oracle = tmp_path / "o.json"
    write_flat_oracle(oracle, {0: 3, 1: 0})
    assert label_phases.main(["--trace", str(trace), "--oracle", str(oracle), "--kernel", "kx",
                              "--unlabeled", "default", "--out", str(out)]) == 0
    df = pd.read_csv(out)
    assert set(df["kernel"]) == {"kx"} and set(df["run_config"]) == {2}
    assert len(df) == 360 and set(df["label"]) == {0, 3}     # region 2 -> majority/smallest
    assert label_phases.main(["--runs-root", str(tmp_path / "none"), "--out", str(out)]) == 1
    assert "no traces found" in capsys.readouterr().err


def test_label_script_entry_point(tmp_path, monkeypatch):
    """Run as a script, label_phases exits with ``main``'s return code."""
    assert _run_main(label_phases, ["--runs-root", str(tmp_path), "--out",
                                    str(tmp_path / "o.csv")], monkeypatch) == 1


# ---------------------------------------------------------------------------
# train_phase_classifier
# ---------------------------------------------------------------------------

def test_load_dataset_missing_columns(tmp_path):
    """A dataset without a feature column or the label is rejected."""
    (tmp_path / "d.csv").write_text("ipc,rob_occ_mean,l1d_mpki,mlp\n1,2,3,4\n")
    with pytest.raises(ValueError, match=r"missing columns \['label'\]"):
        train_phase_classifier.load_dataset(tmp_path / "d.csv")


def test_metrics_empty_weighted_and_cap_samples():
    """Empty metrics, cycle-weighted accuracy and the sample cap."""
    tpc = train_phase_classifier
    assert tpc.metrics(np.array([], int), np.array([], int), 2) == {"n": 0}
    r = tpc.metrics(np.array([0, 1, 1]), np.array([0, 1, 0]), 3, np.array([1.0, 1.0, 2.0]))
    assert r["accuracy"] == pytest.approx(0.6667) and r["cycle_weighted_accuracy"] == 0.5
    assert r["confusion"] == [[1, 0, 0], [1, 1, 0], [0, 0, 0]]
    assert r["per_class"]["2"] == {"precision": 0.0, "recall": 0.0, "support": 0}
    assert "cycle_weighted_accuracy" not in tpc.metrics(np.array([0]), np.array([0]), 1,
                                                        np.array([0.0]))
    idx = np.arange(100)
    assert tpc.cap_samples(idx, 0, 0) is idx and tpc.cap_samples(idx, 200, 0) is idx
    c = tpc.cap_samples(idx, 10, 1)
    assert len(c) == 10 and len(set(c)) == 10 and np.all(np.diff(c) > 0)


def test_split_single_kernel_falls_back_to_random():
    """A kernel split of a single-kernel dataset is a seeded 80/20 random split."""
    df = pd.DataFrame({"kernel": ["a"] * 10})
    tr, te = train_phase_classifier.split_indices(df, "kernel", 0, 4)
    assert len(tr) == 8 and len(te) == 2 and set(tr) | set(te) == set(range(10))


def test_train_empty_dataset(tmp_path, capsys):
    """An empty dataset is exit code 1."""
    cols = ["ipc", "rob_occ_mean", "l1d_mpki", "mlp", "label"]
    (tmp_path / "d.csv").write_text(",".join(cols) + "\n")
    assert train_phase_classifier.main(["--dataset", str(tmp_path / "d.csv"),
                                        "--out-dir", str(tmp_path / "o")]) == 1
    assert "empty dataset" in capsys.readouterr().err


def test_train_extra_test_set_and_final_fit(tmp_path, small_dataset):
    """``--test`` datasets are evaluated and ``--final-fit`` refits on every window."""
    out = _tree(tmp_path, small_dataset, "--split", "kernel", "--folds", "2", "--test",
                str(small_dataset), "--final-fit", "--max-train-samples", "50")
    rep = json.loads((out / "train_report.json").read_text())
    assert rep["final_fit"] is True and rep["n_train_used"] == 50
    assert rep["held_out_kernels"] == ["ka"]
    extra = rep["models"]["tree"][f"extra:{small_dataset}"]
    assert extra["n"] == len(pd.read_csv(small_dataset)) and extra["accuracy"] > 0.9
    assert "cycle_weighted_accuracy" in rep["models"]["tree"]["test"]
    assert rep["majority_baseline"]["test_accuracy"] is not None
    with open(out / "model.pkl", "rb") as fh:
        b = pickle.load(fh)
    assert b["type"] == "tree" and b["n_configs"] == 4


def test_train_script_entry_point(tmp_path, small_dataset, monkeypatch):
    """Run as a script, train_phase_classifier exits 0 and writes the model."""
    out = tmp_path / "s"
    assert _run_main(train_phase_classifier, ["--dataset", str(small_dataset), "--out-dir",
                                              str(out), "--model", "knn", "--split", "none"],
                     monkeypatch) == 0
    assert (out / "model.pkl").is_file()


# ---------------------------------------------------------------------------
# export_lookup_table
# ---------------------------------------------------------------------------

def test_load_model_rejects_foreign_pickle(tmp_path):
    """A pickle that is not a trainer bundle is rejected."""
    p = tmp_path / "m.pkl"
    p.write_bytes(pickle.dumps([1, 2]))
    with pytest.raises(ValueError, match="not a model.pkl"):
        export_lookup_table.load_model(p)


def test_representative_points_empty_bins():
    """Empty bins use the edge midpoint (sample min/max outside, 0..1 without samples)."""
    X = np.array([[1.0], [1.0], [9.0]])
    reps = export_lookup_table.representative_points(X, [np.array([2.0, 4.0, 6.0])])
    assert reps[0].tolist() == [1.0, 3.0, 5.0, 9.0]
    reps = export_lookup_table.representative_points(np.zeros((0, 1)), [np.array([])])
    assert reps[0].tolist() == [0.5]
    reps = export_lookup_table.representative_points(np.zeros((0, 1)), [np.array([2.0])])
    assert reps[0].tolist() == [1.0, 2.0]          # (0 + 2) / 2, (2 + max(1, 2)) / 2


def test_export_needs_an_output(tmp_path, capsys):
    """Without --runtime-lut and --c-header there is nothing to do (exit 2)."""
    assert export_lookup_table.main(["--model", "m.pkl", "--dataset", "d.csv"]) == 2
    assert "nothing to do" in capsys.readouterr().err


def test_export_c_header_only_with_check(tmp_path, small_dataset, capsys):
    """A header-only export checks the in-memory LUT and writes no check file."""
    out = _tree(tmp_path, small_dataset, "--split", "none")
    hdr = tmp_path / "inc" / "lut.h"
    assert export_lookup_table.main(["--model", str(out / "model.pkl"), "--dataset",
                                     str(small_dataset), "--c-header", str(hdr), "--edges", "3",
                                     "--check"]) == 0
    text = hdr.read_text()
    assert "#define WINHINT_LUT_NFEATURES 4" in text
    assert "#define WINHINT_LUT_NCELLS 256" in text          # (3 + 1) ** 4 cells
    chk = json.loads(capsys.readouterr().out.split("[CHECK] ")[1])
    assert chk["model_vs_oracle"] > 0.9 and chk["lut_vs_oracle"] > 0.75
    assert sum(chk["config_counts"]) == 256
    assert not list(tmp_path.rglob("*.check.json"))


def test_export_script_entry_point(tmp_path, monkeypatch):
    """Run as a script, export_lookup_table exits 2 without outputs."""
    assert _run_main(export_lookup_table, ["--model", "m", "--dataset", "d"], monkeypatch) == 2


# ---------------------------------------------------------------------------
# build_lut
# ---------------------------------------------------------------------------

def _build_args(tmp_path, machine_json, sweep_tree, *extra):
    """Return build_lut CLI arguments pointing at the synthetic machine and sweep."""
    return ["--machines-dir", str(machine_json.parent), "--runs-root", str(sweep_tree / "runs"),
            "--oracle-root", str(sweep_tree), "--out-root", str(tmp_path / "b5"), *extra]


def test_build_lut_end_to_end_with_test_input(tmp_path, machine_json, sweep_tree, capsys):
    """build_lut labels, trains a tree (plus the large test set) and exports a checked LUT."""
    _oracle_maps(sweep_tree)
    for kn, k in enumerate(("kern_a_infer", "kern_b_infer")):     # a `large` sweep
        for c in range(4):
            d = sweep_tree / "runs" / "riscv_ooo" / k / "large" / f"c{c}"
            make_trace(d / "window_trace.csv", c, seed=50 + c + kn, rows_per_region=10)
            make_region_stats(d / "region_stats.csv", c)
    _oracle_maps(sweep_tree, ("kern_a_infer", "kern_b_infer"), "large")
    assert build_lut.main(_build_args(tmp_path, machine_json, sweep_tree, "--model", "tree",
                                      "--split", "random", "--edges", "4")) == 0
    out = tmp_path / "b5" / "riscv_ooo"
    assert "[OK] riscv_ooo: B5 LUT" in capsys.readouterr().out
    lut = whdata.read_lut(out / "lut.txt")
    assert lut.shape == (5, 5, 5, 5)
    for _, (cfg, centre) in REGIONS.items():
        assert lut.lookup(np.array([centre]))[0] == cfg
    rep = json.loads((out / "train_report.json").read_text())
    assert rep["final_fit"] is True and rep["n_configs"] == 4
    assert any(k.startswith("extra:") and k.endswith("dataset.large.csv")
               for k in rep["models"]["tree"])
    assert (out / "dataset.large.csv").is_file() and (out / "lut.check.json").is_file()


def test_build_lut_no_labelled_windows(tmp_path, machine_json, sweep_tree, capsys):
    """Without oracle maps the machine fails with label_phases' exit code."""
    assert build_lut.main(_build_args(tmp_path, machine_json, sweep_tree, "--model", "tree",
                                      "--kernels", "kern_a_infer")) == 1
    assert "no labelled windows" in capsys.readouterr().err


def test_build_lut_no_machine_selected(tmp_path, machine_json, sweep_tree, capsys):
    """Selecting no existing machine is exit code 2."""
    assert build_lut.main(_build_args(tmp_path, machine_json, sweep_tree,
                                      "--machines", "nope")) == 2
    assert "no machine JSON selected" in capsys.readouterr().err


def test_build_lut_heavy_lock_and_train_failure(tmp_path, machine_json, sweep_tree, monkeypatch,
                                                capsys):
    """PyTorch models train under the heavy lock (unless --no-lock); trainer errors propagate."""
    _oracle_maps(sweep_tree)
    lock = tmp_path / "build" / ".heavy.lock"
    monkeypatch.setattr(build_lut, "HEAVY_LOCK", lock)
    seen = []

    def fake_train(argv):
        """Record the trainer arguments and whether the lock file exists; fail."""
        seen.append((argv, lock.exists()))
        return 3
    monkeypatch.setattr(build_lut.train_phase_classifier, "main", fake_train)
    args = _build_args(tmp_path, machine_json, sweep_tree, "--test-input", "")
    assert build_lut.main(args + ["--model", "all", "--select", "knn"]) == 3
    argv, locked = seen[-1]
    assert locked and "[LOCK] waiting for" in capsys.readouterr().out
    assert argv[argv.index("--select") + 1] == "knn" and "--test" not in argv
    assert argv[argv.index("--n-configs") + 1] == "4"
    lock.unlink()
    assert build_lut.main(args + ["--model", "mlp", "--no-lock"]) == 3
    assert seen[-1][1] is False and not lock.exists()
    assert not (tmp_path / "b5" / "riscv_ooo" / "lut.txt").exists()


def test_heavy_lock_none_is_noop(tmp_path):
    """``heavy_lock(None)`` takes no lock; a path is created and released."""
    with build_lut.heavy_lock(None):
        pass
    p = tmp_path / "d" / "x.lock"
    with build_lut.heavy_lock(p):
        assert p.exists()
    with build_lut.heavy_lock(p):          # released: can be taken again
        pass


def test_build_lut_script_entry_point(tmp_path, monkeypatch):
    """Run as a script, build_lut exits 2 when no machine is selected."""
    assert _run_main(build_lut, ["--machines-dir", str(tmp_path)], monkeypatch) == 2
