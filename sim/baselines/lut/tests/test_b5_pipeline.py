"""End-to-end B5: oracle labels -> classifier -> runtime LUT (§6)."""
import json
import pickle
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import export_lookup_table
import label_phases
import train_phase_classifier
from conftest import REGIONS
from whdata import Lut, read_lut, write_flat_oracle


@pytest.fixture
def dataset(tmp_path, sweep_tree):
    """Write flat oracle maps for the sweep tree, run label_phases and return the dataset CSV."""
    for k in ("kern_a_infer", "kern_b_infer", "kern_c_infer"):
        write_flat_oracle(sweep_tree / "riscv_ooo" / "small" / f"{k}.json",
                          {r: c for r, (c, _) in REGIONS.items()})
    out = tmp_path / "b5" / "dataset.csv"
    rc = label_phases.main(["--runs-root", str(sweep_tree / "runs"), "--oracle-root",
                            str(sweep_tree), "--machine", "riscv_ooo", "--input", "small",
                            "--threshold-baseline", "--out", str(out)])
    assert rc == 0
    return out


def test_labels_come_from_oracle(dataset):
    """Labels equal the oracle config of each window's region, for every run config."""
    df = pd.read_csv(dataset)
    assert len(df) == 3 * 4 * 2 * 3 * 60
    expect = df["region"].map({r: c for r, (c, _) in REGIONS.items()})
    assert (df["label"] == expect).all()
    assert set(df["run_config"]) == {0, 1, 2, 3}
    assert "threshold_phase" in df.columns  # baseline column only


def test_unknown_region_dropped_or_defaulted(tmp_path, sweep_tree):
    """Windows of a region unknown to the oracle are dropped or get the default config."""
    trace = sweep_tree / "runs" / "riscv_ooo" / "kern_a_infer" / "small" / "c0" / "window_trace.csv"
    oracle = {0: 3, 1: 0}  # region 2 unknown
    d = label_phases.label_trace(trace, oracle, "k", unlabeled="drop")
    assert set(d["region"]) == {0, 1}
    d = label_phases.label_trace(trace, oracle, "k", unlabeled="default", default_config=2)
    assert (d[d["region"] == 2]["label"] == 2).all()


@pytest.mark.parametrize("model", ["tree", "knn", "bins", "mlp", "transformer"])
def test_train_and_export(tmp_path, dataset, model):
    """Each model trains above 90 % held-out accuracy and exports a LUT matching the oracle."""
    out = tmp_path / model
    args = ["--dataset", str(dataset), "--out-dir", str(out), "--model", model,
            "--split", "kernel", "--folds", "3", "--n-configs", "4"]
    if model == "mlp":
        args += ["--epochs", "150"]
    if model == "transformer":
        args += ["--t-epochs", "15"]
    assert train_phase_classifier.main(args) == 0
    rep = json.loads((out / "train_report.json").read_text())
    assert rep["held_out_kernels"]
    assert rep["models"][model]["test"]["accuracy"] > 0.9   # separable synthetic data
    lut_path = out / "lut.txt"
    assert export_lookup_table.main(["--model", str(out / "model.pkl"), "--dataset", str(dataset),
                                     "--runtime-lut", str(lut_path), "--c-header",
                                     str(out / "lut.h"), "--check"]) == 0
    lut = read_lut(lut_path)
    assert lut.features == ["ipc", "rob_occ", "l1d_mpki", "mlp"]
    assert lut.table.min() >= 0 and lut.table.max() <= 3
    chk = json.loads(lut_path.with_suffix(".check.json").read_text())
    assert chk["lut_vs_oracle"] > 0.85
    # LUT lookups at the region centres return the oracle configuration
    for _, (cfg, centre) in REGIONS.items():
        assert lut.lookup(np.array([centre]))[0] == cfg
    assert "winhint_lut_table" in (out / "lut.h").read_text()


def test_kernel_split_rejects_bad_folds():
    """A kernel split needs folds >= 2; more folds than kernels still trains on something."""
    df = pd.DataFrame({"kernel": ["a", "a", "b", "c"]})
    for folds in (0, 1):
        with pytest.raises(ValueError, match="folds"):
            train_phase_classifier.split_indices(df, "kernel", 0, folds)
    for folds in (3, 4, 10):              # 10 > 3 kernels: the build_lut default case
        for seed in range(3):
            tr, te = train_phase_classifier.split_indices(df, "kernel", seed, folds)
            assert len(tr) and len(te)


def test_c_header_rejects_config_index_over_255(tmp_path):
    """write_c_header() refuses indices that an ``unsigned char`` table would truncate."""
    lut = Lut(["ipc"], [np.array([1.0])], np.array([0, 256]))
    with pytest.raises(ValueError, match="0..255"):
        export_lookup_table.write_c_header(tmp_path / "lut.h", lut)
    export_lookup_table.write_c_header(tmp_path / "ok.h", Lut(["ipc"], [np.array([1.0])],
                                                              np.array([0, 255])))
    assert "0, 255," in (tmp_path / "ok.h").read_text()


def test_model_all_reports_every_type(tmp_path, dataset):
    """``--model all`` reports every model type and saves the ``--select`` one."""
    out = tmp_path / "all"
    assert train_phase_classifier.main(["--dataset", str(dataset), "--out-dir", str(out),
                                        "--model", "all", "--select", "tree", "--epochs", "50",
                                        "--t-epochs", "5",
                                        "--split", "random"]) == 0
    rep = json.loads((out / "train_report.json").read_text())
    assert set(rep["models"]) == {"mlp", "tree", "knn", "bins", "transformer"}
    with open(out / "model.pkl", "rb") as fh:
        assert pickle.load(fh)["type"] == "tree"


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not available")
def test_cpp_reader_agrees(tmp_path, dataset):
    """The C++ ``winhint::WindowLut`` reader returns the same configs as ``read_lut``."""
    out = tmp_path / "cpp"
    train_phase_classifier.main(["--dataset", str(dataset), "--out-dir", str(out),
                                 "--model", "tree", "--split", "none"])
    export_lookup_table.main(["--model", str(out / "model.pkl"), "--dataset", str(dataset),
                              "--runtime-lut", str(out / "lut.txt")])
    lut = read_lut(out / "lut.txt")
    pts = np.array([c for _, c in REGIONS.values()] + [[0.0, 0.0, 0.0, 0.0], [9, 999, 999, 99]])
    src = out / "t.cc"
    from conftest import REPO
    hdr = str(REPO / "sim" / "gem5" / "src" / "cpu" / "o3" / "window" / "window_lut.hh")
    src.write_text(f'#include "{hdr}"\n#include <cstdio>\nint main(int c, char**v){{'
                   'winhint::WindowLut l; std::string e; if(!l.load(v[1],e)){puts(e.c_str());return 1;}'
                   'double x[4]; while(scanf("%lf %lf %lf %lf",x,x+1,x+2,x+3)==4) printf("%u\\n",l.lookup(x));}\n')
    exe = out / "t"
    subprocess.run(["g++", "-std=c++17", "-O1", str(src), "-o", str(exe)], check=True)
    inp = "\n".join(" ".join(repr(float(v)) for v in p) for p in pts) + "\n"
    r = subprocess.run([str(exe), str(out / "lut.txt")], input=inp, capture_output=True,
                       text=True, check=True)
    assert [int(x) for x in r.stdout.split()] == lut.lookup(pts).tolist()
