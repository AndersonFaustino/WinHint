"""Unit tests for whdata: window table, trace deltas, LUT format and binning."""

import numpy as np
import pandas as pd

from conftest import TABLE, make_trace
from whdata import (Lut, config_for_window, load_machine, load_window_trace, quantile_edges,
                    read_lut, window_features, write_lut)


def test_config_for_window():
    """``config_for_window`` picks the smallest fitting config, else the largest."""
    assert config_for_window(TABLE, 1) == 0
    assert config_for_window(TABLE, 64) == 0
    assert config_for_window(TABLE, 65) == 1
    assert config_for_window(TABLE, 256) == 3
    assert config_for_window(TABLE, 300) == 3   # too large -> largest
    assert config_for_window(TABLE, 0) == 3     # release -> largest


def test_load_machine_parallel_lists(machine_json):
    """A parallel-list window section is normalised to per-config dicts."""
    m = load_machine(machine_json)
    assert m["window_table"] == TABLE


def test_trace_cumulative_deltas(tmp_path):
    """Cumulative ``insts`` counters are differenced into per-period deltas."""
    df0 = make_trace(tmp_path / "t.csv", 2, seed=1, cumulative=True)
    df = load_window_trace(tmp_path / "t.csv")
    assert np.allclose(df["d_cycles"], 1000)
    assert df["d_insts"].iloc[1] == df0["insts"].iloc[1] - df0["insts"].iloc[0]
    assert (df["d_insts"] > 0).all()


def test_trace_per_period(tmp_path):
    """Per-period ``insts`` are used unchanged."""
    make_trace(tmp_path / "t.csv", 0, seed=2, cumulative=False)
    df = load_window_trace(tmp_path / "t.csv")
    assert df["d_insts"].iloc[3] == pd.read_csv(tmp_path / "t.csv")["insts"].iloc[3]


def test_trace_missing_ipc_is_derived(tmp_path):
    """A missing ``ipc`` column is derived from ``d_insts / d_cycles``."""
    df0 = make_trace(tmp_path / "t.csv", 0, seed=3)
    df0.drop(columns=["ipc"]).to_csv(tmp_path / "u.csv", index=False)
    df = load_window_trace(tmp_path / "u.csv")
    assert np.allclose(df["ipc"].iloc[1:], df["d_insts"].iloc[1:] / 1000)
    X = window_features(df)
    assert X.shape == (len(df), 4)


def test_lut_roundtrip_and_bin_rule(tmp_path):
    """Write/read round-trip of a LUT, the bin rule and the flat index layout."""
    edges = [np.array([1.0, 2.0]), np.array([10.0]), np.array([]), np.array([0.5, 1.5, 2.5])]
    shape = tuple(len(e) + 1 for e in edges)
    table = np.arange(np.prod(shape)) % 4
    lut = Lut(["ipc", "rob_occ", "l1d_mpki", "mlp"], edges, table)
    write_lut(tmp_path / "l.txt", lut, comment="test")
    back = read_lut(tmp_path / "l.txt")
    assert back.features == lut.features
    assert all(np.allclose(a, b) for a, b in zip(back.edges, lut.edges))
    assert np.array_equal(back.table, lut.table)
    # value equal to an edge goes in the lower bin; above the last edge -> bin n
    b = back.bin_index(np.array([[1.0, 10.0, 5.0, 3.0], [1.5, 11.0, 0.0, 0.1]]))
    assert b.tolist() == [[0, 0, 0, 3], [1, 1, 0, 0]]
    idx = back.flat_index(b)
    assert idx[0] == ((0 * 2 + 0) * 1 + 0) * 4 + 3
    text = (tmp_path / "l.txt").read_text().splitlines()
    assert text[0] == "WINHINT_LUT 1"
    assert text[2] == "features 4 ipc rob_occ l1d_mpki mlp"
    assert text[3].startswith("edges ipc 2 ")
    assert any(t == f"table {int(np.prod(shape))}" for t in text)


def test_quantile_edges_unique_sorted():
    """Quantile edges are strictly ascending even with repeated values."""
    e = quantile_edges(np.array([1, 1, 1, 2, 3, 4, 5, 5, 5.0]), 7)
    assert np.all(np.diff(e) > 0)
