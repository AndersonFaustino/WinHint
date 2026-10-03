"""Edge-case tests of analysis/plot_results.py: parsing fallbacks, skipped figures, ELF sizes."""

import json
import struct

import numpy as np
import pandas as pd
import pytest

import plot_results as pr


def _elf(path, is64=True, sections=((1, 0x6, 100), (8, 0x6, 50), (1, 0x2, 7)), truncate=False):
    """Write a minimal ELF file holding only a section header table.

    Args:
        path (Path): Output path.
        is64 (bool): Write ELF64 (else ELF32), little endian.
        sections (tuple): ``(sh_type, sh_flags, sh_size)`` per section header.
        truncate (bool): Cut the file inside the section header table.

    Returns:
        (Path): ``path``.
    """
    shent = 64 if is64 else 40
    hdr = bytearray(64)
    hdr[:6] = b"\x7fELF" + bytes([2 if is64 else 1, 1])
    if is64:
        struct.pack_into("<Q", hdr, 0x28, 64)
        struct.pack_into("<HH", hdr, 0x3A, shent, len(sections))
    else:
        struct.pack_into("<I", hdr, 0x20, 64)
        struct.pack_into("<HH", hdr, 0x2E, shent, len(sections))
    body = bytearray()
    for t, fl, sz in sections:
        sh = bytearray(shent)
        if is64:
            struct.pack_into("<IQ", sh, 4, t, fl)
            struct.pack_into("<Q", sh, 0x20, sz)
        else:
            struct.pack_into("<II", sh, 4, t, fl)
            struct.pack_into("<I", sh, 0x14, sz)
        body += sh
    data = bytes(hdr + body)
    path.write_bytes(data[:-10] if truncate else data)
    return path


def test_spearman_degenerate_inputs():
    """Check that spearman() is NaN for fewer than two points and for constant ranks."""
    assert np.isnan(pr.spearman([1.0], [2.0]))
    assert np.isnan(pr.spearman([1, 1, 1], [1, 2, 3]))


def test_canonical_variant_edge_cases():
    """Check unparsable static indices, non-mapped middle indices and other variants."""
    assert pr.canonical_variant("static_cX", 4) == "static_cX"
    assert pr.canonical_variant("static_c1", 8) == "static_c1"
    assert pr.canonical_variant("static_c4", 8) == "static_mid"
    assert pr.canonical_variant("static_c0-O3", 4) == "static_c0"
    assert pr.canonical_variant("winhint", 4) == "winhint"


def test_load_summary_minimal_columns(tmp_path):
    """Check that load_summary() fills base_machine/sens_param and filters status/input."""
    p = tmp_path / "s.csv"
    pd.DataFrame([
        {"machine": "m", "kernel": "k", "variant": "static_c3", "ipc": "2.0"},
        {"machine": "m", "kernel": "k", "variant": "static_c0", "ipc": "bad"},
    ]).to_csv(p, index=False)
    df = pr.load_summary(p, {})
    assert list(df["base_machine"]) == ["m", "m"] and list(df["sens_param"]) == ["", ""]
    assert list(df["v"]) == ["static_max", "static_c0"]
    assert df["ipc"].isna().iloc[1]
    pd.DataFrame([
        {"machine": "m", "kernel": "k", "variant": "winhint", "status": "fail", "input": "large"},
        {"machine": "m", "kernel": "k", "variant": "winhint", "status": None, "input": "small"},
        {"machine": "m", "kernel": "k", "variant": "lut", "status": "ok", "input": None},
    ]).to_csv(p, index=False)
    assert list(pr.load_summary(p, {})["variant"]) == ["lut"]


def test_load_machines_skips_invalid(tmp_path, machine_json):
    """Check that load_machines() keeps valid machines and skips unparsable ones."""
    (machine_json.parent / "broken.json").write_text("{not json")
    ms = pr.load_machines(machine_json.parent)
    assert list(ms) == ["riscv_ooo"]
    assert len(ms["riscv_ooo"]["window_table"]) == 4


def test_size_kb_and_machine_params():
    """Check size string parsing (suffixes, bare bytes, garbage) and machine_params()."""
    assert pr._size_kb("1MB") == 1024 and pr._size_kb("256kB") == 256
    assert pr._size_kb("512KiB") == 512 and pr._size_kb("2048") == 2
    assert pr._size_kb("1GB") == 1024 * 1024 and pr._size_kb("2048B") == 2
    assert np.isnan(pr._size_kb("xxMB")) and np.isnan(pr._size_kb("huge"))
    m = {"cache": {"l2": {"size": "2MB"}}, "memory": {"latency_cycles": 300},
         "window_table": [{"rob": 64}, {"rob": 256}]}
    assert pr.machine_params(m) == {"l2_kb": 2048.0, "mem_latency": 300.0, "rob_max": 256}
    p = pr.machine_params({"cache": {"l2_size": "512kB"}, "window_table": [{"rob": 32}]})
    assert p["l2_kb"] == 512 and np.isnan(p["mem_latency"])


def test_read_regions_json_skips_bad_entries(tmp_path):
    """Check that non-integer ids, non-dict entries and regions without W* are skipped."""
    p = tmp_path / "k.regions.json"
    p.write_text(json.dumps({"x": {"w_star": 1}, "1": 5, "2": {"config": 1},
                             "3": {"W*": 9, "config": 2, "conservative": True}}))
    r = pr.read_regions_json(p)
    assert list(r) == [3]
    assert r[3] == {"w_pred": 9.0, "config_pred": 2, "function": "", "line": None,
                    "conservative": True}


def test_find_regions_json_unparsable_file_is_accepted(tmp_path):
    """Check that an unparsable regions.json without machine level is accepted."""
    (tmp_path / "k.regions.json").write_text("not json")
    assert pr.find_regions_json([tmp_path], "k", "m") == tmp_path / "k.regions.json"


def _write_wstar_tree(root, machine_json):
    """Write oracle and compiler files for :func:`pr.wstar_table`.

    Args:
        root (Path): Temporary directory.
        machine_json (Path): Machine description of the ``riscv_ooo`` machine.

    Returns:
        (tuple): ``(oracle_root, compiler_dir, machines)``.
    """
    od = root / "oracle" / "riscv_ooo" / "large"
    od.mkdir(parents=True)
    (od / "k.json").write_text(json.dumps({"0": 3, "1": 0, "2": 9}))
    (od / "k.ed2p.json").write_text(json.dumps({"0": 1, "1": 2}))
    (od / "nopred.json").write_text(json.dumps({"0": 1}))
    cd = root / "comp"
    cd.mkdir()
    (cd / "k.regions.json").write_text(json.dumps({"regions": {
        "0": {"w_star": 250, "config": 3, "conservative": True},
        "1": {"w_star": 60, "config": 0},
        "2": {"w_star": 100, "config": 1}}}))
    machines = pr.load_machines(machine_json.parent)
    machines["absent"] = machines["riscv_ooo"]          # no oracle dir for it
    return root / "oracle", cd, machines


def test_wstar_table_metrics_and_conservative(tmp_path, machine_json):
    """Check wstar_table() (ipc and ed2p oracle files) and fig_wstar() conservative metrics."""
    oroot, cd, machines = _write_wstar_tree(tmp_path, machine_json)
    wt = pr.wstar_table(oroot, [cd], machines)
    assert sorted(wt["region"]) == [0, 1]                 # region 2: config out of range
    assert set(wt["kernel"]) == {"k"}                     # nopred has no regions.json
    wt2 = pr.wstar_table(oroot, [cd], machines, metric="ed2p")
    assert dict(zip(wt2["region"], wt2["oracle_config"])) == {0: 1, 1: 2}
    wt3 = pd.concat([wt, wt.assign(region=wt["region"] + 10, conservative=False),
                     wt.assign(region=wt["region"] + 20, conservative=False)])
    metrics = {}
    pr.fig_wstar(wt3, tmp_path / "out", "png", metrics)
    assert metrics["wstar"]["n_conservative"] == 1 and metrics["wstar"]["n_regions"] == 6
    assert metrics["wstar"]["config_agreement"] == 1.0
    assert metrics["wstar"]["spearman_rho_non_conservative"] == pytest.approx(1.0)
    assert (tmp_path / "out" / "wstar_vs_oracle.png").is_file()


def test_fig_wstar_skips_empty(tmp_path, capsys):
    """Check that fig_wstar() skips an empty table without writing metrics."""
    metrics = {}
    pr.fig_wstar(pd.DataFrame(), tmp_path, "png", metrics)
    assert metrics == {} and "[SKIP] wstar_vs_oracle" in capsys.readouterr().out


def test_fig_bars_and_switch_skip_reasons(tmp_path, capsys):
    """Check the skip paths of fig_bars() and fig_switch_frequency()."""
    df = pd.DataFrame([{"machine": "m", "kernel": "k", "v": "winhint", "ipc": 1.0,
                        "insts": 1e6, "win.switches": 3}])
    metrics = {}
    pr.fig_bars(df, tmp_path, "png", "ed2p", "y", "ed2p_bars", "m", metrics)
    pr.fig_bars(df, tmp_path, "png", "ipc", "y", "ipc_bars", "m", metrics)
    assert pr._norm_table(df, "ipc", "m").empty
    pr.fig_switch_frequency(df.drop(columns=["win.switches"]), tmp_path, "png", metrics)
    pr.fig_switch_frequency(df.assign(v="static_max"), tmp_path, "png", metrics)
    out = capsys.readouterr().out
    assert "no column ed2p" in out and "no B0-large reference on m" in out
    assert "no window switch counter" in out and "only static runs" in out
    assert metrics == {} and not list(tmp_path.iterdir())
    pr.fig_switch_frequency(df, tmp_path, "png", metrics)
    assert metrics["switch_frequency"]["per_minst"] == {"WinHint": pytest.approx(3.0)}


def test_find_col_precedence():
    """Check that _find_col() only looks at win.* columns and respects word order."""
    df = pd.DataFrame(columns=["switches", "win.resizes", "win.Switches"])
    assert pr._find_col(df, ("switch", "resize")) == "win.Switches"
    assert pr._find_col(df, ("resize", "switch")) == "win.resizes"
    assert pr._find_col(df, ("nothing",)) is None


def test_compiler_stats_skips_bad_json(tmp_path):
    """Check that compiler_stats() skips unparsable files and reads nested ones."""
    (tmp_path / "a.winhint.json").write_text("{bad")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.winhint.json").write_text(json.dumps({"hints_setwin": 3}))
    cs = pr.compiler_stats([tmp_path])
    assert cs == {"b": {"hints_setwin": 3, "hints_region": 0, "compile_time_ms": None,
                        "analysis_time_ms": None}}


def test_elf_exec_bytes_synthetic(tmp_path):
    """Check executable-section sums of ELF64/ELF32 files and truncated tables."""
    assert pr.elf_exec_bytes(_elf(tmp_path / "a64")) == 100     # NOBITS and non-exec ignored
    assert pr.elf_exec_bytes(_elf(tmp_path / "a32", is64=False)) == 100
    assert pr.elf_exec_bytes(_elf(tmp_path / "t", truncate=True)) is None
    (tmp_path / "short").write_bytes(b"\x7fELF")
    assert pr.elf_exec_bytes(tmp_path / "short") is None


def test_code_size_table_and_hint_overhead(tmp_path, capsys):
    """Check code_size_table() deltas and the code-size panel of fig_hint_overhead()."""
    root = tmp_path / "bin"
    (root / "plain").mkdir(parents=True)
    (root / "winhint").mkdir()
    (root / "plain" / "sub").mkdir()                            # not a file
    (root / "plain" / "k.json").write_text("{}")               # has a suffix
    (root / "plain" / "notelf").write_text("x" * 80)           # unreadable size
    _elf(root / "plain" / "k")
    _elf(root / "winhint" / "k", sections=((1, 0x6, 110),))
    t = pr.code_size_table(root)
    assert t.to_dict(orient="records") == [{"kernel": "k", "v": "winhint", "plain_bytes": 100,
                                            "bytes": 110, "delta_bytes": 10,
                                            "delta_pct": pytest.approx(10.0)}]
    assert pr.code_size_table(None).empty and pr.code_size_table(tmp_path).empty

    metrics = {}
    empty = pd.DataFrame({"v": [], "kernel": []})
    pr.fig_hint_overhead(empty, [tmp_path / "none"], tmp_path / "o", "png", metrics)
    assert "[SKIP] hint_overhead" in capsys.readouterr().out and metrics == {}
    pr.fig_hint_overhead(empty, [tmp_path / "none"], tmp_path / "o", "png", metrics, root)
    m = metrics["hint_overhead"]
    assert m["code_size"]["mean_delta_pct"] == {"WinHint": pytest.approx(10.0)}
    assert m["code_size"]["per_kernel"]["winhint/k"] == {"bytes": 110, "plain_bytes": 100,
                                                         "delta_bytes": 10}
    assert "dynamic_per_kinst" not in m and "static_setwin" not in m
    assert (tmp_path / "o" / "hint_overhead.png").is_file()


def _sp_rows(machine, scale, variants=("static_c3", "winhint", "lut")):
    """Return summary rows of one machine with IPC ``scale`` relative to static_max.

    Args:
        machine (str): Machine name.
        scale (float): IPC factor of the non-static variants.
        variants (tuple): Variant names (``static_c3`` maps to ``static_max``).

    Returns:
        (list[dict]): Rows for two kernels.
    """
    return [{"machine": machine, "base_machine": machine, "sens_param": "", "kernel": k,
             "v": pr.canonical_variant(v, 4), "ipc": 1.0 if v == "static_c3" else scale}
            for k in ("a", "b") for v in variants]


def test_fig_sensitivity_from_machine_descriptions(tmp_path, capsys):
    """Check the machine-comparison fallback of fig_sensitivity() and its skip path."""
    machines = {"m1": {"cache": {"l2": {"size": "1MB"}}, "memory": {"latency_cycles": 100},
                       "window_table": [{"rob": 64}]},
                "m2": {"cache": {"l2": {"size": "2MB"}}, "memory": {"latency_cycles": 200},
                       "window_table": [{"rob": 128}]}}
    df = pd.DataFrame(_sp_rows("m1", 1.2) + _sp_rows("m2", 1.5) + _sp_rows("m3", 2.0))
    metrics = {}
    assert pr.fig_sensitivity_sweep(df, tmp_path, "png", metrics) is False
    pr.fig_sensitivity(df, machines, tmp_path, "png", metrics)
    s = metrics["sensitivity"]
    assert s["source"] == "machines"
    pts = {(p["machine"], p["v"]): p for p in s["points"]}
    assert set(pts) == {("m1", "winhint"), ("m1", "lut"), ("m2", "winhint"), ("m2", "lut")}
    assert pts[("m2", "winhint")]["speedup"] == pytest.approx(1.5)
    assert pts[("m2", "lut")]["l2_kb"] == 2048 and pts[("m1", "lut")]["mem_latency"] == 100
    assert (tmp_path / "sensitivity.png").is_file()
    one = {}
    pr.fig_sensitivity(pd.DataFrame(_sp_rows("m1", 1.2)), machines, tmp_path, "png", one)
    assert one == {} and "needs >= 2 machines" in capsys.readouterr().out


def test_fig_sensitivity_sweep_without_reference(tmp_path):
    """Check that sweep points without any B0-large run produce no figure."""
    rows = [{"machine": "m+l2_size=1MB", "base_machine": "m", "sens_param": "l2_size",
             "kernel": "a", "v": "winhint", "ipc": 1.0, "l2_kb": 1024.0}]
    assert pr.fig_sensitivity_sweep(pd.DataFrame(rows), tmp_path, "png", {}) is False


def test_fig_portability_fallback_and_skip(tmp_path, capsys):
    """Check that portability falls back to static_max and skips without usable runs."""
    metrics = {}
    df = pd.DataFrame(_sp_rows("m1", 1.25, ("static_c3", "winhint", "lut_xfer"))
                      + _sp_rows("m2", 1.1, ("winhint",)))
    pr.fig_portability(df, tmp_path, "png", metrics)
    recs = {(r["machine"], r["v"]): r for r in metrics["portability"]}
    assert set(recs) == {("m1", "winhint"), ("m1", "lut_xfer")}
    assert recs[("m1", "winhint")]["ref"] == "static_max"
    assert recs[("m1", "winhint")]["rel"] == pytest.approx(1.25)
    none = {}
    pr.fig_portability(pd.DataFrame(_sp_rows("m1", 1.0, ("static_c3", "mlp"))), tmp_path, "png", none)
    assert none == {} and "[SKIP] portability" in capsys.readouterr().out


def test_fig_hw_skip_paths(tmp_path, capsys):
    """Check the skip paths of fig_hw_edp() and fig_hw_nop_overhead()."""
    metrics = {}
    pr.fig_hw_edp(pd.DataFrame({"config": ["R1"], "smt": ["on"], "kernel": ["k"]}),
                  tmp_path, "png", metrics)
    pr.fig_hw_edp(pd.DataFrame({"config": ["WH"], "smt": ["on"], "kernel": ["k"],
                                "wall_s_mean": [1.0], "wall_s_ci_lo": [0.9],
                                "wall_s_ci_hi": [1.1]}), tmp_path, "png", metrics)
    pr.fig_hw_nop_overhead(pd.DataFrame({"config": ["R1"]}), tmp_path, "png", metrics)
    pr.fig_hw_nop_overhead(pd.DataFrame({"config": ["NOP-asm-P", "R1"], "kernel": ["k", "k"],
                                         "nop_overhead_pct": [float("nan"), 1.0]}),
                           tmp_path, "png", metrics)
    out = capsys.readouterr().out
    assert "no usable columns" in out and "no R1 reference" in out
    assert out.count("[SKIP] hw_nop_overhead") == 2
    assert metrics == {} and not list(tmp_path.iterdir())


def test_hw_order_expands_r2():
    """Check that _hw_order() expands R2-* and drops unknown configurations."""
    assert pr._hw_order(["WH", "R2-b", "R1", "R2-a", "X"]) == ["R1", "R2-a", "R2-b", "WH"]


def test_main_without_inputs(tmp_path, capsys):
    """Check that main() skips every figure when no inputs exist and still writes metrics."""
    (tmp_path / "machines").mkdir()
    rc = pr.main(["--summary", str(tmp_path / "none.csv"), "--hw-summary", str(tmp_path / "no.csv"),
                  "--oracle-root", str(tmp_path / "oracle"), "--machines-dir", str(tmp_path / "machines"),
                  "--compiler-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[SKIP] gem5 figures" in out and "[SKIP] hw figures" in out
    assert json.loads((tmp_path / "out" / "metrics.json").read_text()) == {}


def test_main_metrics_json_is_clean(tmp_path, machine_json):
    """Check that main() writes numpy ints as ints and non-finite floats as null."""
    oroot, cd, _ = _write_wstar_tree(tmp_path, machine_json)
    # one region only -> spearman NaN; config agreement is a numpy float
    (cd / "k.regions.json").write_text(json.dumps({"regions": {"1": {"w_star": 60, "config": 0}}}))
    hw = tmp_path / "hw.csv"
    pd.DataFrame([{"kernel": "k", "config": c, "smt": "on", "wall_s_mean": t,
                   "wall_s_ci_lo": t, "wall_s_ci_hi": t}
                  for c, t in (("R1", 2.0), ("WH", 1.0))]).to_csv(hw, index=False)
    out = tmp_path / "out"
    assert pr.main(["--summary", str(tmp_path / "none.csv"), "--hw-summary", str(hw),
                    "--oracle-root", str(oroot), "--machines-dir", str(machine_json.parent),
                    "--compiler-dir", str(cd), "--out-dir", str(out), "--format", "png"]) == 0
    m = json.loads((out / "metrics.json").read_text())
    assert m["wstar"]["spearman_rho"] is None and m["wstar"]["n_regions"] == 1
    assert m["wstar"]["config_agreement"] == 1.0
    assert m["hw_edp"]["smt-on"] == {"R1 stock": 1.0, "WinHint": 0.5}
