"""Unit tests of pgo_flow.py (B7 PGO baseline) with fake make/gem5/flock (no builds, no gem5)."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PGO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PGO))
import pgo_flow as pf  # noqa: E402

HDR = "region,config,enter_cycle,cycles,insts"


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Replace ROOT, WINHINT_BUILD and the subprocess calls of pgo_flow with recording fakes.

    ``check_output`` answers ``make -s outdir`` with ``<tmp>/out/<variant>``; ``check_call``
    records the command and, for gem5 runs, writes ``region_stats.csv`` into ``--outdir``
    unless ``state.no_stats`` is set.

    Args:
        tmp_path (Path): pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): pytest monkeypatch fixture.

    Returns:
        (SimpleNamespace): ``calls`` (list of commands run), ``outputs`` (make queries),
            ``no_stats`` flag and ``root``.
    """
    state = SimpleNamespace(calls=[], outputs=[], no_stats=False, root=tmp_path, fail_output=False)
    monkeypatch.setattr(pf, "ROOT", str(tmp_path))
    monkeypatch.setenv("WINHINT_BUILD", str(tmp_path / "build"))
    monkeypatch.delenv("WINHINT_GEM5", raising=False)

    def check_call(cmd, **kw):
        """Record a command; emulate a gem5 run producing region_stats.csv."""
        state.calls.append(list(cmd))
        outs = [c for c in cmd if c.startswith("--outdir=")]
        if outs and not state.no_stats:
            d = Path(outs[0].split("=", 1)[1])
            (d / "region_stats.csv").write_text(HDR + "\n0,%s,0,100,100\n" % cmd[-1])
        return 0

    def check_output(cmd, **kw):
        """Answer ``make -s outdir`` with a synthetic output directory."""
        state.outputs.append(list(cmd))
        if state.fail_output:
            raise subprocess.CalledProcessError(2, cmd)
        variant = [c for c in cmd if c.startswith("VARIANT=")][0].split("=")[1]
        return "make: noise\n%s\n" % (tmp_path / "out" / variant)

    monkeypatch.setattr(pf.subprocess, "check_call", check_call)
    monkeypatch.setattr(pf.subprocess, "check_output", check_output)
    return state


def _args(tmp_path, **kw):
    """Return a select/profile/build namespace with defaults overridden by ``kw``.

    Args:
        tmp_path (Path): Directory for the output map.
        **kw (Any): Attributes to override.

    Returns:
        (SimpleNamespace): The namespace.
    """
    d = dict(stats=[], machine=str(tmp_path / "nomachine.json"), metric="ed2p", regions=None,
             power_json=None, energy_weight=0.15, tie=0.01, source="small",
             out=str(tmp_path / "res" / "k.json"), arch="riscv", input="small", configs=None,
             gem5=None, force=False, dry_run=False, make_vars=[], kernel="k")
    d.update(kw)
    return SimpleNamespace(**d)


def _stats(path, rows, header=HDR):
    """Write a region_stats.csv.

    Args:
        path (Path): File to write (parents created).
        rows (list[str]): CSV data lines.
        header (str): Header line.

    Returns:
        (Path): ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


# ----------------------------------------------------------------------------- load_window
def test_load_window_formats(tmp_path):
    """Check every accepted window format, sorting by ROB size and the default fallbacks."""
    assert pf.load_window(None) is pf.DEFAULT_WINDOW
    assert pf.load_window(str(tmp_path / "missing.json")) is pf.DEFAULT_WINDOW
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"window": [{"rob": 96, "iq": 1, "lq": 1, "sq": 1},
                                        {"rob": 32, "iq": 2, "lq": 2, "sq": 2}]}))
    assert [c["rob"] for c in pf.load_window(str(m))] == [32, 96]
    m.write_text(json.dumps({"window": {"configs": [{"rob": 80}, {"rob": 40}]}}))
    assert pf.load_window(str(m)) == [{"rob": 40}, {"rob": 80}]
    m.write_text(json.dumps({"window": {"rob": [200, 100], "iq": [50, 25], "lq": [9]}}))
    assert pf.load_window(str(m)) == [{"rob": 100, "iq": 25}, {"rob": 200, "iq": 50, "lq": 9}]
    m.write_text(json.dumps({"window": "bogus"}))
    assert pf.load_window(str(m)) is pf.DEFAULT_WINDOW
    m.write_text(json.dumps({}))
    assert pf.load_window(str(m)) is pf.DEFAULT_WINDOW


# ----------------------------------------------------------------------------- read_stats / power
def test_read_stats_aggregates_and_skips(tmp_path):
    """Check per-(region, config) aggregation, energy columns and skipping of bad rows."""
    a = _stats(tmp_path / "a.csv", ["0,0,0,100,50", "0,0,9,100,50", "x,1,0,5,5", "1,1,0,,"])
    b = _stats(tmp_path / "b.csv", ["0,0,0,10,10,2.5", "1,1,0,1,1,"],
               header=HDR + ",energy")
    c = _stats(tmp_path / "c.csv", ["2,0,0,1,1,0.5"], header=HDR + ",energy_j")
    agg = pf.read_stats([str(a), str(b), str(c)])
    assert agg[(0, 0)] == {"cycles": 210.0, "insts": 110.0, "energy": 2.5, "visits": 3}
    assert agg[(1, 1)] == {"cycles": 1.0, "insts": 1.0, "energy": None, "visits": 2}
    assert agg[(2, 0)]["energy"] == 0.5
    assert set(agg) == {(0, 0), (1, 1), (2, 0)}


def test_power_model(tmp_path):
    """Check the ROB proxy and the per-config watts read from a JSON file."""
    w = [{"rob": 64}, {"rob": 128}]
    assert pf.power_model(w, None, 0.5) == {0: 1.25, 1: 1.5}
    pj = tmp_path / "p.json"
    pj.write_text(json.dumps({"0": 2, "1": "3.5"}))
    assert pf.power_model(w, str(pj), 0.5) == {0: 2.0, 1: 3.5}


# ----------------------------------------------------------------------------- select
def _two_region_stats(tmp_path):
    """Write a run directory where region 0 prefers config 3 and region 1 config 0.

    Args:
        tmp_path (Path): Base directory.

    Returns:
        (Path): The run directory.
    """
    runs = tmp_path / "runs"
    for c, (c0, c1) in enumerate([(400, 100), (300, 100), (200, 100), (100, 100)]):
        _stats(runs / f"cfg{c}" / "region_stats.csv", [f"0,{c},0,{c0},100", f"1,{c},0,{c1},100"])
    return runs


def test_select_ipc_and_ties(fake, tmp_path):
    """Check per-region choice by IPC, the tie rule (smallest config) and the output document."""
    runs = _two_region_stats(tmp_path)
    a = _args(tmp_path, stats=[str(runs)], metric="ipc", regions=str(tmp_path / "none.json"))
    pf.select(a)
    doc = json.loads(Path(a.out).read_text())
    assert doc["regions"]["0"]["config"] == 3 and doc["regions"]["0"]["W"] == 256
    assert doc["regions"]["1"]["config"] == 0          # all equal -> smallest config
    assert doc["regions"]["0"]["candidates"]["3"] == {"ipc": 1.0, "cycles": 100.0}
    assert doc["kernel"] == "k" and doc["metric"] == "ipc" and doc["regions_table"] is None
    assert len(doc["stats"]) == 4 and doc["unprofiled_regions"] == []


@pytest.mark.parametrize("metric", ["ed2p", "edp", "cycles", "energy"])
def test_select_metrics_with_energy_column(fake, tmp_path, metric):
    """Check that each metric uses measured energy when present.

    Config 0 is slow but very low energy; config 1 is fast but energy-hungry, so
    ``cycles`` picks 1 and ``energy`` picks 0.
    """
    f = _stats(tmp_path / "s.csv", ["0,0,0,200,100,1", "0,1,0,100,100,100"], header=HDR + ",energy")
    a = _args(tmp_path, stats=[str(f)], metric=metric, regions=str(tmp_path / "none.json"))
    pf.select(a)
    best = json.loads(Path(a.out).read_text())["regions"]["0"]["config"]
    assert best == {"ed2p": 0, "edp": 0, "cycles": 1, "energy": 0}[metric]


def test_select_power_proxy_and_window_index(fake, tmp_path):
    """Check proxy energy (no energy column), a config beyond the window (W=0) and zero-cycle rows."""
    f = _stats(tmp_path / "s.csv", ["0,7,0,100,100", "0,0,0,101,100", "1,0,0,0,10"])
    a = _args(tmp_path, stats=[str(tmp_path / "*.csv")], metric="cycles", tie=0.0,
              regions=str(tmp_path / "none.json"))
    pf.select(a)
    regions = json.loads(Path(a.out).read_text())["regions"]
    assert regions["0"]["config"] == 7 and regions["0"]["W"] == 0
    assert "1" not in regions                          # only zero-cycle samples
    assert f.exists()


def test_select_annotates_and_warns(fake, tmp_path, capsys):
    """Check function/line annotation, unprofiled regions and the stale-profile warning."""
    runs = _two_region_stats(tmp_path)
    table = tmp_path / "k.regions.json"
    table.write_text(json.dumps({"regions": {"0": {"function": "f", "line": 3},
                                             "5": {"function": "g", "line": 9}}}))
    a = _args(tmp_path, stats=[str(runs)], metric="cycles", regions=str(table))
    pf.select(a)
    doc = json.loads(Path(a.out).read_text())
    assert doc["regions"]["0"]["function"] == "f" and doc["regions"]["0"]["line"] == 3
    assert "function" not in doc["regions"]["1"]
    assert doc["unprofiled_regions"] == [5] and doc["regions_table"] == str(table)
    err = capsys.readouterr().err
    assert "profiled regions [1] are not in" in err and "regions [5] were not visited" in err


def test_select_default_regions_from_make(fake, tmp_path):
    """Check that without --regions the table next to the oracle binary is used."""
    runs = _two_region_stats(tmp_path)
    table = tmp_path / "out" / "oracle" / "k.regions.json"
    table.parent.mkdir(parents=True)
    table.write_text(json.dumps({"regions": {"0": {"function": "f", "line": 1},
                                             "1": {"function": "h", "line": 2}}}))
    a = _args(tmp_path, stats=[str(runs)])
    pf.select(a)
    doc = json.loads(Path(a.out).read_text())
    assert doc["regions_table"] == str(table) and doc["regions"]["1"]["function"] == "h"
    assert "outdir" in fake.outputs[0]


def test_select_without_stats_exits(fake, tmp_path):
    """Check the error when no region_stats.csv is found."""
    with pytest.raises(SystemExit, match="no region_stats.csv"):
        pf.select(_args(tmp_path, stats=[str(tmp_path / "empty")]))


# ----------------------------------------------------------------------------- helpers
def test_default_regions_json_fallback(fake, tmp_path):
    """Check the default layout when make fails."""
    fake.fail_output = True
    a = _args(tmp_path, arch="x86")
    assert pf.default_regions_json(a) == str(tmp_path / "build" / "benchmarks" / "x86" / "oracle"
                                             / "k.regions.json")


def test_make_cmd_binary_path_and_dry_run_cmd(fake, tmp_path, capsys):
    """Check the make command line, binary_path() and that dry-run only prints."""
    a = _args(tmp_path, make_vars=["OPT=O3"])
    cmd = pf.make_cmd(a, "pgo", ["PGO_DIR=/x"])
    assert cmd[:4] == ["make", "-C", str(tmp_path / "benchmarks"), "-j1"]
    assert cmd[4:] == ["ARCH=riscv", "VARIANT=pgo", "KERNELS=k", "MACHINE=" + a.machine,
                       "PGO_DIR=/x", "OPT=O3"]
    assert pf.binary_path(a, "oracle") == str(tmp_path / "out" / "oracle" / "k")
    pf.run_cmd(["echo", "hi"], True)
    assert fake.calls == [] and "pgo_flow: echo hi" in capsys.readouterr().out
    pf.run_cmd(["echo", "hi"], False)
    assert fake.calls == [["echo", "hi"]]


def test_build(fake, tmp_path):
    """Check that build makes the pgo variant with PGO_DIR = directory of --out."""
    a = _args(tmp_path)
    pf.build(a)
    assert "VARIANT=pgo" in fake.calls[0] and "PGO_DIR=" + str(tmp_path / "res") in fake.calls[0]


# ----------------------------------------------------------------------------- profile
def _gem5(tmp_path):
    """Create an executable fake gem5.opt.

    Args:
        tmp_path (Path): Directory to create it in.

    Returns:
        (str): Its path.
    """
    g = tmp_path / "gem5.opt"
    g.write_text("#!/bin/sh\n")
    g.chmod(0o755)
    return str(g)


def test_profile_runs_each_config(fake, tmp_path, capsys):
    """Check the oracle build, one flock'ed gem5 run per config, skipping and --force."""
    a = _args(tmp_path, gem5=_gem5(tmp_path))
    pf.profile(a)
    gem5_runs = [c for c in fake.calls if c[0] == "flock"]
    assert "VARIANT=oracle" in fake.calls[0]
    assert len(gem5_runs) == 4
    r = gem5_runs[2]
    assert r[1] == str(tmp_path / "build" / ".heavy.lock") and r[2] == a.gem5
    assert r[r.index("--cmd") + 1] == str(tmp_path / "out" / "oracle" / "k")
    assert r[-4:] == ["--window-policy", "static", "--window-initial", "2"]
    rd = tmp_path / "results" / "pgo" / "runs" / "k" / "small"
    assert a.stats == [str(rd)] and (rd / "cfg3" / "region_stats.csv").exists()
    fake.calls.clear()
    a.configs = [1]
    pf.profile(a)
    assert not [c for c in fake.calls if c[0] == "flock"]
    assert "cfg1 done" in capsys.readouterr().out
    a.force = True
    pf.profile(a)
    assert len([c for c in fake.calls if c[0] == "flock"]) == 1


def test_profile_errors(fake, tmp_path, monkeypatch):
    """Check the non-riscv, missing-gem5 and missing-stats errors."""
    with pytest.raises(SystemExit, match="needs ARCH=riscv"):
        pf.profile(_args(tmp_path, arch="x86"))
    with pytest.raises(SystemExit, match="gem5 not found"):
        pf.profile(_args(tmp_path))
    monkeypatch.setenv("WINHINT_GEM5", _gem5(tmp_path))
    fake.no_stats = True
    with pytest.raises(SystemExit, match="has no region_stats.csv"):
        pf.profile(_args(tmp_path, configs=[0]))


def test_profile_dry_run(fake, tmp_path, capsys):
    """Check that dry-run prints the gem5 commands, runs nothing and creates no directory."""
    a = _args(tmp_path, dry_run=True, configs=[0, 2], stats=["given"])
    pf.profile(a)
    out = capsys.readouterr().out
    assert fake.calls == [] and out.count("--window-initial") == 2
    assert not (tmp_path / "results").exists() and a.stats == ["given"]


# ----------------------------------------------------------------------------- main
def _main(monkeypatch, *argv):
    """Run pf.main() with the given command line.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest monkeypatch fixture.
        *argv (str): Arguments after the program name.
    """
    monkeypatch.setattr(sys, "argv", ["pgo_flow.py", *argv])
    pf.main()


def test_main_select_default_stats_and_out(fake, tmp_path, monkeypatch):
    """Check that `select` reads results/pgo/runs/<kernel>/<input> and writes results/pgo/<kernel>.json."""
    rd = tmp_path / "results" / "pgo" / "runs" / "k" / "small"
    _stats(rd / "cfg0" / "region_stats.csv", ["0,0,0,10,10"])
    _main(monkeypatch, "select", "--kernel", "k", "--regions", str(tmp_path / "none.json"))
    doc = json.loads((tmp_path / "results" / "pgo" / "k.json").read_text())
    assert doc["source"] == "small" and doc["regions"]["0"]["config"] == 0
    assert os.path.isabs(doc["machine"])


def test_main_all_and_build(fake, tmp_path, monkeypatch, capsys):
    """Check `all` (profile + select + build) and `build` with extra make variables."""
    out = tmp_path / "m" / "k.json"
    _main(monkeypatch, "all", "--kernel", "k", "--gem5", _gem5(tmp_path), "--configs", "0", "1",
          "--out", str(out), "--regions", str(tmp_path / "none.json"), "--source", "large")
    doc = json.loads(out.read_text())
    assert doc["source"] == "large" and set(doc["regions"]) == {"0"}
    assert "VARIANT=pgo" in fake.calls[-1] and "PGO_DIR=" + str(out.parent) in fake.calls[-1]
    fake.calls.clear()
    _main(monkeypatch, "build", "--kernel", "k", "OPT=O3")
    assert fake.calls[0][-1] == "OPT=O3" and "VARIANT=pgo" in fake.calls[0]


def test_main_all_dry_run(fake, tmp_path, monkeypatch, capsys):
    """Check that `all --dry-run` prints the select step and writes no map."""
    _main(monkeypatch, "all", "--kernel", "k", "--dry-run", "--configs", "0")
    out = capsys.readouterr().out
    assert "pgo_flow: select" in out and "VARIANT=pgo" in out
    assert fake.calls == [] and not (tmp_path / "results" / "pgo" / "k.json").exists()
