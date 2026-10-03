"""Tests of sim/estimate_energy.py: McPAT path (fake binary), proxy path, CLI and error cases.

No real McPAT is run: ``FAKE_MCPAT`` is a stand-in executable that reads the
``-infile`` XML and prints McPAT-style processor power whose leakage grows
with ``ROB_size``, so the residency weighting can be checked exactly.
``TEMPLATE`` is a minimal McPAT XML with the components the script fills in.
"""
import csv
import json
import runpy
import shutil
import sys
from pathlib import Path

import pytest

import estimate_energy as ee

STATS = """
---------- Begin Simulation Statistics ----------
simSeconds                                   0.001000                       # seconds
simInsts                                      3000000                       # insts
system.cpu.numCycles                          2000000                       # cycles
system.cpu.dcache.overallMisses::total          20000                       # misses
system.cpu.dcache.overallAccesses::total      1000000                       # accesses
system.l2.overallMisses::total                   5000                       # misses
system.cpu.window.cyclesInConfig::0            500000                       # residency
system.cpu.window.cyclesInConfig::3           1500000                       # residency
---------- End Simulation Statistics   ----------
"""

#: minimal McPAT template: every component build_mcpat_xml() writes to, plus NoC and MC
TEMPLATE = """<?xml version="1.0" ?>
<component id="root" name="root">
  <component id="system" name="system">
    <param name="core_tech_node" value="65"/>
    <stat name="total_cycles" value="1"/>
    <component id="system.core0" name="core0">
      <param name="ROB_size" value="80"/>
      <component id="system.core0.icache" name="icache"/>
      <component id="system.core0.dcache" name="dcache"/>
      <component id="system.core0.dtlb" name="dtlb"/>
      <component id="system.core0.itlb" name="itlb"/>
      <component id="system.core0.BTB" name="BTB"/>
    </component>
    <component id="system.L20" name="L20"/>
    <component id="system.NoC0" name="noc0"/>
    <component id="system.mc" name="mc"/>
  </component>
</component>
"""

#: fake McPAT: leakage = ROB_size / 1000 W (+ 0.01 W gate), dynamic 0.5 W
FAKE_MCPAT = r'''#!/usr/bin/env python3
"""Fake McPAT: prints processor power derived from the input XML's ROB_size."""
import os, re, sys
xml = open(sys.argv[sys.argv.index("-infile") + 1]).read()
rob = int(re.search(r'name="ROB_size" value="(\d+)"', xml).group(1))
print("threads=" + os.environ.get("OMP_NUM_THREADS", "?"))
print("Processor:")
print(f"  Subthreshold Leakage = {rob / 1000:.6f} W")
print("  Gate Leakage = 0.01 W")
print("  Runtime Dynamic = 0.5 W")
print("Core:")
print("  Subthreshold Leakage = 99 W")
'''


def _exe(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` as an executable script and return the path."""
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def mcpat(tmp_path, monkeypatch):
    """Install the minimal template and a fake McPAT; return the fake binary's path.

    The template becomes both ``MCPAT_TEMPLATE`` (checked by ``--model auto``)
    and the default ``template`` of ``build_mcpat_xml``; the heavy lock is
    disabled.
    """
    tpl = tmp_path / "tpl.xml"
    tpl.write_text(TEMPLATE)
    monkeypatch.setattr(ee, "MCPAT_TEMPLATE", str(tpl))
    monkeypatch.setattr(ee.build_mcpat_xml, "__defaults__", (str(tpl),))
    monkeypatch.setattr(ee, "HEAVY_LOCK", None)
    monkeypatch.setattr(ee, "MCPAT_THREADS", "3")
    return _exe(tmp_path / "mcpat", FAKE_MCPAT)


@pytest.fixture
def run_dir(tmp_path):
    """Write ``STATS`` (25 % config 0, 75 % config 3) into a run directory and return it."""
    d = tmp_path / "run"
    d.mkdir()
    (d / "stats.txt").write_text(STATS)
    return d


def _leak(rob):
    """Return the fake McPAT's processor leakage (W) for a ROB size."""
    return rob / 1000 + 0.01


def test_mcpat_path_weights_power_by_residency(run_dir, machine_json, mcpat):
    """McPAT runs once per resident config; energy is the residency-weighted power x time."""
    r = ee.estimate_run(run_dir, machine_json, model="mcpat", mcpat_bin=str(mcpat),
                        keep_mcpat=True)
    assert r["model"] == "mcpat" and r["residency"] == [0.25, 0.0, 0.0, 0.75]
    leak = 0.25 * _leak(64) + 0.75 * _leak(256)
    assert r["leakage_w"] == pytest.approx(leak)
    assert r["dynamic_w"] == pytest.approx(0.5)
    assert r["energy_j"] == pytest.approx((leak + 0.5) * 1e-3)
    assert r["power_w"] == pytest.approx(leak + 0.5)
    assert r["edp"] == pytest.approx(r["energy_j"] * 1e-3)
    # only configs with residency > 0 were evaluated; XML and output kept per ROB size
    kept = sorted(p.name for p in run_dir.glob("mcpat_rob*"))
    assert kept == ["mcpat_rob256.txt", "mcpat_rob256.xml", "mcpat_rob64.txt", "mcpat_rob64.xml"]
    assert '<param name="ROB_size" value="256"/>' in (run_dir / "mcpat_rob256.xml").read_text()
    assert "threads=3" in (run_dir / "mcpat_rob64.txt").read_text()


def test_auto_model_picks_mcpat_only_when_available(run_dir, machine_json, mcpat, tmp_path,
                                                   monkeypatch):
    """``auto`` uses McPAT with an executable binary and a template, else the proxy."""
    assert ee.estimate_run(run_dir, machine_json, mcpat_bin=str(mcpat))["model"] == "mcpat"
    assert ee.estimate_run(run_dir, machine_json,
                           mcpat_bin=str(tmp_path / "absent"))["model"] == "proxy"
    noexec = tmp_path / "noexec"
    noexec.write_text("")
    assert ee.estimate_run(run_dir, machine_json, mcpat_bin=str(noexec))["model"] == "proxy"
    monkeypatch.setattr(ee, "MCPAT_TEMPLATE", str(tmp_path / "no-template.xml"))
    assert ee.estimate_run(run_dir, machine_json, mcpat_bin=str(mcpat))["model"] == "proxy"


@pytest.mark.skipif(shutil.which("flock") is None, reason="needs flock(1)")
def test_mcpat_runs_under_heavy_lock(run_dir, machine_json, mcpat, tmp_path, monkeypatch):
    """With HEAVY_LOCK set McPAT runs under flock; the lock's directory is created."""
    lock = tmp_path / "locks" / ".heavy.lock"
    monkeypatch.setattr(ee, "HEAVY_LOCK", lock)
    m = ee.load_machine(machine_json)
    ks = ee.key_stats(ee.parse_stats(run_dir / "stats.txt"))
    leak, dyn = ee.mcpat_power(ks, m, m["window_table"][1], str(mcpat))
    assert leak == pytest.approx(_leak(128)) and dyn == 0.5
    assert lock.exists()


def test_mcpat_failures_raise(run_dir, machine_json, mcpat, tmp_path):
    """A non-zero exit or unparsable McPAT output raises RuntimeError (output still kept)."""
    m = ee.load_machine(machine_json)
    ks = ee.key_stats(ee.parse_stats(run_dir / "stats.txt"))
    bad = _exe(tmp_path / "bad", "#!/bin/sh\necho boom >&2\nexit 3\n")
    with pytest.raises(RuntimeError, match="McPAT exited 3"):
        ee.mcpat_power(ks, m, m["window_table"][0], str(bad), keep_dir=run_dir)
    assert "boom" in (run_dir / "mcpat_rob64.txt").read_text()
    junk = _exe(tmp_path / "junk", "#!/bin/sh\necho no power here\n")
    with pytest.raises(RuntimeError, match="could not parse"):
        ee.mcpat_power(ks, m, m["window_table"][0], str(junk))


def test_parse_mcpat_output():
    """Processor-level (first) values are used; missing values count as zero."""
    text = ("Subthreshold Leakage = 1.5 W\nGate Leakage = 2.5e-1 W\nRuntime Dynamic = 3 W\n"
            "Subthreshold Leakage = 7 W\n")
    assert ee.parse_mcpat_output(text) == (1.75, 3.0)
    assert ee.parse_mcpat_output("Runtime Dynamic = 2 W") == (0.0, 2.0)
    assert ee.parse_mcpat_output("") == (0.0, 0.0)


def test_build_xml_machine_fallbacks_and_new_elements(tmp_path):
    """Sizes/clocks in odd formats fall back to defaults; absent stats are created."""
    tpl = tmp_path / "tpl.xml"
    tpl.write_text(TEMPLATE.replace('<component id="system.NoC0" name="noc0"/>', "")
                   .replace('<component id="system.mc" name="mc"/>', ""))
    ks = {k: 0 for k in ("insts", "fp_insts", "int_insts", "loads", "stores", "cycles",
                         "l1d_accesses", "l1d_misses", "l2_accesses", "l2_misses", "branches",
                         "branch_mispred", "rob_reads", "rob_writes", "l1i_accesses",
                         "l1i_misses")}
    machine = {"cpu": {"clock": "fast"},                  # unparsable -> 2000 MHz
               "cache": {"l1i": {"size": 16384}, "l1d": {"size": "48 KiB"},
                         "l2": {"size": "huge"}}}         # unparsable -> 1 MiB
    xml = ee.build_mcpat_xml(ks, machine, {"rob": 96}, template=tpl)
    assert '<param name="target_core_clockrate" value="2000"/>' in xml
    assert '<param name="icache_config" value="16384,64,4,1,1,2,64,0"/>' in xml
    assert '<param name="dcache_config" value="49152,64,8,1,1,4,64,1"/>' in xml
    assert '<param name="L2_config" value="1048576,64,16,1,1,20,64,1"/>' in xml
    assert '<param name="instruction_window_size" value="64"/>' in xml   # cfg defaults
    assert '<stat name="ROB_reads" value="1"/>' in xml                  # created element
    assert "memory_accesses" not in xml                                  # no MC component
    assert ee._clock_mhz({"cpu": {"clock": "800MHz"}}) == 800
    assert ee._size_bytes("2M") == 2 << 20 and ee._size_bytes(4096.0) == 4096
    tpl.write_text('<component id="root"><component id="system"/></component>')
    with pytest.raises(KeyError, match="system.core0"):
        ee.build_mcpat_xml(ks, machine, {"rob": 96}, template=tpl)


def test_stats_parsing_edge_cases(tmp_path):
    """Last dump wins, non-numeric lines are skipped, NaN becomes 0, simTicks gives seconds."""
    (tmp_path / "stats.txt").write_text(
        "---------- Begin Simulation Statistics ----------\nsimInsts 1\n"
        "---------- Begin Simulation Statistics ----------\n"
        "simTicks 5000000000\nsystem.cpu.numCycles 1000\nsimInsts 500\n"
        "system.cpu.rob.reads 77\nsystem.cpu.l1.reads 9\nbogus text here\nlonely\n"
        "system.cpu.cpi nan\n")
    st = ee.parse_stats(tmp_path / "stats.txt")
    assert "bogus" not in st and "lonely" not in st and st["system.cpu.cpi"] == 0.0
    ks = ee.key_stats(st)
    assert ks["sim_seconds"] == pytest.approx(5e-3)
    assert ks["ipc"] == 0.5 and ks["rob_reads"] == 77 and ks["rob_writes"] == 0.0


def test_residency_static_fallbacks(tmp_path):
    """Without stats residency or a usable trace all time goes to the (clamped) static config."""
    assert ee.window_residency(tmp_path, {}, 4) == ([0, 0, 0, 1.0], "static")
    assert ee.window_residency(tmp_path, {}, 4, static_config=9)[0] == [0, 0, 0, 1.0]
    assert ee.window_residency(tmp_path, {}, 4, static_config=-2)[0] == [1.0, 0, 0, 0]
    (tmp_path / "window_trace.csv").write_text("not,a\ntrace,file\n")   # load error ignored
    assert ee.window_residency(tmp_path, {}, 4, static_config=1) == ([0, 1.0, 0, 0], "static")
    # out-of-range stats indices are ignored
    st = {"system.cpu.window.cyclesInConfig::7": 10.0, "system.cpu.window.residency::1": 5.0}
    assert ee.window_residency(tmp_path, st, 4) == ([0, 1.0, 0, 0], "stats")


def test_proxy_region_energy_and_static_variant():
    """Region proxy energy grows with the window; static_c<i> names give their config."""
    table = [{"rob": 64}, {"rob": 256}]
    small = ee.proxy_region_energy(2e9, 0, 0, table)
    assert small == pytest.approx(1.0 - 0.15 + 0.15 * 64 / 256)
    assert ee.proxy_region_energy(2e9, 0, 1, table) == pytest.approx(1.0)
    assert ee.proxy_region_energy(0, 10, 0, table) == pytest.approx(10 * ee.E_INST)
    assert ee._static_from_variant("static_c2.small") == 2
    assert ee._static_from_variant("winhint") is None


def test_cli_run_dir(run_dir, machine_json, capsys, monkeypatch):
    """``--run-dir`` writes energy.json and prints it; --no-lock clears the heavy lock."""
    monkeypatch.setattr(ee, "HEAVY_LOCK", Path("/nonexistent/.heavy.lock"))
    assert ee.main(["--run-dir", str(run_dir), "--machines-dir", str(machine_json.parent),
                    "--model", "proxy", "--no-lock"]) == 0
    assert ee.HEAVY_LOCK is None
    r = json.loads((run_dir / "energy.json").read_text())
    assert r["model"] == "proxy" and r["machine"] == "riscv_ooo"
    assert json.loads(capsys.readouterr().out) == r


def test_cli_results_root(tmp_path, machine_json, capsys):
    """``--results-root`` reuses energy.json unless --force and records per-run errors."""
    root = tmp_path / "gem5"
    for v in ("static_c1", "winhint"):
        d = root / "riscv_ooo" / "k" / v
        d.mkdir(parents=True)
        (d / "stats.txt").write_text(STATS.replace("cyclesInConfig", "other"))
    cached = root / "riscv_ooo" / "k" / "winhint" / "energy.json"
    cached.write_text(json.dumps({"energy_j": 123.0, "residency": [1.0], "model": "cached"}))
    (root / "nomachine" / "k" / "v").mkdir(parents=True)
    (root / "nomachine" / "k" / "v" / "stats.txt").write_text(STATS)
    args = ["--results-root", str(root), "--machines-dir", str(machine_json.parent),
            "--model", "proxy"]
    assert ee.main(args) == 0
    rows = {r["variant"]: r for r in csv.DictReader(open(root / "energy.csv"))}
    assert rows["winhint"]["energy_j"] == "123.0" and rows["winhint"]["model"] == "cached"
    assert rows["static_c1"]["residency"] == "0.000 1.000 0.000 0.000"   # static_c1 fallback
    assert rows["static_c1"]["model"] == "proxy" and not rows["static_c1"]["error"]
    assert rows["v"]["error"] and rows["v"]["machine"] == "nomachine"
    assert (root / "riscv_ooo" / "k" / "static_c1" / "energy.json").exists()
    out = capsys.readouterr().out
    assert "[ERR] nomachine/k/v" in out and "[OK] 3 runs" in out
    # --force recomputes the cached run; --out-csv moves the CSV
    out_csv = tmp_path / "x" / "e.csv"
    assert ee.main(args + ["--force", "--out-csv", str(out_csv)]) == 0
    rows = {r["variant"]: r for r in csv.DictReader(open(out_csv))}
    assert rows["winhint"]["model"] == "proxy" and float(rows["winhint"]["energy_j"]) != 123.0


def test_cli_requires_a_mode():
    """Neither --run-dir nor --results-root is a usage error."""
    with pytest.raises(SystemExit):
        ee.parse_args([])


def test_script_entry_point(run_dir, machine_json, monkeypatch):
    """Running the file as a script exits with main()'s status."""
    monkeypatch.setattr(sys, "argv", ["estimate_energy.py", "--run-dir", str(run_dir),
                                      "--machine", str(machine_json), "--model", "proxy"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(ee.__file__, run_name="__main__")
    assert e.value.code == 0 and (run_dir / "energy.json").exists()
