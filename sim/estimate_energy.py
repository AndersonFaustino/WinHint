#!/usr/bin/env python3
r"""
estimate_energy.py - energy, EDP and ED²P for WinHint gem5 runs.

For each gem5 run directory (stats.txt [+ window_trace.csv]) this computes

  energy_j   total core+cache energy of the run
  power_w    average power
  edp        energy x delay              (J*s)
  ed2p       energy x delay^2            (J*s^2)

Window configurations (docs/interfaces.md §3) change the active ROB/IQ/LQ/SQ
sizes during a run. The run's time is split by the fraction of cycles spent
in each configuration ("residency"), taken from, in order of preference:
  1. stats.txt counters  system.cpu.window.cyclesInConfig::<config>
  2. window_trace.csv    (sum of per-period cycles grouped by `config`)
  3. the static configuration of the run (``--config``, or the largest)
and energy = sum_i residency_i * E(config i).

Two models:
  mcpat  McPAT ($MCPAT_BIN, default $WINHINT_BUILD/mcpat/mcpat) on an XML built from the gem5
         stats and the machine JSON, once per window configuration (ROB, IQ,
         LQ/SQ sizes substituted). E(config i) = (leakage_i + dynamic_i) * T.
  proxy  documented analytic model, used when McPAT is unavailable:
           P_static(i) = P_CORE * (1 - W_FRAC + W_FRAC * rob_i / rob_max)
           E = T * sum_i residency_i * P_static(i)
               + insts * E_INST + l1d_misses * E_L1MISS + l2_misses * E_L2MISS
         with P_CORE = 1.0 W, W_FRAC = 0.15 (window structures' share of core
         power at full size; the same weight the WinHint cost model uses),
         E_INST = 0.2 nJ, E_L1MISS = 1 nJ, E_L2MISS = 15 nJ. Absolute values
         are not meaningful; relative comparisons on one machine are.
``--model auto`` (default) uses McPAT when the binary exists, else the proxy.

Library use (sim/run_experiments.py, sim/baselines/oracle/oracle_sweep.py):
  parse_stats(path) -> dict, key_stats(stats) -> dict,
  estimate_run(run_dir, machine, model=...) -> dict

CLI:
  python sim/estimate_energy.py --results-root results/gem5        # every run
  python sim/estimate_energy.py --run-dir results/gem5/riscv_ooo/k/winhint \
      --machine sim/machines/riscv_ooo.json
Writes <run_dir>/energy.json and, for --results-root, <root>/energy.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim" / "baselines" / "lut"))
from whdata import load_machine, load_window_trace  # noqa: E402

BUILD = Path(os.environ.get("WINHINT_BUILD", str(REPO / "build")))
MCPAT_BIN = os.environ.get("MCPAT_BIN", str(BUILD / "mcpat" / "mcpat"))
#: technology node (nm) written into the McPAT XML
TECH_NODE = 22

# proxy model constants (see module docstring)
P_CORE = 1.0
W_FRAC = 0.15
E_INST = 0.2e-9
E_L1MISS = 1.0e-9
E_L2MISS = 15.0e-9

# McPAT XML: McPAT's own out-of-order single-core description (Alpha21364.xml,
# shipped with McPAT in ProcessorDescriptionFiles/) is used as the template;
# technology node, clock, widths, window sizes (ROB/IQ/LQ/SQ, per window
# configuration), register files, caches and every activity counter are
# overwritten from the machine JSON and gem5's stats.txt. The template supplies
# only structure the gem5 model does not describe (predictor, TLBs, NoC, MC).
MCPAT_TEMPLATE = os.environ.get(
    "MCPAT_TEMPLATE", str(BUILD / "mcpat" / "src" / "ProcessorDescriptionFiles" / "Alpha21364.xml"))
MCPAT_THREADS = os.environ.get("MCPAT_THREADS", "2")  # McPAT is OpenMP-parallel; 6 GB/12 threads host
# McPAT peaks at ~1.2 GB RSS with this template, so every McPAT call is a heavy
# job: it runs under ``flock $WINHINT_BUILD/.heavy.lock`` (interfaces.md §1).
# This script and sim/run_experiments.py take the lock themselves, per McPAT /
# gem5 call: do NOT wrap them in an outer flock on the same file (deadlock).
# WINHINT_NO_LOCK=1 (or --no-lock) disables it.
HEAVY_LOCK: Path | None = (None if os.environ.get("WINHINT_NO_LOCK") == "1"
                           else BUILD / ".heavy.lock")

# ---------------------------------------------------------------------------
# gem5 stats
# ---------------------------------------------------------------------------

_BEGIN = "Begin Simulation Statistics"


def parse_stats(path: str | Path) -> dict[str, float]:
    """Parse a gem5 stats.txt into ``{stat_name: value}``.

    Only the last dump is kept if there are several (each
    ``Begin Simulation Statistics`` line restarts the dict). Lines whose second
    field is not a number are skipped; NaN values become 0.0.

    Args:
        path: Path to stats.txt.

    Returns:
        Flat mapping of stat names to floats.
    """
    stats: dict[str, float] = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if _BEGIN in line:
                stats = {}
                continue
            parts = line.split()
            if len(parts) < 2 or parts[0].startswith("-"):
                continue
            try:
                v = float(parts[1])
            except ValueError:
                continue
            if math.isnan(v):
                v = 0.0
            stats[parts[0]] = v
    return stats


def _first(stats: dict, names: list[str], default: float = 0.0) -> float:
    """Return the value of the first of ``names`` present in ``stats``, else ``default``."""
    for n in names:
        if n in stats:
            return stats[n]
    return default


def _rx(stats: dict, pattern: str, default: float = 0.0) -> float:
    """Return the value of the first stat whose name matches regex ``pattern``, else ``default``."""
    r = re.compile(pattern)
    for k, v in stats.items():
        if r.search(k):
            return v
    return default


def key_stats(stats: dict) -> dict:
    """Normalise the stats needed for energy across gem5 versions (v25.1 names first).

    Args:
        stats: Output of :func:`parse_stats`.

    Returns:
        Dict with ``sim_seconds``, ``cycles``, ``insts``, ``ipc``, L1I/L1D/L2
        access and miss counts, ``branches``, ``branch_mispred``, ``loads``,
        ``stores``, ``fp_insts``, ``int_insts``, ``rob_reads``, ``rob_writes``
        (0.0 when absent), plus every ``system.cpu.window.*`` counter under a
        ``win.`` prefix. ``sim_seconds`` falls back to ``simTicks`` * 1e-12 and
        ``ipc`` to insts / cycles.
    """
    cycles = _first(stats, ["system.cpu.numCycles", "system.cpu.cpuStats.numCycles"])
    insts = _first(stats, ["simInsts", "system.cpu.committedInsts",
                           "system.cpu.commitStats0.numInsts", "system.cpu.thread_0.numInsts"])
    sim_s = _first(stats, ["simSeconds", "sim_seconds"])
    if not sim_s and "simTicks" in stats:
        sim_s = stats["simTicks"] * 1e-12
    out = {
        "sim_seconds": sim_s,
        "cycles": cycles,
        "insts": insts,
        "ipc": _first(stats, ["system.cpu.ipc"], insts / cycles if cycles else 0.0),
        "l1d_accesses": _first(stats, ["system.cpu.dcache.overallAccesses::total",
                                       "system.cpu.dcache.demandAccesses::total"]),
        "l1d_misses": _first(stats, ["system.cpu.dcache.overallMisses::total",
                                     "system.cpu.dcache.demandMisses::total"]),
        "l1i_accesses": _first(stats, ["system.cpu.icache.overallAccesses::total",
                                       "system.cpu.icache.demandAccesses::total"]),
        "l1i_misses": _first(stats, ["system.cpu.icache.overallMisses::total",
                                     "system.cpu.icache.demandMisses::total"]),
        "l2_accesses": _first(stats, ["system.l2.overallAccesses::total",
                                      "system.l2cache.overallAccesses::total",
                                      "system.l2.demandAccesses::total"]),
        "l2_misses": _first(stats, ["system.l2.overallMisses::total",
                                    "system.l2cache.overallMisses::total",
                                    "system.l2.demandMisses::total"]),
        "branches": _first(stats, ["system.cpu.branchPred.lookups_0::total",
                                   "system.cpu.branchPred.lookups"]),
        "branch_mispred": _first(stats, ["system.cpu.branchPred.condIncorrect",
                                         "system.cpu.commit.branchMispredicts"]),
        "loads": _first(stats, ["system.cpu.commitStats0.numLoadInsts",
                                "system.cpu.commit.loads"]),
        "stores": _first(stats, ["system.cpu.commitStats0.numStoreInsts"]),
        "fp_insts": _first(stats, ["system.cpu.commitStats0.numFpInsts",
                                   "system.cpu.commit.floating"]),
        "int_insts": _first(stats, ["system.cpu.commitStats0.numIntInsts",
                                    "system.cpu.commit.integer"]),
        "rob_reads": _rx(stats, r"system\.cpu\.rob\.reads$"),
        "rob_writes": _rx(stats, r"system\.cpu\.rob\.writes$"),
    }
    for k, v in stats.items():  # all WinHint window counters
        if k.startswith("system.cpu.window."):
            out["win." + k[len("system.cpu.window."):]] = v
    return out


# ---------------------------------------------------------------------------
# Window residency
# ---------------------------------------------------------------------------

# Per-configuration residency: system.cpu.window.cyclesInConfig::<i> (controller.cc).
# Other per-config vectors (robFullCycles::<i>, iqFullCycles::<i>, ...) also end in
# "Cycles" but are NOT residency, so only these names are accepted.
_RES_RX = re.compile(r"^system\.cpu\.window\.(?:cyclesInConfig|residency|configCycles)::(\d+)$")


def window_residency(run_dir: Path, stats: dict, n_configs: int,
                     static_config: int | None = None) -> tuple[list[float], str]:
    """Return the fraction of the run spent in each window configuration.

    Sources, in order of preference: ``system.cpu.window.cyclesInConfig::<i>``
    (also ``residency``/``configCycles``) in ``stats``; ``window_trace.csv`` in
    ``run_dir`` (``d_cycles`` summed per ``config``); otherwise all time in the
    static configuration.

    Args:
        run_dir: gem5 output directory.
        stats: Output of :func:`parse_stats`.
        n_configs: Number of window configurations; out-of-range indices are
            ignored.
        static_config: Configuration used for the static fallback (clamped to
            the valid range); None means the largest.

    Returns:
        ``(residency, source)``: per-config fractions summing to 1 and one of
        ``"stats"``, ``"trace"``, ``"static"``.
    """
    res = [0.0] * n_configs
    for k, v in stats.items():
        m = _RES_RX.match(k)
        if m and int(m.group(1)) < n_configs:
            res[int(m.group(1))] += v
    if sum(res) > 0:
        return [r / sum(res) for r in res], "stats"
    trace = run_dir / "window_trace.csv"
    if trace.exists():
        try:
            df = load_window_trace(trace)
            g = df.groupby("config")["d_cycles"].sum()
            for c, v in g.items():
                if 0 <= int(c) < n_configs:
                    res[int(c)] += float(v)
            if sum(res) > 0:
                return [r / sum(res) for r in res], "trace"
        except Exception:  # noqa: BLE001 - fall through to static
            pass
    c = n_configs - 1 if static_config is None else min(max(static_config, 0), n_configs - 1)
    res[c] = 1.0
    return res, "static"


# ---------------------------------------------------------------------------
# McPAT
# ---------------------------------------------------------------------------

# McPAT output: processor-level leakage and runtime dynamic power (W)
_RE_SUB = re.compile(r"Subthreshold Leakage\s*=\s*([\d.e+\-]+)\s*W")
_RE_GATE = re.compile(r"Gate Leakage\s*=\s*([\d.e+\-]+)\s*W")
_RE_DYN = re.compile(r"Runtime Dynamic\s*=\s*([\d.e+\-]+)\s*W")


def parse_mcpat_output(text: str) -> tuple[float, float]:
    """Extract processor-level power from McPAT text output.

    Args:
        text: McPAT stdout/stderr.

    Returns:
        ``(leakage_w, dynamic_w)`` of the first (processor-level) block;
        leakage = subthreshold + gate leakage. Missing values count as 0.0.
    """
    sub, gate, dyn = _RE_SUB.findall(text), _RE_GATE.findall(text), _RE_DYN.findall(text)
    leak = (float(sub[0]) if sub else 0.0) + (float(gate[0]) if gate else 0.0)
    return leak, (float(dyn[0]) if dyn else 0.0)


def _size_bytes(s) -> int:
    """Convert a size (``"32kB"``, ``"1MiB"``, int) to bytes; unparsable strings give 1 MiB."""
    if isinstance(s, (int, float)):
        return int(s)
    m = re.match(r"^\s*([\d.]+)\s*([kKmMgG]?)i?[bB]?\s*$", str(s))
    if not m:
        return 1 << 20
    return int(float(m.group(1)) * {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30}[m.group(2).lower()])


def _clock_mhz(machine: dict) -> int:
    """Return the machine's CPU clock in MHz (``cpu.clock``; 2000 if absent/unparsable)."""
    clk = str(machine.get("cpu", {}).get("clock", "2GHz"))
    m = re.match(r"^\s*([\d.]+)\s*([GM])Hz", clk, re.I)
    if not m:
        return 2000
    return int(float(m.group(1)) * (1000 if m.group(2).upper() == "G" else 1))


def _set(root: ET.Element, comp_id: str, name: str, value, kind: str | None = None) -> None:
    """Set <param>/<stat> ``name`` of component ``comp_id`` (created if absent).

    Args:
        root: Root of the McPAT XML tree.
        comp_id: ``id`` attribute of the target ``<component>``.
        name: Parameter/stat name.
        value (object): New value (stringified).
        kind: ``"param"`` or ``"stat"`` to restrict the match and the tag of a
            created element; None matches either and creates a ``stat``.

    Raises:
        KeyError: If the template has no component ``comp_id``.
    """
    comp = root.find(f".//component[@id='{comp_id}']")
    if comp is None:
        raise KeyError(f"McPAT template has no component {comp_id}")
    for el in comp.findall("param") + comp.findall("stat"):
        if el.get("name") == name and (kind is None or el.tag == kind):
            el.set("value", str(value))
            return
    ET.SubElement(comp, kind or "stat", name=name, value=str(value))


def _cache_cfg(size, line: int, assoc: int, lat: int, policy: int) -> str:
    # capacity,block_width,associativity,bank,throughput,latency,output_width,policy
    """Format a McPAT cache config string.

    Fields: capacity, block_width, associativity, bank (1), throughput (1),
    latency, output_width (= line size), policy.

    Args:
        size (str | int): Cache size (see :func:`_size_bytes`).
        line: Line size in bytes.
        assoc: Associativity.
        lat: Latency in cycles.
        policy: McPAT cache policy field.

    Returns:
        Comma-separated config string.
    """
    return f"{_size_bytes(size)},{line},{assoc},1,1,{lat},{line},{policy}"


def build_mcpat_xml(ks: dict, machine: dict, cfg: dict, template: str | Path = MCPAT_TEMPLATE) -> str:
    """Build the McPAT input for one window configuration ``cfg`` ({rob, iq, lq, sq}).

    The template's technology node, clock, widths, window sizes, register
    files, caches and activity counters are overwritten from ``machine`` and
    ``ks``; counters gem5 does not provide are approximated from instruction
    counts (e.g. 20% FP if no FP count, 70/30 load/store split of L1D accesses,
    80/20 read/write split at L2).

    Args:
        ks: Output of :func:`key_stats`.
        machine: Machine dict (``cpu``, ``cache`` sections).
        cfg: One window-table entry; ``rob`` is required, ``iq``/``lq``/``sq``
            default to 64/32/32.
        template: McPAT XML template path.

    Returns:
        The XML document as a string.

    Raises:
        KeyError: If the template lacks a required component.
    """
    cpu = machine.get("cpu", {})
    cache = machine.get("cache", {})
    l1i, l1d, l2 = cache.get("l1i", {}), cache.get("l1d", {}), cache.get("l2", {})
    line = int(cache.get("cache_line_size", cache.get("line_size", 64)))
    mhz = _clock_mhz(machine)
    width = int(cpu.get("issue_width", 4))
    insts = max(int(ks["insts"]), 1)
    fp = int(ks["fp_insts"]) or int(insts * 0.2)
    it = int(ks["int_insts"]) or max(insts - fp, 1)
    loads = int(ks["loads"]) or int(ks["l1d_accesses"] * 0.7)
    stores = int(ks["stores"]) or int(ks["l1d_accesses"] * 0.3)
    cycles = max(int(ks["cycles"]), 1)
    l1d_acc = max(int(ks["l1d_accesses"]), loads + stores, 1)
    l1d_miss = int(ks["l1d_misses"])
    rd_frac = loads / max(loads + stores, 1)
    l2_acc = max(int(ks["l2_accesses"]), 1)
    l2_miss = int(ks["l2_misses"])

    root = ET.parse(str(template)).getroot()
    sysc, core = "system", "system.core0"
    for n, v in (("core_tech_node", TECH_NODE), ("target_core_clockrate", mhz),
                 ("number_of_L2Directories", 0), ("number_of_L1Directories", 0),
                 ("number_of_L3s", 0), ("Private_L2", 1), ("number_of_L2s", 1)):
        _set(root, sysc, n, v, "param")
    for n, v in (("total_cycles", cycles), ("idle_cycles", 0), ("busy_cycles", cycles)):
        _set(root, sysc, n, v, "stat")
    params = {
        "clock_rate": mhz, "fetch_width": int(cpu.get("fetch_width", width)),
        "decode_width": int(cpu.get("decode_width", width)), "issue_width": width,
        "peak_issue_width": width, "commit_width": int(cpu.get("commit_width", width)),
        "fp_issue_width": max(width // 2, 1),
        "instruction_window_size": int(cfg.get("iq", 64)),
        "fp_instruction_window_size": max(int(cfg.get("iq", 64)) // 2, 8),
        "ROB_size": int(cfg["rob"]),
        "load_buffer_size": int(cfg.get("lq", 32)), "store_buffer_size": int(cfg.get("sq", 32)),
        "phy_Regs_IRF_size": int(cpu.get("num_int_regs", 256)),
        "phy_Regs_FRF_size": int(cpu.get("num_fp_regs", 256)),
        "archi_Regs_IRF_size": 32, "archi_Regs_FRF_size": 32,
    }
    for n, v in params.items():
        _set(root, core, n, v, "param")
    stats = {
        "total_instructions": insts, "int_instructions": it, "fp_instructions": fp,
        "branch_instructions": int(ks["branches"]), "branch_mispredictions": int(ks["branch_mispred"]),
        "load_instructions": loads, "store_instructions": stores,
        "committed_instructions": insts, "committed_int_instructions": it,
        "committed_fp_instructions": fp,
        "pipeline_duty_cycle": round(min(1.0, insts / cycles / width), 4),
        "total_cycles": cycles, "idle_cycles": 0, "busy_cycles": cycles,
        "ROB_reads": max(int(ks["rob_reads"]) or insts, 1),
        "ROB_writes": max(int(ks["rob_writes"]) or insts, 1),
        "rename_reads": 2 * insts, "rename_writes": insts,
        "fp_rename_reads": 2 * fp, "fp_rename_writes": fp,
        "inst_window_reads": it, "inst_window_writes": it, "inst_window_wakeup_accesses": 2 * it,
        "fp_inst_window_reads": fp, "fp_inst_window_writes": fp,
        "fp_inst_window_wakeup_accesses": 2 * fp,
        "int_regfile_reads": 2 * it, "int_regfile_writes": it,
        "float_regfile_reads": 2 * fp, "float_regfile_writes": fp,
        "function_calls": 0, "context_switches": 0,
        "ialu_accesses": it, "fpu_accesses": fp, "mul_accesses": max(it // 10, 1),
        "cdb_alu_accesses": it, "cdb_fpu_accesses": fp, "cdb_mul_accesses": max(it // 10, 1),
    }
    for n, v in stats.items():
        _set(root, core, n, v, "stat")
    _set(root, "system.core0.icache", "icache_config",
         _cache_cfg(l1i.get("size", cache.get("l1i_size", "32kB")), line,
                    int(l1i.get("assoc", 4)), int(l1i.get("hit_latency_cycles", 2)), 0), "param")
    _set(root, "system.core0.icache", "read_accesses", max(int(ks["l1i_accesses"]), insts // 4, 1))
    _set(root, "system.core0.icache", "read_misses", int(ks["l1i_misses"]))
    _set(root, "system.core0.dcache", "dcache_config",
         _cache_cfg(l1d.get("size", cache.get("l1d_size", "32kB")), line,
                    int(l1d.get("assoc", 8)), int(l1d.get("hit_latency_cycles", 4)), 1), "param")
    for n, v in (("read_accesses", int(l1d_acc * rd_frac)), ("write_accesses", l1d_acc - int(l1d_acc * rd_frac)),
                 ("read_misses", int(l1d_miss * rd_frac)), ("write_misses", l1d_miss - int(l1d_miss * rd_frac))):
        _set(root, "system.core0.dcache", n, v)
    _set(root, "system.core0.dtlb", "total_accesses", l1d_acc)
    _set(root, "system.core0.itlb", "total_accesses", max(int(ks["l1i_accesses"]), insts // 4, 1))
    _set(root, "system.core0.BTB", "read_accesses", max(int(ks["branches"]), 1))
    _set(root, "system.L20", "L2_config",
         _cache_cfg(l2.get("size", cache.get("l2_size", "1MB")), line, int(l2.get("assoc", 16)),
                    int(l2.get("hit_latency_cycles", 20)), 1), "param")
    _set(root, "system.L20", "clockrate", mhz, "param")
    for n, v in (("read_accesses", int(l2_acc * 0.8)), ("write_accesses", l2_acc - int(l2_acc * 0.8)),
                 ("read_misses", int(l2_miss * 0.8)), ("write_misses", l2_miss - int(l2_miss * 0.8))):
        _set(root, "system.L20", n, v)
    if root.find(".//component[@id='system.NoC0']") is not None:
        _set(root, "system.NoC0", "total_accesses", l2_acc)
    if root.find(".//component[@id='system.mc']") is not None:
        _set(root, "system.mc", "memory_accesses", l2_miss)
        _set(root, "system.mc", "memory_reads", int(l2_miss * 0.8))
        _set(root, "system.mc", "memory_writes", l2_miss - int(l2_miss * 0.8))
    # canonical empty-element form (`<param .../>`; ElementTree writes " />")
    return ET.tostring(root, encoding="unicode").replace(" />", "/>")


def mcpat_power(ks: dict, machine: dict, cfg: dict, mcpat_bin: str,
                keep_dir: Path | None = None, timeout: int = 300) -> tuple[float, float]:
    """Run McPAT on one window configuration and return its power.

    McPAT runs with ``OMP_NUM_THREADS=MCPAT_THREADS`` and under
    ``flock HEAVY_LOCK`` unless the lock is disabled.

    Args:
        ks: Output of :func:`key_stats`.
        machine: Machine dict.
        cfg: Window-table entry (``rob``/``iq``/``lq``/``sq``).
        mcpat_bin: McPAT executable.
        keep_dir: If set, the XML and McPAT output are kept there as
            ``mcpat_rob<N>.xml``/``.txt``.
        timeout: Subprocess timeout in seconds.

    Returns:
        ``(leakage_w, dynamic_w)``.

    Raises:
        RuntimeError: If McPAT exits non-zero or its output cannot be parsed.
        subprocess.TimeoutExpired: If McPAT exceeds ``timeout``.
    """
    xml = build_mcpat_xml(ks, machine, cfg)
    with tempfile.TemporaryDirectory() as td:
        xf = Path(td) / "mcpat.xml"
        xf.write_text(xml)
        env = dict(os.environ, OMP_NUM_THREADS=MCPAT_THREADS)
        cmd = [mcpat_bin, "-infile", str(xf), "-print_level", "1"]
        if HEAVY_LOCK is not None:
            HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
            cmd = ["flock", str(HEAVY_LOCK)] + cmd
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        out = r.stdout + r.stderr
    if keep_dir is not None:
        (keep_dir / f"mcpat_rob{cfg['rob']}.xml").write_text(xml)
        (keep_dir / f"mcpat_rob{cfg['rob']}.txt").write_text(out)
    if r.returncode != 0:
        raise RuntimeError(f"McPAT exited {r.returncode}")
    leak, dyn = parse_mcpat_output(out)
    if leak == 0.0 and dyn == 0.0:
        raise RuntimeError("could not parse McPAT output")
    return leak, dyn


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def proxy_energy(ks: dict, table: list[dict], residency: list[float]) -> float:
    """Return the proxy-model energy of a run in joules (see the module docstring).

    Args:
        ks: Output of :func:`key_stats` (``sim_seconds``, ``insts``,
            ``l1d_misses``, ``l2_misses``).
        table: Window table (list of ``{rob, ...}`` dicts).
        residency: Per-config time fractions.

    Returns:
        Static energy weighted by residency plus per-instruction and
        per-miss dynamic energy.
    """
    t = ks["sim_seconds"]
    rob_max = max(c["rob"] for c in table)
    p = sum(f * P_CORE * (1 - W_FRAC + W_FRAC * c["rob"] / rob_max)
            for f, c in zip(residency, table))
    return t * p + ks["insts"] * E_INST + ks["l1d_misses"] * E_L1MISS + ks["l2_misses"] * E_L2MISS


def proxy_region_energy(cycles: float, insts: float, config: int, table: list[dict],
                        clock_hz: float = 2e9) -> float:
    """Return the proxy energy of a region executed entirely in one configuration.

    Used by sim/baselines/oracle/oracle_sweep.py; cache-miss terms are omitted
    because region_stats.csv has no per-region miss counts.

    Args:
        cycles: Region cycles.
        insts: Region instructions.
        config: Window configuration index.
        table: Window table (list of ``{rob, ...}`` dicts).
        clock_hz: Core clock in Hz.

    Returns:
        Energy in joules.
    """
    rob_max = max(c["rob"] for c in table)
    p = P_CORE * (1 - W_FRAC + W_FRAC * table[config]["rob"] / rob_max)
    return cycles / clock_hz * p + insts * E_INST


def estimate_run(run_dir: str | Path, machine: dict | str | Path, model: str = "auto",
                 static_config: int | None = None, mcpat_bin: str = MCPAT_BIN,
                 keep_mcpat: bool = False) -> dict:
    """Estimate energy, power, EDP and ED²P of one gem5 run.

    Args:
        run_dir: gem5 output directory (stats.txt, optional window_trace.csv).
        machine: Machine dict from ``whdata.load_machine`` or a path to its JSON.
        model: ``"mcpat"``, ``"proxy"`` or ``"auto"`` (McPAT if the binary is
            executable and the template exists, else proxy).
        static_config: Static configuration for the residency fallback.
        mcpat_bin: McPAT executable.
        keep_mcpat: Keep McPAT XML/output in ``run_dir``.

    Returns:
        Dict with ``run_dir``, ``machine``, ``model`` (the one used),
        ``residency``, ``residency_source``, ``sim_seconds``, ``cycles``,
        ``insts``, ``ipc``, ``energy_j``, ``power_w``, ``edp``, ``ed2p`` and,
        for McPAT, ``leakage_w``/``dynamic_w`` (residency-weighted).
    """
    run_dir = Path(run_dir)
    if not isinstance(machine, dict):
        machine = load_machine(machine)
    table = machine["window_table"]
    stats = parse_stats(run_dir / "stats.txt")
    ks = key_stats(stats)
    if not ks["sim_seconds"] and ks["cycles"]:
        ks["sim_seconds"] = ks["cycles"] / (_clock_mhz(machine) * 1e6)
    res, res_src = window_residency(run_dir, stats, len(table), static_config)
    use = model
    if model == "auto":
        use = "mcpat" if (Path(mcpat_bin).is_file() and os.access(mcpat_bin, os.X_OK)
                          and Path(MCPAT_TEMPLATE).is_file()) else "proxy"
    out = {"run_dir": str(run_dir), "machine": machine.get("name"), "model": use,
           "residency": [round(r, 6) for r in res], "residency_source": res_src,
           **{k: ks[k] for k in ("sim_seconds", "cycles", "insts", "ipc")}}
    t = ks["sim_seconds"]
    if use == "mcpat":
        e, leak_w, dyn_w = 0.0, 0.0, 0.0
        for f, c in zip(res, table):
            if f <= 0:
                continue
            leak, dyn = mcpat_power(ks, machine, c, mcpat_bin, run_dir if keep_mcpat else None)
            e += f * (leak + dyn) * t
            leak_w += f * leak
            dyn_w += f * dyn
        out.update(leakage_w=leak_w, dynamic_w=dyn_w)
    else:
        e = proxy_energy(ks, table, res)
    out["energy_j"] = e
    out["power_w"] = e / t if t else 0.0
    out["edp"] = e * t
    out["ed2p"] = e * t * t
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _static_from_variant(variant: str) -> int | None:
    """Return ``i`` for a ``static_c<i>`` variant name, else None."""
    m = re.match(r"^static_c(\d+)", variant)
    return int(m.group(1)) if m else None


def find_runs(root: Path) -> list[tuple[str, str, str, Path]]:
    """List the gem5 runs under a results tree.

    Layout: ``<root>/<machine>/<kernel>/<variant>/stats.txt``.

    Args:
        root: Results root (e.g. results/gem5).

    Returns:
        Sorted ``(machine, kernel, variant, run_dir)`` tuples.
    """
    runs = []
    for st in sorted(root.glob("*/*/*/stats.txt")):
        v = st.parent
        runs.append((v.parent.parent.name, v.parent.name, v.name, v))
    return runs


def parse_args(argv=None):
    """Parse the command line (``--results-root`` or ``--run-dir`` required)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--results-root", type=Path, help="results/gem5 tree")
    g.add_argument("--run-dir", type=Path, help="one gem5 outdir")
    p.add_argument("--machine", type=Path, help="machine JSON (for --run-dir)")
    p.add_argument("--machines-dir", type=Path, default=REPO / "sim" / "machines")
    p.add_argument("--config", type=int, default=None, help="static config of --run-dir")
    p.add_argument("--model", default="auto", choices=["auto", "mcpat", "proxy"])
    p.add_argument("--mcpat-bin", default=MCPAT_BIN)
    p.add_argument("--keep-mcpat", action="store_true", help="keep McPAT XML/output in run dirs")
    p.add_argument("--force", action="store_true", help="recompute existing energy.json")
    p.add_argument("--out-csv", type=Path, help="default: <results-root>/energy.csv")
    p.add_argument("--no-lock", action="store_true",
                   help="do not run McPAT under flock $WINHINT_BUILD/.heavy.lock")
    return p.parse_args(argv)


#: column order of energy.csv (--results-root)
CSV_FIELDS = ["machine", "kernel", "variant", "model", "sim_seconds", "cycles", "insts", "ipc",
              "energy_j", "power_w", "edp", "ed2p", "residency_source", "residency", "error"]


def main(argv=None) -> int:
    """CLI entry point.

    With ``--run-dir`` estimates one run and writes ``<run_dir>/energy.json``.
    With ``--results-root`` processes every run (reusing an existing
    energy.json unless ``--force``) and writes a CSV (``--out-csv``, default
    ``<root>/energy.csv``); per-run errors are recorded in the ``error`` column.

    Args:
        argv (list[str] | None): Argument list; None means ``sys.argv[1:]``.

    Returns:
        Exit status (0).
    """
    global HEAVY_LOCK
    a = parse_args(argv)
    if a.no_lock:
        HEAVY_LOCK = None
    if a.run_dir:
        mpath = a.machine or a.machines_dir / "riscv_ooo.json"
        r = estimate_run(a.run_dir, mpath, a.model, a.config, a.mcpat_bin, a.keep_mcpat)
        (a.run_dir / "energy.json").write_text(json.dumps(r, indent=2) + "\n")
        print(json.dumps(r, indent=2))
        return 0
    rows = []
    machines: dict[str, dict] = {}
    for mname, kernel, variant, d in find_runs(a.results_root):
        row = {"machine": mname, "kernel": kernel, "variant": variant}
        cache = d / "energy.json"
        try:
            if cache.exists() and not a.force:
                r = json.loads(cache.read_text())
            else:
                if mname not in machines:
                    machines[mname] = load_machine(a.machines_dir / f"{mname}.json")
                r = estimate_run(d, machines[mname], a.model, _static_from_variant(variant),
                                 a.mcpat_bin, a.keep_mcpat)
                cache.write_text(json.dumps(r, indent=2) + "\n")
            row.update({k: r.get(k) for k in CSV_FIELDS if k in r})
            row["residency"] = " ".join(f"{x:.3f}" for x in r.get("residency", []))
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)
        rows.append(row)
        print(f"[{'ERR' if row.get('error') else 'OK '}] {mname}/{kernel}/{variant} "
              f"E={row.get('energy_j', float('nan'))} {row.get('error', '')}")
    out = a.out_csv or a.results_root / "energy.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"[OK] {len(rows)} runs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
