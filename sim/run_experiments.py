#!/usr/bin/env python3
r"""
run_experiments.py - WinHint gem5 evaluation matrix (PROPOSAL Phase E).

Runs every baseline of PROPOSAL §4.1 plus WinHint and WinHint+HW on every
machine (sim/machines/*.json) through sim/se.py and the RISCV_winhint gem5
build, using only the flags of docs/interfaces.md §4:

  variant               id          binary (Makefile variant)  window policy
  --------------------  ----------  -------------------------  ----------------------
  static_c<i>           B0          plain                      static, initial=i
                                    (every config of the machine's window table:
                                     c0 = small ... c<n-1> = large)
  oracle_hinted         B1          oracle_hinted              hint
  occupancy             B2          plain                      occupancy
  mlp                   B3          plain                      mlp
  bbv                   B4          plain                      bbv
  lut                   B5          plain                      lut (LUT trained on this machine)
  lut_xfer              B5          plain                      lut (LUT of --lut-ref-machine;
                                                               portability, other machines only)
  jones                 B6          jones                      hint, structs=iq (IQ only,
                                                               as published)
  jones_full            B6          jones_full                 hint (extended to ROB/LSQ)
  pgo                   B7          pgo                        hint (profiled on `small`)
  clairvoyance          B8          clairvoyance               static, largest config
  winhint_clairvoyance  B8          winhint_clairvoyance       hint
  ltp                   B9          plain                      ltp (Long-Term Parking), largest config
  ltp_c<i>              B9          plain                      ltp, initial=i (equal resources
                                                               to static_c<i>, i < largest)
  winhint               WinHint     winhint                    hint
  winhint_hw            WinHint+HW  winhint                    hybrid (hint + HW override)
  winhint_nop           overhead    winhint                    static, largest config: the
                                                               hints are decoded but ignored, so
                                                               cycles vs static_c<max> = the
                                                               dynamic hint overhead

A variant whose policy sim/se.py does not accept yet (its WINDOW_POLICIES
list; e.g. ``ltp`` until B9 lands) is reported as MISSING, not run.

Binaries (interfaces.md §1): $WINHINT_BUILD/benchmarks/riscv/<variant><suffix>/
[<machine>/]<kernel>. The <machine>/ level is looked up first: hinted variants
(winhint, oracle_hinted, pgo, jones*, winhint_clairvoyance) depend on the
machine JSON they were compiled for (MACHINE=...), so the portability study
needs one build per machine; for these the build without the <machine>/ level
is used only if it was compiled for this machine (the "target" of its
<kernel>.*.json sidecar, else the default machine riscv_ooo), otherwise the run
is reported as missing. --bin-suffix selects a Makefile build configuration (-O3,
-nounroll, -tiled, ...). Missing binaries are reported and skipped. B5 LUTs
are read from <lut-root>/<machine>/lut.txt (sim/baselines/lut pipeline).

gem5 = $WINHINT_GEM5 or $WINHINT_BUILD/gem5/src/build/RISCV_winhint/gem5.opt.
Every gem5 run is wrapped in ``flock $WINHINT_BUILD/.heavy.lock`` (one heavy
job at a time on the host; --no-lock disables it for tiny runs). Default
--jobs 1, maximum 2.

Sensitivity (PROPOSAL Phase E, cache size and memory latency): ``--sweep
PARAM=v1,v2,...`` (repeatable; PARAM in SWEEP_PARAMS: l2_size, l1d_size,
mem_extra_latency_ns) adds, for every selected machine M and value v, a point
named ``M+PARAM=v``: a derived machine JSON (results/gem5/M+PARAM=v/machine.json)
passed to se.py --machine. Binaries and LUTs are those of M (the compiler is
not re-run: this measures how robust one compiled binary is). summary.csv
gets base_machine, sens_param, sens_value, l2_kb and mem_latency_cycles.

Results: results/gem5/<machine>/<kernel>/<variant>/ (gem5 outdir: stats.txt,
config.ini, simout, simerr, optional window_trace.csv, region_stats.csv) plus
run.json (command, status, wall time) and gem5.log. A run whose run.json says
"ok" and whose stats.txt exists is skipped (resumable); --force reruns.
Inputs other than `large` go to <variant>.<input>/.

After the runs (or with --collect-only) a summary with IPC, window counters
and energy/EDP/ED²P (sim/estimate_energy.py; McPAT at $WINHINT_BUILD/mcpat/mcpat
if built, else the documented proxy) is written to results/gem5/summary.csv.

Run length (sim/run_lengths.json, sim/run_lengths.py): the budget of every
kernel/input comes from that versioned file (``--run-lengths``; ``--run-length
full`` ignores it). ``small`` runs in full. ``large`` uses region-aligned
stratified sampling, executed in three phases that share their products:
  P  region profile: one functional pass of the kernel's ``oracle<suffix>`` build
     (se.py --profile-regions) -> results/runlen/oracle<suffix>/<kernel>.<input>.json
  C  checkpoints: one functional pass per binary (se.py --take-checkpoints) at every
     sample start -> $WINHINT_BUILD/ckpt/<binary sha1>-<input>-<mem>-<starts hash>/,
     shared by every variant and machine running that binary
  R  samples: <outdir>/sNN/ (restore + --warmup-insts + --maxinsts; hint-driven
     policies start in the config of the last setwin before the checkpoint), then
     merged into <outdir>/{stats.txt (whole-program estimate), stats.measured.txt,
     region_stats.csv, window_trace.csv, runlength.json}.
--dry-run lists the pending P/C passes and, per run, its mode, sample count and
detailed instruction budget. The B1 oracle sweep and the baseline tuning use the
same plans (same profile, same starts and budgets).

sim/run_baseline.py is kept as a thin wrapper around ``--policies static``.

Usage (host, `winhint` env active):
  python sim/run_experiments.py --dry-run
  python sim/run_experiments.py --kernels encoder_bert_tiny_infer \
      --policies static winhint lut --machines riscv_ooo
  python sim/run_experiments.py --policies B2 B3 --window-args mlp:mlp_thr=2.0
  python sim/run_experiments.py --machines riscv_ooo --policies static winhint mlp \
      --sweep l2_size=256kB,512kB,2MB --sweep mem_extra_latency_ns=20,40,80
  python sim/run_experiments.py --collect-only
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))
sys.path.insert(0, str(REPO / "sim" / "baselines" / "lut"))
import run_lengths as rl  # noqa: E402
from estimate_energy import estimate_run, key_stats, parse_stats  # noqa: E402
from whdata import load_machine  # noqa: E402

BUILD = Path(os.environ.get("WINHINT_BUILD", str(REPO / "build")))
GEM5_BIN = os.environ.get("WINHINT_GEM5", str(BUILD / "gem5" / "src" / "build" / "RISCV_winhint"
                                              / "gem5.opt"))
#: gem5 config script run by every simulation
SE_PY = REPO / "sim" / "se.py"
BIN_ROOT = Path(os.environ.get("WINHINT_BIN_ROOT", str(BUILD / "benchmarks" / "riscv")))
MACHINES_DIR = REPO / "sim" / "machines"
RESULTS_ROOT = REPO / "results" / "gem5"
LUT_ROOT = REPO / "results" / "b5"
DEFAULT_MACHINE = "riscv_ooo"
#: flock file serialising heavy jobs on the host (interfaces.md §1)
HEAVY_LOCK = BUILD / ".heavy.lock"
RUN_LENGTHS = rl.CONFIG
#: tuned baseline/compiler parameters (sim/baselines/tune/tune_baselines.py --select)
TUNED_DEFAULT = REPO / "results" / "tune" / "tuned.json"
RUNLEN_ROOT = REPO / "results" / "runlen"
#: phase-C checkpoint directories (shared across variants/machines)
CKPT_ROOT = BUILD / "ckpt"
MAX_JOBS = 2  # project-wide limit (6 GB RAM); heavy runs are serialised by HEAVY_LOCK anyway


def all_kernels() -> list[str]:
    """Return the names of all kernels (stems of ``benchmarks/*_infer.c``), sorted."""
    return sorted(p.stem for p in (REPO / "benchmarks").glob("*_infer.c"))


def se_policies(se_py: Path | None = None) -> set[str] | None:
    """Return the window policies accepted by sim/se.py (its WINDOW_POLICIES list).

    The list literal is read from the source text (se.py cannot be imported
    outside gem5). Variants whose policy se.py does not accept yet (e.g. B9
    ``ltp``) are reported as missing instead of failing.

    Args:
        se_py: Path of se.py; None means ``SE_PY``.

    Returns:
        The policy names, or None if they cannot be determined.
    """
    import ast
    import re
    try:
        text = (se_py or SE_PY).read_text()
        m = re.search(r"^WINDOW_POLICIES\s*=\s*(\[[^\]]*\])", text, re.M)
        return set(ast.literal_eval(m.group(1))) if m else None
    except (OSError, ValueError, SyntaxError):
        return None


# ---------------------------------------------------------------------------
# Variant table
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    """One row of the variant table (see the module docstring).

    Attributes:
        name: Variant name; also the results subdirectory (plus suffixes).
        baseline: Baseline id (``B0``..``B9``, ``WinHint``, ``WinHint+HW``,
            ``overhead``).
        binary: Benchmark build variant (directory under the binary root).
        policy: se.py ``--window-policy``.
        initial: ``--window-initial`` config index; None means the largest.
        lut: ``"own"`` (LUT of this machine), ``"ref"`` (LUT of
            ``--lut-ref-machine``) or None (no LUT).
        tags: Extra names ``--policies`` can select the variant by.
        window_args: Built-in se.py ``--window-args`` (merged with the CLI's).
    """
    name: str
    baseline: str
    binary: str
    policy: str
    initial: int | None = None      # None -> largest config
    lut: str | None = None          # "own" | "ref"
    tags: tuple[str, ...] = field(default_factory=tuple)
    window_args: str | None = None  # built-in se.py --window-args (merged with the CLI's)


def variants_for(machine: dict, ref_machine: str) -> list[Variant]:
    """Return every variant to run on one machine.

    ``static_c<i>`` and ``ltp_c<i>`` are generated from the machine's window
    table; ``lut_xfer`` only exists on machines other than ``ref_machine``.

    Args:
        machine: Machine dict from ``whdata.load_machine`` (``name``,
            ``window_table``).
        ref_machine: Machine whose LUT ``lut_xfer`` uses.

    Returns:
        The variants, in table order.
    """
    n = len(machine["window_table"])
    vs = [Variant(f"static_c{i}", "B0", "plain", "static", i, tags=("static", "b0"))
          for i in range(n)]
    vs += [
        Variant("oracle_hinted", "B1", "oracle_hinted", "hint", tags=("oracle", "b1")),
        Variant("occupancy", "B2", "plain", "occupancy", tags=("b2", "hw")),
        Variant("mlp", "B3", "plain", "mlp", tags=("b3", "hw")),
        Variant("bbv", "B4", "plain", "bbv", tags=("b4", "hw")),
        Variant("lut", "B5", "plain", "lut", lut="own", tags=("b5", "hw")),
    ]
    if machine["name"] != ref_machine:
        vs.append(Variant("lut_xfer", "B5", "plain", "lut", lut="ref", tags=("b5", "hw", "xfer")))
    vs += [
        # B6 as published resizes only the IQ: the hint policy caps just the IQ
        # (window_args structs=iq, hint_policy.cc); jones_full moves the whole window
        Variant("jones", "B6", "jones", "hint", tags=("b6",), window_args="structs=iq"),
        Variant("jones_full", "B6", "jones_full", "hint", tags=("b6",)),
        Variant("pgo", "B7", "pgo", "hint", tags=("b7",)),
        Variant("clairvoyance", "B8", "clairvoyance", "static", None, tags=("b8",)),
        Variant("winhint_clairvoyance", "B8", "winhint_clairvoyance", "hint", tags=("b8", "winhint")),
        Variant("ltp", "B9", "plain", "ltp", tags=("b9", "hw")),
        Variant("winhint", "WinHint", "winhint", "hint", tags=("winhint",)),
        Variant("winhint_hw", "WinHint+HW", "winhint", "hybrid", tags=("winhint", "hybrid")),
        Variant("winhint_nop", "overhead", "winhint", "static", None, tags=("overhead",)),
    ]
    # B9 on equal resources: LTP's IQ/LSQ caps follow --window-initial, so pair
    # each smaller static_c<i> with an ltp_c<i> (``ltp`` itself = largest config)
    vs += [Variant(f"ltp_c{i}", "B9", "plain", "ltp", i, tags=("b9", "hw"))
           for i in range(n - 1)]
    return vs


#: binary variants whose code depends on the machine JSON they were compiled for
MACHINE_DEPENDENT = {"winhint", "oracle_hinted", "pgo", "jones", "jones_full",
                     "winhint_clairvoyance"}


# ---------------------------------------------------------------------------
# Sensitivity points (derived machines)
# ---------------------------------------------------------------------------

def _clock_ghz(machine: dict) -> float:
    """Return the machine's CPU clock in GHz (``cpu.clock``; 2.0 if absent/unparsable)."""
    import re
    m = re.match(r"^\s*([\d.]+)\s*([GM])Hz", str(machine.get("cpu", {}).get("clock", "2GHz")), re.I)
    if not m:
        return 2.0
    return float(m.group(1)) / (1.0 if m.group(2).upper() == "G" else 1000.0)


def _set_cache_size(level: str):
    """Return a sweep setter that sets the size of cache ``level`` (e.g. ``"l2"``).

    The returned function writes both ``cache.<level>.size`` and
    ``cache.<level>_size`` of a machine dict in place.
    """
    def f(m: dict, v: str) -> None:
        """Set the cache size of ``m`` to ``v`` in place."""
        c = m.setdefault("cache", {})
        c.setdefault(level, {})["size"] = v
        c[f"{level}_size"] = v
    return f


def _set_mem_extra(m: dict, v: str) -> None:
    """Set ``memory.extra_latency_ns`` of machine ``m`` to ``v`` in place.

    If the machine has ``memory.latency_cycles`` (cost-model estimate of an
    L2-miss round trip), it is shifted by the latency change converted to
    cycles at the machine's clock.
    """
    mem = m.setdefault("memory", {})
    old = float(mem.get("extra_latency_ns", 0) or 0)
    new = float(v)
    mem["extra_latency_ns"] = new
    if "latency_cycles" in mem:  # cost-model estimate of an L2-miss round trip
        mem["latency_cycles"] = int(round(float(mem["latency_cycles"])
                                          + (new - old) * _clock_ghz(m)))


#: --sweep parameter -> in-place setter on a machine dict
SWEEP_PARAMS = {"l2_size": _set_cache_size("l2"), "l1d_size": _set_cache_size("l1d"),
                "mem_extra_latency_ns": _set_mem_extra}


def parse_sweeps(items: list[str] | None) -> list[tuple[str, str]]:
    """Parse ``--sweep PARAM=v1,v2`` options (repeatable).

    Args:
        items: Raw ``--sweep`` values; None for none.

    Returns:
        ``[(param, value), ...]`` in command-line order.

    Raises:
        SystemExit: If an item is malformed or PARAM is not in ``SWEEP_PARAMS``.
    """
    out = []
    for it in items or []:
        par, sep, vals = it.partition("=")
        if not sep or par not in SWEEP_PARAMS or not vals:
            raise SystemExit(f"--sweep expects PARAM=v1,v2 with PARAM in {sorted(SWEEP_PARAMS)}; "
                             f"got {it!r}")
        out += [(par, v.strip()) for v in vals.split(",") if v.strip()]
    return out


def derive_machine(raw: dict, param: str, value: str) -> dict:
    """Return a sensitivity-point copy of a machine JSON with one parameter changed.

    Args:
        raw: Machine JSON as read from disk (not normalised).
        param: Key of ``SWEEP_PARAMS``.
        value: New value (string, as given on the command line).

    Returns:
        A deep copy named ``<base>+<param>=<value>`` with a ``_derived``
        record (``base``, ``param``, ``value``).
    """
    m = copy.deepcopy(raw)
    SWEEP_PARAMS[param](m, value)
    m["name"] = f"{raw.get('name', 'machine')}+{param}={value}"
    m["_derived"] = {"base": raw.get("name"), "param": param, "value": value}
    return m


def machine_point_params(m: dict) -> dict:
    """Return the L2 size (kB) and memory latency (cycles) of a machine, for summary.csv.

    Returns:
        ``{"l2_kb", "mem_latency_cycles"}``; empty strings when unknown.
    """
    import re
    cache = m.get("cache", {})
    l2 = str(cache.get("l2", {}).get("size", cache.get("l2_size", "")))
    mm = re.match(r"^\s*([\d.]+)\s*([kKmMgG]?)i?[bB]?\s*$", l2)
    l2_kb = (float(mm.group(1)) * {"": 1 / 1024, "k": 1, "m": 1024, "g": 1024 ** 2}[
        mm.group(2).lower()]) if mm else ""
    return {"l2_kb": l2_kb, "mem_latency_cycles": m.get("memory", {}).get("latency_cycles", "")}


def select(vs: list[Variant], wanted: list[str] | None) -> list[Variant]:
    """Filter variants by ``--policies``.

    Args:
        vs: Candidate variants.
        wanted: Variant names, baseline ids or tags (case-insensitive for names
            and ids; tags are matched lower-cased); empty/None selects all.

    Returns:
        The matching variants, in their original order.
    """
    if not wanted:
        return vs
    w = {x.lower() for x in wanted}
    return [v for v in vs if v.name.lower() in w or v.baseline.lower() in w
            or w.intersection(v.tags)]


# ---------------------------------------------------------------------------
# Run description
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """One (machine, kernel, variant) simulation and everything needed to run it.

    Attributes:
        machine: Machine or sensitivity-point name (results subdirectory).
        machine_json: Machine JSON passed to se.py (derived JSON for a point).
        kernel: Kernel name.
        variant: The variant.
        input: Input size (``small``/``large``).
        outdir: gem5 output directory.
        binary: Benchmark binary, or None if missing.
        lut: B5 LUT file, or None.
        cmd: gem5 command (the first sample's command for a sampled run).
        missing: Reasons the run cannot be done (empty if runnable).
        lock: Heavy-job lock file, or None (``--no-lock``).
        machine_data: Derived machine JSON to write before running.
        cmd_kw: :func:`gem5_cmd` arguments (sample commands are rebuilt).
        base_machine_json: Real machine file (functional passes).
        runlen: Run-length spec (sim/run_lengths.json).
        plan: Run-length plan, once computable.
        samples: :class:`SampleRun` list of a sampled run.
        profile: Phase P dependency.
        ckpt: Phase C dependency.
        pending: Why the plan is not computable yet (profile missing).
    """
    machine: str
    machine_json: Path
    kernel: str
    variant: Variant
    input: str
    outdir: Path
    binary: Path | None
    lut: Path | None
    cmd: list[str]
    missing: list[str]
    lock: Path | None = None
    machine_data: dict | None = None   # derived machine JSON to write before running
    cmd_kw: dict | None = None         # gem5_cmd() arguments (sample commands are rebuilt)
    base_machine_json: Path | None = None  # real machine file (functional passes)
    runlen: dict | None = None         # run-length spec (sim/run_lengths.json)
    plan: rl.Plan | None = None
    samples: list = field(default_factory=list)   # [SampleRun]
    profile: AuxJob | None = None      # phase P dependency
    ckpt: AuxJob | None = None         # phase C dependency
    pending: str = ""                  # plan not computable yet (profile missing)

    @property
    def key(self) -> str:
        """Return ``<machine>/<kernel>/<outdir name>``, used in progress messages."""
        return f"{self.machine}/{self.kernel}/{self.outdir.name}"

    @property
    def mode(self) -> str:
        """Return the run-length mode: the plan's mode, ``pending`` or ``full``."""
        return self.plan.mode if self.plan else ("pending" if self.pending else "full")


@dataclass
class AuxJob:
    """Functional gem5 pass whose product other runs need (phase P or C).

    Attributes:
        kind: ``"profile"`` or ``"checkpoint"``.
        product: Profile JSON or checkpoint directory.
        outdir: gem5 outdir of the pass.
        cmd: gem5 command.
        lock: Heavy-job lock file, or None.
        starts: Checkpoint instruction counts (checkpoint jobs).
    """
    kind: str            # "profile" | "checkpoint"
    product: Path        # profile JSON | checkpoint directory
    outdir: Path         # gem5 outdir
    cmd: list[str]
    lock: Path | None = None
    starts: list[int] = field(default_factory=list)

    def done(self) -> bool:
        """Return True if the product exists and is complete.

        A profile is done when its JSON exists; a checkpoint job when its
        ``checkpoints.json`` requested exactly ``starts``.
        """
        if self.kind == "profile":
            return self.product.is_file()
        cj = self.product / "checkpoints.json"
        if not cj.is_file():
            return False
        try:
            d = json.loads(cj.read_text())
        except ValueError:
            return False
        return sorted(d.get("requested", [])) == sorted(self.starts)


@dataclass
class SampleRun:
    """One sample of a sampled run.

    Attributes:
        sample: The run-length sample.
        outdir: Sample output directory (``<run outdir>/sNN``).
        cmd: gem5 command of the sample.
    """
    sample: rl.Sample
    outdir: Path
    cmd: list[str]


def binary_target(binary: Path) -> str | None:
    """Return the machine a hinted binary was compiled for.

    This is the "target" of its compiler sidecar (<kernel>.winhint.json,
    .jones.json, .jones_full.json, ...; ``*.regions.json`` is ignored).

    Args:
        binary: Benchmark binary path.

    Returns:
        The target machine name, or None if no sidecar names one.
    """
    for side in sorted(binary.parent.glob(binary.name + ".*.json")):
        if side.name.endswith(".regions.json"):
            continue
        try:
            t = json.loads(side.read_text()).get("target")
        except (OSError, ValueError, AttributeError):
            continue
        if t:
            return str(t)
    return None


def find_binary(bin_root: Path, variant: str, machine: str, kernel: str,
                strict: bool | None = None) -> Path | None:
    """Locate a benchmark binary: <variant>/<machine>/<kernel>, else <variant>/<kernel>.

    For machine-dependent variants (``strict``, default: by variant name) the
    latter is accepted only if it was compiled for ``machine`` (its sidecar
    target; no sidecar = default machine).

    Args:
        bin_root: Binary root ($WINHINT_BUILD/benchmarks/riscv).
        variant: Binary directory name (build variant plus suffixes).
        machine: Machine name.
        kernel: Kernel name.
        strict: Force/disable the compiled-for-this-machine check; None decides
            from ``MACHINE_DEPENDENT`` (Makefile suffix after ``-`` stripped).

    Returns:
        The binary path, or None.
    """
    p = bin_root / variant / machine / kernel
    if p.is_file():
        return p
    p = bin_root / variant / kernel
    if not p.is_file():
        return None
    if strict is None:
        base = variant.split("-")[0]  # strip the Makefile build-config suffix (-O3, ...)
        strict = base in MACHINE_DEPENDENT
    if strict and (binary_target(p) or DEFAULT_MACHINE) != machine:
        return None
    return p


def gem5_cmd(gem5: str, outdir: Path, machine_json: Path, binary: Path | str, input_size: str,
             policy: str, initial: int, period: int | None = None, lut: Path | None = None,
             trace: bool = False, max_insts: int | None = None,
             extra: list[str] | None = None) -> list[str]:
    """Build a gem5 command line using only the sim/se.py flags of interfaces.md §4.

    Args:
        gem5: gem5 binary.
        outdir: gem5 ``--outdir``.
        machine_json: se.py ``--machine``.
        binary: se.py ``--cmd``.
        input_size: se.py ``--options`` (input size).
        policy: ``--window-policy``.
        initial: ``--window-initial``.
        period: ``--window-period``; omitted if falsy.
        lut: ``--window-lut``; omitted if None.
        trace: Add ``--window-trace``.
        max_insts: ``--maxinsts``; omitted if falsy.
        extra: Additional se.py arguments appended at the end.

    Returns:
        The argument list (stdout/stderr redirected into the outdir).
    """
    cmd = [gem5, f"--outdir={outdir}", "--redirect-stdout", "--redirect-stderr", str(SE_PY),
           "--machine", str(machine_json), "--cmd", str(binary), "--options", input_size,
           "--window-policy", policy, "--window-initial", str(initial)]
    if period:
        cmd += ["--window-period", str(period)]
    if lut is not None:
        cmd += ["--window-lut", str(lut)]
    if trace:
        cmd += ["--window-trace"]
    if max_insts:
        cmd += ["--maxinsts", str(max_insts)]
    return cmd + list(extra or [])


def parse_window_args(items: list[str] | None) -> dict[str, str]:
    """Parse ``--window-args KEY:k=v,k=v`` options (repeatable).

    KEY is a window policy (every variant using it) or a variant name (that
    variant only); a later option for the same KEY replaces an earlier one.

    Returns:
        ``{KEY: "k=v,k=v"}``.

    Raises:
        SystemExit: If an item has no ``:``.
    """
    out = {}
    for it in items or []:
        pol, sep, args = it.partition(":")
        if not sep:
            raise SystemExit(f"--window-args expects POLICY:k=v,...; got {it!r}")
        out[pol] = args
    return out


def variant_window_args(v: Variant, wargs: dict[str, str],
                        tuned: dict[str, str] | None = None) -> str | None:
    """Return the merged se.py ``--window-args`` of a variant.

    Sources, in order: built-in args of the variant, then the tuned args of its
    policy and of its name (tuned.json), then --window-args of its policy and
    of its name; a later key=value overrides an earlier one.

    Args:
        v: The variant.
        wargs: Output of :func:`parse_window_args`.
        tuned: Tuned ``window_args`` for this machine (:func:`tuned_for`).

    Returns:
        ``"k=v,..."`` or None if empty.
    """
    merged: dict[str, str] = {}
    tuned = tuned or {}
    for spec in (v.window_args, tuned.get(v.policy), tuned.get(v.name),
                 wargs.get(v.policy), wargs.get(v.name)):
        for kv in (spec or "").split(","):
            k, sep, val = kv.partition("=")
            if k.strip() and sep:
                merged[k.strip()] = val.strip()
    return ",".join(f"{k}={val}" for k, val in merged.items()) or None


def load_tuned(a) -> dict:
    """Load tuned.json of the baseline tuning.

    Uses ``a.tuned``; default ``TUNED_DEFAULT`` if it exists; ``--tuned none``
    disables.

    Args:
        a (argparse.Namespace): Parsed arguments (``tuned`` attribute optional).

    Returns:
        The tuned dict with ``_path`` set, or ``{}``.
    """
    p = getattr(a, "tuned", None)
    if p is not None and str(p) == "none":
        return {}
    if p is None:
        p = TUNED_DEFAULT if TUNED_DEFAULT.is_file() else None
    if p is None:
        return {}
    d = json.loads(Path(p).read_text())
    d["_path"] = str(p)
    return d


def tuned_for(tuned: dict, section: str, machine: str) -> dict:
    """Return the machine-specific entries of a tuned.json section over its "*" entries."""
    sec = tuned.get(section, {}) or {}
    out = dict(sec.get("*", {}) or {})
    out.update(sec.get(machine, {}) or {})
    return out


def tuned_compiler_mismatch(tuned: dict, binary: Path, base: str) -> str | None:
    """Check a WinHint-compiled binary against the tuned compiler knobs.

    Only ``winhint`` and ``winhint_clairvoyance`` binaries are checked, and
    only if tuned.json has ``compiler.winhint.sidecar``.

    Args:
        tuned: Output of :func:`load_tuned`.
        binary: Binary to check (its ``<name>.winhint.json`` sidecar is read).
        base: Build variant of the binary.

    Returns:
        A message (with the rebuild command) if the sidecar is missing or its
        knobs differ, else None.
    """
    want = ((tuned.get("compiler", {}) or {}).get("winhint", {}) or {}).get("sidecar")
    if not want or base not in ("winhint", "winhint_clairvoyance"):
        return None
    side = binary.parent / f"{binary.name}.winhint.json"
    try:
        got = json.loads(side.read_text())
    except (OSError, ValueError):
        return f"no sidecar {side} to check the tuned WinHint knobs"
    bad = {k: (got.get(k), v) for k, v in want.items()
           if got.get(k) is None or abs(float(got[k]) - float(v)) > 1e-9}
    if not bad:
        return None
    make = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in
                    ((tuned["compiler"]["winhint"].get("make") or {}).items()))
    return (f"binary {binary} not built with the tuned WinHint knobs {bad} "
            f"(rebuild: make -C benchmarks VARIANT={base} {make})")


def build_runs(a) -> list[Run]:
    """Build the run matrix selected by the command line.

    Expands machines (plus ``--sweep`` points) x variants (``--policies``) x
    kernels, resolves binaries and LUTs, records missing inputs, builds the
    gem5 commands and attaches run-length plans (:func:`apply_run_length`).

    Args:
        a (argparse.Namespace): Parsed arguments (:func:`parse_args`).

    Returns:
        All runs, including the ones that cannot be run (``missing``).
    """
    runs = []
    supported = se_policies()
    wargs = parse_window_args(getattr(a, "window_args", None))
    tuned = load_tuned(a)
    lock = None if getattr(a, "no_lock", False) else HEAVY_LOCK
    mfiles = sorted(MACHINES_DIR.glob("*.json"))
    if a.machines:
        mfiles = [m for m in mfiles if m.stem in a.machines]
    kernels = a.kernels or all_kernels()
    sweeps = parse_sweeps(getattr(a, "sweep", None))
    points = []  # (point name, base machine name, machine json path, derived data or None)
    for mf in mfiles:
        points.append((mf.stem, mf.stem, mf, None))
        if sweeps:
            raw = json.loads(mf.read_text())
            raw.setdefault("name", mf.stem)
            for par, val in sweeps:
                d = derive_machine(raw, par, val)
                pj = a.results_root / d["name"] / "machine.json"
                points.append((d["name"], mf.stem, pj, d))
    for pname, mname, mf, derived in points:
        machine = load_machine(mf) if derived is None else _normalised(derived)
        machine["name"] = mname
        largest = len(machine["window_table"]) - 1
        twa = tuned_for(tuned, "window_args", mname)
        tsuffix = tuned.get("binary_suffix", {}) or {}
        lut_root = a.lut_root
        if Path(a.lut_root).resolve() == LUT_ROOT.resolve():   # not set on the command line
            lut_root = Path(tuned_for(tuned, "lut_root", mname).get("root", a.lut_root))
        for v in select(variants_for(machine, a.lut_ref_machine), a.policies):
            for k in kernels:
                sub = v.name + a.bin_suffix
                if a.input != "large":
                    sub += f".{a.input}"
                outdir = a.results_root / pname / k / sub
                bdir = v.binary + a.bin_suffix + tsuffix.get(v.binary, "")
                binary = find_binary(a.bin_root, bdir, mname, k)
                missing = []
                if binary is None:
                    why = (" (no build for this machine)" if v.binary in MACHINE_DEPENDENT
                           and (a.bin_root / bdir / k).is_file() else "")
                    missing.append(f"binary {a.bin_root}/{bdir}/[{mname}/]{k}{why}")
                else:
                    bad = tuned_compiler_mismatch(tuned, binary, v.binary)
                    if bad:
                        missing.append(bad)
                if supported is not None and v.policy not in supported:
                    missing.append(f"policy {v.policy!r} not accepted by sim/se.py yet")
                lut = None
                if v.lut:
                    lm = mname if v.lut == "own" else a.lut_ref_machine
                    lut = lut_root / lm / "lut.txt"
                    if not lut.is_file():
                        missing.append(f"LUT {lut}")
                initial = largest if v.initial is None else v.initial
                wa = variant_window_args(v, wargs, twa)
                extra = ["--window-args", wa] if wa else None
                kw = dict(gem5=a.gem5, machine_json=mf,
                          binary=binary or f"<missing:{v.binary}/{k}>", input_size=a.input,
                          policy=v.policy, initial=initial, period=a.period, lut=lut,
                          trace=a.trace, max_insts=a.max_insts, extra=extra)
                cmd = gem5_cmd(outdir=outdir, **kw)
                runs.append(Run(pname, mf, k, v, a.input, outdir, binary, lut, cmd, missing, lock,
                                derived, cmd_kw=kw, base_machine_json=MACHINES_DIR / f"{mname}.json"))
    apply_run_length(runs, a)
    return runs


# ---------------------------------------------------------------------------
# Run length (sim/run_lengths.json): plans, profile and checkpoint passes
# ---------------------------------------------------------------------------

# binary_key() cache: (path, size, mtime_ns) -> sha1 prefix
_SHA: dict[tuple, str] = {}


def binary_key(path: Path) -> str:
    """Return a 16-hex-digit SHA-1 prefix of a binary's contents (cached by path/size/mtime)."""
    st = path.stat()
    k = (str(path), st.st_size, st.st_mtime_ns)
    if k not in _SHA:
        _SHA[k] = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
    return _SHA[k]


def _mem_size(machine_json: Path) -> str:
    """Return the memory size of a machine JSON (``memory.size``; ``512MB`` by default)."""
    try:
        m = json.loads(Path(machine_json).read_text()).get("memory", {})
    except (OSError, ValueError):
        m = {}
    return str(m.get("size", m.get("mem_size", "512MB")))


def _functional_cmd(a, outdir: Path, machine_json: Path, binary: Path, input_size: str,
                    flags: list[str]) -> list[str]:
    """Return a gem5 command for a functional se.py pass (profile/checkpoints).

    Args:
        a (argparse.Namespace): Parsed arguments (``gem5``).
        outdir: gem5 outdir.
        machine_json: se.py ``--machine``.
        binary: se.py ``--cmd``.
        input_size: se.py ``--options``.
        flags: Mode-specific se.py flags.
    """
    return [a.gem5, f"--outdir={outdir}", "--redirect-stdout", "--redirect-stderr", str(SE_PY),
            "--machine", str(machine_json), "--cmd", str(binary), "--options", input_size] + flags


def profile_job(a, profile_dir: str, kernel: str, input_size: str, binary: Path,
                machine_json: Path, cap: int, lock: Path | None) -> AuxJob:
    """Return the phase-P job that profiles the region markers of one kernel/input.

    The product is ``<runlen-root>/<profile_dir>/<kernel>.<input>.json``.

    Args:
        a (argparse.Namespace): Parsed arguments (``runlen_root``, ``gem5``).
        profile_dir: Binary directory of the region-marker build (e.g. ``oracle``).
        kernel: Kernel name.
        input_size: Input size.
        binary: Region-marker binary.
        machine_json: Machine JSON for se.py.
        cap: ``--profile-cap``.
        lock: Heavy-job lock, or None.
    """
    root = Path(getattr(a, "runlen_root", RUNLEN_ROOT))
    out = root / profile_dir / f"{kernel}.{input_size}.json"
    gdir = root / profile_dir / f"{kernel}.{input_size}.gem5"
    return AuxJob("profile", out, gdir, _functional_cmd(
        a, gdir, machine_json, binary, input_size,
        ["--profile-regions", str(out), "--profile-cap", str(cap)]), lock)


def checkpoint_job(a, binary: Path, input_size: str, starts: list[int], machine_json: Path,
                   lock: Path | None) -> AuxJob:
    """Return the phase-C job that writes the checkpoints of one binary/input.

    The product is ``<ckpt-root>/<binary sha1>-<input>-<mem>-<starts hash>/``,
    so every variant and machine running the same binary shares it.

    Args:
        a (argparse.Namespace): Parsed arguments (``ckpt_root``, ``gem5``).
        binary: Binary to checkpoint.
        input_size: Input size.
        starts: Checkpoint instruction counts.
        machine_json: Machine JSON for se.py (memory size is part of the key).
        lock: Heavy-job lock, or None.
    """
    mem = _mem_size(machine_json)
    h = hashlib.sha1(",".join(map(str, starts)).encode()).hexdigest()[:10]
    d = Path(getattr(a, "ckpt_root", CKPT_ROOT)) / f"{binary_key(binary)}-{input_size}-{mem}-{h}"
    return AuxJob("checkpoint", d, d / "gem5", _functional_cmd(
        a, d / "gem5", machine_json, binary, input_size,
        ["--take-checkpoints", ",".join(map(str, starts)), "--checkpoint-dir", str(d)]),
        lock, list(starts))


def run_length_config(a) -> dict | None:
    """Return the run-length config, or None with ``--run-length full``."""
    if getattr(a, "run_length", "config") == "full":
        return None
    return rl.load_config(getattr(a, "run_lengths", None) or RUN_LENGTHS)


def apply_run_length(runs: list[Run], a, profile_variant: str | None = None) -> None:
    """Attach the run-length plan of sim/run_lengths.json to every run.

    Sets each run's spec, the profile/checkpoint passes it needs (deduplicated
    by product) and its sample commands; problems go to ``missing`` and a
    missing profile to ``pending``. No-op with ``--run-length full``.

    Args:
        runs: Runs to update in place.
        a (argparse.Namespace): Parsed arguments.
        profile_variant: Binary directory of the region-marker build (default
            ``oracle<--bin-suffix>``).
    """
    cfg = run_length_config(a)
    if cfg is None:
        return
    pvar = profile_variant or ("oracle" + getattr(a, "bin_suffix", ""))
    jobs: dict[Path, AuxJob] = {}       # dedup: one pass per product

    def dedup(j: AuxJob) -> AuxJob:
        """Return the already registered job with the same product, or register ``j``."""
        return jobs.setdefault(j.product, j)

    for r in runs:
        spec = rl.spec_for(cfg, r.kernel, r.input)
        r.runlen = spec
        if spec["mode"] == "full":
            r.plan = rl.Plan("full", spec["version"])
            continue
        if getattr(a, "max_insts", None):
            r.missing.append("--max-insts conflicts with the sampled run length "
                             "(use --run-length full)")
            continue
        mjson = r.base_machine_json or r.machine_json
        prof = None
        if spec["mode"] == "regions":
            pbin = find_binary(a.bin_root, pvar, r.machine.split("+")[0], r.kernel, strict=False)
            if pbin is None:
                r.missing.append(f"binary {a.bin_root}/{pvar}/{r.kernel} (region profile)")
                continue
            r.profile = dedup(profile_job(a, pvar, r.kernel, r.input, pbin, mjson,
                                          int(spec.get("profile_cap", 2048)), r.lock))
            if not r.profile.done():
                r.pending = f"region profile {r.profile.product}"
                continue
            prof = json.loads(r.profile.product.read_text())
        try:
            r.plan = rl.plan(spec, prof)
        except ValueError as exc:
            r.missing.append(f"run length: {exc}")
            continue
        if r.plan.mode == "full" or r.binary is None:
            continue
        if r.plan.starts:
            r.ckpt = dedup(checkpoint_job(a, r.binary, r.input, r.plan.starts, mjson, r.lock))
        kw = dict(r.cmd_kw or {})
        kw.pop("max_insts", None)
        r.samples = []
        for smp in r.plan.samples:
            od = r.outdir / smp.name
            cmd = gem5_cmd(outdir=od, max_insts=smp.measure, **kw) + rl.sample_args(
                smp, r.ckpt.product if r.ckpt else None)
            r.samples.append(SampleRun(smp, od, cmd))
        if r.samples:
            r.cmd = r.samples[0].cmd


def _normalised(data: dict) -> dict:
    """Return a copy of a derived machine dict with ``window_table`` added."""
    from whdata import window_table
    m = copy.deepcopy(data)
    m["window_table"] = window_table(m)
    return m


def is_done(outdir: Path) -> bool:
    """Return True if ``outdir`` has stats.txt and a run.json with status ``ok``."""
    rj = outdir / "run.json"
    if not (rj.exists() and (outdir / "stats.txt").exists()):
        return False
    try:
        return json.loads(rj.read_text()).get("status") == "ok"
    except (OSError, ValueError):
        return False


def _run_gem5(cmd: list[str], outdir: Path, lock: Path | None, timeout: int | None,
              meta: dict) -> int:
    """Run one gem5 command (under the heavy lock), log to outdir/gem5.log.

    Args:
        cmd: gem5 command.
        outdir: Output directory (created).
        lock: Lock file for ``flock``, or None.
        timeout: Timeout in seconds, or None.
        meta: Run metadata, updated with ``lock``/``timeout``.

    Returns:
        The return code (-9 on timeout).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = list(cmd)
    if lock is not None:
        # interfaces.md §1: gem5 simulations run one at a time under the heavy lock
        lock.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["flock", str(lock)] + cmd
        meta["lock"] = str(lock)
    try:
        p = subprocess.run(cmd, cwd=str(REPO), timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        rc, log = p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        rc, log = -9, (e.stdout or "") if isinstance(e.stdout, str) else ""
        meta["timeout"] = timeout
    (outdir / "gem5.log").write_text(log or "")
    return rc


def _stats_ok(d: Path) -> bool:
    """Return True if ``d/stats.txt`` exists and is non-empty."""
    return (d / "stats.txt").exists() and (d / "stats.txt").stat().st_size > 0


def execute(run: Run, timeout: int | None, force: bool = False) -> dict:
    """Execute one run and write its run.json.

    A full run executes its command; a sampled run executes every sample not
    yet done (unless ``force``) and then merges them
    (``run_lengths.merge_samples``). A derived machine JSON is written first.

    Args:
        run: The run.
        timeout: Per-gem5-call timeout in seconds, or None.
        force: Rerun samples that are already done.

    Returns:
        The run.json metadata (``status`` ``ok``/``failed``, ``wall_seconds``,
        ...).
    """
    run.outdir.mkdir(parents=True, exist_ok=True)
    if run.machine_data is not None:
        run.machine_json.parent.mkdir(parents=True, exist_ok=True)
        run.machine_json.write_text(json.dumps(run.machine_data, indent=2) + "\n")
    t0 = time.time()
    meta = {"machine": run.machine, "machine_json": str(run.machine_json),
            "kernel": run.kernel, "variant": run.variant.name,
            "baseline": run.variant.baseline, "policy": run.variant.policy, "input": run.input,
            "binary": str(run.binary), "lut": str(run.lut) if run.lut else None,
            "command": " ".join(shlex.quote(c) for c in run.cmd),
            "started": dt.datetime.now().isoformat(timespec="seconds"),
            "runlen": run.mode, "runlen_version": (run.runlen or {}).get("version")}
    if not run.samples:
        rc = _run_gem5(run.cmd, run.outdir, run.lock, timeout, meta)
        ok = rc == 0 and _stats_ok(run.outdir)
        meta.update(returncode=rc, status="ok" if ok else "failed",
                    wall_seconds=round(time.time() - t0, 1))
        (run.outdir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
        return meta
    # sampled run: every sample (resumable), then the merge
    failed = []
    for sr in run.samples:
        if not force and is_done(sr.outdir):
            continue
        sm = {"sample": sr.sample.name, "command": " ".join(shlex.quote(c) for c in sr.cmd),
              "started": dt.datetime.now().isoformat(timespec="seconds")}
        rc = _run_gem5(sr.cmd, sr.outdir, run.lock, timeout, sm)
        sm.update(returncode=rc, status="ok" if rc == 0 and _stats_ok(sr.outdir) else "failed")
        (sr.outdir / "run.json").write_text(json.dumps(sm, indent=2) + "\n")
        if sm["status"] != "ok":
            failed.append(sr.sample.name)
    status = "failed" if failed else "ok"
    if not failed:
        try:
            summ = rl.merge_samples(run.outdir, run.plan, [sr.outdir for sr in run.samples])
            meta.update(samples=len(run.samples), measured_insts=sum(summ["measured_insts"]),
                        total_insts=run.plan.total_insts,
                        unsampled_frac=run.plan.unsampled_frac,
                        hint_state_exact=summ["hint_state_exact"])
        except Exception as exc:  # noqa: BLE001
            status, meta["merge_error"] = "failed", str(exc)
    meta.update(returncode=0 if not failed else 1, status=status, failed_samples=failed,
                wall_seconds=round(time.time() - t0, 1))
    (run.outdir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def execute_aux(job: AuxJob, timeout: int | None) -> dict:
    """Execute one phase-P/C job and write ``aux.json`` to its outdir.

    Returns:
        The job metadata; ``status`` is ``ok`` only if gem5 succeeded and
        :meth:`AuxJob.done` holds.
    """
    t0 = time.time()
    meta = {"kind": job.kind, "product": str(job.product),
            "command": " ".join(shlex.quote(c) for c in job.cmd)}
    if job.kind == "profile":
        job.product.parent.mkdir(parents=True, exist_ok=True)
    rc = _run_gem5(job.cmd, job.outdir, job.lock, timeout, meta)
    meta.update(returncode=rc, status="ok" if rc == 0 and job.done() else "failed",
                wall_seconds=round(time.time() - t0, 1))
    (job.outdir / "aux.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def aux_jobs(runs: list[Run], kind: str) -> list[AuxJob]:
    """Return the distinct profile or checkpoint jobs of the runnable runs.

    Args:
        runs: Runs.
        kind: ``"profile"`` or ``"checkpoint"``.
    """
    seen, out = set(), []
    for r in runs:
        j = r.profile if kind == "profile" else r.ckpt
        if j is not None and j.product not in seen and not r.missing:
            seen.add(j.product)
            out.append(j)
    return out


def run_phases(build, a, label=lambda r: r.key) -> tuple[list[Run], int]:
    """Run phases P (profiles), C (checkpoints) and R (runs).

    The runs come from ``build()``, which is called again after P so the plans
    exist. Runs already done (unless ``--force``) or with missing inputs are
    skipped; phase R uses ``a.jobs`` threads.

    Args:
        build (Callable[[], list[Run]]): Callable returning the run list.
        a (argparse.Namespace): Parsed arguments (``gem5``, ``timeout``, ``jobs``, ``force``).
        label (Callable[[Run], str]): Run -> name used in progress messages.

    Returns:
        ``(runs, failures)``; failures also counts jobs not run because gem5 is
        missing.
    """
    failed = 0
    runs = build()
    for kind in ("profile", "checkpoint"):
        todo = [j for j in aux_jobs(runs, kind) if not j.done()]
        if todo and not Path(a.gem5).is_file():
            print(f"[ERROR] gem5 binary not found: {a.gem5}", file=sys.stderr)
            return runs, len(todo)
        for i, j in enumerate(todo, 1):
            m = execute_aux(j, a.timeout)
            failed += m["status"] != "ok"
            print(f"[{kind} {i}/{len(todo)}] {m['status'].upper():<6} {j.product} "
                  f"({m['wall_seconds']} s)")
        if kind == "profile" and todo:
            runs = build()
    todo = []
    for r in runs:
        if is_done(r.outdir) and not a.force:
            continue
        if r.missing or r.pending:
            print(f"[SKIP] {label(r)}: missing " + "; ".join(r.missing or [r.pending]))
            continue
        todo.append(r)
    if todo and not Path(a.gem5).is_file():
        print(f"[ERROR] gem5 binary not found: {a.gem5}", file=sys.stderr)
        return runs, failed + len(todo)
    print(f"[INFO] {len(todo)} runs to do, jobs={a.jobs}")
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(execute, r, a.timeout, a.force): r for r in todo}
        for i, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            meta = f.result()
            failed += meta["status"] != "ok"
            print(f"[{i}/{len(todo)}] {meta['status'].upper():<6} {label(r)} "
                  f"({meta['wall_seconds']} s)")
    return runs, failed


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

#: fixed columns of summary.csv (win.* counters are appended)
SUMMARY_FIELDS = ["machine", "kernel", "variant", "baseline", "policy", "input", "status",
                  "sim_seconds", "cycles", "insts", "ipc", "l1d_misses", "l2_misses",
                  "energy_j", "power_w", "edp", "ed2p", "energy_model", "wall_seconds",
                  "base_machine", "sens_param", "sens_value", "l2_kb", "mem_latency_cycles",
                  "runlen", "runlen_version", "samples", "measured_insts", "unsampled_frac",
                  "hint_state_exact"]


def load_point_machine(results_root: Path, name: str) -> dict:
    """Return the machine of a result directory.

    That is sim/machines/<name>.json, or the derived JSON of a sensitivity
    point (results_root/<name>/machine.json), normalised and named ``name``.
    """
    p = results_root / name / "machine.json"
    m = load_machine(p if p.is_file() else MACHINES_DIR / f"{name}.json")
    m["name"] = name
    return m


def collect(results_root: Path, energy_model: str, out: Path) -> int:
    """Write summary.csv for every run under ``results_root``.

    One row per ``<machine>/<kernel>/<variant>/run.json``: run metadata,
    sensitivity-point parameters, key stats and every ``win.*`` counter (extra
    columns), and energy from ``energy.json`` (computed and cached if absent)
    for successful runs.

    Args:
        results_root: results/gem5 tree.
        energy_model: ``estimate_energy`` model, or ``"none"``.
        out: Output CSV path.

    Returns:
        The number of rows written.
    """
    rows, extra_keys = [], set()
    machines: dict[str, dict] = {}
    for rj in sorted(results_root.glob("*/*/*/run.json")):
        d = rj.parent
        try:
            meta = json.loads(rj.read_text())
        except ValueError:
            continue
        row = {k: meta.get(k) for k in ("machine", "kernel", "variant", "baseline", "policy",
                                        "input", "status", "wall_seconds", "runlen",
                                        "runlen_version", "samples", "measured_insts",
                                        "unsampled_frac", "hint_state_exact")}
        m = meta.get("machine")
        if m not in machines:
            try:
                machines[m] = load_point_machine(results_root, m)
            except (OSError, ValueError):
                machines[m] = None
        if machines[m] is not None:
            der = machines[m].get("_derived") or {}
            row.update(base_machine=der.get("base", m), sens_param=der.get("param", ""),
                       sens_value=der.get("value", ""), **machine_point_params(machines[m]))
        if (d / "stats.txt").exists():
            ks = key_stats(parse_stats(d / "stats.txt"))
            row.update({k: v for k, v in ks.items() if k in SUMMARY_FIELDS or k.startswith("win.")})
            extra_keys.update(k for k in ks if k.startswith("win."))
            if meta.get("status") == "ok" and energy_model != "none" and machines[m] is not None:
                static = None
                if meta.get("policy") == "static":
                    v = meta.get("variant", "")
                    if v.startswith("static_c"):
                        static = int(v.split("static_c")[1].split(".")[0].split("-")[0])
                try:
                    cache = d / "energy.json"
                    e = json.loads(cache.read_text()) if cache.exists() else None
                    if e is None:
                        e = estimate_run(d, machines[m], energy_model, static)
                        cache.write_text(json.dumps(e, indent=2) + "\n")
                    row.update(energy_j=e["energy_j"], power_w=e["power_w"], edp=e["edp"],
                               ed2p=e["ed2p"], energy_model=e["model"])
                except Exception as exc:  # noqa: BLE001
                    row["energy_model"] = f"error: {exc}"
        rows.append(row)
    fields = SUMMARY_FIELDS + sorted(extra_keys)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[OK] summary of {len(rows)} runs -> {out}")
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """Parse the command line (see the module docstring and ``--help``)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernels", nargs="*", help="kernel names (default: all benchmarks/*_infer.c)")
    p.add_argument("--policies", nargs="*",
                   help="variant names, baseline ids (B0..B9, WinHint, WinHint+HW) or tags "
                        "(static, hw, winhint, b5, oracle, ...); default: all")
    p.add_argument("--machines", nargs="*", help="machine names (sim/machines/<name>.json); default: all")
    p.add_argument("--input", default="large", choices=["small", "large"])
    p.add_argument("--jobs", type=int, default=1, help=f"parallel gem5 runs (max {MAX_JOBS})")
    p.add_argument("--dry-run", action="store_true", help="print the run matrix and exit")
    p.add_argument("--force", action="store_true", help="rerun finished runs")
    p.add_argument("--collect-only", action="store_true", help="only write summary.csv")
    p.add_argument("--no-collect", action="store_true")
    p.add_argument("--energy-model", default="auto", choices=["auto", "mcpat", "proxy", "none"])
    p.add_argument("--trace", action="store_true", help="pass --window-trace (large files!)")
    p.add_argument("--period", type=int, default=None, help="--window-period (default: machine/gem5)")
    p.add_argument("--max-insts", type=int, default=None,
                   help="stop after N instructions (only with --run-length full)")
    p.add_argument("--run-lengths", type=Path, default=RUN_LENGTHS,
                   help="run-length config (sim/run_lengths.json)")
    p.add_argument("--run-length", default="config", choices=["config", "full"],
                   help="config: budgets of --run-lengths (large = sampled); full: "
                        "simulate every run in full")
    p.add_argument("--runlen-root", type=Path, default=RUNLEN_ROOT,
                   help="region profiles (phase P)")
    p.add_argument("--ckpt-root", type=Path, default=CKPT_ROOT,
                   help="checkpoints (phase C)")
    p.add_argument("--tuned", type=Path, default=None,
                   help="tuned parameters (sim/baselines/tune/tune_baselines.py output, "
                        f"default {TUNED_DEFAULT} if it exists; 'none' disables)")
    p.add_argument("--timeout", type=int, default=None, help="per-run timeout, seconds")
    p.add_argument("--gem5", default=GEM5_BIN)
    p.add_argument("--bin-root", type=Path, default=BIN_ROOT)
    p.add_argument("--bin-suffix", default="",
                   help="build-config suffix of the Makefile output dirs, e.g. -O3, "
                        "-nounroll, -tiled, -O3-tiled (also appended to the variant name)")
    p.add_argument("--lut-root", type=Path, default=LUT_ROOT)
    p.add_argument("--lut-ref-machine", default="riscv_ooo",
                   help="machine whose LUT the lut_xfer variant uses")
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    p.add_argument("--window-args", action="append", metavar="POLICY:k=v,...",
                   help="per-policy tunables passed as se.py --window-args (repeatable)")
    p.add_argument("--sweep", action="append", metavar="PARAM=v1,v2,...",
                   help="sensitivity points per selected machine (repeatable); PARAM in "
                        + ", ".join(sorted(SWEEP_PARAMS)))
    p.add_argument("--no-lock", action="store_true",
                   help=f"do not wrap gem5 in flock {HEAVY_LOCK} (only for runs < ~1 min)")
    return p.parse_args(argv)


def print_matrix(runs: list[Run], a) -> None:
    """Print the pending P/C passes and the run matrix (``--dry-run``).

    Each run line shows its state (done/MISSING/pending/todo), run-length mode
    (sample count and detailed instruction budget) and command or missing
    inputs.
    """
    for kind in ("profile", "checkpoint"):
        jobs = aux_jobs(runs, kind)
        pend = [j for j in jobs if not j.done()]
        if jobs:
            print(f"# phase {'P' if kind == 'profile' else 'C'}: {len(pend)} {kind} passes to run "
                  f"({len(jobs) - len(pend)} done)")
        for j in pend:
            print(f"  {kind:<10} {j.product}  " + " ".join(shlex.quote(c) for c in j.cmd))
    print(f"{'machine':<16} {'kernel':<30} {'variant':<22} {'id':<10} {'state':<8} "
          f"{'runlen':<16} details")
    n_todo = n_done = n_blocked = 0
    for r in runs:
        if is_done(r.outdir) and not a.force:
            st, n_done = "done", n_done + 1
        elif r.missing:
            st, n_blocked = "MISSING", n_blocked + 1
        elif r.pending:
            st, n_todo = "pending", n_todo + 1
        else:
            st, n_todo = "todo", n_todo + 1
        rlen = "-" if r.missing else r.mode
        if r.samples:
            rlen = f"{r.mode}:{len(r.samples)}x/{r.plan.detailed_insts() / 1e6:.0f}M"
        if r.missing:
            det = "; ".join(r.missing)
        elif r.pending:
            det = f"after phase P ({r.pending})"
        else:
            det = " ".join(shlex.quote(c) for c in r.cmd)
        print(f"{r.machine:<16} {r.kernel:<30} {r.variant.name:<22} {r.variant.baseline:<10} "
              f"{st:<8} {rlen:<16} {det}")
    print(f"\n{len(runs)} runs: {n_todo} to run, {n_done} done, "
          f"{n_blocked} missing inputs; jobs={a.jobs}")


def main(argv=None) -> int:
    """CLI entry point.

    Clamps ``--jobs`` to ``MAX_JOBS``, makes paths absolute, then either
    collects only, prints the matrix (``--dry-run``) or runs all phases and
    collects.

    Args:
        argv (list[str] | None): Argument list; None means ``sys.argv[1:]``.

    Returns:
        1 if any run or pass failed, else 0.
    """
    a = parse_args(argv)
    if a.jobs > MAX_JOBS:
        print(f"[WARN] --jobs {a.jobs} > {MAX_JOBS}; clamped (6 GB RAM shared by all agents)")
        a.jobs = MAX_JOBS
    a.jobs = max(1, a.jobs)
    # gem5 runs with cwd = repo root: make every path absolute
    a.results_root, a.bin_root, a.lut_root = (a.results_root.resolve(), a.bin_root.resolve(),
                                              a.lut_root.resolve())
    a.runlen_root, a.ckpt_root = a.runlen_root.resolve(), a.ckpt_root.resolve()
    if a.collect_only:
        collect(a.results_root, a.energy_model, a.results_root / "summary.csv")
        return 0

    if a.dry_run:
        print_matrix(build_runs(a), a)
        return 0

    _, failed = run_phases(lambda: build_runs(a), a)
    if not a.no_collect:
        collect(a.results_root, a.energy_model, a.results_root / "summary.csv")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
