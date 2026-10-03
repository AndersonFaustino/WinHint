#!/usr/bin/env python3
"""Baseline-fidelity study on microbenchmarks (fidelity.py, docs/reference/proposal.md §7).

"Each reimplemented baseline reproduces the qualitative trend reported in its
paper on a microbenchmark before it is used." For every gem5-side baseline
(B2, B3, B4, B5, B6, B7, B9) this script

    1. defines the gem5 runs it needs: the baseline itself plus static
       reference windows (policy, window-args, initial config, binary variant),
       on the microbenchmarks of benchmarks/micro/ (built with
       ``make -C benchmarks micro``);
    2. runs them (RISCV_winhint gem5 through sim/se.py, one at a time under
       ``flock $WINHINT_BUILD/.heavy.lock``; resumable);
    3. derives what some baselines need from earlier runs: the per-region
       oracle maps of micro_phased (B4/B5/B7 reference), the B5 runtime LUT
       (trained on the `small` input with the sim/baselines/lut pipeline) and
       the B7 `pgo` binary (per-region configs profiled on `small`);
    4. evaluates a machine-checkable qualitative-trend predicate per baseline
       and writes results/fidelity/<baseline>.json:
         PASS        every precondition and every trend check holds
         FAIL        the microbenchmark shows the targeted condition
                     (preconditions hold) but the baseline does not reproduce
                     the paper's trend
         INVALID     a precondition fails: the microbenchmark does not exhibit
                     the condition on this machine, so the test says nothing
         INCOMPLETE  runs are missing or failed

The trends, predicates and threshold rationale are in docs/guide/fidelity.md.

Layout (machine M; the default machine riscv_ooo writes the verdicts at the
top level, any other machine under results/fidelity/M/):

    results/fidelity/runs/M/<kernel>/<input>/<tag>/   gem5 outdir + run.json
         tag c<i> = static config i (oracle binary for micro_phased, so the
         c<i> runs are also the B5 training traces, label_phases.py layout)
    results/fidelity/oracle/M/<input>/micro_phased.{json,table.csv}
    results/fidelity/b5/M/{dataset.csv,model.pkl,lut.txt}
    results/fidelity/[M/]<baseline>.json, results/fidelity/[M/]summary.json

Usage (host, `winhint` env active):

    make -C benchmarks micro                          # build the microbenchmarks
    python sim/fidelity/fidelity.py --dry-run         # list runs, derive steps
    python sim/fidelity/fidelity.py                   # everything (resumable)
    python sim/fidelity/fidelity.py --baselines B3    # one baseline
    python sim/fidelity/fidelity.py --evaluate-only   # re-evaluate existing runs
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "sim"))
sys.path.insert(0, str(REPO / "sim" / "baselines" / "lut"))
from estimate_energy import key_stats, parse_stats  # noqa: E402
from whdata import (aggregate_regions, load_machine, load_region_stats,  # noqa: E402
                    load_window_trace, write_flat_oracle)

#: Build root (``$WINHINT_BUILD``, default ``<repo>/build``).
BUILD = Path(os.environ.get("WINHINT_BUILD", str(REPO / "build")))
#: gem5 binary (``$WINHINT_GEM5``, default the RISCV_winhint build under ``BUILD``).
GEM5 = os.environ.get("WINHINT_GEM5", str(BUILD / "gem5" / "src" / "build" / "RISCV_winhint"
                                          / "gem5.opt"))
SE_PY = REPO / "sim" / "se.py"
MACHINES_DIR = REPO / "sim" / "machines"
#: Default output root (``--results``).
RESULTS = REPO / "results" / "fidelity"
#: Default root of the microbenchmark binaries: ``<BIN_ROOT>/<variant>/<kernel>``.
BIN_ROOT = BUILD / "benchmarks" / "micro" / "riscv"
#: Lock file serialising gem5 runs on the host.
HEAVY_LOCK = BUILD / ".heavy.lock"
LUT_DIR = REPO / "sim" / "baselines" / "lut"
#: Machine whose verdicts are written at the top level of the results root.
DEFAULT_MACHINE = "riscv_ooo"

#: Microbenchmark kernel names (benchmarks/micro/).
GATHER, CHASE, COMPUTE, LOWILP, PHASED = ("micro_gather", "micro_chase", "micro_compute",
                                          "micro_lowilp", "micro_phased")
#: binary variant used for the static references and the reactive policies of a
#: kernel: micro_phased needs region(id) markers (per-region references)
BASE_VARIANT = {PHASED: "oracle"}

# ---------------------------------------------------------------------------
# Thresholds (README.md explains each one). Ratios are IPC ratios unless noted;
# "cap" is the residency-weighted allocated size as a fraction of the largest.
# ---------------------------------------------------------------------------
THRESHOLDS: dict[str, dict[str, float]] = {
    "common": {
        "mlp_bench_gain": 1.20,     # precondition: gather IPC(large)/IPC(small window)
        "tie_tol": 0.02,            # oracle-best = smallest config within 2% of best IPC
        "near_best": 0.97,          # a config is "near-best" for a region at >= 97% of best IPC
        "region_min_share": 0.05,   # regions with >= 5% of the cycles define the phases
    },
    "B2": {"underused_occ": 0.50, "max_cap": 0.60, "min_ipc": 0.95,
           "grow_cap": 0.60, "grow_recovery": 0.50},
    "B3": {"chase_no_gain": 1.05, "speedup": 1.10, "recovery": 0.50,
           "no_gain": 1.03, "no_loss": 0.97, "ilp_res_chase": 0.60, "ilp_res_compute": 0.90},
    "B4": {"top3_cover": 0.80, "max_phases": 12, "pred_acc": 0.80, "warmup": 0.25,
           "near_frac": 0.75, "min_ipc": 0.93, "max_cap": 0.85},
    "B5": {"near_frac": 0.75, "min_ipc": 0.95, "max_cap": 0.85},
    "B6": {"min_ipc": 0.97, "max_iq_cap": 0.75},
    "B7": {"min_oracle": 0.95, "min_ipc": 0.97, "max_cap": 0.85},
    "B9": {"small_iq_loss": 1.15, "recovery": 0.40, "speedup": 1.10, "no_harm": 0.95},
}


# ---------------------------------------------------------------------------
# Run specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    """One gem5 run of the fidelity study.

    Attributes:
        kernel: Microbenchmark name (``micro_*``).
        tag: Output directory name under ``runs/<M>/<kernel>/<input>/``.
        variant: Binary build variant (benchmarks/micro, micro.mk).
        policy: se.py ``--window-policy``.
        initial: Initial config index, or ``"min"``/``"max"`` (see ``resolve``).
        window_args: se.py ``--window-args`` (empty for none).
        input: Input size (``small``/``large``).
        needs: Derived artifact needed before the run: ``""``, ``"lut"`` (B5
            LUT) or ``"pgo"`` (B7 binary).
    """
    kernel: str
    tag: str                 # outdir name under runs/M/<kernel>/<input>/
    variant: str             # binary build variant (benchmarks/micro, micro.mk)
    policy: str              # se.py --window-policy
    initial: int | str       # config index, or "min"/"max" (resolved per machine)
    window_args: str = ""
    input: str = "large"
    needs: str = ""          # "" | "lut" (B5 LUT) | "pgo" (B7 binary): derived first

    @property
    def key(self) -> tuple[str, str, str]:
        """Identity of the run: ``(kernel, input, tag)``."""
        return (self.kernel, self.input, self.tag)

    def label(self) -> str:
        """Return the ``<kernel>/<input>/<tag>`` label (matched by ``--only``)."""
        return f"{self.kernel}/{self.input}/{self.tag}"


def static(kernel: str, cfg: int | str, input_size: str = "large") -> RunSpec:
    """Return a static-window reference run of ``kernel``.

    Args:
        kernel: Microbenchmark name.
        cfg: Config index or ``"min"``/``"max"`` (tag ``c<i>``, ``c_min`` or
            ``c_max``).
        input_size: Input size.

    Returns:
        A ``static`` policy run on the kernel's base variant (``BASE_VARIANT``,
        default ``plain``).
    """
    tag = {"min": "c_min", "max": "c_max"}.get(cfg, f"c{cfg}") if isinstance(cfg, str) else f"c{cfg}"
    return RunSpec(kernel, tag, BASE_VARIANT.get(kernel, "plain"), "static", cfg,
                   input=input_size)


def policy_run(kernel: str, tag: str, policy: str, initial: int | str, window_args: str = "",
               variant: str | None = None, input_size: str = "large", needs: str = "") -> RunSpec:
    """Return a run of ``kernel`` under a window policy.

    Args:
        kernel: Microbenchmark name.
        tag: Output directory name (``_imin``/``_imax`` suffixes are resolved
            with the initial config).
        policy: se.py window policy.
        initial: Initial config index or ``"min"``/``"max"``.
        window_args: se.py ``--window-args``.
        variant: Binary variant; ``None`` uses the kernel's base variant.
        input_size: Input size.
        needs: Derived artifact the run needs (``""``, ``"lut"``, ``"pgo"``).

    Returns:
        The run spec.
    """
    return RunSpec(kernel, tag, variant or BASE_VARIANT.get(kernel, "plain"), policy, initial,
                   window_args, input_size, needs)


def resolve(spec: RunSpec, n_configs: int) -> RunSpec:
    """Replace symbolic initial configs ("min"/"max") and their tags by indices.

    Args:
        spec: Run spec, possibly with ``initial`` ``"min"``/``"max"``.
        n_configs: Number of window configs of the machine.

    Returns:
        ``spec`` unchanged if ``initial`` is an index; otherwise a copy with
        ``initial`` = 0 or ``n_configs - 1`` and ``c_min``/``c_max``/``_imin``/
        ``_imax`` in the tag replaced accordingly.
    """
    if not isinstance(spec.initial, str):
        return spec
    idx = 0 if spec.initial == "min" else n_configs - 1
    tag = spec.tag.replace("c_min", f"c{idx}").replace("c_max", f"c{idx}")
    tag = tag.replace("_imin", f"_i{idx}").replace("_imax", f"_i{idx}")
    return RunSpec(spec.kernel, tag, spec.variant, spec.policy, idx, spec.window_args,
                   spec.input, spec.needs)


def phased_statics(input_size: str, n: int) -> list[RunSpec]:
    """Return the static runs of micro_phased on ``input_size`` for all ``n`` configs."""
    return [static(PHASED, i, input_size) for i in range(n)]


def runs_B2(n: int) -> list[RunSpec]:
    """Return the runs of B2 (occupancy): lowilp from the largest, gather from the smallest."""
    return [static(LOWILP, "max"), policy_run(LOWILP, "occupancy_imax", "occupancy", "max"),
            static(GATHER, "min"), static(GATHER, "max"),
            policy_run(GATHER, "occupancy_imin", "occupancy", "min")]


def runs_B3(n: int) -> list[RunSpec]:
    """Return the runs of B3 (mlp, ``ilp=0``): static min/max and mlp on gather, chase, compute."""
    out = []
    for k in (GATHER, CHASE, COMPUTE):
        out += [static(k, "min"), static(k, "max"), policy_run(k, "mlp_imin", "mlp", "min", "ilp=0")]
    return out


def runs_B4(n: int) -> list[RunSpec]:
    """Return the runs of B4 (bbv): large-input phased statics and bbv from the largest window."""
    return phased_statics("large", n) + [policy_run(PHASED, "bbv_imax", "bbv", "max")]


def runs_B5(n: int) -> list[RunSpec]:
    """Return the runs of B5 (lut): small and large phased statics and the LUT run."""
    return (phased_statics("small", n) + phased_statics("large", n)
            + [policy_run(PHASED, "lut_imax", "lut", "max", needs="lut")])


def runs_B6(n: int) -> list[RunSpec]:
    """Return the runs of B6 (jones, ``structs=iq``) and static max on compute, lowilp, gather."""
    out = []
    for k in (COMPUTE, LOWILP, GATHER):
        out += [static(k, "max"),
                policy_run(k, "jones_iq_imax", "hint", "max", "structs=iq", variant="jones")]
    return out


def runs_B7(n: int) -> list[RunSpec]:
    """Return the runs of B7 (pgo): small and large phased statics and the pgo-binary run."""
    return (phased_statics("small", n) + phased_statics("large", n)
            + [policy_run(PHASED, "pgo_imax", "hint", "max", variant="pgo", needs="pgo")])


def runs_B9(n: int) -> list[RunSpec]:
    """Return the runs of B9 (ltp): static max, small IQ/LSQ hint run and LTP on gather, compute."""
    out = []
    for k in (GATHER, COMPUTE):
        out += [static(k, "max"),
                policy_run(k, "smalliq_imin", "hint", "min", "structs=iq+lq+sq"),
                policy_run(k, "ltp_imin", "ltp", "min")]
    return out


# ---------------------------------------------------------------------------
# Metrics of one run
# ---------------------------------------------------------------------------

def resized_structs(policy: str, window_args: str) -> set[str]:
    """Return the structures that follow the configuration index (others stay at the largest).

    ``hint`` honours ``structs=`` (default ``all``), ``ltp`` defaults to
    ``iq+lsq``; every other policy resizes all structures. Tokens are separated
    by ``+`` or ``:``; ``lsq`` expands to ``lq`` and ``sq``.

    Args:
        policy: se.py window policy.
        window_args: ``--window-args`` string (``k=v,...``).

    Returns:
        Subset of ``{"rob", "iq", "lq", "sq"}``.
    """
    args = dict(kv.split("=", 1) for kv in window_args.split(",") if "=" in kv)
    if policy == "hint":
        spec = args.get("structs", "all")
    elif policy == "ltp":
        spec = args.get("structs", "iq+lsq")
    else:
        spec = "all"
    out: set[str] = set()
    for tok in spec.replace(":", "+").split("+"):
        tok = tok.strip()
        if tok == "all":
            out |= {"rob", "iq", "lq", "sq"}
        elif tok == "lsq":
            out |= {"lq", "sq"}
        elif tok:
            out.add(tok)
    return out


def _vec(stats: dict, name: str, n: int) -> list[float]:
    """Return the per-config vector stat ``system.cpu.window.<name>::<i>`` (missing -> 0)."""
    return [float(stats.get(f"system.cpu.window.{name}::{i}", 0.0)) for i in range(n)]


def run_metrics(outdir: Path, spec: RunSpec, table: list[dict]) -> dict:
    """Return IPC, residency, mean allocated caps and mean occupancies of one run.

    Residency comes from the ``cyclesInConfig`` stats, else from
    ``window_trace.csv``, else 100 % in the initial config (``"static"``). The
    cap of a resized structure is the residency-weighted size; other
    structures count at their largest size. Occupancies (``<s>_occ``) are only
    reported when ``<s>OccSum`` stats exist and residency came from stats.

    Args:
        outdir: Finished run directory.
        spec: Run spec (policy, window args, initial config).
        table: Machine window table.

    Returns:
        Dict with ``ipc``, ``cycles``, ``insts``, ``l1d_misses``,
        ``l2_misses``, ``residency``, ``residency_source``, ``switches``,
        ``<s>_cap``/``<s>_cap_frac`` per structure, optional
        ``<s>_occ``/``<s>_occ_frac`` and ``l1d_outstanding_mean``.
    """
    n = len(table)
    stats = parse_stats(outdir / "stats.txt")
    ks = key_stats(stats)
    cyc = _vec(stats, "cyclesInConfig", n)
    src = "stats"
    if sum(cyc) <= 0 and (outdir / "window_trace.csv").exists():
        df = load_window_trace(outdir / "window_trace.csv")
        cyc = [float(df.loc[df.config == i, "d_cycles"].sum()) for i in range(n)]
        src = "trace"
    if sum(cyc) <= 0:
        cyc = [0.0] * n
        cyc[int(spec.initial) if not isinstance(spec.initial, str) else n - 1] = 1.0
        src = "static"
    tot = sum(cyc)
    res = [c / tot for c in cyc]
    resized = resized_structs(spec.policy, spec.window_args)
    m = {"ipc": ks["ipc"], "cycles": ks["cycles"], "insts": ks["insts"],
         "l1d_misses": ks["l1d_misses"], "l2_misses": ks["l2_misses"],
         "residency": [round(r, 6) for r in res], "residency_source": src,
         "switches": ks.get("win.switches", 0.0)}
    for s in ("rob", "iq", "lq", "sq"):
        mx = max(c[s] for c in table)
        cap = sum(r * c[s] for r, c in zip(res, table)) if s in resized else mx
        m[f"{s}_cap"] = cap
        m[f"{s}_cap_frac"] = cap / mx if mx else 0.0
        occ = _vec(stats, f"{s}OccSum", n)
        if sum(occ) > 0 and src == "stats":
            m[f"{s}_occ"] = sum(occ) / tot
            m[f"{s}_occ_frac"] = m[f"{s}_occ"] / mx if mx else 0.0
    mlp = stats.get("system.cpu.window.l1dOutstandingDist::mean")
    if mlp is not None:
        m["l1d_outstanding_mean"] = mlp
    return m


def region_table(outdirs: dict[int, Path]) -> dict[int, dict[int, dict]]:
    """Return {region: {config: {cycles, insts, ipc}}} from the static runs' region_stats.csv.

    Args:
        outdirs: Config index -> static run directory; directories without
            ``region_stats.csv`` are skipped.

    Returns:
        The per-region, per-config aggregate table.
    """
    tab: dict[int, dict[int, dict]] = {}
    for cfg, d in outdirs.items():
        rs = d / "region_stats.csv"
        if not rs.exists():
            continue
        for r in aggregate_regions(load_region_stats(rs)).itertuples():
            tab.setdefault(int(r.region), {})[cfg] = {
                "cycles": float(r.cycles), "insts": float(r.insts), "ipc": float(r.ipc)}
    return tab


def best_configs(tab: dict[int, dict[int, dict]], tie_tol: float) -> dict[int, int]:
    """Return, per region, the smallest config within ``tie_tol`` of the best IPC.

    Args:
        tab: Table from ``region_table``.
        tie_tol: Relative IPC tolerance.

    Returns:
        ``{region: config}``.
    """
    out = {}
    for r, by in tab.items():
        top = max(v["ipc"] for v in by.values())
        out[r] = min(c for c, v in by.items() if v["ipc"] >= top * (1 - tie_tol))
    return out


def phase_regions(tab: dict[int, dict[int, dict]], share: float) -> list[int]:
    """Return the regions with at least ``share`` of the cycles (largest static config).

    Args:
        tab: Table from ``region_table``.
        share: Minimum cycle share.

    Returns:
        Sorted region ids (empty for an empty table).
    """
    if not tab:
        return []
    cmax = max(c for by in tab.values() for c in by)
    tot = sum(by.get(cmax, {}).get("cycles", 0.0) for by in tab.values())
    return sorted(r for r, by in tab.items()
                  if tot and by.get(cmax, {}).get("cycles", 0.0) >= share * tot)


def near_best_fraction(trace: Path, tab: dict[int, dict[int, dict]], near: float,
                       skip_frac: float = 0.0) -> dict:
    """Return the cycle-weighted share of trace periods in a near-best config.

    Cycle-weighted fraction of trace periods (inside a known region) whose
    (region, config) reaches ``near`` x the region's best static IPC; periods
    before ``skip_frac`` of the run's instructions are skipped (warm-up).

    Args:
        trace: ``window_trace.csv`` of the run.
        tab: Static region table (``region_table``).
        near: Near-best IPC fraction.
        skip_frac: Leading instruction fraction to skip.

    Returns:
        ``near_best_frac``, ``exact_frac`` (share in the oracle-best config;
        both NaN when no period is covered) and ``covered_cycles``.
    """
    df = load_window_trace(trace)
    cum = df["d_insts"].cumsum()
    total = float(df["d_insts"].sum()) or 1.0
    df = df[cum >= skip_frac * total]
    best = {r: max(v["ipc"] for v in by.values()) for r, by in tab.items()}
    bcfg = best_configs(tab, THRESHOLDS["common"]["tie_tol"])
    w_all = w_near = w_exact = 0.0
    for r, c, w in zip(df["region"], df["config"], df["d_cycles"]):
        r, c = int(r), int(c)
        if r not in tab or c not in tab[r]:
            continue
        w_all += w
        if tab[r][c]["ipc"] >= near * best[r]:
            w_near += w
        if bcfg.get(r) == c:
            w_exact += w
    return {"near_best_frac": w_near / w_all if w_all else float("nan"),
            "exact_frac": w_exact / w_all if w_all else float("nan"),
            "covered_cycles": w_all}


def read_bbv_phases(path: Path) -> dict:
    """Parse bbv_phases.csv (docs/guide/gem5/policies.md B4): header counters + per-phase visits.

    ``#`` lines carry ``key=value`` counters; the remaining lines are a CSV
    with one row per phase.

    Args:
        path: ``bbv_phases.csv`` of a bbv run.

    Returns:
        ``intervals``, ``predicted_correct``, ``phases_allocated``,
        ``accuracy`` (predicted_correct / intervals), descending ``visits``,
        ``top3_cover`` (share of visits in the 3 most visited phases) and
        ``learned`` (phase -> learned config).
    """
    head, rows = {}, []
    lines = path.read_text().splitlines()
    for ln in lines:
        if ln.startswith("#"):
            for tok in ln[1:].split():
                k, _, v = tok.partition("=")
                if v:
                    head[k] = float(v)
    body = [ln for ln in lines if ln and not ln.startswith("#")]
    if body:
        cols = body[0].split(",")
        for ln in body[1:]:
            rows.append(dict(zip(cols, ln.split(","))))
    visits = sorted((int(float(r.get("visits", 0))) for r in rows), reverse=True)
    iv = head.get("intervals", 0.0)
    return {"intervals": iv, "predicted_correct": head.get("predicted_correct", 0.0),
            "phases_allocated": head.get("phases_allocated", float(len(rows))),
            "accuracy": head.get("predicted_correct", 0.0) / iv if iv else float("nan"),
            "visits": visits,
            "top3_cover": sum(visits[:3]) / sum(visits) if sum(visits) else float("nan"),
            "learned": {int(float(r["phase"])): int(float(r["learned_config"])) for r in rows
                        if "phase" in r and "learned_config" in r}}


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

class Missing(Exception):
    """A run (or derived artifact) needed by a predicate is not available."""


@dataclass
class Check:
    """One machine-checkable condition of a baseline's predicate.

    Attributes:
        name: Human-readable description.
        kind: ``"precondition"`` or ``"trend"``.
        value: Measured value; None or NaN never passes.
        op: ``">="`` or ``"<="``.
        threshold: Bound compared against ``value``.
        detail: Optional explanation.
        missing: Label of the missing run/artifact, if the value could not be
            computed.
    """
    name: str
    kind: str          # "precondition" | "trend"
    value: float | None
    op: str            # ">=" | "<="
    threshold: float
    detail: str = ""
    missing: str = ""  # set when a run needed by this check is not available

    @property
    def ok(self) -> bool:
        """Whether ``value op threshold`` holds (False for None/NaN values)."""
        if self.value is None or (isinstance(self.value, float) and math.isnan(self.value)):
            return False
        return self.value >= self.threshold if self.op == ">=" else self.value <= self.threshold

    def as_dict(self) -> dict:
        """Return the JSON form of the check (value rounded to 6 digits, None for NaN)."""
        v = self.value
        d = {"name": self.name, "kind": self.kind,
             "value": None if v is None or (isinstance(v, float) and math.isnan(v))
             else round(float(v), 6),
             "op": self.op, "threshold": self.threshold, "ok": self.ok, "detail": self.detail}
        if self.missing:
            d["missing"] = self.missing
        return d


def chk(name: str, kind: str, value: Callable[[], float], op: str, threshold: float,
        detail: Callable[[], str] | str = "") -> Check:
    """Build a check whose value is computed now; a missing run makes it 'missing'.

    Args:
        name: Check description.
        kind: ``"precondition"`` or ``"trend"``.
        value: Callable computing the value; may raise ``Missing``.
        op: ``">="`` or ``"<="``.
        threshold: Bound.
        detail: Detail string or callable producing it.

    Returns:
        The evaluated ``Check`` (value None and ``missing`` set if ``Missing``
        was raised).
    """
    try:
        v = value()
        det = detail() if callable(detail) else detail
    except Missing as e:
        return Check(name, kind, None, op, threshold, "", missing=str(e))
    return Check(name, kind, v, op, threshold, det)


def verdict(checks: list[Check]) -> str:
    """Return the verdict of a baseline from its checks.

    Args:
        checks: All checks of the baseline.

    Returns:
        ``"INVALID"`` if a computed precondition fails, else ``"INCOMPLETE"`` if
        any check is missing, else ``"PASS"`` if all checks hold, else
        ``"FAIL"``.
    """
    if any(c.kind == "precondition" and not c.ok and not c.missing for c in checks):
        return "INVALID"
    if any(c.missing for c in checks):
        return "INCOMPLETE"
    return "PASS" if all(c.ok for c in checks) else "FAIL"


def ratio(a: float, b: float) -> float:
    """Return ``a / b``, or NaN when ``b`` is zero."""
    return a / b if b else float("nan")


def recovery(x: float, lo: float, hi: float) -> float:
    """Return the fraction of the lo->hi IPC gap that x recovers (NaN if hi == lo)."""
    return (x - lo) / (hi - lo) if hi != lo else float("nan")


class Ctx:
    """Access to run metrics and derived data for the predicates.

    Attributes:
        root: Results root.
        machine: Machine dict (with ``name`` and ``window_table``).
        n: Number of window configs.
        used: Label -> metrics of every run accessed (reported per baseline).
    """

    def __init__(self, root: Path, machine: dict, n: int):
        """Create the context.

        Args:
            root: Results root (``--results``).
            machine: Machine dict with ``name`` and ``window_table``.
            n: Number of window configs.
        """
        self.root, self.machine, self.n = root, machine, n
        self._cache: dict = {}
        self.used: dict[str, dict] = {}

    def outdir(self, kernel: str, tag: str, input_size: str = "large") -> Path:
        """Return the run directory ``<root>/runs/<machine>/<kernel>/<input_size>/<tag>``."""
        return self.root / "runs" / self.machine["name"] / kernel / input_size / tag

    def m(self, kernel: str, tag: str, input_size: str = "large") -> dict:
        """Return (and cache) the ``run_metrics`` of a finished run.

        The run's spec is read back from its ``run.json``. Every accessed run is
        recorded in ``used``.

        Raises:
            Missing: If the run is not done.
        """
        key = (kernel, tag, input_size)
        if key not in self._cache:
            d = self.outdir(kernel, tag, input_size)
            if not is_done(d):
                raise Missing(f"{kernel}/{input_size}/{tag}")
            spec = json.loads((d / "run.json").read_text()).get("spec", {})
            rs = RunSpec(kernel, tag, spec.get("variant", "plain"), spec.get("policy", "static"),
                         spec.get("initial", self.n - 1), spec.get("window_args", ""), input_size)
            self._cache[key] = run_metrics(d, rs, self.machine["window_table"])
            self.used[f"{kernel}/{input_size}/{tag}"] = self._cache[key]
        return self._cache[key]

    def ipc(self, kernel: str, tag: str, input_size: str = "large") -> float:
        """Return the IPC of a finished run (see ``m``)."""
        return self.m(kernel, tag, input_size)["ipc"]

    def table(self, input_size: str) -> dict[int, dict[int, dict]]:
        """Return (and cache) the micro_phased region table of the static runs on ``input_size``.

        Raises:
            Missing: If a static run is not done or no ``region_stats.csv`` exists.
        """
        key = ("table", input_size)
        if key not in self._cache:
            dirs = {i: self.outdir(PHASED, f"c{i}", input_size) for i in range(self.n)}
            missing = [str(d) for d in dirs.values() if not is_done(d)]
            if missing:
                raise Missing(", ".join(missing))
            self._cache[key] = region_table(dirs)
            if not self._cache[key]:
                raise Missing(f"no region_stats.csv in the {PHASED} {input_size} static runs")
        return self._cache[key]


def lo(n: int) -> str:
    """Return the tag of the smallest static config (``c0``)."""
    return "c0"


def hi(n: int) -> str:
    """Return the tag of the largest static config (``c<n-1>``)."""
    return f"c{n - 1}"


def phases_matter(ctx: Ctx, input_size: str = "large") -> Check:
    """Return the precondition that micro_phased's phases want different configs.

    Passes when the phase regions (``phase_regions``) have at least two distinct
    oracle-best configs on ``input_size``.

    Args:
        ctx: Evaluation context.
        input_size: Input whose static runs define the regions.

    Returns:
        The precondition check.
    """
    def regs_best():
        """Return the phase regions and per-region best configs."""
        tab = ctx.table(input_size)
        regs = phase_regions(tab, THRESHOLDS["common"]["region_min_share"])
        best = best_configs(tab, THRESHOLDS["common"]["tie_tol"])
        return regs, best
    return chk("phased: distinct oracle-best configs among the phase regions", "precondition",
               lambda: len({regs_best()[1][r] for r in regs_best()[0]}), ">=", 2,
               lambda: f"regions {regs_best()[0]} -> {[regs_best()[1][r] for r in regs_best()[0]]}")


def eval_B2(ctx: Ctx) -> list[Check]:
    """Return the B2 (occupancy) checks.

    Preconditions: lowilp under-uses the ROB at the largest window and gather
    gains from a large window. Trends: on lowilp the occupancy policy shrinks
    the allocated window with a small IPC loss; on gather (starting small) it
    grows back and recovers part of the small->large IPC gap.

    Args:
        ctx: Evaluation context.

    Returns:
        The checks.
    """
    t, c, n = THRESHOLDS["B2"], THRESHOLDS["common"], ctx.n
    big, small = hi(n), lo(n)
    ipc = ctx.ipc
    occ = lambda: ctx.m(LOWILP, f"occupancy_i{n - 1}")  # noqa: E731
    g_occ = lambda: ctx.m(GATHER, "occupancy_i0")  # noqa: E731
    return [
        chk("lowilp: mean ROB occupancy / ROB size at the largest static window", "precondition",
            lambda: ctx.m(LOWILP, big).get("rob_occ_frac", float("nan")), "<=",
            t["underused_occ"]),
        chk("gather: IPC(static large) / IPC(static small)", "precondition",
            lambda: ratio(ipc(GATHER, big), ipc(GATHER, small)), ">=", c["mlp_bench_gain"]),
        chk("lowilp: occupancy policy mean allocated ROB/IQ/LQ/SQ (largest fraction of the four)",
            "trend", lambda: max(occ()[f"{s}_cap_frac"] for s in ("rob", "iq", "lq", "sq")),
            "<=", t["max_cap"]),
        chk("lowilp: IPC(occupancy) / IPC(static large)", "trend",
            lambda: ratio(occ()["ipc"], ipc(LOWILP, big)), ">=", t["min_ipc"]),
        chk("gather (start small): occupancy policy mean allocated window (fraction)", "trend",
            lambda: g_occ()["rob_cap_frac"], ">=", t["grow_cap"]),
        chk("gather (start small): IPC gap static small->large recovered", "trend",
            lambda: recovery(g_occ()["ipc"], ipc(GATHER, small), ipc(GATHER, big)), ">=",
            t["grow_recovery"]),
    ]


def eval_B3(ctx: Ctx) -> list[Check]:
    """Return the B3 (mlp) checks.

    Preconditions: gather gains from a large window, chase does not. Trends:
    mlp speeds up gather over the small (ILP-mode) window and recovers part of
    the gap; on chase it neither gains nor loses; chase and compute stay mostly
    in config 0.

    Args:
        ctx: Evaluation context.

    Returns:
        The checks.
    """
    t, c, n = THRESHOLDS["B3"], THRESHOLDS["common"], ctx.n
    big, small = hi(n), lo(n)
    ipc = ctx.ipc
    mlp = lambda k: ctx.m(k, "mlp_i0")  # noqa: E731
    return [
        chk("gather: IPC(static large) / IPC(static small)", "precondition",
            lambda: ratio(ipc(GATHER, big), ipc(GATHER, small)), ">=", c["mlp_bench_gain"]),
        chk("chase: IPC(static large) / IPC(static small)", "precondition",
            lambda: ratio(ipc(CHASE, big), ipc(CHASE, small)), "<=", t["chase_no_gain"]),
        chk("gather: IPC(mlp) / IPC(static small = ILP mode)", "trend",
            lambda: ratio(mlp(GATHER)["ipc"], ipc(GATHER, small)), ">=", t["speedup"]),
        chk("gather: IPC gap static small->large recovered by mlp", "trend",
            lambda: recovery(mlp(GATHER)["ipc"], ipc(GATHER, small), ipc(GATHER, big)), ">=",
            t["recovery"]),
        chk("chase: IPC(mlp) / IPC(static small) (no gain)", "trend",
            lambda: ratio(mlp(CHASE)["ipc"], ipc(CHASE, small)), "<=", t["no_gain"]),
        chk("chase: IPC(mlp) / IPC(static small) (no loss)", "trend",
            lambda: ratio(mlp(CHASE)["ipc"], ipc(CHASE, small)), ">=", t["no_loss"]),
        chk("chase: mlp residency in the ILP-mode config", "trend",
            lambda: mlp(CHASE)["residency"][0], ">=", t["ilp_res_chase"]),
        chk("compute: mlp residency in the ILP-mode config", "trend",
            lambda: mlp(COMPUTE)["residency"][0], ">=", t["ilp_res_compute"]),
    ]


def best_static_ipc(ctx: Ctx, kernel: str = PHASED) -> float:
    """Return the best IPC over the large-input static runs of ``kernel``."""
    return max(ctx.ipc(kernel, f"c{i}") for i in range(ctx.n))


def _trace_near(ctx: Ctx, tag: str, skip: float) -> dict:
    """Return (and cache) ``near_best_fraction`` of a micro_phased run's trace.

    Args:
        ctx: Evaluation context.
        tag: Run tag on the large input.
        skip: Leading fraction of instructions skipped as warm-up.

    Returns:
        The ``near_best_fraction`` dict (also merged into ``ctx.used``).

    Raises:
        Missing: If the run or its ``window_trace.csv`` is missing.
    """
    d = ctx.outdir(PHASED, tag)
    ctx.m(PHASED, tag)
    if not (d / "window_trace.csv").exists():
        raise Missing(f"{d}/window_trace.csv")
    key = ("near", tag, skip)
    if key not in ctx._cache:
        ctx._cache[key] = near_best_fraction(d / "window_trace.csv", ctx.table("large"),
                                             THRESHOLDS["common"]["near_best"], skip)
        ctx.used[f"{PHASED}/large/{tag}"].update(ctx._cache[key])
    return ctx._cache[key]


def eval_B4(ctx: Ctx) -> list[Check]:
    """Return the B4 (bbv) checks on micro_phased.

    Precondition: ``phases_matter``. Trends: few phases cover most intervals,
    the number of phase ids is bounded, next-phase prediction is accurate, the
    run is near-best after warm-up, IPC is close to the best static and the
    mean allocated window is reduced.

    Args:
        ctx: Evaluation context.

    Returns:
        The checks.
    """
    t, n = THRESHOLDS["B4"], ctx.n
    tag = f"bbv_i{n - 1}"

    def ph() -> dict:
        """Return ``read_bbv_phases`` of the bbv run (recorded in ``ctx.used``)."""
        d = ctx.outdir(PHASED, tag)
        ctx.m(PHASED, tag)
        if not (d / "bbv_phases.csv").exists():
            raise Missing(f"{d}/bbv_phases.csv")
        out = read_bbv_phases(d / "bbv_phases.csv")
        ctx.used[f"{PHASED}/large/{tag}"]["bbv"] = out
        return out
    return [
        phases_matter(ctx),
        chk("BBV: share of intervals in the 3 most visited phases", "trend",
            lambda: ph()["top3_cover"], ">=", t["top3_cover"]),
        chk("BBV: phase ids allocated", "trend", lambda: ph()["phases_allocated"], "<=",
            t["max_phases"]),
        chk("BBV: next-phase prediction accuracy", "trend", lambda: ph()["accuracy"], ">=",
            t["pred_acc"]),
        chk("after warm-up: cycle share in a near-best config for the current region", "trend",
            lambda: _trace_near(ctx, tag, t["warmup"])["near_best_frac"], ">=", t["near_frac"]),
        chk("IPC(bbv) / IPC(best static)", "trend",
            lambda: ratio(ctx.ipc(PHASED, tag), best_static_ipc(ctx)), ">=", t["min_ipc"]),
        chk("bbv mean allocated window (fraction of largest)", "trend",
            lambda: ctx.m(PHASED, tag)["rob_cap_frac"], "<=", t["max_cap"]),
    ]


def eval_B5(ctx: Ctx) -> list[Check]:
    """Return the B5 (lut) checks on micro_phased (LUT trained on ``small``, run on ``large``).

    Precondition: ``phases_matter``. Trends: near-best cycle share, IPC close to
    the best static and a reduced mean allocated window.

    Args:
        ctx: Evaluation context.

    Returns:
        The checks.
    """
    t, n = THRESHOLDS["B5"], ctx.n
    tag = f"lut_i{n - 1}"
    return [
        phases_matter(ctx),
        chk("LUT (trained on small) on large: cycle share in a near-best config", "trend",
            lambda: _trace_near(ctx, tag, 0.0)["near_best_frac"], ">=", t["near_frac"]),
        chk("IPC(lut) / IPC(best static)", "trend",
            lambda: ratio(ctx.ipc(PHASED, tag), best_static_ipc(ctx)), ">=", t["min_ipc"]),
        chk("lut mean allocated window (fraction of largest)", "trend",
            lambda: ctx.m(PHASED, tag)["rob_cap_frac"], "<=", t["max_cap"]),
    ]


def eval_B6(ctx: Ctx) -> list[Check]:
    """Return the B6 (jones, IQ-only hints) checks.

    Trends: IPC close to the largest static window on compute, lowilp and
    gather, and a reduced mean allocated IQ on compute and lowilp.

    Args:
        ctx: Evaluation context.

    Returns:
        The checks.
    """
    t, n = THRESHOLDS["B6"], ctx.n
    tag, big = f"jones_iq_i{n - 1}", hi(n)
    out = []
    for k in (COMPUTE, LOWILP, GATHER):
        out.append(chk(f"{k.split('_', 1)[1]}: IPC(jones, IQ only) / IPC(static large)", "trend",
                       lambda k=k: ratio(ctx.ipc(k, tag), ctx.ipc(k, big)), ">=", t["min_ipc"]))
    for k in (COMPUTE, LOWILP):
        out.append(chk(f"{k.split('_', 1)[1]}: jones mean allocated IQ (fraction of largest)",
                       "trend", lambda k=k: ctx.m(k, tag)["iq_cap_frac"], "<=", t["max_iq_cap"]))
    return out


def eval_B7(ctx: Ctx) -> list[Check]:
    """Return the B7 (pgo) checks on micro_phased.

    Precondition: ``phases_matter``. Trends: every phase region's small-input
    best config is near-best on large, the pgo run reaches the per-region
    oracle IPC and the largest static IPC, with a reduced mean allocated window.

    Args:
        ctx: Evaluation context.

    Returns:
        The checks.
    """
    t, c, n = THRESHOLDS["B7"], THRESHOLDS["common"], ctx.n
    tag = f"pgo_i{n - 1}"

    def transfer() -> tuple[list[int], dict, dict, list[int]]:
        """Return phase regions, small profile, large oracle and the non-transferring regions."""
        tab_l, tab_s = ctx.table("large"), ctx.table("small")
        prof = best_configs(tab_s, c["tie_tol"])
        regs = phase_regions(tab_l, c["region_min_share"])
        bad = [r for r in regs if r not in prof or prof[r] not in tab_l[r]
               or tab_l[r][prof[r]]["ipc"] < c["near_best"] * max(v["ipc"] for v in tab_l[r].values())]
        return regs, prof, best_configs(tab_l, c["tie_tol"]), bad

    def oracle_ipc() -> float:
        """Return the IPC of running every region in its large-input best config."""
        tab_l = ctx.table("large")
        best_l = best_configs(tab_l, c["tie_tol"])
        cyc = sum(tab_l[r][best_l[r]]["cycles"] for r in tab_l)
        ins = sum(tab_l[r][best_l[r]]["insts"] for r in tab_l)
        return ratio(ins, cyc)

    def pgo_vs_oracle() -> float:
        """Return IPC(pgo) / oracle IPC, recording the maps in the run's metrics."""
        run = ctx.m(PHASED, tag)
        o = oracle_ipc()
        regs, prof, best_l, _ = transfer()
        run.update(oracle_ipc=o, profile_small=prof, oracle_large=best_l)
        return ratio(run["ipc"], o)
    return [
        phases_matter(ctx),
        chk("phase regions whose small-input config is not near-best on large", "trend",
            lambda: len(transfer()[3]), "<=", 0,
            lambda: "regions {}; profile(small) {}; oracle(large) {}".format(*transfer()[:3])),
        chk("IPC(pgo, large) / per-region oracle IPC (large)", "trend", pgo_vs_oracle, ">=",
            t["min_oracle"]),
        chk("IPC(pgo) / IPC(static large)", "trend",
            lambda: ratio(ctx.ipc(PHASED, tag), ctx.ipc(PHASED, hi(n))), ">=", t["min_ipc"]),
        chk("pgo mean allocated window (fraction of largest)", "trend",
            lambda: ctx.m(PHASED, tag)["rob_cap_frac"], "<=", t["max_cap"]),
    ]


def eval_B9(ctx: Ctx) -> list[Check]:
    """Return the B9 (ltp) checks.

    Precondition: a small IQ/LSQ hurts gather. Trends: LTP recovers part of the
    gap and speeds gather up over the small IQ/LSQ, without hurting compute.

    Args:
        ctx: Evaluation context.

    Returns:
        The checks.
    """
    t, n = THRESHOLDS["B9"], ctx.n
    big = hi(n)
    ipc = ctx.ipc
    return [
        chk("gather: IPC(large IQ/LSQ) / IPC(small IQ/LSQ, large ROB, no LTP)", "precondition",
            lambda: ratio(ipc(GATHER, big), ipc(GATHER, "smalliq_i0")), ">=", t["small_iq_loss"]),
        chk("gather: IPC gap small->large IQ/LSQ recovered by LTP", "trend",
            lambda: recovery(ipc(GATHER, "ltp_i0"), ipc(GATHER, "smalliq_i0"), ipc(GATHER, big)),
            ">=", t["recovery"]),
        chk("gather: IPC(LTP) / IPC(small IQ/LSQ)", "trend",
            lambda: ratio(ipc(GATHER, "ltp_i0"), ipc(GATHER, "smalliq_i0")), ">=", t["speedup"]),
        chk("compute: IPC(LTP) / IPC(small IQ/LSQ) (no harm)", "trend",
            lambda: ratio(ipc(COMPUTE, "ltp_i0"), ipc(COMPUTE, "smalliq_i0")), ">=",
            t["no_harm"]),
    ]


@dataclass
class Baseline:
    """A gem5-side baseline under test.

    Attributes:
        id: Baseline id (``B2`` ... ``B9``).
        policy: Policy (and variant) evaluated, for the report.
        paper: Reference publication(s).
        trend: Qualitative trend from the paper that is checked.
        runs: Returns the baseline's run specs for ``n`` configs.
        evaluate: Returns the baseline's checks for a context.
    """
    id: str
    policy: str
    paper: str
    trend: str
    runs: Callable[[int], list[RunSpec]]
    evaluate: Callable[[Ctx], list[Check]]


BASELINES: dict[str, Baseline] = {b.id: b for b in [
    Baseline("B2", "occupancy",
             "Ponomarev, Kucuk, Ghose, MICRO-34 2001",
             "occupancy-driven resizing gives large savings in allocated IQ/ROB/LSQ with a small "
             "IPC loss where the window is under-used, and grows back on dispatch stalls",
             runs_B2, eval_B2),
    Baseline("B3", "mlp",
             "Kora, Yamaguchi, Ando, MICRO-46 2013",
             "the window enlarges under memory-level parallelism: big speedup over the small "
             "(ILP-mode) window on independent misses, no gain and no enlargement on pointer "
             "chasing or compute",
             runs_B3, eval_B3),
    Baseline("B4", "bbv",
             "Sherwood, Sair, Calder, ISCA 2003",
             "BBV signatures detect the program's phases, the run-length Markov predictor "
             "predicts the next phase accurately, and recurring phases get their best "
             "configuration after warm-up",
             runs_B4, eval_B4),
    Baseline("B5", "lut",
             "Dubach, Jones, Bonilla, TACO 10(4) 2013 (our LUT pipeline)",
             "a counter-based model trained offline predicts a near-best configuration per "
             "window on a phased program (trained on another input)",
             runs_B5, eval_B5),
    Baseline("B6", "hint (jones, structs=iq)",
             "Jones, O'Boyle, Abella, Gonzalez, HPCA 2005",
             "compiler-directed IQ resizing reduces the IQ size with negligible IPC loss",
             runs_B6, eval_B6),
    Baseline("B7", "hint (pgo)",
             "Huang, Renau, Torrellas, ISCA 2003; Lau, Perelman, Calder, CGO 2006",
             "per-region configurations chosen from a profiling run transfer to another input, "
             "close to the per-region oracle",
             runs_B7, eval_B7),
    Baseline("B9", "ltp",
             "Sembrant et al., MICRO-48 2015",
             "with a small IQ/LSQ, Long-Term Parking recovers much of the IPC lost versus a large "
             "IQ/LSQ on MLP-rich code, and does not hurt code without long-latency loads",
             runs_B9, eval_B9),
]}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def binary_path(bin_root: Path, spec: RunSpec, machine: str) -> Path:
    """Return the binary of a run.

    ``pgo`` binaries are per machine for non-default machines; ``jones``,
    ``jones_full`` and ``winhint`` use a per-machine build when it exists.
    Everything else is ``<bin_root>/<variant>/<kernel>``.

    Args:
        bin_root: Root of the micro binaries.
        spec: Run spec.
        machine: Machine name.

    Returns:
        The binary path (not checked for existence).
    """
    if spec.variant == "pgo" and machine != DEFAULT_MACHINE:
        return bin_root / "pgo" / machine / spec.kernel
    if spec.variant in ("jones", "jones_full", "winhint") and machine != DEFAULT_MACHINE:
        p = bin_root / spec.variant / machine / spec.kernel
        if p.exists():
            return p
    return bin_root / spec.variant / spec.kernel


def gem5_cmd(gem5: str, outdir: Path, machine_json: Path, binary: Path, spec: RunSpec,
             lut: Path | None, period: int | None) -> list[str]:
    """Return the gem5 command line of a run (se.py with ``--window-trace`` always on).

    Args:
        gem5: gem5 binary.
        outdir: Run output directory.
        machine_json: Machine JSON.
        binary: Benchmark binary.
        spec: Run spec (policy, initial config, input, window args).
        lut: ``--window-lut`` file, or None.
        period: ``--window-period``, or None for the JSON default.

    Returns:
        The argument list.
    """
    cmd = [gem5, f"--outdir={outdir}", "--redirect-stdout", "--redirect-stderr", str(SE_PY),
           "--machine", str(machine_json), "--cmd", str(binary), "--options", spec.input,
           "--window-policy", spec.policy, "--window-initial", str(spec.initial),
           "--window-trace"]
    if period:
        cmd += ["--window-period", str(period)]
    if spec.window_args:
        cmd += ["--window-args", spec.window_args]
    if lut is not None:
        cmd += ["--window-lut", str(lut)]
    return cmd


def is_done(outdir: Path) -> bool:
    """Return whether ``outdir`` holds ``stats.txt`` and a ``run.json`` with status ``ok``."""
    rj = outdir / "run.json"
    if not (rj.exists() and (outdir / "stats.txt").exists()):
        return False
    try:
        return json.loads(rj.read_text()).get("status") == "ok"
    except (OSError, ValueError):
        return False


def execute(cmd: list[str], outdir: Path, spec: RunSpec, lock: Path | None,
            timeout: int | None) -> dict:
    """Run one gem5 command and record it.

    Writes ``gem5.log`` (combined stdout/stderr) and ``run.json`` (spec,
    command, start time, return code, status, wall time) into ``outdir``.
    A timeout gives return code -9.

    Args:
        cmd: gem5 command line.
        outdir: Run output directory (created).
        spec: Run spec stored in ``run.json``.
        lock: Lock file to wrap the command in ``flock``, or None.
        timeout: Timeout in seconds, or None.

    Returns:
        The ``run.json`` metadata; ``status`` is ``ok`` iff the return code is 0
        and ``stats.txt`` is non-empty.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"spec": {"kernel": spec.kernel, "tag": spec.tag, "variant": spec.variant,
                     "policy": spec.policy, "initial": spec.initial,
                     "window_args": spec.window_args, "input": spec.input},
            "command": " ".join(shlex.quote(c) for c in cmd),
            "started": dt.datetime.now().isoformat(timespec="seconds")}
    if lock is not None:
        lock.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["flock", str(lock)] + cmd
        meta["lock"] = str(lock)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(REPO), timeout=timeout, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        rc, log = p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        rc, log = -9, e.stdout if isinstance(e.stdout, str) else ""
        meta["timeout"] = timeout
    (outdir / "gem5.log").write_text(log or "")
    st = outdir / "stats.txt"
    ok = rc == 0 and st.exists() and st.stat().st_size > 0
    meta.update(returncode=rc, status="ok" if ok else "failed",
                wall_seconds=round(time.time() - t0, 1))
    (outdir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# ---------------------------------------------------------------------------
# Derived artifacts: oracle maps, B5 LUT, B7 pgo binary
# ---------------------------------------------------------------------------

def derive_oracle(ctx: Ctx, input_size: str) -> Path:
    """Write the micro_phased oracle map and table for ``input_size``.

    Writes ``<root>/oracle/<machine>/<input>/micro_phased.json`` (flat
    ``{region: config}``) and ``micro_phased.table.csv``.

    Args:
        ctx: Evaluation context.
        input_size: Input whose static runs are used.

    Returns:
        Path of the flat map.

    Raises:
        Missing: If the static runs are not available.
    """
    tab = ctx.table(input_size)
    best = best_configs(tab, THRESHOLDS["common"]["tie_tol"])
    out = ctx.root / "oracle" / ctx.machine["name"] / input_size
    out.mkdir(parents=True, exist_ok=True)
    write_flat_oracle(out / f"{PHASED}.json", best)
    lines = ["region,config,cycles,insts,ipc,best"]
    for r in sorted(tab):
        for cfg in sorted(tab[r]):
            v = tab[r][cfg]
            lines.append(f"{r},{cfg},{v['cycles']:.0f},{v['insts']:.0f},{v['ipc']:.6f},"
                         f"{int(best[r] == cfg)}")
    (out / f"{PHASED}.table.csv").write_text("\n".join(lines) + "\n")
    return out / f"{PHASED}.json"


def derive_lut(ctx: Ctx, python: str = sys.executable) -> Path:
    """Derive the B5 runtime LUT of micro_phased.

    Labels the small-input static traces of micro_phased with the small-input
    oracle map, trains (tree model, no held-out split: one kernel) and exports
    the runtime LUT (interfaces.md §6). Output goes to
    ``<root>/b5/<machine>/`` (``dataset.csv``, ``model.pkl``, ``lut.txt``,
    ``pipeline.log``).

    Args:
        ctx: Evaluation context.
        python: Interpreter used for the training/export scripts.

    Returns:
        Path of ``lut.txt``.

    Raises:
        Missing: If a small-input static run or trace is missing.
        RuntimeError: If training or export fails.
    """
    import pandas as pd
    from label_phases import label_trace
    from whdata import load_oracle
    oracle_json = derive_oracle(ctx, "small")
    oracle = load_oracle(oracle_json)
    out = ctx.root / "b5" / ctx.machine["name"]
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(ctx.n):
        tr = ctx.outdir(PHASED, f"c{i}", "small") / "window_trace.csv"
        if not tr.exists():
            raise Missing(str(tr))
        frames.append(label_trace(tr, oracle, PHASED, ctx.machine["name"], "small", i))
    ds = out / "dataset.csv"
    pd.concat(frames, ignore_index=True).to_csv(ds, index=False)
    for argv in ([python, str(LUT_DIR / "train_phase_classifier.py"), "--dataset", str(ds),
                  "--out-dir", str(out), "--model", "tree", "--split", "none",
                  "--n-configs", str(ctx.n)],
                 [python, str(LUT_DIR / "export_lookup_table.py"), "--model",
                  str(out / "model.pkl"), "--dataset", str(ds), "--runtime-lut",
                  str(out / "lut.txt"), "--check"]):
        p = subprocess.run(argv, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        with (out / "pipeline.log").open("a") as fh:
            fh.write(" ".join(argv) + "\n" + p.stdout + "\n")
        if p.returncode != 0:
            raise RuntimeError(f"B5 pipeline step failed ({p.returncode}): {' '.join(argv)}\n"
                               f"{p.stdout[-2000:]}")
    return out / "lut.txt"


def pgo_make_cmd(map_dir: Path, machine_json: Path) -> list[str]:
    """Return the make command that builds the B7 pgo micro_phased from the maps in ``map_dir``."""
    return ["make", "-j1", "-C", str(REPO / "benchmarks"), "micro-pgo", "MICRO_HINTED=pgo",
            f"MICRO_MAP_DIR={map_dir}", f"MACHINE={machine_json}", f"MICRO_KERNELS={PHASED}"]


def derive_pgo(ctx: Ctx, machine_json: Path) -> Path:
    """B7: per-region best config profiled on `small` -> pgo build of micro_phased.

    Args:
        ctx: Evaluation context.
        machine_json: Machine JSON passed to the build.

    Returns:
        Path of the small-input oracle map used as profile.

    Raises:
        Missing: If the small-input static runs are missing.
        RuntimeError: If the make command fails.
    """
    oracle_json = derive_oracle(ctx, "small")
    cmd = pgo_make_cmd(oracle_json.parent, machine_json)
    p = subprocess.run(cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True)
    if p.returncode != 0:
        raise RuntimeError(f"pgo build failed: {' '.join(cmd)}\n{p.stdout[-2000:]}")
    return oracle_json


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def collect_runs(baselines: list[str], n: int) -> list[RunSpec]:
    """Return the deduplicated, resolved runs of the given baselines.

    Args:
        baselines: Baseline ids.
        n: Number of window configs.

    Returns:
        One spec per run key, in first-use order.

    Raises:
        ValueError: If two baselines define different specs for the same key.
    """
    seen: dict[tuple, RunSpec] = {}
    for b in baselines:
        for s in BASELINES[b].runs(n):
            s = resolve(s, n)
            if s.key in seen and seen[s.key] != s:
                raise ValueError(f"conflicting run specs for {s.label()}: {seen[s.key]} vs {s}")
            seen.setdefault(s.key, s)
    return list(seen.values())


def verdict_path(root: Path, machine: str, name: str) -> Path:
    """Return ``<root>[/<machine>]/<name>.json`` (no machine level for the default machine)."""
    base = root if machine == DEFAULT_MACHINE else root / machine
    return base / f"{name}.json"


def evaluate(ctx: Ctx, bid: str) -> dict:
    """Evaluate one baseline and return its verdict record.

    Args:
        ctx: Evaluation context (``used`` is reset).
        bid: Baseline id.

    Returns:
        Dict with baseline metadata, thresholds, ``status``, ``checks``, the
        metrics of the ``runs`` used and, if any, ``missing``.
    """
    b = BASELINES[bid]
    ctx.used = {}
    out = {"baseline": bid, "policy": b.policy, "paper": b.paper, "trend": b.trend,
           "machine": ctx.machine["name"],
           "thresholds": {**THRESHOLDS["common"], **THRESHOLDS[bid]},
           "evaluated": dt.datetime.now().isoformat(timespec="seconds")}
    checks = b.evaluate(ctx)
    out.update(status=verdict(checks), checks=[c.as_dict() for c in checks], runs=ctx.used)
    missing = sorted({c.missing for c in checks if c.missing})
    if missing:
        out["missing"] = missing
    return out


def parse_args(argv=None):
    """Parse the command line.

    Args:
        argv (list[str] | None): Argument list; ``None`` reads ``sys.argv``.

    Returns:
        args (argparse.Namespace): The parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baselines", nargs="*", default=list(BASELINES),
                   help=f"subset of {list(BASELINES)} (default: all)")
    p.add_argument("--machine", default=DEFAULT_MACHINE, help="sim/machines/<name>.json")
    p.add_argument("--machines-dir", type=Path, default=MACHINES_DIR)
    p.add_argument("--results", type=Path, default=RESULTS, help="output root")
    p.add_argument("--bin-root", type=Path, default=BIN_ROOT,
                   help="micro binaries: <bin-root>/<variant>/<kernel>")
    p.add_argument("--gem5", default=GEM5)
    p.add_argument("--period", type=int, default=None, help="--window-period (default: JSON)")
    p.add_argument("--timeout", type=int, default=None, help="per-run timeout (s)")
    p.add_argument("--only", nargs="*", default=None, metavar="GLOB",
                   help="restrict the gem5 runs to labels <kernel>/<input>/<tag> matching any "
                        "glob (e.g. 'micro_gather/*'); predicates needing other runs are "
                        "reported INCOMPLETE")
    p.add_argument("--dry-run", action="store_true", help="list runs and derive steps only")
    p.add_argument("--evaluate-only", action="store_true", help="no gem5 runs, no derive steps")
    p.add_argument("--build", action="store_true",
                   help="run `make -C benchmarks micro` first (cheap, -j1)")
    p.add_argument("--force", action="store_true", help="rerun runs that are already done")
    p.add_argument("--no-lock", action="store_true",
                   help=f"do not wrap gem5 in flock {HEAVY_LOCK} (tests only)")
    a = p.parse_args(argv)
    bad = [b for b in a.baselines if b not in BASELINES]
    if bad:
        p.error(f"unknown baseline(s) {bad}; choose from {list(BASELINES)}")
    return a


def main(argv=None) -> int:
    """Run the fidelity study (see the module docstring).

    Runs phase-1 runs, derives the B5 LUT / B7 pgo binary, runs the dependent
    runs, then writes ``<baseline>.json`` per baseline and merges the statuses
    into ``summary.json``.

    Args:
        argv (list[str] | None): Argument list; ``None`` reads ``sys.argv``.

    Returns:
        0 if every run and derive step succeeded (or ``--dry-run``), else 1.
        Verdicts other than PASS do not affect the exit code.
    """
    a = parse_args(argv)
    mjson = a.machines_dir / f"{a.machine}.json"
    machine = load_machine(mjson)
    machine["name"] = a.machine
    n = len(machine["window_table"])
    runs = collect_runs(a.baselines, n)
    if a.only:
        runs = [s for s in runs if any(fnmatch.fnmatchcase(s.label(), g) for g in a.only)]
    ctx = Ctx(a.results, machine, n)
    lock = None if a.no_lock else HEAVY_LOCK
    lut_path = a.results / "b5" / a.machine / "lut.txt"
    need_lut = any(s.needs == "lut" for s in runs)
    need_pgo = any(s.needs == "pgo" for s in runs)
    if a.only and not runs:
        print(f"[WARN] --only {a.only} selects no run", file=sys.stderr)

    def cmd_for(s: RunSpec) -> list[str]:
        """Return the gem5 command of run ``s``."""
        return gem5_cmd(a.gem5, ctx.outdir(s.kernel, s.tag, s.input), mjson,
                        binary_path(a.bin_root, s, a.machine), s,
                        lut_path if s.needs == "lut" else None, a.period)

    if a.dry_run:
        print(f"# machine {a.machine} ({n} configs), baselines {' '.join(a.baselines)}: "
              f"{len(runs)} gem5 runs (shared runs listed once)")
        for phase, sel in (("1", [s for s in runs if not s.needs]),
                           ("2", [s for s in runs if s.needs])):
            if phase == "2" and (need_lut or need_pgo):
                print("# derive: per-region oracle maps of micro_phased "
                      f"-> {a.results / 'oracle' / a.machine}/<input>/")
                if need_lut:
                    print(f"# derive: B5 LUT (label_phases + train tree + export) -> {lut_path}")
                if need_pgo:
                    print("# derive: B7 pgo build: " + " ".join(
                        pgo_make_cmd(a.results / "oracle" / a.machine / "small", mjson)))
            for s in sel:
                done = "done" if is_done(ctx.outdir(s.kernel, s.tag, s.input)) else "todo"
                users = [b for b in a.baselines if s.key in {resolve(r, n).key
                                                              for r in BASELINES[b].runs(n)}]
                print(f"[{phase}] {done:4s} {s.label():40s} {','.join(users):10s} "
                      f"{' '.join(shlex.quote(c) for c in cmd_for(s))}")
        return 0

    if a.build:
        subprocess.run(["make", "-j1", "-C", str(REPO / "benchmarks"), "micro",
                        f"MACHINE={mjson}"], check=True)

    def run_all(sel: list[RunSpec]) -> int:
        """Run the specs not yet done (or all with ``--force``); return the number of failures."""
        failed = 0
        for s in sel:
            od = ctx.outdir(s.kernel, s.tag, s.input)
            if is_done(od) and not a.force:
                print(f"[SKIP] {s.label()} (done)")
                continue
            b = binary_path(a.bin_root, s, a.machine)
            if not b.is_file():
                print(f"[MISS] {s.label()}: binary {b} (make -C benchmarks micro)", file=sys.stderr)
                failed += 1
                continue
            print(f"[RUN ] {s.label()} ...", flush=True)
            meta = execute(cmd_for(s), od, s, lock, a.timeout)
            print(f"[{'OK' if meta['status'] == 'ok' else 'FAIL':4s}] {s.label()} "
                  f"({meta['wall_seconds']} s)", flush=True)
            failed += meta["status"] != "ok"
        return failed

    failed = 0
    if not a.evaluate_only:
        failed += run_all([s for s in runs if not s.needs])
        derived_ok = True
        try:
            if need_lut:
                print(f"[DERIVE] B5 LUT -> {derive_lut(ctx)}")
            if need_pgo:
                print(f"[DERIVE] B7 profile map {derive_pgo(ctx, mjson)} + pgo build")
        except (Missing, RuntimeError) as e:
            print(f"[ERROR] derive step: {e}", file=sys.stderr)
            derived_ok = False
            failed += 1
        if derived_ok:
            failed += run_all([s for s in runs if s.needs])
    summary = {}
    for bid in a.baselines:
        res = evaluate(ctx, bid)
        path = verdict_path(a.results, a.machine, bid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=2, default=float) + "\n")
        summary[bid] = res["status"]
        print(f"[{res['status']}] {bid} {res['policy']}: {path}")
        for c in res.get("checks", []):
            mark = "-- " if "missing" in c else ("ok " if c["ok"] else "NO ")
            print(f"    {mark} {c['kind'][:5]} {c['name']}: {c['value']} {c['op']} {c['threshold']}")
        for m in res.get("missing", []):
            print(f"    missing: {m}")
    sp = verdict_path(a.results, a.machine, "summary")
    old = json.loads(sp.read_text()) if sp.exists() else {}
    old.update(summary)
    sp.write_text(json.dumps(old, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
