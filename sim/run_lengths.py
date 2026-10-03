#!/usr/bin/env python3
r"""
run_lengths.py - run-length planning and sample merging for gem5 runs.

The method and the per-kernel budgets are in sim/run_lengths.json (versioned;
its "_doc" field is the rationale). This module:

  spec_for(cfg, kernel, input)   budget of one kernel/input (defaults + overrides)
  plan(spec, profile)            the samples of a run: checkpoint start, warm-up and
                                 measured instructions, stratum (region) and the
                                 instructions the stratum represents
  merge_samples(outdir, plan, sample_dirs)
                                 stats.txt (stratified whole-program estimate),
                                 stats.measured.txt, region_stats.csv (measured rows),
                                 window_trace.csv (measured periods), runlength.json

It is used by sim/run_experiments.py (which also schedules the profile and checkpoint
passes, see there), sim/baselines/oracle/oracle_sweep.py and
sim/baselines/tune/tune_baselines.py, so every consumer simulates the same windows.

CLI (inspection only, runs nothing):
  python sim/run_lengths.py show --kernel encoder_bert_tiny_infer --input large \
      [--profile results/runlen/oracle/encoder_bert_tiny_infer.large.json]
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: default run-length config (method + per-kernel budgets)
CONFIG = REPO / "sim" / "run_lengths.json"
#: the measurement starts MARGIN - 1 instructions before the region marker, so the
#: marker is still ahead when the stats are reset (instruction counts are exact to +-1)
MARGIN = 2
#: run-length modes accepted in a spec
MODES = ("full", "regions", "fixed")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path | str | None = None) -> dict:
    """Load the run-length configuration (sim/run_lengths.json by default).

    Args:
        path: Config file; None/empty selects ``CONFIG``.

    Returns:
        The config dict with ``_path`` set to the file it was read from.

    Raises:
        ValueError: If the file has no ``defaults`` section.
    """
    p = Path(path) if path else CONFIG
    cfg = json.loads(p.read_text())
    if "defaults" not in cfg:
        raise ValueError(f"{p}: no 'defaults' section")
    cfg["_path"] = str(p)
    return cfg


def spec_for(cfg: dict, kernel: str, input_size: str) -> dict:
    """Return the run-length budget of one kernel/input.

    The defaults for the input are overridden by ``kernels.<kernel>.<input>``.

    Args:
        cfg: Config from :func:`load_config`.
        kernel: Kernel name.
        input_size: Input size (e.g. ``"large"``).

    Returns:
        A fresh spec dict with ``mode`` (default ``"full"``) and ``version``
        (the config's version, 0 if absent).

    Raises:
        ValueError: If ``mode`` is not one of ``MODES``.
    """
    spec = copy.deepcopy(cfg.get("defaults", {}).get(input_size, {"mode": "full"}))
    spec.update(cfg.get("kernels", {}).get(kernel, {}).get(input_size, {}))
    spec.setdefault("mode", "full")
    if spec["mode"] not in MODES:
        raise ValueError(f"run length of {kernel}/{input_size}: unknown mode {spec['mode']!r}")
    spec["version"] = cfg.get("version", 0)
    return spec


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """One simulated sample of a run.

    Attributes:
        idx: Sample index within the plan (used in :attr:`name`).
        start: Checkpoint instruction count (0 = from the program start).
        warmup: Detailed warm-up instructions after the restore.
        measure: Measured instructions.
        stratum: Region id (-1 = code before the first marker; fixed mode: -2).
        stratum_insts: Dynamic instructions of the whole stratum (profile).
        visit: Visit number of the region (1-based; 0 = n/a).
    """
    idx: int
    start: int          # checkpoint instruction count (0 = from the program start)
    warmup: int         # detailed warm-up instructions after the restore
    measure: int        # measured instructions
    stratum: int        # region id (-1 = code before the first marker; fixed mode: -2)
    stratum_insts: int  # dynamic instructions of the whole stratum (profile)
    visit: int = 0      # visit number of the region (1-based; 0 = n/a)

    @property
    def name(self) -> str:
        """Return the sample's directory name, ``s<idx>`` (two-digit zero-padded)."""
        return f"s{self.idx:02d}"


@dataclass
class Plan:
    """The samples of one run and how they represent the whole program.

    Attributes:
        mode: ``"full"``, ``"regions"`` or ``"fixed"``.
        version: Version of the run-length config the plan was made from.
        samples: Samples to simulate (empty for ``full``).
        total_insts: Dynamic instructions of the whole program, if known.
        unsampled_frac: Fraction of the program's instructions in strata with
            no sample (``regions`` mode).
        approximate: True if the region profile was truncated (``profile_cap``
            reached).
        reason: Why the plan fell back to ``full`` (empty otherwise).
    """
    mode: str                       # "full" | "regions" | "fixed"
    version: int = 0
    samples: list[Sample] = field(default_factory=list)
    total_insts: int | None = None
    unsampled_frac: float = 0.0
    approximate: bool = False       # profile truncated (profile_cap reached)
    reason: str = ""

    @property
    def starts(self) -> list[int]:
        """Return the distinct checkpoint instruction counts (> 0) the samples need, sorted."""
        return sorted({s.start for s in self.samples if s.start > 0})

    def detailed_insts(self) -> int:
        """Return the total detailed (warm-up + measured) instructions of all samples."""
        return sum(s.warmup + s.measure for s in self.samples)

    def to_json(self) -> dict:
        """Return the plan as a JSON-serialisable dict (dataclass fields plus ``starts``)."""
        d = asdict(self)
        d["starts"] = self.starts
        return d


def _evenly(n_items: int, k: int) -> list[int]:
    """Return ``k`` (at most ``n_items``) indices spread evenly over ``range(n_items)``.

    All indices when ``k`` >= ``n_items``; otherwise ``[0]`` when ``k`` <= 1, else
    a set including the first and last index (rounding may merge duplicates).
    """
    if k >= n_items:
        return list(range(n_items))
    if k <= 1:
        return [0]
    return sorted({round(j * (n_items - 1) / (k - 1)) for j in range(k)})


def segments(profile: dict) -> list[tuple[int, int, int, int]]:
    """Split a region profile into strata segments.

    A visit's enter count (profile) is the retired-instruction count right
    after its marker, so the marker is instruction index enter - 1 and the
    visit spans [enter - 1, next marker index). Code before the first marker
    forms stratum -1 (only if non-empty).

    Args:
        profile: Region profile JSON written by ``se.py --profile-regions``
            (``total_insts``, ``visits``).

    Returns:
        ``(region, visit, marker index, length)`` of stratum -1 and every
        recorded visit, ordered by marker index.
    """
    total = int(profile["total_insts"])
    visits = sorted(((int(r), int(c), int(n) - 1) for r, c, n in profile.get("visits", [])),
                    key=lambda v: v[2])
    segs = []
    first = visits[0][2] if visits else total
    if first > 0:
        segs.append((-1, 0, 0, first))
    for i, (r, c, p) in enumerate(visits):
        end = visits[i + 1][2] if i + 1 < len(visits) else total
        segs.append((r, c, p, max(0, end - p)))
    return segs


def plan(spec: dict, profile: dict | None = None) -> Plan:
    """Return the samples of one run (see sim/run_lengths.json).

    ``full``: no samples. ``fixed``: one sample per ``starts`` entry, each
    warmed for up to ``warmup_insts`` and measuring ``measure_insts``.
    ``regions``: falls back to ``full`` if the program has at most
    ``full_below_insts`` instructions; otherwise each region whose share of
    the instructions is at least ``min_region_frac`` gets up to ``per_region``
    evenly spaced visits, strata are taken in order of decreasing share and cut
    at ``max_samples``, and each sample's measurement starts ``MARGIN - 1``
    instructions before its marker and is capped at ``measure_insts``.

    Args:
        spec: Spec from :func:`spec_for`.
        profile: Region profile (required for ``regions``).

    Returns:
        The :class:`Plan`; samples are ordered by start position.

    Raises:
        ValueError: If ``measure_insts`` <= 0 (non-full modes), ``fixed`` has no
            ``starts`` or ``regions`` has no profile.
    """
    mode, ver = spec.get("mode", "full"), spec.get("version", 0)
    if mode == "full":
        return Plan("full", ver)
    w = int(spec.get("warmup_insts", 0))
    m = int(spec.get("measure_insts", 0))
    if m <= 0:
        raise ValueError("run length: measure_insts must be > 0")
    if mode == "fixed":
        starts = [int(x) for x in spec.get("starts", [])]
        if not starts:
            raise ValueError("run length mode 'fixed' needs a 'starts' list")
        samples = [Sample(i, max(0, s - w), s - max(0, s - w), m, -2, m)
                   for i, s in enumerate(sorted(starts))]
        total = profile.get("total_insts") if profile else None
        return Plan("fixed", ver, samples,
                    int(total) if total is not None else None)
    # regions
    if profile is None:
        raise ValueError("run length mode 'regions' needs a region profile")
    total = int(profile["total_insts"])
    if total <= int(spec.get("full_below_insts", 0)):
        return Plan("full", ver, total_insts=total,
                    reason=f"total {total} <= full_below_insts")
    by_region: dict[int, list] = {}
    for seg in segments(profile):
        by_region.setdefault(seg[0], []).append(seg)
    share = {r: sum(s[3] for s in v) for r, v in by_region.items()}
    min_frac = float(spec.get("min_region_frac", 0.0))
    per = max(1, int(spec.get("per_region", 1)))
    chosen = []
    for r in sorted(by_region, key=lambda r: (-share[r], r)):
        if share[r] <= 0 or share[r] / max(total, 1) < min_frac:
            continue
        segs = [s for s in by_region[r] if s[3] > 0]
        for j in _evenly(len(segs), per):
            chosen.append(segs[j])
    cap = int(spec.get("max_samples", 0) or 0)
    if cap and len(chosen) > cap:
        chosen = chosen[:cap]           # strata are ordered by share: keep the largest
    kept = {s[0] for s in chosen}
    unsampled = sum(v for r, v in share.items() if r not in kept)
    samples = []
    lead = MARGIN - 1          # instructions measured before the marker
    for i, (r, c, p, length) in enumerate(sorted(chosen, key=lambda s: s[2])):
        boundary = max(0, p - lead)
        start = max(0, boundary - w)
        samples.append(Sample(i, start, boundary - start, min(m, length + (p - boundary)),
                              r, share[r], c))
    return Plan("regions", ver, samples, total, unsampled / max(total, 1),
                bool(profile.get("truncated")))


def sample_args(s: Sample, ckpt_dir: Path | None) -> list[str]:
    """Return the se.py flags of one sample (added to the run's normal command line).

    Args:
        s: The sample.
        ckpt_dir: Directory holding ``cpt.<N>`` checkpoints.

    Returns:
        ``--restore-checkpoint <ckpt_dir>/cpt.<start>`` (if ``start`` > 0) and
        ``--warmup-insts`` (if ``warmup`` > 0).

    Raises:
        ValueError: If the sample needs a checkpoint but ``ckpt_dir`` is None.
    """
    out = []
    if s.start > 0:
        if ckpt_dir is None:
            raise ValueError("sample with start > 0 needs a checkpoint directory")
        out += ["--restore-checkpoint", str(Path(ckpt_dir) / f"cpt.{s.start}")]
    if s.warmup:
        out += ["--warmup-insts", str(s.warmup)]
    return out


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

_BEGIN = "---------- Begin Simulation Statistics ----------"
_END = "---------- End Simulation Statistics   ----------"
_MAX_RX = re.compile(r"(OccMax|::max$|\.max$|maxBytes|::max_value)")
# means, rates and ratios (cycle-weighted); everything else is an additive counter
_AVG_RX = re.compile(r"([Mm]ean\b|[Rr]ate\b|[Rr]atio\b|\bipc\b|\bcpi\b|[Aa]vg|::stdev|"
                     r"[Bb]andwidth\b|[Uu]tilization\b|[Ff]rac\b)")
_CONST = {"simFreq"}


def read_stats(path: Path) -> list[tuple[str, float]]:
    """Return the ordered (name, value) pairs of the last stats block of a gem5 stats.txt.

    Non-numeric lines are skipped; non-finite values become 0.0.
    """
    out: list[tuple[str, float]] = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if _BEGIN in line:
                out = []
                continue
            parts = line.split()
            if len(parts) < 2 or parts[0].startswith("-"):
                continue
            try:
                v = float(parts[1])
            except ValueError:
                continue
            out.append((parts[0], v if math.isfinite(v) else 0.0))
    return out


def merge_stats(blocks: list[list[tuple[str, float]]], scales: list[float]) -> dict[str, float]:
    """Merge several stats blocks into one, scaling each block.

    Counters: sum of scale x value; maxima: max; means/rates: weighted by the
    scaled cycles of each block; recomputed where the parts are known
    (``system.cpu.ipc``, ``system.cpu.cpi``, ``*OccMean::<i>`` from
    ``*OccSum::<i>`` / ``cyclesInConfig::<i>``). ``simFreq`` is copied.

    Args:
        blocks: Outputs of :func:`read_stats`.
        scales: Per-block weight (same length as ``blocks``).

    Returns:
        Merged stats in first-seen name order.
    """
    order: list[str] = []
    acc: dict[str, float] = {}
    wsum: dict[str, float] = {}
    for blk, sc in zip(blocks, scales):
        d = dict(blk)
        cyc = d.get("system.cpu.numCycles", 1.0) * sc
        for k, v in blk:
            if k not in acc:
                order.append(k)
                acc[k], wsum[k] = 0.0, 0.0
                if _MAX_RX.search(k):
                    acc[k] = -math.inf
            if k in _CONST:
                acc[k] = v
            elif _MAX_RX.search(k):
                acc[k] = max(acc[k], v)
            elif _AVG_RX.search(k):
                acc[k] += v * cyc
                wsum[k] += cyc
            else:
                acc[k] += v * sc
    for k in order:
        if wsum.get(k):
            acc[k] /= wsum[k]
        if acc[k] == -math.inf:
            acc[k] = 0.0
    insts = acc.get("simInsts", 0.0)
    cycles = acc.get("system.cpu.numCycles", 0.0)
    if cycles:
        for k in ("system.cpu.ipc",):
            if k in acc:
                acc[k] = insts / cycles
        if "system.cpu.cpi" in acc and insts:
            acc["system.cpu.cpi"] = cycles / insts
    for k in list(acc):
        mm = re.match(r"^(.*)\.(rob|iq|lq|sq)OccMean::(\d+)$", k)
        if mm:
            s = acc.get(f"{mm.group(1)}.{mm.group(2)}OccSum::{mm.group(3)}")
            c = acc.get(f"{mm.group(1)}.cyclesInConfig::{mm.group(3)}")
            if s is not None and c:
                acc[k] = s / c
    return {k: acc[k] for k in order}


def write_stats(path: Path, stats: dict[str, float], note: str) -> None:
    """Write stats in gem5 stats.txt format (one Begin/End block).

    Integral values are written without decimals, others with 6; non-finite
    values as 0.

    Args:
        path: Output file.
        stats: Stat name -> value.
        note: Text written as the description column of every line.
    """
    with open(path, "w") as fh:
        fh.write(f"\n{_BEGIN}\n")
        for k, v in stats.items():
            v = v if math.isfinite(v) else 0.0
            txt = f"{v:.6f}" if abs(v - round(v)) > 1e-9 else f"{int(round(v))}"
            fh.write(f"{k:<60} {txt:>24}   # {note}\n")
        fh.write(f"\n{_END}\n")


def _measured_rows(path: Path, start_cycle: int) -> list[dict]:
    """Return the region_stats.csv rows of the measurement.

    Rows entered during the warm-up are dropped; a visit straddling the
    measurement start is clipped (its insts in proportion to its cycles,
    assuming a uniform IPC inside the visit).

    Args:
        path: A sample's region_stats.csv.
        start_cycle: ``measure_start_cycle`` of the sample.

    Returns:
        The kept rows as dicts (string values).
    """
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            enter, cyc, ins = int(float(r["enter_cycle"])), float(r["cycles"]), float(r["insts"])
            if enter >= start_cycle:
                rows.append(r)
            elif enter + cyc > start_cycle and cyc > 0:
                keep = enter + cyc - start_cycle
                rows.append(dict(r, enter_cycle=str(start_cycle), cycles=str(int(keep)),
                                 insts=str(int(round(ins * keep / cyc)))))
    return rows


def merge_samples(outdir: Path, pl: Plan, sample_dirs: list[Path]) -> dict:
    """Combine the sample outdirs of one run into ``outdir``.

    Writes ``stats.txt`` (in ``regions`` mode each sample is scaled by
    stratum_insts / measured instructions of its stratum; otherwise unscaled),
    ``stats.measured.txt`` (unweighted sum), ``region_stats.csv`` (measured
    rows), ``window_trace.csv`` (measured periods, cycles rebased to one
    increasing timeline) and ``runlength.json``.

    Args:
        outdir: The run's output directory.
        pl: The run's plan; ``pl.samples`` must align with ``sample_dirs``.
        sample_dirs: Sample output directories (stats.txt, optional
            runlength.json, region_stats.csv, window_trace.csv).

    Returns:
        The runlength.json summary (``plan``, ``scales``, ``measured_insts``,
        ``samples``, ``hint_state_exact``).
    """
    outdir = Path(outdir)
    infos, blocks, measured = [], [], []
    for d in sample_dirs:
        rl = d / "runlength.json"
        info = json.loads(rl.read_text()) if rl.is_file() else {}
        blk = read_stats(d / "stats.txt")
        n = info.get("measured_insts")
        if n is None:
            n = dict(blk).get("simInsts", 0.0)
        infos.append(info)
        blocks.append(blk)
        measured.append(float(n))
    if pl.mode == "regions":
        per_stratum: dict[int, float] = {}
        for s, n in zip(pl.samples, measured):
            per_stratum[s.stratum] = per_stratum.get(s.stratum, 0.0) + n
        scales = [s.stratum_insts / per_stratum[s.stratum] if per_stratum[s.stratum] else 0.0
                  for s in pl.samples]
    else:
        scales = [1.0] * len(blocks)
    note = (f"run-length v{pl.version} {pl.mode}: weighted sum of {len(blocks)} samples "
            "(sim/run_lengths.py)")
    write_stats(outdir / "stats.txt", merge_stats(blocks, scales), note)
    write_stats(outdir / "stats.measured.txt", merge_stats(blocks, [1.0] * len(blocks)),
                f"run-length v{pl.version} {pl.mode}: unweighted sum of {len(blocks)} samples")
    # region_stats.csv: measured rows of every sample
    rows = []
    for d, info in zip(sample_dirs, infos):
        p = d / "region_stats.csv"
        if p.is_file():
            rows += _measured_rows(p, int(info.get("measure_start_cycle", 0)))
    if rows:
        with open(outdir / "region_stats.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["region", "config", "enter_cycle", "cycles", "insts"],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    # window_trace.csv: measured periods, cycle rebased to one increasing timeline
    header, out_rows, base = None, [], 0
    for d, info in zip(sample_dirs, infos):
        p = d / "window_trace.csv"
        if not p.is_file():
            continue
        start = int(info.get("measure_start_cycle", 0))
        with open(p, newline="") as fh:
            rd = csv.reader(fh)
            h = next(rd, None)
            if h is None:
                continue
            header = header or h
            ci = h.index("cycle")
            last = base
            for r in rd:
                c = int(float(r[ci]))
                if c <= start:
                    continue
                r[ci] = str(base + c - start)
                last = int(r[ci])
                out_rows.append(r)
            base = last
    if header and out_rows:
        with open(outdir / "window_trace.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(out_rows)
    summary = {"plan": pl.to_json(), "scales": scales, "measured_insts": measured,
               "samples": [dict(name=s.name, **{k: v for k, v in i.items()
                                                if k not in ("ff_state",)})
                           for s, i in zip(pl.samples, infos)],
               "hint_state_exact": all(i.get("hint_state_exact", True) for i in infos)}
    (outdir / "runlength.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """CLI entry point (``show`` subcommand; inspection only).

    Prints the spec and, when a plan can be made, one line per sample.

    Args:
        argv (list[str] | None): Argument list; None means ``sys.argv[1:]``.

    Returns:
        Exit status (0).
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="print the budget and (with a profile) the sample plan")
    s.add_argument("--kernel", required=True)
    s.add_argument("--input", default="large", choices=["small", "large"])
    s.add_argument("--config", default=str(CONFIG))
    s.add_argument("--profile", default=None, help="region profile JSON (se.py --profile-regions)")
    a = p.parse_args(argv)
    spec = spec_for(load_config(a.config), a.kernel, a.input)
    print(json.dumps({"spec": spec}, indent=2))
    prof = json.loads(Path(a.profile).read_text()) if a.profile else None
    if spec["mode"] == "regions" and prof is None:
        print("(mode 'regions': pass --profile to see the samples)")
        return 0
    pl = plan(spec, prof)
    print(f"mode {pl.mode}: {len(pl.samples)} samples, {pl.detailed_insts()} detailed insts"
          + (f" of {pl.total_insts} total" if pl.total_insts else "")
          + (f", unsampled {pl.unsampled_frac:.3%}" if pl.unsampled_frac else "")
          + (" (approximate: truncated profile)" if pl.approximate else ""))
    for x in pl.samples:
        print(f"  {x.name} region {x.stratum:>3} visit {x.visit:>4}  start {x.start:>12}  "
              f"warmup {x.warmup:>9}  measure {x.measure:>9}  stratum {x.stratum_insts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
