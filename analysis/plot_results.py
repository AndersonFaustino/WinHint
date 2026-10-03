#!/usr/bin/env python3
"""
plot_results.py - WinHint evaluation figures (PROPOSAL Phase E).

Figures (PDF/PNG in --out-dir) and a metrics.json with the numbers behind them:

      wstar_vs_oracle   predicted W* (compiler) vs oracle-best window per region,
                        with Spearman rho (success criterion: rho >= ~0.8)
      ipc_bars          IPC per kernel and baseline (B0..B9, WinHint), normalised to B0-large
      ed2p_bars         ED²P per kernel and baseline, normalised to B0-large
      switch_frequency  window switches per million instructions per baseline
      hint_overhead     dynamic hints per kilo-instruction, static hint count and the
                        dynamic overhead (cycles of the winhint binary with the hints
                        ignored, variant winhint_nop, vs the plain binary, B0-large)
                        and code size (executable ELF bytes of each hinted build vs
                        plain, from --bin-root)
      sensitivity       speedup over B0-large vs L2 size and memory latency: the
                        sensitivity points of run_experiments.py --sweep (machine
                        "M+PARAM=v", summary columns sens_param/l2_kb/
                        mem_latency_cycles) when present, else across machines
      portability       B5 retrained (lut) vs not retrained (lut_xfer) vs WinHint,
                        relative to the oracle, per machine
      hw_edp            real silicon (hw/run_hw_experiments.py): package EDP of
                        R0-P/R0-E/R2/R3/R4/R5/WinHint normalised to R1, 95 % CI
      hw_nop_overhead   x86 NOP-hint overhead (asm vs plain, pinned P / E core)

Inputs
------

      --summary        results/gem5/summary.csv (sim/run_experiments.py)
      --oracle-root    results/oracle (sim/oracle_sweep.py):
                       <root>/<machine>/<input>/<kernel>.json (+ .summary.json)
      --compiler-dir   directories searched for the compiler's per-region model
                       output <kernel>.regions.json (benchmarks/Makefile writes it next
                       to the binaries: $WINHINT_BUILD/benchmarks/riscv/winhint[/<machine>]);
                       a file found without the <machine>/ level is used only for the
                       machine named in its "target" field
      --hw-summary     results/hw/summary.csv (hw/run_hw_experiments.py)
      --machines-dir   sim/machines (window tables, L2 size, memory latency)

Compiler schema: docs/reference/stats-schema.md

      <kernel>.regions.json = {"kernel": str, "target": str,
                               "regions": {"<id>": {"w_star": int,
                                                    "nest_w_star": float,
                                                    "config": int, "nest_config": int,
                                                    "function": str, "line": int, ...}}}
      <kernel>.winhint.json = {"schema": "winhint-stats/1", "hints_setwin": int,
                               "hints_region": int, "compile_time_ms": num, ...}

The predicted window of a region is nest_w_star (dynamic-instruction weighted
over the loop nest) when present, else w_star. A list of region objects with
an "id" field is accepted as well.

Window stats in summary.csv are the gem5 counters system.cpu.window.* (columns
`win.<name>`); the switch counter is the first column whose name contains
"switch" (or "resize"/"reconfig"), the dynamic hint counter the first one
containing "setwinhint" (else "hint"); with gem5's WindowController these are
win.switches and win.setwinHints.

      python3 plot_results.py --out-dir results/figures
      python3 plot_results.py --synthetic --out-dir /tmp/figs     # self-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Repository root.
REPO = Path(__file__).resolve().parents[1]
# shared readers of the gem5/oracle artifacts (kept stable by the sim/ owner)
sys.path.insert(0, str(REPO / "sim" / "baselines" / "lut"))
from whdata import load_machine, load_oracle  # noqa: E402

#: Build root (``$WINHINT_BUILD``, default ``<repo>/build``).
BUILD = Path(os.environ.get("WINHINT_BUILD") or REPO / "build")

#: Display order and labels of the variants of sim/run_experiments.py.
BASELINES = [
    ("static_c0", "B0-small"), ("static_mid", "B0-medium"), ("static_max", "B0-large"),
    ("oracle_hinted", "B1 oracle"), ("occupancy", "B2 occupancy"), ("mlp", "B3 MLP-aware"),
    ("bbv", "B4 BBV"), ("lut", "B5 LUT"), ("lut_xfer", "B5 LUT (no retrain)"),
    ("jones", "B6 Jones IQ"), ("jones_full", "B6 Jones full"), ("pgo", "B7 PGO"),
    ("clairvoyance", "B8 Clairvoyance"), ("winhint_clairvoyance", "WinHint+B8"),
    ("ltp", "B9 LTP"), ("winhint", "WinHint"), ("winhint_hw", "WinHint+HW"),
]
#: Variant name -> display label (BASELINES plus ``winhint_nop``).
LABEL = dict(BASELINES)
LABEL["winhint_nop"] = "WinHint (hints ignored)"


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def rankdata(x: np.ndarray) -> np.ndarray:
    """Return average ranks (1-based) of ``x``; ties share the mean rank.

    Args:
        x: 1-D array-like of values to rank.

    Returns:
        Float array of the same length as ``x`` with the rank of each element.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Return Spearman's rank correlation coefficient of ``x`` and ``y``.

    Computed as the Pearson correlation of the tie-averaged ranks (see :func:`rankdata`).

    Args:
        x (array-like): 1-D array-like of values.
        y (array-like): 1-D array-like of values, same length as ``x``.

    Returns:
        Spearman's rho, or NaN when fewer than two points are given or either
            rank vector is constant.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def geomean(v) -> float:
    """Return the geometric mean of the positive, finite, non-zero entries of ``v``.

    Args:
        v (iterable of float): Iterable of numbers; zero, negative, NaN, infinite and falsy entries
            are ignored.

    Returns:
        The geometric mean, or NaN when no entry qualifies.
    """
    v = np.asarray([x for x in v if x and x > 0 and np.isfinite(x)], float)
    return float(np.exp(np.log(v).mean())) if len(v) else float("nan")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def canonical_variant(variant: str, n_configs: int) -> str:
    """Map ``static_c<i>`` to ``static_c0`` / ``static_mid`` / ``static_max`` for plotting.

    Index 0 maps to ``static_c0``, the last configuration (``n_configs - 1``) to
    ``static_max`` and the middle one(s) to ``static_mid``. Any suffix after a ``.``
    or ``-`` in the index is ignored.

    Args:
        variant: Variant name as written by ``sim/run_experiments.py``.
        n_configs: Number of window configurations of the machine.

    Returns:
        The canonical variant name; other variants (and unparsable or other
            static indices) are returned unchanged.
    """
    if variant.startswith("static_c"):
        try:
            i = int(variant[len("static_c"):].split(".")[0].split("-")[0])
        except ValueError:
            return variant
        if i == n_configs - 1:
            return "static_max"
        if i == 0:
            return "static_c0"
        if i == (n_configs - 1) // 2 or i == n_configs // 2:
            return "static_mid"
    return variant


def load_summary(path: Path, machines: dict) -> pd.DataFrame:
    """Load the gem5 ``summary.csv`` and normalise it for plotting.

    Keeps only rows with ``status == "ok"`` and ``input == "large"`` (missing values
    count as such), fills ``base_machine`` (from ``machine``) and ``sens_param``
    (empty string), adds the canonical variant column ``v`` (see
    :func:`canonical_variant`; machines not in ``machines`` assume 4 configurations)
    and converts the numeric columns to numbers.

    Args:
        path: Path of ``summary.csv`` written by ``sim/run_experiments.py``.
        machines: Machine descriptions keyed by name, as returned by :func:`load_machines`.

    Returns:
        The filtered data frame with the extra ``v`` column.
    """
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"].fillna("ok") == "ok"].copy()
    if "input" in df.columns:
        df = df[df["input"].fillna("large") == "large"].copy()
    if "base_machine" not in df.columns:
        df["base_machine"] = df["machine"]
    df["base_machine"] = df["base_machine"].fillna(df["machine"])
    if "sens_param" not in df.columns:
        df["sens_param"] = ""
    df["sens_param"] = df["sens_param"].fillna("")
    n_cfg = {m: len(machines[m]["window_table"]) if m in machines else 4
             for m in df["base_machine"].unique()}
    df["v"] = [canonical_variant(v, n_cfg.get(m, 4))
               for m, v in zip(df["base_machine"], df["variant"])]
    for c in ("ipc", "ed2p", "insts", "cycles", "l2_kb", "mem_latency_cycles"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_machines(d: Path) -> dict:
    """Load every machine description ``*.json`` of a directory.

    Files that ``whdata.load_machine`` rejects (``ValueError``/``OSError``) are skipped.

    Args:
        d: Directory with the machine JSON files (``sim/machines``).

    Returns:
        Dict mapping the file stem (machine name) to the parsed machine dict.
    """
    out = {}
    for f in sorted(Path(d).glob("*.json")):
        try:
            m = load_machine(f)
            out[f.stem] = m
        except (ValueError, OSError):
            continue
    return out


def _size_kb(s) -> float:
    """Convert a size string such as ``"1MB"``, ``"256kB"`` or ``"512KiB"`` to kB.

    A bare number is taken as bytes.

    Args:
        s (object): Size value (any type; converted with ``str``).

    Returns:
        The size in kB (1 kB = 1024 B), or NaN if it cannot be parsed.
    """
    s = str(s).strip().lower().replace("ib", "b")
    for suf, mul in (("kb", 1), ("mb", 1024), ("gb", 1024 * 1024), ("b", 1 / 1024)):
        if s.endswith(suf):
            try:
                return float(s[: -len(suf)]) * mul
            except ValueError:
                return float("nan")
    try:
        return float(s) / 1024
    except ValueError:
        return float("nan")


def machine_params(m: dict) -> dict:
    """Return the parameters of a machine used on the sensitivity x-axes.

    Args:
        m: Machine dict as returned by ``whdata.load_machine``.

    Returns:
        Dict with ``l2_kb`` (L2 size in kB, NaN if absent), ``mem_latency``
            (``memory.latency_cycles``, NaN if absent) and ``rob_max`` (largest ROB
            size in the window table).
    """
    cache = m.get("cache", {})
    l2 = cache.get("l2", {}).get("size", cache.get("l2_size", "nan"))
    mem = m.get("memory", {})
    lat = mem.get("latency_cycles", float("nan"))
    return {"l2_kb": _size_kb(l2), "mem_latency": float(lat),
            "rob_max": max(c["rob"] for c in m["window_table"])}


def read_regions_json(path: Path) -> dict[int, dict]:
    """Read the compiler's per-region predictions from a ``<kernel>.regions.json``.

    Accepts the ``{"regions": {"<id>": {...}}}`` object, a bare ``{"<id>": {...}}``
    mapping or a list of region objects carrying an ``id`` (or ``region``) field.
    The predicted window is the first non-null of ``nest_w_star``, ``w_star``,
    ``wstar`` and ``W*``; the predicted configuration is ``nest_config`` if
    non-null, else ``config``. Regions with a non-integer id or without a
    predicted window are skipped.

    Args:
        path: Path of the ``.regions.json`` file.

    Returns:
        Dict mapping region id to ``{"w_pred", "config_pred", "function", "line",
            "conservative"}``.
    """
    d = json.loads(Path(path).read_text())
    regs = d.get("regions", d) if isinstance(d, dict) else d
    out = {}
    if isinstance(regs, dict):
        items = regs.items()
    else:
        items = ((r.get("id", r.get("region")), r) for r in regs)
    for k, r in items:
        try:
            rid = int(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(r, dict):
            continue
        w = next((r[k] for k in ("nest_w_star", "w_star", "wstar", "W*") if r.get(k) is not None), None)
        if w is None:
            continue
        cfg = r.get("nest_config") if r.get("nest_config") is not None else r.get("config")
        out[rid] = {"w_pred": float(w), "config_pred": cfg,
                    "function": r.get("function", ""), "line": r.get("line"),
                    "conservative": bool(r.get("conservative", False))}
    return out


def find_regions_json(dirs: list[Path], kernel: str, machine: str) -> Path | None:
    """Locate the ``<kernel>.regions.json`` of a kernel for a given machine.

    ``<dir>/<machine>/<kernel>.regions.json`` is searched first in every directory;
    a file without the machine level is accepted only if its ``"target"`` is this
    machine (or absent/empty, or the file cannot be parsed).

    Args:
        dirs: Directories to search, in order.
        kernel: Kernel name.
        machine: Machine name.

    Returns:
        Path of the first matching file, or ``None``.
    """
    for d in dirs:
        p = Path(d) / machine / f"{kernel}.regions.json"
        if p.is_file():
            return p
    for d in dirs:
        p = Path(d) / f"{kernel}.regions.json"
        if p.is_file():
            try:
                tgt = json.loads(p.read_text()).get("target")
            except (ValueError, AttributeError, OSError):
                tgt = None
            if tgt in (None, "", machine):
                return p
    return None


def wstar_table(oracle_root: Path, compiler_dirs: list[Path], machines: dict,
                input_size: str = "large", metric: str = "ipc") -> pd.DataFrame:
    """Join oracle-best window configurations with the compiler's predicted W*.

    For every machine, every oracle file ``<oracle_root>/<machine>/<input_size>/<kernel>.json``
    (files with more than one ``.`` in the name are skipped) is paired with the
    kernel's ``.regions.json`` (see :func:`find_regions_json`). For a metric other
    than ``ipc``, ``<kernel>.<metric>.json`` is used when it exists. Only regions
    present in both with a valid configuration index are kept.

    Args:
        oracle_root: Root of the oracle results (``results/oracle``).
        compiler_dirs: Directories searched for ``<kernel>.regions.json``.
        machines: Machine descriptions keyed by name.
        input_size: Input size subdirectory of the oracle results.
        metric: Oracle metric (``"ipc"`` or ``"ed2p"``).

    Returns:
        Data frame with one row per region and columns ``machine``, ``kernel``,
            ``region``, ``oracle_config``, ``oracle_rob``, ``w_pred``, ``config_pred``
            and ``conservative``.
    """
    rows = []
    for mname, m in machines.items():
        tab = m["window_table"]
        base = oracle_root / mname / input_size
        if not base.is_dir():
            continue
        for oj in sorted(base.glob("*.json")):
            if oj.name.count(".") > 1:  # .ipc.json/.ed2p.json/.summary.json
                continue
            kernel = oj.stem
            rj = find_regions_json(compiler_dirs, kernel, mname)
            if rj is None:
                continue
            best = load_oracle(oj, metric)
            if metric != "ipc" and (base / f"{kernel}.{metric}.json").exists():
                best = load_oracle(base / f"{kernel}.{metric}.json")
            pred = read_regions_json(rj)
            for rid, cfg in best.items():
                if rid in pred and 0 <= cfg < len(tab):
                    rows.append({"machine": mname, "kernel": kernel, "region": rid,
                                 "oracle_config": cfg, "oracle_rob": tab[cfg]["rob"],
                                 "w_pred": pred[rid]["w_pred"],
                                 "config_pred": pred[rid]["config_pred"],
                                 "conservative": pred[rid].get("conservative", False)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save(fig, out: Path, name: str, fmt: str) -> Path:
    """Lay out, save and close a figure.

    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure.
        out: Output directory (created if missing).
        name: File name without extension.
        fmt: File format/extension (``pdf``, ``png`` or ``svg``).

    Returns:
        Path of the written file.
    """
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.{fmt}"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[FIG] {p}")
    return p


def fig_wstar(wt: pd.DataFrame, out: Path, fmt: str, metrics: dict) -> None:
    """Plot predicted W* against the oracle-best ROB size (``wstar_vs_oracle``).

    Records in ``metrics["wstar"]`` the overall and per-machine Spearman rho, the
    number of regions, the configuration agreement (NaN if any prediction lacks a
    configuration) and, when conservative regions exist, their count and the rho
    over the non-conservative regions only. Skips the figure if ``wt`` is empty.

    Args:
        wt: Table returned by :func:`wstar_table`.
        out: Output directory.
        fmt: Figure file format.
        metrics: Metrics dict updated in place.
    """
    if wt.empty:
        print("[SKIP] wstar_vs_oracle: no (oracle, regions.json) pairs")
        return
    rho = spearman(wt["w_pred"], wt["oracle_rob"])
    agree = float((wt["config_pred"] == wt["oracle_config"]).mean()) \
        if wt["config_pred"].notna().all() else float("nan")
    per_m = {m: spearman(g["w_pred"], g["oracle_rob"]) for m, g in wt.groupby("machine")}
    metrics["wstar"] = {"spearman_rho": rho, "n_regions": int(len(wt)),
                        "config_agreement": agree, "rho_per_machine": per_m}
    if "conservative" in wt.columns and wt["conservative"].any():
        ex = wt[~wt["conservative"].astype(bool)]
        # regions whose trip count/stride the model had to assume (docs/reference/stats-schema.md)
        metrics["wstar"].update(n_conservative=int(wt["conservative"].astype(bool).sum()),
                                spearman_rho_non_conservative=spearman(ex["w_pred"], ex["oracle_rob"])
                                if len(ex) > 2 else float("nan"))
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    rng = np.random.default_rng(0)
    for (mname, g), mk in zip(wt.groupby("machine"), "osD^v<>"):
        jit = rng.uniform(-6, 6, len(g))
        ax.scatter(g["oracle_rob"] + jit, g["w_pred"], s=18, alpha=0.7, marker=mk, label=mname)
    lim = max(wt["oracle_rob"].max(), wt["w_pred"].max()) * 1.08
    ax.plot([0, lim], [0, lim], color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("oracle-best window (ROB entries)")
    ax.set_ylabel("predicted W* (entries)")
    ax.set_title(f"Spearman $\\rho$ = {rho:.2f}  (n = {len(wt)})", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    _save(fig, out, "wstar_vs_oracle", fmt)


def _norm_table(df: pd.DataFrame, col: str, machine: str) -> pd.DataFrame:
    """Return per-kernel values of ``col`` on ``machine`` normalised to B0-large.

    Args:
        df: Summary as returned by :func:`load_summary`.
        col: Metric column to normalise.
        machine: Machine whose rows are used.

    Returns:
        Kernel x variant table (variants in :data:`BASELINES` order) divided by the
            ``static_max`` column, with an extra ``geomean`` row; an empty frame when
            there is no ``static_max`` run.
    """
    d = df[(df["machine"] == machine) & df[col].notna()]
    piv = d.pivot_table(index="kernel", columns="v", values=col, aggfunc="mean")
    if "static_max" not in piv.columns:
        return pd.DataFrame()
    norm = piv.div(piv["static_max"], axis=0)
    order = [v for v, _ in BASELINES if v in norm.columns]  # winhint_nop excluded
    norm = norm[order]
    gm = norm.apply(geomean, axis=0)
    norm.loc["geomean"] = gm
    return norm


def fig_bars(df: pd.DataFrame, out: Path, fmt: str, col: str, ylabel: str, name: str,
             machine: str, metrics: dict) -> None:
    """Plot a per-kernel bar chart of a metric normalised to B0-large.

    Values are plotted as raw ratios to B0-large (not inverted); the direction in
    which the metric is better belongs in ``ylabel``. Stores the geomean per variant in ``metrics[name]``. Skips the figure when
    the column or the B0-large reference is missing.

    Args:
        df: Summary as returned by :func:`load_summary`.
        out: Output directory.
        fmt: Figure file format.
        col: Metric column (``ipc`` or ``ed2p``).
        ylabel: Y-axis label.
        name: Figure and metrics key.
        machine: Machine to plot.
        metrics: Metrics dict updated in place.
    """
    if col not in df.columns:
        print(f"[SKIP] {name}: no column {col}")
        return
    norm = _norm_table(df, col, machine)
    if norm.empty:
        print(f"[SKIP] {name}: no B0-large reference on {machine}")
        return
    metrics[name] = {"machine": machine,
                     "geomean": {LABEL.get(k, k): v for k, v in norm.loc["geomean"].items()}}
    kernels = list(norm.index)
    nv = len(norm.columns)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(kernels) * max(nv, 4) / 4), 3.4))
    width = 0.8 / nv
    x = np.arange(len(kernels))
    cmap = plt.get_cmap("tab20")
    for j, v in enumerate(norm.columns):
        ax.bar(x + (j - nv / 2 + 0.5) * width, norm[v].to_numpy(), width,
               label=LABEL.get(v, v), color=cmap(j % 20))
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_infer", "") for k in kernels], rotation=40, ha="right",
                       fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{machine}", fontsize=9)
    ax.legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    _save(fig, out, name, fmt)


def _find_col(df: pd.DataFrame, words: tuple[str, ...]) -> str | None:
    """Return the first ``win.*`` column whose lower-cased name contains one of ``words``.

    Words are tried in order, so earlier words take precedence.

    Args:
        df: Summary data frame.
        words: Substrings to look for.

    Returns:
        The column name, or ``None``.
    """
    for w in words:
        for c in df.columns:
            if c.startswith("win.") and w in c.lower():
                return c
    return None


def fig_switch_frequency(df: pd.DataFrame, out: Path, fmt: str, metrics: dict) -> None:
    """Plot window switches per million instructions per non-static variant.

    The switch counter is found with :func:`_find_col` (``switch``, ``resize``,
    ``reconfig``). Stores the counter name and per-variant means in
    ``metrics["switch_frequency"]``.

    Args:
        df: Summary as returned by :func:`load_summary`.
        out: Output directory.
        fmt: Figure file format.
        metrics: Metrics dict updated in place.
    """
    col = _find_col(df, ("switch", "resize", "reconfig"))
    if col is None or "insts" not in df.columns:
        print("[SKIP] switch_frequency: no window switch counter in summary")
        return
    d = df.copy()
    d["spmi"] = pd.to_numeric(d[col], errors="coerce") / d["insts"] * 1e6
    g = d.groupby("v")["spmi"].mean()
    order = [v for v, _ in BASELINES if v in g.index and not v.startswith("static")]
    if not order:
        print("[SKIP] switch_frequency: only static runs")
        return
    metrics["switch_frequency"] = {"counter": col,
                                   "per_minst": {LABEL.get(v, v): float(g[v]) for v in order}}
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(order)), [g[v] for v in order], color="tab:blue")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL.get(v, v) for v in order], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("window switches / M instructions")
    ax.set_yscale("symlog", linthresh=1)
    _save(fig, out, "switch_frequency", fmt)


def compiler_stats(compiler_dirs: list[Path]) -> dict[str, dict]:
    """Read static hint counts and compile times from ``<kernel>.winhint.json`` files.

    Files of schema ``winhint-stats/1`` are searched directly in each directory
    and one level below; the first file found for a kernel wins and unparsable
    files are skipped.

    Args:
        compiler_dirs: Directories to search.

    Returns:
        Dict mapping kernel name to ``{"hints_setwin", "hints_region",
            "compile_time_ms", "analysis_time_ms"}``.
    """
    out = {}
    for d in compiler_dirs:
        for p in sorted(Path(d).glob("*.winhint.json")) + sorted(Path(d).glob("*/*.winhint.json")):
            try:
                j = json.loads(p.read_text())
            except ValueError:
                continue
            out.setdefault(p.name[: -len(".winhint.json")], {
                "hints_setwin": int(j.get("hints_setwin", 0)),
                "hints_region": int(j.get("hints_region", 0)),
                "compile_time_ms": j.get("compile_time_ms"),
                "analysis_time_ms": j.get("analysis_time_ms")})
    return out


def static_hint_counts(compiler_dirs: list[Path]) -> dict[str, int]:
    """Return the static ``setwin`` hint count per kernel from ``<kernel>.winhint.json``.

    Thin view of :func:`compiler_stats` (same file search).

    Args:
        compiler_dirs: Directories to search.

    Returns:
        Dict mapping kernel name to ``hints_setwin`` (0 if absent).
    """
    return {k: v["hints_setwin"] for k, v in compiler_stats(compiler_dirs).items()}


def elf_exec_bytes(path: Path) -> int | None:
    """Return the total size of the executable sections of an ELF file.

    Sums ``sh_size`` of all sections with ``SHF_EXECINSTR`` that are not
    ``SHT_NOBITS``. Pure ``struct`` parsing (ELF32/64, little/big endian).

    Args:
        path: File to inspect.

    Returns:
        Number of bytes, or ``None`` if the file cannot be read, is not an ELF
            file or has a truncated section header table.
    """
    import struct
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if len(data) < 64 or data[:4] != b"\x7fELF":
        return None
    is64, end = data[4] == 2, "<" if data[5] == 1 else ">"
    if is64:
        shoff, = struct.unpack_from(end + "Q", data, 0x28)
        shentsize, shnum = struct.unpack_from(end + "HH", data, 0x3A)
    else:
        shoff, = struct.unpack_from(end + "I", data, 0x20)
        shentsize, shnum = struct.unpack_from(end + "HH", data, 0x2E)
    total = 0
    for i in range(shnum):
        off = shoff + i * shentsize
        if off + shentsize > len(data):
            return None
        if is64:
            sh_type, sh_flags = struct.unpack_from(end + "IQ", data, off + 4)
            sh_size, = struct.unpack_from(end + "Q", data, off + 0x20)
        else:
            sh_type, sh_flags = struct.unpack_from(end + "II", data, off + 4)
            sh_size, = struct.unpack_from(end + "I", data, off + 0x14)
        if sh_type != 8 and sh_flags & 0x4:      # not NOBITS, SHF_EXECINSTR
            total += sh_size
    return total


#: hinted binary variants whose code size is compared with ``plain``
HINTED_BINARIES = ("winhint", "jones", "jones_full", "pgo", "oracle_hinted")


def code_size_table(bin_root: Path | None) -> pd.DataFrame:
    """Compare executable bytes of each hinted variant with the plain build, per kernel.

    Binaries are ``<bin_root>/<variant>/<kernel>`` (default-machine builds); files
    with an extension are ignored and only variants in :data:`HINTED_BINARIES` are
    compared.

    Args:
        bin_root: Root of the benchmark binaries, or ``None``.

    Returns:
        Data frame with columns ``kernel``, ``v``, ``plain_bytes``, ``bytes``,
            ``delta_bytes`` and ``delta_pct``; empty when ``bin_root`` is ``None`` or
            has no ``plain`` directory.
    """
    rows = []
    if bin_root is None or not (bin_root / "plain").is_dir():
        return pd.DataFrame(rows)
    for pb in sorted((bin_root / "plain").iterdir()):
        if not pb.is_file() or pb.suffix:
            continue
        base = elf_exec_bytes(pb)
        if not base:
            continue
        for v in HINTED_BINARIES:
            hb = bin_root / v / pb.name
            n = elf_exec_bytes(hb) if hb.is_file() else None
            if n:
                rows.append({"kernel": pb.name, "v": v, "plain_bytes": base, "bytes": n,
                             "delta_bytes": n - base, "delta_pct": (n / base - 1.0) * 100.0})
    return pd.DataFrame(rows)


def fig_hint_overhead(df: pd.DataFrame, compiler_dirs: list[Path], out: Path, fmt: str,
                      metrics: dict, bin_root: Path | None = None) -> None:
    """Plot the three-panel hint overhead figure (``hint_overhead``).

    Panels: dynamic hints per kilo-instruction of the hinted variants (titled
    with the geomean cycle overhead of ``winhint_nop`` vs ``static_max`` when both
    exist), static ``setwin`` hints per kernel, and mean code-size growth vs plain
    per variant. Results go to ``metrics["hint_overhead"]``.

    Args:
        df: Summary as returned by :func:`load_summary`.
        compiler_dirs: Directories with ``<kernel>.winhint.json``.
        out: Output directory.
        fmt: Figure file format.
        metrics: Metrics dict updated in place.
        bin_root: Root of the benchmark binaries for the code-size panel, or ``None``.
    """
    col = _find_col(df, ("setwinhint", "hint"))
    static = static_hint_counts(compiler_dirs)
    cs_tab = code_size_table(bin_root)
    if col is None and not static and cs_tab.empty:
        print("[SKIP] hint_overhead: no hint counters, <kernel>.winhint.json or binaries")
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    m = {}
    if col is not None and "insts" in df.columns:
        d = df[df["v"].isin(["winhint", "winhint_hw", "jones", "jones_full", "pgo",
                             "oracle_hinted", "winhint_clairvoyance"])].copy()
        d = d[d["insts"] > 0]
        d["hpki"] = pd.to_numeric(d[col], errors="coerce") / d["insts"] * 1e3
        g = d.groupby("v")["hpki"].mean()
        order = [v for v, _ in BASELINES if v in g.index]
        axes[0].bar(range(len(order)), [g[v] for v in order], color="tab:orange")
        axes[0].set_xticks(range(len(order)))
        axes[0].set_xticklabels([LABEL.get(v, v) for v in order], rotation=40, ha="right",
                                fontsize=7)
        axes[0].set_ylabel("dynamic hints / kilo-instr.")
        m["dynamic_per_kinst"] = {LABEL.get(v, v): float(g[v]) for v in order}
    else:
        axes[0].set_axis_off()
    if "cycles" in df.columns:
        piv = df.pivot_table(index=["machine", "kernel"], columns="v", values="cycles",
                             aggfunc="mean")
        if {"winhint_nop", "static_max"} <= set(piv.columns):
            ov = (piv["winhint_nop"] / piv["static_max"] - 1.0).dropna() * 100.0
            if len(ov):
                m["dynamic_overhead_pct"] = {f"{mm}/{k}": float(v) for (mm, k), v in ov.items()}
                m["dynamic_overhead_pct_geomean"] = float(
                    (geomean((1 + ov / 100.0).to_numpy()) - 1.0) * 100.0)
                axes[0].set_title("cycle overhead of the hints (ignored): "
                                  f"{m['dynamic_overhead_pct_geomean']:+.2f} % geomean",
                                  fontsize=7)
    if static:
        ks = sorted(static)
        axes[1].bar(range(len(ks)), [static[k] for k in ks], color="tab:green")
        axes[1].set_xticks(range(len(ks)))
        axes[1].set_xticklabels([k.replace("_infer", "") for k in ks], rotation=40, ha="right",
                                fontsize=7)
        axes[1].set_ylabel("static setwin hints (WinHint)")
        m["static_setwin"] = static
        cs = compiler_stats(compiler_dirs)
        m["compile_time_ms"] = {k: v["compile_time_ms"] for k, v in cs.items()
                                if v["compile_time_ms"] is not None}
    else:
        axes[1].set_axis_off()
    if not cs_tab.empty:
        g = cs_tab.groupby("v")["delta_pct"].mean()
        order = [v for v, _ in BASELINES if v in g.index]
        axes[2].bar(range(len(order)), [g[v] for v in order], color="tab:purple")
        axes[2].set_xticks(range(len(order)))
        axes[2].set_xticklabels([LABEL.get(v, v) for v in order], rotation=40, ha="right",
                                fontsize=7)
        axes[2].set_ylabel("code size vs plain (%)")
        m["code_size"] = {
            "mean_delta_pct": {LABEL.get(v, v): float(g[v]) for v in order},
            "per_kernel": {f"{r.v}/{r.kernel}": {"bytes": int(r.bytes),
                                                 "plain_bytes": int(r.plain_bytes),
                                                 "delta_bytes": int(r.delta_bytes)}
                           for r in cs_tab.itertuples()}}
    else:
        axes[2].set_axis_off()
    metrics["hint_overhead"] = m
    _save(fig, out, "hint_overhead", fmt)


def _speedup_by_machine(df: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    """Return the geomean IPC speedup over B0-large per machine and variant.

    Args:
        df: Summary with columns ``machine``, ``kernel``, ``v`` and ``ipc``.
        variants: Variants to include.

    Returns:
        Data frame with columns ``machine``, ``v`` and ``speedup``; machines
            without a ``static_max`` run are skipped.
    """
    rows = []
    for mname, g in df.groupby("machine"):
        piv = g.pivot_table(index="kernel", columns="v", values="ipc", aggfunc="mean")
        if "static_max" not in piv.columns:
            continue
        for v in variants:
            if v in piv.columns:
                rows.append({"machine": mname, "v": v,
                             "speedup": geomean(piv[v] / piv["static_max"])})
    return pd.DataFrame(rows)


#: Sweep parameter -> (summary column, axis label) for the sensitivity figure.
SENS_X = {"l2_size": ("l2_kb", "L2 size (kB)"), "l1d_size": ("l1d_kb", "L1D size (kB)"),
          "mem_extra_latency_ns": ("mem_latency_cycles", "memory latency (cycles)")}


def fig_sensitivity_sweep(df: pd.DataFrame, out: Path, fmt: str, metrics: dict) -> bool:
    """Plot sensitivity from the ``run_experiments.py --sweep`` points.

    Each sweep point is one base machine with one parameter changed. One panel per
    swept parameter in :data:`SENS_X`: speedup over B0-large of the same point vs
    the swept parameter; the base machine is the unmodified point. Stores the
    points in ``metrics["sensitivity"]`` with ``source == "sweep"``.

    Args:
        df: Full summary including sensitivity rows (``sens_param`` non-empty).
        out: Output directory.
        fmt: Figure file format.
        metrics: Metrics dict updated in place.

    Returns:
        ``True`` if the figure was written, ``False`` if there are no usable sweep points.
    """
    vs = ["oracle_hinted", "mlp", "lut", "winhint", "winhint_hw"]
    pts = df[df["sens_param"].astype(str) != ""]
    params = [p for p in SENS_X if p in set(pts["sens_param"]) and SENS_X[p][0] in df.columns]
    if not params:
        return False
    sp = _speedup_by_machine(df, vs)
    if sp.empty:
        return False
    info = df.groupby("machine").agg(base=("base_machine", "first"), param=("sens_param", "first"),
                                     **{c: (c, "first") for c in ("l2_kb", "mem_latency_cycles")
                                        if c in df.columns})
    sp = sp.join(info, on="machine")
    rec, fig, axes = [], *plt.subplots(1, len(params), figsize=(4 * len(params), 3),
                                       sharey=True, squeeze=False)
    for ax, par in zip(axes[0], params):
        xcol, xl = SENS_X[par]
        for base in sorted(pts[pts["sens_param"] == par]["base_machine"].unique()):
            g = sp[(sp["base"] == base) & sp["param"].isin([par, ""])]
            for v, gv in g.groupby("v"):
                gv = gv.sort_values(xcol)
                ax.plot(gv[xcol], gv["speedup"], marker="o",
                        label=f"{LABEL.get(v, v)}" + (f" ({base})" if pts["base_machine"].nunique() > 1 else ""))
                rec += [{"param": par, "base": base, "v": v, "x": float(r[xcol]),
                         "speedup": float(r["speedup"])} for _, r in gv.iterrows()]
        ax.axhline(1.0, color="k", lw=0.6)
        ax.set_xlabel(xl)
        if par != "mem_extra_latency_ns":
            ax.set_xscale("log", base=2)
    axes[0][0].set_ylabel("geomean IPC / B0-large")
    axes[0][-1].legend(fontsize=6)
    metrics["sensitivity"] = {"source": "sweep", "points": rec}
    _save(fig, out, "sensitivity", fmt)
    return True


def fig_sensitivity(df: pd.DataFrame, machines: dict, out: Path, fmt: str, metrics: dict) -> None:
    """Plot speedup over B0-large vs L2 size and memory latency (``sensitivity``).

    Uses :func:`fig_sensitivity_sweep` when sweep points exist; otherwise compares
    machines, taking their parameters from the machine descriptions (needs at
    least two machines with a B0-large run).

    Args:
        df: Full summary including sensitivity rows.
        machines: Machine descriptions keyed by name.
        out: Output directory.
        fmt: Figure file format.
        metrics: Metrics dict updated in place.
    """
    if fig_sensitivity_sweep(df, out, fmt, metrics):
        return
    vs = ["oracle_hinted", "mlp", "lut", "winhint", "winhint_hw"]
    sp = _speedup_by_machine(df, vs)
    if sp.empty or sp["machine"].nunique() < 2:
        print("[SKIP] sensitivity: needs >= 2 machines with B0-large")
        return
    params = {m: machine_params(machines[m]) for m in sp["machine"].unique() if m in machines}
    sp = sp[sp["machine"].isin(params)]
    sp["l2_kb"] = sp["machine"].map(lambda m: params[m]["l2_kb"])
    sp["mem_latency"] = sp["machine"].map(lambda m: params[m]["mem_latency"])
    metrics["sensitivity"] = {"source": "machines", "points": sp.to_dict(orient="records")}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, xcol, xl in ((axes[0], "l2_kb", "L2 size (kB)"),
                         (axes[1], "mem_latency", "memory latency (cycles)")):
        for v, g in sp.groupby("v"):
            g = g.sort_values(xcol)
            ax.plot(g[xcol], g["speedup"], marker="o", label=LABEL.get(v, v))
            for _, r in g.iterrows():
                ax.annotate(r["machine"], (r[xcol], r["speedup"]), fontsize=5, alpha=0.6)
        ax.axhline(1.0, color="k", lw=0.6)
        ax.set_xlabel(xl)
    axes[0].set_ylabel("geomean IPC / B0-large")
    axes[1].legend(fontsize=7)
    _save(fig, out, "sensitivity", fmt)


def fig_portability(df: pd.DataFrame, out: Path, fmt: str, metrics: dict,
                    ref: str = "oracle_hinted") -> None:
    """Plot B5 retrained (``lut``) vs not retrained (``lut_xfer``) vs WinHint per machine.

    Each bar is the geomean IPC relative to ``ref`` (``static_max`` if ``ref`` was
    not run on that machine). Stores the records in ``metrics["portability"]``.

    Args:
        df: Summary as returned by :func:`load_summary`.
        out: Output directory.
        fmt: Figure file format.
        metrics: Metrics dict updated in place.
        ref: Reference variant.
    """
    rows = []
    for mname, g in df.groupby("machine"):
        piv = g.pivot_table(index="kernel", columns="v", values="ipc", aggfunc="mean")
        base = ref if ref in piv.columns else ("static_max" if "static_max" in piv.columns else None)
        if base is None:
            continue
        for v in ("lut", "lut_xfer", "winhint"):
            if v in piv.columns and v != base:
                rows.append({"machine": mname, "v": v, "ref": base,
                             "rel": geomean(piv[v] / piv[base])})
    p = pd.DataFrame(rows)
    if p.empty:
        print("[SKIP] portability: no lut/lut_xfer/winhint runs")
        return
    metrics["portability"] = p.to_dict(orient="records")
    ms = sorted(p["machine"].unique())
    vs = [v for v in ("lut", "lut_xfer", "winhint") if v in set(p["v"])]
    fig, ax = plt.subplots(figsize=(5, 3))
    w = 0.8 / len(vs)
    for j, v in enumerate(vs):
        vals = [p[(p.machine == m) & (p.v == v)]["rel"].mean() for m in ms]
        ax.bar(np.arange(len(ms)) + (j - len(vs) / 2 + 0.5) * w, vals, w, label=LABEL.get(v, v))
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(ms, fontsize=8)
    ax.set_ylabel(f"geomean IPC / {LABEL.get(p['ref'].iloc[0], p['ref'].iloc[0])}")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.legend(fontsize=7)
    _save(fig, out, "portability", fmt)


# ---------------------------------------------------------------------------
# Real-silicon figures (results/hw/summary.csv from hw/run_hw_experiments.py)
# ---------------------------------------------------------------------------

#: Display order of the real-silicon configurations (``R2-*`` = all R2 variants).
HW_ORDER = ["R0-P", "R0-E", "R1", "R2-*", "R3-lpmd", "R4-PIE", "R5-Sondag", "WH", "WH-off"]
#: Real-silicon configuration -> display label.
HW_LABEL = {"R0-P": "R0 P-only", "R0-E": "R0 E-only", "R1": "R1 stock", "R3-lpmd": "R3 lpmd",
            "R4-PIE": "R4 PIE", "R5-Sondag": "R5 Sondag", "WH": "WinHint", "WH-off": "WinHint (off)"}


def _hw_order(configs) -> list[str]:
    """Return the configurations of ``configs`` in :data:`HW_ORDER` display order.

    ``R2-*`` expands to all present ``R2-`` configurations, sorted.

    Args:
        configs (iterable of str): Configuration names present in the data.

    Returns:
        Ordered list of configuration names; names not in :data:`HW_ORDER` are dropped.
    """
    out = []
    for c in HW_ORDER:
        if c == "R2-*":
            out += sorted(x for x in configs if str(x).startswith("R2-"))
        elif c in configs:
            out.append(c)
    return out


def load_hw_summary(path: Path) -> pd.DataFrame:
    """Load ``results/hw/summary.csv`` and convert the statistic columns to numbers.

    Columns ending in ``_mean``, ``_ci_lo``, ``_ci_hi``, ``_vs_R1`` or ``_pct`` are
    converted with ``errors="coerce"``.

    Args:
        path: Path of the summary written by ``hw/run_hw_experiments.py``.

    Returns:
        The summary data frame.
    """
    df = pd.read_csv(path)
    for c in df.columns:
        if c.endswith(("_mean", "_ci_lo", "_ci_hi", "_vs_R1", "_pct")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fig_hw_edp(hw: pd.DataFrame, out: Path, fmt: str, metrics: dict) -> None:
    """Plot package EDP of every R-baseline and WinHint normalised to R1 (``hw_edp``).

    One panel per SMT state, per kernel plus geomean, with 95 % CI error bars.
    Falls back to wall time when no energy was measured. ``NOP-*`` configurations
    are excluded. Stores the metric name and per-SMT geomeans in ``metrics["hw_edp"]``.

    Args:
        hw: Summary as returned by :func:`load_hw_summary`.
        out: Output directory.
        fmt: Figure file format.
        metrics: Metrics dict updated in place.
    """
    metric, ylabel = "edp_pkg_js", "package EDP / R1 (lower is better)"
    if f"{metric}_mean" not in hw.columns or hw[f"{metric}_mean"].notna().sum() == 0:
        metric, ylabel = "wall_s", "time / R1 (no RAPL energy in these runs)"
    if f"{metric}_mean" not in hw.columns:
        print("[SKIP] hw_edp: no usable columns in hw summary")
        return
    d = hw[~hw["config"].astype(str).str.startswith("NOP-")]
    smts = sorted(d["smt"].astype(str).unique(), key=lambda v: ({"on": 0, "off": 1}.get(v, 2), v))
    panels, m = [], {"metric": metric}
    for smt in smts:
        g = d[d["smt"].astype(str) == smt]
        piv = g.pivot_table(index="kernel", columns="config", values=f"{metric}_mean", aggfunc="mean")
        if "R1" not in piv.columns:
            continue
        lo = g.pivot_table(index="kernel", columns="config", values=f"{metric}_ci_lo", aggfunc="mean")
        hi = g.pivot_table(index="kernel", columns="config", values=f"{metric}_ci_hi", aggfunc="mean")
        ref = piv["R1"]
        cfgs = _hw_order(list(piv.columns))
        norm = piv[cfgs].div(ref, axis=0)
        norm.loc["geomean"] = norm.apply(geomean, axis=0)
        lo_n = lo.reindex(columns=cfgs).div(ref, axis=0)
        hi_n = hi.reindex(columns=cfgs).div(ref, axis=0)
        panels.append((smt, norm, lo_n, hi_n))
        m[f"smt-{smt}"] = {HW_LABEL.get(c, c): float(norm.loc["geomean", c]) for c in cfgs}
    if not panels:
        print("[SKIP] hw_edp: no R1 reference")
        return
    metrics["hw_edp"] = m
    fig, axes = plt.subplots(1, len(panels), figsize=(max(5, 1.2 * len(panels[0][1]) + 3) * len(panels) / 1.4,
                                                       3.4), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, (smt, norm, lo_n, hi_n) in zip(axes[0], panels):
        kernels, cfgs = list(norm.index), list(norm.columns)
        x = np.arange(len(kernels))
        w = 0.8 / len(cfgs)
        for j, c in enumerate(cfgs):
            y = norm[c].to_numpy(float)
            err = None
            if c in lo_n.columns:
                l = lo_n[c].reindex(kernels).to_numpy(float)
                h = hi_n[c].reindex(kernels).to_numpy(float)
                if np.isfinite(l).any():
                    err = np.vstack([np.where(np.isfinite(l), y - l, 0), np.where(np.isfinite(h), h - y, 0)])
                    err = np.clip(err, 0, None)
            ax.bar(x + (j - len(cfgs) / 2 + 0.5) * w, y, w, yerr=err, capsize=1.5,
                   label=HW_LABEL.get(c, c), color=cmap(j % 10),
                   error_kw={"elinewidth": 0.6})
        ax.axhline(1.0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("_infer", "") for k in kernels], rotation=40, ha="right", fontsize=7)
        ax.set_title(f"SMT {smt}", fontsize=9)
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(fontsize=6, ncol=5, loc="upper right", bbox_to_anchor=(1.0, -0.38))
    _save(fig, out, "hw_edp", fmt)


def fig_hw_nop_overhead(hw: pd.DataFrame, out: Path, fmt: str, metrics: dict) -> None:
    """Plot the x86 NOP-hint overhead per kernel on the P and E core (``hw_nop_overhead``).

    Uses the ``nop_overhead_pct`` of the ``NOP-asm-<P|E>`` rows; the core side is
    the last character of the configuration name. Stores the values in
    ``metrics["hw_nop_overhead"]``.

    Args:
        hw: Summary as returned by :func:`load_hw_summary`.
        out: Output directory.
        fmt: Figure file format.
        metrics: Metrics dict updated in place.
    """
    if "nop_overhead_pct" not in hw.columns:
        print("[SKIP] hw_nop_overhead: no NOP-asm runs")
        return
    d = hw[hw["config"].astype(str).str.startswith("NOP-asm-") & hw["nop_overhead_pct"].notna()]
    if d.empty:
        print("[SKIP] hw_nop_overhead: no NOP-asm runs")
        return
    d = d.assign(side=d["config"].str[-1])
    piv = d.pivot_table(index="kernel", columns="side", values="nop_overhead_pct", aggfunc="mean")
    metrics["hw_nop_overhead"] = {s: {k: float(v) for k, v in piv[s].dropna().items()} for s in piv.columns}
    fig, ax = plt.subplots(figsize=(max(4, 0.5 * len(piv) + 2), 3))
    x = np.arange(len(piv))
    for j, side in enumerate(piv.columns):
        ax.bar(x + (j - 0.5) * 0.4, piv[side], 0.4, label=f"{side}-core")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_infer", "") for k in piv.index], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("x86 NOP-hint overhead (% wall time)")
    ax.legend(fontsize=7)
    _save(fig, out, "hw_nop_overhead", fmt)


# ---------------------------------------------------------------------------
# Synthetic data (self-test)
# ---------------------------------------------------------------------------

def make_synthetic(root: Path, seed: int = 0) -> dict:
    """Write a small but complete fake result tree under ``root``.

    Creates three machine descriptions, a gem5 ``summary.csv`` (all variants plus
    sensitivity sweep points of ``riscv_ooo``), oracle and compiler outputs
    (``.regions.json``, ``.winhint.json``) and a real-silicon ``hw/summary.csv``.

    Args:
        root: Output directory.
        seed: Seed of the random generator.

    Returns:
        Dict with the paths ``summary``, ``oracle_root``, ``compiler_dirs`` (list),
            ``machines_dir`` and ``hw_summary``.
    """
    rng = np.random.default_rng(seed)
    mdir = root / "machines"
    mdir.mkdir(parents=True, exist_ok=True)
    specs = {"riscv_ooo": ("1MB", 200, [64, 128, 192, 256]),
             "riscv_ooo_small": ("256kB", 150, [32, 64, 96, 128]),
             "riscv_ooo_big": ("2MB", 300, [96, 192, 288, 384])}
    for name, (l2, lat, robs) in specs.items():
        (mdir / f"{name}.json").write_text(json.dumps({
            "name": name, "cache": {"l2": {"size": l2}}, "memory": {"latency_cycles": lat},
            "window": {"rob": robs, "iq": [r // 2 for r in robs], "lq": [r // 4 for r in robs],
                       "sq": [r // 4 for r in robs]}}))
    kernels = ["encoder_bert_tiny_infer", "decoder_gpt2_infer", "regression_svr_infer"]
    rows = []
    comp = root / "compiler"
    for name, (_, _, robs) in specs.items():
        for k in kernels:
            base_ipc = rng.uniform(0.8, 2.0)
            ipcs = {f"static_c{i}": base_ipc * (0.8 + 0.07 * i) for i in range(4)}
            ipcs.update(oracle_hinted=base_ipc * 1.10, occupancy=base_ipc * 1.0, mlp=base_ipc * 1.03,
                        bbv=base_ipc * 1.01, lut=base_ipc * 1.04, jones=base_ipc * 1.0,
                        jones_full=base_ipc * 1.02, pgo=base_ipc * 1.06, clairvoyance=base_ipc * 1.02,
                        winhint_clairvoyance=base_ipc * 1.09, winhint=base_ipc * 1.07,
                        winhint_hw=base_ipc * 1.08, ltp=base_ipc * 1.05,
                        winhint_nop=base_ipc * 1.01 / 1.002)
            if name != "riscv_ooo":
                ipcs["lut_xfer"] = base_ipc * 0.93
            for v, ipc in ipcs.items():
                insts = 1e9
                cyc = insts / ipc
                t = cyc / 2e9
                e = t * (0.85 + 0.15 * rng.uniform(0.3, 1.0))
                rows.append({"machine": name, "kernel": k, "variant": v, "status": "ok",
                             "base_machine": name, "sens_param": "", "sens_value": "",
                             "l2_kb": _size_kb(specs[name][0]), "mem_latency_cycles": specs[name][1],
                             "input": "large", "insts": insts, "cycles": cyc, "ipc": ipc,
                             "sim_seconds": t, "energy_j": e, "ed2p": e * t * t,
                             "win.switches": 0 if v.startswith("static") else rng.integers(10, 5000),
                             "win.hints": rng.integers(100, 10000)
                             if v in ("winhint", "winhint_hw", "jones", "pgo") else 0,
                             "win.setwinHints": rng.integers(50, 5000)
                             if v in ("winhint", "winhint_hw", "jones", "pgo") else 0})
            # oracle + compiler predictions
            od = root / "oracle" / name / "large"
            od.mkdir(parents=True, exist_ok=True)
            best, regs = {}, {}
            for rid in range(6):
                c = int(rng.integers(0, 4))
                best[str(rid)] = c
                regs[str(rid)] = {"w_star": int(robs[c] + rng.normal(0, 20)),
                                  "nest_w_star": float(robs[c] + rng.normal(0, 25)),
                                  "config": c, "nest_config": c, "function": "f", "line": rid}
            (od / f"{k}.json").write_text(json.dumps(best))
            cd = comp / name
            cd.mkdir(parents=True, exist_ok=True)
            (cd / f"{k}.regions.json").write_text(json.dumps({"kernel": k, "target": name,
                                                              "regions": regs}))
            (comp / f"{k}.winhint.json").write_text(json.dumps(
                {"schema": "winhint-stats/1", "kernel": k, "hints_setwin": int(rng.integers(2, 20)),
                 "hints_region": 6, "compile_time_ms": float(rng.uniform(1, 20))}))
    # sensitivity points of riscv_ooo (run_experiments.py --sweep)
    for par, vals in (("l2_size", ("256kB", "512kB", "2MB")),
                      ("mem_extra_latency_ns", ("20", "40", "80"))):
        for val in vals:
            pname = f"riscv_ooo+{par}={val}"
            l2 = _size_kb(val) if par == "l2_size" else 1024.0
            lat = 200 + 2 * float(val) if par != "l2_size" else 200
            press = (1024.0 / l2) ** 0.3 * (lat / 200.0) ** 0.5  # more memory-bound
            for k in kernels:
                b = 1.0 / press
                for v, f in (("static_c3", 1.0), ("winhint", 1.0 + 0.07 * press),
                             ("mlp", 1.0 + 0.03 * press), ("oracle_hinted", 1.0 + 0.1 * press)):
                    ipc = b * f
                    rows.append({"machine": pname, "kernel": k, "variant": v, "status": "ok",
                                 "base_machine": "riscv_ooo", "sens_param": par, "sens_value": val,
                                 "l2_kb": l2, "mem_latency_cycles": lat, "input": "large",
                                 "insts": 1e9, "cycles": 1e9 / ipc, "ipc": ipc})
    summ = root / "gem5" / "summary.csv"
    summ.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summ, index=False)
    # real-silicon summary in the format of hw/run_hw_experiments.py summarize()
    hrows = []
    rel = {"R0-P": (0.80, 1.35), "R0-E": (1.35, 0.85), "R1": (1.0, 1.0), "R2-bpfland_powersave": (1.1, 0.95),
           "R3-lpmd": (1.2, 0.9), "R4-PIE": (1.05, 0.9), "R5-Sondag": (1.02, 0.88), "WH": (0.98, 0.84),
           "WH-off": (1.0, 1.0), "NOP-plain-P": (0.8, 1.3), "NOP-asm-P": (0.802, 1.3),
           "NOP-plain-E": (1.3, 0.8), "NOP-asm-E": (1.303, 0.8)}
    for smt in ("on", "off"):
        for k in kernels:
            t0, e0 = rng.uniform(0.5, 3.0), rng.uniform(5, 30)
            for c, (ft, fe) in rel.items():
                t, e = t0 * ft, e0 * fe
                r = {"kernel": k, "config": c, "smt": smt, "n": 10, "output_ok": True,
                     "wall_s_mean": t, "wall_s_ci_lo": t * 0.98, "wall_s_ci_hi": t * 1.02,
                     "energy_pkg_j_mean": e, "edp_pkg_js_mean": e * t,
                     "edp_pkg_js_ci_lo": e * t * 0.96, "edp_pkg_js_ci_hi": e * t * 1.04}
                if c.startswith("NOP-asm-"):
                    r["nop_overhead_pct"] = 100 * (ft / rel["NOP-plain-" + c[-1]][0] - 1)
                hrows.append(r)
    hsum = root / "hw" / "summary.csv"
    hsum.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hrows).to_csv(hsum, index=False)
    return {"summary": summ, "oracle_root": root / "oracle", "compiler_dirs": [comp],
            "machines_dir": mdir, "hw_summary": hsum}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """Parse the command line.

    Args:
        argv (list[str] | None): Argument list (``None`` uses ``sys.argv``).

    Returns:
        (argparse.Namespace): The parsed ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", type=Path, default=REPO / "results" / "gem5" / "summary.csv")
    p.add_argument("--oracle-root", type=Path, default=REPO / "results" / "oracle")
    p.add_argument("--compiler-dir", type=Path, nargs="*",
                   default=[BUILD / "benchmarks" / "riscv" / "winhint"])
    p.add_argument("--hw-summary", type=Path, default=REPO / "results" / "hw" / "summary.csv")
    p.add_argument("--machines-dir", type=Path, default=REPO / "sim" / "machines")
    p.add_argument("--bin-root", type=Path, default=BUILD / "benchmarks" / "riscv",
                   help="benchmark binaries (<variant>/<kernel>) for the code-size panel")
    p.add_argument("--machine", default="riscv_ooo", help="machine of the per-kernel bar charts")
    p.add_argument("--oracle-metric", default="ipc", choices=["ipc", "ed2p"])
    p.add_argument("--out-dir", type=Path, default=REPO / "results" / "figures")
    p.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--synthetic", action="store_true",
                   help="generate fake inputs under <out-dir>/synthetic and plot them")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Generate all figures and ``metrics.json`` in ``--out-dir``.

    Figures whose inputs are missing are skipped with a ``[SKIP]`` message.

    Args:
        argv (list[str] | None): Argument list (``None`` uses ``sys.argv``).

    Returns:
        Process exit code (always 0).
    """
    a = parse_args(argv)
    if a.synthetic:
        s = make_synthetic(a.out_dir / "synthetic")
        a.summary, a.oracle_root = s["summary"], s["oracle_root"]
        a.compiler_dir, a.machines_dir = s["compiler_dirs"], s["machines_dir"]
        a.hw_summary = s["hw_summary"]
    machines = load_machines(a.machines_dir)
    metrics: dict = {}
    wt = wstar_table(a.oracle_root, a.compiler_dir, machines, metric=a.oracle_metric)
    fig_wstar(wt, a.out_dir, a.format, metrics)
    if a.summary.is_file():
        full = load_summary(a.summary, machines)
        df = full[full["sens_param"] == ""]  # sensitivity points only in fig_sensitivity
        fig_bars(df, a.out_dir, a.format, "ipc", "IPC / B0-large", "ipc_bars", a.machine,
                 metrics)
        fig_bars(df, a.out_dir, a.format, "ed2p", "ED$^2$P / B0-large (lower is better)",
                 "ed2p_bars", a.machine, metrics)
        fig_switch_frequency(df, a.out_dir, a.format, metrics)
        fig_hint_overhead(df, a.compiler_dir, a.out_dir, a.format, metrics,
                          None if a.synthetic else a.bin_root)
        fig_sensitivity(full, machines, a.out_dir, a.format, metrics)
        fig_portability(df, a.out_dir, a.format, metrics)
    else:
        print(f"[SKIP] gem5 figures: {a.summary} not found")
    if a.hw_summary.is_file():
        hw = load_hw_summary(a.hw_summary)
        fig_hw_edp(hw, a.out_dir, a.format, metrics)
        fig_hw_nop_overhead(hw, a.out_dir, a.format, metrics)
    else:
        print(f"[SKIP] hw figures: {a.hw_summary} not found")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    def clean(o):
        """Convert numpy scalars to Python types and non-finite floats to ``None`` (recursively)."""
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not math.isfinite(float(o)) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o

    (a.out_dir / "metrics.json").write_text(json.dumps(clean(metrics), indent=2) + "\n")
    print(f"[OK] metrics -> {a.out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
