"""Shared data helpers for the WinHint analysis pipeline.

Library module (no CLI); imported by the B5 LUT scripts in ``sim/baselines/lut/``.
Readers/writers for the artifacts defined in docs/interfaces.md:

    * window_trace.csv   (gem5 outdir, §4)  one row per sampling period
    * region_stats.csv   (gem5 outdir, §4)  one row per region visit
    * oracle maps        (results/oracle/..., written by sim/baselines/oracle/oracle_sweep.py)
    * machine JSONs      (sim/machines/*.json, "window" table, §3)
    * the B5 runtime LUT (text format, §6)

Only numpy and pandas are required.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Interfaces constants
# ---------------------------------------------------------------------------

#: Columns of window_trace.csv (interfaces.md §4); missing ones are added as NaN.
TRACE_COLUMNS = [
    "cycle", "insts", "ipc", "rob_occ_mean", "iq_occ_mean", "lq_occ_mean",
    "l1d_mpki", "l2_mpki", "mlp", "branch_mpki", "config", "region",
]
#: Required columns of region_stats.csv (interfaces.md §4).
REGION_COLUMNS = ["region", "config", "enter_cycle", "cycles", "insts"]

#: LUT feature names (interfaces.md §6) and the trace column each one reads.
FEATURES: tuple[str, ...] = ("ipc", "rob_occ", "l1d_mpki", "mlp")
FEATURE_SOURCE = {"ipc": "ipc", "rob_occ": "rob_occ_mean",
                  "l1d_mpki": "l1d_mpki", "mlp": "mlp"}

#: Default window table (interfaces.md §3) used when a machine has none.
DEFAULT_WINDOW = [
    {"rob": 64, "iq": 32, "lq": 16, "sq": 16},
    {"rob": 128, "iq": 64, "lq": 32, "sq": 32},
    {"rob": 192, "iq": 96, "lq": 48, "sq": 48},
    {"rob": 256, "iq": 128, "lq": 64, "sq": 64},
]

NO_REGION = -1
"""Region id used for trace rows that are outside every annotated region."""


# ---------------------------------------------------------------------------
# Machine description
# ---------------------------------------------------------------------------

def load_machine(path: str | Path) -> dict:
    """Load a machine JSON and normalise its window table.

    The "window" section may be either parallel lists
    (``{"rob": [...], "iq": [...], ...}``), a list of per-config dicts or a dict
    with a ``configs`` list (see ``window_table``).

    Args:
        path: Machine JSON file (e.g. ``sim/machines/<name>.json``).

    Returns:
        The JSON dict with two extra keys: ``name`` (machine name; the file stem
        if absent) and ``window_table`` (list of ``{"rob", "iq", "lq", "sq"}``
        dicts, one per config).
    """
    path = Path(path)
    with open(path) as fh:
        m = json.load(fh)
    m.setdefault("name", path.stem)
    m["window_table"] = window_table(m)
    return m


def window_table(machine: dict) -> list[dict]:
    """Return the per-config window table of a machine description.

    Accepts the ``window`` section in any of three layouts: a list of per-config
    dicts, parallel lists (``{"rob": [...], "iq": [...], ...}``) or a dict with a
    ``configs`` list. Only the ``rob``, ``iq``, ``lq`` and ``sq`` keys are kept and
    cast to ``int``.

    Args:
        machine: Parsed machine JSON.

    Returns:
        One ``{"rob", "iq", "lq", "sq"}`` dict per config, in config order. A copy of
        ``DEFAULT_WINDOW`` when the section is missing, empty or unrecognised.
    """
    w = machine.get("window")
    if not w:
        return [dict(c) for c in DEFAULT_WINDOW]
    if isinstance(w, list):
        return [{k: int(c[k]) for k in ("rob", "iq", "lq", "sq") if k in c} for c in w]
    if isinstance(w, dict) and isinstance(w.get("rob"), list):
        n = len(w["rob"])
        out = []
        for i in range(n):
            out.append({k: int(w[k][i]) for k in ("rob", "iq", "lq", "sq")
                        if isinstance(w.get(k), list) and i < len(w[k])})
        return out
    if isinstance(w, dict) and isinstance(w.get("configs"), list):
        return [{k: int(c[k]) for k in ("rob", "iq", "lq", "sq") if k in c}
                for c in w["configs"]]
    return [dict(c) for c in DEFAULT_WINDOW]


def config_for_window(table: Sequence[dict], w: float) -> int:
    """Map a requested window size to a config index (``setwin(W)``, §3).

    Selects the smallest config whose ROB is >= ``w``. ``w`` that is None, NaN,
    <= 0 or larger than every config selects the largest (last) config.

    Args:
        table: Window table as returned by ``window_table``.
        w: Requested window (ROB entries).

    Returns:
        Index into ``table``.
    """
    if w is None or (isinstance(w, float) and math.isnan(w)) or w <= 0:
        return len(table) - 1
    for i, c in enumerate(table):
        if c["rob"] >= w:
            return i
    return len(table) - 1


# ---------------------------------------------------------------------------
# gem5 outputs
# ---------------------------------------------------------------------------

def load_window_trace(path: str | Path) -> pd.DataFrame:
    """Read window_trace.csv and derive per-window deltas.

    The trace has one row per sampling period. ``cycle`` and ``insts`` may be
    cumulative counters (end of period) or per-period counts; both cases are
    handled: a column that strictly increases over more than two rows is
    treated as cumulative and differenced. Adds columns
    ``d_cycles``, ``d_insts`` and fills a missing/NaN ``ipc`` from them.
    When ``ipc`` is present, the ``insts`` interpretation (raw or differenced)
    whose ``insts / d_cycles`` best reproduces it is chosen. Missing
    ``TRACE_COLUMNS`` are added as NaN and a missing ``region`` becomes
    ``NO_REGION``.

    Args:
        path: Path to ``window_trace.csv``.

    Returns:
        The trace with ``d_cycles``, ``d_insts`` and a filled ``ipc`` column.

    Raises:
        ValueError: If ``cycle``, ``insts`` or ``config`` is missing.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in ("cycle", "insts", "config") if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: window_trace.csv lacks columns {missing}")
    for c in TRACE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df["region"] = df["region"].fillna(NO_REGION).astype(int)
    df["config"] = df["config"].astype(int)

    def deltas(col: str) -> np.ndarray:
        """Return per-period values of trace column ``col``.

        A column with more than two rows that strictly increases is treated as a
        cumulative counter and differenced (``insts`` from 0, ``cycle`` from
        ``_first_start``); otherwise it is returned unchanged.
        """
        v = df[col].to_numpy(dtype=float)
        if len(v) > 2 and np.all(np.diff(v) > 0):
            return np.diff(v, prepend=0.0 if col == "insts" else _first_start(v))
        return v

    df["d_cycles"] = deltas("cycle")
    # gem5's WindowController writes a cumulative ``cycle`` (end of period) and
    # per-period ``insts``. A per-period column can look strictly increasing by
    # chance on short traces, so when ``ipc`` is present it decides: the
    # interpretation whose insts/d_cycles reproduces ipc wins.
    ins = df["insts"].to_numpy(dtype=float)
    d_ins = deltas("insts")
    ipc = pd.to_numeric(df["ipc"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(ipc) & (df["d_cycles"].to_numpy(float) > 0)
    if ok.any():
        dc = df["d_cycles"].to_numpy(float)[ok]
        err_raw = np.nanmean(np.abs(ins[ok] / dc - ipc[ok]))
        err_diff = np.nanmean(np.abs(d_ins[ok] / dc - ipc[ok]))
        if err_raw < err_diff:
            d_ins = ins
    df["d_insts"] = d_ins
    ipc_derived = np.where(df["d_cycles"] > 0, df["d_insts"] / df["d_cycles"].replace(0, np.nan), 0.0)
    df["ipc"] = df["ipc"].where(df["ipc"].notna(), ipc_derived)
    return df


def _first_start(v: np.ndarray) -> float:
    """Return the start cycle of the first period of a cumulative cycle counter.

    Assumes the first period is as long as the second one; clamps at 0 and returns
    0.0 for traces with fewer than two rows.
    """
    # cumulative cycle counter: the first period starts one period earlier
    if len(v) > 1:
        return max(0.0, v[0] - (v[1] - v[0]))
    return 0.0


def window_features(df: pd.DataFrame, features: Sequence[str] = FEATURES) -> np.ndarray:
    """Return the ``(n, len(features))`` float matrix of per-window LUT features.

    Each feature reads its trace column via ``FEATURE_SOURCE`` (falling back to
    the feature name); non-numeric values and NaN become 0.

    Args:
        df: Trace as returned by ``load_window_trace``.
        features: Feature names.

    Returns:
        Feature matrix; shape ``(n, 0)`` when ``features`` is empty.

    Raises:
        KeyError: If a feature's source column is not in ``df``.
    """
    cols = []
    for f in features:
        src = FEATURE_SOURCE.get(f, f)
        if src not in df.columns:
            raise KeyError(f"feature {f!r} needs trace column {src!r}")
        cols.append(pd.to_numeric(df[src], errors="coerce").fillna(0.0).to_numpy(float))
    return np.stack(cols, axis=1) if cols else np.zeros((len(df), 0))


def load_region_stats(path: str | Path) -> pd.DataFrame:
    """Read a gem5 ``region_stats.csv`` (one row per region visit).

    Strips column names, casts ``region``/``config`` to ``int`` and coerces
    ``enter_cycle``/``cycles``/``insts`` to numbers (non-numeric values become 0).

    Args:
        path: Path to ``region_stats.csv``.

    Returns:
        The visit table with at least the ``REGION_COLUMNS`` columns.

    Raises:
        ValueError: If any of ``REGION_COLUMNS`` is missing.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REGION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: region_stats.csv lacks columns {missing}")
    for c in ("region", "config"):
        df[c] = df[c].astype(int)
    for c in ("enter_cycle", "cycles", "insts"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def aggregate_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Sum region visits: one row per region with cycles, insts, visits, ipc.

    Args:
        df: Visit table as returned by ``load_region_stats``.

    Returns:
        One row per region; ``ipc`` is ``insts / cycles`` (0 when no cycles).
    """
    g = df.groupby("region", as_index=False).agg(
        cycles=("cycles", "sum"), insts=("insts", "sum"), visits=("cycles", "size"))
    g["ipc"] = np.where(g["cycles"] > 0, g["insts"] / g["cycles"].where(g["cycles"] > 0, 1), 0.0)
    return g


# ---------------------------------------------------------------------------
# Oracle maps (written by sim/baselines/oracle/oracle_sweep.py)
# ---------------------------------------------------------------------------

def load_oracle(path: str | Path, metric: str = "ipc") -> dict[int, int]:
    """Read an oracle map (best config per region).

    Accepts the flat ``{region: config}`` form (``results/oracle/<kernel>.json``) or
    the full summary written by oracle_sweep.py
    (``{"best": {"ipc": {...}, "ed2p": {...}}, ...}``). Values may be config
    ints or dicts with a ``config`` key; entries that cannot be parsed are
    skipped.

    Args:
        path: Oracle JSON file.
        metric: Which ``best`` map to read from a full summary; falls back to
            ``ipc`` when absent.

    Returns:
        ``{region_id: config_index}``.
    """
    with open(path) as fh:
        d = json.load(fh)
    if isinstance(d, dict) and "best" in d and isinstance(d["best"], dict):
        d = d["best"].get(metric, d["best"].get("ipc", {}))
    out = {}
    for k, v in d.items():
        try:
            out[int(k)] = int(v["config"] if isinstance(v, dict) else v)
        except (TypeError, ValueError, KeyError):
            continue
    return out


def write_flat_oracle(path: str | Path, best: dict[int, int]) -> None:
    """Write a flat oracle map ``{"<region>": <config>}`` as indented JSON.

    Keys are sorted by region id; parent directories are created.

    Args:
        path: Output JSON path (e.g. ``results/oracle/<kernel>.json``).
        best: Best config index per region id.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({str(k): int(v) for k, v in sorted(best.items())}, fh, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Runtime LUT (interfaces.md §6)
# ---------------------------------------------------------------------------

@dataclass
class Lut:
    """Runtime lookup table of the B5 baseline (interfaces.md §6).

    Each feature is binned by its ascending upper edges (``len(edges[i]) + 1``
    bins); the table holds one config index per cell of the cartesian product of
    bins.

    Attributes:
        features: Feature names, in table-axis order (see ``FEATURES``).
        edges: Ascending upper bin edges, one array per feature.
        table: Flat, row-major config table; the first feature varies slowest.
        meta: Free-form metadata; not serialised by ``write_lut``.
    """
    features: list[str]
    edges: list[np.ndarray]
    table: np.ndarray                      # flat, row-major, first feature slowest
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """Number of bins per feature (``len(edges) + 1`` each)."""
        return tuple(len(e) + 1 for e in self.edges)

    def bin_index(self, x: np.ndarray) -> np.ndarray:
        """Return the per-feature bin of each sample.

        The bin is the number of edges strictly below x (edges are upper bin
        edges, inclusive; values above the last edge go to bin n).

        Args:
            x: ``(n, n_features)`` feature values (a 1-D sample is accepted).

        Returns:
            ``(n, n_features)`` integer bin indices.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.stack([np.searchsorted(e, x[:, i], side="left")
                         for i, e in enumerate(self.edges)], axis=1)

    def flat_index(self, bins: np.ndarray) -> np.ndarray:
        """Convert per-feature bin indices to flat row-major table indices.

        Args:
            bins: ``(n, n_features)`` integer bin indices, e.g. from ``bin_index``.

        Returns:
            ``(n,)`` indices into ``table``.
        """
        return np.ravel_multi_index(tuple(np.asarray(bins).T), self.shape)

    def lookup(self, x: np.ndarray) -> np.ndarray:
        """Return the table entry (config index) for each sample.

        Args:
            x: ``(n, n_features)`` feature values (a single 1-D sample is accepted).

        Returns:
            ``(n,)`` config indices.
        """
        return self.table[self.flat_index(self.bin_index(x))]


def write_lut(path: str | Path, lut: Lut, comment: str | None = None) -> None:
    """Serialise a LUT in the ``WINHINT_LUT 1`` text format (interfaces.md §6).

    Writes the header, optional ``#`` comment lines, the ``features`` line, one
    ``edges`` line per feature (values formatted with ``%.9g``) and the ``table``
    entries, 32 per line. ``lut.meta`` is not written. Parent directories are
    created.

    Args:
        path: Output file path.
        lut: Table to write.
        comment: Optional text, emitted as one ``#`` comment line per input line.

    Raises:
        ValueError: If ``lut.table`` does not have exactly ``prod(lut.shape)``
            entries.
    """
    n_cells = int(np.prod(lut.shape))
    if lut.table.size != n_cells:
        raise ValueError(f"table has {lut.table.size} entries, expected {n_cells}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["WINHINT_LUT 1"]
    if comment:
        lines += [f"# {c}" for c in comment.splitlines()]
    lines.append(f"features {len(lut.features)} " + " ".join(lut.features))
    for name, e in zip(lut.features, lut.edges):
        lines.append(f"edges {name} {len(e)}" + "".join(f" {v:.9g}" for v in e))
    lines.append(f"table {n_cells}")
    tab = [str(int(v)) for v in lut.table]
    for i in range(0, len(tab), 32):
        lines.append(" ".join(tab[i:i + 32]))
    path.write_text("\n".join(lines) + "\n")


def read_lut(path: str | Path) -> Lut:
    """Parse the §6 ``WINHINT_LUT 1`` text format.

    '#' starts a comment; table values may span any number of lines.

    Args:
        path: LUT file.

    Returns:
        The parsed table (``meta`` empty).

    Raises:
        ValueError: On a bad header, unexpected keyword, truncated file,
            non-ascending edges or a table size that does not match the bins.
    """
    tokens: list[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            tokens.extend(line.split())
    it = iter(tokens)

    def nxt() -> str:
        """Return the next token; raise ``ValueError`` if the file ends early."""
        try:
            return next(it)
        except StopIteration:
            raise ValueError(f"{path}: truncated LUT") from None

    if nxt() != "WINHINT_LUT" or nxt() != "1":
        raise ValueError(f"{path}: not a WINHINT_LUT 1 file")
    if nxt() != "features":
        raise ValueError(f"{path}: expected 'features'")
    nf = int(nxt())
    feats = [nxt() for _ in range(nf)]
    edges = []
    for name in feats:
        if nxt() != "edges" or nxt() != name:
            raise ValueError(f"{path}: expected 'edges {name}'")
        n = int(nxt())
        e = np.array([float(nxt()) for _ in range(n)])
        if np.any(np.diff(e) < 0):
            raise ValueError(f"{path}: edges of {name} not ascending")
        edges.append(e)
    if nxt() != "table":
        raise ValueError(f"{path}: expected 'table'")
    n = int(nxt())
    table = np.array([int(nxt()) for _ in range(n)], dtype=np.int64)
    lut = Lut(feats, edges, table)
    if n != int(np.prod(lut.shape)):
        raise ValueError(f"{path}: table size {n} != product of bins {np.prod(lut.shape)}")
    return lut


def quantile_edges(x: np.ndarray, n_edges: int) -> np.ndarray:
    """Return ascending, unique upper bin edges at evenly spaced quantiles.

    Non-finite values are ignored. Duplicate quantiles are merged, so fewer
    than ``n_edges`` edges may be returned.

    Args:
        x: Sample values.
        n_edges: Requested number of edges (``n_edges + 1`` bins).

    Returns:
        Edge array; empty when ``x`` has no finite values or ``n_edges <= 0``.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or n_edges <= 0:
        return np.array([])
    qs = np.linspace(0, 1, n_edges + 2)[1:-1]
    e = np.unique(np.quantile(x, qs))
    return e


def iter_trace_dirs(root: str | Path, name: str = "window_trace.csv") -> Iterable[Path]:
    """Yield trace files found under ``root``.

    Args:
        root: A directory to search recursively, or a single file (yielded as is).
        name: File name to match.

    Yields:
        Matching paths, in sorted order.
    """
    root = Path(root)
    if root.is_file():
        yield root
        return
    yield from sorted(root.rglob(name))
