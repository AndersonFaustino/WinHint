#!/usr/bin/env python3
"""Tune every baseline's parameters with the same effort as WinHint (tune_baselines.py).

Tunes on the training inputs (PROPOSAL §8, "reimplemented baselines could be
accused of being weak").

Methods and their search spaces (keys = the policies' --window-args, see
docs/guide/gem5/policies.md, docs/guide/gem5/patches.md and docs/guide/baselines/ltp.md;
the published/default point is always part of the grid):

    method      variant (run_experiments)  knobs
    ----------  -------------------------  ------------------------------------------------
    B2          occupancy                  update, up_frac, down_factor
    B3          mlp                        mlp_thr, gain, miss_min, shrink_delay
    B4          bbv                        interval_insts, thr, tol, explore
    B5          lut                        LUT training: --model, --edges (build_lut.py)
    B8          clairvoyance               CV_TYPE, CV_UNROLL, CV_INDIR
                                           (compiler/baselines/clairvoyance/knobs.json)
    B8+WinHint  winhint_clairvoyance       same Clairvoyance grid
    B9          ltp                        entries, lll, wake, room
    hybrid      winhint_hw                 miss_mpki, mlp_thr, up_frac, down_margin
    WinHint     winhint                    compiler knobs: switch cost (-winhint-switch-cost,
                                           make SWITCH_COST; 0 = model default) and
                                           hysteresis (-winhint-hysteresis)

Equal effort: every method evaluates at most ``--budget`` parameter points
(default: the size of WinHint's own grid, 16), always including its default
point; a larger grid is subsampled uniformly at random (``--seed``). Every point is
evaluated on the same kernels, machines and inputs (``small``, the training input;
the run length comes from sim/run_lengths.json, i.e. full runs), through the
same runner (sim/run_experiments.py: heavy lock, resumable outdirs).

Selection: per machine, the point with the lowest geometric mean over kernels of
its cost relative to the default point (``--objective ed2p``: ED²P from
sim/estimate_energy.py; ``ipc``: cycles per instruction). Ties keep the default.
Points with a failed or missing kernel are not eligible.

Output (``--out``, default results/tune/tuned.json; read by run_experiments.py
--tuned, so the final campaign uses the tuned parameters):

    window_args.<machine>.<policy>   tuned --window-args (B2, B3, B4, B9, hybrid)
    lut_root.<machine>.root          LUT root of the tuned B5 training
    binary_suffix.<variant>          B8 build dir suffix (-cv_<type>-u<N>-i<K>)
    compiler.winhint.{make,sidecar}  WinHint knobs: make variables for the final build
                                     and the values its <kernel>.winhint.json must show
    "*" entries = the choice on the first machine (fallback for other machines);
    points, scores, budget, seed, objective and run-length version for the record.

Steps (each skipped when its product exists; --dry-run prints them all and the
gem5 job list, running nothing):

    build   compiler-knob binaries (make -j1 into $WINHINT_BUILD/tune/bins/<method>/<pid>
            for WinHint; the Makefile's -cv_* dirs for B8) and B5 LUTs
            (build_lut.py --out-root results/tune/b5/<pid>); --no-build skips this
    run     gem5 runs: results/tune/runs/<machine>/<kernel>/<method>/<pid>/
    select  scores and tuned.json (--select-only: only this step)

Usage (host, `winhint` env active):

    python sim/baselines/tune/tune_baselines.py --dry-run
    python sim/baselines/tune/tune_baselines.py --methods B3 hybrid --kernels encoder_bert_tiny_infer
    python sim/baselines/tune/tune_baselines.py --select-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import os
import random
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "sim"))
sys.path.insert(0, str(REPO / "sim" / "baselines" / "lut"))
import run_experiments as rx  # noqa: E402
from estimate_energy import estimate_run, key_stats, parse_stats  # noqa: E402
from whdata import load_machine  # noqa: E402

#: Default results root of the tuning campaign (``--results-root``).
TUNE_ROOT = REPO / "results" / "tune"
#: Default root of the WinHint compiler-knob binaries (``--tune-bins``).
TUNE_BINS = rx.BUILD / "tune" / "bins"
#: Clairvoyance knob description (B8 grid and default).
KNOBS_B8 = REPO / "compiler" / "baselines" / "clairvoyance" / "knobs.json"
#: Default ``--machines``; the first one is the reference machine.
DEFAULT_MACHINES = ["riscv_ooo"]

# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

#: WinHint compiler-knob grid; its size (16) is the default ``--budget``.
WINHINT_GRID = {"switch_cost": [0, 100, 500, 2000], "hysteresis": [0.0, 0.1, 0.25, 0.5]}


@dataclass
class Method:
    """One tunable baseline and its search space.

    Attributes:
        name: Method name (``B2`` ... ``WinHint``; key of ``--methods``).
        variant: run_experiments variant name.
        kind: How a point is applied: ``window_args``, ``lut``, ``winhint`` or
            ``cv``.
        grid: Knob -> candidate values.
        default: Knob -> published/default value.
    """
    name: str
    variant: str          # run_experiments variant name
    kind: str             # window_args | lut | winhint | cv
    grid: dict            # knob -> values
    default: dict         # knob -> default value

    def points(self) -> list[dict]:
        """Return every point of the full grid (cartesian product, knobs in ``grid`` order)."""
        keys = list(self.grid)
        return [dict(zip(keys, vals)) for vals in itertools.product(*(self.grid[k] for k in keys))]


def _cv_grid() -> tuple[dict, dict]:
    """Return the B8 Clairvoyance grid and default point.

    Read from ``KNOBS_B8`` (knobs ``CV_TYPE``, ``CV_UNROLL``, ``CV_INDIR``; a knob
    with ``"tunable": false`` is left out). Falls back to a built-in grid if the
    file is missing or malformed.

    Returns:
        ``(grid, default)``.
    """
    try:
        k = json.loads(KNOBS_B8.read_text())["knobs"]
        grid = {n: list(k[n]["values"]) for n in ("CV_TYPE", "CV_UNROLL", "CV_INDIR")
                if k.get(n, {}).get("tunable", True)}
        default = {n: k[n]["default"] for n in grid}
        return grid, default
    except (OSError, ValueError, KeyError):
        return ({"CV_TYPE": ["consv", "specsafe", "spec", "multispecsafe", "multispec"],
                 "CV_UNROLL": [1, 2, 4], "CV_INDIR": [0, 1, 2, 3]},
                {"CV_TYPE": "consv", "CV_UNROLL": 2, "CV_INDIR": 1})


def methods() -> dict[str, Method]:
    """Return every tunable method keyed by name, with its grid and default point."""
    cvg, cvd = _cv_grid()
    ms = [
        Method("B2", "occupancy", "window_args",
               {"update": [1, 2, 4], "up_frac": [0.01, 0.05, 0.1], "down_factor": [0.5, 1.0, 1.5]},
               {"update": 2, "up_frac": 0.05, "down_factor": 1.0}),
        Method("B3", "mlp", "window_args",
               {"mlp_thr": [1.2, 1.5, 2.0, 3.0], "gain": [0.05, 0.1, 0.2], "miss_min": [1, 4],
                "shrink_delay": [1, 2, 4]},
               {"mlp_thr": 1.5, "gain": 0.1, "miss_min": 1, "shrink_delay": 2}),
        Method("B4", "bbv", "window_args",
               {"interval_insts": [50000, 100000, 200000], "thr": [0.1, 0.25, 0.5],
                "tol": [0.01, 0.02, 0.05], "explore": [1, 2]},
               {"interval_insts": 100000, "thr": 0.25, "tol": 0.02, "explore": 1}),
        Method("B5", "lut", "lut",
               {"model": ["mlp", "tree", "knn", "bins"], "edges": [5, 7, 9]},
               {"model": "mlp", "edges": 7}),
        Method("B8", "clairvoyance", "cv", cvg, cvd),
        Method("B8+WinHint", "winhint_clairvoyance", "cv", cvg, cvd),
        Method("B9", "ltp", "window_args",
               {"entries": [64, 128], "lll": [20, 30, 50], "wake": [8, 16, 32],
                "room": [0.25, 0.5, 1.0]},
               {"entries": 128, "lll": 30, "wake": 16, "room": 0.5}),
        Method("hybrid", "winhint_hw", "window_args",
               {"miss_mpki": [0.5, 1.0, 2.0], "mlp_thr": [1.2, 1.5, 2.0],
                "up_frac": [0.02, 0.05, 0.1], "down_margin": [0.8, 0.9]},
               {"miss_mpki": 1.0, "mlp_thr": 1.5, "up_frac": 0.05, "down_margin": 0.9}),
        Method("WinHint", "winhint", "winhint", WINHINT_GRID, {"switch_cost": 0, "hysteresis": 0.25}),
    ]
    return {m.name: m for m in ms}


def point_id(params: dict) -> str:
    """Return the identifier of a parameter point (``k1=v1-k2=v2``, ``/`` -> ``_``).

    Used as directory name for runs, binaries and LUTs of the point.
    """
    return "-".join(f"{k}={v}" for k, v in params.items()).replace("/", "_")


def choose_points(m: Method, budget: int, seed: int) -> list[dict]:
    """Return the default point plus up to budget-1 random others (whole grid if it fits).

    The sample is seeded by ``seed`` and the search space, so methods sharing a
    grid (B8, B8+WinHint) evaluate the same points.

    Args:
        m: Method.
        budget: Maximum number of points.
        seed: Sampling seed (``--seed``).

    Returns:
        Points, the default first.
    """
    pts = m.points()
    default = dict(m.default)
    if default not in pts:
        pts.append(default)
    rest = [p for p in pts if p != default]
    if len(pts) > budget:
        # seeded by the search space, so methods sharing a grid (B8, B8+WinHint)
        # evaluate the same points
        key = json.dumps([m.kind, m.grid], sort_keys=True)
        rest = random.Random(f"{seed}:{key}").sample(rest, max(0, budget - 1))
    return [default] + rest


def window_args_str(params: dict) -> str:
    """Return ``params`` as a ``--window-args`` string (``k1=v1,k2=v2``)."""
    return ",".join(f"{k}={v}" for k, v in params.items())


# ---------------------------------------------------------------------------
# Binaries / LUTs per point
# ---------------------------------------------------------------------------

def cv_suffix(params: dict, default: dict) -> str:
    """Return the B8 build directory suffix of a Clairvoyance point.

    Args:
        params: Point with ``CV_TYPE``, ``CV_UNROLL`` and ``CV_INDIR``.
        default: Default point of the method.

    Returns:
        ``""`` for the default point, else ``-cv_<type>-u<N>-i<K>``.
    """
    if params == default:
        return ""
    return f"-cv_{params['CV_TYPE']}-u{params['CV_UNROLL']}-i{params['CV_INDIR']}"


def make_vars_winhint(params: dict) -> dict:
    """Return the make variables that build WinHint with the point's compiler knobs.

    ``SWITCH_COST`` is only set for a non-zero switch cost (0 = model default);
    the hysteresis is passed as ``CFLAGS_EXTRA=-mllvm -winhint-hysteresis=<h>``.

    Args:
        params: Point with ``switch_cost`` and ``hysteresis``.

    Returns:
        ``{make_variable: value}``.
    """
    out = {}
    if params.get("switch_cost"):
        out["SWITCH_COST"] = str(params["switch_cost"])
    out["CFLAGS_EXTRA"] = f"-mllvm -winhint-hysteresis={params['hysteresis']}"
    return out


@dataclass
class Prep:
    """How to obtain what one point needs (binary root / LUT) and how to build it.

    Attributes:
        bin_root: Root of the binary variant directories.
        bin_dir: Binary variant directory under ``bin_root``.
        lut_root: B5 LUT root of the point, or None.
        build: Command that builds the missing product, or None.
        product: File/directory whose existence means "built", or None.
    """
    bin_root: Path
    bin_dir: str             # binary variant directory under bin_root
    lut_root: Path | None
    build: list[str] | None  # command that builds the missing product, or None
    product: Path | None     # file whose existence means "built"


def prepare(m: Method, params: dict, a, machine: str, kernels: list[str]) -> Prep:
    """Describe where a point's binaries/LUT live and how to build them.

    ``winhint`` points build into ``<tune-bins>/<method>/<pid>``; ``cv`` points
    use the Makefile's ``<variant><bin-suffix>-cv_*`` directories under
    ``--bin-root``; ``lut`` points run build_lut.py into
    ``<results-root>/b5/<pid>``; ``window_args`` points reuse the existing base
    binaries and need no build.

    Args:
        m: Method.
        params: Parameter point.
        a (argparse.Namespace): Parsed CLI namespace.
        machine: Machine name.
        kernels: Kernels to build.

    Returns:
        The ``Prep`` for the point.
    """
    pid = point_id(params)
    base_bin = {"winhint_hw": "winhint", "occupancy": "plain", "mlp": "plain", "bbv": "plain",
                "ltp": "plain", "lut": "plain", "winhint": "winhint",
                "clairvoyance": "clairvoyance",
                "winhint_clairvoyance": "winhint_clairvoyance"}[m.variant]
    mjson = rx.MACHINES_DIR / f"{machine}.json"
    if m.kind == "winhint":
        root = a.tune_bins / m.name / pid
        mv = make_vars_winhint(params)
        cmd = (["make", "-C", str(REPO / "benchmarks"), "-j1", "ARCH=riscv", "VARIANT=winhint",
                f"OUT_ROOT={root}", f"MACHINE={mjson}"] + [f"{k}={v}" for k, v in mv.items()]
               + [f"KERNELS={' '.join(kernels)}"])
        prod = root / "riscv" / ("winhint" + a.bin_suffix)
        return Prep(root / "riscv", "winhint" + a.bin_suffix, None, cmd, prod)
    if m.kind == "cv":
        suffix = cv_suffix(params, m.default)
        cmd = (["make", "-C", str(REPO / "benchmarks"), "-j1", "ARCH=riscv",
                f"VARIANT={m.variant}", f"MACHINE={mjson}"]
               + [f"{k}={v}" for k, v in params.items()] + [f"KERNELS={' '.join(kernels)}"])
        return Prep(a.bin_root, m.variant + a.bin_suffix + suffix, None, cmd,
                    a.bin_root / (m.variant + a.bin_suffix + suffix))
    if m.kind == "lut":
        root = a.results_root / "b5" / pid
        cmd = [sys.executable, str(REPO / "sim" / "baselines" / "lut" / "build_lut.py"),
               "--machines", machine, "--model", str(params["model"]),
               "--select", str(params["model"]), "--edges", str(params["edges"]),
               "--out-root", str(root)]
        return Prep(a.bin_root, base_bin + a.bin_suffix, root, cmd, root / machine / "lut.txt")
    return Prep(a.bin_root, base_bin + a.bin_suffix, None, None, None)


def built(prep: Prep, machine: str, kernels: list[str]) -> bool:
    """Return whether the point's product already exists.

    Args:
        prep: Result of ``prepare``.
        machine: Machine name (for the binary lookup).
        kernels: Kernels whose binaries must exist.

    Returns:
        True if nothing needs building, the LUT file exists, or every kernel's
        binary is found.
    """
    if prep.product is None:
        return True
    if prep.lut_root is not None:
        return prep.product.is_file()
    return all(rx.find_binary(prep.bin_root, prep.bin_dir, machine, k) is not None
               for k in kernels)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def build_runs(a, plan: dict) -> list:
    """Return ``rx.Run`` objects for every (method, point, machine, kernel).

    Runs go to ``<results-root>/runs/<machine>/<kernel>/<method>/<pid>/`` and
    carry ``r.tune = (method, pid)``. Missing binaries, LUTs or se.py policies
    are recorded in ``r.missing``. Run lengths are applied via
    ``rx.apply_run_length``.

    Args:
        a (argparse.Namespace): Parsed CLI namespace.
        plan: Method -> points to evaluate.

    Returns:
        The run list.
    """
    runs = []
    supported = rx.se_policies()
    lock = None if a.no_lock else rx.HEAVY_LOCK
    for mname in a.machines:
        mf = rx.MACHINES_DIR / f"{mname}.json"
        machine = load_machine(mf)
        machine["name"] = mname
        largest = len(machine["window_table"]) - 1
        variants = {v.name: v for v in rx.variants_for(machine, mname)}
        for meth, pts in plan.items():
            m = a.methods_def[meth]
            v = variants[m.variant]
            for params in pts:
                pid = point_id(params)
                prep = prepare(m, params, a, mname, a.kernels)
                for k in a.kernels:
                    outdir = a.results_root / "runs" / mname / k / meth / pid
                    binary = rx.find_binary(prep.bin_root, prep.bin_dir, mname, k)
                    missing = []
                    if binary is None:
                        missing.append(f"binary {prep.bin_root}/{prep.bin_dir}/[{mname}/]{k}")
                    if supported is not None and v.policy not in supported:
                        missing.append(f"policy {v.policy!r} not accepted by sim/se.py yet")
                    lut = None
                    if v.lut:
                        lut = (prep.lut_root or a.lut_root) / mname / "lut.txt"
                        if not lut.is_file():
                            missing.append(f"LUT {lut}")
                    wa = v.window_args or ""
                    if m.kind == "window_args":
                        wa = rx.variant_window_args(v, {v.name: window_args_str(params)}) or ""
                    kw = dict(gem5=a.gem5, machine_json=mf,
                              binary=binary or f"<missing:{prep.bin_dir}/{k}>",
                              input_size=a.input, policy=v.policy,
                              initial=largest if v.initial is None else v.initial,
                              period=None, lut=lut, trace=False, max_insts=None,
                              extra=["--window-args", wa] if wa else None)
                    r = rx.Run(mname, mf, k, v, a.input, outdir, binary, lut,
                               rx.gem5_cmd(outdir=outdir, **kw), missing, lock, cmd_kw=kw,
                               base_machine_json=mf)
                    r.tune = (meth, pid)
                    runs.append(r)
    rx.apply_run_length(runs, a)
    return runs


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def run_cost(d: Path, machine: dict, objective: str, energy_model: str) -> float | None:
    """Return the cost of one finished run under the tuning objective.

    Args:
        d: Run output directory.
        machine: Machine dict.
        objective: ``"ipc"`` (cost = cycles per instruction) or ``"ed2p"``.
        energy_model: Energy model passed to ``estimate_energy.estimate_run``.

    Returns:
        The cost (lower is better), or None if the run is missing, not
        ``status == "ok"`` in ``run.json``, or yields a zero/undefined cost.
    """
    rj = d / "run.json"
    if not (rj.is_file() and (d / "stats.txt").is_file()):
        return None
    try:
        if json.loads(rj.read_text()).get("status") != "ok":
            return None
    except ValueError:
        return None
    ks = key_stats(parse_stats(d / "stats.txt"))
    if objective == "ipc":
        return ks["cycles"] / ks["insts"] if ks["insts"] else None
    e = estimate_run(d, machine, energy_model)
    return e["ed2p"] or None


def select(a, plan: dict) -> dict:
    """Score every point and pick the best per method and machine.

    A point's score is the geometric mean over kernels of its cost relative to
    the default point (``run_cost``); points lacking any kernel (or a default
    lacking any kernel) are not scored. The lowest score wins; ties keep the
    default.

    Args:
        a (argparse.Namespace): Parsed CLI namespace.
        plan: Method -> points; the first point is the default.

    Returns:
        ``{method: {machine: (best pid or None, {pid: score})}}``.
    """
    out: dict = {}
    for mname in a.machines:
        machine = load_machine(rx.MACHINES_DIR / f"{mname}.json")
        machine["name"] = mname
        for meth, pts in plan.items():
            pids = [point_id(p) for p in pts]
            costs = {pid: {} for pid in pids}
            for pid in pids:
                for k in a.kernels:
                    c = run_cost(a.results_root / "runs" / mname / k / meth / pid, machine,
                                 a.objective, a.energy_model)
                    if c is not None:
                        costs[pid][k] = c
            base = costs[pids[0]]                     # pids[0] = the default point
            scores = {}
            for pid in pids:
                if base and all(k in costs[pid] and k in base for k in a.kernels):
                    scores[pid] = math.exp(sum(math.log(costs[pid][k] / base[k])
                                               for k in a.kernels) / len(a.kernels))
            best = min(scores, key=lambda p: (scores[p], p != pids[0])) if scores else None
            out.setdefault(meth, {})[mname] = (best, scores)
    return out


def write_tuned(a, plan: dict, chosen: dict) -> dict:
    """Write ``tuned.json`` (``--out``) from the selected points.

    Per machine, ``window_args`` and ``lut_root`` entries are written for the
    chosen point; the reference (first) machine's choice is also stored under
    ``"*"``. ``binary_suffix`` (B8) and ``compiler.winhint`` (make variables
    plus the knob values read from the first available ``.winhint.json``
    sidecar) come from the reference machine only. Methods without an eligible
    point are skipped.

    Args:
        a (argparse.Namespace): Parsed CLI namespace.
        plan: Method -> evaluated points.
        chosen: Result of ``select``.

    Returns:
        The written dict.
    """
    by_pid = {meth: {point_id(p): p for p in pts} for meth, pts in plan.items()}
    ref = a.machines[0]
    t = {"version": 1, "created": dt.datetime.now().isoformat(timespec="seconds"),
         "objective": a.objective, "energy_model": a.energy_model, "input": a.input,
         "kernels": a.kernels, "machines": a.machines, "budget": a.budget, "seed": a.seed,
         "run_length_version": rx.rl.load_config(a.run_lengths).get("version"),
         "window_args": {}, "lut_root": {}, "binary_suffix": {}, "compiler": {},
         "points": {meth: {pid: p for pid, p in d.items()} for meth, d in by_pid.items()},
         "scores": {meth: {m: sc for m, (_, sc) in d.items()} for meth, d in chosen.items()},
         "chosen": {meth: {m: b for m, (b, _) in d.items()} for meth, d in chosen.items()}}
    for meth, per_m in chosen.items():
        m = a.methods_def[meth]
        for mname, (best, _) in per_m.items():
            if best is None:
                continue
            params = by_pid[meth][best]
            keys = [mname] + (["*"] if mname == ref else [])
            for key in keys:
                if m.kind == "window_args":
                    pol = {"occupancy": "occupancy", "mlp": "mlp", "bbv": "bbv", "ltp": "ltp",
                           "winhint_hw": "hybrid"}[m.variant]
                    t["window_args"].setdefault(key, {})[pol] = window_args_str(params)
                elif m.kind == "lut":
                    t["lut_root"].setdefault(key, {})["root"] = str(a.results_root / "b5" / best)
            if mname != ref:
                continue
            if m.kind == "cv":
                t["binary_suffix"][m.variant] = cv_suffix(params, m.default)
            elif m.kind == "winhint":
                ent = {"make": make_vars_winhint(params), "params": params}
                prep = prepare(m, params, a, mname, a.kernels)
                for k in a.kernels:
                    b = rx.find_binary(prep.bin_root, prep.bin_dir, mname, k)
                    side = b.parent / f"{b.name}.winhint.json" if b else None
                    if side and side.is_file():
                        s = json.loads(side.read_text())
                        ent["sidecar"] = {"switch_cost_cycles": s.get("switch_cost_cycles"),
                                          "hysteresis": s.get("hysteresis")}
                        break
                t["compiler"]["winhint"] = ent
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(t, indent=2) + "\n")
    print(f"[OK] tuned parameters -> {a.out}")
    return t


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """Parse and normalise the command line.

    Fills defaults (all methods, all kernels, ``DEFAULT_MACHINES``, budget =
    size of WinHint's grid, ``--out`` = ``<results-root>/tuned.json``), clamps
    ``--jobs`` to ``run_experiments.MAX_JOBS``, attaches ``methods_def`` and
    resolves the path options.

    Args:
        argv (list[str] | None): Argument list; ``None`` reads ``sys.argv``.

    Returns:
        args (argparse.Namespace): The parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="*", help="default: all (" + ", ".join(methods()) + ")")
    p.add_argument("--kernels", nargs="*", help="default: all benchmarks/*_infer.c")
    p.add_argument("--machines", nargs="*", default=None,
                   help=f"default: {' '.join(DEFAULT_MACHINES)} (the first is the reference)")
    p.add_argument("--input", default="small", choices=["small", "large"],
                   help="tuning input (PROPOSAL §8: the training input, small)")
    p.add_argument("--budget", type=int, default=None,
                   help="points per method (default: size of WinHint's grid)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--objective", default="ed2p", choices=["ed2p", "ipc"])
    p.add_argument("--energy-model", default="auto", choices=["auto", "mcpat", "proxy"])
    p.add_argument("--dry-run", action="store_true", help="print the steps and jobs, run nothing")
    p.add_argument("--select-only", action="store_true", help="only score and write --out")
    p.add_argument("--no-build", action="store_true", help="do not build binaries/LUTs")
    p.add_argument("--force", action="store_true")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--gem5", default=rx.GEM5_BIN)
    p.add_argument("--bin-root", type=Path, default=rx.BIN_ROOT)
    p.add_argument("--bin-suffix", default="")
    p.add_argument("--lut-root", type=Path, default=rx.LUT_ROOT)
    p.add_argument("--tune-bins", type=Path, default=TUNE_BINS)
    p.add_argument("--results-root", type=Path, default=TUNE_ROOT)
    p.add_argument("--out", type=Path, default=None, help="default <results-root>/tuned.json")
    p.add_argument("--run-lengths", type=Path, default=rx.RUN_LENGTHS)
    p.add_argument("--run-length", default="config", choices=["config", "full"])
    p.add_argument("--runlen-root", type=Path, default=rx.RUNLEN_ROOT)
    p.add_argument("--ckpt-root", type=Path, default=rx.CKPT_ROOT)
    p.add_argument("--no-lock", action="store_true")
    a = p.parse_args(argv)
    a.methods_def = methods()
    unknown = set(a.methods or []) - set(a.methods_def)
    if unknown:
        p.error(f"unknown methods {sorted(unknown)}; choose from {list(a.methods_def)}")
    a.methods = a.methods or list(a.methods_def)
    a.kernels = a.kernels or rx.all_kernels()
    a.machines = a.machines or list(DEFAULT_MACHINES)
    if a.budget is None:
        a.budget = len(Method("WinHint", "winhint", "winhint", WINHINT_GRID, {}).points())
    a.out = a.out or a.results_root / "tuned.json"
    a.jobs = max(1, min(a.jobs, rx.MAX_JOBS))
    a.max_insts = None
    for f in ("bin_root", "lut_root", "tune_bins", "results_root", "runlen_root", "ckpt_root"):
        setattr(a, f, getattr(a, f).resolve())
    return a


def main(argv=None) -> int:
    """Run the build, run and select steps (see the module docstring).

    Args:
        argv (list[str] | None): Argument list; ``None`` reads ``sys.argv``.

    Returns:
        0 on success or for ``--dry-run``/``--select-only``; 1 if any gem5 run
        failed. Failed builds only print a warning.
    """
    a = parse_args(argv)
    plan = {meth: choose_points(a.methods_def[meth], a.budget, a.seed) for meth in a.methods}
    if a.select_only:
        write_tuned(a, plan, select(a, plan))
        return 0

    # build step (compiler-knob binaries, B5 LUTs)
    builds, seen = [], set()
    for meth, pts in plan.items():
        m = a.methods_def[meth]
        for params in pts:
            for mname in (a.machines if m.kind in ("lut", "winhint", "cv") else []):
                prep = prepare(m, params, a, mname, a.kernels)
                key = " ".join(prep.build or [])
                if prep.build and key not in seen and not built(prep, mname, a.kernels):
                    seen.add(key)
                    builds.append((meth, point_id(params), mname, prep))

    runs = build_runs(a, plan)
    if a.dry_run:
        print(f"# budget {a.budget} points per method, seed {a.seed}, objective {a.objective}, "
              f"input {a.input}, {len(a.kernels)} kernels x {len(a.machines)} machines")
        for meth, pts in plan.items():
            m = a.methods_def[meth]
            print(f"# {meth} ({m.variant}, {m.kind}): {len(pts)} of {len(m.points())} grid points")
            for params in pts:
                print(f"    {point_id(params)}")
        print(f"# build step: {len(builds)} commands")
        for meth, pid, mname, prep in builds:
            print(f"  build {meth:<10} {pid} [{mname}]: " + " ".join(shlex.quote(c) for c in prep.build))
        print(f"# run step: {len(runs)} gem5 runs")
        for r in runs:
            st = ("done" if rx.is_done(r.outdir) else "MISSING" if r.missing
                  else "pending" if r.pending else "todo")
            meth, pid = r.tune
            det = "; ".join(r.missing) if r.missing else " ".join(shlex.quote(c) for c in r.cmd)
            print(f"  {r.machine:<12} {r.kernel:<28} {meth:<10} {pid:<48} {st:<8} {det}")
        return 0

    if not a.no_build:
        env = dict(os.environ)
        for meth, pid, mname, prep in builds:
            print(f"[build] {meth} {pid} [{mname}]")
            if subprocess.run(prep.build, cwd=str(REPO), env=env).returncode != 0:
                print(f"[WARN] build failed: {meth} {pid}")
    _, failed = rx.run_phases(lambda: build_runs(a, plan), a,
                              label=lambda r: f"{r.machine}/{r.kernel}/{r.tune[0]}/{r.tune[1]}")
    write_tuned(a, plan, select(a, plan))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
