#!/usr/bin/env python3
"""B7 baseline: profile-guided positional adaptation of the instruction window.

Positional adaptation after Huang, Renau, Torrellas (ISCA'03); region markers
after Lau, Perelman, Calder (CGO'06).

Host only (no Docker). Run inside the `winhint` micromamba env (`WINHINT_ROOT`,
`WINHINT_BUILD` set by its activation hook; defaults: repo root and `<root>/build`).

Flow:

1. `profile`: build the `oracle` variant (`region(id)` markers, docs/interfaces.md §5)
   with benchmarks/Makefile, then run it in gem5 (RISCV_winhint build, sim/se.py)
   on the *small* input once per window configuration
   (`--window-policy static --window-initial <c>`). Each run writes
   `region_stats.csv` (region,config,enter_cycle,cycles,insts) to
   `results/pgo/runs/<kernel>/<input>/cfg<c>/`. Simulations run serially, each under
   `flock $WINHINT_BUILD/.heavy.lock` (docs/interfaces.md §1).
2. `select`: aggregate cycles/insts per (region, config), pick the best
   configuration per region (ED²P by default, or EDP / IPC / cycles / energy) and
   write the map consumed by `-winhint-mode=from-json=<map>`
   (`results/pgo/<kernel>.json`).
3. `build`: build the `pgo` variant (setwin per region from the map) with
   benchmarks/Makefile; it is then evaluated on the *large* input
   (sim/run_experiments.py).

`all` = profile + select + build.

The same `select` logic can produce the oracle map (B1) from runs on the
large input (`--input large`).

Energy per region: if `region_stats.csv` has an `energy` (J) column it is used;
otherwise `E = P(config) * cycles` with a power proxy
`P(c) = 1 + energy_weight * ROB_c / ROB_max` (the same proxy as the WinHint cost
model), or per-config watts from `--power-json {"0": W0, "1": W1, ...}`
(e.g. derived from McPAT, sim/estimate_energy.py).

Usage:

    pgo_flow.py profile --kernel K [--machine sim/machines/riscv_ooo.json] [--input small]
    pgo_flow.py select  --kernel K [--stats results/pgo/runs/K] [--metric ed2p|ipc|...]
                        [--regions <kernel>.regions.json]
    pgo_flow.py build   --kernel K [--arch riscv] [make vars, e.g. OPT=O3]
    pgo_flow.py all     --kernel K [--metric ed2p]

`--dry-run` prints the gem5/make commands without running them; it still runs the
read-only query `make -s outdir` to resolve the binary path.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict

#: Repository root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
#: Window table used when no machine JSON is available.
DEFAULT_WINDOW = [
    {"rob": 64, "iq": 32, "lq": 16, "sq": 16},
    {"rob": 128, "iq": 64, "lq": 32, "sq": 32},
    {"rob": 192, "iq": 96, "lq": 48, "sq": 48},
    {"rob": 256, "iq": 128, "lq": 64, "sq": 64},
]


def load_window(machine):
    """Return the window table of a machine JSON (same formats the plugin accepts).

    Accepts ``window`` as a list of configuration dicts, as ``{"configs": [...]}``
    or as a dict of per-field lists (``{"rob": [...], "iq": [...], ...}``).

    Args:
        machine (str | None): Path of the machine JSON, or a false value.

    Returns:
        (list[dict]): List of ``{"rob", "iq", "lq", "sq"}`` dicts sorted by ROB size;
            :data:`DEFAULT_WINDOW` if the file is missing or has no usable window.
    """
    if not machine or not os.path.exists(machine):
        return DEFAULT_WINDOW
    with open(machine) as f:
        m = json.load(f)
    w = m.get("window")
    if isinstance(w, dict) and "configs" in w:
        w = w["configs"]
    if isinstance(w, list) and w:
        return sorted(w, key=lambda c: c["rob"])
    if isinstance(w, dict) and "rob" in w:
        n = len(w["rob"])
        cfgs = []
        for i in range(n):
            cfgs.append({k: w[k][i] for k in ("rob", "iq", "lq", "sq") if k in w and i < len(w[k])})
        return sorted(cfgs, key=lambda c: c["rob"])
    return DEFAULT_WINDOW


def read_stats(paths):
    """Aggregate ``region_stats.csv`` files per (region, config).

    Rows without an integer ``region``/``config`` are skipped. Energy comes from an
    ``energy`` or ``energy_j`` column when present.

    Args:
        paths (iterable of str): Iterable of CSV paths.

    Returns:
        (dict): ``{(region, config): {"cycles", "insts", "energy", "visits"}}``; ``energy``
            stays ``None`` when no row of that key had an energy value.
    """
    agg = defaultdict(lambda: {"cycles": 0.0, "insts": 0.0, "energy": None, "visits": 0})
    for p in paths:
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    key = (int(row["region"]), int(row["config"]))
                except (KeyError, ValueError):
                    continue
                a = agg[key]
                a["cycles"] += float(row.get("cycles", 0) or 0)
                a["insts"] += float(row.get("insts", 0) or 0)
                a["visits"] += 1
                e = row.get("energy") or row.get("energy_j")
                if e not in (None, ""):
                    a["energy"] = (a["energy"] or 0.0) + float(e)
    return agg


def power_model(window, power_json, energy_weight):
    """Return the per-configuration power used to estimate energy.

    Args:
        window (list[dict]): Window table as returned by :func:`load_window`.
        power_json (str | None): Path of a ``{"<config>": watts}`` JSON, or a false value.
        energy_weight (float): Weight of the ROB-size term of the proxy.

    Returns:
        (dict[int, float]): ``{config: power}`` from ``power_json`` if given, else the proxy
            ``1 + energy_weight * ROB_c / ROB_max``.
    """
    if power_json:
        with open(power_json) as f:
            pw = json.load(f)
        return {int(k): float(v) for k, v in pw.items()}
    rmax = max(c["rob"] for c in window)
    return {i: 1.0 + energy_weight * c["rob"] / rmax for i, c in enumerate(window)}


def select(args):
    """Pick the best configuration per region and write the PGO map (``select`` step).

    The metric (``--metric``) is computed per instruction; configurations within
    ``--tie`` (relative) of the best go to the smallest index. Entries are annotated
    with ``function``/``line`` from the region table (``--regions`` or
    :func:`default_regions_json`), and regions missing from the profile or from the
    table are reported on stderr.

    Args:
        args (argparse.Namespace): Parsed command-line namespace (``stats``, ``machine``,
            ``power_json``,
            ``energy_weight``, ``metric``, ``tie``, ``regions``, ``kernel``,
            ``source``, ``out``, ...).

    Raises:
        SystemExit: If no ``region_stats.csv`` is found.
    """
    paths = []
    for s in args.stats:
        if os.path.isdir(s):
            paths += glob.glob(os.path.join(s, "**", "region_stats.csv"), recursive=True)
        else:
            paths += glob.glob(s)
    if not paths:
        sys.exit("pgo_flow: no region_stats.csv found in %s" % args.stats)
    window = load_window(args.machine)
    power = power_model(window, args.power_json, args.energy_weight)
    agg = read_stats(paths)
    regions = sorted({r for r, _ in agg})
    out = {}
    for r in regions:
        cands = {}
        for (rr, c), a in agg.items():
            if rr != r or a["cycles"] <= 0:
                continue
            cyc = a["cycles"]
            energy = a["energy"] if a["energy"] is not None else power.get(c, 1.0) * cyc
            # Normalize per instruction so configs with slightly different
            # instruction counts per region (timing-dependent) compare fairly.
            insts = max(a["insts"], 1.0)
            d = cyc / insts
            e = energy / insts
            metric = {
                "ed2p": e * d * d,
                "edp": e * d,
                "ipc": -insts / cyc,
                "cycles": d,
                "energy": e,
            }[args.metric]
            cands[c] = {"metric": metric, "ipc": insts / cyc, "cycles": cyc, "insts": a["insts"]}
        if not cands:
            continue
        best_val = min(v["metric"] for v in cands.values())
        tol = abs(best_val) * args.tie
        # Ties (within --tie) go to the smallest configuration.
        best = min(c for c, v in cands.items() if v["metric"] <= best_val + tol)
        out[str(r)] = {
            "config": best,
            "W": window[best]["rob"] if best < len(window) else 0,
            "metric": args.metric,
            "candidates": {str(c): {"ipc": round(v["ipc"], 4), "cycles": v["cycles"]}
                           for c, v in sorted(cands.items())},
        }
    # Cross-check against the region table of the profiled build
    # (<kernel>.regions.json, docs/interfaces.md §5): annotate each entry with
    # its source location, and report regions that the profile never reached
    # (they get no hint, i.e. the default window, on the large input).
    unprofiled = []
    rpath = args.regions or default_regions_json(args)
    if rpath and os.path.exists(rpath):
        with open(rpath) as f:
            table = json.load(f).get("regions", {})
        for rid, info in table.items():
            if rid in out:
                out[rid]["function"] = info.get("function")
                out[rid]["line"] = info.get("line")
            else:
                unprofiled.append(int(rid))
        stray = sorted(int(r) for r in out if r not in table)
        if stray:
            print("pgo_flow: warning: profiled regions %s are not in %s (stale profile?)"
                  % (stray, rpath), file=sys.stderr)
        if unprofiled:
            print("pgo_flow: note: regions %s were not visited on the profiling input; "
                  "they keep the default window" % sorted(unprofiled), file=sys.stderr)
    else:
        rpath = None
    doc = {
        "kernel": args.kernel,
        "source": args.source,
        "metric": args.metric,
        "machine": args.machine,
        "stats": sorted(paths),
        "regions_table": rpath,
        "unprofiled_regions": sorted(unprofiled),
        "regions": out,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print("pgo_flow: %d regions -> %s" % (len(out), args.out))


def build_root():
    """Return the absolute build root (``$WINHINT_BUILD``, default ``<repo>/build``)."""
    return os.path.abspath(os.environ.get("WINHINT_BUILD", os.path.join(ROOT, "build")))


def default_regions_json(args):
    """Return the ``<kernel>.regions.json`` written next to the oracle binary (benchmarks/Makefile).

    Asks the Makefile for the oracle output directory (``make -s outdir``), so
    extra make variables (OPT=..., TILED=on) select the matching build; falls
    back to the default layout if make is unavailable.

    Args:
        args (argparse.Namespace): Parsed command-line namespace.

    Returns:
        (str): Path of the region table (it may not exist).
    """
    try:
        return binary_path(args, "oracle") + ".regions.json"
    except (OSError, subprocess.CalledProcessError, IndexError):
        return os.path.join(build_root(), "benchmarks", args.arch, "oracle",
                            args.kernel + ".regions.json")


def run_cmd(cmd, dry, **kw):
    """Print a command and run it unless ``dry`` is set.

    Args:
        cmd (list[str]): Command as an argument list.
        dry (bool): If true, only print the command.
        **kw (Any): Passed to ``subprocess.check_call``.

    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    print("pgo_flow: " + " ".join(cmd), flush=True)
    if not dry:
        subprocess.check_call(cmd, **kw)


def make_cmd(args, variant, extra=()):
    """Return the ``make`` command that builds ``variant`` of the kernel in benchmarks/.

    Args:
        args (argparse.Namespace): Parsed command-line namespace (``arch``, ``kernel``, ``machine``,
            ``make_vars``).
        variant (str): Benchmark variant (``VARIANT=``).
        extra (sequence of str): Additional make arguments inserted before ``args.make_vars``.

    Returns:
        (list[str]): The command as an argument list.
    """
    return (["make", "-C", os.path.join(ROOT, "benchmarks"), "-j1", "ARCH=" + args.arch,
             "VARIANT=" + variant, "KERNELS=" + args.kernel, "MACHINE=" + args.machine]
            + list(extra) + args.make_vars)


def binary_path(args, variant):
    """Return the path of the kernel binary of ``variant`` as reported by ``make -s outdir``.

    Runs make even in dry-run mode. This is side-effect free: the ``outdir`` target
    only echoes ``$(OUT_DIR)``, no rule can remake an included makefile, and the
    parse-time ``$(shell ...)`` calls only read files (``grep``, a JSON read). Running
    it keeps the dry-run path identical to the real one for any ``make_vars``.

    Args:
        args (argparse.Namespace): Parsed command-line namespace.
        variant (str): Benchmark variant.

    Returns:
        (str): ``<outdir>/<kernel>``.

    Raises:
        subprocess.CalledProcessError: If make fails.
    """
    out = subprocess.check_output(make_cmd(args, variant, ["-s", "outdir"]), text=True)
    return os.path.join(out.strip().splitlines()[-1], args.kernel)


def runs_dir(args):
    """Return ``results/pgo/runs/<kernel>/<input>``, the profiling run directory."""
    return os.path.join(ROOT, "results", "pgo", "runs", args.kernel, args.input)


def profile(args):
    """Build the oracle variant and run it in gem5 once per window configuration (``profile`` step).

    Each configuration ``c`` runs with ``--window-policy static --window-initial c``
    into ``runs_dir/cfg<c>``, under ``flock $WINHINT_BUILD/.heavy.lock``; finished
    runs are skipped unless ``--force``. Sets ``args.stats`` to the run directory
    when no ``--stats`` was given.

    Args:
        args (argparse.Namespace): Parsed command-line namespace.

    Raises:
        SystemExit: If ``--arch`` is not riscv, gem5 is missing or a run produced no
            ``region_stats.csv``.
    """
    if args.arch != "riscv":
        sys.exit("pgo_flow: profiling runs in gem5, which needs ARCH=riscv")
    run_cmd(make_cmd(args, "oracle"), args.dry_run)
    binary = binary_path(args, "oracle")
    gem5 = args.gem5 or os.environ.get(
        "WINHINT_GEM5", os.path.join(build_root(), "gem5", "src", "build", "RISCV_winhint", "gem5.opt"))
    if not args.dry_run and not os.access(gem5, os.X_OK):
        sys.exit("pgo_flow: gem5 not found at %s (tooling/winhint.sh gem5:build winhint)" % gem5)
    lock = os.path.join(build_root(), ".heavy.lock")
    window = load_window(args.machine)
    configs = args.configs if args.configs else list(range(len(window)))
    for c in configs:
        outdir = os.path.join(runs_dir(args), "cfg%d" % c)
        if os.path.exists(os.path.join(outdir, "region_stats.csv")) and not args.force:
            print("pgo_flow: cfg%d done (%s); --force to rerun" % (c, outdir))
            continue
        if not args.dry_run:
            os.makedirs(outdir, exist_ok=True)
        cmd = ["flock", lock, gem5, "--outdir=" + outdir, os.path.join(ROOT, "sim", "se.py"),
               "--machine", args.machine, "--cmd", binary, "--options", args.input,
               "--window-policy", "static", "--window-initial", str(c)]
        run_cmd(cmd, args.dry_run)
        if not args.dry_run and not os.path.exists(os.path.join(outdir, "region_stats.csv")):
            sys.exit("pgo_flow: %s has no region_stats.csv (region markers missing, or the "
                     "gem5 build lacks the WinHint patch)" % outdir)
    if not args.stats:
        args.stats = [runs_dir(args)]


def build(args):
    """Build the ``pgo`` variant, ``PGO_DIR`` = directory of ``--out`` (``build`` step).

    Args:
        args (argparse.Namespace): Parsed command-line namespace.
    """
    pgo_dir = os.path.dirname(os.path.abspath(args.out))
    run_cmd(make_cmd(args, "pgo", ["PGO_DIR=" + pgo_dir]), args.dry_run)


def main():
    """Parse the command line and run the requested step(s)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["profile", "select", "build", "all"])
    p.add_argument("--kernel", required=True)
    p.add_argument("--stats", nargs="*", default=[], help="region_stats.csv files, globs or run dirs")
    p.add_argument("--machine", default=os.path.join(ROOT, "sim", "machines", "riscv_ooo.json"))
    p.add_argument("--metric", default="ed2p", choices=["ed2p", "edp", "ipc", "cycles", "energy"])
    p.add_argument("--regions", default=None,
                   help="<kernel>.regions.json of the profiled build (default: the oracle build's)")
    p.add_argument("--power-json", default=None)
    p.add_argument("--energy-weight", type=float, default=0.15)
    p.add_argument("--tie", type=float, default=0.01, help="relative tie margin (prefer smaller)")
    p.add_argument("--source", default=None, help="input the profile was taken on (default: --input)")
    p.add_argument("--out", default=None)
    p.add_argument("--arch", default="riscv")
    p.add_argument("--input", default="small", help="input size for the profiling runs")
    p.add_argument("--configs", type=int, nargs="*", default=None, help="config indices (default: all)")
    p.add_argument("--gem5", default=None, help="gem5.opt (default: RISCV_winhint build)")
    p.add_argument("--force", action="store_true", help="rerun finished profiling runs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("make_vars", nargs="*", help="extra make variables (OPT=O3 ...)")
    args = p.parse_args()
    if args.out is None:
        args.out = os.path.join(ROOT, "results", "pgo", args.kernel + ".json")
    args.machine = os.path.abspath(args.machine)
    args.source = args.source or args.input
    if args.command in ("profile", "all"):
        profile(args)
    if args.command == "select" and not args.stats:
        args.stats = [runs_dir(args)]
    if args.command in ("select", "all"):
        if args.dry_run:
            print("pgo_flow: select %s -> %s" % (args.stats, args.out))
        else:
            select(args)
    if args.command in ("build", "all"):
        build(args)


if __name__ == "__main__":
    main()
