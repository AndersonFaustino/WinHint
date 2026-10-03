#!/usr/bin/env python3
"""B1 oracle: best window configuration per code region (oracle_sweep.py).

Runs a kernel built in the ``oracle`` variant (region(id) markers only,
interfaces.md §5) under EVERY static window configuration of a machine
(window_policy=static, window_initial=i), reads each run's
region_stats.csv (one row per region visit) and picks, per region, the
configuration that maximises IPC and the one that minimises ED²P.

Per-region energy for ED²P:

    proxy (default)  E_r,i = cycles_r,i / f * P_core * (1 - w + w * rob_i/rob_max)
                             + insts_r * E_inst
                     (sim/estimate_energy.py proxy constants; w = 0.15)
    mcpat            the whole-run average power of configuration i from McPAT
                     (sim/estimate_energy.py) times the region's time,
                     E_r,i = P_i * cycles_r,i / f (assumes uniform power inside a
                     run - documented approximation; McPAT has no per-region stats)

ED²P_r,i = E_r,i * (cycles_r,i / f)^2.

Outputs (machine M, input I, kernel K):

    results/oracle/M/I/K.json         flat {region: config} by --primary metric
                                      (default ipc) - the oracle_hinted input
    results/oracle/M/I/K.ed2p.json    flat {region: config} by ED²P
    results/oracle/M/I/K.ipc.json     flat {region: config} by IPC
    results/oracle/M/I/K.table.csv    per (region, config): cycles, insts, ipc,
                                      energy, ed2p, best flags
    results/oracle/M/I/K.summary.json metadata + both maps + whole-program
                                      oracle vs. best static totals
    results/oracle/K.json             copy of the flat map for the default
                                      machine/input (riscv_ooo, large): the
                                      path named in interfaces.md §5 and read by
                                      the Makefile's oracle_hinted variant
    results/pgo/K.json                copy for riscv_ooo + input small: the B7
                                      profile (best config per region measured on
                                      the training input; Makefile PGO_DIR);
                                      with --out-root R it goes to R/../pgo/K.json

Map values are config indices (the compiler's from-json mode maps an integer
to the ROB size of that configuration).
gem5 runs go to ``results/oracle/runs/M/K/I/c<i>/`` (resumable; --force reruns).
--trace (default on) also records window_trace.csv in every run: these
traces are the B5 training data (sim/baselines/lut/label_phases.py).

Every gem5 run is wrapped in ``flock $WINHINT_BUILD/.heavy.lock`` (shared
with sim/run_experiments.py; --no-lock for tiny runs).

Run length: the same sim/run_lengths.json plans as sim/run_experiments.py
(``large`` = region-aligned samples from the profile of the oracle build; the
profile and checkpoint passes are shared with the campaign), so the oracle
compares configurations on exactly the windows the campaign simulates. Each
config dir then holds the merged region_stats.csv (measured rows only) and
window_trace.csv. ``--run-length full`` simulates whole programs.

Usage (host, `winhint` env active):

    python sim/baselines/oracle/oracle_sweep.py --kernels encoder_bert_tiny_infer --input small
    python sim/baselines/oracle/oracle_sweep.py --kernels all --dry-run
    python sim/baselines/oracle/oracle_sweep.py --kernels k --analyze-only   # re-derive maps
    python sim/baselines/oracle/oracle_sweep.py --kernels k --energy mcpat   # ED²P from McPAT
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "sim"))
sys.path.insert(0, str(REPO / "sim" / "baselines" / "lut"))
import run_experiments as rx  # noqa: E402
from estimate_energy import _clock_mhz, estimate_run, proxy_region_energy  # noqa: E402
from whdata import aggregate_regions, load_machine, load_region_stats, write_flat_oracle  # noqa: E402

#: Default output root of the oracle maps and runs (``--out-root``).
ORACLE_ROOT = REPO / "results" / "oracle"
#: Name of the B7 PGO profile directory, a sibling of the oracle root
#: (``results/pgo`` for the default ``--out-root``; see ``pgo_root``).
PGO_DIRNAME = "pgo"
#: Machine/input whose primary map is also copied to ``<out-root>/<kernel>.json``.
DEFAULT_MACHINE = "riscv_ooo"
DEFAULT_INPUT = "large"


def pgo_root(root: Path) -> Path:
    """Return the B7 profile directory for oracle root ``root`` (``<root>/../pgo``)."""
    return root.parent / PGO_DIRNAME


def run_dirs(root: Path, machine: str, kernel: str, input_size: str, n: int) -> list[Path]:
    """Return the gem5 run directories of one kernel, one per static config.

    Args:
        root: Oracle output root.
        machine: Machine name.
        kernel: Kernel name.
        input_size: Input size.
        n: Number of configs.

    Returns:
        ``<root>/runs/<machine>/<kernel>/<input_size>/c<i>`` for ``i`` in
        ``range(n)``.
    """
    return [root / "runs" / machine / kernel / input_size / f"c{i}" for i in range(n)]


def analyze(kernel: str, machine: dict, dirs: list[Path], energy: str = "proxy",
            tie_tol: float = 0.0) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Return (table, best_ipc, best_ed2p, totals). ``dirs[i]`` is config i.

    Reads ``region_stats.csv`` of every config directory that has one,
    aggregates visits per region and computes per-region energy (proxy or
    McPAT, see the module docstring) and ED²P. The best IPC config is the
    highest IPC, with configs within ``tie_tol`` (relative) of the maximum
    resolved to the smallest index; the best ED²P config is the lowest ED²P
    (ties -> smallest index).

    Args:
        kernel: Kernel name (for error messages).
        machine: Machine dict from ``whdata.load_machine``.
        dirs: Run directory per config index.
        energy: ``"proxy"`` or ``"mcpat"``.
        tie_tol: Relative IPC tolerance for the tie rule.

    Returns:
        ``(table, best_ipc, best_ed2p, totals)``: one row per (region, config)
        with ``rob``, ``visits``, ``cycles``, ``insts``, ``ipc``, ``energy``,
        ``ed2p``, ``best_ipc`` and ``best_ed2p``; the two ``{region: config}``
        maps; and totals with per-config ``static`` cycles/ED²P sums, oracle
        cycles, ``complete_configs`` (configs with a row for every region of
        the table) and, when that list is non-empty, ``best_static_config``
        (fewest cycles among them) and ``oracle_speedup_vs_best_static``. A
        config missing regions is never the best static one, since its cycle
        sum would leave out work.

    Raises:
        RuntimeError: If no ``region_stats.csv`` was found.
    """
    table_cfg = machine["window_table"]
    f_hz = _clock_mhz(machine) * 1e6
    rows = []
    power = {}
    for i, d in enumerate(dirs):
        rs = d / "region_stats.csv"
        if not rs.exists():
            continue
        agg = aggregate_regions(load_region_stats(rs))
        if energy == "mcpat":
            e = estimate_run(d, machine, "mcpat", static_config=i)
            power[i] = e["power_w"]
        for r in agg.itertuples():
            t = r.cycles / f_hz
            if energy == "mcpat":
                en = power[i] * t
            else:
                en = proxy_region_energy(r.cycles, r.insts, i, table_cfg, f_hz)
            rows.append({"region": int(r.region), "config": i, "rob": table_cfg[i]["rob"],
                         "visits": int(r.visits), "cycles": float(r.cycles),
                         "insts": float(r.insts), "ipc": float(r.ipc),
                         "energy": en, "ed2p": en * t * t})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"{kernel}: no region_stats.csv found in {dirs[0].parent}")
    best_ipc, best_ed2p = {}, {}
    df["best_ipc"] = False
    df["best_ed2p"] = False
    for reg, g in df.groupby("region"):
        # IPC: highest; ties within tie_tol (relative) -> smallest window
        top = g["ipc"].max()
        cand = g[g["ipc"] >= top * (1 - tie_tol)].sort_values("config")
        bi = int(cand.iloc[0]["config"])
        be = int(g.sort_values(["ed2p", "config"]).iloc[0]["config"])
        best_ipc[int(reg)], best_ed2p[int(reg)] = bi, be
        df.loc[(df.region == reg) & (df.config == bi), "best_ipc"] = True
        df.loc[(df.region == reg) & (df.config == be), "best_ed2p"] = True
    # whole-program totals: oracle (per-region best) vs every static config
    tot = {"static": {}}
    for c, g in df.groupby("config"):
        tot["static"][int(c)] = {"cycles": float(g.cycles.sum()), "ed2p_sum": float(g.ed2p.sum())}
    tot["oracle_ipc_cycles"] = float(df[df.best_ipc].cycles.sum())
    tot["oracle_ed2p_cycles"] = float(df[df.best_ed2p].cycles.sum())
    regions = set(df.region)
    full = [int(c) for c, g in df.groupby("config") if set(g.region) == regions]
    tot["complete_configs"] = full
    if full:
        bs = min(full, key=lambda c: tot["static"][c]["cycles"])
        tot["best_static_config"] = bs
        tot["oracle_speedup_vs_best_static"] = (tot["static"][bs]["cycles"]
                                                / max(tot["oracle_ipc_cycles"], 1.0))
    return df.sort_values(["region", "config"]).reset_index(drop=True), best_ipc, best_ed2p, tot


def write_outputs(root: Path, kernel: str, mname: str, input_size: str, df: pd.DataFrame,
                  best_ipc: dict, best_ed2p: dict, tot: dict, primary: str, meta: dict,
                  pgo_copy: bool = True) -> Path:
    """Write the oracle maps, per-(region, config) table and summary of one kernel.

    Writes ``<kernel>.json`` (``primary`` map), ``<kernel>.ipc.json``,
    ``<kernel>.ed2p.json``, ``<kernel>.table.csv`` and ``<kernel>.summary.json``
    under ``<root>/<mname>/<input_size>/``. For the default machine and input
    the primary map is also copied to ``<root>/<kernel>.json``; for the default
    machine with input ``small`` (and ``pgo_copy``) it is copied to
    ``pgo_root(root)/<kernel>.json`` (``results/pgo`` for the default root).

    Args:
        root: Oracle output root.
        kernel: Kernel name.
        mname: Machine name.
        input_size: Input size.
        df: Table returned by ``analyze``.
        best_ipc: Best-IPC config per region.
        best_ed2p: Best-ED²P config per region.
        tot: Whole-program totals returned by ``analyze``.
        primary: ``"ipc"`` or ``"ed2p"``; selects the map of ``<kernel>.json``.
        meta: Extra keys merged into the summary.
        pgo_copy: Allow the B7 profile copy.

    Returns:
        Path of the primary flat map.
    """
    out = root / mname / input_size
    out.mkdir(parents=True, exist_ok=True)
    primary_map = best_ipc if primary == "ipc" else best_ed2p
    write_flat_oracle(out / f"{kernel}.json", primary_map)
    write_flat_oracle(out / f"{kernel}.ipc.json", best_ipc)
    write_flat_oracle(out / f"{kernel}.ed2p.json", best_ed2p)
    df.to_csv(out / f"{kernel}.table.csv", index=False)
    summary = {**meta, "kernel": kernel, "machine": mname, "input": input_size,
               "primary": primary,
               "best": {"ipc": {str(k): v for k, v in sorted(best_ipc.items())},
                        "ed2p": {str(k): v for k, v in sorted(best_ed2p.items())}},
               "totals": tot}
    (out / f"{kernel}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if mname == DEFAULT_MACHINE and input_size == DEFAULT_INPUT:
        shutil.copyfile(out / f"{kernel}.json", root / f"{kernel}.json")
    if pgo_copy and mname == DEFAULT_MACHINE and input_size == "small":
        pgo = pgo_root(root)
        pgo.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / f"{kernel}.json", pgo / f"{kernel}.json")
    return out / f"{kernel}.json"


def parse_args(argv=None):
    """Parse the command line.

    Args:
        argv (list[str] | None): Argument list; ``None`` reads ``sys.argv``.

    Returns:
        args (argparse.Namespace): The parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernels", nargs="+", required=True, help="kernel names or 'all'")
    p.add_argument("--machine", type=Path, default=REPO / "sim" / "machines" / f"{DEFAULT_MACHINE}.json")
    p.add_argument("--input", default=DEFAULT_INPUT, choices=["small", "large"])
    p.add_argument("--configs", default=None, help="comma list of config indices (default: all)")
    p.add_argument("--jobs", type=int, default=1, help=f"parallel gem5 runs (max {rx.MAX_JOBS})")
    p.add_argument("--primary", default="ipc", choices=["ipc", "ed2p"],
                   help="metric of the flat <kernel>.json map")
    p.add_argument("--energy", default="proxy", choices=["proxy", "mcpat"])
    p.add_argument("--tie-tolerance", type=float, default=0.0,
                   help="relative IPC tolerance within which the smaller window wins")
    p.add_argument("--no-trace", dest="trace", action="store_false",
                   help="do not record window_trace.csv (no B5 training data)")
    p.add_argument("--period", type=int, default=None)
    p.add_argument("--max-insts", type=int, default=None)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--gem5", default=rx.GEM5_BIN)
    p.add_argument("--bin-root", type=Path, default=rx.BIN_ROOT)
    p.add_argument("--variant", default="oracle",
                   help="binary variant directory incl. build suffix (default oracle, e.g. oracle-O3)")
    p.add_argument("--no-pgo-copy", dest="pgo_copy", action="store_false",
                   help="with --input small on the default machine, do not copy the map to "
                        "<out-root>/../pgo/<kernel>.json, i.e. results/pgo/<kernel>.json by "
                        "default (the B7 profile, Makefile PGO_DIR)")
    p.add_argument("--out-root", type=Path, default=ORACLE_ROOT)
    p.add_argument("--no-lock", action="store_true",
                   help=f"do not wrap gem5 in flock {rx.HEAVY_LOCK}")
    p.add_argument("--run-lengths", type=Path, default=rx.RUN_LENGTHS,
                   help="run-length config (shared with run_experiments.py)")
    p.add_argument("--run-length", default="config", choices=["config", "full"])
    p.add_argument("--runlen-root", type=Path, default=rx.RUNLEN_ROOT)
    p.add_argument("--ckpt-root", type=Path, default=rx.CKPT_ROOT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--analyze-only", action="store_true", help="skip gem5, re-derive maps")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Run (or list) the static-config sweep and derive the oracle maps.

    With ``--dry-run`` only prints the profile/checkpoint jobs and the status
    and command of every run. Unless ``--analyze-only``, runs the gem5 jobs via
    ``run_experiments.run_phases``; then analyses every kernel and writes its
    outputs.

    Args:
        argv (list[str] | None): Argument list; ``None`` reads ``sys.argv``.

    Returns:
        0 on success, 1 if any kernel could not be analysed.
    """
    a = parse_args(argv)
    a.jobs = max(1, min(a.jobs, rx.MAX_JOBS))
    # gem5 runs with cwd = repo root: make every path absolute
    a.machine, a.out_root, a.bin_root = a.machine.resolve(), a.out_root.resolve(), a.bin_root.resolve()
    machine = load_machine(a.machine)
    mname = a.machine.stem
    machine["name"] = mname
    n = len(machine["window_table"])
    cfgs = [int(c) for c in a.configs.split(",")] if a.configs else list(range(n))
    kernels = rx.all_kernels() if a.kernels == ["all"] else a.kernels

    a.runlen_root, a.ckpt_root = a.runlen_root.resolve(), a.ckpt_root.resolve()

    def build() -> list:
        """Return the ``run_experiments.Run`` jobs (kernels x configs) with run lengths applied."""
        jobs = []
        for k in kernels:
            binary = rx.find_binary(a.bin_root, a.variant, mname, k)
            dirs = run_dirs(a.out_root, mname, k, a.input, n)
            for i in cfgs:
                kw = dict(gem5=a.gem5, machine_json=a.machine,
                          binary=binary or f"<missing:{a.variant}/{k}>", input_size=a.input,
                          policy="static", initial=i, period=a.period, lut=None,
                          trace=a.trace, max_insts=a.max_insts, extra=None)
                run = rx.Run(mname, a.machine, k, rx.Variant(f"c{i}", "B1", a.variant, "static", i),
                             a.input, dirs[i], binary, None, rx.gem5_cmd(outdir=dirs[i], **kw),
                             [] if binary else [f"binary {a.bin_root}/{a.variant}/[{mname}/]{k}"],
                             None if a.no_lock else rx.HEAVY_LOCK, cmd_kw=kw,
                             base_machine_json=a.machine)
                jobs.append(run)
        # the oracle binary carries the region markers itself: it is its own profile build
        rx.apply_run_length(jobs, a, profile_variant=a.variant)
        return jobs

    if a.dry_run:
        jobs = build()
        for kind in ("profile", "checkpoint"):
            for j in rx.aux_jobs(jobs, kind):
                if not j.done():
                    print(f"{kind:<10} {j.product}  " + " ".join(j.cmd))
        for r in jobs:
            st = ("done" if rx.is_done(r.outdir) else "MISSING" if r.missing
                  else "pending" if r.pending else "todo")
            det = ("; ".join(r.missing) if r.missing else f"after phase P ({r.pending})"
                   if r.pending else " ".join(r.cmd))
            rlen = (f"{r.mode}:{len(r.samples)}x" if r.samples else r.mode)
            print(f"{r.kernel:<30} {r.variant.name:<4} {st:<8} {rlen:<12} {det}")
        return 0

    if not a.analyze_only:
        a.jobs = max(1, min(a.jobs, rx.MAX_JOBS))
        rx.run_phases(build, a, label=lambda r: f"{r.kernel} {r.variant.name}")

    rc = 0
    for k in kernels:
        dirs = run_dirs(a.out_root, mname, k, a.input, n)
        try:
            df, bi, be, tot = analyze(k, machine, dirs, a.energy, a.tie_tolerance)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {k}: {exc}")
            rc = 1
            continue
        meta = {"configs_run": sorted(int(c) for c in df.config.unique()),
                "window_table": machine["window_table"], "energy_model": a.energy,
                "tie_tolerance": a.tie_tolerance}
        path = write_outputs(a.out_root, k, mname, a.input, df, bi, be, tot, a.primary, meta,
                             a.pgo_copy)
        print(f"[OK] {k}: {len(bi)} regions -> {path}  "
              f"(ipc map {bi}, ed2p map {be})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
