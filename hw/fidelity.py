#!/usr/bin/env python3
"""Real-hardware baseline fidelity (PROPOSAL §7) and µarch-parameter probes (§8).

Two modes of hw/run_hw_experiments.py (no campaign, no system changes, no root):

      --uarch     run hw/tools/uarch_probe (ROB / load buffer / store buffer / MLP) on one
                  P-core and one E-core CPU -> <out>/uarch/uarch_{P,E}.json, then compare the
                  knees with the documented values (DOCUMENTED, docs/guide/hardware/uarch-params.md)
                  -> <out>/uarch/uarch_check.json.  --uarch-smoke: tiny ranges, functional only.
      --fidelity  microbenchmark-driven trend checks of the reimplemented baselines
                  -> <out>/fidelity/fidelity.json (+ fidelity.csv), one row per predicate:
         ground truth  phase_workload {ilp, compute, memory} pinned to one P and one E CPU:
                       P/E speedup = work rate on P / work rate on E
         R4 (PIE, Van Craeynest et al., ISCA'12): memory-intensive code gains less from the big
                       core, PIE predicts it and places it on the small core; compute-intensive
                       (high-ILP) code goes to the big core. pie_daemon on ilp, memory and the
                       alternating workload.
         R5 (Sondag & Rajan, CGO'11): each region type is assigned to the core type that runs it
                       better. libwinhint WINHINT_MODE=sondag on phase_workload_hinted alt-ilp
                       (region 2 = ilp, region 1 = memory), threshold placed between the two
                       ground-truth speedups (geometric mean), so the expected assignment is
                       ilp -> P, memory -> E.

Every predicate is a pure function of the parsed outputs (unit-tested in hw/tests). A
predicate of kind "trend" is the paper's qualitative claim and must hold before the
baseline is used; kind "info" is reported only. The numbers are functional/fidelity
checks, not paper results.
"""
import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

#: Directory of this file (``hw/``).
HW = Path(__file__).resolve().parent
#: Repository root (``$WINHINT_ROOT``, default the parent of ``hw/``).
ROOT = Path(os.environ.get("WINHINT_ROOT") or HW.parent)
#: Build root (``$WINHINT_BUILD``, default ``<root>/build``).
BUILD = Path(os.environ.get("WINHINT_BUILD") or ROOT / "build")

# ----------------------------------------------------------------------------- µarch
# Intel Core 5 120U = Raptor Lake-U refresh (family 6, model 186): Raptor Cove P-cores
# (Golden Cove core) + Gracemont E-cores. Sources and confidence: docs/guide/hardware/uarch-params.md.
#: Documented µarch sizes per core type; None = no documented value we could verify
#: ("to confirm" by the probe).
DOCUMENTED = {
    "P": {"rob": 512, "lb": 192, "sb": 114, "mlp": 16},    # Intel ORM, Golden Cove tables
    "E": {"rob": 256, "lb": None, "sb": None, "mlp": None},  # ROB from Intel's Gracemont disclosure
}
#: uarch_probe arguments of --uarch-smoke (tiny ranges, functional check only).
UARCH_SMOKE = ["-p", "rob,lb,sb,mlp", "-n", "0:16:8", "-k", "2", "-i", "200", "-m", "16", "-r", "1"]
#: uarch_probe arguments of a full --uarch run.
UARCH_FULL = ["-p", "rob,lb,sb,mlp", "-i", "20000", "-m", "128", "-r", "5"]
#: Probe step (entries) per buffer probe; lower bound of the knee tolerance.
STEP = {"rob": 8, "lb": 4, "sb": 2}


def confirm_uarch(probe, documented, measured, kmax=12):
    """Compare one probe result with a documented value.

    For ``rob``/``lb``/``sb`` the knee ``[size_lo, size_hi]`` must contain the
    documented value within a tolerance of ``max(STEP[probe], 10 %)``. For ``mlp``
    the effective MLP must be within 25 % of the documented value; if the
    documented value exceeds the largest probed ``k``, an effective MLP of at
    least ``0.75 * k`` counts as ``consistent_bound``.

    Args:
        probe (str): Probe name (``rob``, ``lb``, ``sb`` or ``mlp``).
        documented (int | None): Documented size/MLP, or ``None`` if unknown.
        measured (dict | None): The probe's entry of the uarch_probe JSON ``probes`` object, or
            ``None``.
        kmax (int): Assumed largest MLP chain count when the result has no ``x`` values.

    Returns:
        (tuple[str, str]): ``(status, detail)``; status is one of ``confirmed``, ``differs``,
            ``no_knee``, ``undocumented``, ``consistent_bound`` or ``missing``, detail a
            human-readable explanation.
    """
    if measured is None:
        return "missing", "probe not run"
    if probe == "mlp":
        eff = measured.get("mlp_effective")
        if eff is None:
            return "missing", "no mlp_effective"
        if documented is None:
            return "undocumented", f"effective MLP {eff:.1f} (k <= {len(measured.get('x', [])) or kmax})"
        k = max(measured.get("x") or [kmax])
        if documented > k:
            ok = eff >= 0.75 * k
            return ("consistent_bound" if ok else "differs"), \
                f"effective MLP {eff:.1f} with k <= {k}; documented {documented} > k"
        ok = abs(eff - documented) <= 0.25 * documented
        return ("confirmed" if ok else "differs"), f"effective MLP {eff:.1f} vs documented {documented}"
    lo, hi = measured.get("size_lo"), measured.get("size_hi")
    if lo is None or hi is None:
        xs = measured.get("x") or [0]
        return "no_knee", f"no knee for N in [{min(xs)}, {max(xs)}]"
    if documented is None:
        return "undocumented", f"knee: {lo}-{hi} entries"
    tol = max(STEP.get(probe, 1), 0.10 * documented)
    ok = lo - tol <= documented <= hi + tol
    return ("confirmed" if ok else "differs"), f"knee {lo}-{hi} entries vs documented {documented} (tol {tol:.0f})"


def uarch_commands(a, topo):
    """Return the ``uarch_probe`` command lines for the first P-core and first E-core CPU.

    Args:
        a (argparse.Namespace): Parsed namespace of run_hw_experiments.py (``tool_dir``, ``out``,
            optional ``uarch_smoke``).
        topo (dict): Topology dict from ``run_hw_experiments.topology()`` (``pcpus``, ``ecpus``).

    Returns:
        (dict[str, list[str]]): ``{"P": argv, "E": argv}``; each writes
            ``<out>/uarch/uarch_<side>.json``.
    """
    tool = Path(a.tool_dir) / "uarch_probe"
    out = Path(a.out) / "uarch"
    p1 = topo["pcpus"].split(",")[0].split("-")[0]
    e1 = topo["ecpus"].split(",")[0].split("-")[0]
    flags = UARCH_SMOKE if getattr(a, "uarch_smoke", False) else UARCH_FULL
    return {side: [str(tool), "-c", cpu] + flags + ["-o", str(out / f"uarch_{side}.json")]
            for side, cpu in (("P", p1), ("E", e1))}


def run_uarch(a):
    """Run the µarch probes on one P and one E CPU and check them (``--uarch``).

    The knees are compared with :data:`DOCUMENTED` (see :func:`confirm_uarch`).

    Writes ``<out>/uarch/uarch_{P,E}.{csv,json}`` and ``uarch_check.json``. With
    ``--dry-run`` only prints the commands.

    Args:
        a (argparse.Namespace): Parsed namespace of run_hw_experiments.py.

    Raises:
        SystemExit: If the machine is not hybrid or ``uarch_probe`` is not built.
        subprocess.CalledProcessError: If a probe fails.
    """
    import run_hw_experiments as rhe
    topo = rhe.topology()
    rhe.require_hybrid(topo)
    cmds = uarch_commands(a, topo)
    out = Path(a.out) / "uarch"
    if a.dry_run:
        for side, c in cmds.items():
            print(f"[dry-run] uarch {side}: {' '.join(c)}")
        return
    if not Path(cmds["P"][0]).exists():
        sys.exit(f"{cmds['P'][0]} missing: make -C {HW} -j1")
    out.mkdir(parents=True, exist_ok=True)
    report = {"model": topo["model"], "smoke": bool(getattr(a, "uarch_smoke", False)), "sides": {}}
    for side, c in cmds.items():
        with open(out / f"uarch_{side}.csv", "w") as f:
            subprocess.run(c, stdout=f, check=True)
        m = json.loads(Path(c[-1]).read_text())
        rows = {}
        for probe in ("rob", "lb", "sb", "mlp"):
            st, det = confirm_uarch(probe, DOCUMENTED[side].get(probe), m["probes"].get(probe))
            rows[probe] = {"documented": DOCUMENTED[side].get(probe), "status": st, "detail": det}
            print(f"uarch {side} cpu {m['cpu']} {probe:4s}: {st:16s} {det}")
        report["sides"][side] = {"cpu": m["cpu"], "probes": rows}
    (out / "uarch_check.json").write_text(json.dumps(report, indent=1) + "\n")
    if report["smoke"]:
        print("note: --uarch-smoke ranges are too small to find any knee; functional check only")
    print(f"-> {out / 'uarch_check.json'}")


# ----------------------------------------------------------------------------- fidelity
#: Ground-truth workload classes of phase_workload.
CLASSES = ("ilp", "compute", "memory")
#: Output key with the step count per class.
STEPS_KEY = {"ilp": "ilp_steps", "compute": "fp_steps", "memory": "mem_steps"}
#: Output key with the elapsed ns per class.
NS_KEY = {"ilp": "ns_ilp", "compute": "ns_compute", "memory": "ns_memory"}
#: Region id per class in phase_workload_hinted.
REGION = {"compute": 0, "memory": 1, "ilp": 2}   # phase_workload_hinted region ids
#: Pointer-chase footprint in MB.
FID_MB = 32          # pointer-chase footprint (> 12 MB LLC; its set-up is part of each PIE run,
                     # so keep --fidelity-secs >> the set-up time, ~0.2 s)
#: Phase length in ms passed to phase_workload.
FID_PHASE_MS = 50


def parse_workload(stdout):
    """Parse a ``phase_workload`` output line into numbers.

    Args:
        stdout (str): Program output with whitespace-separated ``key=value`` tokens.

    Returns:
        (dict[str, float]): Dict of the tokens whose value parses as a float.
    """
    d = {}
    for tok in stdout.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                d[k] = float(v)
            except ValueError:
                pass
    return d


def work_rate(d, cls):
    """Return the work rate (steps per ns) of a workload class.

    Args:
        d (dict[str, float]): Parsed workload output (see :func:`parse_workload`).
        cls (str): Workload class (``ilp``, ``compute`` or ``memory``).

    Returns:
        (float | None): ``steps / ns`` from the class's keys in :data:`STEPS_KEY` and
            :data:`NS_KEY`,
            or ``None`` if either is missing or zero.
    """
    steps, ns = d.get(STEPS_KEY[cls]), d.get(NS_KEY[cls])
    return steps / ns if steps and ns else None


def speedup(rates_p, rates_e):
    """Return the median rate on P divided by the median rate on E.

    Args:
        rates_p (list[float | None]): Work rates measured on the P-core (falsy entries ignored).
        rates_e (list[float | None]): Work rates measured on the E-core (falsy entries ignored).

    Returns:
        (float | None): The P/E speedup, or ``None`` if either side has no rate.
    """
    rp = [r for r in rates_p if r]
    re_ = [r for r in rates_e if r]
    if not rp or not re_:
        return None
    return statistics.median(rp) / statistics.median(re_)


def workload_argv(tool_dir, cls, secs, hinted=False):
    """Return the ``phase_workload`` command line for one workload class.

    Args:
        tool_dir (str | Path): Directory with the hw tools.
        cls (str): Workload mode (e.g. ``ilp``, ``memory``, ``alt``, ``alt-ilp``).
        secs (float): Run time in seconds.
        hinted (bool): Use ``phase_workload_hinted`` (region markers) instead.

    Returns:
        (list[str]): Argument list ``[binary, secs, FID_PHASE_MS, FID_MB, cls]``.
    """
    b = "phase_workload_hinted" if hinted else "phase_workload"
    return [str(Path(tool_dir) / b), str(secs), str(FID_PHASE_MS), str(FID_MB), cls]


def pie_argv(a, rundir, prog):
    """Return the ``pie_daemon`` command line that runs and steers ``prog``.

    Args:
        a (argparse.Namespace): Parsed namespace (``tool_dir``, ``pie_interval_ms``, ``pie_slack``,
            ``pie_hyst``).
        rundir (Path): Directory receiving ``pie.csv`` and ``pie.json``.
        prog (list[str]): Command line of the workload.

    Returns:
        (list[str]): The argument list (perf backend, initial side P).
    """
    return [str(Path(a.tool_dir) / "pie_daemon"), "-b", "perf", "-i", str(a.pie_interval_ms),
            "-s", str(a.pie_slack), "-H", str(a.pie_hyst), "-I", "P",
            "-o", str(rundir / "pie.csv"), "-S", str(rundir / "pie.json"), "--"] + prog


def sondag_env(base_env, log, threshold, k):
    """Return the environment that runs a hinted binary under libwinhint's sondag mode.

    All ``WINHINT_*`` variables of ``base_env`` except ``WINHINT_PCPUS`` and
    ``WINHINT_ECPUS`` are removed before the sondag settings are added.

    Args:
        base_env (dict[str, str]): Environment to start from.
        log (str | Path): libwinhint log path (``WINHINT_LOG``).
        threshold (float): E/P ratio threshold (``WINHINT_SONDAG_THRESHOLD``).
        k (int): Samples per region type and core type (``WINHINT_SONDAG_K``).

    Returns:
        (dict[str, str]): The new environment dict.
    """
    env = {k_: v for k_, v in base_env.items() if not k_.startswith("WINHINT_")
           or k_ in ("WINHINT_PCPUS", "WINHINT_ECPUS")}
    env.update(WINHINT_MODE="sondag", WINHINT_LOG=str(log), WINHINT_PERF="1",
               WINHINT_REQUIRE_HYBRID="1", WINHINT_SONDAG_K=str(k),
               WINHINT_SONDAG_THRESHOLD=f"{threshold:.4f}")
    return env


def separating_threshold(s_ilp, s_mem, fallback):
    """Return the R5 assignment threshold separating the ilp and memory ground truths.

    The threshold is the geometric mean of the two ground-truth P/E speedups when
    they are ordered as the papers assume (ilp gains more); otherwise the fallback.

    Args:
        s_ilp (float | None): Ground-truth P/E speedup of the ilp class, or ``None``.
        s_mem (float | None): Ground-truth P/E speedup of the memory class, or ``None``.
        fallback (float): Threshold used when the speedups do not separate.

    Returns:
        (tuple[float, bool]): ``(threshold, separating)`` where ``separating`` tells whether the
            geometric mean was used.
    """
    if s_ilp and s_mem and s_ilp > s_mem:
        return math.sqrt(s_ilp * s_mem), True
    return fallback, False


def parse_pie_csv(path):
    """Read the per-interval rows of a ``pie_daemon`` CSV.

    Truncated or malformed rows (killed run) and an unreadable file are ignored.

    Args:
        path (str | Path): Path of ``pie.csv``.

    Returns:
        (list[dict]): List of ``{"side", "decision", "mpki", "ratio"}`` dicts (``ratio`` is the
            predicted ``T_E/T_P``, column ``ratio_e_over_p``).
    """
    rows = []
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({"side": r["side"], "decision": r["decision"], "mpki": float(r["mpki"]),
                                 "ratio": float(r["ratio_e_over_p"])})
                except (KeyError, ValueError, TypeError):   # truncated line (run killed)
                    pass
    except OSError:
        pass
    return rows


def pie_stats(rows):
    """Summarise PIE interval rows.

    Args:
        rows (list[dict]): Rows from :func:`parse_pie_csv`.

    Returns:
        (dict): Dict with ``n``, ``frac_on_e`` (fraction of intervals run on E),
            ``frac_want_e`` (fraction with decision E) and ``ratio_mean`` (mean
            predicted ``T_E/T_P``); the fractions and mean are ``None`` without rows.
    """
    if not rows:
        return {"n": 0, "frac_on_e": None, "frac_want_e": None, "ratio_mean": None}
    n = len(rows)
    return {"n": n, "frac_on_e": sum(r["side"] == "E" for r in rows) / n,
            "frac_want_e": sum(r["decision"] == "E" for r in rows) / n,
            "ratio_mean": statistics.fmean(r["ratio"] for r in rows)}


def pie_phase_tracking(rows):
    """Return P(decision=E | mpki above median) and P(decision=E | mpki at/below median).

    Args:
        rows (list[dict]): Rows from :func:`parse_pie_csv`.

    Returns:
        (tuple[float | None, float | None]): ``(p_high, p_low)``, or ``(None, None)`` with fewer
            than four rows or when
            one group is empty.
    """
    if len(rows) < 4:
        return None, None
    med = statistics.median(r["mpki"] for r in rows)
    hi = [r for r in rows if r["mpki"] > med]
    lo = [r for r in rows if r["mpki"] <= med]
    if not hi or not lo:
        return None, None
    f = lambda rs: sum(r["decision"] == "E" for r in rs) / len(rs)  # noqa: E731
    return f(hi), f(lo)


def sondag_types(summary):
    """Return the decided region types of a libwinhint sondag summary.

    Types still undecided (not sampled K times on both sides) are left out.

    Args:
        summary (dict | None): Parsed ``<log>.summary.json`` of libwinhint, or ``None``.

    Returns:
        (dict[int, dict]): ``{type: entry}`` where ``entry`` is the summary's dict for that type
            (with ``e_over_p`` and ``assigned`` in ``{"P", "E"}``).
    """
    return {int(t["type"]): t for t in (summary or {}).get("sondag", [])
            if t.get("assigned") in ("P", "E")}


def _pred(pid, baseline, kind, claim, value, ok):
    """Build one predicate row.

    Args:
        pid (str): Predicate id.
        baseline (str): Baseline(s) the predicate concerns (e.g. ``"R4"``).
        kind (str): ``"trend"`` (must hold) or ``"info"`` (reported only).
        claim (str): Human-readable claim.
        value (object): Measured value(s) behind the claim.
        ok (bool | None): Outcome, or ``None`` if not evaluable.

    Returns:
        (dict): Dict with ``id``, ``baseline``, ``kind``, ``claim``, ``value`` and ``pass``
            (bool or ``None``).
    """
    return {"id": pid, "baseline": baseline, "kind": kind, "claim": claim, "value": value,
            "pass": None if ok is None else bool(ok)}


def evaluate(data):
    """Evaluate all fidelity predicates on the parsed measurements.

    Covers the ground truth (GT-*), PIE (R4-*) and Sondag (R5-*) claims; R5
    assignment predicates are ``trend`` only when the threshold is separating.

    Args:
        data (dict): ``{"speedup": {cls: x}, "pie": {"ilp": rows, "memory": rows, "alt": rows},
            "sondag": summary_json, "sondag_threshold": thr, "threshold_separating": bool}``;
            missing keys make the affected predicates not evaluable.

    Returns:
        (list[dict]): List of predicate rows (see :func:`_pred`); ``pass=None`` means "not
            evaluable" (missing data).
    """
    sp = data.get("speedup", {})
    s_ilp, s_cmp, s_mem = sp.get("ilp"), sp.get("compute"), sp.get("memory")
    out = []
    both = lambda *v: all(x is not None for x in v)  # noqa: E731
    out.append(_pred("GT-mem-gains-less", "R4,R5", "trend",
                     "memory-intensive code gains less from the P-core than high-ILP code "
                     "(speedup_P/E(memory) < speedup_P/E(ilp))",
                     {"memory": s_mem, "ilp": s_ilp}, s_mem < s_ilp if both(s_mem, s_ilp) else None))
    out.append(_pred("GT-mem-vs-compute", "R4,R5", "info",
                     "speedup_P/E(memory) < speedup_P/E(compute, ILP 1)",
                     {"memory": s_mem, "compute": s_cmp}, s_mem < s_cmp if both(s_mem, s_cmp) else None))
    pie = data.get("pie", {})
    st_i, st_m = pie_stats(pie.get("ilp", [])), pie_stats(pie.get("memory", []))
    out.append(_pred("R4-pred-order", "R4", "trend",
                     "PIE predicts a smaller E-core slowdown for memory-intensive code "
                     "(mean predicted T_E/T_P: memory < ilp)",
                     {"memory": st_m["ratio_mean"], "ilp": st_i["ratio_mean"]},
                     st_m["ratio_mean"] < st_i["ratio_mean"] if both(st_m["ratio_mean"], st_i["ratio_mean"])
                     else None))
    out.append(_pred("R4-place-order", "R4", "trend",
                     "PIE runs memory-intensive code on the E-cores more than compute-intensive code "
                     "(fraction of intervals on E: memory > ilp)",
                     {"memory": st_m["frac_on_e"], "ilp": st_i["frac_on_e"]},
                     st_m["frac_on_e"] > st_i["frac_on_e"] if both(st_m["frac_on_e"], st_i["frac_on_e"])
                     else None))
    out.append(_pred("R4-place-memory-E", "R4", "info",
                     "memory-intensive run mostly on E (>= 50% of intervals) with the campaign slack",
                     st_m["frac_on_e"], st_m["frac_on_e"] >= 0.5 if st_m["frac_on_e"] is not None else None))
    out.append(_pred("R4-place-ilp-P", "R4", "info",
                     "high-ILP run mostly on P (>= 50% of intervals) with the campaign slack",
                     st_i["frac_on_e"], st_i["frac_on_e"] <= 0.5 if st_i["frac_on_e"] is not None else None))
    hi, lo = pie_phase_tracking(pie.get("alt", []))
    out.append(_pred("R4-phase-tracking", "R4", "trend",
                     "on the alternating workload, high-MPKI intervals are sent to E more often "
                     "than low-MPKI intervals",
                     {"want_e_high_mpki": hi, "want_e_low_mpki": lo}, hi > lo if both(hi, lo) else None))
    ty = sondag_types(data.get("sondag"))
    ti, tm = ty.get(REGION["ilp"]), ty.get(REGION["memory"])
    ri = ti.get("e_over_p") if ti else None
    rm = tm.get("e_over_p") if tm else None
    out.append(_pred("R5-order", "R5", "trend",
                     "sampled E/P time-per-instruction ratio: ilp type > memory type",
                     {"ilp": ri, "memory": rm}, ri > rm if both(ri, rm) else None))
    sep = data.get("threshold_separating", False)
    out.append(_pred("R5-assign-ilp-P", "R5", "trend" if sep else "info",
                     "the ilp region type is assigned to the P-cores (the core type that runs it better)",
                     {"assigned": ti.get("assigned") if ti else None, "threshold": data.get("sondag_threshold")},
                     (ti.get("assigned") == "P") if ti else None))
    out.append(_pred("R5-assign-mem-E", "R5", "trend" if sep else "info",
                     "the memory region type is assigned to the E-cores (smaller P advantage)",
                     {"assigned": tm.get("assigned") if tm else None, "threshold": data.get("sondag_threshold")},
                     (tm.get("assigned") == "E") if tm else None))
    for cls, t, r in (("ilp", ti, ri), ("memory", tm, rm)):
        g = sp.get(cls)
        out.append(_pred(f"R5-agrees-gt-{cls}", "R5", "info",
                         f"sampled ratio of the {cls} type within 30% of the pinned ground truth",
                         {"sondag": r, "ground_truth": g},
                         abs(r / g - 1) <= 0.3 if both(r, g) and g else None))
    return out


def run_fidelity(a):
    """Run the fidelity microbenchmarks and evaluate the predicates (``--fidelity``).

    Runs the ground truth (each class pinned to one P and one E CPU, ``reps``
    times), ``pie_daemon`` on the ilp, memory and alternating workloads, then
    ``phase_workload_hinted alt-ilp`` under libwinhint sondag mode with the
    separating threshold. Writes per-run stdout, ``fidelity.json`` and
    ``fidelity.csv`` under ``<out>/fidelity``. With ``--dry-run`` only prints the plan.

    Args:
        a (argparse.Namespace): Parsed namespace of run_hw_experiments.py (``fidelity_secs``,
            ``fidelity_reps``, ``sondag_k``, ``sondag_threshold``, PIE options, ...).

    Raises:
        SystemExit: If the machine is not hybrid or a tool binary is missing.
        subprocess.TimeoutExpired: If a run exceeds ``10 * secs + 60`` s.
    """
    import run_hw_experiments as rhe
    topo = rhe.topology()
    rhe.require_hybrid(topo)
    out = Path(a.out) / "fidelity"
    secs = a.fidelity_secs
    reps = a.fidelity_reps
    need = (4 * a.sondag_k + 2) * FID_PHASE_MS / 1000.0 + 0.5
    if secs < need:
        print(f"warning: --fidelity-secs {secs} < {need:.1f}: R5 cannot sample every region type "
              f"{a.sondag_k}x on both core types, and set-up time dominates the PIE runs")
    p1 = int(topo["pcpus"].split(",")[0].split("-")[0])
    e1 = int(topo["ecpus"].split(",")[0].split("-")[0])
    plan = []   # (name, argv, env, cpu or None)
    base_env = dict(os.environ)
    for cls in CLASSES:
        for side, cpu in (("P", p1), ("E", e1)):
            for r in range(reps):
                plan.append((f"gt_{cls}_{side}_{r}", workload_argv(a.tool_dir, cls, secs), base_env, cpu))
    for cls in ("ilp", "memory", "alt"):
        rd = out / f"pie_{cls}"
        plan.append((f"pie_{cls}", pie_argv(a, rd, workload_argv(a.tool_dir, cls, secs)), base_env, None))
    sondag_argv = workload_argv(a.tool_dir, "alt-ilp", secs, hinted=True)
    if a.dry_run:
        for name, argv, _, cpu in plan:
            pin = f"[cpu {cpu}] " if cpu is not None else ""
            print(f"[dry-run] fidelity {name}: {pin}{' '.join(shlex.quote(x) for x in argv)}")
        print(f"[dry-run] fidelity sondag: WINHINT_MODE=sondag WINHINT_SONDAG_THRESHOLD=<separating> "
              f"{' '.join(sondag_argv)}")
        return
    for b in ("phase_workload", "phase_workload_hinted", "pie_daemon"):
        if not (Path(a.tool_dir) / b).exists():
            sys.exit(f"{Path(a.tool_dir) / b} missing: make -C {HW} -j1")
    out.mkdir(parents=True, exist_ok=True)
    rates = {(c, s): [] for c in CLASSES for s in "PE"}
    for name, argv, env, cpu in plan:
        if name.startswith("pie_"):
            (out / name).mkdir(parents=True, exist_ok=True)
        pre = (lambda c=cpu: os.sched_setaffinity(0, {c})) if cpu is not None else None
        p = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           preexec_fn=pre, timeout=secs * 10 + 60)
        (out / f"{name}.stdout").write_text(p.stdout)
        if p.returncode:
            print(f"FAIL {name} ({p.returncode}): {p.stderr.strip()[-200:]}")
            continue
        if name.startswith("gt_"):
            _, cls, side, _ = name.split("_")
            rates[(cls, side)].append(work_rate(parse_workload(p.stdout), cls))
        print(f"ok   {name}")
    data = {"speedup": {c: speedup(rates[(c, "P")], rates[(c, "E")]) for c in CLASSES},
            "pie": {c: parse_pie_csv(out / f"pie_{c}" / "pie.csv") for c in ("ilp", "memory", "alt")}}
    thr, sep = separating_threshold(data["speedup"]["ilp"], data["speedup"]["memory"], a.sondag_threshold)
    data["sondag_threshold"], data["threshold_separating"] = thr, sep
    log = out / "sondag" / "winhint.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(sondag_argv, env=sondag_env(base_env, log, thr, a.sondag_k),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=secs * 10 + 60)
    (out / "sondag" / "stdout.txt").write_text(p.stdout)
    (out / "sondag" / "stderr.txt").write_text(p.stderr)
    try:
        data["sondag"] = json.loads(Path(str(log) + ".summary.json").read_text())
    except (OSError, ValueError):
        data["sondag"] = None
        print(f"FAIL sondag ({p.returncode}): {p.stderr.strip()[-200:]}")
    preds = evaluate(data)
    (out / "fidelity.json").write_text(json.dumps(
        {"model": topo["model"], "secs": secs, "reps": reps, "speedup": data["speedup"],
         "sondag_threshold": thr, "threshold_separating": sep, "predicates": preds}, indent=1) + "\n")
    with open(out / "fidelity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "baseline", "kind", "pass", "claim", "value"])
        w.writeheader()
        for r in preds:
            w.writerow(dict(r, value=json.dumps(r["value"])))
    for r in preds:
        st = {True: "PASS", False: "FAIL", None: "n/a "}[r["pass"]]
        print(f"{st} [{r['kind']:5s}] {r['id']:22s} {r['claim']}")
    bad = [r["id"] for r in preds if r["kind"] == "trend" and r["pass"] is False]
    print(f"-> {out / 'fidelity.json'}" + (f"  (trend predicates failing: {', '.join(bad)})" if bad else ""))
