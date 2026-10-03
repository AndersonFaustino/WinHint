#!/usr/bin/env python3
"""WinHint Phase E2 -- real-silicon campaign driver (Intel hybrid, e.g. Core 5 120U).

Runs kernels x configurations x SMT states x repetitions, measuring wall time,
RAPL package/core energy, EDP and user-mode cycles/instructions per core type
(with hw/tools/wh_measure), and writes:

      results/hw/raw.csv          one row per run (append-only, resumable)
      results/hw/summary.csv      mean, 95% CI (Student t), normalized to R1
      results/hw/runs/<run_id>/   per-run logs (wh_measure JSON, libwinhint CSV/JSON, PIE log)
      results/hw/system/          system snapshots (record_system.sh), migration microbenchmark

Configurations (see docs/guide/hardware/index.md):

      R0-P, R0-E           plain binary pinned to all P / all E cores (taskset semantics)
      R1                   plain binary, stock scheduler
      R2-<preset>          plain binary under a sched_ext scheduler (needs --enable-scx)
      R3-lpmd              plain binary with intel-lpmd running (needs --enable-lpmd)
      R4-PIE               plain binary managed by pie_daemon (perf or pmctrack backend)
      R5-Sondag            region-marker call-mode binary, WINHINT_MODE=sondag
      WH                   WinHint call-mode binary, WINHINT_MODE=migrate
      WH-off               WinHint call-mode binary, WINHINT_MODE=off (call overhead only)
      NOP-plain-{P,E}      plain binary pinned to one P (E) cpu   } x86 NOP-hint overhead,
      NOP-asm-{P,E}        asm-hint binary pinned to one P (E) cpu} perf stat when available

Optional workloads instead of benchmarks/*.c:

      --synthetic          hw/tools/phase_workload* (pipeline check only)
      --llamacpp           llama.cpp llama-simple with the ggml hooks (hw/integrations/llamacpp):
                           plain = build-vanilla, call/regions = build-winhint; single ggml thread
                           (OMP_THREAD_LIMIT=1: only thread 0 calls libwinhint and libwinhint moves
                           only the calling thread); R5-Sondag adds GGML_WINHINT_SETWIN=none

Other modes (no campaign): --uarch (µarch window probes), --fidelity (R4/R5 trend checks),
see hw/fidelity.py.

Host workflow (conda env `winhint`, no containers), see docs/guide/hardware/index.md:

      python hw/run_hw_experiments.py --dry-run                         # plan + privilege check
      python hw/run_hw_experiments.py --synthetic --reps 1 --allow-no-rapl \
          --configs R0-P,R0-E,R1,WH --cooldown 0                       # unprivileged smoke run

Energy (RAPL energy_uj is root-only on this kernel) needs either root:

      sudo -E env "PATH=$PATH" python hw/run_hw_experiments.py ...     # results chowned back

or the opt-in read grant (hw/baselines/r1_stock/rapl_access.sh, run by the user with sudo).
Unprivileged, the driver runs everything except energy (and R2/R3, which need root).
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

#: Directory of this file (``hw/``).
HW = Path(__file__).resolve().parent
#: Repository root (``$WINHINT_ROOT``, default the parent of ``hw/``).
ROOT = Path(os.environ.get("WINHINT_ROOT") or HW.parent)
#: Build root (``$WINHINT_BUILD``, default ``<root>/build``).
BUILD = Path(os.environ.get("WINHINT_BUILD") or ROOT / "build")
#: Directory of the R-baseline helper scripts.
BASE = HW / "baselines"

#: Kernel names of benchmarks/*.c (``--kernels all``).
ALL_KERNELS = sorted(p.stem for p in (ROOT / "benchmarks").glob("*.c"))
#: Pseudo-kernel name of --synthetic.
SYNTHETIC = "phase_workload"   # hw/tools/phase_workload.c (built by hw/Makefile)
#: --synthetic: variant role -> binary name in --tool-dir.
SYNTHETIC_BINS = {"plain": "phase_workload", "asm": "phase_workload_nop",
                  "call": "phase_workload_hinted", "regions": "phase_workload_hinted"}
#: Pseudo-kernel name of --llamacpp.
LLAMACPP = "llamacpp"          # hw/integrations/llamacpp (build_llamacpp.sh)
#: --llamacpp: variant role -> build dir (llama-simple has no NOP-hint build).
LLAMACPP_BUILDS = {"plain": "build-vanilla", "call": "build-winhint", "regions": "build-winhint"}
#: Default llama-simple arguments (``{dir}`` = --llamacpp-dir).
LLAMACPP_ARGS = "-m {dir}/models/stories15M-q4_0.gguf -n 256 'Once upon a time'"
#: sched_ext presets of the R2 configurations (``--configs all``).
SCX_PRESETS = ["bpfland_powersave", "cosmos_powersave", "lavd_powersave"]
#: Configurations of ``--configs core``.
CORE_CONFIGS = ["R0-P", "R0-E", "R1", "R4-PIE", "R5-Sondag", "WH", "WH-off",
                "NOP-plain-P", "NOP-asm-P", "NOP-plain-E", "NOP-asm-E"]

#: Column order of raw.csv.
RAW_FIELDS = ["run_id", "kernel", "config", "group", "smt", "rep", "input", "timestamp",
              "exit_code", "wall_s", "energy_pkg_j", "energy_core_j", "edp_pkg_js", "edp_core_js",
              "cycles_p", "instructions_p", "run_s_p", "cycles_e", "instructions_e", "run_s_e",
              "migrations", "perf_stat_cycles", "perf_stat_instructions", "stdout_sha256",
              "output_ok", "governor", "epp", "no_turbo", "note"]


# ----------------------------------------------------------------------------- helpers
def rd(path, default=""):
    """Return the stripped contents of a text file, or ``default`` if it cannot be read.

    Args:
        path (str | Path): File path (str or Path).
        default (str): Value returned on ``OSError``.

    Returns:
        (str): The file contents without surrounding whitespace, or ``default``.
    """
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def cpulist(path, env):
    """Return a CPU list from the environment variable ``env`` or, if unset/empty, from ``path``.

    Args:
        path (str): sysfs file with a CPU list (e.g. ``/sys/devices/cpu_core/cpus``).
        env (str): Name of the overriding environment variable.

    Returns:
        (str): The CPU list string (``""`` if neither is available).
    """
    return os.environ.get(env) or rd(path)


def smt_state():
    """Return the SMT state from sysfs: ``"on"``, ``"off"`` or ``"unknown"``."""
    return {"1": "on", "0": "off"}.get(rd("/sys/devices/system/cpu/smt/active"), "unknown")


def freq_policy():
    """Return the current frequency policy of all CPUs with cpufreq.

    Returns:
        (tuple[str, str, str]): ``(governors, epps, no_turbo)``; differing values across CPUs are
            joined
            with ``|`` (``"unknown"`` if none), ``no_turbo`` is the intel_pstate value
            or ``"NA"``.
    """
    govs = {rd(p / "cpufreq/scaling_governor") for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*")
            if (p / "cpufreq").exists()}
    epps = {rd(p / "cpufreq/energy_performance_preference")
            for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*") if (p / "cpufreq").exists()}
    return ("|".join(sorted(govs)) or "unknown", "|".join(sorted(epps)) or "unknown",
            rd("/sys/devices/system/cpu/intel_pstate/no_turbo", "NA"))


def topology():
    """Return the P/E core sets from sysfs or WINHINT_PCPUS/WINHINT_ECPUS.

    The sysfs sources are the cpu_core / cpu_atom PMUs.

    Both sets are restricted to online CPUs (SMT off => P siblings offline).

    Returns:
        (dict): Dict with ``pcpus`` and ``ecpus`` (CPU list strings), ``online``, ``hybrid``
            (both sets non-empty), ``model`` (CPU model name), ``smt`` (see
            :func:`smt_state`) and ``source`` (``"env"`` or ``"sysfs"``).
    """
    pc = cpulist("/sys/devices/cpu_core/cpus", "WINHINT_PCPUS")
    ec = cpulist("/sys/devices/cpu_atom/cpus", "WINHINT_ECPUS")
    online = rd("/sys/devices/system/cpu/online")
    if online:
        pc = intersect_cpulist(pc, online) if pc else ""
        ec = intersect_cpulist(ec, online) if ec else ""
    model = ""
    for line in rd("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    src = "env" if os.environ.get("WINHINT_PCPUS") or os.environ.get("WINHINT_ECPUS") else "sysfs"
    return {"pcpus": pc, "ecpus": ec, "online": online, "hybrid": bool(pc and ec), "model": model,
            "smt": smt_state(), "source": src}


def require_hybrid(topo):
    """Exit with an explanatory error unless the topology is a hybrid P/E CPU.

    Args:
        topo (dict): Dict from :func:`topology`.

    Raises:
        SystemExit: If ``topo["hybrid"]`` is false.
    """
    if not topo["hybrid"]:
        sys.exit(f"ERROR: not a hybrid P/E CPU ({topo['model'] or 'unknown CPU'}): P-cores='{topo['pcpus']}' "
                 f"E-cores='{topo['ecpus']}' (online {topo['online']}).\n"
                 "The real-hardware evaluation needs an Intel hybrid CPU exposing /sys/devices/cpu_core/cpus "
                 "and /sys/devices/cpu_atom/cpus (e.g. Core 5 120U); WINHINT_PCPUS/WINHINT_ECPUS override "
                 "them for testing only.")


def t_crit95(df):
    """Return the two-sided 95 % Student-t critical value for ``df`` degrees of freedom.

    Uses scipy when available; otherwise the smallest tabulated ``df`` not below
    the requested one (1.96 beyond 30).

    Args:
        df (int): Degrees of freedom.

    Returns:
        (float): The critical value ``t_{0.975, df}``.
    """
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:  # small table fallback
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
                 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}
        for k in sorted(table):
            if df <= k:
                return table[k]
        return 1.96


def mean_ci(xs):
    """Return the mean and the 95 % Student-t confidence interval of a sample.

    ``None`` and NaN entries are ignored.

    Args:
        xs (iterable of float | None): Iterable of numbers.

    Returns:
        (tuple[float, float, float]): ``(mean, lo, hi)``; all NaN for an empty sample, ``lo``/``hi``
            NaN for a
            single value.
    """
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return (float("nan"),) * 3
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, float("nan"), float("nan")
    h = t_crit95(len(xs) - 1) * statistics.stdev(xs) / math.sqrt(len(xs))
    return m, m - h, m + h


def fnum(v):
    """Return ``v`` as a float, or ``None`` if it cannot be converted."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- campaign
class Campaign:
    """Real-silicon measurement campaign over kernels x configurations x SMT states x reps.

    Resumable: runs already in ``raw.csv`` are skipped (failed ones are retried
    with ``--retry-failed``). The constructor reads the topology and exits if the
    CPU is not hybrid.

    Attributes:
        a: Parsed command-line namespace.
        out: Output directory (``--out``).
        raw: Path of ``raw.csv``.
        topo: Topology dict from :func:`topology`.
        pcpus: P-core CPU list.
        ecpus: E-core CPU list.
        p1: CPU used by the ``NOP-*-P`` configurations (``--nop-pcpu`` or the first P CPU).
        e1: CPU used by the ``NOP-*-E`` configurations (``--nop-ecpu`` or the first E CPU).
        measure: Path of ``wh_measure``.
        pie: Path of ``pie_daemon``.
        perf: Path of ``perf`` for the NOP rows, or ``None``.
        done: Rows of ``raw.csv`` keyed by run id.
        ref_sha: R1 reference stdout SHA-256 per kernel.
        sys_changed: Whether the campaign changed system settings that must be restored.
        policy0: Frequency policy ``(governor, epp, no_turbo)`` the campaign runs under.
    """
    def __init__(self, a):
        """Set up the campaign and load the already finished runs.

        Args:
            a (argparse.Namespace): Parsed command-line namespace.

        Raises:
            SystemExit: If the CPU is not hybrid.
        """
        self.a = a
        self.out = Path(a.out)
        self.raw = self.out / "raw.csv"
        self.topo = topology()
        require_hybrid(self.topo)
        self.pcpus, self.ecpus = self.topo["pcpus"], self.topo["ecpus"]
        self.p1 = str(a.nop_pcpu if a.nop_pcpu is not None else int(self.pcpus.split(",")[0].split("-")[0]))
        self.e1 = str(a.nop_ecpu if a.nop_ecpu is not None else int(self.ecpus.split(",")[0].split("-")[0]))
        self.measure = Path(a.tool_dir) / "wh_measure"
        self.pie = Path(a.tool_dir) / "pie_daemon"
        self.perf = shutil.which("perf") if not a.no_perf_stat else None
        self.done = self._load_done()
        self.ref_sha = {}
        for r in self.done.values():   # resumed campaign: R1 reference output from earlier runs
            if r["config"] == "R1" and r["exit_code"] == "0" and r.get("input") == a.input:
                self.ref_sha.setdefault(r["kernel"], r["stdout_sha256"])
        self.sys_changed = False
        self.policy0 = None            # frequency policy the campaign runs under (checked per run)

    # ---------------- binaries
    def role(self, variant):
        """Return the role (``plain``, ``asm``, ``call`` or ``regions``) of a build variant name.

        Args:
            variant (str): Variant directory name (``--variant-*``).

        Returns:
            (str): The role; unknown variants are ``plain``.
        """
        a = self.a
        return {a.variant_plain: "plain", a.variant_asm: "asm", a.variant_call: "call",
                a.variant_regions: "regions"}.get(variant, "plain")

    def binary(self, kernel, variant):
        """Return the binary of a kernel in a given variant.

        Args:
            kernel (str): Kernel name (or the synthetic / llama.cpp pseudo-kernel).
            variant (str): Build variant name.

        Returns:
            (Path): ``<bench_root>/<variant>/<kernel>``, the phase_workload binary of the
                variant's role (``--synthetic``) or the llama-simple of the role's build
                (``--llamacpp``).
        """
        a = self.a
        if kernel == SYNTHETIC and a.synthetic:
            return Path(a.tool_dir) / SYNTHETIC_BINS[self.role(variant)]
        if kernel == LLAMACPP and getattr(a, "llamacpp", False):
            build = LLAMACPP_BUILDS.get(self.role(variant), "build-vanilla")
            return Path(a.llamacpp_dir) / build / "bin" / "llama-simple"
        return Path(a.bench_root) / variant / kernel

    def prog_args(self, kernel):
        """Return the command-line arguments of a kernel.

        Args:
            kernel (str): Kernel name.

        Returns:
            (list[str]): ``--synthetic-args`` / ``--llamacpp-args`` (or :data:`LLAMACPP_ARGS`) for
                the
                pseudo-kernels, else ``[--input]``.
        """
        a = self.a
        if kernel == SYNTHETIC and a.synthetic:
            return a.synthetic_args.split()
        if kernel == LLAMACPP and getattr(a, "llamacpp", False):
            import shlex
            return shlex.split(a.llamacpp_args or LLAMACPP_ARGS.format(dir=a.llamacpp_dir))
        return [a.input]

    def variant_of(self, config):
        """Return the build variant a configuration runs.

        Args:
            config (str): Configuration name.

        Returns:
            (str): ``--variant-call`` for WH/WH-off, ``--variant-regions`` for R5-Sondag,
                ``--variant-asm`` for NOP-asm-*, else ``--variant-plain``.
        """
        a = self.a
        if config in ("WH", "WH-off"):
            return a.variant_call
        if config == "R5-Sondag":
            return a.variant_regions
        if config.startswith("NOP-asm"):
            return a.variant_asm
        return a.variant_plain

    # ---------------- resumability
    def _load_done(self):
        """Return the rows of an existing ``raw.csv`` keyed by run id (empty if none)."""
        done = {}
        if self.raw.exists():
            with open(self.raw) as f:
                for r in csv.DictReader(f):
                    done[r["run_id"]] = r
        return done

    def is_done(self, run_id):
        """Return whether a run can be skipped.

        Args:
            run_id (str): Run id ``<kernel>|<config>|smt-<s>|<rep>``.

        Returns:
            (bool): True if the run is recorded and succeeded, or failed and ``--retry-failed`` is
                off.
        """
        r = self.done.get(run_id)
        return r is not None and (r["exit_code"] == "0" or not self.a.retry_failed)

    def append(self, row):
        """Append one row to ``raw.csv`` (writing the header for a new file) and record it as done.

        Args:
            row (dict): Dict with the :data:`RAW_FIELDS` keys.
        """
        new = not self.raw.exists()
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.raw, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        self.done[row["run_id"]] = row

    # ---------------- command construction
    def command(self, kernel, config, rundir):
        """Return the command and environment of one run, wrapped by ``wh_measure``.

        Applies the configuration: CPU pinning (R0-*, NOP-*), ``pie_daemon`` (R4-PIE),
        libwinhint ``WINHINT_*`` variables (WH, WH-off, R5-Sondag), single-threaded
        ggml for llama.cpp, and ``perf stat`` for NOP-* when perf is available.
        Inherited ``WINHINT_*`` variables other than ``WINHINT_PCPUS``/``WINHINT_ECPUS``
        are dropped.

        Args:
            kernel (str): Kernel name.
            config (str): Configuration name.
            rundir (Path): Per-run output directory.

        Returns:
            (tuple[list[str], dict[str, str]]): ``(argv, env)``.
        """
        a = self.a
        binp = str(self.binary(kernel, self.variant_of(config)))
        prog = [binp] + self.prog_args(kernel)
        env = dict(os.environ)
        for k in list(env):
            if k.startswith("WINHINT_") and k not in ("WINHINT_PCPUS", "WINHINT_ECPUS"):
                del env[k]
        wm = [str(self.measure), "-o", str(rundir / "measure.json")]
        if a.require_rapl:
            wm.append("-r")
        pin = None
        if config == "R0-P":
            pin = self.pcpus
        elif config == "R0-E":
            pin = self.ecpus
        elif config.startswith("NOP-") and config.endswith("-P"):
            pin = self.p1
        elif config.startswith("NOP-") and config.endswith("-E"):
            pin = self.e1
        if pin:
            wm += ["-c", pin]
        if config == "R4-PIE":
            pie = [str(self.pie), "-b", a.pie_backend, "-i", str(a.pie_interval_ms), "-s", str(a.pie_slack),
                   "-H", str(a.pie_hyst), "-o", str(rundir / "pie.csv"), "-S", str(rundir / "pie.json")]
            if a.pie_backend == "pmctrack":
                pmc = shutil.which("pmctrack") or self.pmctrack_cli()
                if pmc:
                    pie += ["-C", pmc + " -T %.3f -c %s -p %d"]
                    env.setdefault("PMCTRACK_ROOT", str(Path(pmc).resolve().parents[1]))
            prog = pie + ["--"] + prog
        if config in ("WH", "WH-off", "R5-Sondag"):
            env["WINHINT_MODE"] = {"WH": "migrate", "WH-off": "off", "R5-Sondag": "sondag"}[config]
            env["WINHINT_LOG"] = str(rundir / "winhint.csv")
            env["WINHINT_REQUIRE_HYBRID"] = "1"   # never silently degrade to "no migration"
            # Per-region counters cost a perf_event_open at start-up (several ms on this kernel);
            # wh_measure already counts per core type, so WH runs without them unless asked.
            # R5 needs them (time per instruction of each sampled region type).
            if config != "R5-Sondag" and not a.winhint_perf:
                env["WINHINT_PERF"] = "0"
            env["WINHINT_THRESHOLD"] = str(a.threshold)
            if a.min_dwell_us:
                env["WINHINT_MIN_DWELL_US"] = str(a.min_dwell_us)
            if a.hyst > 1:
                env["WINHINT_HYST"] = str(a.hyst)
            if config == "R5-Sondag":
                env["WINHINT_SONDAG_K"] = str(a.sondag_k)
                env["WINHINT_SONDAG_THRESHOLD"] = str(a.sondag_threshold)
                types = Path(a.sondag_types_dir) / f"{kernel}.types" if a.sondag_types_dir else None
                if types and types.exists():
                    env["WINHINT_SONDAG_TYPES"] = str(types)
        if kernel == LLAMACPP and getattr(a, "llamacpp", False):
            # One ggml thread: only thread 0 calls libwinhint and libwinhint migrates only the
            # calling thread (llama-simple has no -t; OMP_THREAD_LIMIT caps ggml's OpenMP team).
            # Applied to every config so all of them run the same single-threaded work.
            env["OMP_THREAD_LIMIT"] = "1"
            env["OMP_NUM_THREADS"] = "1"
            if config == "R5-Sondag":
                env["GGML_WINHINT_SETWIN"] = "none"   # region markers only (R5 decides)
            else:
                env.pop("GGML_WINHINT_SETWIN", None)
        if config.startswith("NOP-") and self.perf:
            # perf stat inside wh_measure (pinned with it): it counts exactly the kernel binary
            # (not the wh_measure wrapper); C locale for a stable CSV.
            env["LC_ALL"] = "C"
            prog = [self.perf, "stat", "-x", ",", "-o", str(rundir / "perf_stat.csv"),
                    "-e", "cycles:u,instructions:u", "--"] + prog
        argv = wm + ["--"] + prog
        return argv, env

    def pmctrack_cli(self):
        """Return the pmctrack CLI built by install_pmctrack.sh (``--hw-baselines``), or None."""
        p = Path(self.a.hw_baselines) / "pmctrack" / "bin" / "pmctrack"   # install_pmctrack.sh build
        return str(p) if p.is_file() and os.access(p, os.X_OK) else None

    # ---------------- one run
    def run_one(self, kernel, config, group, smt, rep):
        """Execute (or, with ``--dry-run``, print) one run and append its row to ``raw.csv``.

        Collects wall time and RAPL energy from ``measure.json``, migrations from the
        libwinhint or PIE summary, ``perf stat`` counters, the stdout hash (checked
        against the R1 reference output, except for the synthetic workload) and the
        frequency policy.

        Args:
            kernel (str): Kernel name.
            config (str): Configuration name.
            group (str): System-state group (see :meth:`plan`).
            smt (str): SMT state label.
            rep (int): Repetition index.

        Returns:
            (str): ``"skip"`` (already done), ``"dry"``, ``"ok"`` or ``"FAIL(<exit code>)"``
                (124 on timeout).
        """
        a = self.a
        run_id = f"{kernel}|{config}|smt-{smt}|{rep}"
        if self.is_done(run_id):
            return "skip"
        rundir = self.out / "runs" / run_id.replace("|", "__")
        argv, env = self.command(kernel, config, rundir)
        if a.dry_run:
            extra = " ".join(f"{k}={v}" for k, v in env.items()
                             if k.startswith(("WINHINT_", "GGML_WINHINT_", "OMP_")) and os.environ.get(k) != v)
            print(f"[dry-run] {run_id}: {extra} {' '.join(argv)}")
            return "dry"
        rundir.mkdir(parents=True, exist_ok=True)
        time.sleep(a.cooldown)
        t0 = time.time()
        note = ""
        try:
            p = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=a.timeout)
            code, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            code, out, err, note = 124, e.stdout or b"", e.stderr or b"", "timeout"
        (rundir / "stdout.txt").write_bytes(out)
        (rundir / "stderr.txt").write_bytes(err)
        m = {}
        try:
            m = json.loads((rundir / "measure.json").read_text())
        except (OSError, ValueError):
            note = note or "no measure.json"
        sha = hashlib.sha256(out).hexdigest()
        ref_key = kernel
        if config == "R1" and code == 0 and ref_key not in self.ref_sha:
            self.ref_sha[ref_key] = sha
        ref = self.ref_sha.get(ref_key)
        if kernel == SYNTHETIC and a.synthetic:
            ref = None   # time-based workload: its output (step counts) varies by design
        wall = (m.get("wall_ns") or 0) / 1e9 or None
        epkg = m.get("energy_pkg_uj")
        ecore = m.get("energy_core_uj")
        epkg = epkg / 1e6 if epkg is not None else None
        ecore = ecore / 1e6 if ecore is not None else None
        migr = ""
        for js in ("winhint.csv.summary.json", "pie.json"):
            try:
                d = json.loads((rundir / js).read_text())
                migr = d.get("n_migrations", d.get("migrations", ""))
            except (OSError, ValueError):
                pass
        pc = pi = ""
        if (rundir / "perf_stat.csv").exists():
            for line in (rundir / "perf_stat.csv").read_text().splitlines():
                f = line.split(",")
                if len(f) > 2 and f[0].replace(".", "").isdigit():
                    if f[2].startswith("cycles") or "cycles" in f[2]:
                        pc = str(int(float(f[0])) + int(float(pc or 0)))  # sums cpu_core + cpu_atom rows
                    elif "instructions" in f[2]:
                        pi = str(int(float(f[0])) + int(float(pi or 0)))
        gov, epp, nt = freq_policy()
        if self.policy0 and (gov, epp, nt) != self.policy0:
            note = (note + "; " if note else "") + \
                f"frequency policy changed during the campaign (was {'/'.join(self.policy0)})"
        row = {
            "run_id": run_id, "kernel": kernel, "config": config, "group": group, "smt": smt, "rep": rep,
            "input": a.input, "timestamp": int(t0), "exit_code": code,
            "wall_s": wall, "energy_pkg_j": epkg, "energy_core_j": ecore,
            "edp_pkg_js": epkg * wall if epkg is not None and wall else None,
            "edp_core_js": ecore * wall if ecore is not None and wall else None,
            "cycles_p": m.get("cycles_p"), "instructions_p": m.get("instructions_p"),
            "run_s_p": (m.get("run_ns_p") or 0) / 1e9 if "run_ns_p" in m else None,
            "cycles_e": m.get("cycles_e"), "instructions_e": m.get("instructions_e"),
            "run_s_e": (m.get("run_ns_e") or 0) / 1e9 if "run_ns_e" in m else None,
            "migrations": migr, "perf_stat_cycles": pc, "perf_stat_instructions": pi,
            "stdout_sha256": sha, "output_ok": "" if ref is None else int(sha == ref),
            "governor": gov, "epp": epp, "no_turbo": nt,
            "note": (note + ("; " if note and m.get("rapl_msg") else "") + (m.get("rapl_msg") or ""))[:240],
        }
        self.append(row)
        status = "ok" if code == 0 else f"FAIL({code})"
        print(f"{status:8s} {run_id}  wall={wall if wall is None else round(wall, 4)}s "
              f"Epkg={epkg if epkg is None else round(epkg, 3)}J migr={migr}", flush=True)
        err_lines = err.decode(errors="replace").strip().splitlines()
        if code != 0 and err_lines:   # whitespace-only stderr has no line to show
            print("    stderr: " + err_lines[-1][:200], flush=True)
        return status

    # ---------------- system-state groups
    def group_ctx(self, group):
        """Return start/stop callables for the system-wide scheduler daemons of a group (R2/R3).

        Args:
            group (str): Group name (``stock``, ``scx:<preset>`` or ``lpmd``).

        Returns:
            (tuple[Callable | None, Callable | None]): ``(start, stop)``; ``(None, None)`` for
                groups without a daemon. ``start``
                raises ``subprocess.CalledProcessError`` on failure, ``stop`` does not check.
        """
        a = self.a
        env = dict(os.environ, WINHINT_ALLOW_SYSTEM_CHANGES="1")
        if group.startswith("scx:"):
            preset = group[4:]
            start = ["bash", str(BASE / "r2_sched_ext/run_scx.sh"), "start", preset]
            stop = ["bash", str(BASE / "r2_sched_ext/run_scx.sh"), "stop"]
        elif group == "lpmd":
            start = ["bash", str(BASE / "r3_lpmd/run_lpmd.sh"), "start", a.lpmd_mode]
            stop = ["bash", str(BASE / "r3_lpmd/run_lpmd.sh"), "stop"]
        else:
            return None, None
        return (lambda: subprocess.run(start, env=env, check=True),
                lambda: subprocess.run(stop, env=env, check=False))

    def plan(self):
        """Group the configurations by the system state they need.

        Returns:
            (list[tuple[str, list[str]]]): ``[(group, [configs])]``: one ``stock`` group with all
                configurations that
                need no daemon, then one group per R2 preset and one for R3-lpmd.
        """
        a = self.a
        configs = a.configs
        groups = []
        stock = [c for c in configs if not c.startswith(("R2-", "R3-"))]
        if stock:
            groups.append(("stock", stock))
        for c in configs:
            if c.startswith("R2-"):
                groups.append(("scx:" + c[3:], [c]))
            elif c == "R3-lpmd":
                groups.append(("lpmd", [c]))
        return groups

    def bench_migration(self, sysdir, smt):
        """Measure the P<->E migration cost for this SMT state.

        The result is the input of the compiler's P/E switch cost.

        ``migration_cost_smt-<s>.json`` is measured the way libwinhint migrates by default
        (WINHINT_PIN=set: affinity to the whole P / E set, ``bench_migration -s``); this is the
        file benchmarks/Makefile reads (switch_cost_us -> -winhint-migration-us).
        ``migration_cost_single_smt-<s>.json`` pins to one CPU per side (WINHINT_PIN=single).
        Does nothing with ``--no-bench-migration`` or when the tool is missing.

        Args:
            sysdir (Path): Output directory (``<out>/system``).
            smt (str): SMT state label used in the file names.
        """
        bm = Path(self.a.lib_dir) / "bench_migration"
        if not self.a.bench_migration:
            return
        if not bm.exists():
            print(f"note: {bm} missing, migration cost not measured")
            return
        n = str(self.a.bench_migration_n)
        subprocess.run([str(bm), "-n", n, "-s", "-o", str(sysdir / f"migration_cost_smt-{smt}.json")],
                       check=False)
        subprocess.run([str(bm), "-n", n, "-o", str(sysdir / f"migration_cost_single_smt-{smt}.json")],
                       check=False)

    def set_smt(self, want):
        """Switch SMT to the requested state and re-read the topology.

        Args:
            want (str): ``"on"``, ``"off"`` or ``"current"`` (no change).

        Returns:
            (str): The SMT state after the change.

        Raises:
            SystemExit: If a change is needed without ``--allow-system-changes``, or the
                CPU is no longer hybrid.
            subprocess.CalledProcessError: If host_setup.sh fails.
        """
        if want == "current" or smt_state() == want:
            return smt_state()
        if not self.a.allow_system_changes:
            sys.exit(f"SMT is {smt_state()}, campaign wants {want}: pass --allow-system-changes "
                     "(needs root: run the driver with sudo) or switch SMT yourself, e.g. "
                     f"sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 {BASE / 'r1_stock/host_setup.sh'} apply --smt {want}")
        subprocess.run(["bash", str(BASE / "r1_stock/host_setup.sh"), "apply", "--smt", want],
                       env=dict(os.environ, WINHINT_ALLOW_SYSTEM_CHANGES="1"), check=True)
        self.sys_changed = True
        time.sleep(2)
        # topology changed (siblings on/offline): re-read the P list
        self.topo = topology()
        require_hybrid(self.topo)
        self.pcpus, self.ecpus = self.topo["pcpus"], self.topo["ecpus"]
        return smt_state()

    def preflight(self):
        """Check tools, binaries, privileges, RAPL access and frequency policy before a campaign.

        Whether a finding is a problem or a note depends on the options (e.g.
        ``--require-rapl``, ``--allow-unfixed-governor``); fewer than 10 reps is only noted.

        Returns:
            (tuple[list[str], list[str]]): ``(problems, notes)``, two lists of messages; any problem
                aborts a non-dry run.
        """
        a = self.a
        problems, notes = [], []
        if not self.pcpus or not self.ecpus:
            problems.append("no P/E topology (sysfs cpu_core/cpu_atom); set WINHINT_PCPUS/WINHINT_ECPUS")
        if not self.measure.exists():
            problems.append(f"{self.measure} missing: micromamba run -n winhint make -C {HW} -j1")
        if any(c == "R4-PIE" for c in a.configs) and not self.pie.exists():
            problems.append(f"{self.pie} missing: micromamba run -n winhint make -C {HW} -j1")
        if "R4-PIE" in a.configs and a.pie_backend == "pmctrack":
            if not (shutil.which("pmctrack") or self.pmctrack_cli()):
                problems.append("pmctrack CLI not built: hw/baselines/r4_pie/install_pmctrack.sh build")
            if not Path("/proc/pmc").exists():
                problems.append("PMCTrack kernel module not loaded (/proc/pmc missing): "
                                "sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r4_pie/install_pmctrack.sh load")
        missing = sorted({str(self.binary(k, self.variant_of(c))) for k in a.kernels for c in a.configs
                          if not self.binary(k, self.variant_of(c)).exists()})
        if missing:
            problems.append(f"{len(missing)} benchmark binaries missing, e.g. {missing[0]} "
                            f"(make -C benchmarks ARCH=x86 VARIANT=..., or use --synthetic)")
        if any(c.startswith("NOP-asm") for c in a.configs):
            for k in a.kernels:
                b = self.binary(k, a.variant_asm)
                if b.exists() and not x86_nop_hints(b):
                    problems.append(f"{b} contains no WinHint x86 NOP hints (0F 1F 80 .. .. 48 57): "
                                    "the NOP-overhead rows would compare two plain binaries")
        if self.measure.exists():
            p = subprocess.run([str(self.measure), "-r", "-k", "-o", "/dev/stdout", "--", "true"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode == 4:
                msg = p.stderr.decode().strip()
                (notes if not a.require_rapl else problems).append(msg)
            elif p.returncode == 5:
                notes.append(p.stderr.decode().strip() + " (per-core-type counters will be empty)")
        priv = privileges()
        notes.append("privileges: " + ", ".join(f"{k}={v}" for k, v in priv.items()))
        if not priv["rapl_readable"]:
            msg = ("RAPL energy_uj not readable by this user: energy columns will be empty. Either run "
                   "the driver as root (sudo -E env \"PATH=$PATH\" python hw/run_hw_experiments.py ...) "
                   "or grant read access (sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 "
                   "hw/baselines/r1_stock/rapl_access.sh grant), see docs/guide/hardware/index.md §4")
            (problems if a.require_rapl else notes).append(msg)
        if any(c.startswith("R2-") for c in a.configs):
            if not a.enable_scx:
                problems.append("R2 configs requested without --enable-scx")
            if not priv["root"]:
                problems.append("R2 (sched_ext attach) needs root: run the driver with sudo")
        if "R3-lpmd" in a.configs:
            if not a.enable_lpmd:
                problems.append("R3-lpmd requested without --enable-lpmd")
            if not priv["root"]:
                problems.append("R3 (intel-lpmd) needs root: run the driver with sudo")
        if (a.governor or a.epp or a.no_turbo is not None or any(s != "current" for s in a.smt)) \
                and not priv["root"] and not a.dry_run:
            problems.append("--governor/--epp/--no-turbo/--smt changes need root: run the driver with sudo")
        if any(c.startswith("NOP-") for c in a.configs) and not self.perf:
            notes.append("perf not found: NOP overhead uses wh_measure counters only")
        t = self.topo
        notes.append(f"topology: {t['model']}: P-cores {t['pcpus']}  E-cores {t['ecpus']}  online {t['online']} "
                     f"smt={t['smt']} (from {t['source']})")
        gov, epp, nt = freq_policy()
        notes.append(f"frequency policy: governor={gov} epp={epp} no_turbo={nt} smt={smt_state()}")
        fixing = bool(a.governor or a.epp or a.no_turbo is not None)
        if not fixing and ("|" in gov or "|" in epp):
            msg = f"CPUs have different governors/EPPs ({gov} / {epp}): fix them (--governor/--epp) first"
            (notes if a.allow_unfixed_governor else problems).append(msg)
        if a.reps < 10:
            notes.append(f"--reps {a.reps} < 10: functional/smoke run only, not a measurement for the paper")
        elif not fixing and gov != "performance":
            msg = (f"governor is '{gov}' and the campaign does not fix it: pass --governor performance "
                   "(+ --epp/--no-turbo, root) or set it yourself (hw/baselines/r1_stock/host_setup.sh), "
                   "or --allow-unfixed-governor to record it only")
            (notes if a.allow_unfixed_governor else problems).append(msg)
        return problems, notes

    def check_policy(self):
        """Read back the frequency policy after --governor/--epp/--no-turbo and record it.

        Raises:
            SystemExit: If a requested setting is not in effect and
                ``--allow-unfixed-governor`` is not given.
        """
        a = self.a
        gov, epp, nt = freq_policy()
        bad = []
        if a.governor and gov != a.governor:
            bad.append(f"governor={gov} (wanted {a.governor})")
        if a.epp and epp != a.epp:
            bad.append(f"epp={epp} (wanted {a.epp})")
        if a.no_turbo is not None and nt != str(a.no_turbo):
            bad.append(f"no_turbo={nt} (wanted {a.no_turbo})")
        if bad and not a.allow_unfixed_governor:
            raise SystemExit("frequency policy not applied: " + ", ".join(bad))
        self.policy0 = (gov, epp, nt)

    def run(self):
        """Run the whole campaign.

        Steps: preflight, system setup, all SMT states/groups/reps, restore, summary.

        Runs within a repetition are shuffled (seeded by ``--seed`` and the rep) with
        R1 first, so the reference output exists. Saved host settings are restored
        even on failure. Without ``--dry-run`` the results are summarised and, under
        sudo, chowned back to the repository owner.

        Raises:
            SystemExit: If preflight finds problems (non-dry run) or a system change is refused.
        """
        a = self.a
        problems, notes = self.preflight()
        for n in notes:
            print("note:", n)
        for p in problems:
            print("PROBLEM:", p)
        if problems and not a.dry_run:
            sys.exit("preflight failed (use --dry-run to see the plan)")
        groups = self.plan()
        sysdir = self.out / "system"
        state_file = sysdir / "host_state.txt"
        if not a.dry_run:
            sysdir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["bash", str(BASE / "r1_stock/record_system.sh"),
                            str(sysdir / f"system_{int(time.time())}.txt")], check=False)
            (sysdir / "topology.json").write_text(json.dumps(self.topo, indent=1) + "\n")
            if a.governor or a.epp or a.no_turbo is not None:
                if not a.allow_system_changes:
                    sys.exit("--governor/--epp/--no-turbo need --allow-system-changes")
                subprocess.run(["bash", str(BASE / "r1_stock/host_setup.sh"), "save", str(state_file)], check=True)
                cmd = ["bash", str(BASE / "r1_stock/host_setup.sh"), "apply"]
                if a.governor:
                    cmd += ["--governor", a.governor]
                if a.epp:
                    cmd += ["--epp", a.epp]
                if a.no_turbo is not None:
                    cmd += ["--no-turbo", str(a.no_turbo)]
                subprocess.run(cmd, env=dict(os.environ, WINHINT_ALLOW_SYSTEM_CHANGES="1"), check=True)
                self.sys_changed = True
            elif a.allow_system_changes and a.smt != ["current"]:
                subprocess.run(["bash", str(BASE / "r1_stock/host_setup.sh"), "save", str(state_file)], check=True)
        try:
            if not a.dry_run:
                self.check_policy()
            for smt_want in a.smt:
                smt = smt_state() if a.dry_run else self.set_smt(smt_want)
                if a.dry_run and smt_want != "current":
                    smt = smt_want
                if not a.dry_run:
                    self.bench_migration(sysdir, smt)
                for group, configs in groups:
                    start, stop = self.group_ctx(group)
                    if start and not a.dry_run:
                        start()
                    try:
                        for rep in range(a.reps):
                            items = list(itertools.product(a.kernels, configs))
                            random.Random(a.seed * 1000 + rep).shuffle(items)
                            # R1 first in each rep so the reference output exists
                            items.sort(key=lambda kc: kc[1] != "R1")
                            for kernel, config in items:
                                self.run_one(kernel, config, group, smt, rep)
                    finally:
                        if stop and not a.dry_run:
                            stop()
        finally:
            if self.sys_changed and state_file.exists():
                subprocess.run(["bash", str(BASE / "r1_stock/host_setup.sh"), "restore", str(state_file)],
                               env=dict(os.environ, WINHINT_ALLOW_SYSTEM_CHANGES="1"), check=False)
        if not a.dry_run:
            summarize(self.out)
            chown_results(self.out)


def intersect_cpulist(a, b):
    """Return the intersection of two CPU lists (e.g. ``"0-3,8"``).

    Args:
        a (str): CPU list string.
        b (str): CPU list string.

    Returns:
        (str): Sorted comma-separated CPU numbers present in both (ranges expanded).
    """
    def parse(s):
        """Expand a CPU list string into a set of CPU numbers."""
        out = set()
        for part in s.split(","):
            if not part:
                continue
            if "-" in part:
                x, y = part.split("-")
                out.update(range(int(x), int(y) + 1))
            else:
                out.add(int(part))
        return out
    return ",".join(str(c) for c in sorted(parse(a) & parse(b)))


# x86 NOP hint (docs/interfaces.md §2): nopl DISP32(%rax) = 0F 1F 80 <disp32 LE>,
# DISP32 = 0x57480000 | kind << 12 | payload  ->  bytes 0F 1F 80 b0 b1 48 57.
#: Regex of the hint byte pattern, compiled lazily by x86_nop_hints().
_X86_HINT_RE = None


def x86_nop_hints(path):
    """Count the WinHint x86 NOP hints in a binary (static scan).

    The NOPs are architectural no-ops: on real hardware they only cost fetch/decode, which
    is what the NOP-asm-* overhead rows measure; libwinhint acts only on the call-mode
    hooks (`__winhint_setwin`/`__winhint_region`).

    Args:
        path (str | Path): Binary to scan.

    Returns:
        (dict[tuple[int, int], int]): ``{(kind, payload): count}`` decoded from the low 16 bits of
            the displacement
            (``kind = lo >> 12``, ``payload = lo & 0xFFF``); empty if unreadable.
    """
    global _X86_HINT_RE
    if _X86_HINT_RE is None:
        import re
        _X86_HINT_RE = re.compile(rb"\x0f\x1f\x80(..)\x48\x57", re.S)
    out = {}
    try:
        data = Path(path).read_bytes()
    except OSError:
        return out
    for m in _X86_HINT_RE.finditer(data):
        lo = int.from_bytes(m.group(1), "little")
        key = (lo >> 12, lo & 0xFFF)
        out[key] = out.get(key, 0) + 1
    return out


def privileges():
    """Return what this process may do (host, no containers).

    Returns:
        (dict): Dict with ``root``, ``rapl_readable`` (package-0 ``energy_uj``),
            ``perf_event_paranoid``, ``own_process_perf`` (root or paranoid <= 2) and
            ``sched_ext`` (sysfs state or ``"unavailable"``).
    """
    def readable(path):
        """Return whether ``path`` can be opened and read."""
        try:
            with open(path) as f:
                f.read(32)
            return True
        except OSError:
            return False
    paranoid = rd("/proc/sys/kernel/perf_event_paranoid", "?")
    return {
        "root": os.geteuid() == 0,
        "rapl_readable": readable("/sys/class/powercap/intel-rapl:0/energy_uj"),
        "perf_event_paranoid": paranoid,
        "own_process_perf": os.geteuid() == 0 or (paranoid.lstrip("-").isdigit() and int(paranoid) <= 2),
        "sched_ext": rd("/sys/kernel/sched_ext/state", "unavailable"),
    }


def chown_results(out):
    """Hand the results back to the repository owner when running as root (under sudo).

    Args:
        out (Path): Results directory (Path); it and everything below it are chowned to
            the owner of the repository root. Errors are ignored.
    """
    if os.geteuid() != 0:
        return
    st = ROOT.stat()
    for p in [out] + list(out.rglob("*")):
        try:
            os.chown(p, st.st_uid, st.st_gid)
        except OSError:
            pass


# ----------------------------------------------------------------------------- summary
def nop_overhead(hinted, plain):
    """Return the NOP-hint overhead of ``mean(hinted)/mean(plain) - 1`` with its 95 % CI.

    The CI half-width uses first-order error propagation of the two independent CIs.

    Args:
        hinted (tuple[float, float, float] | None): ``(mean, lo, hi)`` of the hinted runs, or a
            falsy value.
        plain (tuple[float, float, float] | None): ``(mean, lo, hi)`` of the plain runs, or a falsy
            value.

    Returns:
        (tuple[float | str, float | str]): ``(overhead_pct, ci_half_width_pct)``; empty strings if a
            side or its mean is
            missing, NaN half-width if a CI bound is NaN.
    """
    if not hinted or not plain:
        return "", ""
    (mh, loh, hih), (mp, lop, hip) = hinted, plain
    if any(x is None or math.isnan(x) for x in (mh, mp)) or not mp:
        return "", ""
    ov = 100 * (mh / mp - 1)
    if any(math.isnan(x) for x in (loh, hih, lop, hip)) or not mh:
        return ov, float("nan")
    rh, rp = (hih - loh) / 2 / mh, (hip - lop) / 2 / mp
    return ov, 100 * (mh / mp) * math.sqrt(rh * rh + rp * rp)


def summarize(out):
    """Aggregate ``raw.csv`` into ``summary.csv``.

    Successful runs are grouped by (kernel, config, smt); each metric gets mean
    and 95 % CI, ``wall_s``/``energy_pkg_j``/``edp_pkg_js`` are normalised to R1
    (``*_vs_R1``), and NOP-asm-* rows get the overhead vs NOP-plain-* on the same
    core type (``nop_overhead*_pct``, ``*_ci_pct``, ``nop_within_noise``).

    Args:
        out (str | Path): Results directory containing ``raw.csv``.
    """
    raw = Path(out) / "raw.csv"
    if not raw.exists():
        print("no raw.csv yet")
        return
    rows = [r for r in csv.DictReader(open(raw)) if r["exit_code"] == "0"]
    groups = {}
    for r in rows:
        groups.setdefault((r["kernel"], r["config"], r["smt"]), []).append(r)
    metrics = ["wall_s", "energy_pkg_j", "energy_core_j", "edp_pkg_js", "edp_core_js",
               "instructions_p", "instructions_e", "cycles_p", "cycles_e", "perf_stat_instructions",
               "perf_stat_cycles"]
    means, cis = {}, {}
    out_rows = []
    for (k, c, s), rs in sorted(groups.items()):
        row = {"kernel": k, "config": c, "smt": s, "n": len(rs), "n_ge_10": int(len(rs) >= 10),
               "output_ok": all(r["output_ok"] in ("1", "") for r in rs)}
        mig = [fnum(r["migrations"]) for r in rs if fnum(r["migrations"]) is not None]
        row["migrations_mean"] = statistics.fmean(mig) if mig else ""
        for m in metrics:
            mu, lo, hi = mean_ci([fnum(r[m]) for r in rs])
            row[m + "_mean"], row[m + "_ci_lo"], row[m + "_ci_hi"] = mu, lo, hi
            means[(k, c, s, m)] = mu
            cis[(k, c, s, m)] = (mu, lo, hi)
        out_rows.append(row)
    for row in out_rows:
        for m in ("wall_s", "energy_pkg_j", "edp_pkg_js"):
            ref = means.get((row["kernel"], "R1", row["smt"], m))
            v = row[m + "_mean"]
            row[m + "_vs_R1"] = v / ref if ref and not math.isnan(ref) and not math.isnan(v) else ""
        # NOP overhead: asm vs plain on the same core type (wall time; cycles/instructions from
        # perf stat). "within noise" = the 95% CI of the ratio (first-order) contains 0.
        if row["config"].startswith("NOP-asm-"):
            side = row["config"][-1]
            key = (row["kernel"], f"NOP-plain-{side}", row["smt"])
            for m, tag in (("wall_s", ""), ("perf_stat_cycles", "_cycles"),
                           ("perf_stat_instructions", "_instructions")):
                ov, half = nop_overhead(cis.get((row["kernel"], row["config"], row["smt"], m)),
                                        cis.get(key + (m,)))
                row["nop_overhead" + tag + "_pct"] = ov
                row["nop_overhead" + tag + "_ci_pct"] = half
                if not tag:
                    row["nop_within_noise"] = "" if half in ("", None) or math.isnan(half) \
                        else int(abs(ov) <= half)
    if not out_rows:
        print("no successful runs to summarize")
        return
    fields = list(out_rows[0].keys())
    for r in out_rows:
        for f in r:
            if f not in fields:
                fields.append(f)
    with open(Path(out) / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"summary: {len(out_rows)} groups -> {Path(out) / 'summary.csv'}")


# ----------------------------------------------------------------------------- CLI
def main():
    """Parse the command line and run the campaign or another mode.

    Other modes: ``--summarize-only``, ``--uarch`` and ``--fidelity``.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernels", default="all", help="comma list or 'all' (benchmarks/*.c)")
    ap.add_argument("--synthetic", action="store_true",
                    help="use the synthetic phase workload (build/hw/phase_workload*) as the only kernel")
    ap.add_argument("--synthetic-args", default="1 50 64",
                    help="phase_workload arguments: seconds phase_ms mem_mb")
    ap.add_argument("--llamacpp", action="store_true",
                    help="use llama.cpp llama-simple (hw/integrations/llamacpp) as the only kernel")
    ap.add_argument("--llamacpp-dir", default=str(BUILD / "integrations" / "llamacpp"))
    ap.add_argument("--llamacpp-args", default="",
                    help="llama-simple arguments (default: " + LLAMACPP_ARGS.format(dir="<llamacpp-dir>") + ")")
    ap.add_argument("--uarch", action="store_true",
                    help="run the µarch window probes (hw/tools/uarch_probe) on a P and an E CPU; no campaign")
    ap.add_argument("--uarch-smoke", action="store_true", help="with --uarch: tiny ranges (functional only)")
    ap.add_argument("--fidelity", action="store_true",
                    help="R4/R5 baseline-fidelity trend checks on phase_workload (hw/fidelity.py); no campaign")
    ap.add_argument("--fidelity-secs", type=float, default=4.0, help="seconds per fidelity run")
    ap.add_argument("--fidelity-reps", type=int, default=3, help="pinned ground-truth runs per class and side")
    ap.add_argument("--configs", default="core",
                    help="comma list; 'core' = " + ",".join(CORE_CONFIGS) +
                         "; 'all' = core + R2-{" + ",".join(SCX_PRESETS) + "} + R3-lpmd")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--input", default="large", choices=["small", "large"])
    ap.add_argument("--smt", default="current", help="comma list of on,off,current")
    ap.add_argument("--out", default=str(ROOT / "results" / "hw"))
    ap.add_argument("--bench-root", default=str(BUILD / "benchmarks" / "x86"))
    ap.add_argument("--variant-plain", default="plain")
    ap.add_argument("--variant-asm", default="winhint", help="x86 NOP-hint build")
    ap.add_argument("--variant-call", default="winhint_call", help="WinHint call-mode build (libwinhint)")
    ap.add_argument("--variant-regions", default="oracle_call", help="region(id) markers, call mode (R5)")
    ap.add_argument("--tool-dir", default=str(BUILD / "hw"))
    ap.add_argument("--lib-dir", default=str(BUILD / "libwinhint"))
    ap.add_argument("--hw-baselines", default=str(BUILD / "hw-baselines"))
    ap.add_argument("--threshold", type=int, default=192, help="WINHINT_THRESHOLD (W >= thr -> P)")
    ap.add_argument("--min-dwell-us", type=int, default=0)
    ap.add_argument("--hyst", type=int, default=1)
    ap.add_argument("--winhint-perf", action="store_true",
                    help="per-region perf counters in WH/WH-off runs (adds perf_event_open start-up cost)")
    ap.add_argument("--allow-unfixed-governor", action="store_true",
                    help="only record the frequency policy (no fixed-governor check)")
    ap.add_argument("--sondag-k", type=int, default=2)
    ap.add_argument("--sondag-threshold", type=float, default=1.4)
    ap.add_argument("--sondag-types-dir", default="", help="dir with <kernel>.types (region_types.py)")
    ap.add_argument("--pie-backend", default="perf", choices=["perf", "pmctrack"])
    ap.add_argument("--pie-interval-ms", type=float, default=10)
    ap.add_argument("--pie-slack", type=float, default=0.15)
    ap.add_argument("--pie-hyst", type=int, default=2)
    ap.add_argument("--nop-pcpu", type=int, default=None)
    ap.add_argument("--nop-ecpu", type=int, default=None)
    ap.add_argument("--enable-scx", action="store_true", help="allow R2 (attaches sched_ext system-wide)")
    ap.add_argument("--enable-lpmd", action="store_true", help="allow R3 (starts intel-lpmd)")
    ap.add_argument("--lpmd-mode", default="AUTO")
    ap.add_argument("--governor", default="", help="fix the cpufreq governor (needs --allow-system-changes)")
    ap.add_argument("--epp", default="", help="fix EPP (needs --allow-system-changes)")
    ap.add_argument("--no-turbo", type=int, choices=[0, 1], default=None)
    ap.add_argument("--allow-system-changes", action="store_true",
                    help="permit governor/EPP/turbo/SMT changes (restored at the end)")
    ap.add_argument("--cooldown", type=float, default=2.0, help="seconds idle before each run")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--require-rapl", dest="require_rapl", action="store_true", default=True)
    ap.add_argument("--allow-no-rapl", dest="require_rapl", action="store_false")
    ap.add_argument("--no-perf-stat", action="store_true")
    ap.add_argument("--no-bench-migration", dest="bench_migration", action="store_false", default=True)
    ap.add_argument("--bench-migration-n", type=int, default=200)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    if a.synthetic and a.llamacpp:
        ap.error("--synthetic and --llamacpp are exclusive")
    if a.uarch or a.fidelity:
        import fidelity
        if a.uarch:
            fidelity.run_uarch(a)
        if a.fidelity:
            fidelity.run_fidelity(a)
        return
    if a.synthetic:
        a.kernels = [SYNTHETIC]
    elif a.llamacpp:
        a.kernels = [LLAMACPP]
    else:
        a.kernels = ALL_KERNELS if a.kernels == "all" else [k for k in a.kernels.split(",") if k]
    a.configs_given = a.configs not in ("core", "all")
    if a.configs in ("core", "all"):
        cfg = list(CORE_CONFIGS)
        if a.configs == "all":
            cfg += ["R2-" + p for p in SCX_PRESETS] + ["R3-lpmd"]
        a.configs = cfg
    else:
        a.configs = [c for c in a.configs.split(",") if c]
    known = set(CORE_CONFIGS) | {"R3-lpmd"}
    for c in a.configs:
        if c not in known and not c.startswith("R2-"):
            ap.error(f"unknown config {c}")
    if a.llamacpp:
        nop = [c for c in a.configs if c.startswith("NOP-")]
        if nop and a.configs_given:
            ap.error(f"--llamacpp has no NOP-hint build (NOP-* rows): drop {','.join(nop)}")
        a.configs = [c for c in a.configs if not c.startswith("NOP-")]
    a.smt = [s for s in a.smt.split(",") if s]
    if a.summarize_only:
        summarize(Path(a.out))
        return
    Campaign(a).run()


if __name__ == "__main__":
    main()
