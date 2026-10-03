r"""
se.py — gem5 SE-mode config for the WinHint project (RISC-V out-of-order core).

Usage (host, conda env ``winhint``):

    $WINHINT_BUILD/gem5/src/build/RISCV_winhint/gem5.opt --outdir=<dir> \
        $WINHINT_ROOT/sim/se.py \
        --machine $WINHINT_ROOT/sim/machines/riscv_ooo.json \
        --cmd $WINHINT_BUILD/benchmarks/riscv/plain/<kernel> --options large \
        --window-policy hint --window-trace

Every hardware parameter defaults to the machine JSON (``--machine``, see
docs/interfaces.md §3/§5). Any explicit command-line flag overrides the JSON.

Window resizing (RISCV_winhint build only, docs/interfaces.md §4):
    --window-policy {static,occupancy,mlp,bbv,lut,hint,hybrid,ltp}
    --window-initial N        starting config index (default: JSON window.initial)
    --window-period CYCLES    sampling period for the reactive policies
    --window-lut FILE         B5 runtime LUT (window_policy=lut)
    --window-trace            write window_trace.csv to the outdir
    --window-args "k=v,..."   policy tunables (see docs/guide/gem5/patches.md);
                              e.g. hint with structs=iq for B6 (IQ only)

On the RISCV_clean build only ``--window-policy static`` is accepted; the
selected config is then applied physically (ROB/IQ/LQ/SQ sized to it).

Run length (``large`` inputs are 0.2-14 G instructions; see sim/run_lengths.json
for the method and the per-kernel budgets, and sim/run_lengths.py):
    --maxinsts M              measured instructions (after any fast-forward/warm-up)
    --warmup-insts W          detailed O3 warm-up before the measurement; the
                              stats are reset at its end (m5.stats.reset())
    --fast-forward N          in-process: AtomicSimpleCPU (functional, warms the
                              caches) for N instructions, then switch to the O3
                              CPU (``system.cpu``; stats names unchanged)
    --take-checkpoints N1,N2  functional AtomicSimpleCPU run (no caches) that writes
      --checkpoint-dir D      D/cpt.<Ni>/ (gem5 checkpoint + winhint_ff.json) at
                              each instruction count, then exits
    --restore-checkpoint C    restore C (a cpt.<N> dir) into the O3 CPU
    --window-seed auto|off    with --restore-checkpoint: hint-driven policies (hint,
                              hybrid) start in the configuration of the last
                              setwin executed before the checkpoint
    --profile-regions FILE    functional run that records every region(id) marker
                              visit (instruction count at entry) -> FILE (JSON)

Hint/region markers during functional execution: the window controller only sees
instructions committed by the O3 CPU. se.py scans the binary for hint PCs
(sim/winhint_elf.py) and counts their retirements with gem5's PcCountTracker
probes, so it knows the last setwin/region executed before a checkpoint or a
switch. With --restore-checkpoint the last setwin becomes the hint policy's
starting configuration (exact: the policy state is a function of the last
setwin only; hybrid also restores its window but its ceiling is re-learned at
the next setwin). With in-process --fast-forward the O3 parameters are fixed
before the fast-forward runs, so the hint state is exact only if a setwin commits
during the warm-up; runlength.json records whether it did (hint_state_exact).
Region markers seen before the switch are not replayed: the controller attributes
cycles from the first marker committed on the O3 CPU.

Every run with fast-forward, warm-up, checkpoint or restore writes
runlength.json to the outdir (instruction counts, measure start tick/cycle,
seed, setwin/region counts).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import winhint_elf  # noqa: E402  (stdlib only; hint PCs of the binary)

import m5  # noqa: E402
from m5.objects import (
    AddrRange,
    AtomicSimpleCPU,
    Cache,
    DDR3_1600_8x8,
    DDR4_2400_8x8,
    L2XBar,
    LPDDR3_1600_1x32,
    MemCtrl,
    MinorCPU,
    Process,
    Root,
    SEWorkload,
    SimpleMemory,
    SrcClockDomain,
    StridePrefetcher,
    System,
    SystemXBar,
    TimingSimpleCPU,
    VoltageDomain,
)

try:  # gem5 >= 23: ISA-specific CPU classes
    from m5.objects import RiscvO3CPU as O3CPUClass
except ImportError:  # pragma: no cover
    from m5.objects import DerivO3CPU as O3CPUClass

try:
    from m5.objects import RiscvAtomicSimpleCPU as AtomicSimpleCPU  # noqa: F811
    from m5.objects import RiscvMinorCPU as MinorCPU  # noqa: F811
    from m5.objects import RiscvTimingSimpleCPU as TimingSimpleCPU  # noqa: F811
except ImportError:  # pragma: no cover
    pass

#: values accepted by --window-policy (docs/interfaces.md §4)
WINDOW_POLICIES = ["static", "occupancy", "mlp", "bbv", "lut", "hint", "hybrid",
                   "ltp"]
#: policies whose state is set by setwin hints (seeded across a checkpoint)
SEED_POLICIES = ("hint", "hybrid")
#: count that is never reached: PcCountTracker targets that only count
NEVER = 2 ** 63
#: per-checkpoint hint-state file written into each cpt.<N>/ directory
FF_STATE = "winhint_ff.json"

# ---------------------------------------------------------------------------
# Machine JSON
# ---------------------------------------------------------------------------

# Built-in fallback (matches sim/machines/riscv_ooo.json) used when no
# --machine is given.
DEFAULT_MACHINE = {
    "cpu": {
        "type": "DerivO3CPU", "clock": "2GHz",
        "fetch_width": 4, "decode_width": 4, "rename_width": 4,
        "dispatch_width": 4, "issue_width": 4, "commit_width": 4,
        "squash_width": 4, "wb_width": 8,
        "num_int_regs": 288, "num_fp_regs": 288,
        "num_rob_entries": 256, "num_iq_entries": 128,
        "lq_entries": 64, "sq_entries": 64,
    },
    "cache": {
        "line_size": 64,
        "l1i": {"size": "32kB", "assoc": 4, "tag_latency": 1,
                "data_latency": 1, "response_latency": 1, "mshrs": 8,
                "tgts_per_mshr": 20},
        "l1d": {"size": "32kB", "assoc": 8, "tag_latency": 2,
                "data_latency": 2, "response_latency": 1, "mshrs": 16,
                "tgts_per_mshr": 20},
        "l2": {"size": "1MB", "assoc": 16, "tag_latency": 10,
               "data_latency": 10, "response_latency": 4, "mshrs": 32,
               "tgts_per_mshr": 12},
        "l1d_prefetcher": "none",
    },
    "memory": {"type": "DDR4_2400_8x8", "size": "512MB",
               "extra_latency_ns": 0},
    "window": {"rob": [64, 128, 192, 256], "iq": [32, 64, 96, 128],
               "lq": [16, 32, 48, 64], "sq": [16, 32, 48, 64],
               "initial": 3, "period": 1000},
}


def load_machine(path):
    """Load the machine JSON, or the built-in default when no path is given.

    Args:
        path (str): Path to a ``sim/machines/*.json`` file; empty/None selects
            ``DEFAULT_MACHINE``.

    Returns:
        machine (dict): The machine description as a dict (``cpu``, ``cache``, ``memory``,
            ``window`` sections).
    """
    if not path:
        return DEFAULT_MACHINE
    with open(path) as f:
        return json.load(f)


def machine_defaults(m):
    """Map the machine JSON onto argparse defaults (dest names).

    For each argparse destination the first non-None candidate value wins;
    keys missing from the JSON leave the parser's own default in place. The
    physical ROB/IQ/LQ/SQ sizes fall back to the largest entry of the
    corresponding ``window`` list when ``cpu`` does not set them.

    Args:
        m (dict): Machine description as returned by :func:`load_machine`.

    Returns:
        defaults (dict): Dict of ``{dest: value}`` suitable for ``ArgumentParser.set_defaults``.
    """
    cpu = m.get("cpu", {})
    cache = m.get("cache", {})
    mem = m.get("memory", {})
    win = m.get("window", {})
    l1i = cache.get("l1i", {})
    l1d = cache.get("l1d", {})
    l2 = cache.get("l2", {})
    d = {}

    def put(dest, *vals):
        """Set ``d[dest]`` to the first non-None value in ``vals``, if any."""
        for v in vals:
            if v is not None:
                d[dest] = v
                return

    put("sys_clock", cpu.get("clock"))
    for k in ("fetch", "decode", "rename", "dispatch", "issue", "commit",
              "squash", "wb"):
        put(f"{k}_width", cpu.get(f"{k}_width"))
    put("num_phys_int_regs", cpu.get("num_int_regs"))
    put("num_phys_fp_regs", cpu.get("num_fp_regs"))
    put("num_rob_entries", cpu.get("num_rob_entries"),
        max(win["rob"]) if win.get("rob") else None)
    put("num_iq_entries", cpu.get("num_iq_entries"),
        max(win["iq"]) if win.get("iq") else None)
    put("lq_entries", cpu.get("lq_entries"),
        max(win["lq"]) if win.get("lq") else None)
    put("sq_entries", cpu.get("sq_entries"),
        max(win["sq"]) if win.get("sq") else None)

    put("cacheline_size", cache.get("line_size"), cache.get("cache_line_size"))
    put("l1i_size", l1i.get("size"), cache.get("l1i_size"))
    put("l1d_size", l1d.get("size"), cache.get("l1d_size"))
    put("l2_size", l2.get("size"), cache.get("l2_size"))
    put("l1i_assoc", l1i.get("assoc"))
    put("l1d_assoc", l1d.get("assoc"))
    put("l2_assoc", l2.get("assoc"))
    put("l1d_mshrs", l1d.get("mshrs"))
    put("l2_mshrs", l2.get("mshrs"))
    put("l1d_prefetcher", cache.get("l1d_prefetcher"))

    put("mem_type", mem.get("type"), mem.get("mem_type"))
    put("mem_size", mem.get("size"), mem.get("mem_size"))
    put("mem_extra_latency_ns", mem.get("extra_latency_ns"))

    put("window_initial", win.get("initial"))
    put("window_period", win.get("period"))
    return d


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the se.py command-line parser with hard-coded fallback defaults.

    The defaults here are overridden by the machine JSON in :func:`parse_args`.

    Returns:
        The configured parser (workload, CPU, cache, memory, window and
        run-length options).
    """
    p = argparse.ArgumentParser(
        description="WinHint gem5 SE config — RISC-V OoO pipeline")

    p.add_argument("--machine", default="",
                   help="Machine JSON (sim/machines/*.json); provides defaults")

    # ── Workload ──────────────────────────────────────────────────────────
    p.add_argument("--cmd", required=True, help="Path to RISC-V binary")
    p.add_argument("--options", default="", help="Arguments for the binary")
    p.add_argument("--input", default="", help="stdin redirect")
    p.add_argument("--output", default="", help="stdout redirect (outdir-relative)")
    p.add_argument("--errout", default="", help="stderr redirect (outdir-relative)")

    # ── CPU ───────────────────────────────────────────────────────────────
    p.add_argument("--cpu-type", default="DerivO3CPU",
                   choices=["AtomicSimpleCPU", "TimingSimpleCPU", "MinorCPU",
                            "DerivO3CPU", "O3CPU"])
    p.add_argument("--sys-clock", default="2GHz")
    p.add_argument("--num-cpus", default=1, type=int)

    for k, v in (("fetch", 4), ("decode", 4), ("issue", 4), ("commit", 4),
                 ("squash", 4), ("rename", 4), ("dispatch", 4), ("wb", 8)):
        p.add_argument(f"--{k}-width", default=v, type=int)

    # Physical structure sizes (the largest window config by default).
    p.add_argument("--num-rob-entries", default=256, type=int)
    p.add_argument("--num-iq-entries", default=128, type=int)
    p.add_argument("--lsq-size", default=None, type=int,
                   help="Sets both LQ and SQ (legacy flag)")
    p.add_argument("--lq-entries", default=64, type=int)
    p.add_argument("--sq-entries", default=64, type=int)
    p.add_argument("--num-phys-int-regs", default=288, type=int)
    p.add_argument("--num-phys-fp-regs", default=288, type=int)

    # ── Caches ────────────────────────────────────────────────────────────
    p.add_argument("--caches", action="store_true",
                   help="Attach L1 caches (implied by --machine)")
    p.add_argument("--l2cache", action="store_true",
                   help="Attach unified L2 (implied by --machine)")
    p.add_argument("--no-caches", action="store_true",
                   help="Disable caches even with --machine")
    p.add_argument("--l1i_size", "--l1i-size", dest="l1i_size", default="32kB")
    p.add_argument("--l1d_size", "--l1d-size", dest="l1d_size", default="32kB")
    p.add_argument("--l1i_assoc", "--l1i-assoc", dest="l1i_assoc", default=4, type=int)
    p.add_argument("--l1d_assoc", "--l1d-assoc", dest="l1d_assoc", default=8, type=int)
    p.add_argument("--l2_size", "--l2-size", dest="l2_size", default="1MB")
    p.add_argument("--l2_assoc", "--l2-assoc", dest="l2_assoc", default=16, type=int)
    p.add_argument("--l1d-mshrs", default=16, type=int)
    p.add_argument("--l2-mshrs", default=32, type=int)
    p.add_argument("--cacheline_size", "--cacheline-size", dest="cacheline_size",
                   default=64, type=int)
    p.add_argument("--l1d-prefetcher", default="none", choices=["none", "stride"])

    # ── Memory ────────────────────────────────────────────────────────────
    p.add_argument("--mem-type", default="DDR4_2400_8x8",
                   choices=["DDR3_1600_8x8", "DDR4_2400_8x8",
                            "LPDDR3_1600_1x32", "SimpleMemory"])
    p.add_argument("--mem-size", default="512MB")
    p.add_argument("--mem-extra-latency-ns", default=0, type=float,
                   help="Added to the memory controller frontend latency")

    # ── Window resizing (RISCV_winhint) ───────────────────────────────────
    p.add_argument("--window-policy", default="static", choices=WINDOW_POLICIES)
    p.add_argument("--window-initial", default=None, type=int,
                   help="Starting config index (default: machine JSON "
                        "window.initial, else the largest)")
    p.add_argument("--window-period", default=1000, type=int,
                   help="Sampling period in cycles")
    p.add_argument("--window-lut", default="", help="B5 runtime LUT file")
    p.add_argument("--window-trace", action="store_true",
                   help="Write window_trace.csv to the outdir")
    p.add_argument("--window-args", default="",
                   help="Policy tunables, 'key=value,key=value'")
    p.add_argument("--no-window", action="store_true",
                   help="Do not pass the window table to gem5 at all")

    # ── Simulation limits / run length (sim/run_lengths.json) ─────────────
    p.add_argument("-I", "--maxinsts", default=0, type=int,
                   help="Stop after N committed (measured) instructions (0 = no "
                        "limit); counted after --fast-forward/--warmup-insts")
    p.add_argument("--warmup-insts", default=0, type=int,
                   help="Detailed warm-up instructions before the measurement "
                        "(stats are reset at its end)")
    p.add_argument("--fast-forward", default=0, type=int,
                   help="Functional AtomicSimpleCPU instructions before switching "
                        "to the O3 CPU (in-process)")
    p.add_argument("--take-checkpoints", default="",
                   help="Comma list of instruction counts: functional run that "
                        "writes a checkpoint at each (needs --checkpoint-dir)")
    p.add_argument("--checkpoint-dir", default="",
                   help="Directory for --take-checkpoints (cpt.<N>/ subdirs)")
    p.add_argument("--restore-checkpoint", default="",
                   help="Restore this checkpoint (a cpt.<N> dir) into the O3 CPU")
    p.add_argument("--window-seed", default="auto", choices=["auto", "off"],
                   help="With --restore-checkpoint: start hint/hybrid in the config "
                        "of the last setwin executed before the checkpoint")
    p.add_argument("--profile-regions", default="",
                   help="Functional run recording every region(id) marker visit "
                        "into this JSON file")
    p.add_argument("--profile-cap", default=2048, type=int,
                   help="Visits recorded per region-marker PC (--profile-regions)")
    return p


def parse_args(argv=None):
    """Parse the command line, applying the machine JSON as defaults.

    A first pass reads only ``--machine``; its JSON is mapped onto the full
    parser's defaults so explicit flags still override it. Also resolves
    ``--lsq-size``, enables L1/L2 caches when a machine is given (unless
    ``--no-caches``), stores the machine dict in ``args.machine_cfg`` and
    validates the run-length flags via :func:`_check_run_length`.

    Args:
        argv (list[str] | None): Argument list; None means ``sys.argv[1:]``.

    Returns:
        args (argparse.Namespace): The namespace, extended with ``machine_cfg``,
            ``mode``, ``checkpoints``, ``ff_state`` and (restore mode) ``seed``.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--machine", default="")
    known, _ = pre.parse_known_args(argv)
    machine = load_machine(known.machine)

    parser = build_parser()
    parser.set_defaults(**machine_defaults(machine))
    args = parser.parse_args(argv)
    if args.lsq_size is not None:
        args.lq_entries = args.sq_entries = args.lsq_size
    if args.machine and not args.no_caches:
        args.caches = True
        args.l2cache = True
    if args.no_caches:
        args.caches = args.l2cache = False
    args.machine_cfg = machine
    _check_run_length(parser, args)
    return args


def _check_run_length(parser, args):
    """Validate the run-length flags and derive ``args.mode``.

    ``args.mode`` is one of ``full``, ``ff`` (in-process fast-forward),
    ``restore``, ``checkpoint`` or ``profile``. Functional modes (``profile``,
    ``checkpoint``) force ``AtomicSimpleCPU`` without caches. In restore mode
    the checkpoint's ``winhint_ff.json`` is loaded into ``args.ff_state`` (None
    if absent) and :func:`seed_window` is applied.

    Args:
        parser (argparse.ArgumentParser): Parser used to report errors (``parser.error`` exits).
        args (argparse.Namespace): Parsed namespace; updated in place (``mode``, ``checkpoints``,
            ``ff_state``, possibly ``cpu_type``/``caches``/``l2cache``).
    """
    modes = [m for m, on in (("profile", args.profile_regions),
                             ("checkpoint", args.take_checkpoints),
                             ("restore", args.restore_checkpoint),
                             ("ff", args.fast_forward > 0)) if on]
    if len(modes) > 1:
        parser.error("--profile-regions, --take-checkpoints, --restore-checkpoint "
                     "and --fast-forward are mutually exclusive")
    args.mode = modes[0] if modes else "full"
    for f in ("maxinsts", "warmup_insts", "fast_forward", "profile_cap"):
        if getattr(args, f) < 0:
            parser.error(f"--{f.replace('_', '-')} must be >= 0")
    args.checkpoints = []
    if args.mode == "checkpoint":
        try:
            args.checkpoints = sorted({int(x) for x in
                                       args.take_checkpoints.split(",") if x.strip()})
        except ValueError:
            parser.error("--take-checkpoints expects a comma list of integers")
        if not args.checkpoints or args.checkpoints[0] <= 0:
            parser.error("--take-checkpoints needs instruction counts > 0")
        if not args.checkpoint_dir:
            parser.error("--take-checkpoints needs --checkpoint-dir")
    if args.mode in ("profile", "checkpoint"):
        # functional passes: atomic CPU, no caches (nothing timing-related is
        # kept; caches are not part of a gem5 checkpoint anyway)
        args.cpu_type = "AtomicSimpleCPU"
        args.caches = args.l2cache = False
        if args.maxinsts or args.warmup_insts:
            parser.error("--maxinsts/--warmup-insts do not apply to functional "
                         "--profile-regions/--take-checkpoints runs")
    if args.mode in ("ff", "restore") and not is_o3(args):
        parser.error(f"--{'fast-forward' if args.mode == 'ff' else 'restore-checkpoint'}"
                     " needs the O3 CPU (--cpu-type DerivO3CPU)")
    args.ff_state = None
    if args.mode == "restore":
        p = os.path.join(args.restore_checkpoint, FF_STATE)
        if os.path.isfile(p):
            with open(p) as fh:
                args.ff_state = json.load(fh)
        seed_window(args)


def seed_window(args):
    """Seed the starting window configuration from the checkpoint's last setwin.

    With ``--restore-checkpoint`` and ``--window-seed auto``, a hint-driven
    policy (``SEED_POLICIES``) starts in the configuration selected by the last
    setwin executed before the checkpoint (``winhint_elf.config_for_setwin``).
    Nothing is seeded if no setwin was recorded, ``--no-window`` is set or the
    machine has no ``window.rob`` list.

    Args:
        args (argparse.Namespace): Parsed namespace; sets ``args.seed`` (None if not seeded) and,
            when seeded, ``args.window_initial``.
    """
    args.seed = None
    st = args.ff_state or {}
    if (args.window_seed != "auto" or args.window_policy not in SEED_POLICIES
            or st.get("last_setwin") is None or args.no_window):
        return
    rob = args.machine_cfg.get("window", {}).get("rob", [])
    if not rob:
        return
    args.seed = winhint_elf.config_for_setwin(int(st["last_setwin"]), list(rob))
    args.window_initial = args.seed


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

#: --cpu-type name -> gem5 CPU class
CPU_MAP = {
    "AtomicSimpleCPU": AtomicSimpleCPU,
    "TimingSimpleCPU": TimingSimpleCPU,
    "MinorCPU": MinorCPU,
    "DerivO3CPU": O3CPUClass,
    "O3CPU": O3CPUClass,
}


def has_window_support():
    """Return True if the O3 CPU class has the WinHint window parameters.

    True on the ``RISCV_winhint`` build, False on ``RISCV_clean``.
    """
    return "window_policy" in O3CPUClass._params


def window_table(args):
    """Return the window table (``rob``/``iq``/``lq``/``sq`` lists) from the machine JSON.

    Args:
        args (argparse.Namespace): Parsed namespace with ``machine_cfg``.

    Returns:
        table_and_n (tuple[dict[str, list[int]], int]): ``(table, n)``: dict of
            the four per-config lists and the number of configurations.

    Raises:
        SystemExit: If the lists are empty or of unequal length.
    """
    win = args.machine_cfg.get("window", {})
    table = {k: list(win.get(k, [])) for k in ("rob", "iq", "lq", "sq")}
    n = len(table["rob"])
    if n == 0 or any(len(v) != n for v in table.values()):
        sys.exit("[se.py] machine JSON 'window' must have rob/iq/lq/sq lists "
                 "of equal, non-zero length")
    return table, n


def is_o3(args):
    """Return True if ``args.cpu_type`` selects the O3 CPU."""
    return args.cpu_type in ("DerivO3CPU", "O3CPU")


def set_iq_entries(cpu, entries):
    """Set the IQ size on an O3 CPU across gem5 versions.

    gem5 >= 25.1 splits the IQ into IQUnit objects (``cpu.instQueues``);
    older versions use ``cpu.numIQEntries``. A single IQ unit is used.

    Args:
        cpu (SimObject): O3 CPU SimObject.
        entries (int): Number of IQ entries.
    """
    if "instQueues" in type(cpu)._params:
        from m5.objects import IQUnit
        cpu.instQueues = [IQUnit(numEntries=entries)]
    else:
        cpu.numIQEntries = entries


def param_max_insts(args):
    """Return the ``max_insts_any_thread`` value for the CPU parameter.

    ``--maxinsts`` is passed as the CPU parameter only for a plain run (mode
    ``full`` without warm-up); after a fast-forward, a restore or a warm-up it
    is scheduled at run time instead (``scheduleInstStop``, relative to the
    start of the measurement), and 0 is returned.

    Args:
        args (argparse.Namespace): Parsed namespace.

    Returns:
        max_insts (int): The instruction limit, or 0 for none.
    """
    return args.maxinsts if args.mode == "full" and not args.warmup_insts else 0


def make_cpu(args, switched_out=False):
    # cpu_id 0 explicitly: switchCpus() requires equal ids on both CPUs
    """Create and configure the CPU selected by ``--cpu-type``.

    For the O3 CPU this sets pipeline widths, physical register counts and
    ROB/IQ/LQ/SQ sizes. Unless ``--no-window``: on the winhint build the
    physical structures are sized to at least the largest window config and the
    ``window_*`` parameters are set; on the clean build only ``static`` is
    accepted and the selected config is applied physically. An out-of-range
    ``window_initial`` is replaced by the largest config index.

    Args:
        args (argparse.Namespace): Parsed namespace; ``window_initial`` is normalised and ``phys``
            (dict of physical rob/iq/lq/sq) is set for the O3 CPU.
        switched_out (bool): Create the CPU switched out (in-process fast-forward).

    Returns:
        cpu (SimObject): The CPU SimObject (``cpu_id`` 0).

    Raises:
        SystemExit: If a non-static policy is requested on the clean build or
            with ``--no-window``, or the window table is malformed.
    """
    cpu = CPU_MAP[args.cpu_type](cpu_id=0, switched_out=switched_out)
    if not is_o3(args):
        if param_max_insts(args):
            cpu.max_insts_any_thread = param_max_insts(args)
        return cpu

    cpu.fetchWidth = args.fetch_width
    cpu.decodeWidth = args.decode_width
    cpu.issueWidth = args.issue_width
    cpu.commitWidth = args.commit_width
    cpu.squashWidth = args.squash_width
    cpu.renameWidth = args.rename_width
    cpu.dispatchWidth = args.dispatch_width
    cpu.wbWidth = args.wb_width
    cpu.numPhysIntRegs = args.num_phys_int_regs
    cpu.numPhysFloatRegs = args.num_phys_fp_regs

    rob, iq, lq, sq = (args.num_rob_entries, args.num_iq_entries,
                       args.lq_entries, args.sq_entries)

    if not args.no_window:
        table, n = window_table(args)
        initial = args.window_initial
        if initial is None or initial < 0 or initial >= n:
            initial = n - 1
        args.window_initial = initial
        if has_window_support():
            # Physical structures must hold the largest config.
            rob = max(rob, max(table["rob"]))
            iq = max(iq, max(table["iq"]))
            lq = max(lq, max(table["lq"]))
            sq = max(sq, max(table["sq"]))
            cpu.window_policy = args.window_policy
            cpu.window_rob = table["rob"]
            cpu.window_iq = table["iq"]
            cpu.window_lq = table["lq"]
            cpu.window_sq = table["sq"]
            cpu.window_initial = initial
            cpu.window_period = args.window_period
            cpu.window_lut_file = args.window_lut
            cpu.window_trace = args.window_trace
            cpu.window_args = args.window_args
        else:
            if args.window_policy != "static":
                sys.exit(f"[se.py] --window-policy {args.window_policy} needs "
                         "the RISCV_winhint build")
            # Clean build: apply the static config physically.
            rob, iq = table["rob"][initial], table["iq"][initial]
            lq, sq = table["lq"][initial], table["sq"][initial]
    elif args.window_policy != "static":
        sys.exit("[se.py] --no-window requires --window-policy static")

    cpu.numROBEntries = rob
    set_iq_entries(cpu, iq)
    cpu.LQEntries = lq
    cpu.SQEntries = sq
    args.phys = dict(rob=rob, iq=iq, lq=lq, sq=sq)

    if param_max_insts(args):
        cpu.max_insts_any_thread = param_max_insts(args)
    return cpu


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------


def _cache(spec, size, assoc, mshrs, defaults):
    """Create a gem5 ``Cache`` from a machine-JSON cache spec.

    Args:
        spec (dict): Cache section of the machine JSON (latencies, ``tgts_per_mshr``).
        size (str): Cache size string (e.g. ``"32kB"``).
        assoc (int): Associativity.
        mshrs (int): Number of MSHRs.
        defaults (dict): Fallback values for keys missing from ``spec``.

    Returns:
        cache (Cache): The configured ``Cache``.
    """
    c = Cache()
    c.size = size
    c.assoc = assoc
    c.tag_latency = spec.get("tag_latency", defaults["tag_latency"])
    c.data_latency = spec.get("data_latency", defaults["data_latency"])
    c.response_latency = spec.get("response_latency",
                                  defaults["response_latency"])
    c.mshrs = mshrs
    c.tgts_per_mshr = spec.get("tgts_per_mshr", defaults["tgts_per_mshr"])
    return c


def make_caches(args):
    """Create the L1I, L1D and L2 caches.

    Sizes/associativities/MSHRs come from ``args`` (machine JSON defaults or
    flags); latencies from the JSON with built-in fallbacks. The L1D optionally
    gets a ``StridePrefetcher`` (degree 4); the L2 is ``mostly_incl``.

    Args:
        args (argparse.Namespace): Parsed namespace with ``machine_cfg``.

    Returns:
        caches (tuple[Cache, Cache, Cache]): ``(l1i, l1d, l2)``.
    """
    cache = args.machine_cfg.get("cache", {})
    l1i = _cache(cache.get("l1i", {}), args.l1i_size, args.l1i_assoc,
                 cache.get("l1i", {}).get("mshrs", 8),
                 dict(tag_latency=1, data_latency=1, response_latency=1,
                      tgts_per_mshr=20))
    l1d = _cache(cache.get("l1d", {}), args.l1d_size, args.l1d_assoc,
                 args.l1d_mshrs,
                 dict(tag_latency=2, data_latency=2, response_latency=1,
                      tgts_per_mshr=20))
    l1d.writeback_clean = False
    if args.l1d_prefetcher == "stride":
        l1d.prefetcher = StridePrefetcher(degree=4)
    l2 = _cache(cache.get("l2", {}), args.l2_size, args.l2_assoc,
                args.l2_mshrs,
                dict(tag_latency=10, data_latency=10, response_latency=4,
                     tgts_per_mshr=12))
    l2.clusivity = "mostly_incl"
    l2.writeback_clean = True
    return l1i, l1d, l2


#: --mem-type name -> gem5 memory class
MEM_MAP = {
    "DDR3_1600_8x8": DDR3_1600_8x8,
    "DDR4_2400_8x8": DDR4_2400_8x8,
    "LPDDR3_1600_1x32": LPDDR3_1600_1x32,
    "SimpleMemory": SimpleMemory,
}


def build_system(args):
    """Build the gem5 ``System``: clock, CPU(s), caches, buses and memory.

    In ``ff`` mode the O3 CPU (``system.cpu``) starts switched out and an
    ``AtomicSimpleCPU`` (``system.ff_cpu``) owns the port connections until
    ``switchCpus``. On the winhint O3 build the L1D/L2 are also passed to the
    window controller (``window_l1d``/``window_l2``). ``--mem-extra-latency-ns``
    is added to the memory controller frontend latency (10 ns base) or to the
    ``SimpleMemory`` latency (30 ns base).

    Args:
        args (argparse.Namespace): Parsed namespace from :func:`parse_args`.

    Returns:
        system (System): The configured ``System``.
    """
    system = System()
    system.clk_domain = SrcClockDomain()
    system.clk_domain.clock = args.sys_clock
    system.clk_domain.voltage_domain = VoltageDomain()
    system.cache_line_size = args.cacheline_size

    ff = args.mode == "ff"
    system.mem_mode = ("atomic" if ff or args.cpu_type == "AtomicSimpleCPU"
                       else "timing")
    system.mem_ranges = [AddrRange(args.mem_size)]

    # In-process fast-forward: the O3 CPU keeps the name system.cpu (stats
    # names unchanged) but starts switched out; system.ff_cpu runs first and
    # owns the port connections until switchCpus() hands them over.
    system.cpu = make_cpu(args, switched_out=ff)
    if ff:
        system.ff_cpu = AtomicSimpleCPU(cpu_id=0)
    front = system.ff_cpu if ff else system.cpu
    system.membus = SystemXBar()

    if args.caches:
        l1i, l1d, l2 = make_caches(args)
        system.cpu.icache = l1i
        system.cpu.dcache = l1d
        front.icache_port = system.cpu.icache.cpu_side
        front.dcache_port = system.cpu.dcache.cpu_side
        if args.l2cache:
            system.l2bus = L2XBar()
            system.l2 = l2
            system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
            system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports
            system.l2bus.mem_side_ports = system.l2.cpu_side
            system.l2.mem_side = system.membus.cpu_side_ports
        else:
            system.cpu.icache.mem_side = system.membus.cpu_side_ports
            system.cpu.dcache.mem_side = system.membus.cpu_side_ports
        if is_o3(args) and has_window_support() and not args.no_window:
            system.cpu.window_l1d = system.cpu.dcache
            if args.l2cache:
                system.cpu.window_l2 = system.l2
    else:
        front.icache_port = system.membus.cpu_side_ports
        front.dcache_port = system.membus.cpu_side_ports

    front.createInterruptController()
    front.mmu.connectWalkerPorts(system.membus.cpu_side_ports,
                                 system.membus.cpu_side_ports)

    mem_cls = MEM_MAP[args.mem_type]
    if mem_cls is SimpleMemory:
        system.mem_ctrl = SimpleMemory(latency=f"{30 + args.mem_extra_latency_ns}ns")
        system.mem_ctrl.range = system.mem_ranges[0]
        system.mem_ctrl.port = system.membus.mem_side_ports
    else:
        dram = mem_cls()
        dram.range = system.mem_ranges[0]
        system.mem_ctrl = MemCtrl(dram=dram)
        if args.mem_extra_latency_ns:
            system.mem_ctrl.static_frontend_latency = \
                f"{10 + args.mem_extra_latency_ns}ns"
        system.mem_ctrl.port = system.membus.mem_side_ports

    system.system_port = system.membus.cpu_side_ports
    return system


def attach_workload(system, args):
    """Create the SE ``Process`` for ``--cmd``/``--options`` and attach it.

    In ``ff`` mode the process is attached to both CPUs and the O3 CPU shares
    the fast-forward CPU's ISA object.

    Args:
        system (System): System from :func:`build_system`.
        args (argparse.Namespace): Parsed namespace.
    """
    process = Process()
    process.cmd = [args.cmd] + (args.options.split() if args.options else [])
    if args.input:
        process.input = args.input
    if args.output:
        process.output = args.output
    if args.errout:
        process.errout = args.errout
    system.workload = SEWorkload.init_compatible(args.cmd)
    if args.mode == "ff":
        system.ff_cpu.workload = process
        system.ff_cpu.createThreads()
        system.cpu.workload = process
        system.cpu.isa = system.ff_cpu.isa
        system.cpu.createThreads()
    else:
        system.cpu.workload = process
        system.cpu.createThreads()


# ---------------------------------------------------------------------------
# Hint/region retirement tracking (gem5 PcCountTracker probes)
# ---------------------------------------------------------------------------


class HintTracking:
    """Count the retirements of the binary's hint PCs on the given CPUs.

    One PcCountTrackerManager per hint kind keeps global per-PC counters and
    the last tracked (pc, count) pair. Targets with count ``NEVER`` only count;
    ``exit_counts`` (profile mode) makes every region visit up to that count an
    exit event.

    Attributes:
        hints: ``{"setwin": {pc: W}, "region": {pc: id}}`` from
            ``winhint_elf.scan_hints``.
        mgr: ``{kind: PcCountTrackerManager}`` for each kind with at least one PC.
    """

    def __init__(self, system, cpus, hints, exit_counts=0):
        """Create the tracker managers and attach one PcCountTracker per CPU and kind.

        Args:
            system (System): System; managers are attached as ``winhint_<kind>_mgr``.
            cpus (list[SimObject]): CPUs to probe (both CPUs in ``ff`` mode).
            hints (dict): Hint PCs from ``winhint_elf.scan_hints``.
            exit_counts (int): If > 0, region targets ``(pc, 1..exit_counts)`` raise an
                exit event at each visit (profile mode); otherwise targets only count.
        """
        from m5.objects import PcCountTracker, PcCountTrackerManager
        from m5.params import PcCountPair

        self.hints = hints
        self.mgr = {}
        for kind in ("setwin", "region"):
            pcs = sorted(hints[kind])
            if not pcs:
                continue
            if kind == "region" and exit_counts:
                targets = [PcCountPair(pc, c) for pc in pcs
                           for c in range(1, exit_counts + 1)]
            else:
                targets = [PcCountPair(pc, NEVER) for pc in pcs]
            mgr = PcCountTrackerManager(targets=targets)
            setattr(system, f"winhint_{kind}_mgr", mgr)
            self.mgr[kind] = mgr
            for cpu in cpus:
                setattr(cpu, f"winhint_{kind}_tracker",
                        PcCountTracker(targets=targets, core=cpu, ptmanager=mgr,
                                       manager=cpu))

    def count(self, kind):
        """Return the total number of retired hints of ``kind`` (``setwin``/``region``)."""
        m = self.mgr.get(kind)
        if m is None:
            return 0
        return int(sum(max(0, int(m.getPcCount(pc))) for pc in self.hints[kind]))

    def last(self, kind):
        """Return the last retired hint of ``kind``.

        Args:
            kind (str): ``"setwin"`` or ``"region"``.

        Returns:
            pc_payload (tuple[int | None, int | None]): ``(pc, payload)``
                (payload = W for setwin, id for region), or ``(None, None)`` if
                none retired.
        """
        m = self.mgr.get(kind)
        if m is None:
            return None, None
        pair = m.getCurrentPcCountPair()
        pc, n = int(pair.get_pc()), int(pair.get_count())
        if n <= 0 or pc not in self.hints[kind]:
            return None, None
        return pc, self.hints[kind][pc]

    def state(self):
        """Return a snapshot of the hint state (last setwin/region and counts).

        Returns:
            state (dict): Dict with ``last_setwin``, ``last_setwin_pc``, ``last_region``,
                ``last_region_pc``, ``setwin_count`` and ``region_count``; the format
                stored in ``winhint_ff.json`` and ``runlength.json``.
        """
        spc, w = self.last("setwin")
        rpc, rid = self.last("region")
        return {"last_setwin": w, "last_setwin_pc": spc, "last_region": rid,
                "last_region_pc": rpc, "setwin_count": self.count("setwin"),
                "region_count": self.count("region")}


def scan_binary(args):
    """Scan ``args.cmd`` for hint PCs, tolerating unreadable binaries.

    Returns:
        hints (dict): ``winhint_elf.scan_hints`` result, or empty ``setwin``/``region`` maps
            (with a warning) on ``OSError``/``ValueError``.
    """
    try:
        return winhint_elf.scan_hints(args.cmd)
    except (OSError, ValueError) as exc:
        print(f"[winhint_se] warning: cannot scan {args.cmd} for hints: {exc}")
        return {"setwin": {}, "region": {}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def clock_period_ticks(clock):
    """Convert a clock string to its period in gem5 ticks (1 ps).

    Example: ``'2GHz' -> 500``.

    Args:
        clock (str): Clock frequency string with a GHz/MHz/kHz/Hz suffix.

    Returns:
        ticks (int): Period in ticks, rounded to an integer.

    Raises:
        SystemExit: If ``clock`` is not a positive number with one of those suffixes.
    """
    s = str(clock).strip()
    for unit, mult in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3), ("Hz", 1.0)):
        if s.endswith(unit):
            try:
                freq = float(s[:-len(unit)]) * mult
            except ValueError:
                break
            if freq > 0:
                return int(round(1e12 / freq))
            break
    sys.exit(f"[se.py] cannot parse clock {clock!r} (expected e.g. '2GHz', "
             "'500MHz')")


def outdir_path(name):
    """Return ``name`` joined to the gem5 output directory (``m5.options.outdir``)."""
    return os.path.join(m5.options.outdir, name)


def write_json(path, data):
    """Write ``data`` to ``path`` as indented JSON with a trailing newline."""
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def current_insts(cpu):
    """Return the instruction count of thread 0 of ``cpu``."""
    return int(cpu.getCurrentInstCount(0))


def run_profile(system, args, track):
    """Run the functional region profile and write it to ``--profile-regions``.

    Each ``simpoint starting point found`` exit is one region(id) marker
    retirement; the loop stops at the first other exit cause. Visits are
    recorded as ``[region_id, visit_count, instruction_count]``; at most
    ``--profile-cap`` visits per marker PC raise exits (``truncated`` flags
    markers that reached the cap).

    Args:
        system (System): Instantiated system.
        args (argparse.Namespace): Parsed namespace (``profile`` mode).
        track (HintTracking): :class:`HintTracking` created with ``exit_counts``.

    Returns:
        cause (str): The final exit cause.
    """
    region_of = track.hints["region"]
    visits, cause = [], ""
    while True:
        ev = m5.simulate()
        cause = ev.getCause()
        if cause != "simpoint starting point found":
            break
        pair = track.mgr["region"].getCurrentPcCountPair()
        pc = int(pair.get_pc())
        visits.append([region_of.get(pc, -1), int(pair.get_count()),
                       current_insts(system.cpu)])
    total = int(system.cpu.totalInsts())
    counts = {}
    for pc, rid in region_of.items():
        counts[str(rid)] = counts.get(str(rid), 0) + max(
            0, int(track.mgr["region"].getPcCount(pc)))
    truncated = any(int(track.mgr["region"].getPcCount(pc)) >= args.profile_cap
                    for pc in region_of)
    write_json(args.profile_regions, {
        "version": 1, "binary": os.path.abspath(args.cmd), "options": args.options,
        "total_insts": total, "exit_cause": cause,
        "markers": {hex(pc): rid for pc, rid in sorted(region_of.items())},
        "visits": visits, "visit_counts": counts, "profile_cap": args.profile_cap,
        "truncated": truncated})
    print(f"[winhint_se] Region profile: {len(visits)} visits, {total} insts "
          f"-> {args.profile_regions}")
    return cause


def run_checkpoints(system, args, track):
    """Run functionally, writing a checkpoint at each requested instruction count.

    Each checkpoint goes to ``<checkpoint-dir>/cpt.<N>/`` with a
    ``winhint_ff.json`` (hint state, instruction count, tick, binary, options).
    ``checkpoints.json`` in the checkpoint dir lists requested and written
    counts. Stops early if the program ends first.

    Args:
        system (System): Instantiated system.
        args (argparse.Namespace): Parsed namespace (``checkpoint`` mode).
        track (HintTracking): :class:`HintTracking` (count-only targets).

    Returns:
        cause (str): The last exit cause (empty if no simulation was needed).
    """
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    done, cause = [], ""
    for n in args.checkpoints:
        delta = n - current_insts(system.cpu)
        if delta > 0:
            system.cpu.scheduleInstStop(0, delta, "winhint checkpoint")
            ev = m5.simulate()
            cause = ev.getCause()
            if cause != "winhint checkpoint":
                print(f"[winhint_se] program ended before instruction {n}: {cause}")
                break
        d = os.path.join(args.checkpoint_dir, f"cpt.{n}")
        m5.checkpoint(d)
        st = dict(track.state(), insts=current_insts(system.cpu), target=n,
                  tick=int(m5.curTick()), binary=os.path.abspath(args.cmd),
                  options=args.options)
        write_json(os.path.join(d, FF_STATE), st)
        done.append(n)
        print(f"[winhint_se] checkpoint {d}: {st}")
    write_json(os.path.join(args.checkpoint_dir, "checkpoints.json"),
               {"requested": args.checkpoints, "written": done,
                "binary": os.path.abspath(args.cmd), "options": args.options,
                "exit_cause": cause})
    return cause


def run_detailed(system, args, track, info):
    """Run the optional detailed warm-up, then the measured part.

    The stats are reset at the end of the warm-up. ``info`` is filled with
    ``setwin_in_warmup``, ``measure_start_tick``, ``measure_start_cycle``,
    ``measured_insts`` and ``ended_in`` (``warmup``, ``measure_limit`` or
    ``program_end``).

    Args:
        system (System): Instantiated system.
        args (argparse.Namespace): Parsed namespace.
        track (HintTracking): :class:`HintTracking` or None (``full`` mode).
        info (dict): ``runlength.json`` dict, updated in place.

    Returns:
        exit_event (GlobalSimLoopExitEvent): The last exit event.
    """
    cpu = system.cpu
    period = clock_period_ticks(args.sys_clock)
    sw0 = track.count("setwin") if track else 0
    if args.warmup_insts:
        cpu.scheduleInstStop(0, args.warmup_insts, "winhint warmup done")
        ev = m5.simulate()
        if ev.getCause() != "winhint warmup done":
            info.update(ended_in="warmup", measured_insts=0)
            return ev
        info["setwin_in_warmup"] = (track.count("setwin") - sw0) if track else 0
        m5.stats.reset()
    info["measure_start_tick"] = int(m5.curTick())
    info["measure_start_cycle"] = int(m5.curTick()) // period
    i0 = current_insts(cpu)
    if args.maxinsts and not param_max_insts(args):
        cpu.scheduleInstStop(0, args.maxinsts, "winhint measure done")
    ev = m5.simulate()
    info["measured_insts"] = current_insts(cpu) - i0
    info["ended_in"] = ("measure_limit" if ev.getCause() in (
        "winhint measure done", "a thread reached the max instruction count")
        else "program_end")
    return ev


def main():
    """Entry point under gem5: build, instantiate and run the configured mode.

    Profile and checkpoint modes exit after their functional run. Otherwise
    the optional fast-forward (then ``switchCpus``) or checkpoint restore is
    followed by :func:`run_detailed`; ``runlength.json`` is written to the
    outdir for every non-``full`` run or run with warm-up.
    """
    args = parse_args()
    system = build_system(args)
    attach_workload(system, args)

    track = None
    if args.mode != "full":
        hints = scan_binary(args)
        if args.mode == "profile" and not hints["region"]:
            sys.exit(f"[se.py] --profile-regions: {args.cmd} has no region(id) "
                     "markers (use the oracle build)")
        cpus = [system.ff_cpu, system.cpu] if args.mode == "ff" else [system.cpu]
        track = HintTracking(system, cpus, hints,
                             exit_counts=args.profile_cap
                             if args.mode == "profile" else 0)

    root = Root(full_system=False, system=system)  # noqa: F841
    m5.instantiate(args.restore_checkpoint or None)

    print(f"[winhint_se] Starting simulation: {args.cmd} {args.options} "
          f"(mode {args.mode})")
    print(f"[winhint_se] Machine: {args.machine or '(built-in default)'}")
    if is_o3(args):
        print(f"[winhint_se] Physical ROB/IQ/LQ/SQ: {args.phys}")
        if not args.no_window:
            mode = "winhint" if has_window_support() else "clean (static, physical)"
            print(f"[winhint_se] Window: policy={args.window_policy} "
                  f"initial={args.window_initial} build={mode}")

    if args.mode == "profile":
        cause = run_profile(system, args, track)
        print(f"[winhint_se] Exiting @ tick {m5.curTick()} — {cause}")
        return
    if args.mode == "checkpoint":
        cause = run_checkpoints(system, args, track)
        print(f"[winhint_se] Exiting @ tick {m5.curTick()} — {cause}")
        return

    info = {"version": 1, "mode": args.mode, "fast_forward": args.fast_forward,
            "warmup_insts": args.warmup_insts, "max_insts": args.maxinsts,
            "window_policy": args.window_policy,
            "window_initial": args.window_initial if is_o3(args) else None,
            "clock_period_ticks": clock_period_ticks(args.sys_clock)}
    if args.mode == "ff":
        system.ff_cpu.scheduleInstStop(0, args.fast_forward, "winhint fast-forward done")
        ev = m5.simulate()
        if ev.getCause() != "winhint fast-forward done":
            info["ended_in"] = "fast_forward"
            write_json(outdir_path("runlength.json"), info)
            print(f"[winhint_se] Exiting @ tick {m5.curTick()} — {ev.getCause()}")
            return
        info["ff_state"] = track.state()
        info["switch_tick"] = int(m5.curTick())
        m5.switchCpus(system, [(system.ff_cpu, system.cpu)])
        if args.warmup_insts == 0:
            print("[winhint_se] warning: --fast-forward without --warmup-insts: the "
                  "first window period after the switch spans the fast-forward")
    elif args.mode == "restore":
        info.update(checkpoint=os.path.abspath(args.restore_checkpoint),
                    ff_state=args.ff_state, seed=args.seed,
                    window_seed=args.window_seed)

    exit_event = run_detailed(system, args, track, info)
    if track is not None:
        st = info.get("ff_state") or {}
        if args.window_policy in SEED_POLICIES and is_o3(args):
            w = st.get("last_setwin")
            rob = args.machine_cfg.get("window", {}).get("rob", [])
            seeded_ok = (w is None or not rob or
                         winhint_elf.config_for_setwin(int(w), list(rob))
                         == args.window_initial)
            info["hint_state_exact"] = bool(seeded_ok or info.get("setwin_in_warmup"))
    if args.mode != "full" or args.warmup_insts:
        write_json(outdir_path("runlength.json"), info)
    print(f"[winhint_se] Exiting @ tick {m5.curTick()} — {exit_event.getCause()}")


if __name__ == "__m5_main__":
    main()
