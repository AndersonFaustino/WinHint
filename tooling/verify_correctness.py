#!/usr/bin/env python3
"""Cross-component correctness harness (PROPOSAL §7): hinted variants vs plain, bit for bit.

For every kernel and every non-plain variant present under
`$WINHINT_BUILD/benchmarks/<arch>/`, run the variant and the matching plain
binary on the same input and compare their stdout bit for bit:

      * riscv  under qemu-riscv64                      (platform "qemu")
      * x86    natively                                (platform "native")
      * riscv  on gem5 RISCV_clean (sim/se.py), when
        build/gem5/src/build/RISCV_clean/gem5.opt exists (platform "gem5")

A variant directory is `<variant><cfg>[/<machine>]` (benchmarks/Makefile), e.g.
winhint, winhint-O3-nounroll, winhint/riscv_ooo_big. Its reference is
`plain<cfg>`. On gem5, the reference output is `plain<cfg>` on gem5 too, and the
plain binary itself is checked against its qemu output (row variant=plain).

gem5 runs use small inputs only, run one at a time under
`flock $WINHINT_BUILD/.heavy.lock`, and are skipped for kernels whose qemu run
takes longer than --gem5-max-qemu-sec (a proxy for the instruction count).

Prints a pass/fail table and writes results/correctness.csv.
Exit status: 0 if every compared pair matches (SKIP rows do not count as
failures), 1 otherwise, 2 if --gem5 on is given but gem5.opt is not built.
Run inside the `winhint` env (`tooling/winhint.sh verify [args]`).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

#: Repository root (``$WINHINT_ROOT``, default the parent of ``tooling/``).
ROOT = Path(os.environ.get("WINHINT_ROOT", Path(__file__).resolve().parents[1]))
#: Build root (``$WINHINT_BUILD``, default ``<root>/build``).
BUILD = Path(os.environ.get("WINHINT_BUILD", ROOT / "build"))
#: File suffixes of build side outputs that are never run.
SIDE_SUFFIXES = (".json", ".ll", ".bc", ".s", ".o", ".txt", ".csv", ".log")
#: Output hash printed by the kernels.
HASH_RE = re.compile(r"hash=(0x[0-9a-fA-F]+)")
#: Column order of correctness.csv.
CSV_FIELDS = ["arch", "platform", "variant_dir", "variant", "cfg", "machine", "kernel",
              "input", "status", "hints", "ref_hash", "out_hash", "exit_code", "seconds", "note"]


@dataclass
class Run:
    """Result of one program execution.

    Attributes:
        exit_code: Process exit code; ``None`` = timeout / could not run.
        stdout: Captured standard output.
        seconds: Wall time of the run.
        note: Short diagnostic (last stderr line on failure, timeout or lock message).
    """
    exit_code: int | None          # None = timeout / could not run
    stdout: str = ""
    seconds: float = 0.0
    note: str = ""

    @property
    def ok(self) -> bool:
        """Whether the run exited with 0 and printed something."""
        return self.exit_code == 0 and bool(self.stdout)


@dataclass
class Row:
    """One line of the result table / ``correctness.csv``.

    Attributes:
        arch: ``riscv`` or ``x86``.
        platform: ``qemu``, ``native`` or ``gem5``.
        variant_dir: Variant directory relative to the arch directory (with machine level).
        variant: Variant name without the configuration suffix.
        cfg: Build configuration suffix without the leading ``-``.
        machine: Machine subdirectory, or empty.
        kernel: Kernel name.
        input: Input argument (``--input``).
        status: PASS, DIFF, FAIL, NOREF, SKIP, TIMEOUT or ERROR.
        hints: ``"<setwin>/<region>"`` hint counts of the variant binary (see :func:`count_hints`).
        ref_hash: Output hash of the reference run (see :func:`out_hash`).
        out_hash: Output hash of the variant run.
        exit_code: Exit code of the variant run, empty if it did not finish.
        seconds: Wall time of the variant run.
        note: Diagnostic.
    """
    arch: str
    platform: str
    variant_dir: str
    variant: str
    cfg: str
    machine: str
    kernel: str
    input: str
    status: str
    hints: str = ""
    ref_hash: str = ""
    out_hash: str = ""
    exit_code: str = ""
    seconds: str = ""
    note: str = ""


def out_hash(text: str) -> str:
    """Return the kernel's own FNV-1a hash when printed, else a short sha1 of stdout.

    Args:
        text: Program output.

    Returns:
        The distinct ``hash=0x...`` values joined by ``,``, ``"sha1:<10 hex>"``, or
            ``""`` for empty output.
    """
    hs = HASH_RE.findall(text)
    if hs:
        return ",".join(dict.fromkeys(hs))
    return "sha1:" + hashlib.sha1(text.encode()).hexdigest()[:10] if text else ""


def exec_bytes(path: Path) -> bytes:
    """Return the concatenated executable (SHF_EXECINSTR) sections of an ELF64 LE file.

    Only ``SHT_PROGBITS`` sections are included.

    Args:
        path: ELF file.

    Returns:
        The section bytes, or ``b""`` if the file is not an ELF64 little-endian file.

    Raises:
        OSError: If the file cannot be read.
        struct.error: If the section header table is truncated.
    """
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return b""
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    out = []
    for i in range(shnum):
        off = shoff + i * shentsize
        sh_type, sh_flags, _, sh_offset, sh_size = struct.unpack_from("<IQQQQ", data, off + 4)
        if sh_type == 1 and sh_flags & 0x4:              # PROGBITS, SHF_EXECINSTR
            out.append(data[sh_offset:sh_offset + sh_size])
    return b"".join(out)


def count_hints(path: Path, arch: str) -> str:
    """Count the setwin and region hint encodings in the text of a binary (docs/interfaces.md §2).

    RISC-V: ``ori x0, x0, (payload<<5)|tag`` with tag 21 (setwin) or 23 (region),
    scanned at 2-byte steps. x86: ``nopl 0x5748K0PP(%rax)`` with kind 1 (setwin)
    or 2 (region).

    Args:
        path: Binary to scan.
        arch: ``riscv``; any other value uses the x86 encoding.

    Returns:
        ``"<setwin>/<region>"``, or ``"?"`` if the file cannot be parsed.
    """
    try:
        text = exec_bytes(path)
    except (OSError, struct.error):
        return "?"
    sw = rg = 0
    if arch == "riscv":                                   # ori x0, x0, (payload<<5)|tag
        for i in range(0, len(text) - 3, 2):
            w = int.from_bytes(text[i:i + 4], "little")
            if w & 0xFFFFF == 0x6013:
                tag = (w >> 20) & 0x1F
                sw += tag == 21
                rg += tag == 23
    else:                                                 # nopl 0x5748K0PP(%rax)
        for m in re.finditer(rb"\x0f\x1f\x80(..)\x48\x57", text, re.S):
            kind = m.group(1)[1] >> 4
            sw += kind == 1
            rg += kind == 2
    return f"{sw}/{rg}"


def is_binary(p: Path) -> bool:
    """Return whether ``p`` is an executable that is not a side output.

    Side outputs are dotfiles and names ending in :data:`SIDE_SUFFIXES`.
    """
    return (p.is_file() and os.access(p, os.X_OK)
            and not p.name.endswith(SIDE_SUFFIXES) and not p.name.startswith("."))


def split_dir(name: str) -> tuple[str, str]:
    """Split a variant directory name into variant and configuration suffix.

    Example:
        ``'winhint-O3-nounroll' -> ('winhint', '-O3-nounroll')``.

    Args:
        name: Directory name.

    Returns:
        ``(variant, cfg)``; ``cfg`` keeps its leading ``-`` and is empty without one.
    """
    v, sep, rest = name.partition("-")
    return v, (sep + rest) if sep else ""


def discover(arch: str, variants: set[str] | None, kernels: set[str] | None):
    """Enumerate the hinted binaries of an architecture and their plain references.

    Args:
        arch: ``riscv`` or ``x86``.
        variants: Variant names to keep, or ``None`` for every non-plain variant.
        kernels: Kernel names to keep, or ``None`` for all.

    Yields:
        (tuple): ``(variant_dir_rel, variant, cfg, machine, kernel, binary, plain_binary)``;
            ``plain_binary`` is ``plain<cfg>/<kernel>`` and may not exist.
    """
    base = BUILD / "benchmarks" / arch
    if not base.is_dir():
        return
    for vdir in sorted(p for p in base.iterdir() if p.is_dir()):
        variant, cfg = split_dir(vdir.name)
        if variant == "plain" or (variants and variant not in variants):
            continue
        plain_dir = base / f"plain{cfg}"
        leaves = [(vdir, "")] + [(m, m.name) for m in sorted(vdir.iterdir()) if m.is_dir()]
        for leaf, machine in leaves:
            for b in sorted(p for p in leaf.iterdir() if is_binary(p)):
                if kernels and b.name not in kernels:
                    continue
                rel = f"{vdir.name}/{machine}" if machine else vdir.name
                yield rel, variant, cfg, machine, b.name, b, plain_dir / b.name


def run_cmd(cmd: list[str], timeout: float, env: dict | None = None) -> Run:
    """Run a command and capture its output.

    Args:
        cmd: Command as an argument list.
        timeout: Timeout in seconds.
        env: Environment, or ``None`` to inherit.

    Returns:
        A :class:`Run`; ``exit_code`` is ``None`` on timeout or ``OSError``.
    """
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, errors="replace", timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return Run(None, "", time.monotonic() - t0, f"timeout after {timeout:.0f}s")
    except OSError as e:
        return Run(None, "", time.monotonic() - t0, str(e))
    note = "" if p.returncode == 0 else (p.stderr.strip().splitlines() or [""])[-1][:160]
    return Run(p.returncode, p.stdout, time.monotonic() - t0, note)


class Runner:
    """Runs binaries natively/under qemu and on gem5, caching results per binary.

    Attributes:
        args: Parsed command-line namespace.
        cache: Results keyed by ``(platform, binary path)``.
        env: Environment of host runs (``WINHINT_MODE=log``, libwinhint on ``LD_LIBRARY_PATH``).
        gem5: Path of gem5.opt.
        lock: Path of the heavy-job lock file.
        lock_busy: Set once a lock wait timed out; further gem5 runs are abandoned.
    """
    def __init__(self, args):
        """Set up the run environment and gem5 paths.

        Args:
            args (argparse.Namespace): Parsed command-line namespace.
        """
        self.args = args
        self.cache: dict[tuple, Run] = {}
        self.env = dict(os.environ)
        # Call-mode binaries (libwinhint) decide but never migrate (benchmarks/Makefile check).
        self.env["WINHINT_MODE"] = "log"
        libdir = BUILD / "libwinhint"
        if libdir.is_dir():
            self.env["LD_LIBRARY_PATH"] = f"{libdir}:{self.env.get('LD_LIBRARY_PATH', '')}"
        self.gem5 = Path(args.gem5_binary) if args.gem5_binary else \
            BUILD / "gem5" / "src" / "build" / f"RISCV_{args.gem5_build}" / "gem5.opt"
        self.lock = BUILD / ".heavy.lock"
        self.lock_busy = False      # set once a lock wait times out: stop trying gem5

    def host(self, arch: str, binary: Path) -> Run:
        """Run a binary natively (x86) or under qemu (riscv) with the input arguments (cached).

        Args:
            arch: ``riscv`` or ``x86``.
            binary: Binary to run.

        Returns:
            The :class:`Run` result.
        """
        key = ("host", str(binary))
        if key not in self.cache:
            argv = [str(binary), *self.args.input_args]
            cmd = [self.args.qemu, *argv] if arch == "riscv" else argv
            self.cache[key] = run_cmd(cmd, self.args.timeout, self.env)
        return self.cache[key]

    def gem5_run(self, binary: Path) -> Run:
        """Run a RISC-V binary on gem5 (sim/se.py) under the heavy lock (cached).

        The program output is read from ``<gem5-outdir>/<rel path>/program.out``.
        With the ``clean`` build ``--no-window`` is passed. If the lock cannot be
        taken within ``--lock-wait`` (flock exit 75), :attr:`lock_busy` is set.

        Args:
            binary: Binary under ``$WINHINT_BUILD/benchmarks``.

        Returns:
            The :class:`Run` result.
        """
        key = ("gem5", str(binary))
        if key in self.cache:
            return self.cache[key]
        if self.lock_busy:
            return Run(None, "", 0.0, "heavy lock busy (gem5 runs abandoned)")
        rel = binary.relative_to(BUILD / "benchmarks")
        outdir = Path(self.args.gem5_outdir) / str(rel).replace("/", "__")
        outdir.mkdir(parents=True, exist_ok=True)
        cmd = [str(self.gem5), f"--outdir={outdir}", str(ROOT / "sim" / "se.py"),
               "--cmd", str(binary), "--options", " ".join(self.args.input_args),
               "--cpu-type", self.args.gem5_cpu, "--mem-size", self.args.gem5_mem,
               "--output", "program.out", "--errout", "program.err"]
        if self.args.gem5_build == "clean":
            cmd.append("--no-window")
        if self.args.no_lock:
            full = cmd
        else:
            full = ["flock", "-E", "75", "-w", str(self.args.lock_wait), str(self.lock), *cmd]
        r = run_cmd(full, self.args.gem5_timeout)
        out = outdir / "program.out"
        stdout = out.read_text(errors="replace") if out.exists() else ""
        note = r.note
        if r.exit_code == 75 and not self.args.no_lock:
            note = f"heavy lock busy for {self.args.lock_wait}s"
            self.lock_busy = True
            res = Run(None, "", r.seconds, note)
            self.cache[key] = res
            return res
        res = Run(r.exit_code, stdout, r.seconds, note)
        if r.exit_code not in (0, None):
            # gem5 exits non-zero if the workload does; keep its last stderr line.
            res.note = note or f"gem5 exit {r.exit_code} (see {outdir})"
        self.cache[key] = res
        return res


def compare(ref: Run, out: Run, plain_label: str = "plain") -> tuple[str, str]:
    """Compare a reference run with a variant run.

    Args:
        ref: Reference (plain) run.
        out: Variant run.
        plain_label: Name of the reference used in notes.

    Returns:
        ``(status, note)``; status is SKIP (lock busy), TIMEOUT/ERROR (variant did
            not finish), NOREF (reference did not finish), FAIL (non-zero exit or empty
            reference output), PASS or DIFF.
    """
    for r in (ref, out):
        if r.exit_code is None and "heavy lock busy" in r.note:
            return "SKIP", r.note
    if out.exit_code is None:
        return ("TIMEOUT" if "timeout" in out.note else "ERROR"), out.note
    if ref.exit_code is None:
        return "NOREF", f"{plain_label}: {ref.note}"
    if ref.exit_code != 0 or out.exit_code != 0:
        return "FAIL", f"exit {plain_label}={ref.exit_code} variant={out.exit_code} {out.note or ref.note}".strip()
    if not ref.stdout:
        return "FAIL", f"{plain_label} printed nothing"
    return ("PASS", "") if ref.stdout == out.stdout else ("DIFF", "stdout differs")


def mkrow(arch, platform, item, status, ref: Run | None, out: Run | None, note, input_s) -> Row:
    """Build a result row for a discovered item.

    Args:
        arch (str): Architecture.
        platform (str): ``qemu``, ``native`` or ``gem5``.
        item (tuple): Tuple yielded by :func:`discover`.
        status (str): Comparison status.
        ref: Reference run, or ``None``.
        out: Variant run, or ``None``.
        note (str): Diagnostic.
        input_s (str): Input argument.

    Returns:
        The :class:`Row`; hints are counted only for non-plain variants.
    """
    rel, variant, cfg, machine, kernel, *_ = item
    binary = item[5]
    hints = count_hints(binary, arch) if variant != "plain" else ""
    return Row(arch, platform, rel, variant, cfg.lstrip("-"), machine, kernel, input_s, status,
               hints, out_hash(ref.stdout) if ref else "", out_hash(out.stdout) if out else "",
               "" if not out or out.exit_code is None else str(out.exit_code),
               f"{out.seconds:.2f}" if out else "", note)


def main(argv=None) -> int:
    """Parse the command line, compare all pairs, print the table and write the CSV.

    Args:
        argv (list[str] | None): Argument list (``None`` uses ``sys.argv``).

    Returns:
        0 if nothing failed (only PASS/SKIP) or ``--dry-run``, 1 otherwise, 2 if
            ``--gem5 on`` is given without a gem5 binary.
    """
    ap = argparse.ArgumentParser(
        description="Bit-identical output check of every hinted variant against plain "
                    "(qemu-riscv64, native x86, gem5 RISCV_clean). PROPOSAL §7.")
    ap.add_argument("--arch", nargs="+", default=["riscv", "x86"], choices=["riscv", "x86"])
    ap.add_argument("--variants", nargs="+", help="only these variants (default: every non-plain dir)")
    ap.add_argument("--kernels", nargs="+", help="only these kernels")
    ap.add_argument("--input", default="small", help="argv[1] for every kernel (default small)")
    ap.add_argument("--extra-args", default="", help="further kernel args (e.g. '1' = one layer/rep)")
    ap.add_argument("--timeout", type=float, default=600, help="per qemu/native run, seconds")
    ap.add_argument("--qemu", default="qemu-riscv64")
    ap.add_argument("--gem5", choices=["auto", "on", "off"], default="auto",
                    help="auto: run gem5 if the binary exists (default)")
    ap.add_argument("--gem5-build", choices=["clean", "winhint"], default="clean")
    ap.add_argument("--gem5-binary", default="", help="override the gem5.opt path")
    ap.add_argument("--gem5-cpu", default="AtomicSimpleCPU",
                    choices=["AtomicSimpleCPU", "TimingSimpleCPU", "MinorCPU", "DerivO3CPU"])
    ap.add_argument("--gem5-mem", default="2GB")
    ap.add_argument("--gem5-timeout", type=float, default=3600)
    ap.add_argument("--gem5-max-qemu-sec", type=float, default=1.0,
                    help="skip gem5 for kernels whose plain qemu run takes longer "
                         "(default 1s, roughly <=1e9 instructions, minutes on AtomicSimpleCPU)")
    ap.add_argument("--gem5-outdir", default=str(ROOT / "results" / "correctness" / "gem5"))
    ap.add_argument("--lock-wait", type=int, default=7200,
                    help="max seconds to wait for the heavy lock per gem5 run")
    ap.add_argument("--no-lock", action="store_true", help="do not take the heavy lock (not recommended)")
    ap.add_argument("--csv", default=str(ROOT / "results" / "correctness.csv"))
    ap.add_argument("--dry-run", action="store_true", help="list the pairs, run nothing")
    args = ap.parse_args(argv)
    args.input_args = [args.input, *args.extra_args.split()]

    variants = set(args.variants) if args.variants else None
    kernels = set(args.kernels) if args.kernels else None
    runner = Runner(args)
    gem5_on = args.gem5 == "on" or (args.gem5 == "auto" and runner.gem5.exists())
    if args.gem5 == "on" and not runner.gem5.exists():
        print(f"ERROR: {runner.gem5} not built (tooling/winhint.sh gem5:build {args.gem5_build})",
              file=sys.stderr)
        return 2
    print(f"==> correctness: input='{' '.join(args.input_args)}' arch={','.join(args.arch)} "
          f"gem5={'RISCV_' + args.gem5_build + ' (' + args.gem5_cpu + ')' if gem5_on else 'off'}")

    rows: list[Row] = []
    gem5_plain_done: set[str] = set()
    for arch in args.arch:
        items = list(discover(arch, variants, kernels))
        if not items:
            print(f"  ({arch}: no hinted variants under {BUILD / 'benchmarks' / arch})")
        platform = "qemu" if arch == "riscv" else "native"
        for item in items:
            rel, variant, cfg, machine, kernel, binary, plain = item
            if args.dry_run:
                print(f"  {arch:5} {platform:6} {rel:30} {kernel:28} vs {plain.relative_to(BUILD)}")
                continue
            if not is_binary(plain):
                rows.append(mkrow(arch, platform, item, "NOREF", None, None,
                                  f"missing {plain.relative_to(BUILD)}", args.input))
                continue
            ref, out = runner.host(arch, plain), runner.host(arch, binary)
            status, note = compare(ref, out)
            rows.append(mkrow(arch, platform, item, status, ref, out, note, args.input))
            print(f"  {status:7} {arch:5} {platform:6} {rel:30} {kernel}", flush=True)

            if arch != "riscv" or not gem5_on:
                continue
            if runner.lock_busy or ref.seconds > args.gem5_max_qemu_sec or not ref.ok:
                why = ("heavy lock busy (gem5 runs abandoned)" if runner.lock_busy else
                       f"plain qemu run {ref.seconds:.1f}s > --gem5-max-qemu-sec "
                       f"{args.gem5_max_qemu_sec:g}" if ref.ok else "plain fails under qemu")
                rows.append(mkrow(arch, "gem5", item, "SKIP", None, None, why, args.input))
                continue
            g_ref = runner.gem5_run(plain)
            if str(plain) not in gem5_plain_done:
                gem5_plain_done.add(str(plain))
                st, nt = compare(ref, g_ref, "plain@qemu")
                pitem = (f"plain{cfg}", "plain", cfg, "", kernel, plain, plain)
                rows.append(mkrow(arch, "gem5", pitem, st, ref, g_ref,
                                  (nt + " (gem5 vs qemu)").strip(), args.input))
                print(f"  {st:7} riscv gem5   {'plain' + cfg + ' (vs qemu)':30} {kernel}", flush=True)
            g_out = runner.gem5_run(binary)
            st, nt = compare(g_ref, g_out, "plain@gem5")
            rows.append(mkrow(arch, "gem5", item, st, g_ref, g_out, nt, args.input))
            print(f"  {st:7} riscv gem5   {rel:30} {kernel}", flush=True)

    if args.dry_run:
        return 0

    # ── table ───────────────────────────────────────────────────────────────
    print()
    hdr = (f"{'status':7} {'arch':5} {'platform':8} {'variant_dir':30} {'kernel':28} "
           f"{'hints':9} {'hash':12} note")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        h = r.out_hash.split(",")[0] if r.out_hash else ""
        print(f"{r.status:7} {r.arch:5} {r.platform:8} {r.variant_dir:30} {r.kernel:28} "
              f"{r.hints:9} {h:12} {r.note}")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    print("-" * len(hdr))
    print("summary: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing to compare"))
    empty = sorted({(r.arch, r.variant_dir, r.kernel) for r in rows
                    if r.variant != "plain" and r.hints == "0/0"})
    if empty:
        print(f"note: {len(empty)} variant binaries contain no setwin/region encoding "
              "(call mode, or the pass placed none) — their PASS is trivially true")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for r in rows:
            w.writerow([getattr(r, k) for k in CSV_FIELDS])
    print(f"wrote {csv_path}")
    bad = sum(v for k, v in counts.items() if k not in ("PASS", "SKIP"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
