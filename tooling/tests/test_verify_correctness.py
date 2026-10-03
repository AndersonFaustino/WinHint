"""Tests of tooling/verify_correctness.py on synthetic build trees.

The "binaries" are tiny shell scripts; qemu, gem5.opt and flock are replaced by fake
executables in ``tmp_path``, so nothing is built or simulated.
"""
import csv
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_correctness as vc  # noqa: E402

#: Fake qemu-riscv64: runs its arguments natively.
FAKE_QEMU = '#!/bin/sh\nexec "$@"\n'

#: Fake gem5.opt: runs ``--cmd`` with ``--options`` and writes ``<outdir>/program.out``.
#: Exits 3 without output when ``FAKE_GEM5_FAIL`` is set.
FAKE_GEM5 = """#!/bin/sh
prev=""
for a in "$@"; do
  case $a in --outdir=*) out=${a#--outdir=};; esac
  [ "$prev" = "--cmd" ] && cmd=$a
  [ "$prev" = "--options" ] && opts=$a
  prev=$a
done
if [ -n "$FAKE_GEM5_FAIL" ]; then echo "gem5 boom" >&2; exit 3; fi
"$cmd" $opts > "$out/program.out"
"""

#: Fake flock that always reports the lock as busy (``-E 75``).
BUSY_FLOCK = "#!/bin/sh\nexit 75\n"


def write_exe(path, body):
    """Write an executable script.

    Args:
        path (Path): Destination file (parents are created).
        body (str): Script text.

    Returns:
        (Path): ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def kernel(text, code=0):
    """Return a shell script that prints ``text`` and exits with ``code``.

    Args:
        text (str): Output line.
        code (int): Exit status.

    Returns:
        (str): Script text.
    """
    return f"#!/bin/sh\necho '{text}'\nexit {code}\n"


def elf64(text, flags=0x6):
    """Build a minimal little-endian ELF64 file with one data and one text section.

    Args:
        text (bytes): Contents of the PROGBITS section.
        flags (int): ``sh_flags`` of that section (0x4 = SHF_EXECINSTR).

    Returns:
        (bytes): The file image.
    """
    data = bytearray(64)
    data[0:4] = b"\x7fELF"
    data[4], data[5] = 2, 1
    body = b"DATA" + text
    shoff = 64 + len(body)
    struct.pack_into("<Q", data, 0x28, shoff)
    struct.pack_into("<HH", data, 0x3A, 64, 2)
    sh = bytearray(128)
    struct.pack_into("<IQQQQ", sh, 4, 1, 0x2, 0, 64, 4)                   # data, not exec
    struct.pack_into("<IQQQQ", sh, 64 + 4, 1, flags, 0, 68, len(text))    # text
    return bytes(data) + body + bytes(sh)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """Point the module at an empty build root under ``tmp_path``.

    Args:
        tmp_path (Path): pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): pytest monkeypatch fixture.

    Returns:
        (Path): The build root (``vc.BUILD``).
    """
    build = tmp_path / "build"
    (build / "benchmarks").mkdir(parents=True)
    monkeypatch.setattr(vc, "BUILD", build)
    monkeypatch.setattr(vc, "ROOT", tmp_path)
    monkeypatch.delenv("FAKE_GEM5_FAIL", raising=False)
    return build


def read_csv(path):
    """Read correctness.csv rows.

    Args:
        path (Path): CSV path.

    Returns:
        (list[dict]): One dict per row.
    """
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- helpers
def test_out_hash():
    """Check kernel-hash extraction (deduplicated), the sha1 fallback and empty output."""
    assert vc.out_hash("a hash=0xAB\nb hash=0xAB hash=0x1") == "0xAB,0x1"
    h = vc.out_hash("no hash here")
    assert h.startswith("sha1:") and len(h) == 15
    assert vc.out_hash("") == ""


def test_split_dir_and_is_binary(tmp_path):
    """Check variant/cfg splitting and the side-output filter of is_binary()."""
    assert vc.split_dir("winhint-O3-nounroll") == ("winhint", "-O3-nounroll")
    assert vc.split_dir("plain") == ("plain", "")
    assert vc.is_binary(write_exe(tmp_path / "k", kernel("x")))
    assert not vc.is_binary(write_exe(tmp_path / "k.json", kernel("x")))
    assert not vc.is_binary(write_exe(tmp_path / ".hidden", kernel("x")))
    (tmp_path / "plainfile").write_text("x")
    assert not vc.is_binary(tmp_path / "plainfile")
    assert not vc.is_binary(tmp_path)


def test_exec_bytes(tmp_path):
    """Check that only executable PROGBITS sections are returned and non-ELF gives b''."""
    f = tmp_path / "a.elf"
    f.write_bytes(elf64(b"\x01\x02\x03\x04"))
    assert vc.exec_bytes(f) == b"\x01\x02\x03\x04"
    f.write_bytes(elf64(b"\x01\x02", flags=0x2))
    assert vc.exec_bytes(f) == b""
    f.write_bytes(b"#!/bin/sh\n")
    assert vc.exec_bytes(f) == b""


def test_count_hints_riscv_and_x86(tmp_path):
    """Count setwin/region encodings for both ISAs and report '?' on unreadable files."""
    def ori(tag, payload=3):
        """Encode ``ori x0, x0, (payload<<5)|tag``."""
        return ((((payload << 5) | tag) << 20) | 0x6013).to_bytes(4, "little")

    rv = tmp_path / "rv"
    # 2-byte padding exercises the 2-byte scan step; tag 5 is neither kind.
    rv.write_bytes(elf64(b"\x01\x00" + ori(21) + ori(23) + ori(23) + ori(5)))
    assert vc.count_hints(rv, "riscv") == "1/2"
    x86 = tmp_path / "x86"
    nop = lambda kind: b"\x0f\x1f\x80" + bytes([0x07, kind << 4]) + b"\x48\x57"  # noqa: E731
    x86.write_bytes(elf64(b"\x90" + nop(1) + nop(1) + nop(2) + nop(3)))
    assert vc.count_hints(x86, "x86") == "2/1"
    assert vc.count_hints(tmp_path / "missing", "x86") == "?"
    trunc = tmp_path / "trunc"
    trunc.write_bytes(elf64(b"\x90")[:80])          # section table cut off
    assert vc.count_hints(trunc, "riscv") == "?"


def test_run_cmd_outcomes(tmp_path):
    """Check run_cmd() for success, failure (last stderr line), timeout and OSError."""
    ok = vc.run_cmd([str(write_exe(tmp_path / "ok", kernel("hi")))], 10)
    assert ok.exit_code == 0 and ok.stdout == "hi\n" and ok.ok and ok.note == ""
    bad = write_exe(tmp_path / "bad", "#!/bin/sh\necho one >&2\necho last >&2\nexit 4\n")
    r = vc.run_cmd([str(bad)], 10)
    assert r.exit_code == 4 and r.note == "last" and not r.ok
    silent = vc.run_cmd([str(write_exe(tmp_path / "s", "#!/bin/sh\nexit 2\n"))], 10)
    assert silent.exit_code == 2 and silent.note == ""
    slow = vc.run_cmd([str(write_exe(tmp_path / "slow", "#!/bin/sh\nexec sleep 5\n"))], 0.2)
    assert slow.exit_code is None and slow.note.startswith("timeout after")
    missing = vc.run_cmd([str(tmp_path / "nope")], 10)
    assert missing.exit_code is None and "nope" in missing.note


def test_compare_statuses():
    """Check every status compare() can return."""
    ok = vc.Run(0, "x\n")
    busy = vc.Run(None, "", 0, "heavy lock busy for 1s")
    assert vc.compare(busy, ok) == ("SKIP", busy.note)
    assert vc.compare(ok, busy)[0] == "SKIP"
    assert vc.compare(ok, vc.Run(None, "", 0, "timeout after 1s"))[0] == "TIMEOUT"
    assert vc.compare(ok, vc.Run(None, "", 0, "No such file"))[0] == "ERROR"
    assert vc.compare(vc.Run(None, "", 0, "boom"), ok, "ref") == ("NOREF", "ref: boom")
    st, note = vc.compare(ok, vc.Run(1, "", 0, "err"))
    assert st == "FAIL" and note == "exit plain=0 variant=1 err"
    assert vc.compare(vc.Run(0, ""), vc.Run(0, "")) == ("FAIL", "plain printed nothing")
    assert vc.compare(ok, vc.Run(0, "x\n")) == ("PASS", "")
    assert vc.compare(ok, vc.Run(0, "y\n")) == ("DIFF", "stdout differs")


def test_discover_filters(tree):
    """Check discover(): plain skipped, machine subdirs, variant/kernel filters, missing arch."""
    base = tree / "benchmarks" / "x86"
    write_exe(base / "plain" / "k1", kernel("a"))
    write_exe(base / "winhint" / "k1", kernel("a"))
    write_exe(base / "winhint" / "k2", kernel("a"))
    write_exe(base / "winhint" / "k1.json", kernel("a"))
    write_exe(base / "winhint" / "m1" / "k1", kernel("a"))
    write_exe(base / "oracle-O3" / "k1", kernel("a"))
    items = list(vc.discover("x86", None, None))
    assert [(i[0], i[4]) for i in items] == [("oracle-O3", "k1"), ("winhint", "k1"),
                                             ("winhint", "k2"), ("winhint/m1", "k1")]
    assert items[0][6] == base / "plain-O3" / "k1" and items[0][2] == "-O3"
    assert items[3][3] == "m1"
    assert [i[0] for i in vc.discover("x86", {"winhint"}, {"k1"})] == ["winhint", "winhint/m1"]
    assert list(vc.discover("riscv", None, None)) == []


def test_runner_env_and_gem5_cache(tree, tmp_path):
    """Check the Runner environment and gem5_run's early exit once the lock is busy."""
    (tree / "libwinhint").mkdir()
    args = SimpleNamespace(gem5_binary="", gem5_build="winhint")
    r = vc.Runner(args)
    assert r.env["WINHINT_MODE"] == "log"
    assert r.env["LD_LIBRARY_PATH"].startswith(str(tree / "libwinhint"))
    assert r.gem5 == tree / "gem5" / "src" / "build" / "RISCV_winhint" / "gem5.opt"
    r.lock_busy = True
    res = r.gem5_run(tree / "benchmarks" / "riscv" / "x" / "k")
    assert res.exit_code is None and "abandoned" in res.note


# ----------------------------------------------------------------------------- main
def x86_tree(build):
    """Create an x86 tree with PASS, DIFF, FAIL, TIMEOUT and NOREF kernels.

    Args:
        build (Path): Build root.
    """
    base = build / "benchmarks" / "x86"
    for k in ("kpass", "kdiff", "kfail", "kslow"):
        write_exe(base / "plain" / k, kernel(f"{k} hash=0x11"))
    write_exe(base / "winhint" / "kpass", kernel("kpass hash=0x11"))
    write_exe(base / "winhint" / "kdiff", kernel("kdiff hash=0x22"))
    write_exe(base / "winhint" / "kfail", kernel("kfail", 1))
    write_exe(base / "winhint" / "kslow", "#!/bin/sh\nexec sleep 5\n")
    write_exe(base / "winhint" / "knoref", kernel("x"))


def test_main_x86_statuses_and_csv(tree, tmp_path, capsys):
    """Run main() on native x86 binaries and check statuses, summary, exit code and CSV."""
    x86_tree(tree)
    out_csv = tmp_path / "res" / "c.csv"
    rc = vc.main(["--arch", "x86", "--timeout", "1", "--csv", str(out_csv), "--gem5", "off"])
    assert rc == 1
    rows = {r["kernel"]: r for r in read_csv(out_csv)}
    assert {k: r["status"] for k, r in rows.items()} == {
        "kpass": "PASS", "kdiff": "DIFF", "kfail": "FAIL", "kslow": "TIMEOUT", "knoref": "NOREF"}
    assert rows["kpass"]["ref_hash"] == "0x11" and rows["kdiff"]["out_hash"] == "0x22"
    assert rows["kpass"]["platform"] == "native" and rows["kpass"]["hints"] == "0/0"
    assert rows["kfail"]["exit_code"] == "1" and rows["kslow"]["exit_code"] == ""
    assert rows["knoref"]["note"] == "missing benchmarks/x86/plain/knoref"
    assert rows["kpass"]["input"] == "small"
    out = capsys.readouterr().out
    assert "summary: DIFF=1, FAIL=1, NOREF=1, PASS=1, TIMEOUT=1" in out
    assert "contain no setwin/region encoding" in out


def test_main_dry_run_and_empty(tree, tmp_path, capsys):
    """Check that --dry-run lists pairs without writing, and an empty tree compares nothing."""
    x86_tree(tree)
    out_csv = tmp_path / "c.csv"
    assert vc.main(["--arch", "x86", "--dry-run", "--csv", str(out_csv), "--gem5", "off"]) == 0
    out = capsys.readouterr().out
    assert "vs benchmarks/x86/plain/kpass" in out and not out_csv.exists()
    assert vc.main(["--arch", "riscv", "--csv", str(out_csv), "--gem5", "off"]) == 0
    out = capsys.readouterr().out
    assert "riscv: no hinted variants" in out and "nothing to compare" in out
    assert read_csv(out_csv) == []


def test_main_gem5_on_missing(tree, tmp_path, capsys):
    """Check exit status 2 when --gem5 on is given but gem5.opt does not exist."""
    rc = vc.main(["--gem5", "on", "--gem5-binary", str(tmp_path / "none"), "--csv", str(tmp_path / "c")])
    assert rc == 2 and "not built" in capsys.readouterr().err


def riscv_tree(build, tmp_path):
    """Create a riscv tree (two variants sharing one plain) plus fake qemu and gem5.

    Args:
        build (Path): Build root.
        tmp_path (Path): Where the fake tools go.

    Returns:
        (list[str]): Common main() arguments (qemu, gem5 binary, outdir).
    """
    base = build / "benchmarks" / "riscv"
    write_exe(base / "plain" / "k", "#!/bin/sh\necho \"k $* hash=0x5\"\n")
    write_exe(base / "winhint" / "k", "#!/bin/sh\necho \"k $* hash=0x5\"\n")
    write_exe(base / "winhint" / "m" / "k", "#!/bin/sh\necho \"k $* hash=0x5\"\n")
    qemu = write_exe(tmp_path / "tools" / "qemu", FAKE_QEMU)
    gem5 = write_exe(tmp_path / "tools" / "gem5.opt", FAKE_GEM5)
    return ["--arch", "riscv", "--qemu", str(qemu), "--gem5-binary", str(gem5),
            "--gem5-outdir", str(tmp_path / "g5"), "--gem5-max-qemu-sec", "30",
            "--csv", str(tmp_path / "c.csv")]


def test_main_riscv_gem5_pass(tree, tmp_path, capsys):
    """Run qemu + gem5 (auto, --no-lock): plain checked once against qemu, variants on gem5."""
    args = riscv_tree(tree, tmp_path)
    rc = vc.main(args + ["--no-lock", "--extra-args", "1"])
    assert rc == 0
    rows = read_csv(tmp_path / "c.csv")
    got = [(r["platform"], r["variant_dir"], r["status"]) for r in rows]
    assert got == [("qemu", "winhint", "PASS"), ("gem5", "plain", "PASS"),
                   ("gem5", "winhint", "PASS"), ("qemu", "winhint/m", "PASS"),
                   ("gem5", "winhint/m", "PASS")]
    assert rows[1]["note"] == "(gem5 vs qemu)" and rows[1]["hints"] == ""
    assert (tmp_path / "g5" / "riscv__plain__k" / "program.out").read_text() == "k small 1 hash=0x5\n"
    assert "RISCV_clean (AtomicSimpleCPU)" in capsys.readouterr().out


def test_main_riscv_gem5_failure(tree, tmp_path, monkeypatch):
    """Check that a failing gem5 run is reported with its last stderr line."""
    args = riscv_tree(tree, tmp_path)
    monkeypatch.setenv("FAKE_GEM5_FAIL", "1")
    assert vc.main(args + ["--no-lock", "--kernels", "k", "--variants", "winhint"]) == 1
    gem5_rows = [r for r in read_csv(tmp_path / "c.csv") if r["platform"] == "gem5"]
    assert gem5_rows and all(r["status"] == "FAIL" for r in gem5_rows)
    assert "gem5 boom" in gem5_rows[0]["note"]


def test_main_riscv_gem5_exit_without_stderr(tree, tmp_path):
    """Check the fallback note when gem5 exits non-zero without printing to stderr."""
    args = riscv_tree(tree, tmp_path)
    write_exe(tmp_path / "tools" / "gem5.opt", "#!/bin/sh\nexit 9\n")
    vc.main(args + ["--no-lock"])
    g = [r for r in read_csv(tmp_path / "c.csv") if r["platform"] == "gem5"][0]
    assert g["note"].startswith("exit plain@qemu=0 variant=9 gem5 exit 9 (see ")


def test_main_riscv_lock_busy(tree, tmp_path, monkeypatch, capsys):
    """Check that a busy heavy lock turns gem5 rows into SKIP and abandons later gem5 runs."""
    args = riscv_tree(tree, tmp_path)
    write_exe(tmp_path / "bin" / "flock", BUSY_FLOCK)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    assert vc.main(args + ["--lock-wait", "1"]) == 0
    rows = read_csv(tmp_path / "c.csv")
    gem5 = [r for r in rows if r["platform"] == "gem5"]
    assert [r["status"] for r in gem5] == ["SKIP", "SKIP", "SKIP"]
    assert gem5[0]["note"] == "heavy lock busy for 1s (gem5 vs qemu)"
    assert gem5[1]["note"] == "heavy lock busy for 1s"
    assert gem5[-1]["note"] == "heavy lock busy (gem5 runs abandoned)"
    assert "SKIP=3" in capsys.readouterr().out


def test_main_riscv_gem5_skipped(tree, tmp_path):
    """Check gem5 SKIP rows for a slow plain qemu run and for a plain failing under qemu."""
    args = riscv_tree(tree, tmp_path)
    vc.main(args + ["--no-lock", "--gem5-max-qemu-sec", "-1", "--variants", "winhint", "--kernels", "k"])
    g = [r for r in read_csv(tmp_path / "c.csv") if r["platform"] == "gem5"]
    assert g[0]["status"] == "SKIP" and "> --gem5-max-qemu-sec -1" in g[0]["note"]
    write_exe(tree / "benchmarks" / "riscv" / "plain" / "k", kernel("x", 1))
    assert vc.main(args + ["--no-lock"]) == 1
    g = [r for r in read_csv(tmp_path / "c.csv") if r["platform"] == "gem5"]
    assert g and all(r["note"] == "plain fails under qemu" for r in g)
