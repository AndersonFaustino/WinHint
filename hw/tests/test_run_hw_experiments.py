"""Unit tests of the real-hardware campaign driver (no measurements, no privileges)."""
import csv
import subprocess
import sys
from pathlib import Path

HW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW))
import run_hw_experiments as rhe  # noqa: E402


def test_intersect_cpulist():
    """Check CPU-list intersection with ranges."""
    assert rhe.intersect_cpulist("0-3,8", "0-1,4-11") == "0,1,8"
    assert rhe.intersect_cpulist("4-11", "0-11") == "4,5,6,7,8,9,10,11"


def test_mean_ci():
    """Check mean_ci() on a small sample and a single value."""
    m, lo, hi = rhe.mean_ci([1.0, 2.0, 3.0])
    assert m == 2.0 and lo < 2.0 < hi
    assert rhe.mean_ci([5.0])[0] == 5.0


def test_summarize_ratios_and_nop(tmp_path):
    """Check summarize() R1 normalisation and the NOP-hint overhead."""
    rows = []
    for rep in range(3):
        for cfg, t, e in (("R1", 1.0, 10.0), ("WH", 0.9, 8.0), ("NOP-plain-P", 1.0, 1.0),
                          ("NOP-asm-P", 1.01, 1.0)):
            r = {f: "" for f in rhe.RAW_FIELDS}
            r.update(run_id=f"k|{cfg}|smt-on|{rep}", kernel="k", config=cfg, smt="on", rep=rep,
                     exit_code=0, wall_s=t, energy_pkg_j=e, edp_pkg_js=e * t, output_ok=1)
            rows.append(r)
    with open(tmp_path / "raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rhe.RAW_FIELDS)
        w.writeheader()
        w.writerows(rows)
    rhe.summarize(tmp_path)
    s = {r["config"]: r for r in csv.DictReader(open(tmp_path / "summary.csv"))}
    assert float(s["R1"]["edp_pkg_js_vs_R1"]) == 1.0
    assert abs(float(s["WH"]["edp_pkg_js_vs_R1"]) - 0.72) < 1e-9
    assert abs(float(s["NOP-asm-P"]["nop_overhead_pct"]) - 1.0) < 1e-9


def test_dry_run_synthetic(tmp_path):
    """Check the --synthetic --dry-run plan output and that no raw.csv is written."""
    p = subprocess.run([sys.executable, str(HW / "run_hw_experiments.py"), "--dry-run", "--synthetic",
                        "--reps", "1", "--allow-no-rapl", "--out", str(tmp_path),
                        "--tool-dir", str(tmp_path / "tools"),
                        "--configs", "R0-P,R0-E,R1,R4-PIE,R5-Sondag,WH,NOP-asm-P"],
                       capture_output=True, text=True, check=True)
    out = p.stdout
    assert "phase_workload|R1|" in out and "phase_workload_hinted" in out
    assert "WINHINT_MODE=sondag" in out and "WINHINT_MODE=migrate" in out
    assert "phase_workload_nop" in out and "pie_daemon" in out
    assert "privileges:" in out
    assert not (tmp_path / "raw.csv").exists()


def test_topology_env_override_and_non_hybrid(monkeypatch):
    """Check the WINHINT_PCPUS/ECPUS override and require_hybrid() on a non-hybrid topology."""
    monkeypatch.setenv("WINHINT_PCPUS", "0-1")
    monkeypatch.setenv("WINHINT_ECPUS", "2-3")
    t = rhe.topology()
    assert t["source"] == "env"
    if t["online"]:   # restricted to online CPUs
        assert set(t["pcpus"].split(",")) <= {"0", "1"}
    rhe.require_hybrid(t)   # hybrid: no exit
    monkeypatch.setenv("WINHINT_ECPUS", "100000")   # offline/non-existent -> no E-cores
    t = rhe.topology()
    assert not t["hybrid"]
    try:
        rhe.require_hybrid(t)
    except SystemExit as e:
        assert "not a hybrid P/E CPU" in str(e)
    else:
        raise AssertionError("require_hybrid did not exit")


def test_nop_overhead():
    """Check nop_overhead() value, CI half-width and the missing-side case."""
    ov, half = rhe.nop_overhead((1.01, 1.00, 1.02), (1.0, 0.99, 1.01))
    assert abs(ov - 1.0) < 1e-9 and 0 < half < 2
    assert rhe.nop_overhead(None, (1.0, 0.9, 1.1)) == ("", "")


def test_dry_run_fails_clearly_without_hybrid(tmp_path):
    """Check that a dry run exits with a clear error on a non-hybrid topology."""
    env = dict(__import__("os").environ, WINHINT_PCPUS="0", WINHINT_ECPUS="100000")
    p = subprocess.run([sys.executable, str(HW / "run_hw_experiments.py"), "--dry-run", "--synthetic",
                        "--reps", "1", "--allow-no-rapl", "--out", str(tmp_path)],
                       capture_output=True, text=True, env=env)
    assert p.returncode != 0 and "not a hybrid P/E CPU" in p.stderr


def test_campaign_policy_rules(tmp_path):
    """reps >= 10 without a fixed governor is a preflight problem unless explicitly allowed."""
    class A:  # minimal args
        """Minimal stand-in for the parsed arguments."""
        pass
    a = A()
    for k, v in dict(out=str(tmp_path), nop_pcpu=None, nop_ecpu=None, tool_dir=str(tmp_path),
                     no_perf_stat=True, configs=["R1"], kernels=["k"], synthetic=False,
                     bench_root=str(tmp_path), variant_plain="plain", require_rapl=False,
                     enable_scx=False, enable_lpmd=False, governor="", epp="", no_turbo=None,
                     smt=["current"], dry_run=False, reps=10, allow_unfixed_governor=False,
                     pie_backend="perf", input="large").items():
        setattr(a, k, v)
    c = rhe.Campaign(a)
    probs, _ = c.preflight()
    gov = rhe.freq_policy()[0]
    assert any("governor" in p for p in probs) == (gov != "performance")
    a.allow_unfixed_governor = True
    probs, _ = c.preflight()
    assert not any("governor" in p for p in probs)


def test_x86_nop_hint_scan(tmp_path):
    # setwin(256) = kind 1, payload 32; region(5) = kind 2, payload 5 (docs/interfaces.md §2)
    """Check x86_nop_hints() on a synthetic byte stream and a missing file."""
    def nop(kind, payload):
        """Return the encoded x86 NOP hint bytes for (kind, payload)."""
        return b"\x0f\x1f\x80" + (0x57480000 | kind << 12 | payload).to_bytes(4, "little")
    f = tmp_path / "bin"
    f.write_bytes(b"\x90" * 7 + nop(1, 32) + b"\xc3" + nop(2, 5) + nop(1, 32) + b"\x0f\x1f\x80\0\0\0\0")
    assert rhe.x86_nop_hints(f) == {(1, 32): 2, (2, 5): 1}
    assert rhe.x86_nop_hints(tmp_path / "missing") == {}


def test_run_one_whitespace_only_stderr(tmp_path, monkeypatch, capsys):
    """A failing run whose stderr is only whitespace is recorded without an IndexError."""
    monkeypatch.setattr(rhe, "topology", lambda: {"pcpus": "0", "ecpus": "1", "hybrid": True})
    monkeypatch.setattr(rhe, "require_hybrid", lambda t: None)

    class A:  # minimal args
        """Minimal stand-in for the parsed arguments."""
        pass
    a = A()
    for k, v in dict(out=str(tmp_path), nop_pcpu=None, nop_ecpu=None, tool_dir=str(tmp_path),
                     no_perf_stat=True, input="large", dry_run=False, retry_failed=False,
                     cooldown=0, timeout=30, synthetic=False).items():
        setattr(a, k, v)
    c = rhe.Campaign(a)
    prog = [sys.executable, "-c", "import sys; sys.stderr.write(' \\n\\t\\n'); sys.exit(3)"]
    monkeypatch.setattr(c, "command", lambda kernel, config, rundir: (prog, None))
    assert c.run_one("k", "R1", "base", "on", 0) == "FAIL(3)"
    assert "stderr:" not in capsys.readouterr().out
    assert c.done["k|R1|smt-on|0"]["exit_code"] == 3
