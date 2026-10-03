"""Unit tests of the --uarch / --fidelity / --llamacpp modes (no measurements, no privileges)."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW))
import fidelity as fid  # noqa: E402
import run_hw_experiments as rhe  # noqa: E402

BUILD = Path(os.environ.get("WINHINT_BUILD") or HW.parent / "build")
TOOLS = BUILD / "hw"


# ----------------------------------------------------------------------------- µarch
def test_confirm_uarch_knees():
    """Check confirm_uarch() statuses for buffer knees."""
    m = {"x": [0, 504, 512], "size_lo": 505, "size_hi": 513, "knee_x": 512}
    assert fid.confirm_uarch("rob", 512, m)[0] == "confirmed"
    assert fid.confirm_uarch("rob", 256, m)[0] == "differs"
    assert fid.confirm_uarch("lb", None, m)[0] == "undocumented"
    assert fid.confirm_uarch("rob", 512, {"x": [0, 8], "size_lo": None, "size_hi": None})[0] == "no_knee"
    assert fid.confirm_uarch("rob", 512, None)[0] == "missing"


def test_confirm_uarch_mlp():
    """Check confirm_uarch() statuses for the MLP probe, including the consistent_bound case."""
    assert fid.confirm_uarch("mlp", 16, {"x": list(range(1, 13)), "mlp_effective": 11.0})[0] == "consistent_bound"
    assert fid.confirm_uarch("mlp", 16, {"x": list(range(1, 13)), "mlp_effective": 3.0})[0] == "differs"
    assert fid.confirm_uarch("mlp", 10, {"x": list(range(1, 13)), "mlp_effective": 9.0})[0] == "confirmed"
    assert fid.confirm_uarch("mlp", None, {"x": [1, 2], "mlp_effective": 2.0})[0] == "undocumented"


def test_uarch_commands(tmp_path):
    """Check that uarch_commands() targets the first P and E CPU and uses the smoke ranges."""
    a = SimpleNamespace(tool_dir=str(tmp_path), out=str(tmp_path / "o"), uarch_smoke=True)
    c = fid.uarch_commands(a, {"pcpus": "0-3", "ecpus": "4-11"})
    assert c["P"][:3] == [str(tmp_path / "uarch_probe"), "-c", "0"]
    assert c["E"][2] == "4" and c["E"][-1].endswith("uarch_E.json")
    assert "-n" in c["P"]   # smoke: tiny range


@pytest.mark.skipif(not (TOOLS / "uarch_probe").exists(), reason="make -C hw first")
def test_uarch_probe_binary_smoke(tmp_path):
    """Run the built uarch_probe on tiny ranges and check its CSV/JSON output and usage error."""
    out = tmp_path / "u.json"
    p = subprocess.run([str(TOOLS / "uarch_probe"), "-c", "0", "-p", "rob,mlp", "-n", "0:8:8", "-k", "2",
                        "-i", "50", "-m", "4", "-r", "1", "-q", "-o", str(out)],
                       capture_output=True, text=True, check=True)
    assert p.stdout.startswith("probe,cpu,core_type,x,ns_per_iter")
    d = json.loads(out.read_text())
    assert d["cpu"] == 0 and set(d["probes"]) == {"rob", "mlp"}
    assert d["probes"]["rob"]["x"] == [0, 8] and len(d["probes"]["mlp"]["ns_per_iter"]) == 2
    bad = subprocess.run([str(TOOLS / "uarch_probe"), "-p", "nope", "-c", "0", "-m", "4"], capture_output=True)
    assert bad.returncode == 2


# ----------------------------------------------------------------------------- fidelity helpers
def test_parse_workload_and_rate():
    """Check phase_workload output parsing, work_rate() and speedup()."""
    d = fid.parse_workload("phases=4 mem_steps=100 fp_steps=0 chk=7 1.000 ilp_steps=50 "
                           "ns_compute=0 ns_memory=1000 ns_ilp=500")
    assert fid.work_rate(d, "memory") == 0.1 and fid.work_rate(d, "ilp") == 0.1
    assert fid.work_rate(d, "compute") is None
    assert fid.speedup([2.0, 2.2, 1.8], [1.0]) == 2.0
    assert fid.speedup([None], [1.0]) is None


@pytest.mark.skipif(not (TOOLS / "phase_workload").exists(), reason="make -C hw first")
def test_phase_workload_modes():
    """Run the built phase_workload in every mode and check its rates and usage error."""
    for mode, key in (("ilp", "ilp_steps"), ("memory", "mem_steps"), ("compute", "fp_steps")):
        p = subprocess.run([str(TOOLS / "phase_workload"), "0.02", "5", "4", mode],
                           capture_output=True, text=True, check=True)
        d = fid.parse_workload(p.stdout)
        assert d[key] > 0 and fid.work_rate(d, mode) > 0
    assert subprocess.run([str(TOOLS / "phase_workload"), "0.01", "5", "4", "bogus"],
                          capture_output=True).returncode == 2


def test_separating_threshold():
    """Check the geometric-mean threshold and the fallback cases of separating_threshold()."""
    thr, ok = fid.separating_threshold(2.0, 1.125, 1.4)
    assert ok and abs(thr - 1.5) < 1e-12
    assert fid.separating_threshold(1.0, 1.2, 1.4) == (1.4, False)
    assert fid.separating_threshold(None, 1.2, 1.4) == (1.4, False)


def test_pie_parsing_and_tracking(tmp_path):
    """Check pie_daemon CSV parsing (with a broken row), pie_stats() and pie_phase_tracking()."""
    f = tmp_path / "pie.csv"
    f.write_text("t_ms,side,instructions,cpi,mpki,mlp_cur,mlp_oth,freq_ghz,ratio_e_over_p,decision,migrated\n"
                 "10,P,1e6,0.5,0.1,1,1,4,1.6,P,0\n20,P,1e6,2.0,30,4,3,4,1.05,E,0\n"
                 "30,E,1e6,2.5,28,3,4,3,1.04,E,1\n40,E,1e6,0.6,0.2,1,1,3,1.7,P,0\nbroken\n")
    rows = fid.parse_pie_csv(f)
    assert len(rows) == 4
    st = fid.pie_stats(rows)
    assert st["frac_on_e"] == 0.5 and st["frac_want_e"] == 0.5
    hi, lo = fid.pie_phase_tracking(rows)
    assert hi == 1.0 and lo == 0.0
    assert fid.parse_pie_csv(tmp_path / "missing.csv") == []
    assert fid.pie_phase_tracking(rows[:2]) == (None, None)


def test_sondag_env_and_types():
    """Check the sondag environment (WINHINT_* reset) and the filtering of undecided types."""
    env = fid.sondag_env({"PATH": "/bin", "WINHINT_MODE": "migrate", "WINHINT_PCPUS": "0-1"},
                         "/tmp/x.csv", 1.5, 2)
    assert env["WINHINT_MODE"] == "sondag" and env["WINHINT_SONDAG_THRESHOLD"] == "1.5000"
    assert env["WINHINT_PCPUS"] == "0-1" and env["PATH"] == "/bin" and env["WINHINT_PERF"] == "1"
    ty = fid.sondag_types({"sondag": [{"type": 1, "e_over_p": 1.1, "assigned": "E"},
                                      {"type": 2, "e_over_p": 0.0, "assigned": "undecided"}]})
    assert set(ty) == {1}


def _data(good=True):
    """Return synthetic evaluate() input where every trend holds (``good``) or is inverted."""
    pie_i = [{"side": "P", "decision": "P", "mpki": 0.1, "ratio": 1.6}] * 10
    pie_m = [{"side": "E", "decision": "E", "mpki": 30.0, "ratio": 1.05}] * 10
    alt = pie_i[:5] + pie_m[:5]
    sp = {"ilp": 1.7, "compute": 1.4, "memory": 1.1}
    son = {"sondag": [{"type": 2, "e_over_p": 1.65, "assigned": "P"},
                      {"type": 1, "e_over_p": 1.12, "assigned": "E"}]}
    if not good:   # everything inverted
        pie_i, pie_m = pie_m, pie_i
        alt = [dict(r, decision="P" if r["decision"] == "E" else "E") for r in alt]
        sp = {"ilp": 1.1, "compute": 1.0, "memory": 1.7}
        son = {"sondag": [{"type": 2, "e_over_p": 1.1, "assigned": "E"},
                          {"type": 1, "e_over_p": 1.6, "assigned": "P"}]}
    thr, sep = fid.separating_threshold(sp["ilp"], sp["memory"], 1.4)
    return {"speedup": sp, "pie": {"ilp": pie_i, "memory": pie_m, "alt": alt}, "sondag": son,
            "sondag_threshold": thr, "threshold_separating": sep}


def test_evaluate_trends_pass_and_fail():
    """Check that all predicates pass on good data and the trend ones fail on inverted data."""
    good = {r["id"]: r for r in fid.evaluate(_data(True))}
    assert all(r["pass"] for r in good.values()), [k for k, r in good.items() if not r["pass"]]
    trend = {k for k, r in good.items() if r["kind"] == "trend"}
    assert {"GT-mem-gains-less", "R4-pred-order", "R4-place-order", "R4-phase-tracking",
            "R5-order", "R5-assign-ilp-P", "R5-assign-mem-E"} <= trend
    bad = {r["id"]: r for r in fid.evaluate(_data(False))}
    for k in ("GT-mem-gains-less", "R4-pred-order", "R4-place-order", "R4-phase-tracking", "R5-order",
              "R5-assign-ilp-P", "R5-assign-mem-E"):
        assert bad[k]["pass"] is False, k
    # non-separating threshold demotes the assignment checks to info
    assert bad["R5-assign-ilp-P"]["kind"] == "info"


def test_evaluate_missing_data_is_not_evaluable():
    """Check that evaluate() on empty data marks every predicate not evaluable."""
    rows = fid.evaluate({})
    assert rows and all(r["pass"] is None for r in rows)


def test_dry_run_uarch_fidelity(tmp_path):
    """Check the --uarch --fidelity --dry-run plan output and that nothing is written."""
    p = subprocess.run([sys.executable, str(HW / "run_hw_experiments.py"), "--uarch", "--uarch-smoke",
                        "--fidelity", "--dry-run", "--out", str(tmp_path), "--tool-dir", str(tmp_path / "t")],
                       capture_output=True, text=True, check=True)
    assert "uarch_probe -c" in p.stdout and "uarch_E.json" in p.stdout
    assert "pie_daemon -b perf" in p.stdout and " memory" in p.stdout and " ilp" in p.stdout
    assert "WINHINT_MODE=sondag" in p.stdout and "alt-ilp" in p.stdout
    assert not (tmp_path / "fidelity").exists()


# ----------------------------------------------------------------------------- --llamacpp
def _llama_args(tmp_path, configs):
    """Return a minimal Campaign namespace for --llamacpp with the given configurations."""
    return SimpleNamespace(
        out=str(tmp_path), nop_pcpu=None, nop_ecpu=None, tool_dir=str(tmp_path), no_perf_stat=True,
        configs=configs, kernels=[rhe.LLAMACPP], synthetic=False, llamacpp=True,
        llamacpp_dir=str(tmp_path / "ll"), llamacpp_args="", bench_root=str(tmp_path),
        variant_plain="plain", variant_asm="winhint", variant_call="winhint_call",
        variant_regions="oracle_call", require_rapl=False, input="large", threshold=192,
        min_dwell_us=0, hyst=1, winhint_perf=False, sondag_k=2, sondag_threshold=1.4,
        sondag_types_dir="", pie_backend="perf", pie_interval_ms=10, pie_slack=0.15, pie_hyst=2,
        retry_failed=False)


def test_llamacpp_command_construction(tmp_path):
    """Check llama-simple binaries, arguments and environment per configuration."""
    a = _llama_args(tmp_path, ["R1", "WH", "R5-Sondag"])
    c = rhe.Campaign(a)
    ll = tmp_path / "ll"
    argv, env = c.command(rhe.LLAMACPP, "R1", tmp_path / "r")
    i = argv.index("--")
    assert argv[i + 1] == str(ll / "build-vanilla" / "bin" / "llama-simple")
    assert argv[i + 2:] == ["-m", f"{ll}/models/stories15M-q4_0.gguf", "-n", "256", "Once upon a time"]
    assert env["OMP_THREAD_LIMIT"] == "1" and "WINHINT_MODE" not in env
    argv, env = c.command(rhe.LLAMACPP, "WH", tmp_path / "r")
    assert str(ll / "build-winhint" / "bin" / "llama-simple") in argv
    assert env["WINHINT_MODE"] == "migrate" and env["OMP_THREAD_LIMIT"] == "1"
    assert "GGML_WINHINT_SETWIN" not in env
    argv, env = c.command(rhe.LLAMACPP, "R5-Sondag", tmp_path / "r")
    assert str(ll / "build-winhint" / "bin" / "llama-simple") in argv
    assert env["WINHINT_MODE"] == "sondag" and env["GGML_WINHINT_SETWIN"] == "none"
    a.llamacpp_args = "-m x.gguf -n 8 'hi there'"
    argv, _ = c.command(rhe.LLAMACPP, "R1", tmp_path / "r")
    assert argv[-5:] == ["-m", "x.gguf", "-n", "8", "hi there"]


def test_llamacpp_cli_rejects_nop_and_synthetic(tmp_path):
    """Check that --llamacpp rejects NOP configs and --synthetic, and plans single-threaded runs."""
    base = [sys.executable, str(HW / "run_hw_experiments.py"), "--dry-run", "--reps", "1",
            "--allow-no-rapl", "--out", str(tmp_path)]
    p = subprocess.run(base + ["--llamacpp", "--configs", "R1,NOP-asm-P"], capture_output=True, text=True)
    assert p.returncode != 0 and "NOP" in p.stderr
    p = subprocess.run(base + ["--llamacpp", "--synthetic"], capture_output=True, text=True)
    assert p.returncode != 0 and "exclusive" in p.stderr
    p = subprocess.run(base + ["--llamacpp", "--configs", "R1,WH",
                               "--llamacpp-dir", str(tmp_path / "ll")], capture_output=True, text=True)
    assert "llamacpp|WH|" in p.stdout and "OMP_THREAD_LIMIT=1" in p.stdout
