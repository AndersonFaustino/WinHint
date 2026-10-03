"""Tests of sim/fidelity/fidelity.py.

Covers run generation, metrics, predicates and the end-to-end pipeline
against a fake gem5 (conftest.py).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import fidelity as fd

N = 4


def args(tmp_path, machines_dir, fake_gem5, bin_root, *extra):
    """Return the common CLI arguments pointing fidelity.py at the fake environment."""
    return ["--machines-dir", str(machines_dir), "--results", str(tmp_path / "res"),
            "--bin-root", str(bin_root), "--gem5", str(fake_gem5), "--no-lock", *extra]


@pytest.fixture
def pgo_builder(monkeypatch, bin_root):
    """Replace the B7 pgo build: instead of make, copy the fake oracle binary."""
    def fake_cmd(map_dir, machine_json):
        """Return a command that copies the fake oracle binary to ``<bin_root>/pgo``."""
        assert (map_dir / f"{fd.PHASED}.json").exists()
        dst = bin_root / "pgo"
        return [sys.executable, "-c",
                f"import pathlib,shutil; pathlib.Path({str(dst)!r}).mkdir(exist_ok=True); "
                f"shutil.copy({str(bin_root / 'oracle' / fd.PHASED)!r}, {str(dst)!r})"]
    monkeypatch.setattr(fd, "pgo_make_cmd", fake_cmd)


# ---------------------------------------------------------------------------
# Run generation
# ---------------------------------------------------------------------------

def test_resolve_symbolic_configs():
    """``resolve`` maps min/max to indices and tags; micro_phased uses the oracle variant."""
    s = fd.resolve(fd.static(fd.GATHER, "max"), N)
    assert (s.tag, s.initial, s.variant, s.policy) == ("c3", 3, "plain", "static")
    s = fd.resolve(fd.policy_run(fd.GATHER, "mlp_imin", "mlp", "min", "ilp=0"), N)
    assert (s.tag, s.initial, s.window_args) == ("mlp_i0", 0, "ilp=0")
    assert fd.resolve(fd.static(fd.PHASED, 2, "small"), N).variant == "oracle"


def test_collect_runs_dedupes_shared_references():
    """Runs shared by several baselines are listed once with consistent specs."""
    runs = fd.collect_runs(list(fd.BASELINES), N)
    keys = [r.key for r in runs]
    assert len(keys) == len(set(keys))
    # gather static small/large is shared by B2 and B3; phased statics by B4/B5/B7
    assert (fd.GATHER, "large", "c0") in keys and (fd.PHASED, "small", "c3") in keys
    by = {r.key: r for r in runs}
    assert by[(fd.PHASED, "large", "lut_i3")].needs == "lut"
    assert by[(fd.PHASED, "large", "pgo_i3")].variant == "pgo"
    assert by[(fd.COMPUTE, "large", "jones_iq_i3")].window_args == "structs=iq"
    assert by[(fd.GATHER, "large", "smalliq_i0")].window_args == "structs=iq+lq+sq"
    assert by[(fd.GATHER, "large", "ltp_i0")].policy == "ltp"
    assert {r.policy for r in runs} == {"static", "occupancy", "mlp", "bbv", "lut", "hint", "ltp"}


def test_collect_runs_per_baseline_counts():
    """Each baseline contributes the expected number of distinct runs."""
    assert len(fd.collect_runs(["B3"], N)) == 9
    assert len(fd.collect_runs(["B4"], N)) == N + 1
    assert len(fd.collect_runs(["B5", "B7"], N)) == 2 * N + 2


def test_gem5_cmd_uses_only_se_flags(tmp_path):
    """The gem5 command uses only se.py flags (policy, initial config, LUT, trace)."""
    s = fd.resolve(fd.policy_run(fd.PHASED, "lut_imax", "lut", "max", needs="lut"), N)
    cmd = fd.gem5_cmd("gem5.opt", tmp_path / "o", Path("m.json"), Path("bin/oracle/micro_phased"),
                      s, Path("lut.txt"), None)
    assert cmd[cmd.index("--window-policy") + 1] == "lut"
    assert cmd[cmd.index("--window-initial") + 1] == "3"
    assert cmd[cmd.index("--window-lut") + 1] == "lut.txt"
    assert "--window-trace" in cmd and "--window-args" not in cmd
    assert str(fd.SE_PY) in cmd


def test_binary_path_variants(tmp_path):
    """pgo binaries are looked up per machine for non-default machines."""
    s = fd.resolve(fd.policy_run(fd.PHASED, "pgo_imax", "hint", "max", variant="pgo"), N)
    assert fd.binary_path(tmp_path, s, "riscv_ooo") == tmp_path / "pgo" / fd.PHASED
    assert fd.binary_path(tmp_path, s, "riscv_ooo_big") == tmp_path / "pgo" / "riscv_ooo_big" / fd.PHASED


def test_resized_structs():
    """``resized_structs`` honours ``structs=`` for hint and the ltp default."""
    assert fd.resized_structs("hint", "structs=iq") == {"iq"}
    assert fd.resized_structs("hint", "") == {"rob", "iq", "lq", "sq"}
    assert fd.resized_structs("ltp", "") == {"iq", "lq", "sq"}
    assert fd.resized_structs("hint", "structs=iq+lq+sq") == {"iq", "lq", "sq"}
    assert fd.resized_structs("mlp", "ilp=0") == {"rob", "iq", "lq", "sq"}


def test_dry_run_lists_runs_and_derive_steps(tmp_path, machines_dir, fake_gem5, bin_root, capsys):
    """``--dry-run`` lists runs and derive steps without creating run directories."""
    assert fd.main(args(tmp_path, machines_dir, fake_gem5, bin_root, "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "micro_phased/large/lut_i3" in out and "derive: B5 LUT" in out
    assert "micro-pgo" in out and "--window-policy ltp" in out
    assert not (tmp_path / "res" / "runs").exists()


# ---------------------------------------------------------------------------
# Metrics and predicates
# ---------------------------------------------------------------------------

def _stats(d: Path, cyc, occ=None, ipc=1.0):
    """Write a minimal stats.txt with per-config cycles (and optional ROB occupancy sums)."""
    d.mkdir(parents=True, exist_ok=True)
    tot = sum(cyc)
    lines = [f"simInsts {int(tot * ipc)}", f"system.cpu.numCycles {tot}"]
    lines += [f"system.cpu.window.cyclesInConfig::{i} {c}" for i, c in enumerate(cyc)]
    for i, o in enumerate(occ or []):
        lines.append(f"system.cpu.window.robOccSum::{i} {o}")
    (d / "stats.txt").write_text("\n".join(lines) + "\n")


def test_run_metrics_caps_follow_resized_structs(tmp_path):
    """Allocated caps follow only the resized structures; occupancy is cycle-weighted."""
    table = [{"rob": r, "iq": q, "lq": l, "sq": l} for r, q, l in
             zip([64, 128, 192, 256], [32, 64, 96, 128], [16, 32, 48, 64])]
    _stats(tmp_path / "a", [500, 0, 0, 500], occ=[500 * 30, 0, 0, 500 * 100])
    s = fd.RunSpec(fd.COMPUTE, "x", "jones", "hint", 3, "structs=iq")
    m = fd.run_metrics(tmp_path / "a", s, table)
    assert m["iq_cap"] == pytest.approx(80.0) and m["rob_cap"] == 256
    assert m["residency"] == [0.5, 0, 0, 0.5]
    assert m["rob_occ"] == pytest.approx(65.0)
    m = fd.run_metrics(tmp_path / "a", fd.RunSpec(fd.COMPUTE, "x", "plain", "mlp", 0), table)
    assert m["rob_cap_frac"] == pytest.approx(160 / 256)


class FakeCtx(fd.Ctx):
    """``Ctx`` serving canned metrics instead of reading run directories."""
    def __init__(self, metrics):
        """Create the context with ``metrics`` keyed by ``(kernel, tag)``."""
        super().__init__(Path("/nonexistent"), {"name": "riscv_ooo", "window_table": []}, N)
        self.metrics = metrics

    def m(self, kernel, tag, input_size="large"):
        """Return the canned metrics, raising ``Missing`` for unknown runs."""
        try:
            return self.metrics[(kernel, tag)]
        except KeyError:
            raise fd.Missing(f"{kernel}/{tag}") from None


def _m(ipc, res0=None, cap=None, occ=None, iq_cap=None):
    """Return a metrics dict with the given IPC and optional residency/cap/occupancy fields."""
    d = {"ipc": ipc}
    if res0 is not None:
        d["residency"] = [res0, 0, 0, 1 - res0]
    if cap is not None:
        d.update({f"{s}_cap_frac": cap for s in ("rob", "iq", "lq", "sq")})
    if occ is not None:
        d["rob_occ_frac"] = occ
    if iq_cap is not None:
        d["iq_cap_frac"] = iq_cap
    return d


def b3_metrics(chase_mlp=0.1, chase_res=1.0, gather_large=1.0):
    """Return canned B3 metrics; arguments perturb chase and gather to provoke FAIL/INVALID."""
    return {(fd.GATHER, "c0"): _m(0.4), (fd.GATHER, "c3"): _m(gather_large),
            (fd.GATHER, "mlp_i0"): _m(0.85, 0.1),
            (fd.CHASE, "c0"): _m(0.1), (fd.CHASE, "c3"): _m(0.1),
            (fd.CHASE, "mlp_i0"): _m(chase_mlp, chase_res),
            (fd.COMPUTE, "c0"): _m(2.8), (fd.COMPUTE, "c3"): _m(2.86),
            (fd.COMPUTE, "mlp_i0"): _m(2.8, 1.0)}


def test_b3_predicate_pass_fail_invalid():
    """B3 predicate yields PASS, FAIL (chase gain or enlargement) and INVALID as designed."""
    assert fd.verdict(fd.eval_B3(FakeCtx(b3_metrics()))) == "PASS"
    # enlarging on pointer chasing (residency in ILP mode low) -> FAIL
    checks = fd.eval_B3(FakeCtx(b3_metrics(chase_res=0.2)))
    assert fd.verdict(checks) == "FAIL"
    assert [c.name for c in checks if not c.ok] == ["chase: mlp residency in the ILP-mode config"]
    # speedup on pointer chasing is also a failure of "no gain"
    assert fd.verdict(fd.eval_B3(FakeCtx(b3_metrics(chase_mlp=0.2)))) == "FAIL"
    # gather does not benefit from a large window on this machine -> INVALID
    assert fd.verdict(fd.eval_B3(FakeCtx(b3_metrics(gather_large=0.42)))) == "INVALID"


def test_evaluate_reports_incomplete():
    """Missing runs make every check missing and the verdict INCOMPLETE."""
    ctx = FakeCtx({})
    ctx.used = {}
    res = fd.evaluate(ctx, "B3")
    assert res["status"] == "INCOMPLETE"
    assert any("micro_gather" in m for m in res["missing"])
    assert all("missing" in c for c in res["checks"])


def test_b2_b6_b9_predicates():
    """B2, B6 and B9 predicates distinguish PASS, FAIL and INVALID."""
    m = {(fd.LOWILP, "c3"): _m(0.9, occ=0.12), (fd.LOWILP, "occupancy_i3"): _m(0.89, cap=0.3),
         (fd.GATHER, "c0"): _m(0.4), (fd.GATHER, "c3"): _m(1.0),
         (fd.GATHER, "occupancy_i0"): _m(0.8, cap=0.8)}
    assert fd.verdict(fd.eval_B2(FakeCtx(m))) == "PASS"
    m[(fd.LOWILP, "occupancy_i3")] = _m(0.89, cap=0.3) | {"sq_cap_frac": 0.95}
    assert fd.verdict(fd.eval_B2(FakeCtx(m))) == "FAIL"           # SQ never shrinks
    m[(fd.LOWILP, "occupancy_i3")] = _m(0.89, cap=0.95)     # never shrinks
    assert fd.verdict(fd.eval_B2(FakeCtx(m))) == "FAIL"
    m[(fd.LOWILP, "c3")] = _m(0.9, occ=0.9)                 # window not under-used
    assert fd.verdict(fd.eval_B2(FakeCtx(m))) == "INVALID"

    m = {}
    for k in (fd.COMPUTE, fd.LOWILP, fd.GATHER):
        m[(k, "c3")] = _m(1.0)
        m[(k, "jones_iq_i3")] = _m(0.99, iq_cap=0.3)
    assert fd.verdict(fd.eval_B6(FakeCtx(m))) == "PASS"
    m[(fd.GATHER, "jones_iq_i3")] = _m(0.9, iq_cap=0.3)     # 10% IPC loss
    assert fd.verdict(fd.eval_B6(FakeCtx(m))) == "FAIL"

    m = {(fd.GATHER, "smalliq_i0"): _m(0.5), (fd.GATHER, "ltp_i0"): _m(0.85),
         (fd.GATHER, "c3"): _m(1.0), (fd.COMPUTE, "smalliq_i0"): _m(2.8),
         (fd.COMPUTE, "ltp_i0"): _m(2.79)}
    assert fd.verdict(fd.eval_B9(FakeCtx(m))) == "PASS"
    m[(fd.GATHER, "ltp_i0")] = _m(0.55)                     # recovers only 10%
    assert fd.verdict(fd.eval_B9(FakeCtx(m))) == "FAIL"


def test_region_table_best_and_near_best(tmp_path):
    """Region table, best configs, phase regions and near-best fractions on toy data."""
    for c, cyc in enumerate([(1000, 100), (800, 101), (700, 102), (690, 103)]):
        d = tmp_path / f"c{c}"
        d.mkdir()
        (d / "region_stats.csv").write_text(
            "region,config,enter_cycle,cycles,insts\n"
            f"1,{c},0,{cyc[0]},700\n2,{c},{cyc[0]},{cyc[1]},300\n")
    tab = fd.region_table({c: tmp_path / f"c{c}" for c in range(4)})
    assert fd.best_configs(tab, 0.02) == {1: 2, 2: 0}
    assert fd.phase_regions(tab, 0.05) == [1, 2]
    tr = tmp_path / "window_trace.csv"
    tr.write_text("cycle,insts,ipc,rob_occ_mean,iq_occ_mean,lq_occ_mean,l1d_mpki,l2_mpki,mlp,"
                  "branch_mpki,config,region\n"
                  "1000,700,0.7,1,1,1,0,0,0,0,0,1\n2000,1000,1.0,1,1,1,0,0,0,0,3,1\n"
                  "3000,2900,2.9,1,1,1,0,0,0,0,0,2\n4000,2900,2.9,1,1,1,0,0,0,0,0,-1\n")
    nb = fd.near_best_fraction(tr, tab, 0.97)
    assert nb["near_best_frac"] == pytest.approx(2 / 3)
    assert nb["exact_frac"] == pytest.approx(1 / 3)


def test_read_bbv_phases(tmp_path):
    """``read_bbv_phases`` parses header counters and per-phase rows."""
    p = tmp_path / "bbv_phases.csv"
    p.write_text("# intervals=100 predicted_correct=90 phases_allocated=5\n"
                 "phase,visits,learned_config,ipc_c0,n_c0\n0,40,3,1,1\n1,30,0,1,1\n"
                 "2,20,0,1,1\n3,5,1,1,1\n4,5,1,1,1\n")
    ph = fd.read_bbv_phases(p)
    assert ph["accuracy"] == pytest.approx(0.9)
    assert ph["top3_cover"] == pytest.approx(0.9)
    assert ph["phases_allocated"] == 5 and ph["learned"][0] == 3


# ---------------------------------------------------------------------------
# End to end against the fake gem5
# ---------------------------------------------------------------------------

def test_end_to_end_all_baselines_pass(tmp_path, machines_dir, fake_gem5, bin_root, pgo_builder,
                                       capsys):
    """Every baseline PASSes against the fake gem5, and reruns are skipped."""
    assert fd.main(args(tmp_path, machines_dir, fake_gem5, bin_root)) == 0
    res = tmp_path / "res"
    summary = json.loads((res / "summary.json").read_text())
    assert summary == {b: "PASS" for b in fd.BASELINES}, capsys.readouterr().out
    b3 = json.loads((res / "B3.json").read_text())
    assert b3["status"] == "PASS" and b3["paper"].startswith("Kora")
    assert all(c["ok"] for c in b3["checks"])
    assert "micro_gather/large/mlp_i0" in b3["runs"]
    assert (res / "b5" / "riscv_ooo" / "lut.txt").read_text().startswith("WINHINT_LUT 1")
    assert json.loads((res / "oracle" / "riscv_ooo" / "small" / "micro_phased.json").read_text()) \
        == {"3": 3, "4": 0, "5": 0}
    run = json.loads((res / "runs" / "riscv_ooo" / "micro_gather" / "large" / "mlp_i0"
                      / "run.json").read_text())
    assert run["status"] == "ok" and run["spec"]["policy"] == "mlp"
    # resumable: a second invocation reruns nothing
    capsys.readouterr()
    assert fd.main(args(tmp_path, machines_dir, fake_gem5, bin_root, "--baselines", "B3")) == 0
    out = capsys.readouterr().out
    assert "[RUN ]" not in out and out.count("[SKIP]") == 9


def test_end_to_end_detects_failure(tmp_path, machines_dir, fake_gem5, bin_root, monkeypatch):
    """A misbehaving mlp policy is reported as FAIL."""
    monkeypatch.setenv("FAKE_MODE", "bad_mlp")    # mlp enlarges on pointer chasing
    assert fd.main(args(tmp_path, machines_dir, fake_gem5, bin_root, "--baselines", "B3")) == 0
    b3 = json.loads((tmp_path / "res" / "B3.json").read_text())
    assert b3["status"] == "FAIL"
    assert any(not c["ok"] and "residency" in c["name"] for c in b3["checks"])


def test_only_filter_gives_partial_evaluation(tmp_path, machines_dir, fake_gem5, bin_root, capsys):
    """``--only`` restricts runs; predicates needing other runs are INCOMPLETE."""
    assert fd.main(args(tmp_path, machines_dir, fake_gem5, bin_root, "--baselines", "B3",
                        "--only", "micro_gather/*")) == 0
    assert capsys.readouterr().out.count("[RUN ]") == 3
    b3 = json.loads((tmp_path / "res" / "B3.json").read_text())
    assert b3["status"] == "INCOMPLETE"
    by = {c["name"]: c for c in b3["checks"]}
    assert by["gather: IPC(mlp) / IPC(static small = ILP mode)"]["ok"]
    assert "missing" in by["chase: mlp residency in the ILP-mode config"]
    assert set(b3["runs"]) == {"micro_gather/large/c0", "micro_gather/large/c3",
                               "micro_gather/large/mlp_i0"}


def test_missing_binary_is_reported(tmp_path, machines_dir, fake_gem5, bin_root):
    """A missing binary makes the run fail and the verdict INCOMPLETE (exit code 1)."""
    shutil.rmtree(bin_root / "jones")
    assert fd.main(args(tmp_path, machines_dir, fake_gem5, bin_root, "--baselines", "B6")) == 1
    assert json.loads((tmp_path / "res" / "B6.json").read_text())["status"] == "INCOMPLETE"


def test_cli_help():
    """``fidelity.py --help`` works as a script."""
    p = subprocess.run([sys.executable, str(fd.HERE / "fidelity.py"), "--help"],
                       capture_output=True, text=True)
    assert p.returncode == 0 and "--baselines" in p.stdout
