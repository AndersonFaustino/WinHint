"""Tests of run-length control: sim/run_lengths.{json,py}, se.py flags, sampled pipeline.

Covers the sampled pipeline of run_experiments/oracle_sweep with no gem5
needed: a fake gem5.opt (``FAKE_GEM5``) models a program of region visits
(``SEGMENTS``) and implements se.py's profile, checkpoint and restore
protocol. se.py itself is imported with a stub ``m5`` to test its argument
parsing, and winhint_elf is tested on a hand-built ELF.
"""
import importlib.util
import json
import struct
import sys
import types

import pandas as pd
import pytest

import oracle_sweep
import run_experiments as rx
import run_lengths as rl
import winhint_elf

ROB = [64, 128, 192, 256]
BEST = {0: 3, 1: 0, 2: 1}
#: the fake program: (region, instructions); -1 = code before the first marker
SEGMENTS = [(-1, 1000), (0, 3000), (1, 2000), (2, 5000), (0, 3000), (1, 2000), (2, 4000)]

FAKE_GEM5 = r'''#!/usr/bin/env python3
"""Fake gem5.opt for the run-length protocol of sim/se.py."""
import json, pathlib, sys
argv = sys.argv[1:]
def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default
out = pathlib.Path([a for a in argv if a.startswith("--outdir=")][0].split("=", 1)[1])
out.mkdir(parents=True, exist_ok=True)
SEG = %(segments)s
BEST = {0: 3, 1: 0, 2: 1}
ROB = [64, 128, 192, 256]
binary = opt("--cmd")
hinted = "winhint" in binary
starts, n = [], 0
for r, length in SEG:
    starts.append(n)
    n += length
total = n
def cfg_for(w):
    if w == 0: return 3
    return next((i for i, r in enumerate(ROB) if r >= w), 3)
if opt("--profile-regions"):
    visits, cnt = [], {}
    for (r, length), s in zip(SEG, starts):
        if r >= 0:
            cnt[r] = cnt.get(r, 0) + 1
            visits.append([r, cnt[r], s + 1])
    pathlib.Path(opt("--profile-regions")).write_text(json.dumps(
        {"version": 1, "total_insts": total, "visits": visits, "truncated": False}))
    sys.exit(0)
if opt("--take-checkpoints"):
    d = pathlib.Path(opt("--checkpoint-dir"))
    req = [int(x) for x in opt("--take-checkpoints").split(",")]
    for c in req:
        last = None
        for (r, length), s in zip(SEG, starts):
            if r >= 0 and s + 1 <= c and hinted:
                last = ROB[BEST[r]]
        (d / f"cpt.{c}").mkdir(parents=True, exist_ok=True)
        (d / f"cpt.{c}" / "winhint_ff.json").write_text(json.dumps({"last_setwin": last}))
    (d / "checkpoints.json").write_text(json.dumps({"requested": req, "written": req}))
    sys.exit(0)
pol, cfg = opt("--window-policy"), int(opt("--window-initial"))
start = 0
rest = opt("--restore-checkpoint")
if rest:
    start = int(pathlib.Path(rest).name.split(".")[1])
    ff = json.loads((pathlib.Path(rest) / "winhint_ff.json").read_text())
    if pol == "hint" and ff.get("last_setwin") is not None:
        cfg = cfg_for(ff["last_setwin"])        # se.py --window-seed auto
warm = int(opt("--warmup-insts", 0))
meas = int(opt("--maxinsts", 0)) or total
end = min(total, start + warm + meas)
cycle = float(start)                            # fake: 1 cycle/inst before the restore
rows, cur, meas_start, cyc_meas, cfg_cyc = [], None, None, 0.0, {}
for i in range(start, end):
    if i == start + warm:
        meas_start = int(cycle)
    seg = max(j for j, s in enumerate(starts) if s <= i)
    r = SEG[seg][0]
    if r >= 0 and i == starts[seg]:             # the marker retires: region entry
        if cur: rows.append(cur)
        if hinted and pol == "hint":
            cfg = BEST[r]
        cur = [r, cfg, int(cycle), 0.0, 0]
    cpi = 1.0 if r < 0 else 1.0 + 0.05 * abs(cfg - BEST[r])
    if cur: cur[3] += cpi; cur[4] += 1
    if meas_start is not None:
        cyc_meas += cpi
        cfg_cyc[cfg] = cfg_cyc.get(cfg, 0.0) + cpi
    cycle += cpi
if cur: rows.append(cur)
(out / "region_stats.csv").write_text("region,config,enter_cycle,cycles,insts\n" + "".join(
    f"{r},{c},{e},{int(round(y))},{k}\n" for r, c, e, y, k in rows))
stats = [f"simInsts {end - start - warm}", f"system.cpu.numCycles {cyc_meas:.1f}",
         f"system.cpu.ipc {(end - start - warm) / max(cyc_meas, 1):.6f}"]
stats += [f"system.cpu.window.cyclesInConfig::{c} {v:.1f}" for c, v in sorted(cfg_cyc.items())]
(out / "stats.txt").write_text("---------- Begin Simulation Statistics ----------\n"
                               + "\n".join(stats) + "\n---------- End Simulation Statistics   ----------\n")
if rest or warm:
    (out / "runlength.json").write_text(json.dumps(
        {"measure_start_cycle": meas_start, "measured_insts": end - start - warm,
         "hint_state_exact": True}))
''' % {"segments": repr(SEGMENTS)}


def true_cycles(cfg: int) -> float:
    """Return the exact whole-program cycles of the fake program in static config ``cfg``."""
    return sum(n * (1.0 if r < 0 else 1.0 + 0.05 * abs(cfg - BEST[r])) for r, n in SEGMENTS)


def profile() -> dict:
    """Return the region profile se.py --profile-regions would write for ``SEGMENTS``."""
    visits, cnt, n = [], {}, 0
    for r, length in SEGMENTS:
        if r >= 0:
            cnt[r] = cnt.get(r, 0) + 1
            visits.append([r, cnt[r], n + 1])
        n += length
    return {"total_insts": n, "visits": visits, "truncated": False}


@pytest.fixture
def cfg_file(tmp_path):
    """Write a test run-length config (v7; ``large``: regions, 500 + 1500 insts) and return it."""
    p = tmp_path / "run_lengths.json"
    p.write_text(json.dumps({"version": 7, "defaults": {
        "small": {"mode": "full"},
        "large": {"mode": "regions", "warmup_insts": 500, "measure_insts": 1500,
                  "per_region": 1, "max_samples": 10, "full_below_insts": 0}}}))
    return p


@pytest.fixture
def fake_gem5(tmp_path):
    """Write the ``FAKE_GEM5`` script as an executable gem5.opt and return its path."""
    g = tmp_path / "gem5.opt"
    g.write_text(FAKE_GEM5)
    g.chmod(0o755)
    return g


# ---------------------------------------------------------------------------
# Plan and merge
# ---------------------------------------------------------------------------

def test_spec_defaults_overrides_and_shipped_config():
    """Check spec_for() on the shipped sim/run_lengths.json and that it names only real kernels."""
    cfg = rl.load_config()                        # the shipped sim/run_lengths.json
    assert cfg["version"] >= 1 and "_doc" in cfg
    assert rl.spec_for(cfg, "encoder_bert_tiny_infer", "small")["mode"] == "full"
    big = rl.spec_for(cfg, "encoder_bert_tiny_infer", "large")
    assert big["mode"] == "regions" and big["measure_insts"] > 0 and big["version"] == cfg["version"]
    assert rl.spec_for(cfg, "decoder_gpt2_infer", "large")["per_region"] == 3
    for k in cfg["kernels"]:
        assert k in rx.all_kernels()


def test_plan_regions_strata_margin_and_caps():
    """Check plan(): strata, MARGIN/warm-up, per_region, max_samples, fixed mode, errors."""
    spec = {"mode": "regions", "warmup_insts": 500, "measure_insts": 1500, "per_region": 1,
            "max_samples": 0, "full_below_insts": 0, "version": 3}
    pl = rl.plan(spec, profile())
    assert pl.mode == "regions" and pl.version == 3 and pl.total_insts == 20000
    by = {s.stratum: s for s in pl.samples}
    assert set(by) == {-1, 0, 1, 2}
    assert by[-1].start == 0 and by[-1].warmup == 0 and by[-1].stratum_insts == 1000
    # region 0: marker retires at insts 1001; the checkpoint is MARGIN + warmup before it
    s0 = by[0]
    assert s0.start == 1001 - rl.MARGIN - 500 and s0.warmup == 500 and s0.measure == 1500
    assert s0.stratum_insts == 6000 and s0.visit == 1
    assert pl.starts == sorted(s.start for s in pl.samples if s.start > 0)
    # per_region 2 samples the first and the last visit; max_samples keeps the largest strata
    pl2 = rl.plan(dict(spec, per_region=2, max_samples=4), profile())
    assert len(pl2.samples) == 4 and {s.stratum for s in pl2.samples} == {0, 2}
    assert sorted(s.visit for s in pl2.samples if s.stratum == 2) == [1, 2]
    assert pl2.unsampled_frac == pytest.approx((1000 + 4000) / 20000)
    # short visit: the measurement stays inside it
    pl3 = rl.plan(dict(spec, measure_insts=10 ** 6), profile())
    assert {s.stratum: s.measure for s in pl3.samples}[1] == 2000 + rl.MARGIN - 1
    # small programs are simulated in full; fixed mode; errors
    assert rl.plan(dict(spec, full_below_insts=10 ** 6), profile()).mode == "full"
    fx = rl.plan({"mode": "fixed", "starts": [5000], "warmup_insts": 100, "measure_insts": 10})
    assert [(s.start, s.warmup, s.measure) for s in fx.samples] == [(4900, 100, 10)]
    assert fx.total_insts is None
    fspec = {"mode": "fixed", "starts": [5000], "measure_insts": 10}
    assert rl.plan(fspec, {}).total_insts is None          # empty profile: not {}
    assert rl.plan(fspec, {"total_insts": 7000}).total_insts == 7000
    with pytest.raises(ValueError):
        rl.plan(spec, None)
    assert rl.sample_args(s0, "/ck") == ["--restore-checkpoint", f"/ck/cpt.{s0.start}",
                                         "--warmup-insts", "500"]


def _sample(d, cycles, insts, cfg_cycles, rows, start_cycle, trace=None, occ=None):
    """Write a synthetic sample outdir (stats, region_stats, runlength.json, trace).

    Args:
        d (Path): Sample directory (created).
        cycles (int): ``system.cpu.numCycles``.
        insts (int): ``simInsts`` and ``measured_insts``.
        cfg_cycles (dict[int, int]): ``{config: cyclesInConfig}``.
        rows (list[list[int]]): region_stats.csv rows
            ``[region, config, enter_cycle, cycles, insts]``.
        start_cycle (int): ``measure_start_cycle``.
        trace (list[int] | None): Optional window_trace.csv ``cycle`` values.
        occ (dict[int, int] | None): Optional ``{config: robOccSum}`` (adds a
            placeholder robOccMean).
    """
    d.mkdir(parents=True)
    lines = [f"simInsts {insts}", f"system.cpu.numCycles {cycles}",
             f"system.cpu.ipc {insts / cycles}", "simFreq 1000000000000",
             "system.cpu.window.robOccMax::0 40"]
    lines += [f"system.cpu.window.cyclesInConfig::{c} {v}" for c, v in cfg_cycles.items()]
    for c, v in (occ or {}).items():
        lines += [f"system.cpu.window.robOccSum::{c} {v}", f"system.cpu.window.robOccMean::{c} 1"]
    (d / "stats.txt").write_text("---------- Begin Simulation Statistics ----------\n"
                                 + "\n".join(lines) + "\n")
    (d / "region_stats.csv").write_text("region,config,enter_cycle,cycles,insts\n" + "".join(
        f"{','.join(map(str, r))}\n" for r in rows))
    (d / "runlength.json").write_text(json.dumps({"measure_start_cycle": start_cycle,
                                                  "measured_insts": insts}))
    if trace:
        (d / "window_trace.csv").write_text("cycle,insts,ipc,config,region\n" + "".join(
            f"{c},500,0.5,0,0\n" for c in trace))


def test_merge_samples_weights_regions_and_trace(tmp_path):
    """Check merge_samples(): stratum scaling, max/mean merging, row clipping, trace rebasing."""
    pl = rl.Plan("regions", 1, [rl.Sample(0, 0, 0, 100, 0, 1000),
                                rl.Sample(1, 500, 50, 100, 1, 300)], 1300)
    a, b = tmp_path / "s00", tmp_path / "s01"
    _sample(a, 200, 100, {0: 200}, [[0, 0, 0, 200, 100]], 0, trace=[1000, 2000], occ={0: 2000})
    # s01: a warm-up row (dropped), a straddling row (clipped), a measured row
    _sample(b, 50, 100, {3: 50}, [[2, 3, 10, 40, 40], [1, 3, 90, 20, 40], [1, 3, 110, 40, 60]],
            100, trace=[100, 150])
    summ = rl.merge_samples(tmp_path, pl, [a, b])
    st = dict(rl.read_stats(tmp_path / "stats.txt"))
    # stratum 0: x10 (1000/100), stratum 1: x3 (300/100)
    assert summ["scales"] == [10.0, 3.0]
    assert st["simInsts"] == 1300 and st["system.cpu.numCycles"] == 200 * 10 + 50 * 3
    assert st["system.cpu.ipc"] == pytest.approx(1300 / 2150)
    assert st["system.cpu.window.cyclesInConfig::3"] == 150
    assert st["system.cpu.window.robOccMax::0"] == 40 and st["simFreq"] == 1e12
    assert st["system.cpu.window.robOccMean::0"] == pytest.approx(20000 / 2000)
    meas = dict(rl.read_stats(tmp_path / "stats.measured.txt"))
    assert meas["simInsts"] == 200 and meas["system.cpu.numCycles"] == 250
    rs = pd.read_csv(tmp_path / "region_stats.csv")
    assert rs.region.tolist() == [0, 1, 1]
    assert rs.iloc[1].tolist() == [1, 3, 100, 10, 20]        # clipped to the measurement
    tr = pd.read_csv(tmp_path / "window_trace.csv")
    assert tr.cycle.tolist() == [1000, 2000, 2050]           # s01 rebased after s00, warm-up row dropped
    assert json.loads((tmp_path / "runlength.json").read_text())["plan"]["mode"] == "regions"


# ---------------------------------------------------------------------------
# se.py: real argparse with a stub m5
# ---------------------------------------------------------------------------

def import_real_se(monkeypatch):
    """Import the real sim/se.py with a stub ``m5`` module and return it."""
    m5 = types.ModuleType("m5")
    objs = types.ModuleType("m5.objects")
    objs.__getattr__ = lambda name: type(name, (), {"_params": {"window_policy": None}})
    m5.objects = objs
    monkeypatch.setitem(sys.modules, "m5", m5)
    monkeypatch.setitem(sys.modules, "m5.objects", objs)
    spec = importlib.util.spec_from_file_location("se_real_rl", rx.SE_PY)
    se = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(se)
    return se


def test_se_run_length_flags(tmp_path, monkeypatch, machine_json):
    """Check se.py's run-length modes, param_max_insts() and rejected flag combinations."""
    se = import_real_se(monkeypatch)
    base = ["--machine", str(machine_json), "--cmd", "/bin/true"]
    a = se.parse_args(base + ["--maxinsts", "100"])
    assert a.mode == "full" and se.param_max_insts(a) == 100
    a = se.parse_args(base + ["--maxinsts", "100", "--warmup-insts", "10"])
    assert a.mode == "full" and se.param_max_insts(a) == 0      # scheduled after the warm-up
    a = se.parse_args(base + ["--fast-forward", "5000", "--warmup-insts", "10", "--maxinsts", "7"])
    assert a.mode == "ff" and a.fast_forward == 5000 and se.param_max_insts(a) == 0
    a = se.parse_args(base + ["--take-checkpoints", "30,10,20", "--checkpoint-dir", str(tmp_path)])
    assert a.mode == "checkpoint" and a.checkpoints == [10, 20, 30]
    assert a.cpu_type == "AtomicSimpleCPU" and not a.caches      # functional pass
    a = se.parse_args(base + ["--profile-regions", str(tmp_path / "p.json")])
    assert a.mode == "profile" and a.profile_cap == 2048 and a.cpu_type == "AtomicSimpleCPU"
    for bad in (["--take-checkpoints", "10"],                       # no --checkpoint-dir
                ["--take-checkpoints", "x", "--checkpoint-dir", "d"],
                ["--fast-forward", "10", "--restore-checkpoint", str(tmp_path)],
                ["--profile-regions", "p", "--maxinsts", "5"],
                ["--fast-forward", "10", "--cpu-type", "TimingSimpleCPU"],
                ["--warmup-insts", "-1"]):
        with pytest.raises(SystemExit):
            se.parse_args(base + bad)


def test_se_restore_seeds_hint_policies(tmp_path, monkeypatch, machine_json):
    """Check that a restore seeds hint/hybrid from the checkpoint's last setwin only."""
    se = import_real_se(monkeypatch)
    ck = tmp_path / "cpt.1000"
    ck.mkdir()
    (ck / se.FF_STATE).write_text(json.dumps({"last_setwin": 128, "last_region": 2}))
    base = ["--machine", str(machine_json), "--cmd", "/bin/true", "--restore-checkpoint", str(ck),
            "--window-initial", "3"]
    a = se.parse_args(base + ["--window-policy", "hint"])
    assert a.mode == "restore" and a.seed == 1 and a.window_initial == 1
    assert se.parse_args(base + ["--window-policy", "hybrid"]).window_initial == 1
    assert se.parse_args(base + ["--window-policy", "static"]).window_initial == 3
    assert se.parse_args(base + ["--window-policy", "hint", "--window-seed", "off"]).window_initial == 3
    (ck / se.FF_STATE).write_text(json.dumps({"last_setwin": 0}))   # release -> largest
    assert se.parse_args(base + ["--window-policy", "hint"]).window_initial == 3
    (ck / se.FF_STATE).write_text(json.dumps({"last_setwin": None}))
    a = se.parse_args(base + ["--window-policy", "hint"])
    assert a.seed is None and a.window_initial == 3
    assert se.clock_period_ticks("2GHz") == 500 and se.clock_period_ticks("500MHz") == 2000
    for bad in ("2 GHZ", "fastGHz", "0GHz", "2000"):
        with pytest.raises(SystemExit, match="cannot parse clock"):
            se.clock_period_ticks(bad)


def _tiny_elf(path, base, code: bytes):
    """Minimal little-endian ELF64 with one executable PROGBITS section."""
    shoff = 64 + len(code)
    hdr = bytearray(64)
    hdr[0:4] = b"\x7fELF"
    hdr[4], hdr[5], hdr[6] = 2, 1, 1
    struct.pack_into("<Q", hdr, 0x28, shoff)
    struct.pack_into("<HH", hdr, 0x3A, 64, 2)
    null = bytes(64)
    text = bytearray(64)
    struct.pack_into("<IIQQQQ", text, 0, 1, 1, 0x6, base, 64, len(code))
    path.write_bytes(bytes(hdr) + code + null + bytes(text))


def test_winhint_elf_scan_and_setwin_mapping(tmp_path):
    """Check scan_hints() on a tiny ELF, decode_hint() rejection and config_for_setwin()."""
    def word(imm):
        """Return the little-endian bytes of ``ori x0, x0, imm``."""
        return (0x00006013 | (imm << 20)).to_bytes(4, "little")
    setwin128 = (16 << 5) | 0x15
    region5 = (5 << 5) | 0x17
    prefetch = (3 << 5) | 0x1              # Zicbop tag: not a hint
    code = b"\x01\x00" + word(setwin128) + word(region5) + word(prefetch) + b"\x13\x00\x00\x00"
    _tiny_elf(tmp_path / "a.out", 0x10000, code)
    h = winhint_elf.scan_hints(tmp_path / "a.out")
    assert h == {"setwin": {0x10002: 128}, "region": {0x10006: 5}}  # 2-byte aligned (RVC)
    assert winhint_elf.decode_hint(0x00006013 | ((0x800 | 0x15) << 20)) is None  # IMM[11] = 1
    assert [winhint_elf.config_for_setwin(w, ROB) for w in (0, 8, 64, 65, 128, 256, 504)] == \
        [3, 0, 0, 1, 1, 3, 3]
    with pytest.raises(ValueError):
        (tmp_path / "x").write_bytes(b"nope")
        winhint_elf.scan_hints(tmp_path / "x")


# ---------------------------------------------------------------------------
# Sampled pipeline end to end (fake gem5)
# ---------------------------------------------------------------------------

def _bins(root, variants=("plain", "oracle", "winhint")):
    """Create dummy binaries ``<root>/<variant>/k`` (distinct contents) and return ``root``."""
    for v in variants:
        (root / v).mkdir(parents=True, exist_ok=True)
        (root / v / "k").write_bytes(f"binary {v}".encode())
    return root


def test_sampled_campaign_and_oracle_use_the_same_windows(tmp_path, machine_json, monkeypatch,
                                                          cfg_file, fake_gem5):
    """Run the sampled campaign and the B1 oracle with the fake gem5 and check they share windows.

    Also checks the dry run before phase P, the stratified cycle estimate,
    hint-state seeding, checkpoint sharing and resumability.
    """
    monkeypatch.setattr(rx, "MACHINES_DIR", machine_json.parent)
    b = _bins(tmp_path / "bin")
    common = ["--bin-root", str(b), "--gem5", str(fake_gem5), "--no-lock",
              "--run-lengths", str(cfg_file), "--runlen-root", str(tmp_path / "runlen"),
              "--ckpt-root", str(tmp_path / "ckpt")]
    args = ["--kernels", "k", "--policies", "static", "winhint", "--results-root",
            str(tmp_path / "res"), "--energy-model", "proxy"] + common
    # dry run before the profile exists: phase P is listed, runs are pending, nothing is written
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert rx.main(args + ["--dry-run"]) == 0
    out = buf.getvalue()
    assert "phase P: 1 profile passes" in out and "pending" in out and "--profile-regions" in out
    assert not (tmp_path / "res").exists() and not (tmp_path / "runlen").exists()

    assert rx.main(args) == 0
    prof = json.loads((tmp_path / "runlen" / "oracle" / "k.large.json").read_text())
    assert prof["total_insts"] == 20000
    pl = rl.plan(rl.spec_for(rl.load_config(cfg_file), "k", "large"), prof)
    s = pd.read_csv(tmp_path / "res" / "summary.csv").set_index("variant")
    assert (s.status == "ok").all() and (s.runlen == "regions").all()
    assert (s.samples == len(pl.samples)).all() and (s.runlen_version == 7).all()
    # the stratified estimate reproduces the whole-program cycles of the fake program
    for c in range(4):
        assert s.loc[f"static_c{c}", "cycles"] == pytest.approx(true_cycles(c), rel=2e-3)
    assert s.loc["winhint", "ipc"] > s.loc["static_c3", "ipc"]
    # hint state restored across checkpoints: every winhint sample was seeded
    d = tmp_path / "res" / "riscv_ooo" / "k" / "winhint"
    cmd0 = json.loads((d / "s01" / "run.json").read_text())["command"]
    assert "--restore-checkpoint" in cmd0 and "--warmup-insts 500" in cmd0
    # one checkpoint pass per binary, each with exactly the plan's starts
    cps = sorted((tmp_path / "ckpt").glob("*/checkpoints.json"))
    assert len(cps) == 2      # plain + winhint
    assert all(json.loads(p.read_text())["requested"] == pl.starts for p in cps)
    rj = json.loads((d / "runlength.json").read_text())
    assert rj["plan"]["starts"] == pl.starts and rj["hint_state_exact"]

    # B1 oracle on large: same profile, same starts, same budgets -> the oracle map
    assert oracle_sweep.main(["--kernels", "k", "--machine", str(machine_json), "--input", "large",
                              "--out-root", str(tmp_path / "oracle")] + common) == 0
    flat = json.loads((tmp_path / "oracle" / "riscv_ooo" / "large" / "k.json").read_text())
    assert {int(r): c for r, c in flat.items()} == BEST
    cps = sorted((tmp_path / "ckpt").glob("*/checkpoints.json"))
    assert len(cps) == 3 and all(json.loads(p.read_text())["requested"] == pl.starts for p in cps)
    r = json.loads((tmp_path / "oracle" / "runs" / "riscv_ooo" / "k" / "large" / "c0"
                    / "runlength.json").read_text())
    assert r["plan"]["starts"] == pl.starts

    # resumable: nothing reruns
    assert rx.main(args) == 0
    # dry run now shows the plan of every run
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rx.main(args + ["--dry-run", "--force"])
    assert f"regions:{len(pl.samples)}x" in buf.getvalue()


def test_every_sample_command_parses_with_real_se_py(tmp_path, machine_json, monkeypatch,
                                                     cfg_file):
    """Check that every generated sample command is accepted by the real se.py parser."""
    se = import_real_se(monkeypatch)
    monkeypatch.setattr(rx, "MACHINES_DIR", machine_json.parent)
    b = _bins(tmp_path / "bin", ("plain", "oracle", "winhint", "jones"))
    pdir = tmp_path / "runlen" / "oracle"
    pdir.mkdir(parents=True)
    (pdir / "k.large.json").write_text(json.dumps(profile()))
    a = rx.parse_args(["--kernels", "k", "--bin-root", str(b), "--results-root", str(tmp_path / "r"),
                       "--run-lengths", str(cfg_file), "--runlen-root", str(tmp_path / "runlen"),
                       "--ckpt-root", str(tmp_path / "ckpt"), "--window-args", "mlp:mlp_thr=2.0"])
    runs = [r for r in rx.build_runs(a) if r.samples]
    assert {r.variant.name for r in runs} >= {"static_c0", "mlp", "winhint", "winhint_hw", "jones"}
    for r in runs:
        assert r.ckpt is not None and r.ckpt.starts == r.plan.starts
        for sr in r.samples:
            i = sr.cmd.index(str(rx.SE_PY))
            args = se.parse_args(sr.cmd[i + 1:])
            assert args.maxinsts == sr.sample.measure and args.warmup_insts == sr.sample.warmup
            assert args.mode == ("restore" if sr.sample.start else "full")
            assert args.window_policy == r.variant.policy
            if r.variant.name == "jones":
                assert args.window_args == "structs=iq"
            if r.variant.policy == "mlp":
                assert args.window_args == "mlp_thr=2.0"
    # --max-insts conflicts with a sampled run length; --run-length full ignores the config
    a2 = rx.parse_args(["--kernels", "k", "--bin-root", str(b), "--run-lengths", str(cfg_file),
                        "--runlen-root", str(tmp_path / "runlen"), "--max-insts", "10",
                        "--policies", "static_c0"])
    assert any("--max-insts" in m for m in rx.build_runs(a2)[0].missing)
    a3 = rx.parse_args(["--kernels", "k", "--bin-root", str(b), "--run-length", "full",
                        "--policies", "static_c0"])
    r3 = rx.build_runs(a3)[0]
    assert not r3.samples and r3.mode == "full" and "--restore-checkpoint" not in r3.cmd


def test_tuned_parameters_feed_the_matrix(tmp_path, machine_json, monkeypatch):
    """Check that tuned.json window args, binary suffixes and compiler knobs reach the matrix."""
    monkeypatch.setattr(rx, "MACHINES_DIR", machine_json.parent)
    b = _bins(tmp_path / "bin", ("plain", "winhint", "clairvoyance-cv_spec-u4-i2"))
    (b / "winhint" / "k.winhint.json").write_text(json.dumps(
        {"target": "riscv_ooo", "switch_cost_cycles": 42, "hysteresis": 0.25}))
    tuned = tmp_path / "tuned.json"
    tuned.write_text(json.dumps({
        "version": 1, "window_args": {"*": {"mlp": "mlp_thr=3.0,gain=0.2", "hybrid": "floor=1"},
                                      "riscv_ooo": {"mlp": "gain=0.05"}},
        "binary_suffix": {"clairvoyance": "-cv_spec-u4-i2"},
        "compiler": {"winhint": {"make": {"SWITCH_COST": "500"},
                                 "sidecar": {"switch_cost_cycles": 42, "hysteresis": 0.25}}}}))
    base = ["--kernels", "k", "--bin-root", str(b), "--run-length", "full", "--tuned", str(tuned)]
    runs = {r.variant.name: r for r in rx.build_runs(rx.parse_args(base + [
        "--window-args", "mlp:miss_min=4"]))}
    wa = lambda r: r.cmd[r.cmd.index("--window-args") + 1]  # noqa: E731
    # a tuned point is complete: the machine's entry replaces "*"; the CLI adds/overrides keys
    assert wa(runs["mlp"]) == "gain=0.05,miss_min=4"
    assert wa(runs["winhint_hw"]) == "floor=1"
    assert runs["clairvoyance"].binary == b / "clairvoyance-cv_spec-u4-i2" / "k"
    assert not runs["winhint"].missing
    # a winhint binary built with other knobs than the tuned ones is not used
    (b / "winhint" / "k.winhint.json").write_text(json.dumps(
        {"target": "riscv_ooo", "switch_cost_cycles": 500, "hysteresis": 0.1}))
    runs = {r.variant.name: r for r in rx.build_runs(rx.parse_args(base))}
    assert any("tuned WinHint knobs" in m for m in runs["winhint"].missing)
    assert any("tuned WinHint knobs" in m for m in runs["winhint_nop"].missing)
    # --tuned none disables
    runs = {r.variant.name: r for r in rx.build_runs(rx.parse_args(
        ["--kernels", "k", "--bin-root", str(b), "--run-length", "full", "--tuned", "none"]))}
    assert "--window-args" not in runs["mlp"].cmd and runs["clairvoyance"].binary is None
