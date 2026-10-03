# Real hardware

The second evaluation platform is real silicon: an Intel hybrid laptop CPU with fast P-cores
and efficient E-cores. There the compiler's window hint cannot resize a window, so it becomes
a placement decision: a region that wants a large window runs on a P-core, one that does not
runs on an E-core. This page is the deep reference for that side (PROPOSAL Phase E2): the
platform, the per-host status, the libwinhint runtime, the R0–R5 baselines and their
deviations, the privileges, and the campaign method. What the hardware results are meant to
show, next to the gem5 study, is in [Evaluation methodology](../../concepts/methodology.md). The
commands in run order are in [Running experiments, Part 2](../usage.md#part-2-real-hardware),
the runbook; this page explains them and records what was verified on this host.

**Terms.**

- *P-core / E-core*: the performance and efficiency core types of the hybrid CPU.
- *RAPL*: Intel's energy counters (Running Average Power Limit), read per domain.
- *EDP*: energy-delay product, E × t.
- *Call mode*: the compiler emits calls to `__winhint_setwin(W)` / `__winhint_region(id)`
  in libwinhint instead of NOP hints ([interfaces.md §2](../../interfaces.md#2-the-hint-isa-contract)).
- *NOP hint*: the x86 multi-byte NOP encoding of a hint; it does nothing on the CPU.
- *SMT*: simultaneous multithreading (two hardware threads per P-core).

Other terms are in the [Glossary](../../reference/glossary.md).

## Platform

Target machine: Intel Core 5 120U (Raptor Lake-U refresh, family 6 model 186).
CPUs 0–3 are P-cores (2 cores × SMT, `cpu_core` PMU), CPUs 4–11 are E-cores (`cpu_atom` PMU).
RAPL domains: `package-0`, `core`, `uncore`, `psys`. Kernel 7.0 (Ubuntu), `intel_pstate` (HWP),
`perf_event_paranoid=1`, RAPL `energy_uj` mode 0400 root.

Everything runs **on the host** from the conda env `winhint` (`tooling/create_conda_env.sh`, see
`build/ENV.md` and [Toolchain and versions](../../reference/toolchain.md)). Nothing here needs root to *build*; only some *measurements* do, and every
privileged step is opt-in and run by you with `sudo` (§4). The scripts never call `sudo`.

```
hw/
├── Makefile                      builds into $WINHINT_BUILD/{libwinhint,hw}   (default build/)
├── libwinhint/
│   ├── winhint.h                 __winhint_setwin(W), __winhint_region(id)   (interfaces.md §2)
│   ├── winhint.c                 runtime: setwin → P/E placement; R5 "sondag" mode; per-region log
│   ├── whutil.{h,c}              topology, hybrid perf counters, RAPL (sysfs or perf power PMU)
│   ├── bench_migration.c         P→E / E→P migration cost → JSON (switch cost for the compiler DP)
│   └── test_affinity.c           smoke test: affinity flips on setwin
├── tools/wh_measure.c            runs one command; wall time, RAPL pkg/core, cycles/instr per core type
├── tools/phase_workload.c        synthetic phase workload: compute (ILP 1) / memory / ilp classes,
│                                 modes alt, alt-ilp, compute, memory, ilp
│                                 (_hinted: calls libwinhint; _nop: x86 NOP hints)
├── tools/uarch_probe.c           ROB / load-buffer / store-buffer / MLP probes (Wong-style), pinned
├── baselines/
│   ├── common.sh                 paths, opt-in/root/env guards, heavy-build lock
│   ├── r0_taskset/r0_taskset.sh  R0: P-only / E-only
│   ├── r1_stock/                 R1: record_system.sh (kernel, config, SMT, ITMT/HFI, cpufreq),
│   │                                 host_setup.sh (governor/EPP/turbo/SMT save/apply/restore, opt-in),
│   │                                 rapl_access.sh (opt-in RAPL read grant, §4)
│   ├── r2_sched_ext/             R2: install_scx.sh (build), run_scx.sh (scx_bpfland / scx_cosmos / scx_lavd)
│   ├── r3_lpmd/                  R3: check_platform.sh, install_lpmd.sh, run_lpmd.sh (intel-lpmd),
│   │                                 upower_stub/ (upower-glib stand-in, no conda package)
│   ├── r4_pie/                   R4: pie_daemon.c (perf_event_open or PMCTrack backend), install_pmctrack.sh
│   └── r5_sondag/region_types.py R5: static region typing for WINHINT_MODE=sondag
├── tests/                        pytest: driver, fidelity predicates, --llamacpp commands (no measurements)
├── fidelity.py                   --uarch (µarch probes) and --fidelity (R4/R5 trend checks) modes
└── run_hw_experiments.py         campaign driver → results/hw/
```

## 0. Status on this machine (2026-10-01)

| Part | Built | Runs unprivileged | Notes |
|---|---|---|---|
| libwinhint (off/log/migrate/sondag), test_affinity, bench_migration | yes | yes | smoke-tested; migration cost in `results/hw/system/migration_cost_smt-on.json` (re-measure idle, see §2) |
| wh_measure (time + per-core-type counters) | yes | yes | energy needs RAPL access (§4) |
| R0, R1, R4, R5, WinHint, NOP overhead | — | yes | 3-rep **noisy** smoke run on `regression_linear_infer` (machine busy with gem5/compiler builds) in `results/hw/smoke_2026-10-01/`; outputs identical to R1 in every config |
| R4 PIE, perf backend | yes | yes (own child, paranoid=1) | smoke-tested |
| R4 PIE, PMCTrack backend | CLI yes; **module no** | needs module (root) | built with the `winhint-kmod` GCC 15.2 on 2026-10-01: PMCTrack v4.0 (`00b032d`) **does not compile against kernel 7.0** (API removals, see §3 R4); R4 uses the perf backend |
| R2 sched_ext (`scx_bpfland`, `scx_cosmos`, `scx_lavd`) | **yes** (scx 1.1.3 @ `build/hw-baselines/scx/WINHINT_SCX_COMMIT`) | no (root) | built with the env's cargo/clang/libelf; libbpf is vendored by `libbpf-sys`, so neither a conda libbpf nor bpftool is needed (`tooling/versions.lock`) |
| R3 intel-lpmd | **yes** (`build/hw-baselines/lpmd/prefix`, commit in `WINHINT_LPMD_COMMIT`) | no (root) | platform supported (`check_platform.sh`). glib (conda-forge `libglib` + `glib` + `glib-tools` 2.90.0, pinned in `versions.lock`), libxml-2.0, libnl, libsystemd all come from the env. One conda gap: `upower-glib` has **no conda package** → `upower_stub/` (lpmd sees "upowerd unreachable, AC power"; run on AC power), recorded in `WINHINT_LPMD_DEVIATIONS` |
| µarch probes (`uarch_probe`) | yes | yes | `make -C hw uarch-smoke` (functional); full probe: `--uarch` (§5.1), values in `tools/uarch_params.md` |
| R4/R5 fidelity checks | yes | yes | `--fidelity` (§5.2); pipeline smoke-tested only (short busy-machine runs, not results) |

## 1. Build (host, conda env, one job)

```sh
eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate winhint

# build (without activating: ~/.local/bin/micromamba run -n winhint make -C hw -j1)
make -C hw -j1

# driver unit tests
python -m pytest -q hw/tests
```

Outputs: `build/libwinhint/{libwinhint.a,libwinhint.so,winhint.h,bench_migration,test_affinity}`
and `build/hw/{wh_measure,pie_daemon,phase_workload,phase_workload_hinted,phase_workload_nop,uarch_probe}`
(`make -C hw static` adds statically linked copies in `build/hw/static/`). The compiler is the
env's `$CC` (conda GCC 13); outputs never go into the source tree.

In the runbook this is [Step 1: Build](../usage.md#step-1-build), which uses the
`tooling/winhint.sh hw:build` and `hw-baselines:build` wrappers.

Out-of-tree baselines (all opt-in, user-level, under `flock build/.heavy.lock`, `-j1`):

```sh
# R2 → build/hw-baselines/scx (SCX_REF=<tag> to pin)
bash hw/baselines/r2_sched_ext/install_scx.sh

# R4 → build/hw-baselines/pmctrack/bin/pmctrack
bash hw/baselines/r4_pie/install_pmctrack.sh build

# R4 → mchw_intel_core.ko (built, never loaded)
bash hw/baselines/r4_pie/install_pmctrack.sh module

# R3 → build/hw-baselines/lpmd/prefix (conda-only route, see §0)
bash hw/baselines/r3_lpmd/install_lpmd.sh
```

### Linking call-mode benchmarks (for `benchmarks/Makefile`)

```make
# x86, -mllvm -winhint-emit=call
WINHINT_LIBDIR ?= $(WINHINT_BUILD)/libwinhint
CALL_CFLAGS  = -I$(WINHINT_ROOT)/hw/libwinhint
CALL_LDFLAGS = -L$(WINHINT_LIBDIR) -l:libwinhint.a     # static: no LD_LIBRARY_PATH needed
```

libwinhint needs only libc (no `-lpthread`, no `-lm`). Variant names expected by the driver
(overridable with `--variant-*`): `plain`, `winhint` (x86 NOP hints, asm mode),
`winhint_call` (WinHint hints in call mode), `oracle_call` (region markers only, call mode;
used by R5). Binaries: `build/benchmarks/x86/<variant>/<kernel>` (`--bench-root`).
For R5 typing, `region_types.py` reads the `<kernel>.regions.json` written next to the binaries.

## 2. libwinhint

`__winhint_setwin(W)`: `W ≥ WINHINT_THRESHOLD` → P-core set; `0 < W < threshold` → E-core set;
`W = 0` (release) → `WINHINT_RELEASE` (default: the original affinity, the OS decides).
Only the calling thread is moved (the kernels are single-threaded). Redundant requests are
dropped without a syscall. Every real migration is timed (`sched_setaffinity` latency) and
checked with `sched_getcpu()`.

| Variable | Default | Meaning |
|---|---|---|
| `WINHINT_MODE` | `migrate` | `off` (hooks return at once; call-overhead runs), `log` (decide + count, never migrate), `migrate`, `sondag` (R5) |
| `WINHINT_THRESHOLD` | 192 | window size at/above which the P-cores are requested. 192 = index ≥ 2 of the `riscv_ooo` table; with an x86 target JSON use E-core ROB (256) + 1 or so |
| `WINHINT_PCPUS` / `WINHINT_ECPUS` | sysfs `cpu_core/cpus`, `cpu_atom/cpus` ∩ online | core sets (cpulists) |
| `WINHINT_PIN` | `set` | `single`: pin to one CPU per side (`WINHINT_PCPU`, `WINHINT_ECPU`) |
| `WINHINT_RESPECT_AFFINITY` | 0 | 1: intersect the sets with the inherited affinity |
| `WINHINT_RELEASE` / `WINHINT_INITIAL` | `orig` | side for `W=0` / at program start (`P`, `E`, `orig`) |
| `WINHINT_HYST` | 1 | consecutive requests for the other side needed before migrating |
| `WINHINT_MIN_DWELL_US` | 0 | rate limit: minimum time between migrations (deferred request is applied at the next hook) |
| `WINHINT_LOG` | — | per-region CSV (`region,side,visits,segments,wall_ns,cycles_p,instructions_p,cycles_e,instructions_e,energy_pkg_uj,energy_core_uj`) + `<path>.summary.json` (migration count/cost per direction, suppressed/rate-limited counts, errors) |
| `WINHINT_PERF` | 1 | per-region user-mode cycles/instructions (one counter per core-type PMU) when logging |
| `WINHINT_RAPL` | 0 | per-region RAPL energy (coarse: RAPL updates every ~1 ms) |
| `WINHINT_TRACE` | — | CSV of every migration (`t_ns,from,to,cost_ns,cpu_after,region,ok`) |
| `WINHINT_SONDAG_K` | 2 | R5: visits sampled per core type before a region type is assigned |
| `WINHINT_SONDAG_THRESHOLD` | 1.4 | R5: assign P if (E time/instr) / (P time/instr) ≥ threshold, else E |
| `WINHINT_SONDAG_TYPES` | identity | R5: `region_id type_id` file from `baselines/r5_sondag/region_types.py` |
| `WINHINT_VERBOSE` | 0 | log the configuration and decisions to stderr |

In the per-region CSV, `visits` are counted on the side active when the region is entered, while
`segments`/`wall_ns`/counters are split at every hook call (so a `region(id)` followed by a
`setwin` that migrates shows the visit on the old side and the time on the new one). `region = -1`
is code before the first marker.

If no hybrid topology is found, `migrate`/`sondag` fall back to `log` with a message on stderr.
If perf counters are not permitted, a clear message is printed and logging continues without them.

### How hints reach libwinhint (x86)

The x86 NOP hints (`nopl 0x5748kppp(%rax)`, [interfaces.md §2](../../interfaces.md#x86-64-encoding-unique-multi-byte-nop); `winhint` variant, asm mode)
are architectural no-ops: the CPU never reports them, so libwinhint does **not** scan or trap
them. They exist on x86 only to measure the hint *overhead* (the `NOP-asm-*` rows, which the
driver refuses to run if the binary contains no `0F 1F 80 .. .. 48 57` hint). Migration uses the
compiler's runtime-call mode (`-mllvm -winhint-emit=call`: `winhint_call` = setwin calls,
`oracle_call` = region calls for R5), which calls `__winhint_setwin(W)` / `__winhint_region(id)`
at the same layer/operator (loop-nest) boundaries with the same W / id values as the NOPs.

### Migration cost (switch cost for the compiler DP)

```sh
# set targets (WINHINT_PIN=set, default)
build/libwinhint/bench_migration -n 200 -s -o results/hw/system/migration_cost_smt-on.json

# one CPU per side (WINHINT_PIN=single)
build/libwinhint/bench_migration -n 200 -o results/hw/system/migration_cost_single_smt-on.json
```

`switch_cost_us` (and `switch_cost_cycles_at_pmax`) = mean of the median P→E and E→P costs,
where each cost = `sched_setaffinity` latency + private-cache refill penalty for a 256 KB working
set (`-w`). The driver measures both files once per SMT state into `<out>/system/`; the canonical
`migration_cost_smt-<s>.json` uses `-s`, i.e. the way libwinhint migrates by default (affinity to
the whole P or E set). It feeds the compiler's DP placement as follows: `benchmarks/Makefile`
(`EMIT=call`, i.e. `winhint_call`/`oracle_call`) reads `switch_cost_us` from
`results/hw/system/migration_cost_smt-on.json` (`MIGRATION_US=` overrides) and passes
`-mllvm -winhint-switch-model=pe -mllvm -winhint-migration-us=<switch_cost_us>`; the pass converts
it to cycles with the target JSON's clock (`switch_cost_cycles` in `<kernel>.winhint.json`).
`-mllvm -winhint-switch-cost=<cycles>` (`SWITCH_COST=`) overrides it outright.

Smoke values (machine loaded by gem5/compiler builds, SMT on, 100 iterations, **noisy**):
single CPU 11.7–13.3 µs (P→E ≈ 10–13 µs, E→P ≈ 14 µs); set targets 26–60 µs (the scheduler has
to pick a CPU of the target set, several of which were busy). `results/hw/system/` still holds the
single-CPU value (11.69 µs) that the current `winhint_call` builds used; re-measure on the idle
machine with `-s` (or let the campaign do it) and rebuild the call-mode benchmarks.

## 3. Baselines

| ID | How | Privileges | Notes |
|---|---|---|---|
| R0-P / R0-E | `wh_measure -c <P or E cpulist>` (= `taskset -c`) | none | `r0_taskset.sh` for manual use |
| R1 | stock EEVDF + ITMT/HFI | none | `record_system.sh` records kernel/cmdline/config, ITMT, EAS, sched_ext state, cpufreq, SMT. EAS is disabled on SMT hybrids by `intel_pstate`; run the campaign with SMT on **and** off (SMT switch needs root). |
| R2 | sched_ext `scx_bpfland --primary-domain powersave`, `scx_cosmos --primary-domain powersave`, `scx_lavd --powersave` (presets: `run_scx.sh presets`) | root | flags checked against the built version's `--help` (`build/hw-baselines/scx/scx_*.help.txt`) |
| R3 | intel-lpmd (`intel_lpmd --no-daemon --dbus-enable -c <cfg>`, `intel_lpmd_control AUTO`) | root + D-Bus policy | `check_platform.sh`: supported (cgroup v2 cpuset, HFI, powerclamp); this CPU (F6 M186) has **no model-specific config**, so `run_lpmd.sh` copies the generic config with `<lp_mode_cpus>` = the E-cores (4–11) and passes it with `-c`. Built with conda tools only; deviation: upower-glib stub (no battery/AC tracking, campaign on AC power), see §0. |
| R4 | `pie_daemon -b perf` (or `-b pmctrack`) | perf: none; pmctrack: module (root) | PIE (Van Craeynest et al., ISCA'12) reimplemented; deviations below |
| R5 | `WINHINT_MODE=sondag` on the `oracle_call` binary | none | Sondag & Rajan (CGO'11); deviations below |

### R4 — PIE-style daemon (deviations from the paper)

PIE schedules several threads on a simulated big/small CMP to maximise throughput. Here there is
one job and the objective is energy, so the daemon runs the job on the E-cores whenever the
predicted E/P time ratio is ≤ 1 + slack (`-s`, default 0.15), else on the P-cores, with
hysteresis (`-H 2`) every interval (`-i 10` ms). The PIE model is kept: CPI is split into a base
and a memory part; the base part scales with the width/ILP ratio (`PIE_BASE_RATIO`, `PIE_WIDTH_*`),
the memory part with the MLP ratio. MLP on the current core is measured on P-cores
(`L1D_PEND_MISS.PENDING / PENDING_CYCLES`, raw events 0x0148 and cmask=1; PIE used MSHR occupancy)
and otherwise estimated as LLC-misses/instr × ROB size (PIE's small→big estimate), with ROB
sizes 512 (Raptor Cove) and 256 (Gracemont) and a cap of 16. Cycles are turned into time with the
measured frequency (cycles / time-on-PMU) scaled by the `cpuinfo_max_freq` ratio. Model
parameters should be tuned on the `small` inputs (same effort as WinHint) — see the per-interval
log (`-o pie.csv`).

Smoke run (3 s synthetic workload, 200 ms phases): 325 intervals, 72 migrations — the default
`-s 0.15 -H 2` flips inside phases on this workload; tune `-H`/`-s` on the `small` inputs before
the campaign.

PMCTrack backend: PMCTrack (github.com/jcsaezal/pmctrack, works on vanilla kernels ≥ 5.9 via
its ftrace stub) is reused unchanged. `install_pmctrack.sh build` builds `libpmctrack` and the
CLI (`build/hw-baselines/pmctrack/bin/pmctrack`, static) with the env's GCC.
`install_pmctrack.sh module` builds `mchw_intel_core.ko` against
`/lib/modules/$(uname -r)/build`; **on this host it fails**: the Ubuntu 7.0 kernel was built with
GCC 15.2 and its Kbuild flags include `-fmin-function-alignment=16`, which the env's GCC 13.4
rejects. Fix without leaving conda: a small separate env with conda-forge `gcc_linux-64=15.2`
(available) and `KCC=<that env>/bin/x86_64-conda-linux-gnu-gcc install_pmctrack.sh module`
(the `winhint-kmod` env). Loading is always a manual root step (§4).

Status 2026-10-01 (`build/pmctrack_module.log`): with GCC 15.2 the module **still fails** on
kernel 7.0. `install_pmctrack.sh` now passes upstream's include paths and defines via `KCFLAGS`,
because Kbuild no longer honours upstream's `EXTRA_CFLAGS`. After that, PMCTrack v4.0 (`00b032d`)
uses kernel APIs that 7.0 removed:
- `del_timer_sync` (now `timer_delete_sync`);
- `hrtimer_init` and `hrtimer_clock_base::get_time` (now `hrtimer_setup`);
- direct `vma->vm_flags` writes (now `vm_flags_set`);
- the `cpuinfo_x86` fields `apicid`, `phys_proc_id`, `cpu_core_id`, `logical_die_id` (now in
  `cpuinfo_x86.topo`);
- `rdpmcl` (now `rdpmc`);
- `topology_max_die_per_package`;
- a missing `<linux/vmalloc.h>`.

That is a port of about 25 call sites across 8 files, not a build fix. It is not done here: the
artifact is reused unchanged. R4 uses the perf backend, which implements the same policy. The daemon spawns
`pmctrack -T <s> -c instr,cycles,llc_misses -p <pid>` (template: `-C`, events:
`PIE_PMCTRACK_EVENTS`; the driver passes the built CLI and `PMCTRACK_ROOT`) and parses its
per-sample lines (`nsample pid event pmc0 pmc1 pmc2`). This path is **untested** (module not
loaded); the perf backend implements the same policy and is what the smoke test exercised.

### R5 — Sondag & Rajan (deviations)

The paper types basic-block "sections" by static instruction-mix similarity, inserts phase marks,
samples each type on every core type, then assigns. Here the marks are WinHint's `region(id)`
markers (top-level loop nests, call mode), the static typing is `region_types.py` (k-means on any
numeric per-region features the pass writes into `<kernel>.regions.json`; otherwise one type per
function, or per region), and the runtime samples the first `K` visits of each type on P then on E
and assigns P iff the E-core slowdown in time per instruction is ≥ `WINHINT_SONDAG_THRESHOLD`.
The paper optimised throughput across threads; we use the slowdown threshold as the single-job
energy/performance knob.

## 4. Privileges (host) — everything below is opt-in and run by you

What needs what on this host (`perf_event_paranoid=1`, RAPL `energy_uj` 0400 root):

| Feature | Requirement | Verified |
|---|---|---|
| `sched_setaffinity` of own threads/children (libwinhint, R0, R4 migration, R5) | nothing | yes |
| Own-process user-mode counters (libwinhint per-region, wh_measure, R4 perf backend) | `perf_event_paranoid ≤ 2` | yes (paranoid=1) |
| RAPL energy via powercap sysfs (wh_measure, `WINHINT_RAPL=1`) | root, **or** the read grant below | no (not granted) |
| RAPL via perf `power` PMU (`WINHINT_RAPL_BACKEND=perf`) | system-wide event: root / `CAP_PERFMON` (paranoid ≥ 1) | no |
| governor / EPP / turbo / SMT (`--governor`, `--epp`, `--no-turbo`, `--smt on,off`) | root | — |
| R2 sched_ext attach | root (BPF + sched_ext) | — |
| R3 intel-lpmd | root (cgroup cpusets) + D-Bus policy file in `/etc/dbus-1/system.d` | — |
| R4 PMCTrack backend | `mchw_intel_core` module loaded (root) | — |

The scripts refuse system-wide changes unless `WINHINT_ALLOW_SYSTEM_CHANGES=1` (or the driver's
`--allow-system-changes`/`--enable-scx`/`--enable-lpmd`) is given, and refuse to run them as a
normal user. The runbook's one-time host setup is
[Step 2](../usage.md#step-2-one-time-host-setup). Exact commands:

**RAPL energy, option A — run the measuring driver as root** (results are chowned back to the
repository owner at the end):

```sh
sudo -E env "PATH=$PATH" python hw/run_hw_experiments.py --reps 10
```

(`-E` + `PATH` keep the activated conda env; the hw binaries carry no env-specific runtime deps.)

**RAPL energy, option B — temporary read grant to your group** (until reboot; side-channel
caveat: PLATYPUS / CVE-2020-8694, revoke after the campaign):

```sh
# chgrp <your group>; chmod 0440 energy_uj
sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r1_stock/rapl_access.sh grant

# check (no root)
hw/baselines/r1_stock/rapl_access.sh status

# back to 0400 root
sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r1_stock/rapl_access.sh revoke
```

A persistent variant is a udev rule, e.g. `/etc/udev/rules.d/99-winhint-rapl.rules`:
`SUBSYSTEM=="powercap", KERNEL=="intel-rapl:*", RUN+="/bin/chmod 0440 /sys%p/energy_uj", RUN+="/bin/chgrp <group> /sys%p/energy_uj"`
— not recommended; prefer the temporary grant.

**Fixed frequency policy / SMT** (the driver saves and restores the state; or by hand):

```sh
sudo -E env "PATH=$PATH" python hw/run_hw_experiments.py --allow-system-changes \
     --governor performance --epp performance --no-turbo 1 --smt on,off

# manual equivalents
sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r1_stock/host_setup.sh save    /tmp/wh_state.txt
sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r1_stock/host_setup.sh apply   --governor performance --no-turbo 1 --smt off
sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r1_stock/host_setup.sh restore /tmp/wh_state.txt
```

**R2 sched_ext** (one scheduler system-wide; the stock scheduler returns when it exits or on
any error — sched_ext's watchdog also ejects a misbehaving scheduler):

```sh
sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r2_sched_ext/run_scx.sh start bpfland_powersave
hw/baselines/r2_sched_ext/run_scx.sh status
sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r2_sched_ext/run_scx.sh stop

# or let the driver attach it per group (as root; see §5.4)
sudo -E env "PATH=$PATH" python hw/run_hw_experiments.py \
    --configs R1,R2-bpfland_powersave --enable-scx --allow-system-changes \
    --governor performance --epp performance --no-turbo 1
```

**R3 intel-lpmd** (after a successful `install_lpmd.sh`; one-time D-Bus policy install):

```sh
# one-time D-Bus policy
sudo install -m 644 build/hw-baselines/lpmd/prefix/etc/dbus-1/system.d/org.freedesktop.intel_lpmd.conf /etc/dbus-1/system.d/
sudo systemctl reload dbus

# start and stop the daemon by hand
sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r3_lpmd/run_lpmd.sh start AUTO
sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r3_lpmd/run_lpmd.sh stop

# undo the D-Bus policy
sudo rm /etc/dbus-1/system.d/org.freedesktop.intel_lpmd.conf && sudo systemctl reload dbus
```

`run_lpmd.sh start` writes the config it uses (generic config, `<lp_mode_cpus>` = E-cores; override
with `LPMD_LP_CPUS=` or `LPMD_CONFIG=<file>`) to `/tmp/winhint-lpmd/intel_lpmd_config.xml`.

**R4 PMCTrack module** (after `install_pmctrack.sh module` succeeded; Secure Boot is off and
`module.sig_enforce=N` here, so the unsigned module loads and taints the kernel):

```sh
# insmod; /proc/pmc appears
sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r4_pie/install_pmctrack.sh load

# rmmod
sudo WINHINT_ALLOW_SYSTEM_CHANGES=1 hw/baselines/r4_pie/install_pmctrack.sh unload
```

`perf stat` for the NOP-overhead rows comes from the env (`linux-perf`); without it the driver
uses `wh_measure`'s own counters.

## 5. Campaign

### 5.1 µarch probes (`--uarch`, PROPOSAL §8)

```sh
# functional smoke (tiny ranges, no knees)
make -C hw uarch-smoke

# show the two probe commands: one P CPU, one E CPU
python hw/run_hw_experiments.py --uarch --dry-run

# idle machine: -> results/hw/uarch/
python hw/run_hw_experiments.py --uarch
```

This writes `uarch_{P,E}.json/.csv` with the time-vs-N curves and knees, and `uarch_check.json`
with each probe compared to the documented values (`confirmed` / `differs` / `no_knee` /
`undocumented`). See [Microarchitecture parameters](uarch-params.md). Keep the SMT sibling of the probed P-core idle.

### 5.2 Baseline fidelity (`--fidelity`, PROPOSAL §7)

```sh
python hw/run_hw_experiments.py --fidelity --dry-run

# defaults shown; pass the R4 parameters the campaign will use
python hw/run_hw_experiments.py --fidelity \
    --fidelity-secs 4 --fidelity-reps 3 --pie-slack 0.15 --pie-hyst 2
```

All runs use `phase_workload` (32 MB pointer chase) and need no root:
- **Ground truth:** each class (ilp, compute, memory) runs pinned to one P and one E CPU, and
  the P/E speedup is computed from the work rate.
- **R4:** `pie_daemon -b perf` runs on ilp, memory and the alternating workload.
- **R5:** `WINHINT_MODE=sondag` runs on `phase_workload_hinted ... alt-ilp` (region 2 = ilp,
  1 = memory). The threshold is the geometric mean of the two ground-truth speedups, so the
  correct assignment is unambiguous.

Machine-checkable predicates go to `results/hw/fidelity/fidelity.{json,csv}`. **trend** marks
the paper's claim, which must pass before the baseline is used; **info** is reported only.

| ID | Kind | Claim |
|---|---|---|
| GT-mem-gains-less | trend | speedup_P/E(memory) < speedup_P/E(ilp): the premise of PIE and of Sondag & Rajan |
| R4-pred-order | trend | PIE's predicted T_E/T_P is lower for memory than for ilp |
| R4-place-order | trend | fraction of intervals on E: memory > ilp |
| R4-phase-tracking | trend | on the alternating workload, high-MPKI intervals are sent to E more often than low-MPKI ones |
| R5-order | trend | sampled E/P ratio of the ilp type > memory type |
| R5-assign-ilp-P / R5-assign-mem-E | trend (info if the ground truth is not ordered) | types go to the core type that runs them better |
| GT-mem-vs-compute, R4-place-memory-E, R4-place-ilp-P, R5-agrees-gt-* | info | absolute placement with the campaign slack; agreement of the samples with the ground truth |

Run it on the idle machine before the campaign, with the R4 parameters (`--pie-slack`,
`--pie-hyst`) the campaign will use. A failing trend predicate means the baseline must be tuned
or fixed first. Report it, do not hide it. Short smoke runs on a busy machine are not
meaningful.

### 5.3 llama.cpp (`--llamacpp`, optional)

The integration itself (hooks, build, functional check) is documented in
[llama.cpp integration](llamacpp.md).

`--llamacpp` replaces the kernels with llama.cpp's `llama-simple`. The `plain` variant uses
`build-vanilla`; the call/regions variants use `build-winhint`. The default arguments are
`-m <dir>/models/stories15M-q4_0.gguf -n 256 'Once upon a time'` (`--llamacpp-args`,
`--llamacpp-dir`).
- `llama-simple` has no `-t` option. Only ggml thread 0 calls libwinhint, and libwinhint moves
  only the calling thread. So every config runs with `OMP_THREAD_LIMIT=1 OMP_NUM_THREADS=1`,
  which caps ggml's OpenMP team at one thread.
- R5-Sondag also sets `GGML_WINHINT_SETWIN=none` (region markers only).
- The NOP-* rows are dropped, because there is no NOP-hint build.

```sh
sudo -E env "PATH=$PATH" python hw/run_hw_experiments.py --llamacpp \
    --configs R0-P,R0-E,R1,WH,WH-off,R5-Sondag --reps 10 \
    --governor performance --epp performance --no-turbo 1 \
    --allow-system-changes --out results/hw/llamacpp
```

### 5.4 Main campaign

```sh
# 0. build tools, then the x86 benchmarks (plain, winhint, winhint_call, oracle_call)
make -C hw -j1
make -C benchmarks ARCH=x86 plain winhint winhint_call oracle_call
# 1. plan + privilege check (no measurement)
python hw/run_hw_experiments.py --dry-run
# 2. unprivileged smoke run on the synthetic workload (no energy)
python hw/run_hw_experiments.py --synthetic --reps 1 --allow-no-rapl --cooldown 0.2 --out results/hw/smoke
# 3. R0, R1, R4, R5, WinHint, NOP overhead, SMT on and off, fixed frequency policy, 10 reps (root)
sudo -E env "PATH=$PATH" python hw/run_hw_experiments.py \
    --reps 10 --smt on,off --governor performance --epp performance --no-turbo 1 \
    --allow-system-changes
# 4. R2/R3 groups (root; after install_scx.sh / install_lpmd.sh), same fixed frequency policy
sudo -E env "PATH=$PATH" python hw/run_hw_experiments.py \
    --configs R1,R2-bpfland_powersave,R2-cosmos_powersave,R2-lavd_powersave,R3-lpmd \
    --enable-scx --enable-lpmd --reps 10 --smt on,off \
    --governor performance --epp performance --no-turbo 1 \
    --allow-system-changes
# 5. re-summarize, then figures (hw_edp, hw_nop_overhead)
python hw/run_hw_experiments.py --summarize-only
python analysis/plot_results.py --hw-summary results/hw/summary.csv --out-dir results/figures
```

(`R2-<preset>` takes any preset of `run_scx.sh presets`, e.g. `R2-lavd_powersave`.)

The driver refuses to measure unless the governor is `performance` or the campaign fixes it
(`--governor`), so the R2/R3 command repeats the fixed frequency policy of step 3
(`--allow-unfixed-governor` only records the policy). The runbook version of these steps,
with R5 region typing (`--sondag-types-dir`), is
[Step 3: Campaign](../usage.md#step-3-campaign).

Method:
- Each (kernel, config, SMT, rep) is one run of `wh_measure`: wall time, RAPL package and core
  energy around the process, EDP = E × t, user-mode cycles/instructions on each core type.
- Fixed frequency policy: `--governor/--epp/--no-turbo` (saved and restored; recorded per row).
  Without these flags the current policy is only recorded. Turbo off gives the most stable
  numbers; report which one was used.
- Order is randomised per repetition (seeded); configurations needing a system-wide daemon
  (R2/R3) are grouped so the daemon is started once per group. `--cooldown` (2 s) idles
  between runs. ≥ 10 repetitions; `summary.csv` gives mean and 95 % CI (Student t) and ratios to R1.
- Correctness: stdout SHA-256 is compared to R1's (`output_ok`; not for the time-based synthetic workload).
- NOP overhead: `NOP-asm-{P,E}` vs `NOP-plain-{P,E}`, pinned to one CPU, `perf stat -e cycles:u,instructions:u` plus wall time.
- Resumable: `raw.csv` is append-only and runs already present are skipped (`--retry-failed` reruns failures).
- The migration microbenchmark is run once per SMT state into `<out>/system/`.
- `--synthetic` replaces the kernels with `build/hw/phase_workload*` (`--synthetic-args "s phase_ms MB"`);
  it runs for a fixed time, so it is only for checking the pipeline.

Safety: the scripts change system state only behind explicit flags and only as root, and restore
it at the end (also on errors). Run the campaign on AC power, with the machine idle — not while
gem5 or compiler builds are running.

## Next

- [µarch parameters](uarch-params.md): the documented P/E window sizes and the probe that
  checks them.
- [llama.cpp integration](llamacpp.md): the optional real-application
  workload.
- [Deviations](../../deviations.md#real-hardware): the real-hardware baselines' departures from their
  papers.
