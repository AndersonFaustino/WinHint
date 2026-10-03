# Baselines

WinHint is compared with the methods that resize or place an out-of-order window by other
means. This page lists every baseline, says which family it belongs to and where it is
implemented, and points to its page and to its deviations from the original paper. Why each
family is in the comparison, why most baselines are reimplemented, and how they are tuned with
the same effort as WinHint is in [Evaluation methodology](../../concepts/methodology.md).

## Families

| Family | IDs | What decides the window |
|--------|-----|-------------------------|
| fixed and oracle references | B0, B1 | nothing (static), or a per-region oracle map |
| reactive hardware | B2, B3, B4, B5, B9 | counters sampled at run time in gem5 |
| compiler and profile | B6, B7, B8 | a compile-time analysis or a profiling run |
| real-hardware placement | R0–R5 | the OS scheduler, a daemon or a runtime on an Intel hybrid CPU |

The full list, with citations and the artifact search, is in the
[design proposal](../../reference/proposal.md) §4. None of the hardware window-resizing papers
has a public artifact, so those baselines are reimplemented inside this infrastructure, on the
same resize mechanism as WinHint. Only B8 (Clairvoyance) reuses the authors' artifact. Each
reimplementation must reproduce its paper's qualitative trend before it is used
([Simulator fidelity](../fidelity.md)). Deviations from the original papers are documented next
to each implementation and summarized per baseline in [Deviations](../../deviations.md).

## Simulator and compiler baselines (gem5)

Each row runs as one or more `run_experiments.py` variants on the
[gem5 model](../gem5/index.md#evaluation-matrix-run_experimentspy).

| ID | Baseline | Page | How it runs |
|----|----------|------|-------------|
| B0 | static windows (4 configurations per machine) | [gem5 model](../gem5/index.md#machine-descriptions) | `window_policy=static`, one run per configuration (`static_c<i>`) |
| B1 | per-region oracle (upper bound) | [`oracle_sweep`](../../api/python/sim/baselines/oracle/oracle_sweep.md) | best configuration per region → `oracle_hinted` binary, `hint` policy |
| B2 | occupancy-driven IQ/ROB/LSQ resizing (Ponomarev et al., MICRO'01) | [Window policies](../gem5/policies.md) | `window_policy=occupancy` |
| B3 | MLP-aware window resizing (Kora et al., MICRO'13) | [Window policies](../gem5/policies.md) | `window_policy=mlp` |
| B4 | BBV phase tracking and prediction (Sherwood et al., ISCA'03) | [Window policies](../gem5/policies.md) | `window_policy=bbv` |
| B5 | learned counter LUT (Dubach et al. style) | [Window policies](../gem5/policies.md), [`label_phases`](../../api/python/sim/baselines/lut/label_phases.md) | label → train → export LUT → `window_policy=lut` |
| B6 | compiler-directed IQ resizing (Jones et al., HPCA'05) | [Jones IQ](jones-iq.md) | `jones` (IQ-only) / `jones_full` (ROB/LSQ too) binaries |
| B7 | profile-guided positional adaptation (Huang et al., ISCA'03) | [`pgo_flow`](../../api/python/compiler/baselines/pgo/pgo_flow.md) | `pgo` binary from the `small`-input oracle |
| B8 | Clairvoyance (Tran et al., CGO'17), public artifact | [Clairvoyance](clairvoyance.md) | `clairvoyance` / `winhint_clairvoyance` binaries |
| B9 | Long-Term Parking (Sembrant et al., MICRO'15) | [Long-Term Parking](ltp.md) | `plain` binary, `window_policy=ltp` |
| — | **WinHint**; WinHint + HW | [Compiler](../compiler.md) | `winhint` binary, `hint` or `hybrid` policy |

Implementation directories: B1 `sim/baselines/oracle/`, B2–B5 `sim/gem5/src/cpu/o3/window/`
(B5 training in `sim/baselines/lut/`), B6 `compiler/baselines/jones_iq/`, B7
`compiler/baselines/pgo/`, B8 `compiler/baselines/clairvoyance/`, B9
`sim/gem5/src/cpu/o3/window/ltp/`.

Baseline tunables (`window_args`) can be tuned per machine with
[`sim/baselines/tune/tune_baselines.py`](../../api/python/sim/baselines/tune/tune_baselines.md);
`sim/run_experiments.py` picks the tuned values up
([Baseline tuning](../usage.md#step-6-baseline-tuning)).

## Real-hardware baselines

| ID | Baseline | Where | Privileges |
|----|----------|-------|------------|
| R0-P / R0-E | pinning to the P or E cores (`taskset -c`) | `hw/baselines/r0_taskset/` | none |
| R1 | stock Linux scheduler (EEVDF + ITMT/HFI); `record_system.sh` records the system state | `hw/baselines/r1_stock/` | none |
| R2 | sched_ext schedulers (`scx_bpfland`, `scx_cosmos`, `scx_lavd` in power-saving presets) | `hw/baselines/r2_sched_ext/` | root |
| R3 | intel-lpmd | `hw/baselines/r3_lpmd/` | root + D-Bus policy |
| R4 | PIE (Van Craeynest et al., ISCA'12), reimplemented as `pie_daemon` (perf or PMCTrack backend) | [`hw/baselines/r4_pie/`](../../api/cpp/hw/pie__daemon_8c.md) | perf: none; PMCTrack: root |
| R5 | Sondag & Rajan (CGO'11) static region typing, `WINHINT_MODE=sondag` | [`hw/baselines/r5_sondag/`](../../api/python/hw/baselines/r5_sondag/region_types.md) | none |

Details, deviations and the measurement campaign: [Real hardware](../hardware/index.md) §3.

R2 (sched_ext) and R3 (intel-lpmd) depend on libraries without conda packages
(`libbpf`/`bpftool`, `upower-glib`); their status and workarounds are in
[Toolchain and versions](../../reference/toolchain.md) and [Real hardware](../hardware/index.md).

## Next

- [Jones IQ](jones-iq.md), [Clairvoyance](clairvoyance.md), [Long-Term Parking](ltp.md): the
  baselines with their own page.
- [Simulator fidelity](../fidelity.md): how each reimplementation is checked against its paper.
- [Deviations](../../deviations.md): every departure from the papers, per baseline.
