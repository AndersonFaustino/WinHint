# µarch parameters of the evaluation machine

The real-hardware side of WinHint needs the window sizes of the two core types of the
evaluation CPU: they set the libwinhint P/E threshold, the PIE model of baseline R4 and the x86
target JSON of the WinHint cost model. This page records those values, their sources, and the
microbenchmark that checks them on the machine (PROPOSAL §8). It belongs to the
[Real hardware](index.md) reference; why parameters are verified rather than assumed is in
[Evaluation methodology](../../concepts/methodology.md).

**Terms.** *ROB* is the reorder buffer, the instruction window; the *load buffer* and *store
buffer* hold in-flight memory operations; *fill buffers* track outstanding L1D misses, which
bound memory-level parallelism (*MLP*). A *knee* is the step in a probe's time-vs-N curve.
Other terms are in the [Glossary](../../reference/glossary.md).

The risk in PROPOSAL §8 is that the P-core and E-core window sizes we use (in `pie_daemon`'s PIE model,
`PIE_ROB_P`/`PIE_ROB_E`; in the libwinhint threshold `WINHINT_THRESHOLD`; and in the x86 target
JSON for the WinHint cost model) might be wrong. The mitigation is to cite Intel's documentation
and then check the values with microbenchmarks ([`hw/tools/uarch_probe.c`](../../../hw/tools/uarch_probe.c)).

## Which cores this chip has

`/proc/cpuinfo` on the evaluation host reports `Intel(R) Core(TM) 5 120U`, family 6,
**model 186 (0xBA)**, stepping 3. Model 0xBA is **Raptor Lake-P/U**, and the 120U is its
"Core 5" refresh. It is not Meteor Lake (models 170/172), so the cores are:

| CPUs | Core type | µarch | PMU in sysfs |
|---|---|---|---|
| 0–3 | P (2 cores × SMT) | **Raptor Cove** (Golden Cove core, larger L2) | `cpu_core` |
| 4–11 | E (2 modules × 4 cores) | **Gracemont** | `cpu_atom` |

`lscpu` gives 6.5 MiB of L2 in 4 instances (2 × 1.25 MiB P-core L2 + 2 × 2 MiB per E-module)
and 12 MiB of L3. That matches Raptor Lake-P.

## Documented values

Sources:
- **[ORM]** Intel® 64 and IA-32 Architectures Optimization Reference Manual, Vol. 1. Use the
  chapter on the Golden Cove microarchitecture (Alder Lake / Sapphire Rapids P-core) and its
  table comparing Golden Cove with Ice Lake/Sunny Cove. Raptor Cove keeps the Golden Cove core
  and changes only the L2 and the clocks.
- **[GRT]** The same manual's Gracemont (E-core) chapter, and Intel's Architecture Day 2021
  Gracemont disclosure.
- **[3P]** Third-party measurements (e.g. Chips and Cheese) are used only as a hint. They are
  never cited as the source.

"to confirm" means we could not verify the number against an Intel document when writing
this file. Treat it as unknown until the probe or the manual settles it. `hw/fidelity.py`
stores the confirmed numbers in `DOCUMENTED`, with `None` for "to confirm".

| Parameter | Raptor Cove (P) | src | Gracemont (E) | src | Probe |
|---|---|---|---|---|---|
| Reorder buffer (ROB) | 512 | ORM | 256 | GRT | `rob` |
| Allocation/rename width | 6 | ORM | 5 | GRT | — |
| Decode | 6-wide + 4K-uop cache | ORM | 2 × 3-wide clustered, no uop cache | GRT | — |
| Load buffer | 192 | ORM | to confirm | — | `lb` |
| Store buffer | 114 | ORM | to confirm | — | `sb` |
| Integer / FP physical registers | 280 / 332 | ORM | to confirm | — | (bounds `lb`) |
| L1D outstanding misses (fill buffers) | 16 | ORM | to confirm | — | `mlp` (k ≤ 12) |
| L2 outstanding misses | 48 | ORM | to confirm | — | — |
| L1D / L1I | 48 KB / 32 KB | ORM | 32 KB / 64 KB | GRT | — |
| L2 | 1.25 MB per core (this SKU, lscpu) | lscpu | 2 MB per 4-core module (this SKU, lscpu) | lscpu | — |
| Max frequency on this host | 5.0 GHz (`cpuinfo_max_freq`) | sysfs | 3.8 GHz | sysfs | — |

The PIE model in `hw/baselines/r4_pie/pie_daemon.c` uses ROB 512 / 256 and width 6 / 5. These
are the documented values above. The E-core load buffer, store buffer, register file and miss
handling sizes are **to confirm**, so they are not used anywhere.

## How the probe confirms them

`uarch_probe` generates (JITs) one loop per filler count N:

    loop: mov rax,[rax] ; N × filler ; mov rdx,[rdx] ; N × filler ; dec rcx ; jnz loop

The two loads are independent pointer chases. They walk a single random cycle through a buffer
much larger than the LLC (128 MB in the full run). Every timed point gets fresh, never-cached
segments of that cycle. Both loads miss in every cache level.

The second miss overlaps the first only while the first load, the N fillers and the second
load all fit in the structure the filler uses. So the time per iteration jumps from about one
memory latency to about two at N ≈ the size of that structure (H. Wong, "Measuring Reorder
Buffer Capacity", 2013). The probe reports the largest step in the curve as `size_lo..size_hi`
(the resolution is the step size of N).

| Probe | Filler | Structure | Expected knee (P / E) |
|---|---|---|---|
| `rob` | 1-byte `nop` | ROB | ≈ 512 / ≈ 256 |
| `lb` | `mov r11,[rsp-8]` (L1 hit) | load buffer (also bounded by the int PRF) | ≈ 192 / to confirm |
| `sb` | `mov [rsp-8],r11` | store buffer | ≈ 114 / to confirm |
| `mlp` | k chains, no filler | outstanding L1D misses | effective MLP = k·t(1)/t(k) saturates at min(k, fill buffers) |

`hw/fidelity.py::confirm_uarch` compares each knee with the documented value. Each result is:
- `confirmed`: within max(step, 10 %);
- `differs`;
- `no_knee`: no step in the range;
- `undocumented`: the measured value fills a "to confirm" cell;
- `consistent_bound`: MLP saturates near the largest k the probe can test, which is below the
  documented fill-buffer count.

Caveats:
- Run it on an idle machine, with a fixed frequency policy, pinned to one CPU. The SMT sibling
  of the P-core under test must be idle (or SMT off), because the ROB and buffers are
  partitioned between two active threads.
- Nops may hit decode limits on Gracemont, whose clustered decoders need taken branches.
  Because the probe needs only about N/3 cycles of decode for N nops, which is below one memory
  latency, this does not move the knee.

## Running it

```sh
# build + functional smoke (tiny ranges)
make -C hw -j1 && make -C hw uarch-smoke

# show the two commands (one P, one E CPU)
python hw/run_hw_experiments.py --uarch --dry-run

# full probe -> results/hw/uarch/uarch_{P,E}.{json,csv},
# uarch_check.json (documented vs knee)
python hw/run_hw_experiments.py --uarch

# one probe by hand
build/hw/uarch_probe -c E -p rob -o /tmp/rob_e.json
```

The full run is a measurement. Run it on the idle machine with the rest of the campaign; it is
not part of the build checks. After it runs:
- replace "to confirm" here with the measured value, marked as "probe";
- update `DOCUMENTED` in `hw/fidelity.py` only when an Intel document gives the value.

In the runbook the probe is part of [Step 3: Campaign](../usage.md#step-3-campaign)
(item c); the method is in [Real hardware §5.1](index.md).

## Next

- [Real hardware](index.md): the campaign that uses these values.
- [llama.cpp integration](llamacpp.md): the optional application
  workload.
