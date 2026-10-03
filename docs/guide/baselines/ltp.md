# B9 — Long-Term Parking (`window_policy=ltp`)

Reimplementation, in the gem5 v25.1 O3 CPU, of

> A. Sembrant, T. Carlson, E. Hagersten, D. Black-Schaffer, A. Perais,
> A. Seznec, P. Michaud. *Long Term Parking (LTP): Criticality-aware
> Resource Allocation in OOO Processors.* MICRO-48, 2015.

No public implementation exists (the original was in gem5 x86 full-system). LTP
keeps the ROB large and holds *non-urgent* instructions in a parking queue
before they allocate scheduling resources, so the IQ and the LQ/SQ hold only
the instructions that lead to long-latency loads (and branches). More of
those fit in a small IQ/LSQ, and the core exposes more memory-level parallelism.

B9 is in the comparison because it reaches the benefit of a large window by a different
route: instead of resizing the window, it keeps small scheduling structures and decides which
instructions may occupy them. It runs on the same window table and machines as every other
gem5 baseline, at equal IQ/LSQ resources ([Evaluation methodology](../../concepts/methodology.md),
[Baselines](index.md)).

**Terms.**

- *Parking queue* (the LTP): a FIFO between rename and IQ/LSQ dispatch that holds non-urgent
  instructions.
- *Urgent instruction*: one in the backward slice of a long-latency load or a mispredicted
  branch; its PC is in the *UIT* (Urgent Instruction Table).
- *IBDA*: iterative backward dependency analysis, which learns that slice one producer at a
  time.
- *Release*: moving a parked instruction into the IQ/LSQ.

Other terms are in the [Glossary](../../reference/glossary.md).

## Files

| File | Role |
|------|------|
| `ltp.{hh,cc}` | `LongTermParking`: parking queue, Urgent Instruction Table (UIT), IBDA training, `system.cpu.ltp.*` statistics, `LtpParams` (`window_args`) |
| `ltp_iew.cc` | the pipeline side, as `IEW` member functions: `ltpInit`, `ltpRelease`, `ltpDispatchInsts`, `ltpToQueues`, `ltpSquash` |
| `ltp_policy.cc` | registers `ltp` in the `WindowPolicyRegistry`. Window decisions are static (`window_initial`). It sets which structures follow the configuration (`structs`, default IQ+LQ+SQ) |
| `ltp_hooks.hh` | two free functions that `commit.cc` calls (ROB pointer, commit-time training) |
| `SConscript` | compiles the three `.cc` files; debug flag `LTP` |
| `tests/ltp_test.c`, `tests/run_ltp_tests.sh` | the end-to-end check: `ltp` against `static`, output compared with qemu |
| `ltp_core.hh`, `tests/test_ltp_core.cc` | gem5-independent UIT/IBDA and the release/dispatch rules used by `ltp_iew.cc`; host unit test: `make -C sim/gem5/src/cpu/o3/window/ltp/tests` |
| `../../../../../../patches/gem5_v25.1.0.0_zz_ltp.patch` | hooks in existing gem5 files. It applies **after** `gem5_v25.1.0.0_winhint.patch` (name order) |

### What the `zz_ltp` patch changes (existing gem5 files, 129 added lines, 0 removed)

| File | Hunk |
|------|------|
| `cpu/o3/iew.hh` | `std::shared_ptr<LongTermParking> ltp` (null unless `window_policy=ltp`) and the declarations of the `ltp*` member functions, at the end of the class |
| `cpu/o3/iew.cc` | `ltpInit(params)` at the end of the constructor; `ltpRelease(tid)` at the top of `dispatch()`, which runs every cycle even when dispatch is blocked; `dispatchInsts()` calls `ltpDispatchInsts()` when LTP is on; `ltpSquash(tid)` in `squash()`; `isDrained()` also requires an empty LTP |
| `cpu/o3/inst_queue.{hh,cc}` | `insertParked()` and `forgetParkedProducer()`, appended to the end of the file |
| `cpu/o3/commit.cc` | gives the LTP the ROB in `startupStage()`, and trains the UIT in `commitHead()` after `updateComInstStats()` |

With any policy other than `ltp`, `IEW::ltp` is null. Every hook is then a single
null test, so the behaviour is bit-identical to the winhint patch alone. None of
these hunks touches the lines changed by the winhint patch.

## Design

### Where the parking queue sits (paper vs. this model)

The paper parks instructions after register renaming (to virtual tags) and
before physical-register and IQ/LSQ allocation. Physical registers, IQ entries
and LQ/SQ entries are allocated only when an instruction leaves the queue,
while ROB entries are allocated in order for everyone.

gem5 O3 allocates physical registers in `Rename` and the ROB/IQ/LSQ entries
right after it (ROB in `Commit::getInsts`, IQ/LSQ in `IEW::dispatchInsts`).
This model therefore places the parking queue **between rename and IQ/LSQ
dispatch**, inside IEW dispatch, for these reasons:

- **ROB.** It is allocated in program order for every instruction, as in the
  paper. Commit is unchanged.
- **IQ and LQ/SQ.** They are allocated only at release. This is the resource
  the comparison is about, and it is exactly what the paper parks.
- **Physical registers.** These are allocated at rename, so this is a deviation.
  Deferring them would require virtual-physical registers in gem5's rename map,
  free list and scoreboard, which is a rewrite of rename. It changes nothing
  here: the project's register file is sized so that it never binds
  ([interfaces.md §3](../../interfaces.md#3-window-configuration-table): 288 = 32 architectural + the largest ROB). The paper's
  register-file savings are therefore not credited, and are not needed for
  equal-resource comparisons. `system.cpu.ltp.occMean` approximates the
  registers held by parked instructions.

Parking between decode and rename was rejected. Instructions that have not
been renamed cannot be bypassed, because rename must see producers in program
order. That would need a second rename stage.

### Urgency: UIT + iterative backward dependency analysis

- **UIT** (`uit_entries` = 256, `uit_assoc` = 4, LRU, full-PC tags) holds the
  PCs of urgent instructions.
- **Seeds**, trained at commit and therefore non-speculative:
  - every committed load whose load-to-use latency (`lastWakeDependents −
    firstIssue`) is at least `lll` cycles (default 30, which is above an L2
    hit of about 24 cycles on `riscv_ooo.json`);
  - branches: mispredicted ones (`branch_seed=1`, the default), all of them
    (`2`) or none (`0`).
- **Backward slice (IBDA).** A Register Dependence Table maps each
  architectural register to the PC of its last producer, and is updated in
  program order at dispatch. When an urgent instruction is dispatched, the
  producers of its source registers are inserted in the UIT. One more link of
  the backward slice is learnt per dynamic visit, as in the paper (and the Load
  Slice Core, from the same group).
- **Classification** happens at dispatch: an instruction is urgent if and only if
  its PC hits in the UIT.

### Dispatch (`ltpDispatchInsts`)

New instructions are taken in program order, `dispatchWidth` per cycle.
Each one goes down exactly one of these paths:

| Instruction | Action |
|-------------|--------|
| squashed | dropped, as in stock gem5 |
| drain class (non-speculative, serializing, squash-after, store-conditional, AMO, barrier, HTM) | dispatched only when the thread's LTP is empty. Otherwise dispatch stalls and the LTP drains (`drainStalls`, `releasedBy::drain`) |
| memory op while an older memory op is parked | parked, to keep LSQ order. If it is urgent it is flagged, and older parked memory ops are pulled out for it (`parkedUrgentMem`, `releasedBy::memOrder`) |
| non-urgent | parked, unless the LTP is empty **and** the IQ has room (free > `room` × cap). In that case it bypasses the queue (`bypassed`). If the LTP is full, dispatch stalls (`fullStalls`) |
| urgent, or a nop | dispatched to the IQ/LSQ. While something is parked it must leave `reserve` IQ entries free (`reserveStalls`) |

A parked instruction registers in the IQ as the producer of its destination
physical registers (`IQ::recordProducer`). Consumers that bypass it then wait
for it in the dependency graph, and its later `insertParked()` skips the
producer step.

### Release (`ltpRelease`, every cycle, before new dispatches)

The queue is scanned oldest first. A parked instruction is eligible when the
first of these holds (reason in `releasedBy`):

1. `old`: it is within `wake` (16) sequence numbers of the ROB head. This is
   the paper's wake-up, which makes sure an instruction reaches the IQ before
   it would stall commit.
2. `drain`: a drain-class instruction waits behind the LTP.
3. `urgent`: it was parked urgent (memory order), or its PC has become urgent
   since it was parked.
4. `memOrder`: it is a memory op older than a parked urgent memory op.
5. `full`: the LTP is full and it is the head.
6. `room`: the IQ has room (free > `room` × cap).

There are two more constraints. Memory ops leave in program order: once one
cannot leave, no younger one may. And the IQ write ports (`dispatchWidth` per
cycle) are shared between released and new instructions, with released ones
first. Parking itself uses no IQ port.

### Correctness invariants (gem5 program-order requirements)

1. **ROB.** It is allocated in program order (commit is unchanged), so in-order
   commit, precise traps and squash-by-sequence-number are untouched.
2. **LSQ and memory-dependence unit.** Loads and stores enter the LQ/SQ and the
   `MemDepUnit` in program order. A memory op never bypasses a parked older
   memory op, and parked memory ops leave oldest first. gem5's store-to-load
   forwarding and its violation checks rely on positional age in the LQ/SQ,
   and this rule keeps them exact.
3. **IQ.** `instList` is kept sorted by sequence number (`insertParked` does a
   sorted insertion), because `IQ::doSquash` and `IQ::commit` walk it by age.
   The dependency graph is correct because producers register at park time
   (see above).
4. **Squash.** `IEW::squash` runs `IQ::squash` first, which unlinks squashed
   consumers. It then removes every parked instruction younger than the
   squash point and clears its dependency-graph heads (`forgetParkedProducer`).
   Parked instructions hold no IQ/LSQ state, so nothing else needs undoing.
5. **No deadlock.** Urgent dispatches and every release except one leave
   `reserve` (≥ 1) IQ entries free. Only an `old` release of the *oldest*
   parked instruction may use the reserve. Every instruction that could occupy
   the reserve was the oldest parked instruction when it was released, so it is
   older than the oldest parked instruction now. If that oldest parked
   instruction is the oldest unexecuted one, everything in the reserve has
   already issued, so it always gets an IQ entry. LQ/SQ entries cannot deadlock
   either, because the LSQ holds only memory ops older than every parked one.
   Releases run every cycle, even while dispatch is blocked.

## Equal-resource configuration

LTP shares the window table ([interfaces.md §3](../../interfaces.md#3-window-configuration-table)). With `structs=iq+lsq`
(the default), the policy caps **IQ, LQ and SQ** at configuration
`window_initial`, and the **ROB stays at the largest configuration** (256).
The physical register file is 288+288 in every run. For example,
`--window-policy ltp --window-initial 1` runs ROB 256 / IQ 64 / LQ 32 / SQ 32
plus a 128-entry parking queue. It is compared with `static` at
`--window-initial 1` (ROB 128 / IQ 64 / LQ 32 / SQ 32), which has the same
IQ/LSQ, and at `--window-initial 3` (ROB 256 / IQ 128 / LQ 64 / SQ 64), the
large window.

The parking queue is a FIFO with no wake-up logic, so in the paper's energy
model it is far cheaper than IQ entries. McPAT has no LTP component.
`sim/estimate_energy.py` should model it as a 128-entry RAM FIFO of
instruction-sized entries, plus the UIT as a 256-entry 4-way tagged table.

## Tunables (`--window-args`, comma-separated `k=v`)

| Key | Default | Meaning |
|-----|---------|---------|
| `entries` | 128 | parking-queue entries per thread |
| `uit_entries`, `uit_assoc` | 256, 4 | UIT geometry |
| `lll` | 30 | a load whose load-to-use latency is at least this many cycles is a seed |
| `wake` | 16 | `old` release distance from the ROB head (sequence numbers) |
| `reserve` | 4 | IQ entries kept for `old` releases (≥ 1) |
| `room` | 0.5 | the IQ "has room" when free > `room` × cap. `room=1` disables `room` releases and bypasses (pure paper-style parking) |
| `branch_seed` | 1 | 0 none, 1 mispredicted branches, 2 all branches |
| `structs` | `iq+lsq` | structures capped at `window_initial` (`rob`, `iq`, `lq`, `sq`, `lsq`, `all`, joined with `+` or `:`; empty = `all`; same parser as `hint`, `parseWindowStructs()`) |

## Statistics (`system.cpu.ltp.*`)

| Group | Statistics |
|-------|------------|
| Classification | `urgent`, `nonUrgent` |
| Park and release events | `parked`, `parkedUrgentMem`, `bypassed`, `released`, `releasedBy::{old,drain,urgent,memOrder,full,room,total}`, `squashed` |
| Stalls | `fullStalls`, `reserveStalls`, `drainStalls` |
| UIT | `seedsLoad`, `seedsBranch`, `uitInserts`, `uitEvictions` |
| Occupancy | `occMean`, `occMax`, `occDist`, `parkedCycles`, `meanParkCycles` |

Use `--debug-flags=LTP` for a per-instruction trace of park, release and squash.

## Deviations from the paper (summary)

- Physical registers are allocated at rename, not when an instruction leaves
  the queue (see above). The register file never binds in this project.
- Memory ops keep LSQ program order (gem5 requirement), so an urgent load
  behind a parked store pulls that store out (`memOrder`).
- Releases also happen when the IQ has room (`room`), and non-urgent
  instructions bypass an empty queue when the IQ has room. This avoids a
  one-cycle park/unpark bubble in compute-bound code. Set `room=1` for strict
  parking.
- Seeds come from measured load-to-use latency at commit (≥ `lll`), rather
  than from an LLC-miss flag.
- Distance to the ROB head is measured in sequence numbers, which over-counts
  right after a squash. That only delays an `old` release, and the head itself
  is always eligible.

## Running

```bash
# build (overlay + both patches)
tooling/winhint.sh gem5:build winhint

# one run: ROB 256, IQ/LQ/SQ of configuration 1, tunables at their defaults
$WINHINT_BUILD/gem5/src/build/RISCV_winhint/gem5.opt --outdir=m5out \
    sim/se.py --machine sim/machines/riscv_ooo.json \
    --cmd $WINHINT_BUILD/benchmarks/riscv/plain/encoder_bert_tiny_infer --options small \
    --window-policy ltp --window-initial 1 \
    --window-args "entries=128,wake=16,room=0.5"

# end-to-end check (each gem5 run takes the heavy lock);
# optional arguments: the gem5.opt binary and the iteration count
sim/gem5/src/cpu/o3/window/ltp/tests/run_ltp_tests.sh
```

In the evaluation matrix B9 runs as the `ltp` and `ltp_c<i>` variants
([gem5 model](../gem5/index.md#evaluation-matrix-run_experimentspy)).
Its paper-trend check is in
[Simulator fidelity](../fidelity.md),
and its deviations are summarized in [Deviations](../../deviations.md#b9).

## Next

- [Window policies](../gem5/policies.md): the reactive B2–B5 policies.
- [gem5 patches](../gem5/patches.md): the shared resize mechanism and the
  winhint patch the `zz_ltp` patch applies on top of.
