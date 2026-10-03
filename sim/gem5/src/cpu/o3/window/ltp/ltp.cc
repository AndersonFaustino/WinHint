/**
 * @file
 * @brief WinHint B9: Long-Term Parking -- state, IBDA classification,
 *        statistics. See ltp.hh and README.md.
 */

#include "cpu/o3/window/ltp/ltp.hh"

#include <algorithm>

#include "base/logging.hh"
#include "cpu/o3/cpu.hh"
#include "cpu/o3/dyn_inst.hh"
#include "cpu/o3/rob.hh"
#include "cpu/o3/window/hint_policy.hh"
#include "cpu/o3/window/ltp/ltp_hooks.hh"
#include "cpu/o3/window/policy.hh"
#include "debug/LTP.hh"

namespace gem5
{

namespace o3
{

// See ltp.hh.
LtpParams
LtpParams::fromArgs(const WindowArgs &a)
{
    LtpParams p;
    p.entries = (unsigned)a.getInt("entries", p.entries);
    p.uitEntries = (unsigned)a.getInt("uit_entries", p.uitEntries);
    p.uitAssoc = (unsigned)a.getInt("uit_assoc", p.uitAssoc);
    p.lllCycles = (unsigned)a.getInt("lll", p.lllCycles);
    p.wakeDist = (unsigned)a.getInt("wake", p.wakeDist);
    p.iqReserve = (unsigned)a.getInt("reserve", p.iqReserve);
    p.roomFrac = a.getDouble("room", p.roomFrac);
    p.branchSeed = (unsigned)a.getInt("branch_seed", p.branchSeed);
    p.structs = parseWindowStructs(a.getString("structs", "iq+lsq"));

    fatal_if(p.entries < 1, "ltp: entries must be >= 1");
    fatal_if(p.uitAssoc < 1 || p.uitEntries < p.uitAssoc ||
             p.uitEntries % p.uitAssoc,
             "ltp: uit_entries must be a multiple of uit_assoc");
    fatal_if(p.iqReserve < 1, "ltp: reserve must be >= 1 (deadlock "
             "freedom: the oldest parked instruction needs an IQ entry)");
    fatal_if(p.roomFrac < 0, "ltp: room must be >= 0");
    fatal_if(p.branchSeed > 2, "ltp: branch_seed must be 0, 1 or 2");
    return p;
}

// See ltp.hh.
LongTermParking::LongTermParking(CPU *_cpu, const LtpParams &params,
                                 ThreadID num_threads)
    : stats(_cpu), cpu(_cpu), p(params), numThreads(num_threads),
      urgency(params.uitEntries, params.uitAssoc, num_threads)
{
    stats.occDist.init(0, p.entries, std::max(1u, p.entries / 16));
    inform("LTP (B9): %u-entry parking queue, UIT %u x %u-way, lll=%u, "
           "wake=%u, reserve=%u, room=%.2f, branch_seed=%u",
           p.entries, urgency.numSets(), p.uitAssoc, p.lllCycles, p.wakeDist,
           p.iqReserve, p.roomFrac, p.branchSeed);
}

// See ltp.hh.
uint32_t
LongTermParking::regKey(const RegId &r)
{
    return ((uint32_t)r.classValue() << 16) | (uint32_t)r.index();
}

// See ltp.hh.
bool
LongTermParking::uitHit(Addr pc, bool touch)
{
    return urgency.hit(pc, touch);
}

// See ltp.hh.
void
LongTermParking::syncUitStats()
{
    stats.uitInserts += (double)(urgency.inserts - uitInsertsSeen);
    stats.uitEvictions += (double)(urgency.evictions - uitEvictionsSeen);
    uitInsertsSeen = urgency.inserts;
    uitEvictionsSeen = urgency.evictions;
}

// See ltp.hh.
void
LongTermParking::uitInsert(Addr pc)
{
    urgency.insert(pc);
    syncUitStats();
}

// See ltp.hh.
bool
LongTermParking::classify(const DynInstPtr &inst)
{
    // IBDA (ltp_core.hh): an urgent instruction makes the producers of its
    // sources urgent (they are found next time they are dispatched); every
    // instruction becomes the last producer of its destinations.
    srcKeys.clear();
    dstKeys.clear();
    for (int i = 0; i < inst->numSrcRegs(); i++)
        if (!inst->renamedSrcIdx(i)->isAlwaysReady())
            srcKeys.push_back(regKey(inst->srcRegIdx(i)));
    for (int i = 0; i < inst->numDestRegs(); i++)
        if (!inst->renamedDestIdx(i)->isAlwaysReady())
            dstKeys.push_back(regKey(inst->destRegIdx(i)));

    const bool urgent = urgency.classify(
        (unsigned)inst->threadNumber, inst->pcState().instAddr(),
        srcKeys.data(), (int)srcKeys.size(), dstKeys.data(),
        (int)dstKeys.size());
    syncUitStats();

    if (urgent)
        ++stats.urgent;
    else
        ++stats.nonUrgent;
    return urgent;
}

// See ltp.hh.
void
LongTermParking::commitInst(const DynInstPtr &inst)
{
    const Addr pc = inst->pcState().instAddr();
    if (inst->isLoad() && !inst->isDataPrefetch() &&
        !inst->isInstPrefetch() && inst->firstIssue != (Tick)-1 &&
        inst->lastWakeDependents != (Tick)-1 &&
        inst->lastWakeDependents >= inst->firstIssue) {
        Cycles lat = cpu->ticksToCycles(inst->lastWakeDependents -
                                        inst->firstIssue);
        if (lat >= p.lllCycles) {
            DPRINTF(LTP, "seed: long-latency load pc %#x (%llu cycles)\n",
                    pc, (unsigned long long)lat);
            uitInsert(pc);
            ++stats.seedsLoad;
        }
    }
    if (inst->isControl() &&
        (p.branchSeed == 2 || (p.branchSeed == 1 && inst->mispredicted()))) {
        uitInsert(pc);
        ++stats.seedsBranch;
    }
}

// See ltp.hh.
bool
LongTermParking::mustDrain(const DynInstPtr &inst)
{
    return inst->isNonSpeculative() || inst->isSerializeBefore() ||
        inst->isSerializeAfter() || inst->isSquashAfter() ||
        inst->isStoreConditional() || inst->isAtomic() ||
        inst->isReadBarrier() || inst->isWriteBarrier() ||
        inst->isFullMemBarrier() || inst->isHtmCmd();
}

// See ltp.hh.
bool
LongTermParking::allEmpty() const
{
    for (ThreadID t = 0; t < numThreads; t++)
        if (!q[t].empty())
            return false;
    return true;
}

// See ltp.hh.
void
LongTermParking::park(const DynInstPtr &inst, bool urgent, Cycles now)
{
    const ThreadID tid = inst->threadNumber;
    assert(!full(tid));
    assert(q[tid].empty() || q[tid].back().inst->seqNum < inst->seqNum);
    const bool mem = inst->isMemRef();
    q[tid].push_back(Entry{inst, urgent, mem, now});
    if (mem)
        nMem[tid]++;
    if (urgent) {
        assert(mem);
        urgentMemSn[tid] = inst->seqNum;
        ++stats.parkedUrgentMem;
    }
    ++stats.parked;
    DPRINTF(LTP, "[tid:%i] park [sn:%llu] pc %s%s (occ %u)\n", tid,
            inst->seqNum, inst->pcState(), urgent ? " (urgent, mem order)"
            : "", (unsigned)q[tid].size());
}

// See ltp.hh.
LongTermParking::Queue::iterator
LongTermParking::release(ThreadID tid, Queue::iterator it, Reason why,
                         Cycles now)
{
    DPRINTF(LTP, "[tid:%i] release [sn:%llu] reason %d after %llu cycles\n",
            tid, it->inst->seqNum, (int)why,
            (unsigned long long)(now - it->parkedAt));
    if (it->mem)
        nMem[tid]--;
    if (it->inst->seqNum == urgentMemSn[tid])
        urgentMemSn[tid] = 0;
    ++stats.released;
    stats.releasedBy[why]++;
    stats.parkedLatency += (double)(now - it->parkedAt);
    return q[tid].erase(it);
}

// See ltp.hh.
LongTermParking::Queue::iterator
LongTermParking::drop(ThreadID tid, Queue::iterator it)
{
    if (it->mem)
        nMem[tid]--;
    if (it->inst->seqNum == urgentMemSn[tid])
        urgentMemSn[tid] = 0;
    ++stats.squashed;
    return q[tid].erase(it);
}

// See ltp.hh.
bool
LongTermParking::isOld(ThreadID tid, InstSeqNum sn) const
{
    if (!rob || rob->isEmpty(tid))
        return true;
    return sn <= rob->readHeadInst(tid)->seqNum + p.wakeDist;
}

// See ltp.hh.
void
LongTermParking::sample(ThreadID tid)
{
    const double occ = (double)q[tid].size();
    stats.occSum += occ;
    ++stats.cycles;
    if (occ > 0)
        ++stats.parkedCycles;
    if (occ > stats.occMax.value())
        stats.occMax = occ;
    stats.occDist.sample(occ);
}

// See ltp.hh.
LongTermParking::LtpStats::LtpStats(statistics::Group *parent)
    : statistics::Group(parent, "ltp"),
      ADD_STAT(urgent, statistics::units::Count::get(),
               "Instructions classified urgent at dispatch (UIT hit)"),
      ADD_STAT(nonUrgent, statistics::units::Count::get(),
               "Instructions classified non-urgent at dispatch"),
      ADD_STAT(parked, statistics::units::Count::get(),
               "Instructions parked (park events)"),
      ADD_STAT(parkedUrgentMem, statistics::units::Count::get(),
               "Urgent memory instructions parked to keep LSQ order"),
      ADD_STAT(bypassed, statistics::units::Count::get(),
               "Non-urgent instructions dispatched directly (LTP empty, "
               "IQ has room)"),
      ADD_STAT(released, statistics::units::Count::get(),
               "Parked instructions released to the IQ/LSQ (unpark "
               "events)"),
      ADD_STAT(releasedBy, statistics::units::Count::get(),
               "Releases by reason"),
      ADD_STAT(squashed, statistics::units::Count::get(),
               "Parked instructions squashed in the LTP"),
      ADD_STAT(fullStalls, statistics::units::Count::get(),
               "Dispatch stalls: LTP full"),
      ADD_STAT(reserveStalls, statistics::units::Count::get(),
               "Dispatch stalls: urgent instruction blocked by the IQ "
               "reserve or a full IQ/LSQ"),
      ADD_STAT(drainStalls, statistics::units::Count::get(),
               "Dispatch stalls: serializing/non-speculative instruction "
               "waits for the LTP to drain"),
      ADD_STAT(seedsLoad, statistics::units::Count::get(),
               "UIT seeds: committed long-latency loads"),
      ADD_STAT(seedsBranch, statistics::units::Count::get(),
               "UIT seeds: committed branches"),
      ADD_STAT(uitInserts, statistics::units::Count::get(),
               "UIT insertions (seeds + backward-slice propagation)"),
      ADD_STAT(uitEvictions, statistics::units::Count::get(),
               "UIT evictions"),
      ADD_STAT(occSum, statistics::units::Count::get(),
               "Sum over cycles of the LTP occupancy"),
      ADD_STAT(cycles, statistics::units::Cycle::get(),
               "Cycles sampled"),
      ADD_STAT(parkedCycles, statistics::units::Cycle::get(),
               "Cycles with at least one parked instruction"),
      ADD_STAT(occMax, statistics::units::Count::get(),
               "Maximum LTP occupancy"),
      ADD_STAT(parkedLatency, statistics::units::Cycle::get(),
               "Total cycles spent parked by released instructions"),
      ADD_STAT(occMean, statistics::units::Rate<
                   statistics::units::Count, statistics::units::Cycle>::get(),
               "Mean LTP occupancy", occSum / cycles),
      ADD_STAT(meanParkCycles, statistics::units::Rate<
                   statistics::units::Cycle, statistics::units::Count>::get(),
               "Mean cycles a released instruction stayed parked",
               parkedLatency / released),
      ADD_STAT(occDist, statistics::units::Count::get(),
               "LTP occupancy distribution (per cycle)")
{
    releasedBy.init(NumReasons);
    releasedBy.subname(RelOld, "old");
    releasedBy.subname(RelDrain, "drain");
    releasedBy.subname(RelUrgent, "urgent");
    releasedBy.subname(RelMemOrder, "memOrder");
    releasedBy.subname(RelFull, "full");
    releasedBy.subname(RelRoom, "room");
    releasedBy.flags(statistics::total);
}

// Hooks for commit.cc (ltp_hooks.hh).
// See ltp_hooks.hh.
void
ltpSetROB(LongTermParking *ltp, ROB *rob)
{
    ltp->setROB(rob);
}

// See ltp_hooks.hh.
void
ltpCommit(LongTermParking *ltp, const DynInstPtr &inst)
{
    ltp->commitInst(inst);
}

} // namespace o3
} // namespace gem5
