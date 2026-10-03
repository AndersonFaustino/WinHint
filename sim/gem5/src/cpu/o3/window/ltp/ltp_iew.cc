/**
 * @file
 * @brief WinHint B9: Long-Term Parking -- the IEW (dispatch) side.
 *
 * These are IEW member functions (declared in cpu/o3/iew.hh by
 * gem5_v25.1.0.0_zz_ltp.patch). With window_policy=ltp, IEW::dispatch()
 * calls ltpRelease() every cycle and IEW::dispatchInsts() is replaced by
 * ltpDispatchInsts().
 *
 * Invariants (see README.md, "Correctness"):
 *  1. Every instruction gets its ROB entry in program order (commit stage,
 *     unchanged). Only the IQ and LQ/SQ allocation is deferred.
 *  2. Memory instructions enter the LSQ (and the memory-dependence unit) in
 *     program order: a memory instruction is never dispatched while an
 *     older memory instruction is parked, and parked memory instructions
 *     leave the LTP oldest first.
 *  3. A parked instruction registers as the producer of its destination
 *     physical registers when it is parked (IQ::recordProducer), so
 *     consumers that bypass it wait for it in the IQ dependency graph.
 *  4. Serializing / non-speculative / barrier / atomic / SC instructions
 *     dispatch only when the thread's LTP is empty.
 *  5. Deadlock freedom: urgent dispatch leaves iqReserve >= 1 IQ entries
 *     free while something is parked, releases go oldest first, and an
 *     instruction within wakeDist of the ROB head is always eligible.
 */

#include "cpu/o3/cpu.hh"
#include "cpu/o3/dyn_inst.hh"
#include "cpu/o3/iew.hh"
#include "cpu/o3/window/ltp/ltp.hh"
#include "cpu/o3/window/policy.hh"
#include "debug/IEW.hh"
#include "debug/LTP.hh"
#include "params/BaseO3CPU.hh"

namespace gem5
{

namespace o3
{

/**
 * @brief Create the LTP if window_policy is "ltp" (else leave ltp null).
 * @param params CPU params (window_policy, window_args).
 */
void
IEW::ltpInit(const BaseO3CPUParams &params)
{
    if (params.window_policy != "ltp")
        return;
    WindowArgs args(params.window_args);
    ltp = std::make_shared<LongTermParking>(
        cpu, LtpParams::fromArgs(args), numThreads);
}

/** @return true if there is no LTP or nothing is parked in any thread. */
bool
IEW::ltpEmpty() const
{
    return !ltp || ltp->allEmpty();
}

/**
 * @brief Whether the IQ "has room" for thread tid: free entries > roomFrac
 *        x (free + occupied).
 * @param tid Thread.
 * @return The room test.
 */
bool
IEW::ltpRoom(ThreadID tid)
{
    const unsigned free_e = instQueue.numFreeEntries(tid);
    const unsigned occ = instQueue.getCount(tid);
    return (double)free_e > ltp->params().roomFrac * (double)(free_e + occ);
}

/**
 * @brief Drop the parked instructions younger than commit's doneSeqNum
 *        (and forget them as IQ producers); reset the drain flag and the
 *        classification cache.
 * @param tid Thread being squashed.
 */
void
IEW::ltpSquash(ThreadID tid)
{
    const InstSeqNum done = fromCommit->commitInfo[tid].doneSeqNum;
    auto &q = ltp->queue(tid);
    auto it = q.begin();
    while (it != q.end()) {
        if (it->inst->seqNum > done) {
            DPRINTF(LTP, "[tid:%i] squash parked [sn:%llu]\n", tid,
                    it->inst->seqNum);
            instQueue.forgetParkedProducer(it->inst);
            it = ltp->drop(tid, it);
        } else {
            ++it;
        }
    }
    ltp->draining[tid] = false;
    ltp->lastSn[tid] = 0;
}

/**
 * @brief Insert an instruction into the IQ/LSQ (the body of the stock
 *        dispatchInsts() for one instruction).
 * @param inst Instruction.
 * @param tid Thread.
 * @param released true for a release from the LTP (uses
 *        IQ::insertParked(); such instructions are never atomic, SC,
 *        barrier, nop or non-speculative).
 */
void
IEW::ltpToQueues(const DynInstPtr &inst, ThreadID tid, bool released)
{
    // hardware transactional memory (as in dispatchInsts()).
    const int numHtmStarts = ldstQueue.numHtmStarts(tid);
    const int numHtmStops = ldstQueue.numHtmStops(tid);
    const int htmDepth = numHtmStarts - numHtmStops;
    if (htmDepth > 0) {
        inst->setHtmTransactionalState(ldstQueue.getLatestHtmUid(tid),
                                        htmDepth);
    } else {
        inst->clearHtmTransactionalState();
    }

    bool add_to_iq = false;
    if (inst->isAtomic()) {
        assert(!released);
        ldstQueue.insertStore(inst);
        ++iewStats.dispStoreInsts;
        inst->setCanCommit();
        instQueue.insertNonSpec(inst);
        ++iewStats.dispNonSpecInsts;
    } else if (inst->isLoad()) {
        ldstQueue.insertLoad(inst);
        ++iewStats.dispLoadInsts;
        add_to_iq = true;
    } else if (inst->isStore()) {
        ldstQueue.insertStore(inst);
        ++iewStats.dispStoreInsts;
        if (inst->isStoreConditional()) {
            assert(!released);
            inst->setCanCommit();
            instQueue.insertNonSpec(inst);
            ++iewStats.dispNonSpecInsts;
        } else {
            add_to_iq = true;
        }
    } else if (inst->isReadBarrier() || inst->isWriteBarrier()) {
        assert(!released);
        inst->setCanCommit();
        instQueue.insertBarrier(inst);
    } else if (inst->isNop()) {
        assert(!released);
        inst->setIssued();
        inst->setExecuted();
        inst->setCanCommit();
        instQueue.recordProducer(inst);
        cpu->executeStats[tid]->numNop++;
    } else {
        assert(!inst->isExecuted());
        add_to_iq = true;
    }

    if (add_to_iq && inst->isNonSpeculative()) {
        assert(!released);
        inst->setCanCommit();
        instQueue.insertNonSpec(inst);
        ++iewStats.dispNonSpecInsts;
        add_to_iq = false;
    }

    if (add_to_iq) {
        if (released)
            instQueue.insertParked(inst);
        else
            instQueue.insert(inst);
    }

    inst->dispatchTick = curTick() - inst->fetchTick;
    ppDispatch->notify(inst);
}

/**
 * @brief Per-cycle release stage: sample the LTP, then walk the thread's
 *        FIFO oldest first and move eligible instructions (releaseReason()
 *        with IQ reserve and LQ/SQ checks, memory ops in order) to the
 *        IQ/LSQ, up to dispatchWidth (ltpIqSlots).
 * @param tid Thread.
 */
void
IEW::ltpRelease(ThreadID tid)
{
    LongTermParking &L = *ltp;
    ltpIqSlots[tid] = dispatchWidth;
    L.sample(tid);

    auto &q = L.queue(tid);
    if (q.empty()) {
        L.draining[tid] = false;
        L.urgentMemSn[tid] = 0;
        return;
    }

    const Cycles now = cpu->curCycle();
    const bool full = L.full(tid);
    // Once a parked memory instruction cannot leave, no younger one may
    // (in-order LSQ allocation).
    bool mem_blocked = false;
    bool released_any = false;
    // True while no live (unsquashed) entry has been skipped: the entry
    // under examination is then the oldest parked instruction.
    bool at_head = true;

    for (auto it = q.begin(); it != q.end() && ltpIqSlots[tid] > 0; ) {
        const DynInstPtr inst = it->inst;
        if (inst->isSquashed()) {
            // Commit already squashed it; IEW::squash() will follow.
            ++it;
            continue;
        }

        // Release rule (ltp_core.hh, unit-tested on the host).
        winhint::ltp::ParkedView v;
        v.old = L.isOld(tid, inst->seqNum);
        v.draining = L.draining[tid];
        v.urgent = it->urgent ||
            L.uitHit(inst->pcState().instAddr(), false);
        v.olderThanUrgentMem = it->mem && L.urgentMemSn[tid] &&
            inst->seqNum < L.urgentMemSn[tid];
        v.fullHead = full && it == q.begin();
        v.room = ltpRoom(tid);
        const LongTermParking::Reason why = winhint::ltp::releaseReason(v);

        if (why == LongTermParking::NumReasons ||
            (it->mem && mem_blocked)) {
            if (it->mem)
                mem_blocked = true;
            at_head = false;
            ++it;
            continue;
        }

        // Resources. The last iqReserve IQ entries are kept for the oldest
        // parked instruction once it is near the ROB head: it is older
        // than everything else that could hold them, so it always gets in
        // (deadlock freedom, invariant 5). Memory instructions also need
        // an LQ/SQ entry.
        const unsigned need =
            winhint::ltp::releaseNeed(why, at_head, L.params().iqReserve);
        if (instQueue.numFreeEntries(tid) == 0)
            break;
        if (instQueue.numFreeEntries(inst) < need ||
            (inst->isLoad() && ldstQueue.lqFull(tid)) ||
            (inst->isStore() && ldstQueue.sqFull(tid))) {
            if (it->mem)
                mem_blocked = true;
            at_head = false;
            ++it;
            continue;
        }

        it = L.release(tid, it, why, now);
        ltpToQueues(inst, tid, true);
        --ltpIqSlots[tid];
        released_any = true;
    }

    if (q.empty())
        L.draining[tid] = false;

    if (released_any) {
        updatedQueues = true;
        activityThisCycle();
        if (dispatchStatus[tid] == Idle)
            dispatchStatus[tid] = Running;
    }
}

/**
 * @brief Replacement for IEW::dispatchInsts() under LTP: classify each
 *        instruction once, then park, stall, or dispatch it directly per
 *        dispatchAction(), blocking on a full LTP, the IQ reserve, a full
 *        LQ/SQ or exhausted IQ write slots.
 * @param tid Thread.
 */
void
IEW::ltpDispatchInsts(ThreadID tid)
{
    LongTermParking &L = *ltp;
    std::queue<DynInstPtr> &insts_to_dispatch =
        dispatchStatus[tid] == Unblocking ?
        skidBuffer[tid] : insts[tid];

    const int insts_to_add = insts_to_dispatch.size();
    const Cycles now = cpu->curCycle();
    int dis_num_inst = 0;

    for ( ; dis_num_inst < insts_to_add &&
              dis_num_inst < dispatchWidth;
          ++dis_num_inst)
    {
        DynInstPtr inst = insts_to_dispatch.front();
        assert(inst);

        if (inst->isSquashed()) {
            ++iewStats.dispSquashedInsts;
            insts_to_dispatch.pop();
            if (inst->isLoad())
                toRename->iewInfo[tid].dispatchedToLQ++;
            if (inst->isStore() || inst->isAtomic())
                toRename->iewInfo[tid].dispatchedToSQ++;
            toRename->iewInfo[tid].dispatched++;
            continue;
        }

        // Classify once (a blocked instruction is re-examined later).
        if (L.lastSn[tid] != inst->seqNum) {
            L.lastSn[tid] = inst->seqNum;
            L.lastUrgent[tid] = L.classify(inst);
        }
        const bool urgent = L.lastUrgent[tid];
        const bool drain = LongTermParking::mustDrain(inst);
        const bool mem = inst->isMemRef();

        // Dispatch rule (ltp_core.hh, unit-tested on the host).
        const winhint::ltp::Action act = winhint::ltp::dispatchAction(
            drain, L.empty(tid), mem, L.parkedMem(tid) > 0, urgent,
            inst->isNop(), L.empty(tid) && ltpRoom(tid));
        if (act == winhint::ltp::ActDrainStall) {
            DPRINTF(LTP, "[tid:%i] [sn:%llu] waits for the LTP to "
                    "drain\n", tid, inst->seqNum);
            L.draining[tid] = true;
            ++L.stats.drainStalls;
            block(tid);
            toRename->iewUnblock[tid] = false;
            break;
        }
        const bool park = act == winhint::ltp::ActPark ||
            act == winhint::ltp::ActParkUrgent;
        const bool park_urgent = act == winhint::ltp::ActParkUrgent;
        const bool bypass = act == winhint::ltp::ActBypass;

        if (park) {
            if (L.full(tid)) {
                ++L.stats.fullStalls;
                block(tid);
                toRename->iewUnblock[tid] = false;
                break;
            }
            L.park(inst, park_urgent, now);
            // Consumers that bypass it must wait for it (invariant 3).
            instQueue.recordProducer(inst);
            insts_to_dispatch.pop();
            if (inst->isLoad())
                toRename->iewInfo[tid].dispatchedToLQ++;
            if (inst->isStore())
                toRename->iewInfo[tid].dispatchedToSQ++;
            toRename->iewInfo[tid].dispatched++;
            ++iewStats.dispatchedInsts;
            continue;
        }

        // Direct dispatch: urgent, bypassing, nop, or drain-class with an
        // empty LTP.
        if (ltpIqSlots[tid] == 0)
            break;  // IQ write ports used by releases: bandwidth full
        const unsigned need =
            winhint::ltp::dispatchNeed(L.empty(tid), L.params().iqReserve);
        if (instQueue.numFreeEntries(inst) < need) {
            DPRINTF(IEW, "[tid:%i] Issue: IQ has become full (LTP "
                    "reserve).\n", tid);
            block(tid);
            toRename->iewUnblock[tid] = false;
            ++iewStats.iqFullEvents;
            ++L.stats.reserveStalls;
            break;
        }
        if ((inst->isAtomic() && ldstQueue.sqFull(tid)) ||
            (inst->isLoad() && ldstQueue.lqFull(tid)) ||
            (inst->isStore() && ldstQueue.sqFull(tid))) {
            block(tid);
            toRename->iewUnblock[tid] = false;
            ++iewStats.lsqFullEvents;
            ++L.stats.reserveStalls;
            break;
        }

        if (bypass)
            ++L.stats.bypassed;
        ltpToQueues(inst, tid, false);
        --ltpIqSlots[tid];

        insts_to_dispatch.pop();
        if (inst->isLoad())
            toRename->iewInfo[tid].dispatchedToLQ++;
        if (inst->isStore() || inst->isAtomic())
            toRename->iewInfo[tid].dispatchedToSQ++;
        toRename->iewInfo[tid].dispatched++;
        ++iewStats.dispatchedInsts;
    }

    if (!insts_to_dispatch.empty()) {
        DPRINTF(IEW,"[tid:%i] Issue: Bandwidth Full. Blocking.\n", tid);
        block(tid);
        toRename->iewUnblock[tid] = false;
    }

    if (dispatchStatus[tid] == Idle && dis_num_inst) {
        dispatchStatus[tid] = Running;
        updatedQueues = true;
    }
}

} // namespace o3
} // namespace gem5
