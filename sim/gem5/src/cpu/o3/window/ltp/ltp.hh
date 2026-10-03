/**
 * @file
 * @brief WinHint B9: Long-Term Parking (LTP) for the gem5 O3 CPU: tunables
 *        (LtpParams) and the parking state (LongTermParking).
 *
 * Reimplementation of Sembrant, Carlson, Hagersten, Black-Schaffer, Perais,
 * Seznec, Michaud, "Long Term Parking (LTP): Criticality-aware Resource
 * Allocation in OOO Processors", MICRO-48, 2015.
 *
 * Instructions are classified as urgent / non-urgent in program order at
 * dispatch with an Urgent Instruction Table (UIT, PC-indexed) trained by
 * Iterative Backward Dependency Analysis (IBDA): long-latency loads (and
 * mispredicted branches) are seeds at commit; an urgent instruction marks
 * the last producers of its source registers (Register Dependence Table,
 * arch reg -> producer PC) urgent, so the backward slice is learned over
 * successive iterations. Non-urgent instructions are parked in a FIFO
 * between rename and the IQ/LSQ: they hold a ROB entry (in order, like
 * every instruction) but no IQ or LQ/SQ entry until they are released.
 *
 * The pipeline side lives in IEW (ltp_iew.cc); this class holds the state,
 * the classification and the statistics. See README.md for the design and
 * the deviations from the paper.
 */

#ifndef __CPU_O3_WINDOW_LTP_LTP_HH__
#define __CPU_O3_WINDOW_LTP_LTP_HH__

#include <cstdint>
#include <list>
#include <string>
#include <vector>

#include "base/statistics.hh"
#include "base/types.hh"
#include "cpu/inst_seq.hh"
#include "cpu/o3/dyn_inst_ptr.hh"
#include "cpu/o3/limits.hh"
#include "cpu/o3/window/ltp/ltp_core.hh"
#include "cpu/reg_class.hh"

namespace gem5
{

namespace o3
{

class CPU;
class ROB;
class WindowArgs;

/** Tunables, read from window_args (docs: docs/guide/baselines/ltp.md). */
struct LtpParams
{
    unsigned entries = 128;     //!< parking-queue entries per thread
    unsigned uitEntries = 256;  //!< UIT entries
    unsigned uitAssoc = 4;      //!< UIT associativity (LRU)
    unsigned lllCycles = 30;    //!< load-to-use latency of a long-latency load
    unsigned wakeDist = 16;     //!< release when within this many insts of
                                //!< the ROB head
    unsigned iqReserve = 4;     //!< IQ entries urgent dispatch leaves free
                                //!< for released instructions
    double roomFrac = 0.5;      //!< IQ "has room": free > roomFrac * cap
    unsigned branchSeed = 1;    //!< 0: none, 1: mispredicted, 2: all branches
    unsigned structs = 0xe;     //!< WindowStruct mask resized by the policy
                                //!< (default IQ+LQ+SQ; ROB stays largest)

    /**
     * @brief Parse (and mark as used) the ltp keys of window_args.
     *
     * Keys: entries, uit_entries, uit_assoc, lll, wake, reserve, room,
     * branch_seed, structs (rob/iq/lq/sq/lsq/all joined by '+' or ':',
     * parsed by parseWindowStructs() as for the hint policy; default
     * "iq+lsq", an empty value means all). fatal() on entries < 1, uit_entries not a positive
     * multiple of uit_assoc, reserve < 1, room < 0 or branch_seed > 2.
     *
     * @param args Parsed window_args.
     * @return The tunables (defaults for absent keys).
     */
    static LtpParams fromArgs(const WindowArgs &args);
};

/**
 * LTP state of one O3 CPU: per-thread parking FIFOs, the UIT/RDT urgency
 * table and the statistics group "ltp" (child of the CPU). Created
 * by IEW::ltpInit() when window_policy=ltp; driven by the IEW functions in
 * ltp_iew.cc and by commit through ltp_hooks.hh.
 */
class LongTermParking
{
  public:
    /** Why a parked instruction left the LTP (ltp_core.hh). */
    using Reason = winhint::ltp::Reason;
    /** @name Release reasons (aliases of winhint::ltp::Reason) */
    /** @{ */
    /** Within wakeDist of the ROB head. */
    static constexpr Reason RelOld = winhint::ltp::RelOld;
    /** Drain-class instruction waiting. */
    static constexpr Reason RelDrain = winhint::ltp::RelDrain;
    /** Became urgent. */
    static constexpr Reason RelUrgent = winhint::ltp::RelUrgent;
    /** Older than a parked urgent mem op. */
    static constexpr Reason RelMemOrder = winhint::ltp::RelMemOrder;
    /** LTP full: head leaves. */
    static constexpr Reason RelFull = winhint::ltp::RelFull;
    /** The IQ has room. */
    static constexpr Reason RelRoom = winhint::ltp::RelRoom;
    /** Not eligible / count. */
    static constexpr Reason NumReasons = winhint::ltp::NumReasons;
    /** @} */

    /** One parked instruction. */
    struct Entry
    {
        DynInstPtr inst;  //!< the parked instruction
        bool urgent;   //!< urgent but parked to keep LSQ order
        bool mem;      //!< memory reference (in-order LSQ allocation)
        Cycles parkedAt;  //!< cycle it was parked
    };
    /** Per-thread parking FIFO, oldest first. */
    using Queue = std::list<Entry>;

    /**
     * @brief Build the queues, the urgency table and the statistics.
     * @param cpu The owning CPU (also the statistics parent).
     * @param p Tunables.
     * @param num_threads Number of hardware threads.
     */
    LongTermParking(CPU *cpu, const LtpParams &p, ThreadID num_threads);

    /** @return The tunables. */
    const LtpParams &params() const { return p; }

    /** @param r The ROB, whose head is the wake-up reference (isOld()). */
    void setROB(ROB *r) { rob = r; }

    // ---- classification (IBDA) ------------------------------------------
    /**
     * @brief Urgency of an instruction about to be dispatched; trains the
     *        backward slice. Call once per instruction, in program order.
     *
     * Only source/destination registers that are not always ready are
     * used as RDT keys.
     *
     * @param inst Instruction (renamed).
     * @return Whether its PC hits in the UIT.
     */
    bool classify(const DynInstPtr &inst);
    /**
     * @brief UIT lookup (no LRU update when !touch).
     * @param pc Instruction address.
     * @param touch Update the LRU state on a hit.
     * @return Whether pc is in the UIT.
     */
    bool uitHit(Addr pc, bool touch);
    /**
     * @brief Training at commit: long-latency loads and branches are seeds.
     *
     * A load is a seed if the ticks from its first issue to the last
     * wake-up of its dependents are >= lllCycles cycles; a control
     * instruction is a seed per branchSeed (1: mispredicted, 2: all).
     *
     * @param inst Committed instruction.
     */
    void commitInst(const DynInstPtr &inst);

    /**
     * @brief Instructions that must not bypass or be parked: they dispatch
     *        only when the LTP of their thread is empty.
     *
     * Non-speculative, serializing, squash-after, store-conditional,
     * atomic, barrier and HTM instructions.
     *
     * @param inst Instruction.
     * @return Whether inst is drain-class.
     */
    static bool mustDrain(const DynInstPtr &inst);

    // ---- parking queue --------------------------------------------------
    /** @param tid Thread. @return Its parking FIFO. */
    Queue &queue(ThreadID tid) { return q[tid]; }
    /** @param tid Thread. @return Whether nothing is parked. */
    bool empty(ThreadID tid) const { return q[tid].empty(); }
    /** @return Whether every thread's FIFO is empty. */
    bool allEmpty() const;
    /** @param tid Thread. @return Whether the FIFO holds p.entries. */
    bool full(ThreadID tid) const { return q[tid].size() >= p.entries; }
    /** @param tid Thread. @return Parked memory references. */
    unsigned parkedMem(ThreadID tid) const { return nMem[tid]; }

    /**
     * @brief Append inst to its thread's FIFO (must not be full; seqNum
     *        must be younger than the tail).
     * @param inst Instruction to park.
     * @param urgent Urgent memory instruction parked for LSQ order (must
     *        be a memory reference); recorded in urgentMemSn.
     * @param now Current cycle.
     */
    void park(const DynInstPtr &inst, bool urgent, Cycles now);
    /**
     * @brief Remove *it after a release; returns the next iterator.
     * @param tid Thread.
     * @param it Entry to remove.
     * @param why Release reason (statistics).
     * @param now Current cycle (parked latency).
     * @return Iterator following it.
     */
    Queue::iterator release(ThreadID tid, Queue::iterator it, Reason why,
                            Cycles now);
    /**
     * @brief Remove *it (squashed while parked).
     * @param tid Thread.
     * @param it Entry to remove.
     * @return Iterator following it.
     */
    Queue::iterator drop(ThreadID tid, Queue::iterator it);
    /**
     * @brief Is sn within wakeDist of the ROB head (or the ROB empty)?
     * @param tid Thread.
     * @param sn Sequence number.
     * @return true if sn <= head seqNum + wakeDist, or no ROB/empty ROB.
     */
    bool isOld(ThreadID tid, InstSeqNum sn) const;

    /**
     * @brief Per-cycle occupancy sampling.
     * @param tid Thread whose FIFO is sampled.
     */
    void sample(ThreadID tid);

    /** Classification cache: an instruction blocked at dispatch is
     * re-examined next cycle; classify it only once. */
    InstSeqNum lastSn[MaxThreads] = {};    ///< last classified seqNum
    bool lastUrgent[MaxThreads] = {};      ///< its classification
    /** A drain-class instruction is waiting for the LTP to empty. */
    bool draining[MaxThreads] = {};
    /** Youngest parked urgent memory instruction (0: none). */
    InstSeqNum urgentMemSn[MaxThreads] = {};

    /** LTP statistics (group "ltp"; descriptions in ltp.cc). */
    struct LtpStats : public statistics::Group
    {
        /** @param parent Parent stats group (the CPU). */
        LtpStats(statistics::Group *parent);

        statistics::Scalar urgent;  ///< classified urgent at dispatch
        statistics::Scalar nonUrgent;  ///< classified non-urgent
        statistics::Scalar parked;  ///< park events
        /** Urgent mem ops parked for LSQ order. */
        statistics::Scalar parkedUrgentMem;
        statistics::Scalar bypassed;  ///< non-urgent dispatched directly
        statistics::Scalar released;  ///< release events
        statistics::Vector releasedBy;  ///< releases per Reason
        statistics::Scalar squashed;  ///< squashed while parked
        statistics::Scalar fullStalls;  ///< dispatch stalls: LTP full
        /** Stalls: IQ reserve / full IQ or LSQ. */
        statistics::Scalar reserveStalls;
        /** Stalls: waiting for the LTP to drain. */
        statistics::Scalar drainStalls;
        statistics::Scalar seedsLoad;  ///< UIT seeds: long-latency loads
        statistics::Scalar seedsBranch;  ///< UIT seeds: branches
        statistics::Scalar uitInserts;  ///< UIT insertions
        statistics::Scalar uitEvictions;  ///< UIT evictions
        statistics::Scalar occSum;  ///< sum of per-cycle occupancy
        statistics::Scalar cycles;  ///< cycles sampled
        statistics::Scalar parkedCycles;  ///< cycles with >= 1 parked
        statistics::Scalar occMax;  ///< maximum occupancy
        statistics::Scalar parkedLatency;  ///< total parked cycles (released)
        statistics::Formula occMean;  ///< occSum / cycles
        statistics::Formula meanParkCycles;  ///< parkedLatency / released
        statistics::Distribution occDist;  ///< occupancy histogram
    } stats;  ///< statistics group "ltp"

  private:
    /**
     * @brief Insert pc into the UIT and update the statistics.
     * @param pc Instruction address.
     */
    void uitInsert(Addr pc);
    /**
     * @brief RDT key of an architectural register.
     * @param r Register id.
     * @return (class << 16) | index.
     */
    static uint32_t regKey(const RegId &r);

    CPU *cpu;               ///< owning CPU
    ROB *rob = nullptr;     ///< ROB (set by commit, see ltp_hooks.hh)
    LtpParams p;            ///< tunables
    ThreadID numThreads;    ///< hardware threads

    Queue q[MaxThreads];             ///< parking FIFOs
    unsigned nMem[MaxThreads] = {};  ///< parked memory refs per thread

    // UIT + RDT (IBDA), gem5-independent (ltp_core.hh).
    winhint::ltp::UrgencyTable urgency;  ///< UIT + RDT
    /** UIT counters already moved into the statistics. */
    uint64_t uitInsertsSeen = 0, uitEvictionsSeen = 0;
    std::vector<uint32_t> srcKeys, dstKeys;   //!< classify() scratch

    /** Move new UIT insert/evict counts into the statistics. */
    void syncUitStats();
};

} // namespace o3
} // namespace gem5

#endif // __CPU_O3_WINDOW_LTP_LTP_HH__
