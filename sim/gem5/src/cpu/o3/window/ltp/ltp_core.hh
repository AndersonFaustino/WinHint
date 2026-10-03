/**
 * @file
 * @brief gem5-independent decision logic of B9 Long-Term Parking
 *        (namespace winhint::ltp).
 *
 * ltp_core.hh - the gem5-independent decision logic of B9 Long-Term
 * Parking (Sembrant et al., MICRO-48 2015), shared by the gem5 side
 * (ltp.cc, ltp_iew.cc) and the host unit tests (tests/test_ltp_core.cc).
 *
 * Header-only, C++17, no gem5 dependency (same convention as
 * ../window_lut.hh):
 *
 *  - UrgencyTable: the Urgent Instruction Table (UIT, set-associative,
 *    full-PC tags, LRU) and the per-thread Register Dependence Table (RDT,
 *    architectural register -> PC of its last producer) that together
 *    implement Iterative Backward Dependency Analysis (IBDA): seeds
 *    (long-latency loads, branches) are inserted at commit; an urgent
 *    instruction inserts the producers of its sources at dispatch, so one
 *    more link of the backward slice is learnt per dynamic visit.
 *  - dispatchAction(): what dispatch does with a new instruction.
 *  - releaseReason(): whether, and why, a parked instruction may leave.
 *  - releaseNeed(): IQ entries a release / dispatch needs (deadlock
 *    freedom through the IQ reserve).
 *
 * See README.md for the design and the deviations from the paper.
 */

#ifndef __CPU_O3_WINDOW_LTP_LTP_CORE_HH__
#define __CPU_O3_WINDOW_LTP_LTP_CORE_HH__

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace winhint
{

namespace ltp
{

/** Why a parked instruction left the LTP (priority order). */
enum Reason
{
    RelOld,      //!< within wakeDist of the ROB head
    RelDrain,    //!< a serializing/non-speculative inst waits behind it
    RelUrgent,   //!< became urgent (UIT hit) or parked urgent (mem order)
    RelMemOrder, //!< memory op older than a parked urgent memory op
    RelFull,     //!< LTP full: head leaves
    RelRoom,     //!< the IQ has room
    NumReasons   //!< not eligible
};

/** Facts about one parked instruction, evaluated by the pipeline. */
struct ParkedView
{
    bool old = false;          //!< within wakeDist of the ROB head
    bool draining = false;     //!< a drain-class inst waits for the LTP
    bool urgent = false;       //!< parked urgent, or its PC is now in the UIT
    bool olderThanUrgentMem = false; //!< mem op older than a parked
                                     //!< urgent mem op
    bool fullHead = false;     //!< LTP full and this is its head
    bool room = false;         //!< the IQ has room
};

/**
 * @brief Release rule (README.md, "Release"): first matching reason, or
 *        NumReasons if the instruction stays parked.
 * @param v Facts about the parked instruction.
 * @return Release reason in Reason priority order, or NumReasons.
 */
inline Reason
releaseReason(const ParkedView &v)
{
    if (v.old)
        return RelOld;
    if (v.draining)
        return RelDrain;
    if (v.urgent)
        return RelUrgent;
    if (v.olderThanUrgentMem)
        return RelMemOrder;
    if (v.fullHead)
        return RelFull;
    if (v.room)
        return RelRoom;
    return NumReasons;
}

/**
 * @brief Free IQ entries needed to release a parked instruction: only an
 *        `old` release of the oldest parked instruction may use the
 *        reserve.
 * @param why Release reason.
 * @param atHead Whether it is the oldest live parked instruction.
 * @param reserve IQ reserve (LtpParams::iqReserve).
 * @return 1, or 1 + reserve.
 */
inline unsigned
releaseNeed(Reason why, bool atHead, unsigned reserve)
{
    return (why == RelOld && atHead) ? 1 : 1 + reserve;
}

/**
 * @brief Free IQ entries needed to dispatch directly (urgent, bypass, nop,
 *        drain-class): the reserve is kept while anything is parked.
 * @param ltpEmpty Whether the thread's LTP is empty.
 * @param reserve IQ reserve (LtpParams::iqReserve).
 * @return 1, or 1 + reserve.
 */
inline unsigned
dispatchNeed(bool ltpEmpty, unsigned reserve)
{
    return ltpEmpty ? 1 : 1 + reserve;
}

/** What dispatch does with a new (unsquashed) instruction. */
enum Action
{
    ActDirect,     //!< to the IQ/LSQ now (urgent, nop, drain-class + empty)
    ActBypass,     //!< non-urgent, but the LTP is empty and the IQ has room
    ActPark,       //!< into the parking queue
    ActParkUrgent, //!< urgent memory op parked behind an older parked one
    ActDrainStall, //!< drain-class: stall until the LTP is empty
};

/**
 * @brief Dispatch rule for one instruction.
 *
 * Drain-class instructions go direct only into an empty LTP; a memory op
 * behind a parked memory op is parked (urgent or not) to keep LSQ order;
 * urgent and nop instructions go direct; others bypass when the LTP is
 * empty and the IQ has room, else they are parked.
 *
 * @param drain Drain-class instruction (LongTermParking::mustDrain()).
 * @param ltpEmpty Whether the thread's LTP is empty.
 * @param mem Whether it is a memory reference.
 * @param memParked Whether a memory reference is parked.
 * @param urgent Whether it was classified urgent.
 * @param nop Whether it is a nop.
 * @param room Whether the IQ has room.
 * @return The action.
 */
inline Action
dispatchAction(bool drain, bool ltpEmpty, bool mem, bool memParked,
               bool urgent, bool nop, bool room)
{
    if (drain)
        return ltpEmpty ? ActDirect : ActDrainStall;
    if (mem && memParked)      // keep LSQ program order
        return urgent ? ActParkUrgent : ActPark;
    if (urgent || nop)
        return ActDirect;
    return (ltpEmpty && room) ? ActBypass : ActPark;
}

/** UIT + RDT (IBDA). Addresses and register keys are plain integers. */
class UrgencyTable
{
  public:
    /**
     * @param entries UIT entries (a multiple of assoc; not checked here).
     * @param assoc UIT associativity.
     * @param threads Number of per-thread RDTs.
     */
    UrgencyTable(unsigned entries, unsigned assoc, unsigned threads = 1)
        : assoc(assoc), sets(entries / assoc), uit(entries), rdt(threads)
    {}

    /** @return Number of UIT sets (entries / assoc). */
    unsigned numSets() const { return sets; }

    /**
     * @brief UIT lookup (LRU update only when touch).
     * @param pc Instruction address (full tag).
     * @param touch Update the LRU state on a hit.
     * @return Whether pc is present.
     */
    bool
    hit(uint64_t pc, bool touch)
    {
        UitWay *w = find(pc);
        if (w && touch)
            w->lru = ++clock;
        return w != nullptr;
    }

    /**
     * @brief Insert pc (no-op apart from LRU if already present).
     *
     * Fills an invalid way if any, else evicts the LRU way of the set.
     *
     * @param pc Instruction address.
     */
    void
    insert(uint64_t pc)
    {
        if (hit(pc, true))
            return;
        UitWay *victim = nullptr;
        for (unsigned w = 0; w < assoc; w++) {
            UitWay &way = uit[setOf(pc) * assoc + w];
            if (!way.valid) {
                victim = &way;
                break;
            }
            if (!victim || way.lru < victim->lru)
                victim = &way;
        }
        if (victim->valid)
            evictions++;
        victim->valid = true;
        victim->tag = pc;
        victim->lru = ++clock;
        inserts++;
    }

    /**
     * @brief Dispatch-time classification, in program order per thread.
     *
     * srcs/dsts: architectural register keys of the instruction (only the
     * ones that carry a dependence). Returns whether pc is urgent; if so,
     * the last producers of its sources become urgent (IBDA). In any case
     * pc becomes the last producer of every dst.
     *
     * @param tid Thread (RDT index).
     * @param pc Instruction address.
     * @param srcs Source register keys.
     * @param nsrc Number of srcs.
     * @param dsts Destination register keys.
     * @param ndst Number of dsts.
     * @return Whether pc hits in the UIT.
     */
    bool
    classify(unsigned tid, uint64_t pc, const uint32_t *srcs, int nsrc,
             const uint32_t *dsts, int ndst)
    {
        auto &table = rdt[tid];
        const bool urgent = hit(pc, true);
        if (urgent) {
            for (int i = 0; i < nsrc; i++) {
                auto it = table.find(srcs[i]);
                if (it != table.end())
                    insert(it->second);
            }
        }
        for (int i = 0; i < ndst; i++)
            table[dsts[i]] = pc;
        return urgent;
    }

    uint64_t inserts = 0;      //!< UIT insertions (seeds + propagation)
    uint64_t evictions = 0;    //!< UIT evictions

  private:
    /** One UIT way. */
    struct UitWay
    {
        uint64_t tag = 0;      //!< full PC
        bool valid = false;    //!< way holds a PC
        uint64_t lru = 0;      //!< last-use timestamp
    };

    /** @param pc Address. @return UIT set index ((pc >> 1) % sets). */
    unsigned setOf(uint64_t pc) const { return (unsigned)((pc >> 1) % sets); }

    /**
     * @param pc Address.
     * @return The valid way tagged pc, or nullptr.
     */
    UitWay *
    find(uint64_t pc)
    {
        for (unsigned w = 0; w < assoc; w++) {
            UitWay &way = uit[setOf(pc) * assoc + w];
            if (way.valid && way.tag == pc)
                return &way;
        }
        return nullptr;
    }

    unsigned assoc, sets;      ///< UIT geometry
    std::vector<UitWay> uit;   ///< sets * assoc ways, set-major
    uint64_t clock = 0;        ///< LRU clock
    /** Per-thread RDT: register key -> PC of its last producer. */
    std::vector<std::unordered_map<uint32_t, uint64_t>> rdt;
};

} // namespace ltp
} // namespace winhint

#endif // __CPU_O3_WINDOW_LTP_LTP_CORE_HH__
