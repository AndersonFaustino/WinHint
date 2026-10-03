/**
 * @file
 * @brief WinHint: the O3 window-resizing mechanism and its instrumentation
 *        (class WindowController).
 *
 * The WindowController
 *  - applies a configuration (ROB/IQ/LQ/SQ caps) to the pipeline. The
 *    structures keep their physical (largest) size and report
 *    free = max(0, min(phys_free, cap - occupancy)), so rename/IEW stall
 *    dispatch until occupancy drains to a smaller cap (no flush, cap >= 1);
 *  - samples per-cycle occupancy, outstanding L1D misses (MLP) and
 *    per-structure "full under the cap" cycles, and builds a per-period
 *    WindowSample of deltas (policy.hh);
 *  - decodes committed RISC-V hint instructions (docs/interfaces.md §2);
 *  - records region(id) visits in region_stats.csv under every policy;
 *  - writes window_trace.csv and the system.cpu.window.* statistics;
 *  - delegates every decision to a WindowPolicy found by name in the
 *    WindowPolicyRegistry (window_policy param).
 *
 * All state is in members (no function-local statics).
 */

#ifndef __CPU_O3_WINDOW_CONTROLLER_HH__
#define __CPU_O3_WINDOW_CONTROLLER_HH__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "base/statistics.hh"
#include "base/types.hh"
#include "cpu/o3/dyn_inst_ptr.hh"
#include "cpu/o3/window/policy.hh"

namespace gem5
{

class BaseCache;
struct BaseO3CPUParams;
class OutputStream;

namespace o3
{

class CPU;

/**
 * Per-CPU window controller: owns the caps, sampling, hint decoding, output
 * files and statistics, and delegates decisions to a WindowPolicy. One
 * instance per O3 CPU, driven by CPU::startup(), CPU::tick() and
 * CPU::instDone().
 */
class WindowController
{
  public:
    /** Hint encoding (docs/interfaces.md §2): IMM[4:0] tag of setwin. */
    static constexpr unsigned TagSetwin = 0x15;  // 0b10101
    /** Hint encoding (docs/interfaces.md §2): IMM[4:0] tag of region. */
    static constexpr unsigned TagRegion = 0x17;  // 0b10111

    /**
     * @brief Decode a 32-bit RISC-V instruction word.
     *
     * Returns true for `ori x0, x0, IMM` with a WinHint tag and
     * IMM[11] = 0; sets kind (1 = setwin, 2 = region) and payload
     * (IMM[10:5]). Only the low 32 bits of word are looked at. kind and
     * payload are left untouched when it returns false.
     *
     * @param word Instruction word (machine code).
     * @param[out] kind 1 = setwin, 2 = region.
     * @param[out] payload 6-bit payload (setwin: W / 8; region: id).
     * @return Whether word is a WinHint hint.
     */
    static bool decodeHint(uint64_t word, unsigned &kind, unsigned &payload);

    /**
     * @brief Build the configuration table from the window_* params,
     *        validate it and instantiate the policy.
     *
     * Without window_rob the table has a single configuration equal to the
     * physical sizes. fatal() on mismatched vector lengths, zero-sized or
     * over-physical entries, a non-ascending ROB column, an out-of-range
     * window_initial, window_period = 0, an unknown policy or unused
     * window_args keys. Registers finish() as an exit callback.
     *
     * @param cpu The owning O3 CPU (its IEW/IQ must already exist).
     * @param params The CPU params (window_* fields).
     */
    WindowController(CPU *cpu, const BaseO3CPUParams &params);
    /** Destroy the controller and its policy. */
    ~WindowController();

    /**
     * @brief Called once all pipeline structures exist (CPU::startup()).
     *
     * Applies the policy's initial configuration, snapshots the cache miss
     * counters, opens window_trace.csv (if window_trace) and starts the
     * first period.
     */
    void startup();
    /**
     * @brief Called at the end of CPU::tick(), once per simulated CPU cycle.
     *
     * Samples occupancy, full-under-cap cycles and outstanding L1D misses,
     * weighted by the cycles elapsed since the previous call, and ends the
     * period once window_period cycles have passed.
     */
    void tick();
    /**
     * @brief Called from CPU::instDone() for every committed instruction.
     *
     * Counts instructions and branches, feeds onBranchCommit() (delayed by
     * one instruction), and decodes WinHint hints (setwin -> onSetwin(),
     * region -> region bookkeeping + onRegion()). A region marker is
     * counted in the insts of the visit it starts, not the one it ends.
     *
     * @param inst The committed instruction.
     */
    void commitInst(const DynInstPtr &inst);
    /**
     * @brief Flush open regions and the partial period (at exit).
     *
     * Idempotent. A branch still waiting for its successor is delivered
     * to onBranchCommit() with the fall-through PC (taken = false). The
     * partial period is written to the trace but not passed to the
     * policy; then WindowPolicy::finish() is called.
     */
    void finish();

    /** @return For DPRINTF: "<cpu>.window". */
    std::string name() const;

    /** @return Configuration index currently in effect. */
    int currentConfig() const { return cur; }
    /** @return The configuration table. */
    const WindowTable &configTable() const { return table; }
    /** @return The window_policy name. */
    const std::string &policyName() const { return polName; }
    /** @return The active policy (owned by the controller). */
    WindowPolicy *activePolicy() const { return policy.get(); }

  private:
    /**
     * @brief Set the caps of configuration idx on ROB/IQ/LQ/SQ.
     *
     * Structures outside the policy's resizedStructures() mask get the
     * largest configuration's cap. Counts a switch if idx changes after
     * startup.
     *
     * @param idx Configuration index, must be in range.
     * @param why Reason, for DPRINTF only.
     */
    void applyConfig(int idx, const char *why);
    /**
     * @brief Apply a policy decision: ignore Keep/negative, clamp, and
     *        call applyConfig() if it differs from the current one.
     * @param want Requested configuration index or WindowPolicy::Keep.
     * @param why Reason, for DPRINTF only.
     */
    void request(int want, const char *why);
    /**
     * @brief Build the WindowSample of the period that just ended, pass it
     *        to onPeriod() (unless finished), write the trace row and reset
     *        the period accumulators.
     */
    void endPeriod();
    /**
     * @brief Close the current region (if any) and open region id.
     * @param id Region id from the hint payload.
     */
    void enterRegion(int id);
    /**
     * @brief Write the region_stats.csv row of the open region visit; the
     *        reported config is the one with the most cycles in the visit.
     */
    void closeRegion();
    /** @brief Charge the cycles since cfgMark to the current config of the
     *         open region (if any) and move cfgMark to now. */
    void accountConfigCycles();
    /**
     * @brief Deliver the pending onBranchCommit() now that the next
     *        committed PC is known.
     * @param next_pc PC of the instruction committed after the branch.
     */
    void flushPendingBranch(Addr next_pc);
    /**
     * @param c Cache, may be nullptr.
     * @return Cumulative demand misses of c (0 if c is nullptr).
     */
    uint64_t cacheMisses(BaseCache *c) const;

    CPU *cpu;                              ///< owning CPU
    WindowTable table;                     ///< configuration table
    WindowArgs args;                       ///< parsed window_args
    std::unique_ptr<WindowPolicy> policy;  ///< decision policy
    std::string polName;                   ///< window_policy
    BaseCache *l1d;                        ///< window_l1d (may be null)
    BaseCache *l2;                         ///< window_l2 (may be null)
    Cycles period;                         ///< window_period
    bool traceOn;                          ///< window_trace
    unsigned structMask = WinAll;          ///< policy's resized structures

    int cur = 0;                ///< configuration in effect
    bool started = false;       ///< startup() has run
    bool finished = false;      ///< finish() has run

    // Caps currently applied (per structure).
    /** Caps currently applied to ROB, IQ, LQ, SQ (entries). */
    unsigned capRob = 0, capIq = 0, capLq = 0, capSq = 0;

    // Period accumulators (deltas; reset each period).
    Cycles periodStart = Cycles(0);  ///< cycle the current period began
    Cycles lastSample = Cycles(0);   ///< cycle of the previous tick()
    /** Committed instructions, branches and mispredicts in the period. */
    uint64_t pInsts = 0, pBranches = 0, pMispred = 0;
    /** setwin / region hints committed in the period. */
    uint64_t pSetwin = 0, pRegion = 0;
    /** Cycle-weighted occupancy sums of ROB, IQ, LQ, SQ. */
    double pRobOcc = 0, pIqOcc = 0, pLqOcc = 0, pSqOcc = 0;
    /** Full-under-cap cycles of ROB, IQ, LQ, SQ. */
    uint64_t pRobFull = 0, pIqFull = 0, pLqFull = 0, pSqFull = 0;
    uint64_t pAnyFull = 0;  ///< cycles with any of the four full
    /** Cycle-weighted outstanding L1D misses; cycles with >= 1 miss. */
    uint64_t pMlpSum = 0, pMissCycles = 0;
    /** Cumulative cache misses at the previous period end. */
    uint64_t lastL1dMisses = 0, lastL2Misses = 0;

    // Branch bookkeeping (onBranchCommit is delayed by one committed
    // instruction to learn the taken/not-taken target).
    uint64_t instsSinceBranch = 0;  ///< insts since the last control inst
    bool pendingBranch = false;     ///< an onBranchCommit() is pending
    /** PC and fall-through PC of the pending branch. */
    Addr pendPc = 0, pendFallthrough = 0;
    bool pendMispred = false;       ///< pending branch was mispredicted
    uint64_t pendBlockInsts = 0;    ///< its basic-block length

    // Config-cycle bookkeeping (region rows).
    Cycles cfgMark = Cycles(0);     ///< last accountConfigCycles() cycle

    // Region bookkeeping.
    int region = -1;                ///< open region id (-1: none)
    Cycles regionEnter = Cycles(0); ///< cycle the open visit began
    uint64_t regionInsts = 0;       ///< insts committed in the visit
                                    ///< (including its region marker)
    /** Cycles of the open visit spent in each configuration. */
    std::vector<uint64_t> regionCfgCycles;

    OutputStream *traceOut = nullptr;   ///< window_trace.csv (or null)
    OutputStream *regionOut = nullptr;  ///< region_stats.csv (lazy)

    /** The system.cpu.window.* statistics (descriptions in controller.cc;
     *  per-config vectors are indexed by configuration). */
    struct WindowStats : public statistics::Group
    {
        /**
         * @brief Register the system.cpu.window.* statistics.
         * @param parent Parent stats group (the CPU).
         * @param n_configs Number of configurations (per-config vectors).
         * @param max_rob Physical ROB size (occupancy histogram range).
         * @param max_iq Physical IQ size.
         * @param max_lq Physical LQ size.
         * @param max_sq Physical SQ size.
         */
        WindowStats(statistics::Group *parent, int n_configs,
                    unsigned max_rob, unsigned max_iq, unsigned max_lq,
                    unsigned max_sq);

        statistics::Scalar switches;        ///< configuration changes
        statistics::Scalar periods;         ///< sampling periods
        statistics::Scalar hints;           ///< hints committed (both)
        statistics::Scalar setwinHints;     ///< setwin hints committed
        statistics::Scalar regionHints;     ///< region hints committed
        statistics::Scalar drainCycles;     ///< cycles above some cap
        statistics::Vector cyclesInConfig;  ///< cycles per config
        statistics::Vector robOccSum;       ///< ROB occupancy sum/config
        statistics::Vector iqOccSum;        ///< IQ occupancy sum/config
        statistics::Vector lqOccSum;        ///< LQ occupancy sum/config
        statistics::Vector sqOccSum;        ///< SQ occupancy sum/config
        statistics::Formula robOccMean;     ///< robOccSum / cyclesInConfig
        statistics::Formula iqOccMean;      ///< iqOccSum / cyclesInConfig
        statistics::Formula lqOccMean;      ///< lqOccSum / cyclesInConfig
        statistics::Formula sqOccMean;      ///< sqOccSum / cyclesInConfig
        statistics::Vector robOccMax;       ///< max ROB occ once drained
        statistics::Vector iqOccMax;        ///< max IQ occ once drained
        statistics::Vector lqOccMax;        ///< max LQ occ once drained
        statistics::Vector sqOccMax;        ///< max SQ occ once drained
        statistics::Vector robFullCycles;   ///< ROB full under cap/config
        statistics::Vector iqFullCycles;    ///< IQ full under cap/config
        statistics::Vector lqFullCycles;    ///< LQ full under cap/config
        statistics::Vector sqFullCycles;    ///< SQ full under cap/config
        statistics::Distribution robOccDist;  ///< ROB occupancy histogram
        statistics::Distribution iqOccDist;   ///< IQ occupancy histogram
        statistics::Distribution lqOccDist;   ///< LQ occupancy histogram
        statistics::Distribution sqOccDist;   ///< SQ occupancy histogram
        /** Outstanding L1D misses histogram (0..63). */
        statistics::Distribution l1dOutstandingDist;
    } stats;  ///< statistics group "window"
};

} // namespace o3
} // namespace gem5

#endif // __CPU_O3_WINDOW_CONTROLLER_HH__
