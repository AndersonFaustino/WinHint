/**
 * @file
 * @brief WinHint: the O3 window-resizing mechanism (see controller.hh).
 *
 * Implementation of WindowController: hint decoding, configuration
 * application, per-cycle sampling, per-period WindowSample construction,
 * window_trace.csv / region_stats.csv output and the window statistics.
 */

#include "cpu/o3/window/controller.hh"

#include <algorithm>
#include <iomanip>

#include "base/logging.hh"
#include "base/output.hh"
#include "cpu/o3/cpu.hh"
#include "cpu/o3/dyn_inst.hh"
#include "debug/WinHint.hh"
#include "mem/cache/base.hh"
#include "params/BaseO3CPU.hh"
#include "sim/sim_exit.hh"

namespace gem5
{

namespace o3
{

namespace
{

/**
 * @brief Maximum of v, or dflt if v is empty.
 * @param v Values.
 * @param dflt Result for an empty vector.
 * @return The maximum element or dflt.
 */
unsigned
maxOf(const std::vector<unsigned> &v, unsigned dflt)
{
    return v.empty() ? dflt : *std::max_element(v.begin(), v.end());
}

/**
 * @brief Size a per-configuration stats vector and name its entries
 *        "0".."n-1".
 * @param v Vector to initialise.
 * @param n Number of configurations.
 */
void
initPerConfig(statistics::Vector &v, int n)
{
    v.init(n);
    for (int i = 0; i < n; i++)
        v.subname(i, std::to_string(i));
}

/**
 * @brief Initialise an occupancy distribution over [0, max] with about 32
 *        buckets (bucket size >= 1).
 * @param d Distribution to initialise.
 * @param max Upper bound (raised to 1 if 0).
 */
void
initDist(statistics::Distribution &d, unsigned max)
{
    max = std::max(1u, max);
    unsigned bucket = std::max(1u, (max + 31) / 32);
    d.init(0, max, bucket);
}

} // anonymous namespace

// See controller.hh.
bool
WindowController::decodeHint(uint64_t word, unsigned &kind,
                             unsigned &payload)
{
    // ori x0, x0, IMM: opcode 0x13, rd 0, funct3 6, rs1 0 -> low 20 bits
    // are 0x06013. Compressed encodings never match (bits[1:0] != 0b11).
    word &= 0xffffffffULL;
    if ((word & 0x000fffffULL) != 0x00006013ULL)
        return false;
    unsigned imm = (word >> 20) & 0xfff;
    if (imm & 0x800)
        return false;               // payload is 0..63, IMM[11] = 0
    unsigned tag = imm & 0x1f;
    if (tag == TagSetwin)
        kind = 1;
    else if (tag == TagRegion)
        kind = 2;
    else
        return false;
    payload = (imm >> 5) & 0x3f;
    return true;
}

// See controller.hh.
WindowController::WindowStats::WindowStats(statistics::Group *parent, int n,
                                           unsigned max_rob, unsigned max_iq,
                                           unsigned max_lq, unsigned max_sq)
    : statistics::Group(parent, "window"),
      ADD_STAT(switches, statistics::units::Count::get(),
               "Number of window configuration changes"),
      ADD_STAT(periods, statistics::units::Count::get(),
               "Number of sampling periods"),
      ADD_STAT(hints, statistics::units::Count::get(),
               "WinHint hint instructions committed (setwin + region)"),
      ADD_STAT(setwinHints, statistics::units::Count::get(),
               "setwin hints committed"),
      ADD_STAT(regionHints, statistics::units::Count::get(),
               "region hints committed"),
      ADD_STAT(drainCycles, statistics::units::Cycle::get(),
               "Cycles with some structure above its cap (draining after "
               "a shrink)"),
      ADD_STAT(cyclesInConfig, statistics::units::Cycle::get(),
               "Cycles spent in each window configuration"),
      ADD_STAT(robOccSum, statistics::units::Count::get(),
               "Sum over cycles of ROB occupancy, per configuration"),
      ADD_STAT(iqOccSum, statistics::units::Count::get(),
               "Sum over cycles of IQ occupancy, per configuration"),
      ADD_STAT(lqOccSum, statistics::units::Count::get(),
               "Sum over cycles of LQ occupancy, per configuration"),
      ADD_STAT(sqOccSum, statistics::units::Count::get(),
               "Sum over cycles of SQ occupancy, per configuration"),
      ADD_STAT(robOccMean, statistics::units::Count::get(),
               "Mean ROB occupancy, per configuration"),
      ADD_STAT(iqOccMean, statistics::units::Count::get(),
               "Mean IQ occupancy, per configuration"),
      ADD_STAT(lqOccMean, statistics::units::Count::get(),
               "Mean LQ occupancy, per configuration"),
      ADD_STAT(sqOccMean, statistics::units::Count::get(),
               "Mean SQ occupancy, per configuration"),
      ADD_STAT(robOccMax, statistics::units::Count::get(),
               "Max ROB occupancy once drained to the cap, per config"),
      ADD_STAT(iqOccMax, statistics::units::Count::get(),
               "Max IQ occupancy once drained to the cap, per config"),
      ADD_STAT(lqOccMax, statistics::units::Count::get(),
               "Max LQ occupancy once drained to the cap, per config"),
      ADD_STAT(sqOccMax, statistics::units::Count::get(),
               "Max SQ occupancy once drained to the cap, per config"),
      ADD_STAT(robFullCycles, statistics::units::Cycle::get(),
               "Cycles with no free ROB entry under the cap, per config"),
      ADD_STAT(iqFullCycles, statistics::units::Cycle::get(),
               "Cycles with no free IQ entry under the cap, per config"),
      ADD_STAT(lqFullCycles, statistics::units::Cycle::get(),
               "Cycles with no free LQ entry under the cap, per config"),
      ADD_STAT(sqFullCycles, statistics::units::Cycle::get(),
               "Cycles with no free SQ entry under the cap, per config"),
      ADD_STAT(robOccDist, statistics::units::Count::get(),
               "ROB occupancy distribution (cycle-weighted)"),
      ADD_STAT(iqOccDist, statistics::units::Count::get(),
               "IQ occupancy distribution (cycle-weighted)"),
      ADD_STAT(lqOccDist, statistics::units::Count::get(),
               "LQ occupancy distribution (cycle-weighted)"),
      ADD_STAT(sqOccDist, statistics::units::Count::get(),
               "SQ occupancy distribution (cycle-weighted)"),
      ADD_STAT(l1dOutstandingDist, statistics::units::Count::get(),
               "Outstanding L1D misses (allocated MSHRs), cycle-weighted")
{
    for (auto *v : {&cyclesInConfig, &robOccSum, &iqOccSum, &lqOccSum,
                    &sqOccSum, &robOccMax, &iqOccMax, &lqOccMax, &sqOccMax,
                    &robFullCycles, &iqFullCycles, &lqFullCycles,
                    &sqFullCycles})
        initPerConfig(*v, n);
    robOccMean = robOccSum / cyclesInConfig;
    iqOccMean = iqOccSum / cyclesInConfig;
    lqOccMean = lqOccSum / cyclesInConfig;
    sqOccMean = sqOccSum / cyclesInConfig;
    for (int i = 0; i < n; i++) {
        std::string s = std::to_string(i);
        robOccMean.subname(i, s);
        iqOccMean.subname(i, s);
        lqOccMean.subname(i, s);
        sqOccMean.subname(i, s);
    }
    initDist(robOccDist, max_rob);
    initDist(iqOccDist, max_iq);
    initDist(lqOccDist, max_lq);
    initDist(sqOccDist, max_sq);
    l1dOutstandingDist.init(0, 63, 1);
}

// See controller.hh.
WindowController::WindowController(CPU *_cpu, const BaseO3CPUParams &p)
    : cpu(_cpu),
      args(p.window_args),
      polName(p.window_policy),
      l1d(p.window_l1d),
      l2(p.window_l2),
      period(p.window_period),
      traceOn(p.window_trace),
      stats(_cpu, std::max<int>(1, (int)p.window_rob.size()),
            p.numROBEntries, _cpu->iew.instQueue.physEntries(),
            p.LQEntries, p.SQEntries)
{
    const unsigned physRob = p.numROBEntries;
    const unsigned physIq = cpu->iew.instQueue.physEntries();
    const unsigned physLq = p.LQEntries, physSq = p.SQEntries;

    if (p.window_rob.empty()) {
        // No table: one configuration equal to the physical sizes.
        fatal_if(!p.window_iq.empty() || !p.window_lq.empty() ||
                 !p.window_sq.empty(),
                 "window_iq/lq/sq given without window_rob");
        table.rob = {physRob};
        table.iq = {physIq};
        table.lq = {physLq};
        table.sq = {physSq};
    } else {
        table.rob = p.window_rob;
        table.iq = p.window_iq;
        table.lq = p.window_lq;
        table.sq = p.window_sq;
    }
    const size_t n = table.rob.size();
    fatal_if(table.iq.size() != n || table.lq.size() != n ||
             table.sq.size() != n,
             "window_rob/iq/lq/sq must have the same length");
    for (size_t i = 0; i < n; i++) {
        fatal_if(!table.rob[i] || !table.iq[i] || !table.lq[i] ||
                 !table.sq[i],
                 "window config %d has a zero-sized structure", i);
        fatal_if(i && table.rob[i] < table.rob[i - 1],
                 "window_rob must be ascending");
        fatal_if(table.rob[i] > physRob || table.iq[i] > physIq ||
                 table.lq[i] > physLq || table.sq[i] > physSq,
                 "window config %d (%d/%d/%d/%d) exceeds the physical "
                 "ROB/IQ/LQ/SQ (%d/%d/%d/%d)", i, table.rob[i], table.iq[i],
                 table.lq[i], table.sq[i], physRob, physIq, physLq, physSq);
    }
    fatal_if(p.window_initial >= n, "window_initial %d out of range (%d "
             "configs)", p.window_initial, n);
    fatal_if(period == 0, "window_period must be > 0");

    WindowPolicyEnv env{table, (int)p.window_initial, args,
                        p.window_lut_file, (uint64_t)period,
                        simout.directory(), cpu};
    policy = WindowPolicyRegistry::create(polName, env);
    args.checkAllUsed(polName);
    structMask = policy->resizedStructures() & WinAll;
    regionCfgCycles.assign(n, 0);

    // Unused when not attached; avoid surprising zero MPKIs silently.
    if (!l1d)
        warn("WinHint: window_l1d not set; L1D misses/MLP read as 0");

    registerExitCallback([this]() { finish(); });
}

WindowController::~WindowController() = default;

// See controller.hh.
std::string
WindowController::name() const
{
    return cpu->name() + ".window";
}

// See controller.hh.
uint64_t
WindowController::cacheMisses(BaseCache *c) const
{
    return c ? (uint64_t)c->windowDemandMisses() : 0;
}

// See controller.hh.
void
WindowController::startup()
{
    cur = table.clamp(policy->initialConfig());
    applyConfig(cur, "initial");
    periodStart = lastSample = cfgMark = cpu->curCycle();
    lastL1dMisses = cacheMisses(l1d);
    lastL2Misses = cacheMisses(l2);
    if (traceOn) {
        traceOut = simout.create("window_trace.csv");
        *traceOut->stream()
            << "cycle,insts,ipc,rob_occ_mean,iq_occ_mean,lq_occ_mean,"
               "l1d_mpki,l2_mpki,mlp,branch_mpki,config,region\n";
    }
    started = true;
    inform("WinHint window: policy=%s configs=%d initial=%d period=%d "
           "structs=%#x", polName, table.size(), cur, (int)period,
           structMask);
}

// See controller.hh.
void
WindowController::applyConfig(int idx, const char *why)
{
    assert(idx >= 0 && idx < table.size());
    if (started && idx != cur) {
        accountConfigCycles();
        stats.switches++;
    }
    const int big = table.largest();
    capRob = table.rob[(structMask & WinROB) ? idx : big];
    capIq = table.iq[(structMask & WinIQ) ? idx : big];
    capLq = table.lq[(structMask & WinLQ) ? idx : big];
    capSq = table.sq[(structMask & WinSQ) ? idx : big];
    DPRINTF(WinHint, "config %d -> %d (%s): ROB %d IQ %d LQ %d SQ %d\n",
            cur, idx, why, capRob, capIq, capLq, capSq);
    cur = idx;
    cpu->rob.setCap(capRob);
    cpu->iew.instQueue.setCap(capIq);
    cpu->iew.ldstQueue.setCaps(capLq, capSq);
    // Make rename see the new free counts without waiting for the next
    // occupancy change.
    cpu->commit.windowCapsChanged();
    cpu->iew.windowCapsChanged();
}

// See controller.hh.
void
WindowController::request(int want, const char *why)
{
    if (want == WindowPolicy::Keep || want < 0)
        return;
    want = table.clamp(want);
    if (want != cur)
        applyConfig(want, why);
}

// See controller.hh.
void
WindowController::accountConfigCycles()
{
    Cycles now = cpu->curCycle();
    if (now > cfgMark && region >= 0)
        regionCfgCycles[cur] += now - cfgMark;
    cfgMark = now;
}

// See controller.hh.
void
WindowController::tick()
{
    if (!started || finished)
        return;
    Cycles now = cpu->curCycle();
    // The O3 CPU stops ticking while idle; weight the sample by the cycles
    // elapsed since the last one (the state did not change meanwhile).
    uint64_t dt = now > lastSample ? (uint64_t)(now - lastSample) : 1;
    lastSample = now;

    const unsigned rob = cpu->rob.occupancy();
    const unsigned iq = cpu->iew.instQueue.occupancy();
    const unsigned lq = (unsigned)cpu->iew.ldstQueue.numLoads();
    const unsigned sq = (unsigned)cpu->iew.ldstQueue.numStores();
    pRobOcc += (double)rob * dt;
    pIqOcc += (double)iq * dt;
    pLqOcc += (double)lq * dt;
    pSqOcc += (double)sq * dt;

    const bool robFull = cpu->rob.numFreeEntries() == 0;
    const bool iqFull = cpu->iew.instQueue.numFreeEntries() == 0;
    const bool lqFull = cpu->iew.ldstQueue.numFreeLoadEntries() == 0;
    const bool sqFull = cpu->iew.ldstQueue.numFreeStoreEntries() == 0;
    if (robFull) {
        pRobFull += dt;
        stats.robFullCycles[cur] += dt;
    }
    if (iqFull) {
        pIqFull += dt;
        stats.iqFullCycles[cur] += dt;
    }
    if (lqFull) {
        pLqFull += dt;
        stats.lqFullCycles[cur] += dt;
    }
    if (sqFull) {
        pSqFull += dt;
        stats.sqFullCycles[cur] += dt;
    }
    if (robFull || iqFull || lqFull || sqFull)
        pAnyFull += dt;

    if (l1d) {
        unsigned out = l1d->windowOutstandingMisses();
        stats.l1dOutstandingDist.sample(out, dt);
        if (out) {
            pMlpSum += (uint64_t)out * dt;
            pMissCycles += dt;
        }
    }

    stats.cyclesInConfig[cur] += dt;
    stats.robOccSum[cur] += (double)rob * dt;
    stats.iqOccSum[cur] += (double)iq * dt;
    stats.lqOccSum[cur] += (double)lq * dt;
    stats.sqOccSum[cur] += (double)sq * dt;
    stats.robOccDist.sample(rob, dt);
    stats.iqOccDist.sample(iq, dt);
    stats.lqOccDist.sample(lq, dt);
    stats.sqOccDist.sample(sq, dt);

    const bool settled = rob <= capRob && iq <= capIq && lq <= capLq &&
                         sq <= capSq;
    if (!settled) {
        stats.drainCycles += dt;
    } else {
        auto upd = [](statistics::Vector &v, int i, unsigned x) {
            if (x > v[i].value())
                v[i] = x;
        };
        upd(stats.robOccMax, cur, rob);
        upd(stats.iqOccMax, cur, iq);
        upd(stats.lqOccMax, cur, lq);
        upd(stats.sqOccMax, cur, sq);
    }

    if (now - periodStart >= period)
        endPeriod();
}

// See controller.hh.
void
WindowController::endPeriod()
{
    Cycles now = cpu->curCycle();
    if (now <= periodStart)
        return;
    const uint64_t cycles = now - periodStart;
    WindowSample s;
    s.cycle = now;
    s.cycles = cycles;
    s.insts = pInsts;
    s.ipc = (double)pInsts / cycles;
    s.robOcc = pRobOcc / cycles;
    s.iqOcc = pIqOcc / cycles;
    s.lqOcc = pLqOcc / cycles;
    s.sqOcc = pSqOcc / cycles;
    // Cumulative cache counters may be reset by m5.stats.reset(): a
    // decrease means "reset", so the delta is the current value.
    const uint64_t l1m = cacheMisses(l1d), l2m = cacheMisses(l2);
    s.l1dMisses = l1m >= lastL1dMisses ? l1m - lastL1dMisses : l1m;
    s.l2Misses = l2m >= lastL2Misses ? l2m - lastL2Misses : l2m;
    lastL1dMisses = l1m;
    lastL2Misses = l2m;
    const double kinsts = pInsts / 1000.0;
    s.l1dMpki = kinsts > 0 ? s.l1dMisses / kinsts : 0;
    s.l2Mpki = kinsts > 0 ? s.l2Misses / kinsts : 0;
    s.mlp = pMissCycles ? (double)pMlpSum / pMissCycles : 0;
    s.mlpAll = (double)pMlpSum / cycles;
    s.missCycles = pMissCycles;
    s.branches = pBranches;
    s.mispredicts = pMispred;
    s.branchMpki = kinsts > 0 ? pMispred / kinsts : 0;
    s.robFullCycles = pRobFull;
    s.iqFullCycles = pIqFull;
    s.lqFullCycles = pLqFull;
    s.sqFullCycles = pSqFull;
    s.anyFullCycles = pAnyFull;
    s.fullFrac = (double)pAnyFull / cycles;
    s.config = cur;
    s.region = region;
    s.setwinHints = pSetwin;
    s.regionHints = pRegion;

    stats.periods++;
    const int want = finished ? WindowPolicy::Keep
                              : policy->onPeriod(s, cur);

    if (traceOut) {
        auto &o = *traceOut->stream();
        o << std::setprecision(6) << s.cycle << ',' << s.insts << ','
          << s.ipc << ',' << s.robOcc << ',' << s.iqOcc << ',' << s.lqOcc
          << ',' << s.l1dMpki << ',' << s.l2Mpki << ',' << s.mlp << ','
          << s.branchMpki << ',' << s.config << ',' << s.region << '\n';
    }

    periodStart = now;
    pInsts = pBranches = pMispred = pSetwin = pRegion = 0;
    pRobOcc = pIqOcc = pLqOcc = pSqOcc = 0;
    pRobFull = pIqFull = pLqFull = pSqFull = pAnyFull = 0;
    pMlpSum = pMissCycles = 0;

    request(want, "period");
}

// See controller.hh.
void
WindowController::flushPendingBranch(Addr next_pc)
{
    if (!pendingBranch)
        return;
    pendingBranch = false;
    policy->onBranchCommit(pendPc, next_pc, next_pc != pendFallthrough,
                           pendMispred, pendBlockInsts);
}

// See controller.hh.
void
WindowController::commitInst(const DynInstPtr &inst)
{
    if (!started || finished)
        return;
    const Addr pc = inst->pcState().instAddr();
    const bool wantBranches = policy->wantsBranchCommits();
    if (wantBranches)
        flushPendingBranch(pc);

    const bool counted = !inst->isMicroop() || inst->isLastMicroop();
    if (counted) {
        pInsts++;
        instsSinceBranch++;
        if (region >= 0)
            regionInsts++;
    }

    if (inst->isControl() && counted) {
        pBranches++;
        const bool mis = inst->mispredicted();
        if (mis)
            pMispred++;
        if (wantBranches) {
            pendingBranch = true;
            pendPc = pc;
            pendFallthrough = pc + inst->staticInst->size();
            pendMispred = mis;
            pendBlockInsts = instsSinceBranch;
        }
        instsSinceBranch = 0;
    }

    // RISC-V HINT: ori x0, x0, imm. Architecturally a no-op on any core
    // (gem5 decodes it as ori_hint); only the controller looks at it, and
    // only at commit (non-speculative).
    unsigned kind = 0, payload = 0;
    if (!decodeHint(inst->staticInst->getEMI(), kind, payload))
        return;
    stats.hints++;
    if (kind == 1) {
        stats.setwinHints++;
        pSetwin++;
        const unsigned w = payload * 8;
        const int want = policy->onSetwin(w, cur);
        DPRINTF(WinHint, "setwin(%d) at %#x -> %d\n", w, pc, want);
        request(want, "setwin");
    } else {
        stats.regionHints++;
        pRegion++;
        DPRINTF(WinHint, "region(%d) at %#x\n", payload, pc);
        // The marker was counted above in the region it ends; it belongs
        // to the visit it starts (region_stats.csv: one row per visit).
        if (counted && region >= 0)
            regionInsts--;
        enterRegion((int)payload);
        if (counted)
            regionInsts++;
        request(policy->onRegion((int)payload, cur), "region");
    }
}

// See controller.hh.
void
WindowController::enterRegion(int id)
{
    closeRegion();
    region = id;
    regionEnter = cpu->curCycle();
    regionInsts = 0;
    cfgMark = regionEnter;
    std::fill(regionCfgCycles.begin(), regionCfgCycles.end(), 0);
}

// See controller.hh.
void
WindowController::closeRegion()
{
    if (region < 0)
        return;
    accountConfigCycles();
    const Cycles now = cpu->curCycle();
    // The config reported is the one in effect for most of the visit.
    int cfg = cur;
    uint64_t best = 0;
    for (int i = 0; i < table.size(); i++) {
        if (regionCfgCycles[i] > best) {
            best = regionCfgCycles[i];
            cfg = i;
        }
    }
    if (!regionOut) {
        regionOut = simout.create("region_stats.csv");
        *regionOut->stream() << "region,config,enter_cycle,cycles,insts\n";
    }
    *regionOut->stream() << region << ',' << cfg << ','
                         << (uint64_t)regionEnter << ','
                         << (uint64_t)(now - regionEnter) << ','
                         << regionInsts << '\n';
    region = -1;
}

// See controller.hh.
void
WindowController::finish()
{
    if (!started || finished)
        return;
    finished = true;
    // A control instruction committed last has no successor: deliver it
    // with the fall-through PC (taken = false) so every committed branch
    // reaches onBranchCommit().
    flushPendingBranch(pendFallthrough);
    endPeriod();
    closeRegion();
    policy->finish();
    if (traceOut)
        traceOut->stream()->flush();
    if (regionOut)
        regionOut->stream()->flush();
}

} // namespace o3
} // namespace gem5
