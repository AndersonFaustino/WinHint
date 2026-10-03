/**
 * @file
 * @brief Host unit tests for the B2-B5 window policies.
 *
 * Host unit tests for the B2-B5 window policies (occupancy, mlp, bbv, lut).
 * No gem5 needed: gem5's base/logging.hh and base/types.hh are stubbed in
 * tests/stubs (fatal() throws). Policies are created through the same
 * registry as in gem5, and fed synthetic WindowSamples.
 *
 *   make -C sim/gem5/src/cpu/o3/window/tests      (-> build/tests/)
 *   build/tests/test_window_policies [path/to/sim/tests/test.lut] [tmpdir]
 *
 * Arguments default to sim/tests/test.lut (relative to the cwd) and /tmp
 * (where the malformed-LUT cases are written). Exit status 0 iff every
 * check passed.
 */

#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cpu/o3/window/bbv_policy.hh"
#include "cpu/o3/window/lut_policy.hh"
#include "cpu/o3/window/policy.hh"
#include "cpu/o3/window/window_lut.hh"

using namespace gem5::o3;

/** Failed and total check counters. */
static int failures = 0, checks = 0;

/** Count a check; report file:line and the expression if cond is false. */
#define CHECK(cond)                                                       \
    do {                                                                  \
        checks++;                                                         \
        if (!(cond)) {                                                    \
            failures++;                                                   \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                 \
    } while (0)

/** Count a check comparing a and b as long; report both on mismatch. */
#define CHECK_EQ(a, b)                                                    \
    do {                                                                  \
        checks++;                                                         \
        long _a = (long)(a), _b = (long)(b);                              \
        if (_a != _b) {                                                   \
            failures++;                                                   \
            std::printf("  FAIL %s:%d: %s == %ld, expected %ld\n",        \
                        __FILE__, __LINE__, #a, _a, _b);                  \
        }                                                                 \
    } while (0)

/** Count a check that stmt throws winhint_test::Fatal (stubbed fatal()). */
#define CHECK_FATAL(stmt)                                                 \
    do {                                                                  \
        checks++;                                                         \
        bool _thrown = false;                                             \
        try {                                                             \
            stmt;                                                         \
        } catch (const winhint_test::Fatal &) {                           \
            _thrown = true;                                               \
        }                                                                 \
        if (!_thrown) {                                                   \
            failures++;                                                   \
            std::printf("  FAIL %s:%d: no fatal from %s\n", __FILE__,     \
                        __LINE__, #stmt);                                 \
        }                                                                 \
    } while (0)

/** Short alias of WindowPolicy::Keep. */
static const int K = WindowPolicy::Keep;

/** @brief Default machine table (docs/interfaces.md §3).
 *  @return The 4-config table of riscv_ooo.json. */
static WindowTable
defaultTable()
{
    WindowTable t;
    t.rob = {64, 128, 192, 256};
    t.iq = {32, 64, 96, 128};
    t.lq = {16, 32, 48, 64};
    t.sq = {16, 32, 48, 64};
    return t;
}

/** A policy plus the objects its env refers to. */
struct Harness
{
    WindowTable table = defaultTable();   ///< config table (outlives p)
    WindowArgs args;                      ///< parsed window_args
    std::unique_ptr<WindowPolicy> p;      ///< policy under test
    int cur;                              ///< config "in effect"

    /**
     * @brief Create a policy through the registry, as the controller does
     *        (period 1000, no outdir, no CPU), including checkAllUsed().
     * @param name Policy name.
     * @param a window_args string.
     * @param lut window_lut_file.
     * @param initial window_initial.
     * @param outdir gem5 outdir (policy dumps; "" = none).
     */
    Harness(const std::string &name, const std::string &a,
            const std::string &lut = "", int initial = 3,
            const std::string &outdir = "")
        : args(a)
    {
        WindowPolicyEnv env{table, initial, args, lut, 1000, outdir, nullptr};
        p = WindowPolicyRegistry::create(name, env);
        args.checkAllUsed(name);           // as the controller does
        cur = table.clamp(p->initialConfig());
    }

    /**
     * @brief Feed one period; apply the decision like the controller.
     * @param s0 Sample (its config field is overwritten with cur).
     * @return The policy's raw decision (K or a config index).
     */
    int
    step(const WindowSample &s0)
    {
        WindowSample s = s0;
        s.config = cur;
        int r = p->onPeriod(s, cur);
        if (r != K)
            cur = table.clamp(r);
        return r;
    }
};

/**
 * @brief Build a 1000-cycle, 1500-instruction sample (IPC 1.5).
 * @param rob,iq,lq,sq Mean occupancies.
 * @param robFull,iqFull,lqFull,sqFull Full-under-cap cycles; anyFullCycles
 *        is approximated by their maximum.
 * @return The sample.
 */
static WindowSample
sample(double rob, double iq, double lq, double sq, uint64_t robFull = 0,
       uint64_t iqFull = 0, uint64_t lqFull = 0, uint64_t sqFull = 0)
{
    WindowSample s;
    s.cycles = 1000;
    s.insts = 1500;
    s.ipc = 1.5;
    s.robOcc = rob;
    s.iqOcc = iq;
    s.lqOcc = lq;
    s.sqOcc = sq;
    s.robFullCycles = robFull;
    s.iqFullCycles = iqFull;
    s.lqFullCycles = lqFull;
    s.sqFullCycles = sqFull;
    s.anyFullCycles = std::max({robFull, iqFull, lqFull, sqFull});
    s.fullFrac = s.anyFullCycles / 1000.0;
    return s;
}

// ---------------------------------------------------------------------------
/** Registry lookups and constructor fatal() paths (typo guard etc.). */
static void
testRegistry()
{
    std::printf("registry\n");
    for (const char *n : {"occupancy", "mlp", "bbv", "lut"})
        CHECK(WindowPolicyRegistry::has(n));
    CHECK_FATAL(Harness("occupancy", "bogus_key=1"));   // typo guard
    CHECK_FATAL(Harness("occupancy", "update=0"));
    CHECK_FATAL(Harness("mlp", "miss_level=3"));
    CHECK_FATAL(Harness("bbv", "buckets=0"));
    CHECK_FATAL(Harness("lut", ""));                     // no LUT file
    // names(): ascending, exactly the policies linked into this binary.
    CHECK(WindowPolicyRegistry::names() ==
          std::vector<std::string>({"bbv", "lut", "mlp", "occupancy"}));
    CHECK_FATAL(WindowPolicyRegistry::create("static", WindowPolicyEnv{
        defaultTable(), 3, WindowArgs(), "", 1000, "", nullptr}));
}

// ---------------------------------------------------------------------------
/**
 * @brief Names and the WindowPolicy default hooks the B2/B3 policies keep;
 *        an empty period (0 cycles) is ignored by occupancy, mlp and lut.
 * @param lutPath Path of sim/tests/test.lut.
 */
static void
testDefaults(const std::string &lutPath)
{
    std::printf("names + default hooks\n");
    WindowSample empty;                       // cycles == 0
    struct { const char *name; const char *lut; } cases[] = {
        {"occupancy", ""}, {"mlp", ""}, {"lut", nullptr},
    };
    for (auto &c : cases) {
        Harness h(c.name, "", c.lut ? c.lut : lutPath, 2);
        CHECK(std::string(h.p->name()) == c.name);
        CHECK_EQ(h.p->initialConfig(), 2);
        CHECK_EQ(h.step(empty), K);
        CHECK_EQ(h.p->onSetwin(64, 2), K);    // hints are ignored
        CHECK_EQ(h.p->onRegion(3, 2), K);
        CHECK(!h.p->wantsBranchCommits());
        CHECK_EQ(h.p->resizedStructures(), WinAll);
        h.p->onBranchCommit(0x100, 0x200, true, true, 5);   // no-op
        h.p->finish();                                      // no-op
    }
    Harness b("bbv", "");
    CHECK(std::string(b.p->name()) == "bbv");
    CHECK_EQ(b.step(empty), K);
}

// ---------------------------------------------------------------------------
/** B2 occupancy: shrink per update period, grow on overflow, drain. */
static void
testOccupancy()
{
    std::printf("occupancy (B2)\n");
    {   // Under-used: shrinks one step per update period (update=2).
        Harness h("occupancy", "", "", 3);
        WindowSample s = sample(20, 10, 5, 5);
        CHECK_EQ(h.step(s), K);
        CHECK_EQ(h.step(s), 2);
        CHECK_EQ(h.step(s), K);
        CHECK_EQ(h.step(s), 1);
        CHECK_EQ(h.step(s), K);
        CHECK_EQ(h.step(s), 0);
        CHECK_EQ(h.step(s), K);
        CHECK_EQ(h.step(s), K);              // already the smallest
        CHECK_EQ(h.cur, 0);
    }
    {   // Dispatch stalls on a full IQ: grows at once (no update wait).
        Harness h("occupancy", "", "", 1);
        CHECK_EQ(h.step(sample(120, 64, 10, 10, 0, 300)), 2);
        CHECK_EQ(h.step(sample(150, 96, 10, 10, 0, 300)), 3);
        CHECK_EQ(h.step(sample(150, 96, 10, 10, 0, 300)), K);  // largest
    }
    {   // Overflow below threshold in one period, but accumulated over the
        // update period it crosses up_frac * 2 * 1000 = 100 cycles.
        Harness h("occupancy", "", "", 1);
        CHECK_EQ(h.step(sample(120, 50, 10, 10, 60)), K);
        CHECK_EQ(h.step(sample(120, 50, 10, 10, 60)), 2);
    }
    {   // One structure (LQ) still needs the current size: no shrink.
        Harness h("occupancy", "", "", 3);
        WindowSample s = sample(20, 10, 50, 5);
        h.step(s);
        CHECK_EQ(h.step(s), K);
        CHECK_EQ(h.cur, 3);
    }
    {   // down_factor=0.5: shrink when half a step is free.
        Harness h("occupancy", "down_factor=0.5,update=1", "", 3);
        CHECK_EQ(h.step(sample(210, 10, 5, 5)), 2);   // 256-210=46 >= 32
        CHECK_EQ(h.step(sample(170, 10, 5, 5)), K);   // 192-170=22 < 32
    }
    {   // Full cycles in the period right after a downsize are the drain of
        // the shrink, not overflow; afterwards they count again.
        Harness h("occupancy", "update=1", "", 3);
        CHECK_EQ(h.step(sample(20, 10, 5, 5)), 2);
        CHECK_EQ(h.step(sample(150, 90, 5, 5, 300)), K);
        CHECK_EQ(h.step(sample(150, 90, 5, 5, 300)), 3);
    }
    {   // Closed loop: demand of 150 ROB entries settles at 192 (index 2).
        Harness h("occupancy", "", "", 0);
        int switches = 0;
        for (int i = 0; i < 50; i++) {
            double cap = h.table.rob[h.cur];
            double occ = std::min(150.0, cap);
            uint64_t full = 150.0 >= cap ? 600 : 0;
            if (h.step(sample(occ, occ / 2, occ / 4, occ / 4, full)) != K)
                switches++;
        }
        CHECK_EQ(h.cur, 2);
        CHECK_EQ(switches, 2);
    }
}

// ---------------------------------------------------------------------------
/**
 * @brief Sample with long-latency misses for the mlp policy.
 * @param l2 L2 misses (l2Mpki = l2 / 1.5).
 * @param mlp MLP value.
 * @param l1d L1D misses (default 4 x l2).
 * @return The sample.
 */
static WindowSample
memSample(uint64_t l2, double mlp, uint64_t l1d = 0)
{
    WindowSample s = sample(100, 50, 20, 10);
    s.l2Misses = l2;
    s.l1dMisses = l1d ? l1d : l2 * 4;
    s.l2Mpki = l2 / 1.5;
    s.mlp = mlp;
    return s;
}

/** B3 mlp: ILP mode, level-by-level growth, revert + back-off. */
static void
testMlp()
{
    std::printf("mlp (B3)\n");
    {   // No long-latency misses: back to ILP mode after shrink_delay=2.
        Harness h("mlp", "", "", 3);
        CHECK_EQ(h.step(memSample(0, 0)), K);
        CHECK_EQ(h.step(memSample(0, 0)), 0);
        CHECK_EQ(h.step(memSample(0, 0)), K);
    }
    {   // L2 misses with MLP: grow level by level while MLP keeps rising.
        Harness h("mlp", "", "", 0);
        CHECK_EQ(h.step(memSample(20, 3.0)), 1);
        CHECK_EQ(h.step(memSample(20, 4.0)), 2);   // 4.0 >= 3.0*1.1: keep
        CHECK_EQ(h.step(memSample(20, 6.0)), 3);   // 6.0 >= 4.4: keep
        CHECK_EQ(h.step(memSample(20, 7.0)), K);   // verified, largest
        CHECK_EQ(h.cur, 3);
    }
    {   // MLP saturates: the extra level is not exploited -> revert and
        // back off (no regrowth for `backoff` periods, then retry).
        Harness h("mlp", "backoff=3", "", 0);
        CHECK_EQ(h.step(memSample(20, 3.0)), 1);
        CHECK_EQ(h.step(memSample(20, 3.1)), 0);   // < 3.3: revert
        CHECK_EQ(h.step(memSample(20, 3.0)), K);   // cooldown
        CHECK_EQ(h.step(memSample(20, 3.0)), K);
        CHECK_EQ(h.step(memSample(20, 3.0)), 1);   // retry
        CHECK_EQ(h.step(memSample(20, 3.0)), 0);   // fails again
        int k = 0;                                 // back-off doubled: 6
        while (h.step(memSample(20, 3.0)) == K && k < 20)
            k++;
        CHECK_EQ(k, 5);
    }
    {   // Isolated (dependent) misses, no MLP: step down to ILP mode.
        Harness h("mlp", "", "", 3);
        CHECK_EQ(h.step(memSample(20, 1.0)), 2);
        CHECK_EQ(h.step(memSample(20, 1.0)), 1);
        CHECK_EQ(h.step(memSample(20, 1.0)), 0);
        CHECK_EQ(h.step(memSample(20, 1.0)), K);
    }
    {   // ilp=1: never below 1; L1D-triggered variant.
        Harness h("mlp", "ilp=1,miss_level=1,miss_min=10", "", 3);
        CHECK_EQ(h.step(memSample(0, 0, 5)), K);    // 5 < miss_min
        CHECK_EQ(h.step(memSample(0, 0, 5)), 1);
        CHECK_EQ(h.step(memSample(0, 2.0, 40)), 2);
    }
}

// ---------------------------------------------------------------------------
/**
 * @brief Feed one BBV interval of phase `ph` (0 = A, 1 = B).
 * @param h Harness holding a BbvPolicy.
 * @param ph Phase (selects the synthetic PCs and block lengths).
 * @param insts Instructions to feed (at least).
 */
static void
feedInterval(Harness &h, int ph, uint64_t insts)
{
    auto *bbv = dynamic_cast<BbvPolicy *>(h.p.get());
    const uint64_t base = ph == 0 ? 0x10000 : 0x80000;
    uint64_t done = 0;
    for (int i = 0; done < insts; i++) {
        uint64_t pc = base + 4 * (uint64_t)(i % 8) * (ph == 0 ? 3 : 5);
        uint64_t len = 4 + (i % 8) * (ph == 0 ? 1 : 3);
        bbv->onBranchCommit(pc, pc + 64, true, false, len);
        done += len;
    }
}

/**
 * @brief One BBV interval of synthetic phase ph (distinct PC sets per ph)
 *        fed as a single period of `insts` instructions at IPC 1.
 * @param h Harness holding a BbvPolicy (interval_insts <= insts).
 * @param ph Phase number (0, 1, 2, ...).
 * @param cur Configuration passed as "in effect" (default: h.cur).
 * @param insts Instructions in the interval.
 * @return The policy's decision.
 */
static int
bbvInterval(Harness &h, int ph, int cur = -1, uint64_t insts = 1000)
{
    auto *b = dynamic_cast<BbvPolicy *>(h.p.get());
    for (uint64_t i = 0; i < insts / 4; i++)
        b->onBranchCommit(0x1000 * (ph + 1) + 4 * (i % 3), 0, true, false, 4);
    WindowSample s = sample(10, 5, 2, 2);
    s.insts = insts;
    s.cycles = insts;
    s.ipc = 1;
    if (cur < 0)
        return h.step(s);
    return h.p->onPeriod(s, cur);
}

/** B4 bbv: LRU eviction, Markov hysteresis, learning edge cases. */
static void
testBbvTable()
{
    std::printf("bbv (B4) phase table / predictor\n");
    {   // LRU: A A B A C with 2 entries evicts B (not the lowest id A),
        // and with it only the Markov entries from/to B.
        Harness h("bbv", "interval_insts=1000,max_phases=2", "", 3);
        auto *b = dynamic_cast<BbvPolicy *>(h.p.get());
        bbvInterval(h, 0);
        bbvInterval(h, 0);
        bbvInterval(h, 1);
        bbvInterval(h, 0);
        CHECK_EQ(b->lastPhaseId(), 0);       // A recognised again
        bbvInterval(h, 2);
        CHECK_EQ(b->lastPhaseId(), 2);
        CHECK_EQ(b->numPhases(), 2);
        bbvInterval(h, 0);
        CHECK_EQ(b->lastPhaseId(), 0);       // A survived
        bbvInterval(h, 1);
        CHECK_EQ(b->lastPhaseId(), 3);       // B was evicted: new id
        CHECK_EQ(b->learnedConfig(1), -1);   // evicted phase is unknown
    }
    {   // max_phases=1: the new phase evicts the last one (run restarts).
        Harness h("bbv", "interval_insts=1000,max_phases=1", "", 3);
        auto *b = dynamic_cast<BbvPolicy *>(h.p.get());
        bbvInterval(h, 0);
        bbvInterval(h, 1);
        CHECK_EQ(b->numPhases(), 1);
        CHECK_EQ(b->lastPhaseId(), 1);
        CHECK_EQ(b->predictedPhaseId(), 1);
        bbvInterval(h, 0);
        CHECK_EQ(b->lastPhaseId(), 2);
    }
    {   // RLE-Markov confidence (run_max=0: key = last phase only).
        // A B A B trains (A -> B) to confidence 1; A A first decrements
        // it (prediction stays B), the next A A replaces it with A.
        Harness h("bbv", "interval_insts=1000,run_max=0", "", 3);
        auto *b = dynamic_cast<BbvPolicy *>(h.p.get());
        for (int ph : {0, 1, 0, 1, 0})
            bbvInterval(h, ph);
        CHECK_EQ(b->predictedPhaseId(), 1);
        bbvInterval(h, 0);
        CHECK_EQ(b->predictedPhaseId(), 1);  // hysteresis
        bbvInterval(h, 0);
        CHECK_EQ(b->predictedPhaseId(), 0);  // replaced
        CHECK_EQ(b->intervalsSeen(), 7);
        CHECK_EQ(b->correctPredictions(), 2);   // intervals 4 and 5
    }
    {   // Learning: explore=2 keeps a phase "exploring" until every config
        // has 2 clean intervals; a config change inside an interval is not
        // credited (mixed) and an out-of-range config is ignored.
        Harness h("bbv", "interval_insts=2000,explore=2", "", 3);
        auto *b = dynamic_cast<BbvPolicy *>(h.p.get());
        CHECK_EQ(b->learnedConfig(0), -1);   // unknown phase
        CHECK_EQ(bbvInterval(h, 0, 3), K);   // first half, config 3
        // second half under config 2: mixed interval, nothing learned; the
        // (unlearned) phase asks for the largest config
        CHECK_EQ(bbvInterval(h, 0, 2), 3);
        CHECK_EQ(b->numPhases(), 1);
        CHECK_EQ(b->learnedConfig(0), -1);
        // clean intervals: config 3 explored twice -> 2 is tried next
        h.cur = 3;
        bbvInterval(h, 0, -1, 2000);
        CHECK_EQ(h.cur, 3);
        CHECK_EQ(bbvInterval(h, 0, -1, 2000), 2);
        CHECK_EQ(b->learnedConfig(0), -1);   // still exploring 0..2
    }
    {   // A first interval under a config outside the table is not
        // credited: afterwards exactly one clean interval per config is
        // still needed (explore=1) before the phase is learned.
        Harness h("bbv", "interval_insts=1000", "", 3);
        auto *b = dynamic_cast<BbvPolicy *>(h.p.get());
        CHECK_EQ(bbvInterval(h, 0, 9), 3);   // explore the largest first
        h.cur = 3;
        for (int want : {2, 1, 0}) {
            CHECK_EQ(b->learnedConfig(0), -1);
            CHECK_EQ(bbvInterval(h, 0), want);
        }
        bbvInterval(h, 0);                   // config 0 explored
        CHECK_EQ(b->learnedConfig(0), 0);    // equal IPC: smallest
    }
}

/**
 * @brief bbv finish(): bbv_phases.csv in the outdir (header, one row per
 *        phase); nothing is written without an outdir or into a missing one.
 * @param tmpdir Writable directory.
 */
static void
testBbvFinish(const std::string &tmpdir)
{
    std::printf("bbv (B4) finish dump\n");
    {
        Harness h("bbv", "interval_insts=1000", "", 3, tmpdir);
        bbvInterval(h, 0);                   // phase 0 at config 3
        bbvInterval(h, 0);                   // phase 0 at config 2
        bbvInterval(h, 1);                   // phase 1 at config 1
        h.p->finish();
        std::ifstream in(tmpdir + "/bbv_phases.csv");
        CHECK(in.good());
        std::string l0, l1, r0, r1, extra;
        std::getline(in, l0);
        std::getline(in, l1);
        std::getline(in, r0);
        std::getline(in, r1);
        CHECK(!std::getline(in, extra));
        CHECK(l0 == "# intervals=3 predicted_correct=1 phases_allocated=2");
        CHECK(l1 == "phase,visits,learned_config,ipc_c0,n_c0,ipc_c1,n_c1,"
                    "ipc_c2,n_c2,ipc_c3,n_c3");
        CHECK(r0 == "0,2,-1,0,0,0,0,1,1,1,1");
        CHECK(r1 == "1,1,-1,0,0,1,1,0,0,0,0");
    }
    {   // No outdir: no file (and no crash); unwritable outdir: ignored.
        std::string d = tmpdir + "/no_such_dir";
        Harness h("bbv", "interval_insts=1000", "", 3, d);
        bbvInterval(h, 0);
        h.p->finish();
        CHECK(!std::ifstream(d + "/bbv_phases.csv").good());
        Harness g("bbv", "interval_insts=1000");
        bbvInterval(g, 0);
        g.p->finish();
    }
}

/** B4 bbv: phase detection, per-phase learning, prediction, LRU. */
static void
testBbv()
{
    std::printf("bbv (B4)\n");
    // Phase A gains up to config 2; phase B is window-insensitive.
    const double ipcA[4] = {1.0, 1.5, 2.0, 2.01};
    const double ipcB[4] = {0.5, 0.5, 0.5, 0.5};
    Harness h("bbv", "interval_insts=10000", "", 3);
    auto *bbv = dynamic_cast<BbvPolicy *>(h.p.get());
    CHECK(bbv != nullptr);
    CHECK(h.p->wantsBranchCommits());

    int wrong = 0, switchesLate = 0;
    const int reps = 30, run = 4;
    for (int r = 0; r < reps; r++) {
        for (int ph = 0; ph < 2; ph++) {
            for (int i = 0; i < run; i++) {
                double ipc = (ph == 0 ? ipcA : ipcB)[h.cur];
                // One interval = two periods of 5000 instructions.
                for (int k = 0; k < 2; k++) {
                    feedInterval(h, ph, 5000);
                    WindowSample s = sample(100, 50, 20, 10);
                    s.insts = 5000;
                    s.cycles = (uint64_t)(5000 / ipc);
                    s.ipc = ipc;
                    int cfgBefore = h.cur;
                    int d = h.step(s);
                    if (k == 0)
                        CHECK_EQ(d, K);            // mid-interval
                    if (r >= reps - 5 && d != K && d != cfgBefore)
                        switchesLate++;
                }
                if (r >= reps - 5) {
                    // Config chosen for the NEXT interval must match the
                    // phase that actually comes next.
                    int nextPh = (i + 1 < run) ? ph : 1 - ph;
                    int want = nextPh == 0 ? 2 : 0;
                    if (h.cur != want)
                        wrong++;
                }
            }
        }
    }
    CHECK_EQ(bbv->numPhases(), 2);
    CHECK_EQ(bbv->learnedConfig(0), 2);   // A: smallest within 2% of best
    CHECK_EQ(bbv->learnedConfig(1), 0);   // B: insensitive -> smallest
    CHECK_EQ(wrong, 0);                   // RLE Markov predicts transitions
    CHECK_EQ(switchesLate, 10);           // exactly 2 per repetition
    std::printf("  intervals=%lu correct_predictions=%lu\n",
                (unsigned long)bbv->intervalsSeen(),
                (unsigned long)bbv->correctPredictions());
    CHECK(bbv->correctPredictions() > bbv->intervalsSeen() * 8 / 10);

    {   // Phase table capacity (LRU): 3 distinct phases, max_phases=2.
        Harness g("bbv", "interval_insts=1000,max_phases=2", "", 3);
        auto *b = dynamic_cast<BbvPolicy *>(g.p.get());
        for (int ph = 0; ph < 3; ph++) {
            for (int i = 0; i < 250; i++)
                b->onBranchCommit(0x1000 * (ph + 1) + 4 * (i % 3), 0, true,
                                  false, 4);
            WindowSample s = sample(10, 5, 2, 2);
            s.insts = 1000;
            g.step(s);
        }
        CHECK_EQ(b->numPhases(), 2);
    }
    testBbvTable();
}

// ---------------------------------------------------------------------------
/**
 * @brief Sample carrying the four default LUT features.
 * @param ipc IPC.
 * @param rob Mean ROB occupancy.
 * @param l1d L1D MPKI.
 * @param mlp MLP.
 * @return The sample.
 */
static WindowSample
lutSample(double ipc, double rob, double l1d, double mlp)
{
    WindowSample s = sample(rob, 10, 5, 5);
    s.ipc = ipc;
    s.l1dMpki = l1d;
    s.mlp = mlp;
    return s;
}

/**
 * @brief Write text to dir/name.
 * @param dir Directory.
 * @param name File name.
 * @param text Contents.
 * @return The path.
 */
static std::string
writeTmp(const std::string &dir, const std::string &name,
         const std::string &text)
{
    std::string p = dir + "/" + name;
    std::ofstream(p) << text;
    return p;
}

/**
 * @brief B5 lut: lookups on sim/tests/test.lut, edge semantics, parsing.
 * @param lutPath Path of sim/tests/test.lut.
 * @param tmpdir Directory for the generated LUT files.
 */
static void
testLut(const std::string &lutPath, const std::string &tmpdir)
{
    std::printf("lut (B5) with %s\n", lutPath.c_str());
    // sim/tests/test.lut: edges ipc 1.0 | rob_occ 48 | l1d_mpki 10 | mlp 1.5
    // table: 1 1 3 3 1 1 3 3 0 0 2 2 1 1 2 2
    // idx = ((b_ipc * 2 + b_rob) * 2 + b_l1d) * 2 + b_mlp
    Harness h("lut", "", lutPath, 3);
    auto *lp = dynamic_cast<LutPolicy *>(h.p.get());
    CHECK_EQ(lp->lut().size(), 16);
    struct Case { double ipc, rob, l1d, mlp; int want; } cases[] = {
        {0.5, 30, 5, 1.0, 1},      // idx 0
        {0.5, 30, 20, 1.0, 3},     // idx 2
        {0.5, 30, 20, 2.0, 3},     // idx 3
        {2.0, 30, 5, 2.0, 0},      // idx 9
        {2.0, 30, 20, 1.0, 2},     // idx 10
        {2.0, 60, 20, 2.0, 2},     // idx 15
        {2.0, 60, 5, 1.0, 1},      // idx 12
        {1.0, 48, 10, 1.5, 1},     // all on the edges -> lower bins, idx 0
        {1.0000001, 48, 10, 1.5, 0},   // ipc just above the edge, idx 8
    };
    for (auto &c : cases) {
        h.cur = 3;
        int r = h.step(lutSample(c.ipc, c.rob, c.l1d, c.mlp));
        CHECK_EQ(r == K ? 3 : r, c.want);
    }
    h.cur = 1;
    CHECK_EQ(h.step(lutSample(0.5, 30, 5, 1.0)), K);   // already there

    // Comments and arbitrary line splits are accepted.
    std::string ok = writeTmp(tmpdir, "ok.lut",
        "WINHINT_LUT 1 # header\nfeatures 2 l2_mpki\n branch_mpki\n"
        "edges l2_mpki 2 1 5\nedges branch_mpki 0\ntable 3\n0\n1 2\n");
    Harness g("lut", "", ok, 0);
    WindowSample s = sample(10, 5, 2, 2);
    s.l2Mpki = 3;
    CHECK_EQ(g.step(s), 1);
    s.l2Mpki = 7;
    CHECK_EQ(g.step(s), 2);

    // Malformed files are fatal.
    CHECK_FATAL(Harness("lut", "", writeTmp(tmpdir, "b1.lut",
        "WINHINT_LUT 2\nfeatures 1 ipc\nedges ipc 0\ntable 1\n0\n"), 0));
    CHECK_FATAL(Harness("lut", "", writeTmp(tmpdir, "b2.lut",
        "WINHINT_LUT 1\nfeatures 1 ipc\nedges ipc 1 1\ntable 3\n0 1 2\n"),
        0));
    CHECK_FATAL(Harness("lut", "", writeTmp(tmpdir, "b3.lut",
        "WINHINT_LUT 1\nfeatures 1 bogus\nedges bogus 0\ntable 1\n0\n"),
        0));
    CHECK_FATAL(Harness("lut", "", writeTmp(tmpdir, "b4.lut",
        "WINHINT_LUT 1\nfeatures 1 ipc\nedges ipc 0\ntable 1\n4\n"), 0));
    CHECK_FATAL(Harness("lut", "", writeTmp(tmpdir, "b5.lut",
        "WINHINT_LUT 1\nfeatures 1 ipc\nedges ipc 2 2 1\ntable 3\n0 0 0\n"),
        0));
    CHECK_FATAL(Harness("lut", "", writeTmp(tmpdir, "b6.lut",
        "WINHINT_LUT 1\nfeatures 1 ipc\nedges ipc 0\ntable 1\n0 1\n"), 0));
    CHECK_FATAL(Harness("lut", "", tmpdir + "/missing.lut", 0));
}

/**
 * @brief winhint::WindowLut::parse(): every rejection reason, the error
 *        message, NaN binning and the empty LUT.
 * @param tmpdir Directory for load() of a malformed file.
 */
static void
testWindowLutParse(const std::string &tmpdir)
{
    std::printf("window_lut parser\n");
    const std::string hdr = "WINHINT_LUT 1\n";
    const std::string f1 = hdr + "features 1 ipc\n";
    struct { std::string text, err; } bad[] = {
        {"", "bad header (expected 'WINHINT_LUT 1')"},
        {"WINHINT_LUT", "bad header (expected 'WINHINT_LUT 1')"},
        {"WINHINT_LUT x", "bad header (expected 'WINHINT_LUT 1')"},
        {"LUT 1", "bad header (expected 'WINHINT_LUT 1')"},
        {hdr, "bad 'features' line"},
        {hdr + "features 0", "bad 'features' line"},
        {hdr + "features 1.5 ipc", "bad 'features' line"},
        {hdr + "features -1 ipc", "bad 'features' line"},
        {hdr + "features 2 ipc", "truncated feature list"},
        {f1, "bad 'edges' line for feature 'ipc'"},
        {f1 + "edges mlp 0", "bad 'edges' line for feature 'ipc'"},
        {f1 + "edges ipc x", "bad 'edges' line for feature 'ipc'"},
        {f1 + "edges ipc 2 1", "bad or truncated edges for 'ipc'"},
        {f1 + "edges ipc 1 nan", "bad or truncated edges for 'ipc'"},
        {f1 + "edges ipc 1 1x", "bad or truncated edges for 'ipc'"},
        {f1 + "edges ipc 2 2 1", "edges of 'ipc' are not ascending"},
        {f1 + "edges ipc 1 1", "bad 'table' line"},
        {f1 + "edges ipc 1 1 tab 2", "bad 'table' line"},
        {f1 + "edges ipc 1 1 table 3 0 1 2",
         "table has 3 cells, expected 2"},
        {f1 + "edges ipc 1 1 table 2 0", "bad or truncated table"},
        {f1 + "edges ipc 1 1 table 2 0 -1", "bad or truncated table"},
        {f1 + "edges ipc 1 1 table 2 0 1.5", "bad or truncated table"},
        {f1 + "edges ipc 1 1 table 2 0 1 7", "trailing data after the table"},
    };
    for (auto &c : bad) {
        winhint::WindowLut lut;
        std::string err;
        bool ok = lut.parse(c.text, err);
        CHECK(!ok);
        if (err != c.err)
            std::printf("  got '%s' for '%s'\n", err.c_str(), c.text.c_str());
        CHECK(err == c.err);
    }

    winhint::WindowLut lut;
    double x0[1] = {5.0};
    CHECK_EQ(lut.lookup(x0), 0);              // empty LUT
    CHECK_EQ(lut.maxConfig(), 0);
    CHECK_EQ(lut.size(), 0);
    std::string err;
    CHECK(lut.parse(hdr + "features 2 ipc mlp\nedges ipc 1 1.0\n"
                    "edges mlp 2 1 2\ntable 6\n0 1 2 3 4 5\n", err));
    CHECK_EQ(lut.size(), 6);
    CHECK_EQ(lut.maxConfig(), 5);
    CHECK_EQ(lut.edges()[1].size(), 2);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    double xs[][2] = {{0.5, 0.5}, {2.0, 1.5}, {2.0, 9.0}, {nan, 9.0},
                      {2.0, nan}};
    const unsigned want[] = {0, 4, 5, 2, 3};
    for (int i = 0; i < 5; i++)
        CHECK_EQ(lut.lookup(xs[i]), want[i]);
    // A failed parse leaves the LUT empty again.
    CHECK(!lut.parse("junk", err));
    CHECK_EQ(lut.size(), 0);
    CHECK_EQ(lut.features().size(), 0);

    // load(): parse errors are prefixed with the path.
    std::string p = writeTmp(tmpdir, "b7.lut", "WINHINT_LUT 1\n");
    CHECK(!lut.load(p, err));
    CHECK(err == p + ": bad 'features' line");
    CHECK(!lut.load(tmpdir + "/missing.lut", err));
    CHECK(err == "cannot open '" + tmpdir + "/missing.lut'");
}

/**
 * @brief Run every test group.
 * @param argc Argument count.
 * @param argv [1] LUT path, [2] temporary directory (both optional).
 * @return 0 iff every check passed.
 */
int
main(int argc, char **argv)
{
    std::string lut = argc > 1 ? argv[1] : "sim/tests/test.lut";
    std::string tmp = argc > 2 ? argv[2] : "/tmp";
    testRegistry();
    testDefaults(lut);
    testOccupancy();
    testMlp();
    testBbv();
    testBbvFinish(tmp);
    testLut(lut, tmp);
    testWindowLutParse(tmp);
    std::printf("%d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
