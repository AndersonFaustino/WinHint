/**
 * @file
 * @brief Host unit tests of the gem5-independent B9 LTP logic
 *        (ltp_core.hh). Exit status 0 iff every check passed.
 */
// Host unit tests of the gem5-independent B9 LTP logic (ltp_core.hh):
// UIT (set-assoc, LRU), IBDA backward-slice learning, the release and
// dispatch rules and the IQ-reserve arithmetic used by ltp_iew.cc.
//   make -C sim/gem5/src/cpu/o3/window/ltp/tests
#include <cstdio>
#include <utility>

#include "../ltp_core.hh"

using namespace winhint::ltp;

/** Total and failed check counters. */
static int checks = 0, fails = 0;
/** Count a check; report file:line and the expression if c is false. */
#define CHECK(c)                                                         \
    do {                                                                 \
        checks++;                                                        \
        if (!(c)) {                                                      \
            fails++;                                                     \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);     \
        }                                                                \
    } while (0)

/** UIT geometry, insert/hit, duplicate insert and LRU eviction. */
static void
testUit()
{
    UrgencyTable t(8, 2);              // 4 sets x 2 ways
    CHECK(t.numSets() == 4);
    CHECK(!t.hit(0x100, false));
    t.insert(0x100);
    CHECK(t.hit(0x100, false));
    CHECK(t.inserts == 1);
    t.insert(0x100);                   // already present: no new insert
    CHECK(t.inserts == 1);
    // Same set ((pc >> 1) % 4 == 0): 0x100, 0x108, 0x110.
    t.insert(0x108);
    t.hit(0x100, true);                // 0x100 most recently used
    t.insert(0x110);                   // evicts the LRU way (0x108)
    CHECK(t.evictions == 1);
    CHECK(t.hit(0x100, false));
    CHECK(!t.hit(0x108, false));
    CHECK(t.hit(0x110, false));
}

/** IBDA: the backward slice of a seed is learnt one link per visit;
 *  per-thread RDTs are independent. */
static void
testIbda()
{
    // Program (program order, one thread):
    //   A: r1 = ...         (pc 0x10)
    //   B: r2 = f(r1)       (pc 0x20)
    //   L: load [r2]        (pc 0x30)  <- long-latency seed
    UrgencyTable t(64, 4);
    const uint32_t r1 = 1, r2 = 2, r3 = 3;
    auto visit = [&](bool expectL) {
        bool a = t.classify(0, 0x10, nullptr, 0, &r1, 1);
        bool b = t.classify(0, 0x20, &r1, 1, &r2, 1);
        bool l = t.classify(0, 0x30, &r2, 1, &r3, 1);
        CHECK(l == expectL);
        return std::make_pair(a, b);
    };
    auto v0 = visit(false);
    CHECK(!v0.first && !v0.second);
    t.insert(0x30);                    // commit-time seed
    auto v1 = visit(true);             // L urgent -> B becomes urgent
    CHECK(!v1.first && !v1.second);
    CHECK(t.hit(0x20, false) && !t.hit(0x10, false));
    auto v2 = visit(true);             // B urgent -> A becomes urgent
    CHECK(!v2.first && v2.second);
    CHECK(t.hit(0x10, false));
    auto v3 = visit(true);             // whole slice learnt
    CHECK(v3.first && v3.second);

    // Threads have separate RDTs.
    UrgencyTable m(64, 4, 2);
    m.insert(0x30);
    m.classify(1, 0x20, nullptr, 0, &r2, 1);   // thread 1 producer of r2
    m.classify(0, 0x30, &r2, 1, nullptr, 0);   // thread 0: no producer
    CHECK(!m.hit(0x20, false));
}

/** Release priority, IQ-reserve arithmetic and the dispatch rule. */
static void
testRules()
{
    ParkedView v;
    CHECK(releaseReason(v) == NumReasons);
    v.room = true;
    CHECK(releaseReason(v) == RelRoom);
    v.fullHead = true;
    CHECK(releaseReason(v) == RelFull);
    v.olderThanUrgentMem = true;
    CHECK(releaseReason(v) == RelMemOrder);
    v.urgent = true;
    CHECK(releaseReason(v) == RelUrgent);
    v.draining = true;
    CHECK(releaseReason(v) == RelDrain);
    v.old = true;
    CHECK(releaseReason(v) == RelOld);

    // Only an old release of the oldest parked instruction may use the
    // reserve (deadlock freedom).
    CHECK(releaseNeed(RelOld, true, 4) == 1);
    CHECK(releaseNeed(RelOld, false, 4) == 5);
    CHECK(releaseNeed(RelUrgent, true, 4) == 5);
    CHECK(dispatchNeed(true, 4) == 1);
    CHECK(dispatchNeed(false, 4) == 5);

    // dispatchAction(drain, ltpEmpty, mem, memParked, urgent, nop, room)
    CHECK(dispatchAction(true, true, false, false, false, false, false) ==
          ActDirect);
    CHECK(dispatchAction(true, false, false, false, true, false, true) ==
          ActDrainStall);
    CHECK(dispatchAction(false, false, true, true, true, false, true) ==
          ActParkUrgent);
    CHECK(dispatchAction(false, false, true, true, false, false, true) ==
          ActPark);
    CHECK(dispatchAction(false, false, false, false, true, false, false) ==
          ActDirect);
    CHECK(dispatchAction(false, false, false, false, false, true, false) ==
          ActDirect);
    CHECK(dispatchAction(false, true, false, false, false, false, true) ==
          ActBypass);
    CHECK(dispatchAction(false, true, false, false, false, false, false) ==
          ActPark);
    CHECK(dispatchAction(false, false, true, false, false, false, true) ==
          ActPark);
}

/** @brief Run every test group. @return 0 iff every check passed. */
int
main()
{
    testUit();
    testIbda();
    testRules();
    std::printf("ltp_core: %d/%d checks passed\n", checks - fails, checks);
    return fails ? 1 : 0;
}
