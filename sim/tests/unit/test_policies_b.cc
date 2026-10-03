/**
 * @file
 * @brief Host unit tests of policy.hh (WindowTable, WindowArgs, registry)
 *        and the static / hint / hybrid policies.
 */
// Host unit tests (no gem5) for the policy interface (policy.hh) and the
// static / hint / hybrid policies. Uses the logging/types stubs of
// sim/gem5/src/cpu/o3/window/tests/stubs (fatal() throws).
//   make -C sim/tests unit
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "cpu/o3/window/hint_policy.hh"
#include "cpu/o3/window/policy.hh"

using namespace gem5::o3;

/** Failed check counter. */
static int fails = 0;
/** Report file:line and the expression if c is false. */
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);     \
            fails++;                                                     \
        }                                                                \
    } while (0)

/**
 * @brief Whether f() throws winhint_test::Fatal (stubbed fatal()/panic()).
 * @param f Callable to run.
 * @return true if it threw Fatal.
 */
template <typename F>
static bool
throws(F f)
{
    try {
        f();
    } catch (const winhint_test::Fatal &) {
        return true;
    }
    return false;
}

/** @return The default 4-config table (docs/interfaces.md §3). */
static WindowTable
table4()
{
    WindowTable t;
    t.rob = {64, 128, 192, 256};
    t.iq = {32, 64, 96, 128};
    t.lq = {16, 32, 48, 64};
    t.sq = {16, 32, 48, 64};
    return t;
}

/**
 * @brief Create a policy through the registry, as the controller does
 *        (period 1000, outdir /tmp, no CPU), including checkAllUsed().
 * @param name Policy name.
 * @param t Config table (must outlive the policy).
 * @param a window_args (must outlive the policy).
 * @param initial window_initial.
 * @return The policy.
 */
static std::unique_ptr<WindowPolicy>
make(const std::string &name, const WindowTable &t, const WindowArgs &a,
     int initial = 3)
{
    WindowPolicyEnv env{t, initial, a, "", 1000, "/tmp", nullptr};
    auto p = WindowPolicyRegistry::create(name, env);
    a.checkAllUsed(name);
    return p;
}

/** @brief Run every check. @return 0 iff all passed. */
int
main()
{
    WindowTable t = table4();

    // configForSetwin (docs/interfaces.md §3)
    CHECK(t.configForSetwin(0) == 3);
    CHECK(t.configForSetwin(8) == 0);
    CHECK(t.configForSetwin(64) == 0);
    CHECK(t.configForSetwin(72) == 1);
    CHECK(t.configForSetwin(128) == 1);
    CHECK(t.configForSetwin(192) == 2);
    CHECK(t.configForSetwin(256) == 3);
    CHECK(t.configForSetwin(504) == 3);

    // WindowArgs
    {
        WindowArgs a(" x=1.5 , s=iq+rob,n=7");
        CHECK(a.getDouble("x", 0) == 1.5);
        CHECK(a.getString("s", "") == "iq+rob");
        CHECK(throws([&] { a.checkAllUsed("t"); }));
        CHECK(a.getInt("n", 0) == 7);
        a.checkAllUsed("t");
        CHECK(throws([] { WindowArgs b("novalue"); }));
        WindowArgs c("x=abc");
        CHECK(throws([&] { c.getDouble("x", 0); }));
        // Empty items are skipped; a later duplicate overwrites; has() does
        // not mark a key used; absent keys return the default.
        WindowArgs d(",a=1,, ,a=2,s=x,");
        CHECK(d.has("a") && d.has("s") && !d.has("b"));
        CHECK(throws([&] { d.checkAllUsed("t"); }));
        CHECK(d.get("a", 0) == 2);
        CHECK(d.getString("s", "dflt") == "x");
        CHECK(d.getString("b", "dflt") == "dflt");
        CHECK(d.getInt("b", -4) == -4);
        CHECK(d.get("b", 2.5) == 2.5);
        d.checkAllUsed("t");
        CHECK(throws([] { WindowArgs e("=1"); }));    // empty key
        WindowArgs f("x=1.5y");                       // trailing garbage
        CHECK(throws([&] { f.getInt("x", 0); }));
        CHECK(WindowArgs("x=2.9").getInt("x", 0) == 2);  // truncated
    }

    // WindowTable::clamp
    CHECK(t.size() == 4 && t.largest() == 3);
    CHECK(t.clamp(-1) == 0 && t.clamp(2) == 2 && t.clamp(9) == 3);

    // Registry
    CHECK(WindowPolicyRegistry::has("static"));
    CHECK(WindowPolicyRegistry::has("hint"));
    CHECK(WindowPolicyRegistry::has("hybrid"));
    CHECK(throws([&] { make("nosuch", t, WindowArgs("")); }));
    CHECK(throws([&] { make("static", t, WindowArgs("bogus=1")); }));
    {   // names(): ascending, every policy linked into this binary
        auto n = WindowPolicyRegistry::names();
        CHECK(n == std::vector<std::string>({"hint", "hybrid", "static"}));
        CHECK(!WindowPolicyRegistry::has("ltp"));
        // a duplicate registration is a panic()
        CHECK(throws([] {
            WindowPolicyRegistry::add("static", WindowPolicyRegistry::Factory());
        }));
    }

    // static
    {
        auto p = make("static", t, WindowArgs(""), 1);
        CHECK(p->initialConfig() == 1);
        CHECK(p->onSetwin(64, 1) == WindowPolicy::Keep);
        WindowSample s;
        s.insts = 1000;
        CHECK(p->onPeriod(s, 1) == WindowPolicy::Keep);
        CHECK(p->resizedStructures() == WinAll);
        // WindowPolicy defaults (static overrides only name()).
        CHECK(std::string(p->name()) == "static");
        CHECK(p->onRegion(5, 1) == WindowPolicy::Keep);
        CHECK(!p->wantsBranchCommits());
        p->onBranchCommit(0x1000, 0x2000, true, false, 7);   // no-op
        p->finish();                                          // no-op
        CHECK(p->onPeriod(s, 1) == WindowPolicy::Keep);
    }

    // hint
    {
        auto p = make("hint", t, WindowArgs(""));
        CHECK(p->initialConfig() == 3);
        CHECK(std::string(p->name()) == "hint");
        CHECK(p->onRegion(1, 3) == WindowPolicy::Keep);
        CHECK(p->onSetwin(64, 3) == 0);
        CHECK(p->onSetwin(100, 0) == 1);
        CHECK(p->onSetwin(0, 1) == 3);
        CHECK(p->onSetwin(400, 1) == 3);
        WindowSample s;
        CHECK(p->onPeriod(s, 0) == WindowPolicy::Keep);
        CHECK(p->resizedStructures() == WinAll);
        auto q = make("hint", t, WindowArgs("structs=iq"));
        CHECK(q->resizedStructures() == WinIQ);
        auto r = make("hint", t, WindowArgs("structs=rob+lsq"));
        CHECK(r->resizedStructures() == (WinROB | WinLQ | WinSQ));
        CHECK(throws([&] { make("hint", t, WindowArgs("structs=foo")); }));
    }

    // parseWindowStructs (shared by hint and ltp's LtpParams::fromArgs)
    {
        CHECK(parseWindowStructs("") == WinAll);
        CHECK(parseWindowStructs("all") == WinAll);
        CHECK(parseWindowStructs("iq+lsq") == (WinIQ | WinLQ | WinSQ));
        CHECK(parseWindowStructs("iq:lsq") == (WinIQ | WinLQ | WinSQ));
        CHECK(parseWindowStructs("rob:iq+sq") == (WinROB | WinIQ | WinSQ));
        CHECK(parseWindowStructs("lq") == WinLQ);
        CHECK(parseWindowStructs("sq+lq") == (WinLQ | WinSQ));
        CHECK(parseWindowStructs("iq+all") == WinAll);
        CHECK(throws([] { parseWindowStructs("iq,lq"); }));
        CHECK(throws([] { parseWindowStructs("iq+bogus"); }));
        CHECK(throws([] { parseWindowStructs("+"); }));
    }

    // hybrid
    {
        auto p = make("hybrid", t, WindowArgs("mlp_thr=2,miss_mpki=5"));
        CHECK(std::string(p->name()) == "hybrid");
        WindowSample s;
        // empty period (no committed instructions): keep
        CHECK(p->onPeriod(s, 2) == WindowPolicy::Keep);
        s.insts = 1000;
        s.cycles = 1000;
        // ceiling from the hint
        CHECK(p->onSetwin(128, 3) == 1);
        // memory bound, MLP high -> ceiling (not above it)
        s.l2Misses = 20; s.l2Mpki = 20; s.mlp = 4;
        CHECK(p->onPeriod(s, 0) == 1);
        CHECK(p->onPeriod(s, 3) == 1);       // above ceiling -> ceiling
        // memory bound, MLP low -> shrink one step
        s.mlp = 1.1;
        CHECK(p->onPeriod(s, 1) == 0);
        CHECK(p->onPeriod(s, 0) == 0);       // floor
        // ILP regime, full window -> grow, but not above the ceiling
        s.l2Misses = 0; s.l2Mpki = 0; s.fullFrac = 0.5;
        s.robOcc = 60; s.iqOcc = 30; s.lqOcc = 10; s.sqOcc = 10;
        CHECK(p->onPeriod(s, 0) == 1);
        CHECK(p->onPeriod(s, 1) == 1);
        // ILP regime, low occupancy -> shrink
        s.fullFrac = 0; s.robOcc = 10; s.iqOcc = 5; s.lqOcc = 2; s.sqOcc = 2;
        CHECK(p->onPeriod(s, 1) == 0);
        // middle occupancy -> keep
        s.robOcc = 100;
        CHECK(p->onPeriod(s, 1) == WindowPolicy::Keep);
        // drain after a shrink: full cycles in the next period (in the
        // shrunk config) do not grow it back; afterwards they do
        s.robOcc = 10;
        CHECK(p->onPeriod(s, 1) == 0);
        s.robOcc = 60; s.fullFrac = 0.5;
        CHECK(p->onPeriod(s, 0) == WindowPolicy::Keep);
        CHECK(p->onPeriod(s, 0) == 1);
        s.fullFrac = 0;
        // release -> ceiling = largest, MLP-rich -> largest
        CHECK(p->onSetwin(0, 1) == 3);
        s.l2Misses = 20; s.l2Mpki = 20; s.mlp = 4;
        CHECK(p->onPeriod(s, 1) == 3);
        // floor
        auto q = make("hybrid", t, WindowArgs("floor=1"));
        WindowSample m;
        m.insts = 1000; m.l2Misses = 50; m.l2Mpki = 50; m.mlp = 1.0;
        CHECK(q->onPeriod(m, 1) == 1);
        CHECK(q->onSetwin(64, 3) == 1);
    }

    std::printf("%s (%s)\n", fails ? "SOME UNIT TESTS FAILED" : "unit: ALL OK",
                WindowPolicyRegistry::names().size() >= 3 ? "registry ok"
                                                          : "registry?");
    return fails ? 1 : 0;
}
