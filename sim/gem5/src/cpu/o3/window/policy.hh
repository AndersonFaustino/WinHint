/**
 * @file
 * @brief WinHint: common interface of the O3 window-resizing decision
 *        policies (WindowPolicy, its inputs and the policy registry).
 *
 * INTERFACE VERSION 1 (stable). Policies (static, occupancy, mlp, bbv,
 * lut, hint, hybrid, ltp) are written against this header only. Any later
 * change is listed in the change log below; additions only, no renames.
 *
 * Change log:
 *   v1 (2026-09-30) initial version.
 *
 * Division of labour
 * ------------------
 * A policy only *decides* which window configuration (an index into the
 * ROB/IQ/LQ/SQ table, docs/interfaces.md §3) to use. It never touches the
 * pipeline. The WindowController (controller.hh) owns the mechanism (caps),
 * the sampling, the hint decoding, region_stats.csv / window_trace.csv and
 * the system.cpu.window.* statistics, and calls the policy at these points:
 *
 *   initialConfig()   once at startup
 *   onPeriod()        once per window_period cycles, with per-period DELTAS
 *   onSetwin()        when a setwin(W) hint commits
 *   onRegion()        when a region(id) hint commits
 *   onBranchCommit()  for every committed control instruction, only if
 *                     wantsBranchCommits() (BBV-style policies)
 *
 * Every decision hook returns the requested configuration index, or
 * WindowPolicy::Keep (-1) to keep the current one. Out-of-range indices are
 * clamped by the controller. The controller applies a new configuration
 * immediately: growing takes effect at once, shrinking gates dispatch until
 * the occupancy of each structure is at or below its new cap (no flush).
 *
 * Writing a policy
 * ----------------
 * @code
 *   // my_policy.cc
 *   #include "cpu/o3/window/policy.hh"
 *   namespace gem5 { namespace o3 {
 *   namespace {
 *   class MyPolicy : public WindowPolicy {
 *     public:
 *       explicit MyPolicy(const WindowPolicyEnv &env)
 *           : WindowPolicy(env), thr(env.args.getDouble("thr", 0.5)) {}
 *       const char *name() const override { return "mine"; }
 *       int onPeriod(const WindowSample &s, int cur) override { ... }
 *     private:
 *       double thr;            // all state in members, never statics
 *   };
 *   } // anonymous namespace
 *   WINHINT_REGISTER_POLICY("mine", MyPolicy);
 *   }}
 * @endcode
 *
 * The .cc must be listed in sim/gem5/src/cpu/o3/window/SConscript.
 * After construction the controller calls env.args.checkAllUsed(), so a
 * policy must read every key it accepts in its constructor (typo guard).
 */

#ifndef __CPU_O3_WINDOW_POLICY_HH__
#define __CPU_O3_WINDOW_POLICY_HH__

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "base/logging.hh"
#include "base/types.hh"

namespace gem5
{

namespace o3
{

class CPU;

/** Structures a configuration can cap (bit mask, see resizedStructures). */
enum WindowStruct : unsigned
{
    WinROB = 1u << 0,                          ///< reorder buffer
    WinIQ  = 1u << 1,                          ///< instruction queue
    WinLQ  = 1u << 2,                          ///< load queue
    WinSQ  = 1u << 3,                          ///< store queue
    WinAll = WinROB | WinIQ | WinLQ | WinSQ,   ///< all four structures
};

/**
 * The configuration table: parallel vectors, ascending ROB sizes
 * (window_rob / window_iq / window_lq / window_sq params,
 * docs/interfaces.md §3). Entry i of each vector is the cap of that
 * structure in configuration i.
 */
struct WindowTable
{
    /** Per-configuration caps in entries (ROB, IQ, LQ, SQ). */
    std::vector<unsigned> rob, iq, lq, sq;

    /** @return Number of configurations. */
    int size() const { return (int)rob.size(); }
    /** @return Index of the largest configuration (size() - 1). */
    int largest() const { return size() - 1; }
    /**
     * @brief Clamp a configuration index to [0, largest()].
     * @param c Requested index (may be out of range).
     * @return The clamped index.
     */
    int clamp(int c) const
    {
        return c < 0 ? 0 : (c > largest() ? largest() : c);
    }

    /**
     * @brief Map a setwin(W) hint to a configuration.
     *
     * setwin(W): the smallest config with ROB >= W; W = 0 or W larger
     * than every config selects the largest (docs/interfaces.md §3).
     *
     * @param w Requested window in ROB entries (0 = release).
     * @return The selected configuration index.
     */
    int
    configForSetwin(unsigned w) const
    {
        if (w == 0)
            return largest();
        for (int i = 0; i < size(); i++)
            if (rob[i] >= w)
                return i;
        return largest();
    }
};

/**
 * Per-period measurements. All counts are DELTAS over the period that just
 * ended (never cumulative); means are per cycle of the period.
 */
struct WindowSample
{
    uint64_t cycle = 0;        //!< CPU cycle at the end of the period
    uint64_t cycles = 0;       //!< length of the period in cycles
    uint64_t insts = 0;        //!< committed instructions (incl. hints)
    double ipc = 0;            //!< insts / cycles

    // Mean occupancy (entries) per cycle.
    double robOcc = 0;         //!< mean ROB occupancy (entries)
    double iqOcc = 0;          //!< mean IQ occupancy (entries)
    double lqOcc = 0;          //!< mean LQ occupancy (entries)
    double sqOcc = 0;          //!< mean SQ occupancy (entries)

    // Cache demand misses (BaseCache overallMisses deltas of window_l1d /
    // window_l2; 0 if the cache is not attached).
    uint64_t l1dMisses = 0;    //!< L1D misses in the period
    uint64_t l2Misses = 0;     //!< L2 misses in the period
    double l1dMpki = 0;        //!< L1D misses per 1000 committed insts
    double l2Mpki = 0;         //!< L2 misses per 1000 committed insts

    /** MLP: mean number of outstanding L1D misses (in-flight MSHR
     * targets) over the cycles with at least one outstanding miss
     * (Chou et al.); 0 if none was outstanding in the period. */
    double mlp = 0;
    /** Mean outstanding L1D misses over all cycles of the period. */
    double mlpAll = 0;
    /** Cycles with at least one outstanding L1D miss. */
    uint64_t missCycles = 0;

    uint64_t branches = 0;       //!< committed control instructions
    uint64_t mispredicts = 0;    //!< committed mispredicted branches
    double branchMpki = 0;       //!< mispredicts per 1000 committed insts

    // Dispatch-stall cycles: cycles in which the structure had no free
    // entry under its current cap (i.e. it would block dispatch).
    uint64_t robFullCycles = 0;  //!< cycles the ROB was full under its cap
    uint64_t iqFullCycles = 0;   //!< cycles the IQ was full under its cap
    uint64_t lqFullCycles = 0;   //!< cycles the LQ was full under its cap
    uint64_t sqFullCycles = 0;   //!< cycles the SQ was full under its cap
    uint64_t anyFullCycles = 0;  //!< at least one of the four was full
    double fullFrac = 0;         //!< anyFullCycles / cycles

    int config = 0;            //!< config in effect at the end of the period
    int region = -1;           //!< current region id (-1: none)
    uint64_t setwinHints = 0;  //!< setwin hints committed in the period
    uint64_t regionHints = 0;  //!< region hints committed in the period
};

/**
 * Tunables from the window_args string "k=v,k=v" (values kept as text).
 *
 * Every getter marks its key as used; checkAllUsed() then rejects keys that
 * no getter asked for.
 */
class WindowArgs
{
  public:
    /** Empty argument set. */
    WindowArgs() = default;

    /**
     * @brief Parse a window_args string.
     *
     * Items are separated by ','; whitespace inside an item is removed and
     * empty items are skipped. A later duplicate key overwrites an earlier
     * one. fatal() on an item without '=' or with an empty key.
     *
     * @param spec The "k=v,k=v" string.
     */
    explicit WindowArgs(const std::string &spec)
    {
        std::stringstream ss(spec);
        std::string item;
        while (std::getline(ss, item, ',')) {
            item.erase(std::remove_if(item.begin(), item.end(),
                                      [](unsigned char c) {
                                          return std::isspace(c);
                                      }),
                       item.end());
            if (item.empty())
                continue;
            auto eq = item.find('=');
            fatal_if(eq == std::string::npos || eq == 0,
                     "window_args: expected key=value, got '%s'", item);
            std::string key = item.substr(0, eq);
            vals[key] = item.substr(eq + 1);
            used[key] = false;
        }
    }

    /**
     * @param key Key to look up.
     * @return Whether key was given (does not mark it used).
     */
    bool has(const std::string &key) const { return vals.count(key) != 0; }

    /**
     * @brief Get a value as text and mark the key used.
     * @param key Key to look up.
     * @param dflt Value returned when the key is absent.
     * @return The value, or dflt.
     */
    std::string
    getString(const std::string &key, const std::string &dflt) const
    {
        auto it = vals.find(key);
        if (it == vals.end())
            return dflt;
        used[key] = true;
        return it->second;
    }

    /**
     * @brief Get a value as a number (std::strtod) and mark the key used.
     *
     * fatal() if the whole value does not parse as a number.
     *
     * @param key Key to look up.
     * @param dflt Value returned when the key is absent.
     * @return The value, or dflt.
     */
    double
    getDouble(const std::string &key, double dflt) const
    {
        auto it = vals.find(key);
        if (it == vals.end())
            return dflt;
        used[key] = true;
        char *end = nullptr;
        double v = std::strtod(it->second.c_str(), &end);
        fatal_if(end == it->second.c_str() || *end != '\0',
                 "window_args: '%s=%s' is not a number", key, it->second);
        return v;
    }

    /**
     * @brief Get a value as an integer: getDouble() truncated to long.
     * @param key Key to look up.
     * @param dflt Value returned when the key is absent.
     * @return The value, or dflt.
     */
    long
    getInt(const std::string &key, long dflt) const
    {
        return (long)getDouble(key, (double)dflt);
    }

    /**
     * @brief Alias of getDouble (draft API).
     * @param key Key to look up.
     * @param dflt Value returned when the key is absent.
     * @return The value, or dflt.
     */
    double get(const std::string &key, double dflt) const
    {
        return getDouble(key, dflt);
    }

    /**
     * @brief fatal() on keys no policy consumed (typo protection).
     * @param policy Policy name, used in the error message.
     */
    void
    checkAllUsed(const std::string &policy) const
    {
        for (auto &kv : used)
            fatal_if(!kv.second, "window_args: key '%s' is not used by "
                     "window_policy=%s", kv.first, policy);
    }

  private:
    std::map<std::string, std::string> vals;  ///< key -> raw value text
    mutable std::map<std::string, bool> used; ///< key -> read by a getter
};

/** Everything a policy may need at construction time. */
struct WindowPolicyEnv
{
    const WindowTable &table;  //!< the configuration table
    int initial;               //!< window_initial (already clamped)
    const WindowArgs &args;    //!< window_args
    std::string lutFile;       //!< window_lut_file
    uint64_t period;           //!< window_period in cycles
    std::string outdir;        //!< gem5 outdir (for policy-private dumps)
    CPU *cpu;                  //!< opaque; only for policies that need
                               //!< pipeline hooks of their own (ltp)
};

/**
 * Base class of every window-resizing decision policy.
 *
 * The defaults implement a policy that keeps the initial configuration
 * forever. Decision hooks return a configuration index or Keep; see the
 * file comment for when the controller calls each hook.
 */
class WindowPolicy
{
  public:
    /** Hook return value: keep the current configuration. */
    static constexpr int Keep = -1;

    /** @param env Construction-time environment (table, args, ...). */
    explicit WindowPolicy(const WindowPolicyEnv &env)
        : table(env.table), initial(env.initial)
    {}
    virtual ~WindowPolicy() = default;

    /** @return The registered policy name (e.g. "mlp"). */
    virtual const char *name() const = 0;

    /**
     * @brief Configuration applied at startup.
     * @return Configuration index (default: window_initial).
     */
    virtual int initialConfig() const { return initial; }

    /**
     * @brief End of a sampling period.
     * @param s Per-period deltas of the period that just ended.
     * @param current Configuration in effect.
     * @return Requested configuration index, or Keep.
     */
    virtual int onPeriod(const WindowSample &s, int current) { return Keep; }

    /**
     * @brief A setwin(W) hint committed (W in entries, 0 = release).
     * @param w Requested window in ROB entries.
     * @param current Configuration in effect.
     * @return Requested configuration index, or Keep.
     */
    virtual int onSetwin(unsigned w, int current) { return Keep; }

    /**
     * @brief A region(id) hint committed.
     * @param id Region id (0..63).
     * @param current Configuration in effect.
     * @return Requested configuration index, or Keep.
     */
    virtual int onRegion(int id, int current) { return Keep; }

    /**
     * @brief Opt in to onBranchCommit (costs a virtual call per branch).
     * @return true to receive onBranchCommit() calls.
     */
    virtual bool wantsBranchCommits() const { return false; }

    /**
     * @brief A control instruction committed.
     *
     * blockInsts = instructions committed since the previous control
     * instruction, including this one (the basic-block length for a BBV).
     *
     * @param pc PC of the control instruction.
     * @param target PC of the next committed instruction (at exit, for a
     *        branch with no successor: the fall-through PC).
     * @param taken Whether target differs from the fall-through PC.
     * @param mispredicted Whether the branch was mispredicted.
     * @param blockInsts Basic-block length in instructions.
     */
    virtual void
    onBranchCommit(Addr pc, Addr target, bool taken, bool mispredicted,
                   uint64_t blockInsts)
    {}

    /**
     * @brief Which structures follow the selected configuration; the others
     *        stay at the largest configuration (e.g. B6 IQ-only: WinIQ).
     * @return Bit mask of WindowStruct values.
     */
    virtual unsigned resizedStructures() const { return WinAll; }

    /** Called once at the end of simulation (optional dumps). */
    virtual void finish() {}

  protected:
    const WindowTable &table;  ///< the configuration table
    const int initial;         ///< window_initial (already clamped)

    /**
     * @brief Clamp a configuration index to the table.
     * @param c Requested index.
     * @return c clamped to [0, table.largest()].
     */
    int clamp(int c) const { return table.clamp(c); }
};

/** Name -> factory registry; each policy .cc self-registers. */
class WindowPolicyRegistry
{
  public:
    /** Builds a policy instance from its environment. */
    using Factory =
        std::function<std::unique_ptr<WindowPolicy>(const WindowPolicyEnv &)>;

    /**
     * @brief Register a factory under a name; panic() on a duplicate.
     * @param name Policy name (window_policy value).
     * @param f Factory for the policy.
     * @return Always true (lets registration initialise a static bool).
     */
    static bool
    add(const std::string &name, Factory f)
    {
        auto &m = map();
        panic_if(m.count(name), "window policy '%s' registered twice", name);
        m[name] = std::move(f);
        return true;
    }

    /**
     * @param name Policy name.
     * @return Whether a policy is registered under name.
     */
    static bool has(const std::string &name) { return map().count(name); }

    /** @return All registered names, in ascending order. */
    static std::vector<std::string>
    names()
    {
        std::vector<std::string> v;
        for (auto &kv : map())
            v.push_back(kv.first);
        return v;
    }

    /**
     * @brief Instantiate the policy registered as name.
     *
     * fatal() if the name is unknown (the message lists the known names).
     *
     * @param name Policy name.
     * @param env Environment passed to the factory.
     * @return The new policy.
     */
    static std::unique_ptr<WindowPolicy>
    create(const std::string &name, const WindowPolicyEnv &env)
    {
        auto it = map().find(name);
        if (it == map().end()) {
            std::string known;
            for (auto &n : names())
                known += (known.empty() ? "" : ", ") + n;
            fatal("window_policy='%s' is unknown (built-in: %s)", name,
                  known);
        }
        return it->second(env);
    }

  private:
    /**
     * @brief The name -> factory map (function-local static, so it is
     *        initialised before any static registration uses it).
     * @return Reference to the map.
     */
    static std::map<std::string, Factory> &
    map()
    {
        static std::map<std::string, Factory> m;
        return m;
    }
};

/** Token-paste helper for WINHINT_POLICY_CAT (no argument expansion). */
#define WINHINT_POLICY_CAT2(a, b) a##b
/** Paste a and b after macro-expanding them (e.g. __LINE__). */
#define WINHINT_POLICY_CAT(a, b) WINHINT_POLICY_CAT2(a, b)

/**
 * @brief Register CLASS (constructible from const WindowPolicyEnv &) as
 *        NAME. Use at namespace scope inside gem5::o3.
 *
 * Expands to a static bool initialised by WindowPolicyRegistry::add(); the
 * variable name embeds __LINE__, so use at most once per line.
 *
 * @param NAME Policy name (string literal).
 * @param CLASS WindowPolicy subclass.
 */
#define WINHINT_REGISTER_POLICY(NAME, CLASS)                                \
    [[maybe_unused]] static const bool                                      \
    WINHINT_POLICY_CAT(winhintPolicyReg_, __LINE__) =                       \
        ::gem5::o3::WindowPolicyRegistry::add(                              \
            NAME, [](const ::gem5::o3::WindowPolicyEnv &env)                \
                -> std::unique_ptr<::gem5::o3::WindowPolicy> {              \
                return std::make_unique<CLASS>(env);                        \
            })

} // namespace o3
} // namespace gem5

#endif // __CPU_O3_WINDOW_POLICY_HH__
