/**
 * @file
 * @brief Implementation of the B3 `mlp` window policy (MlpPolicy).
 *
 * WinHint B3: MLP-aware window resizing (Kora et al., MICRO-46 2013).
 *
 * The paper runs the core with a small, fast window ("ILP mode") and
 * enlarges the window level by level when a last-level-cache miss occurs
 * and memory-level parallelism can be exploited ("MLP mode"); the window
 * shrinks back to the ILP level once the memory-intensive phase is over.
 * Enlarging only pays off if the additional entries uncover additional
 * independent misses, so a level is kept only if it raises the MLP.
 *
 * Whole-window, per-period adaptation (see docs/guide/gem5/policies.md for deviations):
 *  - a period is "memory-intensive" if it has >= miss_min long-latency
 *    misses (L2 by default, L1D with miss_level=1);
 *  - memory-intensive and MLP >= mlp_thr: grow one level; the new level
 *    is verified in the next period: if the MLP did not rise by at least
 *    `gain` (relative), revert and cap the window at the old level for a
 *    back-off time that doubles on every failed attempt (MLP-awareness);
 *  - memory-intensive but MLP < mlp_thr (isolated or dependent misses,
 *    e.g. pointer chasing): step down one level towards ILP mode;
 *  - shrink_delay consecutive miss-free periods: return to ILP mode.
 */

#include "cpu/o3/window/mlp_policy.hh"

#include <algorithm>

namespace gem5
{

namespace o3
{

// See mlp_policy.hh.
MlpPolicy::MlpPolicy(const WindowPolicyEnv &env)
    : WindowPolicy(env),
      missLevel((int)env.args.getInt("miss_level", 2)),
      missMin(env.args.getDouble("miss_min", 1)),
      mlpThr(env.args.getDouble("mlp_thr", 1.5)),
      gain(env.args.getDouble("gain", 0.1)),
      ilp(env.table.clamp((int)env.args.getInt("ilp", 0))),
      shrinkDelay((int)env.args.getInt("shrink_delay", 2)),
      backoffBase((int)env.args.getInt("backoff", 8)),
      backoffMax((int)env.args.getInt("backoff_max", 128)),
      backoffLen(backoffBase)
{
    fatal_if(missLevel != 1 && missLevel != 2,
             "mlp: miss_level must be 1 (L1D) or 2 (L2)");
    fatal_if(shrinkDelay < 1, "mlp: shrink_delay must be >= 1");
    fatal_if(backoffBase < 0 || backoffMax < backoffBase,
             "mlp: need 0 <= backoff <= backoff_max");
}

// See mlp_policy.hh.
int
MlpPolicy::onPeriod(const WindowSample &s, int current)
{
    if (s.cycles == 0)
        return Keep;

    const uint64_t misses = missLevel == 1 ? s.l1dMisses : s.l2Misses;
    const bool memPhase = (double)misses >= missMin && misses > 0;
    if (cooldown > 0)
        cooldown--;

    // Verify the level we enlarged to in the previous period.
    if (pendingFrom >= 0) {
        const int from = pendingFrom;
        pendingFrom = -1;
        if (current == from + 1 && memPhase) {
            if (s.mlp >= pendingMlp * (1.0 + gain)) {
                backoffLen = backoffBase;          // level exploited
            } else {
                cooldown = backoffLen;             // not exploited
                capLevel = from;
                backoffLen = std::min(backoffLen * 2, backoffMax);
                quiet = 0;
                return from;
            }
        }
    }

    if (!memPhase) {
        if (++quiet >= shrinkDelay && current != ilp)
            return ilp;
        return Keep;
    }
    quiet = 0;

    if (s.mlp >= mlpThr) {
        const int limit = cooldown > 0 ? capLevel : table.largest();
        if (current < limit) {
            pendingFrom = current;
            pendingMlp = s.mlp;
            return current + 1;
        }
        return Keep;
    }
    if (current > ilp)
        return current - 1;
    return Keep;
}

/** Register MlpPolicy as window_policy=mlp. */
WINHINT_REGISTER_POLICY("mlp", MlpPolicy);

} // namespace o3
} // namespace gem5
