/**
 * @file
 * @brief Implementation of the B2 `occupancy` window policy
 *        (OccupancyPolicy).
 *
 * WinHint B2: occupancy-driven resizing (Ponomarev et al., MICRO-34 2001).
 *
 * The paper partitions IQ, ROB and LSQ and resizes each one independently:
 *  - downsizing: the occupancy of each resource is averaged over an
 *    "update period" (2048 cycles in the paper); at its end, if
 *    (current size - average occupancy) >= one partition, the resource
 *    loses one partition;
 *  - upsizing: an overflow counter counts the cycles in which dispatch is
 *    blocked because the resource has no free entry; as soon as it exceeds
 *    a threshold within the update period the resource gains one
 *    partition and the counters restart.
 *
 * Whole-window adaptation (same configuration table and mechanism as every
 * other policy): a "partition" is one step of the table (ROB/IQ/LQ/SQ move
 * together). Grow one step as soon as ANY structure overflows (checked at
 * every sample period, i.e. window_period granularity); shrink one step at
 * the end of an update period only if EVERY structure passes the
 * downsizing test against the next smaller configuration.
 */

#include "cpu/o3/window/occupancy_policy.hh"

#include <algorithm>

namespace gem5
{

namespace o3
{

// See occupancy_policy.hh.
OccupancyPolicy::OccupancyPolicy(const WindowPolicyEnv &env)
    : WindowPolicy(env),
      update((int)env.args.getInt("update", 2)),
      upFrac(env.args.getDouble("up_frac", 0.05)),
      downFactor(env.args.getDouble("down_factor", 1.0)),
      periodLen(env.period)
{
    fatal_if(update < 1, "occupancy: update must be >= 1");
    fatal_if(upFrac < 0 || upFrac > 1, "occupancy: up_frac must be in "
             "[0, 1]");
    fatal_if(downFactor < 0, "occupancy: down_factor must be >= 0");
}

// See occupancy_policy.hh.
void
OccupancyPolicy::reset(int current)
{
    cfg = current;
    periods = 0;
    cycles = 0;
    std::fill(std::begin(occ), std::end(occ), 0.0);
    std::fill(std::begin(full), std::end(full), 0);
}

// See occupancy_policy.hh.
int
OccupancyPolicy::onPeriod(const WindowSample &s, int current)
{
    if (current != cfg)            // resized (by us or anything else)
        reset(current);
    if (s.cycles == 0)
        return Keep;

    periods++;
    cycles += s.cycles;
    occ[0] += s.robOcc * s.cycles;
    occ[1] += s.iqOcc * s.cycles;
    occ[2] += s.lqOcc * s.cycles;
    occ[3] += s.sqOcc * s.cycles;
    // The first period after a downsize contains the drain of the
    // entries above the new cap, which gates dispatch in our mechanism but
    // not in the paper's (a partition is turned off only once it is
    // empty): its full cycles are not counted as overflow.
    if (!afterShrink) {
        full[0] += s.robFullCycles;
        full[1] += s.iqFullCycles;
        full[2] += s.lqFullCycles;
        full[3] += s.sqFullCycles;
    }
    afterShrink = false;

    // Upsizing: overflow counter against a threshold expressed as a
    // fraction of the update period (nominal length, so the test can
    // fire before the update period ends, as in the paper).
    uint64_t plen = periodLen ? periodLen : s.cycles;
    double thr = upFrac * (double)plen * update;
    uint64_t worst = *std::max_element(std::begin(full), std::end(full));
    if (current < table.largest() && (double)worst > thr) {
        reset(-1);
        return current + 1;
    }

    if (periods < update)
        return Keep;

    // Downsizing at the end of the update period.
    int target = Keep;
    if (current > 0) {
        const int lo = current - 1;
        const std::vector<unsigned> *caps[4] = {
            &table.rob, &table.iq, &table.lq, &table.sq};
        bool shrink = true;
        for (int k = 0; k < 4 && shrink; k++) {
            double mean = occ[k] / (double)cycles;
            double cap = (*caps[k])[current];
            double step = cap - (double)(*caps[k])[lo];
            shrink = (cap - mean) >= downFactor * step;
        }
        if (shrink) {
            target = lo;
            afterShrink = true;
        }
    }
    reset(target == Keep ? current : -1);
    return target;
}

/** Register OccupancyPolicy as window_policy=occupancy. */
WINHINT_REGISTER_POLICY("occupancy", OccupancyPolicy);

} // namespace o3
} // namespace gem5
