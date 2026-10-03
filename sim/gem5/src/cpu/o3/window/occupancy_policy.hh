/**
 * @file
 * @brief WinHint B2 window policy `occupancy` (OccupancyPolicy).
 *
 * WinHint B2: occupancy-driven resizing of the whole window
 * (Ponomarev, Kucuk, Ghose, "Reducing Power Requirements of Instruction
 * Scheduling Through Dynamic Allocation of Multiple Datapath Resources",
 * MICRO-34, 2001). window_policy=occupancy. See docs/guide/gem5/policies.md.
 */

#ifndef __CPU_O3_WINDOW_OCCUPANCY_POLICY_HH__
#define __CPU_O3_WINDOW_OCCUPANCY_POLICY_HH__

#include <cstdint>

#include "cpu/o3/window/policy.hh"

namespace gem5
{

namespace o3
{

/**
 * B2 policy: grow one step when any structure overflows, shrink one step
 * at the end of an update period when every structure is underused
 * (algorithm in occupancy_policy.cc).
 */
class OccupancyPolicy : public WindowPolicy
{
  public:
    /**
     * @brief Read and validate the window_args tunables.
     *
     * Keys (defaults): update (2, >= 1), up_frac (0.05, in [0, 1]),
     * down_factor (1.0, >= 0).
     *
     * @param env Policy environment (env.period is the nominal period).
     */
    explicit OccupancyPolicy(const WindowPolicyEnv &env);

    /** @return "occupancy". */
    const char *name() const override { return "occupancy"; }
    /**
     * @brief Accumulate the period; grow on overflow, else test for a
     *        shrink once `update` periods have been accumulated.
     * @param s Period sample.
     * @param current Configuration in effect.
     * @return Configuration index or Keep.
     */
    int onPeriod(const WindowSample &s, int current) override;

  private:
    /**
     * @brief Restart the update-period accumulators.
     * @param current Config they belong to (-1: force a reset on the next
     *        period).
     */
    void reset(int current);

    // Tunables (window_args).
    int update;          //!< sample periods per update period
    double upFrac;       //!< overflow threshold, fraction of update cycles
    double downFactor;   //!< shrink if cap - occ >= downFactor * step

    // Accumulators over the current update period.
    int cfg = -1;        //!< config the accumulators belong to
    int periods = 0;     //!< sample periods accumulated
    uint64_t cycles = 0; //!< cycles accumulated
    double occ[4] = {0, 0, 0, 0};       //!< sum of occ * cycles
    uint64_t full[4] = {0, 0, 0, 0};    //!< overflow (full) cycles
    uint64_t periodLen = 0;             //!< nominal period length
    bool afterShrink = false;           //!< next period is a drain
};

} // namespace o3
} // namespace gem5

#endif // __CPU_O3_WINDOW_OCCUPANCY_POLICY_HH__
