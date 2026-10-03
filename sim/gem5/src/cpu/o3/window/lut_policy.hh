/**
 * @file
 * @brief WinHint B5 window policy `lut` (LutPolicy).
 *
 * WinHint B5 (run time): learned counter-based LUT, one lookup per period
 * (docs/interfaces.md §6; offline side in sim/baselines/lut/).
 * window_policy=lut, `window_lut_file=<file>`. See docs/guide/gem5/policies.md.
 */

#ifndef __CPU_O3_WINDOW_LUT_POLICY_HH__
#define __CPU_O3_WINDOW_LUT_POLICY_HH__

#include <string>
#include <vector>

#include "cpu/o3/window/policy.hh"
#include "cpu/o3/window/window_lut.hh"

namespace gem5
{

namespace o3
{

/** B5 policy: one LUT lookup per period, no hysteresis. */
class LutPolicy : public WindowPolicy
{
  public:
    /**
     * @brief Load window_lut_file; fatal() if it is unset, does not parse,
     *        names an unknown feature or holds a config index outside the
     *        table.
     * @param env Policy environment.
     */
    explicit LutPolicy(const WindowPolicyEnv &env);

    /** @return "lut". */
    const char *name() const override { return "lut"; }
    /**
     * @brief Look up the period's features in the LUT.
     * @param s Period sample.
     * @param current Configuration in effect.
     * @return The LUT config (clamped), or Keep if unchanged or the period
     *         is empty.
     */
    int onPeriod(const WindowSample &s, int current) override;

    /**
     * @brief Value of a LUT feature (a window_trace.csv column name or its
     *        short alias) in a sample; false if the name is unknown.
     *
     * Names: ipc, rob_occ[_mean], iq_occ[_mean], lq_occ[_mean],
     * sq_occ[_mean], l1d_mpki, l2_mpki, mlp, branch_mpki, insts, config.
     *
     * @param name Feature name.
     * @param s Period sample.
     * @param current Configuration in effect (feature "config").
     * @param[out] out Feature value.
     * @return Whether name is known.
     */
    static bool feature(const std::string &name, const WindowSample &s,
                        int current, double &out);

    /** @return The loaded LUT. */
    const winhint::WindowLut &lut() const { return lut_; }

  private:
    winhint::WindowLut lut_;   //!< loaded LUT
    std::vector<double> x;     //!< feature vector scratch
};

} // namespace o3
} // namespace gem5

#endif // __CPU_O3_WINDOW_LUT_POLICY_HH__
