/**
 * @file
 * @brief WinHint B3 window policy `mlp` (MlpPolicy).
 *
 * WinHint B3: MLP-aware dynamic instruction window resizing
 * (Kora, Yamaguchi, Ando, "MLP-Aware Dynamic Instruction Window Resizing
 * for Adaptively Exploiting Both ILP and MLP", MICRO-46, 2013).
 * window_policy=mlp. See docs/guide/gem5/policies.md.
 */

#ifndef __CPU_O3_WINDOW_MLP_POLICY_HH__
#define __CPU_O3_WINDOW_MLP_POLICY_HH__

#include <cstdint>

#include "cpu/o3/window/policy.hh"

namespace gem5
{

namespace o3
{

/**
 * B3 policy: stays in a small ILP-mode configuration and grows level by
 * level while long-latency misses show MLP, keeping a level only if it
 * raises the MLP (algorithm in mlp_policy.cc).
 */
class MlpPolicy : public WindowPolicy
{
  public:
    /**
     * @brief Read and validate the window_args tunables.
     *
     * Keys (defaults): miss_level (2; 1 or 2), miss_min (1), mlp_thr (1.5),
     * gain (0.1), ilp (0, clamped to the table), shrink_delay (2, >= 1),
     * backoff (8), backoff_max (128; 0 <= backoff <= backoff_max).
     *
     * @param env Policy environment.
     */
    explicit MlpPolicy(const WindowPolicyEnv &env);

    /** @return "mlp". */
    const char *name() const override { return "mlp"; }
    /**
     * @brief Per-period MLP-aware decision (see mlp_policy.cc).
     * @param s Period sample.
     * @param current Configuration in effect.
     * @return Configuration index or Keep.
     */
    int onPeriod(const WindowSample &s, int current) override;

  private:
    // Tunables (window_args).
    int missLevel;         //!< 2: L2 misses are long-latency; 1: L1D
    double missMin;        //!< long-latency misses per period to trigger
    double mlpThr;         //!< MLP needed to call the misses parallel
    double gain;           //!< min relative MLP gain to keep a level
    int ilp;               //!< ILP-mode config (smallest level)
    int shrinkDelay;       //!< miss-free periods before ILP mode
    int backoffBase, backoffMax; //!< back-off length: initial, maximum

    // State.
    int quiet = 0;         //!< consecutive periods without misses
    int pendingFrom = -1;  //!< level we grew from last period (-1: none)
    double pendingMlp = 0; //!< MLP measured at pendingFrom
    int cooldown = 0;      //!< periods left in which growth is capped
    int capLevel = 0;      //!< max level while cooldown > 0
    int backoffLen;        //!< back-off for the next failed attempt
};

} // namespace o3
} // namespace gem5

#endif // __CPU_O3_WINDOW_MLP_POLICY_HH__
