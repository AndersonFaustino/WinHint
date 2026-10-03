/**
 * @file
 * @brief WinHint+HW: hybrid window policy `hybrid` (hint ceiling plus
 *        MLP/occupancy hardware control below it).
 *
 * The compiler hint sets a CEILING; the hardware may only move below it.
 *
 *   setwin(W)   ceiling = smallest config with ROB >= W (W = 0: largest);
 *               the window jumps to the ceiling (trust the hint first).
 *   no hint yet ceiling = largest config (pure hardware behaviour).
 *
 * Every window_period cycles, within [floor, ceiling] (MLP-style override,
 * after Kora et al., MICRO-46 2013, with an occupancy fallback after
 * Ponomarev et al., MICRO-34 2001):
 *   1. long-latency misses present (l2_mpki >= miss_mpki):
 *        MLP >= mlp_thr -> go to the ceiling (the window is exploited);
 *        MLP <  mlp_thr -> shrink one step (isolated/dependent misses: a
 *                          larger window only adds energy).
 *   2. otherwise (ILP regime):
 *        a window structure was full under its cap in more than up_frac
 *        of the cycles -> grow one step (never above the ceiling);
 *        every structure's mean occupancy < down_margin x its size one
 *        config lower -> shrink one step;
 *        else keep.
 *      The period right after a shrink is the drain of the entries above
 *      the new cap (dispatch is gated while they leave): its full cycles
 *      are not taken as a reason to grow (as in the occupancy policy).
 *
 * window_args (defaults): miss_mpki=1.0, mlp_thr=1.5, up_frac=0.05,
 *   down_margin=0.9, floor=0 (lowest config the hardware may pick).
 * MLP is WindowSample::mlp (mean outstanding L1D misses over cycles with at
 * least one outstanding miss). If no L2 is attached (l2Misses always 0)
 * the L1D MPKI is used for the miss test.
 */

#include "cpu/o3/window/policy.hh"

namespace gem5
{

namespace o3
{

namespace
{

/** Hybrid policy; algorithm and window_args in the file comment. */
class HybridPolicy : public WindowPolicy
{
  public:
    /**
     * @brief Read the window_args tunables; floor is clamped to the table
     *        and the ceiling starts at the largest config.
     * @param env Policy environment.
     */
    explicit HybridPolicy(const WindowPolicyEnv &env)
        : WindowPolicy(env),
          missMpki(env.args.getDouble("miss_mpki", 1.0)),
          mlpThr(env.args.getDouble("mlp_thr", 1.5)),
          upFrac(env.args.getDouble("up_frac", 0.05)),
          downMargin(env.args.getDouble("down_margin", 0.9)),
          floorCfg(table.clamp((int)env.args.getInt("floor", 0))),
          ceiling(table.largest())
    {}

    /** @return "hybrid". */
    const char *name() const override { return "hybrid"; }

    /**
     * @brief Set the ceiling to max(floor, configForSetwin(w)) and jump to
     *        it.
     * @param w Requested window in ROB entries (0 = release).
     * @param current Configuration in effect (unused).
     * @return The new ceiling.
     */
    int
    onSetwin(unsigned w, int current) override
    {
        ceiling = std::max(floorCfg, table.configForSetwin(w));
        return ceiling;
    }

    /**
     * @brief Hardware decision within [floor, ceiling] (file comment).
     *
     * Returns the ceiling if cur is above it; Keep if the period committed
     * no instructions.
     *
     * @param s Period sample.
     * @param cur Configuration in effect.
     * @return Configuration index or Keep.
     */
    int
    onPeriod(const WindowSample &s, int cur) override
    {
        if (cur > ceiling)
            return ceiling;
        if (s.insts == 0)
            return Keep;
        if (s.l2Misses)
            sawL2 = true;
        const double mpki = sawL2 ? s.l2Mpki : s.l1dMpki;
        const int lo = std::max(floorCfg, cur - 1);
        const int hi = std::min(ceiling, cur + 1);

        const bool draining = cur == drainCfg;
        drainCfg = -1;

        if (mpki >= missMpki)
            return s.mlp >= mlpThr ? ceiling : shrinkTo(lo, cur);

        if (s.fullFrac > upFrac && !draining)
            return hi;
        if (cur > floorCfg) {
            const int d = cur - 1;
            if (s.robOcc < downMargin * table.rob[d] &&
                s.iqOcc < downMargin * table.iq[d] &&
                s.lqOcc < downMargin * table.lq[d] &&
                s.sqOcc < downMargin * table.sq[d])
                return shrinkTo(d, cur);
        }
        return Keep;
    }

  private:
    /**
     * @brief Return c, remembering it as the drain config if it is a
     *        shrink.
     * @param c Target configuration.
     * @param cur Configuration in effect.
     * @return c.
     */
    int
    shrinkTo(int c, int cur)
    {
        if (c < cur)
            drainCfg = c;
        return c;
    }

    /** window_args miss_mpki, mlp_thr, up_frac, down_margin. */
    const double missMpki, mlpThr, upFrac, downMargin;
    const int floorCfg;    //!< window_args floor (clamped)
    int ceiling;           //!< highest config allowed (last setwin)
    bool sawL2 = false;    //!< an L2 miss was seen: use l2Mpki
    int drainCfg = -1;     //!< config we just shrank to (drain period)
};

} // anonymous namespace

/** Register HybridPolicy as window_policy=hybrid. */
WINHINT_REGISTER_POLICY("hybrid", HybridPolicy);

} // namespace o3
} // namespace gem5
