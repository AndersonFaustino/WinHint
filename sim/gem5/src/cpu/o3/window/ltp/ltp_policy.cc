/**
 * @file
 * @brief WinHint B9 window policy `ltp` (LtpPolicy).
 *
 * WinHint B9: the window policy half of Long-Term Parking.
 *
 * Window decisions are static (window_initial), like B0. The policy only
 * (a) registers the name "ltp", (b) consumes the ltp window_args keys
 * (typo guard), and (c) says which structures follow the configuration:
 * by default IQ+LQ+SQ, so LTP runs with the IQ/LSQ of window_initial while
 * the ROB stays at the largest configuration (the paper's premise: a large
 * ROB with small IQ/LSQ). The parking stage itself is created by IEW
 * (ltp_iew.cc) when window_policy == "ltp".
 */

#include "cpu/o3/window/ltp/ltp.hh"
#include "cpu/o3/window/policy.hh"

namespace gem5
{

namespace o3
{

namespace
{

/** Static window plus the LTP structure mask (see the file comment). */
class LtpPolicy : public WindowPolicy
{
  public:
    /**
     * @brief Consume every ltp window_args key via LtpParams::fromArgs()
     *        and keep its `structs` mask.
     * @param env Policy environment.
     */
    explicit LtpPolicy(const WindowPolicyEnv &env)
        : WindowPolicy(env), structs(LtpParams::fromArgs(env.args).structs)
    {}

    /** @return "ltp". */
    const char *name() const override { return "ltp"; }
    /** @return The ltp `structs` mask (default IQ+LQ+SQ). */
    unsigned resizedStructures() const override { return structs; }

  private:
    unsigned structs;  ///< WindowStruct mask from window_args
};

} // anonymous namespace

/** Register LtpPolicy as window_policy=ltp. */
WINHINT_REGISTER_POLICY("ltp", LtpPolicy);

} // namespace o3
} // namespace gem5
