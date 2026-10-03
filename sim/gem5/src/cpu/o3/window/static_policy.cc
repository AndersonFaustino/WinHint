/**
 * @file
 * @brief WinHint B0 window policy `static` (StaticPolicy).
 *
 * WinHint B0: static window. The configuration window_initial is applied
 * at startup and never changes; setwin hints are ignored (region markers
 * are still recorded by the controller).
 */

#include "cpu/o3/window/policy.hh"

namespace gem5
{

namespace o3
{

namespace
{

/** B0 policy: every hook keeps the WindowPolicy defaults (never moves). */
class StaticPolicy : public WindowPolicy
{
  public:
    /** @param env Policy environment (no window_args). */
    explicit StaticPolicy(const WindowPolicyEnv &env) : WindowPolicy(env) {}
    /** @return "static". */
    const char *name() const override { return "static"; }
};

} // anonymous namespace

/** Register StaticPolicy as window_policy=static. */
WINHINT_REGISTER_POLICY("static", StaticPolicy);

} // namespace o3
} // namespace gem5
