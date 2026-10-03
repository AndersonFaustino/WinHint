/**
 * @file
 * @brief WinHint: hint-driven window policy (see hint_policy.hh).
 */

#include "cpu/o3/window/hint_policy.hh"

#include <sstream>

namespace gem5
{

namespace o3
{

// See hint_policy.hh.
unsigned
parseWindowStructs(const std::string &spec)
{
    if (spec.empty() || spec == "all")
        return WinAll;
    unsigned mask = 0;
    std::string s = spec;
    for (char &c : s)
        if (c == ':')
            c = '+';
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, '+')) {
        if (item == "rob")
            mask |= WinROB;
        else if (item == "iq")
            mask |= WinIQ;
        else if (item == "lq")
            mask |= WinLQ;
        else if (item == "sq")
            mask |= WinSQ;
        else if (item == "lsq")
            mask |= WinLQ | WinSQ;
        else if (item == "all")
            mask |= WinAll;
        else
            fatal("window_args structs: unknown structure '%s' in '%s' "
                  "(use rob, iq, lq, sq, lsq or all)", item, spec);
    }
    fatal_if(!mask, "window_args structs='%s' selects nothing", spec);
    return mask;
}

// See hint_policy.hh.
HintPolicy::HintPolicy(const WindowPolicyEnv &env)
    : WindowPolicy(env),
      structs(parseWindowStructs(env.args.getString("structs", "all")))
{}

// See hint_policy.hh.
int
HintPolicy::onSetwin(unsigned w, int current)
{
    return table.configForSetwin(w);
}

/** Register HintPolicy as window_policy=hint. */
WINHINT_REGISTER_POLICY("hint", HintPolicy);

} // namespace o3
} // namespace gem5
