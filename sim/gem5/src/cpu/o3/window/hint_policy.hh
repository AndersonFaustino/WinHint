/**
 * @file
 * @brief WinHint window policy `hint` (HintPolicy) and the structure-list
 *        parser parseWindowStructs() (shared with the ltp policy).
 *
 * WinHint: hint-driven window policy (WinHint, and B1 via oracle_hinted,
 * B6 jones/jones_full, B7 pgo binaries). See hint_policy.cc.
 */

#ifndef __CPU_O3_WINDOW_HINT_POLICY_HH__
#define __CPU_O3_WINDOW_HINT_POLICY_HH__

#include <string>

#include "cpu/o3/window/policy.hh"

namespace gem5
{

namespace o3
{

/**
 * @brief Parse a structure list "all" or "rob+iq+lq+sq" (any subset, '+' or
 *        ':' separated) into a WindowStruct mask; fatal() on unknown names.
 *
 * Also accepts "lsq" (= lq+sq). An empty spec means "all"; fatal() if the
 * resulting mask is empty.
 *
 * @param spec Structure list.
 * @return Mask of WindowStruct bits.
 */
unsigned parseWindowStructs(const std::string &spec);

/**
 * setwin(W) selects the smallest configuration with ROB >= W (W = 0, or W
 * larger than every configuration, selects the largest). Starts in
 * window_initial (se.py: the largest unless --window-initial is given).
 *
 * window_args:
 *   structs=all|rob+iq+lq+sq   structures that follow the hint; the others
 *                              stay at the largest config. B6 as published
 *                              (IQ only) uses structs=iq.
 */
class HintPolicy : public WindowPolicy
{
  public:
    /**
     * @brief Read window_args `structs` (default "all").
     * @param env Policy environment.
     */
    explicit HintPolicy(const WindowPolicyEnv &env);
    /** @return "hint". */
    const char *name() const override { return "hint"; }
    /**
     * @brief Select WindowTable::configForSetwin(w).
     * @param w Requested window in ROB entries (0 = release).
     * @param current Configuration in effect (unused).
     * @return Configuration index.
     */
    int onSetwin(unsigned w, int current) override;
    /** @return The `structs` mask. */
    unsigned resizedStructures() const override { return structs; }

  protected:
    unsigned structs;  ///< structures that follow the hint (WindowStruct)
};

} // namespace o3
} // namespace gem5

#endif // __CPU_O3_WINDOW_HINT_POLICY_HH__
