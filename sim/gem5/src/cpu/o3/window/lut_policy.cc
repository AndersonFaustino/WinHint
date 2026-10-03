/**
 * @file
 * @brief Implementation of the B5 `lut` window policy (LutPolicy).
 *
 * WinHint B5 (run time): learned LUT lookup (docs/interfaces.md §6).
 *
 * The LUT file is produced offline by
 * sim/baselines/lut/export_lookup_table.py --runtime-lut and loaded once at
 * construction (it replaces the old compiled-in phase_lookup.h and
 * phase_predictor.cc). At the end of every period the per-period features
 * (deltas, the same quantities that window_trace.csv records) are binned
 * with searchsorted(side="left") semantics and the cell's configuration
 * index is applied. No hysteresis: the LUT is the policy.
 */

#include "cpu/o3/window/lut_policy.hh"

namespace gem5
{

namespace o3
{

// See lut_policy.hh.
bool
LutPolicy::feature(const std::string &f, const WindowSample &s, int current,
                   double &out)
{
    if (f == "ipc") out = s.ipc;
    else if (f == "rob_occ" || f == "rob_occ_mean") out = s.robOcc;
    else if (f == "iq_occ" || f == "iq_occ_mean") out = s.iqOcc;
    else if (f == "lq_occ" || f == "lq_occ_mean") out = s.lqOcc;
    else if (f == "sq_occ" || f == "sq_occ_mean") out = s.sqOcc;
    else if (f == "l1d_mpki") out = s.l1dMpki;
    else if (f == "l2_mpki") out = s.l2Mpki;
    else if (f == "mlp") out = s.mlp;
    else if (f == "branch_mpki") out = s.branchMpki;
    else if (f == "insts") out = (double)s.insts;
    else if (f == "config") out = current;
    else return false;
    return true;
}

// See lut_policy.hh.
LutPolicy::LutPolicy(const WindowPolicyEnv &env)
    : WindowPolicy(env)
{
    fatal_if(env.lutFile.empty(),
             "window_policy=lut needs window_lut_file (se.py --window-lut)");
    std::string err;
    fatal_if(!lut_.load(env.lutFile, err), "LUT: %s", err);
    WindowSample probe;
    for (auto &f : lut_.features()) {
        double v;
        fatal_if(!feature(f, probe, 0, v), "LUT '%s': unknown feature '%s'",
                 env.lutFile, f);
    }
    fatal_if((int)lut_.maxConfig() > table.largest(),
             "LUT '%s': config %d out of range (table has %d configs)",
             env.lutFile, lut_.maxConfig(), table.size());
    x.resize(lut_.features().size());
}

// See lut_policy.hh.
int
LutPolicy::onPeriod(const WindowSample &s, int current)
{
    if (s.cycles == 0)
        return Keep;
    const auto &names = lut_.features();
    for (size_t i = 0; i < names.size(); i++)
        feature(names[i], s, current, x[i]);
    int c = clamp((int)lut_.lookup(x.data()));
    return c == current ? Keep : c;
}

/** Register LutPolicy as window_policy=lut. */
WINHINT_REGISTER_POLICY("lut", LutPolicy);

} // namespace o3
} // namespace gem5
