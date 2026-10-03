/**
 * @file
 * @brief Implementation of the B4 `bbv` window policy (BbvPolicy).
 *
 * WinHint B4: BBV phase tracking and prediction (Sherwood, Sair, Calder,
 * ISCA 2003) driving whole-window resizing.
 *
 * Phase tracker (as in the paper's hardware):
 *  - accumulator table: `buckets` counters; each committed control
 *    instruction adds the length of the basic block it ends to the
 *    counter selected by a hash of its PC;
 *  - at the end of each interval (`interval_insts` committed
 *    instructions, checked at window_period boundaries) the vector is
 *    normalized and quantized to `sig_bits` bits per bucket (the
 *    signature) and compared with the signatures in the past footprint
 *    table (`max_phases` entries, LRU) by Manhattan distance; it joins the
 *    closest phase if the distance is below `thr` (on the scale of
 *    normalized vectors, whose maximum distance is 2.0; thr=0.25 is 12.5%
 *    of the maximum), else a new phase id is allocated.
 * Phase predictor: run-length-encoded Markov table indexed by
 *  (last phase id, min(run length, run_max)), each entry a predicted next
 *  phase id with a saturating confidence counter (replaced only at
 *  confidence 0); last-value fallback when there is no entry.
 * Per-phase configuration (the "phase-based adaptation" use case): learned
 *  online. For every phase, each configuration is tried for `explore`
 *  intervals (largest first); afterwards the smallest configuration whose
 *  mean IPC is within `tol` of the best mean is used. The configuration
 *  for the next interval is the one of the *predicted* next phase.
 */

#include "cpu/o3/window/bbv_policy.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

namespace gem5
{

namespace o3
{

// See bbv_policy.hh.
BbvPolicy::BbvPolicy(const WindowPolicyEnv &env)
    : WindowPolicy(env),
      nBuckets((int)env.args.getInt("buckets", 32)),
      sigBits((int)env.args.getInt("sig_bits", 6)),
      thr(env.args.getDouble("thr", 0.25)),
      maxPhases((int)env.args.getInt("max_phases", 32)),
      intervalInsts((uint64_t)env.args.getInt("interval_insts", 100000)),
      explore((int)env.args.getInt("explore", 1)),
      tol(env.args.getDouble("tol", 0.02)),
      runMax((int)env.args.getInt("run_max", 15)),
      confMax((int)env.args.getInt("conf_max", 3)),
      outdir(env.outdir)
{
    fatal_if(nBuckets < 1 || nBuckets > 4096, "bbv: buckets out of range");
    fatal_if(sigBits < 1 || sigBits > 24, "bbv: sig_bits must be 1..24");
    fatal_if(maxPhases < 1, "bbv: max_phases must be >= 1");
    fatal_if(explore < 1, "bbv: explore must be >= 1");
    fatal_if(runMax < 0 || runMax > 255, "bbv: run_max must be 0..255");
    fatal_if(confMax < 0, "bbv: conf_max must be >= 0");
    fatal_if(thr < 0 || tol < 0, "bbv: thr and tol must be >= 0");
    bbv.assign(nBuckets, 0);
}

// See bbv_policy.hh.
void
BbvPolicy::onBranchCommit(Addr pc, Addr target, bool taken,
                          bool mispredicted, uint64_t blockInsts)
{
    uint64_t h = ((uint64_t)pc >> 1) * 0x9E3779B97F4A7C15ULL;
    bbv[(h >> 32) % (uint64_t)nBuckets] += blockInsts;
}

// See bbv_policy.hh.
std::vector<uint32_t>
BbvPolicy::signature() const
{
    const double scale = (double)((1u << sigBits) - 1);
    double tot = 0;
    for (auto v : bbv)
        tot += (double)v;
    std::vector<uint32_t> sig(nBuckets, 0);
    if (tot > 0)
        for (int i = 0; i < nBuckets; i++)
            sig[i] = (uint32_t)std::lround((double)bbv[i] / tot * scale);
    return sig;
}

// See bbv_policy.hh.
int
BbvPolicy::classify(const std::vector<uint32_t> &sig)
{
    const double scale = (double)((1u << sigBits) - 1);
    int best = -1;
    double bestDist = std::numeric_limits<double>::max();
    for (auto &kv : phases) {
        uint64_t d = 0;
        for (int i = 0; i < nBuckets; i++)
            d += (uint64_t)std::abs((long)kv.second.sig[i] - (long)sig[i]);
        double dist = (double)d / scale;
        if (dist < bestDist) {
            bestDist = dist;
            best = kv.first;
        }
    }
    if (best >= 0 && bestDist < thr)
        return best;

    if ((int)phases.size() >= maxPhases) {
        int victim = phases.begin()->first;
        for (auto &kv : phases)
            if (kv.second.lastUse < phases[victim].lastUse)
                victim = kv.first;
        evict(victim);
    }
    int id = nextId++;
    phases[id].sig = sig;
    return id;
}

// See bbv_policy.hh.
void
BbvPolicy::evict(int victim)
{
    phases.erase(victim);
    for (auto it = markov.begin(); it != markov.end();) {
        if ((int)(it->first >> 8) == victim || it->second.next == victim)
            it = markov.erase(it);
        else
            ++it;
    }
    if (lastPhase == victim) {
        lastPhase = -1;
        runLen = 0;
    }
}

// See bbv_policy.hh.
void
BbvPolicy::learn(int phase, int config, double ipc)
{
    Phase &ph = phases[phase];
    if (ph.n.empty()) {
        ph.ipcSum.assign(table.size(), 0.0);
        ph.n.assign(table.size(), 0);
    }
    if (config < 0 || config >= table.size())
        return;
    ph.ipcSum[config] += ipc;
    ph.n[config]++;
}

// See bbv_policy.hh.
int
BbvPolicy::choose(int phase) const
{
    auto it = phases.find(phase);
    if (it == phases.end() || it->second.n.empty())
        return table.largest();
    const Phase &ph = it->second;
    for (int c = table.largest(); c >= 0; c--)          // explore
        if (ph.n[c] < (uint64_t)explore)
            return c;
    double best = 0;
    for (int c = 0; c < table.size(); c++)
        best = std::max(best, ph.ipcSum[c] / ph.n[c]);
    for (int c = 0; c < table.size(); c++)              // exploit
        if (ph.ipcSum[c] / ph.n[c] >= (1.0 - tol) * best)
            return c;
    return table.largest();
}

// See bbv_policy.hh.
int
BbvPolicy::learnedConfig(int phase) const
{
    auto it = phases.find(phase);
    if (it == phases.end() || it->second.n.empty())
        return -1;
    for (auto n : it->second.n)
        if (n < (uint64_t)explore)
            return -1;
    return choose(phase);
}

// See bbv_policy.hh.
int
BbvPolicy::onPeriod(const WindowSample &s, int current)
{
    if (ivConfig < 0)
        ivConfig = current;
    else if (current != ivConfig)
        ivMixed = true;
    ivInsts += s.insts;
    ivCycles += s.cycles;
    if (ivInsts < intervalInsts || ivCycles == 0)
        return Keep;

    // End of interval: build the signature and classify.
    uint64_t tot = 0;
    for (auto v : bbv)
        tot += v;
    int target = Keep;
    if (tot > 0) {
        int phase = classify(signature());
        Phase &ph = phases[phase];
        ph.lastUse = ++useClock;
        ph.visits++;
        intervals++;
        if (predicted == phase)
            predCorrect++;

        // Credit this interval's IPC to (phase, config in effect).
        if (!ivMixed)
            learn(phase, ivConfig, (double)ivInsts / (double)ivCycles);

        // RLE-Markov update: (last phase, run length) -> this phase.
        if (lastPhase >= 0) {
            MarkovEntry &e = markov[key(lastPhase, runLen)];
            if (e.next == phase) {
                e.conf = std::min(e.conf + 1, confMax);
            } else if (e.conf > 0) {
                e.conf--;
            } else {
                e.next = phase;
                e.conf = 0;
            }
        }
        runLen = (phase == lastPhase) ? std::min(runLen + 1, runMax) : 0;
        lastPhase = phase;

        // Predict the next interval's phase (last-value fallback).
        auto it = markov.find(key(phase, runLen));
        predicted = (it != markov.end() && it->second.next >= 0 &&
                     phases.count(it->second.next))
                        ? it->second.next : phase;
        int cfg = choose(predicted);
        target = cfg == current ? Keep : cfg;
    }

    std::fill(bbv.begin(), bbv.end(), 0);
    ivInsts = ivCycles = 0;
    ivConfig = target == Keep ? current : target;
    ivMixed = false;
    return target;
}

// See bbv_policy.hh.
void
BbvPolicy::finish()
{
    if (outdir.empty())
        return;
    std::ofstream out(outdir + "/bbv_phases.csv");
    if (!out)
        return;
    out << "# intervals=" << intervals << " predicted_correct="
        << predCorrect << " phases_allocated=" << nextId << "\n";
    out << "phase,visits,learned_config";
    for (int c = 0; c < table.size(); c++)
        out << ",ipc_c" << c << ",n_c" << c;
    out << "\n";
    for (auto &kv : phases) {
        const Phase &ph = kv.second;
        out << kv.first << "," << ph.visits << "," << learnedConfig(kv.first);
        for (int c = 0; c < table.size(); c++) {
            bool has = !ph.n.empty() && ph.n[c] > 0;
            out << "," << (has ? ph.ipcSum[c] / ph.n[c] : 0.0) << ","
                << (ph.n.empty() ? 0 : ph.n[c]);
        }
        out << "\n";
    }
}

/** Register BbvPolicy as window_policy=bbv. */
WINHINT_REGISTER_POLICY("bbv", BbvPolicy);

} // namespace o3
} // namespace gem5
