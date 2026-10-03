/**
 * @file
 * @brief WinHint B4 window policy `bbv`: BBV phase tracking and prediction.
 *
 * WinHint B4: basic-block-vector phase tracking and prediction
 * (Sherwood, Sair, Calder, "Phase Tracking and Prediction", ISCA 2003),
 * with a per-phase best window configuration learned online.
 * window_policy=bbv. See docs/guide/gem5/policies.md.
 */

#ifndef __CPU_O3_WINDOW_BBV_POLICY_HH__
#define __CPU_O3_WINDOW_BBV_POLICY_HH__

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "cpu/o3/window/policy.hh"

namespace gem5
{

namespace o3
{

/**
 * B4 policy: classifies fixed-length instruction intervals into phases by
 * their basic-block vector, predicts the next phase with an RLE-Markov
 * table and applies the configuration learned for the predicted phase.
 * Algorithm details in bbv_policy.cc and docs/guide/gem5/policies.md.
 */
class BbvPolicy : public WindowPolicy
{
  public:
    /**
     * @brief Read and validate the window_args tunables.
     *
     * Keys (defaults): buckets (32, 1..4096), sig_bits (6, 1..24),
     * thr (0.25), max_phases (32), interval_insts (100000), explore (1),
     * tol (0.02), run_max (15, 0..255), conf_max (3). fatal() if out of
     * range.
     *
     * @param env Policy environment.
     */
    explicit BbvPolicy(const WindowPolicyEnv &env);

    /** @return "bbv". */
    const char *name() const override { return "bbv"; }
    /** @return true: the BBV is built from branch commits. */
    bool wantsBranchCommits() const override { return true; }
    /**
     * @brief Add blockInsts to the accumulator bucket selected by a hash of
     *        pc. The other arguments are unused.
     */
    void onBranchCommit(Addr pc, Addr target, bool taken, bool mispredicted,
                        uint64_t blockInsts) override;
    /**
     * @brief Accumulate the period into the current interval; at the end
     *        of an interval classify it, train the predictor and the
     *        per-phase IPC table, and return the config of the predicted
     *        next phase.
     * @param s Period sample.
     * @param current Configuration in effect.
     * @return New configuration, or Keep (mid-interval, empty BBV or no
     *         change).
     */
    int onPeriod(const WindowSample &s, int current) override;
    /** @brief Write bbv_phases.csv (per-phase visits, learned config and
     *         mean IPC / interval count per config) to the outdir. */
    void finish() override;

    // Introspection (unit tests, finish() dump).
    /** @return Number of phases in the footprint table. */
    int numPhases() const { return (int)phases.size(); }
    /** @return Phase id of the last classified interval (-1: none). */
    int lastPhaseId() const { return lastPhase; }
    /** @return Predicted phase id for the next interval (-1: none). */
    int predictedPhaseId() const { return predicted; }
    /** @return Intervals classified so far. */
    uint64_t intervalsSeen() const { return intervals; }
    /** @return Intervals whose phase matched the prediction. */
    uint64_t correctPredictions() const { return predCorrect; }
    /**
     * @brief Learned config of a phase (-1: unknown phase or still
     *        exploring).
     * @param phase Phase id.
     * @return Configuration index or -1.
     */
    int learnedConfig(int phase) const;

  private:
    /** Entry of the past footprint table. */
    struct Phase
    {
        std::vector<uint32_t> sig;     //!< quantized signature
        std::vector<double> ipcSum;    //!< per config
        std::vector<uint64_t> n;       //!< intervals per config
        uint64_t lastUse = 0;          //!< LRU timestamp (useClock)
        uint64_t visits = 0;           //!< intervals classified here
    };

    /** RLE-Markov predictor entry. */
    struct MarkovEntry
    {
        int next = -1;                 //!< predicted next phase id
        int conf = 0;                  //!< saturating, 0..confMax
    };

    /**
     * @brief Normalise the accumulator and quantise each bucket to sigBits.
     * @return The signature (all zero if the accumulator is empty).
     */
    std::vector<uint32_t> signature() const;
    /**
     * @brief Find the closest phase (Manhattan distance / quantisation
     *        scale) below thr, or allocate a new phase id (evicting the LRU
     *        phase when the table is full).
     * @param sig Interval signature.
     * @return Phase id.
     */
    int classify(const std::vector<uint32_t> &sig);
    /**
     * @brief Drop a phase and every Markov entry from or to it.
     * @param victim Phase id.
     */
    void evict(int victim);
    /**
     * @brief Credit an interval's IPC to (phase, config).
     * @param phase Phase id.
     * @param config Config in effect for the whole interval (ignored if
     *        out of range).
     * @param ipc Interval IPC.
     */
    void learn(int phase, int config, double ipc);
    /**
     * @brief Config for a phase: the largest config not yet tried
     *        `explore` times, else the smallest config with mean IPC
     *        >= (1 - tol) * best mean.
     * @param phase Phase id.
     * @return Configuration index (largest for an unknown phase).
     */
    int choose(int phase) const;
    /**
     * @brief Markov table key.
     * @param phase Last phase id.
     * @param run Run length (0..255).
     * @return (phase << 8) | run.
     */
    static uint64_t key(int phase, int run) { return ((uint64_t)phase << 8) | run; }

    // Tunables (window_args).
    int nBuckets;              ///< buckets: accumulator size
    int sigBits;               ///< sig_bits: bits per signature bucket
    double thr;                ///< thr: max distance to join a phase
    int maxPhases;             ///< max_phases: footprint table size
    uint64_t intervalInsts;    ///< interval_insts: interval length
    int explore;               ///< explore: intervals per config to try
    double tol;                ///< tol: relative IPC tolerance
    int runMax;                ///< run_max: run-length saturation
    int confMax;               ///< conf_max: Markov confidence maximum

    std::string outdir;        ///< where finish() writes bbv_phases.csv

    // Accumulator (the hardware's BBV counters) and interval state.
    std::vector<uint64_t> bbv;          ///< accumulator table
    /** Instructions and cycles of the current interval so far. */
    uint64_t ivInsts = 0, ivCycles = 0;
    int ivConfig = -1;             //!< config at the start of the interval
    bool ivMixed = false;          //!< config changed inside the interval

    // Phase table (past footprint table) and RLE-Markov predictor.
    std::map<int, Phase> phases;             ///< phase id -> entry
    std::map<uint64_t, MarkovEntry> markov;  ///< key() -> prediction
    /** Next phase id, last phase, its run length, predicted next phase. */
    int nextId = 0, lastPhase = -1, runLen = 0, predicted = -1;
    /** LRU clock and prediction counters. */
    uint64_t useClock = 0, intervals = 0, predCorrect = 0;
};

} // namespace o3
} // namespace gem5

#endif // __CPU_O3_WINDOW_BBV_POLICY_HH__
