/**
 * @file micro_phased.c
 * @brief Recurring phases with different best windows.
 *
 * Condition targeted: the program alternates between three code phases,
 * each a separate function (one WinHint region each), and the sequence
 * recurs:  [gather -> compute -> lowilp] x reps.
 *   gather   independent misses: the largest window is best
 *   compute  cache-resident high ILP: a small window is within a few % of best
 *   lowilp   branch-bound, low occupancy: the smallest window suffices
 * Used by
 *   B4 (Sherwood BBV): three distinct BBV phases, a predictable recurrence,
 *       and the per-phase best configuration found after the first round;
 *   B5 (LUT): the LUT trained on `small` picks the right config per window
 *       on `large`;
 *   B7 (positional adaptation): per-region configs profiled on `small`
 *       transfer to `large`.
 * Every phase visit is ~0.25-0.5 M instructions (several 100 k-instruction
 * BBV intervals). Input: large = 6 rounds (~10.8 M instructions), small = 3
 * rounds of shorter visits with other data (~4.8 M instructions).
 */
#include "micro.h"

/**
 * @brief Gather phase of one round (own WinHint region).
 * @param[in] round Round index (start page round * 389, salt round).
 * @param[in] pages Pages to visit.
 * @return micro_gather() result.
 */
static MICRO_NOINLINE uint64_t phase_gather(unsigned round, unsigned pages)
{
    return micro_gather(round * 389u, pages, round);
}

/**
 * @brief Compute phase of one round (own WinHint region).
 * @param[in] reps Matrix-vector products.
 * @return micro_compute() result.
 */
static MICRO_NOINLINE uint64_t phase_compute(unsigned reps)
{
    return micro_compute(reps);
}

/**
 * @brief Low-ILP phase of one round (own WinHint region).
 * @param[in] round Round index (LCG seed 0x51ed + round).
 * @param[in] iters LCG iterations.
 * @return micro_lowilp() result.
 */
static MICRO_NOINLINE uint64_t phase_lowilp(unsigned round, unsigned iters)
{
    return micro_lowilp(iters, 0x51ed + round);
}

/**
 * @brief Run `rounds` x [gather -> compute -> lowilp] and print the checksum.
 *
 * large: 6 rounds of 448 pages / 20 products / 32 Ki iterations;
 * small: 3 rounds of 320 pages / 14 products / 24 Ki iterations, other seeds.
 * @param[in] argc Argument count.
 * @param[in] argv argv[1] = small|large (default large).
 * @return 0 (exits 2 on a bad argument).
 */
int main(int argc, char **argv)
{
    int large = micro_parse(argc, argv);
    unsigned rounds = large ? 6u : 3u;
    unsigned pages = large ? 448u : 320u;     /* x 64 loads x ~12 insts */
    unsigned mv = large ? 20u : 14u;          /* x ~25 k insts */
    unsigned iters = large ? 32u * 1024u : 24u * 1024u;
    micro_table_init(large ? 3u : 11u);
    micro_compute_init(large ? 3u : 11u);
    uint64_t sum = 0;
    for (unsigned r = 0; r < rounds; r++) {
        sum += phase_gather(r, pages);
        sum = micro_rotl(sum, 5) ^ phase_compute(mv);
        sum += phase_lowilp(r, iters);
    }
    micro_report("micro_phased", large, sum);
    return 0;
}
