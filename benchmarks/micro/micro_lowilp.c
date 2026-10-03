/**
 * @file micro_lowilp.c
 * @brief Low-ILP, branch-bound code with low window occupancy.
 *
 * Condition targeted: a serial multiply chain plus ~50% mispredicted
 * data-dependent branches; frequent squashes keep the window mostly empty.
 *   B2 (Ponomarev): large occupancy (power) savings with small IPC loss.
 *   B3 (Kora): no misses -> stays in ILP mode.
 *   B6 (Jones): IQ can shrink with negligible IPC loss.
 * Input: large = 400 Ki iterations (~7.6 M instructions), small = 128 Ki (~2.5 M).
 */
#include "micro.h"

/**
 * @brief Run micro_lowilp() (400 Ki iterations large / 128 Ki small) and print
 *        the checksum.
 * @param[in] argc Argument count.
 * @param[in] argv argv[1] = small|large (default large).
 * @return 0 (exits 2 on a bad argument).
 */
int main(int argc, char **argv)
{
    int large = micro_parse(argc, argv);
    uint64_t sum = micro_lowilp(large ? 400u * 1024u : 128u * 1024u, large ? 42u : 4242u);
    micro_report("micro_lowilp", large, sum);
    return 0;
}
