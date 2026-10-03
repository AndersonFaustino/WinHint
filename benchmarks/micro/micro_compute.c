/**
 * @file micro_compute.c
 * @brief Cache-resident, high-ILP integer work (no long-latency misses).
 *
 * Condition targeted: IPC near the issue width, low window occupancy.
 *   B2 (Ponomarev): occupancy stays low -> the window shrinks, little IPC loss.
 *   B3 (Kora): no misses -> stays in ILP mode, no gain from enlarging.
 *   B6 (Jones): the IQ demand of the loop DAG is small -> IQ can shrink.
 *   B9 (LTP): nothing to park; LTP must not hurt.
 * Input: large = 160 matrix-vector products (~4 M instructions), small = 48.
 */
#include "micro.h"

/**
 * @brief Run micro_compute() (160 products large / 48 small) and print the checksum.
 * @param[in] argc Argument count.
 * @param[in] argv argv[1] = small|large (default large).
 * @return 0 (exits 2 on a bad argument).
 */
int main(int argc, char **argv)
{
    int large = micro_parse(argc, argv);
    micro_compute_init(large ? 1u : 5u);
    uint64_t sum = micro_compute(large ? 160u : 48u);
    micro_report("micro_compute", large, sum);
    return 0;
}
