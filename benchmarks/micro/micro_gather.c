/**
 * @file micro_gather.c
 * @brief Independent long-latency misses (MLP-rich).
 *
 * Condition targeted: many independent L2 misses whose addresses do not
 * depend on loaded data. A larger window exposes more of them at once.
 *   B2 (Ponomarev): the window fills and dispatch stalls -> it must grow.
 *   B3 (Kora): MLP mode pays off -> big speedup over the ILP-mode window.
 *   B9 (Sembrant LTP): with a small IQ/LSQ the dependent chains clog the IQ;
 *       parking them recovers MLP.
 * Input: large = 3 sweeps over the 8 MiB table, small = 1 sweep (other seed).
 * ~1.4 M instructions of init + ~2.25 M per sweep (rv64gc -O2, QEMU count).
 */
#include "micro.h"

/**
 * @brief Sweep the whole table with micro_gather() (3 sweeps large / 1 small,
 *        different start page and salt per sweep) and print the checksum.
 * @param[in] argc Argument count.
 * @param[in] argv argv[1] = small|large (default large).
 * @return 0 (exits 2 on a bad argument).
 */
int main(int argc, char **argv)
{
    int large = micro_parse(argc, argv);
    unsigned sweeps = large ? 3u : 1u;
    micro_table_init(large ? 1u : 7u);
    uint64_t sum = 0;
    for (unsigned s = 0; s < sweeps; s++)
        sum += micro_gather(s * 517u, MICRO_PAGES, s * 5u);
    micro_report("micro_gather", large, sum);
    return 0;
}
