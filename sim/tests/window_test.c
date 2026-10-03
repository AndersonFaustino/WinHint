/**
 * @file
 * @brief WinHint gem5 mechanism/policy test program (RISC-V guest).
 *
 * window_test.c — WinHint gem5 mechanism/policy test program.
 *
 * @verbatim
 *   window_test phases [iters]   three phases (MLP-rich gather, dependent pointer
 *                          chase, compute-bound), each preceded by region()
 *                          and setwin() hints: 256, 64, 128.
 *   window_test force <W> [iters]  setwin(W) (W in {0,64,128,192,256}), then the
 *                          MLP-rich gather only.  Used for the mechanism
 *                          check: ROB/IQ occupancy must be capped by W.
 * @endverbatim
 *
 * Output is a deterministic checksum; it must be identical on RISCV_clean,
 * RISCV_winhint (any policy), qemu-riscv64 and a -DWINHINT_NO_HINTS build.
 * iters defaults to 20000. The phases mode also runs a fourth gather phase
 * (region 4, setwin(0)); force wraps two gathers in region(1) / region(2).
 * Any other W in force mode is treated as 0. Any other mode prints the
 * usage to stderr and exits with status 2.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "winhint_hint.h"

#define N_ELEMS (1u << 20)          /**< 4 MB of uint32: larger than any L2 */
static uint32_t data[N_ELEMS];      /**< gather table (sparsely initialised) */
static uint32_t ring[N_ELEMS];       /**< pointer-chase ring, 4 MB, one node per line */
#define RING_NODES (N_ELEMS / 16)   /**< ring nodes, one per 64-byte line */

static unsigned iters = 20000;       /**< loop iterations (argv) */

/**
 * @brief Independent loads: the next address does not depend on loaded data.
 * @return Checksum.
 */
static uint64_t gather(void)
{
    uint64_t sum = 0;
    uint32_t x = 12345;
    for (unsigned i = 0; i < iters; i++) {
        x = x * 1664525u + 1013904223u;
        uint32_t j = (x >> 8) & (N_ELEMS - 1);
        sum += data[j] + j;
    }
    return sum;
}

/**
 * @brief Dependent loads: each address comes from the previous load
 *        (MLP = 1); iters / 4 steps.
 * @return Checksum.
 */
static uint64_t chase(void)
{
    uint64_t sum = 0;
    uint32_t p = 0;
    for (unsigned i = 0; i < iters / 4; i++) {
        p = ring[p];
        sum += p;
    }
    return sum;
}

/**
 * @brief Compute-bound, cache-resident.
 * @return Checksum.
 */
static uint64_t compute(void)
{
    uint64_t a = 1, b = 2, c = 3, d = 4;
    for (unsigned i = 0; i < iters; i++) {
        a = a * 3 + i; b = b ^ (a >> 3); c += b * 5; d = d + (c >> 7) + i;
    }
    return a + b + c + d;
}

/** @brief Initialise data (sparsely) and the pointer-chase ring. */
static void init(void)
{
    /* Sparse init: touch one word per 4 kB page so pages exist in SE mode
     * without spending millions of instructions. */
    for (unsigned i = 0; i < N_ELEMS; i += 1024)
        data[i] = i * 2654435761u;
    /* One node per 64-byte line; odd stride => a single cycle over all nodes. */
    for (uint32_t k = 0; k < RING_NODES; k++)
        ring[k * 16] = ((k + 40503u) & (RING_NODES - 1)) * 16;
}

/**
 * @brief Parse the mode and iterations, run the phases, print the checksum.
 * @param argc Argument count.
 * @param argv phases [iters] | force W [iters] (default: phases).
 * @return 0.
 */
int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "phases";
    if (strcmp(mode, "phases") != 0 && strcmp(mode, "force") != 0) {
        fprintf(stderr, "usage: %s phases [iters] | force <W> [iters]\n",
                argv[0]);
        return 2;
    }
    int iters_arg = strcmp(mode, "force") == 0 ? 3 : 2;
    if (argc > iters_arg)
        iters = (unsigned)strtoul(argv[iters_arg], NULL, 10);
    init();
    uint64_t s = 0;

    if (strcmp(mode, "force") == 0) {
        int w = argc > 2 ? atoi(argv[2]) : 0;
        switch (w) {
        case 64:  SETWIN(64);  break;
        case 128: SETWIN(128); break;
        case 192: SETWIN(192); break;
        case 256: SETWIN(256); break;
        default:  SETWIN(0);   break;
        }
        REGION(1);
        s += gather();
        REGION(2);
        s += gather();
    } else {
        REGION(1); SETWIN(256); s += gather();
        REGION(2); SETWIN(64);  s += chase();
        REGION(3); SETWIN(128); s += compute();
        REGION(4); SETWIN(0);   s += gather();
    }
    printf("window_test %s checksum=%llu\n", mode, (unsigned long long)s);
    return 0;
}
