/**
 * @file
 * @brief B9 (Long-Term Parking) functional/stress test program (RISC-V
 *        guest, run by run_ltp_tests.sh).
 *
 * ltp_test.c -- B9 (Long-Term Parking) functional/stress test program.
 *
 *   ltp_test mem     [iters]  independent long-latency loads (4 MB table),
 *                             each feeding a non-urgent dependent chain:
 *                             LTP should park the chains (MLP-bound)
 *   ltp_test compute [iters]  cache-resident ALU work with data-dependent,
 *                             hard-to-predict branches (squash stress)
 *   ltp_test mixed   [iters]  loads + stores + store-to-load forwarding +
 *                             unpredictable branches on loaded data (LSQ
 *                             order, squash of parked instructions)
 *   ltp_test all     [iters]  the three, in order
 *
 * The output is a deterministic checksum; it must be identical under
 * qemu-riscv64, window_policy=static and window_policy=ltp.
 * iters defaults to 20000; an unknown mode prints the usage to stderr and
 * exits with status 2.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** Entries of table (uint32_t). */
#define N (1u << 20)                 /* 4 MB of uint32: beyond L2 */
static uint32_t table[N];       /**< large table (misses) */
static uint32_t small[1024];    /**< cache-resident store/load buffer */

/**
 * @brief Independent long-latency loads, each feeding a dependent chain.
 * @param iters Iterations.
 * @return Checksum.
 */
static uint64_t mem_test(unsigned iters)
{
    uint64_t sum = 0;
    uint32_t x = 7;
    for (unsigned i = 0; i < iters; i++) {
        x = x * 1664525u + 1013904223u;         /* urgent: address slice */
        uint32_t v = table[(x >> 9) & (N - 1)];  /* long-latency load */
        /* non-urgent: dependent chain on the loaded value */
        uint32_t t = v ^ (v >> 7);
        t = t * 2654435761u;
        t ^= t >> 13;
        t += i;
        t = (t << 3) | (t >> 29);
        sum += t;
    }
    return sum;
}

/**
 * @brief Cache-resident ALU work with data-dependent branches.
 * @param iters Iterations.
 * @return Checksum.
 */
static uint64_t compute_test(unsigned iters)
{
    uint64_t a = 1, b = 3, acc = 0;
    for (unsigned i = 0; i < iters; i++) {
        a = a * 6364136223846793005ull + 1442695040888963407ull;
        b ^= a >> 17;
        if ((a >> 33) & 1)            /* random: ~50% mispredicted */
            acc += b * 3;
        else
            acc ^= b + i;
        if (((a >> 40) & 7) == 3)
            acc = (acc << 1) | (acc >> 63);
    }
    return acc + a + b;
}

/**
 * @brief Loads, stores, store-to-load forwarding and branches on missed
 *        data (writes table and small).
 * @param iters Iterations.
 * @return Checksum.
 */
static uint64_t mixed_test(unsigned iters)
{
    uint64_t sum = 0;
    uint32_t x = 99;
    for (unsigned i = 0; i < iters; i++) {
        x = x * 1103515245u + 12345u;
        uint32_t j = (x >> 8) & (N - 1);
        uint32_t v = table[j];                   /* long-latency load */
        unsigned k = (x >> 3) & 1023;
        small[k] = v + i;                        /* store */
        uint32_t w = small[(k + (v & 1)) & 1023];/* forwarded or not */
        if (v & 4)                               /* depends on the miss */
            sum += w;
        else
            sum ^= (uint64_t)w << 7;
        table[(j + 64) & (N - 1)] = (uint32_t)sum; /* store far away */
    }
    for (unsigned k = 0; k < 1024; k++)
        sum += small[k] * (k + 1);
    return sum;
}

/**
 * @brief Initialise the tables and run the selected test(s).
 * @param argc Argument count.
 * @param argv [1] mode (mem, compute, mixed, all; default all),
 *             [2] iterations.
 * @return 0.
 */
int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "all";
    unsigned iters = argc > 2 ? (unsigned)strtoul(argv[2], 0, 10) : 20000;

    if (strcmp(mode, "all") && strcmp(mode, "mem") &&
        strcmp(mode, "compute") && strcmp(mode, "mixed")) {
        fprintf(stderr, "usage: %s [mem|compute|mixed|all] [iters]\n",
                argv[0]);
        return 2;
    }

    for (unsigned i = 0; i < N; i++)
        table[i] = i * 2654435761u + (i >> 5);
    for (unsigned k = 0; k < 1024; k++)
        small[k] = k * 7;

    int all = !strcmp(mode, "all");
    if (all || !strcmp(mode, "mem"))
        printf("mem %llu\n", (unsigned long long)mem_test(iters));
    if (all || !strcmp(mode, "compute"))
        printf("compute %llu\n", (unsigned long long)compute_test(iters));
    if (all || !strcmp(mode, "mixed"))
        printf("mixed %llu\n", (unsigned long long)mixed_test(iters));
    return 0;
}
