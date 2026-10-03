/**
 * @file micro.h
 * @brief Shared helpers for the baseline-fidelity microbenchmarks
 * (benchmarks/micro/, docs/reference/proposal.md §7 "Baseline fidelity", sim/fidelity/).
 *
 * Every microbenchmark:
 *   - takes argv[1] = small|large (default large), like the ML kernels
 *     (docs/interfaces.md §5); `small` is a different, shorter input;
 *   - is integer-only and deterministic: one line
 *       [<name>] input=<small|large> checksum=0x<16 hex digits>
 *     bit-identical on qemu-riscv64, gem5 and every build variant;
 *   - keeps each phase in its own noinline function whose body is a
 *     top-level loop nest, so the WinHint pass gives each phase its own
 *     region id (oracle / pgo builds).
 */
#ifndef WINHINT_MICRO_H
#define WINHINT_MICRO_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MICRO_NOINLINE __attribute__((noinline, unused)) ///< Keep a phase in its own function (no inlining, no unused warning).
#define MICRO_UNUSED __attribute__((unused))             ///< Silence unused warnings for helpers/data a benchmark does not use.

/**
 * @brief splitmix64 finalizer: a 64-bit integer hash.
 * @param[in] x Input value.
 * @return The mixed value.
 */
static inline uint64_t micro_mix64(uint64_t x) /* splitmix64 finalizer */
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

/**
 * @brief Rotate left.
 * @param[in] x Value.
 * @param[in] r Rotation count; taken modulo 64 (r = 0 returns x unchanged).
 * @return x rotated left by r bits.
 */
static inline uint64_t micro_rotl(uint64_t x, unsigned r)
{
    return (x << (r & 63u)) | (x >> ((64u - r) & 63u));
}

/**
 * @brief Parse argv[1] = small|large.
 * @param[in] argc Argument count from main().
 * @param[in] argv Argument vector from main().
 * @return 1 = large (default), 0 = small. Exits 2 on a bad argument.
 */
static inline int micro_parse(int argc, char **argv)
{
    if (argc < 2 || strcmp(argv[1], "large") == 0)
        return 1;
    if (strcmp(argv[1], "small") == 0)
        return 0;
    fprintf(stderr, "usage: %s [small|large]\n", argv[0]);
    exit(2);
}

/**
 * @brief Print the result line `[<name>] input=<small|large> checksum=0x<16 hex>`.
 * @param[in] name  Benchmark name.
 * @param[in] large Nonzero for the large input.
 * @param[in] sum   64-bit checksum.
 */
static inline void micro_report(const char *name, int large, uint64_t sum)
{
    printf("[%s] input=%s checksum=0x%016llx\n", name, large ? "large" : "small",
           (unsigned long long)sum);
}

/* ---------------------------------------------------------------------------
 * Building blocks shared by the single-behaviour benchmarks and micro_phased.
 * ------------------------------------------------------------------------- */

/* MLP-rich independent misses: an 8 MiB table (8x the 1 MiB L2), one 8-byte
 * word read per 64-byte line. Lines are visited page by page (4 KiB) in a
 * per-page permutation, so the TLB misses once per 64 loads and the stride
 * prefetcher (off on riscv_ooo anyway) sees no stride. The address slice is
 * 3-4 ALU ops; each loaded value feeds a 6-op dependent, non-urgent chain
 * (what LTP parks). About 12 instructions per load: a 64-entry ROB holds ~5
 * outstanding misses, a 256-entry ROB ~16 (the L1D MSHR limit). */
#define MICRO_PAGES 2048u            ///< Table pages (x 4 KiB = 8 MiB).
#define MICRO_WORDS_PER_PAGE 512u    ///< 8-byte words per 4 KiB page.
MICRO_UNUSED static uint64_t micro_table[MICRO_PAGES * MICRO_WORDS_PER_PAGE]; ///< The 8 MiB gather table.

/**
 * @brief Initialize the first word of every 64-byte line of micro_table to
 *        micro_mix64(line + seed).
 * @param[in] seed Data seed.
 */
static MICRO_NOINLINE void micro_table_init(uint64_t seed)
{
    for (unsigned i = 0; i < MICRO_PAGES * 64u; i++)
        micro_table[(size_t)i * 8u] = micro_mix64(i + seed);
}

/**
 * @brief Gather phase: `pages` pages starting at page `first` (wrapping), all 64
 *        lines of each, in a per-page permutation.
 *
 * Each loaded word feeds a short dependent hash chain whose result is summed.
 * @param[in] first First page index.
 * @param[in] pages Pages to visit.
 * @param[in] salt  Varies the per-page line permutation offset.
 * @return Sum of the hashed loads.
 */
static MICRO_NOINLINE uint64_t micro_gather(unsigned first, unsigned pages, unsigned salt)
{
    uint64_t sum = 0;
    for (unsigned q = 0; q < pages; q++) {
        unsigned p = (first + q) & (MICRO_PAGES - 1u);
        const uint64_t *page = &micro_table[(size_t)p * MICRO_WORDS_PER_PAGE];
        unsigned off = (p * 13u + salt) & 63u;
        for (unsigned j = 0; j < 64u; j++) {
            unsigned l = (j * 37u + off) & 63u; /* odd multiplier: a permutation */
            uint64_t v = page[l * 8u];
            uint64_t t = v ^ (v >> 29);
            t *= 0x9e3779b97f4a7c15ull;
            t ^= t >> 32;
            t += j;
            t = micro_rotl(t, 7);
            sum += t;
        }
    }
    return sum;
}

/* Cache-resident, high-ILP integer work (no misses): a 64x64 int32
 * matrix-vector product, repeated `reps` times, the vector updated in between. */
#define MICRO_MV 64u ///< Matrix/vector dimension of the compute phase.
MICRO_UNUSED static int32_t micro_A[MICRO_MV * MICRO_MV]; ///< 64x64 matrix (row-major).
MICRO_UNUSED static int32_t micro_x[MICRO_MV];            ///< Input vector, rewritten after every product.
MICRO_UNUSED static int32_t micro_y[MICRO_MV];            ///< Product vector.

/**
 * @brief Fill micro_A with signed 16-bit values and micro_x with 8-bit values
 *        derived from micro_mix64().
 * @param[in] seed Data seed.
 */
static MICRO_NOINLINE void micro_compute_init(uint64_t seed)
{
    for (unsigned i = 0; i < MICRO_MV * MICRO_MV; i++)
        micro_A[i] = (int32_t)(micro_mix64(i + seed) & 0xffff) - 0x8000;
    for (unsigned i = 0; i < MICRO_MV; i++)
        micro_x[i] = (int32_t)(micro_mix64(i + seed + 99999u) & 0xff);
}

/**
 * @brief Compute phase: `reps` matrix-vector products y = A x (two partial sums
 *        per row), each followed by x = (y >> 12) & 0xff.
 * @param[in] reps Number of products.
 * @return Rotating sum of all y values.
 */
static MICRO_NOINLINE uint64_t micro_compute(unsigned reps)
{
    uint64_t sum = 0;
    for (unsigned r = 0; r < reps; r++) {
        for (unsigned i = 0; i < MICRO_MV; i++) {
            int32_t s0 = 0, s1 = 0;
            const int32_t *a = &micro_A[i * MICRO_MV];
            for (unsigned j = 0; j < MICRO_MV; j += 2) {
                s0 += a[j] * micro_x[j];
                s1 += a[j + 1] * micro_x[j + 1];
            }
            micro_y[i] = s0 + s1;
        }
        for (unsigned i = 0; i < MICRO_MV; i++) {
            micro_x[i] = (micro_y[i] >> 12) & 0xff;
            sum += (uint32_t)micro_y[i];
        }
        sum = micro_rotl(sum, 3);
    }
    return sum;
}

/* Low-ILP, low-occupancy code: a serial 64-bit LCG (multiply chain) with
 * data-dependent, ~50% mispredicted branches; no memory traffic. Squashes keep
 * the window mostly empty, so a large window buys nothing. */
/**
 * @brief Low-ILP phase: `iters` steps of the serial LCG with data-dependent branches.
 * @param[in] iters Iterations.
 * @param[in] seed  LCG seed (forced odd).
 * @return Combination of the accumulator and the final LCG/xor state.
 */
static MICRO_NOINLINE uint64_t micro_lowilp(unsigned iters, uint64_t seed)
{
    uint64_t a = seed | 1u, b = 3, acc = 0;
    for (unsigned i = 0; i < iters; i++) {
        a = a * 6364136223846793005ull + 1442695040888963407ull;
        b ^= a >> 17;
        if ((a >> 33) & 1u)
            acc += b * 3u;
        else
            acc ^= b + i;
        if (((a >> 40) & 7u) == 3u)
            acc = micro_rotl(acc, 1);
    }
    return acc + a + b;
}

#endif /* WINHINT_MICRO_H */
