/**
 * @file regression_svr_infer.c
 * @brief WinHint Project - Regression group (non-transformer control): support
 * vector regression with an RBF kernel,
 *   y(x) = sum_i alpha_i * exp(-gamma * ||sv_i - x||^2) + b
 *
 * @verbatim
 *   input  | test points | support vectors | features | SV matrix
 *   -------+-------------+-----------------+----------+----------
 *   large  |    1024     |      1024       |    64    | 256 kB
 *   small  |     256     |      1024       |    64    | 256 kB
 * @endverbatim
 *
 *   argv[1] = small|large (default large)   argv[2] = repetitions (default 1)
 *
 * The model (support vectors) is fixed; the input is the set of test points.
 * Phases (separate non-inlined functions / loop nests): sq_distances
 * (compute bound, L2-resident), rbf_inplace (expf), weighted_sum.
 * Output: float checksum + FNV-1a hash of the predictions.
 * Portable C11; deterministic.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* No FMA contraction: identical results on x86-64 (any -march), RISC-V
 * (rv64gc has fmadd) and gem5/QEMU, for the bit-identical output checks. */
#pragma STDC FP_CONTRACT OFF

/**
 * @def WH_NOINLINE
 * @brief `noinline` on GCC/Clang (empty otherwise): keeps each phase in its
 *        own non-inlined function.
 */
/**
 * @def WH_UNUSED
 * @brief `unused` on GCC/Clang (empty otherwise): silences warnings for
 *        helpers a kernel does not call.
 */
#if defined(__GNUC__) || defined(__clang__)
#define WH_NOINLINE __attribute__((noinline))
#define WH_UNUSED __attribute__((unused))
#else
#define WH_NOINLINE
#define WH_UNUSED
#endif

#define KERNEL_NAME "regression_svr" ///< Name printed as the "[name]" prefix of every output line.

#define N_SV 1024 ///< Number of support vectors (fixed model).
#define FEATURES 64 ///< Features per point.
#ifndef TEST_LARGE
#define TEST_LARGE 1024 ///< Test points for the large input.
#endif
#ifndef TEST_SMALL
#define TEST_SMALL 256 ///< Test points for the small input.
#endif
#define GAMMA 0.05f ///< RBF kernel width gamma.
#define BLOCK 64 ///< test points per block

/* ---- Deterministic PRNG: values in [-0.1, 0.1) -------------------------- */
/** @brief xorshift32 state of the deterministic PRNG (fixed per-kernel seed). */
static uint32_t wh_rng = 0xfedcba98u;
/**
 * @brief Draw the next PRNG value (xorshift32, top 24 bits).
 * @return A float uniformly spaced in [-0.1, 0.1).
 */
static float wh_rand(void) {
    wh_rng ^= wh_rng << 13;
    wh_rng ^= wh_rng >> 17;
    wh_rng ^= wh_rng << 5;
    return ((float)(wh_rng >> 8) * (1.0f / 16777216.0f) - 0.5f) * 0.2f;
}

/**
 * @brief malloc() @p n floats; print an error and exit(1) on failure.
 * @param[in] n Number of floats.
 * @return The (uninitialized) buffer.
 */
static float *wh_alloc(size_t n) {
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) { fprintf(stderr, "[%s] out of memory\n", KERNEL_NAME); exit(1); }
    return p;
}
/**
 * @def WH_POOL
 * @brief Length of the PRNG value pool used by wh_fill_rand() (a prime).
 *
 * Fast deterministic fill: a pool of WH_POOL PRNG values (prime length, so no
 * row of any width repeats in phase) is replicated with memcpy, continuing at a
 * global phase so that consecutive arrays differ. About 1-2 instructions per
 * float instead of ~15 for wh_rand(), so weight initialization stays a small
 * fraction of the simulated instructions (gem5 time). */
#define WH_POOL 4093
static float wh_pool[WH_POOL]; ///< Pool of wh_rand() values, filled on first use.
static size_t wh_pool_pos;     ///< Global read phase into wh_pool.
static int wh_pool_ready;      ///< Nonzero once wh_pool has been filled.
/**
 * @brief Fill @p p with @p n deterministic pseudo-random floats from the pool.
 *
 * Copies the pool cyclically from the current phase, then advances the phase by
 * one more element so that the next array starts at a different offset.
 * @param[out] p Destination buffer.
 * @param[in]  n Number of floats.
 */
static void wh_fill_rand(float *p, size_t n) {
    if (!wh_pool_ready) {
        for (size_t i = 0; i < WH_POOL; i++) wh_pool[i] = wh_rand();
        wh_pool_ready = 1;
    }
    while (n > 0) {
        size_t len = WH_POOL - wh_pool_pos;
        if (len > n) len = n;
        memcpy(p, wh_pool + wh_pool_pos, len * sizeof(float));
        p += len; n -= len; wh_pool_pos += len;
        if (wh_pool_pos == WH_POOL) wh_pool_pos = 0;
    }
    wh_pool_pos = (wh_pool_pos + 1) % WH_POOL;
}
/**
 * @brief Allocate @p n floats filled by wh_fill_rand() (values in [-0.1, 0.1)).
 * @param[in] n Number of floats.
 * @return The new buffer (exits on allocation failure).
 */
WH_UNUSED static float *wh_alloc_rand(size_t n) {
    float *p = wh_alloc(n);
    wh_fill_rand(p, n);
    return p;
}
/**
 * @brief Allocate @p n floats, all set to @p v.
 * @param[in] n Number of floats.
 * @param[in] v Fill value.
 * @return The new buffer (exits on allocation failure).
 */
WH_UNUSED static float *wh_alloc_const(size_t n, float v) {
    float *p = wh_alloc(n);
    for (size_t i = 0; i < n; i++) p[i] = v;
    return p;
}

/**
 * @brief Parse the command line: argv[1] = small|large (default large);
 *        argv[2] = repetitions (default 1).
 *
 * Prints usage and exits with status 2 on an unknown size or a count < 1.
 * @param[in]  argc   Argument count from main().
 * @param[in]  argv   Argument vector from main().
 * @param[out] layers Receives the number of repetitions (argv[2], default 1).
 * @return 1 for the large input, 0 for small.
 */
static int wh_parse_args(int argc, char **argv, int *layers) {
    int large = 1;
    *layers = 1;
    if (argc > 1) {
        if (strcmp(argv[1], "small") == 0) large = 0;
        else if (strcmp(argv[1], "large") == 0) large = 1;
        else {
            fprintf(stderr, "usage: %s [small|large] [repetitions]\n", argv[0]);
            exit(2);
        }
    }
    if (argc > 2) {
        *layers = atoi(argv[2]);
        if (*layers < 1) { fprintf(stderr, "repetitions must be >= 1\n"); exit(2); }
    }
    return large;
}

/**
 * @brief Print the result line `[<kernel>] input=<size> checksum=<sum> hash=0x<h>`.
 *
 * The checksum is the double-precision sum of @p x (printed with %.9e); the hash
 * is 32-bit FNV-1a over the bytes of @p x, then mixed with @p extra.
 * @param[in] size  Input name printed after `input=` ("small" or "large").
 * @param[in] x     Output values to checksum.
 * @param[in] n     Number of values in @p x.
 * @param[in] extra Extra 32-bit value folded into the hash (e.g. token ids; 0 if unused).
 */
static void wh_report(const char *size, const float *x, size_t n, uint32_t extra) {
    double sum = 0.0;
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        uint32_t b;
        sum += (double)x[i];
        memcpy(&b, &x[i], sizeof b);
        for (int k = 0; k < 4; k++) { h ^= (b >> (8 * k)) & 0xffu; h *= 16777619u; }
    }
    h ^= extra; h *= 16777619u;
    printf("[%s] input=%s checksum=%.9e hash=0x%08x\n", KERNEL_NAME, size, sum, (unsigned)h);
}

/**
 * @brief D[t][i] = ||X[t] - SV[i]||^2 for a block of nt test points.
 * @param[out] D  Squared distances, [nt][N_SV] row-major.
 * @param[in]  X  Test points of the block, [nt][FEATURES].
 * @param[in]  SV Support vectors, [N_SV][FEATURES].
 * @param[in]  nt Test points in the block (<= BLOCK).
 */
WH_NOINLINE static void sq_distances(float *D, const float *X, const float *SV, int nt) {
    for (int t = 0; t < nt; t++)
        for (int i = 0; i < N_SV; i++) {
            float acc = 0.0f;
            for (int f = 0; f < FEATURES; f++) {
                float d = X[t * FEATURES + f] - SV[i * FEATURES + f];
                acc += d * d;
            }
            D[(size_t)t * N_SV + i] = acc;
        }
}

/**
 * @brief Apply the RBF kernel in place: D[i] = exp(-GAMMA * D[i]).
 * @param[in,out] D Squared distances in, kernel values out.
 * @param[in]     n Number of elements.
 */
WH_NOINLINE static void rbf_inplace(float *D, int n) {
    for (int i = 0; i < n; i++) D[i] = expf(-GAMMA * D[i]);
}

/**
 * @brief y[t] = b + sum_i alpha[i] * K[t][i] for each test point of the block.
 * @param[out] y     Predictions [nt].
 * @param[in]  K     Kernel values, [nt][N_SV] row-major.
 * @param[in]  alpha Dual coefficients [N_SV].
 * @param[in]  b     Bias.
 * @param[in]  nt    Test points in the block.
 */
WH_NOINLINE static void weighted_sum(float *y, const float *K, const float *alpha, float b, int nt) {
    for (int t = 0; t < nt; t++) {
        float acc = b;
        for (int i = 0; i < N_SV; i++) acc += alpha[i] * K[(size_t)t * N_SV + i];
        y[t] = acc;
    }
}

/**
 * @brief Predict all test points in blocks of BLOCK (sq_distances, rbf_inplace,
 *        weighted_sum) per repetition and report the predictions.
 *
 * Prints a configuration line, then wh_report() over the N predictions.
 * @param[in] argc Argument count.
 * @param[in] argv argv[1] = small|large, argv[2] = repetitions.
 * @return 0 on success (exits 1 on allocation failure, 2 on bad arguments).
 */
int main(int argc, char **argv) {
    int reps;
    int large = wh_parse_args(argc, argv, &reps);
    const int N = large ? TEST_LARGE : TEST_SMALL;
    printf("[%s] input=%s test=%d support_vectors=%d features=%d reps=%d\n", KERNEL_NAME,
           large ? "large" : "small", N, N_SV, FEATURES, reps);

    float *SV = wh_alloc_rand((size_t)N_SV * FEATURES);
    float *alpha = wh_alloc_rand(N_SV);
    float b = wh_rand();
    float *X = wh_alloc_rand((size_t)N * FEATURES);
    float *D = wh_alloc((size_t)BLOCK * N_SV);
    float *y = wh_alloc((size_t)N);

    for (int r = 0; r < reps; r++)
        for (int t0 = 0; t0 < N; t0 += BLOCK) {
            int nt = N - t0 < BLOCK ? N - t0 : BLOCK;
            sq_distances(D, X + (size_t)t0 * FEATURES, SV, nt);
            rbf_inplace(D, nt * N_SV);
            weighted_sum(y + t0, D, alpha, b, nt);
        }
    wh_report(large ? "large" : "small", y, (size_t)N, 0u);

    free(SV); free(alpha); free(X); free(D); free(y);
    return 0;
}
