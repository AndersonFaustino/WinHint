/**
 * @file regression_linear_infer.c
 * @brief WinHint Project - Regression group (non-transformer control): batch linear
 * regression, y = X w + b, with a few gradient steps.
 *
 * @verbatim
 *   input  | samples (batch) | features | X size  | iterations
 *   -------+-----------------+----------+---------+-----------
 *   large  |      2048       |   512    | 4 MB    | 10
 *   small  |       512       |   512    | 1 MB    | 10
 * @endverbatim
 *
 *   argv[1] = small|large (default large)   argv[2] = repetitions (default 1)
 *
 * Phases (separate non-inlined functions / loop nests): predict (row-wise
 * dot products, streaming), mse, gradient_step (column-wise walk over X with
 * a stride of 2 kB: cache-hostile, high MLP).
 * Output: float checksum + FNV-1a hash of the weights and final loss.
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

#define KERNEL_NAME "regression_linear" ///< Name printed as the "[name]" prefix of every output line.

#define FEATURES 512 ///< Features per sample (columns of X).
#ifndef BATCH_LARGE
#define BATCH_LARGE 2048 ///< Samples (rows of X) for the large input.
#endif
#ifndef BATCH_SMALL
#define BATCH_SMALL 512 ///< Samples (rows of X) for the small input.
#endif
#define ITERS 10 ///< Gradient-descent iterations per repetition. (---- Deterministic PRNG: values in [-0.1, 0.1) --------------------------)
/** @brief xorshift32 state of the deterministic PRNG (fixed per-kernel seed). */
static uint32_t wh_rng = 0x11223344u;
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
 * @brief Predict y = X w + b (row-wise dot products, streaming over X).
 * @param[out] y Predictions [N].
 * @param[in]  X Samples, [N][FEATURES] row-major.
 * @param[in]  w Weights [FEATURES].
 * @param[in]  b Bias.
 * @param[in]  N Number of samples.
 */
WH_NOINLINE static void predict(float *y, const float *X, const float *w, float b, int N) {
    for (int i = 0; i < N; i++) {
        float acc = b;
        for (int j = 0; j < FEATURES; j++) acc += X[(size_t)i * FEATURES + j] * w[j];
        y[i] = acc;
    }
}

/**
 * @brief Mean squared error between predictions and targets.
 * @param[in] y Predictions [N].
 * @param[in] t Targets [N].
 * @param[in] N Number of samples.
 * @return sum((y - t)^2) / N.
 */
WH_NOINLINE static float mse(const float *y, const float *t, int N) {
    float loss = 0.0f;
    for (int i = 0; i < N; i++) { float d = y[i] - t[i]; loss += d * d; }
    return loss / (float)N;
}

/**
 * @brief One gradient-descent step on w for the MSE loss.
 *
 * For each feature j, walks column j of X (stride FEATURES floats: cache-hostile)
 * to form g = sum(2 (y - t) X[:, j]), then sets w[j] -= lr * g / N. The bias is
 * not updated.
 * @param[in,out] w  Weights [FEATURES].
 * @param[in]     X  Samples, [N][FEATURES] row-major.
 * @param[in]     y  Predictions [N].
 * @param[in]     t  Targets [N].
 * @param[in]     N  Number of samples.
 * @param[in]     lr Learning rate.
 */
WH_NOINLINE static void gradient_step(float *w, const float *X, const float *y,
                                      const float *t, int N, float lr) {
    for (int j = 0; j < FEATURES; j++) {
        float g = 0.0f;
        for (int i = 0; i < N; i++) g += 2.0f * (y[i] - t[i]) * X[(size_t)i * FEATURES + j];
        w[j] -= lr * g / (float)N;
    }
}

/**
 * @brief Run ITERS predict / mse / gradient_step iterations per repetition and
 *        report the weights plus the final loss.
 *
 * Weights restart from zero at every repetition. Prints a configuration line,
 * then wh_report() over w[0..FEATURES-1] followed by the final loss (w[FEATURES]).
 * @param[in] argc Argument count.
 * @param[in] argv argv[1] = small|large, argv[2] = repetitions.
 * @return 0 on success (exits 1 on allocation failure, 2 on bad arguments).
 */
int main(int argc, char **argv) {
    int reps;
    int large = wh_parse_args(argc, argv, &reps);
    const int N = large ? BATCH_LARGE : BATCH_SMALL;
    printf("[%s] input=%s samples=%d features=%d iters=%d reps=%d\n", KERNEL_NAME,
           large ? "large" : "small", N, FEATURES, ITERS, reps);

    float *X = wh_alloc_rand((size_t)N * FEATURES);
    float *t = wh_alloc_rand((size_t)N);
    float *w = wh_alloc((size_t)FEATURES + 1);
    float *y = wh_alloc((size_t)N);
    float b = 0.01f, loss = 0.0f;

    for (int r = 0; r < reps; r++) {
        for (int j = 0; j < FEATURES; j++) w[j] = 0.0f;
        for (int it = 0; it < ITERS; it++) {
            predict(y, X, w, b, N);
            loss = mse(y, t, N);
            gradient_step(w, X, y, t, N, 0.05f);
        }
    }
    w[FEATURES] = loss;
    wh_report(large ? "large" : "small", w, FEATURES + 1, 0u);

    free(X); free(t); free(w); free(y);
    return 0;
}
