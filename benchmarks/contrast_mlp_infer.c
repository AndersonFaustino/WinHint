/**
 * @file contrast_mlp_infer.c
 * @brief WinHint Project - Contrast group (non-transformer), benchmark 2: deep MLP.
 *
 * Phase-free negative control: a uniform stack of fully connected layers
 * (GEMM + bias, then ReLU), no attention, no softmax, no layer norm.
 *
 *   layers: 1024 -> 1536 -> 1536 -> 1536 -> 256   (ReLU between layers)
 *   weights: 6.7M floats = 26.7 MB (every hidden layer's 9.4 MB matrix > L2)
 *
 * @verbatim
 *   input  | batch rows M | repetitions (default)
 *   -------+--------------+----------------------
 *   large  |      32      | 1
 *   small  |       4      | 1
 * @endverbatim
 *
 *   argv[1] = small|large (default large)   argv[2] = repetitions (default 1)
 *
 * Phases (separate non-inlined functions / loop nests): linear, relu_inplace.
 * Build options: -DTILED (register-blocked, N-outer tiled GEMM, bit-identical
 * output), -DWIDTH=... to override the hidden width.
 * Output: float checksum + FNV-1a hash of the final activations.
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

#define KERNEL_NAME "contrast_mlp" ///< Name printed as the "[name]" prefix of every output line.

#define IN_DIM 1024 ///< Input features of the first layer.
#ifndef WIDTH
#define WIDTH 1536 ///< Hidden width (override with -DWIDTH).
#endif
#define OUT_DIM 256 ///< Output features of the last layer.
#define NUM_FC 4 ///< Number of fully connected layers.
#ifndef BATCH_LARGE
#define BATCH_LARGE 32 ///< Batch rows M for the large input.
#endif
#ifndef BATCH_SMALL
#define BATCH_SMALL 4 ///< Batch rows M for the small input.
#endif

/* ---- Deterministic PRNG: values in [-0.1, 0.1) -------------------------- */
/** @brief xorshift32 state of the deterministic PRNG (fixed per-kernel seed). */
static uint32_t wh_rng = 0x5a5a5a5au;
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

/* ---- GEMM: Y[M][N] = X[M][K] * W[N][K]^T + b[N]  (nn.Linear layout) ----- */
#ifndef TILED
/**
 * @brief Dense layer Y = X W^T + b (naive m-n-k loop nest).
 * @param[out] Y Output, [M][N] row-major.
 * @param[in]  X Input, [M][K] row-major.
 * @param[in]  W Weights, [N][K] row-major (nn.Linear layout).
 * @param[in]  b Bias [N], or NULL for no bias.
 * @param[in]  M Rows of X and Y.
 * @param[in]  N Output features.
 * @param[in]  K Input features.
 */
WH_NOINLINE static void linear(float *Y, const float *X, const float *W,
                               const float *b, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = b ? b[n] : 0.0f;
            for (int k = 0; k < K; k++) acc += X[m * K + k] * W[n * K + k];
            Y[m * N + n] = acc;
        }
    }
}
#else
#define TM 4 ///< Rows of X per register block (tiled GEMM).
#define TN 4 ///< Rows of W (output columns) per register block (tiled GEMM).
/**
 * @brief Dense layer Y = X W^T + b, N-outer tiled with a TM x TN register block
 *        (-DTILED). Same per-output summation order as the naive version, so the
 *        output is bit-identical. Parameters as in the naive linear().
 */
WH_NOINLINE static void linear(float *Y, const float *X, const float *W,
                               const float *b, int M, int N, int K) {
    /* N-outer: a 4-row block of W stays in L1 while all M rows of X stream
     * from L2, so W is read from memory once. 4x4 register block. */
    for (int n0 = 0; n0 < N; n0 += TN) {
        int nb = N - n0 < TN ? N - n0 : TN;
        for (int m0 = 0; m0 < M; m0 += TM) {
            int mb = M - m0 < TM ? M - m0 : TM;
            float acc[TM][TN];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] = (b && j < nb) ? b[n0 + j] : 0.0f;
            if (mb == TM && nb == TN) {
                const float *x0 = X + (m0 + 0) * K, *x1 = X + (m0 + 1) * K;
                const float *x2 = X + (m0 + 2) * K, *x3 = X + (m0 + 3) * K;
                const float *w0 = W + (n0 + 0) * K, *w1 = W + (n0 + 1) * K;
                const float *w2 = W + (n0 + 2) * K, *w3 = W + (n0 + 3) * K;
                for (int k = 0; k < K; k++) {
                    float a0 = x0[k], a1 = x1[k], a2 = x2[k], a3 = x3[k];
                    float c0 = w0[k], c1 = w1[k], c2 = w2[k], c3 = w3[k];
                    acc[0][0] += a0 * c0; acc[0][1] += a0 * c1; acc[0][2] += a0 * c2; acc[0][3] += a0 * c3;
                    acc[1][0] += a1 * c0; acc[1][1] += a1 * c1; acc[1][2] += a1 * c2; acc[1][3] += a1 * c3;
                    acc[2][0] += a2 * c0; acc[2][1] += a2 * c1; acc[2][2] += a2 * c2; acc[2][3] += a2 * c3;
                    acc[3][0] += a3 * c0; acc[3][1] += a3 * c1; acc[3][2] += a3 * c2; acc[3][3] += a3 * c3;
                }
            } else {
                for (int i = 0; i < mb; i++)
                    for (int j = 0; j < nb; j++)
                        for (int k = 0; k < K; k++)
                            acc[i][j] += X[(m0 + i) * K + k] * W[(n0 + j) * K + k];
            }
            for (int i = 0; i < mb; i++)
                for (int j = 0; j < nb; j++) Y[(m0 + i) * N + n0 + j] = acc[i][j];
        }
    }
}
#endif

/**
 * @brief ReLU in place: x[i] = max(x[i], 0).
 * @param[in,out] x Activations.
 * @param[in]     n Number of elements.
 */
WH_NOINLINE static void relu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

static const int fc_in[NUM_FC] = {IN_DIM, WIDTH, WIDTH, WIDTH};   ///< Input features per layer.
static const int fc_out[NUM_FC] = {WIDTH, WIDTH, WIDTH, OUT_DIM}; ///< Output features per layer.

/**
 * @brief Run the NUM_FC-layer MLP over M batch rows per repetition and report
 *        the final activations.
 *
 * Every layer is linear(); all but the last are followed by relu_inplace().
 * Activations ping-pong between two [M][WIDTH] buffers. Prints a configuration
 * line, then (after the last repetition) wh_report() over the [M][OUT_DIM] output.
 * @param[in] argc Argument count.
 * @param[in] argv argv[1] = small|large, argv[2] = repetitions.
 * @return 0 on success (exits 1 on allocation failure, 2 on bad arguments).
 */
int main(int argc, char **argv) {
    int reps;
    int large = wh_parse_args(argc, argv, &reps);
    const int M = large ? BATCH_LARGE : BATCH_SMALL;
    printf("[%s] input=%s batch=%d dims=%d-%d-%d-%d-%d reps=%d%s\n", KERNEL_NAME,
           large ? "large" : "small", M, IN_DIM, WIDTH, WIDTH, WIDTH, OUT_DIM, reps,
#ifdef TILED
           " gemm=tiled"
#else
           " gemm=naive"
#endif
    );

    float *W[NUM_FC], *b[NUM_FC];
    for (int l = 0; l < NUM_FC; l++) {
        W[l] = wh_alloc_rand((size_t)fc_out[l] * fc_in[l]);
        b[l] = wh_alloc_rand((size_t)fc_out[l]);
    }
    float *in = wh_alloc_rand((size_t)M * IN_DIM);
    float *a = wh_alloc((size_t)M * WIDTH), *c = wh_alloc((size_t)M * WIDTH);

    for (int r = 0; r < reps; r++) {
        const float *x = in;
        float *y = a;
        for (int l = 0; l < NUM_FC; l++) {
            linear(y, x, W[l], b[l], M, fc_out[l], fc_in[l]);
            if (l < NUM_FC - 1) relu_inplace(y, M * fc_out[l]);
            x = y;
            y = (y == a) ? c : a;
        }
        if (r == reps - 1)
            wh_report(large ? "large" : "small", x, (size_t)M * OUT_DIM, 0u);
    }

    for (int l = 0; l < NUM_FC; l++) { free(W[l]); free(b[l]); }
    free(in); free(a); free(c);
    return 0;
}
