/**
 * @file imggen_vae_decode_infer.c
 * @brief WinHint Project - Image-generation group (non-transformer), benchmark 3:
 * VAE decoder.
 *
 *   FC 64 -> 1024 (ReLU) -> FC 1024 -> 4096 (ReLU), reshape to 16x[16x16],
 *   upsample 2x + conv3x3 16 -> 8 (ReLU), upsample 2x + conv3x3 8 -> 1,
 *   sigmoid; output 64x64x1 per latent
 *
 * @verbatim
 *   input  | latents (batch) | FC2 weights | repetitions (default)
 *   -------+-----------------+-------------+----------------------
 *   large  |        8        | 16 MB       | 1
 *   small  |        2        | 16 MB       | 1
 * @endverbatim
 *
 *   argv[1] = small|large (default large)   argv[2] = repetitions (default 1)
 *
 * Mixed memory pattern: weight-streaming dense layers (M = batch GEMMs over
 * a 16 MB matrix) followed by cache-resident spatial convolutions.
 * Phases (separate non-inlined functions / loop nests): linear, relu_inplace,
 * upsample2, conv3x3, sigmoid_inplace.
 * Build options: -DTILED (register-blocked, N-outer tiled GEMM, bit-identical).
 * Output: float checksum + FNV-1a hash of all decoded images.
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

#define KERNEL_NAME "imggen_vae_decode" ///< Name printed as the "[name]" prefix of every output line.

#define LATENT 64 ///< Latent vector length.
#define HID 1024 ///< Width of the first FC layer.
#define C0 16 ///< Channels after the reshape.
#define S0 16 ///< Spatial side after the reshape.
#define FLAT (C0 * S0 * S0) ///< Output width of the second FC layer, C0 x S0 x S0 (4096)
#define C1 8 ///< Channels after the first conv.
#define OUT_S (S0 * 4) ///< Output image side.
#ifndef BATCH_LARGE
#define BATCH_LARGE 8 ///< Latents (batch rows M) for the large input.
#endif
#ifndef BATCH_SMALL
#define BATCH_SMALL 2 ///< Latents (batch rows M) for the small input.
#endif

/* ---- Deterministic PRNG: values in [-0.1, 0.1) -------------------------- */
/** @brief xorshift32 state of the deterministic PRNG (fixed per-kernel seed). */
static uint32_t wh_rng = 0x87654321u;
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
 * @brief 3x3 conv, stride 1, pad 1: in [Cin][S][S] -> out [Cout][S][S] (+bias).
 * @param[out] out  Output activations, [Cout][S][S].
 * @param[in]  in   Input activations, [Cin][S][S] (CHW).
 * @param[in]  K    Kernels, [Cout][Cin][3][3].
 * @param[in]  bias Per-output-channel bias [Cout].
 * @param[in]  Cin  Input channels.
 * @param[in]  Cout Output channels.
 * @param[in]  S    Spatial side (input and output).
 */
WH_NOINLINE static void conv3x3(float *out, const float *in, const float *K, const float *bias,
                                int Cin, int Cout, int S) {
    for (int co = 0; co < Cout; co++)
        for (int y = 0; y < S; y++)
            for (int x = 0; x < S; x++) {
                float acc = bias[co];
                for (int ci = 0; ci < Cin; ci++)
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = y + ky - 1;
                        if (iy < 0 || iy >= S) continue;
                        for (int kx = 0; kx < 3; kx++) {
                            int ix = x + kx - 1;
                            if (ix < 0 || ix >= S) continue;
                            acc += in[((size_t)ci * S + iy) * S + ix] *
                                   K[((co * Cin + ci) * 3 + ky) * 3 + kx];
                        }
                    }
                out[((size_t)co * S + y) * S + x] = acc;
            }
}

/**
 * @brief ReLU in place: x[i] = max(x[i], 0).
 * @param[in,out] x Activations.
 * @param[in]     n Number of elements.
 */
WH_NOINLINE static void relu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

/**
 * @brief Nearest-neighbour 2x upsample: [C][S][S] -> [C][2S][2S].
 * @param[out] out Output, [C][2S][2S].
 * @param[in]  in  Input, [C][S][S].
 * @param[in]  C   Channels.
 * @param[in]  S   Input side.
 */
WH_NOINLINE static void upsample2(float *out, const float *in, int C, int S) {
    for (int c = 0; c < C; c++)
        for (int y = 0; y < 2 * S; y++)
            for (int x = 0; x < 2 * S; x++)
                out[((size_t)c * 2 * S + y) * 2 * S + x] = in[((size_t)c * S + y / 2) * S + x / 2];
}

/**
 * @brief Logistic sigmoid in place: x[i] = 1 / (1 + exp(-x[i])).
 * @param[in,out] x Values.
 * @param[in]     n Number of elements.
 */
WH_NOINLINE static void sigmoid_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = 1.0f / (1.0f + expf(-x[i]));
}
/**
 * @brief Decode M latents per repetition and report all decoded images.
 *
 * The two FC layers run as batched GEMMs (linear() with M rows); the conv part
 * runs per latent: upsample2, conv3x3 C0 -> C1 + ReLU, upsample2, conv3x3 C1 -> 1;
 * then sigmoid over all images. Prints a configuration line, then wh_report()
 * over the M x OUT_S x OUT_S outputs.
 * @param[in] argc Argument count.
 * @param[in] argv argv[1] = small|large, argv[2] = repetitions.
 * @return 0 on success (exits 1 on allocation failure, 2 on bad arguments).
 */
int main(int argc, char **argv) {
    int reps;
    int large = wh_parse_args(argc, argv, &reps);
    const int M = large ? BATCH_LARGE : BATCH_SMALL;
    printf("[%s] input=%s latents=%d dims=%d-%d-%d output=%dx%d reps=%d%s\n", KERNEL_NAME,
           large ? "large" : "small", M, LATENT, HID, FLAT, OUT_S, OUT_S, reps,
#ifdef TILED
           " gemm=tiled"
#else
           " gemm=naive"
#endif
    );

    float *W1 = wh_alloc_rand((size_t)HID * LATENT), *b1 = wh_alloc_rand(HID);
    float *W2 = wh_alloc_rand((size_t)FLAT * HID), *b2 = wh_alloc_rand(FLAT);
    float *K1 = wh_alloc_rand((size_t)C1 * C0 * 9), *c1 = wh_alloc_rand(C1);
    float *K2 = wh_alloc_rand((size_t)1 * C1 * 9), *c2 = wh_alloc_rand(1);
    float *z = wh_alloc_rand((size_t)M * LATENT);
    float *h1 = wh_alloc((size_t)M * HID), *h2 = wh_alloc((size_t)M * FLAT);
    float *up = wh_alloc((size_t)C1 * OUT_S * OUT_S);
    float *f1 = wh_alloc((size_t)C1 * (2 * S0) * (2 * S0));
    float *img = wh_alloc((size_t)M * OUT_S * OUT_S);

    for (int r = 0; r < reps; r++) {
        linear(h1, z, W1, b1, M, HID, LATENT);
        relu_inplace(h1, M * HID);
        linear(h2, h1, W2, b2, M, FLAT, HID);
        relu_inplace(h2, M * FLAT);
        for (int m = 0; m < M; m++) {
            float *o = img + (size_t)m * OUT_S * OUT_S;
            upsample2(up, h2 + (size_t)m * FLAT, C0, S0);
            conv3x3(f1, up, K1, c1, C0, C1, 2 * S0);
            relu_inplace(f1, C1 * (2 * S0) * (2 * S0));
            upsample2(up, f1, C1, 2 * S0);
            conv3x3(o, up, K2, c2, C1, 1, OUT_S);
        }
        sigmoid_inplace(img, M * OUT_S * OUT_S);
    }
    wh_report(large ? "large" : "small", img, (size_t)M * OUT_S * OUT_S, 0u);

    free(W1); free(b1); free(W2); free(b2); free(K1); free(c1); free(K2); free(c2);
    free(z); free(h1); free(h2); free(up); free(f1); free(img);
    return 0;
}
