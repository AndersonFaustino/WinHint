/**
 * @file contrast_mobilenet_infer.c
 * @brief WinHint Project - Contrast group (non-transformer), benchmark 1: MobileNetV1.
 *
 * MobileNetV1-style CNN (Howard et al., 2017), width multiplier 1.0 for the
 * first seven layers, CHW layout:
 *
 *   stem conv3x3 s2 3->32, then depthwise-separable blocks
 *   (32->64 s1) (64->128 s2) (128->128 s1) (128->256 s2) (256->256 s1)
 *   (256->512 s2), global average pool, FC 512->1000
 *
 * @verbatim
 *   input  | image       | largest activation      | repetitions (default)
 *   -------+-------------+-------------------------+----------------------
 *   large  | 128x128x3   | 64x64x64 floats = 1 MB  | 1
 *   small  |  64x64x3    | 32x32x64 floats = 256 kB| 1
 * @endverbatim
 *
 *   argv[1] = small|large (default large)   argv[2] = repetitions (default 1)
 *
 * Phases (separate non-inlined functions / loop nests): conv3x3_stem,
 * depthwise_conv3x3 (bandwidth bound, spatial reuse), pointwise_conv
 * (GEMM-like, compute bound), bn_relu6, global_avg_pool, fc.
 * Output: float checksum + FNV-1a hash of the 1000 logits.
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

#define KERNEL_NAME "contrast_mobilenet" ///< Name printed as the "[name]" prefix of every output line.

#ifndef IMG_LARGE
#define IMG_LARGE 128 ///< Input image side S for the large input.
#endif
#ifndef IMG_SMALL
#define IMG_SMALL 64 ///< Input image side S for the small input.
#endif
#define BLOCKS 6 ///< Number of depthwise-separable blocks.
#define STEM_C 32 ///< Output channels of the stem convolution.
#define N_CLASSES 1000 ///< Classifier outputs (logits).
static const int CH_IN[BLOCKS] = {32, 64, 128, 128, 256, 256};   ///< Input channels per block.
static const int CH_OUT[BLOCKS] = {64, 128, 128, 256, 256, 512}; ///< Output channels per block.
static const int STRIDE[BLOCKS] = {1, 2, 1, 2, 1, 2};             ///< Depthwise stride per block.

/* ---- Deterministic PRNG: values in [-0.1, 0.1) -------------------------- */
/** @brief xorshift32 state of the deterministic PRNG (fixed per-kernel seed). */
static uint32_t wh_rng = 0x1234abcdu;
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
 * @brief Stem 3x3 conv, stride 2, pad 1: in [3][S][S] -> out [STEM_C][S/2][S/2].
 * @param[out] out Output activations, [STEM_C][S/2][S/2].
 * @param[in]  in  Input image, [3][S][S] (CHW).
 * @param[in]  K   Kernels, [STEM_C][3][3][3].
 * @param[in]  S   Input side.
 */
WH_NOINLINE static void conv3x3_stem(float *out, const float *in, const float *K, int S) {
    const int So = S / 2;
    for (int co = 0; co < STEM_C; co++)
        for (int y = 0; y < So; y++)
            for (int x = 0; x < So; x++) {
                float acc = 0.0f;
                for (int ci = 0; ci < 3; ci++)
                    for (int ky = 0; ky < 3; ky++)
                        for (int kx = 0; kx < 3; kx++) {
                            int iy = 2 * y + ky - 1, ix = 2 * x + kx - 1;
                            if (iy >= 0 && iy < S && ix >= 0 && ix < S)
                                acc += in[(ci * S + iy) * S + ix] * K[((co * 3 + ci) * 3 + ky) * 3 + kx];
                        }
                out[(co * So + y) * So + x] = acc;
            }
}

/**
 * @brief Depthwise 3x3 conv, pad 1, stride st: in [C][S][S] -> out [C][S/st][S/st].
 * @param[out] out Output activations, [C][S/st][S/st].
 * @param[in]  in  Input activations, [C][S][S].
 * @param[in]  K   Per-channel kernels, [C][3][3].
 * @param[in]  C   Channels.
 * @param[in]  S   Input side.
 * @param[in]  st  Stride (1 or 2).
 */
WH_NOINLINE static void depthwise_conv3x3(float *out, const float *in, const float *K,
                                          int C, int S, int st) {
    const int So = S / st;
    for (int c = 0; c < C; c++)
        for (int y = 0; y < So; y++)
            for (int x = 0; x < So; x++) {
                float acc = 0.0f;
                for (int ky = 0; ky < 3; ky++)
                    for (int kx = 0; kx < 3; kx++) {
                        int iy = st * y + ky - 1, ix = st * x + kx - 1;
                        if (iy >= 0 && iy < S && ix >= 0 && ix < S)
                            acc += in[(c * S + iy) * S + ix] * K[(c * 3 + ky) * 3 + kx];
                    }
                out[(c * So + y) * So + x] = acc;
            }
}

/**
 * @brief Pointwise 1x1 conv: out[co][p] = sum_ci W[co][ci] * in[ci][p]
 *        (axpy over pixels).
 * @param[out] out  Output activations, [Cout][HW].
 * @param[in]  in   Input activations, [Cin][HW].
 * @param[in]  W    Weights, [Cout][Cin].
 * @param[in]  Cin  Input channels.
 * @param[in]  Cout Output channels.
 * @param[in]  HW   Pixels per channel.
 */
WH_NOINLINE static void pointwise_conv(float *out, const float *in, const float *W,
                                       int Cin, int Cout, int HW) {
    for (int co = 0; co < Cout; co++) {
        float *o = out + (size_t)co * HW;
        for (int p = 0; p < HW; p++) o[p] = 0.0f;
        for (int ci = 0; ci < Cin; ci++) {
            const float w = W[co * Cin + ci];
            const float *i = in + (size_t)ci * HW;
            for (int p = 0; p < HW; p++) o[p] += w * i[p];
        }
    }
}

/**
 * @brief Folded batch norm (per-channel scale/shift) + ReLU6, in place.
 * @param[in,out] x     Activations, [C][HW].
 * @param[in]     scale Per-channel scale [C].
 * @param[in]     shift Per-channel shift [C].
 * @param[in]     C     Channels.
 * @param[in]     HW    Pixels per channel.
 */
WH_NOINLINE static void bn_relu6(float *x, const float *scale, const float *shift, int C, int HW) {
    for (int c = 0; c < C; c++)
        for (int p = 0; p < HW; p++) {
            float v = x[c * HW + p] * scale[c] + shift[c];
            x[c * HW + p] = v < 0.0f ? 0.0f : (v > 6.0f ? 6.0f : v);
        }
}

/**
 * @brief Global average pool: out[c] = mean of channel c.
 * @param[out] out Pooled features [C].
 * @param[in]  in  Activations, [C][HW].
 * @param[in]  C   Channels.
 * @param[in]  HW  Pixels per channel.
 */
WH_NOINLINE static void global_avg_pool(float *out, const float *in, int C, int HW) {
    for (int c = 0; c < C; c++) {
        float s = 0.0f;
        for (int p = 0; p < HW; p++) s += in[c * HW + p];
        out[c] = s / (float)HW;
    }
}

/**
 * @brief Fully connected layer (GEMV): out = W in + b.
 * @param[out] out Outputs [N].
 * @param[in]  in  Inputs [K].
 * @param[in]  W   Weights, [N][K].
 * @param[in]  b   Bias [N].
 * @param[in]  N   Outputs.
 * @param[in]  K   Inputs.
 */
WH_NOINLINE static void fc(float *out, const float *in, const float *W, const float *b,
                           int N, int K) {
    for (int n = 0; n < N; n++) {
        float acc = b[n];
        for (int k = 0; k < K; k++) acc += W[n * K + k] * in[k];
        out[n] = acc;
    }
}

/**
 * @brief Allocate a batch-norm scale vector with values 1 + wh_rand().
 * @param[in] C Channels.
 * @return The new buffer (exits on allocation failure).
 */
static float *bn_scale(int C) {
    float *p = wh_alloc((size_t)C);
    for (int i = 0; i < C; i++) p[i] = 1.0f + wh_rand();
    return p;
}

/**
 * @brief Run the stem, the BLOCKS depthwise-separable blocks, global average
 *        pooling and the classifier per repetition, and report the logits.
 *
 * Activations ping-pong between two buffers of 64 * (S/2)^2 floats. Prints a
 * configuration line, then wh_report() over the N_CLASSES logits.
 * @param[in] argc Argument count.
 * @param[in] argv argv[1] = small|large, argv[2] = repetitions.
 * @return 0 on success (exits 1 on allocation failure, 2 on bad arguments).
 */
int main(int argc, char **argv) {
    int reps;
    int large = wh_parse_args(argc, argv, &reps);
    const int S = large ? IMG_LARGE : IMG_SMALL;
    printf("[%s] input=%s image=%dx%dx3 blocks=%d reps=%d\n", KERNEL_NAME,
           large ? "large" : "small", S, S, BLOCKS, reps);

    float *K_stem = wh_alloc_rand((size_t)STEM_C * 27);
    float *s_stem = bn_scale(STEM_C), *t_stem = wh_alloc_rand(STEM_C);
    float *Kdw[BLOCKS], *Wpw[BLOCKS], *s_dw[BLOCKS], *t_dw[BLOCKS], *s_pw[BLOCKS], *t_pw[BLOCKS];
    for (int b = 0; b < BLOCKS; b++) {
        Kdw[b] = wh_alloc_rand((size_t)CH_IN[b] * 9);
        Wpw[b] = wh_alloc_rand((size_t)CH_OUT[b] * CH_IN[b]);
        s_dw[b] = bn_scale(CH_IN[b]); t_dw[b] = wh_alloc_rand(CH_IN[b]);
        s_pw[b] = bn_scale(CH_OUT[b]); t_pw[b] = wh_alloc_rand(CH_OUT[b]);
    }
    float *W_fc = wh_alloc_rand((size_t)N_CLASSES * 512), *b_fc = wh_alloc_rand(N_CLASSES);
    float *img = wh_alloc_rand((size_t)3 * S * S);
    size_t maxact = (size_t)64 * (S / 2) * (S / 2);
    float *A = wh_alloc(maxact), *B = wh_alloc(maxact);
    float pooled[512], logits[N_CLASSES];

    for (int r = 0; r < reps; r++) {
        int sp = S / 2;
        conv3x3_stem(A, img, K_stem, S);
        bn_relu6(A, s_stem, t_stem, STEM_C, sp * sp);
        for (int b = 0; b < BLOCKS; b++) {
            depthwise_conv3x3(B, A, Kdw[b], CH_IN[b], sp, STRIDE[b]);
            sp /= STRIDE[b];
            bn_relu6(B, s_dw[b], t_dw[b], CH_IN[b], sp * sp);
            pointwise_conv(A, B, Wpw[b], CH_IN[b], CH_OUT[b], sp * sp);
            bn_relu6(A, s_pw[b], t_pw[b], CH_OUT[b], sp * sp);
        }
        global_avg_pool(pooled, A, 512, sp * sp);
        fc(logits, pooled, W_fc, b_fc, N_CLASSES, 512);
    }
    wh_report(large ? "large" : "small", logits, N_CLASSES, 0u);

    free(K_stem); free(s_stem); free(t_stem);
    for (int b = 0; b < BLOCKS; b++) {
        free(Kdw[b]); free(Wpw[b]); free(s_dw[b]); free(t_dw[b]); free(s_pw[b]); free(t_pw[b]);
    }
    free(W_fc); free(b_fc); free(img); free(A); free(B);
    return 0;
}
