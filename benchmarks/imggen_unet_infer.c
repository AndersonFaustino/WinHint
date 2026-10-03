/**
 * @file imggen_unet_infer.c
 * @brief WinHint Project - Image-generation group (non-transformer), benchmark 2:
 * UNet-style decoder (upsampling path with skip connections).
 *
 *   3 stages of [nearest upsample 2x + conv3x3 + skip add + ReLU]
 *   channels 128 -> 64 -> 32 -> 16, then 1x1 conv 16 -> 1
 *
 * @verbatim
 *   input  | latent     | output   | weights   | repetitions (default)
 *   -------+------------+----------+-----------+----------------------
 *   large  | 8x8x128    | 64x64x1  | 0.4 MB    | 1
 *   small  | 4x4x128    | 32x32x1  | 0.4 MB    | 1
 * @endverbatim
 *
 *   argv[1] = small|large (default large)   argv[2] = repetitions (default 1)
 *
 * Phases (separate non-inlined functions / loop nests): upsample2, conv3x3,
 * skip_add_relu, conv1x1.
 * Output: float checksum + FNV-1a hash of the output image.
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

#define KERNEL_NAME "imggen_unet" ///< Name printed as the "[name]" prefix of every output line.

#ifndef LAT_LARGE
#define LAT_LARGE 8 ///< Latent side L for the large input.
#endif
#ifndef LAT_SMALL
#define LAT_SMALL 4 ///< Latent side L for the small input.
#endif
#define STAGES 3 ///< Upsampling stages (output side = L << STAGES).
static const int CH[STAGES + 1] = {128, 64, 32, 16}; ///< Channels entering each stage, then the final width.

/* ---- Deterministic PRNG: values in [-0.1, 0.1) -------------------------- */
/** @brief xorshift32 state of the deterministic PRNG (fixed per-kernel seed). */
static uint32_t wh_rng = 0x13572468u;
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
 * @brief Skip connection: y = relu(y + skip), in place.
 * @param[in,out] y    Decoder activations.
 * @param[in]     skip Encoder features (same shape).
 * @param[in]     n    Number of elements.
 */
WH_NOINLINE static void skip_add_relu(float *y, const float *skip, int n) {
    for (int i = 0; i < n; i++) {
        float v = y[i] + skip[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

/**
 * @brief 1x1 conv to a single output channel: out[p] = b + sum_c W[c] * in[c][p].
 * @param[out] out Output image [HW].
 * @param[in]  in  Input activations, [Cin][HW].
 * @param[in]  W   Weights [Cin].
 * @param[in]  b   Bias.
 * @param[in]  Cin Input channels.
 * @param[in]  HW  Pixels.
 */
WH_NOINLINE static void conv1x1(float *out, const float *in, const float *W, float b,
                                int Cin, int HW) {
    for (int p = 0; p < HW; p++) out[p] = b;
    for (int c = 0; c < Cin; c++)
        for (int p = 0; p < HW; p++) out[p] += W[c] * in[(size_t)c * HW + p];
}

/**
 * @brief Run the STAGES upsampling stages and the output 1x1 conv per repetition
 *        and report the output image.
 *
 * The skip-connection "encoder features" are fixed pseudo-random tensors. Prints
 * a configuration line, then wh_report() over the (L << STAGES)^2 output pixels.
 * @param[in] argc Argument count.
 * @param[in] argv argv[1] = small|large, argv[2] = repetitions.
 * @return 0 on success (exits 1 on allocation failure, 2 on bad arguments).
 */
int main(int argc, char **argv) {
    int reps;
    int large = wh_parse_args(argc, argv, &reps);
    const int L = large ? LAT_LARGE : LAT_SMALL;
    const int So = L << STAGES;
    printf("[%s] input=%s latent=%dx%dx%d output=%dx%d reps=%d\n", KERNEL_NAME,
           large ? "large" : "small", L, L, CH[0], So, So, reps);

    float *K[STAGES], *b[STAGES], *skip[STAGES];
    for (int s = 0; s < STAGES; s++) {
        int sp = L << (s + 1);
        K[s] = wh_alloc_rand((size_t)CH[s + 1] * CH[s] * 9);
        b[s] = wh_alloc_rand(CH[s + 1]);
        skip[s] = wh_alloc_rand((size_t)CH[s + 1] * sp * sp); /* encoder features */
    }
    float *W_out = wh_alloc_rand(CH[STAGES]);
    float b_out = wh_rand();
    float *lat = wh_alloc_rand((size_t)CH[0] * L * L);
    size_t maxact = 0;
    for (int s = 0; s < STAGES; s++) {
        size_t a = (size_t)CH[s] * (L << (s + 1)) * (L << (s + 1));
        if (a > maxact) maxact = a;
    }
    float *A = wh_alloc(maxact), *B = wh_alloc(maxact);
    float *out = wh_alloc((size_t)So * So);

    for (int r = 0; r < reps; r++) {
        int sp = L;
        memcpy(A, lat, (size_t)CH[0] * L * L * sizeof(float));
        for (int s = 0; s < STAGES; s++) {
            upsample2(B, A, CH[s], sp);
            sp *= 2;
            conv3x3(A, B, K[s], b[s], CH[s], CH[s + 1], sp);
            skip_add_relu(A, skip[s], CH[s + 1] * sp * sp);
        }
        conv1x1(out, A, W_out, b_out, CH[STAGES], sp * sp);
    }
    wh_report(large ? "large" : "small", out, (size_t)So * So, 0u);

    for (int s = 0; s < STAGES; s++) { free(K[s]); free(b[s]); free(skip[s]); }
    free(W_out); free(lat); free(A); free(B); free(out);
    return 0;
}
