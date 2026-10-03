/**
 * @file decoder_beam_search_infer.c
 * @brief WinHint Project - Decoder group, benchmark 3: GPT-2 small with beam search.
 *
 * GPT-2-small layer shapes (hidden H = 768, heads = 12, head_dim = 64,
 * FFN F = 3072, pre-LN, GELU, tied LM head over a reduced 8192 vocabulary)
 * decoded with beam search (B = 4 beams):
 *
 * @verbatim
 *   input  | prompt | generated | context | beams | layers (default)
 *   -------+--------+-----------+---------+-------+-----------------
 *   large  |   64   |    16     |    80   |   4   | 1
 *   small  |    2   |     2     |     4   |   4   | 1
 * @endverbatim
 *
 *   argv[1] = small|large (default large)   argv[2] = number of layers (default 1)
 *
 * Per decoding step the 4 beams are batched (M = 4 GEMMs over 28.3 MB of
 * layer weights + the 25 MB LM head), each beam attends over its own KV cache
 * (4 x ctx x 768 x 2 floats = 2 MB per layer for large), then beam_select
 * runs data-dependent top-k selection and kv_reorder copies the surviving
 * beams' caches (a pure memory phase). Prefill runs on beam 0 only; the
 * first reorder broadcasts its cache.
 *
 * Phases (separate non-inlined functions / loop nests): embed, layer_norm,
 * linear, kv_append, attention_scores, softmax_causal, attention_context,
 * residual_add, gelu_inplace, log_softmax_rows, beam_select, kv_reorder.
 *
 * Build options: -DTILED (register-blocked, N-outer tiled GEMM with the same
 * per-output summation order => bit-identical output); shape overrides
 * -DHIDDEN, -DFFN_DIM, -DNUM_HEADS, -DVOCAB.
 *
 * Output: float checksum + FNV-1a hash over the final log-probabilities,
 * mixed with the best beam's tokens. Portable C11; deterministic.
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
#if defined(__GNUC__) || defined(__clang__)
#define WH_NOINLINE __attribute__((noinline))
#else
#define WH_NOINLINE
#endif

#define KERNEL_NAME "decoder_beam_search" ///< Name printed as the "[name]" prefix of every output line.

/* ---- Model shape (GPT-2 small) ----------------------------------------- */
#ifndef HIDDEN
#define HIDDEN 768 ///< Hidden size H (override with -DHIDDEN).
#endif
#ifndef NUM_HEADS
#define NUM_HEADS 12 ///< Attention (query) heads (override with -DNUM_HEADS).
#endif
#define HEAD_DIM (HIDDEN / NUM_HEADS) ///< Per-head dimension.
#ifndef FFN_DIM
#define FFN_DIM 3072 ///< FFN inner size F (override with -DFFN_DIM).
#endif
#ifndef VOCAB
#define VOCAB 8192 ///< Reduced vocabulary size (override with -DVOCAB).
#endif
#define BEAMS 4 ///< Beam width B.
#ifndef PROMPT_LARGE
#define PROMPT_LARGE 64 ///< Prompt tokens for the large input.
#endif
#ifndef GEN_LARGE
#define GEN_LARGE 16 ///< Generated tokens for the large input.
#endif
#ifndef PROMPT_SMALL
#define PROMPT_SMALL 2 ///< Prompt tokens for the small input.
#endif
#ifndef GEN_SMALL
#define GEN_SMALL 2 ///< Generated tokens for the small input.
#endif

/* ---- Deterministic PRNG: values in [-0.1, 0.1) -------------------------- */
/** @brief xorshift32 state of the deterministic PRNG (fixed per-kernel seed). */
static uint32_t wh_rng = 0x00000007u;
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
static float *wh_alloc_rand(size_t n) {
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
static float *wh_alloc_const(size_t n, float v) {
    float *p = wh_alloc(n);
    for (size_t i = 0; i < n; i++) p[i] = v;
    return p;
}

/**
 * @brief Parse the command line: argv[1] = small|large (default large);
 *        argv[2] = layers (default 1).
 *
 * Prints usage and exits with status 2 on an unknown size or a count < 1.
 * @param[in]  argc   Argument count from main().
 * @param[in]  argv   Argument vector from main().
 * @param[out] layers Receives the number of layers (argv[2], default 1).
 * @return 1 for the large input, 0 for small.
 */
static int wh_parse_args(int argc, char **argv, int *layers) {
    int large = 1;
    *layers = 1;
    if (argc > 1) {
        if (strcmp(argv[1], "small") == 0) large = 0;
        else if (strcmp(argv[1], "large") == 0) large = 1;
        else {
            fprintf(stderr, "usage: %s [small|large] [layers]\n", argv[0]);
            exit(2);
        }
    }
    if (argc > 2) {
        *layers = atoi(argv[2]);
        if (*layers < 1) { fprintf(stderr, "layers must be >= 1\n"); exit(2); }
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

/* ---- Causal attention over the KV cache --------------------------------
 * Rows i = 0..M-1 are at absolute positions pos0 + i and see keys 0..pos0+i.
 * P is [NUM_HEADS][M][ctx] (row stride ctx). */
/**
 * @brief Causal scaled dot-product scores against the KV cache: for query row i
 *        (position pos0 + i), p[j] = <q, Kc[j]> / sqrt(HEAD_DIM) for j <= pos0 + i.
 *
 * Entries beyond the causal limit are left untouched.
 * @param[out] P    Scores, [NUM_HEADS][M][ctx] (row stride ctx).
 * @param[in]  Q    Queries, [M][HIDDEN].
 * @param[in]  Kc   Key cache, [ctx][HIDDEN].
 * @param[in]  M    Query rows.
 * @param[in]  pos0 Absolute position of row 0.
 * @param[in]  ctx  Cache capacity (row stride of P).
 */
WH_NOINLINE static void attention_scores(float *P, const float *Q, const float *Kc,
                                         int M, int pos0, int ctx) {
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (int h = 0; h < NUM_HEADS; h++)
        for (int i = 0; i < M; i++) {
            const float *q = Q + i * HIDDEN + h * HEAD_DIM;
            float *p = P + ((size_t)h * M + i) * ctx;
            for (int j = 0; j <= pos0 + i; j++) {
                const float *k = Kc + (size_t)j * HIDDEN + h * HEAD_DIM;
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) dot += q[d] * k[d];
                p[j] = dot * scale;
            }
        }
}

/**
 * @brief Causal softmax, in place: row r (query i = r % M) is normalized over its
 *        first pos0 + i + 1 entries.
 * @param[in,out] P    Scores in, probabilities out, [NUM_HEADS][M][ctx].
 * @param[in]     M    Query rows.
 * @param[in]     pos0 Absolute position of row 0.
 * @param[in]     ctx  Row stride.
 */
WH_NOINLINE static void softmax_causal(float *P, int M, int pos0, int ctx) {
    for (int r = 0; r < NUM_HEADS * M; r++) {
        float *x = P + (size_t)r * ctx;
        int n = pos0 + (r % M) + 1;
        float mx = x[0];
        for (int j = 1; j < n; j++) if (x[j] > mx) mx = x[j];
        float sum = 0.0f;
        for (int j = 0; j < n; j++) { x[j] = expf(x[j] - mx); sum += x[j]; }
        float inv = 1.0f / sum;
        for (int j = 0; j < n; j++) x[j] *= inv;
    }
}

/**
 * @brief Causal attention context: C[i, h*HEAD_DIM+d] = sum_{j<=pos0+i} P[h][i][j] * Vc[j, h*HEAD_DIM+d].
 * @param[out] C    Context, [M][HIDDEN].
 * @param[in]  P    Probabilities, [NUM_HEADS][M][ctx].
 * @param[in]  Vc   Value cache, [ctx][HIDDEN].
 * @param[in]  M    Query rows.
 * @param[in]  pos0 Absolute position of row 0.
 * @param[in]  ctx  Row stride of P.
 */
WH_NOINLINE static void attention_context(float *C, const float *P, const float *Vc,
                                          int M, int pos0, int ctx) {
    for (int h = 0; h < NUM_HEADS; h++)
        for (int i = 0; i < M; i++) {
            float *c = C + i * HIDDEN + h * HEAD_DIM;
            const float *p = P + ((size_t)h * M + i) * ctx;
            for (int d = 0; d < HEAD_DIM; d++) c[d] = 0.0f;
            for (int j = 0; j <= pos0 + i; j++) {
                const float *v = Vc + (size_t)j * HIDDEN + h * HEAD_DIM;
                float pj = p[j];
                for (int d = 0; d < HEAD_DIM; d++) c[d] += pj * v[d];
            }
        }
}

/**
 * @brief Copy M new K/V rows into the cache at position pos0 (memory phase).
 * @param[in,out] Kc   Key cache, [ctx][HIDDEN].
 * @param[in,out] Vc   Value cache, [ctx][HIDDEN].
 * @param[in]     K    New keys, [M][HIDDEN].
 * @param[in]     V    New values, [M][HIDDEN].
 * @param[in]     M    Rows to append.
 * @param[in]     pos0 Cache row of the first new entry.
 */
WH_NOINLINE static void kv_append(float *Kc, float *Vc, const float *K, const float *V,
                                  int M, int pos0) {
    for (int i = 0; i < M; i++)
        for (int d = 0; d < HIDDEN; d++) {
            Kc[(size_t)(pos0 + i) * HIDDEN + d] = K[i * HIDDEN + d];
            Vc[(size_t)(pos0 + i) * HIDDEN + d] = V[i * HIDDEN + d];
        }
}

/**
 * @brief Residual connection: y += x, element-wise.
 * @param[in,out] y Accumulator.
 * @param[in]     x Addend.
 * @param[in]     n Number of elements.
 */
WH_NOINLINE static void residual_add(float *y, const float *x, int n) {
    for (int i = 0; i < n; i++) y[i] += x[i];
}

/**
 * @brief Row-wise LayerNorm: y = g * (x - mean) / sqrt(var + 1e-5) + b.
 * @param[out] y    Output, [rows][n].
 * @param[in]  x    Input, [rows][n].
 * @param[in]  g    Gain [n].
 * @param[in]  b    Bias [n].
 * @param[in]  rows Number of rows.
 * @param[in]  n    Row length.
 */
WH_NOINLINE static void layer_norm(float *y, const float *x, const float *g,
                                   const float *b, int rows, int n) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + r * n;
        float *yr = y + r * n;
        float mean = 0.0f, var = 0.0f;
        for (int i = 0; i < n; i++) mean += xr[i];
        mean /= (float)n;
        for (int i = 0; i < n; i++) { float d = xr[i] - mean; var += d * d; }
        float inv = 1.0f / sqrtf(var / (float)n + 1e-5f);
        for (int i = 0; i < n; i++) yr[i] = g[i] * ((xr[i] - mean) * inv) + b[i];
    }
}

/**
 * @brief GELU in place, tanh approximation (Hendrycks & Gimpel).
 *
 * Written with the exact identity 0.5*(1 + tanh(u)) = 1 / (1 + exp(-2u)): glibc's
 * tanhf differs between x86-64 and RISC-V in the last bit, expf does not, so this
 * form keeps the output bit-identical across architectures.
 * @param[in,out] x Activations.
 * @param[in]     n Number of elements.
 */
WH_NOINLINE static void gelu_inplace(float *x, int n) {
    const float c2 = 1.5957691216f; /* 2 * sqrt(2/pi) */
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-c2 * (v + 0.044715f * v * v * v)));
    }
}

/**
 * @brief Embedding lookup: x[i] = wte[tokens[i]] + wpe[pos0 + i].
 * @param[out] x      Hidden states, [M][HIDDEN].
 * @param[in]  wte    Token embeddings, [VOCAB][HIDDEN] (also the tied LM head).
 * @param[in]  wpe    Position embeddings, [ctx][HIDDEN].
 * @param[in]  tokens Token ids [M].
 * @param[in]  M      Rows.
 * @param[in]  pos0   Position of row 0.
 */
WH_NOINLINE static void embed(float *x, const float *wte, const float *wpe,
                              const int *tokens, int M, int pos0) {
    for (int i = 0; i < M; i++)
        for (int d = 0; d < HIDDEN; d++)
            x[i * HIDDEN + d] = wte[(size_t)tokens[i] * HIDDEN + d] +
                                wpe[(size_t)(pos0 + i) * HIDDEN + d];
}

/**
 * @brief Row-wise log-softmax over `rows` rows of length n (in place).
 * @param[in,out] x    Logits in, log-probabilities out, [rows][n].
 * @param[in]     rows Number of rows.
 * @param[in]     n    Row length.
 */
WH_NOINLINE static void log_softmax_rows(float *x, int rows, int n) {
    for (int r = 0; r < rows; r++) {
        float *v = x + (size_t)r * n;
        float mx = v[0];
        for (int i = 1; i < n; i++) if (v[i] > mx) mx = v[i];
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += expf(v[i] - mx);
        float lse = mx + logf(sum);
        for (int i = 0; i < n; i++) v[i] -= lse;
    }
}

/**
 * @brief Keep the BEAMS best (score[b] + logp[b][v]) candidates over `live` beams.
 *
 * Data-dependent insertion into a sorted top-k list (descending score; on ties
 * the earlier candidate stays ahead).
 * @param[in]  logp      Log-probabilities, [live][VOCAB].
 * @param[in]  score     Accumulated scores of the live beams [live].
 * @param[in]  live      Live beams (1 after prefill, BEAMS afterwards).
 * @param[out] parent    Source beam of each survivor [BEAMS].
 * @param[out] token     Token chosen for each survivor [BEAMS].
 * @param[out] new_score Scores of the survivors [BEAMS], best first.
 */
WH_NOINLINE static void beam_select(const float *logp, const float *score, int live,
                                    int *parent, int *token, float *new_score) {
    int cnt = 0;
    for (int b = 0; b < live; b++)
        for (int v = 0; v < VOCAB; v++) {
            float s = score[b] + logp[(size_t)b * VOCAB + v];
            if (cnt == BEAMS && s <= new_score[BEAMS - 1]) continue;
            int pos = cnt < BEAMS ? cnt++ : BEAMS - 1;
            while (pos > 0 && new_score[pos - 1] < s) {
                new_score[pos] = new_score[pos - 1];
                parent[pos] = parent[pos - 1];
                token[pos] = token[pos - 1];
                pos--;
            }
            new_score[pos] = s; parent[pos] = b; token[pos] = v;
        }
}

/**
 * @brief KV cache reorder: dst[b][0..len) = src[parent[b]][0..len) (memory bound).
 * @param[out] dst    Destination caches, [BEAMS][ctx][HIDDEN].
 * @param[in]  src    Source caches, [BEAMS][ctx][HIDDEN].
 * @param[in]  parent Source beam per destination beam [BEAMS].
 * @param[in]  len    Cache rows to copy.
 * @param[in]  ctx    Cache capacity per beam.
 */
WH_NOINLINE static void kv_reorder(float *dst, const float *src, const int *parent,
                                   int len, int ctx) {
    for (int b = 0; b < BEAMS; b++) {
        const float *s = src + (size_t)parent[b] * ctx * HIDDEN;
        float *d = dst + (size_t)b * ctx * HIDDEN;
        for (size_t i = 0; i < (size_t)len * HIDDEN; i++) d[i] = s[i];
    }
}

/* ---- Layer ------------------------------------------------------------- */
/** @brief Weights and per-beam KV caches of one GPT-2 block (W* in [out][in] nn.Linear layout). */
typedef struct {
    float *Wq, *Wk, *Wv, *Wo, *bq, *bk, *bv, *bo; ///< Attention Q/K/V/output weights [HIDDEN][HIDDEN] and biases.
    float *W1, *b1, *W2, *b2;                     ///< FFN weights W1 [FFN_DIM][HIDDEN], W2 [HIDDEN][FFN_DIM] and biases.
    float *ln1_g, *ln1_b, *ln2_g, *ln2_b;         ///< LayerNorm gains (1) and biases (0) [HIDDEN].
    float *Kc, *Vc, *Kt, *Vt; ///< KV caches [BEAMS][ctx][HIDDEN] + reorder buffers
} Layer;

/**
 * @brief Activation buffers sized for rows = max(prompt, BEAMS), reused across
 *        layers and steps.
 *
 * H, Q, K, V, C, A: [rows][HIDDEN]; P: scores [NUM_HEADS][rows][ctx];
 * M: FFN hidden activations [rows][FFN_DIM].
 */
typedef struct {
    float *H, *Q, *K, *V, *P, *C, *A, *M;
} Scratch;

/**
 * @brief Allocate one layer: pseudo-random weights and biases, LayerNorm gains 1
 *        and biases 0, zeroed KV cache and reorder buffers [BEAMS][ctx][HIDDEN].
 * @param[out] L   Layer to initialize.
 * @param[in]  ctx Context length (cache rows).
 */
static void init_layer(Layer *L, int ctx) {
    size_t kv = (size_t)BEAMS * ctx * HIDDEN;
    L->Wq = wh_alloc_rand((size_t)HIDDEN * HIDDEN); L->bq = wh_alloc_rand(HIDDEN);
    L->Wk = wh_alloc_rand((size_t)HIDDEN * HIDDEN); L->bk = wh_alloc_rand(HIDDEN);
    L->Wv = wh_alloc_rand((size_t)HIDDEN * HIDDEN); L->bv = wh_alloc_rand(HIDDEN);
    L->Wo = wh_alloc_rand((size_t)HIDDEN * HIDDEN); L->bo = wh_alloc_rand(HIDDEN);
    L->W1 = wh_alloc_rand((size_t)FFN_DIM * HIDDEN); L->b1 = wh_alloc_rand(FFN_DIM);
    L->W2 = wh_alloc_rand((size_t)HIDDEN * FFN_DIM); L->b2 = wh_alloc_rand(HIDDEN);
    L->ln1_g = wh_alloc_const(HIDDEN, 1.0f); L->ln1_b = wh_alloc_const(HIDDEN, 0.0f);
    L->ln2_g = wh_alloc_const(HIDDEN, 1.0f); L->ln2_b = wh_alloc_const(HIDDEN, 0.0f);
    L->Kc = wh_alloc_const(kv, 0.0f); L->Vc = wh_alloc_const(kv, 0.0f);
    L->Kt = wh_alloc_const(kv, 0.0f); L->Vt = wh_alloc_const(kv, 0.0f);
}

/**
 * @brief Free every buffer of a layer allocated by init_layer().
 * @param[in] L Layer to release.
 */
static void free_layer(Layer *L) {
    float *p[] = {L->Wq, L->Wk, L->Wv, L->Wo, L->bq, L->bk, L->bv, L->bo, L->W1, L->b1,
                  L->W2, L->b2, L->ln1_g, L->ln1_b, L->ln2_g, L->ln2_b,
                  L->Kc, L->Vc, L->Kt, L->Vt};
    for (size_t i = 0; i < sizeof p / sizeof p[0]; i++) free(p[i]);
}

/**
 * @brief Pre-LN block. prefill: M prompt rows on beam 0 (causal);
 *        decode (beams != 0): M = BEAMS rows, row b attends over beam b's cache.
 * @param[in,out] x     Hidden states, [M][HIDDEN].
 * @param[in,out] L     Layer weights and caches (M rows appended, or one per beam).
 * @param         s     Scratch buffers.
 * @param[in]     M     Rows.
 * @param[in]     pos0  Absolute position of row 0.
 * @param[in]     ctx   Context length (cache capacity per beam).
 * @param[in]     beams 0 for prefill, nonzero for batched beam decode.
 */
static void decoder_layer(float *x, Layer *L, Scratch *s, int M, int pos0, int ctx,
                          int beams) {
    const size_t bstride = (size_t)ctx * HIDDEN;
    layer_norm(s->H, x, L->ln1_g, L->ln1_b, M, HIDDEN);
    linear(s->Q, s->H, L->Wq, L->bq, M, HIDDEN, HIDDEN);
    linear(s->K, s->H, L->Wk, L->bk, M, HIDDEN, HIDDEN);
    linear(s->V, s->H, L->Wv, L->bv, M, HIDDEN, HIDDEN);
    if (!beams) {
        kv_append(L->Kc, L->Vc, s->K, s->V, M, pos0);
        attention_scores(s->P, s->Q, L->Kc, M, pos0, ctx);
        softmax_causal(s->P, M, pos0, ctx);
        attention_context(s->C, s->P, L->Vc, M, pos0, ctx);
    } else {
        for (int b = 0; b < BEAMS; b++) {
            float *P = s->P + (size_t)b * NUM_HEADS * ctx;
            kv_append(L->Kc + b * bstride, L->Vc + b * bstride, s->K + b * HIDDEN,
                      s->V + b * HIDDEN, 1, pos0);
            attention_scores(P, s->Q + b * HIDDEN, L->Kc + b * bstride, 1, pos0, ctx);
            softmax_causal(P, 1, pos0, ctx);
            attention_context(s->C + b * HIDDEN, P, L->Vc + b * bstride, 1, pos0, ctx);
        }
    }
    linear(s->A, s->C, L->Wo, L->bo, M, HIDDEN, HIDDEN);
    residual_add(x, s->A, M * HIDDEN);
    layer_norm(s->H, x, L->ln2_g, L->ln2_b, M, HIDDEN);
    linear(s->M, s->H, L->W1, L->b1, M, FFN_DIM, HIDDEN);
    gelu_inplace(s->M, M * FFN_DIM);
    linear(s->A, s->M, L->W2, L->b2, M, HIDDEN, FFN_DIM);
    residual_add(x, s->A, M * HIDDEN);
}

/**
 * @brief Prefill the prompt on beam 0, then decode `gen` steps with beam search,
 *        and report the final log-probabilities.
 *
 * Each step: LM head (tied wte) over the live beams, log_softmax_rows,
 * beam_select, token-history update, and kv_reorder of every layer's caches
 * (swapping cache and reorder buffers). Prompt ids are
 * (i * 2246822519 + 3) % VOCAB. Prints a configuration line, the best beam's
 * score and first (up to 8) tokens, then wh_report() over the last step's
 * [BEAMS][VOCAB] log-probabilities with the hash mixed with the best beam's
 * token ids (th = th * 31 + token).
 * @param[in] argc Argument count.
 * @param[in] argv argv[1] = small|large, argv[2] = number of layers.
 * @return 0 on success; 1 if a malloc() in main fails (wh_alloc exits 1; bad
 *         arguments exit 2).
 */
int main(int argc, char **argv) {
    int layers;
    int large = wh_parse_args(argc, argv, &layers);
    const int prompt = large ? PROMPT_LARGE : PROMPT_SMALL;
    const int gen = large ? GEN_LARGE : GEN_SMALL;
    const int ctx = prompt + gen;
    const int rows = prompt > BEAMS ? prompt : BEAMS;
    printf("[%s] input=%s prompt=%d gen=%d ctx=%d beams=%d hidden=%d heads=%d ffn=%d vocab=%d layers=%d%s\n",
           KERNEL_NAME, large ? "large" : "small", prompt, gen, ctx, BEAMS, HIDDEN,
           NUM_HEADS, FFN_DIM, VOCAB, layers,
#ifdef TILED
           " gemm=tiled"
#else
           " gemm=naive"
#endif
    );

    float *wte = wh_alloc_rand((size_t)VOCAB * HIDDEN);
    float *wpe = wh_alloc_rand((size_t)ctx * HIDDEN);
    float *lnf_g = wh_alloc_const(HIDDEN, 1.0f), *lnf_b = wh_alloc_const(HIDDEN, 0.0f);
    Layer *Ls = (Layer *)malloc((size_t)layers * sizeof(Layer));
    if (!Ls) return 1;
    for (int l = 0; l < layers; l++) init_layer(&Ls[l], ctx);

    Scratch s;
    s.H = wh_alloc((size_t)rows * HIDDEN); s.Q = wh_alloc((size_t)rows * HIDDEN);
    s.K = wh_alloc((size_t)rows * HIDDEN); s.V = wh_alloc((size_t)rows * HIDDEN);
    s.C = wh_alloc((size_t)rows * HIDDEN); s.A = wh_alloc((size_t)rows * HIDDEN);
    s.P = wh_alloc((size_t)NUM_HEADS * rows * ctx);
    s.M = wh_alloc((size_t)rows * FFN_DIM);
    float *x = wh_alloc((size_t)rows * HIDDEN);
    float *logp = wh_alloc((size_t)BEAMS * VOCAB);

    int prompt_tok[PROMPT_LARGE > PROMPT_SMALL ? PROMPT_LARGE : PROMPT_SMALL];
    for (int i = 0; i < prompt; i++) prompt_tok[i] = (int)((i * 2246822519u + 3u) % VOCAB);
    int hist[BEAMS][GEN_LARGE > GEN_SMALL ? GEN_LARGE : GEN_SMALL];
    int hist_new[BEAMS][GEN_LARGE > GEN_SMALL ? GEN_LARGE : GEN_SMALL];
    float score[BEAMS] = {0}, new_score[BEAMS];
    int parent[BEAMS], token[BEAMS];

    for (int t = 0; t < gen; t++) {
        int M, pos0, live;
        if (t == 0) { /* prefill on beam 0 */
            M = prompt; pos0 = 0; live = 1;
            embed(x, wte, wpe, prompt_tok, M, 0);
            for (int l = 0; l < layers; l++) decoder_layer(x, &Ls[l], &s, M, 0, ctx, 0);
            layer_norm(s.H, x + (size_t)(M - 1) * HIDDEN, lnf_g, lnf_b, 1, HIDDEN);
        } else {
            M = BEAMS; pos0 = prompt + t - 1; live = BEAMS;
            int cur[BEAMS];
            for (int b = 0; b < BEAMS; b++) cur[b] = hist[b][t - 1];
            for (int b = 0; b < BEAMS; b++) embed(x + b * HIDDEN, wte, wpe, &cur[b], 1, pos0);
            for (int l = 0; l < layers; l++) decoder_layer(x, &Ls[l], &s, M, pos0, ctx, 1);
            layer_norm(s.H, x, lnf_g, lnf_b, BEAMS, HIDDEN);
        }
        linear(logp, s.H, wte, NULL, live, VOCAB, HIDDEN); /* tied LM head */
        log_softmax_rows(logp, live, VOCAB);
        beam_select(logp, score, live, parent, token, new_score);
        for (int b = 0; b < BEAMS; b++) {
            for (int i = 0; i < t; i++) hist_new[b][i] = hist[parent[b]][i];
            hist_new[b][t] = token[b];
            score[b] = new_score[b];
        }
        memcpy(hist, hist_new, sizeof hist);
        int len = t == 0 ? prompt : pos0 + 1;
        for (int l = 0; l < layers; l++) {
            float *tmp;
            kv_reorder(Ls[l].Kt, Ls[l].Kc, parent, len, ctx);
            kv_reorder(Ls[l].Vt, Ls[l].Vc, parent, len, ctx);
            tmp = Ls[l].Kc; Ls[l].Kc = Ls[l].Kt; Ls[l].Kt = tmp;
            tmp = Ls[l].Vc; Ls[l].Vc = Ls[l].Vt; Ls[l].Vt = tmp;
        }
    }

    uint32_t th = 0;
    for (int i = 0; i < gen; i++) th = th * 31u + (uint32_t)hist[0][i];
    printf("[%s] best beam (score %.6f):", KERNEL_NAME, (double)score[0]);
    for (int i = 0; i < gen && i < 8; i++) printf(" %d", hist[0][i]);
    printf("%s\n", gen > 8 ? " ..." : "");
    wh_report(large ? "large" : "small", logp, (size_t)BEAMS * VOCAB, th);

    free(x); free(logp); free(wte); free(wpe); free(lnf_g); free(lnf_b);
    free(s.H); free(s.Q); free(s.K); free(s.V); free(s.C); free(s.A); free(s.P); free(s.M);
    for (int l = 0; l < layers; l++) free_layer(&Ls[l]);
    free(Ls);
    return 0;
}
