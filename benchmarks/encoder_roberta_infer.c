/**
 * @file encoder_roberta_infer.c
 * @brief WinHint Project - Encoder group, benchmark 2: RoBERTa encoder layer(s).
 *
 * RoBERTa-base layer shapes (Liu et al., 2019):
 *
 *   hidden H = 768, heads = 12, head_dim = 64, FFN F = 3072, post-LN, GELU
 *
 * @verbatim
 *   input  | seq S (padded) | valid tokens | layers (default) | weights/layer
 *   -------+----------------+--------------+------------------+--------------
 *   large  |      256       |  224 (7/8)   | 1                | 28.3 MB
 *   small  |        4       |    3 (3/4)   | 1                | 28.3 MB
 * @endverbatim
 *
 *   argv[1] = small|large (default large)   argv[2] = number of layers (default 1)
 *
 * Differences from encoder_bert_tiny: longer sequence (attention scores are
 * 12*S*S floats = 3 MB for large, i.e. attention itself exceeds L2) and a
 * padding attention mask: keys beyond the valid length get probability 0.
 * The last max(1, S/8) tokens are padding (none if S == 1), so the mask is
 * exercised by both inputs.
 * The model is identical for both inputs; only the input sequence changes.
 * Approximate work per layer: 12*S*H^2 + 2*S^2*H MACs (large: ~1.9 GMAC).
 *
 * Phases are separate, non-inlined functions with separate loop nests:
 *   linear, attention_scores, softmax_rows (masked), attention_context,
 *   residual_add, layer_norm, gelu_inplace.
 *
 * Build options: -DTILED (register-blocked, N-outer tiled GEMM with the same
 * per-output summation order, hence bit-identical output), and shape
 * overrides -DHIDDEN, -DFFN_DIM, -DNUM_HEADS, -DSEQ_LARGE, -DSEQ_SMALL.
 *
 * Output: float checksum + FNV-1a hash of the final hidden states.
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
#if defined(__GNUC__) || defined(__clang__)
#define WH_NOINLINE __attribute__((noinline))
#else
#define WH_NOINLINE
#endif

#define KERNEL_NAME "encoder_roberta" ///< Name printed as the "[name]" prefix of every output line.

/* ---- Model shape (BERT-base) ------------------------------------------- */
#ifndef HIDDEN
#define HIDDEN 768 ///< Hidden size H (override with -DHIDDEN).
#endif
#ifndef NUM_HEADS
#define NUM_HEADS 12 ///< Attention heads (override with -DNUM_HEADS).
#endif
#define HEAD_DIM (HIDDEN / NUM_HEADS) ///< Per-head dimension.
#ifndef FFN_DIM
#define FFN_DIM 3072 ///< FFN inner size F (override with -DFFN_DIM).
#endif
#ifndef SEQ_LARGE
#define SEQ_LARGE 256 ///< Padded sequence length S for the large input.
#endif
#ifndef SEQ_SMALL
#define SEQ_SMALL 4 ///< Padded sequence length S for the small input.
#endif
#define VOCAB 1024 ///< reduced vocabulary: only S rows are gathered

/* ---- Deterministic PRNG: values in [-0.1, 0.1) -------------------------- */
/** @brief xorshift32 state of the deterministic PRNG (fixed per-kernel seed). */
static uint32_t wh_rng = 0x1badb002u;
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

/* ---- Attention: P[h][i][j] = scale * <Q[i,h,:], K[j,h,:]> --------------- */
/**
 * @brief Scaled dot-product scores for all heads: P[h][i][j] = <Q[i,h,:], K[j,h,:]> / sqrt(HEAD_DIM).
 * @param[out] P Scores, [NUM_HEADS][S][S].
 * @param[in]  Q Queries, [S][HIDDEN] (head h at columns h*HEAD_DIM..).
 * @param[in]  K Keys, [S][HIDDEN].
 * @param[in]  S Sequence length.
 */
WH_NOINLINE static void attention_scores(float *P, const float *Q, const float *K, int S) {
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (int h = 0; h < NUM_HEADS; h++)
        for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++) {
                const float *q = Q + i * HIDDEN + h * HEAD_DIM;
                const float *k = K + j * HIDDEN + h * HEAD_DIM;
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) dot += q[d] * k[d];
                P[((size_t)h * S + i) * S + j] = dot * scale;
            }
}

/**
 * @brief Row-wise masked softmax, in place: rows of length `cols`, only the first
 *        `valid` keys participate; masked keys get probability 0.
 * @param[in,out] P     Scores in, probabilities out, [rows][cols].
 * @param[in]     rows  Number of rows (NUM_HEADS * S).
 * @param[in]     cols  Row length (S).
 * @param[in]     valid Number of unmasked keys (1 <= valid <= cols).
 */
WH_NOINLINE static void softmax_rows(float *P, int rows, int cols, int valid) {
    for (int r = 0; r < rows; r++) {
        float *x = P + (size_t)r * cols;
        float mx = x[0];
        for (int j = 1; j < valid; j++) if (x[j] > mx) mx = x[j];
        float sum = 0.0f;
        for (int j = 0; j < valid; j++) { x[j] = expf(x[j] - mx); sum += x[j]; }
        float inv = 1.0f / sum;
        for (int j = 0; j < valid; j++) x[j] *= inv;
        for (int j = valid; j < cols; j++) x[j] = 0.0f;
    }
}

/**
 * @brief Attention context: C[i, h*64+d] = sum_j P[h][i][j] * V[j, h*64+d].
 * @param[out] C Context, [S][HIDDEN].
 * @param[in]  P Attention probabilities, [NUM_HEADS][S][S].
 * @param[in]  V Values, [S][HIDDEN].
 * @param[in]  S Sequence length.
 */
WH_NOINLINE static void attention_context(float *C, const float *P, const float *V, int S) {
    for (int h = 0; h < NUM_HEADS; h++)
        for (int i = 0; i < S; i++) {
            float *c = C + i * HIDDEN + h * HEAD_DIM;
            const float *p = P + ((size_t)h * S + i) * S;
            for (int d = 0; d < HEAD_DIM; d++) c[d] = 0.0f;
            for (int j = 0; j < S; j++) {
                const float *v = V + j * HIDDEN + h * HEAD_DIM;
                float pj = p[j];
                for (int d = 0; d < HEAD_DIM; d++) c[d] += pj * v[d];
            }
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
 * @brief Embedding lookup: x[s] = tok_emb[tokens[s]] + pos_emb[s].
 * @param[out] x       Hidden states, [S][HIDDEN].
 * @param[in]  tok_emb Token embeddings, [VOCAB][HIDDEN].
 * @param[in]  pos_emb Position embeddings, [S][HIDDEN].
 * @param[in]  tokens  Token ids [S].
 * @param[in]  S       Sequence length.
 */
WH_NOINLINE static void embed(float *x, const float *tok_emb, const float *pos_emb,
                              const int *tokens, int S) {
    for (int s = 0; s < S; s++)
        for (int d = 0; d < HIDDEN; d++)
            x[s * HIDDEN + d] = tok_emb[tokens[s] * HIDDEN + d] + pos_emb[s * HIDDEN + d];
}

/* ---- Layer ------------------------------------------------------------- */
/** @brief Weights of one encoder layer (W* in [out][in] nn.Linear layout). */
typedef struct {
    float *Wq, *Wk, *Wv, *Wo, *bq, *bk, *bv, *bo; ///< Attention Q/K/V/output weights [HIDDEN][HIDDEN] and biases [HIDDEN].
    float *W1, *b1, *W2, *b2;                     ///< FFN weights W1 [FFN_DIM][HIDDEN], W2 [HIDDEN][FFN_DIM] and biases.
    float *ln1_g, *ln1_b, *ln2_g, *ln2_b;         ///< LayerNorm gains (1) and biases (0) [HIDDEN].
} Layer;

/**
 * @brief Per-layer activation buffers, reused across layers.
 *
 * Q, K, V, C, A, X1, F: [S][HIDDEN]; P: attention scores [NUM_HEADS][S][S];
 * M: FFN hidden activations [S][FFN_DIM].
 */
typedef struct {
    float *Q, *K, *V, *P, *C, *A, *X1, *M, *F;
} Scratch;

/**
 * @brief Allocate one layer's weights: pseudo-random matrices and biases,
 *        LayerNorm gains 1 and biases 0.
 * @param[out] L Layer to initialize.
 */
static void init_layer(Layer *L) {
    L->Wq = wh_alloc_rand((size_t)HIDDEN * HIDDEN); L->bq = wh_alloc_rand(HIDDEN);
    L->Wk = wh_alloc_rand((size_t)HIDDEN * HIDDEN); L->bk = wh_alloc_rand(HIDDEN);
    L->Wv = wh_alloc_rand((size_t)HIDDEN * HIDDEN); L->bv = wh_alloc_rand(HIDDEN);
    L->Wo = wh_alloc_rand((size_t)HIDDEN * HIDDEN); L->bo = wh_alloc_rand(HIDDEN);
    L->W1 = wh_alloc_rand((size_t)FFN_DIM * HIDDEN); L->b1 = wh_alloc_rand(FFN_DIM);
    L->W2 = wh_alloc_rand((size_t)HIDDEN * FFN_DIM); L->b2 = wh_alloc_rand(HIDDEN);
    L->ln1_g = wh_alloc_const(HIDDEN, 1.0f); L->ln1_b = wh_alloc_const(HIDDEN, 0.0f);
    L->ln2_g = wh_alloc_const(HIDDEN, 1.0f); L->ln2_b = wh_alloc_const(HIDDEN, 0.0f);
}

/**
 * @brief Free every buffer of a layer allocated by init_layer().
 * @param[in] L Layer to release.
 */
static void free_layer(Layer *L) {
    float *p[] = {L->Wq, L->Wk, L->Wv, L->Wo, L->bq, L->bk, L->bv, L->bo, L->W1, L->b1,
                  L->W2, L->b2, L->ln1_g, L->ln1_b, L->ln2_g, L->ln2_b};
    for (size_t i = 0; i < sizeof p / sizeof p[0]; i++) free(p[i]);
}

/**
 * @brief Number of valid (non-padding) tokens of an S-token sequence.
 *
 * The last S/8 tokens are padding, but at least one (unless S == 1), so that
 * short sequences still exercise the padding mask: 256 -> 224, 4 -> 3.
 * @param[in] S Sequence length (>= 1).
 * @return The valid length, 1 <= valid <= S.
 */
static int valid_tokens(int S) {
    int pad = S / 8;
    if (pad < 1 && S > 1) pad = 1;
    return S - pad;
}

/**
 * @brief Post-LN RoBERTa layer; x is updated in place.
 *
 * Same structure as the BERT layer, with the padding mask: the softmax keeps
 * only the first valid_tokens(S) keys.
 * @param[in,out] x Hidden states, [S][HIDDEN]; updated in place.
 * @param[in]     L Layer weights.
 * @param         s Scratch buffers.
 * @param[in]     S Sequence length.
 */
static void encoder_layer(float *x, const Layer *L, Scratch *s, int S) {
    linear(s->Q, x, L->Wq, L->bq, S, HIDDEN, HIDDEN);
    linear(s->K, x, L->Wk, L->bk, S, HIDDEN, HIDDEN);
    linear(s->V, x, L->Wv, L->bv, S, HIDDEN, HIDDEN);
    attention_scores(s->P, s->Q, s->K, S);
    softmax_rows(s->P, NUM_HEADS * S, S, valid_tokens(S));
    attention_context(s->C, s->P, s->V, S);
    linear(s->A, s->C, L->Wo, L->bo, S, HIDDEN, HIDDEN);
    residual_add(s->A, x, S * HIDDEN);
    layer_norm(s->X1, s->A, L->ln1_g, L->ln1_b, S, HIDDEN);
    linear(s->M, s->X1, L->W1, L->b1, S, FFN_DIM, HIDDEN);
    gelu_inplace(s->M, S * FFN_DIM);
    linear(s->F, s->M, L->W2, L->b2, S, HIDDEN, FFN_DIM);
    residual_add(s->F, s->X1, S * HIDDEN);
    layer_norm(x, s->F, L->ln2_g, L->ln2_b, S, HIDDEN);
}

/**
 * @brief Embed S tokens (the last S - valid_tokens(S) are padding, id 1), run `layers`
 *        encoder layers, and report the final hidden states.
 *
 * Prints a configuration line, then wh_report() over the [S][HIDDEN] output
 * (padding rows included).
 * @param[in] argc Argument count.
 * @param[in] argv argv[1] = small|large, argv[2] = number of layers.
 * @return 0 on success; 1 if a malloc() in main fails (wh_alloc exits 1; bad
 *         arguments exit 2).
 */
int main(int argc, char **argv) {
    int layers;
    int large = wh_parse_args(argc, argv, &layers);
    const int S = large ? SEQ_LARGE : SEQ_SMALL;
    printf("[%s] input=%s seq=%d hidden=%d heads=%d ffn=%d layers=%d%s\n", KERNEL_NAME,
           large ? "large" : "small", S, HIDDEN, NUM_HEADS, FFN_DIM, layers,
#ifdef TILED
           " gemm=tiled"
#else
           " gemm=naive"
#endif
    );

    float *tok_emb = wh_alloc_rand((size_t)VOCAB * HIDDEN);
    float *pos_emb = wh_alloc_rand((size_t)S * HIDDEN);
    Layer *Ls = (Layer *)malloc((size_t)layers * sizeof(Layer));
    if (!Ls) return 1;
    for (int l = 0; l < layers; l++) init_layer(&Ls[l]);

    Scratch s;
    s.Q = wh_alloc((size_t)S * HIDDEN); s.K = wh_alloc((size_t)S * HIDDEN);
    s.V = wh_alloc((size_t)S * HIDDEN); s.C = wh_alloc((size_t)S * HIDDEN);
    s.A = wh_alloc((size_t)S * HIDDEN); s.X1 = wh_alloc((size_t)S * HIDDEN);
    s.F = wh_alloc((size_t)S * HIDDEN);
    s.P = wh_alloc((size_t)NUM_HEADS * S * S);
    s.M = wh_alloc((size_t)S * FFN_DIM);

    int *tokens = (int *)malloc((size_t)S * sizeof(int));
    if (!tokens) return 1;
    const int valid = valid_tokens(S);
    for (int i = 0; i < S; i++) /* padding tokens (id 1) at the end */
        tokens[i] = i < valid ? (int)((i * 104729u + 7u) % VOCAB) : 1;

    float *x = wh_alloc((size_t)S * HIDDEN);
    embed(x, tok_emb, pos_emb, tokens, S);
    for (int l = 0; l < layers; l++) encoder_layer(x, &Ls[l], &s, S);

    wh_report(large ? "large" : "small", x, (size_t)S * HIDDEN, 0u);

    free(x); free(tokens); free(tok_emb); free(pos_emb);
    free(s.Q); free(s.K); free(s.V); free(s.C); free(s.A); free(s.X1); free(s.F);
    free(s.P); free(s.M);
    for (int l = 0; l < layers; l++) free_layer(&Ls[l]);
    free(Ls);
    return 0;
}
