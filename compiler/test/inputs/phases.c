/**
 * @file
 * @brief Placement test input: a streaming and a compute phase inside one
 *        loop nest.
 *
 * Two phases in sequence inside one loop nest: a streaming phase and a
 * compute phase -> HintPlacement should place one setwin per phase (hoisted
 * to the phase loops' preheaders), not inside the inner loops.
 * Prints `phases <sum>`.
 */
#include <math.h>
#include <stdio.h>
/** @brief Length of the streamed arrays (4 MiB each). */
#define N (1 << 20)
/** @brief Length of the L1-resident array. */
#define M 512
/** @brief Streamed arrays (A, B) and L1-resident compute array (S). */
static float A[N], B[N], S[M];

/**
 * @brief Run iters times: a streaming pass over A/B, then 64 compute passes
 *        (sqrtf and a divide) over S.
 * @param iters Outer iterations.
 */
__attribute__((noinline)) void step(int iters) {
  for (int t = 0; t < iters; t++) {
    for (int i = 0; i < N; i++)        /* streaming */
      A[i] = A[i] * 0.5f + B[i];
    for (int r = 0; r < 64; r++)       /* compute, L1-resident */
      for (int j = 0; j < M; j++)
        S[j] = sqrtf(S[j] * 0.9f + 1.0f) / (S[j] + 2.0f);
  }
}

/** @brief Initialize the arrays, run step(4) and print a checksum. */
int main(void) {
  for (int i = 0; i < N; i++) { A[i] = (float)(i & 255); B[i] = 1.0f; }
  for (int j = 0; j < M; j++) S[j] = (float)j;
  step(4);
  double s = 0;
  for (int i = 0; i < N; i += 1024) s += A[i];
  for (int j = 0; j < M; j++) s += S[j];
  printf("phases %.4f\n", s);
  return 0;
}
