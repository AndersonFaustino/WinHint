/**
 * @file
 * @brief Model test input: compute-bound, L1-resident Horner polynomial.
 *
 * Compute-bound: small, L1-resident data; each element runs a chain of
 * dependent fused multiply-adds (Horner polynomial). No long-latency loads,
 * so W* is set by the critical path: W_cp = ceil(CP * min(issue_width, body / II)).
 * Prints `compute <sum>`.
 */
#include <stdio.h>
/** @brief Array length (1 KiB of floats). */
#define M 256
/** @brief L1-resident data, updated in place. */
static float X[M];

/**
 * @brief Apply ((((v*a + b)*v + c)*v + d) * 0.25) to every element of X,
 *        reps times (the loop nest analyzed by the tests).
 * @param reps    Outer repetitions.
 * @param a,b,c,d Polynomial coefficients.
 */
__attribute__((noinline)) void poly(int reps, float a, float b, float c, float d) {
  for (int r = 0; r < reps; r++)
    for (int i = 0; i < M; i++) {
      float v = X[i];
      X[i] = (((v * a + b) * v + c) * v + d) * 0.25f;
    }
}

/** @brief Initialize X, run poly(2000, ...) and print the sum. */
int main(void) {
  for (int i = 0; i < M; i++) X[i] = (float)(i % 17) * 0.0625f;
  poly(2000, 0.5f, 0.25f, -0.125f, 0.75f);
  double s = 0;
  for (int i = 0; i < M; i++) s += X[i];
  printf("compute %.6f\n", s);
  return 0;
}
