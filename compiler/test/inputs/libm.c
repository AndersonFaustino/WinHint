/**
 * @file
 * @brief Model test input: L1-resident loop dominated by expf().
 *
 * L1-resident loop dominated by a libm call (softmax numerator). The call is
 * summarized (call-graph/libm summary): it must count as ~20+ instructions and
 * its latency must sit on the critical path, not as an opaque 5-inst call.
 * Prints `libm <sum>`.
 */
#include <math.h>
#include <stdio.h>
/** @brief Array length. */
#define M 1024
/** @brief Input (X) and output (Y) of the softmax numerator. */
static float X[M], Y[M];

/**
 * @brief Compute Y[i] = expf(X[i] - mx), reps times.
 * @param reps Outer repetitions.
 * @param mx   Value subtracted before the exponential.
 */
__attribute__((noinline)) void softmax_num(int reps, float mx) {
  for (int r = 0; r < reps; r++)
    for (int i = 0; i < M; i++)
      Y[i] = expf(X[i] - mx);
}

/** @brief Initialize X, run softmax_num(200, 1.5) and print the sum of Y. */
int main(void) {
  for (int i = 0; i < M; i++) X[i] = (float)(i % 13) * 0.125f;
  softmax_num(200, 1.5f);
  double s = 0;
  for (int i = 0; i < M; i++) s += Y[i];
  printf("libm %.5f\n", s);
  return 0;
}
