/**
 * @file
 * @brief Model test input: STREAM-triad loop over DRAM-resident arrays.
 *
 * Streaming load loop over arrays far larger than L2: many independent
 * misses per window -> MLP-driven, expect the largest window.
 * Prints `stream <sum>`.
 */
#include <stdio.h>
/** @brief Array length. */
#define N (1 << 21) /* 8 MiB per array */
/** @brief Triad arrays: A = B + s * C. */
static float A[N], B[N], C[N];

/**
 * @brief Compute A[i] = B[i] + s * C[i] over all elements.
 * @param s Scalar factor.
 */
__attribute__((noinline)) void triad(float s) {
  for (int i = 0; i < N; i++)
    A[i] = B[i] + s * C[i];
}

/** @brief Initialize B and C, run triad() three times and print a checksum. */
int main(void) {
  for (int i = 0; i < N; i++) {
    B[i] = (float)(i & 1023);
    C[i] = (float)(i % 7);
  }
  for (int r = 0; r < 3; r++)
    triad(0.5f + r);
  double sum = 0;
  for (int i = 0; i < N; i += 4096)
    sum += A[i];
  printf("stream %.3f\n", sum);
  return 0;
}
