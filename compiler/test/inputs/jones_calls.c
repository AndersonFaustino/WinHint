/**
 * @file
 * @brief B6 (JonesIQ) test input for redundant-hint elimination across calls.
 *
 * Loops A and B have the same IQ demand; g() in between sets a different
 * one, so B's hint is NOT redundant, while C (same demand as B, straight
 * after it) is. Prints `jones_calls <sum>`.
 */
#include <stdio.h>
/** @brief Array length. */
#define N 4096
/** @brief Operands of loops A, B and C in f(). */
float a[N], b[N];
/** @brief Divisors of the serial divide chain in g(). */
double d[N];

/**
 * @brief Serial double-precision divide chain over d (small IQ demand);
 *        stores the result in d[0].
 * @param n Number of elements.
 */
__attribute__((noinline)) void g(int n) {
  double x = 1.0; // serial divide chain: small IQ demand
  for (int i = 0; i < n; i++)
    x = x / (d[i] + 1.0);
  d[0] = x;
}

/**
 * @brief Loops A, call g(), loops B and C (B and C have the same demand).
 * @param n Number of elements per loop.
 */
__attribute__((noinline)) void f(int n) {
  for (int i = 0; i < n; i++) // A
    a[i] = a[i] * 2.0f + b[i];
  g(n);
  for (int i = 0; i < n; i++) // B
    b[i] = b[i] * 3.0f + a[i];
  for (int i = 0; i < n; i++) // C: same demand as B, no call between: redundant
    a[i] = a[i] * 5.0f + b[i];
}

/**
 * @brief Initialize the arrays, run f() on N elements (N/2 if any argument
 *        is given) and print a checksum.
 */
int main(int argc, char **argv) {
  for (int i = 0; i < N; i++) {
    a[i] = (float)(i % 7);
    b[i] = (float)(i % 5);
    d[i] = (double)(i % 3);
  }
  f(argc > 1 ? N / 2 : N);
  double s = d[0];
  for (int i = 0; i < N; i++)
    s += a[i] + b[i];
  printf("jones_calls %.6f\n", s);
  return 0;
}
