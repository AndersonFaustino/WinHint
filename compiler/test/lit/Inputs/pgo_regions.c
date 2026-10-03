/**
 * @file
 * @brief Input of pgo-select.test: three functions, one region each
 *        (f=0, g=1, h=2).
 */
/** @brief Region 0: add 1 to n elements of p. */
void f(float *p, int n) { for (int i = 0; i < n; i++) p[i] += 1.0f; }
/** @brief Region 1: add 2 to n elements of p. */
void g(float *p, int n) { for (int i = 0; i < n; i++) p[i] += 2.0f; }
/** @brief Region 2: add 3 to n elements of p. */
void h(float *p, int n) { for (int i = 0; i < n; i++) p[i] += 3.0f; }
