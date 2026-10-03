/**
 * @file
 * @brief Input of placement-options.test: loops that call functions defined
 *        in the module (with and without hints, and recursive), for the
 *        call-graph summaries of HintPlacement and the option paths.
 *
 * No headers, so it compiles for any triple (riscv64, x86-64, aarch64).
 * Regions (functions sorted by name): a_driver 0, b_small_calls 1,
 * c_rec 2, ksmall 3, kstream 4.
 */
/** @brief Length of the streamed arrays (4 MiB each). */
#define N (1 << 20)
/** @brief Length of the L1-resident array. */
#define M 512
/** @brief Streamed (A, B) and L1-resident (S) arrays. */
float A[N], B[N], S[M];

/** @brief Streaming loop: gets the largest window (a hinted callee). */
__attribute__((noinline)) void kstream(void) {
  for (int i = 0; i < N; i++)
    A[i] += B[i];
}

/** @brief Short L1-resident loop: too small to be hinted on its own. */
__attribute__((noinline)) float ksmall(int k) {
  float s = 0;
  for (int j = 0; j < 64; j++)
    s += S[j] * (float)k;
  return s;
}

/** @brief A compute phase followed by a call to the hinted kstream() in
 *         the same loop body. */
void a_driver(int reps) {
  for (int r = 0; r < reps; r++) {
    for (int j = 0; j < M; j++)
      S[j] = S[j] * 1.5f + 1.0f;
    kstream();
  }
}

/** @brief A loop whose body is a call to an unhinted callee. */
void b_small_calls(int reps) {
  for (int r = 0; r < reps; r++)
    S[r & (M - 1)] += ksmall(r);
}

/** @brief Self-recursive call inside a loop (no callee summary yet). */
__attribute__((noinline)) int c_rec(int n) {
  int s = 0;
  for (int i = 0; i < n; i++)
    s += i > 2 ? c_rec(i - 3) : (int)S[i];
  return s;
}
