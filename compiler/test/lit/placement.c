/**
 * @file
 * @brief Lit test of HintPlacementPass: per-phase hints, hoisting, callee
 *        summaries, libm summaries, the minimum region size and the
 *        switch-cost variants.
 */
// HintPlacement (PROPOSAL §3.2): DP over the region tree with call-graph and
// libm summaries, parametric switch cost and hysteresis.
//   phases  : streaming + compute phase inside a hot loop -> one setwin per
//             phase loop (depth 2), none inside the inner compute loops
//   hoist   : uniform demand inside a hot loop -> one setwin hoisted to the
//             outer loop (depth 1)
//   caller  : kstream() leaves the window at 256 (callee summary), so the
//             following loop with the same demand gets no redundant hint
//   act     : tanhf is summarized (34 insts, 44-cycle latency in the DAG), so
//             the loop is critical-path bound -> large window, hoisted
//   tiny    : below -winhint-min-region-insts -> never hinted
// Switch-cost variants: 1e12 cycles or a huge hysteresis -> no hints; P/E
// migration (50 us) -> the phase switch inside the hot loop is not worth it.
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %rvcc -O2 -fno-unroll-loops -fno-vectorize -gline-tables-only %whflags -mllvm -winhint-verbose -mllvm -winhint-out-dir=%t -c %s -o %t/a.o 2>&1 | FileCheck %s
// RUN: llvm-objdump -d %t/a.o | grep -c 'ori[[:space:]]*zero, zero' | FileCheck %s --check-prefix=COUNT
// RUN: %python -c "import json; d=json.load(open('%t/placement.winhint.json')); h={(x['function'], x['depth']) for x in d['hints']}; assert h == {('act', 1), ('hoist', 1), ('kstream', 1), ('phases', 2)}, h; assert d['hints_setwin'] == 5"
// RUN: %rvcc -O2 -fno-unroll-loops -fno-vectorize -gline-tables-only %whflags -mllvm -winhint-verbose -mllvm -winhint-switch-cost=1e12 -c %s -o %t/b.o 2>&1 | FileCheck %s --check-prefix=NONE
// RUN: %rvcc -O2 -fno-unroll-loops -fno-vectorize -gline-tables-only %whflags -mllvm -winhint-verbose -mllvm -winhint-hysteresis=1e9 -c %s -o %t/c.o 2>&1 | FileCheck %s --check-prefix=NONE
// RUN: %rvcc -O2 -fno-unroll-loops -fno-vectorize -gline-tables-only %whflags -mllvm -winhint-verbose -mllvm -winhint-switch-model=pe -mllvm -winhint-migration-us=50 -mllvm -winhint-out-dir=%t/pe -c %s -o %t/d.o 2>&1 | FileCheck %s --check-prefix=PE
// RUN: %python -c "import json; d=json.load(open('%t/pe/placement.winhint.json')); assert d['switch_cost_cycles'] == 100000.0, d['switch_cost_cycles']"
// The libm summary in the model.
// RUN: %rvcc -O2 -fno-unroll-loops -fno-vectorize -gline-tables-only -S -emit-llvm %s -o %t/p.ll
// RUN: opt -load-pass-plugin=%wh -passes='print<winhint-demand>' -disable-output -winhint-target=%machine %t/p.ll 2>&1 | FileCheck %s --check-prefix=LIBM

#include <math.h>
/** @brief Length of the streamed arrays. */
#define N (1 << 20)
/** @brief Length of the L1-resident arrays. */
#define M 512
/** @brief Streamed (A, B) and L1-resident (S, T) arrays. */
float A[N], B[N], S[M], T[M];

/** @brief T[j] = tanhf(S[j]) over an L1-resident array, reps times. */
// CHECK:     winhint: act line [[@LINE+5]] loop {{.*}}: setwin(256)
// CHECK-NOT: winhint: caller
// LIBM:      WINHINT fn=act {{.*}} line=[[@LINE+4]] depth=2 L_mem=4.0 D_indep=inf MLP=0.0 CP=50.0 {{.*}} W*=200 config=3
// PE:        winhint: act line [[@LINE+2]] loop {{.*}}: setwin(256)
void act(int reps) {
  for (int r = 0; r < reps; r++)
    for (int j = 0; j < M; j++)
      T[j] = tanhf(S[j]);
}

/** @brief Uniform streaming demand inside a hot outer loop (hint hoisted). */
// CHECK:     winhint: hoist line [[@LINE+3]] loop {{.*}}: setwin(256)
// CHECK-NOT: winhint: hoist
void hoist(void) {
  for (int r = 0; r < 100; r++)
    for (int i = 0; i < N; i++)
      A[i] += B[i];
}

/** @brief Streaming loop; leaves the window at its setting on return. */
// CHECK:     winhint: kstream line [[@LINE+2]] loop {{.*}}: setwin(256)
__attribute__((noinline)) void kstream(void) {
  for (int i = 0; i < N; i++)
    A[i] = A[i] * 0.25f + B[i];
}
/** @brief Call kstream(), then a loop with the same demand (no redundant hint). */
void caller(void) {
  kstream();
  for (int i = 0; i < N; i++)
    B[i] = B[i] * 0.75f + A[i];
}

/** @brief Streaming phase and L1-resident compute phase inside a hot loop. */
// CHECK:     winhint: phases line [[@LINE+6]] loop {{.*}}: setwin(256)
// CHECK:     winhint: phases line [[@LINE+7]] loop {{.*}}: setwin(64)
// PE-NOT:    winhint: phases line [[@LINE+4]]
// PE:        winhint: phases line [[@LINE+2]] loop {{.*}}: setwin(256)
void phases(int iters) {
  for (int t = 0; t < iters; t++) {
    for (int i = 0; i < N; i++)
      A[i] = A[i] * 0.5f + B[i];
    for (int r = 0; r < 64; r++)
      for (int j = 0; j < M; j++)
        S[j] = sqrtf(S[j] * 0.9f + 1.0f) / (S[j] + 2.0f);
  }
}

// CHECK-NOT: winhint: tiny
// CHECK:     winhint: placement: 5 setwin, 0 region hints
// PE-NOT:    winhint: tiny
// PE:        winhint: placement: 4 setwin
// NONE-NOT:  setwin(
// NONE:      winhint: placement: 0 setwin
// COUNT:     5
/** @brief Loop below -winhint-min-region-insts (never hinted). */
void tiny(void) {
  for (int i = 0; i < 8; i++)
    S[i] = 0;
}
