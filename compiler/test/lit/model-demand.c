/**
 * @file
 * @brief Lit test of WindowDemandAnalysis (print<winhint-demand>) on
 *        canonical loops: streaming, gather, pointer chase, short and long
 *        critical path, memory recurrence and an outer-loop miss.
 */
// WindowDemandAnalysis (PROPOSAL §3.1) on canonical loops, default machine
// (sim/machines/riscv_ooo.json: 4-wide, L1D 32 KiB / 4 cy, L2 1 MiB / 20 cy,
// memory 200 cy, 16 L1D MSHRs, ROB configs 64/128/192/256):
//   streaming (independent DRAM misses)  -> MLP > 1, W* = W_max (config 3)
//   gather    (indirect, 1 miss / iter)  -> MLP bounded by the MSHRs (16)
//   pointer chase (dependent misses)     -> MLP = 1, smallest config
//   L1-resident, short CP                -> no long-latency loads, W* = W_cp, config 0
//   L1-resident, long CP (Horner)        -> W* = W_cp = CP * issue_width, config 1
//   memory recurrence a[i+8] = f(a[i])   -> DependenceAnalysis marks the load
//                                           loop-carried: it adds no MLP (D_indep x2)
//   outer-loop miss around an inner loop -> D_indep counts the inner loop's work
//   -winhint-cp-model=width               -> W_cp = CP * issue_width (ablation)
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %rvcc -O2 -fno-unroll-loops -fno-vectorize -gline-tables-only -S -emit-llvm %s -o %t/m.ll
// RUN: opt -load-pass-plugin=%wh -passes='print<winhint-demand>' -disable-output -winhint-target=%machine %t/m.ll 2>&1 | FileCheck %s
// RUN: opt -load-pass-plugin=%wh -passes='print<winhint-demand>' -disable-output -winhint-target=%machine -winhint-cp-model=width %t/m.ll 2>&1 | FileCheck %s --check-prefix=WIDTH

/** @brief Length of the DRAM-resident arrays. */
#define N (1 << 22) /* 16 MiB per float array: DRAM-resident */
/** @brief Length of the L1-resident arrays. */
#define S 1024      /* 4 KiB: L1-resident */
/** @brief DRAM-resident float arrays. */
float A[N], B[N], C[N];
/** @brief Gather indices for m_gather(). */
int Idx[N];
/** @brief L1-resident arrays. */
float X[S], Y[S];
/** @brief List node for m_chase(); the value sits after 64 bytes of link and padding. */
struct node { struct node *next; long pad[7]; float v; };

// CHECK-LABEL: window demand for function 'm_stream'
// CHECK:       load  stream x1 stride=4.0 fp=16.0MiB reuse=none level=MEM
// CHECK:       WINHINT fn=m_stream {{.*}} L_mem=220.0 {{.*}} MLP={{[2-9]|1[0-6]}}.{{[0-9]}} {{.*}} W*=256 config=3 (ROB 256)
/** @brief Streaming triad over DRAM-resident arrays. */
void m_stream(void) {
  for (int i = 0; i < N; i++)
    A[i] = B[i] + 3.0f * C[i];
}

// CHECK-LABEL: window demand for function 'm_gather'
// CHECK:       load  indirect x1
// CHECK:       WINHINT fn=m_gather {{.*}} L_mem=220.0 {{.*}} MLP=16.0 {{.*}} config={{[123]}}
/** @brief Indirect gather A[Idx[i]]; @return the sum. */
float m_gather(void) {
  float s = 0;
  for (int i = 0; i < N; i++)
    s += A[Idx[i]];
  return s;
}

// CHECK-LABEL: window demand for function 'm_chase'
// CHECK:       load  chase x1
// CHECK:       WINHINT fn=m_chase {{.*}} L_mem=220.0 D_indep=inf MLP=1.0 CP=6.0 {{.*}} config=0 (ROB 64)
// WIDTH-LABEL: window demand for function 'm_chase'
// WIDTH:       WINHINT fn=m_chase {{.*}} CP=6.0 {{.*}} W_cp=24 W*=24 config=0
/** @brief Pointer chase from p until null; @return the sum of the values. */
float m_chase(struct node *p) {
  float s = 0;
  while (p) {
    s += p->v;
    p = p->next;
  }
  return s;
}

// CHECK-LABEL: window demand for function 'm_short_cp'
// CHECK:       load  stream x1 stride=4.0 fp=4.0KiB reuse=8.0KiB level=L1D misses/iter=0.000
// CHECK:       WINHINT fn=m_short_cp {{.*}} depth=2 L_mem=4.0 D_indep=inf MLP=0.0 CP=8.0 {{.*}} W_mlp=0 W_cp=32 W*=32 config=0 (ROB 64)
/** @brief L1-resident copy-add with a short critical path. */
void m_short_cp(void) {
  for (int r = 0; r < 1000; r++)
    for (int i = 0; i < S; i++)
      Y[i] = X[i] + 1.0f;
}

// CHECK-LABEL: window demand for function 'm_long_cp'
// CHECK:       WINHINT fn=m_long_cp {{.*}} depth=2 L_mem=4.0 D_indep=inf MLP=0.0 CP=31.0 {{.*}} W_mlp=0 W_cp=124 W*=124 config=1 (ROB 128)
// WIDTH-LABEL: window demand for function 'm_long_cp'
// WIDTH:       WINHINT fn=m_long_cp {{.*}} depth=2 {{.*}} W_cp=124 W*=124 config=1
/** @brief L1-resident degree-5 Horner polynomial (long critical path). */
void m_long_cp(void) {
  for (int r = 0; r < 1000; r++)
    for (int i = 0; i < S; i++) {
      float x = X[i];
      Y[i] = ((((x * x + 1.0f) * x + 2.0f) * x + 3.0f) * x + 4.0f) * x + 5.0f;
    }
}

// CHECK-LABEL: window demand for function 'm_memrec'
// CHECK:       load  stream x1 {{.*}} loop-carried
// CHECK:       WINHINT fn=m_memrec {{.*}} D_indep=176.0 MLP=5.0
// CHECK-LABEL: window demand for function 'm_nomemrec'
// CHECK-NOT:   loop-carried
// CHECK:       WINHINT fn=m_nomemrec {{.*}} D_indep=88.0 MLP=10.0
/** @brief Memory recurrence a[i + 8] = a[i] * 0.5 + b[i] (loop-carried load). */
void m_memrec(float *restrict a, const float *restrict b, int n) {
  for (int i = 0; i < n - 8; i++)
    a[i + 8] = a[i] * 0.5f + b[i];
}
/** @brief Same body as m_memrec() without the recurrence (a[i] in place). */
void m_nomemrec(float *restrict a, const float *restrict b, int n) {
  for (int i = 0; i < n; i++)
    a[i] = a[i] * 0.5f + b[i];
}

// One DRAM miss per outer iteration, 64 dependent FMAs in the inner loop:
// D_indep = outer body + inner loop (7 + 64 * 4).
// CHECK-LABEL: window demand for function 'm_rare'
// CHECK:       load  strided x1 stride=16384.0 {{.*}} level=MEM misses/iter=1.000
// CHECK:       WINHINT fn=m_rare {{.*}} depth=1 L_mem=220.0 D_indep=263.0 {{.*}} W*=256 config=3
// CHECK:       WINHINT fn=m_rare {{.*}} depth=2 L_mem=4.0 D_indep=inf {{.*}} config=0
/** @brief One DRAM miss per outer iteration around a 64-step dependent inner loop;
 *  @return the accumulated sum. */
float m_rare(void) {
  float s = 0;
  for (int i = 0; i < N; i += 4096) {
    float x = A[i];
    for (int k = 0; k < 64; k++)
      x = x * 0.999f + 1.0f;
    s += x;
  }
  return s;
}
