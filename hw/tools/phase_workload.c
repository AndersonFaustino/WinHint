/**
 * @file phase_workload.c
 * @brief Synthetic two-phase workload for smoke-testing the hw/
 * runtime and baselines: alternates a compute-bound phase (dependent FP chain,
 * cache-resident) and a memory-bound phase (random pointer chasing over a
 * large array, several independent chains -> MLP).
 *
 *   phase_workload [seconds=2] [phase_ms=100] [mem_mb=64] [mode=alt]
 *
 * Phase classes (region id = class id in the hinted build):
 *   0 compute  dependent FP multiply-add chain (ILP 1, cache-resident)
 *   1 memory   8 independent random pointer chases over mem_mb (LLC misses, MLP <= 8)
 *   2 ilp      8 independent FP multiply-add chains (high ILP, cache-resident)
 * Modes: alt (compute/memory alternating, the default), alt-ilp (ilp/memory),
 * compute, memory, ilp (one class only; used by hw/fidelity.py for the R4/R5 checks).
 * Output: one line "phases=.. mem_steps=.. fp_steps=.. chk=.." followed by
 * " ilp_steps=.. ns_compute=.. ns_memory=.. ns_ilp=.." (time spent per class), so
 * the work rate of a class is steps / ns (mem step = 8 loads, fp step = 1 op,
 * ilp step = 8 ops).
 *
 * Built three times: plain; -DWITH_HINTS (calls __winhint_region/setwin at
 * each phase boundary like WinHint call-mode code: memory phase -> large
 * window, compute/ilp phase -> small window); and -DWITH_NOP_HINTS (the same hints
 * as x86 multi-byte NOPs, docs/interfaces.md §2, for the NOP-overhead path).
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef WITH_HINTS
#include "winhint.h"
#endif
#if defined(WITH_NOP_HINTS) && (defined(__x86_64__) || defined(__i386__))
/**
 * @brief Emit a WinHint x86 NOP hint.
 *
 * nopl DISP32(%rax), DISP32 = 0x57480000 | kind<<12 | payload (kind 1 setwin W/8, 2 region id)
 *
 * @param kind    1 = setwin, 2 = region (compile-time constant).
 * @param payload W/8 for setwin, region id for region (compile-time constant).
 */
#define WH_NOP(kind, payload) \
    __asm__ volatile(".byte 0x0f,0x1f,0x80; .long %c0" ::"i"(0x57480000u | ((kind) << 12) | (payload)))
#endif

/**
 * @brief Current CLOCK_MONOTONIC time.
 * @return Nanoseconds since an unspecified epoch.
 */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/**
 * @brief Run the phase loop for the requested time and print the step counters.
 *
 * Positional arguments: seconds, phase_ms, mem_mb, mode (see file header).
 * The pointer-chase array is a Sattolo random cyclic permutation of mem_mb MiB
 * of uint32_t indices; it is built in every mode.
 *
 * @param[in] argc Argument count.
 * @param[in] argv Arguments.
 * @return 0 on success, 1 if the array cannot be allocated, 2 on an unknown mode
 *         or mem_mb < 1.
 */
int main(int argc, char **argv) {
    double secs = argc > 1 ? atof(argv[1]) : 2.0;
    double phase_ms = argc > 2 ? atof(argv[2]) : 100.0;
    long mb_arg = argc > 3 ? atol(argv[3]) : 64;
    const char *mode = argc > 4 ? argv[4] : "alt";
    enum { C_COMPUTE = 0, C_MEMORY = 1, C_ILP = 2 };
    int cls[2];
    if (!strcmp(mode, "alt")) { cls[0] = C_COMPUTE; cls[1] = C_MEMORY; }
    else if (!strcmp(mode, "alt-ilp")) { cls[0] = C_ILP; cls[1] = C_MEMORY; }
    else if (!strcmp(mode, "compute")) cls[0] = cls[1] = C_COMPUTE;
    else if (!strcmp(mode, "memory")) cls[0] = cls[1] = C_MEMORY;
    else if (!strcmp(mode, "ilp")) cls[0] = cls[1] = C_ILP;
    else {
        fprintf(stderr, "phase_workload: unknown mode '%s' (alt, alt-ilp, compute, memory, ilp)\n", mode);
        return 2;
    }
    if (mb_arg < 1) {
        fprintf(stderr, "phase_workload: mem_mb must be >= 1 (got '%s')\n", argv[3]);
        return 2;
    }
    size_t mb = (size_t)mb_arg;
    size_t n = mb * 1024 * 1024 / sizeof(uint32_t);
    uint32_t *next = malloc(n * sizeof *next);
    if (!next) return 1;
    /* random cyclic permutation (Sattolo) */
    for (size_t i = 0; i < n; i++) next[i] = (uint32_t)i;
    uint64_t s = 88172645463325252ull;
    for (size_t i = n - 1; i > 0; i--) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        size_t j = s % i;
        uint32_t t = next[i]; next[i] = next[j]; next[j] = t;
    }
    uint64_t end = now_ns() + (uint64_t)(secs * 1e9), ph = (uint64_t)(phase_ms * 1e6);
    uint32_t p[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    double x = 1.0, y[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    unsigned long phases = 0, mem_steps = 0, fp_steps = 0, ilp_steps = 0;
    uint64_t ns_cls[3] = {0, 0, 0};
    while (now_ns() < end) {
        int c = cls[phases & 1];
#ifdef WITH_HINTS
        __winhint_region((unsigned)c);
        __winhint_setwin(c == C_MEMORY ? 256 : 64);
#elif defined(WH_NOP)
        if (c == C_MEMORY) { WH_NOP(2, 1); WH_NOP(1, 256 / 8); }
        else if (c == C_ILP) { WH_NOP(2, 2); WH_NOP(1, 64 / 8); }
        else { WH_NOP(2, 0); WH_NOP(1, 64 / 8); }
#endif
        uint64_t ps = now_ns(), pe = ps + ph;
        if (c == C_MEMORY) {
            while (now_ns() < pe)
                for (int k = 0; k < 4096; k++, mem_steps++)
                    for (int j = 0; j < 8; j++) p[j] = next[p[j]];
        } else if (c == C_ILP) {
            while (now_ns() < pe)
                for (int k = 0; k < 8192; k++, ilp_steps++)
                    for (int j = 0; j < 8; j++) y[j] = y[j] * 1.000000001 + 1e-12;
        } else {
            while (now_ns() < pe)
                for (int k = 0; k < 65536; k++, fp_steps++) x = x * 1.000000001 + 1e-12;
        }
        ns_cls[c] += now_ns() - ps;
        phases++;
    }
#ifdef WITH_HINTS
    __winhint_setwin(0);
#elif defined(WH_NOP)
    WH_NOP(1, 0);
#endif
    printf("phases=%lu mem_steps=%lu fp_steps=%lu chk=%u %.6f ilp_steps=%lu ns_compute=%llu "
           "ns_memory=%llu ns_ilp=%llu\n", phases, mem_steps, fp_steps, p[0] ^ p[7], x + y[0] + y[7],
           ilp_steps, (unsigned long long)ns_cls[C_COMPUTE], (unsigned long long)ns_cls[C_MEMORY],
           (unsigned long long)ns_cls[C_ILP]);
    free(next);
    return 0;
}
