/**
 * @file bench_migration.c
 * @brief Measures the cost of migrating a thread between a
 * P-core and an E-core with sched_setaffinity (P->E and E->P).
 *
 * Two components are measured per migration:
 *   1. syscall: wall time of sched_setaffinity() on the calling thread, which
 *      returns after the thread is running on an allowed CPU (the migration
 *      itself, including the context switch);
 *   2. refill:  extra time to touch a working set of --ws-kb right after the
 *      migration, versus touching it again warm on the same core (private
 *      L1/L2 refill penalty).
 * The sum is the effective switch cost the WinHint DP uses for hardware
 * (compiler switch-cost parameter, PROPOSAL §3.2).
 *
 * Usage: bench_migration [-n iters] [-w ws_kb] [-p pcpu] [-e ecpu] [-s] [-o out.json]
 *   -s   migrate to the whole P/E sets instead of single CPUs
 *   -n   iterations (default 200, minimum 2); -w working set in KiB (default 256)
 *   -p/-e P/E CPU for single-CPU mode (default: first CPU of each set);
 *        must be in 0..CPU_SETSIZE-1 (bad usage otherwise)
 *   -o   JSON output file (default stdout); a one-line summary goes to stderr
 *
 * JSON statistics are in microseconds. Exit status: 0 ok, 1 setup error,
 * 2 bad usage, 3 if any migration landed on the wrong core type.
 */
#define _GNU_SOURCE
#include "whutil.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <unistd.h>

/** @brief qsort comparator for ascending uint64_t. */
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

/** @brief Summary statistics of a sample (same unit as the input, ns). */
typedef struct { double mean, median, p10, p90, min, max; } st_t;

/**
 * @brief Compute mean, median, p10, p90, min and max.
 * @param[in,out] v Samples; sorted in place.
 * @param[in]     n Number of samples.
 * @return Statistics (all zero if @p n <= 0).
 */
static st_t stats(uint64_t *v, int n) {
    st_t s = {0};
    if (n <= 0) return s;
    qsort(v, (size_t)n, sizeof *v, cmp_u64);
    double sum = 0;
    for (int i = 0; i < n; i++) sum += (double)v[i];
    s.mean = sum / n;
    s.median = (double)v[n / 2];
    s.p10 = (double)v[n / 10];
    s.p90 = (double)v[(n * 9) / 10];
    s.min = (double)v[0];
    s.max = (double)v[n - 1];
    return s;
}

/** @brief Sink that keeps touch() from being optimized away. */
static volatile uint64_t sink;
/**
 * @brief Read-modify-write one word per 64-byte line of a buffer.
 * @param[in,out] buf Working set.
 * @param[in]     n   Buffer length in uint64_t words.
 * @return Elapsed time in ns.
 */
static uint64_t touch(volatile uint64_t *buf, size_t n) {
    uint64_t t0 = wh_now_ns(), acc = 0;
    for (size_t i = 0; i < n; i += 8) { acc += buf[i]; buf[i] = acc; }  /* one line per 64B */
    sink += acc;
    return wh_now_ns() - t0;
}

/**
 * @brief Busy-wait for a number of microseconds.
 * @param[in] us Duration in microseconds.
 */
static void spin_us(unsigned us) {
    uint64_t end = wh_now_ns() + us * 1000ull;
    while (wh_now_ns() < end) { }
}

/**
 * @brief Print one statistics object as a JSON member, converted from ns to us.
 * @param[in] f    Output stream.
 * @param[in] name JSON key.
 * @param[in] s    Statistics in ns.
 * @param[in] last Non-zero to omit the trailing comma.
 */
static void print_st(FILE *f, const char *name, st_t s, int last) {
    fprintf(f, "    \"%s\": {\"mean\": %.3f, \"median\": %.3f, \"p10\": %.3f, \"p90\": %.3f, "
               "\"min\": %.3f, \"max\": %.3f}%s\n",
            name, s.mean / 1e3, s.median / 1e3, s.p10 / 1e3, s.p90 / 1e3, s.min / 1e3,
            s.max / 1e3, last ? "" : ",");
}

/**
 * @brief Run the P<->E migration benchmark and emit JSON (see file header for options).
 * @param[in] argc Argument count.
 * @param[in] argv Arguments.
 * @return Process exit status (0, 1, 2 or 3; see file header).
 */
int main(int argc, char **argv) {
    int iters = 200, ws_kb = 256, use_sets = 0, pcpu = -1, ecpu = -1;
    const char *out = NULL;
    int opt;
    while ((opt = getopt(argc, argv, "n:w:p:e:so:h")) != -1) {
        switch (opt) {
        case 'n': iters = atoi(optarg); break;
        case 'w': ws_kb = atoi(optarg); break;
        case 'p':
        case 'e': {
            int c = atoi(optarg);
            if (c < 0 || c >= CPU_SETSIZE) {
                fprintf(stderr, "bench_migration: -%c %s: CPU must be in 0..%d\n", opt, optarg,
                        CPU_SETSIZE - 1);
                return 2;
            }
            if (opt == 'p') pcpu = c; else ecpu = c;
            break;
        }
        case 's': use_sets = 1; break;
        case 'o': out = optarg; break;
        default:
            fprintf(stderr, "usage: %s [-n iters] [-w ws_kb] [-p pcpu] [-e ecpu] [-s] [-o out.json]\n",
                    argv[0]);
            return opt == 'h' ? 0 : 2;
        }
    }
    if (iters < 2) iters = 2;
    wh_topo t;
    if (wh_topo_detect(&t) != 0) {
        fprintf(stderr, "bench_migration: no hybrid P/E topology (set WINHINT_PCPUS/WINHINT_ECPUS)\n");
        return 1;
    }
    cpu_set_t mp, me;
    if (use_sets) { mp = t.p; me = t.e; }
    else {
        if (pcpu < 0) pcpu = wh_cpuset_first(&t.p);
        if (ecpu < 0) ecpu = wh_cpuset_first(&t.e);
        CPU_ZERO(&mp); CPU_SET(pcpu, &mp);
        CPU_ZERO(&me); CPU_SET(ecpu, &me);
    }
    size_t n = (size_t)ws_kb * 1024 / sizeof(uint64_t);
    volatile uint64_t *buf = calloc(n, sizeof(uint64_t));
    uint64_t *sys_pe = calloc((size_t)iters, 8), *sys_ep = calloc((size_t)iters, 8);
    uint64_t *ref_pe = calloc((size_t)iters, 8), *ref_ep = calloc((size_t)iters, 8);
    uint64_t *noop = calloc((size_t)iters, 8);
    if (!buf || !sys_pe || !sys_ep || !ref_pe || !ref_ep || !noop) return 1;

    if (sched_setaffinity(0, sizeof mp, &mp)) { perror("sched_setaffinity"); return 1; }
    int bad = 0;
    for (int i = 0; i < iters; i++) {
        /* on P: warm, then noop (redundant) affinity call as a baseline */
        touch(buf, n); touch(buf, n);
        uint64_t t0 = wh_now_ns();
        sched_setaffinity(0, sizeof mp, &mp);
        noop[i] = wh_now_ns() - t0;
        /* P -> E */
        t0 = wh_now_ns();
        sched_setaffinity(0, sizeof me, &me);
        sys_pe[i] = wh_now_ns() - t0;
        if (wh_topo_side_of_cpu(&t, sched_getcpu()) != WH_SIDE_E) bad++;
        uint64_t cold = touch(buf, n), warm = touch(buf, n);
        ref_pe[i] = cold > warm ? cold - warm : 0;
        spin_us(50);
        /* E -> P */
        t0 = wh_now_ns();
        sched_setaffinity(0, sizeof mp, &mp);
        sys_ep[i] = wh_now_ns() - t0;
        if (wh_topo_side_of_cpu(&t, sched_getcpu()) != WH_SIDE_P) bad++;
        cold = touch(buf, n); warm = touch(buf, n);
        ref_ep[i] = cold > warm ? cold - warm : 0;
        spin_us(50);
    }
    uint64_t *tot_pe = calloc((size_t)iters, 8), *tot_ep = calloc((size_t)iters, 8);
    for (int i = 0; i < iters; i++) { tot_pe[i] = sys_pe[i] + ref_pe[i]; tot_ep[i] = sys_ep[i] + ref_ep[i]; }
    st_t s_pe = stats(sys_pe, iters), s_ep = stats(sys_ep, iters);
    st_t r_pe = stats(ref_pe, iters), r_ep = stats(ref_ep, iters);
    st_t t_pe = stats(tot_pe, iters), t_ep = stats(tot_ep, iters), s_no = stats(noop, iters);
    double rec_us = (t_pe.median + t_ep.median) / 2.0 / 1e3;
    double fghz = t.fmax_p_khz > 0 ? t.fmax_p_khz / 1e6 : 0;

    FILE *f = out ? fopen(out, "w") : stdout;
    if (!f) { perror(out); return 1; }
    struct utsname u;
    uname(&u);
    char a[128], b[128];
    fprintf(f, "{\n  \"tool\": \"bench_migration\",\n  \"kernel\": \"%s\",\n  \"iters\": %d,\n"
               "  \"ws_kb\": %d,\n  \"target\": \"%s\",\n  \"pcpus\": \"%s\",\n  \"ecpus\": \"%s\",\n"
               "  \"smt_active\": %d,\n  \"fmax_p_khz\": %ld,\n  \"fmax_e_khz\": %ld,\n"
               "  \"wrong_side\": %d,\n  \"units\": \"us\",\n",
            u.release, iters, ws_kb, use_sets ? "sets" : "single",
            wh_cpuset_str(&mp, a, sizeof a), wh_cpuset_str(&me, b, sizeof b), t.smt_on,
            t.fmax_p_khz, t.fmax_e_khz, bad);
    fprintf(f, "  \"noop_setaffinity\": {\n");
    print_st(f, "syscall", s_no, 1);
    fprintf(f, "  },\n  \"p_to_e\": {\n");
    print_st(f, "syscall", s_pe, 0);
    print_st(f, "refill", r_pe, 0);
    print_st(f, "total", t_pe, 1);
    fprintf(f, "  },\n  \"e_to_p\": {\n");
    print_st(f, "syscall", s_ep, 0);
    print_st(f, "refill", r_ep, 0);
    print_st(f, "total", t_ep, 1);
    fprintf(f, "  },\n  \"switch_cost_us\": %.3f,\n  \"switch_cost_cycles_at_pmax\": %.0f\n}\n",
            rec_us, rec_us * 1e3 * fghz);
    if (out) fclose(f);
    fprintf(stderr, "bench_migration: P->E %.1f us (syscall %.1f), E->P %.1f us (syscall %.1f); "
                    "switch cost %.1f us%s%s\n",
            t_pe.median / 1e3, s_pe.median / 1e3, t_ep.median / 1e3, s_ep.median / 1e3, rec_us,
            out ? " -> " : "", out ? out : "");
    return bad ? 3 : 0;
}
