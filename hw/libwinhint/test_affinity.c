/**
 * @file test_affinity.c
 * @brief Smoke test for libwinhint (WINHINT_MODE=migrate):
 * calls __winhint_setwin with large/small/release windows and checks the
 * thread's affinity (sched_getaffinity, /proc/self/status) and current CPU.
 * Also exercises __winhint_region. Exit status 0 = all checks passed.
 *
 * The expected P/E affinity mirrors libwinhint's targets: the full P/E sets,
 * intersected with the initial affinity under WINHINT_RESPECT_AFFINITY=1, and
 * reduced to one CPU per side under WINHINT_PIN=single (WINHINT_PCPU /
 * WINHINT_ECPU, default the first CPU of each target set). Exit status 1 = a
 * check failed or no hybrid topology, 2 = WINHINT_MODE set to something other
 * than migrate.
 */
#define _GNU_SOURCE
#include "winhint.h"
#include "whutil.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/** @brief Sink that keeps work() from being optimized away. */
static volatile double sink;
/**
 * @brief Spin on a floating-point recurrence so the scheduler can settle.
 * @param[in] n Iterations.
 */
static void work(int n) {
    double x = 1.0;
    for (int i = 0; i < n; i++) x = x * 1.0000001 + 1e-9;
    sink = x;
}

/**
 * @brief Copy the Cpus_allowed_list value from /proc/self/status.
 * @param[out] out Destination (empty string if not found).
 * @param[in]  n   Size of @p out.
 */
static void proc_allowed(char *out, size_t n) {
    FILE *f = fopen("/proc/self/status", "r");
    char line[512];
    out[0] = 0;
    while (f && fgets(line, sizeof line, f))
        if (!strncmp(line, "Cpus_allowed_list:", 18)) {
            char *p = line + 18;
            while (*p == ' ' || *p == '\t') p++;
            p[strcspn(p, "\n")] = 0;
            snprintf(out, n, "%s", p);
        }
    if (f) fclose(f);
}

static cpu_set_t want_p; ///< Expected affinity for WH_SIDE_P (see expected_masks()).
static cpu_set_t want_e; ///< Expected affinity for WH_SIDE_E (see expected_masks()).

/**
 * @brief Compute the P/E target masks the same way libwinhint's init does.
 *
 * Full topology sets; AND-ed with @p orig if WINHINT_RESPECT_AFFINITY is non-zero;
 * then, under WINHINT_PIN=single, one CPU per side (WINHINT_PCPU / WINHINT_ECPU,
 * default the first CPU of the set computed so far).
 *
 * @param[in] t    Detected topology.
 * @param[in] orig Affinity at program start.
 */
static void expected_masks(const wh_topo *t, const cpu_set_t *orig) {
    want_p = t->p;
    want_e = t->e;
    if (wh_env_long("WINHINT_RESPECT_AFFINITY", 0)) {
        CPU_AND(&want_p, &want_p, orig);
        CPU_AND(&want_e, &want_e, orig);
    }
    const char *pin = getenv("WINHINT_PIN");
    if (pin && !strcasecmp(pin, "single")) {
        int cp = (int)wh_env_long("WINHINT_PCPU", wh_cpuset_first(&want_p));
        int ce = (int)wh_env_long("WINHINT_ECPU", wh_cpuset_first(&want_e));
        CPU_ZERO(&want_p);
        CPU_ZERO(&want_e);
        if (cp >= 0) CPU_SET(cp, &want_p);
        if (ce >= 0) CPU_SET(ce, &want_e);
    }
}

/**
 * @brief Check the thread's current affinity and CPU against the expected side and print a line.
 * @param[in] what      Label for the output line.
 * @param[in] t         Detected topology.
 * @param[in] want_side WH_SIDE_P / WH_SIDE_E (affinity must equal the expected target mask,
 *                      see expected_masks(), and the current CPU be on that side) or
 *                      WH_SIDE_ORIG (affinity must equal @p orig).
 * @param[in] orig      Affinity at program start.
 * @return 1 if the check passed, else 0.
 */
static int check(const char *what, const wh_topo *t, int want_side, const cpu_set_t *orig) {
    cpu_set_t cur;
    sched_getaffinity(0, sizeof cur, &cur);
    work(2000000);
    int cpu = sched_getcpu();
    char a[256], st[512];
    proc_allowed(st, sizeof st);
    wh_cpuset_str(&cur, a, sizeof a);
    int ok;
    if (want_side == WH_SIDE_P) ok = CPU_EQUAL(&cur, &want_p) && wh_topo_side_of_cpu(t, cpu) == WH_SIDE_P;
    else if (want_side == WH_SIDE_E) ok = CPU_EQUAL(&cur, &want_e) && wh_topo_side_of_cpu(t, cpu) == WH_SIDE_E;
    else ok = CPU_EQUAL(&cur, orig);
    printf("%-28s affinity={%s} /proc={%s} cpu=%d expect=%s  %s\n", what, a, st, cpu,
           wh_side_name(want_side), ok ? "OK" : "FAIL");
    return ok;
}

/**
 * @brief Drive setwin/region through P, E, threshold and release cases and verify placement.
 * @return 0 if all checks passed, 1 on failure or no hybrid topology, 2 on a wrong WINHINT_MODE.
 */
int main(void) {
    wh_topo t;
    if (wh_topo_detect(&t) != 0) { fprintf(stderr, "no hybrid topology\n"); return 1; }
    const char *m = getenv("WINHINT_MODE");
    if (m && strcmp(m, "migrate")) { fprintf(stderr, "run with WINHINT_MODE=migrate (or unset)\n"); return 2; }
    unsigned thr = (unsigned)wh_env_long("WINHINT_THRESHOLD", 192);
    cpu_set_t orig;
    sched_getaffinity(0, sizeof orig, &orig);
    expected_masks(&t, &orig);
    int ok = 1;
    ok &= check("start", &t, WH_SIDE_ORIG, &orig);
    __winhint_region(0);
    __winhint_setwin(thr + 64);
    ok &= check("setwin(large) -> P", &t, WH_SIDE_P, &orig);
    __winhint_setwin(thr + 64);                       /* redundant: no migration */
    __winhint_region(1);
    __winhint_setwin(64);
    ok &= check("setwin(64) -> E", &t, WH_SIDE_E, &orig);
    __winhint_setwin(thr);
    ok &= check("setwin(threshold) -> P", &t, WH_SIDE_P, &orig);
    __winhint_setwin(0);
    ok &= check("setwin(0) -> release", &t, WH_SIDE_ORIG, &orig);
    for (int i = 0; i < 20; i++) {  /* flip-flop: exercises the migration stats */
        __winhint_region(2 + (i & 1));
        __winhint_setwin((i & 1) ? 32 : 512);
        work(200000);
    }
    __winhint_setwin(0);
    winhint_flush();
    printf("%s\n", ok ? "ALL OK" : "SOME CHECKS FAILED");
    return ok ? 0 : 1;
}
