/**
 * @file winhint.c
 * @brief libwinhint: maps WinHint runtime-call hints to P/E-core placement
 * on Intel hybrid CPUs (PROPOSAL Phase E2, docs/interfaces.md §2).
 *
 *   __winhint_setwin(W): W >= WINHINT_THRESHOLD -> P-core set,
 *                        0 < W < threshold      -> E-core set,
 *                        W == 0 (release)       -> WINHINT_RELEASE (default: the
 *                                                  original affinity, OS decides)
 *   __winhint_region(id): region marker; per-region accounting, and the R5
 *                        (Sondag & Rajan, CGO'11) sampling/assignment policy in
 *                        WINHINT_MODE=sondag.
 *
 * Modes (WINHINT_MODE):
 *   off      hooks return immediately (call-overhead measurement)
 *   log      decisions and per-region counters, no affinity change
 *   migrate  (default) setwin -> sched_setaffinity on the calling thread
 *   sondag   ignore setwin; sample each region type on P and E for the first
 *            K visits each, then pin the type to the better core type
 *
 * Only the calling thread is migrated (the kernels are single-threaded).
 * All configuration is via environment variables -- see docs/guide/hardware/index.md.
 *
 * State lives in the single global ::G, serialized by a spin lock. The library
 * initializes itself from a constructor (or lazily on the first hook) and writes
 * its logs from a destructor. Outputs (when configured):
 *   WINHINT_LOG    per-(region, side) CSV + <path>.summary.json
 *   WINHINT_TRACE  one CSV row per applied/attempted migration
 */
#define _GNU_SOURCE
#include "winhint.h"
#include "whutil.h"

#include <errno.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#define NREG 65            /**< region ids 0..63, 64 = "no region yet" */
#define REG_NONE 64        /**< Bucket for "no region yet" / out-of-range ids. */
#define NSIDE 3            /**< WH_SIDE_ORIG, WH_SIDE_P, WH_SIDE_E */

/** @brief Runtime modes (WINHINT_MODE); values index mode_names. */
enum { M_OFF = 0, M_LOG, M_MIGRATE, M_SONDAG };
/** @brief WINHINT_MODE spellings, indexed by the mode enum. */
static const char *mode_names[] = {"off", "log", "migrate", "sondag"};

/** @brief Per-(region, side) accounting bucket; one row of the WINHINT_LOG CSV. */
typedef struct {
    uint64_t visits, segs, wall_ns;  ///< Region entries, closed segments, wall time (ns).
    uint64_t cyc[WH_PMU_MAX], ins[WH_PMU_MAX]; ///< User-mode cycles / instructions per core-type PMU.
    uint64_t e_pkg_uj, e_core_uj;    ///< RAPL package / core energy (uJ).
} bucket_t;

/** @brief Cost statistics of sched_setaffinity calls for one (from, to) side pair. */
typedef struct {
    uint64_t n, sum_ns, min_ns, max_ns; ///< Count, total, min and max call time (ns).
} mstat_t;

/**
 * @brief Global runtime state (configuration, placement state, counters, accounting).
 *
 * Fields mirror the WINHINT_* environment variables documented in docs/guide/hardware/index.md;
 * all access except the M_OFF fast path is under lock(); n_setwin and n_region
 * are also bumped by that fast path, so they are only updated/read atomically.
 */
static struct {
    volatile int inited;
    int lock;
    int mode;
    unsigned threshold;
    int release_side, initial_side;
    long hyst;
    uint64_t dwell_ns;
    int pin_single;
    int verbose;

    wh_topo topo;
    cpu_set_t orig, mask_p, mask_e;
    int topo_ok;

    /* placement state */
    int cur_side;          /**< logical side currently requested/applied */
    int cand_side;
    long cand_streak;
    int pending_side;      /**< deferred by min-dwell, -1 none */
    uint64_t last_mig_ns;

    /* counters */
    uint64_t n_setwin, n_region, n_redundant, n_rate_limited, n_hyst_held;
    uint64_t n_errors, n_mismatch, n_logical_moves;
    int last_errno;
    mstat_t mig[NSIDE][NSIDE];

    /* accounting */
    int account;           /**< per-region accounting enabled */
    int cur_region;
    uint64_t seg_start_ns, t_init_ns, init_ns;
    wh_perf perf;
    int perf_on;
    wh_perf_vals pv_last;
    wh_rapl rapl;
    int rapl_seg, rapl_on;
    uint64_t rp_last, rc_last, rp_init, rc_init;
    bucket_t b[NREG][NSIDE];
    char log_path[1024];
    FILE *trace;

    /* sondag (R5) */
    long sk;
    double s_thr;
    int type_of[NREG];
    int assigned[NREG];    /**< per type: -1 undecided, else side */
    uint64_t s_n[NREG][NSIDE], s_ns[NREG][NSIDE], s_ins[NREG][NSIDE];
    double s_ratio[NREG];
    int flushed;
} G;

/** @brief Acquire the global spin lock (yields while contended). */
static void lock(void) { while (__atomic_exchange_n(&G.lock, 1, __ATOMIC_ACQUIRE)) sched_yield(); }
/** @brief Release the global spin lock. */
static void unlock(void) { __atomic_store_n(&G.lock, 0, __ATOMIC_RELEASE); }

/**
 * @brief Parse a side name ("P", "E", "orig"/"os"; case-insensitive).
 * @param[in] s    String to parse (NULL/empty allowed).
 * @param[in] dflt Value for NULL, empty or unknown strings.
 * @return WH_SIDE_P, WH_SIDE_E, WH_SIDE_ORIG or @p dflt.
 */
static int parse_side(const char *s, int dflt) {
    if (!s || !*s) return dflt;
    if (!strcasecmp(s, "p")) return WH_SIDE_P;
    if (!strcasecmp(s, "e")) return WH_SIDE_E;
    if (!strcasecmp(s, "orig") || !strcasecmp(s, "os")) return WH_SIDE_ORIG;
    return dflt;
}

/**
 * @brief Load the R5 region-to-type map (WINHINT_SONDAG_TYPES).
 *
 * Starts from the identity map, then applies "id type" or "id,type" lines
 * ('#' lines are comments; ids/types outside 0..63 are ignored).
 *
 * @param[in] path Map file; NULL/empty keeps the identity map.
 */
static void load_sondag_types(const char *path) {
    for (int i = 0; i < NREG; i++) G.type_of[i] = i;  /* identity typing */
    if (!path || !*path) return;
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[winhint] WINHINT_SONDAG_TYPES=%s: %s (using one type per region)\n",
                path, strerror(errno));
        return;
    }
    int id, ty;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        if (sscanf(line, "%d %d", &id, &ty) == 2 || sscanf(line, "%d,%d", &id, &ty) == 2)
            if (id >= 0 && id < REG_NONE && ty >= 0 && ty < REG_NONE) G.type_of[id] = ty;
    }
    fclose(f);
}

/**
 * @brief Affinity mask for a side.
 * @param[in] side WH_SIDE_P, WH_SIDE_E, anything else = original affinity.
 * @return Pointer into ::G (mask_p, mask_e or orig).
 */
static const cpu_set_t *side_mask(int side) {
    return side == WH_SIDE_P ? &G.mask_p : side == WH_SIDE_E ? &G.mask_e : &G.orig;
}

/**
 * @brief One-time initialization from the environment (idempotent; call under lock()).
 *
 * Reads the configuration, records the original affinity, detects the P/E
 * topology (falling back to log mode, or exiting with status 3 under
 * WINHINT_REQUIRE_HYBRID=1, if none is found), opens per-region perf counters
 * and RAPL when accounting is enabled, and opens the migration trace.
 * In M_OFF mode only the log path is recorded.
 */
static void init(void) {
    if (G.inited) return;
    G.inited = 1;
    const char *m = getenv("WINHINT_MODE");
    G.mode = M_MIGRATE;
    if (m && *m) {
        for (int i = 0; i < 4; i++)
            if (!strcasecmp(m, mode_names[i])) G.mode = i;
        if (!strcmp(m, "0")) G.mode = M_OFF;
    }
    G.verbose = (int)wh_env_long("WINHINT_VERBOSE", 0);
    G.threshold = (unsigned)wh_env_long("WINHINT_THRESHOLD", 192);
    G.release_side = parse_side(getenv("WINHINT_RELEASE"), WH_SIDE_ORIG);
    G.initial_side = parse_side(getenv("WINHINT_INITIAL"), WH_SIDE_ORIG);
    G.hyst = wh_env_long("WINHINT_HYST", 1);
    if (G.hyst < 1) G.hyst = 1;
    G.dwell_ns = (uint64_t)wh_env_long("WINHINT_MIN_DWELL_US", 0) * 1000ull;
    const char *pin = getenv("WINHINT_PIN");
    G.pin_single = pin && !strcasecmp(pin, "single");
    G.cur_side = WH_SIDE_ORIG;
    G.cand_side = -1;
    G.pending_side = -1;
    G.cur_region = REG_NONE;
    for (int i = 0; i < NREG; i++) G.assigned[i] = -1;
    G.t_init_ns = G.seg_start_ns = wh_now_ns();
    sched_getaffinity(0, sizeof G.orig, &G.orig);

    if (G.mode == M_OFF) {
        const char *lp = getenv("WINHINT_LOG");
        if (lp && *lp) snprintf(G.log_path, sizeof G.log_path, "%s", lp);
        return;
    }

    G.topo_ok = wh_topo_detect(&G.topo) == 0;
    G.mask_p = G.topo.p;
    G.mask_e = G.topo.e;
    if (wh_env_long("WINHINT_RESPECT_AFFINITY", 0)) {
        CPU_AND(&G.mask_p, &G.mask_p, &G.orig);
        CPU_AND(&G.mask_e, &G.mask_e, &G.orig);
    }
    if (G.pin_single) {
        int cp = (int)wh_env_long("WINHINT_PCPU", wh_cpuset_first(&G.mask_p));
        int ce = (int)wh_env_long("WINHINT_ECPU", wh_cpuset_first(&G.mask_e));
        CPU_ZERO(&G.mask_p);
        CPU_ZERO(&G.mask_e);
        if (cp >= 0) CPU_SET(cp, &G.mask_p);
        if (ce >= 0) CPU_SET(ce, &G.mask_e);
    }
    if (!G.topo_ok || !CPU_COUNT(&G.mask_p) || !CPU_COUNT(&G.mask_e)) {
        char a[256], b[256];
        fprintf(stderr, "[winhint] no hybrid P/E topology found (P={%s} E={%s} from %s; needs the "
                        "cpu_core/cpu_atom PMUs in sysfs, i.e. an Intel hybrid CPU, or "
                        "WINHINT_PCPUS/WINHINT_ECPUS)\n",
                wh_cpuset_str(&G.mask_p, a, sizeof a), wh_cpuset_str(&G.mask_e, b, sizeof b), G.topo.src);
        /* Campaign runs must not silently degrade to "no migration" (the driver sets this). */
        if (wh_env_long("WINHINT_REQUIRE_HYBRID", 0) && (G.mode == M_MIGRATE || G.mode == M_SONDAG)) {
            fprintf(stderr, "[winhint] WINHINT_REQUIRE_HYBRID=1: aborting (mode %s needs P and E cores)\n",
                    mode_names[G.mode]);
            _exit(3);
        }
        fprintf(stderr, "[winhint] migrations disabled, falling back to log mode\n");
        if (G.mode == M_MIGRATE || G.mode == M_SONDAG) G.mode = M_LOG;
    }

    G.sk = wh_env_long("WINHINT_SONDAG_K", 2);
    if (G.sk < 1) G.sk = 1;
    G.s_thr = wh_env_double("WINHINT_SONDAG_THRESHOLD", 1.4);
    load_sondag_types(getenv("WINHINT_SONDAG_TYPES"));

    const char *lp = getenv("WINHINT_LOG");
    if (lp && *lp) snprintf(G.log_path, sizeof G.log_path, "%s", lp);
    G.account = G.log_path[0] != 0 || G.mode == M_SONDAG;

    if (G.account && wh_env_long("WINHINT_PERF", 1)) {
        wh_evspec ev[2] = {
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1, "cycles"},
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, -1, "instructions"},
        };
        G.perf_on = wh_perf_open(&G.perf, 0, &G.topo, ev, 2, 0) == 0;
        if (!G.perf_on)
            fprintf(stderr, "[winhint] per-region perf counters unavailable: %s\n", G.perf.msg);
        else
            wh_perf_read(&G.perf, &G.pv_last);
    }
    if (G.log_path[0] && wh_env_long("WINHINT_RAPL", 0)) {
        G.rapl_on = wh_rapl_open(&G.rapl) == 0;
        if (!G.rapl_on)
            fprintf(stderr, "[winhint] RAPL unavailable: %s\n", G.rapl.msg);
        else {
            wh_rapl_read(&G.rapl, &G.rp_last, &G.rc_last);
            G.rp_init = G.rp_last;
            G.rc_init = G.rc_last;
            G.rapl_seg = 1;
        }
    }
    const char *tp = getenv("WINHINT_TRACE");
    if (tp && *tp) {
        G.trace = fopen(tp, "w");
        if (G.trace) fprintf(G.trace, "t_ns,from,to,cost_ns,cpu_after,region,ok\n");
    }
    /* Setup (topology, perf_event_open -- the first open can take several ms on this
     * kernel) is reported as init_ns and not charged to region -1. */
    G.seg_start_ns = wh_now_ns();
    G.init_ns = G.seg_start_ns - G.t_init_ns;
    if (G.perf_on) wh_perf_read(&G.perf, &G.pv_last);
    if (G.rapl_seg) wh_rapl_read(&G.rapl, &G.rp_last, &G.rc_last);
    if (G.verbose) {
        char a[256], b[256];
        fprintf(stderr, "[winhint] mode=%s thr=%u P={%s} E={%s} hyst=%ld dwell_us=%llu\n",
                mode_names[G.mode], G.threshold, wh_cpuset_str(&G.mask_p, a, sizeof a),
                wh_cpuset_str(&G.mask_e, b, sizeof b), G.hyst,
                (unsigned long long)(G.dwell_ns / 1000));
    }
}

/* ------------------------------------------------------------ accounting */
/**
 * @brief Close the current accounting segment and charge it to (cur_region, cur_side).
 *
 * Adds wall time, perf deltas and RAPL deltas to the bucket; in sondag mode also
 * adds time/instructions to the sampling totals of an undecided region type.
 * No-op unless accounting is enabled.
 *
 * @param[in] now Current wh_now_ns() timestamp; becomes the next segment start.
 */
static void close_segment(uint64_t now) {
    if (!G.account) return;
    bucket_t *bk = &G.b[G.cur_region][G.cur_side];
    uint64_t dt = now - G.seg_start_ns;
    bk->segs++;
    bk->wall_ns += dt;
    uint64_t dins = 0;
    if (G.perf_on) {
        wh_perf_vals pv;
        wh_perf_read(&G.perf, &pv);
        for (int i = 0; i < G.perf.npmu; i++) {
            bk->cyc[i] += pv.v[i][0] - G.pv_last.v[i][0];
            uint64_t di = pv.v[i][1] - G.pv_last.v[i][1];
            bk->ins[i] += di;
            dins += di;
        }
        G.pv_last = pv;
    }
    if (G.rapl_seg) {
        uint64_t p, c;
        if (wh_rapl_read(&G.rapl, &p, &c) == 0) {
            bk->e_pkg_uj += wh_rapl_delta(&G.rapl, G.rp_last, p, 0);
            bk->e_core_uj += wh_rapl_delta(&G.rapl, G.rc_last, c, 1);
            G.rp_last = p;
            G.rc_last = c;
        }
    }
    if (G.mode == M_SONDAG && G.cur_region != REG_NONE && G.cur_side != WH_SIDE_ORIG) {
        int ty = G.type_of[G.cur_region];
        if (G.assigned[ty] < 0) {   /* visit count is bumped at region exit */
            G.s_ns[ty][G.cur_side] += dt;
            G.s_ins[ty][G.cur_side] += dins;
        }
    }
    G.seg_start_ns = now;
}

/* ------------------------------------------------------------ migration */
/**
 * @brief Move the calling thread to a side (unconditionally).
 *
 * In log mode only the logical side changes. Otherwise calls sched_setaffinity,
 * records its cost in G.mig[from][side], counts errors and CPU/side mismatches,
 * and appends a WINHINT_TRACE row.
 *
 * @param[in] side Target side.
 * @param[in] now  Current timestamp (unused).
 */
static void do_apply(int side, uint64_t now) {
    int from = G.cur_side;
    if (G.mode == M_LOG) {
        G.n_logical_moves++;
        G.cur_side = side;
        return;
    }
    const cpu_set_t *mask = side_mask(side);
    uint64_t t0 = wh_now_ns();
    int r = sched_setaffinity(0, sizeof(cpu_set_t), mask);
    uint64_t t1 = wh_now_ns();
    int cpu = sched_getcpu();
    if (r != 0) {
        G.n_errors++;
        G.last_errno = errno;
        if (G.n_errors == 1)
            fprintf(stderr, "[winhint] sched_setaffinity(%s) failed: %s\n", wh_side_name(side),
                    strerror(errno));
    } else {
        if (side != WH_SIDE_ORIG && wh_topo_side_of_cpu(&G.topo, cpu) != side) G.n_mismatch++;
        mstat_t *ms = &G.mig[from][side];
        uint64_t c = t1 - t0;
        if (!ms->n || c < ms->min_ns) ms->min_ns = c;
        if (c > ms->max_ns) ms->max_ns = c;
        ms->n++;
        ms->sum_ns += c;
        G.cur_side = side;
        G.last_mig_ns = t1;
    }
    if (G.trace)
        fprintf(G.trace, "%llu,%s,%s,%llu,%d,%d,%d\n", (unsigned long long)(t0 - G.t_init_ns),
                wh_side_name(from), wh_side_name(side), (unsigned long long)(t1 - t0), cpu,
                G.cur_region == REG_NONE ? -1 : G.cur_region, r == 0);
    (void)now;
}

/**
 * @brief Request a side, honoring redundancy elimination and min-dwell rate limiting.
 *
 * A request within WINHINT_MIN_DWELL_US of the last migration is stored as
 * pending and applied later by service_pending().
 *
 * @param[in] side Target side.
 * @param[in] now  Current timestamp (ns).
 */
static void request_side(int side, uint64_t now) {
    if (side == G.cur_side) {
        G.n_redundant++;
        G.pending_side = -1;
        return;
    }
    if (G.dwell_ns && G.last_mig_ns && now - G.last_mig_ns < G.dwell_ns) {
        G.n_rate_limited++;
        G.pending_side = side;
        return;
    }
    G.pending_side = -1;
    do_apply(side, now);
}

/**
 * @brief Apply a rate-limited pending request once the min-dwell time has passed.
 * @param[in] now Current timestamp (ns).
 */
static void service_pending(uint64_t now) {
    if (G.pending_side >= 0 && (!G.dwell_ns || now - G.last_mig_ns >= G.dwell_ns)) {
        int s = G.pending_side;
        G.pending_side = -1;
        if (s != G.cur_side) do_apply(s, now);
    }
}

/** @brief Set once the WINHINT_INITIAL placement has been applied. */
static int initial_applied;
/**
 * @brief Apply the WINHINT_INITIAL side (if not "orig" and the mode is not off).
 * @param[in] now Current timestamp (ns).
 */
static void apply_initial(uint64_t now) {
    initial_applied = 1;
    if (G.initial_side != WH_SIDE_ORIG && G.mode != M_OFF) do_apply(G.initial_side, now);
}

/* ------------------------------------------------------------ hooks */
/** @brief See winhint.h: __winhint_setwin(); applies the threshold and hysteresis. */
void __winhint_setwin(unsigned w) {
    if (__builtin_expect(G.inited && G.mode == M_OFF, 1)) {
        __atomic_fetch_add(&G.n_setwin, 1, __ATOMIC_RELAXED);
        return;
    }
    lock();
    init();
    __atomic_fetch_add(&G.n_setwin, 1, __ATOMIC_RELAXED);
    if (G.mode == M_OFF) { unlock(); return; }
    uint64_t now = wh_now_ns();
    if (!initial_applied) apply_initial(now);
    close_segment(now);
    service_pending(now);
    if (G.mode == M_LOG || G.mode == M_MIGRATE) {
        int want = (w == 0) ? G.release_side : (w >= G.threshold ? WH_SIDE_P : WH_SIDE_E);
        if (want == G.cur_side) {
            G.cand_side = -1;
            G.cand_streak = 0;
            G.n_redundant++;
            G.pending_side = -1;
        } else {
            if (want == G.cand_side) G.cand_streak++;
            else { G.cand_side = want; G.cand_streak = 1; }
            if (G.cand_streak >= G.hyst) {
                G.cand_side = -1;
                G.cand_streak = 0;
                request_side(want, now);
            } else {
                G.n_hyst_held++;
            }
        }
    }
    unlock();
}

/**
 * @brief R5 (Sondag & Rajan) placement for a region type.
 *
 * Samples the type on P until K visits, then on E until K visits; afterwards
 * computes the E/P ratio of time per instruction (time per visit without
 * counters) and assigns P if the ratio is >= WINHINT_SONDAG_THRESHOLD, else E.
 * The assignment is final.
 *
 * @param[in] ty Region type (0..63).
 * @return Side to run the type on.
 */
static int sondag_choose(int ty) {
    if (G.assigned[ty] >= 0) return G.assigned[ty];
    if (G.s_n[ty][WH_SIDE_P] < (uint64_t)G.sk) return WH_SIDE_P;
    if (G.s_n[ty][WH_SIDE_E] < (uint64_t)G.sk) return WH_SIDE_E;
    /* both sampled K times: E-core slowdown ratio from time per instruction
     * (or per visit if no counters); assign P if the slowdown is large. */
    double mp, me;
    if (G.perf_on && G.s_ins[ty][WH_SIDE_P] && G.s_ins[ty][WH_SIDE_E]) {
        mp = (double)G.s_ns[ty][WH_SIDE_P] / (double)G.s_ins[ty][WH_SIDE_P];
        me = (double)G.s_ns[ty][WH_SIDE_E] / (double)G.s_ins[ty][WH_SIDE_E];
    } else {
        mp = (double)G.s_ns[ty][WH_SIDE_P] / (double)G.s_n[ty][WH_SIDE_P];
        me = (double)G.s_ns[ty][WH_SIDE_E] / (double)G.s_n[ty][WH_SIDE_E];
    }
    double ratio = mp > 0 ? me / mp : 1.0;
    G.s_ratio[ty] = ratio;
    G.assigned[ty] = ratio >= G.s_thr ? WH_SIDE_P : WH_SIDE_E;
    if (G.verbose)
        fprintf(stderr, "[winhint] sondag: type %d E/P time ratio %.3f -> %s\n", ty, ratio,
                wh_side_name(G.assigned[ty]));
    return G.assigned[ty];
}

/** @brief See winhint.h: __winhint_region(); closes the segment and runs the sondag policy. */
void __winhint_region(unsigned id) {
    if (__builtin_expect(G.inited && G.mode == M_OFF, 1)) {
        __atomic_fetch_add(&G.n_region, 1, __ATOMIC_RELAXED);
        return;
    }
    lock();
    init();
    __atomic_fetch_add(&G.n_region, 1, __ATOMIC_RELAXED);
    if (G.mode == M_OFF) { unlock(); return; }
    uint64_t now = wh_now_ns();
    if (!initial_applied) apply_initial(now);
    close_segment(now);
    service_pending(now);
    int rid = id < REG_NONE ? (int)id : REG_NONE;
    if (G.mode == M_SONDAG && G.cur_region != REG_NONE && G.cur_side != WH_SIDE_ORIG &&
        G.assigned[G.type_of[G.cur_region]] < 0)
        G.s_n[G.type_of[G.cur_region]][G.cur_side]++;   /* one completed sampled visit */
    if (G.mode == M_SONDAG && rid != REG_NONE) {
        int side = sondag_choose(G.type_of[rid]);
        request_side(side, now);
    }
    G.cur_region = rid;
    G.b[rid][G.cur_side].visits++;
    unlock();
}

/* ------------------------------------------------------------ output */
/**
 * @brief Write the WINHINT_LOG CSV and its `<path>.summary.json` (no-op without a log path).
 *
 * The CSV has one row per non-empty (region, side) bucket (region 64 is written
 * as -1). The JSON holds the configuration, hook/migration counters, per-direction
 * migration costs, total RAPL energy and the sondag decisions.
 */
static void write_outputs(void) {
    if (!G.log_path[0]) return;
    FILE *f = fopen(G.log_path, "w");
    if (!f) {
        fprintf(stderr, "[winhint] cannot write WINHINT_LOG=%s: %s\n", G.log_path, strerror(errno));
        return;
    }
    fprintf(f, "region,side,visits,segments,wall_ns,cycles_p,instructions_p,cycles_e,"
               "instructions_e,energy_pkg_uj,energy_core_uj\n");
    for (int r = 0; r < NREG; r++)
        for (int s = 0; s < NSIDE; s++) {
            bucket_t *b = &G.b[r][s];
            if (!b->visits && !b->segs) continue;
            fprintf(f, "%d,%s,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                    r == REG_NONE ? -1 : r, wh_side_name(s), (unsigned long long)b->visits,
                    (unsigned long long)b->segs, (unsigned long long)b->wall_ns,
                    (unsigned long long)b->cyc[0], (unsigned long long)b->ins[0],
                    (unsigned long long)b->cyc[1], (unsigned long long)b->ins[1],
                    (unsigned long long)b->e_pkg_uj, (unsigned long long)b->e_core_uj);
        }
    fclose(f);

    char jp[1100];
    snprintf(jp, sizeof jp, "%s.summary.json", G.log_path);
    f = fopen(jp, "w");
    if (!f) return;
    char a[256], b[256], o[256];
    uint64_t now = wh_now_ns();
    fprintf(f, "{\n  \"mode\": \"%s\",\n  \"threshold\": %u,\n  \"release\": \"%s\",\n"
               "  \"initial\": \"%s\",\n  \"hyst\": %ld,\n  \"min_dwell_us\": %llu,\n"
               "  \"pin\": \"%s\",\n  \"pcpus\": \"%s\",\n  \"ecpus\": \"%s\",\n"
               "  \"orig_affinity\": \"%s\",\n  \"smt_active\": %d,\n",
            mode_names[G.mode], G.threshold, wh_side_name(G.release_side),
            wh_side_name(G.initial_side), G.hyst, (unsigned long long)(G.dwell_ns / 1000),
            G.pin_single ? "single" : "set", wh_cpuset_str(&G.mask_p, a, sizeof a),
            wh_cpuset_str(&G.mask_e, b, sizeof b), wh_cpuset_str(&G.orig, o, sizeof o),
            G.topo.smt_on);
    fprintf(f, "  \"init_ns\": %llu,\n", (unsigned long long)G.init_ns);
    fprintf(f, "  \"wall_ns\": %llu,\n  \"n_setwin\": %llu,\n  \"n_region\": %llu,\n"
               "  \"n_redundant\": %llu,\n  \"n_rate_limited\": %llu,\n  \"n_hyst_held\": %llu,\n"
               "  \"n_logical_moves\": %llu,\n  \"n_errors\": %llu,\n  \"last_errno\": %d,\n"
               "  \"n_cpu_mismatch\": %llu,\n",
            (unsigned long long)(now - G.t_init_ns),
            (unsigned long long)__atomic_load_n(&G.n_setwin, __ATOMIC_RELAXED),
            (unsigned long long)__atomic_load_n(&G.n_region, __ATOMIC_RELAXED),
            (unsigned long long)G.n_redundant,
            (unsigned long long)G.n_rate_limited, (unsigned long long)G.n_hyst_held,
            (unsigned long long)G.n_logical_moves, (unsigned long long)G.n_errors, G.last_errno,
            (unsigned long long)G.n_mismatch);
    fprintf(f, "  \"migrations\": [");
    int first = 1;
    uint64_t tot_n = 0, tot_ns = 0;
    for (int i = 0; i < NSIDE; i++)
        for (int j = 0; j < NSIDE; j++) {
            mstat_t *m = &G.mig[i][j];
            if (!m->n) continue;
            tot_n += m->n;
            tot_ns += m->sum_ns;
            fprintf(f, "%s\n    {\"from\": \"%s\", \"to\": \"%s\", \"n\": %llu, \"mean_ns\": %.1f, "
                       "\"min_ns\": %llu, \"max_ns\": %llu}",
                    first ? "" : ",", wh_side_name(i), wh_side_name(j), (unsigned long long)m->n,
                    (double)m->sum_ns / (double)m->n, (unsigned long long)m->min_ns,
                    (unsigned long long)m->max_ns);
            first = 0;
        }
    fprintf(f, "%s],\n  \"n_migrations\": %llu,\n  \"migration_total_ns\": %llu,\n",
            first ? "" : "\n  ", (unsigned long long)tot_n, (unsigned long long)tot_ns);
    fprintf(f, "  \"perf\": %s,\n  \"perf_msg\": \"%s\",\n  \"rapl\": %s,\n  \"rapl_msg\": \"%s\",\n",
            G.perf_on ? "true" : "false", G.perf.msg, G.rapl_on ? "true" : "false", G.rapl.msg);
    if (G.rapl_on) {
        uint64_t p, c;
        wh_rapl_read(&G.rapl, &p, &c);
        fprintf(f, "  \"energy_pkg_uj\": %llu,\n  \"energy_core_uj\": %llu,\n",
                (unsigned long long)wh_rapl_delta(&G.rapl, G.rp_init, p, 0),
                (unsigned long long)wh_rapl_delta(&G.rapl, G.rc_init, c, 1));
    }
    fprintf(f, "  \"sondag\": [");
    first = 1;
    for (int t = 0; t < NREG; t++) {
        if (!G.s_n[t][WH_SIDE_P] && !G.s_n[t][WH_SIDE_E]) continue;
        fprintf(f, "%s\n    {\"type\": %d, \"samples_p\": %llu, \"samples_e\": %llu, "
                   "\"e_over_p\": %.4f, \"assigned\": \"%s\"}",
                first ? "" : ",", t, (unsigned long long)G.s_n[t][WH_SIDE_P],
                (unsigned long long)G.s_n[t][WH_SIDE_E], G.s_ratio[t],
                G.assigned[t] < 0 ? "undecided" : wh_side_name(G.assigned[t]));
        first = 0;
    }
    fprintf(f, "%s]\n}\n", first ? "" : "\n  ");
    fclose(f);
}

/** @brief See winhint.h: winhint_flush(). */
void winhint_flush(void) {
    lock();
    if (G.inited && G.mode != M_OFF) close_segment(wh_now_ns());
    write_outputs();
    if (G.trace) fflush(G.trace);
    unlock();
}

/** @brief Load-time constructor: initialize and apply the initial placement. */
__attribute__((constructor)) static void winhint_ctor(void) {
    lock();
    init();
    if (!initial_applied) apply_initial(wh_now_ns());
    unlock();
}

/** @brief Exit-time destructor: flush outputs once and release perf/RAPL/trace resources. */
__attribute__((destructor)) static void winhint_dtor(void) {
    if (G.flushed) return;
    G.flushed = 1;
    winhint_flush();
    if (G.trace) { fclose(G.trace); G.trace = NULL; }
    if (G.perf_on) wh_perf_close(&G.perf);
    if (G.rapl_on) wh_rapl_close(&G.rapl);
}
