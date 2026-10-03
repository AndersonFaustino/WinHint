/**
 * @file whutil.h
 * @brief Shared helpers for the WinHint real-hardware runtime (hw/).
 *
 * Hybrid topology detection (P/E cpusets), hybrid-aware perf_event_open
 * counters (one event per core-type PMU: cpu_core / cpu_atom), RAPL energy
 * (powercap sysfs, with a perf "power" PMU fallback) and timing.
 *
 * Used by libwinhint, bench_migration and the R4 PIE daemon.
 */
#ifndef WINHINT_WHUTIL_H
#define WINHINT_WHUTIL_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------- time */
/**
 * @brief Current CLOCK_MONOTONIC time.
 * @return Nanoseconds since an unspecified monotonic epoch.
 */
static inline uint64_t wh_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* ---------------------------------------------------------------- cpusets */
/**
 * @brief Parse a Linux cpulist ("0-3,8,10-11"). Returns #cpus, or -1 on error.
 *
 * Separators may be ',', ' ' or '\n'. Rejects non-digits, reversed ranges and
 * CPUs >= CPU_SETSIZE.
 *
 * @param[in]  s   NUL-terminated cpulist string (NULL is an error).
 * @param[out] out Cleared, then filled with the listed CPUs (partially filled on error).
 * @return Number of CPUs in @p out, or -1 on a parse error.
 */
int wh_parse_cpulist(const char *s, cpu_set_t *out);
/**
 * @brief Read a cpulist from a sysfs file. Returns #cpus, or -1.
 * @param[in]  path File to read (at most 1023 bytes are used).
 * @param[out] out  Cleared, then filled with the parsed CPUs.
 * @return Number of CPUs, or -1 if the file cannot be read or parsed.
 */
int wh_read_cpulist_file(const char *path, cpu_set_t *out);
/**
 * @brief Format a cpuset as a cpulist into buf (always NUL-terminated).
 *
 * Consecutive CPUs are collapsed to ranges ("0-3,8"). Output is truncated at
 * the last range that fits.
 *
 * @param[in]  s   Set to format.
 * @param[out] buf Destination buffer.
 * @param[in]  n   Size of @p buf in bytes (must be > 0).
 * @return @p buf.
 */
const char *wh_cpuset_str(const cpu_set_t *s, char *buf, size_t n);
/**
 * @brief First CPU in the set or -1.
 * @param[in] s Set to scan.
 * @return Lowest CPU number in @p s, or -1 if the set is empty.
 */
int wh_cpuset_first(const cpu_set_t *s);

/** @name Placement sides
 *  Logical placement targets used throughout hw/ (and as array indices 0..2).
 *  @{ */
#define WH_SIDE_ORIG 0   /**< original/inherited affinity ("let the OS decide") */
#define WH_SIDE_P    1   /**< Performance (P) core set. */
#define WH_SIDE_E    2   /**< Efficiency (E) core set. */
/** @} */
/**
 * @brief Short name of a placement side.
 * @param[in] side WH_SIDE_ORIG, WH_SIDE_P or WH_SIDE_E.
 * @return "orig", "P", "E", or "?" for any other value (static string).
 */
const char *wh_side_name(int side);

/** @brief Detected hybrid CPU topology (filled by wh_topo_detect()). */
typedef struct {
    cpu_set_t online;       ///< Online CPUs (/sys/devices/system/cpu/online, else 0.._SC_NPROCESSORS_ONLN-1).
    cpu_set_t p;            /**< performance cores (cpu_core) ∩ online */
    cpu_set_t e;            /**< efficiency cores  (cpu_atom) ∩ online */
    int n_p, n_e;           ///< CPU_COUNT of @c p and @c e.
    int hybrid;             /**< both sets non-empty */
    int pmu_type_p;         /**< /sys/devices/cpu_core/type, -1 if absent */
    int pmu_type_e;         /**< /sys/devices/cpu_atom/type, -1 if absent */
    long fmax_p_khz, fmax_e_khz; ///< cpuinfo_max_freq (kHz) of the first P / E CPU, 0 if unknown.
    int smt_on;             /**< /sys/devices/system/cpu/smt/active */
    char src[128];          /**< where the sets came from (sysfs / env) */
} wh_topo;

/**
 * @brief Detect the hybrid P/E topology.
 *
 * Detect the topology. Env overrides (cpulists):
 *   WINHINT_PCPUS, WINHINT_ECPUS
 * Otherwise the sets come from /sys/devices/cpu_core/cpus and
 * /sys/devices/cpu_atom/cpus. Both sets are intersected with the online CPUs
 * (so SMT siblings drop out when SMT is off). PMU types, max frequencies and
 * the SMT state are filled in best-effort.
 *
 * @param[out] t Zeroed, then filled (also on failure).
 * @return 0 on success, -1 if no usable P and E sets were found.
 */
int wh_topo_detect(wh_topo *t);

/**
 * @brief Which side a cpu belongs to (WH_SIDE_P / WH_SIDE_E / -1).
 * @param[in] t   Topology from wh_topo_detect().
 * @param[in] cpu CPU number.
 * @return WH_SIDE_P, WH_SIDE_E, or -1 if @p cpu is in neither set or out of range.
 */
int wh_topo_side_of_cpu(const wh_topo *t, int cpu);

/* ---------------------------------------------------------------- perf */
#define WH_EV_MAX   6      /**< Maximum events per wh_perf (extra events are dropped). */
#define WH_PMU_MAX  2      /**< index 0 = P (cpu_core), 1 = E (cpu_atom) */

/** @brief Description of one perf event to open with wh_perf_open(). */
typedef struct {
    uint64_t type;         /**< PERF_TYPE_HARDWARE / HW_CACHE / RAW */
    uint64_t config;       /**< without the hybrid PMU-type bits */
    int pmu_only;          /**< -1: all PMUs; 0: P PMU only; 1: E PMU only */
    const char *name;      ///< Label used in error messages (may be NULL).
} wh_evspec;

/** @brief A set of open perf events, one fd per (core-type PMU, event). */
typedef struct {
    int fd[WH_PMU_MAX][WH_EV_MAX]; ///< Event fds [pmu][event]; -1 if not opened.
    int npmu;              ///< 2 if both hybrid PMU types are known, else 1 (generic PMU).
    int nev;               ///< Number of events (<= WH_EV_MAX).
    wh_evspec ev[WH_EV_MAX]; ///< Copy of the requested event specs.
    int ok;                ///< Non-zero if at least one event opened on some PMU.
    int err;               /**< errno of the first failure */
    char msg[256];         ///< Human-readable reason for the first failure (the error when ok == 0).
} wh_perf;

/** @brief Counter snapshot from wh_perf_read(), indexed [pmu][event]. */
typedef struct {
    uint64_t v[WH_PMU_MAX][WH_EV_MAX];         ///< Raw counts (not scaled for multiplexing).
    uint64_t running[WH_PMU_MAX][WH_EV_MAX];   /**< ns the event was on a PMU */
} wh_perf_vals;

/** @name wh_perf_open() flags
 *  @{ */
#define WH_PERF_INHERIT         1   /**< count child threads/processes too */
#define WH_PERF_ENABLE_ON_EXEC  2   /**< start disabled, enable at exec() */
#define WH_PERF_WITH_KERNEL     4   /**< do not exclude kernel mode */
/** @} */
/**
 * @brief Open @p nev counting events on a task, one per core-type PMU.
 *
 * Open nev events on `pid` (0 = self, current thread), any CPU, user-space only
 * (unless WH_PERF_WITH_KERNEL). On a hybrid machine each event is opened once per
 * core-type PMU (the extended PERF_TYPE_HARDWARE encoding, or the PMU type itself
 * for PERF_TYPE_RAW), so counts follow migrations. Events that fail on one PMU
 * (e.g. P-only raw events) get fd = -1 and read as 0. Hypervisor mode is always
 * excluded.
 *
 * @param[out] p     Zeroed, then filled; close with wh_perf_close().
 * @param[in]  pid   Target task (0 = calling thread).
 * @param[in]  t     Topology providing the PMU types; NULL or missing types = one generic PMU.
 * @param[in]  ev    Event specs (copied).
 * @param[in]  nev   Number of specs; clamped to WH_EV_MAX.
 * @param[in]  flags Bitwise OR of WH_PERF_INHERIT, WH_PERF_ENABLE_ON_EXEC, WH_PERF_WITH_KERNEL.
 * @return 0 if at least one event opened on a PMU it was requested for (so an
 *         E-only event counts on a hybrid system), else -1 (reason for the first
 *         failure in p->err / p->msg).
 */
int  wh_perf_open(wh_perf *p, pid_t pid, const wh_topo *t,
                  const wh_evspec *ev, int nev, int flags);
/**
 * @brief Read all open events.
 * @param[in]  p   Events from wh_perf_open().
 * @param[out] out Zeroed, then filled; unopened or failed reads stay 0.
 * @return Always 0.
 */
int  wh_perf_read(const wh_perf *p, wh_perf_vals *out);
/**
 * @brief Close all event fds and clear @c ok.
 * @param[in,out] p Events from wh_perf_open().
 */
void wh_perf_close(wh_perf *p);
/**
 * @brief Human-readable explanation of a perf_event_open failure.
 * @param[in] err errno value from perf_event_open.
 * @return Static message string.
 */
const char *wh_perf_errhint(int err);

/* ---------------------------------------------------------------- RAPL */
/** @brief RAPL energy reader (package-0 and its core domain). */
typedef struct {
    int backend;            /**< 0 none, 1 sysfs powercap, 2 perf power PMU */
    int fd_pkg, fd_core;    ///< Package / core energy fds; -1 if not open (core is optional).
    uint64_t max_pkg, max_core;  /**< wrap range (uJ), sysfs only */
    double scale_pkg, scale_core;/**< Joules per count, perf only */
    char msg[512];          ///< Reason why opening failed.
} wh_rapl;

/**
 * @brief Open package + core energy. WINHINT_RAPL_BACKEND=sysfs|perf|auto (auto).
 *
 * `auto` tries the powercap sysfs files first (intel-rapl package-0 and
 * intel-rapl:0:* "core"), then the perf `power` PMU (system-wide event on the
 * first online CPU, needs CAP_PERFMON/root).
 *
 * @param[out] r Zeroed, then filled; close with wh_rapl_close().
 * @return 0 if at least the package domain opened, else -1 (reason in r->msg).
 */
int  wh_rapl_open(wh_rapl *r);
/**
 * @brief Monotonic readings in microjoules (wrap handled by wh_rapl_delta).
 * @param[in]  r       Reader from wh_rapl_open().
 * @param[out] pkg_uj  Package energy (uJ).
 * @param[out] core_uj Core energy (uJ); 0 if the core domain is unavailable.
 * @return 0 on success, -1 if the package read failed or no backend is open.
 */
int  wh_rapl_read(const wh_rapl *r, uint64_t *pkg_uj, uint64_t *core_uj);
/**
 * @brief Energy between two readings, handling a sysfs counter wrap.
 * @param[in] r    Reader the readings came from.
 * @param[in] a    Earlier reading (uJ).
 * @param[in] b    Later reading (uJ).
 * @param[in] core Non-zero for the core domain (selects max_core), 0 for package.
 * @return @p b - @p a, or the wrapped difference (max - @p a + @p b + 1, since the sysfs
 *         counter ranges over [0, max_energy_range_uj]); 0 if @p b < @p a and the wrap
 *         range is unknown.
 */
uint64_t wh_rapl_delta(const wh_rapl *r, uint64_t a, uint64_t b, int core);
/**
 * @brief Close the energy fds and reset the backend to none.
 * @param[in,out] r Reader from a successful wh_rapl_open().
 */
void wh_rapl_close(wh_rapl *r);

/* ---------------------------------------------------------------- misc */
/**
 * @brief Integer environment variable (strtol base 0: decimal, 0x hex, 0 octal).
 * @param[in] name Variable name.
 * @param[in] dflt Value if unset, empty or not a number.
 * @return Parsed value or @p dflt.
 */
long wh_env_long(const char *name, long dflt);
/**
 * @brief Floating-point environment variable (strtod).
 * @param[in] name Variable name.
 * @param[in] dflt Value if unset, empty or not a number.
 * @return Parsed value or @p dflt.
 */
double wh_env_double(const char *name, double dflt);

#ifdef __cplusplus
}
#endif
#endif
