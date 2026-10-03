/**
 * @file whutil.c
 * @brief Implementation of the shared hw/ helpers declared in whutil.h.
 *
 * Small sysfs readers, environment parsing, cpulist handling, hybrid topology
 * detection, per-core-type perf counters and RAPL energy readers. Public
 * functions are documented in whutil.h.
 */
#define _GNU_SOURCE
#include "whutil.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

/* ------------------------------------------------------------ small io */
/**
 * @brief Read a small text file and strip trailing newlines/spaces.
 * @param[in]  path File to read (one read() call).
 * @param[out] buf  Destination, NUL-terminated on success.
 * @param[in]  n    Size of @p buf; at most n-1 bytes are read.
 * @return Length of the stripped content, or -1 (errno set) on open/read failure.
 */
static int read_small_file(const char *path, char *buf, size_t n) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    ssize_t k = read(fd, buf, n - 1);
    int e = errno;
    close(fd);
    if (k < 0) { errno = e; return -1; }
    buf[k] = 0;
    while (k > 0 && (buf[k - 1] == '\n' || buf[k - 1] == ' ')) buf[--k] = 0;
    return (int)k;
}

/**
 * @brief Read an integer (strtol base 0) from a small file.
 * @param[in] path File to read.
 * @param[in] dflt Value if the file is missing, unreadable or empty.
 * @return Parsed value or @p dflt.
 */
static long read_long_file(const char *path, long dflt) {
    char b[64];
    if (read_small_file(path, b, sizeof b) <= 0) return dflt;
    return strtol(b, NULL, 0);
}

/** @brief See whutil.h: wh_env_long(). */
long wh_env_long(const char *name, long dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    char *end;
    long v = strtol(s, &end, 0);
    return end == s ? dflt : v;
}

/** @brief See whutil.h: wh_env_double(). */
double wh_env_double(const char *name, double dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    char *end;
    double v = strtod(s, &end);
    return end == s ? dflt : v;
}

/* ------------------------------------------------------------ cpusets */
/** @brief See whutil.h: wh_parse_cpulist(). */
int wh_parse_cpulist(const char *s, cpu_set_t *out) {
    CPU_ZERO(out);
    if (!s) return -1;
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == ',' || *p == '\n') p++;
        if (!*p) break;
        if (!isdigit((unsigned char)*p)) return -1;
        char *end;
        long a = strtol(p, &end, 10), b = a;
        p = end;
        if (*p == '-') {
            p++;
            if (!isdigit((unsigned char)*p)) return -1;
            b = strtol(p, &end, 10);
            p = end;
        }
        if (a < 0 || b < a || b >= CPU_SETSIZE) return -1;
        for (long c = a; c <= b; c++) CPU_SET((int)c, out);
        if (*p && *p != ',' && *p != '\n' && *p != ' ') return -1;
    }
    return CPU_COUNT(out);
}

/** @brief See whutil.h: wh_read_cpulist_file(). */
int wh_read_cpulist_file(const char *path, cpu_set_t *out) {
    char buf[1024];
    CPU_ZERO(out);
    if (read_small_file(path, buf, sizeof buf) < 0) return -1;
    return wh_parse_cpulist(buf, out);
}

/** @brief See whutil.h: wh_cpuset_str(). */
const char *wh_cpuset_str(const cpu_set_t *s, char *buf, size_t n) {
    size_t off = 0;
    buf[0] = 0;
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if (!CPU_ISSET(c, s)) continue;
        int d = c;
        while (d + 1 < CPU_SETSIZE && CPU_ISSET(d + 1, s)) d++;
        int k = (d == c) ? snprintf(buf + off, n - off, "%s%d", off ? "," : "", c)
                         : snprintf(buf + off, n - off, "%s%d-%d", off ? "," : "", c, d);
        if (k < 0 || (size_t)k >= n - off) break;
        off += (size_t)k;
        c = d;
    }
    return buf;
}

/** @brief See whutil.h: wh_cpuset_first(). */
int wh_cpuset_first(const cpu_set_t *s) {
    for (int c = 0; c < CPU_SETSIZE; c++)
        if (CPU_ISSET(c, s)) return c;
    return -1;
}

/** @brief See whutil.h: wh_side_name(). */
const char *wh_side_name(int side) {
    switch (side) {
    case WH_SIDE_P: return "P";
    case WH_SIDE_E: return "E";
    case WH_SIDE_ORIG: return "orig";
    default: return "?";
    }
}

/* ------------------------------------------------------------ topology */
/** @brief See whutil.h: wh_topo_detect(). */
int wh_topo_detect(wh_topo *t) {
    memset(t, 0, sizeof *t);
    if (wh_read_cpulist_file("/sys/devices/system/cpu/online", &t->online) <= 0) {
        CPU_ZERO(&t->online);
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        for (long c = 0; c < n && c < CPU_SETSIZE; c++) CPU_SET((int)c, &t->online);
    }
    const char *ep = getenv("WINHINT_PCPUS"), *ee = getenv("WINHINT_ECPUS");
    int np = -1, ne = -1;
    if (ep && *ep) np = wh_parse_cpulist(ep, &t->p);
    else np = wh_read_cpulist_file("/sys/devices/cpu_core/cpus", &t->p);
    if (ee && *ee) ne = wh_parse_cpulist(ee, &t->e);
    else ne = wh_read_cpulist_file("/sys/devices/cpu_atom/cpus", &t->e);
    snprintf(t->src, sizeof t->src, "P:%s E:%s",
             (ep && *ep) ? "env" : "sysfs", (ee && *ee) ? "env" : "sysfs");
    if (np < 0) CPU_ZERO(&t->p);
    if (ne < 0) CPU_ZERO(&t->e);
    CPU_AND(&t->p, &t->p, &t->online);   /* SMT off => siblings are offline */
    CPU_AND(&t->e, &t->e, &t->online);
    t->n_p = CPU_COUNT(&t->p);
    t->n_e = CPU_COUNT(&t->e);
    t->hybrid = t->n_p > 0 && t->n_e > 0;
    t->pmu_type_p = (int)read_long_file("/sys/devices/cpu_core/type", -1);
    t->pmu_type_e = (int)read_long_file("/sys/devices/cpu_atom/type", -1);
    char path[128];
    int cp = wh_cpuset_first(&t->p), ce = wh_cpuset_first(&t->e);
    if (cp >= 0) {
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cp);
        t->fmax_p_khz = read_long_file(path, 0);
    }
    if (ce >= 0) {
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", ce);
        t->fmax_e_khz = read_long_file(path, 0);
    }
    t->smt_on = (int)read_long_file("/sys/devices/system/cpu/smt/active", -1);
    return t->hybrid ? 0 : -1;
}

/** @brief See whutil.h: wh_topo_side_of_cpu(). */
int wh_topo_side_of_cpu(const wh_topo *t, int cpu) {
    if (cpu < 0 || cpu >= CPU_SETSIZE) return -1;
    if (CPU_ISSET(cpu, &t->p)) return WH_SIDE_P;
    if (CPU_ISSET(cpu, &t->e)) return WH_SIDE_E;
    return -1;
}

/* ------------------------------------------------------------ perf */
/**
 * @brief Raw perf_event_open(2) syscall wrapper (glibc has none).
 * @param[in] a     Event attributes.
 * @param[in] pid   Target task (-1 = any, 0 = self).
 * @param[in] cpu   CPU (-1 = any).
 * @param[in] group Group leader fd (-1 = none).
 * @param[in] flags PERF_FLAG_* flags.
 * @return New fd, or -1 with errno set.
 */
static long sys_perf_event_open(struct perf_event_attr *a, pid_t pid, int cpu,
                                int group, unsigned long flags) {
    return syscall(SYS_perf_event_open, a, pid, cpu, group, flags);
}

/** @brief See whutil.h: wh_perf_errhint(). */
const char *wh_perf_errhint(int err) {
    switch (err) {
    case EACCES:
    case EPERM:
        return "perf_event_open denied (check /proc/sys/kernel/perf_event_paranoid: "
               "<= 2 for own-process user-mode counters, see docs/guide/hardware/index.md)";
    case ENOENT:
    case EINVAL:
        return "event not supported by this PMU";
    case ENOSYS:
        return "perf_event_open not available (seccomp or kernel config)";
    default:
        return "perf_event_open failed";
    }
}

/** @brief See whutil.h: wh_perf_open(). */
int wh_perf_open(wh_perf *p, pid_t pid, const wh_topo *t,
                 const wh_evspec *ev, int nev, int flags) {
    memset(p, 0, sizeof *p);
    for (int i = 0; i < WH_PMU_MAX; i++)
        for (int j = 0; j < WH_EV_MAX; j++) p->fd[i][j] = -1;
    if (nev > WH_EV_MAX) nev = WH_EV_MAX;
    p->nev = nev;
    memcpy(p->ev, ev, sizeof(wh_evspec) * (size_t)nev);
    int pmu_types[WH_PMU_MAX] = {-1, -1};
    if (t && t->pmu_type_p >= 0 && t->pmu_type_e >= 0) {
        p->npmu = 2;
        pmu_types[0] = t->pmu_type_p;
        pmu_types[1] = t->pmu_type_e;
    } else {
        p->npmu = 1;
    }
    for (int i = 0; i < p->npmu; i++) {
        for (int j = 0; j < nev; j++) {
            if (ev[j].pmu_only >= 0 && ev[j].pmu_only != i && p->npmu > 1) continue;
            struct perf_event_attr a;
            memset(&a, 0, sizeof a);
            a.size = sizeof a;
            a.type = (uint32_t)ev[j].type;
            a.config = ev[j].config;
            if (pmu_types[i] >= 0) {
                if (ev[j].type == PERF_TYPE_RAW)
                    a.type = (uint32_t)pmu_types[i];
                else /* PERF_PMU_TYPE_SHIFT = 32: extended hybrid encoding */
                    a.config |= ((uint64_t)pmu_types[i]) << 32;
            }
            a.disabled = (flags & WH_PERF_ENABLE_ON_EXEC) ? 1 : 0;
            a.enable_on_exec = (flags & WH_PERF_ENABLE_ON_EXEC) ? 1 : 0;
            a.exclude_kernel = (flags & WH_PERF_WITH_KERNEL) ? 0 : 1;
            a.exclude_hv = 1;
            a.inherit = (flags & WH_PERF_INHERIT) ? 1 : 0;
            a.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
            long fd = sys_perf_event_open(&a, pid, -1, -1, PERF_FLAG_FD_CLOEXEC);
            if (fd < 0) {
                int e = errno;
                if (!p->msg[0]) {   /* remember the first failure */
                    p->err = e;
                    snprintf(p->msg, sizeof p->msg, "%s: %s (errno %d)",
                             ev[j].name ? ev[j].name : "event", wh_perf_errhint(e), e);
                }
                continue;
            }
            p->fd[i][j] = (int)fd;
        }
    }
    /* success = at least one event opened on a PMU it was requested for (an
     * E-only first event never has fd[0][0] on a hybrid system) */
    p->ok = 0;
    for (int i = 0; i < p->npmu && !p->ok; i++)
        for (int j = 0; j < nev; j++)
            if (p->fd[i][j] >= 0) { p->ok = 1; break; }
    if (!p->ok && !p->msg[0])
        snprintf(p->msg, sizeof p->msg, "no event opened");
    return p->ok ? 0 : -1;
}

/** @brief See whutil.h: wh_perf_read(). */
int wh_perf_read(const wh_perf *p, wh_perf_vals *out) {
    memset(out, 0, sizeof *out);
    for (int i = 0; i < p->npmu; i++)
        for (int j = 0; j < p->nev; j++) {
            if (p->fd[i][j] < 0) continue;
            uint64_t buf[3];
            if (read(p->fd[i][j], buf, sizeof buf) == (ssize_t)sizeof buf) {
                out->v[i][j] = buf[0];
                out->running[i][j] = buf[2];
            }
        }
    return 0;
}

/** @brief See whutil.h: wh_perf_close(). */
void wh_perf_close(wh_perf *p) {
    for (int i = 0; i < WH_PMU_MAX; i++)
        for (int j = 0; j < WH_EV_MAX; j++)
            if (p->fd[i][j] >= 0) { close(p->fd[i][j]); p->fd[i][j] = -1; }
    p->ok = 0;
}

/* ------------------------------------------------------------ RAPL */
/**
 * @brief Read a decimal unsigned integer from offset 0 of an open file.
 * @param[in]  fd File descriptor (e.g. a powercap energy_uj file).
 * @param[out] v  Parsed value.
 * @return 0 on success, -1 if nothing could be read.
 */
static int pread_u64(int fd, uint64_t *v) {
    char b[32];
    ssize_t k = pread(fd, b, sizeof b - 1, 0);
    if (k <= 0) return -1;
    b[k] = 0;
    *v = strtoull(b, NULL, 10);
    return 0;
}

/**
 * @brief Open the powercap sysfs RAPL backend.
 *
 * Scans /sys/class/powercap for the intel-rapl "package-0" domain and the
 * "core" subdomain of intel-rapl:0, opening their energy_uj files and reading
 * max_energy_range_uj as the wrap range.
 *
 * @param[in,out] r Reader with fd_pkg/fd_core preset to -1; on failure r->msg says why.
 * @return 0 if the package domain opened (backend = 1), else -1.
 */
static int rapl_open_sysfs(wh_rapl *r) {
    const char *base = "/sys/class/powercap";
    DIR *d = opendir(base);
    if (!d) {
        snprintf(r->msg, sizeof r->msg,
                 "%s not present (kernel without the intel_rapl powercap driver)", base);
        return -1;
    }
    struct dirent *de;
    char path[512], name[64];
    int denied = 0, masked = 0;
    while ((de = readdir(d))) {
        /* intel-rapl:N or intel-rapl:N:M (skip intel-rapl-mmio duplicates) */
        if (strncmp(de->d_name, "intel-rapl:", 11) != 0) continue;
        snprintf(path, sizeof path, "%s/%s/name", base, de->d_name);
        if (read_small_file(path, name, sizeof name) <= 0) {
            if (errno == ENOENT) masked = 1;   /* dangling symlink: hidden/masked sysfs */
            continue;
        }
        int is_pkg = strncmp(name, "package-0", 9) == 0;
        int is_core = strcmp(name, "core") == 0 && strchr(de->d_name + 11, ':') != NULL;
        if (!is_pkg && !is_core) continue;
        if (is_core && strncmp(de->d_name, "intel-rapl:0:", 13) != 0) continue;
        snprintf(path, sizeof path, "%s/%s/energy_uj", base, de->d_name);
        int fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) { if (errno == EACCES || errno == EPERM) denied = 1; continue; }
        snprintf(path, sizeof path, "%s/%s/max_energy_range_uj", base, de->d_name);
        uint64_t mx = (uint64_t)read_long_file(path, 0);
        if (is_pkg) { r->fd_pkg = fd; r->max_pkg = mx; }
        else { r->fd_core = fd; r->max_core = mx; }
    }
    closedir(d);
    if (r->fd_pkg < 0) {
        snprintf(r->msg, sizeof r->msg, "%s",
                 denied ? "RAPL energy_uj is root-only (0400) since CVE-2020-8694: run as "
                          "root (sudo), or opt in to a read grant (docs/guide/hardware/index.md, "
                          "'RAPL access'), or use the perf power PMU backend"
                 : masked ? "powercap energy files are not reachable (masked sysfs)"
                          : "no intel-rapl package domain found in /sys/class/powercap");
        if (r->fd_core >= 0) { close(r->fd_core); r->fd_core = -1; }
        return -1;
    }
    r->backend = 1;
    return 0;
}

/**
 * @brief Read a floating-point value (strtod) from a small file.
 * @param[in] path File to read.
 * @param[in] dflt Value if the file is missing, unreadable or empty.
 * @return Parsed value or @p dflt.
 */
static double read_double_file(const char *path, double dflt) {
    char b[64];
    if (read_small_file(path, b, sizeof b) <= 0) return dflt;
    return strtod(b, NULL);
}

/**
 * @brief Open one perf `power` PMU energy event system-wide on the first online CPU.
 * @param[in]  ev    Event name under /sys/bus/event_source/devices/power/events/
 *                   (e.g. "energy-pkg", "energy-cores").
 * @param[out] scale Joules per count from `<ev>.scale` (default 2^-32).
 * @return Event fd, or -1 with errno set (ENOENT if the PMU/event is absent).
 */
static int rapl_open_perf_event(const char *ev, double *scale) {
    char path[256], buf[128];
    long type = read_long_file("/sys/bus/event_source/devices/power/type", -1);
    if (type < 0) { errno = ENOENT; return -1; }
    snprintf(path, sizeof path, "/sys/bus/event_source/devices/power/events/%s", ev);
    if (read_small_file(path, buf, sizeof buf) <= 0) { errno = ENOENT; return -1; }
    char *q = strstr(buf, "event=");
    if (!q) { errno = EINVAL; return -1; }
    uint64_t cfg = strtoull(q + 6, NULL, 0);
    snprintf(path, sizeof path, "/sys/bus/event_source/devices/power/events/%s.scale", ev);
    *scale = read_double_file(path, 2.3283064365386962890625e-10);
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a;
    a.type = (uint32_t)type;
    a.config = cfg;
    cpu_set_t on;
    int cpu = 0;
    if (wh_read_cpulist_file("/sys/devices/system/cpu/online", &on) > 0) cpu = wh_cpuset_first(&on);
    long fd = sys_perf_event_open(&a, -1, cpu, -1, PERF_FLAG_FD_CLOEXEC);
    return (int)fd;
}

/** @brief See whutil.h: wh_rapl_open(). */
int wh_rapl_open(wh_rapl *r) {
    memset(r, 0, sizeof *r);
    r->fd_pkg = r->fd_core = -1;
    const char *be = getenv("WINHINT_RAPL_BACKEND");
    int want_sysfs = !be || !*be || !strcmp(be, "auto") || !strcmp(be, "sysfs");
    int want_perf = !be || !*be || !strcmp(be, "auto") || !strcmp(be, "perf");
    char m1[sizeof r->msg] = "";
    if (want_sysfs) {
        if (rapl_open_sysfs(r) == 0) return 0;
        snprintf(m1, sizeof m1, "%s", r->msg);
    }
    if (want_perf) {
        int fd = rapl_open_perf_event("energy-pkg", &r->scale_pkg);
        if (fd >= 0) {
            r->fd_pkg = fd;
            r->fd_core = rapl_open_perf_event("energy-cores", &r->scale_core);
            r->backend = 2;
            r->msg[0] = 0;
            return 0;
        }
        int e = errno;
        snprintf(r->msg, sizeof r->msg, "sysfs: %.300s; perf power PMU: %.150s", m1[0] ? m1 : "skipped",
                 (e == EACCES || e == EPERM)
                     ? "denied (system-wide events need CAP_PERFMON with perf_event_paranoid>0)"
                     : strerror(e));
    }
    return -1;
}

/** @brief See whutil.h: wh_rapl_read(). */
int wh_rapl_read(const wh_rapl *r, uint64_t *pkg_uj, uint64_t *core_uj) {
    *pkg_uj = *core_uj = 0;
    if (r->backend == 1) {
        if (pread_u64(r->fd_pkg, pkg_uj)) return -1;
        if (r->fd_core >= 0) pread_u64(r->fd_core, core_uj);
        return 0;
    }
    if (r->backend == 2) {
        uint64_t c;
        if (read(r->fd_pkg, &c, sizeof c) != (ssize_t)sizeof c) return -1;
        *pkg_uj = (uint64_t)((double)c * r->scale_pkg * 1e6);
        if (r->fd_core >= 0 && read(r->fd_core, &c, sizeof c) == (ssize_t)sizeof c)
            *core_uj = (uint64_t)((double)c * r->scale_core * 1e6);
        return 0;
    }
    return -1;
}

/** @brief See whutil.h: wh_rapl_delta(). */
uint64_t wh_rapl_delta(const wh_rapl *r, uint64_t a, uint64_t b, int core) {
    if (b >= a) return b - a;
    uint64_t mx = r->backend == 1 ? (core ? r->max_core : r->max_pkg) : 0;
    /* sysfs counter wrapped: energy_uj takes values in [0, max_energy_range_uj],
     * so it wraps modulo max_energy_range_uj + 1 (max -> 0 is one more uJ). */
    return mx ? (mx - a) + b + 1 : 0;
}

/** @brief See whutil.h: wh_rapl_close(). */
void wh_rapl_close(wh_rapl *r) {
    if (r->fd_pkg >= 0) close(r->fd_pkg);
    if (r->fd_core >= 0) close(r->fd_core);
    r->fd_pkg = r->fd_core = -1;
    r->backend = 0;
}
