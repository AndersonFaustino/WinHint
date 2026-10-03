/**
 * @file pie_daemon.c
 * @brief R4: reactive counter-driven P/E migration, PIE-style.
 *
 * Reimplementation (no public artifact) of the policy of
 *   K. Van Craeynest, A. Jaleel, L. Eeckhout, P. Narvaez, J. Emer,
 *   "Scheduling Heterogeneous Multi-Cores through Performance Impact
 *    Estimation (PIE)", ISCA 2012,
 * adapted to one single-threaded job on an Intel hybrid CPU with an energy
 * objective (see docs/guide/hardware/index.md, "R4 deviations").
 *
 * Every interval the daemon reads counters of the target on the core type it
 * is running on, splits CPI into a base and a memory component, and predicts
 * the CPI on the *other* core type (PIE eqs.: base CPI scales with the
 * width/ILP ratio, memory CPI scales with the MLP ratio, MLP estimated from
 * LLC misses per instruction times the ROB size, or measured on the P core
 * from L1D_PEND_MISS when available). It converts CPI into time with the
 * measured frequency, and runs the job on the E cores whenever the predicted
 * E/P time ratio is <= 1 + slack (else on the P cores). Migration uses
 * sched_setaffinity on every thread of the target.
 *
 * Backends:
 *   perf      perf_event_open on the target (default; testable)
 *   pmctrack  PMCTrack (github.com/jcsaezal/pmctrack, needs its kernel
 *             module): the `pmctrack` CLI is spawned in attach mode and its
 *             per-sample output is parsed. Command template: -C (see README).
 *
 * Usage:
 *   pie_daemon [opts] -- cmd [args...]      launch and manage cmd
 *   pie_daemon [opts] -p PID                manage an existing process
 * Options:
 *   -b perf|pmctrack  backend                  -i ms      interval (10)
 *   -s slack          E allowed if T_E/T_P <= 1+slack (0.15)
 *   -H n              hysteresis: consecutive agreeing intervals (2)
 *   -L ns             memory latency (110)       -I P|E|orig initial side (P)
 *   -o log.csv        per-interval log          -S sum.json summary
 *   -C template       pmctrack command template ("pmctrack -T %.3f -c %s -p %d")
 * Model parameters (env): PIE_ROB_P (512) PIE_ROB_E (256) PIE_WIDTH_P (6)
 *   PIE_WIDTH_E (5) PIE_BASE_RATIO (1.25: base CPI E/P) PIE_MLP_CAP (16)
 *   PIE_RAW_MLP (1; 0 = do not open the L1D_PEND_MISS raw events)
 *   PIE_PMCTRACK_EVENTS ("instr,cycles,llc_misses", pmctrack backend)
 *
 * Intervals with fewer than 1000 instructions are skipped. The CSV log (-o) has
 * one row per evaluated interval; the JSON summary (-S) has interval/migration
 * counts and the job's exit code. Exit status: the launched command's status
 * (0 in -p mode), 1 setup error / no hybrid topology, 2 bad usage, 5 backend
 * failed to open.
 */
#define _GNU_SOURCE
#include "whutil.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/wait.h>
#include <unistd.h>

/** @brief Counter deltas for one interval, taken on the core type the target ran on. */
typedef struct {
    double cyc, ins, llc, pend, pend_cyc; ///< Cycles, instructions, LLC misses, L1D_PEND_MISS.PENDING(_CYCLES) (P only, else 0).
    int side;          /**< core type the interval ran on */
    double run_ns;     /**< time on that core type */
} sample_t;

/** @brief Detected P/E topology. */
static wh_topo T;
/** @brief Managed process. */
static pid_t target;
/** @brief Set by SIGINT/SIGTERM to end the control loop. */
static volatile sig_atomic_t stop_flag;

/* ------------------------------------------------------------ perf backend */
/** @brief Indices of the perf-backend events (NEV = count). */
enum { EV_CYC, EV_INS, EV_LLC, EV_PEND, EV_PENDC, NEV };
/** @brief Perf-backend event set. */
static wh_perf P;
/** @brief Previous perf snapshot (for interval deltas). */
static wh_perf_vals last;

/**
 * @brief Open the perf backend on the target (inherited by its threads/children).
 *
 * Cycles, instructions and LLC misses on every core-type PMU, plus (unless
 * PIE_RAW_MLP=0) the P-core-only L1D_PEND_MISS raw events for measured MLP.
 *
 * @param[in] pid            Target process.
 * @param[in] enable_on_exec Non-zero to start counting at the target's exec().
 * @return 0 on success, -1 if the counters could not be opened.
 */
static int perf_backend_open(pid_t pid, int enable_on_exec) {
    wh_evspec ev[NEV] = {
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1, "cycles"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, -1, "instructions"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, -1, "LLC misses"},
        /* Golden/Raptor Cove: L1D_PEND_MISS.PENDING (0x48/0x01) and
         * L1D_PEND_MISS.PENDING_CYCLES (cmask=1): outstanding L1D misses -> MLP */
        {PERF_TYPE_RAW, 0x0148, 0, "L1D_PEND_MISS.PENDING"},
        {PERF_TYPE_RAW, 0x01000148, 0, "L1D_PEND_MISS.PENDING_CYCLES"},
    };
    int nev = wh_env_long("PIE_RAW_MLP", 1) ? NEV : EV_PEND;
    int fl = WH_PERF_INHERIT | (enable_on_exec ? WH_PERF_ENABLE_ON_EXEC : 0);
    if (wh_perf_open(&P, pid, &T, ev, nev, fl)) {
        fprintf(stderr, "pie_daemon: perf backend: %s\n", P.msg);
        return -1;
    }
    memset(&last, 0, sizeof last);
    return 0;
}

/**
 * @brief Read the perf counters and build the interval sample.
 *
 * The side is the PMU whose cycles event ran longer during the interval
 * (always P on a single-PMU system).
 *
 * @param[out] s Interval sample.
 * @return Always 0.
 */
static int perf_backend_sample(sample_t *s) {
    wh_perf_vals v;
    wh_perf_read(&P, &v);
    memset(s, 0, sizeof *s);
    double run[WH_PMU_MAX] = {0};
    for (int i = 0; i < P.npmu; i++)
        run[i] = (double)(v.running[i][EV_CYC] - last.running[i][EV_CYC]);
    int k = (P.npmu > 1 && run[1] > run[0]) ? 1 : 0;
    s->side = (P.npmu > 1) ? (k ? WH_SIDE_E : WH_SIDE_P) : WH_SIDE_P;
    s->run_ns = run[k];
/** @brief Interval delta of event @p e on PMU k (local helper). */
#define D(e) ((double)(v.v[k][e] - last.v[k][e]))
    s->cyc = D(EV_CYC);
    s->ins = D(EV_INS);
    s->llc = D(EV_LLC);
    if (k == 0 && P.fd[0][EV_PEND] >= 0 && P.fd[0][EV_PENDC] >= 0) {
        s->pend = D(EV_PEND);
        s->pend_cyc = D(EV_PENDC);
    }
#undef D
    last = v;
    return 0;
}

/* ------------------------------------------------------------ pmctrack backend */
/** @brief stdout of the spawned pmctrack process. */
static FILE *pmc_pipe;
/** @brief pid of the spawned pmctrack shell (0 if none). */
static pid_t pmc_pid;

/**
 * @brief CPU a task last ran on (field 39 of /proc/PID/stat).
 * @param[in] pid Task id.
 * @return CPU number, or -1 if unavailable.
 */
static int read_task_cpu(pid_t pid) {
    char path[64], buf[2048];
    snprintf(path, sizeof path, "/proc/%d/stat", pid);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    char *p = strrchr(buf, ')');
    if (!p) return -1;
    int field = 2, cpu = -1;
    for (char *tok = strtok(p + 1, " "); tok; tok = strtok(NULL, " ")) {
        field++;
        if (field == 39) { cpu = atoi(tok); break; }
    }
    return cpu;
}

/**
 * @brief Spawn `pmctrack` (via /bin/sh) in attach mode and open a pipe to its output.
 * @param[in] pid        Target process.
 * @param[in] tmpl       printf template taking (interval_s, events, pid).
 * @param[in] interval_s Sampling interval in seconds.
 * @return 0 if the pipe was opened, -1 otherwise.
 */
static int pmctrack_backend_open(pid_t pid, const char *tmpl, double interval_s) {
    char cmd[512];
    const char *events = getenv("PIE_PMCTRACK_EVENTS");
    if (!events) events = "instr,cycles,llc_misses";
    snprintf(cmd, sizeof cmd, tmpl, interval_s, events, (int)pid);
    int fds[2];
    if (pipe(fds)) return -1;
    pmc_pid = fork();
    if (pmc_pid == 0) {
        dup2(fds[1], 1);
        close(fds[0]);
        close(fds[1]);
        execl("/bin/sh", "sh", "-c", cmd, (char *)NULL);
        _exit(127);
    }
    close(fds[1]);
    pmc_pipe = fdopen(fds[0], "r");
    if (access("/proc/pmc", F_OK) != 0)
        fprintf(stderr, "pie_daemon: warning: /proc/pmc missing -- is the PMCTrack kernel module loaded?\n");
    fprintf(stderr, "pie_daemon: pmctrack backend: %s\n", cmd);
    return pmc_pipe ? 0 : -1;
}

/**
 * @brief Read the next pmctrack sample.
 *
 * Parses `pmctrack` sample lines: "nsample pid event pmc0 pmc1 pmc2 ..." where the
 * pmc columns follow PIE_PMCTRACK_EVENTS order (instr, cycles, llc_misses).
 * Header and non-numeric lines are skipped. Blocks until one sample arrives.
 * The side comes from the target's current CPU (P if unknown); run_ns is 0.
 *
 * @param[out] s Interval sample.
 * @return 0 on a sample, -1 at end of the pmctrack output.
 */
static int pmctrack_backend_sample(sample_t *s) {
    char line[1024];
    memset(s, 0, sizeof *s);
    while (fgets(line, sizeof line, pmc_pipe)) {
        char *tok[32];
        int n = 0;
        for (char *t = strtok(line, " \t\n"); t && n < 32; t = strtok(NULL, " \t\n")) tok[n++] = t;
        if (n < 6) continue;
        char *end;
        strtod(tok[0], &end);
        if (*end) continue;                      /* header */
        double v[3];
        int ok = 1;
        for (int i = 0; i < 3; i++) {
            v[i] = strtod(tok[3 + i], &end);
            if (*end) ok = 0;
        }
        if (!ok) continue;
        s->ins = v[0];
        s->cyc = v[1];
        s->llc = v[2];
        int cpu = read_task_cpu(target);
        s->side = wh_topo_side_of_cpu(&T, cpu);
        if (s->side < 0) s->side = WH_SIDE_P;
        s->run_ns = 0;                           /* frequency falls back to fmax */
        return 0;
    }
    return -1;
}

/* ------------------------------------------------------------ migration */
/**
 * @brief Set the affinity of every thread of a process (/proc/PID/task).
 * @param[in] pid Process id (falls back to the pid alone if its task dir is unreadable).
 * @param[in] m   New affinity mask.
 * @return 0 if every call succeeded, else -1.
 */
static int migrate_all(pid_t pid, const cpu_set_t *m) {
    char path[64];
    snprintf(path, sizeof path, "/proc/%d/task", pid);
    DIR *d = opendir(path);
    if (!d) return sched_setaffinity(pid, sizeof *m, m);
    struct dirent *de;
    int rc = 0;
    while ((de = readdir(d)))
        if (de->d_name[0] != '.' && sched_setaffinity(atoi(de->d_name), sizeof *m, m)) rc = -1;
    closedir(d);
    return rc;
}

/** @brief SIGINT/SIGTERM handler: request a stop. */
static void on_sig(int s) { (void)s; stop_flag = 1; }

/* ------------------------------------------------------------ model */
/** @brief PIE prediction for one interval ("cur" = core type measured on, "oth" = the other).
 *
 * CPI, time per instruction (ns), E/P time ratio, LLC MPKI, MLP estimates and the
 * measured (or fmax) frequency in GHz.
 */
typedef struct { double cpi_cur, cpi_oth, t_cur, t_oth, ratio_e_over_p, mpki, mlp_cur, mlp_oth, f_cur; } pred_t;

/**
 * @brief Predict the CPI and time per instruction on the other core type (PIE model).
 *
 * Splits the measured CPI into a memory component (MPI * latency * f / MLP, capped
 * at 95% of CPI) and a base component; scales base CPI by PIE_BASE_RATIO (bounded
 * below by 1/width) and memory CPI by the MLP ratio; MLP = clamp(MPI * ROB, 1,
 * PIE_MLP_CAP), or measured from L1D_PEND_MISS on P. The other core's MLP is never
 * above P's when predicting E, and never below E's when predicting P.
 *
 * @param[in] s      Interval sample (instructions > 0).
 * @param[in] lat_ns Memory latency in ns.
 * @return Prediction; ratio_e_over_p is T_E / T_P regardless of the current side.
 */
static pred_t pie_predict(const sample_t *s, double lat_ns) {
    pred_t p = {0};
    double rob_p = wh_env_double("PIE_ROB_P", 512), rob_e = wh_env_double("PIE_ROB_E", 256);
    double w_p = wh_env_double("PIE_WIDTH_P", 6), w_e = wh_env_double("PIE_WIDTH_E", 5);
    double k_base = wh_env_double("PIE_BASE_RATIO", 1.25), cap = wh_env_double("PIE_MLP_CAP", 16);
    int on_p = s->side == WH_SIDE_P;
    double fmax_cur = (on_p ? T.fmax_p_khz : T.fmax_e_khz) / 1e6;   /* GHz */
    double fmax_oth = (on_p ? T.fmax_e_khz : T.fmax_p_khz) / 1e6;
    if (fmax_cur <= 0) fmax_cur = 1;
    if (fmax_oth <= 0) fmax_oth = fmax_cur;
    double f_cur = (s->run_ns > 0 && s->cyc > 0) ? s->cyc / s->run_ns : fmax_cur;
    double f_oth = f_cur * fmax_oth / fmax_cur;
    double cpi = s->cyc / s->ins;
    double mpi = s->llc / s->ins;
    double rob_cur = on_p ? rob_p : rob_e, rob_oth = on_p ? rob_e : rob_p;
    double mlp_cur = fmin(cap, fmax(1.0, mpi * rob_cur));
    if (on_p && s->pend_cyc > 0) mlp_cur = fmax(1.0, s->pend / s->pend_cyc);   /* measured */
    double mlp_oth = fmin(cap, fmax(1.0, mpi * rob_oth));
    if (on_p) mlp_oth = fmin(mlp_oth, mlp_cur);    /* smaller window: never more MLP */
    else mlp_oth = fmax(mlp_oth, mlp_cur);
    double cpi_mem = fmin(0.95 * cpi, mpi * lat_ns * f_cur / mlp_cur);
    double cpi_base = cpi - cpi_mem;
    double cpi_base_oth = on_p ? fmax(cpi_base * k_base, 1.0 / w_e) : fmax(cpi_base / k_base, 1.0 / w_p);
    double cpi_mem_oth = mpi * lat_ns * f_oth / mlp_oth;
    p.cpi_cur = cpi;
    p.cpi_oth = cpi_base_oth + cpi_mem_oth;
    p.t_cur = cpi / f_cur;           /* ns per instruction */
    p.t_oth = p.cpi_oth / f_oth;
    p.ratio_e_over_p = on_p ? p.t_oth / p.t_cur : p.t_cur / p.t_oth;
    p.mpki = mpi * 1000;
    p.mlp_cur = mlp_cur;
    p.mlp_oth = mlp_oth;
    p.f_cur = f_cur;
    return p;
}

/**
 * @brief Launch or attach to the target and run the PIE control loop until it exits.
 * @param[in] argc Argument count.
 * @param[in] argv Arguments (see file header).
 * @return See the exit status in the file header.
 */
int main(int argc, char **argv) {
    const char *backend = "perf", *logp = NULL, *sump = NULL;
    const char *tmpl = "pmctrack -T %.3f -c %s -p %d";
    double interval_ms = 10, slack = 0.15, lat_ns = 110;
    long hyst = 2;
    int init_side = WH_SIDE_P, opt;
    pid_t attach = 0;
    while ((opt = getopt(argc, argv, "+b:i:s:H:L:I:o:S:C:p:h")) != -1) {
        switch (opt) {
        case 'b': backend = optarg; break;
        case 'i': interval_ms = atof(optarg); break;
        case 's': slack = atof(optarg); break;
        case 'H': hyst = atol(optarg); break;
        case 'L': lat_ns = atof(optarg); break;
        case 'I':
            if (!strcasecmp(optarg, "P")) init_side = WH_SIDE_P;
            else if (!strcasecmp(optarg, "E")) init_side = WH_SIDE_E;
            else if (!strcasecmp(optarg, "orig")) init_side = WH_SIDE_ORIG;
            else {
                fprintf(stderr, "pie_daemon: -I %s: initial side must be P, E or orig\n", optarg);
                return 2;
            }
            break;
        case 'o': logp = optarg; break;
        case 'S': sump = optarg; break;
        case 'C': tmpl = optarg; break;
        case 'p': attach = atoi(optarg); break;
        default:
            fprintf(stderr, "usage: %s [-b perf|pmctrack] [-i ms] [-s slack] [-H n] [-L ns] [-I P|E|orig] "
                            "[-o log.csv] [-S summary.json] [-C tmpl] (-p pid | -- cmd args...)\n", argv[0]);
            return opt == 'h' ? 0 : 2;
        }
    }
    if (hyst < 1) hyst = 1;
    if (wh_topo_detect(&T) != 0) {
        fprintf(stderr, "pie_daemon: no hybrid P/E topology (set WINHINT_PCPUS/WINHINT_ECPUS)\n");
        return 1;
    }
    int use_perf = !strcmp(backend, "perf");
    if (!use_perf && strcmp(backend, "pmctrack")) { fprintf(stderr, "pie_daemon: unknown backend %s\n", backend); return 2; }

    int sync[2] = {-1, -1};
    if (attach) target = attach;
    else {
        if (optind >= argc) { fprintf(stderr, "pie_daemon: need -p PID or -- cmd\n"); return 2; }
        if (pipe2(sync, O_CLOEXEC)) { perror("pipe"); return 1; }
        target = fork();
        if (target == 0) {
            close(sync[1]);
            char c;
            if (read(sync[0], &c, 1) != 1) _exit(126);
            execvp(argv[optind], argv + optind);
            fprintf(stderr, "pie_daemon: exec %s: %s\n", argv[optind], strerror(errno));
            _exit(127);
        }
        close(sync[0]);
    }
    int cur = init_side;
    if (init_side != WH_SIDE_ORIG)
        migrate_all(target, init_side == WH_SIDE_P ? &T.p : &T.e);
    int rc = use_perf ? perf_backend_open(target, !attach)
                      : pmctrack_backend_open(target, tmpl, interval_ms / 1000.0);
    if (rc) {
        if (!attach) { kill(target, SIGKILL); waitpid(target, NULL, 0); }
        return 5;
    }
    if (!attach) {
        if (write(sync[1], "g", 1) != 1) { perror("write"); return 1; }
        close(sync[1]);
    }
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    FILE *lf = logp ? fopen(logp, "w") : NULL;
    if (lf) fprintf(lf, "t_ms,side,instructions,cpi,mpki,mlp_cur,mlp_oth,freq_ghz,ratio_e_over_p,decision,migrated\n");
    uint64_t t0 = wh_now_ns(), mig_ns = 0;
    long n_int = 0, n_mig = 0, streak = 0, n_side[3] = {0};
    int cand = -1, status = 0, done = 0;
    struct timespec ts = {(time_t)(interval_ms / 1000), (long)(fmod(interval_ms, 1000.0) * 1e6)};
    while (!done && !stop_flag) {
        if (use_perf) nanosleep(&ts, NULL);
        pid_t w = attach ? (kill(target, 0) ? target : 0) : waitpid(target, &status, WNOHANG);
        if (w == target) { done = 1; break; }
        sample_t s;
        if ((use_perf ? perf_backend_sample(&s) : pmctrack_backend_sample(&s)) != 0) break;
        n_int++;
        if (s.ins < 1000 || s.cyc <= 0) continue;   /* idle / not scheduled */
        n_side[s.side]++;
        pred_t p = pie_predict(&s, lat_ns);
        int want = p.ratio_e_over_p <= 1.0 + slack ? WH_SIDE_E : WH_SIDE_P;
        int migrated = 0;
        if (want == cur) { cand = -1; streak = 0; }
        else {
            if (want == cand) streak++; else { cand = want; streak = 1; }
            if (streak >= hyst) {
                uint64_t a = wh_now_ns();
                if (migrate_all(target, want == WH_SIDE_P ? &T.p : &T.e) == 0) {
                    mig_ns += wh_now_ns() - a;
                    n_mig++;
                    cur = want;
                    migrated = 1;
                }
                cand = -1;
                streak = 0;
            }
        }
        if (lf)
            fprintf(lf, "%.3f,%s,%.0f,%.4f,%.3f,%.2f,%.2f,%.3f,%.4f,%s,%d\n",
                    (wh_now_ns() - t0) / 1e6, wh_side_name(s.side), s.ins, p.cpi_cur, p.mpki,
                    p.mlp_cur, p.mlp_oth, p.f_cur, p.ratio_e_over_p, wh_side_name(want), migrated);
    }
    if (!attach && !done) {
        if (stop_flag) kill(target, SIGTERM);
        while (waitpid(target, &status, 0) < 0 && errno == EINTR) { }
    }
    if (pmc_pid > 0) { kill(pmc_pid, SIGTERM); waitpid(pmc_pid, NULL, 0); }
    if (lf) fclose(lf);
    int code = attach ? 0 : (WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status));
    if (sump) {
        FILE *f = fopen(sump, "w");
        if (f) {
            fprintf(f, "{\"policy\": \"R4-PIE\", \"backend\": \"%s\", \"interval_ms\": %.3f, \"slack\": %.3f, "
                       "\"hyst\": %ld, \"lat_ns\": %.1f, \"intervals\": %ld, \"intervals_on_p\": %ld, "
                       "\"intervals_on_e\": %ld, \"migrations\": %ld, \"migration_total_ns\": %llu, "
                       "\"raw_mlp_events\": %s, \"exit_code\": %d, \"wall_ns\": %llu}\n",
                    backend, interval_ms, slack, hyst, lat_ns, n_int, n_side[WH_SIDE_P], n_side[WH_SIDE_E],
                    n_mig, (unsigned long long)mig_ns,
                    (use_perf && P.fd[0][EV_PENDC] >= 0) ? "true" : "false", code,
                    (unsigned long long)(wh_now_ns() - t0));
            fclose(f);
        }
    }
    if (use_perf) wh_perf_close(&P);
    return code;
}
