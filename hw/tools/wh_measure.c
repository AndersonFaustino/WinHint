/**
 * @file wh_measure.c
 * @brief Run one command and measure it: wall time, rusage, RAPL
 * package/core energy, and user-mode cycles/instructions split by core type
 * (cpu_core vs cpu_atom PMU, counting child threads). Emits one JSON object.
 *
 * Usage: wh_measure [-o out.json] [-c cpulist] [-P] [-R] [-r] [-k] -- cmd [args...]
 *   -c cpulist  pin the command (like taskset -c) before exec
 *   -P          no perf counters      -R  no RAPL
 *   -r          require RAPL (exit 4 with a clear message if unavailable)
 *   -k          require perf counters (exit 5 if unavailable)
 * The command's stdout/stderr are passed through. Exit status: the child's.
 *
 * The JSON goes to -o (default: stderr). The child blocks on a pipe until the
 * counters are opened (enabled at exec, inherited by its threads/children) and
 * RAPL has been sampled. Exit 2 = bad usage/cpulist; 1 = pipe/fork/wait4 failure
 * (no JSON is written); a child killed by signal S yields 128+S; 126/127 =
 * affinity/exec failure in the child.
 */
#define _GNU_SOURCE
#include "whutil.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

/**
 * @brief Write @p s as a JSON string literal (quotes/backslashes escaped,
 *        control characters replaced by spaces).
 * @param[in] f Output stream.
 * @param[in] s String (NULL prints "").
 */
static void json_str(FILE *f, const char *s) {
    fputc('"', f);
    for (; s && *s; s++) {
        if (*s == '"' || *s == '\\') fputc('\\', f);
        if ((unsigned char)*s < 0x20) { fputc(' ', f); continue; }
        fputc(*s, f);
    }
    fputc('"', f);
}

/**
 * @brief Parse options, fork/exec the command under measurement and emit the JSON record.
 * @param[in] argc Argument count.
 * @param[in] argv Arguments; the command starts at the first non-option.
 * @return The child's exit code, or 1/2/4/5 on wh_measure's own errors (see file header).
 */
int main(int argc, char **argv) {
    const char *out = NULL, *cpus = NULL;
    int no_perf = 0, no_rapl = 0, need_rapl = 0, need_perf = 0, opt;
    while ((opt = getopt(argc, argv, "+o:c:PRrkh")) != -1) {
        switch (opt) {
        case 'o': out = optarg; break;
        case 'c': cpus = optarg; break;
        case 'P': no_perf = 1; break;
        case 'R': no_rapl = 1; break;
        case 'r': need_rapl = 1; break;
        case 'k': need_perf = 1; break;
        default:
            fprintf(stderr, "usage: %s [-o out.json] [-c cpulist] [-P] [-R] [-r] [-k] -- cmd ...\n", argv[0]);
            return opt == 'h' ? 0 : 2;
        }
    }
    if (optind >= argc) { fprintf(stderr, "wh_measure: missing command\n"); return 2; }
    cpu_set_t pin;
    if (cpus && wh_parse_cpulist(cpus, &pin) <= 0) { fprintf(stderr, "wh_measure: bad cpulist %s\n", cpus); return 2; }

    wh_topo t;
    wh_topo_detect(&t);
    wh_rapl r;
    memset(&r, 0, sizeof r);
    r.fd_pkg = r.fd_core = -1;
    int rapl_ok = 0;
    if (!no_rapl) {
        rapl_ok = wh_rapl_open(&r) == 0;
        if (!rapl_ok && need_rapl) {
            fprintf(stderr, "wh_measure: RAPL required but unavailable: %s\n", r.msg);
            return 4;
        }
    }

    int sync[2];
    if (pipe2(sync, O_CLOEXEC)) { perror("pipe"); return 1; }
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }
    if (pid == 0) {
        close(sync[1]);
        char c;
        if (read(sync[0], &c, 1) != 1) _exit(126);
        if (cpus && sched_setaffinity(0, sizeof pin, &pin)) { perror("wh_measure: sched_setaffinity"); _exit(126); }
        execvp(argv[optind], argv + optind);
        fprintf(stderr, "wh_measure: exec %s: %s\n", argv[optind], strerror(errno));
        _exit(127);
    }
    close(sync[0]);
    wh_perf perf;
    memset(&perf, 0, sizeof perf);
    int perf_ok = 0;
    if (!no_perf) {
        wh_evspec ev[2] = {
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1, "cycles"},
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, -1, "instructions"},
        };
        perf_ok = wh_perf_open(&perf, pid, &t, ev, 2, WH_PERF_INHERIT | WH_PERF_ENABLE_ON_EXEC) == 0;
        if (!perf_ok && need_perf) {
            fprintf(stderr, "wh_measure: perf counters required but unavailable: %s\n", perf.msg);
            kill(pid, SIGKILL);
            waitpid(pid, NULL, 0);
            return 5;
        }
    }
    uint64_t p0 = 0, c0 = 0, p1 = 0, c1 = 0;
    if (rapl_ok) wh_rapl_read(&r, &p0, &c0);
    uint64_t t0 = wh_now_ns();
    if (write(sync[1], "g", 1) != 1) { perror("write"); return 1; }
    close(sync[1]);
    int status = 0;
    struct rusage ru;
    pid_t w;
    while ((w = wait4(pid, &status, 0, &ru)) < 0 && errno == EINTR) { }
    if (w < 0) {   /* status and ru are meaningless: do not report a fake result */
        perror("wh_measure: wait4");
        return 1;
    }
    uint64_t t1 = wh_now_ns();
    if (rapl_ok) wh_rapl_read(&r, &p1, &c1);
    wh_perf_vals pv;
    memset(&pv, 0, sizeof pv);
    if (perf_ok) wh_perf_read(&perf, &pv);
    int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);

    FILE *f = out ? fopen(out, "w") : stderr;
    if (!f) { perror(out); f = stderr; }
    fprintf(f, "{\"exit_code\": %d, \"wall_ns\": %llu, \"utime_us\": %lld, \"stime_us\": %lld, "
               "\"maxrss_kb\": %ld, \"nvcsw\": %ld, \"nivcsw\": %ld, ",
            code, (unsigned long long)(t1 - t0),
            (long long)ru.ru_utime.tv_sec * 1000000 + ru.ru_utime.tv_usec,
            (long long)ru.ru_stime.tv_sec * 1000000 + ru.ru_stime.tv_usec, ru.ru_maxrss,
            ru.ru_nvcsw, ru.ru_nivcsw);
    if (rapl_ok)
        fprintf(f, "\"energy_pkg_uj\": %llu, \"energy_core_uj\": %llu, ",
                (unsigned long long)wh_rapl_delta(&r, p0, p1, 0),
                (unsigned long long)wh_rapl_delta(&r, c0, c1, 1));
    else
        fprintf(f, "\"energy_pkg_uj\": null, \"energy_core_uj\": null, ");
    fprintf(f, "\"rapl_backend\": \"%s\", \"rapl_msg\": ",
            r.backend == 1 ? "sysfs" : r.backend == 2 ? "perf" : "none");
    json_str(f, r.msg);
    if (perf_ok)
        fprintf(f, ", \"cycles_p\": %llu, \"instructions_p\": %llu, \"run_ns_p\": %llu, "
                   "\"cycles_e\": %llu, \"instructions_e\": %llu, \"run_ns_e\": %llu",
                (unsigned long long)pv.v[0][0], (unsigned long long)pv.v[0][1],
                (unsigned long long)pv.running[0][0], (unsigned long long)pv.v[1][0],
                (unsigned long long)pv.v[1][1], (unsigned long long)pv.running[1][0]);
    fprintf(f, ", \"perf_msg\": ");
    json_str(f, perf_ok ? "" : (no_perf ? "disabled" : perf.msg));
    fprintf(f, "}\n");
    if (f != stderr) fclose(f);
    if (perf_ok) wh_perf_close(&perf);
    if (rapl_ok) wh_rapl_close(&r);
    return code;
}
