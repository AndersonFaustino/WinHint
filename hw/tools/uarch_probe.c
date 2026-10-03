/**
 * @file uarch_probe.c
 * @brief Microbenchmarks that confirm the out-of-order window
 * parameters of one core type (PROPOSAL §8: "cite Intel's optimization manual
 * and confirm with microbenchmarks"; documented values in docs/guide/hardware/uarch-params.md).
 *
 * Method (H. Wong, "Measuring Reorder Buffer Capacity", 2013): two independent
 * pointer chases that miss in every cache level (A and B) are separated by N
 * filler instructions:
 *
 *     loop:  mov rax,[rax] ; N x filler ; mov rdx,[rdx] ; N x filler ; dec rcx ; jnz loop
 *
 * While load A waits at the head of the window, load B overlaps with it only
 * if A, the N fillers and B all fit in the structure the filler occupies, so
 * the time per iteration jumps from ~1 to ~2 memory latencies at N ~= size of
 * that structure. Fillers:
 *   rob  1-byte nop                -> reorder buffer (every uop takes a ROB entry)
 *   lb   mov r11,[rsp-8]  (L1 hit) -> load buffer (bounded also by the int PRF)
 *   sb   mov [rsp-8],r11           -> store buffer
 * A fourth probe measures memory-level parallelism:
 *   mlp  k independent chases per iteration (k = 1..12, no filler); effective
 *        MLP(k) = k * t(1) / t(k) saturates at the number of outstanding L1D
 *        misses the core sustains (fill buffers / MSHRs).
 * The loops are generated at run time (x86-64 machine code in an mmap'd page).
 *
 *   uarch_probe [-c CPU|P|E] [-p rob,lb,sb,mlp|all] [-n MIN:MAX:STEP] [-k KMAX]
 *               [-i ITERS] [-m MB] [-r REPS] [-o out.json] [-q]
 *
 *   -c  CPU to pin to: a number, or P / E = first CPU of that core type (default P)
 *   -n  filler range for rob/lb/sb (default per probe: rob 0:640:8, lb 0:256:4, sb 0:160:2)
 *   -k  max chains for mlp (default 12)     -i  iterations per point (20000)
 *   -m  pointer-chase buffer in MB (64; must exceed the LLC)
 *   -r  repetitions per point, the minimum is kept (3)
 * Output: CSV on stdout (probe,cpu,core_type,x,ns_per_iter) and, with -o, JSON
 * with the curves and the knee estimates (hw/fidelity.py reads it).
 *
 * Knee estimate (rob/lb/sb): the largest increase of t between consecutive
 * points, accepted if it is >= 25% of the plateau before it; the structure
 * size is ~ x_knee + 1 (A, the fillers and B must fit: N + 2 entries, B's slot
 * is the one that no longer fits). Resolution = the step. These are
 * confirmations of documented values, not measurements for the paper.
 *
 * The -n range, when given, applies to every selected rob/lb/sb probe. An
 * unknown name in the -p list is a usage error (exit 2).
 * Exit status: 0 ok, 1 pinning/allocation/output error, 2 bad usage or a
 * non-x86-64 build.
 */
#define _GNU_SOURCE
#include "whutil.h"

#include <ctype.h>
#include <errno.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <unistd.h>

#if !defined(__x86_64__)
/** @brief Non-x86-64 build: print an error and exit with status 2. */
int main(void) {
    fprintf(stderr, "uarch_probe: x86-64 only\n");
    return 2;
}
#else

/** @brief x86-64 general-purpose register numbers (ModRM/REX encoding). */
enum { RAX = 0, RCX = 1, RDX = 2, RBX = 3, RSP = 4, RBP = 5, RSI = 6, RDI = 7,
       R8, R9, R10, R11, R12, R13, R14, R15 };

/* ---------------------------------------------------------------- JIT */
/** @brief Run-time code buffer (an mmap'd page toggled between RW and RX). */
typedef struct {
    uint8_t *p;      ///< Buffer start.
    size_t n, cap;   ///< Bytes emitted / capacity.
} code_t;

/**
 * @brief Emit one byte; exits the process on buffer overflow.
 * @param[in,out] c Code buffer.
 * @param[in]     b Byte value.
 */
static void put(code_t *c, int b) {
    if (c->n >= c->cap) { fprintf(stderr, "uarch_probe: code buffer overflow\n"); exit(1); }
    c->p[c->n++] = (uint8_t)b;
}

/**
 * @brief Emit a 64-bit register <-> memory move.
 *
 * op r64 <-> [base + disp8]; op = 0x8B (load) or 0x89 (store)
 *
 * @param[in,out] c    Code buffer.
 * @param[in]     op   Opcode byte (0x8B load, 0x89 store).
 * @param[in]     reg  Register operand.
 * @param[in]     base Base register (RSP gets a SIB byte, RBP always a disp8).
 * @param[in]     disp Signed 8-bit displacement.
 */
static void mem_op(code_t *c, int op, int reg, int base, int disp) {
    put(c, 0x48 | ((reg >> 3) & 1) << 2 | ((base >> 3) & 1));
    put(c, op);
    int mod = (disp == 0 && (base & 7) != RBP) ? 0 : 1;
    put(c, mod << 6 | (reg & 7) << 3 | (base & 7));
    if ((base & 7) == RSP) put(c, 0x24);   /* SIB: base only */
    if (mod == 1) put(c, disp & 0xff);
}
/** @brief Emit `mov dst, [base+disp]`. */
static void load(code_t *c, int dst, int base, int disp) { mem_op(c, 0x8B, dst, base, disp); }
/** @brief Emit `mov [base+disp], src`. */
static void store(code_t *c, int base, int disp, int src) { mem_op(c, 0x89, src, base, disp); }
/** @brief Emit `push r`. */
static void push(code_t *c, int r) { if (r >= 8) put(c, 0x41); put(c, 0x50 + (r & 7)); }
/** @brief Emit `pop r`. */
static void pop(code_t *c, int r) { if (r >= 8) put(c, 0x41); put(c, 0x58 + (r & 7)); }

/** @brief Filler kinds: 1-byte nop (ROB), L1-hit load (LB), store (SB). */
enum { F_NOP, F_LOAD, F_STORE };
/**
 * @brief Emit one filler instruction.
 * @param[in,out] c    Code buffer.
 * @param[in]     kind F_NOP, F_LOAD (`mov r11,[rsp-8]`) or F_STORE (`mov [rsp-8],r11`).
 */
static void filler(code_t *c, int kind) {
    if (kind == F_NOP) put(c, 0x90);
    else if (kind == F_LOAD) load(c, R11, RSP, -8);
    else store(c, RSP, -8, R11);
}

/**
 * @brief Emit `dec rcx; jnz top`.
 * @param[in,out] c   Code buffer.
 * @param[in]     top Offset of the loop head.
 */
static void loop_tail(code_t *c, size_t top) {
    put(c, 0x48); put(c, 0xFF); put(c, 0xC9);                 /* dec rcx */
    int32_t rel = (int32_t)((long)top - (long)(c->n + 6));
    put(c, 0x0F); put(c, 0x85);                               /* jnz rel32 */
    for (int i = 0; i < 4; i++) put(c, (rel >> (8 * i)) & 0xff);
}

/**
 * @brief Generate the two-chain ROB/LB/SB probe loop.
 *
 * uint64_t f(uint64_t iters, void **pos): chains A = pos[0], B = pos[1]; the final
 * positions are stored back so the next call continues on lines not yet cached.
 *
 * @param[in,out] c    Code buffer (reset).
 * @param[in]     kind Filler kind (F_NOP, F_LOAD, F_STORE).
 * @param[in]     n    Fillers after each chase load.
 */
static void gen_pair(code_t *c, int kind, int n) {
    c->n = 0;
    put(c, 0x48); put(c, 0x89); put(c, 0xF9);                 /* mov rcx, rdi */
    load(c, RAX, RSI, 0);
    load(c, RDX, RSI, 8);
    while (c->n % 16) put(c, 0x90);
    size_t top = c->n;
    load(c, RAX, RAX, 0);
    for (int i = 0; i < n; i++) filler(c, kind);
    load(c, RDX, RDX, 0);
    for (int i = 0; i < n; i++) filler(c, kind);
    loop_tail(c, top);
    store(c, RSI, 0, RAX);
    store(c, RSI, 8, RDX);
    put(c, 0xC3);
}

/** @brief Registers holding the MLP chains (callee-saved ones are pushed/popped). */
static const int CHAIN_REG[] = {RAX, RDX, R8, R9, R10, R11, RDI, RBX, R12, R13, R14, R15};
/** @brief Maximum number of MLP chains (one per CHAIN_REG entry). */
#define KMAX_CHAINS ((int)(sizeof CHAIN_REG / sizeof CHAIN_REG[0]))
/** @brief Callee-saved registers used by gen_mlp(). */
static const int SAVED[] = {RBX, R12, R13, R14, R15};

/**
 * @brief Generate the k-chain MLP probe loop.
 *
 * uint64_t f(uint64_t iters, void **pos): k chains, one load each per iteration;
 * final positions stored back to pos[]
 *
 * @param[in,out] c Code buffer (reset).
 * @param[in]     k Number of chains, 1..KMAX_CHAINS.
 */
static void gen_mlp(code_t *c, int k) {
    c->n = 0;
    for (int i = 0; i < 5; i++) push(c, SAVED[i]);
    put(c, 0x48); put(c, 0x89); put(c, 0xF9);                 /* mov rcx, rdi */
    for (int i = 0; i < k; i++) load(c, CHAIN_REG[i], RSI, 8 * i);
    while (c->n % 16) put(c, 0x90);
    size_t top = c->n;
    for (int i = 0; i < k; i++) load(c, CHAIN_REG[i], CHAIN_REG[i], 0);
    loop_tail(c, top);
    for (int i = 0; i < k; i++) store(c, RSI, 8 * i, CHAIN_REG[i]);
    for (int i = 4; i >= 0; i--) pop(c, SAVED[i]);
    put(c, 0xC3);                                             /* rax = chain 0 */
}

/* ---------------------------------------------------------------- chase buffer */
/**
 * @brief Pointer-chase buffer.
 *
 * One random cycle through every 64-byte line of the buffer. Each timed point hands
 * every chain a *fresh* segment of the cycle (chain j starts where chain j-1's
 * segment ends, the frontier then moves past all of them), so every load goes to a
 * line last touched ~ a whole buffer of chasing ago -- i.e. not in any cache if the
 * buffer is well above the LLC.
 */
typedef struct {
    void **lines;      /**< one pointer per 64-byte line */
    uint32_t *cyc;     /**< cyc[t] = t-th line along the cycle */
    size_t n;          /**< lines */
    size_t front;      /**< next unused position along the cycle */
} chase_t;

/** @brief Pointers per 64-byte line. */
#define LINE_WORDS 8

/**
 * @brief Allocate the chase buffer (mmap, MADV_HUGEPAGE best effort) and link
 *        all lines into one Sattolo random cycle.
 * @param[out] ch Buffer to initialize.
 * @param[in]  mb Size in MiB.
 * @return 0 on success, -1 on allocation failure.
 */
static int chase_init(chase_t *ch, size_t mb) {
    size_t bytes = mb << 20;
    void *m = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return -1;
#ifdef MADV_HUGEPAGE
    madvise(m, bytes, MADV_HUGEPAGE);   /* fewer TLB misses; best effort */
#endif
    ch->lines = m;
    ch->n = bytes / 64;
    ch->front = 0;
    uint32_t *perm = malloc(ch->n * sizeof *perm);
    ch->cyc = malloc(ch->n * sizeof *ch->cyc);
    if (!perm || !ch->cyc) return -1;
    for (size_t i = 0; i < ch->n; i++) perm[i] = (uint32_t)i;
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (size_t i = ch->n - 1; i > 0; i--) {   /* Sattolo: one cycle through every line */
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        size_t j = s % i;
        uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for (size_t i = 0; i < ch->n; i++)
        ch->lines[i * LINE_WORDS] = &ch->lines[(size_t)perm[i] * LINE_WORDS];
    uint32_t cur = 0;
    for (size_t t = 0; t < ch->n; t++) { ch->cyc[t] = cur; cur = perm[cur]; }
    free(perm);
    return 0;
}

/**
 * @brief Hand out fresh, never-recently-touched segments of the cycle.
 *
 * k fresh segments of `steps` lines each; out[j] = start of segment j
 *
 * @param[in,out] ch    Chase buffer (frontier advanced by k * steps).
 * @param[in]     k     Number of segments.
 * @param[in]     steps Lines per segment.
 * @param[out]    out   k start pointers.
 */
static void chase_fresh(chase_t *ch, int k, size_t steps, void **out) {
    for (int j = 0; j < k; j++) {
        out[j] = &ch->lines[(size_t)ch->cyc[ch->front % ch->n] * LINE_WORDS];
        ch->front += steps;
    }
}

/* ---------------------------------------------------------------- measurement */
/** @brief Signature of the generated probe loops. */
typedef uint64_t (*chase_fn)(uint64_t, void **);
/** @brief Sink for the generated functions' return values. */
static volatile uint64_t sink;

/**
 * @brief Time a generated loop.
 *
 * best-of-reps ns per iteration; the chains advance through fresh lines on every call
 * (after one warm-up call of iters/4+1 iterations).
 *
 * @param[in]     c     Code buffer holding an RX loop.
 * @param[in,out] pos   Chain positions (updated by the loop).
 * @param[in]     iters Iterations per timed call.
 * @param[in]     reps  Timed calls; the minimum is returned.
 * @return Best time per iteration in ns.
 */
static double time_chase(code_t *c, void **pos, long iters, int reps) {
    chase_fn f = (chase_fn)(void *)c->p;
    sink += f((uint64_t)(iters / 4 + 1), pos);                /* warm-up (code, TLB) */
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        uint64_t t0 = wh_now_ns();
        sink += f((uint64_t)iters, pos);
        double t = (double)(wh_now_ns() - t0) / (double)iters;
        if (t < best) best = t;
    }
    return best;
}

/**
 * @brief Make the code buffer writable.
 * @param[in] c Code buffer.
 * @return mprotect() result.
 */
static int code_rw(code_t *c) { return mprotect(c->p, c->cap, PROT_READ | PROT_WRITE); }
/**
 * @brief Flush the instruction cache and make the code buffer executable.
 * @param[in] c Code buffer.
 * @return mprotect() result.
 */
static int code_rx(code_t *c) {
    __builtin___clear_cache((char *)c->p, (char *)c->p + c->n);
    return mprotect(c->p, c->cap, PROT_READ | PROT_EXEC);
}

/**
 * @brief Find the knee of a probe curve.
 *
 * largest jump between consecutive points; -1 if none >= 25% of the plateau
 * (the minimum before the jump) or if a later point falls back below its midpoint.
 *
 * @param[in] t Time per iteration for each point.
 * @param[in] n Number of points.
 * @return Index of the first point after the jump, or -1.
 */
static int knee_index(const double *t, int n) {
    int best = -1;
    double bj = 0;
    for (int i = 0; i + 1 < n; i++) {
        double j = t[i + 1] - t[i];
        if (j > bj) { bj = j; best = i + 1; }
    }
    if (best < 1) return -1;
    double plateau = t[0];
    for (int i = 0; i < best; i++) if (t[i] < plateau) plateau = t[i];
    if (bj < 0.25 * plateau) return -1;
    /* a real knee is a step: every later point stays above the midpoint of the jump
     * (rejects single noisy points on a busy machine) */
    for (int i = best; i < n; i++) if (t[i] < t[best - 1] + 0.5 * bj) return -1;
    return best;
}

/**
 * @brief Print the usage line to stderr.
 * @param[in] a0 Program name.
 */
static void usage(const char *a0) {
    fprintf(stderr, "usage: %s [-c CPU|P|E] [-p rob,lb,sb,mlp|all] [-n MIN:MAX:STEP] [-k KMAX] "
                    "[-i ITERS] [-m MB] [-r REPS] [-o out.json] [-q]\n", a0);
}

/**
 * @brief Parse options, pin to the CPU, run the selected probes, print CSV (and JSON with -o).
 * @param[in] argc Argument count.
 * @param[in] argv Arguments (see file header).
 * @return 0 on success, 1 on a pinning/allocation/output error, 2 on bad usage.
 */
int main(int argc, char **argv) {
    const char *cpu_arg = "P", *probes = "all", *range = NULL, *outp = NULL;
    long iters = 20000;
    int kmax = KMAX_CHAINS, reps = 3, quiet = 0;
    size_t mb = 64;
    int opt;
    while ((opt = getopt(argc, argv, "c:p:n:k:i:m:r:o:qh")) != -1) {
        switch (opt) {
        case 'c': cpu_arg = optarg; break;
        case 'p': probes = optarg; break;
        case 'n': range = optarg; break;
        case 'k': kmax = atoi(optarg); break;
        case 'i': iters = atol(optarg); break;
        case 'm': mb = (size_t)atol(optarg); break;
        case 'r': reps = atoi(optarg); break;
        case 'o': outp = optarg; break;
        case 'q': quiet = 1; break;
        default: usage(argv[0]); return opt == 'h' ? 0 : 2;
        }
    }
    if (kmax < 1 || kmax > KMAX_CHAINS || iters < 1 || reps < 1 || mb < 1) { usage(argv[0]); return 2; }
    if (strcmp(probes, "all")) {   /* every name in the -p list must be a known probe */
        char lst[128], *save = NULL;
        if (strlen(probes) >= sizeof lst) { fprintf(stderr, "uarch_probe: probe list too long\n"); return 2; }
        snprintf(lst, sizeof lst, "%s", probes);
        for (char *tok = strtok_r(lst, ",", &save); tok; tok = strtok_r(NULL, ",", &save))
            if (strcmp(tok, "rob") && strcmp(tok, "lb") && strcmp(tok, "sb") && strcmp(tok, "mlp")) {
                fprintf(stderr, "uarch_probe: unknown probe '%s' (rob, lb, sb, mlp or all)\n", tok);
                return 2;
            }
    }

    wh_topo T;
    int hybrid = wh_topo_detect(&T) == 0;
    int cpu;
    if (!strcasecmp(cpu_arg, "P") || !strcasecmp(cpu_arg, "E")) {
        if (!hybrid) { fprintf(stderr, "uarch_probe: no hybrid topology; give a CPU number\n"); return 2; }
        cpu = wh_cpuset_first(toupper((unsigned char)cpu_arg[0]) == 'P' ? &T.p : &T.e);
    } else {
        char *end;
        cpu = (int)strtol(cpu_arg, &end, 10);
        if (*end || cpu < 0) { usage(argv[0]); return 2; }
    }
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(cpu, &one);
    if (sched_setaffinity(0, sizeof one, &one)) {
        fprintf(stderr, "uarch_probe: pin to cpu %d: %s\n", cpu, strerror(errno));
        return 1;
    }
    int side = hybrid ? wh_topo_side_of_cpu(&T, cpu) : -1;
    const char *ctype = side == WH_SIDE_P ? "P" : side == WH_SIDE_E ? "E" : "unknown";

    chase_t ch;
    if (chase_init(&ch, mb)) { fprintf(stderr, "uarch_probe: cannot allocate %zu MB\n", mb); return 1; }
    code_t c = {0};
    c.cap = 1 << 16;
    c.p = mmap(NULL, c.cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (c.p == MAP_FAILED) { perror("mmap"); return 1; }

    size_t seg = (size_t)(iters / 4 + 1) + (size_t)iters * (size_t)reps;   /* steps per chain per point */
    if (seg * (size_t)kmax > ch.n || seg * 2 > ch.n)
        fprintf(stderr, "uarch_probe: warning: -i/-r too large for -m %zu: chains revisit cached lines\n", mb);

    FILE *js = NULL;
    if (outp && !(js = fopen(outp, "w"))) { perror(outp); return 1; }
    char model[128] = "";
    FILE *ci = fopen("/proc/cpuinfo", "r");
    if (ci) {
        char line[256];
        while (fgets(line, sizeof line, ci))
            if (!strncmp(line, "model name", 10)) {
                char *v = strchr(line, ':');
                if (v) { snprintf(model, sizeof model, "%s", v + 2); model[strcspn(model, "\n\"")] = 0; }
                break;
            }
        fclose(ci);
    }
    if (js)
        fprintf(js, "{\n  \"tool\": \"uarch_probe\", \"model\": \"%s\", \"cpu\": %d, \"core_type\": \"%s\",\n"
                    "  \"iters\": %ld, \"reps\": %d, \"buffer_mb\": %zu,\n  \"probes\": {",
                model, cpu, ctype, iters, reps, mb);
    printf("probe,cpu,core_type,x,ns_per_iter\n");

    static const struct { const char *name; int kind; int lo, hi, step; } PAIR[] = {
        {"rob", F_NOP, 0, 640, 8}, {"lb", F_LOAD, 0, 256, 4}, {"sb", F_STORE, 0, 160, 2}};
    int all = !strcmp(probes, "all"), first = 1, ran = 0;
    for (int pi = 0; pi < 3; pi++) {
        char pat[16];
        snprintf(pat, sizeof pat, ",%s,", PAIR[pi].name);
        char lst[128];
        snprintf(lst, sizeof lst, ",%s,", probes);
        if (!all && !strstr(lst, pat)) continue;
        int lo = PAIR[pi].lo, hi = PAIR[pi].hi, step = PAIR[pi].step;
        if (range && sscanf(range, "%d:%d:%d", &lo, &hi, &step) != 3) { usage(argv[0]); return 2; }
        if (step < 1 || lo < 0 || hi < lo || 2L * hi * 5 + 64 > (long)c.cap) {
            fprintf(stderr, "uarch_probe: bad range %d:%d:%d\n", lo, hi, step);
            return 2;
        }
        int np = (hi - lo) / step + 1;
        double *t = calloc((size_t)np, sizeof *t);
        int *xs = calloc((size_t)np, sizeof *xs);
        void *st[2];
        for (int i = 0; i < np; i++) {
            xs[i] = lo + i * step;
            chase_fresh(&ch, 2, seg, st);
            code_rw(&c);
            gen_pair(&c, PAIR[pi].kind, xs[i]);
            code_rx(&c);
            t[i] = time_chase(&c, st, iters, reps);
            printf("%s,%d,%s,%d,%.3f\n", PAIR[pi].name, cpu, ctype, xs[i], t[i]);
            fflush(stdout);
        }
        int k = knee_index(t, np);
        if (!quiet)
            fprintf(stderr, "uarch_probe: %s cpu %d (%s): %s", PAIR[pi].name, cpu, ctype,
                    k < 0 ? "no knee in range\n" : "");
        if (!quiet && k >= 0)
            fprintf(stderr, "knee at N=%d (t %.1f -> %.1f ns/iter): size ~ %d-%d entries\n",
                    xs[k], t[k - 1], t[k], xs[k - 1] + 1, xs[k] + 1);
        if (js) {
            fprintf(js, "%s\n    \"%s\": {\"x\": [", first ? "" : ",", PAIR[pi].name);
            for (int i = 0; i < np; i++) fprintf(js, "%s%d", i ? ", " : "", xs[i]);
            fprintf(js, "],\n      \"ns_per_iter\": [");
            for (int i = 0; i < np; i++) fprintf(js, "%s%.3f", i ? ", " : "", t[i]);
            if (k >= 0)
                fprintf(js, "],\n      \"knee_x\": %d, \"size_lo\": %d, \"size_hi\": %d}", xs[k],
                        xs[k - 1] + 1, xs[k] + 1);
            else
                fprintf(js, "],\n      \"knee_x\": null, \"size_lo\": null, \"size_hi\": null}");
        }
        first = 0;
        ran++;
        free(t);
        free(xs);
    }
    if (all || strstr(probes, "mlp")) {
        double t[KMAX_CHAINS + 1], best = 0;
        void *st[KMAX_CHAINS];
        int bestk = 1;
        for (int k = 1; k <= kmax; k++) {
            chase_fresh(&ch, k, seg, st);
            code_rw(&c);
            gen_mlp(&c, k);
            code_rx(&c);
            t[k] = time_chase(&c, st, iters, reps);
            double mlp = k * t[1] / t[k];
            if (mlp > best) { best = mlp; bestk = k; }
            printf("mlp,%d,%s,%d,%.3f\n", cpu, ctype, k, t[k]);
            fflush(stdout);
        }
        if (!quiet)
            fprintf(stderr, "uarch_probe: mlp cpu %d (%s): effective MLP %.1f (at k=%d), latency %.1f ns\n",
                    cpu, ctype, best, bestk, t[1]);
        if (js) {
            fprintf(js, "%s\n    \"mlp\": {\"x\": [", first ? "" : ",");
            for (int k = 1; k <= kmax; k++) fprintf(js, "%s%d", k > 1 ? ", " : "", k);
            fprintf(js, "],\n      \"ns_per_iter\": [");
            for (int k = 1; k <= kmax; k++) fprintf(js, "%s%.3f", k > 1 ? ", " : "", t[k]);
            fprintf(js, "],\n      \"latency_ns\": %.3f, \"mlp_effective\": %.3f, \"mlp_at_k\": %d}",
                    t[1], best, bestk);
        }
        ran++;
    }
    if (js) {
        fprintf(js, "\n  }\n}\n");
        fclose(js);
    }
    if (!ran) { fprintf(stderr, "uarch_probe: unknown probe list '%s'\n", probes); return 2; }
    return 0;
}
#endif
