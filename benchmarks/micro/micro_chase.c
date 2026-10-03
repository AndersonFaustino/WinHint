/**
 * @file micro_chase.c
 * @brief Dependent long-latency misses (pointer chasing, MLP = 1).
 *
 * Condition targeted: every miss address comes from the previous miss, so a
 * larger window cannot overlap them.
 *   B3 (Kora): MLP mode must NOT pay off (no speedup, window stays small).
 * The nodes (one per 64-byte line, 8 MiB) form one random cycle (Sattolo's
 * algorithm, fixed LCG seed). Each step also runs the same 6-op dependent
 * chain on the payload as micro_gather, so the two differ only in whether
 * the misses are independent.
 * Input: large = 96 Ki steps, small = 32 Ki steps (other seed).
 */
#include "micro.h"

#define NODES (MICRO_PAGES * 64u)   ///< 128 Ki nodes x 64 B = 8 MiB

/** @brief One list node, exactly one 64-byte cache line. */
struct node {
    struct node *next; ///< Next node of the single random cycle.
    uint64_t val;      ///< Payload hashed at every step.
    uint64_t pad[6];   ///< Padding to 64 bytes.
};

static struct node nodes[NODES]; ///< The 8 MiB node pool.
static uint32_t perm[NODES];     ///< Visit order: perm[i] is linked to perm[i + 1].

/**
 * @brief Link all nodes into one random cycle (Sattolo's algorithm, LCG) and set
 *        each node's payload to micro_mix64(index + seed).
 * @param[in] seed LCG and payload seed.
 */
static MICRO_NOINLINE void build_cycle(uint64_t seed)
{
    uint64_t s = seed;
    for (unsigned i = 0; i < NODES; i++)
        perm[i] = i;
    for (unsigned i = NODES - 1u; i > 0; i--) { /* Sattolo: one single cycle */
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        unsigned j = (unsigned)((s >> 33) % i);
        uint32_t t = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
    }
    for (unsigned i = 0; i < NODES; i++) {
        nodes[perm[i]].next = &nodes[perm[(i + 1u) % NODES]];
        nodes[i].val = micro_mix64(i + seed);
    }
}

/**
 * @brief Follow `steps` links from nodes[perm[0]], hashing each payload with the
 *        same dependent chain as micro_gather().
 * @param[in] steps Number of dependent loads.
 * @return Sum of the hashed payloads plus the final node index.
 */
static MICRO_NOINLINE uint64_t chase(unsigned steps)
{
    const struct node *p = &nodes[perm[0]];
    uint64_t sum = 0;
    for (unsigned k = 0; k < steps; k++) {
        uint64_t v = p->val;
        uint64_t t = v ^ (v >> 29);
        t *= 0x9e3779b97f4a7c15ull;
        t ^= t >> 32;
        t += k;
        t = micro_rotl(t, 7);
        sum += t;
        p = p->next;
    }
    return sum + (uint64_t)(p - nodes);
}

/**
 * @brief Build the cycle, chase it (96 Ki steps large / 32 Ki small) and print
 *        the checksum.
 * @param[in] argc Argument count.
 * @param[in] argv argv[1] = small|large (default large).
 * @return 0 (exits 2 on a bad argument).
 */
int main(int argc, char **argv)
{
    int large = micro_parse(argc, argv);
    build_cycle(large ? 12345u : 777u);
    uint64_t sum = chase(large ? 96u * 1024u : 32u * 1024u);
    micro_report("micro_chase", large, sum);
    return 0;
}
