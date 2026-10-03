/**
 * @file
 * @brief Model test input: dependent pointer chase over 16 MiB of nodes.
 *
 * Dependent pointer chase: every miss depends on the previous one, so no
 * memory-level parallelism -> expect the smallest window.
 * Prints `chase <sum>`.
 */
#include <stdio.h>
#include <stdlib.h>
/** @brief 64-byte list node (one cache line per node). */
typedef struct node { struct node *next; long val; long pad[6]; } node;
/** @brief Number of nodes. */
#define NN (1 << 18) /* 16 MiB of nodes */

/**
 * @brief Follow the list for a number of steps, summing the node values
 *        (the loop analyzed by the tests).
 * @param p     Start node.
 * @param steps Number of nodes to visit.
 * @return Sum of the visited `val` fields.
 */
__attribute__((noinline)) long walk(node *p, long steps) {
  long s = 0;
  for (long i = 0; i < steps; i++) {
    s += p->val;
    p = p->next;
  }
  return s;
}

/** @brief Build a random cyclic permutation list and walk it 10^6 steps. */
int main(void) {
  node *nodes = malloc(sizeof(node) * NN);
  unsigned *perm = malloc(sizeof(unsigned) * NN);
  for (unsigned i = 0; i < NN; i++) perm[i] = i;
  unsigned x = 12345u;
  for (unsigned i = NN - 1; i > 0; i--) {
    x = x * 1103515245u + 12345u;
    unsigned j = (x >> 8) % (i + 1), t = perm[i];
    perm[i] = perm[j]; perm[j] = t;
  }
  for (unsigned i = 0; i < NN; i++) {
    nodes[perm[i]].next = &nodes[perm[(i + 1) % NN]];
    nodes[perm[i]].val = i;
  }
  printf("chase %ld\n", walk(&nodes[perm[0]], 1000000));
  free(nodes); free(perm);
  return 0;
}
