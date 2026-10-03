#!/usr/bin/env python3
"""Print a C file with N functions `f000..f<N-1>`, one loop each (N regions).

Usage:

    gen_loops.py N
"""
import sys

n = int(sys.argv[1])
for i in range(n):
    print("void f%03d(float *p, int n) { for (int i = 0; i < n; i++) p[i] += %d.0f; }" % (i, i + 1))
