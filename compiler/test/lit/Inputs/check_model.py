#!/usr/bin/env python3
"""Check the WinHint demand model output of three kernels (lit test helper).

Usage:

    check_model.py <machine.json> <stream.txt> <chase.txt> <compute.txt> [<prev W*(compute)>]

Checks the `print<winhint-demand>` output of the three model kernels against
the machine description (so the test follows sim/machines/*.json edits):

    stream  (triad): independent DRAM misses -> W* = W_max, last config, MLP-driven,
                     L_mem = L2 hit latency + memory latency (JSON parsed)
    chase   (walk) : dependent misses        -> config 0
    compute (poly) : no long-latency loads   -> W* = W_cp < W_max

Prints W*(compute) so the caller can check that it scales with the core.
"""
import json
import re
import sys


def line(path, fn):
    """Return the key=value fields of the last ``WINHINT fn=<fn>`` line of a file.

    Args:
        path (str): Text file with the ``print<winhint-demand>`` output.
        fn (str): Function name to look for.

    Returns:
        (dict[str, str]): Dict of the ``key=value`` tokens of that line (values as strings).

    Raises:
        AssertionError: If the file has no such line.
    """
    rows = [l for l in open(path) if re.search(r"WINHINT fn=%s " % fn, l)]
    assert rows, "no WINHINT line for %s in %s" % (fn, path)
    return dict(kv.split("=", 1) for kv in rows[-1].split() if "=" in kv)


m = json.load(open(sys.argv[1]))
rob = m["window"]["rob"]
cache = m["cache"]
l2 = cache["l2"]["hit_latency_cycles"] if "l2" in cache else None
mem = m["memory"]["latency_cycles"]

s = line(sys.argv[2], "triad")
assert int(s["W*"]) == max(rob), s
assert int(s["config"]) == len(rob) - 1, s
assert int(s["W_mlp"]) > int(s["W_cp"]), s
if l2 is not None:
    assert float(s["L_mem"]) == l2 + mem, (s["L_mem"], l2, mem)

c = line(sys.argv[3], "walk")
assert int(c["config"]) == 0, c
assert float(c["MLP"]) <= 1.0, c

p = line(sys.argv[4], "poly")
assert int(p["W_mlp"]) == 0, p
assert int(p["W*"]) == int(p["W_cp"]) < max(rob), p
assert int(c["W*"]) < int(p["W*"]), (c, p)
if len(sys.argv) > 5:
    assert int(p["W*"]) > int(sys.argv[5]), ("W*(compute) does not grow with the core", p)
print(p["W*"])
