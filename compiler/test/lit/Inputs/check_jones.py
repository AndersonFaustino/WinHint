#!/usr/bin/env python3
"""B6 (Jones IQ) stats sanity check for inputs/jones_calls.c (lit test helper).

Usage:

    check_jones.py <kernel.jones*.json> <mode> <encoding>

Asserts the region/hint log of the pass in the given mode and encoding (the
`setiq` encoding also checks redundant-hint removal) and prints `ok` with the log.
"""
import json
import sys

d = json.load(open(sys.argv[1]))
mode, enc = sys.argv[2], sys.argv[3]
assert d["mode"] == mode and d["iq_only"] == (mode == "iq") and d["encoding"] == enc, d
log = d["log"]
assert d["regions"] == len(log) and d["hints"] + d["redundant_removed"] == len(log), d
assert all(x["iq_demand"] % 8 == 0 and 8 <= x["iq_demand"] <= 128 for x in log), log
f = sorted((x for x in log if x["function"] == "f"), key=lambda x: x["line"])
assert len(f) == 3, f  # loops A, B, C
a, b, c = f
if enc == "setiq":
    g = [x for x in log if x["function"] == "g"][0]
    assert g["hint"] != b["hint"], (g, b)
    # B follows the call to g(), which sets another value: not redundant.
    assert not a["redundant"] and not b["redundant"], f
    # C has B's demand and follows B directly: redundant.
    assert c["hint"] == b["hint"] and c["redundant"], f
print("ok", [(x["function"], x["line"], x["iq_demand"], x["hint"], x["redundant"]) for x in log])
