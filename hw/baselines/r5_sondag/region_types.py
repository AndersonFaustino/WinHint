#!/usr/bin/env python3
"""R5 (Sondag & Rajan, CGO'11) -- static region typing for libwinhint's sondag mode.

The published flow clusters code sections by static similarity ("section
types"), samples one representative visit of each *type* on every core type at
run time, and then assigns the whole type to the core type where it performs
best. libwinhint implements the runtime part (`WINHINT_MODE=sondag`); this script
produces the static part: a "region_id type_id" file for `WINHINT_SONDAG_TYPES`.

Input: `<kernel>.regions.json` written by the WinHint pass (docs/interfaces.md §5):
`id -> {function, loop header, source line, ...}`. Typing strategies:

- `features` (default): if regions carry numeric static features (any numeric
  fields, e.g. instruction-mix counts from the pass) they are clustered with a
  deterministic k-means (k = `--k`); this is the closest match to the paper's
  instruction-mix clustering.
- `function`: regions of the same function share a type.
- `identity`: one type per region (no clustering; also the fallback).

Usage:

    region_types.py regions.json -o types.txt [--strategy features|function|identity] [--k 4]
"""
import argparse
import json
import math
import sys

#: Region fields that are identifiers or source positions, never clustering features.
NON_FEATURE_KEYS = {"id", "line", "source_line", "column", "col", "header_line", "encoded_id"}


def load_regions(path):
    """Load the region table of a ``<kernel>.regions.json``.

    Accepts ``{"regions": ...}`` or the bare table, either as an ``{"<id>": {...}}``
    mapping or as a list (ids from each entry's ``id`` field, else the list index).
    Non-dict entries are wrapped as ``{"value": v}``.

    Args:
        path (str): JSON file path.

    Returns:
        (dict[int, dict]): Dict mapping integer region id to the region's attribute dict.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "regions" in data:
        data = data["regions"]
    regions = {}
    if isinstance(data, dict):
        for k, v in data.items():
            regions[int(k)] = v if isinstance(v, dict) else {"value": v}
    else:
        for i, v in enumerate(data):
            rid = int(v.get("id", i)) if isinstance(v, dict) else i
            regions[rid] = v if isinstance(v, dict) else {"value": v}
    return regions


def numeric_features(r, prefix=""):
    """Return the numeric (non-bool) fields of a region, flattening nested dicts.

    Keys in :data:`NON_FEATURE_KEYS` are ignored; nested keys are joined with ``.``.

    Args:
        r (dict): Region attribute dict.
        prefix (str): Prefix prepended to every key (used for recursion).

    Returns:
        (dict[str, float]): Dict mapping feature name to float value.
    """
    out = {}
    for k, v in r.items():
        if k in NON_FEATURE_KEYS:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[prefix + k] = float(v)
        elif isinstance(v, dict):
            out.update(numeric_features(v, prefix + k + "."))
    return out


def kmeans(points, k, iters=50):
    """Cluster points with a deterministic k-means (farthest-point init) on z-scored features.

    Args:
        points (list[list[float]]): List of equal-length numeric feature vectors (non-empty).
        k (int): Number of clusters (clamped to ``[1, len(points)]``).
        iters (int): Maximum number of iterations.

    Returns:
        (list[int]): Cluster label per point, renumbered in order of first appearance.
    """
    n = len(points)
    k = max(1, min(k, n))
    dims = len(points[0])
    mean = [sum(p[d] for p in points) / n for d in range(dims)]
    std = [math.sqrt(sum((p[d] - mean[d]) ** 2 for p in points) / n) or 1.0 for d in range(dims)]
    z = [[(p[d] - mean[d]) / std[d] for d in range(dims)] for p in points]
    dist = lambda a, b: sum((x - y) ** 2 for x, y in zip(a, b))
    cent = [z[0]]
    while len(cent) < k:
        cent.append(max(z, key=lambda p: min(dist(p, c) for c in cent)))
    lab = [0] * n
    for _ in range(iters):
        new = [min(range(len(cent)), key=lambda c: dist(p, cent[c])) for p in z]
        if new == lab and _:
            break
        lab = new
        for c in range(len(cent)):
            mem = [z[i] for i in range(n) if lab[i] == c]
            if mem:
                cent[c] = [sum(m[d] for m in mem) / len(mem) for d in range(dims)]
    # renumber types in order of first appearance for stable output
    remap, out = {}, []
    for l in lab:
        remap.setdefault(l, len(remap))
        out.append(remap[l])
    return out


def main():
    """Parse the command line and write the ``region_id type_id`` file.

    The ``features`` strategy falls back to ``function`` when the regions share no
    numeric feature or there is only one region.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("regions_json")
    ap.add_argument("-o", "--out", default="-")
    ap.add_argument("--strategy", choices=["features", "function", "identity"], default="features")
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    regions = load_regions(a.regions_json)
    ids = sorted(regions)
    types = {i: i for i in ids}
    used = "identity"
    if a.strategy == "features" and ids:
        feats = [numeric_features(regions[i]) for i in ids]
        keys = sorted(set.intersection(*(set(f) for f in feats))) if feats else []
        if keys and len(ids) > 1:
            pts = [[f[k] for k in keys] for f in feats]
            for i, t in zip(ids, kmeans(pts, a.k)):
                types[i] = t
            used = "features(%s)" % ",".join(keys)
        else:
            a.strategy = "function"
            print("region_types: no common numeric features; falling back to --strategy function",
                  file=sys.stderr)
    if a.strategy == "function":
        fn = {}
        for i in ids:
            name = regions[i].get("function", str(i))
            types[i] = fn.setdefault(name, len(fn))
        used = "function"
    out = sys.stdout if a.out == "-" else open(a.out, "w")
    out.write("# region_id type_id  (strategy: %s, source: %s)\n" % (used, a.regions_json))
    for i in ids:
        out.write("%d %d\n" % (i, types[i]))
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
