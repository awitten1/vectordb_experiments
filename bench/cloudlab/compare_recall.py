#!/usr/bin/env python3
"""Fail unless two arms give the same recall at every search value.

  python3 bench/cloudlab/compare_recall.py <results.jsonl> <ref-label-prefix> <new-label-prefix> [--tol 0.002]

Used to check that a re-implementation (pgvectorfast + tq4) searches exactly
like the original (pgvector_ntq): same graph, same distance code, so recall at
each ef_search must agree up to tie order. Rounds are averaged per value.
"""
import json
import sys
from collections import defaultdict


def by_value(rows, prefix):
    acc = defaultdict(list)
    for r in rows:
        if (r.get("label") or "").startswith(prefix) and r.get("clients") == 1:
            acc[r["search_value"]].append(r["recall"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main() -> None:
    args = sys.argv[1:]
    tol = 0.002
    if "--tol" in args:
        i = args.index("--tol")
        tol = float(args[i + 1])
        del args[i:i + 2]
    path, ref_p, new_p = args
    rows = [json.loads(line) for line in open(path)]
    ref, new = by_value(rows, ref_p), by_value(rows, new_p)
    if not ref or not new:
        sys.exit(f"error: no rows for {ref_p if not ref else new_p} in {path}")
    worst = 0.0
    print(f"{'value':>8} {ref_p:>14} {new_p:>16}")
    for k in sorted(new):
        if k not in ref:
            sys.exit(f"error: {ref_p} has no row at {k}")
        d = abs(new[k] - ref[k])
        worst = max(worst, d)
        print(f"{k:>8} {ref[k]:>14.4f} {new[k]:>16.4f}")
    if worst > tol:
        sys.exit(f"error: recall differs by up to {worst:.4f} (> {tol})")
    print(f"recall matches: worst difference {worst:.4f}")


if __name__ == "__main__":
    main()
