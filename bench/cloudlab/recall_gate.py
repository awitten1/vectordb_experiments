#!/usr/bin/env python3
"""Fail unless two result labels give the same 1-client recall at every
ef_search they share (within --tol). Used by hn_pipnn_stock.sh: stock pgvector
and the pgvectorfast fork run the same float layer-0 search on the same
imported graph, so their recall must agree.

  recall_gate.py <label> <reference label> [--tol 0.002] [--results FILE]
"""
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("ref")
ap.add_argument("--tol", type=float, default=0.002)
ap.add_argument("--results", default="bench/results/hackernews_minilm_384.jsonl")
args = ap.parse_args()

with open(args.results) as f:
    rows = [json.loads(line) for line in f if line.strip()]


def by_ef(label):
    return {r["search_value"]: r["recall"] for r in rows
            if r.get("label") == label and r.get("clients") == 1}


a, b = by_ef(args.label), by_ef(args.ref)
common = sorted(set(a) & set(b))
if len(common) < 4:
    sys.exit(f"error: only {len(common)} ef_search values in common between {args.label} and {args.ref}")
worst = 0.0
for ef in common:
    d = a[ef] - b[ef]
    worst = max(worst, abs(d))
    print(f"ef_search {ef:5}: {args.label} {a[ef]:.4f}  {args.ref} {b[ef]:.4f}  diff {d:+.4f}")
print(f"largest difference: {worst:.4f} (allowed {args.tol})")
if worst > args.tol:
    sys.exit(f"error: {args.label} does not match {args.ref}: not the same search on the same graph")
