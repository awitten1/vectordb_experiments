#!/usr/bin/env python3
"""Recall and QPS of two result labels at each ef_search they share, at one
client count; e.g. the same index measured on two node pairs.

  compare_qps.py <label> <reference label> [--clients 40] [--results FILE]
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("ref")
ap.add_argument("--clients", type=int, default=40)
ap.add_argument("--results", default="bench/results/hackernews_minilm_384.jsonl")
args = ap.parse_args()

with open(args.results) as f:
    rows = [json.loads(line) for line in f if line.strip()]


def get(label):
    return {r["search_value"]: r for r in rows
            if r.get("label") == label and r.get("clients") == args.clients}


a, b = get(args.label), get(args.ref)
for ef in sorted(set(a) & set(b)):
    x, y = a[ef], b[ef]
    print(f"ef_search {ef:5}: recall {x['recall']:.4f} vs {y['recall']:.4f}  "
          f"QPS {x['qps']:8,.0f} vs {y['qps']:8,.0f}  ratio {x['qps'] / y['qps']:.3f}")
if not set(a) & set(b):
    print(f"no ef_search in common between {args.label} and {args.ref} at {args.clients} clients")
