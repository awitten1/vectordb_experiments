#!/usr/bin/env python3
"""Check that a Ginger run really used (or really didn't use) transparent huge
pages, then append its rows to the results file under a new label.

  ginger_thp_promote.py <rows.jsonl> <sampler.csv> <label> <thp|4k>

<sampler.csv> holds samples of the ginger_bench process's smaps_rollup
(t, rss_kb, anon_kb, anon_huge_kb), written by hn_ginger_thp.sh while the run
went on. A `thp` run passes if, at its largest, at least 80% of its anonymous
memory was on huge pages; a `4k` run passes if at most 1% was. Rows of a run
that fails are not appended, so they can't be mistaken for a valid arm.
"""
import csv
import json
import sys

RESULTS = "bench/results/hackernews_minilm_384.jsonl"

rows_path, sampler_path, label, mode = sys.argv[1:5]
with open(sampler_path) as f:
    samples = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f)]
if len(samples) < 3:
    sys.exit(f"error: only {len(samples)} samples in {sampler_path}; the run was not observed")

anon = max(s["anon_kb"] for s in samples)
huge = max(s["anon_huge_kb"] for s in samples)
share = huge / anon if anon else 0.0
print(f"{len(samples)} samples; peak anonymous {anon / 2**20:.1f} GB, "
      f"peak on huge pages {huge / 2**20:.1f} GB ({share:.1%})")
if mode == "thp" and share < 0.80:
    sys.exit("error: under 80% of Ginger's memory was on huge pages; rows not kept")
if mode == "4k" and share > 0.01:
    sys.exit("error: the 4 KB control had memory on huge pages; rows not kept")

with open(rows_path) as f:
    rows = [json.loads(line) for line in f if line.strip()]
if not rows:
    sys.exit(f"error: no rows in {rows_path}")
with open(RESULTS, "a") as out:
    for r in rows:
        r["label"] = label
        r["pages"] = "huge pages (THP via jemalloc thp:always)" if mode == "thp" else "4 KB"
        r["anon_gb"] = round(anon / 2**20, 2)
        r["anon_huge_gb"] = round(huge / 2**20, 2)
        out.write(json.dumps(r) + "\n")
print(f"appended {len(rows)} rows to {RESULTS} as {label}")
