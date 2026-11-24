#!/usr/bin/env python3
"""Is it the graph? QPS at matched recall for every arm of hn_pipnn.sh and
hn_pipnn_ginger.sh, next to the main study's Ginger and pgvectorfast + tq4.

  python3 bench/cloudlab/pipnn_compare.py

Reads bench/profiles/hn-pipnn/{client-results,ginger-graph}.jsonl,
bench/profiles/hn-pipnn/counts-*-ef*.{csv,sum.txt} and the main results in
bench/results/*/hackernews_minilm_384.jsonl. Interpolates log(QPS) and
log(width) against -log10(1 - recall), as the main study does.
"""
import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PD = ROOT / "bench/profiles/hn-pipnn"
LEVELS = (0.90, 0.95, 0.97, 0.98, 0.99)

ARMS = [  # (key, label in the jsonl, description)
    ("hnsw", "piphnsw-fork", "pgvectorfast fork, HNSW graph (m=16)"),
    ("pipnn", "pippipnn-fork", "pgvectorfast fork, PiPNN graph (single layer, degree <= 64)"),
    ("hnswhp", "piphp-hnsw-fork", "pgvectorfast fork, HNSW graph, huge pages"),
    ("pipnnhp", "piphp-pipnn-fork", "pgvectorfast fork, PiPNN graph, huge pages"),
    ("gfloat", "ginger-graph-float", "Ginger on the same PiPNN graph, float search, no re-rank"),
    ("gnative", "ginger-graph-native", "Ginger on the same PiPNN graph, native search (4-bit + 12-bit re-rank)"),
    ("ginger", "ginger-recommended", "Ginger on its own PiPNN graph (main study)"),
    ("fastntq", "thpon-fastntq", "pgvectorfast + tq4, huge pages, 2 MB translation arrays (main study)"),
]


def load_rows():
    rows = []
    for f in [PD / "client-results.jsonl", PD / "ginger-graph.jsonl",
              *map(Path, glob.glob(str(ROOT / "bench/results/*/hackernews_minilm_384.jsonl")))]:
        if f.exists():
            rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def curves(rows):
    out = {}
    for r in rows:
        c = r.get("clients") or r.get("threads")
        for key, label, _ in ARMS:
            if r.get("label") == label and c in (1, 40):
                out.setdefault((key, c), {})[r["search_value"]] = (r["recall"], r["qps"])
    return out


def x(r):
    return -math.log10(1 - r)


def at(pts, R):
    p = sorted((v[0], w, v[1]) for w, v in pts.items())
    for (r1, w1, q1), (r2, w2, q2) in zip(p, p[1:]):
        if r1 <= R <= r2 and r2 > r1:
            t = (x(R) - x(r1)) / (x(r2) - x(r1))
            return math.exp(math.log(w1) + t * math.log(w2 / w1)), math.exp(math.log(q1) + t * math.log(q2 / q1))
    return None


def counts():
    out = {}
    for g in ("hnsw", "pipnn"):
        for csv in sorted(PD.glob(f"counts-{g}-ef*.csv")):
            ef = int(csv.stem.split("ef")[1])
            n = {}
            for line in csv.read_text().splitlines():
                parts = line.split(",")
                if len(parts) > 2 and "probe_vectorfast" in parts[2]:
                    n[parts[2].split(":")[1]] = float(parts[0])
            s = (PD / f"counts-{g}-ef{ef}.sum.txt").read_text().split()
            q = n["hnswrescan"]
            out[(g, ef)] = (n["HnswLoadUnvisitedRaw"] / q, float(s[1].split("=")[1]) / q)
    return out


def ginger_stats(rows, label):
    return {r["search_value"]: r["stats"] for r in rows if r.get("label") == label and "stats" in r}


def main():
    rows = load_rows()
    cv = curves(rows)
    print("## Recall and QPS by width (1 client / thread)\n")
    widths = sorted({w for (k, c), pts in cv.items() if c == 1 for w in pts})
    keys = [k for k, _, _ in ARMS if (k, 1) in cv]
    print("| width | " + " | ".join(keys) + " |")
    print("|---:|" + "---:|" * len(keys))
    for w in widths:
        cells = []
        for k in keys:
            v = cv[(k, 1)].get(w)
            cells.append(f"{v[0]:.4f} / {v[1]:,.0f}" if v else "")
        print(f"| {w} | " + " | ".join(cells) + " |")
    print()
    for c in (1, 40):
        print(f"## QPS at matched recall, {c} client(s) / thread(s) (width in brackets)\n")
        ks = [k for k, _, _ in ARMS if (k, c) in cv]
        print("| recall | " + " | ".join(ks) + " | pipnn / hnsw | gfloat / pipnn | gnative / pipnn |")
        print("|---:|" + "---:|" * (len(ks) + 3))
        for R in LEVELS:
            v = {k: at(cv[(k, c)], R) for k in ks}
            cells = [f"{v[k][1]:,.0f} ({v[k][0]:.0f})" if v[k] else "–" for k in ks]
            def ratio(a, b):
                return f"{v[a][1] / v[b][1]:.2f}×" if v.get(a) and v.get(b) else "–"
            print(f"| {R} | " + " | ".join(cells) + f" | {ratio('pipnn', 'hnsw')} | {ratio('gfloat', 'pipnn')} | {ratio('gnative', 'pipnn')} |")
        print()
    cn = counts()
    gf = ginger_stats(rows, "ginger-graph-float")
    gn = ginger_stats(rows, "ginger-graph-native")
    print("## Work per query (pgvectorfast: uprobes; Ginger: its statistics)\n")
    print("| width | hnsw hops · distances | pipnn hops · distances | Ginger float on the same graph | Ginger native |")
    print("|---:|---:|---:|---:|---:|")
    for ef in sorted({e for _, e in cn}):
        def g(st):
            s = st.get(ef)
            return f"{s['nodes_visited']:.1f} · {s['distance_computations']:,.0f}" if s else ""
        h, p = cn.get(("hnsw", ef)), cn.get(("pipnn", ef))
        print(f"| {ef} | {h[0]:.1f} · {h[1]:,.0f} | {p[0]:.1f} · {p[1]:,.0f} | {g(gf)} | {g(gn)} |")


if __name__ == "__main__":
    main()
