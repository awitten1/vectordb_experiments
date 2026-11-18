#!/usr/bin/env python3
"""Recall/QPS curves for every index *variant* on one dataset, as a standalone SVG.

Usage:
  bench/plot_variants.py bigann_18000000                 # 1 client
  bench/plot_variants.py bigann_18000000 --clients 8     # only variants with 8-client sweeps
  bench/plot_variants.py bigann_18000000 -o /tmp/out.svg

plot_sweep.py keys a series on the index type, which can't tell apart stock
pgvector, pgvectorfast and the RaBitQ builds -- they are all `hnsw`. Here each
series is a filter over result rows (index, label, rabitq_bits, vector_so).

Rows that repeat a series at the same search value (the stock/fast rounds of
the pgvectorfast phase, the two plain-HNSW sweeps of the rabitq phase) are
averaged; for PRISM and IVFFlat the latest row per value wins, since their
early sweeps include points where the planner flipped plans.

The x axis is -log10(1 - recall) ("nines"), so 0.9, 0.99 and 0.999 are evenly
spaced: on a linear axis everything above 0.97 lands in the last few pixels.
Points under MIN_RECALL are dropped and the footer says so.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results"
MIN_RECALL = 0.5
MAX_RECALL = 0.999

# Categorical slots 1-8 of the reference palette, in fixed order. Validated with
# the dataviz skill's validator (adjacent pairs, all checks pass both modes):
#   node scripts/validate_palette.js "<light slots>" --mode light --surface "#fcfcfb"
#   node scripts/validate_palette.js "<dark slots>"  --mode dark  --surface "#1a1a19"
# Light aqua, yellow and magenta are under 3:1 on the surface, so the relief
# rule applies: every series is direct-labelled at its line end, and the
# FINDINGS.md tables carry the numbers.
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
CHROME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835"},
}


def is_label(prefix):
    return lambda r: (r.get("label") or "").startswith(prefix)


def rq(bits):
    return lambda r: (r.get("build_params") or {}).get("rabitq_bits") == bits


# Order = colour slot. A series keeps its slot whether or not it has data, so a
# hue always means the same variant across --clients settings. `dash` marks the
# RaBitQ family as a second channel next to hue.
SERIES = [
    {"name": "pgvector 0.8.7 HNSW", "short": "pgvector 0.8.7", "index": "hnsw", "agg": "mean",
     "match": is_label("pvf-stock")},
    {"name": "PRISM", "short": "PRISM", "index": "prism", "agg": "latest",
     "match": lambda r: True},
    {"name": "pgvector 0.8.7 IVFFlat", "short": "IVFFlat", "index": "ivfflat", "agg": "latest",
     "match": lambda r: True},
    {"name": "pgvectorfast HNSW", "short": "pgvectorfast", "index": "hnsw", "agg": "mean",
     "match": is_label("pvf-fast")},
    {"name": "RaBitQ 4-bit HNSW", "short": "RaBitQ 4-bit", "index": "hnsw", "agg": "mean",
     "match": rq(4), "dash": True},
    {"name": "RaBitQ 2-bit HNSW", "short": "RaBitQ 2-bit", "index": "hnsw", "agg": "mean",
     "match": rq(2), "dash": True},
    {"name": "RaBitQ 1-bit HNSW", "short": "RaBitQ 1-bit", "index": "hnsw", "agg": "mean",
     "match": rq(1), "dash": True},
    {"name": "rabitq branch, plain HNSW (1de2e56)", "short": "rabitq branch, plain", "index": "hnsw",
     "agg": "mean",
     "match": lambda r: (r.get("pgvector_git") or "").startswith("1de2e56")
     and not r.get("label") and not (r.get("build_params") or {}).get("rabitq_bits")},
]

BIGANN_NOTE = ("Measured: PRISM, IVFFlat 2026-11-05 · RaBitQ, rabitq-branch plain HNSW 2026-11-13 · "
               "pgvector 0.8.7 vs pgvectorfast back-to-back 2026-11-16, rounds averaged.")

# The CloudLab four-way comparison. Rows come from two nodes (see load()); the
# stock-pgvector rows measured on the baseline host are drawn as a dashed line
# in pgvector's own hue, so node-to-node agreement reads as two overlapping
# lines. Ginger runs in-process: its curve does not pay the Postgres round
# trip that the `rtt` rows of the overhead phase measure.
# Which Postgres rows a view uses. At 1 client the client shares the server's
# machine (node-0), as in the original sweeps; from 8 clients up it runs on
# node-1 over the private LAN (labels rc-<arm>-remote at 8, sc-<arm> at 12/16),
# because a local client competes with the backends for cores and cost both
# HNSW arms 15-30% of their throughput at moderate recall. A row's `host` is
# the client's machine, so the 1-client rows are picked by host node-0.
def pg_arm(arm):
    return lambda r: (
        (r["clients"] == 1 and is_label(f"pvf-{arm}")(r) and (r.get("host") or "").startswith("node-0"))
        or (r["clients"] >= 8 and r.get("label") in (f"rc-{arm}-remote", f"sc-{arm}")))


def prism_rows(r):
    return ((r["clients"] == 1 and r.get("label") == "prism-solo")
            or (r["clients"] >= 8 and r.get("label") == "sc-prism"))


def ginger(cfg):
    # ginger_bench writes the config *name* into the label (ginger-<cfg>) and
    # the resolved GingerSearcherConfig proto text into `ginger_config`.
    return lambda r: r.get("engine") == "ginger" and r.get("label") == f"ginger-{cfg}"


# node-1's own stock pgvector run is left off: it only checked that the two
# nodes are interchangeable (they agree within 1-5% at matched recall; see
# FINDINGS-dbpedia.md), so it would plot the same curve twice.
DBPEDIA_SERIES = [
    {"name": "pgvector 0.8.7 HNSW", "short": "pgvector 0.8.7", "index": "hnsw", "agg": "mean",
     "slot": 1, "match": pg_arm("stock")},
    {"name": "PRISM (pg_vectorsearch)", "short": "PRISM", "index": "prism", "agg": "mean",
     "slot": 2, "match": prism_rows},
    {"name": "Ginger (PCA, 4+8-bit TQ)", "short": "Ginger", "index": None,
     "agg": "mean", "slot": 3, "match": ginger("recommended")},
    {"name": "pgvectorfast HNSW", "short": "pgvectorfast", "index": "hnsw", "agg": "mean",
     "slot": 4, "match": pg_arm("fast")},
    {"name": "Ginger, float rerank", "short": "Ginger float-rerank", "index": None,
     "agg": "mean", "slot": 3, "dash": True, "match": ginger("float-rerank")},
]
DBPEDIA_NOTE = ("m510, Xeon D-1548 @ 2.0 GHz pinned, AVX2. Ginger in-process. Postgres client local at 1 "
                "client, on a 2nd node over 10 GbE from 8 up. Rounds averaged.")

BIGANN_TITLE = "18M BIGANN (SIFT), {n} index variants"
DBPEDIA_TITLE = "DBpedia OpenAI 1M (ada-002, 1536-d), {n} variants"
SERIES_SETS = {"bigann": (SERIES, BIGANN_NOTE, BIGANN_TITLE),
               "dbpedia": (DBPEDIA_SERIES, DBPEDIA_NOTE, DBPEDIA_TITLE)}

W, H = 1000, 660
PAD = {"l": 78, "r": 200, "t": 128, "b": 106}


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nines(recall: float) -> float:
    return -math.log10(1 - min(recall, 0.99999))


def load(dataset: str, clients: int, series_spec=SERIES):
    """Rows from bench/results/<dataset>.jsonl and from every per-node copy
    under bench/results/<node>/ (what bench/cloudlab/sync.sh pull creates).
    Each row is tagged with `_node`, the directory it came from."""
    paths = [RESULTS_DIR / f"{dataset}.jsonl", *sorted(RESULTS_DIR.glob(f"*/{dataset}.jsonl"))]
    paths = [p for p in paths if p.exists()]
    if not paths:
        sys.exit(f"no results for {dataset} under {RESULTS_DIR}")
    rows = []
    for path in paths:
        node = "local" if path.parent == RESULTS_DIR else path.parent.name
        rows += [{**json.loads(l), "_node": node} for l in path.read_text().splitlines() if l.strip()]
    # Deadline-based runs only: one pass over the queries is too short to time.
    rows = [r for r in rows if r.get("queries_executed") and r.get("search_value") is not None
            and r.get("clients", 1) == clients]
    out, dropped = [], 0
    for pos, spec in enumerate(series_spec, start=1):
        slot = spec.get("slot", pos)
        hits = [r for r in rows if spec["index"] in (None, r.get("index")) and spec["match"](r)]
        if not hits:
            continue
        by_val = defaultdict(list)
        for r in hits:
            by_val[r["search_value"]].append(r)
        pts = []
        for v, rs in by_val.items():
            if spec["agg"] == "latest":
                rs = [max(rs, key=lambda r: r["timestamp"])]
            p = {"value": v, "recall": sum(r["recall"] for r in rs) / len(rs),
                 "qps": sum(r["qps"] for r in rs) / len(rs), "n": len(rs),
                 "qps_min": min(r["qps"] for r in rs), "qps_max": max(r["qps"] for r in rs)}
            if p["recall"] < MIN_RECALL:
                dropped += 1
                continue
            pts.append(p)
        pts.sort(key=lambda p: p["value"])
        if pts:
            out.append({**spec, "slot": slot, "rows": pts, "param": hits[0].get("search_param", ""),
                        "meta": hits[-1]})
    return out, dropped


def log_ticks(lo, hi):
    out, e = [], math.floor(math.log10(lo))
    while 10 ** e <= hi * 10:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lo <= v <= hi:
                out.append(v)
        e += 1
    return out


def spread(ys, lo, hi, gap):
    """Nudge label baselines apart by at least `gap`, keeping them inside [lo, hi]."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = [ys[i] for i in order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + gap)
    overflow = placed[-1] - hi if placed else 0
    if overflow > 0:
        placed = [y - overflow for y in placed]
        for k in range(len(placed) - 2, -1, -1):
            placed[k] = min(placed[k], placed[k + 1] - gap)
    placed = [max(lo, y) for y in placed]
    res = [0.0] * len(ys)
    for k, i in enumerate(order):
        res[i] = placed[k]
    return res


def build_svg(series, dataset, clients, dropped, note1=BIGANN_NOTE, title=BIGANN_TITLE):
    qpss = [p["qps"] for s in series for p in s["rows"]]
    x0, x1 = nines(MIN_RECALL), nines(MAX_RECALL)
    y0, y1 = min(qpss) / 1.4, max(qpss) * 1.4
    px0, px1 = PAD["l"], W - PAD["r"]
    py0, py1 = H - PAD["b"], PAD["t"]

    def sx(r): return px0 + (nines(r) - x0) / (x1 - x0) * (px1 - px0)
    def sy(q): return py0 + (math.log10(q) - math.log10(y0)) / (
        math.log10(y1) - math.log10(y0)) * (py1 - py0)

    m = series[0]["meta"]
    p, a = [], None
    a = p.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
      f'role="img" aria-labelledby="t d" font-family="Inter, ui-sans-serif, system-ui, '
      f'-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif">')
    a(f'<title id="t">Recall vs throughput by index variant, {esc(dataset)}, {clients} client(s)</title>')
    desc = "; ".join(s["name"] + ": " + ", ".join(
        f'{s["param"].split(".")[-1]} {q["value"]} recall {q["recall"]:.4f} {q["qps"]:,.0f} QPS'
        for q in s["rows"]) for s in series)
    a(f'<desc id="d">{esc(desc)}</desc>')

    def toks(mode, cols):
        c = CHROME[mode]
        return ";".join([f"--{k}:{v}" for k, v in c.items()]
                        + [f"--s{i}:{h}" for i, h in enumerate(cols, start=1)])
    a("<style>")
    a(f':root{{{toks("light", LIGHT)}}}')
    a(f'@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{{toks("dark", DARK)}}}}}')
    a(f':root[data-theme="dark"]{{{toks("dark", DARK)}}}')
    a('.bg{fill:var(--surface)} .ttl{fill:var(--ink);font-size:19px;font-weight:600}'
      '.sub{fill:var(--ink2);font-size:12.5px} .foot{fill:var(--muted);font-size:11.5px}'
      '.axlbl{fill:var(--ink2);font-size:12.5px;font-weight:500}'
      '.tick{fill:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums}'
      '.grid{stroke:var(--grid);stroke-width:1} .axis{stroke:var(--axis);stroke-width:1}'
      '.lgnd{fill:var(--ink);font-size:12px;font-weight:500}'
      '.end{fill:var(--ink);font-size:12px;font-weight:600}'
      '.lead{stroke:var(--axis);stroke-width:1;fill:none}'
      '.dot:hover{r:7}')
    for i in range(1, 9):
        a(f'.l{i}{{fill:none;stroke:var(--s{i});stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}'
          f'.d{i}{{fill:var(--s{i});stroke:var(--surface);stroke-width:2}}')
    a('.dash{stroke-dasharray:7 4}')
    a("</style>")
    a(f'<rect class="bg" x="0" y="0" width="{W}" height="{H}"/>')
    a(f'<text class="ttl" x="{PAD["l"]}" y="34">Recall vs throughput — '
      f'{esc(title.format(n=len(series)))}, {clients} client{"s" if clients > 1 else ""}</text>')
    a(f'<text class="sub" x="{PAD["l"]}" y="55">{m["rows"]:,} × {m["dim"]}-d vectors · '
      f'indexes fully resident in shared_buffers · Postgres {esc(m.get("postgres_version"))} · '
      f'k={m.get("k")} · {m.get("duration_s") or 60:g}s per point · hover a point for its value</text>')

    # legend, two rows of up to four
    for k, s in enumerate(series):
        lx = PAD["l"] + (k % 4) * 215
        ly = 80 + (k // 4) * 20
        dash = " dash" if s.get("dash") else ""
        a(f'<line class="l{s["slot"]}{dash}" x1="{lx}" y1="{ly}" x2="{lx + 24}" y2="{ly}"/>')
        a(f'<circle class="d{s["slot"]}" cx="{lx + 12}" cy="{ly}" r="4.5"/>')
        a(f'<text class="lgnd" x="{lx + 32}" y="{ly + 4}">{esc(s["name"])}</text>')

    for v in log_ticks(y0, y1):
        y = sy(v)
        a(f'<line class="grid" x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}"/>')
        a(f'<text class="tick" x="{px0 - 10}" y="{y + 4:.1f}" text-anchor="end">{v:,.0f}</text>')
    for r in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999):
        x = sx(r)
        a(f'<line class="grid" x1="{x:.1f}" y1="{py0}" x2="{x:.1f}" y2="{py1}"/>')
        a(f'<text class="tick" x="{x:.1f}" y="{py0 + 20}" text-anchor="middle">{r:g}</text>')
    a(f'<line class="axis" x1="{px0}" y1="{py0}" x2="{px1}" y2="{py0}"/>')
    a(f'<line class="axis" x1="{px0}" y1="{py0}" x2="{px0}" y2="{py1}"/>')
    a(f'<text class="axlbl" x="{(px0 + px1) / 2:.0f}" y="{H - 62}" text-anchor="middle">'
      f'recall@{m.get("k")}  (scale spaces 0.9, 0.99, 0.999 evenly)</text>')
    a(f'<text class="axlbl" transform="translate(22,{(py0 + py1) / 2:.0f}) rotate(-90)" '
      f'text-anchor="middle">queries per second  (log scale)</text>')

    for s in series:
        pts = " ".join(f"{sx(q['recall']):.1f},{sy(q['qps']):.1f}" for q in s["rows"])
        dash = " dash" if s.get("dash") else ""
        a(f'<polyline class="l{s["slot"]}{dash}" points="{pts}"/>')
    for s in series:
        prm = s["param"].split(".")[-1]
        for q in s["rows"]:
            rng = (f' (n={q["n"]}, {q["qps_min"]:,.0f}–{q["qps_max"]:,.0f})' if q["n"] > 1 else "")
            a(f'<circle class="dot d{s["slot"]}" cx="{sx(q["recall"]):.1f}" cy="{sy(q["qps"]):.1f}" r="4.5">'
              f'<title>{esc(s["name"])} — {prm} {q["value"]}: recall {q["recall"]:.4f}, '
              f'{q["qps"]:,.0f} QPS{rng}</title></circle>')

    # direct labels at each line's end, de-overlapped in the right gutter
    ends = [(sx(s["rows"][-1]["recall"]), sy(s["rows"][-1]["qps"])) for s in series]
    ys = spread([y for _, y in ends], py1 + 4, py0 - 2, 15)
    lx = px1 + 14
    for s, (ex, ey), ty in zip(series, ends, ys):
        a(f'<polyline class="lead" points="{ex + 6:.1f},{ey:.1f} {lx - 4:.1f},{ty - 4:.1f}"/>')
        a(f'<text class="end" x="{lx}" y="{ty:.1f}">{esc(s["short"])}</text>')

    sizes = " · ".join(f'{s["short"]} {(s["meta"].get("index_bytes") or 0) / 1024 ** 3:.1f}'
                       for s in series)
    note2 = f"{dropped} points under recall {MIN_RECALL} not shown." if dropped else ""
    for k, line in enumerate((f"Index size (GiB): {sizes}", note1, note2)):
        if line:
            a(f'<text class="foot" x="{PAD["l"]}" y="{H - 40 + 16 * k}">{esc(line)}</text>')
    a("</svg>")
    return "\n".join(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("-o", "--out")
    ap.add_argument("--clients", type=int, default=1)
    ap.add_argument("--series", choices=sorted(SERIES_SETS),
                    help="which variant set to draw (default: bigann for bigann_* datasets, "
                         "dbpedia otherwise)")
    args = ap.parse_args()
    spec, note, title = SERIES_SETS[args.series or ("bigann" if args.dataset.startswith("bigann") else "dbpedia")]
    series, dropped = load(args.dataset, args.clients, spec)
    if not series:
        sys.exit(f"no sweeps for {args.dataset} with clients={args.clients}")
    out = Path(args.out) if args.out else (
        REPO_ROOT / "bench" / "report"
        / f"{args.dataset.replace('_', '-')}-variants-{args.clients}c-recall-qps.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_svg(series, args.dataset, args.clients, dropped, note, title))
    for s in series:
        r = s["rows"]
        print(f'{s["name"]:>38}: {len(r)} pts, recall {r[0]["recall"]:.4f}–{r[-1]["recall"]:.4f}, '
              f'{r[-1]["qps"]:,.0f}–{r[0]["qps"]:,.0f} QPS, n={max(q["n"] for q in r)}')
    print(out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out)


if __name__ == "__main__":
    main()
