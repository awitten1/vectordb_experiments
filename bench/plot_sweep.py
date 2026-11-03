#!/usr/bin/env python3
"""Plot a recall/QPS tradeoff curve from bench/results/*.jsonl as a standalone SVG.

Usage:
  bench/plot_sweep.py bigann_18000000 [-o bench/report/bigann-18m-hnsw-recall-qps.svg]

Picks the most recent single-client sweep for the dataset, puts recall@k on x and
QPS on a log y axis, and labels each point with its ef_search value. The SVG
carries its own light/dark styling, so it stays readable either way.

Written by hand rather than with matplotlib so the repo keeps a 4-package
requirements.txt; the output is a few hundred lines of plain SVG.
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOW_OVERRIDE = None  # set by --window for runs made before duration_s was recorded
RESULTS_DIR = REPO_ROOT / "bench" / "results"

# Chart chrome and the one categorical slot we need, from the validated
# reference palette (slot 1, blue). Both modes pass all six checks against
# their own surface; see bench/REPORT.md for the validator invocation.
TOKENS = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": "#3987e5",
    },
}

W, H = 900, 540
PAD = {"l": 78, "r": 52, "t": 86, "b": 78}


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def load_sweep(dataset: str, index: str, clients: int):
    path = RESULTS_DIR / f"{dataset}.jsonl"
    if not path.exists():
        sys.exit(f"no results at {path}")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("index") == index and r.get("clients", 1) == clients
            and r.get("search_value") is not None]
    if not rows:
        sys.exit(f"no {index} rows with clients={clients} in {path}")
    # Prefer deadline-based runs (they carry queries_executed); a single pass over
    # the query set is too short to be a trustworthy throughput measurement.
    looped = [r for r in rows if r.get("queries_executed")]
    rows = looped or rows
    # Keep the newest sweep only: group by timestamp, take the latest group.
    newest = max(r["timestamp"] for r in rows)
    same = [r for r in rows if r["timestamp"] == newest]
    if len(same) < 2:  # timestamps differ per point in some runs; fall back to tail
        by_val, out = {}, []
        for r in rows:
            by_val[r["search_value"]] = r
        same = list(by_val.values())
    return sorted(same, key=lambda r: r["search_value"])


def log_ticks(lo: float, hi: float):
    out = []
    e = math.floor(math.log10(lo))
    while 10 ** e <= hi * 10:
        for m in (1, 2, 3, 5):
            v = m * 10 ** e
            if lo <= v <= hi:
                out.append(v)
        e += 1
    return out or [lo, hi]


def fmt_qps(v: float) -> str:
    return f"{v:,.0f}"


def build_svg(rows, dataset: str) -> str:
    recalls = [r["recall"] for r in rows]
    qpss = [r["qps"] for r in rows]

    x0 = math.floor(min(recalls) * 20) / 20          # snap to 0.05
    x1 = 1.0
    y0 = min(qpss) / 1.35
    y1 = max(qpss) * 1.35

    px0, px1 = PAD["l"], W - PAD["r"]
    py0, py1 = H - PAD["b"], PAD["t"]

    def sx(v): return px0 + (v - x0) / (x1 - x0) * (px1 - px0)
    def sy(v): return py0 + (math.log10(v) - math.log10(y0)) / (
        math.log10(y1) - math.log10(y0)) * (py1 - py0)

    meta = rows[0]
    bp = meta.get("build_params") or {}
    idx_gib = (meta.get("index_bytes") or 0) / 1024 ** 3
    window = meta.get("duration_s") or WINDOW_OVERRIDE
    nq = meta.get("num_queries")
    ex = [r["queries_executed"] for r in rows if r.get("queries_executed")]

    sub = (f'{meta["rows"]:,} × {meta["dim"]}-d vectors · HNSW m={bp.get("m")}, '
           f'ef_construction={bp.get("ef_construction")} · {idx_gib:.1f} GiB index, fully resident '
           f'· 1 client')
    if window:
        sub += f' · {window:g}s per point'
    foot = (f'pgvector {meta.get("pgvector_version")} · Postgres {meta.get("postgres_version")} '
            f'· k={meta.get("k")} · {nq:,} distinct queries')
    if ex:
        foot += f', {min(ex):,}–{max(ex):,} executed per point'

    # x ticks every 0.05, labelled every 0.1
    xt = []
    v = x0
    while v <= x1 + 1e-9:
        xt.append(round(v, 2))
        v += 0.05

    p = []
    a = p.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
      f'role="img" aria-labelledby="t d" font-family="Inter, ui-sans-serif, system-ui, '
      f'-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif">')
    a(f'<title id="t">Recall vs throughput for HNSW on {esc(dataset)}</title>')
    desc = "; ".join(f'ef_search {r["search_value"]}: recall {r["recall"]:.4f}, {r["qps"]:,.0f} QPS'
                     for r in rows)
    a(f'<desc id="d">Single-client recall/throughput tradeoff. {esc(desc)}.</desc>')

    # light/dark tokens; the media query covers the OS setting, the data-theme
    # scopes let an explicit choice win either way
    L, D = TOKENS["light"], TOKENS["dark"]
    def block(t, scope):
        return (f'{scope}{{--surface:{t["surface"]};--ink:{t["ink"]};--ink2:{t["ink2"]};'
                f'--muted:{t["muted"]};--grid:{t["grid"]};--axis:{t["axis"]};'
                f'--series:{t["series"]};}}')
    a("<style>")
    a(block(L, ":root"))
    a(f'@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{'
      f'--surface:{D["surface"]};--ink:{D["ink"]};--ink2:{D["ink2"]};--muted:{D["muted"]};'
      f'--grid:{D["grid"]};--axis:{D["axis"]};--series:{D["series"]};}}}}')
    a(f':root[data-theme="dark"]{{--surface:{D["surface"]};--ink:{D["ink"]};--ink2:{D["ink2"]};'
      f'--muted:{D["muted"]};--grid:{D["grid"]};--axis:{D["axis"]};--series:{D["series"]};}}')
    a('.bg{fill:var(--surface)}')
    a('.ttl{fill:var(--ink);font-size:19px;font-weight:600}')
    a('.sub{fill:var(--ink2);font-size:12.5px}')
    a('.foot{fill:var(--muted);font-size:11.5px}')
    a('.axlbl{fill:var(--ink2);font-size:12.5px;font-weight:500}')
    a('.tick{fill:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums}')
    a('.grid{stroke:var(--grid);stroke-width:1}')
    a('.axis{stroke:var(--axis);stroke-width:1}')
    a('.line{fill:none;stroke:var(--series);stroke-width:2;stroke-linejoin:round;'
      'stroke-linecap:round}')
    a('.dot{fill:var(--series);stroke:var(--surface);stroke-width:2}')
    a('.plbl{fill:var(--ink);font-size:11.5px;font-weight:600;'
      'font-variant-numeric:tabular-nums}')
    a("</style>")

    a(f'<rect class="bg" x="0" y="0" width="{W}" height="{H}"/>')
    a(f'<text class="ttl" x="{PAD["l"]}" y="34">Recall vs throughput — HNSW, 18M BIGANN (SIFT) vectors</text>')
    a(f'<text class="sub" x="{PAD["l"]}" y="55">{esc(sub)}</text>')

    # y gridlines + ticks
    for v in log_ticks(y0, y1):
        y = sy(v)
        a(f'<line class="grid" x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}"/>')
        a(f'<text class="tick" x="{px0 - 10}" y="{y + 4:.1f}" text-anchor="end">{fmt_qps(v)}</text>')
    # x ticks
    for v in xt:
        x = sx(v)
        a(f'<line class="axis" x1="{x:.1f}" y1="{py0}" x2="{x:.1f}" y2="{py0 + 5}"/>')
        if abs(round(v * 10) / 10 - v) < 1e-9:
            a(f'<text class="tick" x="{x:.1f}" y="{py0 + 20}" text-anchor="middle">{v:.1f}</text>')
    a(f'<line class="axis" x1="{px0}" y1="{py0}" x2="{px1}" y2="{py0}"/>')
    a(f'<line class="axis" x1="{px0}" y1="{py0}" x2="{px0}" y2="{py1}"/>')

    a(f'<text class="axlbl" x="{(px0 + px1) / 2:.0f}" y="{H - 32}" text-anchor="middle">'
      f'recall@{meta.get("k")}  (fraction of true nearest neighbours returned)</text>')
    a(f'<text class="axlbl" transform="translate(22,{(py0 + py1) / 2:.0f}) rotate(-90)" '
      f'text-anchor="middle">queries per second  (log scale)</text>')

    pts = " ".join(f"{sx(r['recall']):.1f},{sy(r['qps']):.1f}" for r in rows)
    a(f'<polyline class="line" points="{pts}"/>')

    for r in rows:
        x, y = sx(r["recall"]), sy(r["qps"])
        a(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
        # The curve enters each point from the upper left and leaves to the
        # lower right, so the free quadrants are upper-right and lower-left.
        # Default to upper-right; where that would overflow, drop to lower-left
        # rather than upper-left, which the incoming line already occupies.
        if x > px1 - 64:
            a(f'<text class="plbl" x="{x - 11:.1f}" y="{y + 17:.1f}" text-anchor="end">'
              f'ef {r["search_value"]}</text>')
        else:
            a(f'<text class="plbl" x="{x + 11:.1f}" y="{y - 10:.1f}">ef {r["search_value"]}</text>')

    a(f'<text class="foot" x="{PAD["l"]}" y="{H - 10}">{esc(foot)}</text>')
    a("</svg>")
    return "\n".join(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="dataset name, e.g. bigann_18000000")
    ap.add_argument("-o", "--out", help="output .svg path")
    ap.add_argument("--index", default="hnsw")
    ap.add_argument("--clients", type=int, default=1)
    ap.add_argument("--window", type=float,
                    help="seconds per sweep point, for results recorded before "
                         "duration_s was saved in the metadata")
    args = ap.parse_args()

    global WINDOW_OVERRIDE
    WINDOW_OVERRIDE = args.window

    rows = load_sweep(args.dataset, args.index, args.clients)
    svg = build_svg(rows, args.dataset)
    out = Path(args.out) if args.out else (
        REPO_ROOT / "bench" / "report" / f"{args.dataset.replace('_', '-')}-{args.index}-recall-qps.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    print(f"{out.relative_to(REPO_ROOT)}  ({len(rows)} points, "
          f"recall {rows[0]['recall']:.4f}–{rows[-1]['recall']:.4f}, "
          f"{rows[-1]['qps']:,.0f}–{rows[0]['qps']:,.0f} QPS)")


if __name__ == "__main__":
    main()
