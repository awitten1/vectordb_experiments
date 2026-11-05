#!/usr/bin/env python3
"""Plot recall/QPS tradeoff curves from bench/results/*.jsonl as a standalone SVG.

Usage:
  bench/plot_sweep.py bigann_18000000                      # hnsw + prism
  bench/plot_sweep.py bigann_18000000 --index hnsw         # one series
  bench/plot_sweep.py bigann_18000000 -o /tmp/out.svg

Picks the most recent single-client sweep per index, puts recall@k on x and QPS
on a log y axis, and labels each point with its search parameter. The SVG
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

# Chart chrome plus categorical slots 1 and 2 from the validated reference
# palette (blue, orange). Both modes pass all six checks against their own
# surface; re-check with the dataviz skill's validator before changing a hue:
#   node scripts/validate_palette.js "#2a78d6,#eb6834" --mode light --surface "#fcfcfb"
#   node scripts/validate_palette.js "#3987e5,#d95926" --mode dark  --surface "#1a1a19"
TOKENS = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series1": "#2a78d6",
        "series2": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series1": "#3987e5",
        "series2": "#d95926",
    },
}

# Per-index presentation: display name and the prefix for each point's label.
INDEX_STYLE = {
    "hnsw": {"name": "pgvector HNSW", "prefix": "ef"},
    "prism": {"name": "PRISM", "prefix": "np"},
    "ivfflat": {"name": "pgvector IVFFlat", "prefix": "pr"},
}

W, H = 900, 560
PAD = {"l": 78, "r": 52, "t": 112, "b": 78}


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
        return []
    # Prefer deadline-based runs (they carry queries_executed); a single pass over
    # the query set is too short to be a trustworthy throughput measurement.
    looped = [r for r in rows if r.get("queries_executed")]
    rows = looped or rows
    # Keep the newest sweep only: group by timestamp, take the latest group.
    newest = max(r["timestamp"] for r in rows)
    same = [r for r in rows if r["timestamp"] == newest]
    if len(same) < 2:  # timestamps differ per point in some runs; fall back to tail
        by_val = {}
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


def gib(n) -> str:
    return f"{(n or 0) / 1024 ** 3:.2f} GiB"


def build_svg(series, dataset: str) -> str:
    recalls = [r["recall"] for s in series for r in s["rows"]]
    qpss = [r["qps"] for s in series for r in s["rows"]]

    x0 = math.floor(min(recalls) * 20) / 20          # snap to 0.05
    x1 = 1.0
    y0 = min(qpss) / 1.35
    y1 = max(qpss) * 1.35

    px0, px1 = PAD["l"], W - PAD["r"]
    py0, py1 = H - PAD["b"], PAD["t"]

    def sx(v): return px0 + (v - x0) / (x1 - x0) * (px1 - px0)
    def sy(v): return py0 + (math.log10(v) - math.log10(y0)) / (
        math.log10(y1) - math.log10(y0)) * (py1 - py0)

    meta = series[0]["rows"][0]
    window = meta.get("duration_s") or WINDOW_OVERRIDE
    # Not "same buffer pool": the HNSW arm runs with the prism index dropped,
    # because raising hnsw.ef_search eventually makes the planner prefer prism.
    sub = (f'{meta["rows"]:,} × {meta["dim"]}-d vectors, same table · index fully '
           f'resident in shared_buffers · 1 client')
    if window:
        sub += f' · {window:g}s per point'

    foot = (f'Postgres {meta.get("postgres_version")} · k={meta.get("k")} · '
            f'{meta.get("num_queries"):,} distinct queries · ')
    foot += " · ".join(
        f'{INDEX_STYLE[s["index"]]["name"]} {gib(s["rows"][0].get("index_bytes"))}'
        for s in series)

    # x ticks every 0.05, labelled every 0.1
    xt, v = [], x0
    while v <= x1 + 1e-9:
        xt.append(round(v, 2))
        v += 0.05

    p = []
    a = p.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
      f'role="img" aria-labelledby="t d" font-family="Inter, ui-sans-serif, system-ui, '
      f'-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif">')
    names = " vs ".join(INDEX_STYLE[s["index"]]["name"] for s in series)
    a(f'<title id="t">Recall vs throughput — {esc(names)} on {esc(dataset)}</title>')
    desc = "; ".join(
        f'{INDEX_STYLE[s["index"]]["name"]} '
        + ", ".join(f'{INDEX_STYLE[s["index"]]["prefix"]} {r["search_value"]}: '
                    f'recall {r["recall"]:.4f}, {r["qps"]:,.0f} QPS' for r in s["rows"])
        for s in series)
    a(f'<desc id="d">Single-client recall/throughput tradeoff. {esc(desc)}.</desc>')

    # light/dark tokens; the media query covers the OS setting, the data-theme
    # scopes let an explicit choice win either way
    L, D = TOKENS["light"], TOKENS["dark"]

    def vars_(t):
        return ";".join(f"--{k}:{v}" for k, v in t.items())

    a("<style>")
    a(f':root{{{vars_(L)};}}')
    a(f'@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{{vars_(D)};}}}}')
    a(f':root[data-theme="dark"]{{{vars_(D)};}}')
    a('.bg{fill:var(--surface)}')
    a('.ttl{fill:var(--ink);font-size:19px;font-weight:600}')
    a('.sub{fill:var(--ink2);font-size:12.5px}')
    a('.foot{fill:var(--muted);font-size:11.5px}')
    a('.axlbl{fill:var(--ink2);font-size:12.5px;font-weight:500}')
    a('.tick{fill:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums}')
    a('.grid{stroke:var(--grid);stroke-width:1}')
    a('.axis{stroke:var(--axis);stroke-width:1}')
    a('.lgnd{fill:var(--ink);font-size:12.5px;font-weight:500}')
    # Series identity is carried by the legend AND by every point's own label, so
    # it never rests on color alone.
    for i in (1, 2):
        a(f'.line{i}{{fill:none;stroke:var(--series{i});stroke-width:2;'
          f'stroke-linejoin:round;stroke-linecap:round}}')
        a(f'.dot{i}{{fill:var(--series{i});stroke:var(--surface);stroke-width:2}}')
        a(f'.sw{i}{{fill:var(--series{i})}}')
    a('.plbl{fill:var(--ink);font-size:11.5px;font-weight:600;'
      'font-variant-numeric:tabular-nums}')
    a("</style>")

    a(f'<rect class="bg" x="0" y="0" width="{W}" height="{H}"/>')
    a(f'<text class="ttl" x="{PAD["l"]}" y="34">Recall vs throughput — {esc(names)}, '
      f'18M BIGANN (SIFT) vectors</text>')
    a(f'<text class="sub" x="{PAD["l"]}" y="55">{esc(sub)}</text>')

    # legend: mandatory for two or more series
    lx = PAD["l"]
    for i, s in enumerate(series, start=1):
        st = INDEX_STYLE[s["index"]]
        a(f'<rect class="sw{i}" x="{lx}" y="74" width="22" height="4" rx="2"/>')
        a(f'<circle class="dot{i}" cx="{lx + 11}" cy="76" r="5"/>')
        label = f'{st["name"]}  ({s["param"]})'
        a(f'<text class="lgnd" x="{lx + 32}" y="80">{esc(label)}</text>')
        lx += 32 + 7.1 * len(label) + 34

    # y gridlines + ticks
    for v in log_ticks(y0, y1):
        y = sy(v)
        a(f'<line class="grid" x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}"/>')
        a(f'<text class="tick" x="{px0 - 10}" y="{y + 4:.1f}" text-anchor="end">{fmt_qps(v)}</text>')
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

    for i, s in enumerate(series, start=1):
        pts = " ".join(f"{sx(r['recall']):.1f},{sy(r['qps']):.1f}" for r in s["rows"])
        a(f'<polyline class="line{i}" points="{pts}"/>')

    for i, s in enumerate(series, start=1):
        prefix = INDEX_STYLE[s["index"]]["prefix"]
        for r in s["rows"]:
            x, y = sx(r["recall"]), sy(r["qps"])
            a(f'<circle class="dot{i}" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
            # Each curve enters a point from the upper left and leaves to the
            # lower right, so the free quadrants are upper-right and lower-left.
            # Series 1 takes upper-right, series 2 lower-left, which also keeps
            # the two label sets apart where the curves cross.
            above = (i == 1)
            if x > px1 - 64:
                above = False
            if above:
                a(f'<text class="plbl" x="{x + 11:.1f}" y="{y - 10:.1f}">{prefix} '
                  f'{r["search_value"]}</text>')
            else:
                a(f'<text class="plbl" x="{x - 11:.1f}" y="{y + 17:.1f}" text-anchor="end">'
                  f'{prefix} {r["search_value"]}</text>')

    a(f'<text class="foot" x="{PAD["l"]}" y="{H - 10}">{esc(foot)}</text>')
    a("</svg>")
    return "\n".join(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="dataset name, e.g. bigann_18000000")
    ap.add_argument("-o", "--out", help="output .svg path")
    ap.add_argument("--index", action="append",
                    help="index to plot; repeatable (default: hnsw then prism)")
    ap.add_argument("--clients", type=int, default=1)
    ap.add_argument("--window", type=float,
                    help="seconds per sweep point, for results recorded before "
                         "duration_s was saved in the metadata")
    args = ap.parse_args()

    global WINDOW_OVERRIDE
    WINDOW_OVERRIDE = args.window

    wanted = args.index or ["hnsw", "prism"]
    if len(wanted) > 2:
        sys.exit("error: at most two series (the palette defines two validated slots)")

    series = []
    for name in wanted:
        if name not in INDEX_STYLE:
            sys.exit(f"error: unknown index {name!r}, expected one of {list(INDEX_STYLE)}")
        rows = load_sweep(args.dataset, name, args.clients)
        if not rows:
            # Asked for explicitly -> hard error; part of the default pair -> skip,
            # so a dataset with only an HNSW sweep still plots.
            if args.index:
                sys.exit(f"no {name} rows with clients={args.clients} "
                         f"in {RESULTS_DIR / (args.dataset + '.jsonl')}")
            continue
        param = rows[0].get("search_param") or ""
        series.append({"index": name, "rows": rows, "param": param})
    if not series:
        sys.exit(f"no sweeps found for {args.dataset}")

    svg = build_svg(series, args.dataset)
    stem = "-".join(s["index"] for s in series)
    out = Path(args.out) if args.out else (
        REPO_ROOT / "bench" / "report"
        / f"{args.dataset.replace('_', '-')}-{stem}-recall-qps.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    for s in series:
        r = s["rows"]
        print(f'{s["index"]:>7}: {len(r)} points, recall {r[0]["recall"]:.4f}–'
              f'{r[-1]["recall"]:.4f}, {r[-1]["qps"]:,.0f}–{r[0]["qps"]:,.0f} QPS')
    print(out.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
