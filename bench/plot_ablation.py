#!/usr/bin/env python3
"""Chart the pgvectorfast ablation ladder (run_experiment.sh pvf_ablation) as an SVG.

Usage:
  bench/plot_ablation.py                         # ef_search=40
  bench/plot_ablation.py --ef 160 -o /tmp/x.svg  # throughput panels only exist per ef

Four small multiples, one row per arm (each arm adds one layer to the one above):
QPS at 1 client, QPS at 8 clients, CPU cycles per query and instructions per
query. The counter panels come from the arm's perf window (1 client, ef 40):
`perf stat -I 2000`, steady-state median of the intervals divided by the
window's QPS.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "bench" / "results" / "bigann_18000000.jsonl"
PERF_DIR = REPO_ROOT / "bench" / "profiles" / "ablation"

ARMS = [
    ("A0", "stock pgvector 0.8.7"),
    ("A1", "+ rewritten search loop, SIMD"),
    ("A2", "+ Calico page translation"),
    ("A3", "+ optimistic reads (no pins, locks)"),
    ("A4", "+ neighbour prefetch  = pgvectorfast"),
]

# Slot 1 of the reference categorical palette; one hue, since every panel is a
# single measure across the same five arms.
TOKENS = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "bar": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "bar": "#3987e5"},
}

W, H = 1060, 560
LABEL_W = 268            # row-label column
PANEL_W = 330
GAP = 60
ROW_H = 30
BAR_H = 18
TOP = 104                # first panel row's title baseline
PANEL_H = 40 + ROW_H * len(ARMS)


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_qps(ef):
    out = {}
    for line in RESULTS.read_text().splitlines():
        r = json.loads(line)
        lab = r.get("label") or ""
        if lab.startswith("pva-A") and r["search_value"] == ef:
            out[(lab[4:], r["clients"])] = r
    return out


def load_perf():
    out = {}
    for arm, _ in ARMS:
        qps = None
        for line in (PERF_DIR / f"{arm}.bench.txt").read_text().splitlines():
            p = line.split()
            if len(p) >= 3 and p[0].isdigit():
                qps = float(p[2])
        ev = {}
        with open(PERF_DIR / f"{arm}.stat.csv") as f:
            for row in csv.reader(l for l in f if l[:1].isdigit()):
                if row[1][:1].isdigit():
                    ev.setdefault(row[3], []).append(float(row[1]))
        # perf stat -I 2000: drop warm-up and tail intervals, median of the rest
        rate = {k: statistics.median(v[3:-2]) / 2.0 for k, v in ev.items()}
        out[arm] = {"qps": qps, "cycles": rate["cycles"] / qps,
                    "instructions": rate["instructions"] / qps,
                    "ipc": rate["instructions"] / rate["cycles"]}
    return out


def nice_max(v):
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        e = 10 ** (len(str(int(v))) - 1)
        if m * e >= v:
            return m * e
    return v


def bar(x, y, w, h, r=4):
    """Bar anchored square at the baseline, 4px rounded data end."""
    r = min(r, w / 2, h / 2)
    return (f'M{x:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}'
            f'a{r},{r} 0 0 1 {-r},{r}h{-(w - r):.1f}z')


def panel(a, px, py, title, values, fmt, notes, unit_tip):
    a(f'<text class="ptl" x="{px}" y="{py}">{esc(title)}</text>')
    top = py + 14
    vmax = nice_max(max(values))
    plot_w = PANEL_W - 70           # room for the value label after the longest bar
    for i in range(5):
        v = vmax * i / 4
        x = px + plot_w * i / 4
        a(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + ROW_H * len(values)}"/>')
        a(f'<text class="tick" x="{x:.1f}" y="{top + ROW_H * len(values) + 15}" '
          f'text-anchor="middle">{fmt(v, tick=True)}</text>')
    for k, v in enumerate(values):
        y = top + k * ROW_H + (ROW_H - BAR_H) / 2
        w = max(plot_w * v / vmax, 1)
        a(f'<path class="bar" d="{bar(px, y, w, BAR_H)}"><title>{esc(ARMS[k][1])}: '
          f'{esc(fmt(v))} {esc(unit_tip)}{esc(notes[k] if notes else "")}</title></path>')
        label = fmt(v)
        a(f'<text class="val" x="{px + w + 6:.1f}" y="{y + BAR_H / 2 + 4:.1f}">{esc(label)}'
          + (f'<tspan class="note"> {esc(notes[k])}</tspan>' if notes and notes[k] else "")
          + '</text>')
    a(f'<line class="axis" x1="{px}" y1="{top}" x2="{px}" y2="{top + ROW_H * len(values)}"/>')


def build(ef, qps, perf):
    p = []
    a = p.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
      f'role="img" aria-labelledby="t d" font-family="Inter, ui-sans-serif, system-ui, '
      f'-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif">')
    a('<title id="t">Where pgvectorfast\'s speedup comes from</title>')
    desc = "; ".join(
        f'{name}: {qps[(arm, 1)]["qps"]:,.0f} QPS at 1 client, {qps[(arm, 8)]["qps"]:,.0f} at 8, '
        f'{perf[arm]["cycles"] / 1e6:.2f}M cycles and {perf[arm]["instructions"] / 1e6:.2f}M '
        f'instructions per query' for arm, name in ARMS)
    a(f'<desc id="d">{esc(desc)}</desc>')

    def toks(t):
        return ";".join(f"--{k}:{v}" for k, v in t.items())
    a("<style>")
    a(f':root{{{toks(TOKENS["light"])}}}')
    a(f'@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{{toks(TOKENS["dark"])}}}}}')
    a(f':root[data-theme="dark"]{{{toks(TOKENS["dark"])}}}')
    a('.bg{fill:var(--surface)} .ttl{fill:var(--ink);font-size:19px;font-weight:600}'
      '.sub{fill:var(--ink2);font-size:12.5px} .foot{fill:var(--muted);font-size:11.5px}'
      '.ptl{fill:var(--ink);font-size:13px;font-weight:600}'
      '.row{fill:var(--ink);font-size:12.5px;font-weight:500}'
      '.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}'
      '.val{fill:var(--ink);font-size:12px;font-weight:600;font-variant-numeric:tabular-nums}'
      '.note{fill:var(--muted);font-weight:500}'
      '.grid{stroke:var(--grid);stroke-width:1} .axis{stroke:var(--axis);stroke-width:1}'
      '.bar{fill:var(--bar)} .bar:hover{opacity:.8}')
    a("</style>")
    a(f'<rect class="bg" x="0" y="0" width="{W}" height="{H}"/>')
    a('<text class="ttl" x="24" y="34">Where pgvectorfast\'s speedup over pgvector comes from</text>')
    rec = qps[("A0", 1)]["recall"]
    a(f'<text class="sub" x="24" y="56">HNSW on 18M BIGANN, ef_search={ef}, recall {rec:.3f} in every arm · '
      f'each row adds one layer to the row above · same index, fully in shared_buffers</text>')
    a('<text class="sub" x="24" y="74">Grey figures are the step over the row above. '
      'Hover a bar for its value.</text>')

    def mult(vals, inverse=False):
        out = [""]
        for i in range(1, len(vals)):
            r = vals[i - 1] / vals[i] if inverse else vals[i] / vals[i - 1]
            out.append(f"{'÷' if inverse else '×'}{r:.2f}")
        return out

    def fq(v, tick=False):
        return f"{v / 1000:g}k" if tick and v >= 1000 else f"{v:,.0f}"

    def fm(v, tick=False):
        return f"{v / 1e6:g}" if tick else f"{v / 1e6:.2f}M"

    cols = [LABEL_W, LABEL_W + PANEL_W + GAP]
    rows = [TOP, TOP + PANEL_H + 34]
    q1 = [qps[(arm, 1)]["qps"] for arm, _ in ARMS]
    q8 = [qps[(arm, 8)]["qps"] for arm, _ in ARMS]
    cyc = [perf[arm]["cycles"] for arm, _ in ARMS]
    ins = [perf[arm]["instructions"] for arm, _ in ARMS]
    ipc = [f'IPC {perf[arm]["ipc"]:.2f}' for arm, _ in ARMS]

    for ry in rows:
        for k, (_, name) in enumerate(ARMS):
            y = ry + 14 + k * ROW_H + ROW_H / 2 + 4
            a(f'<text class="row" x="{LABEL_W - 14}" y="{y:.1f}" text-anchor="end">{esc(name)}</text>')

    panel(a, cols[0], rows[0], "Queries per second, 1 client", q1, fq, mult(q1), "QPS")
    panel(a, cols[1], rows[0], "Queries per second, 8 clients", q8, fq, mult(q8), "QPS")
    panel(a, cols[0], rows[1], "CPU cycles per query (millions), 1 client", cyc, fm,
          mult(cyc, inverse=True), "cycles per query")
    panel(a, cols[1], rows[1], "Instructions per query (millions), 1 client", ins, fm, ipc,
          "instructions per query · ")

    a(f'<text class="foot" x="24" y="{H - 30}">Optimistic reads cut instructions ~60%; prefetch leaves '
      f'instructions and cache misses (~25k/query) unchanged but lifts instructions per cycle (IPC) '
      f'0.38 → 1.31 by overlapping misses.</text>')
    a(f'<text class="foot" x="24" y="{H - 12}">Postgres 18.6 · pgvector 0.8.7 vs pgvectorfast a704e32 · '
      f'60s per QPS point · counters: perf stat over a 40s 1-client window, steady-state median, '
      f'multiplexed (scaled) · AMD Ryzen 9 9900X</text>')
    a("</svg>")
    return "\n".join(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ef", type=int, default=40)
    ap.add_argument("-o", "--out")
    args = ap.parse_args()
    out = Path(args.out) if args.out else (
        REPO_ROOT / "bench" / "report" / f"bigann-18000000-pgvectorfast-ablation-ef{args.ef}.svg")
    out.write_text(build(args.ef, load_qps(args.ef), load_perf()))
    print(out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out)


if __name__ == "__main__":
    main()
