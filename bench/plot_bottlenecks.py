#!/usr/bin/env python3
"""Figures for bench/PGVECTOR-BOTTLENECKS.md, from bench/report/dbpedia-pgvector-bottlenecks.json.

  bench/plot_bottlenecks.py      # writes bench/report/bottleneck-*.svg

Same visual system as plot_variants.py: its categorical palette (pgvector is
slot 1, pgvectorfast slot 4 in every chart), light and dark themes switched by
prefers-color-scheme, direct labels instead of relying on hue.
"""

import json
from pathlib import Path

from plot_variants import CHROME, DARK, LIGHT

REPO = Path(__file__).resolve().parent.parent
DATA = json.loads((REPO / "bench/report/dbpedia-pgvector-bottlenecks.json").read_text())
OUT = REPO / "bench/report"
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    """A themed SVG canvas with a title block; marks are appended as raw elements."""

    def __init__(self, w, h, title, subtitle, desc):
        self.w, self.h, self.parts = w, h, []
        a = self.parts.append
        a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
          f'role="img" aria-labelledby="t d" font-family="{FONT}">')
        a(f'<title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>')

        def toks(mode, cols):
            return ";".join([f"--{k}:{v}" for k, v in CHROME[mode].items()]
                            + [f"--s{i}:{c}" for i, c in enumerate(cols, start=1)])
        a("<style>")
        a(f':root{{{toks("light", LIGHT)}}}')
        a(f'@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{{toks("dark", DARK)}}}}}')
        a(f':root[data-theme="dark"]{{{toks("dark", DARK)}}}')
        a(".bg{fill:var(--surface)} .ttl{fill:var(--ink);font-size:18px;font-weight:600}"
          ".sub{fill:var(--ink2);font-size:12.5px} .tick{fill:var(--muted);font-size:11.5px;"
          "font-variant-numeric:tabular-nums} .axlbl{fill:var(--ink2);font-size:12.5px;font-weight:500}"
          ".lbl{fill:var(--ink);font-size:12.5px;font-weight:600} .val{fill:var(--ink2);font-size:12px;"
          "font-variant-numeric:tabular-nums} .note{fill:var(--muted);font-size:11.5px}"
          ".grid{stroke:var(--grid);stroke-width:1} .axis{stroke:var(--axis);stroke-width:1}"
          ".ref{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4;fill:none}"
          ".band{fill:var(--muted);opacity:.14}")
        for i in range(1, 9):
            a(f".l{i}{{fill:none;stroke:var(--s{i});stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}"
              f".d{i}{{fill:var(--s{i});stroke:var(--surface);stroke-width:2}} .f{i}{{fill:var(--s{i})}}")
        a(".dash{stroke-dasharray:7 4}")
        a("</style>")
        a(f'<rect class="bg" x="0" y="0" width="{w}" height="{h}"/>')
        a(f'<text class="ttl" x="24" y="32">{esc(title)}</text>')
        a(f'<text class="sub" x="24" y="52">{esc(subtitle)}</text>')

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, cls="val", anchor="start"):
        self.add(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{esc(s)}</text>')

    def save(self, name):
        path = OUT / name
        path.write_text("\n".join(self.parts + ["</svg>"]) + "\n")
        print(path.relative_to(REPO))


def xy_axes(svg, box, xs, x_ticks, ymax, y_ticks, xlabel, ylabel):
    """Linear axes; returns scale functions."""
    l, t, r, b = box
    sx = lambda v: l + (v - xs[0]) / (xs[1] - xs[0]) * (r - l)
    sy = lambda v: b - v / ymax * (b - t)
    for v in y_ticks:
        svg.add(f'<line class="grid" x1="{l}" x2="{r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>')
        svg.text(l - 8, sy(v) + 4, f"{v:,}", "tick", "end")
    for v in x_ticks:
        svg.text(sx(v), b + 18, str(v), "tick", "middle")
    svg.add(f'<line class="axis" x1="{l}" x2="{r}" y1="{b}" y2="{b}"/>')
    svg.add(f'<line class="axis" x1="{l}" x2="{l}" y1="{t}" y2="{b}"/>')
    svg.text((l + r) / 2, b + 40, xlabel, "axlbl", "middle")
    svg.add(f'<text class="axlbl" transform="translate({l - 52} {(t + b) / 2}) rotate(-90)" '
            f'text-anchor="middle">{esc(ylabel)}</text>')
    return sx, sy


def line(svg, pts, slot, sx, sy, dash=False):
    d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
    svg.add(f'<polyline class="l{slot}{" dash" if dash else ""}" points="{d}"/>')
    for x, y in pts:
        svg.add(f'<circle class="d{slot}" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5"/>')


def fig_throughput():
    """QPS vs clients at ef_search=640 against the bandwidth-implied ceiling."""
    th = [t for t in DATA["throughput"] if t["ef_search"] == 640]
    mb = sum(c["dram_mb_per_query"] for c in DATA["counters"] if c["ef_search"] == 640) / 3
    lo = DATA["ceilings_gbps"]["random_6kb_chunks"]["8"] * 1e3 / mb
    hi = DATA["ceilings_gbps"]["sequential"]["8"] * 1e3 / mb
    svg = Svg(860, 500, "Throughput stops at the memory-bandwidth ceiling",
              f"HNSW at recall 0.989 (ef_search=640). Each query reads ~{mb:.0f} MB from DRAM; at 30–35 GB/s that caps "
              f"throughput at {lo:.0f}–{hi:.0f} QPS.",
              "QPS by client count for pgvector and pgvectorfast, with the ceiling implied by DRAM bandwidth.")
    box = (96, 84, 640, 420)
    sx, sy = xy_axes(svg, box, (0, 16), [1, 4, 8, 12, 16], 700, [0, 100, 200, 300, 400, 500, 600, 700],
                     "concurrent clients", "queries per second")
    svg.add(f'<rect class="band" x="{box[0]}" y="{sy(hi):.1f}" width="{box[2] - box[0]}" height="{sy(lo) - sy(hi):.1f}"/>')
    svg.text(box[2] - 8, sy(lo) - 8, f"DRAM ceiling {lo:.0f}–{hi:.0f} QPS", "val", "end")
    ends = {}
    for name, slot in (("pgvector 0.8.7", 1), ("pgvectorfast", 4)):
        pts = sorted((t["clients"], t["qps"]) for t in th if t["engine"] == name)
        line(svg, pts, slot, sx, sy)
        ends[name] = pts[-1]
    lx = box[2] + 14
    for name, (x, y) in ends.items():
        svg.text(lx, sy(y) + 4, f"{name}  {y:,.0f}", "lbl")
    svg.text(24, 482, "1 client: client on the server's machine. 8–16: client on a second node, so the server keeps "
             "its 8 cores (16 hardware threads).", "note")
    svg.save("bottleneck-throughput.svg")


def fig_bandwidth():
    """DRAM read bandwidth vs clients against the machine's measured curve."""
    ceil = DATA["ceilings_gbps"]["random_6kb_chunks"]
    svg = Svg(860, 500, "DRAM reads reach what the machine can deliver",
              "Memory-controller read traffic while serving queries, against microbenchmarks of this machine's read ceilings.",
              "DRAM read GB/s for pgvector at ef_search 40 and 640 and pgvectorfast at 640, by client count.")
    box = (96, 84, 640, 420)
    sx, sy = xy_axes(svg, box, (0, 16), [1, 4, 8, 12, 16], 40, [0, 5, 10, 15, 20, 25, 30, 35, 40],
                     "concurrent clients (threads, for the microbenchmark)", "DRAM read GB/s")
    ys = []
    for label, key in (("ceiling: sequential reads", "sequential"), ("ceiling: random 6 KB reads", "random_6kb_chunks")):
        pts = sorted((int(k), v) for k, v in DATA["ceilings_gbps"][key].items())
        d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        svg.add(f'<polyline class="ref" points="{d}"/>')
        ys.append((label, pts[-1][1]))
    c = DATA["counters"]
    series = [
        ("pgvector, ef 640", 1, False, [(r["clients"], r["dram_read_gbps"]) for r in c if r["ef_search"] == 640]),
        ("pgvector, ef 40", 1, True, [(r["clients"], r["dram_read_gbps"]) for r in c if r["ef_search"] == 40]),
        ("pgvectorfast, ef 640", 4, False, [(r["clients"], r["dram_read_gbps"]) for r in DATA["counters_pgvectorfast_ef640"]]),
    ]
    for name, slot, dash, p in series:
        p = sorted(p)
        line(svg, p, slot, sx, sy, dash)
        ys.append((name, p[-1][1]))
    # end labels, nudged apart
    placed = []
    for name, y in sorted(ys, key=lambda t: -t[1]):
        py = sy(y) + 4
        while any(abs(py - q) < 15 for q in placed):
            py += 15
        placed.append(py)
        svg.text(sx(16) + 14, py, f"{name}  {y:.1f}", "lbl")
    svg.text(24, 482, "pgvectorfast is at 32.9 GB/s with 8 clients, between the two ceilings (its prefetching overlaps "
             "reads), and gains nothing from 16.", "note")
    svg.save("bottleneck-bandwidth.svg")


def fig_breakdown():
    """Where server CPU time goes, per visited element."""
    b = DATA["breakdown_1client_ef640"]
    per_node_us = b["server_cpu_ms_per_query"] * 1e3 / b["nodes_per_query"]
    items = sorted(b["share_pct"].items(), key=lambda kv: -kv[1])
    svg = Svg(860, 470, f"Each visited element costs {per_node_us:.1f} µs of server CPU",
              f"1 client, ef_search=640: {b['nodes_per_query']:,} elements and {b['server_cpu_ms_per_query']:.1f} ms "
              f"of server CPU per query. Self time by category, from call-graph samples.",
              "Horizontal bars of microseconds per visited element by category.")
    l, top, r, row = 300, 84, 700, 34
    vmax = 0.8
    for i, (name, pct) in enumerate(items):
        us = pct / 100 * per_node_us
        y = top + i * row
        w = max(us / vmax * (r - l), 2)
        svg.text(l - 12, y + 17, name, "lbl", "end")
        svg.add(f'<rect class="f1" x="{l}" y="{y + 4}" width="{w:.1f}" height="{row - 12}" rx="3"/>')
        svg.text(l + w + 8, y + 17, f"{us:.2f} µs  ({pct:.0f}%)", "val")
    y = top + len(items) * row + 14
    svg.text(24, y, f"Overlapping all of these: page-table walks take {b['page_walk_cycles_pct']}% of cycles "
             f"(~{b['dtlb_misses_per_node']} dTLB misses per element; shared_buffers is mapped with 4 KB pages).", "note")
    svg.text(24, y + 18, "The two buffer-manager rows are one cost: finding, pinning and locking each element's page. "
             "Together, 33%.", "note")
    svg.save("bottleneck-breakdown.svg")


def fig_kernel():
    """Distance kernel: cached vs DRAM, one vs four accumulators."""
    k = DATA["kernel_ns"]
    svg = Svg(860, 446, "The distance arithmetic is not the bottleneck",
              "Time per 1536-d L2 distance, one thread. pgvector's kernel verbatim, built with pgvector's flags.",
              "Grouped bars: cached and from-DRAM time for pgvector's kernel and a four-accumulator version.")
    l, b, t = 96, 350, 104
    vmax = 800
    sy = lambda v: b - v / vmax * (b - t)
    for v in range(0, 801, 200):
        svg.add(f'<line class="grid" x1="{l}" x2="640" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>')
        svg.text(l - 8, sy(v) + 4, str(v), "tick", "end")
    svg.add(f'<line class="axis" x1="{l}" x2="640" y1="{b}" y2="{b}"/>')
    svg.add(f'<text class="axlbl" transform="translate({l - 52} {(t + b) / 2}) rotate(-90)" '
            f'text-anchor="middle">ns per distance</text>')
    groups = [("vectors in cache", "cached"), ("vectors from DRAM (the real case)", "dram")]
    kernels = list(k.items())
    gw, bw = 260, 72
    for gi, (glabel, key) in enumerate(groups):
        gx = l + 40 + gi * gw
        for ki, (kname, vals) in enumerate(kernels):
            v = vals[key]
            x = gx + ki * (bw + 4)
            svg.add(f'<rect class="f{1 if ki == 0 else 3}" x="{x}" y="{sy(v):.1f}" width="{bw}" '
                    f'height="{b - sy(v):.1f}" rx="3"/>')
            svg.text(x + bw / 2, sy(v) - 8, f"{v}", "val", "middle")
        svg.text(gx + bw + 2, b + 20, glabel, "lbl", "middle")
    lx = l
    for ki, (kname, _) in enumerate(kernels):
        svg.add(f'<rect class="f{1 if ki == 0 else 3}" x="{lx}" y="{t - 22}" width="12" height="12" rx="2"/>')
        svg.text(lx + 18, t - 12, kname, "val")
        lx += 30 + 7 * len(kname)
    svg.text(24, 404, "In cache, pgvector's single accumulator serializes 192 FMAs (≈4.7 cycles each); four accumulators "
             "are 3.4× faster.", "note")
    svg.text(24, 422, "From DRAM both take ~0.73 µs, the time to stream 6 KB, and that matches the 0.67 µs measured "
             "inside real queries.", "note")
    svg.save("bottleneck-kernel.svg")


if __name__ == "__main__":
    fig_throughput()
    fig_bandwidth()
    fig_breakdown()
    fig_kernel()
