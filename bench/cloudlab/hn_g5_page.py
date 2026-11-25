#!/usr/bin/env python3
"""The HN c220g5 study's page (the claude.ai artifact), from hn_g5_plot.py's output.

  hn_g5_page.py PLOTDIR OUT.html [ORDER_LOG [GAP_DIR [SIMD_DIR]]]

PLOTDIR holds hn_g5_plot.py's SVG figures and summary.json. ORDER_LOG, if
given, is hn_g5_32k_ro.sh's order step log (graph_compare.py and
graph_layout.py output), quoted in the reordering section. GAP_DIR, if given,
is hn_g5_gap.sh's profiles (bench/profiles/hn-gap), read with gap_attrib.py
for the section on why Ginger is still faster. SIMD_DIR is hn_g5_simd.sh's
output (bench/profiles/hn-g5-simd): the kernel comparison and micro-benchmark.

The figures stay matplotlib's own SVG, inlined; the page adds engine
checkboxes, legends that list only the engines shown, and a hover tooltip.
"""
import base64
import html
import json
import math
import re
import sys
from pathlib import Path

plotdir, out = Path(sys.argv[1]), Path(sys.argv[2])
order_log = Path(sys.argv[3]) if len(sys.argv) > 3 else None
gap_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else None
simd_dir = Path(sys.argv[5]) if len(sys.argv) > 5 else None
# The second tab: the account of the experiments, in order (static HTML)
HISTORY = (Path(__file__).resolve().parent / "hn_g5_history.html")
history_html = HISTORY.read_text() if HISTORY.exists() else ""
s = json.loads((plotdir / "summary.json").read_text())

ARMS = [
    ("pgvector-hnsw", "pgvector, HNSW m=32"),
    ("pgvectorfast-hnsw", "pgvectorfast, HNSW m=32"),
    ("pgvector-pipnn", "pgvector, PiPNN graph"),
    ("pgvectorfast-pipnn", "pgvectorfast, PiPNN graph"),
    ("pgvectorfast-tq4", "pgvectorfast, PiPNN graph + tq4"),
    ("ginger", "Ginger"),
    ("pgvectorfast-tq4-32k", "pgvectorfast, PiPNN graph + tq4, 32 KB pages"),
    ("pgvectorfast-tq4-32k-ro", "pgvectorfast, PiPNN graph + tq4, 32 KB pages, reordered"),
    ("pgvectorfast-tq4-32k-ro-rr10", "pgvectorfast, PiPNN graph + tq4, 32 KB pages, reordered, rerank ≤ 100"),
    ("pgvectorfast-tq4-32k-ro-rr10simd",
     "pgvectorfast, PiPNN graph + tq4, 32 KB pages, reordered, rerank ≤ 100, AVX-512 rerank"),
    ("pgvectorfast-tq4-32k-ro-rr10simd-nolock90", "pgvectorfast, as above + no scan lock, 90 clients"),
]
PASS_NAME = {"hp": "Huge pages", "4k": "4 KB pages"}
TARGETS = ["0.90", "0.95", "0.99"]


def fmt(q):
    return "–" if q is None else f"{q:,.0f}"


# Swatches for the checkboxes: the series' tab10 colour, line style and marker,
# as matplotlib draws them in the legend.
STYLE = {
    "pgvector-hnsw": ("#1f77b4", "o", True),
    "pgvectorfast-hnsw": ("#ff7f0e", "s", False),
    "pgvector-pipnn": ("#2ca02c", "^", True),
    "pgvectorfast-pipnn": ("#d62728", "D", False),
    "pgvectorfast-tq4": ("#9467bd", "v", False),
    "ginger": ("#8c564b", "*", False),
    "pgvectorfast-tq4-32k": ("#e377c2", "X", False),
    "pgvectorfast-tq4-32k-ro": ("#7f7f7f", "P", False),
    "pgvectorfast-tq4-32k-ro-rr10": ("#bcbd22", "h", False),
    "pgvectorfast-tq4-32k-ro-rr10simd": ("#17becf", ">", False),
    "pgvectorfast-tq4-32k-ro-rr10simd-nolock90": ("#393b79", "p", False),
}


def star(cx, cy, r_out, r_in):
    pts = []
    for i in range(10):
        r = r_out if i % 2 == 0 else r_in
        a = math.pi / 2 + i * math.pi / 5
        pts.append(f"{cx + r * math.cos(a):.2f},{cy - r * math.sin(a):.2f}")
    return " ".join(pts)


def swatch(arm):
    c, m, dashed = STYLE[arm]
    dash = ' stroke-dasharray="4.4,1.9"' if dashed else ""
    shape = {
        "o": f'<circle cx="17" cy="7" r="3.4" fill="{c}"/>',
        "s": f'<rect x="13.8" y="3.8" width="6.4" height="6.4" fill="{c}"/>',
        "^": f'<polygon points="17,3 20.8,10.2 13.2,10.2" fill="{c}"/>',
        "D": f'<polygon points="17,2.8 21,7 17,11.2 13,7" fill="{c}"/>',
        "v": f'<polygon points="13.2,3.8 20.8,3.8 17,11" fill="{c}"/>',
        "*": f'<polygon points="{star(17, 7.4, 5.4, 2.2)}" fill="{c}"/>',
        "X": f'<path d="M13.4,3.6 L20.6,10.4 M20.6,3.6 L13.4,10.4" stroke="{c}" stroke-width="2.6"/>',
        "P": f'<path d="M15.6,2.6 h2.8 v3 h3 v2.8 h-3 v3 h-2.8 v-3 h-3 v-2.8 h3 z" fill="{c}"/>',
        "h": f'<polygon points="17,2.4 21,4.7 21,9.3 17,11.6 13,9.3 13,4.7" fill="{c}"/>',
        ">": f'<polygon points="13.4,3.2 21,7 13.4,10.8" fill="{c}"/>',
        "p": f'<polygon points="17,2.6 21.2,5.6 19.6,10.6 14.4,10.6 12.8,5.6" fill="{c}"/>',
    }[m]
    return (f'<svg class="sw" viewBox="0 0 34 14" width="34" height="14" aria-hidden="true">'
            f'<line x1="1" y1="7" x2="33" y2="7" stroke="{c}" stroke-width="1.5"{dash}/>{shape}</svg>')


def controls(pas):
    boxes = "".join(
        f'<label class="arm"><input type="checkbox" id="show-{pas}-{arm}" data-arm="{arm}" checked>'
        f'{swatch(arm)}<span>{html.escape(name)}</span></label>'
        for arm, name in ARMS
        if any(p["pass"] == pas and p["arm"] == arm for p in s["points"]))
    return (f'<fieldset class="arms"><legend>Engines shown in every chart</legend>{boxes}'
            f'<button type="button" class="showall" id="showall-{pas}">Show all</button></fieldset>')


def figure(pas):
    """The matplotlib SVG inline (not an <img>), so the page can toggle its series"""
    svg = (plotdir / f"{pas}.svg").read_text()
    svg = svg[svg.index("<svg"):]
    svg = re.sub(r'<svg ([^>]*?)width="[^"]*" height="[^"]*"', r'<svg \1', svg, count=1)
    label = (f"{PASS_NAME[pas]}: QPS (log scale) against recall@10 for six engines, "
             "at 1 client and at 60 clients (Ginger 40 threads)")
    svg = svg.replace("<svg ", f'<svg class="mpl" data-pass="{pas}" role="img" aria-label="{html.escape(label)}" ', 1)
    return f'{controls(pas)}<div class="fig">{svg}</div>'


def matched(pas):
    m = {(r["arm"], r["clients"] > 1): r for r in s["matched"] if r["pass"] == pas}
    rows = []
    for arm, name in ARMS:
        if not any((arm, multi) in m for multi in (False, True)):
            continue
        cells = []
        for multi in (False, True):
            r = m.get((arm, multi))
            cells += [f"<td>{fmt(r[t] if r else None)}</td>" for t in TARGETS]
        rows.append(f"<tr><th scope=\"row\">{html.escape(name)}</th>{''.join(cells)}</tr>")
    head1 = ('<tr><th rowspan="2" scope="col">Engine</th>'
             '<th colspan="3" scope="colgroup">1 client</th>'
             '<th colspan="3" scope="colgroup">60 clients (Ginger: 40 threads)</th></tr>')
    head2 = "<tr>" + "".join(f"<th scope=\"col\">@ {t}</th>" for t in TARGETS) * 2 + "</tr>"
    return (f'<div class="tbl"><table><thead>{head1}{head2}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def points(pas):
    m = {}
    for p in s["points"]:
        if p["pass"] == pas:
            m.setdefault((p["arm"], p["clients"] > 1), []).append(p)
    efs = sorted({p["ef"] for p in s["points"] if p["pass"] == pas})
    rows = []
    for multi in (False, True):
        for arm, name in ARMS:
            pts = {p["ef"]: p for p in m.get((arm, multi), [])}
            if not pts:
                continue
            c = next(iter(pts.values()))["clients"]
            who = f"{c} thread{'s' if c > 1 else ''}" if arm == "ginger" else f"{c} client{'s' if c > 1 else ''}"
            cells = "".join(
                f"<td>{pts[e]['recall']:.4f}<br><span>{pts[e]['qps']:,.0f}</span></td>" if e in pts else "<td>–</td>"
                for e in efs)
            rows.append(f"<tr><th scope=\"row\">{html.escape(name)}<br><span>{who}</span></th>{cells}</tr>")
    head = "<tr><th scope=\"col\">ef_search →</th>" + "".join(f"<th scope=\"col\">{e}</th>" for e in efs) + "</tr>"
    return (f'<div class="tbl"><table class="pts"><thead>{head}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')



def scaling(pas, arm, ef):
    pts = {(p["clients"] > 1): p["qps"] for p in s["points"] if p["pass"] == pas and p["arm"] == arm and p["ef"] == ef}
    return pts[True] / pts[False]


def qps(pas, arm, multi, ef):
    return next(p["qps"] for p in s["points"]
                if p["pass"] == pas and p["arm"] == arm and (p["clients"] > 1) == multi and p["ef"] == ef)


def bottlenecks(pas):
    """What limits each engine, from the scaling, the CPU trace and the machine's measured memory bandwidth"""
    def sc(arms):
        lo = min(scaling(pas, a, ef) for a in arms for ef in (160, 1000))
        hi = max(scaling(pas, a, ef) for a in arms for ef in (160, 1000))
        return f"{lo:.1f}–{hi:.1f}×" if f"{lo:.1f}" != f"{hi:.1f}" else f"{lo:.1f}×"
    f160 = qps(pas, "pgvectorfast-pipnn", True, 160)
    t160 = qps(pas, "pgvectorfast-tq4", True, 160)
    f1_160 = qps(pas, "pgvectorfast-pipnn", False, 160)
    t1_160 = qps(pas, "pgvectorfast-tq4", False, 160)
    mb = 77e9 / f160 / 1e6
    tuples = round(77e9 / f160 / 1664, -3)
    s1000 = qps(pas, "pgvector-pipnn", True, 1000) / qps(pas, "pgvectorfast-pipnn", True, 1000)
    tq10 = qps(pas, "pgvectorfast-tq4", True, 10)
    rows = [
        ("pgvector (HNSW and PiPNN)", "1,616 B float tuple", sc(["pgvector-hnsw", "pgvector-pipnn"]), "97–100%", "CPU"),
        ("pgvectorfast (HNSW and PiPNN)", "1,616 B float tuple", sc(["pgvectorfast-hnsw", "pgvectorfast-pipnn"]), "90–100%", "Memory bandwidth"),
        ("pgvectorfast + tq4", "288 B of 4-bit codes, then ~400 B per candidate to rerank", sc(["pgvectorfast-tq4"]),
         "95–99% from ef 80; 63% and 76% at ef 10 and 20", "CPU; at ef 10–20, too few clients"),
        ("Ginger (1 → 40 threads)", "4-bit TurboQuant codes, in-process", sc(["ginger"]), "not sampled (client node)", "CPU"),
    ]
    body = "".join(
        f'<tr><th scope="row">{a}</th><td class="l">{b}</td><td>{c}</td><td class="l">{d}</td><td class="l"><strong>{e}</strong></td></tr>'
        for a, b, c, d, e in rows)
    return f"""
<section>
  <h2>What limits each engine</h2>
  <p class="body">The server has 192 GB in six 32 GB DDR4 DIMMs, one on each of channels 0–2 per socket, so half of the
  twelve memory channels are empty. The CPUs run the DIMMs at 2400 MT/s, a theoretical 115 GB/s for the machine.
  Measured on the idle server with <code>bench/cloudlab/membw.c</code>:</p>
  <div class="tbl"><table class="kv">
    <tbody>
      <tr><th scope="row">Read, 40 threads, memory interleaved across both sockets (as Postgres ran)</th><td>77 GB/s</td></tr>
      <tr><th scope="row">Read, 20 threads on one socket, local memory</th><td>48 GB/s</td></tr>
      <tr><th scope="row">STREAM triad, 40 threads</th><td>55 GB/s</td></tr>
      <tr><th scope="row">Read, 1 thread</th><td>8.7 GB/s</td></tr>
      <tr><th scope="row">One dependent random access (4 KB pages): local / interleaved</th><td>136 / 166 ns</td></tr>
    </tbody>
  </table></div>
  <p class="body">The clearest sign of what limits an engine is how far it scales from 1 to 60 clients. With 20 cores and
  hyper-threading, compute-bound code gets about 20–25×. An engine that stops well short of that while the server is
  fully busy is waiting on something the cores share.</p>
  <div class="tbl"><table class="lim">
    <thead><tr><th scope="col">Engine</th><th scope="col">Read per graph node</th><th scope="col">1 → 60 clients<br><span>ef 160 and ef 1000</span></th>
      <th scope="col">Server busy at 60 clients</th><th scope="col">Limit</th></tr></thead>
    <tbody>{body}</tbody>
  </table></div>
  <div class="why">
    <h3>pgvector: CPU</h3>
    <p>It scales like compute-bound code and keeps every hardware thread busy. It walks the same graph as pgvectorfast:
    recall is identical at every ef_search, so both read the same vectors per query. It is 3–4× slower per query,
    though, so it runs out of cores first. At ef 1000 and 60 clients it moves {s1000:.0%} of the bytes per second
    pgvectorfast moves.</p>
    <h3>pgvectorfast: memory bandwidth</h3>
    <p>It is 3–4× faster than pgvector at 1 client, but from 1 to 60 clients it scales only 8× at ef 160 and 1000
    (12× at ef 10), with the server busy throughout. A thread
    stalled on a memory read still counts as busy. Two things point at bandwidth:</p>
    <ul>
      <li>tq4 runs the same code on the same graph, reading 288-byte tuples instead of 1,616-byte ones. At 1 client
      it is about as fast ({t1_160:,.0f} against {f1_160:,.0f} QPS at ef 160). At 60 clients it is
      {t160 / f160:.1f}× faster ({t160:,.0f} against {f160:,.0f}). A gap that opens only under load, and only for the
      engine reading more bytes, comes from a resource the cores share.</li>
      <li>The numbers fit. At the measured 77 GB/s, {f160:,.0f} QPS is {mb:.0f} MB per query, about
      {tuples:,.0f} float tuples, or roughly {round(tuples / 64, -1):.0f} layer-0 nodes expanded with 64 neighbours each. That
      is the scale of walk ef_search 160 needs.</li>
    </ul>
    <p class="note">This is inferred from scaling and arithmetic. It was not measured with the memory controllers'
    counters. With all twelve channels populated, theoretical bandwidth would double to 230 GB/s.</p>
    <h3>pgvectorfast + tq4: CPU, and at low ef_search the number of clients</h3>
    <p>With about a fifth of the bytes per node, it scales like compute-bound code again (20–22×). From ef 80 up the
    server is 95–99% busy. At ef 10 and 20 it is only 63% and 76% busy. At {tq10:,.0f} QPS, 60 clients give each query a
    round trip of {60 / tq10 * 1000:.2f} ms, and a little under half of that is server CPU. The rest is the network, the
    client and wake-ups. So 60 connections can't keep 40 hardware threads busy, and those two points are lower
    bounds.</p>
    <h3>Ginger: CPU</h3>
    <p>It runs inside the benchmark process on the client node, so it skips Postgres's per-query work: the executor,
    buffer pins, fetching the ten result rows from the heap, and the wire protocol. It also has no network round trip.
    It reads 4-bit codes and scales 27–29× on 40 threads. At recall 0.95 it is about twice tq4's QPS at full load.
    Part of that gap is the cost of serving a query through Postgres. Part is layout: Ginger's recommended
    configuration renumbers the graph so neighbours sit together in memory (recursive graph partitioning), which the
    Postgres arms don't do. Their elements are stored in table order. An earlier ablation put reordering at
    1.08–1.15× within Ginger itself.</p>
  </div>
</section>"""


def pages32(pas):
    """The tq4 arm on 8 KB and on 32 KB pages, point by point"""
    efs = sorted({p["ef"] for p in s["points"] if p["pass"] == pas and p["arm"] == "pgvectorfast-tq4-32k"})
    if not efs:
        return ""
    rows = []
    for ef in efs:
        cells = []
        for multi in (False, True):
            a, b = qps(pas, "pgvectorfast-tq4", multi, ef), qps(pas, "pgvectorfast-tq4-32k", multi, ef)
            cells.append(f"<td>{a:,.0f}</td><td>{b:,.0f}</td><td><strong>{b / a - 1:+.1%}</strong></td>")
        rec = next(p["recall"] for p in s["points"] if p["pass"] == pas and p["arm"] == "pgvectorfast-tq4-32k" and p["ef"] == ef)
        rows.append(f'<tr><th scope="row">{ef}</th><td>{rec:.4f}</td>{"".join(cells)}</tr>')
    head = ('<tr><th rowspan="2" scope="col">ef_search</th><th rowspan="2" scope="col">recall@10<br><span>both</span></th>'
            '<th colspan="3" scope="colgroup">1 client</th><th colspan="3" scope="colgroup">60 clients</th></tr>'
            '<tr>' + '<th scope="col">8 KB</th><th scope="col">32 KB</th><th scope="col">change</th>' * 2 + '</tr>')
    return f"""
<section>
  <h2>tq4 on 32 KB pages</h2>
  <p class="body">The same graph and the same 4-bit and rerank codes, served from Postgres built with 32 KB pages instead
  of 8 KB. Recall is identical at every ef_search, so the difference is the page size alone.</p>
  <div class="tbl"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>
  <div class="why">
    <p><strong>No change at 1 client; about 8% more QPS at 60 clients wherever the server is the limit</strong>
    (ef_search 40 and up). At ef_search 10 and 20 the 60 clients, not the server, set the pace
    (63–76% busy at both page sizes), so those points barely move.</p>
    <p>A search reads the same tuples either way. Graph neighbours sit on effectively random pages, so a bigger page
    doesn't put more of them together. Two things do shrink with the page count:</p>
    <ul>
      <li>calico's lookup table, one 8-byte entry per index block: about 30 MB at 8 KB, more than the 27.5 MB of L3
      on both sockets combined, and about 7 MB at 32 KB.</li>
      <li>The index: 29.5 GiB at 8 KB and 28.2 GiB at 32 KB (−4.6%), because fewer bytes are left over at the end
      of each page.</li>
    </ul>
    <p>A gain that appears only when all cores compete for cache and memory fits the smaller lookup table, but this
    run doesn't prove it. A single client sees no difference. Reading the hardware's cache-miss counters during a
    60-client run would settle it.</p>
    <p class="note">The table and index also loaded into memory faster at 32 KB: about 6 minutes against 15.</p>
  </div>
</section>"""


# Filled in once the reordered run's numbers are known: the section's reading.
REORDER_NOTES = """
    <p><strong>Reading.</strong> The reordered layout is faster at every point. Compared at the same recall@10, which
    sets aside the small recall differences between the two graphs, it gains 6% at 0.90, 6% at 0.95 and 9% at 0.99
    with 60 clients (38,109, 25,906 and 9,890 QPS), and 1–5% with one client. The prediction was 12% fewer index
    pages per expansion. At 60 clients the server is 94–100% busy from ef 80 up, so the saving shows up as less
    CPU time per query: fewer pages to pin, look up in calico and miss in cache per node expanded.</p>
    <p><strong>With 32 KB pages.</strong> The two changes add up. Against the tq4 arm on 8 KB pages in table
    order, at 60 clients: +16% at recall 0.95 (22,383 to 25,906 QPS) and +19% at 0.99 (8,337 to 9,890).
    Ginger at 40 threads is still 1.7× faster at 0.95 (43,972).</p>
    <p class="note"><strong>Caveat.</strong> The graph was rebuilt for the reordered export, and PiPNN's build is not
    deterministic: 57% of rows have identical neighbour sets and 96% of edges are shared. Recall moves by at most
    0.0016 from ef 40 up and by +0.0105 at ef 10. Part of the gain could come from the graph rather than the layout.
    Importing the reordered graph into a table in id order would separate the two (about 1.3 h).</p>"""


def order_facts():
    """graph_compare.py and graph_layout.py lines from hn_g5_32k_ro.sh's order step log"""
    if order_log is None or not order_log.exists():
        return {}
    t = order_log.read_text()
    f = {}
    for key, pat in [("rows", r"rows with identical neighbour sets: ([\d.]+%)"),
                     ("edges", r"edges in common: [\d,]+ of [\d,]+ \(([\d.]+%) of the union\)"),
                     ("entry", r"entry point: row \d+ and row \d+ \((\w+)\)"),
                     ("fewer", r"node layout touches ([-\d.]+%) fewer pages")]:
        m = re.search(pat, t)
        if m:
            v = m.group(1)
            f[key] = f"{float(v[:-1]):.1f}%" if v.endswith("%") else v
    for lay in ("heap", "node", "random"):
        m = re.search(rf"^\| {lay} \| ([\d.]+) \| ([\d.]+%) \|", t, re.M)
        if m:
            f[lay] = (m.group(1), m.group(2))
    return f


def reorder32(pas):
    """32 KB pages, table order against Ginger's reordered order"""
    ro, base = "pgvectorfast-tq4-32k-ro", "pgvectorfast-tq4-32k"
    efs = sorted({p["ef"] for p in s["points"] if p["pass"] == pas and p["arm"] == ro})
    if not efs:
        return ""
    def rec(arm, ef):
        return next(p["recall"] for p in s["points"]
                    if p["pass"] == pas and p["arm"] == arm and p["ef"] == ef and p["clients"] == 1)
    rows = []
    for ef in efs:
        cells = [f"<td>{rec(base, ef):.4f}</td><td>{rec(ro, ef):.4f}</td>"]
        for multi in (False, True):
            a, b = qps(pas, base, multi, ef), qps(pas, ro, multi, ef)
            cells.append(f"<td>{a:,.0f}</td><td>{b:,.0f}</td><td><strong>{b / a - 1:+.1%}</strong></td>")
        rows.append(f'<tr><th scope="row">{ef}</th>{"".join(cells)}</tr>')
    head = ('<tr><th rowspan="2" scope="col">ef_search</th><th colspan="2" scope="colgroup">recall@10</th>'
            '<th colspan="3" scope="colgroup">1 client</th><th colspan="3" scope="colgroup">60 clients</th></tr>'
            '<tr><th scope="col">table order</th><th scope="col">reordered</th>'
            + '<th scope="col">table order</th><th scope="col">reordered</th><th scope="col">change</th>' * 2 + '</tr>')
    f = order_facts()
    facts = []
    if "rows" in f:
        facts.append(f"<li><strong>Same graph?</strong> Rebuilding the graph for the reordered export gave "
                     f"{f['rows']} of rows the same neighbour set as the graph behind every other result, "
                     f"{f.get('edges', '?')} of edges in common, and the {f.get('entry', '?')} entry point.</li>")
    if "heap" in f and "node" in f:
        facts.append(f"<li><strong>Predicted beforehand</strong> (<code>graph_layout.py</code>, 46 tq4 elements per "
                     f"32 KB page): expanding a node touches {f['heap'][0]} distinct index pages in table order "
                     f"and {f['node'][0]} reordered ({f.get('fewer', '?')} fewer). The share of neighbours on the "
                     f"node's own page goes from {f['heap'][1]} to {f['node'][1]}.</li>")
    facts_html = f"<ul>{''.join(facts)}</ul>" if facts else ""
    return f"""
<section>
  <h2>Reordered graph on 32 KB pages</h2>
  <p class="body">Ginger renumbers its graph so that neighbours sit together in memory (recursive graph
  partitioning, as in its recommended configuration). Here the table was loaded in that order, so the import laid
  the tq4 index out in it too. Both runs are on 32 KB pages, measured the same way. The only change is the layout,
  plus whatever differs in the rebuilt graph (below).</p>
  <div class="tbl"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>
  <div class="why">{facts_html}{REORDER_NOTES}</div>
</section>"""

# Filled in once the capped run's numbers are known: the section's reading.
RR_NOTES = """
    <p><strong>Reading.</strong> Recall@10 is identical to four decimals at every ef_search, including the four
    where the cap binds: the true top 10 always ranked inside the top 100 on 4-bit distance. Throughput rises with
    the candidates dropped: +15% to +75% at one client (ef 160 to 1000) and +8% to +28% at 60. At recall 0.99 that
    is 417 → 530 QPS at one client and 9,890 → 11,172 at 60, and Ginger's lead there drops from 1.42× to 1.12× and
    from 1.59× to 1.41×. Up to ef 80 the two runs agree within 3%, the run-to-run noise.</p>
    <p>The gain is larger at one client than at 60. That fits the gap analysis: a reranked candidate costs about 4 µs
    of CPU either way, and under load more of each query's time is the traversal and the statement path. At recall
    0.95 (ef ≈ 64) the cap never binds. There only a faster 12-bit kernel would help.</p>"""


def rerank_cap(pas):
    """The reordered 32 KB arm with its rerank capped at 100 candidates, against uncapped"""
    rr, base = "pgvectorfast-tq4-32k-ro-rr10", "pgvectorfast-tq4-32k-ro"
    efs = sorted({p["ef"] for p in s["points"] if p["pass"] == pas and p["arm"] == rr})
    if not efs:
        return ""

    def rec(arm, ef, multi):
        return next(p["recall"] for p in s["points"]
                    if p["pass"] == pas and p["arm"] == arm and p["ef"] == ef and (p["clients"] > 1) == multi)
    rows = []
    for ef in efs:
        cells = [f"<td>{min(ef, 100)} of {ef}</td><td>{rec(base, ef, False):.4f}</td><td>{rec(rr, ef, False):.4f}</td>"]
        for multi in (False, True):
            a, b = qps(pas, base, multi, ef), qps(pas, rr, multi, ef)
            cells.append(f"<td>{a:,.0f}</td><td>{b:,.0f}</td><td><strong>{b / a - 1:+.1%}</strong></td>")
        rows.append(f'<tr><th scope="row">{ef}</th>{"".join(cells)}</tr>')
    head = ('<tr><th rowspan="2" scope="col">ef_search</th><th rowspan="2" scope="col">reranked</th>'
            '<th colspan="2" scope="colgroup">recall@10</th>'
            '<th colspan="3" scope="colgroup">1 client</th><th colspan="3" scope="colgroup">60 clients</th></tr>'
            '<tr><th scope="col">all</th><th scope="col">≤ 100</th>'
            + '<th scope="col">all</th><th scope="col">≤ 100</th><th scope="col">change</th>' * 2 + '</tr>')
    m = {(x["arm"], x["clients"]): x for x in s["matched"] if x["pass"] == pas}
    mrows = []
    for c in (1, 60):
        cells = []
        for t in TARGETS:
            a, b = m.get((base, c), {}).get(t), m.get((rr, c), {}).get(t)
            cells.append(f"<td>{fmt(a)}</td><td>{fmt(b)}</td>"
                         f"<td><strong>{'–' if not a or not b else f'{b / a - 1:+.1%}'}</strong></td>")
        mrows.append(f'<tr><th scope="row">{c} client{"s" if c > 1 else ""}</th>{"".join(cells)}</tr>')
    mhead = ('<tr><th rowspan="2" scope="col">QPS at recall@10</th>'
             + "".join(f'<th colspan="3" scope="colgroup">{t}</th>' for t in TARGETS) + '</tr><tr>'
             + '<th scope="col">all</th><th scope="col">≤ 100</th><th scope="col">change</th>' * len(TARGETS) + '</tr>')
    return f"""
<section>
  <h2>Rerank capped at 10 × k</h2>
  <p class="body">The reordered 32 KB tq4 arm again, with a new setting in pgvectorfast_ntq,
  <code>hnsw.rerank_limit</code>, at 100 = 10 × k. The search still keeps ef_search candidates, but only the
  nearest 100 by 4-bit distance are re-scored with the 12-bit codes. The rest follow them in 4-bit order. Before,
  every candidate was re-scored. Ginger re-scores 6 × k. The index is unchanged; only the server's library and this
  setting differ. Up to ef 100 the cap doesn't bind, so those rows are a control.</p>
  <div class="tbl"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>
  <div class="tbl"><table><thead>{mhead}</thead><tbody>{"".join(mrows)}</tbody></table></div>
  <div class="why">{RR_NOTES}</div>
</section>"""


LOWREC_NOTES = """
    <p><strong>Reading.</strong> With 60 clients, dropping the lock changes nothing (within 4%): the lock-manager
    waits disappear (none in the samples), but socket 0 stays full (20.0 of 20 CPUs busy, socket 1 at 3.3) and
    the backends wait on their clients. Each of the 60 clients has only one query in flight. With 90 clients and
    the lock it doesn't help either, because the extra clients queue on the lock. With 90 clients and no lock
    the work spills onto socket 1 and the server is 99–100% busy at every width: +39% at ef 10 (53,441 →
    74,495 QPS), +31% at 20, +18% at 40 and +5–10% from 80 up.</p>
    <p>Ginger's lead at low recall shrinks from 2.9× to 1.98× at ef 10 and from 2.2× to 1.70× at 20. With the
    server now fully busy, the rest is CPU per query: 537 µs against Ginger's 272 at ef 10. Most of the difference
    is the client/server statement path (about 240 µs per query, see "Why Ginger is still faster"). A real fix for
    the lock would keep VACUUM's guarantee with something cheaper per query, for example an LWLock per index taken
    in shared mode (one atomic operation) or a scan counter VACUUM waits on. Spreading the network card's
    interrupts over both sockets would let 60 clients use the whole server too.</p>"""


# Filled in once the vector kernel's numbers are known: the section's reading.
SIMD_NOTES = """
    <p><strong>Reading.</strong> Recall@10 is identical at every ef_search. With one client the vector kernel gives
    what the profile predicted: +32% at recall 0.95 (1,051 → 1,387 QPS) and +23% at 0.99 (530 → 650), where
    Postgres is now ahead of Ginger's 594. With 60 clients it gives only +8–9% (0.95: 25,809 → 28,010;
    Ginger 43,972), against about +30% predicted. At ef 10 it is 9% slower (55,567 → 50,715 QPS; two earlier runs
    gave 55,728 and 55,567), although only 10 candidates are re-ranked there. Each query used 477 µs of CPU instead
    of 444.</p>
    <p>That points at the clock rather than the kernel. A Skylake-SP core lowers its frequency while it runs heavy
    512-bit floating point, less so for 256-bit, and keeps it lower for a while afterwards. That slows the whole
    query, not just the rerank. At 60 clients every core does it. In the same session at 60 clients, AVX2 is as fast
    as AVX-512 (ef 320: 9,299 against 9,269) and ahead at ef 64 (30,286 against about 29,000). This is not measured
    yet: counting cycles against reference cycles in each kernel would settle it. A kernel that stays in integer
    arithmetic, like the 4-bit traversal kernel and Ginger's scorers, would avoid it.</p>"""


def qps_at_ef(pas, arm, multi, ef):
    """QPS at an ef_search, interpolated in log QPS between the sweep's neighbouring points if not measured"""
    pts = sorted((p["ef"], p["qps"]) for p in s["points"]
                 if p["pass"] == pas and p["arm"] == arm and (p["clients"] > 1) == multi)
    for (e0, q0), (e1, q1) in zip(pts, pts[1:]):
        if e0 <= ef <= e1:
            f = (ef - e0) / (e1 - e0)
            return math.exp(math.log(q0) + f * (math.log(q1) - math.log(q0)))
    return None


def simd_section(pas):
    """The rerank's 4+8 bit distance with a vector kernel, against the scalar loop"""
    new, base = "pgvectorfast-tq4-32k-ro-rr10simd", "pgvectorfast-tq4-32k-ro-rr10"
    efs = sorted({p["ef"] for p in s["points"] if p["pass"] == pas and p["arm"] == new})
    if not efs:
        return ""

    def rec(arm, ef, multi):
        return next(p["recall"] for p in s["points"]
                    if p["pass"] == pas and p["arm"] == arm and p["ef"] == ef and (p["clients"] > 1) == multi)
    rows = []
    for ef in efs:
        cells = [f"<td>{rec(base, ef, False):.4f}</td><td>{rec(new, ef, False):.4f}</td>"]
        for multi in (False, True):
            a, b = qps(pas, base, multi, ef), qps(pas, new, multi, ef)
            cells.append(f"<td>{a:,.0f}</td><td>{b:,.0f}</td><td><strong>{b / a - 1:+.1%}</strong></td>")
        rows.append(f'<tr><th scope="row">{ef}</th>{"".join(cells)}</tr>')
    head = ('<tr><th rowspan="2" scope="col">ef_search</th><th colspan="2" scope="colgroup">recall@10</th>'
            '<th colspan="3" scope="colgroup">1 client</th><th colspan="3" scope="colgroup">60 clients</th></tr>'
            '<tr><th scope="col">scalar</th><th scope="col">AVX-512</th>'
            + '<th scope="col">scalar</th><th scope="col">AVX-512</th><th scope="col">change</th>' * 2 + '</tr>')
    m = {(x["arm"], x["clients"]): x for x in s["matched"] if x["pass"] == pas}
    g = {x["clients"]: x for x in s["matched"] if x["pass"] == pas and x["arm"] == "ginger"}
    mrows = []
    for c in (1, 60):
        cells = []
        for t in TARGETS:
            a, b = m.get((base, c), {}).get(t), m.get((new, c), {}).get(t)
            gg = g.get(40 if c > 1 else 1, {}).get(t)
            cells.append(f"<td>{fmt(a)}</td><td>{fmt(b)}</td>"
                         f"<td><strong>{'–' if not a or not b else f'{b / a - 1:+.1%}'}</strong></td>"
                         f"<td>{'–' if not b or not gg else f'{gg / b:.2f}×'}</td>")
        mrows.append(f'<tr><th scope="row">{c} client{"s" if c > 1 else ""}</th>{"".join(cells)}</tr>')
    mhead = ('<tr><th rowspan="2" scope="col">QPS at recall@10</th>'
             + "".join(f'<th colspan="4" scope="colgroup">{t}</th>' for t in TARGETS) + '</tr><tr>'
             + ('<th scope="col">scalar</th><th scope="col">AVX-512</th><th scope="col">change</th>'
                '<th scope="col">Ginger ahead</th>') * len(TARGETS) + '</tr>')
    extra = ""
    if simd_dir is not None:
        bl = next(iter(sorted(simd_dir.glob("*-build.log"))), None)
        bench = re.findall(r"^(scalar|avx2|avx512)\s+([\d.]+) ns per distance", bl.read_text(), re.M) if bl else []
        kj = simd_dir / "kernels.jsonl"
        krows = []
        if kj.exists():
            ab = {}
            for line in kj.read_text().splitlines():
                r = json.loads(line)
                ab[(r["label"].rsplit("-k", 1)[1], r["clients"], r["search_value"])] = r
            for ef in (64, 320):
                for c in (1, 60):
                    q512 = qps_at_ef(pas, new, c > 1, ef)
                    q2 = ab.get(("avx2", c, ef), {}).get("qps")
                    qs = ab.get(("scalar", c, ef), {}).get("qps")
                    approx = "≈ " if not any(p["ef"] == ef for p in s["points"] if p["arm"] == new) else ""
                    krows.append(f'<tr><th scope="row">{ef}</th><td>{c}</td><td>{fmt(qs)}</td><td>{fmt(q2)}</td>'
                                 f'<td>{approx}{fmt(q512)}</td></tr>')
        if bench:
            extra += ("<p><strong>Kernel alone</strong> (micro-benchmark, dimension 384, warm cache, one core): " +
                      ", ".join(f"{k} {float(v):,.0f} ns" for k, v in bench) + " per distance.</p>")
        if krows:
            extra += ('<p>The three kernels in the same server session (QPS; scalar and AVX2 measured after the '
                      'sweep, AVX-512 from it, ≈ where interpolated between the sweep\'s neighbouring widths):</p><div class="tbl"><table><thead><tr><th scope="col">ef_search</th>'
                      '<th scope="col">clients</th><th scope="col">scalar</th><th scope="col">AVX2</th>'
                      '<th scope="col">AVX-512</th></tr></thead><tbody>' + "".join(krows) + "</tbody></table></div>")
    return f"""
<section>
  <h2>A vector kernel for the rerank</h2>
  <p class="body">The rerank's 4+8 bit distance (<code>HnswTqDistance4k</code>) was a scalar loop over the 384
  dimensions. It is now an AVX-512 kernel: 16 dimensions per step, gathering each sub-code's centroid from the 8 KB
  codebook, flipping its sign, and multiplying it by the float query. There is an AVX2 version and the scalar loop
  too, chosen by <code>hnsw.rerank_kernel</code>. The kernels sum in another order, so distances agree with the
  scalar loop to float rounding (the codec self-test checks each). The rerank cap stays at 100 and the index is
  unchanged.</p>
  <div class="tbl"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>
  <div class="tbl"><table><thead>{mhead}</thead><tbody>{"".join(mrows)}</tbody></table></div>
  <div class="why">{extra}{SIMD_NOTES}</div>
</section>"""


REPO = Path(__file__).resolve().parents[2]


def sat_busy(path):
    """{(label, clients, ef): server busy share} from a saturation_report.py table"""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"\| (\S+) \| (\d+) \| (\d+) \| [\d.–-]+ \| [\d,]+ \| (\d+)% ", line)
        if m:
            out[(m.group(1), int(m.group(2)), int(m.group(3)))] = int(m.group(4))
    return out


def lowrecall_section(pas):
    """More clients and the scan lock at low recall (hn_g5_c90.sh, hn_g5_nolock.sh)"""
    nl, c90 = REPO / "bench/profiles/hn-nolock", REPO / "bench/profiles/hn-g5-c90"
    if not (nl / "results.jsonl").exists():
        return ""
    rows = {}
    for f in (nl / "results.jsonl", c90 / "results.jsonl"):
        if f.exists():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                rows[(r["label"], r["clients"], r["search_value"])] = r
    busy = {**sat_busy(nl / "saturation.md"), **sat_busy(c90 / "saturation.md")}
    gq = {p["ef"]: p["qps"] for p in s["points"] if p["pass"] == pas and p["arm"] == "ginger" and p["clients"] > 1}
    cols = [("g5nolock-on", 60, "60, lock on"), ("g5nolock-off", 60, "60, lock off"),
            ("g5hp32kro-rr10simd-c90-pgvectorfast-tq4", 90, "90, lock on"), ("g5nolock-off", 90, "90, lock off")]
    body = []
    for ef in (10, 20, 40, 80, 160):
        rec = next((rows[(lab, c, ef)]["recall"] for lab, c, _ in cols if (lab, c, ef) in rows), None)
        cells = []
        for lab, c, _ in cols:
            r = rows.get((lab, c, ef))
            b = busy.get((lab, c, ef))
            cells.append("<td>–</td>" if r is None else
                         f"<td>{r['qps']:,.0f}<span class=\"sub\">{'' if b is None else f' {b}%'}</span></td>")
        best = rows.get(("g5nolock-off", 90, ef))
        g = gq.get(ef)
        ratio = f"{g / best['qps']:.2f}×" if g and best else "–"
        body.append(f'<tr><th scope="row">{ef}</th><td>{rec:.4f}</td><td>{fmt(g)}</td>{"".join(cells)}'
                    f'<td><strong>{ratio}</strong></td></tr>')
    head = ('<tr><th scope="col">ef_search</th><th scope="col">recall@10</th><th scope="col">Ginger, 40 threads</th>'
            + "".join(f'<th scope="col">Postgres, {h}</th>' for _, _, h in cols)
            + '<th scope="col">Ginger ahead of the last</th></tr>')
    return f"""
<section>
  <h2>Low recall: the socket and the scan lock</h2>
  <p class="body">At ef_search 10–40 the server was only 60–85% busy at 60 clients, and 90 clients did not help.
  Sampling the server per CPU and the backends' wait events showed two causes:</p>
  <ul class="body">
    <li><strong>One socket did the work.</strong> The network card sits on socket 0, and its interrupts are pinned
    there (CPUs 0–9, 20–29; irqbalance off). Each query's packet wakes its backend from that socket, and the
    scheduler keeps it there. At ef 10 socket 0 was 19–20 of 20 CPUs busy and socket 1 only 3–8.</li>
    <li><strong>Every query took the same heavyweight lock.</strong> The scan takes a shared page lock
    (<code>HNSW_SCAN_LOCK</code>) twice per query, so that VACUUM can wait out in-flight scans. Shared locks don't
    conflict, but a page lock can't use the fast path. Taking and releasing one means an exclusive pass through
    one partition of the shared lock table, and every query uses the same one. Backends waited on
    <code>LWLock:LockManager</code> in 21% of samples at 60 clients and 46% at 90.</li>
  </ul>
  <p class="body">A new setting, <code>hnsw.scan_lock</code> (superuser, on by default), skips the lock. It is for
  this measurement only: with it off, a concurrent VACUUM could free elements a scan is reading. Server CPU busy
  is shown after each QPS; all columns use the vector rerank kernel and the cap of 100.</p>
  <div class="tbl"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>
  <div class="why">{LOWREC_NOTES}</div>
</section>"""


def gap_runs():
    """gap_attrib.Run for every profiled run under gap_dir, by name"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gap_attrib  # noqa: E402
    runs = {}
    for wf in gap_dir.rglob("*.windows.csv"):
        r = gap_attrib.Run(wf.parent, wf.name[: -len(".windows.csv")])
        runs[r.name] = r
    return runs


def gap_groups(r):
    """CPU µs per query in four groups, from the run's call-stack shares"""
    engine, other, _, _ = r.buckets()
    k = r.cpu_us() / (sum(engine.values()) + sum(other.values()))
    g = {"rerank": 0.0, "traversal": 0.0, "query encoding": 0.0, "statement": 0.0}
    for b, n in list(engine.items()) + [("other", sum(other.values()))]:
        if b.startswith("rerank"):
            g["rerank"] += n * k
        elif b.startswith(("traversal", "4-bit", "beam", "graph decode")):
            g["traversal"] += n * k
        elif b.startswith("query encoding"):
            g["query encoding"] += n * k
        else:
            g["statement"] += n * k
    return g


def gap_section():
    """Why Ginger is still faster: hn_g5_gap.sh's measurements"""
    if gap_dir is None or not gap_dir.exists():
        return ""
    R = gap_runs()
    need = ["ginger-b64-t1", "pg-batch-ef64-c1", "pg-lan-ef64-c1", "ginger-b64-t40", "pg-batch-ef64-c40",
            "pg-lan-ef64-c60", "ginger-b160-t40", "pg-lan-ef160-c60"]
    if any(n not in R for n in need):
        return ""
    G = {n: gap_groups(R[n]) for n in need}
    cols = [("ginger-b64-t1", "Ginger"), ("pg-batch-ef64-c1", "Postgres, in server"),
            ("pg-lan-ef64-c1", "Postgres, over LAN"), ("ginger-b64-t40", "Ginger"),
            ("pg-batch-ef64-c40", "Postgres, in server"), ("pg-lan-ef64-c60", "Postgres, over LAN")]
    names = [("rerank", "Rerank (12-bit distances, reads, sort)"),
             ("traversal", "Graph traversal (4-bit distances, neighbour lists, candidates)"),
             ("query encoding", "Query encoding"),
             ("statement", "Statement path: protocol, executor, kernel, other")]
    body = []
    for key, label in names:
        body.append(f'<tr><th scope="row">{label}</th>' +
                    "".join(f"<td>{G[n][key]:,.0f}</td>" for n, _ in cols) + "</tr>")
    body.append('<tr class="tot"><th scope="row">CPU µs per query</th>' +
                "".join(f"<td>{R[n].cpu_us():,.0f}</td>" for n, _ in cols) + "</tr>")
    body.append('<tr><th scope="row">Busy CPUs (of 40)</th>' +
                "".join(f"<td>{R[n].busy():.1f}</td>" for n, _ in cols) + "</tr>")
    body.append('<tr class="tot"><th scope="row">QPS</th>' +
                "".join(f"<td>{R[n].qps:,.0f}</td>" for n, _ in cols) + "</tr>")
    head = ('<tr><th rowspan="2" scope="col">width 64, recall 0.95</th>'
            '<th colspan="3" scope="colgroup">1 thread / client</th>'
            '<th colspan="3" scope="colgroup">full load*</th></tr><tr>' + "".join(f'<th scope="col">{h}</th>' for _, h in cols) + "</tr>")

    def ctr(n, w, ev):
        v = R[n].stats[w].get(ev)
        return v[0] / (R[n].qps * R[n].dur[w]) if v else float("nan")

    def dram(n):
        st = R[n].stats["stat1"]
        b = (st["uncore_imc/cas_count_read/"][0] + st["uncore_imc/cas_count_write/"][0]) * 1048576
        return b / (R[n].qps * R[n].dur["stat1"]) / 1024, b / R[n].dur["stat1"] / 1e9

    crow = []
    for n, label in [("ginger-b64-t1", "Ginger, 1 thread"), ("pg-batch-ef64-c1", "Postgres in server, 1"),
                     ("ginger-b64-t40", "Ginger, 40 threads"), ("pg-batch-ef64-c40", "Postgres in server, 40"),
                     ("pg-lan-ef64-c60", "Postgres over LAN, 60")]:
        cyc, ins = ctr(n, "stat1", "cycles"), ctr(n, "stat1", "instructions")
        kb, gbs = dram(n)
        c4 = R[n].stats["stat4"]["cycles"][0]
        stall = R[n].stats["stat4"]["cycle_activity.stalls_mem_any"][0] / c4
        crow.append(f'<tr><th scope="row">{label}</th><td>{ins / 1e6:.2f}</td><td>{ins / cyc:.2f}</td>'
                    f'<td>{stall:.0%}</td><td>{kb:,.0f}</td><td>{gbs:.0f}</td></tr>')
    chead = ('<tr><th scope="col">width 64</th><th scope="col">M instructions / query</th><th scope="col">IPC</th>'
             '<th scope="col">cycles stalled on memory</th><th scope="col">DRAM KB / query</th>'
             '<th scope="col">DRAM GB/s</th></tr>')

    g1, b1, l1 = (G[n] for n in ("ginger-b64-t1", "pg-batch-ef64-c1", "pg-lan-ef64-c1"))
    g40, b40, l60 = (G[n] for n in ("ginger-b64-t40", "pg-batch-ef64-c40", "pg-lan-ef64-c60"))
    rr = lambda n: R[n].cpu_us()  # noqa: E731
    per_cand_pg = G["pg-batch-ef64-c1"]["rerank"] / 64
    per_cand_g = G["ginger-b64-t1"]["rerank"] / 60
    off_cpu = 1e6 / R["pg-lan-ef64-c1"].qps - rr("pg-lan-ef64-c1")
    q = lambda n: R[n].qps  # noqa: E731
    fixed = rr("pg-lan-ef64-c60") - l60["rerank"] + g40["rerank"]
    proj = R["pg-lan-ef64-c60"].busy() / fixed * 1e6
    proj_b = 40 / (rr("pg-batch-ef64-c40") - b40["rerank"] + g40["rerank"]) * 1e6
    return f"""
<section>
  <h2>Why Ginger is still faster</h2>
  <p class="body">Measured by <code>hn_g5_gap.sh</code> at width 64, where both engines reach recall@10 0.952 and do
  the same work: {R["pg-lan-ef64-c1"].qps / 1:,.0f} and {q("ginger-b64-t1"):,.0f} QPS with one client or thread, and
  {q("pg-lan-ef64-c60"):,.0f} against {q("ginger-b64-t40"):,.0f} under full load
  ({q("ginger-b64-t40") / q("pg-lan-ef64-c60"):.2f}×). On average each query expands the same nodes (73), scores the
  same neighbours (2,500 against 2,492). The Postgres arm is the best one:
  PiPNN + tq4 on 32 KB pages, reordered. "In server" runs 100 queries per statement through pgbench on the server,
  so it has no network and no per-query protocol or transaction. CPU time per query is split by sampled call stacks
  (perf, 10 s per run).</p>
  <div class="tbl"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>
  <p class="note">* Full load: Ginger 40 threads; Postgres 40 backends in server, 60 clients over the LAN.
  CPU µs per query is the node's busy CPU time divided by QPS.</p>
  <div class="why">
    <h3>1. Rerank: a scalar kernel, {per_cand_pg / per_cand_g:.0f}× slower per candidate</h3>
    <p>Postgres reranks each candidate with <code>HnswTqDistance4k</code>, a plain C loop over all 384 dimensions that
    pulls out each 8-bit sub-code and looks up its centroid one dimension at a time: {per_cand_pg:.1f} µs per
    candidate, its reads and the final sort included. Ginger's AVX-512 <code>ExtraPrecisionScoring</code> takes {per_cand_g:.2f} µs, likewise. The rerank is
    {l60["rerank"] / rr("pg-lan-ef64-c60"):.0%} of Postgres's CPU at width 64 and
    {G["pg-lan-ef160-c60"]["rerank"] / rr("pg-lan-ef160-c60"):.0%} at 160, because Postgres reranks all ef candidates
    (160) where Ginger stops at 6 × k = 60. Under full load it costs {l60["rerank"] - g40["rerank"]:,.0f} µs per query
    more than Ginger's.</p>
    <p><strong>What it would take:</strong> a SIMD version of the 4+8-bit kernel, as in Ginger. At Ginger's cost per
    candidate, the LAN arm would reach about {round(proj, -3):,.0f} QPS at width 64 (+{proj / q("pg-lan-ef64-c60") - 1:.0%}), and
    the in-server path about {round(proj_b, -3):,.0f}, level with Ginger. Capping the rerank count helps only at large widths. At 10 × k it is now
    measured (next section): recall unchanged, +13% at recall 0.99 with 60 clients and +27% with one.</p>

    <h3>2. The client/server boundary: {l60["statement"] - g40["statement"]:,.0f} µs of CPU per query under load</h3>
    <p>Each query is a statement over TCP: protocol and socket work, the kernel's network stack, the executor,
    catalog and lock-table lookups, memory contexts, a transaction. That is {l1["statement"]:,.0f} µs of server CPU
    with one client ({b1["statement"]:,.0f} µs in server) and {l60["statement"]:,.0f} µs at 60 clients
    ({b40["statement"]:,.0f} in server). A single client also waits {off_cpu:,.0f} µs per query off the server's CPU
    for the round trip and the client. And 60 clients leave {40 - R["pg-lan-ef64-c60"].busy():.1f} of the 40 CPUs
    idle; 40 clients leave {40 - R["pg-lan-ef64-c40"].busy():.0f} idle ({q("pg-lan-ef64-c40"):,.0f} QPS). Ginger,
    in process, pays none of this. Most of it is inherent to a database server.</p>

    <h3>3. Traversal is faster than Ginger's, but loses most of its lead under load</h3>
    <p>Postgres walks the graph in {b1["traversal"]:,.0f} µs per query on one thread against Ginger's
    {g1["traversal"]:,.0f}: it prefetches every neighbour's tuple and runs an AVX2 4-bit kernel at about 50 ns per
    neighbour, while Ginger stalls on memory at each code row and pays to decode its compressed lists. Under load
    the walk grows to {b40["traversal"]:,.0f} µs ({b40["traversal"] / b1["traversal"]:.2f}×) against Ginger's
    {g40["traversal"]:,.0f} ({g40["traversal"] / g1["traversal"]:.2f}×). Postgres moves
    {dram("pg-batch-ef64-c40")[0] / dram("ginger-b64-t40")[0]:.1f}× the DRAM bytes per query (a 288-byte element tuple
    with 60 bytes of heap TIDs, line pointers and neighbour tuples, against Ginger's 192-byte code rows). It also
    runs at high IPC, so a second hyperthread adds little; Ginger's memory stalls leave room for it. Once the rerank
    is fixed, bandwidth comes next: 47–55 GB/s here, against 77 GB/s measured for reads on this machine.</p>
  </div>
  <div class="tbl"><table><thead>{chead}</thead><tbody>{"".join(crow)}</tbody></table></div>
  <div class="why">

    <h3>Not bottlenecks</h3>
    <ul>
      <li><strong>Contention:</strong> about 80 loads per query hit a line another core had modified, over the LAN at
      60 clients (lock manager hash table, scheduler, IOMMU), roughly 10 µs. In server it is 6.</li>
      <li><strong>Calico, heap fetch, executor:</strong> under 10 µs each.</li>
      <li><strong>Query encoding</strong> costs both engines about 15 µs.</li>
      <li><strong>The kernel's IOMMU mapping</strong> for NIC DMA is under 1% of CPU.</li>
    </ul>
    <p class="note">Ginger ran on the client node and Postgres on the server node, which are identical c220g5s. Profiled
    windows ran a few % slower than the counter windows, so their shares are scaled to the counters' CPU per query.
    vectorfast.so is built without frame pointers, so index time goes by leaf function. Raw tables:
    <code>bench/profiles/hn-gap/report.md</code>.</p>
  </div>
</section>"""


passes = [f["pass"] for f in s["figures"]]
sections = []
for pas in ("hp", "4k"):
    if pas not in passes:
        continue  # the 4 KB pass was stopped before it finished
    if True:
        sections.append(f"""
<section>
  <h2>{PASS_NAME[pas]}</h2>
  {figure(pas)}
  <h3>QPS at matched recall@10</h3>
  <p class="note">Interpolated between neighbouring ef_search points, linear in recall and in log QPS. A dash means the sweep never reached that recall.</p>
  {matched(pas)}
  <details>
    <summary>Every measured point (recall@10 over QPS)</summary>
    {points(pas)}
  </details>
</section>""")
        if pas == "hp":
            sections.append(bottlenecks(pas))
            sections.append(pages32(pas))
            sections.append(reorder32(pas))
            sections.append(gap_section())
            sections.append(rerank_cap(pas))
            sections.append(simd_section(pas))
            sections.append(lowrecall_section(pas))
    else:
        sections.append(f"""
<section>
  <h2>{PASS_NAME[pas]}</h2>
  <p class="pending">The 4 KB pass is still running on the nodes. Its figure and table will be added here when it finishes.</p>
</section>""")

POINTS_JS = {}
for _p in sorted(s["points"], key=lambda q: q["ef"]):
    POINTS_JS.setdefault(f'{_p["pass"]}-{"cN" if _p["clients"] > 1 else "c1"}-{_p["arm"]}', []).append(
        {"ef": _p["ef"], "recall": _p["recall"], "qps": _p["qps"], "clients": _p["clients"]})

SCRIPT = '<script>\n(() => {\n  const ARMS = __ARMS__;\n  const hidden = new Set();\n  const figs = [...document.querySelectorAll("svg.mpl")];\n  // Each panel\'s legend, measured once while every entry is showing. Rows are\n  // anchored at the bottom (loc="lower left"): visible rows take the bottom\n  // slots in their order, and the frame\'s top and right edges move in, the\n  // way matplotlib would draw a legend of only those rows.\n  const legends = [];\n  for (const svg of figs) {\n    const pas = svg.dataset.pass;\n    for (const side of ["c1", "cN"]) {\n      const rows = [];\n      for (const [arm] of ARMS) {\n        const h = svg.getElementById(`${pas}-${side}-leg-${arm}`);\n        const t = svg.getElementById(`${pas}-${side}-legtext-${arm}`);\n        if (!h || !t) continue;\n        const m = /M\\s*(-?[\\d.]+)\\s+(-?[\\d.]+)/.exec(h.querySelector("path").getAttribute("d"));\n        const tb = t.getBBox();\n        rows.push({ arm, h, t, y: +m[2], right: tb.x + tb.width });\n      }\n      if (!rows.length) continue;\n      const box = rows[0].h.parentNode;\n      const frame = box.querySelector(\'g[id^="patch_"] path\');\n      const d0 = frame.getAttribute("d");\n      const nums = d0.match(/-?[\\d.]+/g).map(Number);\n      const xs = nums.filter((_, i) => i % 2 === 0), ys = nums.filter((_, i) => i % 2 === 1);\n      const midX = (Math.min(...xs) + Math.max(...xs)) / 2, midY = (Math.min(...ys) + Math.max(...ys)) / 2;\n      const step = rows.length > 1 ? rows[1].y - rows[0].y : 0;\n      legends.push({ box, frame, d0, midX, midY, step, rows,\n                     slots: rows.map((r) => r.y), right: Math.max(...rows.map((r) => r.right)) });\n    }\n  }\n  const layout = (L) => {\n    const vis = L.rows.filter((r) => !hidden.has(r.arm));\n    L.box.classList.toggle("is-hidden", vis.length === 0);\n    for (const r of L.rows) {\n      const off = hidden.has(r.arm);\n      r.h.classList.toggle("is-hidden", off);\n      r.t.classList.toggle("is-hidden", off);\n    }\n    if (!vis.length) return;\n    const first = L.rows.length - vis.length;\n    vis.forEach((r, k) => {\n      const dy = L.slots[first + k] - r.y;\n      for (const g of [r.h, r.t]) dy ? g.setAttribute("transform", `translate(0 ${dy})`) : g.removeAttribute("transform");\n    });\n    const dTop = first * L.step, dRight = L.right - Math.max(...vis.map((r) => r.right));\n    let i = 0;\n    L.frame.setAttribute("d", L.d0.replace(/-?[\\d.]+/g, (v) => {\n      const n = +v, isY = i++ % 2 === 1;\n      if (isY) return String(n < L.midY ? n + dTop : n);\n      return String(n > L.midX ? n - dRight : n);\n    }));\n  };\n  const render = () => {\n    for (const svg of figs) {\n      const pas = svg.dataset.pass;\n      for (const [arm] of ARMS) for (const side of ["c1", "cN"]) {\n        const g = svg.getElementById(`${pas}-${side}-${arm}`);\n        if (g) g.classList.toggle("is-hidden", hidden.has(arm));\n      }\n    }\n    legends.forEach(layout);\n    for (const box of document.querySelectorAll(".arm input")) box.checked = !hidden.has(box.dataset.arm);\n  };\n  for (const box of document.querySelectorAll(".arm input")) {\n    box.addEventListener("change", () => {\n      box.checked ? hidden.delete(box.dataset.arm) : hidden.add(box.dataset.arm);\n      render();\n    });\n  }\n  document.querySelectorAll(".showall").forEach((b) => b.addEventListener("click", () => { hidden.clear(); render(); }));\n  // Hover: the nearest visible point within 14 px shows its values. Marker\n  // positions come from matplotlib\'s own <use> elements, in plotting order,\n  // which is the order of POINTS (by ef_search).\n  const POINTS = __POINTS__;\n  const COLORS = __COLORS__;\n  const NAMES = Object.fromEntries(ARMS);\n  const NS = "http://www.w3.org/2000/svg";\n  const tip = document.createElement("div");\n  tip.className = "tip";\n  tip.hidden = true;\n  document.body.appendChild(tip);\n  const fmt = (n) => Math.round(n).toLocaleString("en-US");\n  const line = (cls, text) => { const d = document.createElement("div"); d.className = cls; d.textContent = text; return d; };\n  for (const svg of figs) {\n    const pas = svg.dataset.pass;\n    const pts = [];\n    for (const [arm] of ARMS) for (const side of ["c1", "cN"]) {\n      const g = svg.getElementById(`${pas}-${side}-${arm}`);\n      const data = POINTS[`${pas}-${side}-${arm}`];\n      if (!g || !data) continue;\n      const uses = g.querySelectorAll("use");\n      if (uses.length !== data.length) continue;\n      uses.forEach((u, i) => pts.push({ arm, x: +u.getAttribute("x"), y: +u.getAttribute("y"), d: data[i] }));\n    }\n    const ring = document.createElementNS(NS, "circle");\n    ring.setAttribute("r", "6.5");\n    ring.setAttribute("class", "ring");\n    ring.style.display = "none";\n    svg.appendChild(ring);\n    const hide = () => { tip.hidden = true; ring.style.display = "none"; };\n    const show = (e) => {\n      const m = svg.getScreenCTM();\n      if (!m) return;\n      const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());\n      const lim = 14 / m.a;\n      let best = null, bd = lim * lim;\n      for (const q of pts) {\n        if (hidden.has(q.arm)) continue;\n        const dd = (q.x - p.x) ** 2 + (q.y - p.y) ** 2;\n        if (dd < bd) { bd = dd; best = q; }\n      }\n      if (!best) return hide();\n      const d = best.d;\n      const who = best.arm === "ginger" ? `${d.clients} thread${d.clients > 1 ? "s" : ""}` : `${d.clients} client${d.clients > 1 ? "s" : ""}`;\n      const head = line("tip-h", NAMES[best.arm]);\n      const dot = document.createElement("span");\n      dot.className = "dot";\n      dot.style.background = COLORS[best.arm];\n      head.prepend(dot);\n      tip.replaceChildren(head, line("tip-q", `${fmt(d.qps)} QPS`),\n        line("tip-m", `recall@10 ${d.recall.toFixed(4)} · ef_search ${d.ef} · ${who}`));\n      tip.hidden = false;\n      ring.setAttribute("cx", best.x);\n      ring.setAttribute("cy", best.y);\n      ring.style.stroke = COLORS[best.arm];\n      ring.style.display = "";\n      const w = tip.offsetWidth, h = tip.offsetHeight;\n      let left = e.clientX + 14, top = e.clientY - h - 12;\n      if (left + w > window.innerWidth - 8) left = e.clientX - w - 14;\n      if (top < 8) top = e.clientY + 18;\n      tip.style.left = `${Math.max(8, left)}px`;\n      tip.style.top = `${top}px`;\n    };\n    svg.addEventListener("pointermove", show);\n    svg.addEventListener("pointerdown", show);\n    svg.addEventListener("pointerleave", hide);\n  }\n  render();\n})();\n</script>'.replace("__ARMS__", json.dumps(ARMS)).replace("__POINTS__", json.dumps(POINTS_JS)).replace("__COLORS__", json.dumps({a: c for a, (c, _, _) in STYLE.items()}))

TABS_SCRIPT = """<script>
(() => {
  // Runs after the chart script, which measures each legend while Results is showing.
  const tabs = { results: document.getElementById("tab-btn-results"), log: document.getElementById("tab-btn-log") };
  const panels = { results: document.getElementById("tab-results"), log: document.getElementById("tab-log") };
  const show = (name, push) => {
    if (!panels[name]) name = "results";
    for (const k of Object.keys(panels)) {
      panels[k].hidden = k !== name;
      tabs[k].setAttribute("aria-selected", String(k === name));
      tabs[k].tabIndex = k === name ? 0 : -1;
    }
    if (push && location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  };
  for (const [k, b] of Object.entries(tabs)) {
    b.addEventListener("click", () => { show(k, true); window.scrollTo(0, 0); });
    b.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const next = k === "results" ? "log" : "results";
      show(next, true); tabs[next].focus();
    });
  }
  window.addEventListener("hashchange", () => show(location.hash.slice(1), false));
  show(location.hash.slice(1) || "results", false);
})();
</script>"""

page = f"""<title>HN 27.4M on c220g5</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* Layout: one reading column; each figure is a matplotlib SVG at its native width, scrolling sideways on a phone.
   Single light theme by choice: the figures are white matplotlib output. */
:root {{
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #5b5f66;
  --rule: #d9dbe0;
  --head: #f3f4f6;
  --accent: #1f77b4; /* matplotlib C0 */
  --sans: "DejaVu Sans", "Bitstream Vera Sans", -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  --mono: "DejaVu Sans Mono", ui-monospace, Menlo, Consolas, monospace;
  color-scheme: light;
}}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.55 var(--sans); }}
main {{ max-width: 1320px; margin: 0 auto; padding-block: 28px 56px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 36px; }}
header {{ display: grid; gap: 8px; }}
h1 {{ font-size: 1.6rem; line-height: 1.2; margin: 0; text-wrap: balance; }}
h2 {{ font-size: 1.25rem; margin: 0; padding-bottom: 6px; border-bottom: 1px solid var(--rule); }}
h3 {{ font-size: 1rem; margin: 0; }}
p {{ margin: 0; }}
.lede {{ color: var(--muted); max-width: 80ch; }}
section {{ display: grid; gap: 14px; min-width: 0; }}
.fig {{ overflow-x: auto; }}
.fig svg.mpl {{ display: block; width: 100%; min-width: 760px; height: auto; }}
.arms {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px 20px; margin: 0; padding: 10px 14px;
  border: 1px solid var(--rule); border-radius: 4px; }}
.arms legend {{ padding: 0 6px; font-size: 0.8rem; color: var(--muted); letter-spacing: 0.02em; }}
.arm {{ display: inline-flex; align-items: center; gap: 6px; font-size: 0.9rem; cursor: pointer; user-select: none; }}
.arm input {{ margin: 0; accent-color: var(--accent); width: 15px; height: 15px; cursor: pointer; }}
.arm input:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.arm:has(input:not(:checked)) span {{ color: var(--muted); text-decoration: line-through; }}
.arm:has(input:not(:checked)) .sw {{ opacity: 0.3; }}
.showall {{ font: inherit; font-size: 0.85rem; padding: 3px 12px; border: 1px solid var(--rule); border-radius: 4px;
  background: var(--bg); color: var(--fg); cursor: pointer; margin-left: auto; }}
.showall:hover {{ background: var(--head); }}
.showall:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
svg.mpl .is-hidden {{ display: none; }}
svg.mpl .ring {{ fill: none; stroke-width: 2; pointer-events: none; }}
.tip {{ position: fixed; z-index: 20; pointer-events: none; max-width: 300px; padding: 7px 10px;
  background: var(--bg); color: var(--fg); border: 1px solid var(--rule); border-radius: 4px;
  box-shadow: 0 2px 10px rgba(20, 24, 31, 0.14); font-size: 0.82rem; line-height: 1.4; font-variant-numeric: tabular-nums; }}
.tip-h {{ display: flex; align-items: center; gap: 6px; color: var(--muted); }}
.tip-q {{ font-size: 1rem; font-weight: 600; }}
.tip-m {{ color: var(--muted); }}
.tip .dot {{ width: 9px; height: 9px; border-radius: 50%; flex: none; }}
.body {{ max-width: 80ch; }}
.why {{ display: grid; gap: 10px; max-width: 80ch; }}
.why h3 {{ margin-top: 8px; }}
.why ul {{ margin: 0; padding-left: 1.2em; display: grid; gap: 6px; }}
table.kv th {{ font-weight: 400; }}
table.lim td.l, table.lim th {{ text-align: left; white-space: normal; }}
table.lim td.l {{ min-width: 14ch; }}
table.lim thead th span {{ font-weight: 400; }}
.note, .pending {{ color: var(--muted); font-size: 0.9rem; max-width: 80ch; }}
.pending {{ padding: 14px 16px; background: var(--head); border-radius: 4px; }}
.tbl {{ overflow-x: auto; min-width: 0; }}
tr.tot th, tr.tot td {{ font-weight: 600; }}
.sub {{ color: #6b7280; font-size: 0.85em; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 0.9rem; }}
th, td {{ padding: 6px 12px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }}
thead th {{ background: var(--head); font-weight: 600; text-align: center; }}
tbody th {{ text-align: left; font-weight: 500; }}
table span {{ color: var(--muted); font-size: 0.82rem; }}
table.pts td {{ line-height: 1.3; }}
details summary {{ cursor: pointer; color: var(--accent); font-size: 0.9rem; }}
details[open] summary {{ margin-bottom: 10px; }}
summary:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 6px 18px; margin: 0; font-size: 0.92rem; }}
dt {{ color: var(--muted); }}
dd {{ margin: 0; min-width: 0; }}
code {{ font-family: var(--mono); font-size: 0.88em; }}
@media (max-width: 560px) {{ dl {{ grid-template-columns: 1fr; gap: 2px; }} dd {{ margin-bottom: 8px; }} }}
.tabs {{ display: flex; gap: 4px; border-bottom: 1px solid var(--rule); margin-top: -12px; }}
.tabs button {{ font: inherit; font-size: 0.95rem; padding: 8px 16px; border: 1px solid transparent;
  border-bottom: none; border-radius: 4px 4px 0 0; background: none; color: var(--muted); cursor: pointer;
  margin-bottom: -1px; }}
.tabs button[aria-selected="true"] {{ color: var(--fg); border-color: var(--rule); background: var(--bg); font-weight: 600; }}
.tabs button:hover {{ color: var(--fg); }}
.tabs button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.panel {{ display: grid; gap: 36px; min-width: 0; }}
.panel[hidden] {{ display: none; }}
.paper {{ display: grid; gap: 26px; max-width: 74ch; font-family: Georgia, "Times New Roman", serif;
  font-size: 1.02rem; line-height: 1.65; }}
.paper h2, .paper h3, .paper caption, .paper table, .paper .paper-meta {{ font-family: var(--sans); }}
.paper-head {{ display: grid; gap: 6px; }}
.paper-title {{ font-size: 1.35rem; line-height: 1.3; border: none; padding: 0; text-wrap: balance; }}
.paper-meta {{ color: var(--muted); font-size: 0.9rem; }}
.paper section {{ gap: 10px; }}
.paper h3 {{ font-size: 1.05rem; }}
.paper ul, .paper ol {{ margin: 0; padding-left: 1.4em; display: grid; gap: 6px; }}
.paper .abstract {{ padding: 14px 18px; background: var(--head); border-radius: 4px; }}
.paper .abstract p {{ font-size: 0.97rem; }}
.paper caption {{ caption-side: top; text-align: left; font-size: 0.85rem; color: var(--muted); padding-bottom: 6px; }}
.paper em {{ font-style: italic; }}
</style>
<main>
<header>
  <h1>Hacker News 27.4M: QPS vs recall@10 on c220g5</h1>
  <p class="lede">Six engines on 27.4M MiniLM embeddings (384-d, L2), k = 10, at 1 client and at 60 clients
  (Ginger: 1 and 40 threads), plus the tq4 arm again on Postgres built with 32 KB pages, in table order and in Ginger's reordered graph order. Recall@10 runs along a linear axis with a tick every 0.05; QPS is on a log scale.
  Each point is one ef_search value held for 60 s.</p>
</header>
<nav class="tabs" role="tablist" aria-label="Views">
  <button type="button" role="tab" id="tab-btn-results" aria-controls="tab-results" aria-selected="true">Results</button>
  <button type="button" role="tab" id="tab-btn-log" aria-controls="tab-log" aria-selected="false">How we got here</button>
</nav>
<div id="tab-results" role="tabpanel" aria-labelledby="tab-btn-results" class="panel">
{''.join(sections)}
<section>
  <h2>Setup</h2>
  <dl>
    <dt>Data</dt><dd>Hacker News comments, MiniLM-L6 embeddings, 27.4M rows after deduplication, 1,000 held-out queries, exact top-10 ground truth</dd>
    <dt>Server</dt><dd>CloudLab c220g5: 2× Xeon Silver 4114 (20 cores, 40 threads, AVX-512), 192 GB DDR4-2400 on 6 of 12 channels; clock pinned at the 2.2 GHz base, turbo and deep C-states off; memory interleaved across both sockets</dd>
    <dt>Clients</dt><dd>A second c220g5 on the same LAN runs the Postgres clients. Ginger runs in-process on that node, with no network or database in its path.</dd>
    <dt>Postgres</dt><dd>18.6, <code>shared_buffers</code> 112 GB, table and index prewarmed and checked resident before every sweep</dd>
    <dt>HNSW m=32</dt><dd>Built by stock pgvector 0.8.7 (ef_construction 64) on the main table; served by pgvector and by pgvectorfast</dd>
    <dt>PiPNN graph</dt><dd>Built by PiPNN on a copy of the table and imported into pgvectorfast's HNSW format (m = 32, 64 neighbours on layer 0); served by both</dd>
    <dt>tq4</dt><dd>The same PiPNN graph with 4-bit TurboQuant navigation codes, re-ranked with ntq12 codes</dd>
    <dt>Ginger</dt><dd>Its recommended configuration and its own index</dd>
    <dt>Huge pages</dt><dd>Postgres with <code>huge_pages=on</code> from a reserved 2 MB pool, checked on every serve. Ginger through jemalloc with transparent huge pages, kept only if its memory was confirmed on huge pages.</dd>
    <dt>32 KB pages</dt><dd>A second Postgres 18.6 built with <code>-Dblocksize=32</code>, in its own data directory. The
    table was loaded the same way, and the same PiPNN graph was imported, with its row locations rewritten for the
    32 KB table (<code>bench/cloudlab/pipnn_retid.py</code>). Measured like the 8 KB tq4 arm: huge pages, the same
    <code>shared_buffers</code>, sweep and clients.</dd>
    <dt>Checks</dt><dd>pgvector and pgvectorfast must give identical recall at every ef_search on each graph. Both graphs passed: 16 of 16 points identical on each.</dd>
    <dt>Saturation</dt><dd>At 60 clients the server is 90–100% busy for every float arm at every ef_search, and for tq4 from ef_search 80 up. tq4 at ef_search 10 and 20 leaves the server 63% and 76% busy, so those two QPS figures are lower bounds. No link drops or CRC errors in any 60-client point.</dd>
  </dl>
</section>
</div>
<div id="tab-log" role="tabpanel" aria-labelledby="tab-btn-log" class="panel" hidden>
{history_html}
</div>
</main>
{SCRIPT}
{TABS_SCRIPT}
"""
out.write_text(page)
print(out, len(page))
