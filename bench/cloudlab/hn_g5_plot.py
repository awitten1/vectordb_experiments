#!/usr/bin/env python3
"""QPS vs recall@10 for the c220g5 Hacker News study (hn_g5.sh), drawn with
matplotlib, plus QPS at matched recall.

  hn_g5_plot.py RESULTS.jsonl OUTDIR [--prefix smoke-]

RESULTS.jsonl holds ann_bench / ginger_bench rows labelled g5hp-<arm> (huge
pages) and g5-<arm> (4 KB pages). Writes, for each pass that has rows,
OUTDIR/<pass>.svg (1 client and 60 clients side by side: x is recall@10,
linear, a tick every 0.05; y is QPS, log), and OUTDIR/summary.json with the
points and the matched-recall table (QPS interpolated in log space between
sweep points).
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator  # noqa: E402

# Series in plotting (and tab10 colour) order: arm label suffix, legend name, marker, line style.
ARMS = [
    ("pgvector-hnsw", "pgvector, HNSW m=32", "o", "--"),
    ("pgvectorfast-hnsw", "pgvectorfast, HNSW m=32", "s", "-"),
    ("pgvector-pipnn", "pgvector, PiPNN graph", "^", "--"),
    ("pgvectorfast-pipnn", "pgvectorfast, PiPNN graph", "D", "-"),
    ("pgvectorfast-tq4", "pgvectorfast, PiPNN graph + tq4", "v", "-"),
    ("ginger", "Ginger", "*", "-"),
    ("pgvectorfast-tq4-32k", "pgvectorfast, PiPNN graph + tq4, 32 KB pages", "X", "-"),
    ("pgvectorfast-tq4-32k-ro", "pgvectorfast, PiPNN graph + tq4, 32 KB pages, reordered", "P", "-"),
    ("pgvectorfast-tq4-32k-ro-rr10", "pgvectorfast, PiPNN graph + tq4, 32 KB pages, reordered, rerank ≤ 100",
     "h", "-"),
    ("pgvectorfast-tq4-32k-ro-rr10simd",
     "pgvectorfast, PiPNN graph + tq4, 32 KB pages, reordered, rerank ≤ 100, AVX-512 rerank", ">", "-"),
    ("pgvectorfast-tq4-32k-ro-rr10simd-nolock90",
     "pgvectorfast, as above + no scan lock, 90 clients", "p", "-"),
]
# tab10 has ten colours; later arms get their own
COLOR = {"pgvectorfast-tq4-32k-ro-rr10simd-nolock90": "#393b79"}
# Labels outside the <pass label>-<arm> pattern: hn_g5_32k.sh's 32 KB tq4 run.
EXTRA = {"g5hp32k-pgvectorfast-tq4": ("hp", "pgvectorfast-tq4-32k"),
         "g5hp32kro-pgvectorfast-tq4": ("hp", "pgvectorfast-tq4-32k-ro"),
         "g5hp32kro-rr10-pgvectorfast-tq4": ("hp", "pgvectorfast-tq4-32k-ro-rr10"),
         "g5hp32kro-rr10simd-pgvectorfast-tq4": ("hp", "pgvectorfast-tq4-32k-ro-rr10simd"),
         # label@clients: only that client count's rows
         "g5nolock-off@90": ("hp", "pgvectorfast-tq4-32k-ro-rr10simd-nolock90"),
         # the full sweep of the same (hn_g5_nolock_full.sh); the latest row per point wins
         "g5nolock-full@90": ("hp", "pgvectorfast-tq4-32k-ro-rr10simd-nolock90")}
PASSES = [("hp", "g5hp", "Huge pages"), ("4k", "g5", "4 KB pages")]
TARGETS = [0.90, 0.95, 0.99]


def load(path, prefix):
    """{(pass, arm, clients): [(ef, recall, qps)]}, the latest row per point"""
    latest = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        label = r["label"]
        bare = label.removeprefix(prefix)
        if f"{bare}@{r['clients']}" in EXTRA:
            bare = f"{bare}@{r['clients']}"
        if bare in EXTRA:
            pas, arm = EXTRA[bare]
            key = (pas, arm, r["clients"], r["search_value"])
            if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
                latest[key] = r
            continue
        for pas, lp, _ in PASSES:
            for arm, *_ in ARMS:
                if label == f"{prefix}{lp}-{arm}":
                    key = (pas, arm, r["clients"], r["search_value"])
                    if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
                        latest[key] = r
    series = {}
    for (pas, arm, c, ef), r in latest.items():
        series.setdefault((pas, arm, c), []).append((ef, r["recall"], r["qps"]))
    for v in series.values():
        v.sort()
    return series


def at_recall(pts, target):
    """QPS at a recall, interpolating log QPS linearly in recall between sweep points"""
    pts = sorted(pts, key=lambda p: p[1])
    for (_, r0, q0), (_, r1, q1) in zip(pts, pts[1:]):
        if r0 <= target <= r1 and r1 > r0:
            f = (target - r0) / (r1 - r0)
            return math.exp(math.log(q0) + f * (math.log(q1) - math.log(q0)))
    return None


def qps_fmt(y, _):
    if y >= 1000:
        return f"{y / 1000:g}k"
    return f"{y:g}"


def axes(ax, series, pas, load_name, multi):
    side = "cN" if multi else "c1"
    lo = 1.0
    for i, (arm, name, marker, ls) in enumerate(ARMS):
        keys = [k for k in series if k[0] == pas and k[1] == arm and (k[2] > 1) == multi]
        if not keys:
            continue
        pts = series[keys[0]]
        lo = min(lo, min(p[1] for p in pts))
        ax.plot([p[1] for p in pts], [p[2] for p in pts], marker=marker, linestyle=ls,
                color=COLOR.get(arm, f"C{i}"), label=name,
                markersize={"*": 10, "X": 7, "P": 7, "h": 7, ">": 7, "p": 7}.get(marker, 6),
                gid=f"{pas}-{side}-{arm}")  # the page toggles series by these ids
    ax.set_yscale("log")
    ax.set_xlim(math.floor(lo / 0.05) * 0.05 - 0.005, 1.005)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(qps_fmt))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", axis="y", alpha=0.1)
    ax.set_xlabel("recall@10")
    ax.set_ylabel("QPS (log scale)")
    ax.set_title(load_name)
    # Each panel carries its own legend, in the lower left, which the curves
    # (QPS falling as recall rises) leave empty.
    leg = ax.legend(loc="lower left", fontsize="small")
    arm_of = {name: arm for arm, name, *_ in ARMS}
    for h, t in zip(leg.legend_handles, leg.get_texts()):
        h.set_gid(f"{pas}-{side}-leg-{arm_of[t.get_text()]}")
        t.set_gid(f"{pas}-{side}-legtext-{arm_of[t.get_text()]}")


def figure(series, pas, out):
    """One pass: 1 client and 60 clients side by side"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.2))
    axes(a1, series, pas, "1 client (Ginger: 1 thread)", False)
    axes(a2, series, pas, "60 clients (Ginger: 40 threads)", True)
    fig.suptitle(dict((p, t) for p, _, t in PASSES)[pas], fontsize="x-large")
    fig.tight_layout()
    # Salt the SVG's generated ids per pass, so two figures inlined in one
    # page don't share clip-path or marker ids.
    with matplotlib.rc_context({"svg.hashsalt": f"hn-g5-{pas}"}):
        fig.savefig(out, format="svg")
    plt.close(fig)


def main():
    args = sys.argv[1:]
    prefix = ""
    if "--prefix" in args:
        i = args.index("--prefix")
        prefix = args[i + 1]
        del args[i:i + 2]
    results, outdir = args
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    series = load(results, prefix)
    summary = {"points": [], "matched": [], "figures": []}
    for pas, _, _ in PASSES:
        if not any(k[0] == pas for k in series):
            continue
        f = outdir / f"{pas}.svg"
        figure(series, pas, f)
        summary["figures"].append({"pass": pas, "file": f.name})
    for (pas, arm, c), pts in sorted(series.items()):
        for ef, r, q in pts:
            summary["points"].append({"pass": pas, "arm": arm, "clients": c, "ef": ef,
                                      "recall": r, "qps": q})
        summary["matched"].append({"pass": pas, "arm": arm, "clients": c,
                                   **{f"{t:.2f}": at_recall(pts, t) for t in TARGETS}})
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    for m in summary["matched"]:
        print(m["pass"], m["arm"], m["clients"],
              *[("-" if m[f"{t:.2f}"] is None else f"{m[f'{t:.2f}']:,.0f}") for t in TARGETS])


if __name__ == "__main__":
    main()
