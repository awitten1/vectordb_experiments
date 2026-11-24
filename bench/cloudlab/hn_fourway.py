#!/usr/bin/env python3
"""pgvector, pgvectorfast, each on the HNSW graph and on Ginger's PiPNN graph,
and Ginger, on Hacker News 27.4M: QPS at matched recall@10, 4 KB pages and
huge pages, at every client count measured.

  python3 bench/cloudlab/hn_fourway.py > bench/profiles/hn-pipnn/fourway.md

Reads the same rows as pipnn_compare.py (bench/results/*/ and
bench/profiles/hn-pipnn/), deduplicated, since client-results.jsonl is a copy
of the client node's results. Rows of one series at the same ef_search
(rounds) are averaged. Interpolation is pipnn_compare.at: log(QPS) against
-log10(1 - recall).

Source rule: at each client count, take the rows that share a .so and a
protocol with the PiPNN arms. Those ran at 1 and 40 clients, with the client
on the other node at 1 client too. So:
  - pgvectorfast is the fork's .so on the HNSW graph (piphnsw-fork) at 1 and 40
    clients, and upstream pgvectorfast (sc-fast) at 10 and 20;
  - pgvector at 1 client is the LAN run of hn_pipnn_stock.sh (pipstock-hnsw).
The rows that this rule leaves out are in a second table, to show the deltas.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipnn_compare import at, load_rows  # noqa: E402

LEVELS = (0.90, 0.95, 0.97, 0.98, 0.99)
CLIENTS = (1, 10, 20, 40)
ENGINES = ("pgvector", "pgvector + PiPNN", "pgvectorfast", "pgvectorfast + PiPNN", "Ginger")


def lan(label):
    return lambda r: r.get("label") == label


def local(prefix):
    """1-client rounds with the client on the server (main study)."""
    return lambda r: (r.get("label") or "").startswith(prefix) and r.get("server_host") in ("localhost", "127.0.0.1")


GINGER = lan("ginger-recommended")

# pages -> engine -> {clients: (source tag, row filter)}
ARMS = {
    "4 KB": {
        "pgvector": {1: ("pipstock-hnsw", lan("pipstock-hnsw")),
                     **{c: ("sc-stock", lan("sc-stock")) for c in (10, 20, 40)}},
        "pgvector + PiPNN": {c: ("pipstock-pipnn", lan("pipstock-pipnn")) for c in (1, 40)},
        "pgvectorfast": {**{c: ("piphnsw-fork", lan("piphnsw-fork")) for c in (1, 40)},
                         **{c: ("sc-fast", lan("sc-fast")) for c in (10, 20)}},
        "pgvectorfast + PiPNN": {c: ("pippipnn-fork", lan("pippipnn-fork")) for c in (1, 40)},
        "Ginger": {c: ("ginger-recommended", GINGER) for c in CLIENTS},
    },
    "huge pages": {
        "pgvector": {1: ("piphp-hnsw-stock", lan("piphp-hnsw-stock")),
                     **{c: ("hpsc-stock", lan("hpsc-stock")) for c in (10, 20, 40)}},
        "pgvector + PiPNN": {c: ("piphp-pipnn-stock", lan("piphp-pipnn-stock")) for c in (1, 40)},
        "pgvectorfast": {c: ("piphp-hnsw-fork", lan("piphp-hnsw-fork")) for c in (1, 40)},
        "pgvectorfast + PiPNN": {c: ("piphp-pipnn-fork", lan("piphp-pipnn-fork")) for c in (1, 40)},
        # Transparent huge pages via jemalloc thp:always (hn_ginger_thp.sh).
        "Ginger": {c: ("ginger-thp-recommended", lan("ginger-thp-recommended")) for c in CLIENTS},
    },
}

# What the source rule leaves out: (description, pages, clients, filter)
ALTERNATIVES = [
    ("pgvector, client local (pvf-stock-r1/r3)", "4 KB", 1, local("pvf-stock-r")),
    ("pgvectorfast upstream .so, client local (pvf-fast-r2/r4)", "4 KB", 1, local("pvf-fast-r")),
    ("pgvectorfast upstream .so (sc-fast)", "4 KB", 40, lan("sc-fast")),
    ("pgvector, client local (hp-stock-r1/r2)", "huge pages", 1, local("hp-stock-r")),
    ("pgvectorfast upstream .so, client local (hp-fast-r1/r2)", "huge pages", 1, local("hp-fast-r")),
    ("pgvectorfast upstream .so (hpsc-fast)", "huge pages", 40, lan("hpsc-fast")),
    ("Ginger, 4 KB control run beside the huge-pages run (ginger-4kctl-recommended)", "4 KB", 1, lan("ginger-4kctl-recommended")),
    ("Ginger, 4 KB control run beside the huge-pages run (ginger-4kctl-recommended)", "4 KB", 40, lan("ginger-4kctl-recommended")),
]


def dedup(rows):
    seen, out = set(), []
    for r in rows:
        k = json.dumps(r, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def curve(rows, match, clients):
    """{ef: (recall, qps)}, rounds averaged."""
    acc = {}
    for r in rows:
        if (r.get("clients") or r.get("threads")) == clients and match(r):
            acc.setdefault(r["search_value"], []).append((r["recall"], r["qps"]))
    return {w: (sum(p[0] for p in v) / len(v), sum(p[1] for p in v) / len(v)) for w, v in acc.items()}


def qps_at(pts, R):
    v = at(pts, R) if pts else None
    return v[1] if v else None


def fmt(q):
    return f"{q:,.0f}" if q else "–"


def main():
    rows = dedup(load_rows())
    table = {}  # (pages, engine, clients) -> {R: qps}
    for pages, arms in ARMS.items():
        print(f"## QPS at matched recall@10, {pages}\n")
        print("| clients | engine | " + " | ".join(f"{R:.2f}" for R in LEVELS) + " | source |")
        print("|---:|---|" + "---:|" * len(LEVELS) + "---|")
        for c in CLIENTS:
            for e in ENGINES:
                src = arms[e].get(c)
                if not src:
                    print(f"| {c} | {e} | " + " | ".join("–" for _ in LEVELS) + " | not measured |")
                    continue
                pts = curve(rows, src[1], c)
                q = {R: qps_at(pts, R) for R in LEVELS}
                table[(pages, e, c)] = q
                tag = src[0] if pts else f"{src[0]} (no rows yet)"
                print(f"| {c} | {e} | " + " | ".join(fmt(q[R]) for R in LEVELS) + f" | `{tag}` |")
        print()

    for pages in ARMS:
        print(f"## Speedup over pgvector on the HNSW graph, {pages} (recall 0.95 / 0.97)\n")
        print("| clients | " + " | ".join(ENGINES[1:]) + " | Ginger / pgvectorfast + PiPNN |")
        print("|---:|" + "---:|" * len(ENGINES))
        for c in CLIENTS:
            base = table.get((pages, "pgvector", c), {})
            cells = []
            for e in ENGINES[1:]:
                q = table.get((pages, e, c), {})
                cells.append(" / ".join(f"{q[R] / base[R]:.2f}×" if q.get(R) and base.get(R) else "–"
                                        for R in (0.95, 0.97)))
            g, p = table.get((pages, "Ginger", c), {}), table.get((pages, "pgvectorfast + PiPNN", c), {})
            cells.append(" / ".join(f"{g[R] / p[R]:.2f}×" if g.get(R) and p.get(R) else "–" for R in (0.95, 0.97)))
            print(f"| {c} | " + " | ".join(cells) + " |")
        print()

    print("## Rows the source rule leaves out\n")
    print("| series | pages | clients | " + " | ".join(f"{R:.2f}" for R in LEVELS) + " |")
    print("|---|---|---:|" + "---:|" * len(LEVELS))
    for desc, pages, c, match in ALTERNATIVES:
        pts = curve(rows, match, c)
        print(f"| {desc} | {pages} | {c} | " + " | ".join(fmt(qps_at(pts, R)) for R in LEVELS) + " |")
    print()

    print("## Builds (27.4M rows, 40 threads)\n")
    print("| engine | build | index |")
    print("|---|---:|---:|")
    print("| HNSW, m=16, ef_construction=64 (pgvector and pgvectorfast) | 4,105 s (+531 s table load) | 52 GB + 42 GB heap |")
    print("| PiPNN import: COPY out 152 s, Ginger 1,055 s, CREATE INDEX 365 s (serial) | ~1,580 s | 52 GB + 42 GB heap |")
    print("| Ginger recommended | 1,206 s | 17.6 GB, all in |")


if __name__ == "__main__":
    main()
