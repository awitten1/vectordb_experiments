#!/usr/bin/env python3
"""How many index pages does expanding one node touch, for a graph file
written by pipnn_export, under three layouts of the same graph?

  graph_layout.py FILE [--per-page 4] [--chunk 1000000]

A search that expands node u reads the element tuple of every neighbour of u.
In the index, element tuples sit `--per-page` to a page (4 for 384-d vectors
at m = 32: 27,445,759 elements on 6,861,440 pages). So the pages one
expansion touches are the distinct pages among u's neighbours' positions.

Layouts:
  heap    where today's import puts node u: the rank of u's heap TID, since
          CREATE INDEX scans the heap in TID order
  node    position u, the file's own order (Ginger's reordered order when the
          file was written with --reorder; otherwise the same as heap)
  random  a random permutation, the no-locality baseline

Per layout, averaged over nodes and weighted by in-degree (how often a node
is someone's neighbour, a proxy for how often a search reaches it):
  pages     distinct index pages among a node's neighbours
  same      share of neighbours on the node's own page
  regions   distinct 2 MB regions (256 pages of 8 KB) among them: TLB reach
"""
import argparse
import struct
import sys

import numpy as np

REGION_PAGES = 256  # 2 MB of 8 KB pages
INVALID = 0xFFFFFFFF


def distinct_per_row(vals, valid):
    """Number of distinct values per row, among the valid entries."""
    sentinel = np.iinfo(np.int64).max
    v = np.where(valid, vals, sentinel)
    v.sort(axis=1)
    new = np.ones_like(v, dtype=bool)
    new[:, 1:] = v[:, 1:] != v[:, :-1]
    return (new & (v != sentinel)).sum(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--per-page", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    with open(a.file, "rb") as f:
        magic, version, max_degree, n, entry, dims, flags = struct.unpack("<8sIIQQII", f.read(40))
    if magic != b"PIPNNGR1" or version != 1:
        sys.exit(f"{a.file} is not a pipnn_export graph file")
    tids = np.fromfile(a.file, dtype="<u8", count=n, offset=40)
    adj = np.memmap(a.file, dtype="<u4", mode="r", offset=40 + 8 * n, shape=(n, max_degree + 1))

    # TID = (uint32 block, uint16 offset, uint16 pad): order by block, then offset.
    key = (tids & 0xFFFFFFFF).astype(np.int64) * 65536 + ((tids >> 32) & 0xFFFF).astype(np.int64)
    heap_pos = np.empty(n, dtype=np.int64)
    heap_pos[np.argsort(key, kind="stable")] = np.arange(n, dtype=np.int64)
    layouts = {
        "heap": heap_pos,
        "node": np.arange(n, dtype=np.int64),
        "random": np.random.default_rng(a.seed).permutation(n).astype(np.int64),
    }

    indeg = np.zeros(n, dtype=np.int64)
    sums = {name: {"pages": [0.0, 0.0], "same": [0.0, 0.0], "regions": [0.0, 0.0]} for name in layouts}
    edges = 0
    for lo in range(0, n, a.chunk):
        hi = min(n, lo + a.chunk)
        block = np.asarray(adj[lo:hi])
        deg = block[:, 0].astype(np.int64)
        nb = block[:, 1:].astype(np.int64)
        valid = np.arange(max_degree)[None, :] < deg[:, None]
        nb = np.where(valid, nb, 0)
        if (nb[valid] >= n).any():
            sys.exit("a neighbour is out of range")
        np.add.at(indeg, nb[valid], 1)
        edges += int(deg.sum())
        for name, pos in layouts.items():
            page = pos[nb] // a.per_page
            own = pos[lo:hi] // a.per_page
            s = sums[name]
            s["pages"][0] += distinct_per_row(page, valid).sum()
            s["same"][0] += ((page == own[:, None]) & valid).sum()
            s["regions"][0] += distinct_per_row(page // REGION_PAGES, valid).sum()
        print(f"  {hi:,} of {n:,} nodes", file=sys.stderr)

    # In-degree weighted pass (needs in-degrees, so a second pass).
    for lo in range(0, n, a.chunk):
        hi = min(n, lo + a.chunk)
        block = np.asarray(adj[lo:hi])
        deg = block[:, 0].astype(np.int64)
        nb = block[:, 1:].astype(np.int64)
        valid = np.arange(max_degree)[None, :] < deg[:, None]
        nb = np.where(valid, nb, 0)
        w = indeg[lo:hi].astype(np.float64)
        for name, pos in layouts.items():
            page = pos[nb] // a.per_page
            own = pos[lo:hi] // a.per_page
            s = sums[name]
            s["pages"][1] += (distinct_per_row(page, valid) * w).sum()
            s["same"][1] += (((page == own[:, None]) & valid).sum(axis=1) * w).sum()
            s["regions"][1] += (distinct_per_row(page // REGION_PAGES, valid) * w).sum()

    wsum = float(indeg.sum())
    wdeg = float((adj[:, 0].astype(np.float64) * indeg).sum())
    print(f"{a.file}: {n:,} nodes, mean degree {edges / n:.2f}, flags {flags} "
          f"({'reordered' if flags & 1 else 'read order'}), {a.per_page} elements per page\n")
    print("| layout | pages / expansion | same page | 2 MB regions / expansion | pages (in-degree weighted) | same page (weighted) | regions (weighted) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name in layouts:
        s = sums[name]
        print(f"| {name} | {s['pages'][0] / n:.2f} | {s['same'][0] / edges:.1%} | {s['regions'][0] / n:.2f} "
              f"| {s['pages'][1] / wsum:.2f} | {s['same'][1] / wdeg:.1%} | {s['regions'][1] / wsum:.2f} |")
    h, d = sums["heap"]["pages"][0], sums["node"]["pages"][0]
    print(f"\nnode layout touches {1 - d / h:.1%} fewer pages per expansion than heap layout "
          f"({'gate passed: 25% or more' if d <= 0.75 * h else 'under the 25% gate'})")


if __name__ == "__main__":
    main()
