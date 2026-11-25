#!/usr/bin/env python3
"""Rewrite a PiPNN graph file's heap TIDs for another copy of the same table.

  pipnn_retid.py GRAPH_IN OLD_TIDS NEW_TIDS GRAPH_OUT
  pipnn_retid.py GRAPH_IN OLD_TIDS --order-out ORDER.npy

A graph file (pipnn_export) keys each node by the heap TID of its row. A copy
of the table with the same rows at other TIDs, e.g. in a cluster built with
another page size, can use the same graph once the TIDs are rewritten. Only
the TID block changes: the header (entry node included) and the neighbour
lists, which are node numbers, are copied as they are.

OLD_TIDS and NEW_TIDS list "<block> <offset> <id>" per row of the old and new
table, from
  \\copy (SELECT (ctid::text::point)[0]::bigint, (ctid::text::point)[1]::int, id
          FROM <table>) TO '<file>' WITH (DELIMITER ' ')
Each node's TID is looked up in OLD_TIDS to find its row id, and that id's TID
in NEW_TIDS replaces it. Fails unless every node maps to a distinct row and
both tables hold exactly the graph's rows. It also reports how many nodes sit
at their own position in the new table (node i is the i-th row on disk), which
is 100% for a table loaded in the graph's node order.

--order-out writes the row id of each node, in node order, as a .npy: loading
a table in that order (run_experiment.sh build_pipnn with PIPNN_ORDER) puts
the rows, and the index the import lays out from them, in the graph's node
order, e.g. Ginger's reordering (pipnn_export --reorder).
"""
import argparse
import struct
import sys

import numpy as np

HEADER = struct.Struct("<8sIIQQII")  # magic, version, max degree, nodes, entry, dims, reserved
TID = np.dtype([("blk", "<u4"), ("off", "<u2"), ("pad", "<u2")])


def die(msg):
    sys.exit(f"error: {msg}")


def read_tids(path, n):
    a = np.fromfile(path, dtype=np.int64, sep=" ")
    if a.size % 3:
        die(f"{path}: not <block> <offset> <id> triples")
    a = a.reshape(-1, 3)
    if len(a) != n:
        die(f"{path} lists {len(a)} rows, the graph has {n} nodes")
    blk, off, ids = a[:, 0], a[:, 1], a[:, 2]
    if ids.min() < 0 or ids.max() >= n or np.unique(ids).size != n:
        die(f"{path}: ids are not exactly 0..{n - 1}")
    return blk, off, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_in")
    ap.add_argument("old_tids")
    ap.add_argument("new_tids", nargs="?")
    ap.add_argument("graph_out", nargs="?")
    ap.add_argument("--order-out", help="write each node's row id, in node order, to this .npy")
    a = ap.parse_args()
    if (a.new_tids is None) != (a.graph_out is None):
        ap.error("give both NEW_TIDS and GRAPH_OUT, or neither")
    if a.new_tids is None and a.order_out is None:
        ap.error("nothing to do: give NEW_TIDS and GRAPH_OUT, or --order-out")
    gin, old_path, new_path, gout = a.graph_in, a.old_tids, a.new_tids, a.graph_out
    with open(gin, "rb") as f:
        head = f.read(HEADER.size)
    magic, version, max_degree, n, entry, dims, _ = HEADER.unpack(head)
    if magic != b"PIPNNGR1" or version != 1:
        die(f"{gin} is not a version-1 PiPNN graph file")
    print(f"{gin}: {n} nodes, max degree {max_degree}, entry node {entry}, {dims} dimensions")

    tids = np.fromfile(gin, dtype=TID, count=n, offset=HEADER.size)
    gkey = (tids["blk"].astype(np.int64) << 16) | tids["off"]

    oblk, ooff, oid = read_tids(old_path, n)
    okey = (oblk << 16) | ooff
    order = np.argsort(okey)
    okey_sorted = okey[order]
    pos = np.searchsorted(okey_sorted, gkey)
    pos[pos == n] = 0
    if not np.array_equal(okey_sorted[pos], gkey):
        bad = int(np.flatnonzero(okey_sorted[pos] != gkey)[0])
        die(f"node {bad}'s TID ({tids['blk'][bad]},{tids['off'][bad]}) is not a row of the old table")
    node_id = oid[order][pos]
    if np.unique(node_id).size != n:
        die("two nodes map to the same row")
    print(f"every node matched a row; node i is row id i for {np.mean(node_id == np.arange(n)):.4%} of nodes")
    if a.order_out:
        np.save(a.order_out, node_id.astype(np.int64))
        print(f"wrote {a.order_out}: the row id of each of the {n} nodes, in node order")
    if new_path is None:
        return

    nblk, noff, nid = read_tids(new_path, n)
    by_id_blk = np.empty(n, np.int64)
    by_id_off = np.empty(n, np.int64)
    by_id_blk[nid] = nblk
    by_id_off[nid] = noff
    out = np.zeros(n, dtype=TID)
    out["blk"] = by_id_blk[node_id]
    out["off"] = by_id_off[node_id]
    key = (out["blk"].astype(np.int64) << 16) | out["off"]
    rank = np.empty(n, np.int64)
    rank[np.argsort(key, kind="stable")] = np.arange(n)
    print(f"node i is the i-th row of the new table for {np.mean(rank == np.arange(n)):.4%} of nodes")
    per_page = np.bincount(out["blk"]).max()
    print(f"new TIDs: blocks 0..{out['blk'].max()}, up to {per_page} rows per page "
          f"(old: up to {np.bincount(tids['blk']).max()})")

    adj_off = HEADER.size + n * TID.itemsize
    expect = adj_off + n * (4 + 4 * max_degree)
    with open(gin, "rb") as src, open(gout, "wb") as dst:
        src.seek(0, 2)
        if src.tell() != expect:
            die(f"{gin} is {src.tell()} bytes, expected {expect}")
        dst.write(head)
        out.tofile(dst)
        src.seek(adj_off)
        while chunk := src.read(64 << 20):
            dst.write(chunk)
    print(f"wrote {gout} ({expect} bytes; neighbour lists and entry node unchanged)")


if __name__ == "__main__":
    main()
