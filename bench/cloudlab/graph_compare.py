#!/usr/bin/env python3
"""Are two PiPNN graph files the same graph, whatever their node numbering?

  graph_compare.py GRAPH_A TIDS_A GRAPH_B TIDS_B

Each graph file (pipnn_export) keys its nodes by heap TID, and TIDS_A / TIDS_B
list "<block> <offset> <id>" for the table each file's TIDs point into (see
pipnn_retid.py), so every node and every neighbour can be named by its row id.
Two files are the same graph if every row has the same set of neighbours in
both. This is the case for a graph written with and without --reorder from one
build, and need not be for two builds (PiPNN's build can differ run to run).

Prints the share of rows whose neighbour sets are identical, the share of edges
the two graphs have in common (intersection over union), and whether the entry
point is the same row.
"""
import struct
import sys

import numpy as np

HEADER = struct.Struct("<8sIIQQII")
TID = np.dtype([("blk", "<u4"), ("off", "<u2"), ("pad", "<u2")])
INVALID = 0xFFFFFFFF
CHUNK = 1_000_000


def die(msg):
    sys.exit(f"error: {msg}")


class Graph:
    def __init__(self, path, tids_path):
        with open(path, "rb") as f:
            magic, version, self.max_degree, self.n, self.entry, dims, flags = HEADER.unpack(f.read(HEADER.size))
        if magic != b"PIPNNGR1" or version != 1:
            die(f"{path} is not a version-1 PiPNN graph file")
        n = self.n
        tids = np.fromfile(path, dtype=TID, count=n, offset=HEADER.size)
        self.adj = np.memmap(path, dtype="<u4", mode="r", offset=HEADER.size + n * TID.itemsize,
                             shape=(n, 1 + self.max_degree))
        t = np.fromfile(tids_path, dtype=np.int64, sep=" ").reshape(-1, 3)
        if len(t) != n:
            die(f"{tids_path} lists {len(t)} rows, {path} has {n} nodes")
        key = (t[:, 0] << 16) | t[:, 1]
        order = np.argsort(key)
        ks = key[order]
        gkey = (tids["blk"].astype(np.int64) << 16) | tids["off"]
        pos = np.searchsorted(ks, gkey)
        pos[pos == n] = 0
        if not np.array_equal(ks[pos], gkey):
            die(f"some TIDs of {path} are not rows listed in {tids_path}")
        self.node_id = t[:, 2][order][pos]          # node -> row id
        self.node_of = np.empty(n, np.int64)        # row id -> node
        self.node_of[self.node_id] = np.arange(n)
        print(f"{path}: {n} nodes, max degree {self.max_degree}, entry node {self.entry} "
              f"(row {self.node_id[self.entry]}), flags {flags:#x}")

    def neighbour_ids(self, ids):
        """Sorted neighbour row ids for these rows; INVALID-padded to max_degree"""
        rows = np.asarray(self.adj[np.sort(self.node_of[ids])])
        # rows come back in node order; put them back in the order of ids
        back = np.argsort(np.argsort(self.node_of[ids]))
        rows = rows[back]
        deg, nb = rows[:, 0], rows[:, 1:].astype(np.int64)
        valid = nb != INVALID
        out = np.where(valid, self.node_id[np.where(valid, nb, 0)], np.int64(INVALID))
        out.sort(axis=1)
        return deg, out


def main():
    a = Graph(sys.argv[1], sys.argv[2])
    b = Graph(sys.argv[3], sys.argv[4])
    if a.n != b.n:
        die(f"the graphs have {a.n} and {b.n} nodes")
    n = a.n
    if a.max_degree != b.max_degree:
        print(f"note: max degree {a.max_degree} against {b.max_degree}")
    same_rows = inter = union = 0
    for lo in range(0, n, CHUNK):
        ids = np.arange(lo, min(n, lo + CHUNK))
        da, na = a.neighbour_ids(ids)
        db, nb = b.neighbour_ids(ids)
        w = max(na.shape[1], nb.shape[1])
        na = np.pad(na, ((0, 0), (0, w - na.shape[1])), constant_values=INVALID)
        nb = np.pad(nb, ((0, 0), (0, w - nb.shape[1])), constant_values=INVALID)
        same_rows += int(np.all(na == nb, axis=1).sum())
        # Edges as (row, neighbour) keys; each row's neighbours are distinct.
        r = np.repeat(ids, w).reshape(-1, w)
        ka = (r[na != INVALID] << 32) | na[na != INVALID]
        kb = (r[nb != INVALID] << 32) | nb[nb != INVALID]
        common = np.intersect1d(ka, kb, assume_unique=True).size
        inter += common
        union += ka.size + kb.size - common
    print(f"rows with identical neighbour sets: {same_rows / n:.4%}")
    print(f"edges in common: {inter:,} of {union:,} ({inter / union:.4%} of the union)")
    print(f"entry point: row {a.node_id[a.entry]} and row {b.node_id[b.entry]}"
          f" ({'same' if a.node_id[a.entry] == b.node_id[b.entry] else 'different'})")


if __name__ == "__main__":
    main()
