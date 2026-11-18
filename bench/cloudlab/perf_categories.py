#!/usr/bin/env python3
"""Group a pgvector HNSW profile's self time into categories.

  python3 bench/cloudlab/perf_categories.py bench/profiles/pgvector/pgv-c1-ef40.perf.data ...

Reads `perf report --no-children --comm postgres --percentage relative`, so each
percentage is a share of time spent in Postgres processes (idle and other
processes excluded). Categories go by function; the call paths behind the two
large ones were checked with `perf report -G`: hash_search_with_hash_value is
BufTableLookup under ReadBufferExtended under HnswLoadElementImpl, i.e. the
buffer manager finding each element's page.
"""

import re
import subprocess
import sys
from collections import defaultdict

CATEGORIES = [
    ("distance (vector_l2_squared_distance)", r"^vector_l2_squared_distance$|^l2_distance|Distance"),
    ("buffer manager: page lookup (BufTableLookup)", r"^hash_search_with_hash_value$|^hash_bytes$|^BufTable"),
    ("buffer manager: pin/unpin, locks, refcounts", r"Buffer|^PinBuffer|^UnpinBuffer|^GetPrivateRefCount|^LWLock|^ResourceOwner|^ReservePrivateRefCount|^ForgetPrivateRefCount|^LockBuf"),
    ("HNSW element/neighbor loading", r"^HnswLoad|^HnswGetNeighbors|^HnswPtrAccess|^HnswSearchLayer$|^HnswInit"),
    ("visited set (tidhash)", r"^tidhash|^pointerhash|^offsethash"),
    ("candidate queues (pairing heap)", r"^pairingheap|^merge_children|^CompareNearer|^CompareFurther|^pairingheap_"),
    ("memory allocation", r"^palloc|^pfree|^AllocSet|^MemoryContext|^repalloc"),
    ("kernel / syscalls / network", r"\[k\]"),
]


def categorize(sym: str, dso_kernel: bool) -> str:
    if dso_kernel:
        return "kernel / syscalls / network"
    for name, pat in CATEGORIES:
        if name.startswith("kernel"):
            continue
        if re.search(pat, sym):
            return name
    return "other (executor, protocol, planner, ...)"


def main() -> None:
    for path in sys.argv[1:]:
        out = subprocess.run(
            ["perf", "report", "-i", path, "--no-children", "--comm", "postgres",
             "--percentage", "relative", "--sort", "symbol", "--stdio", "-g", "none",
             "--percent-limit", "0"],
            capture_output=True, text=True, check=True).stdout
        cats = defaultdict(float)
        for line in out.splitlines():
            m = re.match(r"\s+([\d.]+)%\s+\[(.)\]\s+(\S+)", line)
            if m:
                cats[categorize(m.group(3), m.group(2) == "k")] += float(m.group(1))
        print(f"\n{path.split('/')[-1].removesuffix('.perf.data')}  (share of Postgres time)")
        for name, pct in sorted(cats.items(), key=lambda kv: -kv[1]):
            print(f"  {pct:5.1f}%  {name}")


if __name__ == "__main__":
    main()
