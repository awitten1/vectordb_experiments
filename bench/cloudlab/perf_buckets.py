#!/usr/bin/env python3
"""Where each profiled configuration's server CPU time goes, per query.

  python3 bench/cloudlab/perf_buckets.py [--label PREFIX] [--top N] bench/profiles/hn-tq4 <results.jsonl>...

For each <name>.perf.data written by perf_window.sh (499 Hz, system-wide), the
samples taken in postgres processes are grouped by symbol (self time, no
children) into buckets: distance kernel, buffer manager, HNSW traversal, ntq
rerank and query setup, fmgr, pgvectorfast's own access path, kernel, other.
Each bucket is reported as microseconds of server CPU per query:
samples / 499 / (QPS x 20 s window). QPS comes from the client's result row
(label <PREFIX>-<arm>, matched on clients and ef_search; arm is fast, ntq or
stock from the profile name). The top symbols follow, so unbucketed time can
be spotted.
"""

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

WINDOW_S = 20.0
SAMPLE_HZ = 499

# First match wins, so the specific patterns come first.
BUCKETS = [
    ("distance", r"L2SquaredDistance|l2_squared_distance|l2_distance|InnerProduct|vector_negative_inner|"
                 r"HnswTqCodeDistance|TqNavDot|TqCombine|HnswTqValueDistance"),
    ("ntq rerank+setup", r"HnswTqRerank|TqRefineDot|TqGetBits|TqRefineCells|TqRotate|HnswTqMakeValue|"
                         r"TqNorm|CompareCandidatesDesc|list_sort|HnswTqLoadState"),
    ("buffer manager", r"ReadBuffer|StartReadBuffer|PinBuffer|UnpinBuffer|ReleaseBuffer|BufTableLookup|"
                       r"BufTableHashCode|hash_search_with_hash_value|hash_bytes|LockBuffer|LWLock|"
                       r"ResourceOwner|BufferGetBlockNumber|BufferIsValid|WaitReadBuffers|"
                       r"BufferGetPage|MarkBufferDirtyHint|ConditionalLockBuffer|IncrBufferRefCount|"
                       r"GetPrivateRefCount|ReservePrivateRefCount|NewPrivateRefCount|"
                       r"ForgetPrivateRefCount|TrackNewBufferPin|s_lock|perform_spin_delay|LockBufHdr|UnlockBufHdr|"
                       r"ReadRecentBuffer|PrefetchSharedBuffer|PrefetchBuffer|BufferAlloc|PinBufferForBlock|"
                       r"pgstat_count_backend_io_op|pgstat_count_io_op"),
    ("hnsw traversal", r"^Hnsw|tidhash|offsethash|pointerhash|pairingheap|CompareNearer|CompareFurther|"
                       r"CountElement|AddToVisited|InitVisited"),
    ("fmgr+palloc", r"FunctionCall|fmgr|palloc|pfree|AllocSet|MemoryContext|datumCopy|pg_detoast|"
                    r"detoast|repalloc|GenerationAlloc|SlabAlloc|bump_alloc"),
]
CALICO = "pgvectorfast access path"


def bucket(dso: str, sym: str, kind: str) -> str:
    if kind == "k":
        return "kernel"
    for name, pat in BUCKETS:
        if re.search(pat, sym):
            return name
    if "vectorfast" in dso or sym.startswith("calico"):
        return CALICO
    return "other"


def symbols(perf_data: Path) -> list[tuple[int, str, str, str]]:
    out = subprocess.run(
        ["perf", "report", "-i", str(perf_data), "--no-children", "--comm", "postgres", "--sort", "dso,sym",
         "--stdio", "-g", "none", "-n", "--percent-limit", "0"],
        capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        m = re.match(r"^\s+[\d.]+%\s+(\d+)\s+(\S+)\s+\[(.)\]\s+(.+?)\s*$", line)
        if m:
            rows.append((int(m.group(1)), m.group(2), m.group(3), m.group(4)))
    return rows


def main() -> None:
    argv = sys.argv[1:]
    prefix, top = "prof", 25
    while argv and argv[0].startswith("--"):
        if argv[0] == "--label":
            prefix, argv = argv[1], argv[2:]
        elif argv[0] == "--top":
            top, argv = int(argv[1]), argv[2:]
        else:
            sys.exit(f"unknown option {argv[0]}")
    prof = Path(argv[0])
    qps = {}
    for path in argv[1:]:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            lab = r.get("label") or ""
            if lab.startswith(prefix + "-"):
                qps[(lab[len(prefix) + 1:], r["clients"], r["search_value"])] = r["qps"]

    def arm(name: str) -> str:
        return ("fastntq" if "-fastntq-" in name else "fast" if "-fast-" in name
                else "ntq" if "-ntq-" in name else "stock")

    order = {"stock": 0, "fast": 1, "ntq": 2, "fastntq": 3}
    names = ["distance", "buffer manager", "hnsw traversal", "ntq rerank+setup", "fmgr+palloc", CALICO,
             "kernel", "other"]
    table, tops = [], []
    for pd in sorted(prof.glob("*.perf.data"),
                     key=lambda p: (order[arm(p.name)], tuple(int(x) for x in re.findall(r"\d+", p.name)[-2:]))):
        name = pd.name.removesuffix(".perf.data")
        c, ef = (int(x) for x in re.findall(r"\d+", name)[-2:])
        q = qps.get((arm(name), c, ef))
        rows = symbols(pd)
        total = sum(n for n, *_ in rows)
        per = defaultdict(int)
        for n, dso, kind, sym in rows:
            per[bucket(dso, sym, kind)] += n
        # q bound now: the top-symbol lists are printed after the loop
        us = (lambda n, q=q: n / SAMPLE_HZ / (q * WINDOW_S) * 1e6) if q else (lambda n: float("nan"))
        table.append((f"{arm(name)}, {c} client(s), ef {ef}", q, us(total), [(us(per[b]), per[b] / total) for b in names]))
        tops.append((name, total, rows[:top], us))

    print("config | QPS | server CPU us/query | " + " | ".join(names))
    print("|".join("---" for _ in range(3 + len(names))))
    for label, q, tot, cells in table:
        print(f"{label} | {q:.0f} | {tot:.0f} | " if q else f"{label} | ? | ? | ", end="")
        print(" | ".join(f"{u:.0f} ({f * 100:.0f}%)" for u, f in cells))
    for name, total, rows, us in tops:
        print(f"\n## {name}: top symbols ({total} postgres samples)")
        for n, dso, kind, sym in rows:
            print(f"  {n / total * 100:5.1f}%  {us(n):7.1f} us/q  {bucket(dso, sym, kind):<24} {Path(dso).name:<16} {sym}")


if __name__ == "__main__":
    main()
