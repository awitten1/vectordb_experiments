#!/usr/bin/env python3
"""Per-query costs of Postgres and Ginger from hn_g5_gap.sh's windows.

  python3 bench/cloudlab/gap_attrib.py bench/profiles/hn-gap [--top N] > report.md

Reads every <name>.windows.csv under the directory (server/ and client/ from
gap_window.sh) and prints markdown:
  1. per run: QPS over the counter windows, busy CPUs, CPU µs per query,
     cycles and instructions per query, IPC, topdown L1, DRAM and cross-socket
     (UPI) bytes per query and bandwidth, L3-miss loads per query and where
     they were served, HITM loads, stall and page-walk shares, context
     switches per query
  2. per run: CPU µs per query by cost centre, from the folded call stacks
     (busy CPU time shared out by sample counts, idle and load generators
     taken out)
  3. work per query: Postgres uprobe counts (pg_counts.sh) and Ginger's
     search statistics (--stats)
  4. per loaded run with c2c: the most contended cache lines and their code
  5. per run: the top leaf functions not put in a cost centre (--top), to
     check the buckets
QPS comes from the span of the five counter windows (stat1..stat5), where no
sampler slows the engine; the Postgres transaction counts are flushed by each
backend about once a second, too coarse for one 10 s window.
"""
import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

NCPU = 40
STAT_WINDOWS = ("stat1", "stat2", "stat3", "stat4", "stat5")
IDLE_COMMS = {"swapper"}
LOADGEN_COMMS = {"pgbench", "python3", "python", "perf", "psql", "sleep", "bash", "date", "awk", "gzip"}

# --- Postgres cost centres, decided by the frames of a stack (root first) ------
TRAVERSAL = ("HnswLoadUnvisitedRaw", "HnswSearchLayerDiskFastCore", "HnswSearchLayerUpperDiskFast",
             "HnswSearchLayerFromEntryFastCands", "HnswSearchLayerFastCands", "HnswLoadNeighborTids",
             "HnswLoadUnvisitedMissingOrPinSlow", "HnswLoadFastCandSlow", "HnswEvalAsyncUnvisitedBatchFast",
             "hnsw_fast_visited", "HnswSearchLayer")
INDEX_ROOTS = ("hnswgettuple", "hnswrescan", "hnswbeginscan", "hnswendscan", "index_getnext", "index_rescan",
               "index_beginscan", "index_endscan")
PG_ORDER = [
    # (cost centre, frame patterns; first match anywhere in the stack wins, in this order)
    ("index: rerank (ntq12)", ("HnswTqRerank", "RerankOne", "RerankDistance", "ReadRerankTid", "TqRefine")),
    ("index: tq4 distances", ("HnswTqQueryDistance", "TqNavDot", "TqCodeDistance", "HnswTqDot")),
    ("index: Calico lookup, pin", ("Calico", "calico_")),
    ("index: traversal", TRAVERSAL),
    ("index: other", INDEX_ROOTS + ("Hnsw", "hnsw")),
    ("heap fetch", ("heapam_index_fetch", "index_fetch_heap", "heap_hot_search", "heapam_index_fetch_tuple")),
    ("transaction, snapshot, stats", ("StartTransaction", "CommitTransaction", "AbortTransaction",
                                      "GetSnapshotData", "ProcArrayEndTransaction", "pgstat_", "AtEOXact",
                                      "ResourceOwnerRelease", "CleanupTransaction")),
    ("client I/O: protocol + kernel network", ("pq_", "secure_read", "secure_write", "socket_",
                                               "__sys_sendto", "__sys_recvfrom", "sock_", "tcp_",
                                               "WaitEventSetWait", "epoll_", "ReadCommand", "SocketBackend")),
    ("parse, plan", ("exec_parse_message", "exec_bind_message", "pg_parse_query", "pg_analyze",
                     "pg_plan_query", "GetCachedPlan", "planner", "raw_parser", "parse_analyze",
                     "BuildCachedPlan", "RevalidateCachedQuery")),
    ("executor", ("ExecutorRun", "ExecutorStart", "ExecutorEnd", "ExecutorFinish", "PortalRun",
                  "PortalStart", "ExecProcNode", "ExecInit", "ExecEnd", "standard_Executor",
                  "ExecNestLoop", "ExecScan", "ExecAgg")),
    ("locks, pins (Postgres)", ("LWLock", "s_lock", "perform_spin_delay", "PinBuffer", "UnpinBuffer",
                                "LockBuffer", "LockAcquire", "LockRelease")),
    ("memory contexts", ("AllocSet", "palloc", "pfree", "MemoryContext", "GenerationAlloc", "SlabAlloc")),
]
KERNEL_SCHED = ("schedule", "__schedule", "finish_task_switch", "pick_next_task", "try_to_wake_up",
                "futex_", "do_futex")

def leafname(frame: str) -> str:
    return frame.rsplit("@", 1)[0]


def dso(frame: str) -> str:
    return frame.rsplit("@", 1)[-1] if "@" in frame else ""


# vectorfast.so is built without frame pointers, so its stacks stop at the
# leaf: index code (and libm/postgres helpers it calls) goes by leaf function.
PG_LEAF = [
    ("rerank: 12-bit distance (HnswTqDistance4k, scalar)", ("HnswTqDistance4k",)),
    ("rerank: other (reads, sort)", ("HnswTqRerank", "RerankOne", "RerankDistance", "ReadRerankTid",
                                     "CompareFastCands", "pg_qsort")),
    ("traversal: tq4 distance (Dot4Avx2)", ("Dot4Avx2", "Dot4Scalar", "HnswTqDistance4", "HnswTqDot4",
                                            "HnswTqQueryDistance")),
    ("traversal: graph walk (lists, prefetch, visited, heaps)", TRAVERSAL),
    ("query encoding (rotate, quantize)", ("HnswTqEncodeQuery", "HnswTqRotate", "HnswTqMakeQuery",
                                            "FlipSigns", "roundf", "Householder")),
    ("Calico lookup, pin", ("Calico", "calico_")),
    ("exact distance of the k results (l2_distance)", ("l2_distance", "vector_l2")),
]
GINGER_LEAF = [
    ("4-bit distances (TurboQuantScoringInt8BatchDims)", ("TurboQuantScoringInt8",)),
    ("rerank: 12-bit distance (ExtraPrecisionScoring)", ("ExtraPrecisionScoring",)),
    ("graph decode (compressed neighbour lists)", ("DecodeAdaptive", "CompressedCSRGraph", "Neighbors(")),
    ("beam loop, candidate merge and sort", ("__do_visit", "DispatchBeamSearch", "MergeBeamAndCandidates",
                                             "emplace_back", "__introsort", "__sort<", "__insertion_sort",
                                             "__tuple_cmp", "BeamSearch")),
    ("query encoding (rotate, quantize)", ("EncodeQuery", "PrepareQueryForQuantization", "Hadamard",
                                           "FlipSigns", "SubspaceMix", "Rotate", "roundf")),
    ("allocator", ("je_", "malloc", "free", "operator new", "operator delete", "AllocateAligned")),
]


def leaf_match(name: str, table) -> str | None:
    for b, pats in table:
        if any(p in name for p in pats):
            return b
    return None


def pg_bucket(frames: list[str]) -> str:
    leaf = frames[-1] if frames else "?"
    b = leaf_match(leafname(leaf), PG_LEAF)
    if b:
        return b
    if "vectorfast" in dso(leaf):
        return "index: other"
    names = [leafname(f) for f in frames]
    for b, pats in PG_ORDER:
        if b.startswith("index"):
            continue
        if any(n.startswith(p) or p in n for n in names for p in pats):
            return b
    if "kernel" in dso(leaf):
        if any(n.startswith(KERNEL_SCHED) for n in names):
            return "kernel: scheduler, futex"
        return "kernel: other"
    return "postgres: other"


def ginger_bucket(frames: list[str]) -> str:
    leaf = frames[-1] if frames else "?"
    b = leaf_match(leafname(leaf), GINGER_LEAF)
    if b:
        return b
    if "kernel" in dso(leaf):
        return "kernel: scheduler, futex" if any(leafname(f).startswith(KERNEL_SCHED) for f in frames) else "kernel: other"
    if "libc" in dso(leaf) or "libstdc++" in dso(leaf) or "libm" in dso(leaf):
        return "libc, libstdc++, libm"
    return "ginger: other"


def read_stat(p: Path) -> dict:
    """perf stat -x, output: event -> (value, running %); tma_* metrics -> value"""
    out = {}
    if not p.exists():
        return out
    for row in csv.reader(l for l in p.open() if l.strip() and not l.startswith("#")):
        if len(row) >= 7 and "tma_" in row[6]:
            try:
                out[row[6].split()[-1]] = float(row[5])
            except ValueError:
                pass
        if len(row) >= 5 and row[2] and row[0] not in ("", "<not counted>", "<not supported>"):
            try:
                out[row[2]] = (float(row[0]), float(row[4]) if row[4] else 100.0)
            except ValueError:
                pass
    return out


def cpu_split(c0: str, c1: str, dt: float) -> dict:
    a, b = [list(map(int, x.split())) for x in (c0, c1)]
    d = [y - x for x, y in zip(a, b)]
    tot = sum(d[:8]) or 1
    share = lambda i: d[i] / tot * NCPU  # noqa: E731
    busy = (tot - d[3] - d[4]) / tot * NCPU
    return {"busy": busy, "user": share(0) + share(1), "sys": share(2), "soft": share(6) + share(5)}


class Run:
    def __init__(self, d: Path, name: str):
        self.name, self.dir = name, d
        self.windows = {}
        for r in csv.DictReader((d / f"{name}.windows.csv").open()):
            self.windows[r["window"]] = r
        st = [self.windows[w] for w in STAT_WINDOWS if w in self.windows]
        t0, t1 = float(st[0]["start"]), float(st[-1]["end"])
        self.qps = (int(st[-1]["q1"]) - int(st[0]["q0"])) / (t1 - t0)
        self.cpu = {w: cpu_split(r["cpu0"], r["cpu1"], float(r["end"]) - float(r["start"]))
                    for w, r in self.windows.items()}
        self.stats = {w: read_stat(d / f"{name}.{w}.csv") for w in STAT_WINDOWS}
        self.dur = {w: float(r["end"]) - float(r["start"]) for w, r in self.windows.items()}

    def per_query(self, w: str, ev: str):
        v = self.stats[w].get(ev)
        if not v:
            return None
        return v[0] / (self.qps * self.dur[w])

    def coverage(self) -> float:
        return min((v[1] for s in self.stats.values() for v in s.values() if isinstance(v, tuple)), default=100.0)

    def busy(self) -> float:
        xs = [self.cpu[w]["busy"] for w in STAT_WINDOWS if w in self.cpu]
        return sum(xs) / len(xs)

    def cpu_us(self) -> float:
        return self.busy() / self.qps * 1e6

    def buckets(self):
        p = self.dir / f"{self.name}.folded.gz"
        engine, other, idle = Counter(), Counter(), 0
        unbucketed = Counter()
        if not p.exists():
            return None
        is_pg = self.name.startswith("pg-")
        with gzip.open(p, "rt") as fh:
            for line in fh:
                stack, _, n = line.rstrip("\n").rpartition(" ")
                n = int(n)
                frames = stack.split(";")
                comm, frames = frames[0], frames[1:]
                if comm in IDLE_COMMS:
                    idle += n
                elif is_pg and comm == "postgres" or not is_pg and comm.startswith("ginger"):
                    b = pg_bucket(frames) if is_pg else ginger_bucket(frames)
                    engine[b] += n
                    if b.endswith("other"):
                        unbucketed[leafname(frames[-1]) if frames else "?"] += n
                else:
                    other[comm] += n
        return engine, other, idle, unbucketed


def fmt(x, nd=0, unit=""):
    if x is None:
        return "–"
    return f"{x:,.{nd}f}{unit}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(head))) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    root = Path(a.dir)
    runs = []
    for wf in sorted(root.rglob("*.windows.csv")):
        runs.append(Run(wf.parent, wf.name[: -len(".windows.csv")]))
    order = lambda r: (r.name.split("-")[0], r.name.split("-")[1], int(re.search(r"(?:ef|b)(\d+)", r.name).group(1)),  # noqa: E731
                       int(re.search(r"-[ct](\d+)$", r.name).group(1)))
    runs.sort(key=order)

    print("## Counters per query\n")
    rows = []
    for r in runs:
        g = lambda w, e: r.per_query(w, e)  # noqa: E731
        cyc, ins = g("stat1", "cycles"), g("stat1", "instructions")
        rd = r.stats["stat1"].get("uncore_imc/cas_count_read/")
        wr = r.stats["stat1"].get("uncore_imc/cas_count_write/")
        dram = (rd[0] + wr[0]) * 1048576 if rd and wr else None
        upi = r.stats["stat1"].get("unc_upi_txl_flits.all_data")
        l3 = g("stat2", "mem_load_retired.l3_miss")
        loc, rem, hitm = (g("stat3", f"mem_load_l3_miss_retired.{e}") for e in ("local_dram", "remote_dram", "remote_hitm"))
        c4 = r.stats["stat4"].get("cycles")
        sm = r.stats["stat4"].get("cycle_activity.stalls_mem_any")
        s3 = r.stats["stat4"].get("cycle_activity.stalls_l3_miss")
        wk = r.stats["stat4"].get("dtlb_load_misses.walk_active")
        t5 = r.stats["stat5"]
        td = "/".join(fmt(t5.get(f"tma_{k}")) for k in ("retiring", "bad_speculation", "frontend_bound", "backend_bound"))
        rows.append([
            r.name, fmt(r.qps), fmt(r.busy(), 1), fmt(r.cpu_us()),
            fmt(cyc / 1e6 if cyc else None, 2), fmt(ins / 1e6 if ins else None, 2),
            fmt(ins / cyc if cyc and ins else None, 2), td,
            fmt(dram / (r.qps * r.dur["stat1"]) / 1024 if dram else None, 0),
            fmt(dram / r.dur["stat1"] / 1e9 if dram else None, 1),
            fmt(upi[0] * 8 / (r.qps * r.dur["stat1"]) / 1024 if upi else None, 0),
            fmt(l3), fmt(rem / (loc + rem) * 100 if loc is not None and rem is not None and loc + rem else None, 0, "%"),
            fmt(hitm, 1),
            fmt(sm[0] / c4[0] * 100 if sm and c4 else None, 0, "%"), fmt(s3[0] / c4[0] * 100 if s3 and c4 else None, 0, "%"),
            fmt(wk[0] / c4[0] * 100 if wk and c4 else None, 1, "%"),
            fmt(g("stat1", "context-switches"), 2), fmt(r.coverage(), 0, "%"),
        ])
    print(table(["run", "QPS", "busy CPUs", "CPU µs/q", "Mcycles/q", "Minstr/q", "IPC",
                 "retire/badspec/FE/BE %", "DRAM KB/q", "DRAM GB/s", "UPI KB/q", "L3-miss loads/q",
                 "remote", "HITM loads/q", "stalled on memory", "stalled on L3 miss", "page walk",
                 "ctx switches/q", "min coverage"], rows))
    print("\nBusy CPUs and CPU µs per query come from /proc/stat over the counter windows (everything on "
          "the node, including pgbench for the batch runs). Counters are system wide. UPI bytes assume "
          "8 bytes per data flit.\n")

    print("## CPU µs per query by cost centre (from call stacks)\n")
    print("Each run's sampled CPU shares, scaled to its CPU µs per query from the counter windows. "
          "vectorfast.so has no frame pointers, so index code goes by leaf function.\n")
    allb, data = set(), {}
    for r in runs:
        res = r.buckets()
        if res is None:
            continue
        engine, other, idle, unb = res
        busy_samples = sum(engine.values()) + sum(other.values())
        if not busy_samples:
            continue
        qps_rec = (int(r.windows["record"]["q1"]) - int(r.windows["record"]["q0"])) / r.dur["record"] \
            if "record" in r.windows else r.qps
        # Shares of the sampled window, scaled to the CPU per query of the
        # counter windows (the sampler slows the engine by a few %, and the
        # sampled window's own QPS is noisy for Postgres: stats flush).
        us_per_sample = r.cpu_us() / busy_samples
        d = {b: n * us_per_sample for b, n in engine.items()}
        d["(load generator, profiler, other processes)"] = sum(other.values()) * us_per_sample
        data[r.name] = (d, unb, sum(engine.values()), qps_rec)
        allb |= set(d)
    order_b = sorted(allb, key=lambda b: (b.startswith("("), b.startswith("kernel"), b))
    names = list(data)
    rows = []
    for b in order_b:
        rows.append([b] + [fmt(data[n][0].get(b), 0) for n in names])
    rows.append(["**engine total**"] + [fmt(sum(v for k, v in data[n][0].items() if not k.startswith("(")), 0)
                                        for n in names])
    print(table(["cost centre"] + names, rows))
    print()

    print("## Work per query\n")
    for p in sorted(root.rglob("pgcounts-ef*.csv")):
        n = Counter()
        for row in csv.reader(l for l in p.open() if l.strip() and not l.startswith("#")):
            if len(row) > 2 and "probe_vectorfast" in row[2]:
                try:
                    n[row[2].split(":")[-1]] = float(row[0])
                except ValueError:
                    pass
        q = n.get("hnswrescan") or 1
        print(f"- Postgres {p.stem}: " + ", ".join(f"{k} {v / q:,.1f}" for k, v in n.items() if k != "hnswrescan"))
    for p in sorted(root.rglob("ginger-stats.jsonl")):
        for line in p.open():
            r = json.loads(line)
            print(f"- Ginger beam {r.get('search_value')}: recall {r.get('recall')}, stats {r.get('stats')}")
    print()

    print("## Most contended cache lines (perf c2c)\n")
    for p in sorted(root.rglob("*.c2c.txt.gz")):
        txt = gzip.open(p, "rt", errors="replace").read()
        tot = {k: v for k, v in re.findall(r"^\s+(Load (?:Local|Remote) HITM|Load Operations|Total records)\s+:\s+(\d+)", txt, re.M)}
        print(f"### {p.name[:-len('.c2c.txt.gz')]}\n")
        print("  " + ", ".join(f"{k}: {v}" for k, v in tot.items()))
        m = re.search(r"Shared Cache Line Distribution Pareto.*?\n(.*?)(?:\n=====|\Z)", txt, re.S)
        if m:
            lines = [l for l in m.group(1).splitlines() if l.strip() and not l.lstrip().startswith("#")]
            print("\n```")
            print("\n".join(l[:220] for l in lines[:40]))
            print("```\n")

    print("## Leaf functions outside a cost centre\n")
    for n, (d, unb, tot, _) in data.items():
        if tot:
            top = ", ".join(f"{k} {v / tot:.1%}" for k, v in unb.most_common(a.top))
            print(f"- {n}: {top}")


if __name__ == "__main__":
    main()
