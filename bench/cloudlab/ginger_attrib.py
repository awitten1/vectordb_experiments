#!/usr/bin/env python3
"""What Ginger's queries are made of, from ginger_window.sh and ginger_bench --stats.

  python3 bench/cloudlab/ginger_attrib.py bench/profiles/hn-ginger [--bin ginger_bench] [--objdump llvm-objdump]

For each ginger-t<threads>-b<beam> profile in the directory:

* Work per query from ginger-stats.jsonl (nodes expanded, candidates,
  distance computations, re-ranked points), then counters per query and per
  distance computation (cycles, cache misses, lines leaving the core, page
  walks), with QPS from the progress file over the windows.
* Precise cycle samples by part of the search: bucketed by the innermost
  source file of each sampled instruction (objdump -l on the profiled binary,
  built with -g1), since Ginger's templates inline into a few big functions.
* PEBS load-latency samples by data source and by instruction.
* Huge-page use of the process (smaps_rollup).
"""

import argparse
import gzip
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tlb_attrib import fmt  # noqa: E402
from trav_attrib import LVL, level_kind, read_counters, read_kv, table  # noqa: E402

# Part of the search, from the containing function (objdump's, else perf's
# symbol). The beam-search loop is one big inlined function
# (DispatchBeamSearch / std::__do_visit), split by innermost source file.
FUNC_BUCKETS = [
    (("TurboQuantScoringInt8BatchDims", "ComputeDistances4Bit"), "4-bit distance kernel"),
    (("ExtraPrecisionScoring",), "re-rank (12-bit codes)"),
    (("DecodeAdaptive", "CompressedCSRGraph", "LocalBuffer"), "graph: neighbour-list decode"),
    (("MergeBeamAndCandidates", "__introsort", "__sort<", "__insertion_sort", "nth_element",
      "__unguarded", "SearchToken"), "beam: sort and merge candidates"),
    (("WalshHadamard", "EncodeQuery", "PrepareQuery", "TurboQuantQuery::Create", "Rotat"),
     "query setup (rotation, encoding)"),
]
LOOP_FILES = [
    (("fast_visited", "visited_set"), "visited set"),
    (("compressed", "varint", "adaptive_gap"), "graph: neighbour-list decode"),
    (("prefetch",), "beam-search loop (prefetch, candidate list)"),
]


def bucket(func: str, src: str, sym: str, dso: str) -> str:
    name = func if func and func != "?" else sym
    for keys, b in FUNC_BUCKETS:
        if any(k in name for k in keys):
            return b
    if "DispatchBeamSearch" in name or "__do_visit" in name or "NavigationBeamSearch" in name or \
            "std::vector<ginger::DistanceAndId" in name:
        for keys, b in LOOP_FILES:
            if any(k in src for k in keys):
                return b
        return "beam-search loop (prefetch, candidate list)"
    if dso.startswith("[kernel"):
        return "kernel"
    if "jemalloc" in dso or name.startswith(("je_", "malloc", "free", "operator new", "operator delete")):
        return "allocator"
    if "libc" in dso or "libstdc++" in dso or "libm" in dso:
        return "libc, libstdc++, libm"
    if "ginger_bench" in dso:
        return "other ginger_bench"
    return "other"


def objdump_lines(binary: Path, objdump: str, cache: Path) -> dict:
    """address -> (function, file:line, instruction), cached next to the profiles."""
    if not cache.exists():
        with open(cache, "w") as fh:
            subprocess.run([objdump, "-d", "-l", "-C", "--no-show-raw-insn", str(binary)],
                           stdout=fh, stderr=subprocess.DEVNULL, check=True)
    tab, func, src = {}, "?", "?"
    with open(cache, errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
            if m:
                func = m.group(1)
                continue
            if line.startswith(("; /", "/")) and re.search(r":\d+( \(discriminator \d+\))?$", line):
                path = re.sub(r" \(discriminator \d+\)$", "", line.lstrip("; "))
                p, _, ln = path.rpartition(":")
                src = f"{p.rsplit('/', 1)[-1]}:{ln}"
                continue
            m = re.match(r"^\s+([0-9a-f]+):\s+(.*)$", line)
            if m:
                tab[int(m.group(1), 16)] = (func, src, re.sub(r"\s+", " ", m.group(2)).strip())
    return tab


def text_mappings(smaps: Path):
    """[(lo, hi, file offset)] of ginger_bench's own executable mappings."""
    regs = []
    if smaps.exists():
        for line in smaps.read_text(errors="replace").splitlines():
            m = re.match(r"^([0-9a-f]+)-([0-9a-f]+) (\S+) ([0-9a-f]+) \S+ \d+\s+(\S.*)$", line)
            if m and m.group(5).endswith("/ginger_bench"):
                regs.append((int(m.group(1), 16), int(m.group(2), 16), int(m.group(4), 16)))
    return regs


def file_addr(regs, ip: int):
    for lo, hi, off in regs:
        if lo <= ip < hi:
            return ip - lo + off
    return None


# perf script lines. Ginger's demangled C++ names contain spaces and
# parentheses, so the symbol is everything between the ip and the final
# "(dso)"; in load-latency lines the ip follows the weight after "|BLK ...".
CYC_LINE = re.compile(r"^\s*\S+\s+(\d+)\s+([0-9a-f]+)\s+(.*)\s+\(([^()]*)\)\s*$")
LAT_LINE = re.compile(r"\|BLK\s+\S+\s+(\d+)\s+([0-9a-f]+)\s+(.*)\s+\(([^()]*)\)\s*$")


def parse_samples(path: Path, lat: bool):
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if lat:
                m = LAT_LINE.search(line)
                if not m:
                    continue
                lv = LVL.search(line)
                yield int(m.group(2), 16), m.group(3), m.group(4), int(m.group(1)), (lv.group(1).strip() if lv else None)
            else:
                m = CYC_LINE.match(line)
                if m:
                    yield int(m.group(2), 16), m.group(3), m.group(4), None, None


def read_tlb(p: Path) -> dict:
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split(",")
            try:
                v = float(parts[0])
            except (ValueError, IndexError):
                continue
            ev = parts[2]
            m = re.search(r"name=([^,/]+)", line) if ev.startswith("cpu/") else None
            out[(m.group(1) if m else ev).split(":")[0]] = v
    return out


def load_stats(d: Path) -> dict:
    """beam -> stats dict (and recall) from ginger-stats.jsonl."""
    out = {}
    p = d / "ginger-stats.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if "stats" in r:
                out[r["search_value"]] = dict(r["stats"], recall=r["recall"])
    return out


def report(d: Path, name: str, lines: dict, stats: dict) -> None:
    m = re.match(r"ginger-t(\d+)-b(\d+)", name)
    t, beam = int(m.group(1)), int(m.group(2))
    params = read_kv(d / f"{name}.params")
    win = float(params.get("window", 10))
    qps = float(read_kv(d / f"{name}.qps.txt").get("qps", "nan"))
    nq = qps * win
    st = stats.get(beam, {})
    dist = st.get("distance_computations", float("nan"))
    print(f"## {name}: {t} thread(s), beam_size {beam}, QPS {qps:,.0f} over the windows\n")
    if st:
        table([("recall@10", f"{st['recall']:.4f}"), ("nodes expanded", fmt(st["nodes_visited"])),
               ("candidates considered", fmt(st["candidates_considered"])),
               ("4-bit distance computations", fmt(dist)), ("re-ranked points", fmt(st["reranked"]))],
              ["per query (ginger_bench --stats)", "value"])

    rp = d / f"{name}.smaps_rollup.txt"
    if rp.exists():
        kv = {}
        for line in rp.read_text().splitlines():
            mm = re.match(r"^(\w+):\s+(\d+) kB", line)
            if mm:
                kv[mm.group(1)] = int(mm.group(2))
        thp = (d / f"{name}.thp.txt").read_text().split("\n")[0] if (d / f"{name}.thp.txt").exists() else "?"
        print(f"- resident {kv.get('Rss', 0) / 1048576:.1f} GiB, of which on transparent huge pages "
              f"{kv.get('AnonHugePages', 0) / 1048576:.1f} GiB; THP setting: {thp}\n")

    vals, cyc, cover = read_counters(d, name)
    tlb = read_tlb(d / f"{name}.tlbstat.csv")
    if vals:
        def per_q(ev):
            return vals[ev][0] / nq if ev in vals else float("nan")
        cycles_q = sum(cyc.values()) / len(cyc) / nq
        ins_q = sum(v for k, v in vals.items() if k.startswith("instructions#")) / len(cyc) / nq
        rows = [
            ("cycles", cycles_q), ("instructions", ins_q),
            ("retired loads that missed L1 and allocated a fill buffer", per_q("mem_load_uops_retired.l1_miss")),
            ("retired loads that missed L1 but hit a fill buffer", per_q("mem_load_uops_retired.hit_lfb")),
            ("retired loads missing L2", per_q("mem_load_uops_retired.l2_miss")),
            ("retired loads missing L3", per_q("mem_load_uops_retired.l3_miss")),
            ("— served by local DRAM", per_q("mem_load_uops_l3_miss_retired.local_dram")),
            ("— served by remote DRAM", per_q("mem_load_uops_l3_miss_retired.remote_dram")),
            ("loads that hit a software prefetch in flight", per_q("load_hit_pre.sw_pf")),
            ("L1D line fills", per_q("l1d.replacement")),
            ("demand data reads sent off core", per_q("offcore_requests.demand_data_rd")),
        ]
        if tlb:
            rows += [("load page walks started", tlb.get("dtlb_load_misses.miss_causes_a_walk", float("nan")) / nq),
                     ("· completed for a 4 KB page", tlb.get("dtlb_load_misses.walk_completed_4k", float("nan")) / nq),
                     ("· completed for a 2 MB page", tlb.get("dtlb_load_misses.walk_completed_2m_4m", float("nan")) / nq)]
        table([(k, fmt(v), fmt(v / dist) if dist == dist and dist else "") for k, v in rows],
              ["counter", "per query", "per distance computation"])
        print(f"- IPC {ins_q / cycles_q:.2f} (system-wide)")
        if tlb and "dtlb_load_misses.walk_duration" in tlb:
            tc = next((float(line.split(",")[0]) for line in (d / f"{name}.tlbstat.csv").read_text().splitlines()
                       if ",cycles," in line or line.split(",")[2:3] == ["cycles"]), None)
            if tc:
                print(f"- cycles with a load page walk in progress: {tlb['dtlb_load_misses.walk_duration'] / tc:.1%}")
        print(f"- counter coverage: min {cover:.0f}% of each window" + ("" if cover >= 99 else " (scaled estimates)"))
        print()

    regs = text_mappings(d / f"{name}.smaps.txt")
    period = int(params.get("period_cyc", 500009))
    cp = d / f"{name}.cyc.txt.gz"
    if cp.exists():
        by_b, by_sym, by_line, by_ip = Counter(), Counter(), Counter(), Counter()
        n = 0
        for ip, sym, dso, _, _ in parse_samples(cp, False):
            n += 1
            src, func, fa = "?", "?", None
            if "ginger_bench" in dso:
                fa = file_addr(regs, ip)
                if fa is not None and fa in lines:
                    func, src = lines[fa][0], lines[fa][1]
                    by_line[(func[:60], src)] += 1
                    by_ip[fa] += 1
            by_b[bucket(func, src, sym, dso)] += 1
            by_sym[(sym[:70], dso.rsplit("/", 1)[-1])] += 1
        if n:
            cyc_q = n * period / nq
            print(f"**Precise cycle samples (ginger_bench)**: {n:,}, one per {period:,} cycles, "
                  f"about {fmt(cyc_q)} cycles per query" + (f", {fmt(cyc_q / dist)} per distance computation" if dist == dist else "") + "\n")
            table([(b, f"{k / n:.1%}", f"{k * period / nq / 2.6e3:.0f}") for b, k in by_b.most_common()],
                  ["part", "share", "µs per query at 2.6 GHz"])
            table([(s, o, f"{k / n:.1%}") for (s, o), k in by_sym.most_common(15)], ["function", "object", "share"])
            if by_line:
                print("By source line (innermost inline):\n")
                table([(f, s, f"{k / n:.1%}") for (f, s), k in by_line.most_common(25)], ["function", "source line", "share"])
                print("Hottest instructions:\n")
                table([(f"{o:#x}", lines[o][1], f"`{lines[o][2][:60]}`", f"{k / n:.1%}") for o, k in by_ip.most_common(20)],
                      ["address", "source", "instruction", "share"])

    lp = d / f"{name}.lat.txt.gz"
    if lp.exists():
        plat = int(params.get("period_lat", 2003))
        rows = list(parse_samples(lp, True))
        if rows:
            print(f"**Load-latency samples (loads over 50 cycles)**: {len(rows):,}, one per {plat:,} "
                  f"(about {fmt(len(rows) * plat / nq)} such loads per query)\n")
            agg = defaultdict(lambda: [0, 0])
            for r in rows:
                a = agg[level_kind(r[4])]
                a[0] += 1
                a[1] += r[3] or 0
            tot = sum(a[1] for a in agg.values()) or 1
            table([(k, f"{a[0] / len(rows):.1%}", f"{a[1] / max(a[0], 1):.0f}", f"{a[1] / tot:.1%}")
                   for k, a in sorted(agg.items(), key=lambda x: -x[1][1])],
                  ["served by", "share of samples", "mean latency (cycles)", "share of total latency"])
            per = defaultdict(lambda: [0, 0, Counter()])
            for ip, sym, dso, w, lv in rows:
                fa = file_addr(regs, ip) if "ginger_bench" in dso else None
                key = (lines[fa][0][:50], lines[fa][1], lines[fa][2]) if fa in lines else (sym[:50], dso.rsplit("/", 1)[-1], "")
                p = per[key]
                p[0] += 1
                p[1] += w or 0
                p[2][level_kind(lv)] += 1
            tot = sum(p[1] for p in per.values()) or 1
            out = []
            for (f, s, ins), (k, w, lvl) in sorted(per.items(), key=lambda kv: -kv[1][1])[:15]:
                out.append((f, s, f"`{ins[:60]}`" if ins else "", f"{w / tot:.1%}", f"{w / k:.0f}",
                            ", ".join(f"{a} {b / k:.0%}" for a, b in lvl.most_common(3))))
            table(out, ["function", "source", "instruction", "share of latency", "mean cycles", "served by"])
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--bin", help="profiled ginger_bench (default <dir>/ginger_bench)")
    ap.add_argument("--objdump", default=shutil.which("llvm-objdump") or shutil.which("objdump") or "objdump")
    a = ap.parse_args()
    d = Path(a.dir)
    binary = Path(a.bin) if a.bin else d / "ginger_bench"
    lines = objdump_lines(binary, a.objdump, d / "ginger_bench.objdump.txt") if binary.exists() else {}
    stats = load_stats(d)
    if stats:
        print("## Ginger's work per query by beam_size (one pass, 1 thread)\n")
        table([(b, f"{s['recall']:.4f}", fmt(s["nodes_visited"]), fmt(s["candidates_considered"]),
                fmt(s["distance_computations"]), fmt(s["reranked"])) for b, s in sorted(stats.items())],
              ["beam_size", "recall@10", "nodes expanded", "candidates", "distance computations", "re-ranked"])
    names = sorted({p.name.split(".")[0] for p in d.glob("ginger-t*.params")},
                   key=lambda s: tuple(int(x) for x in re.findall(r"\d+", s)))
    for name in names:
        report(d, name, lines, stats)


if __name__ == "__main__":
    main()
