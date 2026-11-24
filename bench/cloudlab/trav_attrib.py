#!/usr/bin/env python3
"""What pgvectorfast's HNSW traversal is made of, from trav_window.sh output.

  python3 bench/cloudlab/trav_attrib.py bench/profiles/hn-trav [--lib vectorfast.so] [--objdump llvm-objdump]

For each <name> (hn-<arm>-c<clients>-ef<ef>) in the directory:

* Counters per query (system-wide count / (QPS x window), QPS from
  xact_commit over the windows): cache misses of retired loads by level,
  local vs remote DRAM, fill-buffer hits (late prefetches), memory-level
  parallelism (L1D misses outstanding while any is), average demand-read
  latency, and stall shares.
* Precise cycle samples, by function and, inside vectorfast.so, by source
  line (objdump -l on the served library, which keeps its line table).
* PEBS load-latency samples (loads over 50 cycles): by data source, and the
  instructions and source lines with the most total latency.
* uprobe counts per query (per hnswrescan call).
"""

import argparse
import gzip
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tlb_attrib import Maps, fmt  # noqa: E402

KERNEL_START = 0xFFFF800000000000
TRAVERSAL = {"HnswLoadUnvisitedRaw", "HnswSearchLayerDiskFastCore", "HnswSearchLayerUpperDiskFast",
             "HnswLoadNeighborTidsAndHashSlow", "HnswLoadUnvisitedMissingOrPinSlow", "HnswLoadFastCandSlow",
             "HnswEvalAsyncUnvisitedBatchFast", "hnsw_fast_visited_grow", "hnsw_fast_visited_init"}


def bucket(sym: str, dso: str) -> str:
    if sym in TRAVERSAL:
        return "traversal"
    if "TqNavDot" in sym or "TqCodeDistance" in sym:
        return "tq4 navigation distance"
    if "Rerank" in sym or "TqRefine" in sym:
        return "re-rank"
    if sym.startswith(("Calico", "calico")):
        return "Calico (pin, lock, lookup)"
    if dso.startswith("[kernel") or dso == "[unknown]" and sym.startswith(("entry_", "__")):
        return "kernel"
    if "vectorfast.so" in dso:
        return "other vectorfast.so"
    if dso.endswith("postgres"):
        return "postgres executor, protocol, other"
    if "libc" in dso:
        return "libc"
    return "other"


def read_kv(p: Path) -> dict:
    return dict(x.split("=") for x in p.read_text().split()) if p.exists() else {}


def read_counters(d: Path, name: str):
    """{event: value}, {group: cycles}, min coverage %."""
    vals, cyc, cover = {}, {}, 100.0
    for f in sorted(d.glob(f"{name}.travstat*.csv")):
        g = f.stem[-1]
        for line in f.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            try:
                v = float(parts[0])
            except (ValueError, IndexError):
                continue
            ev = parts[2]
            m = re.search(r"name=([^,/]+)", line) if ev.startswith("cpu/") else None
            ev = (m.group(1) if m else ev).split(":")[0]
            pct = re.findall(r",(\d+\.\d+),", line)
            if pct:
                cover = min(cover, float(pct[0]))
            if ev in ("cycles", "instructions"):
                if ev == "cycles":
                    cyc[g] = v
                vals[f"{ev}#{g}"] = v
            else:
                vals[ev] = (v, g)
    return vals, cyc, cover


def objdump_lines(lib: Path, objdump: str) -> dict:
    """offset -> (function, file:line, instruction) for the whole library."""
    out = subprocess.run([objdump, "-d", "-l", "--no-show-raw-insn", str(lib)],
                         capture_output=True, text=True, check=True).stdout
    table, func, src = {}, "?", "?"
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if m:
            func = m.group(1)
            continue
        if line.startswith("; /"):
            path, _, ln = line[2:].rpartition(":")
            src = f"{path.rsplit('/', 1)[-1]}:{ln}"
            continue
        m = re.match(r"^\s+([0-9a-f]+):\s+(.*)$", line)
        if m:
            table[int(m.group(1), 16)] = (func, src, re.sub(r"\s+", " ", m.group(2)).strip())
    return table


# ... <ip> <sym> (<dso>) at the end of a perf script line; ip has 8+ hex digits
# (user or kernel), so it cannot be confused with the pid.
TAIL = re.compile(r"\s([0-9a-f]{8,16})\s+(\S+)\s+\(([^()]*)\)\s*$")


def parse_cyc(path: Path):
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3 or not parts[1].isdigit():
                continue
            m = TAIL.search(line)
            if m:
                yield int(parts[1]), int(m.group(1), 16), m.group(2), m.group(3)


LVL = re.compile(r"\|LVL ([^|]+)")


def parse_lat(path: Path):
    """(pid, ip, sym, dso, addr, weight, level) per sample; weight/level None if absent."""
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4 or not parts[1].isdigit():
                continue
            m = TAIL.search(line)
            if not m:
                continue
            # comm pid addr data_src(hex) |OP ..|LVL ..|..|BLK  N/A  weight  ip sym (dso)
            head = line[:m.start()]
            lv = LVL.search(line)
            level = lv.group(1).strip() if lv else None
            toks = head.split()
            addr = int(toks[2], 16) if len(toks) > 2 and re.fullmatch(r"[0-9a-f]+", toks[2]) else 0
            weight = int(toks[-1]) if lv and toks[-1].isdigit() else None
            yield int(parts[1]), int(m.group(1), 16), m.group(2), m.group(3), addr, weight, level


def level_kind(level: str | None) -> str:
    if level is None:
        return "?"
    s = level.lower()
    if "remote ram" in s:
        return "remote DRAM"
    if "local ram" in s:
        return "local DRAM"
    if "remote cache" in s:
        return "remote cache"
    if "lfb" in s:
        return "fill buffer (line already in flight)"
    if "l3" in s:
        return "L3"
    if "l2" in s:
        return "L2"
    if "l1" in s:
        return "L1"
    if "uncached" in s or "io" in s:
        return "uncached/IO"
    return level


def table(rows, head):
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join("---" if i == 0 else "---:" for i in range(len(head))) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


def report(d: Path, name: str, lines: dict) -> None:
    m = re.match(r"hn-(\w+)-c(\d+)-ef(\d+)", name)
    c, ef = int(m.group(2)), int(m.group(3))
    params = read_kv(d / f"{name}.params")
    win = float(params.get("window", 10))
    qps = float(read_kv(d / f"{name}.qps.txt").get("qps", "nan"))
    nq = qps * win
    print(f"## {name}: {c} client(s), ef_search {ef}, QPS {qps:,.0f} (xact_commit over the windows)\n")
    maps = Maps(d / f"{name}.maps")

    # Counters
    vals, cyc, cover = read_counters(d, name)
    if vals:
        def per_q(ev):
            return vals[ev][0] / nq if ev in vals else float("nan")

        def share(ev):
            return vals[ev][0] / cyc[vals[ev][1]] if ev in vals else float("nan")

        def ratio(a, b):
            return vals[a][0] / vals[b][0] if a in vals and b in vals and vals[b][0] else float("nan")
        cycles_q = sum(cyc.values()) / len(cyc) / nq
        ins_q = sum(v for k, v in vals.items() if k.startswith("instructions#")) / len(cyc) / nq
        rows = [
            ("cycles", fmt(cycles_q)), ("instructions", fmt(ins_q)),
            ("retired loads that missed L1 and allocated a fill buffer", fmt(per_q("mem_load_uops_retired.l1_miss"))),
            ("retired loads that missed L1 but hit a fill buffer (line already in flight)", fmt(per_q("mem_load_uops_retired.hit_lfb"))),
            ("retired loads missing L2", fmt(per_q("mem_load_uops_retired.l2_miss"))),
            ("retired loads missing L3", fmt(per_q("mem_load_uops_retired.l3_miss"))),
            ("— served by local DRAM", fmt(per_q("mem_load_uops_l3_miss_retired.local_dram"))),
            ("— served by remote DRAM", fmt(per_q("mem_load_uops_l3_miss_retired.remote_dram"))),
            ("retired loads served by L3 (no snoop)", fmt(per_q("mem_load_uops_l3_hit_retired.xsnp_none"))),
            ("loads that hit a software prefetch in flight", fmt(per_q("load_hit_pre.sw_pf"))),
            ("L1D line fills", fmt(per_q("l1d.replacement"))),
            ("demand data reads sent off core", fmt(per_q("offcore_requests.demand_data_rd"))),
        ]
        table(rows, ["counter", "per query"])
        mlp = ratio("l1d_pend_miss.pending", "l1d_pend_miss.pending_cycles")
        lat = ratio("offcore_requests_outstanding.demand_data_rd", "offcore_requests.demand_data_rd")
        print(f"- IPC {ins_q / cycles_q:.2f} (system-wide, includes other processes)")
        print(f"- not used, unreliable here: L1D misses outstanding while any is ({mlp:.2f}) and average "
              f"demand-read latency ({lat:.0f} cycles) use counters Haswell miscounts with hyperthreading "
              f"on (erratum HSD78); the cycle_activity stall shares count halted cycles of idle CPUs, "
              f"so system-wide they exceed 100% at 1 client")
        loc, rem = per_q("mem_load_uops_l3_miss_retired.local_dram"), per_q("mem_load_uops_l3_miss_retired.remote_dram")
        print(f"- remote share of DRAM-served loads: {rem / (loc + rem):.0%}")
        print(f"- counter coverage: min {cover:.0f}% of each window" + ("" if cover >= 99 else " (MULTIPLEXED: scaled estimates)"))
        print()

    # Cycle samples
    period = int(params.get("period_cyc", 500009))
    cp = d / f"{name}.cyc.txt.gz"
    if cp.exists():
        by_b, by_sym, by_line, by_ip = Counter(), Counter(), Counter(), Counter()
        n = 0
        for pid, ip, sym, dso in parse_cyc(cp):
            n += 1
            by_b[bucket(sym, dso)] += 1
            by_sym[(sym, dso.rsplit("/", 1)[-1])] += 1
            if "vectorfast.so" in dso:
                co = maps.code_offset(pid, ip)
                if co:
                    f, src, ins = lines.get(co[1], (sym, "?", "?"))
                    by_line[(f, src)] += 1
                    by_ip[co[1]] += 1
        cyc_per_q = n * period / nq
        print(f"**Precise cycle samples (postgres)**: {n:,}, one per {period:,} cycles, "
              f"about {fmt(cyc_per_q)} cycles per query\n")
        table([(b, f"{k / n:.1%}", f"{k * period / nq / 2.6e3:.0f}") for b, k in by_b.most_common()],
              ["bucket", "share", "µs per query at 2.6 GHz"])
        table([(s, o, f"{k / n:.1%}") for (s, o), k in by_sym.most_common(15)], ["function", "object", "share"])
        trav = sum(k for (f, _), k in by_line.items() if f in TRAVERSAL)
        if trav:
            print(f"Traversal by source line (share of traversal samples, {trav:,}):\n")
            table([(f, src, f"{k / trav:.1%}") for (f, src), k in by_line.most_common() if f in TRAVERSAL][:25],
                  ["function", "source line (innermost inline)", "share"])
            print("Hottest traversal instructions:\n")
            hot = [(o, k) for o, k in by_ip.most_common() if lines.get(o, ("",))[0] in TRAVERSAL][:20]
            table([(f"+{o:#x}", lines[o][1], f"`{lines[o][2]}`", f"{k / trav:.1%}") for o, k in hot],
                  ["offset", "source", "instruction", "share of traversal"])

    # Load latency
    lp = d / f"{name}.lat.txt.gz"
    if lp.exists():
        plat = int(params.get("period_lat", 2003))
        rows = list(parse_lat(lp))
        if rows:
            have_w = any(r[5] is not None for r in rows)
            have_l = any(r[6] is not None for r in rows)
            print(f"**Load-latency samples (loads over 50 cycles, postgres)**: {len(rows):,}, one per {plat:,} "
                  f"(about {fmt(len(rows) * plat / nq)} such loads per query); weight {'yes' if have_w else 'no'}, "
                  f"data source {'yes' if have_l else 'no'}\n")
            if have_l:
                agg = defaultdict(lambda: [0, 0])
                for r in rows:
                    a = agg[level_kind(r[6])]
                    a[0] += 1
                    a[1] += r[5] or 0
                tot = sum(a[1] for a in agg.values()) or 1
                table([(k, f"{a[0] / len(rows):.1%}", f"{a[1] / max(a[0], 1):.0f}", f"{a[1] / tot:.1%}")
                       for k, a in sorted(agg.items(), key=lambda x: -x[1][1])],
                      ["served by", "share of samples", "mean latency (cycles)", "share of total latency"])
            per = defaultdict(lambda: [0, 0, Counter()])
            for pid, ip, sym, dso, addr, w, lv in rows:
                key = None
                if "vectorfast.so" in dso:
                    co = maps.code_offset(pid, ip)
                    if co:
                        f, src, ins = lines.get(co[1], (sym, "?", "?"))
                        key = (f, src, ins, f"+{co[1]:#x}")
                if key is None:
                    key = (sym, dso.rsplit("/", 1)[-1], "", "")
                p = per[key]
                p[0] += 1
                p[1] += w or 0
                p[2][level_kind(lv)] += 1
            tot = sum(p[1] for p in per.values()) or sum(p[0] for p in per.values())
            sortkey = (lambda kv: -kv[1][1]) if have_w else (lambda kv: -kv[1][0])
            out = []
            for (f, src, ins, off), (k, w, lvl) in sorted(per.items(), key=sortkey)[:20]:
                mix = ", ".join(f"{a} {b / k:.0%}" for a, b in lvl.most_common(3))
                out.append((f, src, f"`{ins}` {off}" if ins else "", f"{(w if have_w else k) / tot:.1%}",
                            f"{w / k:.0f}" if have_w else "", mix))
            table(out, ["function", "source", "instruction", "share of latency", "mean cycles", "served by"])
            daddrs = len({r[4] for r in rows})
            print(f"- distinct data addresses: {daddrs:,} over {len(rows):,} samples"
                  + (" (unusable, as for the STLB event)" if daddrs < 100 else "") + "\n")

    # Probes
    pp = d / f"{name}.probes.csv"
    if pp.exists():
        cnt = {}
        for line in pp.read_text().splitlines():
            parts = line.split(",")
            if len(parts) > 2 and parts[0].replace(".", "").isdigit():
                cnt[parts[2].split(":")[-1]] = float(parts[0])
        qn = cnt.get("hnswrescan")
        if qn:
            print(f"**Function entries per query** (uprobes, {qn:,.0f} queries in the window):\n")
            table([(k, f"{v / qn:,.1f}") for k, v in cnt.items() if k != "hnswrescan"], ["function", "calls per query"])
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--lib", help="served vectorfast.so (default <dir>/vectorfast.so)")
    ap.add_argument("--objdump", default=shutil.which("llvm-objdump") or shutil.which("objdump") or "objdump")
    a = ap.parse_args()
    d = Path(a.dir)
    lib = Path(a.lib) if a.lib else d / "vectorfast.so"
    lines = objdump_lines(lib, a.objdump) if lib.exists() else {}
    names = sorted({p.name.split(".")[0] for p in d.glob("hn-*.params")}, key=lambda s: int(re.search(r"-c(\d+)-", s).group(1)))
    for name in names:
        report(d, name, lines)


if __name__ == "__main__":
    main()
