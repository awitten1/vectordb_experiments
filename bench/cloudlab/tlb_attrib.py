#!/usr/bin/env python3
"""Where the server's dTLB misses land, from tlb_window.sh output.

  python3 bench/cloudlab/tlb_attrib.py [--label PREFIX] bench/profiles/hn-tlb <results.jsonl>...

For each <name> (hn-<arm>-c<clients>-ef<ef>) in the directory:

* Counters, per query (system-wide count / (QPS x window)): load page walks
  started and completed, split by the page size of the translation (4 KB or
  2 MB), STLB hits, PDE-cache misses, walk cycles, where the walker's
  page-table reads were served (L1/L2/L3/DRAM), the same for stores, and
  retired loads that missed the STLB (the PEBS event below; walks it does not
  account for came from speculative loads or prefetches).
* PEBS samples of retired loads and stores that missed the STLB: each data
  address is placed in the mapping of /proc/<pid>/maps that holds it (main
  shared memory on 2 MB pages, dynamic shared memory, the backend heap,
  anonymous mmaps, ...), then broken down by function. For main shared memory
  it also reports how concentrated the samples are on 2 MB pages: hot pages
  mean metadata, a flat spread over tens of thousands of pages means the
  buffer pool itself.

QPS comes from the client's result rows (label <PREFIX>-<arm>, matched on
clients and ef_search).
"""

import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

KERNEL_START = 0xFFFF800000000000


def load_qps(prefix: str, paths: list[str]) -> dict:
    acc = defaultdict(list)
    for path in paths:
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            lab = r.get("label") or ""
            if lab.startswith(prefix + "-"):
                acc[(lab[len(prefix) + 1:], r["clients"], r["search_value"])].append(r["qps"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def read_params(d: Path, name: str) -> tuple[int, float]:
    period, window = 2003, 10.0
    p = d / f"{name}.params"
    if p.exists():
        kv = dict(x.split("=") for x in p.read_text().split())
        period, window = int(kv["period"]), float(kv["window"])
    return period, window


def read_stats(d: Path, name: str) -> dict:
    vals = {}
    for f in sorted(d.glob(f"{name}.tlbstat*.csv")):
        for line in f.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                v = float(parts[0])
            except ValueError:
                continue  # <not counted> / <not supported>
            ev = parts[2]
            # A raw event (cpu/event=..,umask=..,name=x/) has commas of its own.
            m = re.search(r"name=([^,/]+)", line) if ev.startswith("cpu/") else None
            ev = (m.group(1) if m else ev).split(":")[0]
            if ev in ("cycles", "instructions"):
                ev = f"{ev}#{f.stem[-1]}"
            vals[ev] = v
    return vals


class Maps:
    """Mappings of every postgres process, end snapshot preferred."""

    def __init__(self, d: Path):
        self.by_pid = {}
        for f in sorted(d.glob("*.start")) + sorted(d.glob("*.end")):  # end overrides start
            regs = []
            for line in f.read_text().splitlines():
                parts = line.split(None, 5)
                if len(parts) < 5:
                    continue
                lo, hi = (int(x, 16) for x in parts[0].split("-"))
                regs.append((lo, hi, parts[5].strip() if len(parts) > 5 else "", int(parts[2], 16)))
            self.by_pid[int(f.name.split(".")[0])] = regs
        # Shared mappings are inherited from the postmaster, so any process
        # will do for a pid that exited before its maps were read.
        self.any_regs = max(self.by_pid.values(), key=len) if self.by_pid else []

    def find(self, pid: int, addr: int):
        for lo, hi, path, _ in self.by_pid.get(pid, self.any_regs):
            if lo <= addr < hi:
                return lo, hi, path
        if pid in self.by_pid:
            for lo, hi, path, _ in self.any_regs:
                if lo <= addr < hi and ("anon_hugepage" in path or path.startswith(("/dev/shm", "/SYSV"))):
                    return lo, hi, path
        return None

    def code_offset(self, pid: int, ip: int):
        """(library basename, offset in the file) for a code address, for objdump."""
        for lo, hi, path, off in self.by_pid.get(pid, self.any_regs):
            if lo <= ip < hi and path.startswith("/"):
                return path.rsplit("/", 1)[-1], ip - lo + off
        return None


def region_kind(path: str) -> str:
    if "anon_hugepage" in path:
        return "main shared memory (2 MB pages)"
    if path.startswith("/dev/shm/PostgreSQL"):
        return "dynamic shared memory"
    if path.startswith("/SYSV"):
        return "SysV shared memory"
    if path == "[heap]":
        return "backend heap (brk)"
    if path.startswith("[stack"):
        return "stack"
    if path == "":
        return "anonymous mmap (large allocations)"
    if path.startswith("["):
        return path
    return "file: " + path.rsplit("/", 1)[-1]


LINE = re.compile(r"^\s*(\S+)\s+(\d+)\s+(\S+?):\s+([0-9a-f]+)\s+([0-9a-f]+)\s+(.*?)\s*(?:\((.*)\))?\s*$")


def samples(path: Path):
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            m = LINE.match(line)
            if not m:
                continue
            yield int(m.group(2)), m.group(3), int(m.group(4), 16), int(m.group(5), 16), m.group(6) or "[unknown]"


def smaps_summary(path: Path) -> dict:
    """Per region kind of the busiest backend: (size kB, kernel page kB, kB mapped by 2 MB THP)."""
    out = defaultdict(lambda: [0, set(), 0])
    if not path.exists():
        return out
    kind = None
    for line in path.read_text().splitlines():
        if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
            parts = line.split(None, 5)
            kind = region_kind(parts[5].strip() if len(parts) > 5 else "")
        elif kind and line.startswith("Size:"):
            out[kind][0] += int(line.split()[1])
        elif kind and line.startswith("KernelPageSize:"):
            out[kind][1].add(int(line.split()[1]))
        elif kind and line.split(":")[0] in ("AnonHugePages", "ShmemPmdMapped", "FilePmdMapped"):
            out[kind][2] += int(line.split()[1])
    return out


def fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.1f}" if abs(x) >= 10 else f"{x:.2f}"


def report(d: Path, name: str, qps_map: dict) -> None:
    m = re.match(r"hn-(\w+)-c(\d+)-ef(\d+)", name)
    arm, c, ef = m.group(1), int(m.group(2)), int(m.group(3))
    period, window = read_params(d, name)
    qps = qps_map.get((arm, c, ef))
    print(f"## {name}: {c} client(s), ef_search {ef}, QPS {fmt(qps) if qps else '?'}\n")
    st = read_stats(d, name)
    per_q = (lambda v: v / (qps * window)) if qps else (lambda v: float("nan"))

    if st:
        g = st.get
        started = g("dtlb_load_misses.miss_causes_a_walk", 0)
        rows = [
            ("load walks started", g("dtlb_load_misses.miss_causes_a_walk")),
            ("load walks completed, 4 KB translation", g("dtlb_load_misses.walk_completed_4k")),
            ("load walks completed, 2 MB translation", g("dtlb_load_misses.walk_completed_2m_4m")),
            ("load STLB hits, 4 KB", g("dtlb_load_misses.stlb_hit_4k")),
            ("load STLB hits, 2 MB", g("dtlb_load_misses.stlb_hit_2m")),
            ("PDE-cache misses", g("dtlb_load_misses.pde_cache_miss")),
            ("retired loads that missed the STLB", g("mem_uops_retired.stlb_miss_loads")),
            ("load walk cycles", g("dtlb_load_misses.walk_duration")),
            ("walker reads served by L1", g("page_walker_loads.dtlb_l1")),
            ("walker reads served by L2", g("page_walker_loads.dtlb_l2")),
            ("walker reads served by L3", g("page_walker_loads.dtlb_l3")),
            ("walker reads served by DRAM", g("page_walker_loads.dtlb_memory")),
            ("store walks started", g("dtlb_store_misses.miss_causes_a_walk")),
            ("store walks completed, 4 KB", g("dtlb_store_misses.walk_completed_4k")),
            ("store walks completed, 2 MB", g("dtlb_store_misses.walk_completed_2m_4m")),
            ("store walk cycles", g("dtlb_store_misses.walk_duration")),
        ]
        print("| counter | per query |\n|---|---:|")
        for label, v in rows:
            print(f"| {label} | {fmt(per_q(v)) if v is not None else 'n/a'} |")
        if started:
            dur = g("dtlb_load_misses.walk_duration", 0)
            cyc = g("cycles#1", 0)
            wl = [g(f"page_walker_loads.dtlb_{k}", 0) for k in ("l1", "l2", "l3", "memory")]
            print(f"\n- cycles per started load walk: {fmt(dur / started)}; "
                  f"load walk cycles are {100 * dur / cyc:.1f}% of all cycles" if cyc else "")
            if sum(wl):
                print("- walker reads per started walk: " + ", ".join(
                    f"{k} {v / started:.2f}" for k, v in zip(("L1", "L2", "L3", "DRAM"), wl)))
            ret = g("mem_uops_retired.stlb_miss_loads")
            if ret:
                print(f"- retired loads that missed the STLB are {100 * ret / started:.0f}% of started walks "
                      "(the rest: speculative loads, hardware or software prefetches)")
        print()

    txt = d / f"{name}.stlb.txt.gz"
    if not txt.exists():
        print("(no PEBS samples)\n")
        return
    maps = Maps(d / f"{name}.maps")
    by_ev = defaultdict(Counter)            # event -> region kind -> n
    sym_by = defaultdict(Counter)           # (event, kind) -> sym -> n
    hp_pages = Counter()                    # 2 MB page index within main shared memory -> n
    hp_page_syms = defaultdict(Counter)
    hp_gb = Counter()
    zero = 0
    ips = defaultdict(Counter)              # event -> (lib, offset, sym) -> n
    addrs = defaultdict(set)
    for pid, ev, addr, ip, sym in samples(txt):
        co = maps.code_offset(pid, ip)
        ips[ev][(co[0], co[1], sym) if co else ("?", ip, sym)] += 1
        if len(addrs[ev]) <= 1000:
            addrs[ev].add(addr)
        if addr == 0:
            zero += 1
            continue
        if addr >= KERNEL_START:
            kind, reg = "kernel address", None
        else:
            reg = maps.find(pid, addr)
            kind = region_kind(reg[2]) if reg else "not mapped in snapshot"
        by_ev[ev][kind] += 1
        sym_by[(ev, kind)][sym] += 1
        if ev == "stlb_miss_loads" and reg and "anon_hugepage" in reg[2]:
            off = addr - reg[0]
            hp_pages[off >> 21] += 1
            hp_page_syms[off >> 21][sym] += 1
            hp_gb[off >> 30] += 1

    for ev in sorted(by_ev):
        tot = sum(by_ev[ev].values())
        print(f"**{ev}**: {tot:,} samples, one per {period:,} events"
              f" (about {fmt(per_q(tot * period))} per query)\n")
        print("| instruction (library + offset, for objdump) | function | share |\n|---|---|---:|")
        for (lib, off, s), n in ips[ev].most_common(10):
            print(f"| {lib} +0x{off:x} | {s} | {100 * n / tot:.1f}% |")
        print()
        if tot > 1000 and len(addrs[ev]) < 100:
            print(f"Data addresses are not usable for this event here ({len(addrs[ev])} distinct values "
                  f"over {tot:,} samples), so the region breakdown is skipped; use the instructions above.\n")
            continue
        print("| region | share | per query | top functions |\n|---|---:|---:|---|")
        for kind, n in by_ev[ev].most_common():
            tops = ", ".join(f"{s} {100 * k / n:.0f}%" for s, k in sym_by[(ev, kind)].most_common(4))
            print(f"| {kind} | {100 * n / tot:.1f}% | {fmt(per_q(n * period))} | {tops} |")
        print()
        allsym = Counter()
        for (e, kind), cnt in sym_by.items():
            if e == ev:
                allsym.update(cnt)
        print("| function | share |\n|---|---:|")
        for s, n in allsym.most_common(12):
            print(f"| {s} | {100 * n / tot:.1f}% |")
        print()
    if zero:
        print(f"({zero:,} samples had no data address)\n")

    if hp_pages:
        tot = sum(hp_pages.values())
        hot = hp_pages.most_common()
        top64 = sum(n for _, n in hot[:64])
        print(f"**Main shared memory, load samples:** {tot:,} over {len(hp_pages):,} distinct 2 MB pages; "
              f"the hottest 64 pages hold {100 * top64 / tot:.1f}%, the hottest 1,024 "
              f"{100 * sum(n for _, n in hot[:1024]) / tot:.1f}%.\n")
        print("| hottest 2 MB pages (offset) | share | top functions |\n|---|---:|---|")
        for pg, n in hot[:8]:
            tops = ", ".join(f"{s} {100 * k / n:.0f}%" for s, k in hp_page_syms[pg].most_common(3))
            print(f"| {pg * 2:,} MB | {100 * n / tot:.2f}% | {tops} |")
        lo_gb, hi_gb = min(hp_gb), max(hp_gb)
        print(f"\nSpread by GB of offset: samples from {lo_gb} to {hi_gb} GB; "
              f"GB bins with any samples: {len(hp_gb)}; largest bin {100 * max(hp_gb.values()) / tot:.1f}%.\n")

    mi = d / f"{name}.meminfo.txt"
    if mi.exists():
        kv = {l.split(":")[0]: l.split(":")[1].strip() for l in mi.read_text().splitlines() if ":" in l}
        print(f"meminfo: ShmemHugePages {kv.get('ShmemHugePages', '?')}, ShmemPmdMapped {kv.get('ShmemPmdMapped', '?')}, "
              f"Shmem {kv.get('Shmem', '?')}\n")
    sm = smaps_summary(d / f"{name}.smaps.txt")
    # Calico's translation extents, one row instead of one per 512 MB file.
    cal = [v for k, v in sm.items() if k.startswith("file: calico.")]
    if cal:
        for k in [k for k in sm if k.startswith("file: calico.")]:
            del sm[k]
        sm[f"Calico translation extents ({len(cal)} files)"] = [sum(v[0] for v in cal), set().union(*(v[1] for v in cal)),
                                                               sum(v[2] for v in cal)]
    if sm:
        print("Busiest backend's mappings (smaps):\n\n| region | size | kernel page size | mapped with 2 MB THP |\n|---|---:|---|---:|")
        for kind, (sz, kps, thp) in sorted(sm.items(), key=lambda kv: -kv[1][0]):
            if sz >= 1024:
                print(f"| {kind} | {sz / 1024:,.0f} MB | {'/'.join(f'{k} kB' for k in sorted(kps))} | {thp / 1024:,.0f} MB |")
        print()


def main() -> None:
    argv = sys.argv[1:]
    prefix = "tlbhp"
    if argv and argv[0] == "--label":
        prefix, argv = argv[1], argv[2:]
    d = Path(argv[0])
    qps_map = load_qps(prefix, argv[1:])
    names = sorted({p.name.split(".")[0] for p in d.glob("hn-*.params")},
                   key=lambda n: int(re.search(r"-c(\d+)-", n).group(1)))
    for name in names:
        report(d, name, qps_map)


if __name__ == "__main__":
    main()
