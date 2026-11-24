#!/usr/bin/env python3
"""Summarize perf_window.sh output: one row per profiled configuration.

  python3 bench/cloudlab/perf_summary.py [--label PREFIX] bench/profiles/pgvector <results.jsonl>...

QPS comes from the client's result rows (label <PREFIX>-stock, or <PREFIX>-fast
for <tag>-fast-* windows, matched on clients and ef_search; PREFIX defaults to
prof). Per configuration it reports busy CPUs and server CPU time per query (from
call-graph samples in Postgres processes), IPC, DRAM read bandwidth from the
IMC counters, DRAM bytes per query, and the
Top-Down level-1/2 split. The windows are 20 s each, back to back during one
steady-state client run, so the counters and QPS describe the same load.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

WINDOW_S = 20.0
SAMPLE_HZ = 499  # perf_window.sh's call-graph sampling rate


def postgres_samples(perf_data: Path) -> int:
    """Call-graph samples taken in Postgres processes: 499 per busy CPU-second.

    CPU time comes from these rather than from the `cycles` counter, which
    under-reported busy CPUs here (2.5x at one client, 2x for pgvectorfast at
    16) while the samples and `ref-cycles` agreed with each other."""
    out = subprocess.run(["perf", "report", "-i", str(perf_data), "--no-children", "--sort", "comm",
                          "--stdio", "-g", "none", "-n"], capture_output=True, text=True, check=True).stdout
    m = re.search(r"^\s+[\d.]+%\s+(\d+)\s+postgres\s*$", out, re.M)
    return int(m.group(1)) if m else 0


def num(s: str) -> float:
    return float(s.replace(",", ""))


def main() -> None:
    argv = sys.argv[1:]
    prefix = "prof"
    if argv[0] == "--label":
        prefix, argv = argv[1], argv[2:]
    prof = Path(argv[0])
    qps = {}
    for path in argv[1:]:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r.get("label") in (f"{prefix}-stock", f"{prefix}-fast", f"{prefix}-ntq", f"{prefix}-fastntq"):
                # latest run wins: the profiled run follows any unprofiled one
                qps[(r["label"].rsplit("-", 1)[1], r["clients"], r["search_value"])] = r["qps"]

    cols = ["config", "QPS", "busy CPUs", "IPC", "DRAM rd GB/s", "MB/query", "server CPU ms/query",
            "retiring", "frontend", "bad spec", "backend", "mem bound", "core bound",
            "walk cyc %", "walks/query", "cyc/walk"]
    print(" | ".join(cols))
    print("|".join("---" for _ in cols))
    def arm(name: str) -> str:
        return ("fastntq" if "-fastntq-" in name else "fast" if "-fast-" in name
                else "ntq" if "-ntq-" in name else "stock")

    order = {"stock": 0, "fast": 1, "ntq": 2, "fastntq": 3}
    for stat in sorted(prof.glob("*.stat.txt"), key=lambda p: (order[arm(p.name)], tuple(int(x) for x in re.findall(r"\d+", p.name)[-2:]))):
        name = stat.name.removesuffix(".stat.txt")
        c, ef = (int(x) for x in re.findall(r"\d+", name)[-2:])
        label = arm(name)
        s = stat.read_text()
        cyc = num(re.search(r"([\d,]+)\s+cycles", s).group(1))
        ins = num(re.search(r"([\d,]+)\s+instructions", s).group(1))
        rd_mib = sum(num(x) for x in re.findall(r"([\d,.]+) MiB\s+uncore_imc_\d+/cas_count_read/", s))
        rd_bytes = rd_mib * 2**20
        td = (prof / f"{name}.topdown.txt").read_text()

        def pct(k: str) -> str:
            m = re.search(r"([\d.]+) %\s+tma_" + k + r"\b", td)
            return f"{float(m.group(1)):.0f}%" if m else "?"

        # Page-table walks (tlb.txt, from perf_window.sh's fourth window): the
        # walk share is against ref-cycles, which unlike `cycles` matched the
        # call-graph samples here; at the pinned clock both tick at 2.0 GHz.
        tlb_path = prof / f"{name}.tlb.txt"
        walk = ("?", "?", "?")
        if tlb_path.exists():
            t = tlb_path.read_text()
            g = lambda ev: num(re.search(r"([\d,]+)\s+" + re.escape(ev), t).group(1))
            dur, done, ref = g("dtlb_load_misses.walk_duration"), g("dtlb_load_misses.walk_completed"), g("ref-cycles")
            walk = (f"{dur / ref * 100:.1f}%", None, f"{dur / done:.0f}")
        q = qps.get((label, c, ef))
        cpu_s = postgres_samples(prof / f"{name}.perf.data") / SAMPLE_HZ
        per_q = (lambda v: f"{v / (q * WINDOW_S):.2f}") if q else (lambda v: "?")
        print(" | ".join([
            f"{ {'fast': 'pgvectorfast', 'ntq': 'pgvector_ntq', 'fastntq': 'pgvectorfast+tq4'}.get(label, 'pgvector')}, {c} client(s), ef {ef}",
            f"{q:.0f}" if q else "?",
            f"{cpu_s / WINDOW_S:.1f}", f"{ins / cyc:.2f}",
            f"{rd_bytes / 1e9 / WINDOW_S:.1f}", per_q(rd_bytes / 1e6), per_q(cpu_s * 1e3),
            pct("retiring"), pct("frontend_bound"), pct("bad_speculation"), pct("backend_bound"),
            pct("memory_bound"), pct("core_bound"),
            walk[0], (f"{done / (q * WINDOW_S):.0f}" if q and walk[1] is None and tlb_path.exists() else "?"), walk[2],
        ]))


if __name__ == "__main__":
    main()
