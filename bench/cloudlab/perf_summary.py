#!/usr/bin/env python3
"""Summarize perf_window.sh output: one row per profiled configuration.

  python3 bench/cloudlab/perf_summary.py bench/profiles/pgvector <results.jsonl>...

QPS comes from the client's result rows (label prof-stock, matched on clients
and ef_search). Per configuration it reports busy cores, IPC, DRAM read
bandwidth from the IMC counters, DRAM bytes and cycles per query, and the
Top-Down level-1/2 split. The windows are 20 s each, back to back during one
steady-state client run, so the counters and QPS describe the same load.
"""

import json
import re
import sys
from pathlib import Path

WINDOW_S = 20.0
CLOCK_HZ = 2.0e9  # pinned by bench/cloudlab/pin_cpu.sh


def num(s: str) -> float:
    return float(s.replace(",", ""))


def main() -> None:
    prof = Path(sys.argv[1])
    qps = {}
    for path in sys.argv[2:]:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r.get("label") == "prof-stock":
                qps[(r["clients"], r["search_value"])] = r["qps"]  # latest run wins

    cols = ["config", "QPS", "busy cores", "IPC", "DRAM rd GB/s", "MB/query", "Mcycles/query",
            "retiring", "frontend", "bad spec", "backend", "mem bound", "core bound"]
    print(" | ".join(cols))
    print("|".join("---" for _ in cols))
    for stat in sorted(prof.glob("*.stat.txt"), key=lambda p: tuple(int(x) for x in re.findall(r"\d+", p.name))):
        name = stat.name.removesuffix(".stat.txt")
        c, ef = (int(x) for x in re.findall(r"\d+", name))
        s = stat.read_text()
        cyc = num(re.search(r"([\d,]+)\s+cycles", s).group(1))
        ins = num(re.search(r"([\d,]+)\s+instructions", s).group(1))
        rd_mib = sum(num(x) for x in re.findall(r"([\d,.]+) MiB\s+uncore_imc_\d+/cas_count_read/", s))
        rd_bytes = rd_mib * 2**20
        td = (prof / f"{name}.topdown.txt").read_text()

        def pct(k: str) -> str:
            m = re.search(r"([\d.]+) %\s+tma_" + k + r"\b", td)
            return f"{float(m.group(1)):.0f}%" if m else "?"

        q = qps.get((c, ef))
        per_q = (lambda v: f"{v / (q * WINDOW_S):.2f}") if q else (lambda v: "?")
        print(" | ".join([
            f"{c} client(s), ef {ef}", f"{q:.0f}" if q else "?",
            f"{cyc / CLOCK_HZ / WINDOW_S:.1f}", f"{ins / cyc:.2f}",
            f"{rd_bytes / 1e9 / WINDOW_S:.1f}", per_q(rd_bytes / 1e6), per_q(cyc / 1e6),
            pct("retiring"), pct("frontend_bound"), pct("bad_speculation"), pct("backend_bound"),
            pct("memory_bound"), pct("core_bound"),
        ]))


if __name__ == "__main__":
    main()
