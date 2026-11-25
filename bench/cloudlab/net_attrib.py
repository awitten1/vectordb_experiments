#!/usr/bin/env python3
"""Per-CPU load, network interrupts and Postgres waits from net_diag.sh windows.

  python3 bench/cloudlab/net_attrib.py bench/profiles/hn-netdiag [--top 8]

For each <name> window (server/ and client/ subdirectories):
  - busy CPUs in total and per CPU: the busiest CPUs with their user, system,
    irq and softirq shares, and how many CPUs are over 90% busy
  - NIC interrupts per second per CPU (lines of /proc/interrupts naming the NIC
    or a common NIC driver), and NET_RX / NET_TX softirqs per CPU
  - softnet: packets processed, dropped, and times the budget ran out
    (time_squeeze) per CPU
  - with PGWAIT: the share of sampled client backends in each state and wait
QPS comes from results.jsonl next to the subdirectories (label and clients).
"""
import argparse
import re
from collections import Counter
from pathlib import Path

NIC_RX = re.compile(r"mlx|i40e|ice|bnxt|enp|eth|ixgbe|qede", re.I)


def cpu_lines(p: Path) -> dict:
    out = {}
    for line in p.read_text().splitlines():
        f = line.split()
        if f and re.fullmatch(r"cpu\d+", f[0]):
            out[int(f[0][3:])] = list(map(int, f[1:9]))
    return out


def per_cpu(d: Path, name: str, dt: float) -> dict:
    a, b = cpu_lines(d / f"{name}.stat0"), cpu_lines(d / f"{name}.stat1")
    res = {}
    for c in a:
        x = [j - i for i, j in zip(a[c], b[c])]
        tot = sum(x) or 1
        res[c] = {"user": (x[0] + x[1]) / tot, "sys": x[2] / tot, "irq": x[5] / tot, "soft": x[6] / tot,
                  "busy": (tot - x[3] - x[4]) / tot}
    return res


def irq_per_cpu(d: Path, name: str, dt: float):
    def parse(p):
        lines = p.read_text().splitlines()
        ncpu = len(lines[0].split())
        rows = {}
        for line in lines[1:]:
            f = line.split()
            if not f or not f[0].endswith(":") or not f[0][:-1].isdigit():
                continue
            counts = list(map(int, f[1:1 + ncpu]))
            rows[f[0][:-1]] = (counts, " ".join(f[1 + ncpu:]))
        return rows, ncpu
    a, n = parse(d / f"{name}.irq0")
    b, _ = parse(d / f"{name}.irq1")
    per = [0] * n
    queues = 0
    for irq, (cb, label) in b.items():
        if not NIC_RX.search(label) or irq not in a:
            continue
        delta = [y - x for x, y in zip(a[irq][0], cb)]
        if sum(delta):
            queues += 1
        per = [p + q / dt for p, q in zip(per, delta)]
    return per, queues


def softirqs(d: Path, name: str, dt: float):
    def parse(p):
        lines = p.read_text().splitlines()
        rows = {}
        for line in lines[1:]:
            f = line.split()
            if f and f[0].endswith(":"):
                rows[f[0][:-1]] = list(map(int, f[1:]))
        return rows
    a, b = parse(d / f"{name}.softirq0"), parse(d / f"{name}.softirq1")
    return {k: [(y - x) / dt for x, y in zip(a[k], b[k])] for k in ("NET_RX", "NET_TX") if k in a}


def softnet(d: Path, name: str):
    def parse(p):
        return [[int(x, 16) for x in line.split()[:3]] for line in p.read_text().splitlines()]
    a, b = parse(d / f"{name}.softnet0"), parse(d / f"{name}.softnet1")
    return [[y - x for x, y in zip(r0, r1)] for r0, r1 in zip(a, b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    root = Path(a.dir)
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name in ("server", "client")):
        print(f"\n# {d.name}\n")
        nic = d.glob("*.nic.txt")
        first = next(iter(sorted(nic)), None)
        if first:
            t = first.read_text()
            for sec in ("## ethtool -l", "## ethtool -c", "## NIC IRQs", "## RPS / XPS", "## irqbalance", "## NIC NUMA node", "## driver"):
                m = re.search(re.escape(sec) + r".*?\n(.*?)(?=\n## |\Z)", t, re.S)
                if m:
                    body = "\n".join(m.group(1).splitlines()[:24])
                    print(f"{sec}\n```\n{body}\n```")
        for t0 in sorted(d.glob("*.t0")):
            name = t0.name[:-3]
            dt = float((d / f"{name}.t1").read_text()) - float(t0.read_text())
            pc = per_cpu(d, name, dt)
            busy = sum(v["busy"] for v in pc.values())
            hot = sorted(pc.items(), key=lambda kv: -kv[1]["busy"])
            print(f"\n## {name}: {busy:.1f} of {len(pc)} CPUs busy over {dt:.0f} s; "
                  f"{sum(v['busy'] > 0.9 for v in pc.values())} CPUs over 90% busy; "
                  f"softirq {sum(v['soft'] for v in pc.values()):.2f} CPUs, irq {sum(v['irq'] for v in pc.values()):.2f} CPUs")
            print("busiest CPUs (busy = user + sys + irq + softirq):")
            for c, v in hot[: a.top]:
                print(f"  cpu{c:<3} busy {v['busy']:.0%}  user {v['user']:.0%}  sys {v['sys']:.0%}  irq {v['irq']:.0%}  softirq {v['soft']:.0%}")
            soft = sorted(pc.items(), key=lambda kv: -kv[1]["soft"])
            print("most softirq: " + ", ".join(f"cpu{c} {v['soft']:.0%}" for c, v in soft[:6]))
            per, queues = irq_per_cpu(d, name, dt)
            tot = sum(per)
            if tot:
                top = sorted(enumerate(per), key=lambda kv: -kv[1])
                n_any = sum(1 for x in per if x > 0.01 * tot)
                print(f"NIC interrupts: {tot:,.0f}/s from {queues} IRQ lines, on {n_any} CPUs (>1% each); top: " +
                      ", ".join(f"cpu{c} {x:,.0f}/s" for c, x in top[:6]))
            sq = softirqs(d, name, dt)
            for k, v in sq.items():
                top = sorted(enumerate(v), key=lambda kv: -kv[1])
                print(f"{k}: {sum(v):,.0f}/s; top: " + ", ".join(f"cpu{c} {x:,.0f}/s" for c, x in top[:6]))
            sn = softnet(d, name)
            print(f"softnet: processed {sum(r[0] for r in sn) / dt:,.0f}/s, dropped {sum(r[1] for r in sn)}, "
                  f"time_squeeze {sum(r[2] for r in sn)}")
            w = d / f"{name}.waits.txt"
            if w.exists() and w.stat().st_size:
                c = Counter(line.strip() for line in w.read_text().splitlines() if line.strip())
                n = sum(c.values())
                print("backends (share of samples): " + ", ".join(f"{k} {v / n:.0%}" for k, v in c.most_common(8)))


if __name__ == "__main__":
    main()
