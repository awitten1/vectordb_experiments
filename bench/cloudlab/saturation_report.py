#!/usr/bin/env python3
"""QPS and CPU busy share per client count, from hn_saturation_probe.sh.

  saturation_report.py DIR

DIR holds results.jsonl (ann_bench rows), server-cpu.txt and client-cpu.txt
("<epoch> cpu <user> <nice> <system> <idle> <iowait> <irq> <softirq> <steal>
..." every 2 s).

ann_bench stamps a whole sweep with one timestamp, taken before its first
point, so each point's window is found in the server trace: points run back
to back, each busy for its duration, with a few idle seconds between them
(connecting the clients, the plan check). A point starts at the first sample
over START_BUSY after the previous point's end (its start + duration, then the
first sample under IDLE_BUSY). The busy share is taken over [start + 5 s,
start + duration - 5 s]. "stalled" counts the seconds in that window when the
server was under IDLE_BUSY, i.e. every client was waiting on something else.

If DIR also holds nic.txt (the server's link monitor: "<epoch> <carrier>
<rx_crc_errors> <rx_packets>" every 2 s), "link" gives, for each point's
window, the samples with the link down and the new receive CRC errors; "–"
where the monitor doesn't cover the window.
"""
import json
import sys
from datetime import datetime
from itertools import groupby
from pathlib import Path

START_BUSY = 0.30
IDLE_BUSY = 0.10


def samples(path):
    out = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) < 9 or f[1] != "cpu":
            continue
        v = [int(x) for x in f[2:]]
        idle = v[3] + v[4]
        out.append((int(f[0]), sum(v[:8]), idle))
    return out


def busy(s, t0, t1):
    """Busy share between the samples nearest t0 and t1"""
    a = min(s, key=lambda x: abs(x[0] - t0))
    b = min(s, key=lambda x: abs(x[0] - t1))
    if b[1] <= a[1]:
        return None
    return 1 - (b[2] - a[2]) / (b[1] - a[1])


def intervals(s):
    """(t, busy share over the next interval) for each sample"""
    return [(a[0], 1 - (b[2] - a[2]) / (b[1] - a[1])) for a, b in zip(s, s[1:]) if b[1] > a[1]]


def windows(iv, t0, n, duration):
    """Start times of n back-to-back points after t0, or None where not found"""
    out, t = [], t0 - 2
    for _ in range(n):
        start = next((u for u, b in iv if u >= t and b > START_BUSY), None)
        out.append(start)
        if start is None:
            continue
        end = start + duration
        t = next((u for u, b in iv if u >= end and b < IDLE_BUSY), end)
    return out


def nic_samples(path):
    out = []
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) == 4 and all(x.isdigit() for x in f):
            out.append(tuple(int(x) for x in f[:3]))
    return out


def link(nic, a, b):
    """(down samples, new CRC errors) in [a, b], or None if not covered"""
    if not nic or nic[0][0] > a or nic[-1][0] < b:
        return None
    w = [x for x in nic if a <= x[0] <= b]
    if not w:
        return None
    return sum(1 for x in w if x[1] != 1), w[-1][2] - w[0][2]


def main():
    d = Path(sys.argv[1])
    rows = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()]
    srv, cli = samples(d / "server-cpu.txt"), samples(d / "client-cpu.txt")
    iv = intervals(srv)
    nic = nic_samples(d / "nic.txt")
    # The recall checks run 5 s per point at 1 client: no CPU to report
    rows = [r for r in rows if (r.get("duration_s") or 0) >= 30]
    out = []
    # One sweep: the rows sharing a label and timestamp, in the order they ran
    for _, sweep in groupby(rows, key=lambda r: (r["label"], r["timestamp"])):
        sweep = list(sweep)
        dur = sweep[0].get("duration_s") or 60
        t0 = datetime.fromisoformat(sweep[0]["timestamp"]).timestamp()
        for r, start in zip(sweep, windows(iv, t0, len(sweep), dur)):
            if start is None:
                out.append((r, None, None, None, None))
                continue
            a, b = start + 5, start + dur - 5
            stalled = sum(2 for u, x in iv if a <= u < b and x < IDLE_BUSY)
            out.append((r, busy(srv, a, b), busy(cli, a, b), stalled, link(nic, start, start + dur)))
    head = "| label | clients | ef_search | recall@10 | QPS | server busy | client busy | stalled |"
    print(head + (" link (down / CRC) |" if nic is not None else ""))
    print("|---|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if nic is not None else ""))
    fmt = lambda x: "–" if x is None else f"{x:.0%} ({x * 40:.1f} of 40)"
    for r, sb, cb, st, lk in sorted(out, key=lambda o: (o[0]["label"], o[0]["clients"], o[0]["search_value"])):
        row = f"| {r['label']} | {r['clients']} | {r['search_value']} | {r['recall']:.4f} | {r['qps']:,.0f} | {fmt(sb)} | {fmt(cb)} | {'–' if st is None else f'{st} s'} |"
        if nic is not None:
            row += (" – |" if lk is None else f" {lk[0]} / {lk[1]} |")
        print(row)


if __name__ == "__main__":
    main()
