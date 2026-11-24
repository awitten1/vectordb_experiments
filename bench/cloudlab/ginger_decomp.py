#!/usr/bin/env python3
"""Where Ginger's lead over pgvectorfast + tq4 comes from, on HN 27.4M.

  python3 bench/cloudlab/ginger_decomp.py

Combines:
* work per query: bench/profiles/hn-ginger/ginger-stats.jsonl (Ginger's own
  search statistics) and pgvectorfast's uprobe counts from pg_counts.sh
  (bench/profiles/hn-ginger/pgcounts-ef*.csv, copied in below as PG);
* recall/QPS curves: ginger-recommended and thpon-fastntq rows in
  bench/results/*/hackernews_minilm_384.jsonl (pgvectorfast + tq4 with huge
  pages and 2 MB translation arrays, the best measured configuration);
* per-query time by part from the profiles: ginger_attrib.py on
  bench/profiles/hn-ginger (Ginger, beam 64 and 160) and trav_attrib.py on
  bench/profiles/hn-pipe/off (pgvectorfast at ef_search 160), typed in below.

pgvectorfast at ef_search 227 (recall 0.95) is a model: the ef 160 profile's
cost per distance and per re-ranked candidate times the measured work at 227,
plus its fixed per-query costs; it is checked against the measured QPS.
"""
import json, glob, math
GHZ = 2.6e3  # cycles per µs
# --- work counts
PG = {10:(27.6,552.9,15),20:(38.0,774.0,25),40:(58.2,1166.9,45),80:(97.9,1877.4,85),160:(177.7,3179.7,165),
      320:(337.4,5572.3,325.1),640:(655.0,9913.9,645.1),1000:(1016.1,14508.6,1005)}
gi = {}
for l in open('bench/profiles/hn-ginger/ginger-stats.jsonl'):
    r = json.loads(l); s = r['stats']; gi[r['search_value']] = (s['nodes_visited'], s['distance_computations'], s['reranked'], r['recall'])
curves = {}
for f in glob.glob('bench/results/*/hackernews_minilm_384.jsonl'):
    for l in open(f):
        r = json.loads(l); c = r.get('clients') or r.get('threads')
        if r['label'] in ('thpon-fastntq', 'ginger-recommended'):
            curves.setdefault((r['label'], c), {})[r['search_value']] = (r['recall'], r['qps'])
x = lambda r: -math.log10(1 - r)
def at_recall(pts, R):  # -> (width, qps), interpolated in log(width), log(qps) against x(recall)
    p = sorted((v[0], k, v[1]) for k, v in pts.items())
    for (r1, w1, q1), (r2, w2, q2) in zip(p, p[1:]):
        if r1 <= R <= r2:
            t = (x(R) - x(r1)) / (x(r2) - x(r1))
            return math.exp(math.log(w1) + t * math.log(w2 / w1)), math.exp(math.log(q1) + t * math.log(q2 / q1))
def lin(tab, w, i):  # linear in width between measured widths
    ks = sorted(tab)
    for a, b in zip(ks, ks[1:]):
        if a <= w <= b:
            return tab[a][i] + (w - a) / (b - a) * (tab[b][i] - tab[a][i])
print("== work at matched recall (1 thread curves)")
for R in (0.90, 0.95, 0.97):
    wg, qg = at_recall(curves[('ginger-recommended', 1)], R); wp, qp = at_recall(curves[('thpon-fastntq', 1)], R)
    dg, dp = lin(gi, wg, 1), lin(PG, wp, 1)
    _, qg40 = at_recall(curves[('ginger-recommended', 40)], R); _, qp40 = at_recall(curves[('thpon-fastntq', 40)], R)
    print(f"R {R}: ginger beam {wg:.0f} dist {dg:.0f} exp {lin(gi,wg,0):.0f} rr {min(60,wg):.0f} | pg ef {wp:.0f} dist {dp:.0f} hops {lin(PG,wp,0):.0f} rr {lin(PG,wp,2):.0f} | dist ratio {dp/dg:.2f} | qps1 {qg:.0f}/{qp:.0f}={qg/qp:.2f} qps40 {qg40:.0f}/{qp40:.0f}={qg40/qp40:.2f} | non-graph factor 1t {qg/qp/(dp/dg):.2f}")
# --- per-query time model, 1 thread, from the profiles at width 160 (pg) and 64/160 (ginger)
pg160 = dict(wall=1e6/834, search=406+264, rerank=156+79, server_other=73+37+28+5+3, cpu=2732243/GHZ)
pg160['off'] = pg160['wall'] - pg160['cpu']
ns_dist = pg160['search'] / 3179.7 * 1e3; us_rr = pg160['rerank'] / 165
print(f"\n== pg ef160 1 client: wall {pg160['wall']:.0f} µs, server CPU {pg160['cpu']:.0f}, off-server {pg160['off']:.0f}; search {pg160['search']} µs = {ns_dist:.0f} ns = {ns_dist*2.6:.0f} cyc per distance; re-rank {us_rr:.2f} µs per candidate")
g64 = dict(wall=1e6/1391.8, kernel=547, decode=72, beam=29, loop=18, rerank=25, cpu=1838264/GHZ)
g160 = dict(wall=1e6/687.2, kernel=1134, decode=159, beam=55, loop=38, rerank=26, cpu=3728239/GHZ)
for n, g, d in (('g64', g64, gi[64][1]), ('g160', g160, gi[160][1])):
    s = g['kernel'] + g['decode'] + g['beam'] + g['loop']
    print(f"{n}: wall {g['wall']:.0f} cpu {g['cpu']:.0f} search {s} µs = {s/d*1e3:.0f} ns = {s/d*1e3*2.6:.0f} cyc per distance; rerank {g['rerank']} µs = {g['rerank']/60:.2f} µs/cand; other {g['cpu']-s-g['rerank']:.0f}")
wp, qp = at_recall(curves[('thpon-fastntq', 1)], 0.95); dp = lin(PG, wp, 1); rp = lin(PG, wp, 2)
m = dict(search=dp*ns_dist/1e3, rerank=rp*us_rr, other=pg160['server_other'], off=pg160['off'])
print(f"\n== pg at recall 0.95 (ef {wp:.0f}): model {sum(m.values()):.0f} µs vs measured {1e6/qp:.0f}: " + ", ".join(f"{k} {v:.0f}" for k, v in m.items()))
gs = g64['kernel'] + g64['decode'] + g64['beam'] + g64['loop']
print(f"ginger at 0.95 (beam 64): wall {g64['wall']:.0f}: search {gs}, rerank {g64['rerank']}, rest {g64['wall']-gs-g64['rerank']:.0f}")
dg = gi[64][1]
print(f"gap {1e6/qp - g64['wall']:.0f} µs: fewer distances {(dp-dg)*ns_dist/1e3:.0f}; rerank {m['rerank']-g64['rerank']:.0f}; postgres+client {m['other']+m['off']-(g64['wall']-gs-g64['rerank']):.0f}; ginger's dearer distances {-(gs-dg*ns_dist/1e3):.0f}")
# --- 40 threads
pg40 = dict(search=(636+403)*GHZ, rerank=(234+114)*GHZ, other=(126+100+48+10+9)*GHZ, cpu=4367686, qps=15286)
c_dist40 = pg40['search']/3179.7; c_rr40 = pg40['rerank']/165
busy_pg = pg40['cpu']*pg40['qps']/2.6e9
g40 = 2839123; busy_g = g40*36513.4/2.6e9
cpu_pg_095 = dp*c_dist40 + rp*c_rr40 + pg40['other']
print(f"\n== 40: pg search {c_dist40:.0f} cyc/dist, rerank {c_rr40/GHZ:.2f} µs/cand; server busy {busy_pg:.1f} of 40 CPUs; ginger busy {busy_g:.1f}")
print(f"ginger 40 b64: {g40} cyc/query = {g40/dg:.0f}/dist all-in; search {0.942*g40/dg:.0f} cyc/dist")
print(f"pg CPU per query at 0.95 (model) {cpu_pg_095/1e6:.2f}M vs ginger {g40/1e6:.2f}M -> {cpu_pg_095/g40:.2f}x; with busy {busy_pg/40:.0%} vs {busy_g/40:.0%} -> predicted QPS ratio {cpu_pg_095/g40*(busy_g/busy_pg):.2f}")
