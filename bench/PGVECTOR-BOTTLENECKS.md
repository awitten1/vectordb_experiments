# What limits pgvector's HNSW at 1536 dimensions

Profiling of stock pgvector 0.8.7 HNSW on DBpedia-OpenAI 1M, with everything in
RAM, to find what bounds its query performance. The broader four-engine
comparison this came out of is in [FINDINGS-dbpedia.md](FINDINGS-dbpedia.md).

## Summary

- **Every element HNSW visits costs a 6 KB read from DRAM.** pgvector stores the
  full float32 vector in each graph element's index page, and nearly every visit
  misses all caches. A query at recall 0.99 visits about 8,500 elements and reads
  about **60 MB from DRAM**.
- **Under concurrency, DRAM bandwidth is the ceiling.** This machine delivers
  30–35 GB/s, which caps any engine reading those pages at **500–580 QPS** at
  recall 0.99. pgvectorfast hits that ceiling at 8 clients (33 GB/s) and gains
  nothing beyond. Stock pgvector reaches 27.5 GB/s at 16 clients.
- **On a single query, a third of the CPU time is the buffer manager.** Finding,
  pinning and locking each element's page costs as much as the distance itself.
  This is the part pgvectorfast removes, worth 2× at one client.
- **Page-table walks take 13% of cycles**, because `shared_buffers` is mapped
  with 4 KB pages.
- **The distance arithmetic is not a bottleneck.** pgvector's kernel would be
  3.4× faster with more accumulators when the data is in cache. But the data
  comes from DRAM, where any kernel waits the same 0.7 µs per vector.

**The lever that matters is bytes per visited element**, which the
buffer-manager work cannot get past. Ginger, a C++ library whose graph search
reads 4-bit compressed codes (a few hundred bytes per element, estimated from
its index size), ran **8,886 QPS at recall 0.99 with 16 threads on the same
machine**.

## Setup

| | |
|---|---|
| Data | DBpedia-OpenAI 1M (Qdrant): 975,000 × 1536-d `text-embedding-ada-002`, L2 (vectors are unit-norm, so L2 ranks like cosine) |
| Index | pgvector 0.8.7 HNSW, m=16, ef_construction=64; 7.4 GiB index plus a 7.4 GiB TOAST table |
| Machine | CloudLab m510: Xeon D-1548, 8 cores / 16 threads, AVX2, 12 MiB L3, 2-channel DDR4, 62 GiB RAM |
| Conditions | Clock pinned at 2.0 GHz (turbo and deep C-states off), swap off, every table and index block resident in `shared_buffers` (28 GB, checked after every run) |
| Client | 1 client: on the server's machine. 8–16 clients: on an identical second node over 10 GbE, so the server keeps its cores |

`ef_search=640` gives recall@10 = 0.989 and `ef_search=40` gives 0.913. Both
are used throughout.

## 1. Each visited element is a 6 KB DRAM read

| ef_search | recall@10 | pages per query | DRAM read per query |
|---:|---:|---:|---:|
| 40 | 0.913 | 1,060 | 6.5–7.0 MB |
| 640 | 0.989 | 8,555 | 58–62 MB |

- **Pages per query** are buffer accesses from `EXPLAIN (ANALYZE, BUFFERS)`
  over 100 queries.
- **DRAM reads** come from the memory controllers' counters during sustained
  load, divided by queries served.
- A 1536-d `vector` is 6 KB, so each 8 KB index page holds one graph element.
  DRAM traffic works out to 6–7 KB per page visited: essentially the vector
  itself, every time.
- The index is 600× the size of the L3 cache, and HNSW's access pattern is
  random. A visited element is essentially never in cache.

## 2. Under concurrency, DRAM bandwidth is the ceiling

![Throughput vs clients against the DRAM ceiling](report/bottleneck-throughput.svg)

- **Where the ceiling comes from:** the machine's read bandwidth, measured with
  a microbenchmark (`membw.c`), divided by bytes per query. Random 6 KB reads
  reach 30 GB/s; sequential reads reach 35 GB/s.
- **pgvectorfast** reaches 585 QPS at 8 clients and then stays flat.
- **Stock pgvector** spends more CPU per element, so it climbs more slowly. At
  16 clients it reaches 458 QPS.

The memory controllers show both engines arriving at the ceiling:

![DRAM read bandwidth vs clients](report/bottleneck-bandwidth.svg)

| | clients | QPS | DRAM read | IPC |
|---|---:|---:|---:|---:|
| pgvector | 1 | 56 | 3.4 GB/s | 0.73 |
| pgvector | 8 | 325 | 18.9 GB/s | 0.62 |
| pgvector | 16 | 446 | 27.5 GB/s | 0.35 |
| pgvectorfast | 8 | 574 | 32.9 GB/s | 0.46 |
| pgvectorfast | 16 | 562 | 33.4 GB/s | 0.21 |

(`ef_search=640`, measured during profiling, so QPS is 3–8% below the regular
runs in the figure above.)

Going from 8 to 16 clients, pgvectorfast's IPC halves and its cycles per query
double, with no extra throughput: the signature of a memory-bandwidth wall.
Both engines read the same pages, so they share the ceiling. pgvectorfast just
reaches it sooner.

## 3. Within a query: where the CPU time goes

![CPU time per visited element by category](report/bottleneck-breakdown.svg)

At one client and `ef_search=640`, a query takes 17.8 ms of server CPU, or
**2.1 µs per visited element**. These shares come from system-wide call-graph
sampling and stay within a few points at 8 and 16 clients and at
`ef_search=40`.

- **Distance (32%)** is the cost of bringing the 6 KB vector in from DRAM (see
  section 4).
- **Buffer manager (33%), split across two rows:**
  - `ReadBuffer` → `BufTableLookup` → `hash_search_with_hash_value` probes the
    shared buffer mapping table for every element's page (18%).
  - Pinning, unpinning, content locks and private refcounts make up the rest
    (15%).

  This is the cost Calico (pgvectorfast) replaces with array lookups and
  optimistic reads.
- **Element and neighbor loading (18%)** is reading element and neighbor
  tuples out of the page. That mostly means cache misses on the page itself.
- **Visited set (5%) and candidate queues (3%)** are pgvector's per-search
  bookkeeping.
- **Page-table walks (12.7% of all cycles)** overlap all of the above. There
  are ~5.5 dTLB misses per element at ~94 cycles per walk, because the 28 GB
  buffer pool is mapped with 4 KB pages: about 7 million of them, with page
  tables far larger than the cache.

**The CPU's own breakdown** (Top-Down, at one client and `ef_search=640`):

| | share of cycles |
|---|---:|
| retiring useful work | 21% |
| backend-bound | 65% |
| … of which waiting on memory | 45% |
| front-end-bound | 10% |
| bad speculation | 4% |

At 16 clients, the memory-bound share rises to 56%.

## 4. The distance arithmetic is not the bottleneck

![Distance kernel: cached vs DRAM](report/bottleneck-kernel.svg)

pgvector's `VectorL2SquaredDistance` is a plain loop the compiler vectorizes
with a single accumulator. A microbenchmark (`distbench.c`) runs it verbatim,
built with pgvector's flags:

- **With both vectors in cache:** 447 ns. Each of 192 AVX2 multiply-adds waits
  for the previous one. Four independent accumulators bring it to 131 ns.
- **With a fresh vector from DRAM each time** (HNSW's case): both versions take
  ~0.73 µs, the time to stream 6 KB on one core. The distance time measured
  inside real queries is 0.67 µs per element.

A better kernel would help a cache-resident dataset. It would not help this
one.

## What would move it

These are predictions from the measurements, not yet tested, ranked by
expected effect:

1. **Fewer bytes per visited element.** This raises the throughput ceiling in
   proportion and shortens single queries.
   - A `halfvec` index (3 KB per element, built into pgvector) should
     roughly double the ceiling.
   - Compressed codes in the graph with a short rerank, as Ginger does at a
     few hundred bytes per element, raise it by an order of magnitude.
2. **Less buffer-manager work per element.** pgvectorfast shows the effect:
   2× at one client and 1.7× at 8. It stops at the bandwidth ceiling, so it
   pays off most combined with (1).
3. **Huge pages for `shared_buffers`** (`huge_pages=on` with reserved huge
   pages) to remove most of the 12.7% spent in page-table walks.
4. **Fewer elements per query at the same recall.** A better-connected graph
   needs fewer hops, which cuts both bytes and per-element overhead.
5. **Not worth it here:** a faster distance kernel. It only helps when vectors
   are already in cache.

The first and third are configuration-level experiments on the existing setup.

## Method

- **Profiled configurations:** 1, 8 and 16 clients at `ef_search` 40 and 640.
  The client ran on the second node, so the server node ran only Postgres.
- **Each configuration:** one 80-second steady-state run with three
  consecutive 20-second `perf` windows (`bench/cloudlab/perf_window.sh`):
  1. core counters plus memory-controller (`uncore_imc`) read traffic
  2. Top-Down levels 1–2
  3. system-wide call graphs from a frame-pointer build at 499 Hz
- **Page-walk cost:** `dtlb_load_misses.walk_duration` in a separate run.
- **Every number above** is in
  [report/dbpedia-pgvector-bottlenecks.json](report/dbpedia-pgvector-bottlenecks.json),
  and `bench/plot_bottlenecks.py` draws the figures from it.

```sh
# on the server node: frame pointers, and perf access (resets on reboot)
FRAME_POINTERS=1 ./build_postgres.sh && ./build_extensions.sh
sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0
# from the workstation (see FINDINGS-dbpedia.md for node setup)
bench/cloudlab/profile_pgvector.sh
python3 bench/cloudlab/perf_summary.py bench/profiles/pgvector <client results.jsonl>   # on the server
python3 bench/cloudlab/perf_categories.py bench/profiles/pgvector/*.perf.data           # on the server
```

## Caveats

- **One machine.** A two-channel, AVX2-only Xeon D. A server with more memory
  channels raises the bandwidth ceiling but not the bytes per query, so the
  ranking above should hold. The absolute ceiling will not.
- **Profiling costs throughput.** Call-graph sampling cost 8% at one client and
  `ef_search=640` (60.9 → 55.9 QPS). The breakdown is a share of time and is
  unaffected; absolute QPS during profiling is slightly low.
- **One run per configuration.** Repeated runs elsewhere in this study agreed
  within 5% at 8 clients and within ~15% at 1 client. The 16-client runs were
  not repeated.
