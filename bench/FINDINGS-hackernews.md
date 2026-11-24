# pgvector vs pgvectorfast vs pgvector_ntq vs Ginger on Hacker News 28.7M (384-d)

Session of 2026-11-19 on two CloudLab c220g2 nodes (Wisconsin). It is a
companion to `FINDINGS-dbpedia.md` (1M × 1536-d, m510 nodes). Here the dataset
is 27× larger and its vectors are 4× smaller. The hardware is different, so
absolute numbers are not comparable with that study.

![recall vs QPS, 1 client](report/hackernews-minilm-384-variants-1c-recall-qps.svg)
![recall vs QPS, 10 clients](report/hackernews-minilm-384-variants-10c-recall-qps.svg)
![recall vs QPS, 20 clients](report/hackernews-minilm-384-variants-20c-recall-qps.svg)
![recall vs QPS, 40 clients](report/hackernews-minilm-384-variants-40c-recall-qps.svg)

## Setup

- **Hardware:** two identical c220g2 nodes. Each has:
  - 2× Xeon E5-2660 v3 (Haswell EP), 10 cores / 20 threads per socket, so 2
    NUMA nodes and 40 threads in total
  - 25 MiB L3 per socket and 160 GB DDR4-2133
  - a 480 GB SATA SSD, which holds `/mydata`; the boot disk is an HDD
  - a 10 GbE experiment LAN
  - **AVX2 only:** no AVX-512 and no VNNI

  The OS is Ubuntu 24.04.5 with kernel 6.8.
- **Roles:**
  - `c220g2-011111` is the Postgres server. CloudLab calls it `node-1`, and
    its LAN address is 10.10.1.2.
  - `c220g2-011114` runs the network clients and Ginger. CloudLab calls it
    `node-0`.
- **Clock and memory settings:**
  - The clock is pinned at the 2.6 GHz base frequency, with turbo off and idle
    states deeper than C1 disabled (`pin_cpu.sh`, `FREQ_KHZ=2600000`).
  - Swap is off.
  - Postgres and Ginger both run under `numactl --interleave=all`, so memory
    is spread across both sockets. Every server start is checked for the
    interleave policy in its `numa_maps`.
- **Software:**
  - PostgreSQL 18.6.
  - Stock pgvector 0.8.7 and pgvectorfast, both built `-march=native`. They are
    swapped behind `$libdir/vector` and serve the same index.
  - Ginger (submodule `13c0929`), driven in-process by `bench/ginger`.
- **Dataset:** ClickHouse's Hacker News embeddings (`all-MiniLM-L6-v2`,
  384-d, published as cosine). The loader is `datasets.py`, spec `hackernews`.
  - **Source:** 28,737,557 vectors. **1,281,798 exact duplicates (4.5%) were
    dropped**, keeping the lowest id of each. Hacker News repeats a lot of text,
    such as empty comments and reposts.
  - **Held-out queries:** a seeded pool of 10,000 rows. The base is the
    remaining **27,445,759** vectors.
  - **Metric:** vectors are unit-norm (max |norm − 1| = 1.8e-7), so L2 ranks
    like cosine and every engine computes L2.
  - **Ground truth:** exact top-100, recomputed locally. It is byte-identical on
    both nodes (sha256).
  - **Unambiguous queries only.** The same text embedded twice can differ by
    float noise (~1e-7), which exact deduplication misses. A query is kept only
    if:
    - its nearest neighbour is ≥ 1e-5 away, and
    - its 10th and 11th neighbours differ by ≥ 1e-5 (exact float64
      distances).

    38 of the first 1,266 pool queries (3.0%) failed that check. The first
    1,000 that passed are used.
- **Protocol:**
  - Each point runs for 60 s, with k=10, sweeping `ef_search` (HNSW) or beam
    size (Ginger) over 10–1000.
  - At 1 client the Python client runs on the server node. From 10 clients up
    it runs on the client node over the LAN, as in the DBpedia study.
  - `shared_buffers=112GB`. The table and index are prewarmed, and a residency
    check confirms every block is in memory before each arm.
  - At 1 client, the arms alternate stock / fast / stock / fast. At 10–40
    clients each arm runs once.
  - Ginger runs on the client node with Postgres stopped there.

## Results: QPS at matched recall@10

Interpolated in log(QPS) against −log10(1 − recall); rounds are averaged.

| engine | clients | 0.90 | 0.95 | 0.97 | 0.98 | 0.99 |
|---|---:|---:|---:|---:|---:|---:|
| pgvector HNSW | 1 | 257 | 113 | 55 | – | – |
| pgvector HNSW, huge pages | 1 | 341 | 146 | 71 | – | – |
| pgvectorfast HNSW | 1 | 692 | 315 | 159 | – | – |
| pgvectorfast HNSW, huge pages | 1 | 885 | 410 | 209 | – | – |
| pgvector_ntq HNSW (tq4 + ntq12) | 1 | 251 | 113 | 53 | 31 | – |
| pgvectorfast + tq4 (ntq12) | 1 | 635 | 298 | 140 | 83 | – |
| pgvectorfast + tq4, optimized kernels | 1 | 1,003 | 483 | 233 | 140 | – |
| pgvectorfast + tq4, optimized kernels, huge pages | 1 | 1,361 | 653 | 314 | 187 | – |
| Ginger recommended | 1 | 2,309 | 1,436 | 1,037 | 808 | 539 |
| Ginger float-rerank | 1 | 2,387 | 1,503 | 1,083 | 842 | 559 |
| pgvector HNSW | 10 | 2,154 | 908 | 429 | – | – |
| pgvector HNSW, huge pages | 10 | 2,981 | 1,263 | 607 | – | – |
| pgvectorfast HNSW | 10 | 5,791 | 2,655 | 1,342 | – | – |
| pgvector_ntq HNSW (tq4 + ntq12) | 10 | 2,238 | 1,000 | 455 | 258 | – |
| pgvectorfast + tq4 (ntq12) | 10 | 6,088 | 2,792 | 1,300 | 771 | – |
| pgvectorfast + tq4, optimized kernels | 10 | 8,847 | 4,368 | 2,084 | 1,233 | – |
| Ginger recommended | 10 | 22,392 | 14,009 | 10,143 | 7,934 | 5,242 |
| pgvector HNSW | 20 | 3,242 | 1,328 | 627 | – | – |
| pgvector HNSW, huge pages | 20 | 4,369 | 1,806 | 858 | – | – |
| pgvectorfast HNSW | 20 | 7,005 | 2,970 | 1,447 | – | – |
| pgvector_ntq HNSW (tq4 + ntq12) | 20 | 3,300 | 1,450 | 654 | 370 | – |
| pgvectorfast + tq4 (ntq12) | 20 | 8,146 | 3,668 | 1,690 | 997 | – |
| pgvectorfast + tq4, optimized kernels | 20 | 12,031 | 5,589 | 2,597 | 1,534 | – |
| Ginger recommended | 20 | 43,078 | 26,902 | 19,452 | 15,207 | 9,867 |
| pgvector HNSW | 40 | 5,197 | 2,330 | 1,144 | – | – |
| pgvector HNSW, huge pages | 40 | 6,644 | 3,030 | 1,510 | – | – |
| pgvectorfast HNSW | 40 | 10,884 | 4,853 | 2,426 | – | – |
| pgvectorfast HNSW, huge pages | 40 | 11,338 | 4,870 | 2,416 | – | – |
| pgvector_ntq HNSW (tq4 + ntq12) | 40 | 5,275 | 2,511 | 1,186 | 693 | – |
| pgvectorfast + tq4 (ntq12) | 40 | 12,849 | 6,126 | 3,074 | 1,892 | – |
| pgvectorfast + tq4, optimized kernels | 40 | 16,953 | 8,970 | 4,453 | 2,748 | – |
| pgvectorfast + tq4, optimized kernels, huge pages | 40 | 20,350 | 11,550 | 5,831 | 3,597 | – |
| Ginger recommended | 40 | 58,251 | 36,354 | 26,301 | 20,574 | 13,109 |

Ginger float-rerank is within 3–7% of the recommended config at every
thread count; the full curves are in the charts and in `bench/results/`.

**Builds and sizes:**

| | build | threads | size |
|---|---:|---:|---|
| pgvector / pgvectorfast HNSW (m=16, ef_construction=64) | 4,105 s, plus 531 s to load the table | 39 workers | index 52 GB + heap 42 GB |
| pgvector_ntq HNSW (tq4 + ntq12, m=16, ef_construction=64) | 4,586 s, plus 604 s to load its table copy | 39 workers | index 24 GB + heap 42 GB |
| Ginger recommended | 1,206 s | 40 | 17.6 GB, everything included (float data dropped) |
| Ginger float-rerank | 1,162 s | 40 | 46.9 GB, including the floats |

The HNSW build used `maintenance_work_mem=80GB`, and the graph stayed in memory
with no spill.

## Four-way: pgvector, pgvectorfast, each on the HNSW and the PiPNN graph, and Ginger

Session of 2026-11-23 (`hn_pipnn_stock.sh`), on top of the runs of the
"Is it the graph?" section below. It adds the one arm that was missing:
**stock pgvector 0.8.7 searching the PiPNN graph** that the pgvectorfast fork
imported (`items_hn_pipnn_hnsw_m32_ef_construction64`). With it, each engine
runs on both graphs, so the graph's gain and Calico's gain can be read apart.

![four-way, 1 client, 4 KB pages](report/hackernews-minilm-384-fourway-1c-recall-qps.svg)
![four-way, 40 clients, 4 KB pages](report/hackernews-minilm-384-fourway-40c-recall-qps.svg)
![four-way, 1 client, huge pages](report/hackernews-minilm-384-fourway-hp-1c-recall-qps.svg)
![four-way, 40 clients, huge pages](report/hackernews-minilm-384-fourway-hp-40c-recall-qps.svg)

**QPS at matched recall@10** (`hn_fourway.py`; full tables, with 10 and 20
clients, in `bench/profiles/hn-pipnn/fourway.md`):

| pages | clients | engine | 0.90 | 0.95 | 0.97 | 0.98 | 0.99 |
|---|---:|---|---:|---:|---:|---:|---:|
| 4 KB | 1 | pgvector, HNSW graph | 251 | 109 | 54 | – | – |
| 4 KB | 1 | pgvector, PiPNN graph | 339 | 221 | 163 | 125 | 77 |
| 4 KB | 1 | pgvectorfast, HNSW graph | 750 | 350 | 180 | – | – |
| 4 KB | 1 | pgvectorfast, PiPNN graph | 1,025 | 725 | 557 | 442 | 285 |
| 4 KB | 1 | Ginger | 2,309 | 1,436 | 1,037 | 808 | 539 |
| 4 KB | 40 | pgvector, HNSW graph | 5,197 | 2,330 | 1,144 | – | – |
| 4 KB | 40 | pgvector, PiPNN graph | 6,767 | 4,535 | 3,394 | 2,622 | 1,637 |
| 4 KB | 40 | pgvectorfast, HNSW graph | 10,198 | 4,360 | 2,161 | – | – |
| 4 KB | 40 | pgvectorfast, PiPNN graph | 15,254 | 10,480 | 7,816 | 6,044 | 3,752 |
| 4 KB | 40 | Ginger | 58,251 | 36,354 | 26,301 | 20,574 | 13,109 |
| huge | 1 | pgvector, HNSW graph | 335 | 145 | 70 | – | – |
| huge | 1 | pgvector, PiPNN graph | 439 | 293 | 218 | 166 | 102 |
| huge | 1 | pgvectorfast, HNSW graph | 915 | 439 | 227 | – | – |
| huge | 1 | pgvectorfast, PiPNN graph | 1,205 | 857 | 662 | 527 | 340 |
| huge | 40 | pgvector, HNSW graph | 6,644 | 3,030 | 1,510 | – | – |
| huge | 40 | pgvector, PiPNN graph | 8,817 | 5,873 | 4,427 | 3,447 | 2,196 |
| huge | 40 | pgvectorfast, HNSW graph | 12,584 | 5,340 | 2,663 | – | – |
| huge | 40 | pgvectorfast, PiPNN graph | 16,305 | 11,024 | 8,186 | 6,354 | 3,921 |
| huge | 1 | Ginger | 2,830 | 1,751 | 1,261 | 985 | 651 |
| huge | 40 | Ginger | 73,938 | 45,707 | 32,921 | 25,778 | 16,334 |

On "huge" pages, Postgres runs with `huge_pages=on` and Ginger on transparent
huge pages (see "Ginger on huge pages" below). The PiPNN arms ran at 1 and 40
clients only.

**Graph and engine, read apart** (4 KB pages, recall 0.95; each cell divides
two rows above):

| | 1 client | 40 clients |
|---|---:|---:|
| graph, HNSW → PiPNN, under stock pgvector | 2.03× | 1.95× |
| graph, HNSW → PiPNN, under pgvectorfast | 2.07× | 2.40× |
| engine, pgvector → pgvectorfast, on the HNSW graph | 3.22× | 1.87× |
| engine, pgvector → pgvectorfast, on the PiPNN graph | 3.28× | 2.31× |
| both: pgvector on HNSW → pgvectorfast on PiPNN | 6.66× | 4.50× |
| Ginger over pgvectorfast on PiPNN | 1.98× | 3.47× |

- **The two gains multiply.** At 1 client the graph is worth ~2× and Calico
  ~3.2× whichever the other is (2.03 × 3.28 = 6.66). At 40 clients they
  compound a little more: on the PiPNN graph Calico gives 2.31× instead of
  1.87×. PiPNN scores 25–39 new neighbours per hop against HNSW's 15–18, so
  more of each query is page access, which is what Calico speeds up.
- **The graph's gain grows with recall.** Under stock pgvector it is 2.0× at
  recall 0.95 and 3.0× at 0.97 (1 and 40 clients), and the HNSW graph never
  reaches 0.98.
- **Stock pgvector on the PiPNN graph matches pgvectorfast on the HNSW
  graph at 40 clients** (4,535 against 4,360 at recall 0.95, within noise)
  and passes it above (3,394 against 2,161 at 0.97). At 1 client
  pgvectorfast still leads (350 against 221).
- **Huge pages** give stock pgvector 1.33× on the PiPNN graph at 1 client and
  1.30× at 40, about as much as on the HNSW graph (1.33×, 1.30×). The graph's
  gain holds on them: 2.02× at 1 client and 1.94× at 40.
- **Ginger's remaining lead** over the best Postgres arm is 2.0× at 1 client
  and 3.5× at 40 on 4 KB pages. With both on 2 MB pages it is 2.0× and 4.1×.
  The "Is it the graph?" section factors it: per-query overhead in Postgres
  at 1 client, and quantized distances at 40.

**Checks:**
- Stock pgvector reads the fork-built index correctly. Its recall equals the
  fork's at every `ef_search`, on both page sizes (difference 0.0000;
  `recall_gate.py` stops the driver above 0.002).
- The new 1-client pgvector baseline runs its client over the LAN, like every
  PiPNN arm. It gives 109 QPS at recall 0.95 (4 KB pages) and 145 (huge
  pages), against 113 and 146 with the client on the server (main study).
- Every server start passed the residency check (all blocks of the table and
  index in `shared_buffers`) and the NUMA interleave check. With huge pages,
  114 GB (PiPNN table) and 96 GB (main table) of the 115 GB pool were in use.

**Source rule** (`hn_fourway.py`, and the `hackernews-fourway` series in
`plot_variants.py`): at each client count, the rows that share a library and a
protocol with the PiPNN arms. So:
- pgvectorfast is the fork's library at 1 and 40 clients. That is the same
  library as the PiPNN arm, so only the graph differs. At 10 and 20 clients it
  is the upstream library (`sc-fast`). At 40 clients upstream measured 4,853
  QPS at recall 0.95 against the fork's 4,360, in a different session.
- The main study's 1-client rows (client on the server) are left out. They
  are listed in `fourway.md`.

### Ginger on huge pages

Session of 2026-11-23 (`hn_ginger_thp.sh`), on the client node, searching the
cached recommended index the main study measured.

Ginger doesn't ask for huge pages where it matters. Its one
`madvise(MADV_HUGEPAGE)` (`ginger/dataset.h`) covers the float dataset, which
the recommended config drops after the build. The codes and the graph that
search reads come from jemalloc, and THP is `madvise` on the nodes, so they sit
on 4 KB pages. Ginger has no setting for it. Running `ginger_bench` with
`MALLOC_CONF=thp:always,metadata_thp:always` makes jemalloc ask for huge pages
on every mapping it makes, for that process only, with no change to Ginger or
to the system's THP mode.

Each thread count ran a 4 KB control and then the huge-pages run, after
dropping the page cache and compacting memory. A sampler read the process's
`smaps_rollup` every 15 s, and `ginger_thp_promote.py` kept a run only if its
memory was where it should be. Every huge-pages run had 99.4–99.7% of its
17.5 GB on 2 MB pages, and every control 0%.

| threads | recall | 0.90 | 0.95 | 0.97 | 0.98 | 0.99 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 4 KB (control) | 2,285 | 1,425 | 1,032 | 810 | 538 |
| 1 | 2 MB pages | 2,830 | 1,751 | 1,261 | 985 | 651 |
| 1 | gain | 1.24× | 1.23× | 1.22× | 1.22× | 1.21× |
| 10 | 4 KB (control) | 22,364 | 13,974 | 10,123 | 7,924 | 5,251 |
| 10 | 2 MB pages | 27,613 | 17,035 | 12,236 | 9,528 | 6,264 |
| 10 | gain | 1.23× | 1.22× | 1.21× | 1.20× | 1.19× |
| 20 | 4 KB (control) | 43,210 | 27,075 | 19,614 | 15,324 | 9,994 |
| 20 | 2 MB pages | 53,646 | 33,298 | 23,920 | 18,548 | 12,077 |
| 20 | gain | 1.24× | 1.23× | 1.22× | 1.21× | 1.21× |
| 40 | 4 KB (control) | 59,310 | 37,185 | 26,908 | 20,984 | 13,355 |
| 40 | 2 MB pages | 73,938 | 45,707 | 32,921 | 25,778 | 16,334 |
| 40 | gain | 1.25× | 1.23× | 1.22× | 1.23× | 1.22× |

- **Huge pages make Ginger 1.19–1.25× faster** at every thread count and
  recall. Recall is the same at every beam size, so the gain is all page
  size.
- **On the same PiPNN graph, Ginger keeps its gain at 40 threads and
  pgvectorfast doesn't.** Float pgvectorfast gets 1.18× from huge pages at 1
  client and 1.05× at 40, where moving float vectors is the limit. Ginger
  reads a 192-byte code row per distance instead of a 1,536-byte vector, and
  keeps 1.23×. (On the HNSW graph, pgvectorfast's gain holds: 1.25× and
  1.22×.)
- **The control agrees with the main study** (Nov 20, same index) within 2.3%
  at every thread count and recall.

### Ginger's node order for the Postgres index (measured offline: not worth building)

Ginger can renumber the graph so neighbours get nearby numbers. The
`pipnn-reorder` branch of pgvectorfast lets `pipnn_export --reorder
partitioning` write the graph in that order. The import still lays elements
out in heap order, so it would also need an import that follows the file's
order. Before building that import, `bench/cloudlab/graph_layout.py` measured
offline what it would buy: the distinct index pages a search touches when
it expands one node (4 elements per 8 KB page).

| 27.4M PiPNN graph | pages per expansion | neighbours on the node's page | 2 MB regions per expansion |
|---|---:|---:|---:|
| heap order (today's import) | 39.38 | 0.0% | 38.92 |
| Ginger's order (`--reorder partitioning`) | 38.70 | 0.7% | 25.82 |
| random order | 39.39 | 0.0% | 39.36 |

- **The pages a hop reads barely change: −1.7%** (−2.3% at 1M), far below the
  25% gate set beforehand. A node keeps about 39 neighbours, chosen by
  α-pruning to point in different directions. Reordering puts them near
  their node but almost never on its page.
- **Larger pages don't change that.** At 1M, 16 KB and 32 KB pages (8 and 16
  elements) give −4.8% and −8.6%.
- **Heap order is no better than random**, so today's import gives up no
  locality.
- **What reordering does cut is 2 MB regions per hop (−34%).** That could
  only help with `huge_pages=on`, by reducing TLB misses. That isn't the
  limit measured here, which is the bytes read per neighbour ("What
  traversal is made of", "A pipelined hop").
- Ginger's reordering costs almost nothing: 3.1 s of a 1,194 s build.

So the order-aware import wasn't built. The export option stays on the branch,
and the file records its order (header flag bit 0), in case the
huge-pages TLB effect is ever worth testing.

### Where Ginger's speed comes from (ablation)

`bench/cloudlab/hn_ginger_ablation.sh` ran Ginger's recommended configuration
with one piece turned off at a time, at 1 and 40 threads on the client node,
with the full beam sweep and 60 s per point. Huge pages come from
`hn_ginger_thp.sh` (THP against its own 4 KB control run). The values are QPS
relative to the recommended configuration at the same recall, interpolated as
elsewhere. The dashboard's "Ginger ablation" tab has the curves.

| piece turned off | 1 thread, 0.95 / 0.97 / 0.99 | 40 threads, 0.95 / 0.97 / 0.99 |
|---|---|---|
| 4-bit traversal (float traversal, no re-rank) | 0.96× / 0.96× / 0.86× | **0.46× / 0.46× / 0.43×** |
| re-rank (4-bit distances only) | recall stops at 0.922 | recall stops at 0.922 |
| 12-bit re-rank, replaced by float re-rank | 1.08× / 1.07× / 1.05× | 1.06× / 1.06× / 1.05× |
| graph compression | 1.07× / 1.07× / 1.02× | 1.04× / 1.03× / 1.00× |
| reordering (and compression), against compression off | 0.90× / 0.89× / 0.87× | 0.92× / 0.91× / 0.88× |
| huge pages (THP, against its 4 KB control) | 0.81× / 0.82× / 0.83× | 0.81× / 0.82× / 0.82× |

At 0.95, the recommended configuration runs at 1,382 QPS with 1 thread and
36,692 with 40.

- **4-bit traversal is the large piece, and only under load:** 2.2× at 40
  threads, 1.04× at 1. One thread waits on memory latency, so smaller codes
  barely help it. Forty threads are limited by bandwidth, and the codes are
  8× smaller (192 B against 1,536 B per vector).
- **Re-rank is required.** Without it recall stops at 0.92, whatever the beam.
- **Floats re-rank 5–8% faster than 12-bit codes,** at the same recall. The
  12-bit codes save memory (576 B against 1,536 B per vector), not time.
- **Compression costs a little**, 1.00–1.07×, since decoding takes CPU and the
  graph fits in memory either way.
- **Reordering is worth 1.08–1.15×**, more than the offline page count above
  suggests (−1.7% index pages per hop). Ginger lays vectors out contiguously
  in node order, so its gain is cache lines and TLB reach, not 8 KB pages.
- **Huge pages are worth 1.22×** at both thread counts (previous subsection).
- **Caveats:**
  - The float-re-rank, no-quantization, no-compression and no-reorder arms
    each run on their own build, so a few percent of each gap can be build
    noise.
  - What Ginger can't turn off is not here: in-process execution, the
    contiguous code layout, batching.

**What this means for pgvectorfast:**
1. **4-bit traversal on the PiPNN graph** is the piece to adopt first. The
   `pgvectorfast-ntq` branch ports Ginger's codec and adds
   `quantizer = 'tq4'` indexes; `hn_pipnn_tq4.sh` measures it.
2. **Re-ranking from floats** would be as fast or faster, but needs the
   floats on separate pages, or a fetch of the heap row.
3. **Reordering deserves a second look with tq4.** A tq4 element and its
   neighbour tuple take 688 B, so 11 elements fit on a page instead of 4.
   That makes node order matter more for pages per hop.
4. **Huge pages** are already a setting (`huge_pages=on`).
5. **Compression** isn't worth porting.

## What the numbers say

**pgvector cannot reach 0.98 recall on this dataset at its default build
settings.**
- With m=16 and ef_construction=64, recall tops out at 0.979 at
  `ef_search=1000`, which is pgvector's maximum. pgvector_ntq's graph, built
  with the same settings, only just clears it (0.9802).
- At the same search width, Ginger's graph is far more accurate: at 160,
  pgvector reaches 0.933 and Ginger 0.985.
- On DBpedia at 1M, the same settings reached 0.993. At 27× the size, the
  graph is the limit. A stronger build (higher ef_construction) is the
  obvious next test; it is not run here.

**Ginger is about 9–19× stock pgvector and 3–7× pgvectorfast at one thread,
across recall 0.90–0.97. At 40 threads it is 11–23× and 5–11×.**
- The gap widens with recall, because pgvector's curve falls off as it
  approaches its ceiling.
- The Postgres boundary does not explain it. A trivial prepared query
  carrying the same vector costs 0.114 ms at 1 client (the `rtt` row: 8,745
  QPS). Adding that to Ginger's 0.70 ms at 0.95 still leaves it 3.9×
  pgvectorfast and 11× stock.
- From 10 clients up, the round-trip floor (54k / 87k / 115k QPS at
  10 / 20 / 40) is far above every Postgres result.

**pgvectorfast is 2.7–2.9× stock at one client, and still about 2.1× at 40
clients.**
- Both arms return identical recall at every `ef_search`. They search the same
  graph, so the difference is purely in the buffer manager.
- The lead is larger than on DBpedia, where it was 1.45–2.0×.
- On DBpedia, pgvectorfast stopped scaling past 8 clients, attributed there to
  DRAM bandwidth. Here it keeps scaling to 40. A plausible reason is that a
  384-d element is 1.5 KB instead of 6 KB, so bandwidth runs out much later.
  That is not measured here.

**Postgres scaling is uneven between 10 and 40 clients; this is not profiled.**
At recall 0.95:

| clients | stock | fast |
|---|---:|---:|
| 10 → 20 | +46% | +12% |
| 20 → 40 | +75% | +63% |

Twenty clients is one backend per physical core, so the small step for
pgvectorfast from 10 to 20 is unexpected. Two candidate causes are where the
scheduler places backends across the two sockets, and interleaved memory under
cross-socket traffic.

**Ginger keeps scaling.** Its throughput grows 9.8× from 1 to 10 threads, 1.9×
from 10 to 20, and 1.35× from 20 to 40, where the extra threads are
hyperthreads.

**Ginger's two configs search the same graph at 384-d.**
- `RecommendGingerConfig` applies no PCA at this dimension. The resolved
  configs differ only in the re-rank: 4-bit navigation with 8 extra
  TurboQuant bits (floats dropped) versus 4-bit navigation with an exact float
  rerank.
- So they run at the same speed. The float rerank is 3–7% faster, with
  marginally higher recall.
- The recommended config's index is 2.7× smaller (17.6 GB vs 46.9 GB).
- That differs from DBpedia, where the recommended config's PCA made it
  1.5–1.7× faster.

## pgvector_ntq (tq4 + ntq12)

`pgvector_ntq` (`awitten1/pgvector`, branch `hnsw-tq4-nav`, built
`-march=native`) stores 4-bit TurboQuant navigation codes in the HNSW graph
instead of float vectors, and re-ranks the final candidates from a separate
ntq12 page chain. This arm uses
`WITH (m = 16, ef_construction = 64, quantizer = 'tq4', rerank = 'ntq12')`.
- It runs on its own copy of the table, `items_hn_tq4`, so the planner sees one
  HNSW index per table and each arm loads only its own relations.
- Autovacuum is off on that table, because the stock and fast libraries do not
  understand the format: both write the same `HNSW_VERSION` (1).
- The protocol matches the other Postgres arms: two 1-client rounds on the
  server node, then one round each at 10/20/40 clients over the LAN, with
  everything prewarmed and resident and the interleave checked.

**It runs at stock pgvector's speed, with an index less than half the size.**

| clients | vs stock at 0.90 / 0.95 / 0.97 | vs pgvectorfast at 0.90 / 0.95 / 0.97 |
|---:|---|---|
| 1 | 0.98× / 1.00× / 0.95× | 0.36× / 0.36× / 0.33× |
| 10 | 1.04× / 1.10× / 1.06× | 0.39× / 0.38× / 0.34× |
| 20 | 1.02× / 1.09× / 1.04× | 0.47× / 0.49× / 0.45× |
| 40 | 1.02× / 1.08× / 1.04× | 0.48× / 0.52× / 0.49× |

- **Index:** 24 GB against 52 GB (0.46×). The build took 4,586 s, 12% longer
  than the float index's 4,105 s.
- **Recall:** at the same `ef_search` it is marginally higher than the float
  graph (0.9366 vs 0.9326 at 160; 0.9802 vs 0.979 at 1000), so the 12-bit
  rerank costs no accuracy. That takes it just past 0.98 at `ef_search=1000`,
  at 31 QPS with 1 client and 693 with 40.
- **Speed:** at 1 client it is indistinguishable from stock. From 10 clients
  up it is 2–10% ahead, about the size of the single-round noise. It is
  2–3× slower than pgvectorfast at every client count.
- **Why:** see the next section. In short, the cost of each visit is the same
  for both formats, and the 4-bit distance kernel is slower than the float one
  it replaces.

## Why tq4 gives no speedup here

Profiled with `bench/cloudlab/hn_profile_tq4.sh`. Each arm was served in turn
(prewarmed, residency and interleave checked), at `ef_search=160`, where stock
recall is 0.933 and tq4 recall is 0.937.
- **Load:** the client ran on the client node for 130 s at 1 and at 40
  clients. Meanwhile `perf` sampled the whole server in four 20 s windows:
  counters with IMC DRAM traffic, Top-Down, call stacks (LBR, so the
  benchmarked binaries were used unchanged), and TLB walks.
- **Perf overhead:** throughput was within 1% of the unprofiled runs.
- **Buffer accesses:** counted from `EXPLAIN (ANALYZE, BUFFERS)` over 100
  queries.
- **Kernels:** with Postgres stopped, the two distance loops were timed
  outside Postgres (`tq_kernel_bench.c`).
- **Outputs:** `bench/profiles/hn-tq4/`, including `summary.md`, `buckets.md`,
  `explain-*.txt`, `kernel.txt` and `disasm*.txt`.

**Per query at 1 client:**

| | stock | pgvector_ntq | pgvectorfast |
|---|---:|---:|---:|
| QPS | 167 | 152 | 455 |
| server CPU per query | 5.95 ms | 6.62 ms | 2.15 ms |
| shared-buffer accesses per query (EXPLAIN) | 3,291 | 3,630 | (same graph as stock) |
| **CPU per buffer access** | **1.81 µs** | **1.82 µs** | 0.65 µs |
| DRAM read per query | 9.4 MB | 3.4 MB | 8.4 MB |
| dTLB walks per query; share of cycles walking | 18.8k; 15.7% | 19.9k; 10.9% | 12.6k; 24.0% |
| IPC; Top-Down memory-bound | 0.42; 70% | 0.51; 59% | 0.45; 69% |

**Where the CPU time goes** (self time, µs per query, 1 client; the buckets
cover 96% of samples):

| | stock | pgvector_ntq | change |
|---|---:|---:|---:|
| buffer manager (hash lookup, pin, header lock, LWLocks) | 2,732 | 2,304 | −428 |
| HNSW traversal (mostly `HnswLoadElementImpl`) | 1,763 | 1,710 | −53 |
| distance kernel | 936 | 1,610 | **+674** |
| ntq rerank (`HnswTqRerank`) | – | 517 | **+517** |
| everything else | 520 | 478 | −42 |
| **total** | **5,951** | **6,619** | **+668** |

**Distance kernels outside Postgres** (384-d, one thread, ns per distance;
random rows out of 8M):

| | in cache | random, 4 KB pages | dependent chain, 4 KB | random, huge pages |
|---|---:|---:|---:|---:|
| float L2 (pgvector) | 68 | 492 | 585 | 312 |
| tq4 `TqNavDot` | 279 | 599 | 588 | 565 |

What this shows:

1. **Shrinking the tuples did reduce memory traffic, but bandwidth was not
   the bottleneck.** The workload is memory-bound in the latency sense (Top-Down:
   70% of cycles stalled on loads), not in the bandwidth sense.
   - DRAM read per query fell 64%, from 9.4 to 3.4 MB.
   - Even stock reads only 1.6 GB/s at 1 client and 34 GB/s at 40, far below
     what two sockets of DDR4 deliver.
   - The extra 21 cache lines per float vector are contiguous, so the
     hardware prefetcher streams them alongside the first line. They were
     cheap to begin with.
2. **Each visit costs the same, and the bytes are not where that cost is.**
   - CPU per buffer access is 1.81 µs for stock and 1.82 µs for ntq.
   - About 46% of stock's time is the buffer manager: the `ReadBuffer` hash
     lookup (`hash_search_with_hash_value` alone is 26%) plus the pin. A
     further 23% is `HnswLoadElementImpl`'s first touch of the tuple.
   - Those are dependent cache and TLB misses on the buffer-mapping hash table,
     the buffer descriptor, the page header and the tuple header. Every visit
     pays them whatever the tuple size: both arms take about 19–20k TLB walks
     per query, roughly 5.7 per buffer access.
   - The smaller index made each walk cheaper (113 vs 155 cycles) and trimmed
     the buffer-manager time, but not the number of misses.
3. **The 4-bit kernel is slower than the float kernel it replaces.**
   - GCC compiles `TqNavDot`'s per-nibble table lookup into AVX2 gathers (16
     `vpgatherdd` in `HnswTqCodeDistance`), and gathers are slow on Haswell.
   - In cache it is 4.1× slower than pgvector's float L2 (279 vs 68 ns).
   - Out of cache its compute costs about as much as the miss it saves. So
     even with no Postgres in the way it never beats float L2 on this CPU: it
     ties on a dependent chain and loses everywhere else.
   - In Postgres the distance bucket grows from 0.94 to 1.61 ms per query.
4. **The rerank adds about as much again.**
   - It does 2 buffer accesses for each of the `ef_search` candidates. That is
     the 10% more accesses EXPLAIN shows (3,630 vs 3,291, ≈ 2 × 160).
   - Its scalar bit-unpacking 12-bit dot product (`TqRefineDot`, inlined)
     brings it to 0.52 ms per query, about 3.2 µs per candidate.
5. **The graph is not the cause.** At the same `ef_search`, ntq's recall is
   slightly higher and it touches the same number of graph pages, once the
   rerank's are excluded.

**Net:**
- The −0.43 ms in the buffer manager is outweighed by +0.67 ms in the distance
  kernel and +0.52 ms in the rerank, so ntq costs 11% more CPU per query at the
  same `ef_search`.
- Its slightly better recall per `ef_search` buys that back, which is why it
  matches stock at matched recall.
- At 40 clients the picture is the same: CPU per query is 10.07 ms for stock
  and 10.38 ms for ntq, and stock's 34 GB/s is not a bandwidth wall that tq4
  could relieve.

**Confound:**
- pgvector_ntq branches from pgvector master as of 2026-11-18, not 0.8.7.
  Master's PG18 HNSW prefetch path (`PrefetchBuffer` + `ReadRecentBuffer`)
  shows up in its profile as `LockBufHdr` in place of `PinBuffer`.
- Part of the −0.43 ms in the buffer manager may come from that, not from tq4.
  It does not change the conclusion.

**What would make tq4 pay:**
- **Register-resident lookup table.** A shuffle-based kernel would replace the
  gathers: a `vpshufb` / `vpermps` lookup table, or an int8 query table as in
  FAISS fast-scan and Ginger. Estimate: it would bring the distance bucket
  under stock's, for about 1.15–1.2× over stock at 1 client.
  - The ceiling is low. The ~3.6 ms per query of buffer-manager work and
    first-touch misses (55–60%) does not depend on tuple size.
- **Rerank fewer candidates.** For example, only the best few × k instead of
  all `ef_search`, and vectorize `TqRefineDot`.
- **Put tq4 on pgvectorfast's access path.** That removes the per-access cost
  entirely, leaving vector bytes and distance compute a much larger share of
  what remains.
  - pgvectorfast's 0.65 µs per access sits in its traversal functions
    (`HnswLoadUnvisitedRaw`, `HnswGetDistanceOptimistic`).
  - There, smaller tuples and a fast kernel should show up.
  - Measured since: 2.4–2.9× over pgvector_ntq, and 1.2–1.3× over float
    pgvectorfast at 20–40 clients (section "pgvectorfast + tq4").
- **Huge pages for `shared_buffers`.** Stock spends 15.7% of cycles walking
  page tables at 1 client and 25.8% at 40. Measured since: huge pages make
  stock 1.3–1.4× faster (section "Huge pages for `shared_buffers`").
  pgvectorfast + tq4 gains 1.34–1.36× at 1 client and 1.20–1.31× at 40.
  Float pgvectorfast gains 1.28–1.31× at 1 client but nothing at 40.

## pgvectorfast + tq4 (ntq12)

pgvectorfast with pgvector_ntq's TurboQuant read path ported in. It lives on
branch `tq4-read` of the pgvectorfast submodule, checked out as the worktree
`pgvectorfast_ntq/`, and is built by `build_fastntq.sh`. It serves the same
24 GB `items_hn_tq4` index that pgvector_ntq built, with no rebuild. It is
measured by `bench/cloudlab/hn_fastntq.sh`, with the same protocol as the
other Postgres arms.

**What changed in pgvectorfast** (8 files, about 270 lines, plus pgvector_ntq's
`hnswtq.c` and its tables):
- **Search:** the lock-free path scores a 4-bit code in place
  (`HnswTqCodeDistanceOptimistic`) instead of a float vector. It checks the
  code's marker, padded dimensions and length before reading, then validates
  the frame version afterwards, as it does for vectors.
- **Query and fallbacks:** the query is rotated once per scan. The locked
  fallback paths and upper layers use pgvector_ntq's code distance.
- **Re-rank:** the final candidates are re-ranked from the ntq12 chain with
  pgvector_ntq's arithmetic, through Calico's buffer path
  (`HnswTqRerankFast`), and then re-sorted.
- **Read-only:** building, inserting into or vacuuming a tq4 index raises an
  error in this build.
- **Correctness check:** recall equals pgvector_ntq's at every `ef_search`
  (worst difference 0.0001, from tie order), on 1M and 27.4M rows.
  `compare_recall.py` enforces this before any multi-client measurement.

**QPS at matched recall:**

| clients | 0.90 | 0.95 | 0.97 | 0.98 | vs pgvectorfast (floats) | vs pgvector_ntq |
|---:|---:|---:|---:|---:|---|---|
| 1 | 635 | 298 | 140 | 83 | 0.88–0.95× | 2.5–2.6× |
| 10 | 6,088 | 2,792 | 1,300 | 771 | 0.97–1.05× | 2.7–2.9× |
| 20 | 8,146 | 3,668 | 1,690 | 997 | 1.16–1.24× | 2.5–2.6× |
| 40 | 12,849 | 6,126 | 3,074 | 1,892 | 1.18–1.27× | 2.4–2.6× |

The two 1-client rounds agree within 1%. Unlike the float graph, it reaches
0.98 at every client count.

**Profile, `ef_search=160`:**

| | pgvectorfast (floats), 1 client | **+ tq4, 1 client** | pgvectorfast (floats), 40 clients | **+ tq4, 40 clients** |
|---|---:|---:|---:|---:|
| QPS (profiled run) | 455 | 392 | 6,805 | 7,724 |
| server CPU per query | 2.15 ms | 2.50 ms | 4.23 ms | 3.64 ms |
| DRAM read per query | 8.4 MB | 2.4 MB | 8.6 MB | 2.5 MB |
| DRAM read bandwidth | 3.8 GB/s | 0.9 GB/s | 58.7 GB/s | 19.6 GB/s |
| IPC | 0.45 | 0.80 | 0.22 | 0.58 |
| Top-Down memory-bound | 69% | 34% | 52% | 18% |
| Top-Down retiring | 12% | 39% | 9% | 45% |

Where the combination's CPU time goes, in µs per query at 1 client (total
2,500):

| | µs | share |
|---|---:|---:|
| `TqNavDot`, the 4-bit navigation distance | 930 | 37% |
| traversal (`HnswLoadUnvisitedRaw`, `HnswSearchLayerDiskFastCore`) | 771 | 31% |
| `TqRefineDot`, the 12-bit re-rank dot product | 449 | 18% |
| re-rank buffer reads and sort | about 50 | 2% |
| Calico pins and lookups | about 100 | 4% |
| everything else | about 200 | 8% |

What this shows:

- **pgvectorfast's access path is what tq4 was missing.** On it, tq4 runs
  2.4–2.9× faster than on stock pgvector's buffer manager. The per-access
  cost that hid the smaller tuples is gone.
- **The bottleneck has moved from memory to compute.**
  - Memory-bound falls from 69% to 34% of cycles, and IPC nearly doubles,
    from 0.45 to 0.80.
  - The two tq4 kernels now take 55% of the CPU time. `TqNavDot` is the
    gather-based 4-bit kernel the microbenchmark measured at 4.1× slower
    than float L2; `TqRefineDot` is the scalar bit-unpacking re-rank.
  - At 1 client this extra compute costs more than the memory it saves, so
    the combination is 5–12% behind float pgvectorfast.
- **Under load the memory saving wins.**
  - Float pgvectorfast reads 59 GB/s at 40 clients and its IPC drops to 0.22.
  - The combination reads 20 GB/s, with IPC 0.58 and memory-bound 18%.
  - That is 1.2–1.3× more throughput at 20–40 clients.
- **Next step: vectorize the two kernels.** Done; see the next section.

## pgvectorfast + tq4 with optimized kernels, and with huge pages

The two tq4 kernels, rewritten in `pgvectorfast_ntq/src/hnswtq_simd.h` and
called from `hnswtq.c`. Same index and the same protocol. Measured by
`bench/cloudlab/hn_fastntq_k2.sh`, first on 4 KB pages, then with
`huge_pages=on` (40 LAN clients only).

**The kernels:**
- **`TqNavDot`** (4-bit navigation). The compiler had turned the scalar
  lookup into AVX2 gathers, which are slow on Haswell. The new loop keeps the
  8 positive centroids in one ymm register. It splits 8 nibbles into lanes
  with one `vpsrlvd`, picks each centroid with `vpermps` (which reads only the
  low 3 bits), and uses nibble bit 3 as the float sign.
- **`TqRefineDot`** (ntq12 re-rank). The generic loop re-derived a k-bit
  offset and chose the table for every coordinate. The new loop is
  specialized for k = 8, where the index is a byte, and k = 6, where four
  indices pack into three bytes. It uses four accumulators.
- **Checks.**
  - The disassembly of the built library has 6 `vpermps` and no gathers in
    the tq code (`bench/profiles/hn-fastntq-k2/disasm.txt`).
  - Recall equals pgvector_ntq's at every `ef_search`, on both page sizes
    (worst difference 0.0001).

**Microbenchmark** (`bench/cloudlab/tq_kernel_opt_bench.c` on the server,
384 dims, in cache, same compiler flags; the original kernels are copied
verbatim). All are within 1.6e-7 of a double-precision reference at 384, 1536,
120, 13 and 1 dimensions, so they differ only in summation order.

| kernel | original | optimized | speedup |
|---|---:|---:|---:|
| `TqNavDot` | 298 ns | 100 ns | 3.0× |
| `TqRefineDot`, k = 8 (ntq12, used here) | 2,613 ns | 652 ns | 4.0× |
| `TqRefineDot`, k = 6 (ntq10) | 2,794 ns | 832 ns | 3.4× |

**QPS at matched recall:**

| clients | variant | 0.90 | 0.95 | 0.97 | 0.98 | vs original kernels | vs pgvectorfast (floats), same page size |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | optimized kernels | 1,003 | 483 | 233 | 140 | 1.58–1.69× | 1.45–1.53× |
| 1 | + huge pages | 1,361 | 653 | 314 | 187 | 2.14–2.25× | 1.50–1.59× |
| 10 | optimized kernels | 8,847 | 4,368 | 2,084 | 1,233 | 1.45–1.60× | 1.53–1.65× |
| 20 | optimized kernels | 12,031 | 5,589 | 2,597 | 1,534 | 1.48–1.54× | 1.72–1.88× |
| 40 | optimized kernels | 16,953 | 8,970 | 4,453 | 2,748 | 1.32–1.46× | 1.56–1.85× |
| 40 | + huge pages | 20,350 | 11,550 | 5,831 | 3,597 | 1.58–1.90× | 1.79–2.41× |

- **Against stock pgvector:** 5.3–5.8× at 1 client and 3.9–5.1× at
  40 clients.
- **Against Ginger:** Ginger is still 1.7–4.3× faster at 1 thread and
  2.9–5.7× at 40, widening with recall.
- **Repeatability:** the two 1-client rounds agree within 1.7% on 4 KB pages
  and within 2.8% on huge pages.
- **The huge-page rows compare against float pgvectorfast on huge pages**
  (measured in the section "Huge pages for float pgvectorfast"). Floats stop
  at recall 0.979 with `ef_search` 1000, so the ratios cover 0.90–0.97.

**Profiles at `ef_search=160`:**

| | original kernels | optimized | optimized + huge pages |
|---|---:|---:|---:|
| QPS, 1 client (profiled run) | 392 | 615 | 805 |
| server CPU per query, 1 client | 2.51 ms | 1.59 ms | 1.19 ms |
| — tq4 navigation distance | 975 µs | 289 µs | 284 µs |
| — traversal (`HnswLoadUnvisitedRaw`, `HnswSearchLayerDiskFastCore`) | 771 µs | 855 µs | 489 µs |
| — re-rank (`TqRefineDot`, buffer reads, sort) | 496 µs | 175 µs | 174 µs |
| IPC, 1 client | 0.80 | 1.10 | 1.41 |
| dTLB walk cycles, 1 client | 17.6% | 29.2% | 11.7% |
| dTLB walks per query, 1 client | 13,000 | 12,777 | 6,936 |
| QPS, 40 clients (profiled run) | 7,724 | 11,240 | 14,198 |
| server CPU per query, 40 clients | 3.64 ms | 2.38 ms | 1.82 ms |
| IPC, 40 clients | 0.58 | 0.79 | 1.03 |
| DRAM read, 40 clients | 19.6 GB/s | 28.8 GB/s | 32.3 GB/s |

What this shows:

- **The kernels delivered what the microbenchmark predicted.**
  - Navigation distance fell 3.4× per query, and the re-rank 2.8×.
  - Total CPU per query fell from 2.51 to 1.59 ms. The projection from the
    microbenchmark was about 1.5 ms.
- **This made the walk TLB-bound, and huge pages pay off.**
  - Once the kernels were cheaper, page walks rose to 29% of cycles; the
    number of walks did not change.
  - With huge pages, walks per query halve and traversal time falls by 43%.
  - Of the ~6,900 walks per query that remain, about half are in Calico's
    translation arrays, which are on 4 KB pages, and half in the buffer
    pool itself, on 2 MB pages (measured since: section "Where the
    remaining page walks come from"). An earlier version of this note
    guessed backend-local memory; that was wrong. Putting the translation
    arrays on 2 MB pages removes most of the first half, for 2–4% more QPS
    at 1 client and 4–7% at 40 (section "Calico's translation arrays on
    2 MB pages").
- **Navigation distance now runs at in-cache kernel speed.** 284 µs over
  about 3,600 buffer accesses per query (EXPLAIN on the tq4 index) is
  roughly 80–100 ns per call, the microbenchmark's figure.
- **Where 1.19 ms per query now goes** (1 client, huge pages):
  - traversal 41%, mostly loading neighbour lists and probing the visited
    set
  - navigation distance 24%
  - re-rank 15%
  - Calico and buffer pins 12%

**Huge-page check.** After each server start and at the end of the run:
- `huge_pages_status` was on.
- `shared_buffers` was one hugetlb mapping with `KernelPageSize` 2048 kB.
- 34,683 of 59,175 reserved 2 MB pages were in use (67 GB: the tq4 table
  plus its index).
- The mapping kept its interleave policy across both nodes.

## Huge pages for `shared_buffers` (stock pgvector)

Stock pgvector re-measured with `huge_pages=on`, run by
`bench/cloudlab/hn_hugepages.sh`. Everything else matched the original stock
arm: same index, settings and protocol, two 1-client rounds and one round each
at 10/20/40 clients. The perf profiles at `ef_search=160` used the same windows
as above.

**Setup and verification.**
- **Reservation:** 59,175 pages of 2 MB (115 GB) were reserved, split evenly
  across the two NUMA nodes. Postgres needs 58,574 for these settings.
- **Checks.** These ran after the 1-client rounds, after the LAN restart, and
  again at the end. Any failure stops the run. Each requires:
  - `huge_pages_status = on`;
  - that the postmaster's shared-memory mapping, which every backend
    inherits, is a single 114 GB hugetlb mapping with 2 MB kernel pages;
  - that 96 GB of the pool is actually faulted in (the prewarmed 94 GB of
    table and index);
  - that the mapping is still interleaved across both nodes.
- **Release:** the pool was released afterwards (`vm.nr_hugepages=0`).

**QPS at matched recall:**

| clients | 4 KB pages (0.90 / 0.95 / 0.97) | huge pages | gain |
|---:|---|---|---|
| 1 | 257 / 113 / 55 | 341 / 146 / 71 | 1.29–1.32× |
| 10 | 2,154 / 908 / 429 | 2,981 / 1,263 / 607 | 1.38–1.42× |
| 20 | 3,242 / 1,328 / 627 | 4,369 / 1,806 / 858 | 1.35–1.37× |
| 40 | 5,197 / 2,330 / 1,144 | 6,644 / 3,030 / 1,510 | 1.28–1.32× |

- Recall is identical at every `ef_search`, and the two 1-client rounds
  agree within 1%.
- pgvectorfast stays ahead of stock on huge pages: 2.0–2.2× at 1 client and
  1.6× at 40 on 4 KB pages, and 2.6–2.9× at 1 client and 1.6–1.7× at 40
  with both on huge pages (next section).

**Profile, `ef_search=160`:**

| | 1 client, 4 KB | 1 client, huge | 40 clients, 4 KB | 40 clients, huge |
|---|---:|---:|---:|---:|
| QPS (profiled run, client on LAN) | 167 | 211 | 3,302 | 4,216 |
| server CPU per query | 5.95 ms | 4.71 ms | 10.07 ms | 7.67 ms |
| IPC | 0.42 | 0.51 | 0.26 | 0.33 |
| cycles walking page tables | 15.7% | 2.5% | 25.8% | 7.4% |
| dTLB walks per query | 18.8k | 5.3k | 24.5k | 7.9k |
| cycles per walk | 155 | 69 | 251 | 172 |
| Top-Down memory-bound | 70% | 67% | 74% | 67% |
| DRAM read | 1.6 GB/s | 1.9 GB/s | 34 GB/s | 40 GB/s |

Where the CPU time went, in µs per query:

| | 1 client, 4 KB | 1 client, huge | 40 clients, 4 KB | 40 clients, huge |
|---|---:|---:|---:|---:|
| buffer manager | 2,732 | 2,078 | 4,770 | 3,002 |
| HNSW traversal (tuple first touch) | 1,763 | 1,308 | 2,730 | 2,139 |
| distance kernel | 936 | 851 | 1,722 | 1,723 |
| other | 520 | 489 | 852 | 805 |

What this shows:

- **Page-table walks were the single largest avoidable cost.**
  - Walk cycles fall from 15.7% to 2.5% at 1 client, and from 25.8% to 7.4%
    at 40.
  - Walks per query fall 3.5×, and each remaining walk is cheaper, because the
    page tables for 2 MB pages are 512× smaller and stay cached.
  - The time comes out of the buffer manager and the tuple first touch (−1.1
    ms per query at 1 client), the two places that dereference
    `shared_buffers` addresses.
  - The distance kernel barely moves: it reads a contiguous run of 24 cache
    lines inside one page.
- **The workload stays latency-bound.**
  - Top-Down still shows two thirds of cycles stalled on memory.
  - DRAM bandwidth is still far from any limit.
  - IPC rises from 0.42 to 0.51.
  - Huge pages shorten each dependent miss chain; they do not remove it.
- **This explains part of the slowdown under load, not all of it.**
  - With huge pages, CPU per query still rises 1.6× from 1 to 40 clients
    (4.71 to 7.67 ms).
  - The distance kernel, which touches no shared state, doubles from 0.85 to
    1.72 ms.
  - That remainder points at hyperthread sharing (40 threads on 20 cores),
    L3 contention and cross-socket misses rather than page walks. This is
    inferred, not separately measured.
- **The gain is largest at 10 clients (1.4×).** That is where the uneven 10→20
  step was, so part of that anomaly may be TLB pressure. Not profiled at
  10 or 20 clients.

## Huge pages for float pgvectorfast, and tq4 against floats like for like

Float pgvectorfast re-measured with `huge_pages=on`, run by
`bench/cloudlab/hn_fast_hp.sh`: same float index, settings and protocol, two
1-client rounds (client on the server, as for every 1-client row), one round
at 40 LAN clients, and profiles at `ef_search=160`. It was run to give
pgvectorfast + tq4 on huge pages a baseline on the same page size.

**Huge-page check.** After the 1-client rounds, after the LAN restart and at
the end, the same checks as for stock passed: `huge_pages_status` on, one
114 GB hugetlb mapping with 2 MB kernel pages, 49,381 of 59,175 pages in use
(96 GB: the float table plus its index), interleave kept. The pool was
released afterwards. Recall equals the 4 KB rounds at every `ef_search`, and
the two 1-client rounds agree within 2.7%.

**QPS at matched recall (0.90 / 0.95 / 0.97):**

| clients | pgvectorfast, 4 KB | pgvectorfast, huge pages | huge-page gain | pgvectorfast + tq4 (optimized), huge pages | tq4 vs floats, both on huge pages |
|---:|---|---|---|---|---|
| 1 | 692 / 315 / 159 | 885 / 410 / 209 | 1.28–1.31× | 1,361 / 653 / 314 | 1.50–1.59× |
| 40 | 10,884 / 4,853 / 2,426 | 11,338 / 4,870 / 2,416 | 1.00–1.04× | 20,350 / 11,550 / 5,831 | 1.79–2.41× |

tq4 against floats, on the same page size:

| clients | 4 KB pages | huge pages |
|---:|---|---|
| 1 | 1.45–1.53× | 1.50–1.59× |
| 40 | 1.56–1.85× | 1.79–2.41× |

**Profile, `ef_search=160`:**

| | 1 client, 4 KB | 1 client, huge | 40 clients, 4 KB | 40 clients, huge |
|---|---:|---:|---:|---:|
| QPS (profiled run) | 455 | 565 | 6,805 | 7,153 |
| server CPU per query | 2.15 ms | 1.73 ms | 4.23 ms | 3.99 ms |
| — `HnswLoadUnvisitedRaw` (neighbour lists, visited set) | 735 µs | 363 µs | 1,603 µs | 1,388 µs |
| — `HnswGetDistanceOptimistic` (float distance) | 725 µs | 724 µs | 1,212 µs | 1,106 µs |
| — `HnswSearchLayerDiskFastCore` | 543 µs | 502 µs | 1,002 µs | 1,043 µs |
| IPC | 0.45 | 0.52 | 0.22 | 0.24 |
| cycles walking page tables | 24.0% | 9.5% | 28.2% | 22.3% |
| dTLB walks per query | 12.6k | 6.9k | 14.9k | 10.6k |
| cycles per walk | 128 | 75 | 198 | 207 |
| DRAM read | 3.8 GB/s | 4.5 GB/s | 58.7 GB/s | 61.6 GB/s |
| DRAM read per query | 8.4 MB | 7.9 MB | 8.6 MB | 8.6 MB |

For comparison, pgvectorfast + tq4 (optimized kernels, huge pages) at 40
clients: 1.82 ms CPU per query, 2.3 MB per query, 32 GB/s, 94 cycles per
walk, 17.4% walk cycles.

What this shows:

- **At 1 client, floats gain about as much as tq4 does** (1.28–1.31× against
  1.34–1.36×), so tq4's lead over floats barely changes: 1.45–1.53× on 4 KB
  pages, 1.50–1.59× on huge pages.
  - Huge pages halve the neighbour-list and visited-set time (735 → 363 µs),
    the chain of dependent misses.
  - The float distance does not move (725 µs). It reads 24 contiguous cache
    lines inside one page, so it is bound by the misses themselves, not by
    the TLB.
- **At 40 clients floats gain nothing**, and tq4's lead widens to
  1.79–2.41×.
  - CPU per query falls only 6%, and walk cycles only from 28.2% to 22.3%.
  - Each walk got no cheaper (198 → 207 cycles), unlike at 1 client
    (128 → 75). The likely reason: at 62 GB/s of float reads, the L3 no
    longer holds the page-table entries, so a 2 MB walk still goes to DRAM.
    tq4 reads a quarter of the bytes, and its walks cost 94 cycles. This is
    inferred from the counters, not measured directly.
  - 61.6 GB/s is the highest read bandwidth seen in any run here, about 45%
    of the two CPUs' rated peak (2 × 68 GB/s for the E5-2660 v3). Random 64-byte reads across two
    sockets cannot reach peak, but where the ceiling is on this machine was
    not measured.
- **So the 4 KB comparison understated tq4's lead under load, and the earlier
  "about 2×" figure on huge pages was roughly right at 40 clients for the
  wrong reason.** That figure compared tq4 on huge pages with floats on 4 KB
  pages. Like for like it is 1.5–1.6× at 1 client and 1.8–2.4× at 40.
- **Caveats.** Read-only workload. Floats reach recall 0.979 at most with this
  graph, so there is no 0.98 comparison. tq4's re-rank reads the ntq12 codes,
  not the floats.

## Where the remaining page walks come from (pgvectorfast + tq4, huge pages)

pgvectorfast + tq4 (optimized kernels, `huge_pages=on`) still walked page
tables ~6,900 times per query at 1 client, 11.7% of cycles. Measured by
`bench/cloudlab/hn_tlb_diag.sh` at `ef_search=160`, 1 and 40 LAN clients,
same index and serve setup (QPS in the profiled windows: 813 and 14,365).
`bench/cloudlab/tlb_window.sh` takes the counters in groups of four (meant
to avoid multiplexing; see the method notes below), then PEBS samples of retired loads and stores that
missed the second-level TLB (STLB), with their exact instructions.
`bench/cloudlab/tlb_attrib.py` writes `bench/profiles/hn-tlb/attrib.md`.

**Counters, per query:**

| | 1 client | 40 clients |
|---|---:|---:|
| load page walks started | 7,063 | 8,211 |
| — completed for a 4 KB page | 3,540 | 4,795 |
| — completed for a 2 MB page | 3,338 | 3,232 |
| retired loads (incl. software prefetches) that missed the STLB | 6,312 | 6,522 |
| cycles per walk | 59 | 92 |
| walk cycles, share of all cycles | 11.4% | 17.1% |
| walker's page-table reads served by L1 / L2 / L3 / DRAM, per walk | 0.40 / 0.23 / 0.68 / 0.02 | 0.48 / 0.21 / 0.48 / 0.05 |
| store page walks | 70 | 112 |

**The instructions that miss** (share of sampled STLB-missing loads; the
same at 1 and 40 clients, from disassembly of the served `vectorfast.so`
with its line table):

| instruction | source | what it touches | share |
|---|---|---|---:|
| `prefetcht0 (%rcx,%rdi,8)` | `calico_internal.h:608`, called from `hnswutils.c:1912` | the neighbour block's 8-byte entry in Calico's block → frame translation array | 42% |
| four `prefetcht0` and a loop tail | `calico_internal.h:770–776, 793` | the neighbour's line pointer inside its buffer page in `shared_buffers` | 44% |
| loads in `CalicoLookupAndPin`, `calico_active_LockBuffer` | | translation entries and frames on the pin path | 6% |
| loads in `HnswSearchLayerDiskFastCore` | | | 3% |

**Calico's memory layout** (from the backends' `/proc/<pid>/maps` and
`smaps`): besides the 114 GB hugetlb mapping, every backend maps 52
translation extents of 512 MB in `/dev/shm`, each a flat array of 8-byte
entries indexed by block number. `calico_pool.c:345,421` calls
`madvise(MADV_NOHUGEPAGE)` on them, and shmem THP is `never` on the node, so
they are on 4 KB pages. The tq4 index's ~3M blocks use about 24 MB of its
array: some 6,000 4 KB pages, against an STLB of 1,024 entries.

What this shows:

- **Each visited neighbour costs two page walks**, one per stage of
  Calico's batched lookup in `HnswLoadUnvisitedRaw`: the prefetch of its
  translation entry (4 KB page), then the prefetch of its tuple in the
  buffer page (2 MB page). The two counts match: 3,540 and 3,338 walks per
  query, against about 3,600 buffer accesses per query.
- **Huge pages for `shared_buffers` already took the walk count from
  12,800 to 6,900**, but the STLB covers about 2 GB with 2 MB pages,
  against 67 GB of resident tq4 table and index, so random tuple accesses
  still miss it.
- **The visited set and the rest of backend-local memory do not matter.**
  The backend heap is 1 MB and anonymous mappings 5 MB. The instructions
  behind 95% of the samples touch translation entries, frames or tuples,
  not the visited set. Store walks are 1–2% of load walks.
- **The walks are short.** Page-table entries come from cache (DRAM 0.02
  reads per walk at 1 client), and most walks start from software
  prefetches, which overlap other work. So the 11.4% (1 client) and 17.1%
  (40 clients) of cycles with a walk in progress are upper bounds on what
  removing them would save, not the saving.
- **The two fixes, one per half:**
  - **Translation arrays on 2 MB pages.** `MADV_HUGEPAGE` instead of
    `MADV_NOHUGEPAGE` on the extents, with `/dev/shm` mounted
    `huge=advise`. The index's 24 MB of entries would fit in 12 pages,
    which stay in the TLB, so ~3,500 walks per query should go. The cost
    is 2 MB fault granularity for sparse arrays, which is why Calico chose
    4 KB. Measured in the next section but one: most of those walks go,
    for 2–4% more QPS at 1 client and 4–7% at 40.
  - **1 GB pages for `shared_buffers`** (`huge_page_size=1GB`; the CPU has
    `pdpe1gb`). Haswell's TLBs hold few 1 GB translations, so walks may
    not disappear, but each would read only the top two levels. Whether
    that is a net gain has to be measured, and reserving ~115 GB of 1 GB
    pages at runtime may fail on a fragmented node.
- **Float pgvectorfast has the same structure** with twice the translation
  entries (52 GB index). That may be part of why huge pages did nothing for
  it at 40 clients; not measured.
- **Method note.** On this node (Haswell-EP, kernel 6.8, perf with
  `-d`), the data addresses of the PEBS load samples were invalid: 6
  distinct values over 26,730 samples. The store samples' addresses and
  all instruction addresses were valid. The attribution therefore rests on
  the instructions, the page-size split of the walk counters, and the
  memory maps, which agree.
- **Counter coverage.** Contrary to the intent above, perf did multiplex
  the counter windows. The four TLB events were counted 83% of each window
  (59–63% in group 2), and cycles 42–66%. The counts are perf's scaled
  estimates over a 10 s steady state, so they should be close. The
  comparisons in the next section use identical windows on both sides.
  Cause, established by the traversal run below:
  - The NMI watchdog holds the fixed cycles counter, so `cycles` takes a
    programmable one. That explains the 83% coverage.
  - Linux schedules the 0xD0–0xD3 load events exclusively across
    hyperthread siblings on Haswell, as an erratum workaround. That
    explains the ~60% in group 2.

## Calico's translation arrays on 2 MB pages (measured)

The first of the two fixes above, as a blanket change. A new server setting,
`calico.huge_translation`, is read only at server start
(`pgvectorfast_ntq/src/calico.c`). It is off by default, which keeps
`MADV_NOHUGEPAGE`. On, the two `madvise` calls in `calico_pool.c` ask for
`MADV_HUGEPAGE` instead. For user tmpfs mounts such as `/dev/shm`, that
takes effect only if the mount has `huge=advise`. The sysfs `shmem_enabled`
knob mentioned above does not govern them.

`bench/cloudlab/hn_calico_thp.sh` ran both arms with one library, in one
session, on the same tq4 index:
- **off:** setting off, `/dev/shm` with its default `huge=never`; the
  same as everything above.
- **on:** setting on, `/dev/shm` remounted `huge=advise`.

Both arms ran with `huge_pages=on` (67 GB of 2 MB hugetlb pages in use,
NUMA interleave), prewarmed. Each arm had sweeps at 1 and 40 LAN clients,
then the TLB windows of the previous section at `ef_search=160`. The off
arm ran first. Each arm ran once.

**Checks that each arm was what it claimed:**
- **Setting and mount:** the server's `SHOW calico.huge_translation`
  and `/dev/shm`'s mount options.
- **Huge pages allocated:**
  - off: `ShmemHugePages` 0;
  - on: 206 MB just after the start and 218 MB during the windows.
- **In the busiest backend:** 166 MB of the 52 Calico extents mapped with
  2 MB pages (`ShmemPmdMapped`).
- **Cost:** total shared memory in `/dev/shm` went from 72 MB to 228 MB.
  That is the 2 MB fault granularity on the sparse arrays.
- **Teardown:** `/dev/shm` went back to `huge=never` and the hugetlb pool
  was released.

**QPS, on / off.** Recall is identical in both arms at every `ef_search`, so
these are matched-recall ratios:

| ef_search | 10 | 20 | 40 | 80 | 160 | 320 | 640 | 1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recall@10 | 0.624 | 0.751 | 0.836 | 0.898 | 0.937 | 0.961 | 0.972 | 0.980 |
| 1 client | 1.037 | 1.023 | 1.032 | 1.029 | 1.018 | 1.030 | 1.022 | 1.021 |
| 40 clients | 1.065 | 1.071 | 1.061 | 1.041 | 1.046 | 1.058 | 1.042 | 1.040 |

At `ef_search=160`, QPS goes from 816 to 831 at 1 client and from 14,781 to
15,466 at 40.

**Counters, per query, `ef_search=160`** (profiled windows):

| | 1 client off | 1 client on | 40 clients off | 40 clients on |
|---|---:|---:|---:|---:|
| QPS in the window | 806 | 826 | 14,431 | 15,042 |
| load page walks started | 6,923 | 3,763 | 8,359 | 4,723 |
| — completed for a 4 KB page | 3,482 | 475 | 4,891 | 1,406 |
| — completed for a 2 MB page | 3,280 | 3,157 | 3,308 | 3,183 |
| retired loads that missed the STLB | 6,745 | 3,263 | 6,478 | 3,400 |
| translation-entry prefetch, share of those (PEBS) | 42% | <1.1% | 42% | <0.8% |
| walk cycles, share of all cycles | 11.7% | 6.9% | 17.2% | 11.3% |
| cycles per query (millions) | 3.66 | 3.58 | 4.50 | 4.24 |
| IPC | 1.40 | 1.45 | 1.03 | 1.08 |

What this shows:

- **The fix does what the diagnosis predicted.**
  - The 4 KB walks fall by 86% at 1 client and 71% at 40.
  - The translation-entry prefetch drops out of the STLB-missing
    instructions: from 42% of samples to below the top ten.
  - The 2 MB walks, in `shared_buffers`, are unchanged.
- **Some 4 KB walks remain:** 475 per query at 1 client and 1,406 at 40.
  - Backend-local memory and the kernel are both on 4 KB pages.
  - The busiest backend had 166 MB of extents on 2 MB pages, against
    218 MB of shmem huge pages system-wide, so it may still have touched
    some 4 KB parts of the extents.
  - Not attributed further.
- **The gain is 2–4% at 1 client and 4–7% at 40.**
  - Walk cycles fell by 5.0% (1 client) and 6.5% (40 clients) of all
    cycles. Cycles per query fell by 2.4% and 5.7%.
  - So at 1 client about half the removed walk time had overlapped other
    work, as expected for walks that start from software prefetches. At
    40 clients, where walks took longer (92 cycles each against 62), most
    of it was on the critical path.
- **How sure.** The 1-client gain is the size of the run-to-run noise
  (max 2.8% for this arm on huge pages). It has the same sign at all eight
  `ef_search` values and agrees with the drop in cycles per query. The
  40-client gain is beyond the noise. One run per arm, off before on.
- **Memory.** About 2 MB per touched translation array instead of 4 KB:
  156 MB more here, for 52 extents. In a database with thousands of
  relations, a blanket setting would cost proportionally more. That is why
  a production version would ask for 2 MB pages only on the populated
  ranges of large relations.
- **What is left.** The other half of the walks, ~3,200 per query on
  2 MB pages in `shared_buffers`. 1 GB pages are the remaining lever for
  them.

## What traversal is made of (pgvectorfast + tq4, huge pages, 2 MB translation arrays)

Traversal (`HnswLoadUnvisitedRaw` and `HnswSearchLayerDiskFastCore`) was
the largest share of query time, so it was profiled on its own. Measured
by `bench/cloudlab/hn_trav_prof.sh` at `ef_search=160`, 1 and 40 LAN
clients, with the better configuration from the previous section.
`bench/cloudlab/trav_window.sh` collected, in steady state:
- **Counters**, five groups of four. The NMI watchdog was off for these
  windows and restored afterwards; groups 3–5 had 100% coverage.
- **Precise cycle samples** (`cycles:pp`). Haswell's PEBS records the
  instruction that was holding up retirement.
- **PEBS load-latency samples:** loads over 50 cycles, with their latency
  and where they were served from.
- **Function-entry counts**, from uprobes.

`bench/cloudlab/trav_attrib.py` maps the samples to source lines with the
served library's line table. QPS in the windows was 843 and 15,235
(`xact_commit`); the probes are in `bench/profiles/hn-trav/attrib.md`.

**Where a query's cycles go** (precise cycle samples, postgres processes):

| | 1 client | 40 clients |
|---|---:|---:|
| cycles per query | 2.70 M | 4.36 M |
| traversal | 38% (399 µs) | 38% (630 µs) |
| tq4 navigation distance | 25% (260 µs) | 25% (414 µs) |
| re-rank | 15% (156 µs) | 14% (234 µs) |
| Calico pin, lock, lookup | 7.5% | 6.2% |
| Postgres executor, protocol | 6.9% | 7.8% |
| kernel | 3.5% | 5.9% |

**Calls per query** (uprobes, the same at 1 and 40 clients):
- 177.8 hops (calls of `HnswLoadUnvisitedRaw`);
- one upper-layer search and one re-rank;
- 330 calls of `CalicoLookupAndPin`, about two per re-rank candidate.
  Traversal itself pins nothing; it reads optimistically.
- No slow-path calls (neighbour-list fallback, missing blocks, async
  reads) and no visited-set growth.

With about 3,600 buffer accesses per query, that is ~17 unvisited
neighbours evaluated per hop.

**Inside traversal** (share of traversal's cycle samples):

| code | source | 1 client | 40 clients |
|---|---|---:|---:|
| tuple prefetch: read the line pointer, prefetch each tuple line | `hnswutils.c:1522–1566` | 30% | 24% |
| batch lookup: read the translation entries just prefetched, prefetch every neighbour's line pointer | `calico_internal.h:690–823` | 22% | 31% |
| candidate heaps (sift up/down) | `hnswutils.c:2069–2200` | 12% | 11% |
| visited-set probe | `hnswutils.c:1870–1917` | 9% | 8% |
| translation-entry prefetches | `calico_internal.h:609` | 4% | 4% |
| read neighbour list, hash TIDs | `hnswutils.c:1796–1862` | 3% | 5% |
| loop control, element checks, next-hop prefetch, other | | 20% | 17% |
| **of which on a `prefetcht0` instruction** | | **50%** | **54%** |

**Counters, per query** (groups 1–2 are scaled estimates; see below):

| | 1 client | 40 clients |
|---|---:|---:|
| retired loads that missed L1 and allocated a fill buffer | 13,057 | 9,692 |
| retired loads that missed L1 and hit a line already in flight | 16,674 | 11,458 |
| — of which hit a software prefetch still in flight | 8,478 | 4,263 |
| retired loads served by L3 | 2,775 | 1,635 |
| retired loads served by DRAM: local / remote | 333 / 305 | 316 / 304 |
| L1D line fills | 54,662 | 68,677 |

What this shows:

- **Half of traversal is the core stalled at prefetch instructions.**
  - The precise cycle samples land on `prefetcht0`, and the load-latency
    samples give those prefetches 120–310 cycles each.
  - A software prefetch needs one of the L1's 10 fill buffers, and the
    core cannot retire past it until it has one.
  - Each hop issues its prefetches in bursts, one stage at a time, for
    all ~17 neighbours:
    1. a translation-entry prefetch per neighbour (`calico_internal.h:609`);
    2. straight away, the batch lookup reads those entries and prefetches
       every neighbour's line pointer (`calico_internal.h:771–777`);
    3. straight away, the priming loop reads three line pointers and
       prefetches those tuples' ~5 lines each (`hnswutils.c:1560–1566`).
  - That is 17 or more lines requested at once against 10 buffers, with
    no distance work in between to overlap. The two burst stages are ~50%
    of traversal, about 20% of all query cycles (~200 µs at 1 client).
- **Prefetches are issued too late, not too few.**
  - Demand loads seldom go to DRAM: ~640 per query, against ~3,000
    neighbour tuples.
  - But 8,500 loads per query at 1 client (2.7 per neighbour) find their
    software prefetch still in flight. Each stage reads data prefetched a
    few instructions earlier.
  - Only the tuple stage runs ahead, by three neighbours (`HNSW_PREFETCH_AHEAD`).
- **Memory placement is not the issue.**
  - Translation-entry walks were removed by 2 MB pages (previous section).
  - Of the loads served by DRAM, 48% are remote, the expected result of
    interleaving `shared_buffers` across the two sockets.
- **The candidate heaps and the visited set are ~20% of traversal**
  (~8% of the query). The visited set is the open-addressing table and
  stays in cache.
- **The re-rank pays the same way.**
  - Its code loads (`TqRefineDotFast`, `hnswtq_simd.h:142–145`) take
    220–290 cycles, 50–85% of them waiting on a line already in flight.
  - So do the descriptor reads in `CalicoTryPinBuffer` and
    `CalicoLookupAndPin`.
  - Calico's pin path (7.5% of cycles) is all re-rank, so a smaller
    re-rank set shrinks it too.
- **Optimizations this points to, in order:**
  1. **Pipeline each hop instead of bursting it.** *(Measured since: no
     gain. See the next section.)* One loop over the
     unvisited neighbours, with each stage running a few neighbours ahead
     of the next: translation entry at i+6, line pointer at i+4, tuple
     lines at i+2, distance at i. That keeps about 8–10 lines in flight
     and overlaps them with the ~100 ns `TqNavDot` per neighbour. The
     next candidate's neighbour list is already prefetched during the
     current hop. Its visited-set filtering and first translation-entry
     prefetches could start in the tail of the current hop, so the
     pipeline doesn't drain between hops. This is a code change only,
     with no index rebuild, and it goes after the ~200 µs per query of
     burst stalls.
  2. **Re-rank:** prefetch codes and buffer descriptors a few candidates
     ahead, and use a smaller set (see "Open").
  3. **Fewer lines per neighbour.** An element tuple is a 72-byte header,
     60 bytes of it heap TIDs, then the ~200-byte tq4 code: about 5.4
     lines on average. The heap TIDs are only needed for the final top-k.
     Moving them out of the element tuple would save about one line per
     neighbour (~18%), but needs a new index format and a rebuild.
  4. **The candidate heaps** (a 4-ary heap, or a sorted beam for small
     ef): a few percent at most.
- **Method notes.**
  - The data addresses in the load-latency samples were invalid again,
    like the STLB samples, so attribution rests on instructions and data
    sources.
  - Counter groups 1–2 use the 0xD1–0xD3 load events. They were counted
    41–59% of each window because of the hyperthread exclusion, so they
    are scaled estimates.
  - Not used, as unreliable here:
    - Haswell's outstanding-miss counters (`L1D_PEND_MISS`,
      `OFFCORE_REQUESTS_OUTSTANDING`), which miscount with hyperthreading
      on (erratum HSD78);
    - the `CYCLE_ACTIVITY` stall shares, which also count halted cycles
      of idle CPUs and exceed 100% system-wide at 1 client.
  - The latency of a prefetch instruction (120–310 cycles) is read as
    time waiting for a fill buffer. The cycle samples that land on the
    prefetches support that reading, but it is an inference.

## A pipelined hop (measured: no gain)

`hnsw.pipeline_hop`, a new per-session setting in pgvectorfast_ntq that is
off by default, replaces each hop's per-stage prefetch bursts with one
staged loop:
- the translation entry 6 neighbours ahead of the distance computation;
- the line pointer 4 ahead;
- the tuple lines 2 ahead.

Neighbours that the fast path can't read take the existing slow path.
Since the setting is per session, one server start served both versions
on the same warm pool. Each client run set it with `PGOPTIONS`, in the
order off, on, on, off. The configuration was huge pages,
`calico.huge_translation=on` and `/dev/shm` mounted `huge=advise`. Every
check passed:
- huge pages in use (67 GB);
- translation arrays on 2 MB pages;
- the served library is the new build;
- each session reports the setting it asked for.

QPS with the pipelined hop on, relative to off. Each version is the mean
of its two rounds; the largest round-to-round spread was 1.8%:

| ef_search | recall@10 (both) | 1 client | 40 clients |
|---:|---:|---:|---:|
| 20 | 0.7505 | 1.000 | 0.995 |
| 80 | 0.8978 | 1.007 | 1.010 |
| 160 | 0.9366 | 1.000 | 1.003 |
| 640 | 0.9722 | 0.995 | 0.999 |

- **Recall is identical** to six decimals at every ef_search, as
  expected: the same neighbours are evaluated in the same order.
- **Throughput doesn't change.** Every ratio is within the run-to-run
  spread.
- **The new path did run, and the stall moved instead of shrinking.**
  Profiles at ef_search 160:

  | | 1 client, off | 1 client, on | 40 clients, off | 40 clients, on |
  |---|---:|---:|---:|---:|
  | traversal, µs per query | 406 | 410 | 636 | 632 |
  | batch-lookup prefetches (`calico_internal.h:771–777`), share of traversal | 18% | gone | 28% | gone |
  | new line-pointer prefetch (`hnswutils.c:1614`), share of traversal | n/a | 23% | n/a | 33% |
  | loads that found their line already in flight, per query | 10.9k | 20.9k | 11.2k | 11.3k |
  | loads that hit a software prefetch in flight, per query | 6.5k | 13.3k | 4.2k | 5.6k |
  | L1D line fills, per query | 54.0k | 52.7k | 67.5k | 63.6k |

  The load counters are scaled estimates (counted 35–44% of each window).
- **Reading it.**
  - What costs time is how many lines each neighbour needs, not the order
    they are requested in. The same lines are fetched either way (L1D
    fills are within the counters' noise).
  - The stall lands on whichever prefetch finds all ten fill buffers
    busy: before, the batch-lookup burst; now, the line-pointer prefetch,
    whose latency samples average ~195 cycles while hitting L1, the same
    sign of waiting for a fill buffer as before.
  - At 1 client, twice as many demand loads now find their line still in
    flight. The requests queue behind each other instead of overlapping
    any better.
  - About 43k demand reads leave the core per query, or about 14 lines
    per evaluated neighbour (178 hops × ~17 neighbours ≈ 3,000 per query).
    That is about twice the ~7.4 lines an element tuple, its translation
    entry and its line pointer account for. Part of the excess is other
    traffic (neighbour lists, re-rank, heaps), and the counter may
    include software prefetches. Even so, it is the best lead for where
    traversal's time goes.
- **What it changes in the list above.** Reordering is ruled out.
  Fewer lines per neighbour (item 3) becomes the main traversal lever,
  together with finding out what the other half of the lines leaving the
  core are. The setting stays in the code, off by default. It costs
  nothing off and is worth re-testing once each neighbour needs fewer
  lines.

## Why Ginger is faster than pgvectorfast + tq4 (measured)

Ginger (recommended config) against pgvectorfast + tq4 in its best measured
configuration (huge pages, 2 MB translation arrays; the `thpon` rows):

| recall@10 | Ginger beam | pgvectorfast ef_search | QPS ratio, 1 thread | 40 threads |
|---:|---:|---:|---:|---:|
| 0.90 | 32 | 83 | 1.78× | 2.75× |
| 0.95 | 63 | 227 | 2.22× | 3.01× |
| 0.97 | 96 | 551 | 3.24× | 4.31× |

At equal width the order reverses: at 160, pgvectorfast runs 831 QPS (recall
0.937) and Ginger 690 (recall 0.985).

**Work per query** (`hn_ginger_prof.sh`): Ginger's own search statistics
(`ginger_bench --stats`), and uprobe counts on pgvectorfast (`pg_counts.sh`):

| per query, 1 thread | recall 0.90 | 0.95 | 0.97 |
|---|---:|---:|---:|
| Ginger: 4-bit distances · nodes expanded · re-ranked | 1,541 · 42 · 32 | 2,459 · 72 · 60 | 3,398 · 105 · 60 |
| pgvectorfast: 4-bit distances · layer-0 hops · re-ranked | 1,919 · 100 · 88 | 4,175 · 244 · 232 | 8,702 · 566 · 556 |
| distances, pgvectorfast ÷ Ginger | 1.25× | 1.70× | 2.56× |

**Time per query at recall 0.95, 1 thread** (µs). Ginger at beam 64 is
measured. pgvectorfast at ef_search 227 is modelled: the ef 160 profile's cost
per distance (211 ns) and per re-ranked candidate (1.42 µs), times the
measured work, plus its fixed costs. That gives 1,504 µs against 1,543 µs
measured.

| | graph search | re-rank | executor, protocol, kernel, other | off the server's CPU | total |
|---|---:|---:|---:|---:|---:|
| pgvectorfast + tq4 | 880 | 330 | 146 | 148 | 1,504 (1,543 measured) |
| Ginger | 666 | 25 | 27 (all) | | 718 |

The 825 µs gap breaks down as:
- +354 µs: pgvectorfast scores 1,681 more neighbours;
- +305 µs: it re-ranks 232 candidates at 1.42 µs each, where Ginger re-ranks
  60 at 0.42 µs;
- +267 µs: Postgres executor, protocol and kernel, plus the client round trip;
- −141 µs: Ginger pays more per neighbour (267 ns against 211 ns);
- +40 µs: model shortfall.

- **The graph is the largest factor, and it grows with recall** (1.25×, 1.7×,
  2.6× fewer distances), matching the widening QPS gap.
  - Ginger expands beam + 8 nodes and scores ~34 new neighbours per expansion.
    pgvectorfast takes ef + 18 hops at ~18 each.
  - Ginger's graph is PiPNN with up to 64 neighbours, α-pruned at 1.08. HNSW's
    layer 0 has 32, from ef_construction 64.
  - These runs don't separate degree from construction. A stronger HNSW build
    would.
- **The re-rank and the Postgres boundary are the rest.**
  - pgvectorfast re-ranks every result candidate (ef + 5), each through Calico
    lookups, pins and locks on two pages. Ginger re-ranks at most 6 × k = 60,
    from a flat 12-bit array.
  - At 40 clients the server's CPUs are 25.7 of 40 busy, against Ginger's 39.9.
    Per query, pgvectorfast uses 1.96× Ginger's CPU (5.58 M against 2.84 M
    cycles at 0.95). Divided by the busy shares, that predicts a 3.05× QPS
    ratio, against 3.01× measured.
- **Per neighbour, pgvectorfast is cheaper.** At width 160:

  | per neighbour | pgvectorfast 1c | Ginger 1t | pgvectorfast 40c | Ginger 40t |
  |---|---:|---:|---:|---:|
  | cycles in the graph search | 548 | 704 | 850 | 1,094 |
  | demand reads leaving the core | 13.6 | 6.3 | 14.3 | 6.6 |
  | page walks | 1.18 | 0.83 | 1.49 | 1.03 |
  | cycles walking | 6.9% | 18.0% | 11.3% | 12.4% |
  | loads hitting a software prefetch in flight, per query | 6,450 | 6 | 4,197 | 6 |

  - 77–79% of Ginger's cycles are in its 4-bit kernel. 56% of all cycles sit
    on the first load of each 192-byte code row.
  - The kernel issues no software prefetches. The source requests them
    (`score_4bit_soa.cc:140–165`), but no compiled copy of the kernel contains
    one (GCC 13.3, `-O3 -march=native`); the 12-bit re-rank kernel does. Why
    is not established.
  - Ginger runs on 4 KB pages: THP is `madvise` and Ginger doesn't ask, so
    0 GB of its 17.4 GB is on huge pages.
  - So Ginger has headroom of its own on this machine. pgvectorfast's
    per-neighbour path is not what separates the two.
- **Implications for pgvectorfast, in order:**
  1. A better graph: a stronger HNSW build first, then α-pruning or a
     Vamana-style build. Measure distances at matched recall.
  2. A bounded re-rank (top c × k). At c = 6 it would save ~245 µs of 1,543
     at 0.95, if recall holds.
  3. Enough clients to keep every server CPU busy at high concurrency.
- **Method notes.**
  - pgvectorfast's profile is `hn-pipe/off` at ef_search 160, the
    `thpon` configuration. Its page-walk figures come from `hn-calico-thp/on`.
  - The ginger_bench binary was rebuilt with `-g1`, which generates the same
    code. QPS stayed within 1.2% of the main runs.
  - Counters using the 0xD1–0xD3 events were counted 36–45% of each window
    and are scaled.
  - Report: https://claude.ai/artifact/Bf1kvr9XNJb8PTeewVcemj

## Is it the graph? Ginger's PiPNN graph in pgvectorfast (measured)

The awitten1/pgvectorfast fork can build an `hnsw` index from a graph built
elsewhere (PR awitten1/pgvectorfast#1):
- **`tools/pipnn_export`** links Ginger (a submodule). It `SELECT`s
  `ctid, embedding` out of the table and builds a PiPNN graph with Ginger's
  recommended config: degree ≤ 64, two passes, α 1.08.
- **`hnsw.build_graph_file`** makes `CREATE INDEX` store that graph as layer
  0, with every element on that layer, starting searches at Ginger's entry
  point. It streams, at about 42 bytes of memory per row.

So this is a single-layer PiPNN graph in an `hnsw` index, not an HNSW graph.

Searching it, the same engine (the fork, float vectors) can be compared on
the two graphs. Ginger can also be compared on the identical graph, with
`ginger_bench --graph`.

The setup (`hn_pipnn.sh`, `hn_pipnn_ginger.sh`, `hn_pipnn_hp.sh`):
- the PiPNN index is on an unlogged copy of the table (`items_hn_pipnn`), so
  neither the load nor the index writes WAL, which `/mydata` had no room for;
- the HNSW arm is the main study's index (m=16, ef_construction=64);
- both arms were run with 4 KB pages and with `huge_pages=on`.

**Build, 27.4M rows, 40 threads:**

| part | time |
|---|---:|
| export: binary COPY of 27.4M rows | 152 s |
| Ginger: 4-bit quantization | 28 s |
| Ginger: PiPNN graph | 999 s |
| Ginger: 12-bit codes (not needed here) | 28 s |
| `CREATE INDEX` from the file (serial, streaming) | 365 s |
| **total** | **~1,580 s (26 min), against 4,105 s for the HNSW build** |

- The graph has mean degree 39.4 and maximum 64.
- 27,445,749 of 27,445,759 nodes are reachable from the entry point, all
  within 11 hops.
- The index is 52 GB, the same size as the m=16 HNSW index, at 4 elements
  per page either way.
- At 1M rows the streaming import and an earlier in-memory version gave
  identical recall from the same graph file.
- Two PiPNN builds of the same rows aren't bit-identical (multithreaded); at
  1M their recall differed by at most 0.003.

**Recall by width (1 client):** the PiPNN graph in Postgres matches Ginger's
own index, and Ginger's float search on the same graph matches Postgres
exactly:

| width | HNSW graph | PiPNN graph (pgvectorfast) | same graph, Ginger float | Ginger, own index |
|---:|---:|---:|---:|---:|
| 20 | 0.7487 | 0.8504 | 0.8504 | 0.8475 |
| 80 | 0.8950 | 0.9662 | 0.9662 | 0.9618 |
| 160 | 0.9326 | 0.9850 | 0.9850 | 0.9845 |
| 640 | 0.9710 | 0.9973 | 0.9973 | 0.9968 |

**Work per query** (pgvectorfast: uprobes; Ginger: its statistics):

| width | HNSW graph: hops · distances | PiPNN graph, pgvectorfast | PiPNN graph, Ginger float |
|---:|---:|---:|---:|
| 20 | 37.0 · 615 | 29.6 · 1,154 | 29.6 · 1,157 |
| 160 | 176.2 · 3,008 | 167.7 · 5,101 | 167.7 · 5,108 |
| 640 | 654.9 · 9,813 | 646.5 · 16,513 | 646.4 · 16,471 |

- pgvectorfast's layer-0 search and Ginger's beam search do the same work on
  the same graph, within 0.3%. Ginger's candidate batching barely matters.
- PiPNN scores 25–39 new neighbours per hop against HNSW's 15–18. It costs
  more at equal width, but it needs a much smaller width for the same
  recall.

**QPS at matched recall:**

| recall | HNSW, 4 KB | PiPNN, 4 KB | HNSW, huge pages | PiPNN, huge pages | Ginger float, same graph | Ginger native, same graph | Ginger, own index |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **1 client / thread** | | | | | | | |
| 0.90 | 750 | 1,025 | 915 | 1,205 | 2,092 | 2,317 | 2,309 |
| 0.95 | 350 | 725 | 439 | 857 | 1,332 | 1,438 | 1,436 |
| 0.97 | 180 | 557 | 227 | 662 | 959 | 1,030 | 1,037 |
| 0.98 | – | 442 | – | 527 | 717 | 797 | 808 |
| 0.99 | – | 285 | – | 340 | 432 | 488 | 539 |
| **40 clients / threads** | | | | | | | |
| 0.95 | 4,360 | 10,480 | 5,340 | 11,024 | 13,435 | 32,658 | 36,354 |
| 0.97 | 2,161 | 7,816 | 2,663 | 8,186 | 9,712 | 23,319 | 26,301 |
| 0.98 | – | 6,044 | – | 6,354 | 7,342 | 17,993 | 20,574 |

(Ginger runs on 4 KB pages, as in the main study. "Native" is 4-bit
traversal with a 12-bit re-rank of the top 60, on the same graph without
Ginger's reordering or compression.)

**Ginger's lead over pgvectorfast's HNSW graph at recall 0.95, factored**
(each step multiplies the one above; the products match the measured totals):

| step | 1 thread | 40 threads |
|---|---:|---:|
| graph: HNSW → PiPNN (pgvectorfast, 4 KB pages) | 2.07× | 2.40× |
| engine: pgvectorfast → Ginger, float distances, same graph | 1.84× | 1.28× |
| quantization: Ginger float → native | 1.08× | 2.43× |
| Ginger's reordering and graph compression | 1.00× | 1.11× |
| **total** | **4.1× (350 → 1,436)** | **8.3× (4,360 → 36,354)** |

- **It is largely the graph.** On the same engine, PiPNN's graph gives 1.4×,
  2.1× and 3.1× at recall 0.90, 0.95 and 0.97 (1 client), and 1.5× to 3.6×
  at 40. pgvectorfast on the PiPNN graph, even without tq4 or huge pages,
  beats the best-tuned pgvectorfast + tq4 at 1 client from recall 0.95 up.
  It also reaches 0.99, which the HNSW graph can't.
- **The engine's share is per-query overhead, and it fades with
  concurrency.** That's the Postgres executor, protocol and client round
  trip, plus pgvectorfast's search loop. Huge pages narrow it from 1.84× to
  1.55× at 1 client. At 40 clients both engines are limited by moving float
  vectors (~1.5 KB per distance), so the gap is 1.2–1.3×.
- **Quantized distances matter at high concurrency** (2.4× at 40 threads),
  where they cut the bytes moved, and hardly at 1 thread (1.08×).
- **Huge pages for `shared_buffers`** give 1.18× (PiPNN) and 1.25× (HNSW) at
  1 client, and 1.05× and 1.22× at 40. The graph gain holds with them: 1.95×
  and 2.06× at recall 0.95.
- **What it points to for pgvectorfast:**
  1. the PiPNN graph (the PR);
  2. tq4 navigation codes on it, for high concurrency (the import exists only
     in the float fork so far);
  3. lower per-query overhead, for single clients.
- **Caveats.**
  - Rows inserted after the import go through HNSW insertion: new upper
    layers grow from inserted rows only, and their neighbour lists are
    HNSW-pruned. The graph drifts from PiPNN's as rows are added.
  - The unlogged copy is emptied after a crash.

## pgvectorfast + PiPNN + tq4 (measured)

The `pgvectorfast-ntq` branch (awitten1/pgvectorfast#3, stacked on #1) ports
Ginger's TurboQuant codec and adds `quantizer = 'tq4'` indexes. Element
tuples hold 4-bit codes instead of floats. The candidates are re-scored with
4+8 bit codes (`rerank = 'ntq12'`) kept on separate pages.
`bench/cloudlab/hn_pipnn_tq4.sh` imported the same PiPNN graph file as tq4
codes into `items_hn_pipnn` and swept at 1 and 40 LAN clients on 4 KB pages
(label `piptq4-pipnn`). It then imported the float index again and checked it
against `pippipnn-fork`: identical recall at every `ef_search`.

| clients | recall | PiPNN, floats | **PiPNN, tq4** | tq4 ÷ floats | Ginger, same graph and codec | Ginger, own index |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.90 | 1,025 | **1,183** | 1.15× | 2,317 | 2,309 |
| 1 | 0.95 | 725 | **788** | 1.09× | 1,438 | 1,436 |
| 1 | 0.97 | 557 | **606** | 1.09× | 1,030 | 1,037 |
| 1 | 0.99 | 285 | **322** | 1.13× | 488 | 539 |
| 40 | 0.90 | 15,254 | **20,440** | 1.34× | – | 58,251 |
| 40 | 0.95 | 10,480 | **15,449** | **1.47×** | 32,658 | 36,354 |
| 40 | 0.97 | 7,816 | **12,109** | 1.55× | 23,319 | 26,301 |
| 40 | 0.99 | 3,752 | **6,181** | 1.65× | – | 13,109 |

(Missing cells weren't in the earlier "Is it the graph?" runs.)

- **tq4 helps under load, as in Ginger, but by less.** It gives 1.09–1.15× at
  1 client and 1.34–1.65× at 40, growing with recall. In Ginger it gives
  1.08× and 2.43× (ablation and "Is it the graph?").
- **Recall per `ef_search` is the float graph's, give or take the codes:**
  0.7186 against 0.7388 at 10, then within 0.003 from 20 up (0.9630 against
  0.9662 at 80, 0.9980 against 0.9987 at 1000). The 12-bit re-rank recovers
  what the 4-bit walk loses.
- **Index:** 29.5 GiB (31.7 GB) against the float index's ~55 GB.
- **Build:** `CREATE INDEX` from the file took 1,163 s, against 365 s for
  floats. The import is serial, so encoding each row (rotation and 4+8 bit
  codes, about 29 µs) adds about 13 minutes at 27.4M rows. It could run in
  parallel.
- **Ginger's lead at recall 0.95 is now 1.82× at 1 client and 2.35× at 40**,
  down from 1.98× and 3.47× against floats. Factored at 40 clients:
  - 1.11× is Ginger's reordering and compression (own index against the same
    graph);
  - 2.11× is the engine, with the same graph, codes and re-rank precision.

  At 1 client the engine gap is what it was with floats (1.82× against
  1.84×), so it is per-query overhead. The 2.11× at 40 is larger than with
  floats (1.28×), because tq4 gains less here than in Ginger. Not measured on
  this configuration yet, the candidates are:
  1. **Busy CPUs:** the server ran 25.7 of 40 CPUs busy with tq4 on the HNSW
     graph, against Ginger's 39.9.
  2. **Bytes per neighbour:** a translation entry, a line pointer and a
     288-byte tuple per neighbour (72 bytes of tuple header and heap TIDs
     before the codes), plus a 392-byte neighbour tuple per hop, against
     Ginger's contiguous 192-byte code rows.
  3. **Re-rank:** each candidate needs two page lookups.

  A profile of this configuration at 40 clients would separate them.

## HNSW at m = 32 (measured)

Is PiPNN's lead over the HNSW graph its degree (64 neighbours at layer 0,
against HNSW m = 16's 32)? `bench/cloudlab/hn_hnsw_m32.sh` built an HNSW
index with m = 32 (64 at layer 0, the same as the PiPNN graph) and
ef_construction = 64 on `items_hn_pipnn`, in place of the float PiPNN index.
It then swept pgvectorfast (labels `pipm32-fork`) and stock pgvector
(`pipm32-stock`) at 1 and 40 LAN clients on 4 KB pages, and finally imported
the PiPNN graph again. The restored index matched `pippipnn-fork` exactly at
every `ef_search`.

- **Build:** 4,190 s, entirely in `maintenance_work_mem`, against 4,105 s for
  m = 16. **Index:** 54 GB, about the same as the PiPNN index.
- **Recall per `ef_search`:** 0.8979 at 40, 0.9646 at 160, 0.9864 at 640 and
  0.9896 at 1000. m = 16 tops out at 0.979, PiPNN at 0.9987.

**QPS at matched recall@10:**

| engine | clients | recall | HNSW m = 16 | **HNSW m = 32** | PiPNN | m = 32 ÷ m = 16 | PiPNN ÷ m = 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pgvectorfast | 1 | 0.90 | 750 | **809** | 1,025 | 1.08× | 1.27× |
| pgvectorfast | 1 | 0.95 | 350 | **451** | 725 | 1.29× | 1.61× |
| pgvectorfast | 1 | 0.97 | 180 | **277** | 557 | 1.54× | 2.01× |
| pgvectorfast | 1 | 0.98 | – | **171** | 442 | – | 2.58× |
| pgvectorfast | 40 | 0.90 | 10,198 | **9,966** | 15,254 | 0.98× | 1.53× |
| pgvectorfast | 40 | 0.95 | 4,360 | **5,061** | 10,480 | 1.16× | 2.07× |
| pgvectorfast | 40 | 0.97 | 2,161 | **2,954** | 7,816 | 1.37× | 2.65× |
| pgvectorfast | 40 | 0.98 | – | **1,785** | 6,044 | – | 3.39× |
| pgvector | 1 | 0.90 | 251 | **258** | 339 | 1.03× | 1.31× |
| pgvector | 1 | 0.95 | 109 | **131** | 221 | 1.20× | 1.69× |
| pgvector | 1 | 0.97 | 54 | **77** | 163 | 1.43× | 2.12× |
| pgvector | 1 | 0.98 | – | **46** | 125 | – | 2.72× |
| pgvector | 40 | 0.90 | 5,197 | **5,373** | 6,767 | 1.03× | 1.26× |
| pgvector | 40 | 0.95 | 2,330 | **2,785** | 4,535 | 1.20× | 1.63× |
| pgvector | 40 | 0.97 | 1,144 | **1,646** | 3,394 | 1.44× | 2.06× |
| pgvector | 40 | 0.98 | – | **999** | 2,622 | – | 2.62× |

- **Degree explains a small part of PiPNN's lead.** At recall 0.95, doubling
  HNSW's degree gives 1.16–1.29×. PiPNN is still 1.6–2.1× faster than an HNSW
  graph of the same degree and about the same size, so most of its lead
  comes from how it is built.
- **The extra degree pays only at high recall.** At 0.90, m = 32 is no faster
  than m = 16 (0.98–1.08×): each hop reads twice the neighbours. At 0.97 it
  is 1.37–1.54×, and it reaches 0.98, which m = 16 never does. It still
  levels off below 0.99.
- **The engine ratio holds:** pgvectorfast is 3.1–3.7× stock pgvector at 1
  client and 1.8× at 40 on this graph, close to the other two graphs
  (3.0–3.3× and 1.9–2.3×).
- **Caveat:** ef_construction = 64 is the least pgvector accepts with m = 32,
  so this is a cheap HNSW build. A stronger one (ef_construction 200) would
  take roughly 2–3× longer to build and may close some of the remaining gap.

## Measurement quality

- **Repeatability between rounds of the same arm** (1 client, same `ef_search`):
  - stock r1 vs r3: median 4.2%, max 5.3%
  - fast r2 vs r4: median 1.7%, max 2.4%
  - pgvector_ntq r1 vs r2: max 0.5%
  - pgvectorfast + tq4, optimized kernels: max 1.7% (4 KB pages), 2.8% (huge pages)
  - pgvectorfast on huge pages: max 2.7%
  - Calico translation arrays on 2 MB pages: one run per arm, so the
    1-client gain (2–4%) is within the noise above; the 40-client gain
    (4–7%) is not
- **10–40 clients:** one round per arm, so treat differences under about 5% as
  noise.
- **The two nodes are not cross-checked against each other.** Each engine ran
  on only one of them: Postgres on the server node, Ginger on the client node.
  The hardware is identical and so is the CPU pinning. The environment captures
  are in `bench/results/<host>/env-*.txt`.

## Protocol notes

- **Index preservation.** `hn.env` sets `PROTECT_INDEXES=1`. With it,
  `ann_bench.py` refuses to drop, rebuild or reload an existing table or index,
  so the 52 GB HNSW index can be re-measured under other server settings
  (`EXTRA_PG_OPTS`) without rebuilding it.
- **SSH.** Each driver polls a node at most once a minute over one multiplexed
  connection per node. If a node stops answering, the driver backs off 1, 2,
  then 4 minutes and exits on the fourth consecutive failure.
- **Smoke run.** The whole pipeline ran first on a 1M-row subset
  (`hackernews:1000000`) with 5 s windows.

## Reproducing

```sh
. bench/cloudlab/nodes-wisc.sh
ssh -i ~/.ssh/cloudlab awitt@<node>.wisc.cloudlab.us 'bash -s -- disk' < bench/cloudlab/setup_node.sh   # once per node
bench/cloudlab/sync.sh push
SMOKE=1 bench/cloudlab/hn_qps.sh && bench/cloudlab/hn_qps.sh
SMOKE=1 bench/cloudlab/hn_tq4.sh && bench/cloudlab/hn_tq4.sh
bench/cloudlab/hn_profile_tq4.sh          # profiles + kernel bench -> bench/profiles/hn-tq4/
bench/cloudlab/hn_hugepages.sh            # stock with huge_pages=on -> bench/profiles/hn-hugepages/
./build_fastntq.sh && SMOKE=1 bench/cloudlab/hn_fastntq.sh && bench/cloudlab/hn_fastntq.sh   # pgvectorfast + tq4
bench/cloudlab/hn_fastntq_k2.sh           # optimized tq4 kernels, 4 KB and huge pages
bench/cloudlab/hn_fast_hp.sh              # float pgvectorfast with huge_pages=on -> bench/profiles/hn-fast-hp/
bench/cloudlab/hn_tlb_diag.sh             # where pgvectorfast + tq4's page walks come from -> bench/profiles/hn-tlb/
bench/cloudlab/hn_calico_thp.sh           # calico.huge_translation off vs on -> bench/profiles/hn-calico-thp/
bench/cloudlab/hn_trav_prof.sh            # what traversal is made of -> bench/profiles/hn-trav/
python3 bench/cloudlab/trav_attrib.py bench/profiles/hn-trav > bench/profiles/hn-trav/attrib.md
bench/cloudlab/hn_pipe_hop.sh             # hnsw.pipeline_hop off vs on -> bench/profiles/hn-pipe/
for a in off on; do python3 bench/cloudlab/trav_attrib.py bench/profiles/hn-pipe/$a --lib bench/profiles/hn-pipe/vectorfast.so > bench/profiles/hn-pipe/$a/attrib.md; done
bench/cloudlab/hn_ginger_prof.sh          # Ginger vs pgvectorfast: work counts and profiles -> bench/profiles/hn-ginger/
python3 bench/cloudlab/ginger_attrib.py bench/profiles/hn-ginger > bench/profiles/hn-ginger/attrib.md
python3 bench/cloudlab/ginger_decomp.py   # matched-recall work and per-query time model
bench/cloudlab/hn_pipnn.sh                # PiPNN graph in the pgvectorfast fork vs its HNSW graph (SMOKE=1 first)
bench/cloudlab/hn_pipnn_ginger.sh         # Ginger searching the same graph file (float and native)
bench/cloudlab/hn_pipnn_hp.sh             # both graphs again with huge_pages=on
python3 bench/cloudlab/pipnn_compare.py   # matched-recall tables and work per query
bench/cloudlab/hn_pipnn_stock.sh          # stock pgvector on the PiPNN graph, and its 1-client LAN baseline, 4 KB and huge pages
bench/cloudlab/hn_ginger_thp.sh           # Ginger on THP (jemalloc thp:always) against a 4 KB control; THREADS="1 10 20 40" for all four
bench/cloudlab/sync.sh pull
for c in 1 10 20 40; do python3 bench/plot_variants.py hackernews_minilm_384 --clients $c; done
python3 bench/cloudlab/hn_fourway.py > bench/profiles/hn-pipnn/fourway.md   # four-way matched-recall tables
for s in fourway fourway-hp; do for c in 1 40; do
  python3 bench/plot_variants.py hackernews_minilm_384 --series hackernews-$s --clients $c \
    -o bench/report/hackernews-minilm-384-$s-${c}c-recall-qps.svg; done; done
```

## Open

- What the ~14 lines per evaluated neighbour that leave the core are,
  against the ~7.4 an element tuple, its translation entry and its line
  pointer need. Count them by source (tuple, neighbour list, re-rank,
  heaps, hardware prefetchers), for example with the L1 and L2 hardware
  prefetchers switched off in one window (MSR 0x1a4) and per-stage
  counts. The pipelined hop showed that the number of lines, not their
  order, is the cost.
- Fewer lines per neighbour: heap TIDs out of the element tuple (a new
  index format).
- PiPNN's graph with tq4 navigation codes and huge pages: port
  `hnsw.build_graph_file` to the tq4 branch, or tq4 to the fork.
- A stronger HNSW build (m 32, ef_construction 128–200). m = 32 at
  ef_construction 64 is measured ("HNSW at m = 32"): degree gives
  1.16–1.29× of PiPNN's 2.1× at recall 0.95. Also a bounded re-rank
  (top 6 × k) and its recall cost.
- Ginger with working prefetches in its AVX2 4-bit kernel, and why GCC drops
  them. (Ginger on huge pages is measured: 1.19–1.25×, in "Ginger on huge
  pages".)
- Postgres at more than 40 clients: whether the server's idle third at 40
  clients can be filled.
- `TqNavDot` with four accumulators (it is FMA-latency bound at two).
- 1 GB pages for `shared_buffers`: the half of the remaining page walks
  that 2 MB translation arrays leave (~3,200 per query).
- A targeted version of `calico.huge_translation`: 2 MB pages only for the
  populated ranges of large relations, pre-faulted at a safe point. Also,
  a second run of the blanket A/B in the opposite order.
- Fix the counter multiplexing in `tlb_window.sh` (drop the explicit
  `cycles,instructions` from groups, or disable the NMI watchdog for the
  windows).
- Why float pgvectorfast gains nothing from huge pages at 40 clients: a
  random-read bandwidth test on this machine, to locate the ceiling, and
  where the remaining walks come from.
- A smaller re-rank set (a few × k instead of all ef_search).
- pgvector_ntq should bump `HNSW_VERSION` (or its magic number), so stock
  pgvector refuses its indexes instead of misreading them.
- A stronger HNSW build (ef_construction 128–200), to compare pgvector at
  0.98–0.99. It would go on a copy of the table, to keep the current index.
- The uneven Postgres scaling from 10 to 40 clients: profile it, and compare
  `numactl --cpunodebind` on one socket against interleave.
- Huge pages for pgvector_ntq. Stock, pgvectorfast and pgvectorfast + tq4
  are measured above.
- A profile at 10 and 20 clients, with and without huge pages, for the uneven
  10→20 step.
