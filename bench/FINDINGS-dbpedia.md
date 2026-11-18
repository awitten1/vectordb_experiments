# Ginger vs pgvector vs pgvectorfast vs PRISM on DBpedia-OpenAI 1M

Session of 2026-11-17 on two CloudLab nodes. Companion to `FINDINGS.md`, which
covers BIGANN 18M on a different machine; nothing here is comparable to those
numbers in absolute terms.

![recall vs QPS, 1 client](report/dbpedia-openai-1m-1536-angular-variants-1c-recall-qps.svg)
![recall vs QPS, 8 clients](report/dbpedia-openai-1m-1536-angular-variants-8c-recall-qps.svg)

## Setup

- **Hardware:** two identical CloudLab m510 nodes, ms0931 (node-0) and ms0905
  (node-1). Each has a Xeon D-1548 (8 cores / 16 threads, Broadwell), 12 MiB
  L3, 62 GiB RAM and NVMe. The CPU has **AVX2 only: no AVX-512 and no VNNI**,
  so Ginger's TurboQuant runs without the `vpdpbusd` path it is tuned for.
  The OS is Ubuntu 24.04, kernel 6.8.
- **Clock and memory settings:**
  - The clock is pinned with `bench/cloudlab/pin_cpu.sh`: turbo off,
    min = max = 2.0 GHz, and idle states deeper than C1 disabled.
  - Swap is off.
  - The runners refuse to start unless both hold.
- **Dataset:** Qdrant's `dbpedia-openai-1M-1536-angular`. It has 975,000 base
  vectors of `text-embedding-ada-002` at 1536-d, and 1,000 of its 5,000 test
  queries are used.
  - Exact top-100 is recomputed locally. It agrees with Qdrant's published
    top-10 to **1.0000**.
  - No query is also a base vector.
  - The ground-truth file is byte-identical on both nodes (sha256).
- **Metric:** L2 for every engine. The vectors are unit-norm (asserted:
  max |norm − 1| = 1.8e-7), so L2 ranks exactly like cosine. Using L2 means
  Ginger, which has no cosine, computes the same metric as the Postgres
  engines.
- **Engines and configurations:**

  | engine | configuration |
  |---|---|
  | pgvector 0.8.7 | HNSW, m=16, ef_construction=64 |
  | pgvectorfast | the same HNSW index, with Calico preloaded |
  | PRISM | pg_vectorsearch `93567fd`, all auto settings, on its own copy of the table |
  | Ginger (recommended) | `RecommendGingerConfig(1536, SQUARED_L2, 1.08)`: PiPNN graph (degree 64), PCA, 4-bit TurboQuant traversal, 12-bit rerank, float data dropped |
  | Ginger (float-rerank) | the same, minus PCA, with an exact float rerank |

- **Protocol:**
  - Each point runs for 60 s; k=10.
  - 1 and 8 clients for Postgres; 1 and 8 query threads for Ginger.
  - Postgres runs with `shared_buffers=28GB`, and the table, its TOAST table
    and all their indexes are prewarmed.
  - Every saved point is re-checked afterwards: no swapping, and every block
    still resident. Results: **146/146 rows clean**, and 6 system-wide major
    faults in total.
  - pgvector and pgvectorfast alternate stock / fast / stock / fast on one
    index on node-0.
  - node-1 repeats stock pgvector twice on its own build of the same index, as
    a cross-node check.
  - Ginger runs on node-1 with Postgres stopped.

## Results: QPS at matched recall@10

The curves are interpolated in log(QPS) against −log10(1 − recall). Rounds are
averaged.

**1 client / thread:**

| engine | 0.90 | 0.95 | 0.98 | 0.99 | 0.995 | max recall |
|---|---:|---:|---:|---:|---:|---:|
| pgvector HNSW | 338 | 251 | 110 | 57 | – | 0.9928 |
| pgvectorfast HNSW | 538 | 370 | 160 | 115 | – | 0.9928 |
| PRISM | 271 | 144 | 63 | 38 | 24 | 0.9980 |
| Ginger recommended | 3,140 | 2,263 | 1,347 | 865 | 491 | 0.9984 |
| Ginger float-rerank | 2,137 | 1,424 | 827 | 517 | 299 | 0.9987 |

**8 clients / threads:**

| engine | 0.90 | 0.95 | 0.98 | 0.99 | 0.995 | max recall |
|---|---:|---:|---:|---:|---:|---:|
| pgvector HNSW | 1,818 | 1,143 | 615 | 315 | – | 0.9928 |
| pgvectorfast HNSW | 3,059 | 2,018 | 889 | 530 | – | 0.9928 |
| PRISM | 1,263 | 999 | 463 | 287 | 186 | 0.9980 |
| Ginger recommended | 23,025 | 16,375 | 9,640 | 6,226 | 3,536 | 0.9984 |
| Ginger float-rerank | 15,046 | 9,936 | 5,718 | 3,590 | 2,073 | 0.9987 |

HNSW stops at 0.9928 because `ef_search=1000` is pgvector's maximum.

**Builds and sizes:**

| | build | threads | on-disk / in-memory size |
|---|---:|---:|---|
| pgvector / pgvectorfast HNSW | 433 s | 15 workers | index 7.4 GiB, plus the 7.4 GiB TOAST table it returns rows from |
| PRISM | 622 s | 15 workers | index 0.42 GiB, plus its own 7.4 GiB TOAST table for the rerank |
| Ginger recommended | 312 s | 16 | 2.4 GiB, everything included (float data dropped) |
| Ginger float-rerank | 270 s | 16 | 6.3 GiB, including the floats |

The HNSW build time comes from a timed rebuild on node-1, made after its
measurements were finished. The original build-time log lines on both nodes
were lost to a logging bug (see the protocol notes).

## What the numbers say

**Ginger is in a different class: about 7.5× pgvectorfast and 15× pgvector at
0.99 recall with one thread, and 12× / 20× with eight.** The gap is not the
Postgres boundary:
- A trivial prepared query carrying the same 6 KB vector parameter costs
  0.31 ms at 1 client (`index=rtt` rows; 3,084 QPS).
- Adding that floor to Ginger's 1.16 ms at 0.99 still leaves it about 5.9×
  pgvectorfast and 12× pgvector.
- At 8 threads, Ginger's QPS scales 7.2× on 8 cores, against 5.5× for
  pgvector.

**Ginger's recommended config beats its own float rerank by 1.5–1.7× at every
recall.** It also reaches the same top recall (0.998). Reranking from 12-bit
TurboQuant codes loses no measurable recall against exact floats, and it
avoids touching 6 KB of float per candidate. The PCA projection and dropped
floats cost nothing in quality on this dataset.

**pgvectorfast is 1.45–2.0× stock pgvector at matched recall:**
- With 1 client: 1.59× at 0.90, 1.47× at 0.95, 1.45× at 0.98, 2.0× at 0.99.
- With 8 clients: 1.45–1.77×.
- Both engines search the same graph and return identical recall at every
  `ef_search` in every round. The difference is purely the buffer manager.

**PRISM is the slowest engine here.** That reverses BIGANN, where it was 3–5×
faster than HNSW. `EXPLAIN (ANALYZE, VERBOSE)` on a warm backend, for query 0,
puts the time in the posting-list scan, not the rerank:

| prism.nprobe | entries scanned | posting pages | posting scan | rerank (candidates) | total |
|---:|---:|---:|---:|---:|---:|
| 5 | 1,658 | 49 | 0.65 ms | 0.19 ms (17) | 1.07 ms |
| 40 | 23,581 | 672 | 14.6 ms | 0.30 ms (17) | 15.9 ms |

- **The scan costs about 0.62 µs per entry.** A 1536-d 1-bit RaBitQ entry is
  ~234 bytes (35 per page), about 15× BIGANN's 128-d entry, and this CPU has no
  AVX-512 for the fast-scan kernels.
- **The rerank is cheap even though its vectors are TOASTed.** It reranks only
  17 candidates, so detoasting is not the problem.
- **PRISM builds slower than HNSW here as well** (622 s vs 433 s), again the
  reverse of BIGANN.
- **Worth asking the pg_vectorsearch authors** whether the scan path at high
  dimensions on AVX2 is expected to look like this.

## Measurement quality

**Nodes agree** where recall matches:
- Node-1's stock pgvector is within 1–5% of node-0's at 0.90, 0.95 and 0.99.
- At 0.98 it is 13% apart. That target sits in the gap between `ef_search`
  160 and 320, and the two independently built graphs place their points
  differently, so it is mostly interpolation error.
- At `ef_search=1000` both graphs reach recall 0.9928, at 41.9 vs 42.5 QPS.
- Parallel HNSW builds are not deterministic. node-1 reaches slightly higher
  recall at low `ef_search` (0.769 vs 0.761 at 10), so compare the nodes at
  matched recall, not at matched `ef_search`.

**Repeatability between rounds of the same arm:**

| arm | 1 client: median / max diff | 8 clients: median / max diff |
|---|---:|---:|
| node-0 stock (r1 vs r3) | 3.1% / 11.7% | 1.5% / 4.4% |
| node-0 fast (r2 vs r4) | 9.4% / 17.4% | 0.5% / 2.1% |
| node-1 stock (r1 vs r2) | 4.8% / 11.1% | 1.0% / 4.4% |

- **1-client points are noisy**, especially pgvectorfast's. Treat 1-client
  differences under ~15% as unresolved. The 8-client numbers are tight.
- **PRISM was measured in one round only.** In separate 5–10 s smoke runs at
  100K, its `nprobe=5` throughput ranged from 225 to 353 QPS between identical
  runs. That points to a per-backend warm-up: a fresh backend's first query
  took 68 ms against 1 ms warm. The 60 s windows amortize it, but the PRISM
  curve has no repeat to bound its noise.

## Protocol notes and fixes made along the way

- **TOAST.** A 1536-d `vector` is ~6 KB, over the ~2 KB TOAST threshold, so
  the heap holds only pointers: 6,256 pages for 975K rows.
  - The vectors live in `pg_toast_*`: 975,000 pages, one vector per page.
  - `prewarm_or_die` and the per-point residency check now cover the TOAST
    table and its index. Before this fix, "fully resident" silently excluded
    the 7.4 GiB of vectors.
- **Planner.** Because the heap is tiny, the planner prices a seq scan plus
  sort as nearly free; it does not cost detoasting. At `prism.nprobe=80` on
  100K it picked a seq scan costed at 4,051 over the index.
  - `ann_bench.py` now sets `enable_seqscan=off` for index arms and records it
    in `settings`.
  - `check_plan` still asserts the intended index at every point.
  - A cost-model gap worth reporting upstream: neither pgvector's nor PRISM's
    costing accounts for TOASTed vectors.
- **Logging.** `prewarm_or_die` printed its table with `tee /dev/stderr`.
  Under `nohup ... > log 2>&1`, `tee` reopens the log with `O_TRUNC` and wipes
  it, which is how the HNSW build-time lines were lost. It now captures the
  output into a variable and prints it.
- **CPU.** Leaving C1E enabled let idle cores drop to 800 MHz even with the
  clock pinned. Single-client Postgres is a client↔backend ping-pong that pays
  that wake-up on every query, which in-process Ginger never does.
- **Ginger build flags.** `ginger_cc_binary` reads `GINGER_DEFAULT_COPTS`,
  which is only defined in Ginger's directory scope. The first driver build
  therefore silently compiled without `-march=native` and `-funsigned-char`.
  `bench/ginger/CMakeLists.txt` now reads the variable from Ginger's scope, and
  `ginger_bench --version` reports the ISA (AVX2 here).

## Reproducing

```sh
bench/cloudlab/sync.sh push                       # workstation -> both nodes
# on each node, from /mydata/vectordb_experiments:
bench/cloudlab/setup_node.sh                      # /mydata, packages, pin CPU, swapoff, venv
set -a; . bench/cloudlab/dbpedia.env; set +a      # SMOKE=1 first for the 100K subset
# node-0:
./build_postgres.sh && ./build_extensions.sh
bench/run_experiment.sh data build pgvectorfast prism_solo overhead
# node-1:
./build_ginger.sh && bench/run_ginger.sh
./build_postgres.sh && ./build_extensions.sh
PVF_ARMS="stock stock" bench/run_experiment.sh data build pgvectorfast overhead
# workstation:
bench/cloudlab/sync.sh pull
bench/plot_variants.py dbpedia_openai_1m_1536_angular [--clients 8]
```

## Open

- PRISM's posting-scan cost at 1536-d on AVX2: profile it
  (`perf_event_paranoid` is 4 on these nodes and needs lowering), and try
  `fastscan` settings or `vec16`.
- A second PRISM round, to bound its noise the way the HNSW arms are bounded.
- Ginger on a VNNI / AVX-512 machine, where TurboQuant has its fast path.
  These numbers understate it.
- What TOAST costs the Postgres engines: rerun with
  `ALTER TABLE ... ALTER COLUMN embedding SET STORAGE PLAIN`. A 6 KB row still
  fits an 8 KB page, so the vectors would move back into the heap.
