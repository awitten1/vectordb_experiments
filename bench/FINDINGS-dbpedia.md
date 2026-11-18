# Ginger vs pgvector vs pgvectorfast vs PRISM on DBpedia-OpenAI 1M

Session of 2026-11-17 on two CloudLab nodes. Companion to `FINDINGS.md`, which
covers BIGANN 18M on a different machine; nothing here is comparable to those
numbers in absolute terms.

![recall vs QPS, 1 client](report/dbpedia-openai-1m-1536-angular-variants-1c-recall-qps.svg)
![recall vs QPS, 8 clients](report/dbpedia-openai-1m-1536-angular-variants-8c-recall-qps.svg)
![recall vs QPS, 12 clients](report/dbpedia-openai-1m-1536-angular-variants-12c-recall-qps.svg)
![recall vs QPS, 16 clients](report/dbpedia-openai-1m-1536-angular-variants-16c-recall-qps.svg)

From 8 clients up, the Postgres client runs on the second node, over the
private 10 GbE network (see "Where the client runs"). At 1 client it shares the
server's machine.

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
  - 1, 8, 12 and 16 clients for Postgres; the same numbers of query threads
    for Ginger.
  - At 1 client the Python client runs on node-0 beside the server. From 8 up
    it runs on node-1 and connects over the private LAN, so it doesn't compete
    with the backends for node-0's cores.
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

**8, 12 and 16 clients / threads** (Postgres client on node-1):

| engine | 0.95 @ 8 | @ 12 | @ 16 | 8 → 16 | 0.99 @ 8 | @ 12 | @ 16 | 8 → 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pgvector HNSW | 1,452 | 2,060 | 2,454 | +69% | 325 | 385 | 426 | +31% |
| pgvectorfast HNSW | 2,439 | 2,718 | 2,890 | +18% | 546 | 536 | 526 | −4% |
| PRISM | 1,071 | 1,213 | 1,344 | +25% | 296 | 311 | 325 | +10% |
| Ginger recommended | 16,375 | 19,645 | 23,027 | +41% | 6,226 | 7,502 | 8,886 | +43% |
| Ginger float-rerank | 9,936 | 11,555 | 12,914 | +30% | 3,589 | 4,167 | 4,645 | +29% |

The full curves at each count are in the charts above, and in `bench/results/`.
The first published version of this file had an 8-client table measured with
the client on node-0; those numbers are kept under "Where the client runs".

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
0.99 recall with one thread, 11× / 19× with eight, and 17× / 21× with
sixteen.** The gap is not the Postgres boundary:
- A trivial prepared query carrying the same 6 KB vector parameter costs
  0.31 ms at 1 client (`index=rtt` rows; 3,084 QPS).
- Adding that floor to Ginger's 1.16 ms at 0.99 still leaves it about 5.9×
  pgvectorfast and 12× pgvector.
- From 8 clients up, the floor (18,000–29,500 QPS) sits far above every
  Postgres engine's throughput, so the connection is never the limit there.

**Ginger's recommended config beats its own float rerank by 1.5–1.7× at every
recall.** It also reaches the same top recall (0.998), so PCA and the 12-bit
rerank cost nothing in quality on this dataset.

The float-rerank config differs in two ways, though: no PCA, and an exact
float rerank. So the run cannot say which of the two costs the time. The PCA is
the likely one:
- The graph search does most of the work. Without PCA it scores 4-bit codes of
  all 1536 dimensions, about 3× the data per distance.
- The rerank touches only ~60 candidates per query.
- At the same beam size, float-rerank is ~1.9× slower but slightly *more*
  accurate (beam 160: 0.9908 vs 0.9890). That fits a slower, more accurate
  search, not an expensive final step.

**pgvectorfast is 1.45–2.0× stock pgvector up to 8 clients, but its lead
shrinks past that:**
- With 1 client: 1.59× at 0.90, 1.47× at 0.95, 1.45× at 0.98, 2.0× at 0.99.
- With 8 clients: 1.68× at both 0.95 and 0.99.
- With 16 clients: 1.18× at 0.95 and 1.23× at 0.99. From 8 to 16 clients,
  pgvectorfast gains 18% at 0.95 and nothing at 0.99, while stock pgvector
  gains 69% and 31%.
- Both engines search the same graph and return identical recall at every
  `ef_search` in every round. The difference is purely the buffer manager.
- **Likely cause of the plateau, not yet profiled:** at high `ef_search` both
  engines stream a full 6 KB float vector per candidate. Past 8 clients the
  extra parallelism is hyperthreads only, which add no memory bandwidth.
  pgvectorfast, doing the same work faster per query, reaches this machine's
  bandwidth limit first; stock pgvector still has its buffer-manager overhead
  to hide behind.
- **Ginger keeps scaling** (+41–43% from 8 to 16 threads). Its 4-bit codes
  move about 20× less data per candidate.

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

## Where pgvector's time goes

The standalone write-up of this section, with figures, is
[PGVECTOR-BOTTLENECKS.md](PGVECTOR-BOTTLENECKS.md).

Profiled on node-0 with stock pgvector, a frame-pointer build, and the client
on node-1, so node-0 ran nothing but Postgres. For 1, 8 and 16 clients at
`ef_search` 40 (recall 0.913) and 640 (0.989), during one steady-state run per
configuration (`bench/cloudlab/profile_pgvector.sh`), three 20 s windows:
1. core counters plus DRAM traffic from the memory controllers
2. Top-Down level 1/2
3. system-wide call graphs

Plus `EXPLAIN (ANALYZE, BUFFERS)` on 100 queries, the machine's memory
ceilings (`membw.c`) and a distance-kernel microbenchmark (`distbench.c`).

**Per query, what HNSW touches:**

| ef_search | recall | pages per query (EXPLAIN) | DRAM read per query (IMC) |
|---:|---:|---:|---:|
| 40 | 0.913 | 1,060 | 6.5 MB |
| 640 | 0.989 | 8,555 | 61 MB |

Every visited element is one 8 KB index page holding its 6 KB float vector,
and nearly every byte of it comes from DRAM: the 7.4 GiB index is 600× the
12 MiB L3.

**Where the time goes** (share of Postgres CPU time, from the call graphs; the
split is within a few points across all six configurations):

| category | 1 client, ef 640 | 16 clients, ef 640 |
|---|---:|---:|
| distance (`vector_l2_squared_distance`) | 32% | 33% |
| buffer manager: page lookup (`BufTableLookup` → `hash_search_with_hash_value`) | 18% | 19% |
| buffer manager: pin/unpin, LWLocks, refcounts, `ReadBuffer` | 15% | 14% |
| HNSW element and neighbor loading | 18% | 16% |
| visited set (`tidhash`) | 5% | 7% |
| candidate queues (pairing heap) | 3% | 3% |
| executor, protocol, other | 6% | 4% |
| kernel | 3% | 3% |

At 1 client, `ef_search=640` costs 17.8 ms of server CPU per query, **2.1 µs per
visited element**: about 0.67 µs distance, 0.7 µs buffer manager, 0.36 µs
element loading, 0.11 µs visited set. Overlapping all of these, **page-table
walks take 12.7% of all cycles** (about 5.5 dTLB misses per element, 94 cycles
per walk). `shared_buffers` is mapped with 4 KB pages; no huge pages are
reserved.

**Top-Down:** retiring only 16–27% of cycles. Backend-bound 50–75%, of which
memory-bound 34–61%. Front-end-bound 6–21%.

**The distance kernel is memory-bound, not compute-bound, here:**

| ns per 1536-d distance, 1 thread | cached | from DRAM |
|---|---:|---:|
| pgvector's `VectorL2SquaredDistance` (one accumulator) | 447 | 741 |
| same arithmetic, four accumulators | 131 | 719 |

With the data cached, pgvector's loop is limited by its single accumulator's
FMA-latency chain (447 ns ≈ 192 FMAs × 4.7 cycles), and four accumulators are
3.4× faster. From DRAM, the real case, both take about the time to stream
6 KB: 0.72–0.74 µs, matching the 0.67 µs measured inside queries. A faster
kernel would not help this workload.

**Concurrency runs into DRAM bandwidth:**

| clients | ef_search | QPS | DRAM read GB/s | IPC | Mcycles per query |
|---:|---:|---:|---:|---:|---:|
| 1 | 640 | 56 | 3.4 | 0.73 | 38 |
| 8 | 640 | 325 | 18.9 | 0.62 | 37 |
| 16 | 640 | 446 | **27.5** | 0.35 | 69 |
| 16 | 40 | 3,471 | 24.2 | 0.48 | 6.4 |

The measured ceiling for this access pattern (random 6 KB reads) is 30 GB/s,
reached at 8 threads; streaming reads reach 35 GB/s. At 61 MB per 0.99-recall
query, that caps any engine reading these pages at roughly **490–570 QPS**.
Stock pgvector reaches 446 at 16 clients. pgvectorfast reaches 526–585 from 8
clients up and goes no further. That is the plateau in the scaling table:
pgvectorfast removes the buffer-manager third of the CPU cost, so it hits the
bandwidth ceiling sooner, and the ceiling is the same for both.

**Bottlenecks, in order of how much they limit throughput:**

1. **Bytes per visited element.** A 6 KB float vector per element and ~8,500
   elements per 0.99-recall query is 61 MB of DRAM traffic per query. This
   sets the throughput ceiling under concurrency and the largest share of
   single-query time. Halving bytes per element (`halfvec`) should raise the
   ceiling about 2×. Compressed codes in the graph, as Ginger stores (about
   300 B), raise it about 20×.
2. **The buffer manager per element** (~1/3 of CPU at low concurrency): the
   `BufTableLookup` hash probe, pin/unpin, LWLocks and private refcounts, for
   every one of the ~8,500 pages. This is what Calico (pgvectorfast) removes,
   1.5–2× at 1–8 clients, until bandwidth takes over.
3. **Page-table walks** (12.7% of cycles), from mapping 28 GB of
   `shared_buffers` with 4 KB pages. Huge pages for `shared_buffers` should
   recover most of it. Untested.
4. **HNSW bookkeeping:** element and neighbor-tuple loading (~17%), the
   visited set (5–7%), the candidate heaps (~3%).
5. **Not a bottleneck here:** the distance arithmetic (memory-bound, see
   above), the kernel, and the executor/protocol (≤6%).

Caveats:
- Profiling perturbs throughput somewhat. At 1 client:
  - `ef_search=640`: regular build 60.9 QPS, frame-pointer build unprofiled
    60.3, with call-graph sampling 55.9 (−8%).
  - `ef_search=40`: 324, 275 and 245. The unprofiled gap there is within the
    up-to-17% noise of single-client points.

  Read the profiles for shares of time, not absolute QPS.
- pgvectorfast was measured the same way at `ef_search=640`. It moves the same
  57–60 MB per query, and reads **32.9 GB/s at 8 clients and 33.4 GB/s at
  16**: already at the machine's limit with 8 clients. From 8 to 16, its IPC
  halves (0.46 → 0.21) for no extra throughput.
- At light load, `perf stat`'s `cycles` and `ref-cycles` disagree by 2.5×
  (likely idle cores in the POLL state counting differently), so per-query CPU
  time above comes from call-graph samples in Postgres processes, not from
  `cycles`. At 16 clients the counters agree (1.93 GHz).
- Flame graphs were not rendered; the `perf.data` files are on node-0 under
  `bench/profiles/pgvector/`.

## Where the client runs

The first 8-client sweeps ran the Python client on node-0, next to the server:
8 client processes and 8 backends on 8 cores / 16 hardware threads. To measure
what that cost, `bench/cloudlab/remote_client.sh` ran the same 8-client sweep
twice back to back on one warm server per arm, client on node-1 and client on
node-0, with the order flipped between arms.

| ef_search | recall | pgvectorfast: node-1 / node-0 client | stock pgvector: node-1 / node-0 client |
|---:|---:|---:|---:|
| 10 | 0.761 | 5,566 / 4,564 (+22%) | 3,455 / 2,983 (+16%) |
| 40 | 0.913 | 3,397 / 2,823 (+20%) | 2,179 / 1,666 (+31%) |
| 80 | 0.951 | 2,415 / 1,976 (+22%) | 1,435 / 1,105 (+30%) |
| 160 | 0.970 | 1,585 / 1,338 (+18%) | 1,044 / 930 (+12%) |
| 640 | 0.989 | 585 / 564 (+4%) | 350 / 338 (+3%) |
| 1000 | 0.993 | 408 / 399 (+2%) | 236 / 232 (+2%) |

- **A local client cost both HNSW arms 12–31% below 0.98 recall**, falling to
  2–4% at 0.99 and above, where each query spends far longer in the server than
  in the client.
- **The network hop itself is negligible.** The round-trip floor is 0.314 ms
  local and 0.340 ms over the LAN at 1 client (+26 µs). At 8 clients it is
  0.41 ms either way. The gain is the freed cores, not the network.
- **PRISM's 8-client curve rose the same way** at low `nprobe` (+38% at
  `nprobe=5`, +25% at 10) and is unchanged from `nprobe=40` up. Its two
  placements were measured in separate server sessions rather than back to
  back.
- **Conclusions at 0.99 hold either way.** Below ~0.98, the original
  8-client Postgres curves understated every Postgres engine.

The original 8-client table (client on node-0), for reference:

| engine | 0.90 | 0.95 | 0.98 | 0.99 | 0.995 |
|---|---:|---:|---:|---:|---:|
| pgvector HNSW | 1,818 | 1,143 | 615 | 315 | – |
| pgvectorfast HNSW | 3,059 | 2,018 | 889 | 530 | – |
| PRISM | 1,263 | 999 | 463 | 287 | 186 |

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
# workstation, with both nodes set up and node-0's indexes built:
bench/cloudlab/remote_client.sh    # client placement, 8 clients
bench/cloudlab/scaling.sh          # 12 and 16 clients, client on node-1
bench/cloudlab/sync.sh pull
bench/plot_variants.py dbpedia_openai_1m_1536_angular [--clients 8|12|16]
```

`remote_client.sh` and `scaling.sh` run each step detached on its node and
record its completion in `/mydata/steps/`. A dropped connection or a sleeping
workstation only pauses them; re-running continues after the last finished
step.

## Open

- Confirm what stops pgvectorfast scaling past 8 clients at high recall:
  measure DRAM bandwidth during the 8- and 16-client sweeps (`perf stat` with
  uncore counters, after lowering `perf_event_paranoid`).
- Separate Ginger's PCA from its float rerank with a third config (PCA on,
  float rerank). The float-rerank config changes both, and its 1.5–1.7×
  slowdown most likely comes from dropping PCA.

- PRISM's posting-scan cost at 1536-d on AVX2: profile it
  (`perf_event_paranoid` is 4 on these nodes and needs lowering), and try
  `fastscan` settings or `vec16`.
- A second PRISM round, to bound its noise the way the HNSW arms are bounded.
- Ginger on a VNNI / AVX-512 machine, where TurboQuant has its fast path.
  These numbers understate it.
- What TOAST costs the Postgres engines: rerun with
  `ALTER TABLE ... ALTER COLUMN embedding SET STORAGE PLAIN`. A 6 KB row still
  fits an 8 KB page, so the vectors would move back into the heap.
