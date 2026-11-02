# pgvector ANN benchmark: initial results (2026-11-01)

First baseline numbers from `bench/ann_bench.py`: recall@10 against throughput and
latency for pgvector's HNSW and IVFFlat indexes, on SIFT-1M and a synthetic
dataset. These numbers are the reference point for later changes to pgvector.

## Summary

- **HNSW on SIFT-1M** (m=16, ef_construction=64) reaches **0.93 recall@10 at ~990 QPS**
  (ef_search=40, p50 0.97 ms) and **0.99 recall at ~380 QPS** (ef_search=160) with one
  client. The build took 54 s with 7 parallel workers, and the index is 782 MiB.
- **IVFFlat on SIFT-1M** (lists=1000) needs probes=16 to reach 0.92 recall, at 185 QPS.
  That is about 5x slower than HNSW at the same recall. The build is much faster
  (8 s vs. 54 s).
- **Throughput scales with clients.** HNSW at ef_search=40 goes from 990 QPS (1 client)
  to 3,614 QPS (4 clients, 3.7x) and 5,752 QPS (8 clients, 5.8x) on 8 cores / 16 threads.
- **Exact search** (sequential scan) on SIFT-1M runs at 10.7 QPS (94 ms per query).
  HNSW at 0.99 recall is about 35x faster.
- **The HNSW search is bound by memory and the buffer manager, not by vector math**
  (see [CPU profile](#cpu-profile-hnsw-on-sift-1m)). About 45% of backend CPU goes to
  `ReadBuffer`, finding and pinning index pages that are already in `shared_buffers`.
  About 28% goes to touching the element itself, which is mostly cache misses. The L2
  distance computation is under 5%.

## Setup

**Hardware / OS**
- AMD Ryzen AI 7 PRO 350 laptop: 8 cores / 16 threads, up to 5.1 GHz, with frequency
  scaling on (default governor)
- 26 GiB RAM, NVMe SSD
- Ubuntu 26.04.1 LTS, Linux 7.0.0-31-generic

**Software**
- PostgreSQL 17.11 built from source by `build_postgres.sh`: meson `buildtype=debugoptimized`
  (`-O2 -g`), assertions off (`cassert=false`), ICU disabled
- pgvector v0.8.7 (`f37c13f`, git submodule, unmodified), built with pgvector's default
  flags (`-march=native -ftree-vectorize -fassociative-math -ffp-contract=fast ...`)
- Client: Python 3.13, psycopg 3.3.6 (binary), pgvector-python 0.5.1, numpy 2.5.4.
  The client runs on the same machine and connects over TCP to localhost.

**Postgres configuration**: an `initdb` cluster with stock defaults, except:

| Setting | Value | Where it was set |
|---|---|---|
| `shared_buffers` | 4GB | server command line (`PG_OPTS`). Large enough for the whole SIFT table and index |
| `port` | 5433 | command line. Another Postgres already used 5432 on this machine |
| `listen_addresses` | localhost | command line (`run_postgres.sh`) |
| `maintenance_work_mem` | 2GB | session, during `CREATE INDEX` only |
| `max_parallel_maintenance_workers` | 7 | session, during `CREATE INDEX` only |
| `hnsw.ef_search` / `ivfflat.probes` | the value being tested | session |

Everything else is at its default. That includes `work_mem=4MB`, `max_worker_processes=8`,
`max_parallel_workers=8`, `max_parallel_workers_per_gather=2`, `jit=on` and
`effective_cache_size=4GB`. `hnsw.iterative_scan` and `ivfflat.iterative_scan` are off.

**Method**
- Each query is `SELECT id FROM items_<dataset> ORDER BY embedding <-> $1 LIMIT 10`,
  sent as a protocol-level prepared statement.
- Before measuring, the script runs `EXPLAIN` and checks that the plan uses the intended index.
- There is no explicit cache prewarm. The data had just been loaded and indexed, and it
  all fits in `shared_buffers`.
- Latency is measured on the client around `execute` + `fetchall` for each query, so it
  includes the localhost round trip.
- QPS = queries / wall time. Each sweep value runs all 1,000 test queries (100 for the
  exact scan). With multiple clients, the queries are split evenly across connections
  that start together.
- recall@10 = |returned ids ∩ true top-10| / 10, averaged over the queries. The true
  neighbors are the ones shipped in the ann-benchmarks file for SIFT, and brute-force
  numpy results for the synthetic dataset.
- Each configuration was run **once**. See Caveats for what that means for noise.

## Results

### SIFT-1M (`sift-128-euclidean`, 1,000,000 x 128, L2, 1,000 queries)

Loading the table took 2.4 s with binary `COPY`.

**HNSW**, m=16, ef_construction=64. Build 54.1 s (7 workers), index size 820,289,536 bytes (782 MiB). 1 client.

| ef_search | recall@10 | QPS | p50 ms | p95 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.7248 | 1,743.9 | 0.510 | 0.958 | 1.606 |
| 20 | 0.8475 | 1,175.2 | 0.761 | 1.372 | 2.010 |
| 40 | 0.9286 | 989.8 | 0.972 | 1.506 | 1.879 |
| 80 | 0.9734 | 622.1 | 1.530 | 2.448 | 2.920 |
| 160 | 0.9931 | 380.8 | 2.618 | 3.839 | 5.001 |
| 320 | 0.9976 | 196.0 | 5.060 | 7.193 | 8.613 |

**HNSW, concurrent clients** (same index):

| clients | ef_search | recall@10 | QPS | speedup vs. 1 client | p50 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 0.9286 | 989.8 | 1.0x | 0.972 | 1.879 |
| 4 | 40 | 0.9286 | 3,613.7 | 3.7x | 0.946 | 2.876 |
| 8 | 40 | 0.9286 | 5,752.2 | 5.8x | 1.091 | 3.099 |
| 1 | 160 | 0.9931 | 380.8 | 1.0x | 2.618 | 5.001 |
| 4 | 160 | 0.9931 | 1,507.7 | 4.0x | 2.532 | 4.447 |
| 8 | 160 | 0.9931 | 2,122.2 | 5.6x | 3.067 | 6.153 |

**IVFFlat**, lists=1000. Build 8.0 s. 1 client.

| probes | recall@10 | QPS | p50 ms | p95 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3694 | 2,059.1 | 0.421 | 1.013 | 1.318 |
| 2 | 0.5297 | 969.5 | 0.920 | 1.895 | 2.269 |
| 4 | 0.6839 | 588.2 | 1.502 | 3.234 | 4.518 |
| 8 | 0.8253 | 360.4 | 2.477 | 5.046 | 6.532 |
| 16 | 0.9243 | 184.7 | 5.034 | 9.493 | 11.726 |
| 32 | 0.9758 | 62.4 | 15.536 | 24.381 | 28.227 |
| 64 | 0.9945 | 37.5 | 25.564 | 37.710 | 41.109 |

**Exact (no index, sequential scan)**, 100 queries: recall 1.0, **10.7 QPS**, p50 94.0 ms, p99 99.0 ms.

### Synthetic (`synthetic:100000x128`, seed 42, L2, 1,000 queries)

Overlapping Gaussian clusters (316 cluster centers ~ N(0,1), unit noise). This dataset is
small and fast to set up, so it is meant for quick checks, not for comparison with
published numbers.

**HNSW**, m=16, ef_construction=64. Build 3.1 s.

| ef_search | recall@10 | QPS | p50 ms | p95 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.6858 | 5,291.4 | 0.148 | 0.389 | 0.610 |
| 20 | 0.8381 | 4,288.7 | 0.179 | 0.544 | 0.742 |
| 40 | 0.9486 | 1,940.4 | 0.457 | 0.871 | 1.306 |
| 80 | 0.9886 | 1,328.0 | 0.679 | 1.244 | 1.543 |
| 160 | 0.9986 | 1,468.3 | 0.599 | 1.178 | 1.558 |

**IVFFlat**, lists=100 (the default, rows/1000). Build 0.3 s.

| probes | recall@10 | QPS | p50 ms | p95 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9264 | 1,536.7 | 0.537 | 1.499 | 2.071 |
| 2 | 0.9587 | 1,011.4 | 0.967 | 1.546 | 2.087 |
| 4 | 0.9756 | 470.4 | 2.118 | 2.772 | 3.083 |
| 8 | 0.9901 | 186.4 | 5.142 | 7.326 | 8.425 |
| 16 | 0.9975 | 63.2 | 15.790 | 21.054 | 22.811 |
| 32 | 0.9999 | 37.7 | 26.795 | 31.735 | 34.139 |

## Observations and open questions

- **HNSW matches published results.** On SIFT-1M with these build parameters, recall of
  about 0.93 at ef_search=40 and 0.99 at ef_search=160 is in line with pgvector's
  published results for the same parameters.
- **IVFFlat slows down more than expected at high probes.** On SIFT, going from probes=16
  to 32 should double the vectors scanned. Instead QPS drops 3x (185 → 62), and synthetic
  data shows the same jump at 8 → 16. One possible cause is the candidate sort spilling
  past `work_mem=4MB`, but this has **not been verified**. The next step is to re-run with
  a larger `work_mem` and with `EXPLAIN (ANALYZE, BUFFERS)`.
- **Some results are not monotonic.** On synthetic data, HNSW at ef_search=160 measured
  faster than at 80 (1,468 vs. 1,328 QPS). That shouldn't happen, so it is probably
  run-to-run noise (see Caveats).
- **Concurrency scales almost linearly to 4 clients.** At 8 clients it is still 5.6–5.8x,
  which is reasonable given 8 physical cores, with the Python client processes competing
  for the same CPUs.

## CPU profile: HNSW on SIFT-1M

**Flamegraph:** [`report/sift-hnsw-ef40-flamegraph.svg`](report/sift-hnsw-ef40-flamegraph.svg).
Open it in a browser to zoom and search.

![HNSW query flamegraph](report/sift-hnsw-ef40-flamegraph.svg)

**What was profiled**
- HNSW (m=16, ef_construction=64), `ef_search=40`, k=10, all 10,000 SIFT test queries,
  one client. The run got recall 0.935 at 1,405 QPS (p50 0.72 ms).
- `perf record -F 999 --call-graph fp`, attached to the postmaster. The query backend it
  starts inherits the profiling, and the Python client is not profiled. 7,085 samples.
- Postgres and pgvector were rebuilt with frame pointers
  (`FRAME_POINTERS=1 ./build_postgres.sh`, which adds `-fno-omit-frame-pointer
  -mno-omit-leaf-frame-pointer`). perf's DWARF unwinding only reconstructed about 25% of
  backend stacks on this build. With frame pointers, 100% unwind to `main`.
- **Everything was in memory.** `bench/profile.sh` runs `pg_prewarm` on the table and
  index, then checks with `pg_buffercache` before and after the run. All 71,488 table
  blocks and 100,114 index blocks were in `shared_buffers` (4GB) both times. Without the
  prewarm, right after a server restart (9% of the index cached), the same run managed
  958 QPS instead of 1,405.

**Where the backend's CPU time goes** (inclusive % of samples):

| % | Function / area |
|---:|---|
| 96.8 | `exec_execute_message`: running the query. Protocol, bind, planning and executor overhead are about 3% in total |
| 94.7 | `HnswSearchLayer`, the graph search |
| 79.5 | `HnswLoadElementImpl`, which runs for every element visited (`hnswutils.c:538`) |
| 44.5 | ↳ `ReadBuffer`, finding and pinning the element's index page (always a buffer hit here) |
| 21.2 | ↳↳ hashing the page's ID for the buffer table (`hash_bytes` + inlined `pg_rotate_left32`) |
| 10.6 | ↳↳ atomic compare-and-swap on the buffer header (`PinBuffer`) |
| 28.4 | ↳ reading the element tuple (`PageGetItem` 7.8 + `HnswLoadElementImpl` self 20.6) |
| 4.5 | ↳ `UnlockReleaseBuffer` |
| 4.9 | ↳ distance: `vector_l2_squared_distance` 4.6 + fmgr call overhead 0.3 |
| 3.1 | `HnswLoadUnvisitedFromDisk`, reading neighbor lists |
| 2.3 | candidate heaps (`pairingheap_*`) |
| 1.8 | visited set (`tidhash`) |

**Hardware counters.** These come from `perf stat` on a separate 10,000-query run, on
the earlier build without frame pointers, also fully cached:

| Counter | Total | Per query |
|---|---:|---:|
| cycles | 31.3 G | 3.13 M |
| instructions | 16.6 G (**IPC 0.53**) | 1.66 M |
| cache-misses | 266 M | 26.6 K |
| L1-dcache-load-misses | 245 M (2.6% of loads) | 24.5 K |
| dTLB-load-misses | 36.7 M (53% of `dTLB-loads`) | 3.7 K |

**Interpretation**
- Each element the search visits is on a different, effectively random 8 KB page of a
  782 MiB index. Reaching it means hashing the page ID, looking it up in the shared
  buffer table, pinning with an atomic operation, locking, reading the line pointer and
  tuple, then unlocking and unpinning. Almost all of that is waiting on cache and TLB
  misses, which is why IPC is 0.53. The vector math is a small fraction.
- `huge_pages=try` is not in effect (`huge_pages_status=off`, `vm.nr_hugepages=0`), so
  the 4GB of `shared_buffers` is mapped with 4 KB pages. That fits the high dTLB miss
  rate.
- When the CPU stalls on memory, perf can attribute the stall to an instruction near
  the load that missed. So the split between `hash_bytes`, the pin CAS and `PageGetItem`
  is approximate. The total share for buffer access (about 75% of the search) is reliable.

**Ideas to test next.** None of these have been tried yet.
1. Huge pages for `shared_buffers` (reserve `vm.nr_hugepages`, set `huge_pages=on`) to
   cut TLB misses.
2. Prefetch neighbor pages (`PrefetchBuffer` / `__builtin_prefetch`) before loading them,
   so the misses overlap instead of happening one after another.
3. Reduce buffer-manager round trips for each element, for example by keeping pins
   across neighbors on the same page or caching hot upper-layer elements in memory.

## Caveats

- **Each configuration ran once on a laptop with CPU frequency scaling on**, so expect
  roughly ±10% noise in QPS. Before drawing conclusions from small differences, repeat
  the runs, pin the governor to `performance`, and compare medians.
- The benchmark client runs on the same machine as the server. At 8 clients the Python
  processes take CPU away from Postgres backends.
- Latency is measured on the client and includes the round trip and Python overhead,
  about 0.05–0.1 ms going by the fastest queries above.
- Postgres was built `debugoptimized` (`-O2 -g`), not `release`. The CPU profile used a frame-pointer build, which is typically ~1–2% slower. Assertions are off, so
  the difference should be small, but it is not a production-tuned build.
- `shared_buffers=4GB` holds the whole SIFT table and index, so these are in-memory
  numbers. Datasets larger than memory will behave differently.

## How to reproduce

Running these steps in order reproduces every number above (up to noise).
Order matters: the benchmark keeps only one ANN index per table, so switching index
type drops the previous one.

```sh
# 1. Clone and build Postgres 17.11 + pgvector v0.8.7 into ./postgres/install
git clone --recurse-submodules git@github.com:awitten1/vectordb_experiments.git
cd vectordb_experiments
./build_postgres.sh            # needs meson, ninja, a C compiler, curl

# 2. Python environment for the benchmark
python3 -m venv bench/.venv
bench/.venv/bin/pip install -r bench/requirements.txt

# 3. Start Postgres with 4GB shared_buffers (use any free port; this run used 5433)
export PGPORT=5433
PG_OPTS="-c shared_buffers=4GB" ./run_postgres.sh

# 4. Run the benchmarks (SIFT-1M is downloaded automatically, ~525 MB, into bench/data/)
P=bench/.venv/bin/python
D=sift-128-euclidean

# HNSW sweep (loads the table, builds the index: ~1 min)
$P bench/ann_bench.py --dataset $D --index hnsw --m 16 --ef-construction 64 --sweep 10,20,40,80,160,320
# HNSW concurrency (reuses the index)
$P bench/ann_bench.py --dataset $D --index hnsw --sweep 40,160 --clients 4
$P bench/ann_bench.py --dataset $D --index hnsw --sweep 40,160 --clients 8
# IVFFlat (drops the HNSW index)
$P bench/ann_bench.py --dataset $D --index ivfflat --lists 1000 --sweep 1,2,4,8,16,32,64
# Exact baseline
$P bench/ann_bench.py --dataset $D --index none --num-queries 100

# Synthetic dataset
$P bench/ann_bench.py --dataset synthetic:100000x128 --index hnsw --sweep 10,20,40,80,160
$P bench/ann_bench.py --dataset synthetic:100000x128 --index ivfflat --sweep 1,2,4,8,16,32
```

Results are printed and appended to `bench/results/<dataset>.jsonl`. Each line has the
full configuration, including the pgvector git SHA. To get the build time on a run that
would otherwise reuse an existing index, add `--rebuild`. See `bench/README.md` for all
options.

### Reproducing the CPU profile

```sh
# Allow perf for your user and show kernel symbols (both reset on reboot)
sudo sysctl -w kernel.perf_event_paranoid=1 kernel.kptr_restrict=0

# Rebuild with frame pointers (full Postgres rebuild, a few minutes) and restart
./run_postgres.sh stop
FRAME_POINTERS=1 ./build_postgres.sh
PG_OPTS="-c shared_buffers=4GB" ./run_postgres.sh

# Prewarm, check residency, perf record, render the flamegraph
# (fetches Brendan Gregg's FlameGraph scripts at a pinned commit into bench/.tools/)
NAME=sift-hnsw-ef40 bench/profile.sh --dataset sift-128-euclidean --index hnsw \
    --m 16 --ef-construction 64 --sweep 40 --num-queries 10000
# -> bench/profiles/sift-hnsw-ef40.{svg,folded,perf.data,residency.txt,bench.txt}

# Function-level self time
perf report -i bench/profiles/sift-hnsw-ef40.perf.data --no-children --sort symbol --stdio -g none

# Hardware counters for a separate run (Ctrl-C or kill -INT perf when the run finishes)
perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,dTLB-loads,dTLB-load-misses \
    -p "$(head -1 postgres/data/postmaster.pid)" &
bench/.venv/bin/python bench/ann_bench.py --dataset sift-128-euclidean --index hnsw --sweep 40 --num-queries 10000 --no-save
kill -INT %1
```

To go back to the normal build without frame pointers, run `./build_postgres.sh` again
without `FRAME_POINTERS`.
