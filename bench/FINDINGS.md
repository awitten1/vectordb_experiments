# HNSW on pgvector: where the time goes, and what moves it

Working notes from the 2026-11-02/04 session on an 18M-vector BIGANN index.
`bench/REPORT.md` is the SIFT-1M report that predates this; the numbers here
supersede it. Everything below is measured on this machine unless flagged as an
estimate.

## Setup

- AMD Ryzen 9 9900X, 12 cores / 24 threads, 64 MiB L3 (2 instances), 1 NUMA node
- 59 GiB RAM, 3.6 TB disk, Linux 7.0
- PostgreSQL 18.6 (`debugoptimized`), pgvector v0.8.7 (`-march=native`),
  pg_vectorsearch 0.2.0-dev. Earlier numbers in this file were taken on 17.11;
  where both exist they agree (HNSW `ef_search=320`: 0.9893 / 201 QPS on 17.11,
  0.9891 / 204 QPS on 18.6).
- `shared_buffers=28GB`; table and index fully resident, asserted via `pg_buffercache`
- BIGANN (SIFT1B) first 18M vectors, 128-d uint8 → float4, k=10, 2,000 distinct queries
- Client: Python + psycopg, prepared statements, same machine

Sizes: table 1,285,760 blocks (9.81 GiB), index 1,828,467 blocks (13.95 GiB) —
**23.8 GiB total**, which is how 18M rows was chosen. That works out to 832 bytes
of index per element, or 9.84 elements per index block averaged over element and
neighbour pages together.

**Dataset choice matters more than it looks.** The synthetic uniform-random
vectors used first are not a stand-in for real data: their distance distribution
is nearly flat (d100/d1 = 1.104, i.e. the 100th nearest neighbour is only 10%
further away than the 1st), so every graph edge is almost equally good and recall
saturates at trivial `ef_search`. Real SIFT descriptors have structure to find.
Any ANN result on uniform random vectors should be discarded.

## Building the index: the DSM backing type is decisive

A parallel HNSW build puts the whole graph in dynamic shared memory, and
`dynamic_shared_memory_type` decides what that memory actually is:

| setting | backing | outcome here |
|---|---|---|
| `mmap` | a file in `$PGDATA` | **32 tuples/s, 71-hour ETA** |
| `posix` | `/dev/shm` | capped by the mount size *and* the per-user `usrquota` (~23 GiB here) |
| `sysv` | System V shm, RAM | **27 minutes** |

That is a **~1,100× difference** from one GUC. With `mmap` the build is bound by
random writeback to a rotational disk, and no amount of `maintenance_work_mem`
helps — the memory is not the constraint, its backing store is. `posix` fails
less obviously: `/dev/shm` can look big enough while a user quota silently caps
the allocation, so check `repquota` and not just `df`. `sysv` is governed by
`kernel.shmmax` / `shmall`, which are effectively unlimited on a modern kernel.

`bench/run_experiment.sh` now runs a `shmget` preflight before the build and
refuses to proceed with `mmap`.

## Query performance

Single client, 60 s per point, warm backend, stock pgvector:

| ef_search | recall@10 | QPS | µs/query |
|---:|---:|---:|---:|
| 10 | 0.6212 | 2,551 | 392 |
| 20 | 0.7494 | 1,721 | 581 |
| 40 | 0.8541 | 1,092 | 916 |
| 80 | 0.9281 | 636 | 1,573 |
| 160 | 0.9683 | 363 | 2,755 |
| 320 | 0.9893 | 201 | 4,973 |

Plotted in `bench/report/bigann-18000000-hnsw-recall-qps.svg`.

**At one client, QPS is exactly `1/latency`** — Little's Law with a single
in-flight request. So the single-client QPS axis of a recall/QPS chart is a
latency axis wearing a disguise, and comparing it against a vendor benchmark run
with 32 clients is meaningless. ann-benchmarks is single-client by default;
vendor numbers usually are not.

Concurrency, `ef_search=40`, measured on the earlier single-pass protocol (treat
as indicative, the absolute values are ~25% low):

| clients | QPS | scaling |
|---:|---:|---:|
| 1 | 793 | 1.00× |
| 4 | 1,957 | 2.47× |
| 8 | 2,943 | 3.71× |

Sub-linear on a 24-thread box, as expected for a memory-latency-bound workload
contending on one buffer pool.

## The profile: 51% of cycles are inside `ReadBuffer`

`perf record --call-graph fp`, 60 s at `ef_search=40`, warm backend, inclusive
percentages of cycles:

| % | symbol |
|---:|---|
| 96.1 | `HnswSearchLayer` |
| 76.1 | ↳ `HnswLoadElementImpl` |
| 51.3 | ↳ `ReadBuffer` |
| 38.1 | ↳ `BufTableLookup` |
| 37.6 | ↳ `hash_search_with_hash_value` |
| 1.0 | page fault handling |

Every element visited costs one `ReadBuffer` on a page that is **already in
`shared_buffers`**. There is no I/O. The cost is the buffer manager's own
bookkeeping — hash the `BufferTag`, probe a ~1.8M-entry shared hash table, pin
the buffer — and that hash probe is a random access into ~100 MB of metadata,
so it misses to DRAM. Distance computation is under 5%.

### The page-fault finding, and how it nearly misled us

A first profile showed **31.6% of cycles in page-fault handling**
(`handle_mm_fault` 26.8%, `filemap_map_pages` 23.9%). That looked like a major
discovery. It was an artefact of the measurement window: the profile covered a
single pass over 1,000 queries on a **brand-new backend**, so the per-process
cost of populating page tables for the 28 GiB `shared_buffers` mapping landed
inside the measured region.

Settled by measuring `min_flt` directly on the backend: **243,503 minor faults**
on the first pass, then **exactly 0** on every pass after. ~1.4 s of one-time
cost per connection. The steady-state profile above shows 1.0%.

Two things worth keeping:

- `shared_memory_type=mmap` maps `/dev/zero` `MAP_SHARED`, which becomes an
  internal shmem/tmpfs object, so faults go through `do_read_fault` /
  `filemap_map_pages` — the *file*-backed path, which is why the stacks look like
  I/O even though nothing touches a disk.
- **Fault-around hurts here.** `filemap_map_pages` populates ~16 PTEs (64 KiB)
  per fault on the assumption of sequential access. HNSW access is random, so
  most of that work is wasted. It still beats one fault per page overall, but it
  is the wrong heuristic for this workload.
- `huge_pages=try` silently falls back, and shmem THP has a *separate* knob:
  `/sys/kernel/mm/transparent_hugepage/shmem_enabled`, which is `[never]` here.
  Worth revisiting — 28 GiB in 4 KiB pages is 7M PTEs and about half of dTLB
  loads miss.

## Four attempts at the buffer-manager overhead

**Measure the ceiling before writing the optimization.** Three of these four were
settled by a counting run that took minutes, and two of those would have been
days of work for nothing.

### 1. Block-sorted batch loading — dead

Idea: sort the unvisited neighbours by block number and hold a page cursor, so
neighbours sharing a page reuse one pin instead of one `ReadBuffer` each.
Implemented (`HnswPageCursor`, insertion sort of `unvisited[]`) and measured no
gain. The instrumented count says why: **1.000 neighbours per distinct block.**
No two neighbours of any node ever share a page.

That is the signature of placement being uncorrelated with graph adjacency.
Elements are written in insertion order, which for BIGANN is file order, which is
random with respect to vector space. Nothing to batch.

### 2. `ReadRecentBuffer` cache — we built the wrong thing

Idea: remember the last buffer and skip the buffer table on a repeat.
**Hit rate 0.79%.** We called it dead. That was wrong, and the error was in the
data structure, not the idea.

`pg_vectorsearch` ships the same optimization as `prism.recent_buffers`, on by
default, and its comment (`src/pg/bufstorage.c:43`) reports our bottleneck
independently: *"~22% of warm query CPU is BufTableLookup hash probes inside
ReadBuffer, for index pages that essentially never leave shared_buffers."*

The difference is that they do not cache the *recent* buffers. They keep a
**dense array indexed by block number** — one 4-byte `Buffer` per block of the
relation, per backend, 11 MB for a 21 GB index — and re-pin through
`ReadRecentBuffer`, which validates the tag and pins without touching the
mapping table. There is no eviction policy to get wrong: a stale id fails
validation and falls back to `ReadBuffer`, which refreshes the slot, so
correctness never depends on the cache.

On a fully resident index that turns nearly every `BufTableLookup` into an array
index. Our 0.79% measured how often the *last* buffer was reused, which for a
random graph walk is almost never; it says nothing about how often the block is
already in the pool, which is always.

They also measured something we did not: **38% of reads in a fresh backend, and
14% even warmed, miss purely because a slot was never populated.** So they seed
the array eagerly by sweeping all `NBuffers` buffer descriptors for tags
matching the relation, reading tags without the header lock because
`ReadRecentBuffer` re-validates anyway.

`ReadRecentBuffer` is a public API (`bufmgr.h:206`), so unlike the prefetch in
attempt 4 this needs **no core patch**. It attacks the whole 38%, not the single
DRAM miss the prefetch hides. Untried on pgvector HNSW; it should be the next
thing tried.

### 3. Graph-aware page layout — withdrawn before implementing

Idea: reorder the index so a node and its neighbours co-locate. The arithmetic
kills it at 8 KiB blocks: ~10 elements per page against ~21.5 neighbours per
node, so even a *perfect* graph partition leaves most neighbours off-page —
about a 42% ceiling, for a graph-partitioning rewrite. This is the one that
larger blocks would change (see below).

### 4. Buffer-table hash prefetch — **+14% QPS**

The one that worked, and it is ~30 lines. After `HnswLoadUnvisitedFromDisk`
produces the unvisited list, compute the `BufferTag` and hash for all ~21
neighbours and issue a `prefetcht0` for each bucket slot, so the misses overlap
instead of serializing:

```c
static inline void
HnswPrefetchUnvisited(Relation index, HnswUnvisited *unvisited, int unvisitedLength)
{
    for (int i = 0; i < unvisitedLength; i++)
    {
        BufferTag tag;
        InitBufferTag(&tag, &index->rd_locator, MAIN_FORKNUM,
                      ItemPointerGetBlockNumber(&unvisited[i].indextid));
        BufTablePrefetch(BufTableHashCode(&tag));
    }
}
```

`BufTablePrefetch` / `hash_prefetch` do not exist in Postgres, so this needs a
small core patch (`dynahash.c`, `buf_table.c`, two headers) — but it is a
non-faulting hint with no locking, so **the extension side stays an extension**.

| ef_search | stock QPS | prefetch QPS | Δ |
|---:|---:|---:|---:|
| 10 | 2,551 | 2,873 | +12.6% |
| 20 | 1,721 | 1,958 | +13.8% |
| 40 | 1,092 | 1,240 | +13.5% |
| 80 | 636 | 727 | +14.3% |
| 160 | 363 | 411 | +13.2% |
| 320 | 201 | 230 | +14.2% |

At `ef_search=40`: 916 → 807 µs/query. Over ~1,171 element loads per query that
is **≈93 ns saved per load — one DRAM round trip.**

### Why it is +14% and not 2×

dynahash has three levels, and two of them are *just pointers*, which makes it
easy to talk yourself into thinking the prefetch covers more than it does:

```c
typedef struct HASHELEMENT {
    struct HASHELEMENT *link;   /* next entry in the same bucket */
    uint32  hashvalue;
} HASHELEMENT;                  /* then the key (BufferTag), then the data */

typedef HASHELEMENT *HASHBUCKET;    /* a bucket IS a pointer */
typedef HASHBUCKET  *HASHSEGMENT;   /* a segment is an array of those pointers */
```

The lookup chain is:

1. `hashp->dir[segment_num]` → the segment. Small, stays cached.
2. `&segp[segment_ndx]` → the address of **one 8-byte pointer slot**. Computable
   from the hash alone. This is what the prefetch touches. Loading it is miss #1.
3. `currBucket = *prevBucketPtr` (`dynahash.c:1011`) → dereferences that slot to
   reach the `HASHELEMENT`, which lives in a separate pool. **Miss #2.**
4. Compare `hashvalue` and the `BufferTag`; on a collision follow `link` and miss
   again.

Step 3's *address is the value loaded in step 2*, so a prefetch cannot reach it.
That is the definition of a pointer chase. One of the two misses is hidden, which
is exactly the ≈93 ns measured.

A three-pass version (hash all 21 → prefetch all slots → load all 21 head
pointers → prefetch all elements → look up) could in principle hide the second
one too. Untested; ~10 more lines.

## Larger block sizes: what they would and would not fix

Considered as a way to revive attempts 1 and 2.

**64 KiB is not available. Postgres caps `BLCKSZ` at 32 KiB**
(`configure.ac:266`, allowed values 1/2/4/8/16/32 kB). The limit is structural:
`ItemIdData.lp_off` and `lp_len` are 15-bit fields, so an offset must fit in
32767. Going past it means changing the line-pointer format.

So the experiment would be 4×, not 8×, and:

- **Attempt 1 stays dead.** Random placement is scale-free in this respect. At
  32 KiB the index holds ~39 elements per page over ~461k element pages, so the
  expected number of same-page pairs among 21.5 neighbours is
  C(21.5,2)/461k ≈ **0.0005**. Still 21.5 distinct blocks.
- **Attempt 2 stays marginal.** The 0.79% recent-buffer hit rate should scale
  ~4× → ~3%, worth under 1% QPS. Below the noise floor.
- **Attempt 3's ceiling changes a lot.** At ~39 elements per page a node plus all
  ~21.5 of its neighbours could fit on one page, so graph-aware layout goes from
  a 42% ceiling to potentially most of the 21.5 fetches collapsing into one.
  Big blocks are only interesting *together with* reordering.
- **The real win is unrelated to 1 and 2:** the buffer manager's own metadata
  shrinks 4×, which attacks the same misses the prefetch attacked — by making
  them rarer rather than cheaper.

| | 8 KiB | 32 KiB (est.) |
|---|---:|---:|
| index blocks | 1,828,467 | ~461,000 |
| `BufferDesc` array (64 B/ea) | 117 MB | 29 MB |
| buffer hash table (~56 B/entry) | ~102 MB | ~26 MB |
| per-page waste (832 B tuples) | 6.6% | 1.6% |

With 64 MiB of L3, a 26 MB hash table has a real chance of staying resident
where a 102 MB one does not. Estimated 5–15%, untested.

Cost: `BLCKSZ` is recorded in `pg_control`, so a 32 KiB binary cannot read the
existing cluster. Recompile Postgres and pgvector, fresh `initdb`, reload,
rebuild (~2–3 h wall clock), and WAL full-page writes get 4× bigger. Should be
done in a **second `PGDATA`** so the 8 KiB cluster survives as a warm baseline.

## PRISM (pg_vectorsearch) against HNSW, same table

`pg_vectorsearch` 0.2.0-dev (`753e3e74`) builds a PRISM index — IVF-style, with
a hierarchical centroid tree, RaBitQ-quantized posting lists, and two-stage
search where 1-bit estimates with error bounds prune candidates before exact
rerank from the heap. It is the opposite design point from a graph index.

It indexes the **existing pgvector column with no rewrite and no copy**: `vec32`
is binary-compatible with `vector`, and the extension installs a
`WITHOUT FUNCTION` cast, which satisfies PostgreSQL's binary-coercibility rule
for operator classes.

### Build and size

| | HNSW (m=16, efc=64) | PRISM (all auto) |
|---|---:|---:|
| build | 1,811 s | **170 s** |
| size | 14 GB / 1,828,406 blk | **1,345 MB / 172,163 blk** |
| peak build DSM | the whole graph | **26.7 MB** |

Both ratios are 10.7x, coincidentally. Resolved settings: `nlist=26569`,
`fan_out=163`, `nlevels=2`, `centroid_format=fastscan`, auto `nprobe=82`.

Build phases: k-means 64.2 s (root 31.7 + subtrees 32.5), parallel table scan
85.6 s, WAL logging 17.1 s, finalize 2.4 s.

The DSM line is the one that matters operationally. HNSW's build puts the entire
graph in dynamic shared memory, which is what made `dynamic_shared_memory_type`
a 1,100x correctness-of-measurement trap (above). PRISM's build peaks at 26.7 MB
and simply does not have that failure mode.

One cost worth naming: `replicated 16,723,756 vectors (92.9% of 18,000,000,
lambda=1)`. SOAR plus the 0.35 boundary band nearly doubles the posting entries.
That is where the posting pages go, and it is what buys recall at a given
`nprobe`.

### Recall vs throughput

![recall vs QPS](report/bigann-18000000-hnsw-prism-recall-qps.svg)

Single client, 60 s per point, index fully resident, PostgreSQL 18.6.

| prism.nprobe | recall@10 | QPS | | hnsw.ef_search | recall@10 | QPS |
|---:|---:|---:|---|---:|---:|---:|
| 5 | 0.6439 | 6,433 | | 10 | 0.6227 | 2,617 |
| 10 | 0.7838 | 4,505 | | 20 | 0.7532 | 1,762 |
| 20 | 0.8826 | 3,058 | | 40 | 0.8551 | 1,116 |
| 40 | 0.9437 | 2,074 | | 80 | 0.9263 | 650 |
| 80 | 0.9781 | 1,382 | | 160 | 0.9686 | 365 |
| 160 | 0.9923 | 837 | | 320 | 0.9891 | 204 |
| 320 | 0.9970 | 473 | | | | |

**PRISM is roughly 3-5x faster at matched recall, and the gap widens as recall
rises** (interpolating prism onto HNSW's recall points: ~3.1x at 0.855, ~3.6x at
0.926, ~4.3x at 0.969, ~4.7x at 0.989). It also reaches 0.997, which the HNSW
sweep never does — `ef_search` would have to go well past 320.

A single scan at `nprobe=82`: 12.3 ms, 82 posting lists, 106,000 posting entries
scanned, **96 rerank candidates for 10 rows**. The RaBitQ bounds pruned 106,000
to 96.

### Forcing one index or the other

The two arms are not symmetric, and the asymmetry decides the protocol.

`pg_vectorsearch_setup_pgvector_compat()` adds pgvector's `<->`, `<#>` and `<=>`
to prism's operator families *as well as* prism's own, so `embedding <-> $1` is
a candidate for **either** index and the planner decides on cost. The reverse
does not hold: pgvector's `vector_l2_ops` has no `vec32` member, so
`embedding::vec32 <-> $1::vec32` can only reach prism. The cast is
`WITHOUT FUNCTION`, hence a zero-cost relabel, not a conversion.

- **prism** is forced by operand type and is immune to a cost flip.
- **HNSW** cannot be forced that way, and raising `ef_search` raises its
  estimated cost until prism wins. It has to be the only ANN index on the table.

So `phase_compare` runs prism first with both indexes present, then HNSW with
prism dropped, and says so. Note the planner **prefers HNSW at default
settings** on this table despite prism being several times faster — a cost-model
gap worth looking at.

## Measurement protocol

This cost more time than any of the optimizations, so it is worth writing down.

**A/B differences under ~5% are not meaningful unless both arms are measured the
same way, back to back, without restarting the server.** A restart costs ~14%
here: the machine is still hot from the rebuild, and the new backend re-faults
the `shared_buffers` mapping inside the measured window. `vector.so` is
`dlopen`ed per backend, so swapping the file and opening a new connection gives a
clean arm with the buffer pool still warm. No restart needed.

**Verify each arm from the binary, not the source tree:**

```sh
nm -D postgres/install/lib/*/vector.so | grep -c BufTablePrefetch
```

An uncommitted working-tree change rides along across `git checkout`. Without
that assertion it is entirely possible to measure the same build twice and
conclude a patch does nothing — which happened here, and produced a confident
"null result" for a change that is worth +14%. The sequence of reported answers
for this one patch was +13.7% (cross-protocol, invalid), then "no effect"
(both arms patched, invalid), then +14.0% (verified). Also needs `make clean`
per arm, and build output left visible.

**Verify the query plan at every sweep point, not once per sweep.** The search
parameter feeds the index's own cost estimate, so it can change which index the
planner picks part-way up a sweep. With both an HNSW and a prism index on the
table, `hnsw.ef_search=320` raised HNSW's estimated startup cost to 1799 and the
planner switched to prism — producing an "HNSW" row at 1,369 QPS and recall
0.9789, i.e. **3.7x faster than `ef_search=160`**, which is what gave it away.
It was an exact match for prism at its auto `nprobe` of 82 (1,386 QPS, 0.9781,
p50 0.704 ms). `check_plan` now runs per point with that point's GUC applied and
aborts with the offending plan. Cost: 13 extra `EXPLAIN`s made the run 14s
*faster* than the one before it, i.e. nothing, and all of it outside the
measured window.

**A single pass over the query set is not a throughput measurement.** Use a
deadline (`--duration 60`) and record `queries_executed`. The first profile's
page-fault artefact existed because `phase_profile` silently did not forward
`--duration`.

**Other scar tissue:**

- Do not edit a shell script while bash is executing it. Bash reads by byte
  offset, so an edit mid-run produces `command not found` or a syntax error at a
  random later line. This killed two long chains.
- `cmd | awk '...exit...'` with `set -o pipefail -e` dies on SIGPIPE. It killed a
  finish script.
- `ensure_index` drops competing indexes. Chaining an IVFFlat phase after an HNSW
  phase therefore **destroys the HNSW index** — 27 minutes to rebuild. Now
  guarded by `ALLOW_INDEX_SWAP=1`.

## Reproducing

```sh
bench/run_experiment.sh data build query concurrency profile
```

Phases: `data build query concurrency profile ivfflat exact`. The script reuses a
running server when its options match, skips prewarm when the relation is already
resident, and asserts residency before measuring. Patches and their A/B protocol
are in `bench/patches/`.

## Moving 17.11 -> 18.6

`pg_vectorsearch` requires PostgreSQL 18+, so the cluster was rebuilt. Notes for
anyone repeating it:

- `pg_upgrade --copy` would have preserved the heap *and* the index files
  (no `REINDEX`), but the data was deleted and reloaded instead. Reload 542 s,
  HNSW rebuild 1,811 s; the index came out 61 blocks smaller, which is just
  parallel-build nondeterminism.
- **PostgreSQL 18 turns data checksums on by default.** `run_postgres.sh` now
  passes `initdb --no-data-checksums` so the numbers here stay comparable. For a
  `pg_upgrade` this is a hard stop, not a preference: the old and new clusters
  must agree.
- `pkglibdir` moved to `lib/x86_64-linux-gnu`. The `nm -D
  postgres/install/lib/*/vector.so` assertion still globs correctly.
- `build_postgres.sh` guarded the source download with `[ ! -d "$SRC" ]`, which
  made `PG_VERSION` a silent no-op once a tree existed. It now writes a
  `.pg_version` stamp and refuses a mismatch rather than rebuilding the wrong
  version or deleting a tree that may carry patches.
- The hash-prefetch patch applies cleanly to 18.6 (offsets only, one fuzz-2
  hunk). It is **not** applied: both Postgres and pgvector are stock here.

## Open

- `bench/REPORT.md` still describes the SIFT-1M run and needs rewriting around these numbers.
- IVFFlat and exact-scan baselines on BIGANN 18M: deferred, not run.
- 32 KiB block-size experiment: analysed above, not run.
- Port PRISM's dense block->buffer map to pgvector HNSW (attempt 2, above): no
  core patch needed, and it targets the whole 38%.
- Why the planner prefers HNSW at default settings when PRISM is 3-5x faster.
- Three-pass prefetch to hide the second dynahash miss: not tried.
- shmem THP / huge pages for `shared_buffers`: not tried, and dTLB miss rate says it could matter.
