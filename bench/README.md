# ANN query benchmark

Measures recall@k vs. QPS / latency for pgvector's HNSW and IVFFlat indexes,
against the Postgres built by `../build_postgres.sh`.

`FINDINGS.md` holds the 18M-vector BIGANN results and the buffer-manager
investigation; `REPORT.md` is the older SIFT-1M report.

## Setup

```sh
python3 -m venv bench/.venv
bench/.venv/bin/pip install -r bench/requirements.txt

# Give Postgres enough shared memory to hold the index (default is 128MB)
PG_OPTS="-c shared_buffers=8GB" ./run_postgres.sh
```

If something else already listens on 5432, set `PGPORT` (e.g. `PGPORT=5433`) for
both `run_postgres.sh` and `ann_bench.py`. `shared_buffers` only changes on a
restart: `./run_postgres.sh stop`, then start again with `PG_OPTS`.

## Running

```sh
P=bench/.venv/bin/python

# HNSW on SIFT-1M (downloaded once into bench/data/), sweeping ef_search
$P bench/ann_bench.py --dataset sift-128-euclidean --index hnsw --m 16 --ef-construction 64 \
    --sweep 10,20,40,80,160,320

# IVFFlat on synthetic data, sweeping probes
$P bench/ann_bench.py --dataset synthetic:100000x128 --index ivfflat --lists 316 --sweep 1,2,4,8,16,32

# Exact sequential scan (baseline; recall is 1.0 by definition)
$P bench/ann_bench.py --dataset synthetic:100000x128 --index none

# 8 concurrent connections
$P bench/ann_bench.py --dataset sift-128-euclidean --index hnsw --sweep 40 --clients 8
```

`--help` lists every option (k, build memory/parallel workers, query count, DSN, ...).

### Datasets

- `synthetic:<n>x<d>[:<metric>]`: overlapping Gaussian clusters, seeded by `--seed`.
  Metric is `euclidean` (default), `angular` (cosine) or `ip`. Exact neighbors are
  brute-forced in numpy and cached in `bench/data/`.
- Any [ann-benchmarks](https://github.com/erikbern/ann-benchmarks#data-sets) HDF5
  name, e.g. `sift-128-euclidean`, `glove-100-angular`, `fashion-mnist-784-euclidean`,
  `gist-960-euclidean`. These ship with precomputed exact neighbors.

### What gets reused

- The table (`items_<dataset>`) is loaded once, and reloaded only when its row count
  doesn't match or you pass `--reload`.
- An index with the same parameters is reused. Pass `--rebuild` to time a fresh build.
  Building a different index drops any other HNSW/IVFFlat index on that table, so the
  planner can't pick the wrong one. Before measuring, the script checks with `EXPLAIN`
  that the query uses the intended index.

## Output

The script prints a table per run and appends one JSON line per sweep value to
`bench/results/<dataset>.jsonl`. Each line records recall, QPS, latency
(mean/p50/p95/p99), build params, build time, index size, Postgres/pgvector
versions, and the pgvector submodule SHA (`-dirty` if it has local edits), so you
can compare runs before and after changing pgvector:

```sh
make -C pgvector install PG_CONFIG=$PWD/postgres/install/bin/pg_config
./run_postgres.sh stop && PG_OPTS="-c shared_buffers=8GB" ./run_postgres.sh
$P bench/ann_bench.py --dataset sift-128-euclidean --index hnsw --rebuild
```

## Profiling (perf + flamegraph)

`bench/profile.sh` takes the same arguments as `ann_bench.py`. It does the following:
- prewarms the benchmark tables and indexes, and fails unless every block is in
  `shared_buffers`
- runs the benchmark under `perf record`, attached to the postmaster so the query
  backend is profiled and the Python client is not
- writes the profile, residency check and flamegraph to `bench/profiles/<NAME>.*`

```sh
sudo sysctl -w kernel.perf_event_paranoid=1 kernel.kptr_restrict=0   # resets on reboot
FRAME_POINTERS=1 ./build_postgres.sh     # complete call stacks; DWARF unwinding recovered only ~25%
PGPORT=5433 NAME=sift-hnsw-ef40 bench/profile.sh \
    --dataset sift-128-euclidean --index hnsw --sweep 40 --num-queries 10000
```

The flamegraph uses Brendan Gregg's FlameGraph scripts, cloned at a pinned commit
into `bench/.tools/`. Other environment knobs: `PERF_FREQ` (default 999),
`PERF_CALLGRAPH` (default `fp`), and `FLAMEGRAPH=0` to record only.

Notes:
- Latency is measured client-side around each `execute` + `fetchall` and includes the
  network round trip over localhost. Queries use protocol-level prepared statements.
- Recall is |returned ids ∩ true top-k ids| / k, averaged over the queries.
- QPS is total queries divided by wall time (from the first query start to the last query end, across all clients).
