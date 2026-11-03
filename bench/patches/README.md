# Experiment patches

Patches for experiments described in `bench/REPORT.md`. They are kept here
because `postgres/` and the `pgvector/` submodule's branches are not part of
this repository's history.

| patch | applies to | result |
|---|---|---|
| `postgres-17.11-hash-prefetch.patch` | `postgres/source` after `build_postgres.sh` fetches it | adds `hash_prefetch()` / `BufTablePrefetch()` |
| `pgvector-0.8.7-bucket-prefetch.patch` | `pgvector/` at v0.8.7 (`f37c13f`) | **+14% QPS** at ef_search=40; needs the Postgres patch above |
| `pgvector-0.8.7-batch-and-blockstats.patch` | `pgvector/` at v0.8.7 (`f37c13f`) | **no effect**; also carries the `-DHNSW_BLOCKSTATS` instrumentation |

## Applying

```sh
./build_postgres.sh                      # fetches postgres/source
patch -p1 -d postgres/source < bench/patches/postgres-17.11-hash-prefetch.patch
patch -p1 -d pgvector       < bench/patches/pgvector-0.8.7-bucket-prefetch.patch
./build_postgres.sh                      # rebuild both
```

## Measuring them

A/B results below ~5% are not meaningful unless both arms are measured the same
way, back to back, on a server that is **not** restarted between them — a
restart costs ~14% because the machine is still hot from the rebuild and the
new backend re-faults the `shared_buffers` mapping inside the measured window.

Swap `vector.so` and open a new connection instead: it is `dlopen`ed per
backend, so no restart is needed and the buffer pool stays warm. Verify each
arm from the binary rather than from the source tree:

```sh
nm -D postgres/install/lib/*/vector.so | grep -c BufTablePrefetch
```

An uncommitted working-tree change rides along across `git checkout`, so
without that check it is entirely possible to measure the same build twice and
conclude a patch does nothing. That happened here.
