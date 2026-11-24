#!/usr/bin/env bash
# Build the first pgvectorfast + TurboQuant prototype, which reads indexes
# pgvector_ntq built (branch tq4-prototype of awitten1/pgvectorfast), in-tree
# against
# the Postgres from build_postgres.sh. Nothing is installed: the fastntq arm in
# bench/run_experiment.sh swaps pgvectorfast_ntq/vectorfast.so in the way the
# fast arm does pgvectorfast/vectorfast.so. Same flags as pgvectorfast
# (its Makefile's OPTFLAGS=-march=native); calico's control/SQL files are the
# ones build_extensions.sh installed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/pgvectorfast_ntq"
PG_CONFIG="$ROOT/postgres/install/bin/pg_config"
JOBS="$(getconf _NPROCESSORS_ONLN)"
[ -x "$PG_CONFIG" ] || { echo "error: run ./build_postgres.sh first" >&2; exit 1; }
# pgvectorfast_ntq is a submodule pinned to the later port (branch
# pgvectorfast-ntq), which has no hnswtq_simd.h. To rebuild the prototype:
#   git -C pgvectorfast_ntq fetch origin tq4-prototype && git -C pgvectorfast_ntq checkout FETCH_HEAD
# and check the pinned commit out again afterwards (git submodule update).
[ -f "$SRC/src/hnswtq_simd.h" ] || { echo "error: $SRC is not the tq4-prototype branch (see the comment above)" >&2; exit 1; }

make -C "$SRC" clean PG_CONFIG="$PG_CONFIG" >/dev/null
make -C "$SRC" -j"$JOBS" PG_CONFIG="$PG_CONFIG"
so="$SRC/vectorfast.so"
[ "$(nm "$so" | grep -c ' calico_')" -gt 0 ] || { echo "error: $so has no calico symbols" >&2; exit 1; }
[ "$(nm "$so" | grep -c ' HnswTqRerankFast$')" -gt 0 ] || { echo "error: $so has no TurboQuant read path" >&2; exit 1; }
echo "pgvectorfast + tq4: $so"
