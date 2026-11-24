#!/usr/bin/env bash
# Build pgvector_ntq (awitten1/pgvector, branch hnsw-tq4-nav: 4-bit TurboQuant
# navigation codes for HNSW) in-tree against the Postgres from
# build_postgres.sh. Nothing is installed: bench/run_experiment.sh's ntq arm
# swaps pgvector_ntq/vector.so in behind $libdir/vector, as it does for
# pgvectorfast, so the database stays bound to the same extension.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PG_CONFIG="$ROOT/postgres/install/bin/pg_config"
JOBS="$(getconf _NPROCESSORS_ONLN)"
[ -x "$PG_CONFIG" ] || { echo "error: run ./build_postgres.sh first" >&2; exit 1; }
[ -f "$ROOT/pgvector_ntq/src/hnswtq.c" ] || {
  echo "error: $ROOT/pgvector_ntq is missing or not the hnsw-tq4-nav branch" >&2; exit 1; }

# Same flags as stock pgvector in build_postgres.sh.
make -C "$ROOT/pgvector_ntq" clean PG_CONFIG="$PG_CONFIG" >/dev/null
# OPTFLAGS on the command line overrides the Makefile's own (including its
# per-platform resets) and flows into PG_CFLAGS for every object.
LOG="$(mktemp)"
make -C "$ROOT/pgvector_ntq" -j"$JOBS" PG_CONFIG="$PG_CONFIG" OPTFLAGS="-march=native" | tee "$LOG"
# Every C compile must have carried -march=native, the TurboQuant kernels included.
# (JIT bitcode compiles, -emit-llvm, are not the code that runs; skip them.)
compiles="$(grep -E '(gcc|cc|clang).* -c .*\.c' "$LOG" | grep -vc -- '-emit-llvm' || true)"
native="$(grep -E '(gcc|cc|clang).* -c .*\.c' "$LOG" | grep -v -- '-emit-llvm' | grep -c -- '-march=native' || true)"
grep -E ' -c .*src/hnswtq\.c' "$LOG" | grep -v -- '-emit-llvm' | grep -q -- '-march=native' ||
  { echo "error: src/hnswtq.c was not compiled with -march=native" >&2; exit 1; }
[ "$compiles" -gt 0 ] && [ "$native" = "$compiles" ] ||
  { echo "error: $native of $compiles compiles used -march=native" >&2; exit 1; }
echo "all $compiles C compiles used -march=native"
rm -f "$LOG"
[ "$(nm "$ROOT/pgvector_ntq/vector.so" | grep -c ' HnswTqCreateState$')" -gt 0 ] ||
  { echo "error: vector.so has no TurboQuant code" >&2; exit 1; }
echo "pgvector_ntq: $ROOT/pgvector_ntq/vector.so"
