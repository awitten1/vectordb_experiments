#!/usr/bin/env bash
# Run the FAISS index-type matrix on the same dataset the Postgres benchmarks
# use, appending to bench/results/<dataset>.jsonl.
#
# Usage: bench/run_faiss.sh [factory ...]     (no argument = the whole matrix)
#
# FAISS runs single-threaded so its numbers share an axis with the
# single-client Postgres figures. Builds use every core; that is recorded per
# row as build_threads.
#
# This is an in-process library against a client/server database. The raw ratio
# therefore includes the whole Postgres round trip, which `run_experiment.sh
# overhead` measures separately. See bench/FINDINGS.md.
set -euo pipefail

BENCH="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$BENCH")"
BIN="$BENCH/faiss/build/faiss_bench"
DATASET="${DATASET:-bigann_18000000}"
K="${K:-10}"
NUM_QUERIES="${NUM_QUERIES:-1000}"
DURATION="${DURATION:-60}"
THREADS="${THREADS:-1}"

[ -x "$BIN" ] || { echo "error: $BIN not built; run ./build_faiss.sh" >&2; exit 1; }

# A Flat index is 9.2 GB resident and the experiment server holds 28 GB of
# shared_buffers. Together with the mmapped train set that is close enough to
# this machine's 59 GB to start swapping, which would quietly ruin the timings
# rather than fail.
if "$ROOT/postgres/install/bin/pg_ctl" -D "$ROOT/postgres/data" status >/dev/null 2>&1; then
  echo "error: the Postgres experiment server is running. Stop it first:" >&2
  echo "       ./run_postgres.sh stop" >&2
  exit 1
fi

# factory | sweep. nlist values mirror what the Postgres side resolved to:
# 4242 is pgvector IVFFlat's default lists at 18M rows, 26569 is the nlist
# PRISM chose, so the RaBitQ rows differ from PRISM only in implementation.
MATRIX=(
  "Flat|"
  "IVF4242,Flat|1,2,5,10,20,50,100"
  "IVF4242,SQ8|1,2,5,10,20,50,100"
  "IVF4242,PQ16|1,2,5,10,20,50,100"
  "IVF26569,RaBitQ|5,10,20,40,80,160,320"
  "IVF26569,RaBitQfs|5,10,20,40,80,160,320"
  "HNSW16|10,20,40,80,160,320"
)

wanted=("$@")
START=$SECONDS
for entry in "${MATRIX[@]}"; do
  factory="${entry%%|*}"
  sweep="${entry#*|}"

  if [ ${#wanted[@]} -gt 0 ]; then
    match=0
    for w in "${wanted[@]}"; do [ "$w" = "$factory" ] && match=1; done
    [ "$match" = 1 ] || continue
  fi

  printf '\n=== %s\n' "$factory" >&2
  args=(--data "$BENCH/data" --dataset "$DATASET" --factory "$factory"
        --k "$K" --num-queries "$NUM_QUERIES" --duration "$DURATION"
        --threads "$THREADS")
  # Flat has no search parameter, and one pass of exact search over 18M vectors
  # is already minutes; a 60s deadline would only make it re-run.
  if [ -n "$sweep" ]; then
    args+=(--sweep "$sweep")
  else
    args+=(--duration 0 --num-queries 100)
  fi
  "$BIN" "${args[@]}"
done

printf '\n=== done in %dm%ds\n' $(( (SECONDS - START) / 60 )) $(( (SECONDS - START) % 60 )) >&2
