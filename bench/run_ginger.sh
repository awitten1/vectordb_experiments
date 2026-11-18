#!/usr/bin/env bash
# Run Ginger on the same dataset files the Postgres benchmarks use, appending
# to bench/results/<dataset>.jsonl.
#
# Usage: bench/run_ginger.sh [config ...]   (default: recommended float-rerank)
#
# Env: DATASET (a datasets.py spec, default the DBpedia-OpenAI 1M set),
#      GINGER_SWEEP (beam_size values), THREADS (comma list, default 1,8),
#      DURATION, NUM_QUERIES, K, REBUILD=1 to time a fresh index build.
#
# Index builds use every hardware thread (PARLAY_NUM_THREADS unset); queries
# use THREADS threads, which play the role of Postgres's concurrent clients.
set -euo pipefail

BENCH="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$BENCH")"
BIN="$BENCH/ginger/build/ginger_bench"
PY="$BENCH/.venv/bin/python"
DATASET="${DATASET:-qdrant:dbpedia-openai-1M-1536-angular}"
GINGER_SWEEP="${GINGER_SWEEP:-10,20,40,80,160,320,640,1000}"
THREADS="${THREADS:-1,8}"
DURATION="${DURATION:-60}"
NUM_QUERIES="${NUM_QUERIES:-1000}"
K="${K:-10}"

[ -x "$BIN" ] || { echo "error: $BIN not built; run ./build_ginger.sh" >&2; exit 1; }

# Same environment guards as run_experiment.sh (set by bench/cloudlab/dbpedia.env).
if [ "${REQUIRE_PINNED:-0}" = 1 ]; then
  "$BENCH/cloudlab/pin_cpu.sh" check || exit 1
fi
if [ "${REQUIRE_NO_SWAP:-0}" = 1 ] && [ -n "$(tail -n +2 /proc/swaps)" ]; then
  echo "error: swap is enabled; run 'sudo swapoff -a'" >&2
  exit 1
fi

# Postgres and Ginger must not share the machine during a measurement: a
# resident 28 GB shared_buffers plus Ginger's index and the mmapped train set
# would compete for memory and cores, and the timings would quietly suffer.
if "$ROOT/postgres/install/bin/pg_ctl" -D "$ROOT/postgres/data" status >/dev/null 2>&1; then
  echo "error: the Postgres experiment server is running. Stop it first:" >&2
  echo "       ./run_postgres.sh stop" >&2
  exit 1
fi

# The driver reads <name>_train.npy / <name>_meta.npz; materialize them (a
# no-op when cached) and get the file stem datasets.py uses.
NAME="$(cd "$BENCH" && "$PY" -c '
import sys, datasets
print(datasets.load(sys.argv[1], num_queries=int(sys.argv[2])).name)' "$DATASET" "$NUM_QUERIES" | tail -1)"

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(recommended float-rerank)

cd "$ROOT"
"$BIN" --version
for cfg in "${CONFIGS[@]}"; do
  first=1
  for t in ${THREADS//,/ }; do
    echo "=== ginger $cfg, $t thread(s), beam_size $GINGER_SWEEP, ${DURATION}s per point" >&2
    # Only the first run of a config may rebuild; the rest reuse its index.
    rebuild=()
    [ "${REBUILD:-0}" = 1 ] && [ "$first" = 1 ] && rebuild=(--rebuild)
    "$BIN" --data bench/data --dataset "$NAME" --config "$cfg" \
           --sweep "$GINGER_SWEEP" --k "$K" --num-queries "$NUM_QUERIES" \
           --duration "$DURATION" --threads "$t" "${rebuild[@]}"
    first=0
  done
done
