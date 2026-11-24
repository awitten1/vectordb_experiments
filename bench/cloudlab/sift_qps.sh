#!/usr/bin/env bash
# Recall/QPS for every system on SIFT1M (128-d), matching the DBpedia study:
# pgvector, pgvectorfast and PRISM at 1, 8, 12 and 16 clients, Ginger
# (recommended and float-rerank) at 1, 8, 12 and 16 threads.
#
# Run from the workstation once node-0 has the SIFT table and HNSW index
# (ENV_FILE=bench/cloudlab/sift.env bench/run_experiment.sh data build).
#
# Stage 1 runs two chains at once, because they share no machine:
#   node-0: the 1-client sweeps, client on node-0 as in the DBpedia runs:
#           stock/fast/stock/fast (labels pvf-<arm>-r<n>), then PRISM's
#           build on a table copy and its 1-client sweep (label prism-solo),
#           then the 1-client round-trip floor (label rtt)
#   node-1: Ginger, with Postgres stopped there; builds are timed (REBUILD=1)
# Stage 2: 8, 12 and 16 clients from node-1 over the LAN (labels sc-<arm>,
#          sc-prism), and the round-trip floor at each count (sc-rtt).
#
# Steps are detached and resumable (see lib.sh).
#
#   bench/cloudlab/sift_qps.sh
set -euo pipefail
export ENV_FILE=bench/cloudlab/sift.env
# shellcheck source=bench/cloudlab/lib.sh
. "$(dirname "$0")/lib.sh"

PRISM_TABLE=items_sift_128_euclidean_prism
COUNTS="8 12 16"

require_measurement_state

# Stage 1, in parallel.
if ! step_done "$SERVER" sift-local; then
  start_step "$SERVER" sift-local \
    'PVF_ARMS="stock fast stock fast" PVF_CLIENTS=1 CLIENTS=1 bench/run_experiment.sh pgvectorfast prism_solo overhead'
fi
if ! step_done "$CLIENT" sift-ginger; then
  start_step "$CLIENT" sift-ginger 'REBUILD=1 THREADS=1,8,12,16 bench/run_ginger.sh recommended float-rerank'
fi
wait_step "$SERVER" sift-local
wait_step "$CLIENT" sift-ginger

# Stage 2: network clients.
for arm in stock fast; do
  serve_step "sift-serve-$arm" "$arm"
  for c in $COUNTS; do
    bench_step "$CLIENT" "sift-$arm-c$c" "sc-$arm" "$HNSW_ARGS --clients $c"
  done
done
serve_step sift-serve-prism stock "$PRISM_TABLE"
for c in $COUNTS; do
  # shellcheck disable=SC2016
  bench_step "$CLIENT" "sift-prism-c$c" sc-prism \
    "--index prism --table $PRISM_TABLE --sweep \"\$NPROBE_SWEEP\" --clients $c"
done
for c in $COUNTS; do
  bench_step "$CLIENT" "sift-rtt-c$c" sc-rtt "--index rtt --clients $c"
done
step "$SERVER" sift-stop "./run_postgres.sh stop"

say "done; pull with bench/cloudlab/sync.sh pull"
