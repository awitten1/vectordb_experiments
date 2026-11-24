#!/usr/bin/env bash
# Recall/QPS on Hacker News (28.7M x 384-d MiniLM) for stock pgvector,
# pgvectorfast and Ginger, on the two c220g2 nodes in nodes-wisc.sh.
#
# Run from the workstation after the repo is on both nodes:
#   . bench/cloudlab/nodes-wisc.sh
#   ssh ... 'bash -s -- disk' < bench/cloudlab/setup_node.sh   # once per node
#   bench/cloudlab/sync.sh push
#   SMOKE=1 bench/cloudlab/hn_qps.sh     # 1M subset, short windows
#   bench/cloudlab/hn_qps.sh             # the real run
#
# Stages (each step detached and resumable, see lib.sh; smoke steps are
# named smoke-hn-*, real ones hn-*):
#   0. setup_node.sh on both nodes; Postgres + both pgvector builds on the
#      server, Ginger on the client. Shared by smoke and real runs.
#   1. Both nodes materialize the dataset at once (download, dedup, held-out
#      queries, exact top-100); the two copies must be byte-identical.
#   2. In parallel, as they share no machine:
#        server: load + HNSW build (timed), then 1-client stock/fast/stock/fast
#                rounds with the client local (labels pvf-<arm>-r<n>), then the
#                1-client round-trip floor (label rtt)
#        client: Ginger, recommended and float-rerank, 1/10/20/40 threads,
#                fresh timed builds, Postgres not running there
#   3. Clients on the client node over the LAN: 10, 20 and 40 clients against
#      the stock arm, then the fast arm (labels sc-<arm>), then the round-trip
#      floor at each count (sc-rtt).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

if [ "${SMOKE:-0}" = 1 ]; then T=smoke-hn; NAME=hackernews_minilm_384_1000000
else T=hn; NAME=hackernews_minilm_384
fi
COUNTS="10 20 40"

# 0. Node setup and builds (setup_node.sh pins the clock at hn.env's FREQ_KHZ).
for n in "$SERVER" "$CLIENT"; do
  step_done "$n" hn-setup || start_step "$n" hn-setup "bench/cloudlab/setup_node.sh"
done
wait_step "$SERVER" hn-setup
wait_step "$CLIENT" hn-setup
step_done "$SERVER" hn-build-pg ||
  start_step "$SERVER" hn-build-pg "./build_postgres.sh && SKIP_PRISM=1 ./build_extensions.sh"
step_done "$CLIENT" hn-build-ginger || start_step "$CLIENT" hn-build-ginger "./build_ginger.sh"
wait_step "$SERVER" hn-build-pg
wait_step "$CLIENT" hn-build-ginger
require_measurement_state

# 1. Dataset on both nodes, then check the copies agree.
# The loader is called directly: the client has no Postgres build, which
# run_experiment.sh insists on.
# shellcheck disable=SC2016  # expanded on the node, from hn.env
data_cmd='cd bench && .venv/bin/python -c "import sys, datasets; ds = datasets.load(sys.argv[1], num_queries=int(sys.argv[2])); print(ds.name, ds.train.shape, ds.test.shape)" "$DATASET" "$NUM_QUERIES"'
for n in "$SERVER" "$CLIENT"; do
  step_done "$n" "$T-data" || start_step "$n" "$T-data" "$data_cmd"
done
wait_step "$SERVER" "$T-data"
wait_step "$CLIENT" "$T-data"
sums=()
for n in "$SERVER" "$CLIENT"; do
  sums+=("$(ssh_to "$n" "cd $DEST/bench/data && sha256sum ${NAME}_train.npy ${NAME}_meta.npz")")
done
printf '%s\n' "${sums[@]}" >&2
[ "${sums[0]}" = "${sums[1]}" ] || { echo "error: the two nodes' dataset files differ" >&2; exit 1; }

# 2. Server: build and 1-client rounds; client: Ginger. In parallel.
if ! step_done "$SERVER" "$T-build"; then
  start_step "$SERVER" "$T-build" "bench/run_experiment.sh build"
fi
if ! step_done "$CLIENT" "$T-ginger"; then
  start_step "$CLIENT" "$T-ginger" "REBUILD=1 THREADS=1,10,20,40 bench/run_ginger.sh recommended float-rerank"
fi
wait_step "$SERVER" "$T-build"
step "$SERVER" "$T-local" 'PVF_ARMS="stock fast stock fast" bench/run_experiment.sh pgvectorfast overhead'
wait_step "$CLIENT" "$T-ginger"

# 3. Network clients. Each arm also checks that the server really runs with
#    interleaved memory (its shared memory mapping says so in numa_maps).
for arm in stock fast; do
  serve_step "$T-serve-$arm" "$arm"
  step "$SERVER" "$T-numa-$arm" \
    '[ -n "${NUMA_CMD:-}" ] || exit 0; n=$(grep -c " interleave" /proc/$(head -1 postgres/data/postmaster.pid)/numa_maps); echo "postmaster mappings with interleave policy: $n"; [ "$n" -gt 0 ]'
  for c in $COUNTS; do
    bench_step "$CLIENT" "$T-$arm-c$c" "sc-$arm" "$HNSW_ARGS --clients $c"
  done
done
for c in $COUNTS; do
  bench_step "$CLIENT" "$T-rtt-c$c" sc-rtt "--index rtt --clients $c"
done
step "$SERVER" "$T-stop" "./run_postgres.sh stop"

# Keep the step logs (build time, sizes, dedup counts) next to the results.
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $STEPS/hn-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
say "done; pull with: . bench/cloudlab/nodes-wisc.sh && bench/cloudlab/sync.sh pull"
