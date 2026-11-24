#!/usr/bin/env bash
# The tq4 arm of the Hacker News study: pgvector_ntq (4-bit TurboQuant
# navigation codes in the HNSW graph, ntq12 re-ranking) on the same nodes,
# dataset, m / ef_construction and protocol as hn_qps.sh's stock and fast arms.
#
#   SMOKE=1 bench/cloudlab/hn_tq4.sh     # 1M subset, short windows
#   bench/cloudlab/hn_tq4.sh             # the real run
#
# It first waits for hn_qps.sh's last server step (the server must be free;
# building or loading anything during those measurements would disturb them),
# then, each step detached and resumable (see lib.sh):
#   1. build pgvector_ntq in-tree (build_ntq.sh)
#   2. load NTQ_TABLE, a second copy of the dataset, and build the quantized
#      index on it (run_experiment.sh build_tq); the plain index is untouched
#   3. 1-client rounds, client local: ntq twice (labels pvf-ntq-r1, -r2)
#   4. 10, 20 and 40 clients from the client node (label sc-ntq)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

if [ "${SMOKE:-0}" = 1 ]; then T=smoke-hn; else T=hn; fi
COUNTS="10 20 40"

say "waiting for hn_qps.sh to finish with the server (step hn-stop)"
wait_step "$SERVER" hn-stop
require_measurement_state

step "$SERVER" hn-build-ntq "./build_ntq.sh"
step "$SERVER" "$T-tq4-build" "bench/run_experiment.sh build_tq"
step "$SERVER" "$T-tq4-local" 'PVF_ARMS="ntq ntq" bench/run_experiment.sh pgvectorfast'

serve_step "$T-serve-ntq" ntq
step "$SERVER" "$T-numa-ntq" \
  '[ -n "${NUMA_CMD:-}" ] || exit 0; n=$(grep -c " interleave" /proc/$(head -1 postgres/data/postmaster.pid)/numa_maps); echo "postmaster mappings with interleave policy: $n"; [ "$n" -gt 0 ]'
for c in $COUNTS; do
  # shellcheck disable=SC2016  # NTQ_* expand on the node, from hn.env
  bench_step "$CLIENT" "$T-ntq-c$c" sc-ntq "$HNSW_ARGS"' --table "$NTQ_TABLE" $NTQ_ARGS'" --clients $c"
done
step "$SERVER" "$T-tq4-stop" "./run_postgres.sh stop"

for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $STEPS/hn-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
say "done; pull with: . bench/cloudlab/nodes-wisc.sh && bench/cloudlab/sync.sh pull"
