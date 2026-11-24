#!/usr/bin/env bash
# HNSW with PiPNN's layer-0 degree: m = 32 (64 layer-0 neighbours, as PiPNN
# keeps), ef_construction = 64 as in the main study. Against the imported
# PiPNN graph (same m), this separates how the graph is built from how many
# neighbours it keeps. Both engines: pgvectorfast (the fork library of the
# PiPNN arm, so only the graph differs) and stock pgvector. Run with Postgres
# stopped, after hn_pipnn.sh:
#
#   caffeinate -i bench/cloudlab/hn_hnsw_m32.sh
#
# The index goes on PIPNN_TABLE, the dataset's rows in the same order: /mydata
# has no room for a ~60 GB index next to the float PiPNN index (~55 GB), and
# the main table's m = 16 index stays untouched. Both indexes would be named
# <table>_hnsw_m32_ef_construction64, so the PiPNN index is dropped first and
# imported again from its graph file at the end.
#
# Steps are detached and resumable (see lib.sh); bump RUN to redo them.
#   1. drop the PiPNN index; build HNSW m=32, ef_construction=64 on
#      PIPNN_TABLE with stock pgvector, in memory (BUILD_MWM=100GB), 39 workers
#   2. check that step 1 built the index (not reused the PiPNN import, which
#      has the same name) and did not spill out of maintenance_work_mem
#   3. 4 KB pages: serve pgvectorfast, sweep at 1 and 40 LAN clients
#      (label pipm32-fork); serve stock, the same (label pipm32-stock)
#   4. import the PiPNN index again from the graph file; serve the fork and
#      check its recall at 1 client against pippipnn-fork (label pipm32-restore)
#   5. stop Postgres; client results into bench/profiles/hn-pipnn/
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
T="hnm32-r$RUN"
PD=hn-pipnn
JSONL=hackernews_minilm_384.jsonl
IDX='${PIPNN_TABLE}_hnsw_m32_ef_construction64'
IARGS="--table \$PIPNN_TABLE --m 32 --ef-construction 64"

for n in "$SERVER" "$CLIENT"; do
  ( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
      bench/run_experiment.sh bench/cloudlab/hn.env bench/cloudlab/recall_gate.py \
      "$REMOTE_USER@$n.$DOMAIN:$DEST/" )
done

require_measurement_state

step "$CLIENT" "$T-client-idle" \
  'if pgrep -x ginger_bench >/dev/null; then echo "error: a ginger_bench is running"; exit 1; fi; echo idle'

step "$SERVER" "$T-build" \
  "M=32 EF_CONSTRUCTION=64 BUILD_MWM=100GB BUILD_TABLE=\$PIPNN_TABLE BUILD_DROP=\"$IDX\" bench/run_experiment.sh build"

# The build step must have built the index (not reused the PiPNN import, which
# has the same name) and kept the graph in memory, or its time means nothing.
# shellcheck disable=SC2016  # expanded on the node
step "$SERVER" "$T-check-build" '
  l=/mydata/steps/'"$T"'-build.log
  grep -q "building ${PIPNN_TABLE}_hnsw_m32_ef_construction64" $l || { echo "error: the build step did not build the index"; exit 1; }
  grep -q "no spill" $l || { echo "error: the build spilled out of maintenance_work_mem"; exit 1; }
  grep -E "built in|no spill" $l'

# arm <name> <serve arm> <serve env> <label>
arm() {
  step "$SERVER" "$T-serve-$1" \
    "$3 SERVE_ARM=$2 SERVE_TABLE=\$PIPNN_TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
  for c in 1 40; do
    bench_step "$CLIENT" "$T-$1-c$c" "$4" "--index hnsw $IARGS --sweep \"\$EF_SWEEP\" --clients $c"
  done
}
arm fork fast 'FAST_SO=$PIPNN_SO' pipm32-fork
arm stock stock '' pipm32-stock

# Put the PiPNN index back, as hn_pipnn.sh built it, and check it
step "$SERVER" "$T-restore" "FAST_SO=\$PIPNN_SO PIPNN_REBUILD=1 bench/run_experiment.sh build_pipnn"
step "$SERVER" "$T-serve-restore" \
  "FAST_SO=\$PIPNN_SO SERVE_ARM=fast SERVE_TABLE=\$PIPNN_TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
bench_step "$CLIENT" "$T-restore-c1" pipm32-restore "--index hnsw $IARGS --sweep \"\$EF_SWEEP\" --clients 1"
step "$CLIENT" "$T-restore-gate" "bench/.venv/bin/python bench/cloudlab/recall_gate.py pipm32-restore pippipnn-fork"

step "$SERVER" "$T-stop" "./run_postgres.sh stop || true"
ssh_to "$CLIENT" "cat $DEST/bench/results/$JSONL" |
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$SERVER.$DOMAIN" "cat > $DEST/bench/profiles/$PD/client-results.jsonl"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
mkdir -p "$HERE/../profiles/$PD"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/bench/profiles/$PD/client-results.jsonl" "$HERE/../profiles/$PD/"
say "done; labels pipm32-fork, pipm32-stock in bench/profiles/$PD/client-results.jsonl; then bench/cloudlab/sync.sh pull"
