#!/usr/bin/env bash
# pgvectorfast with Ginger's TurboQuant codes (quantizer = 'tq4', ntq12
# re-rank) on the PiPNN graph of hn_pipnn.sh: the same graph file, imported
# as 4-bit codes instead of floats. Run with Postgres stopped, after
# hn_pipnn.sh:
#
#   caffeinate -i bench/cloudlab/hn_pipnn_tq4.sh
#
# FORK is the awitten1/pgvectorfast checkout with quantizer = 'tq4' (default
# the pgvectorfast_ntq submodule, branch pgvectorfast-ntq); it is synced to
# pgvectorfast_ntq/ on the server.
#
# Steps are detached and resumable (see lib.sh); bump RUN to redo them.
#   1. build the fork's vectorfast.so on the server
#   2. /mydata has no room for both indexes (the float one is ~55 GB, the tq4
#      one ~32 GB), so drop the float PiPNN index and import the graph file
#      again as tq4 codes (build_pipnn with PIPNN_QUANTIZER=tq4)
#   3. serve it (4 KB pages); ef_search sweep at 1 and 40 LAN clients, label
#      piptq4-pipnn
#   4. drop the tq4 index and import the float index again from the same
#      file, with the PiPNN fork's library; serve it and check at 1 client
#      that its recall matches pippipnn-fork's (recall_gate.py), label
#      piptq4-restore
#   5. stop Postgres; client results into bench/profiles/hn-pipnn/
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
T="hnptq-r$RUN"
FORK="${FORK:-$(cd "$HERE/../.." && pwd)/pgvectorfast_ntq}"
PD=hn-pipnn
JSONL=hackernews_minilm_384.jsonl
NTQ_SO="$DEST/pgvectorfast_ntq/vectorfast.so"
FLOAT_IDX='${PIPNN_TABLE}_hnsw_m${PIPNN_M}_ef_construction64'
TQ_IDX="${FLOAT_IDX}_tq4_ntq12"
PIPNN_ARGS="--table \$PIPNN_TABLE --m \$PIPNN_M --ef-construction 64"

grep -q 'HNSW_QUANTIZER_TQ4' "$FORK/src/hnsw.h" ||
  { echo "error: $FORK is not the pgvectorfast fork with quantizer = 'tq4'" >&2; exit 1; }

rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
  --exclude .git --exclude '/tools/*/build/' --exclude '*.o' --exclude '*.so' --exclude '*.bc' \
  --exclude third_party --exclude .deps --exclude tmp_check --exclude results --exclude .DS_Store \
  "$FORK/" "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/pgvectorfast_ntq/"
for n in "$SERVER" "$CLIENT"; do
  ( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
      bench/run_experiment.sh bench/cloudlab/hn.env bench/cloudlab/recall_gate.py \
      "$REMOTE_USER@$n.$DOMAIN:$DEST/" )
done

require_measurement_state

# The LAN clients run on the client node: nothing else may be measuring there
step "$CLIENT" "$T-client-idle" \
  'if pgrep -x ginger_bench >/dev/null; then echo "error: a ginger_bench is running"; exit 1; fi; echo idle'

# shellcheck disable=SC2016  # expanded on the node
step "$SERVER" "$T-so" '
  set -eo pipefail
  make -C pgvectorfast_ntq clean PG_CONFIG=$PWD/postgres/install/bin/pg_config > /dev/null
  make -C pgvectorfast_ntq -j"$(nproc)" PG_CONFIG=$PWD/postgres/install/bin/pg_config 2>&1 | grep -E "hnswtq.*(warning|error)| error" || true
  so=pgvectorfast_ntq/vectorfast.so
  st=$(strings $so)
  grep -q hnsw.build_graph_file <<< "$st" || { echo "error: no hnsw.build_graph_file in $so"; exit 1; }
  grep -q "quantizer \"tq4\"" <<< "$st" || { echo "error: no quantizer tq4 in $so"; exit 1; }
  [ "$(nm $so | grep -c " calico_")" -gt 0 ] || { echo "error: no calico symbols in $so"; exit 1; }
  [ -s "$PIPNN_GRAPH" ] || { echo "error: no graph file $PIPNN_GRAPH (run hn_pipnn.sh)"; exit 1; }
  echo "built $so"'

step "$SERVER" "$T-build-tq4" \
  "FAST_SO=$NTQ_SO PIPNN_QUANTIZER=tq4 PIPNN_DROP=\"$FLOAT_IDX\" bench/run_experiment.sh build_pipnn"

step "$SERVER" "$T-serve-tq4" \
  "FAST_SO=$NTQ_SO SERVE_ARM=fast SERVE_TABLE=\$PIPNN_TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
for c in 1 40; do
  bench_step "$CLIENT" "$T-tq4-c$c" piptq4-pipnn \
    "--index hnsw $PIPNN_ARGS --quantizer tq4 --rerank ntq12 --sweep \"\$EF_SWEEP\" --clients $c"
done

# Put the float index back, as hn_pipnn.sh built it, and check it
step "$SERVER" "$T-restore" \
  "FAST_SO=\$PIPNN_SO PIPNN_DROP=\"$TQ_IDX\" bench/run_experiment.sh build_pipnn"
step "$SERVER" "$T-serve-restore" \
  "FAST_SO=\$PIPNN_SO SERVE_ARM=fast SERVE_TABLE=\$PIPNN_TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
bench_step "$CLIENT" "$T-restore-c1" piptq4-restore "--index hnsw $PIPNN_ARGS --sweep \"\$EF_SWEEP\" --clients 1"
step "$CLIENT" "$T-restore-gate" "bench/.venv/bin/python bench/cloudlab/recall_gate.py piptq4-restore pippipnn-fork"

step "$SERVER" "$T-stop" "./run_postgres.sh stop || true"
ssh_to "$CLIENT" "cat $DEST/bench/results/$JSONL" |
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$SERVER.$DOMAIN" "cat > $DEST/bench/profiles/$PD/client-results.jsonl"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
mkdir -p "$HERE/../profiles/$PD"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/bench/profiles/$PD/client-results.jsonl" "$HERE/../profiles/$PD/"
say "done; label piptq4-pipnn in bench/profiles/$PD/client-results.jsonl; then bench/cloudlab/sync.sh pull"
