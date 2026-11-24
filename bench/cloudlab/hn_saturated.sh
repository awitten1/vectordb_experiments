#!/usr/bin/env bash
# Every Postgres arm of the four-way study at a client count that keeps the
# server busy (CLIENTS, from hn_saturation_probe.sh; default 60), 4 KB pages,
# full ef_search sweep, 60 s per point, with server and client CPU sampled
# throughout (saturation_report.py gives the busy share per point). Run with
# Postgres stopped:
#
#   CLIENTS=60 caffeinate -i bench/cloudlab/hn_saturated.sh
#
# Arms (labels sat<N>-*): pgvector and pgvectorfast (the fork library of the
# PiPNN arm) on the main table's HNSW m = 16 index and on the imported PiPNN
# graph; then pgvectorfast + PiPNN + tq4, which needs the tq4 import (the
# float PiPNN index is dropped for room, and imported again at the end, with
# a recall check against pippipnn-fork).
#
# Steps are detached and resumable (see lib.sh); bump RUN to redo them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
CLIENTS="${CLIENTS:-60}"
T="hnsat$CLIENTS-r$RUN"
L="sat$CLIENTS"
PD="$HERE/../profiles/hn-saturated-c$CLIENTS"
JSONL=hackernews_minilm_384.jsonl
NTQ_SO="$DEST/pgvectorfast_ntq/vectorfast.so"
FLOAT_IDX='${PIPNN_TABLE}_hnsw_m${PIPNN_M}_ef_construction64'
TQ_IDX="${FLOAT_IDX}_tq4_ntq12"
HNSW_ARGS="--m 16 --ef-construction 64"
PIPNN_ARGS="--table \$PIPNN_TABLE --m \$PIPNN_M --ef-construction 64"

for n in "$SERVER" "$CLIENT"; do
  ( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
      bench/run_experiment.sh bench/cloudlab/hn.env bench/cloudlab/recall_gate.py \
      "$REMOTE_USER@$n.$DOMAIN:$DEST/" )
done

require_measurement_state

step "$CLIENT" "$T-client-idle" \
  'if pgrep -x ginger_bench >/dev/null; then echo "error: a ginger_bench is running"; exit 1; fi; echo idle'
# shellcheck disable=SC2016  # expanded on the node
step "$SERVER" "$T-check" '
  [ -x pgvectorfast_ntq/vectorfast.so ] || [ -f pgvectorfast_ntq/vectorfast.so ] || { echo "error: no pgvectorfast_ntq/vectorfast.so (run hn_pipnn_tq4.sh)"; exit 1; }
  [ -s "$PIPNN_GRAPH" ] || { echo "error: no graph file $PIPNN_GRAPH"; exit 1; }
  echo ok'

# CPU samplers on both nodes, for the whole run
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "[ -f $STEPS/$T-cpu.pid ] && kill \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; nohup sh -c 'while :; do echo \"\$(date +%s) \$(head -1 /proc/stat)\"; sleep 2; done' >> $STEPS/$T-cpu.txt 2>&1 < /dev/null & echo \$! > $STEPS/$T-cpu.pid; echo sampler \$(cat $STEPS/$T-cpu.pid)"
done

# arm <name> <serve env> <serve arm> <label> <ann_bench index args>
arm() {
  step "$SERVER" "$T-serve-$1" \
    "$2 SERVE_ARM=$3 LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
  bench_step "$CLIENT" "$T-$1" "$4" "--index hnsw $5 --sweep \"\$EF_SWEEP\" --clients $CLIENTS"
}
arm pgv-hnsw   ""                                          stock "$L-pgvector-hnsw"      "$HNSW_ARGS"
arm pvf-hnsw   'FAST_SO=$PIPNN_SO'                         fast  "$L-pgvectorfast-hnsw"  "$HNSW_ARGS"
arm pgv-pipnn  'SERVE_TABLE=$PIPNN_TABLE'                  stock "$L-pgvector-pipnn"     "$PIPNN_ARGS"
arm pvf-pipnn  'FAST_SO=$PIPNN_SO SERVE_TABLE=$PIPNN_TABLE' fast  "$L-pgvectorfast-pipnn" "$PIPNN_ARGS"

# tq4: import the graph file as codes (room: drop the float index first)
step "$SERVER" "$T-build-tq4" \
  "FAST_SO=$NTQ_SO PIPNN_QUANTIZER=tq4 PIPNN_DROP=\"$FLOAT_IDX\" bench/run_experiment.sh build_pipnn"
arm pvf-tq4 "FAST_SO=$NTQ_SO SERVE_TABLE=\$PIPNN_TABLE" fast "$L-pgvectorfast-tq4" "$PIPNN_ARGS --quantizer tq4 --rerank ntq12"

for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "kill \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; rm -f $STEPS/$T-cpu.pid; true"
done

# Put the float PiPNN index back and check its recall (1 client, 5 s per
# point: recall doesn't depend on the duration)
step "$SERVER" "$T-restore" \
  "FAST_SO=\$PIPNN_SO PIPNN_DROP=\"$TQ_IDX\" bench/run_experiment.sh build_pipnn"
step "$SERVER" "$T-serve-restore" \
  "FAST_SO=\$PIPNN_SO SERVE_ARM=fast SERVE_TABLE=\$PIPNN_TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
step "$CLIENT" "$T-restore-c1" \
  "BENCH_LABEL=$L-restore bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw $PIPNN_ARGS --sweep \"\$EF_SWEEP\" --clients 1 --num-queries \"\$NUM_QUERIES\" --duration 5 $REMOTE_DSN"
step "$CLIENT" "$T-restore-gate" "bench/.venv/bin/python bench/cloudlab/recall_gate.py $L-restore pippipnn-fork"
step "$SERVER" "$T-stop" "./run_postgres.sh stop || true"

mkdir -p "$PD"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-cpu.txt" "$PD/server-cpu.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$CLIENT.$DOMAIN:$STEPS/$T-cpu.txt" "$PD/client-cpu.txt"
ssh_to "$CLIENT" "grep -F '\"$L-' $DEST/bench/results/$JSONL" > "$PD/results.jsonl"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
python3 "$HERE/saturation_report.py" "$PD" | tee "$PD/report.md"
say "done; labels $L-* in $PD/results.jsonl; then bench/cloudlab/sync.sh pull"
