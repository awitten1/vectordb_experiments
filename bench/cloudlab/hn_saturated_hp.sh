#!/usr/bin/env bash
# hn_saturated_m32.sh's five Postgres arms at the saturating client count
# (CLIENTS, default 60) again, with shared_buffers on 2 MB huge pages
# (huge_pages=on, pool reserved by hp_lib.sh). Same nodes (the second c220g2
# pair), sweep, 60 s per point, CPU and link sampling, and the same index
# swaps on PIPNN_TABLE, ordered to rebuild as little as possible:
#   1. on the float PiPNN index: pgvectorfast, then stock pgvector
#   2. drop it; import the graph as tq4 + ntq12; pgvectorfast-ntq
#   3. drop it; build HNSW m = 32, ef_construction 64; pgvectorfast, pgvector
#   4. import the float PiPNN index again; recall at 1 client must match
#      pippipnn-fork (recall_gate.py)
# The builds need the memory the pool holds (maintenance_work_mem 100GB), so
# the pool is released before each build and reserved again after it. Every
# serve is checked to be on huge pages (HP_CHECK; the tq4 arm's table and
# index fill ~72 GB of the pool, hence its lower HP_MIN_GB). Run with
# Postgres stopped:
#
#   CLIENTS=60 caffeinate -i bench/cloudlab/hn_saturated_hp.sh
#
# Labels sathp<N>-*. Steps are detached and resumable (see lib.sh); bump RUN
# to redo them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"
# shellcheck source=bench/cloudlab/hp_lib.sh
. "$HERE/hp_lib.sh"

RUN="${RUN:-1}"
CLIENTS="${CLIENTS:-60}"
T="hnsathp${CLIENTS}-r$RUN"
L="sathp${CLIENTS}"
PD="$HERE/../profiles/hn-saturated-c$CLIENTS-hp"
JSONL=hackernews_minilm_384.jsonl
NIC=enp6s0f0
NICLOG="$STEPS/nic-$NIC.txt"
NTQ_SO="$DEST/pgvectorfast_ntq/vectorfast.so"
IDX='${PIPNN_TABLE}_hnsw_m${PIPNN_M}_ef_construction64'
TQ_IDX="${IDX}_tq4_ntq12"
IARGS="--table \$PIPNN_TABLE --m \$PIPNN_M --ef-construction 64"
HPOPT="EXTRA_PG_OPTS='-c huge_pages=on'"

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
  [ -f pgvectorfast_ntq/vectorfast.so ] || { echo "error: no pgvectorfast_ntq/vectorfast.so"; exit 1; }
  [ -f "$PIPNN_SO" ] || { echo "error: no $PIPNN_SO"; exit 1; }
  [ -s "$PIPNN_GRAPH" ] || { echo "error: no graph file $PIPNN_GRAPH"; exit 1; }
  echo ok'

ssh_to "$SERVER" "if [ -f $STEPS/nic-$NIC.pid ] && kill -0 \$(cat $STEPS/nic-$NIC.pid) 2>/dev/null; then echo 'nic monitor running'; else nohup sh -c 'while :; do echo \"\$(date +%s) \$(cat /sys/class/net/$NIC/carrier 2>/dev/null) \$(cat /sys/class/net/$NIC/statistics/rx_crc_errors) \$(cat /sys/class/net/$NIC/statistics/rx_packets)\"; sleep 2; done' >> $NICLOG 2>&1 < /dev/null & echo \$! > $STEPS/nic-$NIC.pid; echo nic monitor started \$(cat $STEPS/nic-$NIC.pid); fi"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "[ -f $STEPS/$T-cpu.pid ] && kill \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; nohup sh -c 'while :; do echo \"\$(date +%s) \$(head -1 /proc/stat)\"; sleep 2; done' >> $STEPS/$T-cpu.txt 2>&1 < /dev/null & echo \$! > $STEPS/$T-cpu.pid; echo sampler \$(cat $STEPS/$T-cpu.pid)"
done

serve() {  # serve <name> <serve env> <serve arm> [HP_MIN_GB]
  step "$SERVER" "$T-serve-$1" \
    "$HPOPT $2 SERVE_ARM=$3 SERVE_TABLE=\$PIPNN_TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
  step "$SERVER" "$T-hpcheck-$1" "HP_MIN_GB=${4:-90}; $HP_CHECK"
}
sweep() {  # sweep <name> <label> <clients> [extra ann_bench args]
  bench_step "$CLIENT" "$T-$1" "$2" "--index hnsw $IARGS ${4:-} --sweep \"\$EF_SWEEP\" --clients $3"
}

# 1. Float PiPNN index (as imported by hn_pipnn.sh)
step "$SERVER" "$T-reserve" "$HP_RESERVE"
serve pvf-pipnn 'FAST_SO=$PIPNN_SO' fast
sweep pvf-pipnn "$L-pgvectorfast-pipnn" "$CLIENTS"
serve pgv-pipnn '' stock
sweep pgv-pipnn "$L-pgvector-pipnn" "$CLIENTS"

# 2. tq4 + ntq12 import of the PiPNN graph in its place
step "$SERVER" "$T-release-tq4" "$HP_RELEASE"
step "$SERVER" "$T-build-tq4" \
  "FAST_SO=$NTQ_SO PIPNN_QUANTIZER=tq4 PIPNN_DROP=\"$IDX\" bench/run_experiment.sh build_pipnn"
step "$SERVER" "$T-reserve-tq4" "$HP_RESERVE"
serve pvf-tq4 "FAST_SO=$NTQ_SO" fast 60
sweep pvf-tq4 "$L-pgvectorfast-tq4" "$CLIENTS" "--quantizer tq4 --rerank ntq12"

# 3. HNSW m = 32 in its place
step "$SERVER" "$T-release-m32" "$HP_RELEASE"
step "$SERVER" "$T-build-m32" \
  "M=32 EF_CONSTRUCTION=64 BUILD_MWM=100GB BUILD_TABLE=\$PIPNN_TABLE BUILD_DROP=\"$TQ_IDX\" bench/run_experiment.sh build"
# shellcheck disable=SC2016  # expanded on the node
step "$SERVER" "$T-check-m32" '
  l=/mydata/steps/'"$T"'-build-m32.log
  grep -q "building ${PIPNN_TABLE}_hnsw_m32_ef_construction64" $l || { echo "error: the build step did not build the index"; exit 1; }
  grep -q "no spill" $l || { echo "error: the build spilled out of maintenance_work_mem"; exit 1; }
  grep -E "built in|no spill" $l'
step "$SERVER" "$T-reserve-m32" "$HP_RESERVE"
serve pvf-m32 'FAST_SO=$PIPNN_SO' fast
sweep pvf-m32 "$L-pgvectorfast-hnsw-m32" "$CLIENTS"
serve pgv-m32 '' stock
sweep pgv-m32 "$L-pgvector-hnsw-m32" "$CLIENTS"

# 4. The float PiPNN index back in its place (same name: PIPNN_REBUILD drops
#    HNSW m = 32), checked at 1 client (5 s per point: recall doesn't depend
#    on the duration)
step "$SERVER" "$T-release-restore" "$HP_RELEASE"
step "$SERVER" "$T-restore" \
  "FAST_SO=\$PIPNN_SO PIPNN_REBUILD=1 bench/run_experiment.sh build_pipnn"
step "$SERVER" "$T-serve-restore" \
  "FAST_SO=\$PIPNN_SO SERVE_ARM=fast SERVE_TABLE=\$PIPNN_TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
step "$CLIENT" "$T-restore-c1" \
  "BENCH_LABEL=$L-restore bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw $IARGS --sweep \"\$EF_SWEEP\" --clients 1 --num-queries \"\$NUM_QUERIES\" --duration 5 $REMOTE_DSN"
step "$CLIENT" "$T-restore-gate" "bench/.venv/bin/python bench/cloudlab/recall_gate.py $L-restore pippipnn-fork"

for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "kill \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; rm -f $STEPS/$T-cpu.pid; true"
done
step "$SERVER" "$T-stop" "./run_postgres.sh stop || true; sudo sysctl -w vm.nr_hugepages=0"

mkdir -p "$PD"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-cpu.txt" "$PD/server-cpu.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$CLIENT.$DOMAIN:$STEPS/$T-cpu.txt" "$PD/client-cpu.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$NICLOG" "$PD/nic.txt"
ssh_to "$CLIENT" "grep -F '\"$L-' $DEST/bench/results/$JSONL" > "$PD/results.jsonl"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
python3 "$HERE/saturation_report.py" "$PD" | tee "$PD/report.md"
say "done; labels $L-* in $PD/results.jsonl; then bench/cloudlab/sync.sh pull"
