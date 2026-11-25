#!/usr/bin/env bash
# The Hacker News study on two c220g5 nodes (nodes-wisc.sh), from scratch: six
# engines at 1 and CLIENTS (default 60) LAN clients, Ginger at 1 and GTHREADS
# (default 40) threads, on 2 MB huge pages and then on 4 KB pages.
#
#   arm                 library                       index
#   pgvector-hnsw       stock pgvector 0.8.7          HNSW m=32, ef_construction 64, main table
#   pgvectorfast-hnsw   pgvectorfast fork (PIPNN_SO)  the same index
#   pgvector-pipnn      stock pgvector 0.8.7          PiPNN graph imported as floats, PIPNN_TABLE
#   pgvectorfast-pipnn  pgvectorfast fork (PIPNN_SO)  the same index
#   pgvectorfast-tq4    pgvectorfast_ntq              PiPNN graph imported as tq4 + ntq12, PIPNN_TABLE
#   ginger              Ginger, recommended config    its own index, on the client node
#
# HNSW m=32 lives on the main table and is never touched once built. The PiPNN
# table holds one index at a time (disk), so the passes are ordered to swap it
# only twice: huge pages float -> tq4, then 4 KB tq4 -> float.
#
#   0. setup_node.sh on both; Postgres, stock pgvector, both pgvectorfast forks
#      and pipnn_export on the server; Ginger on the client (shared by smoke
#      and real runs)
#   1. the full dataset on both nodes, byte-identical (shared)
#   2. server: main table + HNSW m=32, then PIPNN_TABLE + PiPNN graph + float
#      import; client meanwhile: Ginger's recommended index
#   3. huge pages (labels g5hp-*): HNSW arms, float PiPNN arms; then the tq4
#      import on the server while Ginger runs on THP on the idle client; tq4 arm
#   4. 4 KB pages (labels g5-*), huge_pages=off: tq4 arm; float re-import on the
#      server while Ginger's 4 KB run uses the idle client; float PiPNN arms
#      (recall gated against the huge-pages run); HNSW arms
#   5. recall checks (stock = fast on each graph), saturation report, pull
#
# Run from the workstation, after setup_node.sh's disk mode and sync.sh push:
#   SMOKE=1 caffeinate -i bench/cloudlab/hn_g5.sh   # 1M rows, 5 s points
#   caffeinate -i bench/cloudlab/hn_g5.sh
# Steps are detached and resumable (lib.sh); bump RUN to redo them. Don't edit
# this file while it runs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
SMOKE="${SMOKE:-0}"
export SMOKE
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"
# shellcheck source=bench/cloudlab/hp_lib.sh
. "$HERE/hp_lib.sh"

RUN="${RUN:-1}"
CLIENTS="${CLIENTS:-60}"
GTHREADS="${GTHREADS:-40}"
if [ "$SMOKE" = 1 ]; then P=smoke-; DS=hackernews_minilm_384_1000000; HPMIN=1; HPMIN_TQ=1; GATE="--min-common 3"; SAMPLE_S=2
else P=""; DS=hackernews_minilm_384; HPMIN=90; HPMIN_TQ=50; GATE=""; SAMPLE_S=15; fi
T="${P}hng5-r$RUN"
LHP="${P}g5hp"
L4K="${P}g5"
JSONL="bench/results/$DS.jsonl"
PD="bench/profiles/hn-g5${P:+-smoke}"     # same path on the nodes and here
NTQ_SO="$DEST/pgvectorfast_ntq/vectorfast.so"
FLOAT_IDX='${PIPNN_TABLE}_hnsw_m${PIPNN_M}_ef_construction64'
TQ_IDX="${FLOAT_IDX}_tq4_ntq12"
MAIN_ARGS='--index hnsw --m 32 --ef-construction 64'
PIP_ARGS='--index hnsw --table $PIPNN_TABLE --m $PIPNN_M --ef-construction 64'
TQ_ARGS="$PIP_ARGS --quantizer tq4 --rerank ntq12"

# ---- 0. Node setup and builds -------------------------------------------------
for n in "$SERVER" "$CLIENT"; do
  step_done "$n" g5-setup || start_step "$n" g5-setup "bench/cloudlab/setup_node.sh"
done
wait_step "$SERVER" g5-setup
wait_step "$CLIENT" g5-setup

step_done "$SERVER" g5-build-pg ||
  start_step "$SERVER" g5-build-pg "./build_postgres.sh && SKIP_PRISM=1 ./build_extensions.sh"
step_done "$CLIENT" g5-build-ginger || start_step "$CLIENT" g5-build-ginger "./build_ginger.sh"
wait_step "$SERVER" g5-build-pg
# shellcheck disable=SC2016  # expanded on the node
step "$SERVER" g5-build-forks '
  set -eo pipefail
  PGC=$PWD/postgres/install/bin/pg_config
  for d in pgvectorfast_pipnn pgvectorfast_ntq; do
    make -C $d clean PG_CONFIG=$PGC > /dev/null
    make -C $d -j"$(nproc)" PG_CONFIG=$PGC 2>&1 | grep -E " error|error:" || true
    so=$d/vectorfast.so; st=$(strings $so)
    grep -q hnsw.build_graph_file <<< "$st" || { echo "error: no hnsw.build_graph_file in $so"; exit 1; }
    [ "$(nm $so | grep -c " calico_")" -gt 0 ] || { echo "error: no calico symbols in $so"; exit 1; }
    echo "built $so"
  done
  grep -q "quantizer \"tq4\"" <<< "$(strings pgvectorfast_ntq/vectorfast.so)" ||
    { echo "error: no quantizer tq4 in pgvectorfast_ntq/vectorfast.so"; exit 1; }
  [ "$(nm postgres/stock/vector.so | grep -c " calico_" || true)" -eq 0 ] || { echo "error: stock vector.so has calico"; exit 1; }
  b=pgvectorfast_pipnn/tools/pipnn_export/build
  cmake -S pgvectorfast_pipnn/tools/pipnn_export -B $b -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DPG_CONFIG=$PGC > $b.configure.log 2>&1 || { tail -30 $b.configure.log; exit 1; }
  cmake --build $b -j"$(nproc)" --target pipnn_export 2>&1 | tail -3
  [ -x $b/pipnn_export ] && echo "built $b/pipnn_export"'
wait_step "$CLIENT" g5-build-ginger

# ---- 1. The full dataset on both nodes, then check the copies agree ----------
# shellcheck disable=SC2016  # expanded on the node
data_cmd='cd bench && .venv/bin/python -c "import sys, datasets; ds = datasets.load(sys.argv[1], num_queries=int(sys.argv[2])); print(ds.name, ds.train.shape, ds.test.shape)" hackernews "$NUM_QUERIES" && df -h /mydata /mydata_hdd'
for n in "$SERVER" "$CLIENT"; do
  step_done "$n" g5-data || start_step "$n" g5-data "$data_cmd"
done
wait_step "$SERVER" g5-data
wait_step "$CLIENT" g5-data
sums=()
for n in "$SERVER" "$CLIENT"; do
  sums+=("$(ssh_to "$n" "cd $DEST/bench/data && sha256sum hackernews_minilm_384_train.npy hackernews_minilm_384_meta.npz")")
done
printf '%s\n' "${sums[@]}" >&2
[ "${sums[0]}" = "${sums[1]}" ] || { echo "error: the two nodes' dataset files differ" >&2; exit 1; }

# ---- 2. Indexes ----------------------------------------------------------------
require_measurement_state
for n in "$SERVER" "$CLIENT"; do
  ( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
      bench/run_experiment.sh bench/cloudlab/hn.env bench/cloudlab/recall_gate.py \
      bench/cloudlab/ginger_thp_promote.py "$REMOTE_USER@$n.$DOMAIN:$DEST/" )
  ssh_to "$n" "mkdir -p $DEST/$PD"
done

# Client: Ginger's recommended index, timed (one throwaway point), cached for
# the measured runs. The subset's files are written first in a smoke run.
# shellcheck disable=SC2016  # expanded on the node
step_done "$CLIENT" "$T-ginger-build" || start_step "$CLIENT" "$T-ginger-build" '
  set -e
  (cd bench && .venv/bin/python -c "import sys, datasets; datasets.load(sys.argv[1], num_queries=int(sys.argv[2]))" "$DATASET" "$NUM_QUERIES")
  $NUMA_CMD bench/ginger/build/ginger_bench --data bench/data --dataset '"$DS"' --config recommended --rebuild \
    --sweep 10 --k 10 --num-queries "$NUM_QUERIES" --duration 1 --threads 1 --results '"$PD"'/ginger-build.jsonl
  ls -la bench/data/ginger/'

# Server: main table + HNSW m=32 (stock pgvector's CREATE INDEX), then the
# PiPNN table, graph and float import.
step "$SERVER" "$T-build-main" "M=32 EF_CONSTRUCTION=64 BUILD_MWM=100GB bench/run_experiment.sh build"
# shellcheck disable=SC2016  # expanded on the node
step "$SERVER" "$T-check-main" '
  l=/mydata/steps/'"$T"'-build-main.log
  grep -q "no spill" $l || { echo "error: the build spilled out of maintenance_work_mem"; exit 1; }
  grep -E "built in|no spill|index .* GB|table .* GB" $l | head; df -h /mydata'
step "$SERVER" "$T-build-pipnn" "FAST_SO=\$PIPNN_SO bench/run_experiment.sh build_pipnn"
wait_step "$CLIENT" "$T-ginger-build"

# ---- helpers for the measured passes -------------------------------------------
step "$CLIENT" "$T-client-idle" \
  'if pgrep -x ginger_bench >/dev/null; then echo "error: a ginger_bench is running"; exit 1; fi
   if postgres/install/bin/pg_ctl -D postgres/data status >/dev/null 2>&1; then echo "error: Postgres runs on the client"; exit 1; fi; echo idle'

# CPU samplers on both nodes and the server's link monitor, every 2 s.
ssh_to "$SERVER" "if [ -f $STEPS/nic-$NIC.pid ] && kill -0 \$(cat $STEPS/nic-$NIC.pid) 2>/dev/null; then echo 'nic monitor running'; else nohup sh -c 'while :; do echo \"\$(date +%s) \$(cat /sys/class/net/$NIC/carrier 2>/dev/null) \$(cat /sys/class/net/$NIC/statistics/rx_crc_errors) \$(cat /sys/class/net/$NIC/statistics/rx_packets)\"; sleep 2; done' >> $STEPS/nic-$NIC.txt 2>&1 < /dev/null & echo \$! > $STEPS/nic-$NIC.pid; echo nic monitor started; fi"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "if [ -f $STEPS/$T-cpu.pid ] && kill -0 \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; then echo 'cpu sampler running'; else nohup sh -c 'while :; do echo \"\$(date +%s) \$(head -1 /proc/stat)\"; sleep 2; done' >> $STEPS/$T-cpu.txt 2>&1 < /dev/null & echo \$! > $STEPS/$T-cpu.pid; echo cpu sampler started; fi"
done

PAGES=""   # hp | 4k, set per pass
serve() {  # serve <name> <serve env> <serve arm> [table] [HP_MIN_GB]
  local opts
  if [ "$PAGES" = hp ]; then opts="EXTRA_PG_OPTS='-c huge_pages=on'"; else opts="EXTRA_PG_OPTS='-c huge_pages=off'"; fi
  step "$SERVER" "$T-$PAGES-serve-$1" \
    "$opts $2 SERVE_ARM=$3 ${4:+SERVE_TABLE=$4} LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
  if [ "$PAGES" = hp ]; then
    step "$SERVER" "$T-$PAGES-hpcheck-$1" "HP_MIN_GB=${5:-$HPMIN}; $HP_CHECK"
  else
    # shellcheck disable=SC2016  # expanded on the node
    step "$SERVER" "$T-$PAGES-4kcheck-$1" '
      s=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW huge_pages_status")
      pid=$(head -1 postgres/data/postmaster.pid)
      h=$(awk "/^KernelPageSize:/ && \$2 != 4 {n++} END {print n+0}" /proc/$pid/smaps)
      echo "huge_pages_status: $s; mappings with a page size other than 4 KB: $h; shmem THP: $(cat /sys/kernel/mm/transparent_hugepage/shmem_enabled)"
      grep -c " interleave" /proc/$pid/numa_maps | sed "s/^/mappings with interleave policy: /"
      [ "$s" = off ] && [ "$h" = 0 ] || { echo "error: the server is not on 4 KB pages"; exit 1; }'
  fi
}
sweeps() {  # sweeps <arm name> <ann_bench args>: 1 client, then CLIENTS
  local L; if [ "$PAGES" = hp ]; then L=$LHP; else L=$L4K; fi
  bench_step "$CLIENT" "$T-$PAGES-$1-c1" "$L-$1" "$2 --sweep \"\$EF_SWEEP\" --clients 1"
  bench_step "$CLIENT" "$T-$PAGES-$1-c$CLIENTS" "$L-$1" "$2 --sweep \"\$EF_SWEEP\" --clients $CLIENTS"
}
# Ginger on the client node at 1 and GTHREADS threads, sampled; kept only if
# its pages are what the pass says (ginger_thp_promote.py). hp: jemalloc asks
# for THP on every mapping; 4k: the default (THP is madvise, so 4 KB).
# shellcheck disable=SC2016  # expanded on the node
SAMPLER='
  f='"$PD"'/ginger-$ARM-t$THR.smaps.csv
  echo "t,rss_kb,anon_kb,anon_huge_kb" > $f
  pid=""
  for i in $(seq 600); do pid=$(pgrep -x ginger_bench | head -1); [ -n "$pid" ] && break; sleep 1; done
  [ -n "$pid" ] || { echo "error: ginger_bench never started"; exit 1; }
  echo "sampling pid $pid"
  while kill -0 "$pid" 2>/dev/null; do
    awk -v t="$(date +%s)" "/^Rss:/ {r=\$2} /^Anonymous:/ {a=\$2} /^AnonHugePages:/ {h=\$2} END {if (r) print t\",\"r\",\"a\",\"h}" /proc/$pid/smaps_rollup >> $f 2>/dev/null || true
    sleep '"$SAMPLE_S"'
  done
  echo "$(( $(wc -l < $f) - 1 )) samples"'
ginger() {
  local t mc label mode out
  if [ "$PAGES" = hp ]; then mc="MALLOC_CONF=thp:always,metadata_thp:always"; label="$LHP-ginger"; mode=thp
  else mc=""; label="$L4K-ginger"; mode=4k; fi
  for t in 1 "$GTHREADS"; do
    out="$PD/ginger-$PAGES-t$t.jsonl"
    step_done "$CLIENT" "$T-$PAGES-ginger-t$t-promote" && { say "$T-$PAGES-ginger-t$t: already done"; continue; }
    # Not promoted yet: measure again, sampled, even if a run finished (its
    # samples are what promote judges, so a run without them can't be kept).
    start_step "$CLIENT" "$T-$PAGES-ginger-t$t-sample" "ARM=$PAGES THR=$t; $SAMPLER"
    start_step "$CLIENT" "$T-$PAGES-ginger-t$t" \
        "rm -f $out && sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null && echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null && $mc \$NUMA_CMD bench/ginger/build/ginger_bench --data bench/data --dataset $DS --config recommended --sweep \"\$GINGER_SWEEP\" --k 10 --num-queries \"\$NUM_QUERIES\" --duration \"\$DURATION\" --threads $t --results $out"
    wait_step "$CLIENT" "$T-$PAGES-ginger-t$t"
    wait_step "$CLIENT" "$T-$PAGES-ginger-t$t-sample"
    step "$CLIENT" "$T-$PAGES-ginger-t$t-promote" \
      "bench/.venv/bin/python bench/cloudlab/ginger_thp_promote.py $out $PD/ginger-$PAGES-t$t.smaps.csv $label $mode $JSONL"
  done
}

# bg_step <node> <name> <command>: start a background step unless it already
# succeeded or is still running (a log and no marker), e.g. after the driver
# stopped and was restarted while it ran; wait_step joins it later.
bg_step() {
  local st
  st="$(ssh_to "$1" "if [ -f $STEPS/$2.done ]; then echo \"done \$(cat $STEPS/$2.done)\"; elif [ -f $STEPS/$2.log ]; then echo running; else echo new; fi")" || give_up "$1"
  case "$st" in
    "done 0") say "$2: already done, skipping" ;;
    running) say "$2: still running from before, joining it" ;;
    *) start_step "$@" ;;
  esac
}

# ---- 3. Huge pages -------------------------------------------------------------
PAGES=hp
step "$SERVER" "$T-hp-reserve" "$HP_RESERVE"
serve pgv-hnsw '' stock
sweeps pgvector-hnsw "$MAIN_ARGS"
serve pvf-hnsw 'FAST_SO=$PIPNN_SO' fast
sweeps pgvectorfast-hnsw "$MAIN_ARGS"
serve pvf-pipnn 'FAST_SO=$PIPNN_SO' fast '$PIPNN_TABLE'
sweeps pgvectorfast-pipnn "$PIP_ARGS"
serve pgv-pipnn '' stock '$PIPNN_TABLE'
sweeps pgvector-pipnn "$PIP_ARGS"

# The tq4 import in place of the float index (server), while Ginger has the
# idle client node to itself.
step "$SERVER" "$T-hp-release-tq4" "$HP_RELEASE"
bg_step "$SERVER" "$T-build-tq4" \
  "FAST_SO=$NTQ_SO PIPNN_QUANTIZER=tq4 PIPNN_DROP=\"$FLOAT_IDX\" bench/run_experiment.sh build_pipnn"
ginger
wait_step "$SERVER" "$T-build-tq4"
step "$SERVER" "$T-hp-reserve-tq4" "$HP_RESERVE"
serve pvf-tq4 "FAST_SO=$NTQ_SO" fast '$PIPNN_TABLE' "$HPMIN_TQ"
sweeps pgvectorfast-tq4 "$TQ_ARGS"
step "$SERVER" "$T-hp-release" "$HP_RELEASE"

# ---- 4. 4 KB pages -------------------------------------------------------------
PAGES=4k
serve pvf-tq4 "FAST_SO=$NTQ_SO" fast '$PIPNN_TABLE'
sweeps pgvectorfast-tq4 "$TQ_ARGS"
step "$SERVER" "$T-4k-stop-tq4" "./run_postgres.sh stop"

# The float index back in place of tq4 (server), while Ginger's 4 KB run has
# the client node.
bg_step "$SERVER" "$T-restore-float" \
  "FAST_SO=\$PIPNN_SO PIPNN_DROP=\"$TQ_IDX\" bench/run_experiment.sh build_pipnn"
ginger
wait_step "$SERVER" "$T-restore-float"

serve pvf-pipnn 'FAST_SO=$PIPNN_SO' fast '$PIPNN_TABLE'
bench_step "$CLIENT" "$T-4k-pgvectorfast-pipnn-c1" "$L4K-pgvectorfast-pipnn" "$PIP_ARGS --sweep \"\$EF_SWEEP\" --clients 1"
# The re-imported index must search exactly like the one the huge-pages pass measured.
step "$CLIENT" "$T-restore-gate" \
  "bench/.venv/bin/python bench/cloudlab/recall_gate.py $L4K-pgvectorfast-pipnn $LHP-pgvectorfast-pipnn --tol 0 --results $JSONL $GATE"
sweeps pgvectorfast-pipnn "$PIP_ARGS"
serve pgv-pipnn '' stock '$PIPNN_TABLE'
sweeps pgvector-pipnn "$PIP_ARGS"
serve pgv-hnsw '' stock
sweeps pgvector-hnsw "$MAIN_ARGS"
serve pvf-hnsw 'FAST_SO=$PIPNN_SO' fast
sweeps pgvectorfast-hnsw "$MAIN_ARGS"
step "$SERVER" "$T-stop" "./run_postgres.sh stop || true; sudo sysctl -w vm.nr_hugepages=0"

# ---- 5. Checks, reports, results back ------------------------------------------
# Stock and fast search the same graph, so their recall must agree.
for L in "$LHP" "$L4K"; do
  for g in hnsw pipnn; do
    step "$CLIENT" "$T-gate-$L-$g" \
      "bench/.venv/bin/python bench/cloudlab/recall_gate.py $L-pgvector-$g $L-pgvectorfast-$g --results $JSONL $GATE"
  done
done

for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "kill \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; rm -f $STEPS/$T-cpu.pid; mkdir -p $DEST/bench/results/logs && cp $STEPS/g5-*.log $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
ssh_to "$SERVER" "kill \$(cat $STEPS/nic-$NIC.pid) 2>/dev/null; rm -f $STEPS/nic-$NIC.pid; true"

LPD="$HERE/../../$PD"
mkdir -p "$LPD/sat"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-cpu.txt" "$LPD/server-cpu.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$CLIENT.$DOMAIN:$STEPS/$T-cpu.txt" "$LPD/client-cpu.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/nic-$NIC.txt" "$LPD/nic.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$CLIENT.$DOMAIN:$DEST/$PD/" "$LPD/"
ssh_to "$CLIENT" "grep -E '\"label\": \"($LHP|$L4K)-' $DEST/$JSONL" > "$LPD/results.jsonl"
# saturation_report.py finds each point in the server's CPU trace, which only
# works for the saturated Postgres sweeps.
python3 - "$LPD/results.jsonl" "$CLIENTS" > "$LPD/sat/results.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("clients") == int(sys.argv[2]) and "ginger" not in r["label"]:
        print(line, end="")
PY
cp "$LPD/server-cpu.txt" "$LPD/client-cpu.txt" "$LPD/nic.txt" "$LPD/sat/"
python3 "$HERE/saturation_report.py" "$LPD/sat" | tee "$LPD/saturation.md"
say "done; labels $LHP-* and $L4K-* in $PD/results.jsonl; then bench/cloudlab/sync.sh pull"
