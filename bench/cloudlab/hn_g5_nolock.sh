#!/usr/bin/env bash
# What the per-query scan lock costs at low recall. Each search takes a shared
# heavyweight page lock (HNSW_SCAN_LOCK) twice; page locks can't use the fast
# path, so at ~50k QPS every query goes through one partition of the shared lock
# table, and hn_g5_netdiag.sh found backends waiting on LWLock:LockManager in
# 21% (60 clients) to 46% (90 clients) of samples. pgvectorfast_ntq gets a
# setting, hnsw.scan_lock (superuser; default on), that skips the lock: for
# measurement only, since a concurrent VACUUM could then free elements a scan
# is reading.
# Same server and arm as hn_g5_c90.sh (vector rerank, cap 100):
#   ef 10, 20, 40, 80, 160 at 60 clients, scan lock on then off (same session)
#   the same at 90 clients, scan lock off
#   net_diag.sh on both nodes at ef 10, 60 clients, scan lock off
# Labels g5nolock-on, g5nolock-off. Everything into bench/profiles/hn-nolock/.
#
#   caffeinate -i bench/cloudlab/hn_g5_nolock.sh
# Steps are detached and resumable (lib.sh); bump RUN to redo them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
export SMOKE=0
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"
# shellcheck source=bench/cloudlab/hp_lib.sh
. "$HERE/hp_lib.sh"

RUN="${RUN:-1}"
CLIENTS=90
RR=100
T="hng5-nolock-r$RUN"
LABEL=g5nolock
LOW_EF=10,20,40,80,160
CHECKSUM=251cc2ee3b2c0245     # tq_selftest on the codec before this change
D32=/mydata/vectordb_experiments_32k
NTQ32="$D32/pgvectorfast_ntq/vectorfast.so"
TABLE=items_hn_pipnn_ro
TQ_ARGS="--index hnsw --table $TABLE --m \$PIPNN_M --ef-construction 64 --quantizer tq4 --rerank ntq12 --set hnsw.rerank_limit=$RR"
PD=bench/profiles/hn-nolock

ENVS8="$ENVS"
ENVS32="${ENVS/cd $DEST /cd $D32 }"
[ "$ENVS32" != "$ENVS8" ] || { echo "error: could not point ENVS at $D32" >&2; exit 1; }
on32() { local save="$ENVS"; ENVS="$ENVS32"; "$@"; ENVS="$save"; }
NOPG='if pgrep -x postgres >/dev/null; then echo "error: a Postgres is running"; exit 1; fi'

require_measurement_state

# ---- 0. The changed sources; vectorfast.so with hnsw.scan_lock ---------------------
( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    pgvectorfast_ntq/src/hnsw.c pgvectorfast_ntq/src/hnsw.h pgvectorfast_ntq/src/hnswscan.c \
    "$REMOTE_USER@$SERVER.$DOMAIN:$D32/" )
# shellcheck disable=SC2016  # expanded on the node
on32 step "$SERVER" "$T-build" "$NOPG"'
  set -eo pipefail
  PGC=$PWD/postgres/install/bin/pg_config
  so=pgvectorfast_ntq/vectorfast.so
  [ -f $so.simd ] || cp -p $so $so.simd
  make -C pgvectorfast_ntq clean PG_CONFIG=$PGC > /dev/null
  make -C pgvectorfast_ntq -j"$(nproc)" PG_CONFIG=$PGC 2>&1 | grep -E " error|error:|hnswscan.c.*warning:" || true
  st=$(strings $so)
  for s in hnsw.scan_lock hnsw.rerank_kernel hnsw.rerank_limit "quantizer \"tq4\""; do
    grep -q "$s" <<< "$st" || { echo "error: no $s in $so"; exit 1; }
  done
  [ "$(nm $so | grep -c " calico_")" -gt 0 ] || { echo "error: no calico symbols in $so"; exit 1; }
  grep -qE "^#define BLCKSZ 32768$" "$($PGC --includedir-server)/pg_config.h" ||
    { echo "error: this Postgres is not built with 32 KB pages"; exit 1; }
  ls -la $so $so.simd; echo "built $so with hnsw.scan_lock"'

# ---- 1. Huge pages; scan lock on and off -------------------------------------------------
step "$CLIENT" "$T-client-idle" \
  'if pgrep -x ginger_bench >/dev/null; then echo "error: a ginger_bench is running"; exit 1; fi; echo idle'
ssh_to "$SERVER" "if [ -f $STEPS/nic-$NIC.pid ] && kill -0 \$(cat $STEPS/nic-$NIC.pid) 2>/dev/null; then echo 'nic monitor running'; else nohup sh -c 'while :; do echo \"\$(date +%s) \$(cat /sys/class/net/$NIC/carrier 2>/dev/null) \$(cat /sys/class/net/$NIC/statistics/rx_crc_errors) \$(cat /sys/class/net/$NIC/statistics/rx_packets)\"; sleep 2; done' >> $STEPS/$T-nic.txt 2>&1 < /dev/null & echo \$! > $STEPS/nic-$NIC.pid; echo nic monitor started; fi"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "if [ -f $STEPS/$T-cpu.pid ] && kill -0 \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; then echo 'cpu sampler running'; else nohup sh -c 'while :; do echo \"\$(date +%s) \$(head -1 /proc/stat)\"; sleep 2; done' >> $STEPS/$T-cpu.txt 2>&1 < /dev/null & echo \$! > $STEPS/$T-cpu.pid; echo cpu sampler started; fi"
done

on32 step "$SERVER" "$T-hp-reserve" "$HP_RESERVE"
on32 step "$SERVER" "$T-hp-serve" \
  "EXTRA_PG_OPTS='-c huge_pages=on' FAST_SO=$NTQ32 SERVE_ARM=fast SERVE_TABLE=$TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
# shellcheck disable=SC2016  # expanded on the node
on32 step "$SERVER" "$T-hp-check" 'HP_MIN_GB=50; '"$HP_CHECK"'
  b=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW block_size")
  echo "block_size: $b"; [ "$b" = 32768 ] || { echo "error: not a 32 KB cluster"; exit 1; }
  r=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW hnsw.rerank_kernel")
  echo "hnsw.rerank_kernel default: $r"; [ "$r" = auto ] || { echo "error: the served library has no hnsw.rerank_kernel"; exit 1; }
  r=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW hnsw.scan_lock")
  echo "hnsw.scan_lock default: $r"; [ "$r" = on ] || { echo "error: the served library has no hnsw.scan_lock"; exit 1; }
  r=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SET hnsw.scan_lock = off" -c "SHOW hnsw.scan_lock" | tail -1)
  echo "hnsw.scan_lock after SET off: $r"; [ "$r" = off ] || { echo "error: cannot turn hnsw.scan_lock off"; exit 1; }'
( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" bench/cloudlab/net_diag.sh "$REMOTE_USER@$SERVER.$DOMAIN:$D32/" )
( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" bench/cloudlab/net_diag.sh "$REMOTE_USER@$CLIENT.$DOMAIN:$DEST/" )
DIAG="NIC=$NIC PROFILE_DIR=hn-nolock bench/cloudlab/net_diag.sh"

# diag_run <name> <label> <ann_bench args>: net_diag.sh on both nodes around one run
diag_run() {
  local name="$1" label="$2"; shift 2
  if step_done "$CLIENT" "$T-$name"; then say "$T-$name: already done, skipping"; return; fi
  on32 start_step "$SERVER" "$T-diag-s-$name" "PGWAIT=1 $DIAG $name-server 25 20"
  start_step "$CLIENT" "$T-diag-c-$name" "$DIAG $name-client 25 20"
  bench_step "$CLIENT" "$T-$name" "$label" "$*"
  on32 wait_step "$SERVER" "$T-diag-s-$name"
  wait_step "$CLIENT" "$T-diag-c-$name"
}
bench_step "$CLIENT" "$T-on-c60" "$LABEL-on" "$TQ_ARGS --sweep $LOW_EF --clients 60"
bench_step "$CLIENT" "$T-off-c60" "$LABEL-off" "$TQ_ARGS --set hnsw.scan_lock=off --sweep $LOW_EF --clients 60"
bench_step "$CLIENT" "$T-off-c90" "$LABEL-off" "$TQ_ARGS --set hnsw.scan_lock=off --sweep $LOW_EF --clients 90"
diag_run off-ef10-c60 "$LABEL-offdiag" "$TQ_ARGS --set hnsw.scan_lock=off --sweep 10 --clients 60"
on32 step "$SERVER" "$T-hp-release" "$HP_RELEASE"

# ---- 2. Results ---------------------------------------------------------------------
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "kill \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; rm -f $STEPS/$T-cpu.pid; true"
done
ssh_to "$SERVER" "kill \$(cat $STEPS/nic-$NIC.pid) 2>/dev/null; rm -f $STEPS/nic-$NIC.pid; true"
LPD="$HERE/../../$PD"
mkdir -p "$LPD/sat"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-cpu.txt" "$LPD/server-cpu.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$CLIENT.$DOMAIN:$STEPS/$T-cpu.txt" "$LPD/client-cpu.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-nic.txt" "$LPD/nic.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-*.log" "$LPD/"
ssh_to "$CLIENT" "grep -E '\"label\": \"$LABEL-' $DEST/bench/results/hackernews_minilm_384.jsonl" > "$LPD/results.jsonl"
mkdir -p "$LPD/server" "$LPD/client"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$D32/bench/profiles/hn-nolock/" "$LPD/server/"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$CLIENT.$DOMAIN:$DEST/bench/profiles/hn-nolock/" "$LPD/client/"
python3 - "$LPD/results.jsonl" "$CLIENTS" > "$LPD/sat/results.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    if json.loads(line).get("clients") > 1:
        print(line, end="")
PY
cp "$LPD/server-cpu.txt" "$LPD/client-cpu.txt" "$LPD/nic.txt" "$LPD/sat/"
python3 "$HERE/saturation_report.py" "$LPD/sat" | tee "$LPD/saturation.md"
say "done; label $LABEL in $PD/results.jsonl"
