#!/usr/bin/env bash
# The best Postgres arm (pgvectorfast + PiPNN + tq4 on 32 KB pages, reordered;
# label g5hp32kro-pgvectorfast-tq4) with its rerank capped at
# rerank_factor x k = 10 x 10 = 100 candidates (hnsw.rerank_limit=100, a new
# setting in pgvectorfast_ntq; 0, the default, re-ranks all ef_search
# candidates as before). Measured exactly as hn_g5_32k_ro.sh measured the
# uncapped arm: huge pages, QUERY_SB, 1 and 60 clients over the LAN, the same
# sweep. Label g5hp32kro-rr10-pgvectorfast-tq4.
#
# No index rebuild: the cap only changes which candidates a scan re-scores.
#   0. pgvectorfast_ntq's changed sources to the 32 KB checkout
#   1. rebuild its vectorfast.so against the 32 KB Postgres (the previous one
#      is kept as vectorfast.so.nocap)
#   2. huge pages: reserve, serve, check (and SHOW hnsw.rerank_limit = 0)
#   3. sweeps at 1 and 60 clients with --set hnsw.rerank_limit=100
#   4. results, logs, CPU traces and saturation report into bench/profiles/hn-g5-rr/
# At ef_search <= 100 the cap doesn't bind, so those points are a control: same
# recall as g5hp32kro, same QPS within the noise.
#
# Run from the workstation (no sync.sh push: its --delete removes builds that
# exist only on the server):
#   caffeinate -i bench/cloudlab/hn_g5_rr.sh
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
CLIENTS=60
RR=100
T="hng5-rr-r$RUN"
LABEL=g5hp32kro-rr10-pgvectorfast-tq4
D32=/mydata/vectordb_experiments_32k
NTQ32="$D32/pgvectorfast_ntq/vectorfast.so"
TABLE=items_hn_pipnn_ro
TQ_ARGS="--index hnsw --table $TABLE --m \$PIPNN_M --ef-construction 64 --quantizer tq4 --rerank ntq12 --set hnsw.rerank_limit=$RR"
PD=bench/profiles/hn-g5-rr

ENVS8="$ENVS"
ENVS32="${ENVS/cd $DEST /cd $D32 }"
[ "$ENVS32" != "$ENVS8" ] || { echo "error: could not point ENVS at $D32" >&2; exit 1; }
on32() { local save="$ENVS"; ENVS="$ENVS32"; "$@"; ENVS="$save"; }
NOPG='if pgrep -x postgres >/dev/null; then echo "error: a Postgres is running"; exit 1; fi'

require_measurement_state

# ---- 0. The changed sources ------------------------------------------------------
( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    pgvectorfast_ntq/src/hnsw.c pgvectorfast_ntq/src/hnsw.h pgvectorfast_ntq/src/hnswtqam.c \
    "$REMOTE_USER@$SERVER.$DOMAIN:$D32/" )

# ---- 1. vectorfast.so with hnsw.rerank_limit, against the 32 KB Postgres ------------
# shellcheck disable=SC2016  # expanded on the node
on32 step "$SERVER" "$T-build" "$NOPG"'
  set -eo pipefail
  PGC=$PWD/postgres/install/bin/pg_config
  so=pgvectorfast_ntq/vectorfast.so
  [ -f $so.nocap ] || cp -p $so $so.nocap
  rm -f $so
  make -C pgvectorfast_ntq -j"$(nproc)" PG_CONFIG=$PGC 2>&1 | grep -E " error|error:|warning:" || true
  st=$(strings $so)
  grep -q hnsw.rerank_limit <<< "$st" || { echo "error: no hnsw.rerank_limit in $so"; exit 1; }
  grep -q "quantizer \"tq4\"" <<< "$st" || { echo "error: no quantizer tq4 in $so"; exit 1; }
  [ "$(nm $so | grep -c " calico_")" -gt 0 ] || { echo "error: no calico symbols in $so"; exit 1; }
  grep -qE "^#define BLCKSZ 32768$" "$($PGC --includedir-server)/pg_config.h" ||
    { echo "error: this Postgres is not built with 32 KB pages"; exit 1; }
  ls -la $so $so.nocap; echo "built $so with hnsw.rerank_limit"'

# ---- 2-3. Huge pages, 1 and 60 clients -------------------------------------------------
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
  r=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW hnsw.rerank_limit")
  echo "hnsw.rerank_limit default: $r"; [ "$r" = 0 ] || { echo "error: the served library has no hnsw.rerank_limit"; exit 1; }'
bench_step "$CLIENT" "$T-tq4-c1" "$LABEL" "$TQ_ARGS --sweep \"\$EF_SWEEP\" --clients 1"
bench_step "$CLIENT" "$T-tq4-c$CLIENTS" "$LABEL" "$TQ_ARGS --sweep \"\$EF_SWEEP\" --clients $CLIENTS"
on32 step "$SERVER" "$T-hp-release" "$HP_RELEASE"

# ---- 4. Results ---------------------------------------------------------------------
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
ssh_to "$CLIENT" "grep -E '\"label\": \"$LABEL\"' $DEST/bench/results/hackernews_minilm_384.jsonl" > "$LPD/results.jsonl"
python3 - "$LPD/results.jsonl" "$CLIENTS" > "$LPD/sat/results.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    if json.loads(line).get("clients") == int(sys.argv[2]):
        print(line, end="")
PY
cp "$LPD/server-cpu.txt" "$LPD/client-cpu.txt" "$LPD/nic.txt" "$LPD/sat/"
python3 "$HERE/saturation_report.py" "$LPD/sat" | tee "$LPD/saturation.md"
say "done; label $LABEL in $PD/results.jsonl"
