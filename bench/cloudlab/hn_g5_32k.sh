#!/usr/bin/env bash
# pgvectorfast + PiPNN graph + tq4 (ntq12 rerank) on a Postgres built with
# 32 KB pages, on the c220g5 pair, measured exactly as hn_g5.sh measured the
# 8 KB arm with huge pages (label g5hp-pgvectorfast-tq4): huge_pages=on,
# QUERY_SB, 1 and 60 clients over the LAN, the same sweep. Label
# g5hp32k-pgvectorfast-tq4.
#
# The page size is fixed when Postgres is compiled, and the scripts keep
# Postgres under the repo (postgres/install, postgres/data), so the 32 KB
# cluster lives in its own checkout, D32 below, next to the 8 KB one, which
# this driver never writes to. D32 shares the dataset (bench/data) and the
# Python environment (bench/.venv) with it through symlinks.
#
# The graph is the one the 8 KB arm searched, not a new PiPNN build: the graph
# file keys nodes by heap TID, so pipnn_retid.py rewrites its TIDs for the
# 32 KB table, matching rows by id. The TurboQuant seed is fixed, so the codes
# are the same too, and recall should match the 8 KB run (recall_gate.py).
#
#   1. D32: copy of the code; Postgres with PG_BLOCKSIZE=32, extensions, and
#      the pgvectorfast_ntq fork built against it
#   2. load items_hn_pipnn into the 32 KB cluster (unlogged, as before) and
#      list its rows' TIDs; list the 8 KB table's too (that cluster started
#      read-only in effect and stopped cleanly, which keeps its unlogged table)
#   3. rewrite the graph's TIDs; import it as tq4 + ntq12
#   4. huge pages: reserve, serve, check (huge pages, 32 KB block size), sweep
#      at 1 and 60 clients, release
#   5. recall against the 8 KB arm; results, CPU traces and saturation report
#      into bench/profiles/hn-g5-32k/
#
# Run from the workstation, with the code pushed (sync.sh push; no build running):
#   caffeinate -i bench/cloudlab/hn_g5_32k.sh
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
T="hng5-32k-r$RUN"
LABEL=g5hp32k-pgvectorfast-tq4
REF=g5hp-pgvectorfast-tq4
JSONL=bench/results/hackernews_minilm_384.jsonl
D32=/mydata/vectordb_experiments_32k
NTQ32="$D32/pgvectorfast_ntq/vectorfast.so"
G8=/var/tmp/pipnn/items_hn_pipnn.graph
G32=/var/tmp/pipnn/items_hn_pipnn.32k.graph
TQ_ARGS='--index hnsw --table $PIPNN_TABLE --m $PIPNN_M --ef-construction 64 --quantizer tq4 --rerank ntq12'
PD=bench/profiles/hn-g5-32k

# Steps run in the 8 KB checkout (lib.sh's ENVS) unless wrapped in on32,
# which runs them in D32 with D32's hn.env.
ENVS8="$ENVS"
ENVS32="${ENVS/cd $DEST /cd $D32 }"
[ "$ENVS32" != "$ENVS8" ] || { echo "error: could not point ENVS at $D32" >&2; exit 1; }
on32() { local save="$ENVS"; ENVS="$ENVS32"; "$@"; ENVS="$save"; }
TIDS='\copy (SELECT (ctid::text::point)[0]::bigint, (ctid::text::point)[1]::int, id FROM items_hn_pipnn) TO'

require_measurement_state
# Only the files this run changed, not sync.sh push: its --delete would remove
# builds that exist only on the server (pipnn_export's build directory).
( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    build_postgres.sh bench/run_experiment.sh bench/cloudlab/pipnn_retid.py bench/cloudlab/hn.env \
    "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/" )

# ---- 1. The 32 KB checkout and its builds --------------------------------------
# shellcheck disable=SC2016  # expanded on the node
step "$SERVER" "$T-tree" '
  set -e
  if pgrep -x postgres >/dev/null; then echo "error: a Postgres is running"; exit 1; fi
  mkdir -p '"$D32"'
  rsync -a --exclude /postgres/ --exclude /bench/data --exclude /bench/.venv \
    --exclude /bench/results/ --exclude /bench/profiles/ \
    --exclude "*.o" --exclude "*.so" --exclude "*.bc" '"$DEST"'/ '"$D32"'/
  ln -sfn '"$DEST"'/bench/data '"$D32"'/bench/data
  ln -sfn '"$DEST"'/bench/.venv '"$D32"'/bench/.venv
  mkdir -p '"$D32"'/bench/results
  ls -la '"$D32"' | head -40; df -h /mydata'

# shellcheck disable=SC2016  # expanded on the node
on32 step "$SERVER" "$T-build" '
  set -eo pipefail
  PG_BLOCKSIZE=32 ./build_postgres.sh > '"$STEPS/$T"'-build-pg.out 2>&1 ||
    { tail -40 '"$STEPS/$T"'-build-pg.out; exit 1; }
  SKIP_PRISM=1 ./build_extensions.sh > '"$STEPS/$T"'-build-ext.out 2>&1 ||
    { tail -40 '"$STEPS/$T"'-build-ext.out; exit 1; }
  PGC=$PWD/postgres/install/bin/pg_config
  make -C pgvectorfast_ntq clean PG_CONFIG=$PGC > /dev/null
  make -C pgvectorfast_ntq -j"$(nproc)" PG_CONFIG=$PGC 2>&1 | grep -E " error|error:" || true
  so=pgvectorfast_ntq/vectorfast.so; st=$(strings $so)
  grep -q hnsw.build_graph_file <<< "$st" || { echo "error: no hnsw.build_graph_file in $so"; exit 1; }
  grep -q "quantizer \"tq4\"" <<< "$st" || { echo "error: no quantizer tq4 in $so"; exit 1; }
  [ "$(nm $so | grep -c " calico_")" -gt 0 ] || { echo "error: no calico symbols in $so"; exit 1; }
  h="$($PGC --includedir-server)/pg_config.h"
  grep -E "^#define BLCKSZ " "$h"
  grep -qE "^#define BLCKSZ 32768$" "$h" ||
    { echo "error: this Postgres is not built with 32 KB pages"; exit 1; }
  echo "built $so against 32 KB Postgres"'

# ---- 2. Load the 32 KB table; both tables' TIDs ---------------------------------
# shellcheck disable=SC2016  # expanded on the node
on32 step "$SERVER" "$T-load" '
  set -eo pipefail
  FAST_SO='"$NTQ32"' PIPNN_QUANTIZER=tq4 PIPNN_LOAD_ONLY=1 bench/run_experiment.sh build_pipnn
  postgres/install/bin/pg_controldata postgres/data | grep -E "Database block size"
  # run_experiment.sh stops the server when it exits; start it plainly to list the TIDs
  ./run_postgres.sh start
  postgres/install/bin/psql -X -p $PGPORT -U postgres -d postgres -c "'"$TIDS"' '"'/var/tmp/pipnn/tids-32k.txt'"' WITH (DELIMITER '"' '"')"
  ./run_postgres.sh stop
  wc -l /var/tmp/pipnn/tids-32k.txt; df -h /mydata'

# The 8 KB cluster, started plainly and stopped cleanly (a clean stop keeps an
# unlogged table), only to read its table's TIDs.
# shellcheck disable=SC2016  # expanded on the node
step "$SERVER" "$T-tids8k" '
  set -eo pipefail
  if pgrep -x postgres >/dev/null; then echo "error: a Postgres is running"; exit 1; fi
  ./run_postgres.sh start
  postgres/install/bin/psql -X -p $PGPORT -U postgres -d postgres -c "'"$TIDS"' '"'/var/tmp/pipnn/tids-8k.txt'"' WITH (DELIMITER '"' '"')"
  ./run_postgres.sh stop
  wc -l /var/tmp/pipnn/tids-8k.txt'

# ---- 3. The same graph, rewritten for the 32 KB table; tq4 import -------------
step "$SERVER" "$T-retid" \
  "bench/.venv/bin/python bench/cloudlab/pipnn_retid.py $G8 /var/tmp/pipnn/tids-8k.txt /var/tmp/pipnn/tids-32k.txt $G32"
on32 step "$SERVER" "$T-import" \
  "FAST_SO=$NTQ32 PIPNN_QUANTIZER=tq4 PIPNN_GRAPH=$G32 bench/run_experiment.sh build_pipnn && ./run_postgres.sh stop && df -h /mydata"

# ---- 4. Huge pages, 1 and 60 clients -------------------------------------------
step "$CLIENT" "$T-client-idle" \
  'if pgrep -x ginger_bench >/dev/null; then echo "error: a ginger_bench is running"; exit 1; fi; echo idle'
# CPU samplers on both nodes and the server's link monitor, every 2 s.
ssh_to "$SERVER" "if [ -f $STEPS/nic-$NIC.pid ] && kill -0 \$(cat $STEPS/nic-$NIC.pid) 2>/dev/null; then echo 'nic monitor running'; else nohup sh -c 'while :; do echo \"\$(date +%s) \$(cat /sys/class/net/$NIC/carrier 2>/dev/null) \$(cat /sys/class/net/$NIC/statistics/rx_crc_errors) \$(cat /sys/class/net/$NIC/statistics/rx_packets)\"; sleep 2; done' >> $STEPS/$T-nic.txt 2>&1 < /dev/null & echo \$! > $STEPS/nic-$NIC.pid; echo nic monitor started; fi"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "if [ -f $STEPS/$T-cpu.pid ] && kill -0 \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; then echo 'cpu sampler running'; else nohup sh -c 'while :; do echo \"\$(date +%s) \$(head -1 /proc/stat)\"; sleep 2; done' >> $STEPS/$T-cpu.txt 2>&1 < /dev/null & echo \$! > $STEPS/$T-cpu.pid; echo cpu sampler started; fi"
done

on32 step "$SERVER" "$T-hp-reserve" "$HP_RESERVE"
on32 step "$SERVER" "$T-hp-serve" \
  "EXTRA_PG_OPTS='-c huge_pages=on' FAST_SO=$NTQ32 SERVE_ARM=fast SERVE_TABLE=\$PIPNN_TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
# shellcheck disable=SC2016  # expanded on the node
on32 step "$SERVER" "$T-hp-check" 'HP_MIN_GB=50; '"$HP_CHECK"'
  b=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW block_size")
  echo "block_size: $b"; [ "$b" = 32768 ] || { echo "error: not a 32 KB cluster"; exit 1; }'
bench_step "$CLIENT" "$T-tq4-c1" "$LABEL" "$TQ_ARGS --sweep \"\$EF_SWEEP\" --clients 1"
bench_step "$CLIENT" "$T-tq4-c$CLIENTS" "$LABEL" "$TQ_ARGS --sweep \"\$EF_SWEEP\" --clients $CLIENTS"
on32 step "$SERVER" "$T-hp-release" "$HP_RELEASE"

# ---- 5. Checks, results ---------------------------------------------------------
# Same graph, same codes: recall should match the 8 KB arm exactly.
step "$CLIENT" "$T-gate" \
  "bench/.venv/bin/python bench/cloudlab/recall_gate.py $LABEL $REF --tol 0 --results $JSONL || \
   bench/.venv/bin/python bench/cloudlab/recall_gate.py $LABEL $REF --results $JSONL"

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
ssh_to "$CLIENT" "grep -E '\"label\": \"$LABEL\"' $DEST/$JSONL" > "$LPD/results.jsonl"
python3 - "$LPD/results.jsonl" "$CLIENTS" > "$LPD/sat/results.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    if json.loads(line).get("clients") == int(sys.argv[2]):
        print(line, end="")
PY
cp "$LPD/server-cpu.txt" "$LPD/client-cpu.txt" "$LPD/nic.txt" "$LPD/sat/"
python3 "$HERE/saturation_report.py" "$LPD/sat" | tee "$LPD/saturation.md"
say "done; label $LABEL in $PD/results.jsonl"
