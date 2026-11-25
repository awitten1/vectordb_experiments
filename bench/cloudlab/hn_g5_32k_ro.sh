#!/usr/bin/env bash
# pgvectorfast + PiPNN graph + tq4 (ntq12 rerank) on 32 KB pages, with the
# graph in Ginger's reordered node order (RECURSIVE_GRAPH_PARTITIONING, as
# Ginger's recommended config uses), measured exactly as hn_g5_32k.sh measured
# the same arm in table order (label g5hp32k-pgvectorfast-tq4): huge pages,
# QUERY_SB, 1 and 60 clients over the LAN, the same sweep. Label
# g5hp32kro-pgvectorfast-tq4.
#
# No C changes. The graph import lays index elements (and tq4 rerank tuples)
# out in the order it scans the table (pgvectorfast_ntq/src/hnswbuild.c), so a
# table loaded in the graph's node order gives an index in that order:
#   1. dataset files (bench/data) to the HDD, behind a symlink, to make room on
#      /mydata; a tablespace directory on the HDD for the new table. Sweeps run
#      from prewarmed shared_buffers, so the disk only affects setup time.
#   2. pipnn_export from the pgvectorfast pipnn-reorder branch (--reorder),
#      built incrementally in the 8 KB checkout's existing build directory
#      (the old binary is kept as pipnn_export.noreorder)
#   3. export the 32 KB items_hn_pipnn and build its PiPNN graph, reordered
#   4. row ids in node order (pipnn_retid.py --order-out); the new graph
#      against the one every earlier result used (graph_compare.py); pages per
#      expansion predicted for both layouts (graph_layout.py)
#   5. load items_hn_pipnn_ro in that order (PIPNN_ORDER), in the HDD tablespace
#   6. rewrite the graph's TIDs for it (pipnn_retid.py: 100% in physical order)
#   7. import it as tq4 + ntq12 (index on the SSD)
#   8. huge pages: reserve, serve, check, sweep at 1 and 60 clients, release
#   9. results, logs, CPU traces and saturation report into
#      bench/profiles/hn-g5-32k-ro/
# Nothing is dropped: the 32 KB cluster keeps items_hn_pipnn and its index.
#
# Run from the workstation (no sync.sh push: its --delete removes builds that
# exist only on the server):
#   caffeinate -i bench/cloudlab/hn_g5_32k_ro.sh
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
T="hng5-32kro-r$RUN"
LABEL=g5hp32kro-pgvectorfast-tq4
D32=/mydata/vectordb_experiments_32k
NTQ32="$D32/pgvectorfast_ntq/vectorfast.so"
PEXP="$DEST/pgvectorfast_pipnn/tools/pipnn_export/build/pipnn_export"
TABLE=items_hn_pipnn_ro
G32=/var/tmp/pipnn/items_hn_pipnn.32k.graph      # the graph every earlier result used
TIDS32=/var/tmp/pipnn/tids-32k.txt                # the 32 KB items_hn_pipnn's TIDs
GRO=/var/tmp/pipnn/items_hn_pipnn.ro.graph        # reordered, TIDs of items_hn_pipnn
GRO2=/var/tmp/pipnn/items_hn_pipnn.ro.32k.graph   # reordered, TIDs of items_hn_pipnn_ro
TIDSRO=/var/tmp/pipnn/tids-ro.txt
ORDER=/var/tmp/pipnn/ro.order.npy
HDDTS=/mydata_hdd/pg32k_tblspc
TQ_ARGS="--index hnsw --table $TABLE --m \$PIPNN_M --ef-construction 64 --quantizer tq4 --rerank ntq12"
PD=bench/profiles/hn-g5-32k-ro
PY=bench/.venv/bin/python

ENVS8="$ENVS"
ENVS32="${ENVS/cd $DEST /cd $D32 }"
[ "$ENVS32" != "$ENVS8" ] || { echo "error: could not point ENVS at $D32" >&2; exit 1; }
on32() { local save="$ENVS"; ENVS="$ENVS32"; "$@"; ENVS="$save"; }
tids_cmd() {  # tids_cmd <table> <file>: psql \copy of each row's block, offset and id
  echo "postgres/install/bin/psql -X -p \$PGPORT -U postgres -d postgres -c \"\\copy (SELECT (ctid::text::point)[0]::bigint, (ctid::text::point)[1]::int, id FROM $1) TO '$2' WITH (DELIMITER ' ')\""
}
NOPG='if pgrep -x postgres >/dev/null; then echo "error: a Postgres is running"; exit 1; fi'

require_measurement_state

# ---- 0. The changed files, to both checkouts; the branch's exporter source ------
SRC_TMP="$(mktemp -d)"
trap 'rm -rf "$SRC_TMP"' EXIT
mkdir -p "$SRC_TMP/pgvectorfast_pipnn/tools/pipnn_export"
git -C "$HERE/../../pgvectorfast_pipnn" show origin/pipnn-reorder:tools/pipnn_export/pipnn_export.cc \
  > "$SRC_TMP/pgvectorfast_pipnn/tools/pipnn_export/pipnn_export.cc"
for d in "$DEST" "$D32"; do
  ( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
      bench/ann_bench.py bench/run_experiment.sh bench/cloudlab/pipnn_retid.py \
      bench/cloudlab/graph_compare.py bench/cloudlab/graph_layout.py \
      "$REMOTE_USER@$SERVER.$DOMAIN:$d/" )
done
( cd "$SRC_TMP" && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    pgvectorfast_pipnn/tools/pipnn_export/pipnn_export.cc "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/" )

# ---- 1. Room on /mydata: the dataset files to the HDD; a tablespace directory ---
# shellcheck disable=SC2016  # expanded on the node
step "$SERVER" "$T-hdd" "$NOPG"'
  set -eo pipefail
  src='"$DEST"'/bench/data; dst=/mydata_hdd/bench_data
  if [ -L $src ]; then
    echo "already moved: $src -> $(readlink $src)"
  else
    mkdir -p $dst
    rsync -a $src/ $dst/
    a=$(cd $src && find . -type f -printf "%P %s\n" | sort | md5sum)
    b=$(cd $dst && find . -type f -printf "%P %s\n" | sort | md5sum)
    [ "$a" = "$b" ] || { echo "error: the copy on the HDD differs from $src (names or sizes)"; exit 1; }
    mv $src $src.moving && ln -s $dst $src && rm -rf $src.moving
    echo "moved $src -> $dst"
  fi
  mkdir -p '"$HDDTS"' && chmod 700 '"$HDDTS"'
  ls -la '"$DEST"'/bench/ | grep data; df -h /mydata /mydata_hdd'

# ---- 2. pipnn_export with --reorder ---------------------------------------------
# shellcheck disable=SC2016  # expanded on the node
step "$SERVER" "$T-exporter" '
  set -eo pipefail
  b=pgvectorfast_pipnn/tools/pipnn_export/build
  [ -f $b/pipnn_export.noreorder ] || cp -p $b/pipnn_export $b/pipnn_export.noreorder
  grep -q -- "--reorder" pgvectorfast_pipnn/tools/pipnn_export/pipnn_export.cc ||
    { echo "error: pipnn_export.cc is not the pipnn-reorder version"; exit 1; }
  cmake --build $b -j"$(nproc)" --target pipnn_export 2>&1 | tail -3
  # The option parser rejects an unknown algorithm by name (it exits before
  # connecting); an exporter without --reorder calls the option unknown.
  msg=$($b/pipnn_export --reorder none-such 2>&1 || true)
  echo "$msg"
  grep -q "must be partitioning" <<< "$msg" ||
    { echo "error: the rebuilt pipnn_export has no --reorder"; exit 1; }
  echo "pipnn_export has --reorder"'

# ---- 3. The reordered PiPNN graph of the 32 KB items_hn_pipnn -------------------
on32 step "$SERVER" "$T-export" "$NOPG"'
  set -eo pipefail
  PG_OPTS="-c shared_preload_libraries=vectorfast -c shared_buffers=2GB" ./run_postgres.sh start
  if [ ! -s '"$GRO"' ]; then
    '"$PEXP"' --conninfo "host=localhost port=$PGPORT user=postgres dbname=postgres" \
      --table items_hn_pipnn --column embedding --reorder partitioning --out '"$GRO"'
  fi
  ./run_postgres.sh stop
  ls -la /var/tmp/pipnn; df -h /'

# ---- 4. Node order as row ids; same graph as before?; predicted pages ------------
step "$SERVER" "$T-order" "
  set -eo pipefail
  $PY bench/cloudlab/pipnn_retid.py $GRO $TIDS32 --order-out $ORDER
  echo '--- the reordered graph against the graph every earlier result used'
  $PY bench/cloudlab/graph_compare.py $G32 $TIDS32 $GRO $TIDS32
  echo '--- pages per expansion, 46 tq4 elements per 32 KB page (heap: table order; node: reordered)'
  $PY bench/cloudlab/graph_layout.py $GRO --per-page 46"

# ---- 5. items_hn_pipnn_ro in node order, on the HDD; its TIDs -------------------
on32 step "$SERVER" "$T-load" "$NOPG"'
  set -eo pipefail
  PG_OPTS="-c shared_preload_libraries=vectorfast -c shared_buffers=2GB" ./run_postgres.sh start
  if [ -z "$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SELECT 1 FROM pg_tablespace WHERE spcname = '"'hdd'"'")" ]; then
    postgres/install/bin/psql -X -p $PGPORT -U postgres -d postgres -c "CREATE TABLESPACE hdd LOCATION '"'$HDDTS'"'"
  fi
  ./run_postgres.sh stop
  PIPNN_TABLE='"$TABLE"' PIPNN_ORDER='"$ORDER"' PIPNN_TABLESPACE=hdd PIPNN_LOAD_ONLY=1 \
    FAST_SO='"$NTQ32"' PIPNN_QUANTIZER=tq4 bench/run_experiment.sh build_pipnn
  ./run_postgres.sh start
  '"$(tids_cmd "$TABLE" "$TIDSRO")"'
  postgres/install/bin/psql -X -p $PGPORT -U postgres -d postgres -c "SELECT c.relname, t.spcname, pg_size_pretty(pg_relation_size(c.oid)) FROM pg_class c LEFT JOIN pg_tablespace t ON t.oid = c.reltablespace WHERE c.relname LIKE '"'items_hn_pipnn%'"'"
  ./run_postgres.sh stop
  wc -l '"$TIDSRO"'; df -h /mydata /mydata_hdd'

# ---- 6. The graph's TIDs for items_hn_pipnn_ro ------------------------------------
step "$SERVER" "$T-retid" "
  set -eo pipefail
  $PY bench/cloudlab/pipnn_retid.py $GRO $TIDS32 $TIDSRO $GRO2 | tee /dev/stderr |
    grep -q 'i-th row of the new table for 100.0000% of nodes' ||
    { echo 'error: the reordered table is not in the graph node order'; exit 1; }"

# ---- 7. tq4 import -------------------------------------------------------------------
on32 step "$SERVER" "$T-import" \
  "PIPNN_TABLE=$TABLE FAST_SO=$NTQ32 PIPNN_QUANTIZER=tq4 PIPNN_GRAPH=$GRO2 bench/run_experiment.sh build_pipnn && ./run_postgres.sh stop && df -h /mydata /mydata_hdd"

# ---- 8. Huge pages, 1 and 60 clients -------------------------------------------------
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
  echo "block_size: $b"; [ "$b" = 32768 ] || { echo "error: not a 32 KB cluster"; exit 1; }'
bench_step "$CLIENT" "$T-tq4-c1" "$LABEL" "$TQ_ARGS --sweep \"\$EF_SWEEP\" --clients 1"
bench_step "$CLIENT" "$T-tq4-c$CLIENTS" "$LABEL" "$TQ_ARGS --sweep \"\$EF_SWEEP\" --clients $CLIENTS"
on32 step "$SERVER" "$T-hp-release" "$HP_RELEASE"

# ---- 9. Results ---------------------------------------------------------------------
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
