#!/usr/bin/env bash
# Is it the graph? pgvectorfast (the awitten1/pgvectorfast fork) searching the
# HNSW graph it serves today against the same engine searching a graph built by
# Ginger's PiPNN, on HN 27.4M.
#
#   SMOKE=1 caffeinate -i bench/cloudlab/hn_pipnn.sh   # 1M rows: build, recall, probes
#   caffeinate -i bench/cloudlab/hn_pipnn.sh           # 27.4M
#
# Steps are detached and resumable (see lib.sh); bump RUN to redo them.
# Everything is built and run on the server node, except the LAN clients.
#   1. sync the fork (FORK, default the pgvectorfast_pipnn submodule, branch
#      pipnn-graph-import, with third_party/ginger) to pgvectorfast_pipnn/;
#      check free space
#   2. build its vectorfast.so, and tools/pipnn_export (Ginger from the
#      submodule, libpq from this Postgres)
#   3. run_experiment.sh build_pipnn: load PIPNN_TABLE, SELECT ctid, embedding
#      out of it into pipnn_export (Ginger's recommended PiPNN graph), then
#      CREATE INDEX ... WITH (m = 32) with hnsw.build_graph_file
#   SMOKE: 4. a local sweep on the PiPNN index (recall), and probe counts
#   full:  4. serve the fork on the main table (HNSW, m = 16): ef_search sweep
#             at 1 and 40 LAN clients, then hops and distances per query
#          5. the same on PIPNN_TABLE (PiPNN graph, m = 32)
#          6. stop Postgres; client results and counts into bench/profiles/hn-pipnn/
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = 1 ]; then T="hnpips-r$RUN"; else T="hnpip-r$RUN"; fi
FORK="${FORK:-$(cd "$HERE/../.." && pwd)/pgvectorfast_pipnn}"
PD=hn-pipnn
JSONL=hackernews_minilm_384.jsonl
# lib.sh exports SMOKE before sourcing hn.env on the node, so both sides pick
# the same dataset and PIPNN_TABLE.
PIPENV="FAST_SO=\$PIPNN_SO"
PROBES="hnswrescan HnswLoadUnvisitedRaw"
# Distances are computed inline in the fork's search loop, so they are counted
# as the per-hop number of unvisited neighbours, read on the line after
# HnswLoadUnvisitedRaw returns (the `if (tuples != NULL)` before the counter).

[ -f "$FORK/src/hnswbuild.c" ] && grep -q build_graph_file "$FORK/src/hnswbuild.c" ||
  { echo "error: $FORK is not the pgvectorfast fork with hnsw.build_graph_file" >&2; exit 1; }
[ -f "$FORK/third_party/ginger/CMakeLists.txt" ] ||
  { echo "error: $FORK/third_party/ginger is empty; git submodule update --init --recursive" >&2; exit 1; }
tl=$(grep -n '(\*tuples) += unvisitedLength;' "$FORK/src/hnswsearch.c" | head -1 | cut -d: -f1)
[ -n "$tl" ] || { echo "error: cannot find the per-hop counter in $FORK/src/hnswsearch.c" >&2; exit 1; }
SUM="hnswsearch.c:$(( tl - 1 )) unvisitedLength"

rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
  --exclude .git --exclude '/tools/pipnn_export/build/' --exclude '*.o' --exclude '*.so' --exclude '*.bc' \
  "$FORK/" "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/pgvectorfast_pipnn/"
( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    bench/run_experiment.sh bench/cloudlab/hn.env bench/cloudlab/pg_counts.sh \
    "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/" )
( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" bench/cloudlab/hn.env "$REMOTE_USER@$CLIENT.$DOMAIN:$DEST/" )

require_measurement_state

# /mydata takes the unlogged table copy (42 GB) and its index (~52 GB); the
# graph file (~7 GB) goes to the root disk (PIPNN_GRAPH in hn.env).
need_gb=105; [ "$SMOKE" = 1 ] && need_gb=20
# shellcheck disable=SC2016
step "$SERVER" "$T-df" '
  a=$(df -BG --output=avail /mydata | tail -1 | tr -dc 0-9)
  r=$(df -BG --output=avail /var/tmp | tail -1 | tr -dc 0-9)
  echo "free: /mydata ${a} GB (need '"$need_gb"'), /var/tmp ${r} GB (need 10)"
  [ "$a" -ge '"$need_gb"' ] && [ "$r" -ge 10 ]
  command -v cmake && command -v ninja && cmake --version | head -1'

# shellcheck disable=SC2016
step "$SERVER" "$T-so" '
  set -eo pipefail
  make -C pgvectorfast_pipnn clean PG_CONFIG=$PWD/postgres/install/bin/pg_config > /dev/null
  make -C pgvectorfast_pipnn -j"$(nproc)" PG_CONFIG=$PWD/postgres/install/bin/pg_config 2>&1 | grep -E "warning|error" || true
  so=pgvectorfast_pipnn/vectorfast.so
  st=$(strings $so)
  grep -q hnsw.build_graph_file <<< "$st" || { echo "error: no hnsw.build_graph_file in $so"; exit 1; }
  [ "$(nm $so | grep -c " calico_")" -gt 0 ] || { echo "error: no calico symbols in $so"; exit 1; }
  echo "built $so"'

# shellcheck disable=SC2016
step "$SERVER" "$T-tool" '
  set -eo pipefail
  b=pgvectorfast_pipnn/tools/pipnn_export/build
  cmake -S pgvectorfast_pipnn/tools/pipnn_export -B $b -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DPG_CONFIG=$PWD/postgres/install/bin/pg_config > $b.configure.log 2>&1 ||
    { tail -30 $b.configure.log; exit 1; }
  cmake --build $b -j"$(nproc)" --target pipnn_export 2>&1 | tail -3
  [ -x $b/pipnn_export ] && echo "built $b/pipnn_export"'

# The smoke index is rebuilt every run, to exercise the import itself.
rb=""; [ "$SMOKE" = 1 ] && rb="PIPNN_REBUILD=1"
step "$SERVER" "$T-build" "$rb $PIPENV bench/run_experiment.sh build_pipnn"

if [ "$SMOKE" = 1 ]; then
  # build_pipnn again only restarts the server (table and index are reused)
  # and, with KEEP_RUNNING, leaves it up for the queries and probes.
  step "$SERVER" "$T-query" \
    "KEEP_RUNNING=1 BUILD_SB=16GB $PIPENV bench/run_experiment.sh build_pipnn && bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw --table \"\$PIPNN_TABLE\" --m \"\$PIPNN_M\" --ef-construction 64 --sweep 10,40,160 --clients 1 --num-queries \"\$NUM_QUERIES\" --duration 5 --no-save"
  step "$SERVER" "$T-counts" \
    "PROFILE_DIR=$PD PG_COUNTS_NAME=smoke-pipnn PG_COUNTS_PROBES='$PROBES' PG_COUNTS_SUM='$SUM' PG_COUNTS_ARGS=\"--table \$PIPNN_TABLE --m \$PIPNN_M --ef-construction 64\" bench/cloudlab/pg_counts.sh 160"
  step "$SERVER" "$T-stop" "./run_postgres.sh stop || true"
  say "smoke done: check $T-build, $T-query and $T-counts above"
  exit 0
fi

# arm <name> <serve env> <ann_bench index args, evaluated on the node>
arm() {
  local name="$1" senv="$2" iargs="$3"
  step "$SERVER" "$T-serve-$name" \
    "$PIPENV $senv SERVE_ARM=fast LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
  for c in 1 40; do
    step "$CLIENT" "$T-$name-c$c" \
      "BENCH_LABEL=pip$name-fork bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw $iargs --sweep \"\$EF_SWEEP\" --clients $c --num-queries \"\$NUM_QUERIES\" --duration \"\$DURATION\" $REMOTE_DSN"
  done
  step "$SERVER" "$T-counts-$name" \
    "PROFILE_DIR=$PD PG_COUNTS_NAME=counts-$name PG_COUNTS_PROBES='$PROBES' PG_COUNTS_SUM='$SUM' PG_COUNTS_ARGS=\"$iargs\" bench/cloudlab/pg_counts.sh 20,80,160,640"
}
arm hnsw "" "--m 16 --ef-construction 64"
arm pipnn "SERVE_TABLE=\$PIPNN_TABLE" "--table \$PIPNN_TABLE --m \$PIPNN_M --ef-construction 64"

step "$SERVER" "$T-stop" "./run_postgres.sh stop || true"
ssh_to "$CLIENT" "cat $DEST/bench/results/$JSONL" |
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$SERVER.$DOMAIN" "cat > $DEST/bench/profiles/$PD/client-results.jsonl"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
mkdir -p "$HERE/../profiles/$PD"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/bench/profiles/$PD/" "$HERE/../profiles/$PD/"
say "done; results in bench/profiles/$PD/; logs: . bench/cloudlab/nodes-wisc.sh && bench/cloudlab/sync.sh pull"
