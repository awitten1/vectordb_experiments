#!/usr/bin/env bash
# Where stock pgvector HNSW spends its time on DBpedia-OpenAI 1M.
#
# Run from the workstation after a FRAME_POINTERS=1 build on node-0 (and
# build_extensions.sh, so postgres/stock/vector.so is that build) and with
# kernel.perf_event_paranoid <= 0 there. For 1, 8 and 16 clients at
# ef_search 40 (recall 0.91) and 640 (0.99), the client runs on node-1 for 80 s
# while bench/cloudlab/perf_window.sh profiles node-0, which then runs nothing
# but Postgres. Then per-query buffer accesses from EXPLAIN, and the machine's
# memory ceilings (membw.c) with the server stopped.
#
# Another dataset: ENV_FILE=bench/cloudlab/sift.env TAG=sift PROFILE_DIR=sift EFS="40 160" \
#   FAST_EF=160 bench/cloudlab/profile_pgvector.sh
#
# Steps are detached and resumable (see lib.sh); bump RUN to redo a server start.
#
#   bench/cloudlab/profile_pgvector.sh
set -euo pipefail
# shellcheck source=bench/cloudlab/lib.sh
. "$(dirname "$0")/lib.sh"

RUN="${RUN:-1}"
CLIENTS_LIST="${CLIENTS_LIST:-1 8 16}"
EFS="${EFS:-40 640}"
EXPLAIN_EFS="${EXPLAIN_EFS:-10,40,160,640,1000}"
# Per-dataset naming: profile files are <TAG>-c<clients>-ef<ef>.* in
# bench/profiles/$PROFILE_DIR, and every step name carries the tag.
TAG="${TAG:-pgv}"
export PROFILE_DIR="${PROFILE_DIR:-pgvector}"
# FAST_EF: also profile pgvectorfast at 8 and 16 clients at this ef_search.
FAST_EF="${FAST_EF:-}"
# PROF_INDEX: hnsw (float vectors) or hnsw_halfvec (the halfvec expression
# index); either way the table's other indexes stay. Client rows are labelled
# <PROF_LABEL>-stock / <PROF_LABEL>-fast.
PROF_INDEX="${PROF_INDEX:-hnsw}"
PROF_LABEL="${PROF_LABEL:-prof}"
PROF_ARGS="--index $PROF_INDEX --m 16 --ef-construction 64 --keep-other-indexes"

require_measurement_state
ssh_to "$SERVER" '[ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 0 ]' ||
  { echo "error: perf_event_paranoid > 0 on $SERVER" >&2; exit 1; }

# profile_pair <arm> <label> <clients> <ef> <name>: client on node-1 for 100 s,
# perf_window.sh on node-0 during it (12 s warm-up, then four 20 s windows).
profile_pair() {
  local arm="$1" label="$2" c="$3" ef="$4" name="$5"
  if step_done "$CLIENT" "prof-client-$name" && step_done "$SERVER" "prof-perf-$name"; then
    say "$name: already done, skipping"; return
  fi
  # shellcheck disable=SC2016
  start_step "$CLIENT" "prof-client-$name" \
    "BENCH_LABEL=$label bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" $PROF_ARGS --sweep $ef --clients $c --num-queries \"\$NUM_QUERIES\" --duration 100 $REMOTE_DSN"
  start_step "$SERVER" "prof-perf-$name" "PROFILE_DIR=$PROFILE_DIR bench/cloudlab/perf_window.sh $name 12 20"
  wait_step "$SERVER" "prof-perf-$name"
  wait_step "$CLIENT" "prof-client-$name"
}

serve_step "prof-$TAG-serve-stock-r$RUN" stock
for c in $CLIENTS_LIST; do
  for ef in $EFS; do
    profile_pair stock "$PROF_LABEL-stock" "$c" "$ef" "$TAG-c$c-ef$ef"
  done
done
step "$SERVER" "prof-$TAG-explain-r$RUN" \
  "bench/.venv/bin/python bench/cloudlab/explain_buffers.py --dataset \"\$DATASET\" --index $PROF_INDEX --ef $EXPLAIN_EFS --n 100"

if [ -n "$FAST_EF" ]; then
  serve_step "prof-$TAG-serve-fast-r$RUN" fast
  for c in 8 16; do
    profile_pair fast "$PROF_LABEL-fast" "$c" "$FAST_EF" "$TAG-fast-c$c-ef$FAST_EF"
  done
fi

step "$SERVER" "prof-$TAG-stop-r$RUN" "./run_postgres.sh stop"
step "$SERVER" "prof-membw" \
  "gcc -O2 -march=native -pthread -o /mydata/membw bench/cloudlab/membw.c && /mydata/membw 8"

say "done; profiles in $SERVER:$DEST/bench/profiles/$PROFILE_DIR/"
