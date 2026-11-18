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
# Steps are detached and resumable (see lib.sh); bump RUN to redo a server start.
#
#   bench/cloudlab/profile_pgvector.sh
set -euo pipefail
# shellcheck source=bench/cloudlab/lib.sh
. "$(dirname "$0")/lib.sh"

RUN="${RUN:-1}"
CLIENTS_LIST="${CLIENTS_LIST:-1 8 16}"
EFS="${EFS:-40 640}"

require_measurement_state
ssh_to "$SERVER" '[ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 0 ]' ||
  { echo "error: perf_event_paranoid > 0 on $SERVER" >&2; exit 1; }

serve_step "prof-serve-stock-r$RUN" stock

for c in $CLIENTS_LIST; do
  for ef in $EFS; do
    name="pgv-c$c-ef$ef"
    if step_done "$CLIENT" "prof-client-$name" && step_done "$SERVER" "prof-perf-$name"; then
      say "$name: already done, skipping"; continue
    fi
    # shellcheck disable=SC2016
    start_step "$CLIENT" "prof-client-$name" \
      "BENCH_LABEL=prof-stock bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" $HNSW_ARGS --sweep $ef --clients $c --num-queries \"\$NUM_QUERIES\" --duration 80 $REMOTE_DSN"
    start_step "$SERVER" "prof-perf-$name" "bench/cloudlab/perf_window.sh $name 12 20"
    wait_step "$SERVER" "prof-perf-$name"
    wait_step "$CLIENT" "prof-client-$name"
  done
done

# shellcheck disable=SC2016
step "$SERVER" "prof-explain-r$RUN" \
  'bench/.venv/bin/python bench/cloudlab/explain_buffers.py --dataset "$DATASET" --ef 10,40,160,640,1000 --n 100'
step "$SERVER" "prof-stop-r$RUN" "./run_postgres.sh stop"
step "$SERVER" "prof-membw" \
  "gcc -O2 -march=native -pthread -o /mydata/membw bench/cloudlab/membw.c && /mydata/membw 8"

say "done; profiles in $SERVER:$DEST/bench/profiles/pgvector/"
