#!/usr/bin/env bash
# pgvectorfast + pgvector_ntq's TurboQuant read path (build_fastntq.sh, the
# pgvectorfast_ntq submodule with branch tq4-prototype checked out) on the Hacker News tq4 index
# that hn_tq4.sh built with pgvector_ntq, same protocol as the other arms:
#
#   SMOKE=1 caffeinate -i bench/cloudlab/hn_fastntq.sh   # 1M tq4 table, 5 s windows
#   caffeinate -i bench/cloudlab/hn_fastntq.sh           # 27.4M
#
# Steps are detached and resumable (see lib.sh); bump RUN to redo them.
#   1. build pgvectorfast_ntq/vectorfast.so
#   2. 1-client rounds, client local: fastntq twice (pvf-fastntq-r1, -r2)
#   3. check recall at every ef_search equals pgvector_ntq's (pvf-ntq-r*):
#      same graph and distance code, so any difference is a bug
#   4. serve fastntq for LAN clients (NUMA interleave checked), then 10, 20
#      and 40 clients (sc-fastntq); the smoke run does 40 only
#   5. full run: perf profiles at ef_search 160, 1 and 40 clients
#      (prof-fastntq, bench/profiles/hn-fastntq/), as in hn_profile_tq4.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
BUILD_RUN="${BUILD_RUN:-1}"
if [ "${SMOKE:-0}" = 1 ]; then
  T="smoke-hnfq-r$RUN"; JSONL=hackernews_minilm_384_1000000.jsonl; COUNTS="40"
else
  T="hnfq-r$RUN"; JSONL=hackernews_minilm_384.jsonl; COUNTS="10 20 40"
fi
PD=hn-fastntq
OUT="bench/profiles/$PD"
EF=160

require_measurement_state

step "$SERVER" "hnfq-build-r$BUILD_RUN" "./build_fastntq.sh"
step "$SERVER" "$T-local" "PVF_ARMS='fastntq fastntq' bench/run_experiment.sh pgvectorfast"
step "$SERVER" "$T-recall" "python3 bench/cloudlab/compare_recall.py bench/results/$JSONL pvf-ntq-r pvf-fastntq-r"

serve_step "$T-serve" fastntq
# shellcheck disable=SC2016
step "$SERVER" "$T-numa" \
  '[ -n "${NUMA_CMD:-}" ] || exit 0; n=$(grep -c " interleave" /proc/$(head -1 postgres/data/postmaster.pid)/numa_maps); echo "postmaster mappings with interleave policy: $n"; [ "$n" -gt 0 ]'
for c in $COUNTS; do
  # shellcheck disable=SC2016  # NTQ_* expand on the node, from hn.env
  bench_step "$CLIENT" "$T-c$c" sc-fastntq "$HNSW_ARGS"' --table "$NTQ_TABLE" $NTQ_ARGS'" --clients $c"
done

if [ "${SMOKE:-0}" != 1 ]; then
  step "$SERVER" "$T-perf-check" \
    "[ \"\$(cat /proc/sys/kernel/perf_event_paranoid)\" -le 0 ] || sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0; mkdir -p $OUT"
  for c in 1 40; do
    name="hn-fastntq-c$c-ef$EF"
    if step_done "$CLIENT" "$T-client-$name" && step_done "$SERVER" "$T-perf-$name"; then
      say "$name: already done, skipping"; continue
    fi
    # shellcheck disable=SC2016
    start_step "$CLIENT" "$T-client-$name" \
      "BENCH_LABEL=prof-fastntq bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw --m 16 --ef-construction 64 --table \"\$NTQ_TABLE\" \$NTQ_ARGS --sweep $EF --clients $c --num-queries \"\$NUM_QUERIES\" --duration 130 $REMOTE_DSN"
    start_step "$SERVER" "$T-perf-$name" "CALLGRAPH=lbr PROFILE_DIR=$PD bench/cloudlab/perf_window.sh $name 30 20"
    wait_step "$SERVER" "$T-perf-$name"
    wait_step "$CLIENT" "$T-client-$name"
  done
fi

step "$SERVER" "$T-stop" "./run_postgres.sh stop"

if [ "${SMOKE:-0}" != 1 ]; then
  ssh_to "$CLIENT" "cat $DEST/bench/results/$JSONL" |
    ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$SERVER.$DOMAIN" "cat > $DEST/$OUT/client-results.jsonl"
  step "$SERVER" "$T-summary" \
    "set -o pipefail; python3 bench/cloudlab/perf_summary.py $OUT $OUT/client-results.jsonl | tee $OUT/summary.md &&
     python3 bench/cloudlab/perf_buckets.py --top 30 $OUT $OUT/client-results.jsonl | tee $OUT/buckets.md"
  mkdir -p "$HERE/../profiles/$PD"
  rsync -az --exclude '*.perf.data' -e "ssh ${SSH_OPTS[*]}" \
    "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/$OUT/" "$HERE/../profiles/$PD/"
fi
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $STEPS/hnfq-build-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
say "done; results: . bench/cloudlab/nodes-wisc.sh && bench/cloudlab/sync.sh pull"
