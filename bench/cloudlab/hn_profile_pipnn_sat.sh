#!/usr/bin/env bash
# Where stock pgvector and pgvectorfast spend the server's time on the float
# PiPNN graph (imported into PIPNN_TABLE by hn_pipnn.sh), at 1 client and at
# the saturating count, at the ef_search that gives recall@10 ~0.95 there (60:
# 0.921 at 40, 0.966 at 80). The companion of hn_profile_sat.sh (HNSW m = 32),
# same method and scripts: the client holds one ef_search for 130 s while
# perf_window.sh profiles the server (30 s warm-up, then four 20 s windows:
# counters + DRAM traffic, Top-Down, call graph, TLB walks); perf_summary.py
# and perf_buckets.py give CPU time per query by bucket.
#
# Needs the float PiPNN index in place: hn_saturated_m32.sh runs it after its
# restore and recall gate. Leaves Postgres running (stock); the caller serves
# again. Labels prof<N>p-stock, prof<N>p-fast. Steps are resumable; bump RUN
# to redo.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
CLIENTS="${CLIENTS:-60}"
EF="${EF:-60}"
T="hnprofpipnn-r$RUN"
L="prof${CLIENTS}p"
PD=hn-sat-prof-pipnn
OUT="bench/profiles/$PD"
CG_FILE="$STEPS/$T-callgraph"
IARGS="--table \$PIPNN_TABLE --m \$PIPNN_M --ef-construction 64"
RESTORE_STEP="${RESTORE_STEP:-hnsat${CLIENTS}n-r1-restore}"

for n in "$SERVER" "$CLIENT"; do
  ( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
      bench/run_experiment.sh bench/cloudlab/hn.env bench/cloudlab/perf_window.sh \
      bench/cloudlab/perf_summary.py bench/cloudlab/perf_buckets.py bench/cloudlab/explain_buffers.py \
      "$REMOTE_USER@$n.$DOMAIN:$DEST/" )
done

require_measurement_state

# shellcheck disable=SC2016
step "$SERVER" "$T-perf-enable" \
  'sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0 && perf --version &&
   if perf record -a --call-graph lbr -o /mydata/lbr-test.data -- sleep 1 >/dev/null 2>&1; then
     echo lbr > '"$CG_FILE"'; else echo none > '"$CG_FILE"'; fi; rm -f /mydata/lbr-test.data;
   echo "call graphs: $(cat '"$CG_FILE"')"; mkdir -p '"$OUT"

# The index on PIPNN_TABLE must be the imported PiPNN graph, not HNSW m = 32
# (same name): the import logs "imported graph" in the restore step
step "$SERVER" "$T-check-index" \
  "grep 'imported graph' $STEPS/$RESTORE_STEP.log || { echo 'error: no PiPNN import in $RESTORE_STEP'; exit 1; }"

# profile_pair <arm> <clients>: the client for 130 s, perf_window.sh during it
profile_pair() {
  local arm="$1" c="$2" name="hn-$1-c$2-ef$EF"
  if step_done "$CLIENT" "$T-client-$name" && step_done "$SERVER" "$T-perf-$name"; then
    say "$name: already done, skipping"; return
  fi
  start_step "$CLIENT" "$T-client-$name" \
    "BENCH_LABEL=$L-$arm bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw $IARGS --sweep $EF --clients $c --num-queries \"\$NUM_QUERIES\" --duration 130 $REMOTE_DSN"
  start_step "$SERVER" "$T-perf-$name" \
    "CALLGRAPH=\$(cat $CG_FILE) PROFILE_DIR=$PD bench/cloudlab/perf_window.sh $name 30 20"
  wait_step "$SERVER" "$T-perf-$name"
  wait_step "$CLIENT" "$T-client-$name"
}

for arm in fast stock; do
  env=""
  [ "$arm" = fast ] && env='FAST_SO=$PIPNN_SO'
  step "$SERVER" "$T-serve-$arm" \
    "$env SERVE_ARM=$arm SERVE_TABLE=\$PIPNN_TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
  # shellcheck disable=SC2016
  step "$SERVER" "$T-numa-$arm" \
    '[ -n "${NUMA_CMD:-}" ] || exit 0; n=$(grep -c " interleave" /proc/$(head -1 postgres/data/postmaster.pid)/numa_maps); echo "postmaster mappings with interleave policy: $n"; [ "$n" -gt 0 ]'
  for c in 1 "$CLIENTS"; do
    profile_pair "$arm" "$c"
  done
  # Per-query buffer accesses (stock: pgvectorfast's fast path reads pages
  # without going through the counted buffer calls)
  if [ "$arm" = stock ]; then
    step "$SERVER" "$T-explain-$arm" \
      "set -o pipefail; bench/.venv/bin/python bench/cloudlab/explain_buffers.py --dataset \"\$DATASET\" --table \"\$PIPNN_TABLE\" --ef 20,$EF,320 --n 100 | tee $OUT/explain-$arm.txt"
  fi
done

# The client's result rows carry the QPS the summaries divide by
ssh_to "$CLIENT" "grep -F '\"$L-' $DEST/bench/results/hackernews_minilm_384.jsonl" |
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$SERVER.$DOMAIN" "cat > $DEST/$OUT/client-results.jsonl"
step "$SERVER" "$T-summary" \
  "set -o pipefail; python3 bench/cloudlab/perf_summary.py --label $L $OUT $OUT/client-results.jsonl | tee $OUT/summary.md &&
   python3 bench/cloudlab/perf_buckets.py --label $L --top 30 $OUT $OUT/client-results.jsonl | tee $OUT/buckets.md"

for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
mkdir -p "$HERE/../profiles/$PD"
rsync -az --exclude '*.perf.data' -e "ssh ${SSH_OPTS[*]}" \
  "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/$OUT/" "$HERE/../profiles/$PD/"
say "done; profiles in bench/profiles/$PD (perf.data files stay on $SERVER)"
