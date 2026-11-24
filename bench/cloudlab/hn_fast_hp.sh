#!/usr/bin/env bash
# Float pgvectorfast on Hacker News 27.4M with shared_buffers in 2 MB huge
# pages (huge_pages=on): the like-for-like baseline for pgvectorfast + tq4 on
# huge pages (hn_fastntq_k2.sh), on the same float index and protocol as
# hn_qps.sh.
#
#   caffeinate -i bench/cloudlab/hn_fast_hp.sh
#
# Steps are detached and resumable (see lib.sh); bump RUN to redo them.
#   1. reserve the huge pages (hp_lib.sh HP_RESERVE)
#   2. 1-client rounds, client local: fast twice (hp-fast-r1, -r2), then the
#      recall check against the 4 KB rounds (pvf-fast-r*)
#   3. serve fast for LAN clients; 40 clients (hpsc-fast)
#   4. perf profiles at ef_search 160, 1 and 40 clients (profhp-fast,
#      bench/profiles/hn-fast-hp/)
#   HP_CHECK after the 1-client rounds, after the LAN serve and at the end
#   (table 42 GB + index 52 GB prewarmed, so >= 90 GB of the pool in use)
#   5. stop Postgres and release the pool
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"
# shellcheck source=bench/cloudlab/hp_lib.sh
. "$HERE/hp_lib.sh"

RUN="${RUN:-1}"
T="hnfhp-r$RUN"
HP="EXTRA_PG_OPTS='-c huge_pages=on'"
JSONL=hackernews_minilm_384.jsonl
PD=hn-fast-hp
OUT="bench/profiles/$PD"
EF=160

require_measurement_state

step "$SERVER" "$T-reserve" "$HP_RESERVE"

step "$SERVER" "$T-local" "$HP KEEP_RUNNING=1 PVF_LABEL=hp PVF_ARMS='fast fast' bench/run_experiment.sh pgvectorfast"
step "$SERVER" "$T-check-local" "$HP_CHECK"
step "$SERVER" "$T-recall" "python3 bench/cloudlab/compare_recall.py bench/results/$JSONL pvf-fast-r hp-fast-r"

step "$SERVER" "$T-serve" \
  "$HP SERVE_ARM=fast LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
# shellcheck disable=SC2016
step "$SERVER" "$T-numa" \
  '[ -n "${NUMA_CMD:-}" ] || exit 0; n=$(grep -c " interleave" /proc/$(head -1 postgres/data/postmaster.pid)/numa_maps); echo "postmaster mappings with interleave policy: $n"; [ "$n" -gt 0 ]'
step "$SERVER" "$T-check-serve" "$HP_CHECK"

bench_step "$CLIENT" "$T-c40" hpsc-fast "$HNSW_ARGS --clients 40"

step "$SERVER" "$T-perf-check" \
  "[ \"\$(cat /proc/sys/kernel/perf_event_paranoid)\" -le 0 ] || sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0; mkdir -p $OUT"
for c in 1 40; do
  name="hn-fast-c$c-ef$EF"
  if step_done "$CLIENT" "$T-client-$name" && step_done "$SERVER" "$T-perf-$name"; then
    say "$name: already done, skipping"; continue
  fi
  start_step "$CLIENT" "$T-client-$name" \
    "BENCH_LABEL=profhp-fast bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw --m 16 --ef-construction 64 --sweep $EF --clients $c --num-queries \"\$NUM_QUERIES\" --duration 130 $REMOTE_DSN"
  start_step "$SERVER" "$T-perf-$name" "CALLGRAPH=lbr PROFILE_DIR=$PD bench/cloudlab/perf_window.sh $name 30 20"
  wait_step "$SERVER" "$T-perf-$name"
  wait_step "$CLIENT" "$T-client-$name"
done

step "$SERVER" "$T-check-end" "$HP_CHECK"
step "$SERVER" "$T-release" "$HP_RELEASE"

ssh_to "$CLIENT" "cat $DEST/bench/results/$JSONL" |
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$SERVER.$DOMAIN" "cat > $DEST/$OUT/client-results.jsonl"
step "$SERVER" "$T-summary" \
  "set -o pipefail; python3 bench/cloudlab/perf_summary.py --label profhp $OUT $OUT/client-results.jsonl | tee $OUT/summary.md &&
   python3 bench/cloudlab/perf_buckets.py --label profhp --top 30 $OUT $OUT/client-results.jsonl | tee $OUT/buckets.md"

for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
mkdir -p "$HERE/../profiles/$PD"
rsync -az --exclude '*.perf.data' -e "ssh ${SSH_OPTS[*]}" \
  "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/$OUT/" "$HERE/../profiles/$PD/"
say "done; profiles in bench/profiles/$PD; results: . bench/cloudlab/nodes-wisc.sh && bench/cloudlab/sync.sh pull"
