#!/usr/bin/env bash
# pgvectorfast + tq4 with the optimized TurboQuant kernels
# (pgvectorfast_ntq/src/hnswtq_simd.h: AVX2 register-lookup TqNavDot, a
# k-specialized TqRefineDot), on the Hacker News tq4 index:
#
#   caffeinate -i bench/cloudlab/hn_fastntq_k2.sh
#
# Steps are detached and resumable (see lib.sh); bump RUN to redo them.
#   1. microbenchmark the kernels on the server's CPU (tq_kernel_opt_bench.c):
#      correct against a double reference on every size, and faster
#   2. keep the first build (original kernels) as /mydata/vectorfast_ntq_k1.so,
#      rebuild, and check the new library's kernels use no gathers
#   3. 4 KB pages, full protocol as hn_fastntq.sh: 1-client rounds
#      (pvfk2-fastntq-r1, -r2), recall check against pgvector_ntq, 10/20/40
#      LAN clients (sck2-fastntq), perf profiles at ef_search 160, 1 and 40
#      clients (profk2-fastntq, bench/profiles/hn-fastntq-k2/)
#   4. huge pages (hp_lib.sh): 1-client rounds (hpk2-fastntq-r*), recall check,
#      40 LAN clients (hpsck2-fastntq), profiles (profhpk2-fastntq,
#      (the tq4 table + index are 66 GB, so the check wants >= 60 GB in use)
#      bench/profiles/hn-fastntq-k2-hp/), huge-page checks after each server
#      start and at the end, then the pool is released
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
T="hnk2-r$RUN"
JSONL=hackernews_minilm_384.jsonl
EF=160
CFLAGS_PG="-O2 -march=native -ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math -ffp-contract=fast"
# shellcheck disable=SC2016  # NTQ_* expand on the node, from hn.env
TQ_ARGS='--table "$NTQ_TABLE" $NTQ_ARGS'

require_measurement_state

step "$SERVER" "$T-kernel" \
  "set -o pipefail; mkdir -p bench/profiles/hn-fastntq-k2 && gcc $CFLAGS_PG -Ipgvectorfast_ntq/src -o /mydata/tq_kernel_opt_bench bench/cloudlab/tq_kernel_opt_bench.c -lm && /mydata/tq_kernel_opt_bench | tee bench/profiles/hn-fastntq-k2/kernel.txt"

# shellcheck disable=SC2016
step "$SERVER" "$T-build" '
  set -o pipefail
  [ -f /mydata/vectorfast_ntq_k1.so ] || cp pgvectorfast_ntq/vectorfast.so /mydata/vectorfast_ntq_k1.so
  ./build_fastntq.sh
  dis=$(objdump -d --no-show-raw-insn pgvectorfast_ntq/vectorfast.so |
        awk "/^[0-9a-f]+ <.*>:\$/ {p = (\$2 ~ /TqNavDot|TqRefineDot|HnswTqCodeDistance|HnswTqRerank/)} p")
  g=$(grep -c gather <<<"$dis" || true); pm=$(grep -c vpermps <<<"$dis" || true)
  echo "tq kernel code: $(grep -cE "^ +[0-9a-f]+:" <<<"$dis") instructions, $pm vpermps, $g gathers"
  echo "$dis" > bench/profiles/hn-fastntq-k2/disasm.txt
  [ "$g" -eq 0 ] && [ "$pm" -gt 0 ] || { echo "error: expected vpermps and no gathers in the tq kernels"; exit 1; }'

# profile_pair <label-prefix> <profile-dir> <clients>
profile_pair() {
  local lp="$1" pd="$2" c="$3" name="hn-fastntq-c$3-ef$EF"
  local tag="$T-$lp-c$c"
  if step_done "$CLIENT" "$tag-client" && step_done "$SERVER" "$tag-perf"; then
    say "$tag: already done, skipping"; return
  fi
  start_step "$CLIENT" "$tag-client" \
    "BENCH_LABEL=$lp-fastntq bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw --m 16 --ef-construction 64 $TQ_ARGS --sweep $EF --clients $c --num-queries \"\$NUM_QUERIES\" --duration 130 $REMOTE_DSN"
  start_step "$SERVER" "$tag-perf" "mkdir -p bench/profiles/$pd && CALLGRAPH=lbr PROFILE_DIR=$pd bench/cloudlab/perf_window.sh $name 30 20"
  wait_step "$SERVER" "$tag-perf"
  wait_step "$CLIENT" "$tag-client"
}

# measure <phase> <local-prefix> <lan-prefix> <prof-prefix> <profile-dir> <counts> [extra env]
measure() {
  local ph="$1" lp="$2" sp="$3" pp="$4" pd="$5" counts="$6" env="${7:-}"
  step "$SERVER" "$T-$ph-local" "$env PVF_LABEL=$lp PVF_ARMS='fastntq fastntq' KEEP_RUNNING=1 bench/run_experiment.sh pgvectorfast"
  [ -n "$env" ] && step "$SERVER" "$T-$ph-check-local" "HP_MIN_GB=60; $HP_CHECK"
  step "$SERVER" "$T-$ph-recall" "python3 bench/cloudlab/compare_recall.py bench/results/$JSONL pvf-ntq-r $lp-fastntq-r"
  step "$SERVER" "$T-$ph-serve" \
    "$env SERVE_ARM=fastntq LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
  # shellcheck disable=SC2016
  step "$SERVER" "$T-$ph-numa" \
    '[ -n "${NUMA_CMD:-}" ] || exit 0; n=$(grep -c " interleave" /proc/$(head -1 postgres/data/postmaster.pid)/numa_maps); echo "postmaster mappings with interleave policy: $n"; [ "$n" -gt 0 ]'
  [ -n "$env" ] && step "$SERVER" "$T-$ph-check-serve" "HP_MIN_GB=60; $HP_CHECK"
  local c
  for c in $counts; do
    bench_step "$CLIENT" "$T-$ph-c$c" "$sp-fastntq" "$HNSW_ARGS $TQ_ARGS --clients $c"
  done
  step "$SERVER" "$T-$ph-perf-check" \
    "[ \"\$(cat /proc/sys/kernel/perf_event_paranoid)\" -le 0 ] || sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0"
  for c in 1 40; do profile_pair "$pp" "$pd" "$c"; done
  [ -n "$env" ] && step "$SERVER" "$T-$ph-check-end" "HP_MIN_GB=60; $HP_CHECK"
  true
}

measure 4k pvfk2 sck2 profk2 hn-fastntq-k2 "10 20 40"
step "$SERVER" "$T-4k-stop" "./run_postgres.sh stop"

step "$SERVER" "$T-hp-reserve" "$HP_RESERVE"
measure hp hpk2 hpsck2 profhpk2 hn-fastntq-k2-hp "40" "EXTRA_PG_OPTS='-c huge_pages=on'"
step "$SERVER" "$T-hp-release" "$HP_RELEASE"

# Summaries: the client's rows carry the QPS the profiles divide by.
ssh_to "$CLIENT" "cat $DEST/bench/results/$JSONL" |
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$SERVER.$DOMAIN" "cat > $DEST/bench/profiles/hn-fastntq-k2/client-results.jsonl"
step "$SERVER" "$T-summary" "set -o pipefail
  for pd in hn-fastntq-k2:profk2 hn-fastntq-k2-hp:profhpk2; do
    d=bench/profiles/\${pd%%:*}; l=\${pd#*:}
    python3 bench/cloudlab/perf_summary.py --label \$l \$d bench/profiles/hn-fastntq-k2/client-results.jsonl | tee \$d/summary.md
    python3 bench/cloudlab/perf_buckets.py --label \$l --top 30 \$d bench/profiles/hn-fastntq-k2/client-results.jsonl | tee \$d/buckets.md
  done"

for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
for pd in hn-fastntq-k2 hn-fastntq-k2-hp; do
  mkdir -p "$HERE/../profiles/$pd"
  rsync -az --exclude '*.perf.data' -e "ssh ${SSH_OPTS[*]}" \
    "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/bench/profiles/$pd/" "$HERE/../profiles/$pd/"
done
say "done; results: . bench/cloudlab/nodes-wisc.sh && bench/cloudlab/sync.sh pull"
