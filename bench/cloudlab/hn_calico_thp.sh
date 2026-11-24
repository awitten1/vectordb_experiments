#!/usr/bin/env bash
# Calico's translation arrays on 2 MB pages: does removing the ~3,500 4 KB page
# walks per query (half the remaining walks, hn_tlb_diag.sh) speed up
# pgvectorfast + tq4 on huge pages?
#
#   caffeinate -i bench/cloudlab/hn_calico_thp.sh
#
# One library for both arms: calico.huge_translation (PGC_POSTMASTER, default
# off = the old MADV_NOHUGEPAGE) picks MADV_HUGEPAGE on the /dev/shm extents,
# which only takes effect with /dev/shm mounted huge=advise.
# Steps are detached and resumable (see lib.sh); bump RUN to redo them.
#   1. rebuild pgvectorfast_ntq with the setting (old library kept as
#      /mydata/vectorfast_ntq_k2.so); check the optimized kernels are still in
#   2. reserve huge pages for shared_buffers
#   3. arm off: setting off, /dev/shm huge=never (as before); arm on: setting
#      on, /dev/shm remounted huge=advise. Each: serve fastntq on huge pages
#      (prewarmed), HP_CHECK, a check that the arm is what it claims (SHOW,
#      mount options, ShmemHugePages), LAN sweeps at 1 and 40 clients
#      (thpoff-/thpon-fastntq), then tlb_window.sh at ef_search 160, 1 and 40
#      clients (bench/profiles/hn-calico-thp/{off,on}/)
#   4. stop Postgres, release the pool, put /dev/shm back to huge=never
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
T="hnthp-r$RUN"
JSONL=hackernews_minilm_384.jsonl
PD=hn-calico-thp
EF=160
# shellcheck disable=SC2016  # NTQ_* expand on the node, from hn.env
TQ_ARGS='--table "$NTQ_TABLE" $NTQ_ARGS'

# Only the changed files, not a full sync.
( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    pgvectorfast_ntq/src/calico.c pgvectorfast_ntq/src/calico_pool.c pgvectorfast_ntq/src/calico_internal.h \
    bench/cloudlab/tlb_window.sh bench/cloudlab/tlb_attrib.py "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/" )

require_measurement_state

# shellcheck disable=SC2016
step "$SERVER" "$T-build" '
  set -o pipefail
  ./run_postgres.sh stop || true
  [ -f /mydata/vectorfast_ntq_k2.so ] || cp pgvectorfast_ntq/vectorfast.so /mydata/vectorfast_ntq_k2.so
  ./build_fastntq.sh
  pm=$(objdump -d --no-show-raw-insn pgvectorfast_ntq/vectorfast.so | grep -c vpermps || true)
  g=$(grep -c "calico.huge_translation" <(strings pgvectorfast_ntq/vectorfast.so) || true)
  echo "vpermps: $pm; calico.huge_translation strings: $g"
  [ "$pm" -gt 0 ] && [ "$g" -gt 0 ] || { echo "error: build lacks the optimized kernels or the new setting"; exit 1; }'

step "$SERVER" "$T-reserve" "$HP_RESERVE"

# arm <off|on>
arm() {
  local a="$1" opts="-c huge_pages=on" mnt=never
  [ "$a" = on ] && { opts="$opts -c calico.huge_translation=on"; mnt=advise; }
  step "$SERVER" "$T-$a-mount" \
    "./run_postgres.sh stop || true; sudo mount -o remount,huge=$mnt /dev/shm && findmnt -no OPTIONS /dev/shm"
  step "$SERVER" "$T-$a-serve" \
    "EXTRA_PG_OPTS='$opts' SERVE_ARM=fastntq LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
  # shellcheck disable=SC2016
  step "$SERVER" "$T-$a-numa" \
    '[ -n "${NUMA_CMD:-}" ] || exit 0; n=$(grep -c " interleave" /proc/$(head -1 postgres/data/postmaster.pid)/numa_maps); echo "postmaster mappings with interleave policy: $n"; [ "$n" -gt 0 ]'
  step "$SERVER" "$T-$a-check-hp" "HP_MIN_GB=60; $HP_CHECK"
  # shellcheck disable=SC2016
  step "$SERVER" "$T-$a-check-thp" '
    s=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW calico.huge_translation")
    m=$(findmnt -no OPTIONS /dev/shm); h=$(awk "/^ShmemHugePages:/ {print \$2}" /proc/meminfo)
    echo "calico.huge_translation: $s; /dev/shm: $m; ShmemHugePages: $h kB"
    ls /dev/shm | grep -c "^calico\." | sed "s/^/calico extents: /"
    if [ '"$a"' = on ]; then
      [ "$s" = on ] && [[ "$m" == *huge=advise* ]] && [ "$h" -gt 0 ] || { echo "error: arm on is not on 2 MB pages"; exit 1; }
    else
      [ "$s" = off ] || { echo "error: arm off has the setting on"; exit 1; }
    fi'
  local c
  for c in 1 40; do
    bench_step "$CLIENT" "$T-$a-c$c" "thp$a-fastntq" "$HNSW_ARGS $TQ_ARGS --clients $c"
  done
  step "$SERVER" "$T-$a-perf-check" \
    "[ \"\$(cat /proc/sys/kernel/perf_event_paranoid)\" -le 0 ] || sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0"
  local cp period name
  for cp in 1:2003 40:20011; do
    c=${cp%%:*} period=${cp#*:} name="hn-fastntq-c${cp%%:*}-ef$EF"
    if step_done "$CLIENT" "$T-$a-client-$name" && step_done "$SERVER" "$T-$a-tlb-$name"; then
      say "$a $name: already done, skipping"; continue
    fi
    start_step "$CLIENT" "$T-$a-client-$name" \
      "BENCH_LABEL=tlbthp$a-fastntq bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw --m 16 --ef-construction 64 $TQ_ARGS --sweep $EF --clients $c --num-queries \"\$NUM_QUERIES\" --duration 150 $REMOTE_DSN"
    start_step "$SERVER" "$T-$a-tlb-$name" "PERIOD=$period PROFILE_DIR=$PD/$a bench/cloudlab/tlb_window.sh $name 30 10"
    wait_step "$SERVER" "$T-$a-tlb-$name"
    wait_step "$CLIENT" "$T-$a-client-$name"
  done
  step "$SERVER" "$T-$a-check-end" "HP_MIN_GB=60; $HP_CHECK"
}

arm off
arm on

step "$SERVER" "$T-release" "$HP_RELEASE"
step "$SERVER" "$T-unmount" "sudo mount -o remount,huge=never /dev/shm && findmnt -no OPTIONS /dev/shm"

ssh_to "$CLIENT" "cat $DEST/bench/results/$JSONL" |
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$SERVER.$DOMAIN" "cat > $DEST/bench/profiles/$PD/client-results.jsonl"
step "$SERVER" "$T-attrib" "set -o pipefail
  for a in off on; do
    python3 bench/cloudlab/tlb_attrib.py --label tlbthp\$a bench/profiles/$PD/\$a bench/profiles/$PD/client-results.jsonl > bench/profiles/$PD/\$a/attrib.md
  done; head -40 bench/profiles/$PD/on/attrib.md"

for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
mkdir -p "$HERE/../profiles/$PD"
rsync -az --exclude '*.data' -e "ssh ${SSH_OPTS[*]}" \
  "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/bench/profiles/$PD/" "$HERE/../profiles/$PD/"
say "done; results: . bench/cloudlab/nodes-wisc.sh && bench/cloudlab/sync.sh pull"
