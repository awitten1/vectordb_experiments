#!/usr/bin/env bash
# The pipelined layer-0 hop (hnsw.pipeline_hop) against today's per-stage
# prefetch bursts, on pgvectorfast + tq4 with huge pages and 2 MB Calico
# translation arrays (the better arm of hn_calico_thp.sh):
#
#   caffeinate -i bench/cloudlab/hn_pipe_hop.sh
#
# hnsw.pipeline_hop is a per-session setting, so one server start serves both
# arms: each client run sets it with PGOPTIONS, and the arms alternate (off,
# on, on, off) on the same warm buffer pool, which cancels drift.
# Steps are detached and resumable (see lib.sh); bump RUN to redo them.
#   1. rebuild pgvectorfast_ntq with the setting (the previous library kept as
#      /mydata/vectorfast_ntq_thp.so); check the kernels and both settings
#   2. reserve huge pages, /dev/shm huge=advise, serve fastntq with
#      huge_pages=on and calico.huge_translation=on; check both; check that a
#      client session sees hnsw.pipeline_hop off and on as asked
#   3. ef_search 20, 80, 160, 640 at 1 and 40 LAN clients, off/on/on/off
#      (labels pipeoff1, pipeon1, pipeon2, pipeoff2 -fastntq)
#   4. trav_window.sh at ef_search 160, 1 and 40 clients, each arm
#      (bench/profiles/hn-pipe/{off,on}/)
#   5. stop Postgres, release the pool, /dev/shm back to huge=never
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
T="hnpipe-r$RUN"
JSONL=hackernews_minilm_384.jsonl
PD=hn-pipe
EF=160
SWEEP=20,80,160,640
# shellcheck disable=SC2016  # NTQ_* expand on the node, from hn.env
TQ_ARGS='--table "$NTQ_TABLE" $NTQ_ARGS'
LIB=postgres/install/lib/x86_64-linux-gnu/vectorfast.so
DSN="host=$SERVER_IP port=$PORT user=postgres dbname=postgres"

( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    pgvectorfast_ntq/src/hnsw.c pgvectorfast_ntq/src/hnsw.h pgvectorfast_ntq/src/hnswutils.c \
    bench/cloudlab/trav_window.sh "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/" )

require_measurement_state

# shellcheck disable=SC2016
step "$SERVER" "$T-build" '
  set -o pipefail
  ./run_postgres.sh stop || true
  [ -f /mydata/vectorfast_ntq_thp.so ] || cp pgvectorfast_ntq/vectorfast.so /mydata/vectorfast_ntq_thp.so
  ./build_fastntq.sh
  so=pgvectorfast_ntq/vectorfast.so  # what the fastntq arm installs at serve time
  pm=$(objdump -d --no-show-raw-insn $so | grep -c vpermps || true)
  echo "built library: vpermps $pm"
  # Not `strings | grep -q`: grep -q quits at the first match, strings dies of
  # SIGPIPE, and pipefail fails the check.
  st=$(strings $so)
  [ "$pm" -gt 0 ] && grep -q hnsw.pipeline_hop <<< "$st" && grep -q calico.huge_translation <<< "$st" ||
    { echo "error: build lacks the optimized kernels or a setting"; exit 1; }'

step "$SERVER" "$T-reserve" "$HP_RESERVE"
step "$SERVER" "$T-mount" \
  "./run_postgres.sh stop || true; sudo mount -o remount,huge=advise /dev/shm && findmnt -no OPTIONS /dev/shm"
step "$SERVER" "$T-serve" \
  "EXTRA_PG_OPTS='-c huge_pages=on -c calico.huge_translation=on' SERVE_ARM=fastntq LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
# shellcheck disable=SC2016
step "$SERVER" "$T-numa" \
  '[ -n "${NUMA_CMD:-}" ] || exit 0; n=$(grep -c " interleave" /proc/$(head -1 postgres/data/postmaster.pid)/numa_maps); echo "postmaster mappings with interleave policy: $n"; [ "$n" -gt 0 ]'
step "$SERVER" "$T-check-hp" "HP_MIN_GB=60; $HP_CHECK"
step "$SERVER" "$T-check-lib" "cmp pgvectorfast_ntq/vectorfast.so $LIB && echo served library is the new build"
# shellcheck disable=SC2016
step "$SERVER" "$T-check-thp" '
  s=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW calico.huge_translation")
  m=$(findmnt -no OPTIONS /dev/shm); h=$(awk "/^ShmemHugePages:/ {print \$2}" /proc/meminfo)
  echo "calico.huge_translation: $s; /dev/shm: $m; ShmemHugePages: $h kB"
  [ "$s" = on ] && [[ "$m" == *huge=advise* ]] && [ "$h" -gt 0 ] || { echo "error: translation arrays are not on 2 MB pages"; exit 1; }'
step "$CLIENT" "$T-check-setting" "
  for a in off on; do
    got=\$(PGOPTIONS=\"-c hnsw.pipeline_hop=\$a\" bench/.venv/bin/python -c \"import psycopg; print(psycopg.connect('$DSN').execute('SHOW hnsw.pipeline_hop').fetchone()[0])\")
    echo \"asked \$a, session has \$got\"; [ \"\$got\" = \"\$a\" ] || exit 1
  done"

# sweep <round label> <off|on> <clients>
sweep() {
  step "$CLIENT" "$T-$1-c$3" \
    "PGOPTIONS='-c hnsw.pipeline_hop=$2' BENCH_LABEL=$1-fastntq bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw --m 16 --ef-construction 64 $TQ_ARGS --sweep $SWEEP --clients $3 --num-queries \"\$NUM_QUERIES\" --duration \"\$DURATION\" $REMOTE_DSN"
}
for c in 1 40; do
  sweep pipeoff1 off $c
  sweep pipeon1 on $c
  sweep pipeon2 on $c
  sweep pipeoff2 off $c
done

step "$SERVER" "$T-perf-check" \
  "[ \"\$(cat /proc/sys/kernel/perf_event_paranoid)\" -le 0 ] || sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0"
for a in off on; do
  for cp in 1:500009:2003 40:20000003:20011; do
    IFS=: read -r c pcyc plat <<< "$cp"
    name="hn-fastntq-c$c-ef$EF"
    if step_done "$CLIENT" "$T-$a-client-$name" && step_done "$SERVER" "$T-$a-trav-$name"; then
      say "$a $name: already done, skipping"; continue
    fi
    start_step "$CLIENT" "$T-$a-client-$name" \
      "PGOPTIONS='-c hnsw.pipeline_hop=$a' BENCH_LABEL=travpipe$a-fastntq bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw --m 16 --ef-construction 64 $TQ_ARGS --sweep $EF --clients $c --num-queries \"\$NUM_QUERIES\" --duration 180 $REMOTE_DSN"
    start_step "$SERVER" "$T-$a-trav-$name" "PERIOD_CYC=$pcyc PERIOD_LAT=$plat PROFILE_DIR=$PD/$a bench/cloudlab/trav_window.sh $name 30 10"
    wait_step "$SERVER" "$T-$a-trav-$name"
    wait_step "$CLIENT" "$T-$a-client-$name"
  done
done

step "$SERVER" "$T-check-end" "HP_MIN_GB=60; $HP_CHECK"
step "$SERVER" "$T-check-watchdog" "echo nmi_watchdog: \$(cat /proc/sys/kernel/nmi_watchdog); ! sudo -n perf probe -l 2>/dev/null | grep -q probe_vectorfast"
step "$SERVER" "$T-release" "$HP_RELEASE"
step "$SERVER" "$T-unmount" "sudo mount -o remount,huge=never /dev/shm && findmnt -no OPTIONS /dev/shm; cp $LIB bench/profiles/$PD/vectorfast.so"

ssh_to "$CLIENT" "cat $DEST/bench/results/$JSONL" |
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$SERVER.$DOMAIN" "cat > $DEST/bench/profiles/$PD/client-results.jsonl"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
mkdir -p "$HERE/../profiles/$PD"
rsync -az --exclude '*.data' -e "ssh ${SSH_OPTS[*]}" \
  "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/bench/profiles/$PD/" "$HERE/../profiles/$PD/"
say "done; profiles in bench/profiles/$PD/; results: . bench/cloudlab/nodes-wisc.sh && bench/cloudlab/sync.sh pull"
