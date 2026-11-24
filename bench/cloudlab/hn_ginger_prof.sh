#!/usr/bin/env bash
# Why Ginger is faster than pgvectorfast + tq4 on HN 27.4M: the work each does
# per query, and what that work costs.
#
#   caffeinate -i bench/cloudlab/hn_ginger_prof.sh
#
# Steps are detached and resumable (see lib.sh); bump RUN to redo them.
# Client node (Ginger's node), started first and run alongside the server steps:
#   1. rebuild ginger_bench with line tables (-g1: same code, so profiles can be
#      read by source line) and with --stats / GINGER_PROGRESS; the previous
#      binary is kept as /mydata/ginger_bench.orig
# Server node:
#   2. serve pgvectorfast + tq4 as measured last (huge pages, 2 MB Calico
#      translation arrays); check both
#   3. pg_counts.sh: hops, tq4 distances and re-ranked candidates per query at
#      each ef_search (uprobes; a local one-client run)
#   4. stop Postgres, release the pool, /dev/shm back to huge=never
# Client node again:
#   5. Ginger's search statistics per query at each beam_size (one pass, 1 thread)
#   6. ginger_window.sh profiles at beam_size 64 (recall ~0.95) and 160 (the
#      ef_search of the pgvectorfast profiles), 1 and 40 threads
# Everything lands in bench/profiles/hn-ginger/.
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
T="hngin-r$RUN"
PD=hn-ginger
DS=hackernews_minilm_384
BIN=bench/ginger/build/ginger_bench
EF_SWEEP=10,20,40,80,160,320,640,1000
BEAM_SWEEP=10,20,40,64,80,160,320,640,1000
GARGS="--data bench/data --dataset $DS --config recommended --k 10 --num-queries 1000"

( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    bench/ginger/ginger_bench.cc bench/cloudlab/ginger_window.sh bench/cloudlab/pg_counts.sh \
    "$REMOTE_USER@$CLIENT.$DOMAIN:$DEST/" )
( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    bench/cloudlab/pg_counts.sh "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/" )

require_measurement_state

# 1. Ginger rebuild, in the background on the client node.
# shellcheck disable=SC2016
if ! step_done "$CLIENT" "$T-gbuild"; then
  start_step "$CLIENT" "$T-gbuild" '
    set -eo pipefail
    if postgres/install/bin/pg_ctl -D postgres/data status >/dev/null 2>&1; then
      echo "error: Postgres is running on the Ginger node; stop it first"; exit 1; fi
    [ -f /mydata/ginger_bench.orig ] || cp '"$BIN"' /mydata/ginger_bench.orig
    cmake -B bench/ginger/build bench/ginger -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-g1 > /dev/null
    cmake --build bench/ginger/build -j"$(nproc)" --target ginger_bench | tail -3
    '"$BIN"' --version
    n=$(readelf -S '"$BIN"' | grep -c debug_line || true)
    echo "line-table sections: $n"; [ "$n" -gt 0 ] || { echo "error: no line tables"; exit 1; }'
fi

# 2-4. pgvectorfast + tq4 work counts on the server node.
step "$SERVER" "$T-reserve" "$HP_RESERVE"
step "$SERVER" "$T-mount" \
  "./run_postgres.sh stop || true; sudo mount -o remount,huge=advise /dev/shm && findmnt -no OPTIONS /dev/shm"
step "$SERVER" "$T-serve" \
  "EXTRA_PG_OPTS='-c huge_pages=on -c calico.huge_translation=on' SERVE_ARM=fastntq LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
step "$SERVER" "$T-check-hp" "HP_MIN_GB=60; $HP_CHECK"
# shellcheck disable=SC2016
step "$SERVER" "$T-check-thp" '
  s=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW calico.huge_translation")
  h=$(awk "/^ShmemHugePages:/ {print \$2}" /proc/meminfo)
  echo "calico.huge_translation: $s; ShmemHugePages: $h kB"
  [ "$s" = on ] && [ "$h" -gt 0 ] || { echo "error: translation arrays are not on 2 MB pages"; exit 1; }'
step "$SERVER" "$T-pgcounts" "PROFILE_DIR=$PD bench/cloudlab/pg_counts.sh $EF_SWEEP"
step "$SERVER" "$T-release" "$HP_RELEASE"
step "$SERVER" "$T-unmount" "sudo mount -o remount,huge=never /dev/shm && findmnt -no OPTIONS /dev/shm"

wait_step "$CLIENT" "$T-gbuild"

# 5-6. Ginger.
step "$CLIENT" "$T-perf-check" \
  "[ \"\$(cat /proc/sys/kernel/perf_event_paranoid)\" -le 0 ] || sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0"
step "$CLIENT" "$T-gstats" \
  "mkdir -p bench/profiles/$PD && \$NUMA_CMD $BIN $GARGS --sweep $BEAM_SWEEP --threads 1 --duration 0 --stats --results bench/profiles/$PD/ginger-stats.jsonl"
for bt in 64:1:500009:2003 160:1:500009:2003 64:40:20000003:20011 160:40:20000003:20011; do
  IFS=: read -r b t pcyc plat <<< "$bt"
  name="ginger-t$t-b$b"
  if step_done "$CLIENT" "$T-run-$name" && step_done "$CLIENT" "$T-win-$name"; then
    say "$name: already done, skipping"; continue
  fi
  prog="$STEPS/$T-$name.progress"
  start_step "$CLIENT" "$T-run-$name" \
    "rm -f $prog; GINGER_PROGRESS=$prog \$NUMA_CMD $BIN $GARGS --sweep $b --threads $t --duration 150 --results bench/profiles/$PD/ginger-prof.jsonl"
  start_step "$CLIENT" "$T-win-$name" \
    "PERIOD_CYC=$pcyc PERIOD_LAT=$plat PROFILE_DIR=$PD bench/cloudlab/ginger_window.sh $name $prog 20 10"
  wait_step "$CLIENT" "$T-win-$name"
  wait_step "$CLIENT" "$T-run-$name"
done
step "$CLIENT" "$T-collect" \
  "cp $BIN bench/profiles/$PD/ginger_bench; cat /sys/kernel/mm/transparent_hugepage/enabled > bench/profiles/$PD/thp-enabled.txt"

ssh_to "$SERVER" "cd $DEST && tar cf - bench/profiles/$PD" | ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$CLIENT.$DOMAIN" "cd $DEST && tar xf -"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
mkdir -p "$HERE/../profiles/$PD"
rsync -az --exclude '*.data' -e "ssh ${SSH_OPTS[*]}" \
  "$REMOTE_USER@$CLIENT.$DOMAIN:$DEST/bench/profiles/$PD/" "$HERE/../profiles/$PD/"
say "done; profiles in bench/profiles/$PD/; logs: . bench/cloudlab/nodes-wisc.sh && bench/cloudlab/sync.sh pull"
