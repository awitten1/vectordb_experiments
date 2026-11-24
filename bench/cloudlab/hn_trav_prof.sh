#!/usr/bin/env bash
# What pgvectorfast + tq4's HNSW traversal (41% of server CPU per query on
# huge pages) is made of: which instructions stall, which loads are slow and
# where they are served from (L3, local or remote DRAM), how many misses are
# in flight at once, how often prefetches arrive late, and hops per query.
#
#   caffeinate -i bench/cloudlab/hn_trav_prof.sh
#
# Same tq4 index and library as hn_calico_thp.sh, in its better arm:
# huge_pages=on, calico.huge_translation=on, /dev/shm mounted huge=advise.
# Steps are detached and resumable (see lib.sh); bump RUN to redo them.
#   1. preflight: perf can take precise cycles, PEBS load latency (and
#      perf script prints weight and data source), and uprobes on the served
#      vectorfast.so; the library is the one with the optimized kernels
#   2. reserve huge pages, remount /dev/shm huge=advise, serve fastntq,
#      check huge pages and the translation-array setting
#   3. at 1 and 40 LAN clients, ef_search 160 (label trav-fastntq):
#      trav_window.sh (bench/profiles/hn-trav/)
#   4. stop Postgres, release the pool, /dev/shm back to huge=never; copy the
#      served vectorfast.so next to the profiles, for objdump -l
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
T="hntrav-r$RUN"
JSONL=hackernews_minilm_384.jsonl
PD=hn-trav
OUT="bench/profiles/$PD"
EF=160
# shellcheck disable=SC2016  # NTQ_* expand on the node, from hn.env
TQ_ARGS='--table "$NTQ_TABLE" $NTQ_ARGS'
LIB=postgres/install/lib/x86_64-linux-gnu/vectorfast.so

( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    bench/cloudlab/trav_window.sh "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/" )

require_measurement_state

# shellcheck disable=SC2016
step "$SERVER" "$T-preflight" '
  set -o pipefail
  [ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 0 ] || sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0
  mkdir -p '"$OUT"'
  pm=$(objdump -d --no-show-raw-insn '"$LIB"' | grep -c vpermps || true)
  g=$(grep -c "calico.huge_translation" <(strings '"$LIB"') || true)
  echo "served library: vpermps $pm, calico.huge_translation strings $g"
  [ "$pm" -gt 0 ] && [ "$g" -gt 0 ] || { echo "error: served library is not the optimized build with the setting"; exit 1; }
  echo "nmi_watchdog: $(cat /proc/sys/kernel/nmi_watchdog)"
  perf record -a -c 500009 -e cycles:pp -o /tmp/trav-test.data -- sleep 1 2>&1 | tail -1
  echo "precise cycles: ok"
  LAT="cpu/mem-loads,ldlat=50/P"
  perf record -e "$LAT" -- true >/dev/null 2>&1 || LAT="cpu/event=0xcd,umask=0x01,ldlat=50,name=load_latency/pp"
  perf record -a -W -d -c 2003 -e "$LAT" -o /tmp/trav-test.data -- sleep 1 2>&1 | tail -1
  echo "load latency event: $LAT"
  out=$(perf script -i /tmp/trav-test.data -F comm,pid,ip,sym,dso,addr,weight,data_src 2>&1 | head -3 || true)
  echo "$out"
  if [ -n "$out" ] && ! grep -qi "invalid\|not supported\|usage" <<< "$out"; then echo "perf script weight + data_src: ok"
  else echo "perf script weight + data_src: not confirmed (an idle 1 s test may have no samples to print)"; fi
  sudo -n true && echo "sudo: ok"
  rm -f /tmp/trav-test.data
  sudo -n perf probe -q -x '"$LIB"' -a HnswLoadUnvisitedRaw && sudo -n perf probe -q -d "probe_vectorfast:*" && echo "uprobes: ok" || echo "uprobes: refused (no hop counts)"
  uname -r; perf version'

step "$SERVER" "$T-reserve" "$HP_RESERVE"
step "$SERVER" "$T-mount" \
  "./run_postgres.sh stop || true; sudo mount -o remount,huge=advise /dev/shm && findmnt -no OPTIONS /dev/shm"
step "$SERVER" "$T-serve" \
  "EXTRA_PG_OPTS='-c huge_pages=on -c calico.huge_translation=on' SERVE_ARM=fastntq LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
# shellcheck disable=SC2016
step "$SERVER" "$T-numa" \
  '[ -n "${NUMA_CMD:-}" ] || exit 0; n=$(grep -c " interleave" /proc/$(head -1 postgres/data/postmaster.pid)/numa_maps); echo "postmaster mappings with interleave policy: $n"; [ "$n" -gt 0 ]'
step "$SERVER" "$T-check-hp" "HP_MIN_GB=60; $HP_CHECK"
# shellcheck disable=SC2016
step "$SERVER" "$T-check-thp" '
  s=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW calico.huge_translation")
  m=$(findmnt -no OPTIONS /dev/shm); h=$(awk "/^ShmemHugePages:/ {print \$2}" /proc/meminfo)
  echo "calico.huge_translation: $s; /dev/shm: $m; ShmemHugePages: $h kB"
  [ "$s" = on ] && [[ "$m" == *huge=advise* ]] && [ "$h" -gt 0 ] || { echo "error: translation arrays are not on 2 MB pages"; exit 1; }'

# Windows: 30 s warm-up, 5 counter groups + 2 recordings of 10 s (qps from
# xact_commit over those), then a 10 s uprobe window and perf script: the
# client runs 180 s so it outlasts all of them.
for cp in 1:500009:2003 40:20000003:20011; do
  IFS=: read -r c pcyc plat <<< "$cp"
  name="hn-fastntq-c$c-ef$EF"
  if step_done "$CLIENT" "$T-client-$name" && step_done "$SERVER" "$T-trav-$name"; then
    say "$name: already done, skipping"; continue
  fi
  start_step "$CLIENT" "$T-client-$name" \
    "BENCH_LABEL=trav-fastntq bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw --m 16 --ef-construction 64 $TQ_ARGS --sweep $EF --clients $c --num-queries \"\$NUM_QUERIES\" --duration 180 $REMOTE_DSN"
  start_step "$SERVER" "$T-trav-$name" "PERIOD_CYC=$pcyc PERIOD_LAT=$plat PROFILE_DIR=$PD bench/cloudlab/trav_window.sh $name 30 10"
  wait_step "$SERVER" "$T-trav-$name"
  wait_step "$CLIENT" "$T-client-$name"
done

step "$SERVER" "$T-check-end" "HP_MIN_GB=60; $HP_CHECK"
step "$SERVER" "$T-check-watchdog" "echo nmi_watchdog: \$(cat /proc/sys/kernel/nmi_watchdog); ! sudo -n perf probe -l 2>/dev/null | grep -q probe_vectorfast"
step "$SERVER" "$T-release" "$HP_RELEASE"
step "$SERVER" "$T-unmount" "sudo mount -o remount,huge=never /dev/shm && findmnt -no OPTIONS /dev/shm; cp $LIB $OUT/vectorfast.so"

ssh_to "$CLIENT" "cat $DEST/bench/results/$JSONL" |
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$SERVER.$DOMAIN" "cat > $DEST/$OUT/client-results.jsonl"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
mkdir -p "$HERE/../profiles/$PD"
rsync -az --exclude '*.data' -e "ssh ${SSH_OPTS[*]}" \
  "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/$OUT/" "$HERE/../profiles/$PD/"
say "done; profiles in bench/profiles/$PD/; results: . bench/cloudlab/nodes-wisc.sh && bench/cloudlab/sync.sh pull"
