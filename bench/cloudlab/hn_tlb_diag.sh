#!/usr/bin/env bash
# Where do pgvectorfast + tq4's remaining page walks come from? On huge pages
# (optimized kernels) it still walks ~6,900 times per query at 1 client and
# ~8,400 at 40, 12-17% of cycles. This attributes them, on the same tq4 index
# and serve setup as hn_fastntq_k2.sh's huge-page phase:
#
#   caffeinate -i bench/cloudlab/hn_tlb_diag.sh
#
# Steps are detached and resumable (see lib.sh); bump RUN to redo them.
#   1. preflight: perf can sample MEM_UOPS_RETIRED.STLB_MISS_* with PEBS and
#      data addresses; the served library is the optimized-kernel build; log
#      THP mode and 1 GB page support (for whatever the fix turns out to be)
#   2. reserve huge pages, serve fastntq with huge_pages=on, prewarmed;
#      HP_CHECK after the start and at the end
#   3. at 1 and 40 LAN clients, ef_search 160 (label tlbhp-fastntq):
#      tlb_window.sh: walk counters split by page size, where the walker's
#      reads hit, and PEBS samples of STLB-missing loads/stores with their data
#      addresses, placed against /proc/<pid>/maps (bench/profiles/hn-tlb/)
#   4. stop Postgres, release the pool, write bench/profiles/hn-tlb/attrib.md
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
T="hntlb-r$RUN"
HP="EXTRA_PG_OPTS='-c huge_pages=on'"
JSONL=hackernews_minilm_384.jsonl
PD=hn-tlb
OUT="bench/profiles/$PD"
EF=160
# shellcheck disable=SC2016  # NTQ_* expand on the node, from hn.env
TQ_ARGS='--table "$NTQ_TABLE" $NTQ_ARGS'

# The two server-side scripts are new: copy just them (not a full sync).
( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    bench/cloudlab/tlb_window.sh bench/cloudlab/tlb_attrib.py "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/" )

require_measurement_state

# shellcheck disable=SC2016
step "$SERVER" "$T-preflight" '
  set -o pipefail
  [ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 0 ] || sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0
  mkdir -p '"$OUT"'
  pm=$(objdump -d --no-show-raw-insn pgvectorfast_ntq/vectorfast.so | grep -c vpermps || true)
  echo "vpermps in pgvectorfast_ntq/vectorfast.so: $pm"
  [ "$pm" -gt 0 ] || { echo "error: pgvectorfast_ntq/vectorfast.so is not the optimized-kernel build"; exit 1; }
  ev="cpu/event=0xd0,umask=0x11,name=stlb_miss_loads/pp"
  perf record -a -d -c 10007 -e "$ev" -o /tmp/pebs-test.data -- sleep 1 2>&1 | tail -2
  echo "PEBS with data addresses: ok"
  if perf record -a -d -c 10007 -e "$ev" --call-graph lbr -o /tmp/pebs-test.data -- sleep 1 >/dev/null 2>&1
  then echo "PEBS with LBR call stacks: ok"; else echo "PEBS with LBR call stacks: refused (flat samples only)"; fi
  rm -f /tmp/pebs-test.data
  for e in dtlb_load_misses.walk_completed_4k dtlb_load_misses.walk_completed_2m_4m page_walker_loads.dtlb_memory; do
    perf stat -e $e -- true >/dev/null 2>&1 && echo "named event $e: ok" || echo "named event $e: missing, raw encoding used"
  done
  echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled); defrag: $(cat /sys/kernel/mm/transparent_hugepage/defrag)"
  echo "1 GB pages: CPU $(grep -qw pdpe1gb /proc/cpuinfo && echo supports || echo lacks) pdpe1gb; reserved now: $(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages 2>/dev/null || echo n/a)"
  uname -r'

step "$SERVER" "$T-reserve" "$HP_RESERVE"
step "$SERVER" "$T-serve" \
  "$HP SERVE_ARM=fastntq LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
# shellcheck disable=SC2016
step "$SERVER" "$T-numa" \
  '[ -n "${NUMA_CMD:-}" ] || exit 0; n=$(grep -c " interleave" /proc/$(head -1 postgres/data/postmaster.pid)/numa_maps); echo "postmaster mappings with interleave policy: $n"; [ "$n" -gt 0 ]'
step "$SERVER" "$T-check-serve" "HP_MIN_GB=60; $HP_CHECK"

# 1 client: ~5.6M walks/s, so one sample per 2,003 STLB-missing loads gives
# ~25k samples in 10 s; 40 clients walk ~20x as often, so sample 10x sparser.
for cp in 1:2003 40:20011; do
  c=${cp%%:*} period=${cp#*:}
  name="hn-fastntq-c$c-ef$EF"
  if step_done "$CLIENT" "$T-client-$name" && step_done "$SERVER" "$T-tlb-$name"; then
    say "$name: already done, skipping"; continue
  fi
  start_step "$CLIENT" "$T-client-$name" \
    "BENCH_LABEL=tlbhp-fastntq bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw --m 16 --ef-construction 64 $TQ_ARGS --sweep $EF --clients $c --num-queries \"\$NUM_QUERIES\" --duration 150 $REMOTE_DSN"
  start_step "$SERVER" "$T-tlb-$name" "PERIOD=$period PROFILE_DIR=$PD bench/cloudlab/tlb_window.sh $name 30 10"
  wait_step "$SERVER" "$T-tlb-$name"
  wait_step "$CLIENT" "$T-client-$name"
done

step "$SERVER" "$T-check-end" "HP_MIN_GB=60; $HP_CHECK"
step "$SERVER" "$T-release" "$HP_RELEASE"

ssh_to "$CLIENT" "cat $DEST/bench/results/$JSONL" |
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$SERVER.$DOMAIN" "cat > $DEST/$OUT/client-results.jsonl"
step "$SERVER" "$T-attrib" \
  "set -o pipefail; python3 bench/cloudlab/tlb_attrib.py --label tlbhp $OUT $OUT/client-results.jsonl | tee $OUT/attrib.md"

for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
mkdir -p "$HERE/../profiles/$PD"
rsync -az --exclude '*.data' -e "ssh ${SSH_OPTS[*]}" \
  "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/$OUT/" "$HERE/../profiles/$PD/"
say "done; attribution in bench/profiles/$PD/attrib.md; results: . bench/cloudlab/nodes-wisc.sh && bench/cloudlab/sync.sh pull"
