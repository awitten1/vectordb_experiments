#!/usr/bin/env bash
# Stock pgvector on Hacker News 27.4M again, with shared_buffers in 2 MB huge
# pages (huge_pages=on), on the same index and protocol as hn_qps.sh. The
# profiles showed 16% of cycles walking page tables at 1 client and 26% at 40
# with 4 KB pages over 112 GB of shared_buffers.
#
#   caffeinate -i bench/cloudlab/hn_hugepages.sh
#
# Steps are detached and resumable (see lib.sh); bump RUN to redo them all.
#   1. reserve the huge pages: Postgres stopped, page cache dropped, memory
#      compacted, vm.nr_hugepages = what `postgres -C
#      shared_memory_size_in_huge_pages` reports for these settings (+1%);
#      fails if the kernel cannot reserve them all
#   2. 1-client rounds, client local: stock twice (labels hp-stock-r1, -r2)
#   3. serve stock for LAN clients
#   hp_check after the 1-client rounds, after the LAN serve and at the end:
#      huge_pages_status=on, a >100 GB 2 MB-page shared mapping, >=90 GB of the
#      pool in use, NUMA interleave still applied (fails the run otherwise)
#   4. 10, 20 and 40 clients from the client node (label hpsc-stock)
#   5. perf profiles at ef_search 160, 1 and 40 clients (label profhp-stock,
#      bench/profiles/hn-hugepages/), the same windows as hn_profile_tq4.sh
#   6. stop Postgres and release the huge pages (vm.nr_hugepages=0), so a
#      later run with 4 KB pages has the memory back
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
T="hnhp-r$RUN"
HP="EXTRA_PG_OPTS='-c huge_pages=on'"
PD=hn-hugepages
OUT="bench/profiles/$PD"
EF=160

require_measurement_state

# hp_check <step>: prove the running server's shared memory is in huge pages:
#   - Postgres says so (huge_pages_status = on)
#   - the postmaster's shared-memory mapping, which every backend inherits at
#     fork, is a hugetlb mapping with 2 MB kernel pages and over 100 GB
#   - the pages are really in use: at least 90 GB of the pool is faulted in
#     (table 42 GB + index 52 GB were prewarmed into shared_buffers)
#   - the NUMA interleave policy still applies to it
# shellcheck disable=SC2016  # expands on the node
HP_CHECK='
  s=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW huge_pages_status")
  echo "huge_pages_status: $s"
  pid=$(head -1 postgres/data/postmaster.pid)
  big=$(awk "/^[0-9a-f]+-[0-9a-f]+ / {m=\$0} /^Size:/ {sz=\$2} /^KernelPageSize:/ {if (\$2 == 2048 && sz > 100*1024*1024) print sz\" kB, 2 MB pages: \"m}" /proc/$pid/smaps)
  echo "hugetlb shared-memory mapping: ${big:-none}"
  read -r tot free < <(awk "/^HugePages_Total:/ {t=\$2} /^HugePages_Free:/ {f=\$2} END {print t, f}" /proc/meminfo)
  used_gb=$(( (tot - free) * 2 / 1024 ))
  echo "huge pages in use: $(( tot - free )) of $tot (${used_gb} GB)"
  n=$(grep -c " interleave" /proc/$pid/numa_maps); h=$(grep " huge" /proc/$pid/numa_maps | grep -c interleave)
  echo "postmaster mappings with interleave policy: $n (huge ones: $h)"
  grep " huge" /proc/$pid/numa_maps | head -2
  [ "$s" = on ] && [ -n "$big" ] && [ "$used_gb" -ge 90 ] && [ "$h" -gt 0 ] ||
    { echo "error: the server is not running on huge pages as expected"; exit 1; }'
hp_check() { step "$SERVER" "$T-check-$1" "$HP_CHECK"; }

# shellcheck disable=SC2016  # expands on the node
step "$SERVER" "$T-reserve" '
  ./run_postgres.sh stop
  n=$(postgres/install/bin/postgres -D postgres/data -C shared_memory_size_in_huge_pages \
        -c shared_buffers=$QUERY_SB -c huge_pages=on \
        -c max_worker_processes=$(( WORKERS + 9 )) -c max_parallel_workers=$(( WORKERS + 1 )))
  [ "$n" -gt 0 ] || { echo "error: could not size the huge-page pool"; exit 1; }
  want=$(( n + n / 100 + 16 ))
  echo "postgres needs $n huge pages; reserving $want ($(( want * 2 / 1024 )) GB)"
  sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null && echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null
  sudo sysctl -w vm.nr_hugepages=$want
  got=$(awk "/^HugePages_Total:/ {print \$2}" /proc/meminfo)
  grep -E "^(HugePages_|Hugepagesize|MemFree|MemAvailable)" /proc/meminfo
  cat /sys/devices/system/node/node*/hugepages/hugepages-2048kB/nr_hugepages | paste -sd" " | sed "s/^/per NUMA node: /"
  [ "$got" -ge "$want" ] || { echo "error: only $got of $want huge pages reserved"; sudo sysctl -w vm.nr_hugepages=0; exit 1; }'

step "$SERVER" "$T-local" "$HP KEEP_RUNNING=1 PVF_LABEL=hp PVF_ARMS='stock stock' bench/run_experiment.sh pgvectorfast"
hp_check local   # the 1-client server is still up

step "$SERVER" "$T-serve" \
  "$HP SERVE_ARM=stock LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
hp_check serve


for c in 10 20 40; do
  # shellcheck disable=SC2016
  bench_step "$CLIENT" "$T-c$c" hpsc-stock "$HNSW_ARGS"" --clients $c"
done

# Profiles: same windows as hn_profile_tq4.sh (30 s warm-up, four 20 s windows
# during a 130 s client run). perf stays enabled from that run.
step "$SERVER" "$T-perf-check" \
  "[ \"\$(cat /proc/sys/kernel/perf_event_paranoid)\" -le 0 ] || sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0; mkdir -p $OUT"
for c in 1 40; do
  name="hn-stock-c$c-ef$EF"
  if step_done "$CLIENT" "$T-client-$name" && step_done "$SERVER" "$T-perf-$name"; then
    say "$name: already done, skipping"; continue
  fi
  start_step "$CLIENT" "$T-client-$name" \
    "BENCH_LABEL=profhp-stock bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw --m 16 --ef-construction 64 --sweep $EF --clients $c --num-queries \"\$NUM_QUERIES\" --duration 130 $REMOTE_DSN"
  start_step "$SERVER" "$T-perf-$name" "CALLGRAPH=lbr PROFILE_DIR=$PD bench/cloudlab/perf_window.sh $name 30 20"
  wait_step "$SERVER" "$T-perf-$name"
  wait_step "$CLIENT" "$T-client-$name"
done

hp_check end     # still on huge pages after every LAN and profiled run
step "$SERVER" "$T-release" "./run_postgres.sh stop && sudo sysctl -w vm.nr_hugepages=0 && grep -E '^HugePages_Total' /proc/meminfo"

ssh_to "$CLIENT" "cat $DEST/bench/results/hackernews_minilm_384.jsonl" |
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
