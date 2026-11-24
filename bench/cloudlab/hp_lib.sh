# Huge-page helpers shared by drivers (sourced after lib.sh; uses SERVER, step).
# Same commands as hn_hugepages.sh, which ran with its own copy.
#
# HP_RESERVE: Postgres stopped, page cache dropped, memory compacted, then
#   vm.nr_hugepages = what `postgres -C shared_memory_size_in_huge_pages`
#   reports for QUERY_SB (+1%); fails if the kernel cannot reserve them all.
# HP_CHECK: the running server is on huge pages (huge_pages_status=on, a
#   >100 GB 2 MB-page shared mapping, at least HP_MIN_GB (default 90) GB of the
#   pool in use, i.e. the prewarmed table and index) with NUMA interleave;
#   fails otherwise.
# HP_RELEASE: stop Postgres and give the pool back.
# shellcheck disable=SC2016  # expands on the node
HP_RESERVE='
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
  [ "$s" = on ] && [ -n "$big" ] && [ "$used_gb" -ge "${HP_MIN_GB:-90}" ] && [ "$h" -gt 0 ] ||
    { echo "error: the server is not running on huge pages as expected"; exit 1; }'
HP_RELEASE="./run_postgres.sh stop && sudo sysctl -w vm.nr_hugepages=0 && grep -E '^HugePages_Total' /proc/meminfo"
