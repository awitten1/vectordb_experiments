#!/usr/bin/env bash
# Ginger on 2 MB transparent huge pages, against a 4 KB control in the same
# session, on HN 27.4M (recommended config, the cached index the main study
# measured). Run from the workstation, with Postgres stopped on the client node
# (it never runs there):
#
#   caffeinate -i bench/cloudlab/hn_ginger_thp.sh
#
# Ginger asks for huge pages only for its float dataset (SpanDataset), which
# the recommended config drops. Its codes and graph come from jemalloc, and THP
# is `madvise` on the nodes, so it runs on 4 KB pages by default. jemalloc's
# MALLOC_CONF=thp:always makes it madvise(MADV_HUGEPAGE) every mapping it
# makes: huge pages for this process only, with no change to Ginger or to the
# system's THP mode.
#
# On the client node, as in the main study, for 1 and 40 threads (THREADS), the
# control then the THP run (labels ginger-4kctl-recommended,
# ginger-thp-recommended). Before each: drop the page cache and compact memory,
# so 2 MB pages are there to be had. During each: sample the process's
# smaps_rollup every 15 s. After: ginger_thp_promote.py keeps the rows only if
# the THP run had >= 80% of its anonymous memory on huge pages and the control
# none. Steps are resumable (lib.sh); bump RUN to redo them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
THREADS="${THREADS:-1 40}"
T="hngthp-r$RUN"
DS=hackernews_minilm_384
PD=bench/profiles/hn-ginger-thp
INDEX="bench/data/ginger/${DS}_ginger_recommended_a1.08.ginger"

( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    bench/cloudlab/hn.env bench/cloudlab/ginger_thp_promote.py "$REMOTE_USER@$CLIENT.$DOMAIN:$DEST/" )

require_measurement_state

# shellcheck disable=SC2016  # expanded on the node
step "$CLIENT" "$T-check" '
  set -e
  [ -x bench/ginger/build/ginger_bench ] || { echo "error: no ginger_bench"; exit 1; }
  [ -s '"$INDEX"' ] || { echo "error: no cached index '"$INDEX"'"; exit 1; }
  ldd bench/ginger/build/ginger_bench | grep -q libjemalloc || { echo "error: ginger_bench is not linked with jemalloc"; exit 1; }
  if postgres/install/bin/pg_ctl -D postgres/data status >/dev/null 2>&1; then echo "error: Postgres is running"; exit 1; fi
  if pgrep -x ginger_bench >/dev/null; then echo "error: a ginger_bench is already running"; exit 1; fi
  thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled); echo "THP enabled: $thp"
  case "$thp" in *"[never]"*) echo "error: THP is off"; exit 1 ;; esac
  dpkg -l libjemalloc2 | tail -1
  mkdir -p '"$PD"' && ls -la '"$INDEX"''

# shellcheck disable=SC2016  # expanded on the node
SAMPLER='
  f='"$PD"'/$ARM-t$THR.smaps.csv
  echo "t,rss_kb,anon_kb,anon_huge_kb" > $f
  pid=""
  for i in $(seq 600); do pid=$(pgrep -x ginger_bench | head -1); [ -n "$pid" ] && break; sleep 1; done
  [ -n "$pid" ] || { echo "error: ginger_bench never started"; exit 1; }
  echo "sampling pid $pid"
  while kill -0 "$pid" 2>/dev/null; do
    awk -v t="$(date +%s)" "/^Rss:/ {r=\$2} /^Anonymous:/ {a=\$2} /^AnonHugePages:/ {h=\$2} END {if (r) print t\",\"r\",\"a\",\"h}" /proc/$pid/smaps_rollup >> $f 2>/dev/null || true
    sleep 15
  done
  echo "$(( $(wc -l < $f) - 1 )) samples"'

for t in $THREADS; do
  for arm in 4k thp; do
    if [ "$arm" = thp ]; then mc="MALLOC_CONF=thp:always,metadata_thp:always"; label=ginger-thp-recommended
    else mc=""; label=ginger-4kctl-recommended; fi
    out="$PD/$arm-t$t.jsonl"
    if ! step_done "$CLIENT" "$T-$arm-t$t-promote"; then
      # The sampler only goes with a fresh run: a resumed driver whose run is
      # done goes straight to the check, on the samples it already has.
      if ! step_done "$CLIENT" "$T-$arm-t$t"; then
        start_step "$CLIENT" "$T-$arm-t$t-sample" "ARM=$arm THR=$t; $SAMPLER"
        step "$CLIENT" "$T-$arm-t$t" \
          "rm -f $out && sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null && echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null && $mc \$NUMA_CMD bench/ginger/build/ginger_bench --data bench/data --dataset $DS --config recommended --sweep \"\$GINGER_SWEEP\" --k 10 --num-queries \"\$NUM_QUERIES\" --duration \"\$DURATION\" --threads $t --results $out"
        wait_step "$CLIENT" "$T-$arm-t$t-sample"
      fi
      step "$CLIENT" "$T-$arm-t$t-promote" \
        "bench/.venv/bin/python bench/cloudlab/ginger_thp_promote.py $out $PD/$arm-t$t.smaps.csv $label $arm"
    else
      say "$T-$arm-t$t: already done, skipping"
    fi
  done
done

ssh_to "$CLIENT" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
mkdir -p "$HERE/../profiles/hn-ginger-thp"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$CLIENT.$DOMAIN:$DEST/$PD/" "$HERE/../profiles/hn-ginger-thp/"
say "done; labels ginger-thp-recommended and ginger-4kctl-recommended; then bench/cloudlab/sync.sh pull"
