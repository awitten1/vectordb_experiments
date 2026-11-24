#!/usr/bin/env bash
# Where Ginger's speed comes from: its recommended config on HN 27.4M, with
# one piece turned off at a time, at 1 and 40 threads, on the client node (as
# in the main study). Postgres never runs there.
#
#   caffeinate -i bench/cloudlab/hn_ginger_ablation.sh
#
# Arms (labels ginger-abl-<arm>):
#   full         recommended: 4-bit TurboQuant traversal, 12-bit re-rank of the
#                top 6k, graph reordering + compression, float data dropped
#   norerank     the same index, no re-rank (4-bit distances only)
#   floatrerank  re-rank with floats instead of 12-bit codes (keep-floats index)
#   float        float traversal, no re-rank: no quantization (keep-floats)
#   nocomp       graph compression off (its own index)
#   noreorder    graph reordering and compression off (its own index)
# Huge pages are measured separately (hn_ginger_thp.sh).
#
# The keep-floats, no-compression and no-reorder indexes are built first (one
# short run each, which also saves them); every measured run then loads its
# index from disk. Steps are resumable (lib.sh); bump RUN to redo them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
THREADS="${THREADS:-1 40}"
T="hngabl-r$RUN"
DS=hackernews_minilm_384
PD=bench/profiles/hn-ginger-ablation
BIN=bench/ginger/build/ginger_bench

( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    bench/cloudlab/hn.env bench/ginger/ginger_bench.cc "$REMOTE_USER@$CLIENT.$DOMAIN:$DEST/" )

require_measurement_state

# shellcheck disable=SC2016  # expanded on the node
step "$CLIENT" "$T-check" '
  set -e
  cmake --build bench/ginger/build --target ginger_bench 2>&1 | tail -1
  grep -q "keep-floats" bench/ginger/ginger_bench.cc
  if postgres/install/bin/pg_ctl -D postgres/data status >/dev/null 2>&1; then echo "error: Postgres is running"; exit 1; fi
  if pgrep -x ginger_bench >/dev/null; then echo "error: a ginger_bench is already running"; exit 1; fi
  ls -la bench/data/ginger/'"$DS"'_ginger_recommended_a1.08.ginger
  mkdir -p '"$PD"'; df -h /mydata | tail -1'

# Build the three extra indexes (each run saves its index; its one row, with
# the build time, goes to the profile directory, not the results).
for cfg in keep-floats no-compression no-reorder; do
  step "$CLIENT" "$T-build-$cfg" \
    "\$NUMA_CMD $BIN --data bench/data --dataset $DS --config $cfg --label abl-build-$cfg --sweep 10 --k 10 --num-queries \"\$NUM_QUERIES\" --duration 1 --threads 1 --results $PD/build-$cfg.jsonl"
done

# arm <name> <ginger_bench arguments>
arms=(
  "full|--config recommended"
  "norerank|--config recommended --rerank none"
  "floatrerank|--config keep-floats --rerank float"
  "float|--config keep-floats --traversal float --rerank none"
  "nocomp|--config no-compression"
  "noreorder|--config no-reorder"
)
for t in $THREADS; do
  for a in "${arms[@]}"; do
    name="${a%%|*}" args="${a#*|}"
    step "$CLIENT" "$T-$name-t$t" \
      "\$NUMA_CMD $BIN --data bench/data --dataset $DS $args --label abl-$name --sweep \"\$GINGER_SWEEP\" --k 10 --num-queries \"\$NUM_QUERIES\" --duration \"\$DURATION\" --threads $t"
  done
done

ssh_to "$CLIENT" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
mkdir -p "$HERE/../profiles/hn-ginger-ablation"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$CLIENT.$DOMAIN:$DEST/$PD/" "$HERE/../profiles/hn-ginger-ablation/"
say "done; labels ginger-abl-*; then bench/cloudlab/sync.sh pull"
