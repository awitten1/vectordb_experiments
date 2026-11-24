#!/usr/bin/env bash
# Ginger searching the PiPNN graph that pgvectorfast searched in hn_pipnn.sh
# (PIPNN_GRAPH, exported from PIPNN_TABLE), to compare the two engines on the
# same graph. Run after hn_pipnn.sh, which leaves Postgres stopped:
#
#   caffeinate -i bench/cloudlab/hn_pipnn_ginger.sh
#
# On the server node, where the graph file is:
#   1. build bench/ginger (ginger_bench, with --graph); jemalloc as in the
#      main Ginger runs
#   2. --search float (float distances, no re-rank: what pgvectorfast's float
#      index computes) and --search native (Ginger's 4-bit traversal and 12-bit
#      re-rank), beam_size = the ef_search sweep, 1 and 40 threads; the
#      1-thread runs also record Ginger's search statistics (--stats)
# Results: bench/profiles/hn-pipnn/ginger-graph.jsonl.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
T="hnpipg-r$RUN"
PD=hn-pipnn
DS=hackernews_minilm_384

( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    bench/ginger/ginger_bench.cc bench/ginger/CMakeLists.txt build_ginger.sh \
    "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/" )

require_measurement_state

# shellcheck disable=SC2016
step "$SERVER" "$T-build" '
  set -eo pipefail
  if postgres/install/bin/pg_ctl -D postgres/data status >/dev/null 2>&1; then
    echo "error: Postgres is running; Ginger needs the machine"; exit 1; fi
  [ -s "$PIPNN_GRAPH" ] || { echo "error: no graph file $PIPNN_GRAPH"; exit 1; }
  pkg-config --exists jemalloc || sudo -n apt-get install -y -q libjemalloc-dev > /dev/null
  ./build_ginger.sh > bench/ginger/build.log 2>&1 ||
    { grep -E -A8 "error|FAILED" bench/ginger/build.log | head -60; exit 1; }
  tail -3 bench/ginger/build.log
  # The driver reads <name>_meta.npz (queries, ground truth); datasets.py writes it once.
  [ -f bench/data/'"$DS"'_meta.npz ] || bench/.venv/bin/python -c "
import sys, numpy as np; sys.path.insert(0, \"bench\"); import datasets
ds = datasets.load(\"hackernews\", num_queries=1000)
np.savez(datasets.DATA_DIR / f\"{ds.name}_meta.npz\", test=np.asarray(ds.test, dtype=np.float32), neighbors=np.asarray(ds.neighbors, dtype=np.int64))"
  ls -la bench/data/'"$DS"'_meta.npz'

for mode in float native; do
  for t in 1 40; do
    st=""; [ "$t" = 1 ] && st="--stats"
    step "$SERVER" "$T-$mode-t$t" \
      "mkdir -p bench/profiles/$PD && \$NUMA_CMD bench/ginger/build/ginger_bench --data bench/data --dataset $DS --graph \"\$PIPNN_GRAPH\" --search $mode --sweep \"\$GINGER_SWEEP\" --k 10 --num-queries \"\$NUM_QUERIES\" --duration \"\$DURATION\" --threads $t $st --results bench/profiles/$PD/ginger-graph.jsonl"
  done
done

ssh_to "$SERVER" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
mkdir -p "$HERE/../profiles/$PD"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/bench/profiles/$PD/ginger-graph.jsonl" "$HERE/../profiles/$PD/"
say "done; bench/profiles/$PD/ginger-graph.jsonl"
