#!/usr/bin/env bash
# hn_pipnn.sh's two arms again with shared_buffers on huge pages: the
# pgvectorfast fork on the PiPNN graph, then on the HNSW graph, so the graph
# comparison holds with huge pages too. (The fork has no
# calico.huge_translation; Calico's translation arrays stay on 4 KB pages.)
# Run after hn_pipnn_ginger.sh, with Postgres stopped:
#
#   caffeinate -i bench/cloudlab/hn_pipnn_hp.sh
#
# Steps are detached and resumable (see lib.sh); bump RUN to redo them.
#   1. reserve the huge-page pool (hp_lib.sh HP_RESERVE)
#   2. serve the fork on PIPNN_TABLE with huge_pages=on; check (HP_CHECK: a
#      >100 GB 2 MB-page mapping, >= 90 GB of the pool in use, interleave);
#      ef_search sweep at 1 and 40 LAN clients (label piphp-pipnn-fork)
#   3. the same on the main table's HNSW index (label piphp-hnsw-fork)
#   4. check again, stop Postgres, release the pool
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
T="hnpiph-r$RUN"
PD=hn-pipnn
JSONL=hackernews_minilm_384.jsonl
HP="EXTRA_PG_OPTS='-c huge_pages=on' FAST_SO=\$PIPNN_SO"

require_measurement_state

step "$SERVER" "$T-reserve" "$HP_RESERVE"

# arm <name> <serve env> <ann_bench index args, evaluated on the node>
arm() {
  local name="$1" senv="$2" iargs="$3"
  step "$SERVER" "$T-serve-$name" \
    "$HP $senv SERVE_ARM=fast LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
  step "$SERVER" "$T-check-$name" "$HP_CHECK"
  for c in 1 40; do
    step "$CLIENT" "$T-$name-c$c" \
      "BENCH_LABEL=piphp-$name-fork bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw $iargs --sweep \"\$EF_SWEEP\" --clients $c --num-queries \"\$NUM_QUERIES\" --duration \"\$DURATION\" $REMOTE_DSN"
  done
}
arm pipnn "SERVE_TABLE=\$PIPNN_TABLE" "--table \$PIPNN_TABLE --m \$PIPNN_M --ef-construction 64"
arm hnsw "" "--m 16 --ef-construction 64"

step "$SERVER" "$T-check-end" "$HP_CHECK"
step "$SERVER" "$T-release" "$HP_RELEASE"

ssh_to "$CLIENT" "cat $DEST/bench/results/$JSONL" |
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$SERVER.$DOMAIN" "cat > $DEST/bench/profiles/$PD/client-results.jsonl"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/bench/profiles/$PD/client-results.jsonl" "$HERE/../profiles/$PD/"
say "done; results in bench/profiles/$PD/client-results.jsonl (labels piphp-*)"
