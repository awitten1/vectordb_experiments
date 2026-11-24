#!/usr/bin/env bash
# Stock pgvector 0.8.7 searching the PiPNN graph that the pgvectorfast fork
# imported in hn_pipnn.sh (index items_hn_pipnn_hnsw_m32_ef_construction64).
# This separates the graph's gain from Calico's: the same graph under both
# engines, and both graphs under the same engine. Also a 1-client LAN baseline
# for stock on the HNSW graph, so that every Postgres arm at 1 client uses the
# protocol of hn_pipnn.sh (client on the other node). Run with Postgres
# stopped, after hn_pipnn.sh:
#
#   caffeinate -i bench/cloudlab/hn_pipnn_stock.sh
#
# Steps are detached and resumable (see lib.sh); bump RUN to redo them.
#   1. push run_experiment.sh and hn.env to both nodes
#   2. 4 KB pages: serve stock on PIPNN_TABLE; ef_search sweep at 1 LAN client
#      (label pipstock-pipnn), then a recall gate: stock and the fork run the
#      same float layer-0 search on the same graph, so recall must match the
#      fork's (pippipnn-fork) within 0.002 at every ef_search, or this stops;
#      then 40 clients. Serve stock on the main table: 1 client (pipstock-hnsw).
#   3. huge pages (hp_lib.sh): the same, labels piphp-pipnn-stock and
#      piphp-hnsw-stock, checked with HP_CHECK; release the pool
#   4. stop Postgres; client results into bench/profiles/hn-pipnn/
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
T="hnpis-r$RUN"
PD=hn-pipnn
JSONL=hackernews_minilm_384.jsonl
PIPNN_ARGS="--table \$PIPNN_TABLE --m \$PIPNN_M --ef-construction 64"
HNSW_IARGS="--m 16 --ef-construction 64"

for n in "$SERVER" "$CLIENT"; do
  ( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
      bench/run_experiment.sh bench/cloudlab/hn.env bench/cloudlab/recall_gate.py \
      "$REMOTE_USER@$n.$DOMAIN:$DEST/" )
done

require_measurement_state

# recall_gate <name> <label> <reference label>: on the client node, where the
# LAN results are, compare the 1-client recall of the two labels.
recall_gate() {
  step "$CLIENT" "$1" "bench/.venv/bin/python bench/cloudlab/recall_gate.py $2 $3"
}

# arms <tag> <serve env> <check command> <pipnn label> <hnsw label> <reference label>
arms() {
  local tag="$1" senv="$2" check="$3" plabel="$4" hlabel="$5" ref="$6"
  step "$SERVER" "$T-$tag-serve-pipnn" \
    "$senv SERVE_ARM=stock SERVE_TABLE=\$PIPNN_TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
  step "$SERVER" "$T-$tag-check-pipnn" "$check"
  bench_step "$CLIENT" "$T-$tag-pipnn-c1" "$plabel" "--index hnsw $PIPNN_ARGS --sweep \"\$EF_SWEEP\" --clients 1"
  recall_gate "$T-$tag-gate" "$plabel" "$ref"
  bench_step "$CLIENT" "$T-$tag-pipnn-c40" "$plabel" "--index hnsw $PIPNN_ARGS --sweep \"\$EF_SWEEP\" --clients 40"

  step "$SERVER" "$T-$tag-serve-hnsw" \
    "$senv SERVE_ARM=stock LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
  step "$SERVER" "$T-$tag-check-hnsw" "$check"
  bench_step "$CLIENT" "$T-$tag-hnsw-c1" "$hlabel" "--index hnsw $HNSW_IARGS --sweep \"\$EF_SWEEP\" --clients 1"
}

# shellcheck disable=SC2016  # expanded on the node
NUMA_CHECK='n=$(grep -c " interleave" /proc/$(head -1 postgres/data/postmaster.pid)/numa_maps); echo "postmaster mappings with interleave policy: $n"; [ "$n" -gt 0 ]'

arms 4k "" "$NUMA_CHECK" pipstock-pipnn pipstock-hnsw pippipnn-fork

step "$SERVER" "$T-hp-reserve" "$HP_RESERVE"
arms hp "EXTRA_PG_OPTS='-c huge_pages=on'" "$HP_CHECK" piphp-pipnn-stock piphp-hnsw-stock piphp-pipnn-fork
step "$SERVER" "$T-hp-release" "$HP_RELEASE"

ssh_to "$CLIENT" "cat $DEST/bench/results/$JSONL" |
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$SERVER.$DOMAIN" "cat > $DEST/bench/profiles/$PD/client-results.jsonl"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
mkdir -p "$HERE/../profiles/$PD"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/bench/profiles/$PD/client-results.jsonl" "$HERE/../profiles/$PD/"
say "done; labels pipstock-*, piphp-*-stock in bench/profiles/$PD/client-results.jsonl; then bench/cloudlab/sync.sh pull"
