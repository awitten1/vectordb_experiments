#!/usr/bin/env bash
# Why pgvectorfast + PiPNN + tq4 on 32 KB pages in the reordered layout (label
# g5hp32kro-pgvectorfast-tq4) can't go faster, and why Ginger is still ahead.
# Profiles both engines at equal width, where they do the same work: 64
# (recall ~0.95) and 160 (~0.986).
#
#   A  Postgres over the LAN, as benchmarked (ann_bench on the client node):
#      ef 64 at 1, 60 and 40 clients; ef 160 at 1 and 60 clients
#   B  Postgres without the LAN and without a protocol round trip and a
#      transaction per query: pgbench on the server running a statement that
#      answers 100 queries (gap_queries.py), 1 and 40 backends, ef 64 and 160
#   C  Ginger (recommended config, its own index, THP as in hn_g5.sh) on the
#      client node, beam 64 and 160, 1 and 40 threads; one --stats pass
# plus the work per query in Postgres (pg_counts.sh: uprobes on the served
# library). Each run has a gap_window.sh on its node: counters (topdown, cache
# and DRAM, uncore bandwidth), call stacks, and perf c2c under load.
# Served exactly as hn_g5_32k_ro.sh served the arm (huge pages, QUERY_SB).
# Nothing is built or dropped; the only new object is the small hn_queries
# table. Results go to bench/profiles/hn-gap/ (label g5gap-pg; the chart's
# data is untouched). Analysis: bench/cloudlab/gap_attrib.py.
#
# Run from the workstation (no sync.sh push: its --delete removes builds that
# exist only on the server):
#   caffeinate -i bench/cloudlab/hn_g5_gap.sh
# Steps are detached and resumable (lib.sh); bump RUN to redo them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
export SMOKE=0
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"
# shellcheck source=bench/cloudlab/hp_lib.sh
. "$HERE/hp_lib.sh"

RUN="${RUN:-1}"
T="hngap-r$RUN"
LABEL=g5gap-pg
D32=/mydata/vectordb_experiments_32k
NTQ32="$D32/pgvectorfast_ntq/vectorfast.so"
TABLE=items_hn_pipnn_ro
IDX=items_hn_pipnn_ro_hnsw_m32_ef_construction64_tq4_ntq12
DS=hackernews_minilm_384
TQ_ARGS="--index hnsw --table $TABLE --m \$PIPNN_M --ef-construction 64 --quantizer tq4 --rerank ntq12"
RD=bench/profiles/hn-gap                       # on each node, under its checkout
GW="PROFILE_DIR=hn-gap bench/cloudlab/gap_window.sh"
DUR=150                                        # seconds of load per run
PY=bench/.venv/bin/python
LPD="$HERE/../../bench/profiles/hn-gap"

ENVS8="$ENVS"
ENVS32="${ENVS/cd $DEST /cd $D32 }"
[ "$ENVS32" != "$ENVS8" ] || { echo "error: could not point ENVS at $D32" >&2; exit 1; }
on32() { local save="$ENVS"; ENVS="$ENVS32"; "$@"; ENVS="$save"; }

# measured <wwrap> <wnode> <wname> <lwrap> <lnode> <wenv> <load command>: a
# gap_window.sh step on <wnode> (QPS source etc. in <wenv>) alongside one run
# of load as step $T-<wname> on <lnode>, both detached. A wrap of on32 runs
# that step in the 32 KB checkout, "-" in the default one. Skipped once the
# window's step has succeeded.
measured() {
  local ww="$1" wn="$2" wname="$3" lw="$4" ln="$5" wenv="$6" lcmd="$7"
  [ "$ww" = - ] && ww=""; [ "$lw" = - ] && lw=""
  if step_done "$wn" "$T-w-$wname"; then say "$T-w-$wname: already done, skipping"; return; fi
  $ww start_step "$wn" "$T-w-$wname" "$wenv $GW $wname 20 10"
  $lw step "$ln" "$T-$wname" "$lcmd"
  $ww wait_step "$wn" "$T-w-$wname"
}

require_measurement_state

# ---- 0. The new and changed files, to the 32 KB checkout and the client ----------
( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    bench/cloudlab/gap_window.sh bench/cloudlab/gap_queries.py bench/cloudlab/pg_counts.sh \
    "$REMOTE_USER@$SERVER.$DOMAIN:$D32/" )
( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    bench/cloudlab/gap_window.sh "$REMOTE_USER@$CLIENT.$DOMAIN:$DEST/" )

# ---- 1. perf on both nodes ---------------------------------------------------------
# shellcheck disable=SC2016  # expanded on the node
PREFLIGHT='
  set -eo pipefail
  sudo -n sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0
  perf stat -a -e cycles,uncore_imc/cas_count_read/,unc_upi_txl_flits.all_data -M TopdownL1 -- sleep 0.2 2>&1 | tail -12
  python3 --version
  mkdir -p '"$RD"
for n in "$SERVER" "$CLIENT"; do
  wrap=""; [ "$n" = "$SERVER" ] && wrap=on32
  $wrap step "$n" "$T-preflight" "$PREFLIGHT"'
    if pgrep -x ginger_bench >/dev/null; then echo "error: a ginger_bench is running"; exit 1; fi
    echo preflight ok'
done

# ---- 2. Serve the arm on huge pages, as hn_g5_32k_ro.sh did ------------------------
on32 step "$SERVER" "$T-hp-reserve" "$HP_RESERVE"
on32 step "$SERVER" "$T-hp-serve" \
  "EXTRA_PG_OPTS='-c huge_pages=on' FAST_SO=$NTQ32 SERVE_ARM=fast SERVE_TABLE=$TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
# shellcheck disable=SC2016  # expanded on the node
on32 step "$SERVER" "$T-hp-check" 'HP_MIN_GB=50; '"$HP_CHECK"'
  b=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW block_size")
  echo "block_size: $b"; [ "$b" = 32768 ] || { echo "error: not a 32 KB cluster"; exit 1; }
  lib=postgres/install/lib/x86_64-linux-gnu/vectorfast.so
  cmp -s $lib '"$NTQ32"' && echo "served library is '"$NTQ32"'" ||
    { echo "error: $lib is not '"$NTQ32"'"; exit 1; }'

# ---- 3. The queries as a table; the batch statement's plan and recall ----------------
on32 step "$SERVER" "$T-queries" \
  "$PY bench/cloudlab/gap_queries.py --dataset \"\$DATASET\" --table $TABLE --index $IDX --ef 64 --script $RD/batch.sql"

# ---- 4. A: over the LAN ---------------------------------------------------------------
for run in "64 1" "64 60" "64 40" "160 1" "160 60"; do
  read -r ef c <<< "$run"
  c2c=0; [ "$c" -gt 1 ] && c2c=1
  measured on32 "$SERVER" "pg-lan-ef$ef-c$c" - "$CLIENT" "QPS_SRC=xact C2C=$c2c" \
    "BENCH_LABEL=$LABEL bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" $TQ_ARGS --sweep $ef --clients $c --num-queries \"\$NUM_QUERIES\" --duration $DUR $REMOTE_DSN"
done

# ---- 5. B: inside the server, 100 queries per statement -------------------------------
for run in "64 1" "64 40" "160 1" "160 40"; do
  read -r ef c <<< "$run"
  c2c=0; [ "$c" -gt 1 ] && c2c=1
  measured on32 "$SERVER" "pg-batch-ef$ef-c$c" on32 "$SERVER" "QPS_SRC=xact QPS_MULT=100 C2C=$c2c" \
    "PGOPTIONS='-c hnsw.ef_search=$ef' postgres/install/bin/pgbench -n -M simple -f $RD/batch.sql -c $c -j $c -T $DUR -P 30 -p \$PGPORT -U postgres postgres > $RD/pg-batch-ef$ef-c$c.pgbench.txt 2>&1; cat $RD/pg-batch-ef$ef-c$c.pgbench.txt"
done

# ---- 6. Work per query (uprobes; a local client) ----------------------------------------
on32 step "$SERVER" "$T-counts" \
  "PROFILE_DIR=hn-gap PG_COUNTS_PROBES='hnswrescan HnswLoadUnvisitedRaw HnswTqQueryDistance RerankDistanceOptimistic HnswTqRerankCands CalicoLookupAndPin' PG_COUNTS_ARGS='--m 32 --ef-construction 64 --table $TABLE --quantizer tq4 --rerank ntq12' bench/cloudlab/pg_counts.sh 64,160"
on32 step "$SERVER" "$T-hp-release" "$HP_RELEASE"

# ---- 7. C: Ginger on the client node ----------------------------------------------------
GPREP="sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null && echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null"
GBIN="MALLOC_CONF=thp:always,metadata_thp:always \$NUMA_CMD bench/ginger/build/ginger_bench --data bench/data --dataset $DS --config recommended --k 10 --num-queries \"\$NUM_QUERIES\""
step "$CLIENT" "$T-ginger-stats" "$GPREP && $GBIN --sweep 64,160 --duration 5 --threads 1 --stats --results $RD/ginger-stats.jsonl"
for run in "64 1" "64 40" "160 1" "160 40"; do
  read -r b t <<< "$run"
  c2c=0; [ "$t" -gt 1 ] && c2c=1
  pf="/tmp/gap-progress-b$b-t$t"
  measured - "$CLIENT" "ginger-b$b-t$t" - "$CLIENT" "QPS_SRC=progress:$pf C2C=$c2c" \
    "rm -f $pf && $GPREP && GINGER_PROGRESS=$pf $GBIN --sweep $b --duration $DUR --threads $t --results $RD/ginger-b$b-t$t.jsonl"
done

# ---- 8. Everything back -------------------------------------------------------------------
mkdir -p "$LPD/server" "$LPD/client"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$D32/$RD/" "$LPD/server/"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$CLIENT.$DOMAIN:$DEST/$RD/" "$LPD/client/"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-*.log" "$LPD/server/"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$CLIENT.$DOMAIN:$STEPS/$T-*.log" "$LPD/client/"
ssh_to "$CLIENT" "grep -E '\"label\": \"$LABEL\"' $DEST/bench/results/$DS.jsonl" > "$LPD/results.jsonl"
say "done; profiles in bench/profiles/hn-gap/ (analyse with gap_attrib.py)"
