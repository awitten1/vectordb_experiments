#!/usr/bin/env bash
# Does Postgres catch up at low recall with more clients? The best arm (PiPNN +
# tq4, 32 KB pages, reordered, rerank capped at 100, vector rerank kernel: as
# hn_g5_simd.sh left the server's library) at ef_search 10, 20, 40, 80 with 90
# clients, and with 60 clients in the same session as a control. At 60 clients
# the server was only 61-86% busy at these widths.
# Labels g5hp32kro-rr10simd-c90-pgvectorfast-tq4 (both client counts; kept out
# of the chart's series). Results into bench/profiles/hn-g5-c90/.
#
#   caffeinate -i bench/cloudlab/hn_g5_c90.sh
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
CLIENTS=90
RR=100
T="hng5-c90-r$RUN"
LABEL=g5hp32kro-rr10simd-c90-pgvectorfast-tq4
LOW_EF=10,20,40,80
CHECKSUM=251cc2ee3b2c0245     # tq_selftest on the codec before this change
D32=/mydata/vectordb_experiments_32k
NTQ32="$D32/pgvectorfast_ntq/vectorfast.so"
TABLE=items_hn_pipnn_ro
TQ_ARGS="--index hnsw --table $TABLE --m \$PIPNN_M --ef-construction 64 --quantizer tq4 --rerank ntq12 --set hnsw.rerank_limit=$RR"
PD=bench/profiles/hn-g5-c90

ENVS8="$ENVS"
ENVS32="${ENVS/cd $DEST /cd $D32 }"
[ "$ENVS32" != "$ENVS8" ] || { echo "error: could not point ENVS at $D32" >&2; exit 1; }
on32() { local save="$ENVS"; ENVS="$ENVS32"; "$@"; ENVS="$save"; }
NOPG='if pgrep -x postgres >/dev/null; then echo "error: a Postgres is running"; exit 1; fi'

require_measurement_state

# ---- 1. Huge pages; 90 and 60 clients at low ef_search -------------------------------------------------
step "$CLIENT" "$T-client-idle" \
  'if pgrep -x ginger_bench >/dev/null; then echo "error: a ginger_bench is running"; exit 1; fi; echo idle'
ssh_to "$SERVER" "if [ -f $STEPS/nic-$NIC.pid ] && kill -0 \$(cat $STEPS/nic-$NIC.pid) 2>/dev/null; then echo 'nic monitor running'; else nohup sh -c 'while :; do echo \"\$(date +%s) \$(cat /sys/class/net/$NIC/carrier 2>/dev/null) \$(cat /sys/class/net/$NIC/statistics/rx_crc_errors) \$(cat /sys/class/net/$NIC/statistics/rx_packets)\"; sleep 2; done' >> $STEPS/$T-nic.txt 2>&1 < /dev/null & echo \$! > $STEPS/nic-$NIC.pid; echo nic monitor started; fi"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "if [ -f $STEPS/$T-cpu.pid ] && kill -0 \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; then echo 'cpu sampler running'; else nohup sh -c 'while :; do echo \"\$(date +%s) \$(head -1 /proc/stat)\"; sleep 2; done' >> $STEPS/$T-cpu.txt 2>&1 < /dev/null & echo \$! > $STEPS/$T-cpu.pid; echo cpu sampler started; fi"
done

on32 step "$SERVER" "$T-hp-reserve" "$HP_RESERVE"
on32 step "$SERVER" "$T-hp-serve" \
  "EXTRA_PG_OPTS='-c huge_pages=on' FAST_SO=$NTQ32 SERVE_ARM=fast SERVE_TABLE=$TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
# shellcheck disable=SC2016  # expanded on the node
on32 step "$SERVER" "$T-hp-check" 'HP_MIN_GB=50; '"$HP_CHECK"'
  b=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW block_size")
  echo "block_size: $b"; [ "$b" = 32768 ] || { echo "error: not a 32 KB cluster"; exit 1; }
  r=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW hnsw.rerank_kernel")
  echo "hnsw.rerank_kernel default: $r"; [ "$r" = auto ] || { echo "error: the served library has no hnsw.rerank_kernel"; exit 1; }
  r=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SET hnsw.rerank_kernel = avx512" -c "SHOW hnsw.rerank_kernel" | tail -1)
  echo "hnsw.rerank_kernel after SET avx512: $r"; [ "$r" = avx512 ] || { echo "error: the served library has no AVX-512 kernel"; exit 1; }'
bench_step "$CLIENT" "$T-tq4-c$CLIENTS" "$LABEL" "$TQ_ARGS --sweep $LOW_EF --clients $CLIENTS"
bench_step "$CLIENT" "$T-tq4-c60" "$LABEL" "$TQ_ARGS --sweep $LOW_EF --clients 60"
on32 step "$SERVER" "$T-hp-release" "$HP_RELEASE"

# ---- 2. Results ---------------------------------------------------------------------
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "kill \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; rm -f $STEPS/$T-cpu.pid; true"
done
ssh_to "$SERVER" "kill \$(cat $STEPS/nic-$NIC.pid) 2>/dev/null; rm -f $STEPS/nic-$NIC.pid; true"
LPD="$HERE/../../$PD"
mkdir -p "$LPD/sat"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-cpu.txt" "$LPD/server-cpu.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$CLIENT.$DOMAIN:$STEPS/$T-cpu.txt" "$LPD/client-cpu.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-nic.txt" "$LPD/nic.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-*.log" "$LPD/"
ssh_to "$CLIENT" "grep -E '\"label\": \"$LABEL\"' $DEST/bench/results/hackernews_minilm_384.jsonl" > "$LPD/results.jsonl"
python3 - "$LPD/results.jsonl" "$CLIENTS" > "$LPD/sat/results.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    if json.loads(line).get("clients") > 1:
        print(line, end="")
PY
cp "$LPD/server-cpu.txt" "$LPD/client-cpu.txt" "$LPD/nic.txt" "$LPD/sat/"
python3 "$HERE/saturation_report.py" "$LPD/sat" | tee "$LPD/saturation.md"
say "done; label $LABEL in $PD/results.jsonl"
