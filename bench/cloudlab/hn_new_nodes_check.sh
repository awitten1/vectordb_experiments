#!/usr/bin/env bash
# After moving the study to a new node pair (data rsynced across from the old
# pair, see nodes-wisc.sh), check that it arrived whole and measures the same:
# every relation's size on the server, then pgvectorfast on the imported PiPNN
# graph at 1 client over the new LAN, 5 s per point, whose recall must match
# pippipnn-fork's from the old pair (recall_gate.py). Also starts the server's
# link monitor (nic-enp6s0f0.txt, as hn_saturated_rerun.sh reads it) and
# reports the link's state. Run with Postgres stopped:
#
#   bench/cloudlab/hn_new_nodes_check.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
T="hnmove-r$RUN"
L="move-$SERVER"
NIC=enp6s0f0
NICLOG="$STEPS/nic-$NIC.txt"
PIPNN_ARGS="--table \$PIPNN_TABLE --m \$PIPNN_M --ef-construction 64"

for n in "$SERVER" "$CLIENT"; do
  ( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
      bench/run_experiment.sh bench/cloudlab/hn.env bench/cloudlab/recall_gate.py \
      "$REMOTE_USER@$n.$DOMAIN:$DEST/" )
done

require_measurement_state

ssh_to "$SERVER" "mkdir -p $STEPS; if [ -f $STEPS/nic-$NIC.pid ] && kill -0 \$(cat $STEPS/nic-$NIC.pid) 2>/dev/null; then echo 'nic monitor running'; else nohup sh -c 'while :; do echo \"\$(date +%s) \$(cat /sys/class/net/$NIC/carrier 2>/dev/null) \$(cat /sys/class/net/$NIC/statistics/rx_crc_errors) \$(cat /sys/class/net/$NIC/statistics/rx_packets)\"; sleep 2; done' >> $NICLOG 2>&1 < /dev/null & echo \$! > $STEPS/nic-$NIC.pid; echo nic monitor started \$(cat $STEPS/nic-$NIC.pid); fi"

step "$CLIENT" "$T-client-idle" \
  'if pgrep -x ginger_bench >/dev/null; then echo "error: a ginger_bench is running"; exit 1; fi; echo idle'
step "$SERVER" "$T-check" '
  [ -s "$PIPNN_GRAPH" ] || { echo "error: no graph file $PIPNN_GRAPH"; exit 1; }
  [ -f "$PIPNN_SO" ] || { echo "error: no $PIPNN_SO"; exit 1; }
  echo ok'

# Serving prewarms the PiPNN table and index, i.e. reads them all
step "$SERVER" "$T-serve" \
  "FAST_SO=\$PIPNN_SO SERVE_ARM=fast SERVE_TABLE=\$PIPNN_TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
# shellcheck disable=SC2016  # expanded on the node
step "$SERVER" "$T-sizes" '
  postgres/install/bin/psql -h localhost -U postgres -d postgres -X -c "
    select c.relname, c.relkind, pg_size_pretty(pg_relation_size(c.oid)) as size
    from pg_class c join pg_namespace n on n.oid = c.relnamespace
    where n.nspname = '"'"'public'"'"' and c.relkind in ('"'"'r'"'"', '"'"'i'"'"')
    order by c.relname"'
step "$CLIENT" "$T-c1" \
  "BENCH_LABEL=$L bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw $PIPNN_ARGS --sweep \"\$EF_SWEEP\" --clients 1 --num-queries \"\$NUM_QUERIES\" --duration 5 $REMOTE_DSN"
step "$CLIENT" "$T-gate" "bench/.venv/bin/python bench/cloudlab/recall_gate.py $L pippipnn-fork"
step "$SERVER" "$T-stop" "./run_postgres.sh stop || true"

ssh_to "$SERVER" "cat $STEPS/$T-sizes.log"
ssh_to "$CLIENT" "cat $STEPS/$T-gate.log"
ssh_to "$SERVER" "echo link: \$(cat /sys/class/net/$NIC/carrier) carrier, \$(cat /sys/class/net/$NIC/statistics/rx_crc_errors) CRC errors since boot; sudo ethtool -S $NIC | grep -E 'link|crc' || true"
say "done"
