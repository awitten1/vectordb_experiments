#!/usr/bin/env bash
# Re-run the hn_saturated.sh sweeps that the server's 10 GbE link drops hit
# on 2026-11-23, after hn_saturated.sh has finished and put the float PiPNN
# index back. ARMS picks them, from pvf-hnsw, pgv-hnsw, pvf-pipnn and
# pgv-pipnn (the tq4 arm needs its own import; not here). Run with Postgres
# stopped:
#
#   CLIENTS=60 ARMS="pvf-hnsw pgv-pipnn" caffeinate -i bench/cloudlab/hn_saturated_rerun.sh
#
# Before each sweep the server must have had its link up, with no new receive
# CRC errors, for LINK_CLEAN_S (default 600 s); a step waits up to an hour for
# that, then fails. The link monitor (nic-enp6s0f0.txt: "<epoch> <carrier>
# <rx_crc_errors> <rx_packets>" every 2 s) goes into the report directory with
# the CPU samples, so saturation_report.py can flag every point the link
# touched. Labels sat<N>r-*.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
CLIENTS="${CLIENTS:-60}"
ARMS="${ARMS:-pvf-hnsw pgv-pipnn}"
T="hnsat${CLIENTS}rr-r$RUN"
L="sat${CLIENTS}r"
PD="$HERE/../profiles/hn-saturated-c$CLIENTS-rerun"
JSONL=hackernews_minilm_384.jsonl
NIC=enp6s0f0
NICLOG="$STEPS/nic-$NIC.txt"
LINK_CLEAN_S="${LINK_CLEAN_S:-600}"
HNSW_ARGS="--m 16 --ef-construction 64"
PIPNN_ARGS="--table \$PIPNN_TABLE --m \$PIPNN_M --ef-construction 64"

for n in "$SERVER" "$CLIENT"; do
  ( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
      bench/run_experiment.sh bench/cloudlab/hn.env "$REMOTE_USER@$n.$DOMAIN:$DEST/" )
done

require_measurement_state

step "$CLIENT" "$T-client-idle" \
  'if pgrep -x ginger_bench >/dev/null; then echo "error: a ginger_bench is running"; exit 1; fi; echo idle'

# The link monitor (started by hand on 2026-11-23); start it if it isn't running
ssh_to "$SERVER" "if [ -f $STEPS/nic-$NIC.pid ] && kill -0 \$(cat $STEPS/nic-$NIC.pid) 2>/dev/null; then echo 'nic monitor running'; else nohup sh -c 'while :; do echo \"\$(date +%s) \$(cat /sys/class/net/$NIC/carrier 2>/dev/null) \$(cat /sys/class/net/$NIC/statistics/rx_crc_errors) \$(cat /sys/class/net/$NIC/statistics/rx_packets)\"; sleep 2; done' >> $NICLOG 2>&1 < /dev/null & echo \$! > $STEPS/nic-$NIC.pid; echo nic monitor started \$(cat $STEPS/nic-$NIC.pid); fi"

# CPU samplers on both nodes, for the whole run
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "[ -f $STEPS/$T-cpu.pid ] && kill \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; nohup sh -c 'while :; do echo \"\$(date +%s) \$(head -1 /proc/stat)\"; sleep 2; done' >> $STEPS/$T-cpu.txt 2>&1 < /dev/null & echo \$! > $STEPS/$T-cpu.pid; echo sampler \$(cat $STEPS/$T-cpu.pid)"
done

# link_clean <name>: wait (up to an hour) until the link has been up with no
# new CRC errors for LINK_CLEAN_S
link_clean() {
  # shellcheck disable=SC2016  # expanded on the node
  step "$SERVER" "$T-link-$1" '
    for i in $(seq 360); do
      now=$(date +%s)
      bad=$(awk -v t=$((now - '"$LINK_CLEAN_S"')) "\$1 >= t { if (\$2 != 1) d++; if (n++ && \$3 != c) e++; c = \$3 } END { print (n < '"$LINK_CLEAN_S"' / 2 - 5) ? \"short\" : d + e }" '"$NICLOG"')
      if [ "$bad" = 0 ]; then echo "link clean for '"$LINK_CLEAN_S"' s at $(date +%T)"; exit 0; fi
      [ "$i" = 1 ] && echo "waiting for a clean link ($bad bad samples in the last '"$LINK_CLEAN_S"' s)"
      sleep 10
    done
    echo "error: the link was not clean for '"$LINK_CLEAN_S"' s within an hour"; exit 1'
}

# arm <name> <serve env> <serve arm> <label> <ann_bench index args>
arm() {
  step "$SERVER" "$T-serve-$1" \
    "$2 SERVE_ARM=$3 LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
  link_clean "$1"
  bench_step "$CLIENT" "$T-$1" "$4" "--index hnsw $5 --sweep \"\$EF_SWEEP\" --clients $CLIENTS"
}
for a in $ARMS; do
  case "$a" in
    pgv-hnsw)  arm pgv-hnsw  ''                                          stock "$L-pgvector-hnsw"      "$HNSW_ARGS" ;;
    pvf-hnsw)  arm pvf-hnsw  'FAST_SO=$PIPNN_SO'                         fast  "$L-pgvectorfast-hnsw"  "$HNSW_ARGS" ;;
    pgv-pipnn) arm pgv-pipnn 'SERVE_TABLE=$PIPNN_TABLE'                  stock "$L-pgvector-pipnn"     "$PIPNN_ARGS" ;;
    pvf-pipnn) arm pvf-pipnn 'FAST_SO=$PIPNN_SO SERVE_TABLE=$PIPNN_TABLE' fast  "$L-pgvectorfast-pipnn" "$PIPNN_ARGS" ;;
    *) echo "error: unknown arm $a" >&2; exit 1 ;;
  esac
done

for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "kill \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; rm -f $STEPS/$T-cpu.pid; true"
done
step "$SERVER" "$T-stop" "./run_postgres.sh stop || true"

mkdir -p "$PD"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-cpu.txt" "$PD/server-cpu.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$CLIENT.$DOMAIN:$STEPS/$T-cpu.txt" "$PD/client-cpu.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$NICLOG" "$PD/nic.txt"
ssh_to "$CLIENT" "grep -F '\"$L-' $DEST/bench/results/$JSONL" > "$PD/results.jsonl"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
python3 "$HERE/saturation_report.py" "$PD" | tee "$PD/report.md"
say "done; labels $L-* in $PD/results.jsonl; then bench/cloudlab/sync.sh pull"
