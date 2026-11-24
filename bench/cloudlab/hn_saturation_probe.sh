#!/usr/bin/env bash
# How many LAN clients does it take to keep the server busy? pgvectorfast on
# the PiPNN graph (pippipnn-fork's library and index), at the ef_search that
# gives recall@10 ~0.95, with 40, 60 and 80 clients. Each node samples
# /proc/stat every 2 s; saturation_report.py gives QPS and the busy share of
# both nodes over each point's measurement window. Run with Postgres stopped:
#
#   caffeinate -i bench/cloudlab/hn_saturation_probe.sh
#
# Steps are resumable (see lib.sh); bump RUN to redo them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
T="hnsat-r$RUN"
EF="${EF:-64}"
COUNTS="${COUNTS:-40 60 80}"
LABEL=satprobe-pipnn
PD="$HERE/../profiles/hn-saturation"
JSONL=hackernews_minilm_384.jsonl
PIPNN_ARGS="--table \$PIPNN_TABLE --m \$PIPNN_M --ef-construction 64"

for n in "$SERVER" "$CLIENT"; do
  ( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
      bench/run_experiment.sh bench/cloudlab/hn.env "$REMOTE_USER@$n.$DOMAIN:$DEST/" )
done

require_measurement_state

step "$CLIENT" "$T-client-idle" \
  'if pgrep -x ginger_bench >/dev/null; then echo "error: a ginger_bench is running"; exit 1; fi; echo idle'
step "$SERVER" "$T-serve" \
  "FAST_SO=\$PIPNN_SO SERVE_ARM=fast SERVE_TABLE=\$PIPNN_TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"

# CPU samplers: "<epoch> cpu <user> <nice> <system> <idle> <iowait> ..." every 2 s
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "[ -f $STEPS/$T-cpu.pid ] && kill \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; nohup sh -c 'while :; do echo \"\$(date +%s) \$(head -1 /proc/stat)\"; sleep 2; done' > $STEPS/$T-cpu.txt 2>&1 < /dev/null & echo \$! > $STEPS/$T-cpu.pid; echo sampler \$(cat $STEPS/$T-cpu.pid) on \$(nproc) CPUs"
done

for c in $COUNTS; do
  bench_step "$CLIENT" "$T-c$c" "$LABEL" "--index hnsw $PIPNN_ARGS --sweep $EF --clients $c"
done

for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "kill \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; rm -f $STEPS/$T-cpu.pid; true"
done
step "$SERVER" "$T-stop" "./run_postgres.sh stop || true"

mkdir -p "$PD"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-cpu.txt" "$PD/server-cpu.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$CLIENT.$DOMAIN:$STEPS/$T-cpu.txt" "$PD/client-cpu.txt"
ssh_to "$CLIENT" "grep -F '$LABEL' $DEST/bench/results/$JSONL" > "$PD/results.jsonl"
python3 "$HERE/saturation_report.py" "$PD" | tee "$PD/report.md"
say "done; $PD/report.md"
