# Shared by the workstation-side drivers (remote_client.sh, scaling.sh). Source it.
#
# A driver runs on the workstation, which can reach both nodes, but every long
# step runs detached on its node (nohup) and writes /mydata/steps/<name>.done
# with its exit status. The driver only starts steps and waits for markers, so
# a workstation that sleeps or drops its connection cannot leave a measurement
# half-done or a driver hung on a dead SSH session; it just stops advancing.
# Re-running a driver skips every step whose marker says it succeeded.

SERVER="${SERVER:-ms0931}"
CLIENT="${CLIENT:-ms0905}"
SERVER_IP="${SERVER_IP:-10.10.1.1}"
CLIENT_NET="${CLIENT_NET:-10.10.1.0/24}"
KEY="${KEY:-$HOME/.ssh/cloudlab}"
REMOTE_USER="${REMOTE_USER:-awitt}"
DEST=/mydata/vectordb_experiments
STEPS=/mydata/steps
PORT=5433
ENVS="cd $DEST && set -a && . bench/cloudlab/dbpedia.env && set +a && export PGPORT=$PORT"
REMOTE_DSN="--dsn 'host=$SERVER_IP port=$PORT user=postgres dbname=postgres'"

ssh_to() {
  local node="$1"; shift
  ssh -i "$KEY" -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
    -o ConnectTimeout=20 -n "$REMOTE_USER@$node.utah.cloudlab.us" "$@"
}
say() { printf '\n=== %s  (%s)\n' "$*" "$(date +%H:%M:%S)" >&2; }

step_done() {  # step_done <node> <name>: true if the step's marker says it succeeded
  [ "$(ssh_to "$1" "cat $STEPS/$2.done 2>/dev/null" || true)" = 0 ]
}

# start_step <node> <name> <command>: launch detached in the repo with the
# DBpedia settings loaded, and return at once. Pair with wait_step.
start_step() {
  local node="$1" name="$2" cmd="$3"
  say "$name on $node"
  # Only the redirected nohup goes to the background. Backgrounding an
  # `a && b && nohup ...` chain would keep SSH's output channel open in the
  # background subshell, and ssh would not return until the step finished.
  ssh_to "$node" "mkdir -p $STEPS && rm -f $STEPS/$name.done; nohup bash -c $(printf '%q' "$ENVS && $cmd; echo \$? > $STEPS/$name.done") > $STEPS/$name.log 2>&1 < /dev/null &"
}

# wait_step <node> <name>: wait for the marker, print result rows, fail on error.
wait_step() {
  local node="$1" name="$2" st
  until st="$(ssh_to "$node" "cat $STEPS/$name.done 2>/dev/null" 2>/dev/null)" && [ -n "$st" ]; do sleep 15; done
  ssh_to "$node" "grep -avE '^\s*[0-9.]*%' $STEPS/$name.log | grep -aE '^ +[0-9-]+ +[0-9.-]+ |^error' || true" >&2 || true
  [ "$st" = 0 ] || { echo "error: step $name failed (exit $st); log: $node:$STEPS/$name.log" >&2; exit 1; }
}

# step <node> <name> <command>: start_step + wait_step, skipped if already done.
step() {
  if step_done "$1" "$2"; then
    say "$2: already done, skipping"
    return
  fi
  start_step "$@"
  wait_step "$1" "$2"
}

# serve_step <name> <arm> [table]: bring node-0's server up as one arm for LAN clients.
serve_step() {
  step "$SERVER" "$1" \
    "SERVE_ARM=$2 ${3:+SERVE_TABLE=$3} LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
}

# bench_step <node> <name> <label> <ann_bench args...>: one ann_bench run; a
# step on the client node reaches the server over the LAN.
bench_step() {
  local node="$1" name="$2" label="$3"; shift 3
  local dsn=""
  [ "$node" = "$CLIENT" ] && dsn="$REMOTE_DSN"
  step "$node" "$name" "BENCH_LABEL=$label bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" $* --num-queries \"\$NUM_QUERIES\" --duration \"\$DURATION\" $dsn"
}

require_measurement_state() {
  local n
  for n in "$SERVER" "$CLIENT"; do
    ssh_to "$n" "$ENVS && bench/cloudlab/pin_cpu.sh check && [ -z \"\$(tail -n +2 /proc/swaps)\" ]" ||
      { echo "error: $n is not pinned or has swap on; run bench/cloudlab/setup_node.sh there" >&2; exit 1; }
  done
}

# shellcheck disable=SC2016  # expanded on the node, from dbpedia.env
HNSW_ARGS='--index hnsw --m 16 --ef-construction 64 --sweep "$PVF_EF_SWEEP"'
