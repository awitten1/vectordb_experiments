# Shared by the workstation-side drivers (remote_client.sh, scaling.sh). Source it.
#
# A driver runs on the workstation, which can reach both nodes, but every long
# step runs detached on its node (nohup) and writes /mydata/steps/<name>.done
# with its exit status. The driver only starts steps and waits for markers, so
# a workstation that sleeps or drops its connection cannot leave a measurement
# half-done or a driver hung on a dead SSH session; it just stops advancing.
# Re-running a driver skips every step whose marker says it succeeded.

SERVER="${SERVER:-ms0818}"
CLIENT="${CLIENT:-ms0841}"
SERVER_IP="${SERVER_IP:-10.10.1.1}"
CLIENT_NET="${CLIENT_NET:-10.10.1.0/24}"
KEY="${KEY:-$HOME/.ssh/cloudlab}"
REMOTE_USER="${REMOTE_USER:-awitt}"
# CloudLab cluster domain: utah.cloudlab.us for the m510s, wisc.cloudlab.us for c220g2.
DOMAIN="${DOMAIN:-utah.cloudlab.us}"
DEST=/mydata/vectordb_experiments
STEPS=/mydata/steps
PORT=5433
# Settings file loaded for every step: dbpedia.env, or sift.env for the SIFT study.
ENV_FILE="${ENV_FILE:-bench/cloudlab/dbpedia.env}"
# SMOKE is passed through, so a driver run with SMOKE=1 gets the env file's smoke settings.
ENVS="cd $DEST && export SMOKE=${SMOKE:-0} && set -a && . $ENV_FILE && set +a && export PGPORT=$PORT"
REMOTE_DSN="--dsn 'host=$SERVER_IP port=$PORT user=postgres dbname=postgres'"

# Every command to a node rides one multiplexed SSH connection (ControlMaster),
# so polling a step's marker opens a channel, not a new TCP connection and
# login: many logins in a short time can get the client blocked.
SSH_OPTS=(-i "$KEY" -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4
          -o ConnectTimeout=20 -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-%C"
          -o ControlPersist=30m)
ssh_to() {
  local node="$1"; shift
  ssh "${SSH_OPTS[@]}" -n "$REMOTE_USER@$node.$DOMAIN" "$@"
}
say() { printf '\n=== %s  (%s)\n' "$*" "$(date +%H:%M:%S)" >&2; }

# Not answering: after this many consecutive failed SSH attempts (one retry,
# a minute after the first failure), a driver gives up and exits instead of
# knocking on an unresponsive node; repeated logins can get the client blocked.
MAX_SSH_FAILURES="${MAX_SSH_FAILURES:-2}"

give_up() {  # give_up <node>
  echo "error: $1 is not answering; stopping rather than keep retrying (steps already" >&2
  echo "       launched there keep running; re-run the driver once the node is back)" >&2
  exit 1
}

step_done() {  # step_done <node> <name>: true if the step's marker says it succeeded
  local st
  # An SSH failure must not read as "not done": the caller would relaunch the step.
  st="$(ssh_to "$1" "cat $STEPS/$2.done 2>/dev/null; true")" || give_up "$1"
  [ "$st" = 0 ]
}

# start_step <node> <name> <command>: launch detached in the repo with the
# DBpedia settings loaded, and return at once. Pair with wait_step.
start_step() {
  local node="$1" name="$2" cmd="$3"
  say "$name on $node"
  # Only the redirected nohup goes to the background. Backgrounding an
  # `a && b && nohup ...` chain would keep SSH's output channel open in the
  # background subshell, and ssh would not return until the step finished.
  # The command runs in a subshell so that an `exit` in it still leaves the
  # marker; otherwise wait_step would poll for a marker that never comes.
  ssh_to "$node" "mkdir -p $STEPS && rm -f $STEPS/$name.done; nohup bash -c $(printf '%q' "( $ENVS && $cmd ); echo \$? > $STEPS/$name.done") > $STEPS/$name.log 2>&1 < /dev/null &"
}

# wait_step <node> <name>: wait for the marker, print result rows, fail on error.
wait_step() {
  local node="$1" name="$2" st
  # Poll once a minute. If the node stops answering, retry once a minute
  # later, then give up (MAX_SSH_FAILURES failures in a row).
  local fails=0 pause
  while :; do
    # `; true`: a missing marker (step still running) is not an SSH failure.
    if st="$(ssh_to "$node" "cat $STEPS/$name.done 2>/dev/null; true")"; then
      [ -n "$st" ] && break
      fails=0
      pause=60
    else
      fails=$(( fails + 1 ))
      [ "$fails" -ge "$MAX_SSH_FAILURES" ] && give_up "$node"
      pause=$(( 60 << (fails - 1) ))
      echo "  ($node not answering, $fails of $MAX_SSH_FAILURES; next try in ${pause}s)" >&2
    fi
    sleep "$pause"
  done
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
