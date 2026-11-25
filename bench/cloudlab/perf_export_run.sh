#!/usr/bin/env bash
# Run perf_export.sh on both nodes (detached steps), then pull back only the
# text exports (<name>.export/ directories), never the raw *.data files.
#
#   caffeinate -i bench/cloudlab/perf_export_run.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
T="perfexp-r$RUN"
for n in "$SERVER" "$CLIENT"; do
  ( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" bench/cloudlab/perf_export.sh "$REMOTE_USER@$n.$DOMAIN:$DEST/" )
done
for n in "$SERVER" "$CLIENT"; do
  step_done "$n" "$T" || start_step "$n" "$T" "bench/cloudlab/perf_export.sh bench/profiles"
done
for n in "$SERVER" "$CLIENT"; do
  wait_step "$n" "$T"
  rsync -az -m --include '*/' --include '*.export/**' --exclude '*' -e "ssh ${SSH_OPTS[*]}" \
    "$REMOTE_USER@$n.$DOMAIN:$DEST/bench/profiles/" "$HERE/../profiles/"
  ssh_to "$n" "cp $STEPS/$T.log $DEST/bench/profiles/perf_export-$n.log 2>/dev/null; cat $STEPS/$T.log" > "$HERE/../profiles/perf_export-$n.log"
done
say "done; exports in bench/profiles/*/*.export/"
