#!/usr/bin/env bash
# Push this working tree (submodules included) to the CloudLab nodes, or pull
# their results back.
#
#   bench/cloudlab/sync.sh push            # workstation -> both nodes
#   bench/cloudlab/sync.sh pull            # both nodes' bench/results -> bench/results/<host>/
#
# Env: NODES (default "$SERVER $CLIENT"), DOMAIN (default utah.cloudlab.us), KEY (default ~/.ssh/cloudlab),
#      REMOTE_USER (default awitt), DEST (default /mydata/vectordb_experiments)
#
# rsync rather than git clone: the code under test is usually uncommitted, and
# the submodules carry local branches that are not pushed anywhere.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NODES="${NODES:-${SERVER:-ms0818} ${CLIENT:-ms0841}}"
KEY="${KEY:-$HOME/.ssh/cloudlab}"
REMOTE_USER="${REMOTE_USER:-awitt}"
DEST="${DEST:-/mydata/vectordb_experiments}"
DOMAIN="${DOMAIN:-utah.cloudlab.us}"
SSH="ssh -i $KEY -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%C -o ControlPersist=30m"

case "${1:-}" in
  push)
    for n in $NODES; do
      host="$REMOTE_USER@$n.$DOMAIN"
      echo "=== push -> $n:$DEST" >&2
      # Nothing that is built, downloaded or measured on the node goes over.
      rsync -az --delete -e "$SSH" \
        --exclude '.DS_Store' \
        --exclude '/postgres/' \
        --exclude '/bench/.venv/' --exclude '/bench/data/' \
        --exclude '/bench/results/' --exclude '/bench/profiles/' \
        --exclude '/bench/.tools/' --exclude '__pycache__/' \
        --exclude '/bench/faiss/build/' --exclude '/bench/ginger/build/' \
        --exclude '/faiss/build/' --exclude '/faiss/install/' \
        --exclude '/pg_vectorsearch/builddir/' \
        --exclude '*.o' --exclude '*.so' --exclude '*.bc' --exclude '*.dylib' \
        "$ROOT/" "$host:$DEST/"
    done
    ;;
  pull)
    for n in $NODES; do
      host="$REMOTE_USER@$n.$DOMAIN"
      echo "=== pull <- $n:$DEST/bench/results" >&2
      mkdir -p "$ROOT/bench/results/$n"
      rsync -az -e "$SSH" "$host:$DEST/bench/results/" "$ROOT/bench/results/$n/"
    done
    ;;
  *)
    echo "usage: $0 push|pull" >&2
    exit 1
    ;;
esac
