#!/usr/bin/env bash
# How much work pgvectorfast + tq4 does per query, by ef_search: function-entry
# counts from uprobes on the served vectorfast.so while a local one-client
# benchmark runs (on the server node itself, so this is one detached step).
#
#   PROFILE_DIR=hn-ginger bench/cloudlab/pg_counts.sh 10,20,40,80,160,320,640,1000
#
# Writes bench/profiles/$PROFILE_DIR/<PG_COUNTS_NAME>-ef<ef>.csv per ef_search
# (default name pgcounts). uprobes
# trap into the kernel on every hit, so queries run much slower here; only the
# per-query ratios are used, never the speed:
#   hnswrescan                    one per query
#   HnswLoadUnvisitedRaw          one per layer-0 hop (node expanded)
#   HnswTqCodeDistanceOptimistic  one per tq4 navigation distance
#   CalicoLookupAndPin            two per re-ranked candidate
# PG_COUNTS_PROBES replaces that list (hnswrescan first: it counts queries), and
# PG_COUNTS_ARGS the ann_bench index arguments (default: the tq4 table).
# PG_COUNTS_SUM="<file>:<line> <variable>" adds a line probe that records the
# variable on each hit; its per-query sum is reported as "sum" (for example
# the neighbours scored per hop, where the distance code is inlined and has no
# function to count). Written to <name>-ef<ef>.sum.txt.
# Needs the server up (any arm built from pgvectorfast_ntq), the bench venv,
# and passwordless sudo for perf probe.
set -euo pipefail

SWEEP="${1:-10,20,40,80,160,320,640,1000}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/bench/profiles/${PROFILE_DIR:-ginger}"
LIB="$ROOT/postgres/install/lib/x86_64-linux-gnu/vectorfast.so"
mkdir -p "$OUT"
cleanup() { sudo -n perf probe -q -d 'probe_vectorfast:*' >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

probes=()
NAME="${PG_COUNTS_NAME:-pgcounts}"
# shellcheck disable=SC2016  # NTQ_* expand here, from hn.env
IARGS="${PG_COUNTS_ARGS:---m 16 --ef-construction 64 --table $NTQ_TABLE $NTQ_ARGS}"
for f in ${PG_COUNTS_PROBES:-hnswrescan HnswLoadUnvisitedRaw HnswTqCodeDistanceOptimistic CalicoLookupAndPin}; do
  if sudo -n perf probe -q -x "$LIB" -a "$f" >/dev/null; then probes+=(-e "probe_vectorfast:$f")
  else echo "no probe for $f in $LIB; not counted" >&2; fi
done
[ "${#probes[@]}" -gt 0 ] || { echo "error: no probes could be placed" >&2; exit 1; }
if [ -n "${PG_COUNTS_SUM:-}" ]; then
  sudo -n perf probe -q -x "$LIB" -a "pgsum=$PG_COUNTS_SUM" >/dev/null ||
    { echo "error: cannot place the line probe $PG_COUNTS_SUM" >&2; exit 1; }
fi

cd "$ROOT"
for ef in ${SWEEP//,/ }; do
  # A 60 s run; the probes count 20 s of it after 15 s of warm-up. Not saved:
  # the probes slow every query, so its QPS means nothing.
  # shellcheck disable=SC2086  # IARGS is split on purpose
  BENCH_LABEL="$NAME" bench/.venv/bin/python bench/ann_bench.py --dataset "$DATASET" \
    --index hnsw $IARGS --sweep "$ef" --clients 1 --num-queries "$NUM_QUERIES" --duration 60 --no-save \
    > "$OUT/$NAME-ef$ef.client.log" 2>&1 &
  bench_pid=$!
  sleep 15
  rec=""
  if [ -n "${PG_COUNTS_SUM:-}" ]; then
    sudo -n perf record -q -a -e probe_vectorfast:pgsum -o "$OUT/$NAME-ef$ef.sum.data" -- sleep 20 2>/dev/null &
    rec=$!
  fi
  sudo -n perf stat -a -x, -o "$OUT/$NAME-ef$ef.csv" "${probes[@]}" -- sleep 20
  sudo -n chown "$(id -u):$(id -g)" "$OUT/$NAME-ef$ef.csv"
  if [ -n "$rec" ]; then
    wait "$rec" || true
    var="${PG_COUNTS_SUM##* }"
    sudo -n perf script -i "$OUT/$NAME-ef$ef.sum.data" 2>/dev/null |
      grep -o "$var=[-0-9]*" | awk -F= '{s += $2; n++} END {printf "hits=%d sum=%d\n", n, s}' > "$OUT/$NAME-ef$ef.sum.txt"
    sudo -n rm -f "$OUT/$NAME-ef$ef.sum.data"
  fi
  wait "$bench_pid"
  awk -F, -v ef="$ef" '$3 ~ /probe_vectorfast/ {split($3, a, ":"); n[a[2]] = $1; order[++k] = a[2]}
    END {q = n["hnswrescan"]; if (q < 1) q = 1;
         printf "ef %s: %d queries; per query:", ef, q;
         for (i = 1; i <= k; i++) if (order[i] != "hnswrescan") printf " %s %.1f", order[i], n[order[i]] / q;
         if ((getline line < sumf) > 0) { split(line, a, /[ =]/); printf " sum %.1f (over %d hits)", a[4] / q, a[2] }
         printf "\n"}' sumf="$OUT/$NAME-ef$ef.sum.txt" \
    "$OUT/$NAME-ef$ef.csv"
done
