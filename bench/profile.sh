#!/usr/bin/env bash
# Profile Postgres with perf while ann_bench.py runs, and render a flamegraph.
#
# Usage: bench/profile.sh [ann_bench.py args...]
#   e.g. bench/profile.sh --dataset sift-128-euclidean --index hnsw --sweep 40 --num-queries 10000
#
# Steps:
#   1. Run the benchmark once untimed with a few queries so the table is loaded and
#      the index is built (both are reused afterwards).
#   2. pg_prewarm the benchmark tables and indexes, then check with pg_buffercache that
#      every block is in shared_buffers (fails otherwise).
#   3. Attach `perf record` to the postmaster (new backends inherit it) and run the
#      benchmark. Only the server is profiled, not the Python client.
#   4. Fold the stacks and render bench/profiles/<name>.svg with Brendan Gregg's
#      FlameGraph scripts (fetched once into bench/.tools/ at a pinned commit).
#
# Needs: kernel.perf_event_paranoid <= 1 (and kernel.kptr_restrict=0 for kernel symbols),
# and Postgres + pgvector built with FRAME_POINTERS=1 ./build_postgres.sh. perf's DWARF
# unwinding only recovered ~25% of backend stacks here, so frame pointers are the default.
# Env: PGPORT (default 5432), PERF_FREQ (default 999 Hz), NAME (output name),
#      PERF_CALLGRAPH (default fp; e.g. dwarf,16384), FLAMEGRAPH=0 to record only.
set -euo pipefail

BENCH="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$BENCH")"
PY="$BENCH/.venv/bin/python"
PSQL=("$ROOT/postgres/install/bin/psql" -h localhost -p "${PGPORT:-5432}" -U postgres -d postgres -X -q -At)
DATA="$ROOT/postgres/data"
OUT="$BENCH/profiles"
TOOLS="$BENCH/.tools/FlameGraph"
FLAMEGRAPH_COMMIT=41fee1f99f9276008b7cd112fca19dc3ea84ac32
FREQ="${PERF_FREQ:-999}"
CALLGRAPH="${PERF_CALLGRAPH:-fp}"
NAME="${NAME:-profile-$(date +%Y%m%d-%H%M%S)}"

paranoid="$(cat /proc/sys/kernel/perf_event_paranoid)"
if [ "$paranoid" -gt 1 ]; then
  echo "error: kernel.perf_event_paranoid=$paranoid; run: sudo sysctl -w kernel.perf_event_paranoid=1 kernel.kptr_restrict=0" >&2
  exit 1
fi

if [ "${FLAMEGRAPH:-1}" = 1 ] && [ ! -d "$TOOLS" ]; then
  git clone -q https://github.com/brendangregg/FlameGraph.git "$TOOLS"
  git -C "$TOOLS" checkout -q "$FLAMEGRAPH_COMMIT"
fi

# 1. Load the table and build the index if needed (small, unsaved run)
echo "== setup run" >&2
"$PY" "$BENCH/ann_bench.py" "$@" --num-queries 10 --no-save >/dev/null

# 2. Residency check: every block of the benchmark table and its indexes in shared_buffers
residency() {
  "${PSQL[@]}" -F $'\t' <<'SQL'
SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS pg_buffercache;
SELECT c.relname,
       pg_relation_size(c.oid) / current_setting('block_size')::int AS blocks,
       count(b.bufferid) AS cached
FROM pg_class c
LEFT JOIN pg_buffercache b
  ON b.relfilenode = pg_relation_filenode(c.oid)
 AND b.reldatabase = (SELECT oid FROM pg_database WHERE datname = current_database())
 AND b.relforknumber = 0
WHERE c.relname LIKE 'items\_%' AND c.relkind IN ('r', 'i')
  AND pg_relation_size(c.oid) > 0
GROUP BY c.oid, c.relname
ORDER BY c.relname;
SQL
}
check_resident() {
  echo "== residency ($1): relation, blocks, cached in shared_buffers" >&2
  residency | tee -a "$OUT/$NAME.residency.txt" | column -t >&2
}
mkdir -p "$OUT"
: >"$OUT/$NAME.residency.txt"
echo "shared_buffers=$("${PSQL[@]}" -c 'SHOW shared_buffers')" >>"$OUT/$NAME.residency.txt"

# Load every benchmark table and index into shared_buffers, so the profile shows
# query CPU time rather than buffer misses (e.g. right after a server restart)
"${PSQL[@]}" >/dev/null <<'SQL'
CREATE EXTENSION IF NOT EXISTS pg_prewarm;
SELECT pg_prewarm(c.oid) FROM pg_class c
WHERE c.relname LIKE 'items\_%' AND c.relkind IN ('r', 'i');
SQL
check_resident before
if residency | awk -F '\t' '$2 != $3 {bad=1} END {exit !bad}'; then
  echo "error: not everything fits in shared_buffers; raise it (PG_OPTS=\"-c shared_buffers=...\")" >&2
  exit 1
fi

# 3. Profile the server while the benchmark runs
POSTMASTER="$(head -1 "$DATA/postmaster.pid")"
echo "== perf record -p $POSTMASTER at ${FREQ} Hz" >&2
perf record -F "$FREQ" --call-graph "$CALLGRAPH" -p "$POSTMASTER" -o "$OUT/$NAME.perf.data" 2>"$OUT/$NAME.perf.log" &
PERF=$!
sleep 1  # let perf attach before the benchmark connects
"$PY" "$BENCH/ann_bench.py" "$@" | tee "$OUT/$NAME.bench.txt"
kill -INT "$PERF"
wait "$PERF" || true
check_resident after

# 4. Flamegraph (FLAMEGRAPH=0 skips it; render later from the .perf.data)
if [ "${FLAMEGRAPH:-1}" != 1 ]; then
  echo "perf data: $OUT/$NAME.perf.data (flamegraph skipped)" >&2
  exit 0
fi
echo "== rendering flamegraph" >&2
perf script -i "$OUT/$NAME.perf.data" 2>/dev/null \
  | "$TOOLS/stackcollapse-perf.pl" >"$OUT/$NAME.folded"
"$TOOLS/flamegraph.pl" --title "postgres: ann_bench.py $*" --width 1600 \
  "$OUT/$NAME.folded" >"$OUT/$NAME.svg"
echo "flamegraph: $OUT/$NAME.svg" >&2
