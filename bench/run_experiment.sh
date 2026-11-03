#!/usr/bin/env bash
# Run the whole pgvector ANN experiment end to end and leave the numbers in
# bench/results/ (one JSON line per sweep point) plus bench/profiles/.
#
# Usage: bench/run_experiment.sh [phase ...]          (no phase = all of them)
#   phases: data build query concurrency profile ivfflat exact
#
# NOTE ON ORDER: ann_bench.py keeps at most one vector index per table, so the
# ivfflat phase DROPS the HNSW index. Everything that needs HNSW (query,
# concurrency, profile) therefore runs before ivfflat, and re-running `build`
# afterwards is what restores it.
#   e.g. bench/run_experiment.sh query concurrency
#
# Each phase is resumable and safe to re-run: the dataset, the table and the
# indexes are reused if they already exist (pass REBUILD=1 to force a rebuild,
# which is what you want if you are timing the build).
#
# Env knobs (defaults in the CONFIG block below):
#   DATASET      dataset spec, e.g. bigann:18000000 or synthetic:18000000x128
#   PGPORT       port for the experiment server (default 5433, so it does not
#                collide with a system Postgres on 5432)
#   M, EF_CONSTRUCTION, LISTS          index parameters
#   BUILD_MWM    maintenance_work_mem for CREATE INDEX (default 28GB)
#   QUERY_SB     shared_buffers for the query phases (default 28GB)
#   WORKERS      parallel workers for the build (default: nproc-1)
#   EF_SWEEP, PROBE_SWEEP, CLIENTS     sweep points
#   NUM_QUERIES                        size of the query set (<=10000 for bigann)
#   DURATION                           seconds per sweep point (default 60; 0 = one pass)
#   EXACT_QUERIES                      queries for the exact baseline
#   KEEP_RUNNING=1                     leave the server up at the end
#
# Why this script exists rather than a list of commands in the report: the two
# phases need *different* server configurations, and one of the settings is not
# a tuning preference but a correctness-of-measurement requirement. See the
# dsm_preflight and start_server functions.
set -euo pipefail

BENCH="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$BENCH")"
PY="$BENCH/.venv/bin/python"
BIN="$ROOT/postgres/install/bin"
PGDATA_DIR="$ROOT/postgres/data"

# ---- CONFIG -----------------------------------------------------------------
DATASET="${DATASET:-bigann:18000000}"
export PGPORT="${PGPORT:-5433}"
M="${M:-16}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-64}"
LISTS="${LISTS:-}"                       # empty = ann_bench default, sqrt(n)
BUILD_MWM="${BUILD_MWM:-28GB}"
QUERY_SB="${QUERY_SB:-28GB}"
BUILD_SB="${BUILD_SB:-2GB}"              # small: the build lives in the DSM arena
WORKERS="${WORKERS:-$(( $(nproc) - 1 ))}"
EF_SWEEP="${EF_SWEEP:-10,20,40,80,160,320}"
PROBE_SWEEP="${PROBE_SWEEP:-1,2,5,10,20,50,100}"
CLIENTS="${CLIENTS:-4,8}"
NUM_QUERIES="${NUM_QUERIES:-1000}"
EXACT_QUERIES="${EXACT_QUERIES:-30}"     # each one scans the whole table
# Seconds per sweep point. A single pass over the query set is too short to be a
# trustworthy throughput measurement once there are several clients: 10k queries
# at 8 clients is ~3s, so startup costs and one-off stalls do not average out.
DURATION="${DURATION:-60}"
PROFILE_EF="${PROFILE_EF:-40}"
LOG_DIR="${LOG_DIR:-$BENCH/results}"
# -----------------------------------------------------------------------------

PSQL=("$BIN/psql" -h localhost -p "$PGPORT" -U postgres -d postgres -X -q)
say() { printf '\n=== %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# ---- server control ---------------------------------------------------------

# A shutdown checkpoint after a multi-GB load can take a while, and starting
# again too early fails with "the database system is shutting down".
#
# pg_ctl stop waits only 60s by default and then exits non-zero while the
# shutdown carries on in the background, so under `set -e` it aborts the run
# just as the next phase is about to start. Measured here after the HNSW build:
# a 42s shutdown checkpoint flushing 62% of the buffer pool. So checkpoint
# first to shorten it, give pg_ctl a long timeout, and treat the status loop --
# not pg_ctl's exit code -- as the real answer.
stop_server() {
  if "$BIN/pg_ctl" -D "$PGDATA_DIR" status >/dev/null 2>&1; then
    "${PSQL[@]}" -c "CHECKPOINT" >/dev/null 2>&1 || true
    "$BIN/pg_ctl" -D "$PGDATA_DIR" -w -t "${STOP_TIMEOUT:-3600}" stop >/dev/null 2>&1 || true
  fi
  until ! "$BIN/pg_ctl" -D "$PGDATA_DIR" status >/dev/null 2>&1; do sleep 2; done
  rm -f "$PGDATA_DIR/.bench_server_opts"
}

# dynamic_shared_memory_type is the one setting you must not get wrong.
#
# Parallel HNSW builds put the whole in-memory graph in a DSM segment, and the
# build writes to it randomly as it rewires neighbour lists:
#
#   mmap  - backs the segment with a file in $PGDATA. Every edge update dirties
#           a page the kernel must write back, so the build degenerates into
#           random writes against the data directory. Measured here: 32
#           tuples/s and a 71-hour projection for 18M vectors, with all cores
#           idle at 73% iowait. Never use it for this.
#   posix - a file in /dev/shm. RAM-backed and correct, but /dev/shm is capped
#           (typically at half of RAM, and it may additionally carry a per-user
#           quota, which is what bit us: 23 GiB usable out of a 29.7 GiB mount).
#   sysv  - System V shared memory, RAM-backed and governed by kernel.shmmax /
#           kernel.shmall, which are effectively unlimited on modern Linux.
#
# So: sysv, and verify up front that the arena can actually be allocated
# instead of discovering it 20 minutes into a build.
dsm_preflight() {
  local want_kb want_bytes
  want_kb="$("$PY" - "$BUILD_MWM" <<'PY'
import re, sys
s = sys.argv[1].strip().upper()
m = re.fullmatch(r"(\d+)\s*(KB|MB|GB|TB)?", s)
if not m:
    sys.exit(f"cannot parse size {s!r}")
mult = {None: 1, "KB": 1, "MB": 1024, "GB": 1024**2, "TB": 1024**3}[m.group(2)]
print(int(m.group(1)) * mult)
PY
)"
  want_bytes=$(( want_kb * 1024 ))
  say "preflight: can we allocate a ${BUILD_MWM} System V segment?"
  "$PY" - "$want_bytes" <<'PY' || die "cannot allocate $BUILD_MWM of System V shared memory.
  Lower BUILD_MWM, or raise kernel.shmmax/kernel.shmall:
    sudo sysctl -w kernel.shmmax=<bytes> kernel.shmall=<pages>
  Do not work around this by switching to dynamic_shared_memory_type=mmap:
  the build will appear to run and then crawl (see dsm_preflight in this script)."
import ctypes, ctypes.util, sys
size = int(sys.argv[1])
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.shmget.restype = ctypes.c_int
libc.shmget.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
IPC_PRIVATE, IPC_CREAT, IPC_RMID = 0, 0o1000, 0
shmid = libc.shmget(IPC_PRIVATE, size, IPC_CREAT | 0o600)
if shmid < 0:
    sys.exit(f"shmget({size}) failed: {ctypes.get_errno()}")
libc.shmctl(shmid, IPC_RMID, None)
print(f"  ok: {size / 1024**3:.1f} GiB System V segment reserved and released", file=sys.stderr)
PY
}

# $1 = shared_buffers, $2 = maintenance_work_mem, $3... = extra options
#
# Restarting costs more than the restart: the new backend re-faults the
# shared_buffers mapping (~243k minor faults) and the pool has to be prewarmed
# again, and measurements taken right after are not comparable with ones taken
# on a warm server -- stock pgvector measured 1045-1089 QPS right after a
# restart versus 1236-1243 warm, on identical code. So when the running server
# already has the configuration a phase wants, keep it.
start_server() {
  local sb="$1" mwm="$2"; shift 2
  local want="shared_buffers=$sb maintenance_work_mem=$mwm $*"
  local stamp="$PGDATA_DIR/.bench_server_opts"

  if "$BIN/pg_ctl" -D "$PGDATA_DIR" status >/dev/null 2>&1 &&
     [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ]; then
    echo "reusing the running server (config unchanged: $want)" >&2
    return
  fi

  stop_server
  PG_OPTS="-c shared_buffers=$sb \
           -c maintenance_work_mem=$mwm \
           -c dynamic_shared_memory_type=sysv \
           -c max_worker_processes=$(( WORKERS + 9 )) \
           -c max_parallel_workers=$(( WORKERS + 1 )) \
           -c max_parallel_maintenance_workers=$WORKERS \
           -c max_wal_size=32GB \
           -c checkpoint_timeout=30min \
           -c effective_cache_size=48GB \
           -c log_min_messages=notice \
           $*" \
    "$ROOT/run_postgres.sh" start >/dev/null
  "${PSQL[@]}" -c "CREATE EXTENSION IF NOT EXISTS pg_prewarm; CREATE EXTENSION IF NOT EXISTS pg_buffercache;" >/dev/null
  printf '%s' "$want" > "$stamp"
  printf 'server started on port %s: shared_buffers=%s maintenance_work_mem=%s dsm=sysv\n' \
    "$PGPORT" "$sb" "$mwm" >&2
}

bench() { "$PY" "$BENCH/ann_bench.py" --dataset "$DATASET" "$@"; }

# Only ever touch the relations belonging to $DATASET. bench/data may hold
# several datasets, and prewarming all of them at once would blow past
# shared_buffers and make the residency assert below fail for no good reason.
TABLE=""
dataset_table() {
  if [ -z "$TABLE" ]; then
    TABLE="$(PYTHONPATH="$BENCH" "$PY" -c '
import sys, datasets
print(datasets.load(sys.argv[1], num_queries=1).table)' "$DATASET" 2>/dev/null | tail -1)"
    [ -n "$TABLE" ] || die "could not determine the table name for $DATASET"
  fi
  printf '%s' "$TABLE"
}

index_args() {
  printf '%s' "--m $M --ef-construction $EF_CONSTRUCTION"
  [ -n "$LISTS" ] && printf ' --lists %s' "$LISTS"
}

# Load every benchmark relation into shared_buffers and fail if anything is
# left out: a query phase that misses shared_buffers measures the disk, not the
# index (ef_search=10 went from 8 QPS to 945 QPS here on that alone).
prewarm_or_die() {
  say "prewarming table + index into shared_buffers"
  local t; t="$(dataset_table)"

  # Prewarming an already-resident pool re-reads every block for nothing, so
  # check first. The residency assert below still runs either way.
  if [ "$("${PSQL[@]}" -v t="$t" -At <<'SQL'
SELECT count(*) FROM pg_class c
LEFT JOIN pg_index i ON i.indexrelid = c.oid
LEFT JOIN pg_buffercache b
  ON b.relfilenode = pg_relation_filenode(c.oid)
 AND b.reldatabase = (SELECT oid FROM pg_database WHERE datname = current_database())
 AND b.relforknumber = 0
WHERE (c.relname = :'t' OR i.indrelid = :'t'::regclass)
  AND c.relkind IN ('r','i') AND pg_relation_size(c.oid) > 0
GROUP BY c.oid
HAVING count(b.bufferid) <> pg_relation_size(c.oid) / current_setting('block_size')::int;
SQL
)" = "0" ]; then
    echo "already fully resident; skipping prewarm" >&2
  else
  "${PSQL[@]}" -v t="$t" >/dev/null <<'SQL'
SELECT pg_prewarm(c.oid) FROM pg_class c
LEFT JOIN pg_index i ON i.indexrelid = c.oid
WHERE (c.relname = :'t' OR i.indrelid = :'t'::regclass)
  AND c.relkind IN ('r', 'i') AND pg_relation_size(c.oid) > 0;
SQL
  fi

  "${PSQL[@]}" -v t="$t" -At -F $'\t' <<'SQL' | tee /dev/stderr | awk -F '\t' '
      $2 != $3 { bad = 1 }
      END { if (bad) { print "error: not all blocks are resident; raise QUERY_SB" > "/dev/stderr"; exit 1 } }'
SELECT c.relname,
       pg_relation_size(c.oid) / current_setting('block_size')::int AS blocks,
       count(b.bufferid) AS cached
FROM pg_class c
LEFT JOIN pg_index i ON i.indexrelid = c.oid
LEFT JOIN pg_buffercache b
  ON b.relfilenode = pg_relation_filenode(c.oid)
 AND b.reldatabase = (SELECT oid FROM pg_database WHERE datname = current_database())
 AND b.relforknumber = 0
WHERE (c.relname = :'t' OR i.indrelid = :'t'::regclass)
  AND c.relkind IN ('r', 'i') AND pg_relation_size(c.oid) > 0
GROUP BY c.oid, c.relname
ORDER BY c.relname;
SQL
}

report_sizes() {
  "${PSQL[@]}" -v t="$(dataset_table)" >&2 <<'SQL'
SELECT c.relname, pg_size_pretty(pg_relation_size(c.oid)) AS size,
       pg_relation_size(c.oid) / current_setting('block_size')::int AS blocks
FROM pg_class c
LEFT JOIN pg_index i ON i.indexrelid = c.oid
WHERE (c.relname = :'t' OR i.indrelid = :'t'::regclass)
  AND c.relkind IN ('r','i') AND pg_relation_size(c.oid) > 0
ORDER BY 1;
SQL
}

# ---- phases -----------------------------------------------------------------

phase_data() {
  say "materializing dataset $DATASET (download + exact neighbors, cached in bench/data)"
  PYTHONPATH="$BENCH" "$PY" - "$DATASET" "$NUM_QUERIES" <<'PY'
import sys, time
import datasets
t0 = time.time()
ds = datasets.load(sys.argv[1], num_queries=int(sys.argv[2]))
print(f"{ds.name}: train={ds.train.shape} test={ds.test.shape} "
      f"neighbors={ds.neighbors.shape} metric={ds.metric} ({time.time() - t0:.0f}s)")
PY
}

phase_build() {
  dsm_preflight
  start_server "$BUILD_SB" "$BUILD_MWM"
  say "loading table and building HNSW (m=$M, ef_construction=$EF_CONSTRUCTION, $WORKERS workers)"
  # One sweep point so the run is a complete, saved record of the build.
  bench --index hnsw $(index_args) \
        --maintenance-work-mem "$BUILD_MWM" --parallel-workers "$WORKERS" \
        --sweep "$PROFILE_EF" --num-queries 100 ${REBUILD:+--rebuild} --no-save
  if grep -qi 'no longer fits' "$PGDATA_DIR/log" 2>/dev/null; then
    echo "warning: the graph spilled out of maintenance_work_mem; the build used the" >&2
    echo "         slow on-disk path and the build time is not comparable. Raise BUILD_MWM." >&2
  else
    echo "no spill: the graph stayed in maintenance_work_mem" >&2
  fi
  report_sizes
}

phase_query() {
  start_server "$QUERY_SB" 2GB -c max_parallel_workers_per_gather=0
  prewarm_or_die
  say "HNSW ef_search sweep ($EF_SWEEP), 1 client, ${DURATION}s per point"
  bench --index hnsw $(index_args) --sweep "$EF_SWEEP" \
        --num-queries "$NUM_QUERIES" --duration "$DURATION"
  report_sizes
}

phase_concurrency() {
  start_server "$QUERY_SB" 2GB -c max_parallel_workers_per_gather=0
  prewarm_or_die
  local c
  for c in ${CLIENTS//,/ }; do
    say "HNSW ef_search=$PROFILE_EF with $c concurrent clients, ${DURATION}s"
    bench --index hnsw $(index_args) --sweep "$PROFILE_EF" \
          --clients "$c" --num-queries "$NUM_QUERIES" --duration "$DURATION"
  done
}

phase_ivfflat() {
  # ann_bench.py's ensure_index drops any other vector index on the table, so
  # this phase destroys the HNSW index. It costs ~27 min to rebuild at 18M, so
  # say so plainly rather than letting it happen silently.
  if "$BIN/pg_ctl" -D "$PGDATA_DIR" status >/dev/null 2>&1; then
    local existing
    existing="$("${PSQL[@]}" -At -c "SELECT string_agg(c.relname, ' ')
        FROM pg_class c JOIN pg_index i ON i.indexrelid = c.oid
        WHERE i.indrelid = '$(dataset_table)'::regclass AND c.relname LIKE '%hnsw%'" 2>/dev/null || true)"
    if [ -n "$existing" ]; then
      echo "warning: the ivfflat phase will DROP the HNSW index ($existing)." >&2
      echo "         Re-run the 'build' phase afterwards to restore it." >&2
      if [ "${ALLOW_INDEX_SWAP:-0}" != 1 ]; then
        die "refusing to drop it. Re-run with ALLOW_INDEX_SWAP=1 to proceed,
  or run the ivfflat phase on its own once you are done with HNSW."
      fi
    fi
  fi
  dsm_preflight
  start_server "$BUILD_SB" "$BUILD_MWM"
  say "building IVFFlat"
  bench --index ivfflat $(index_args) \
        --maintenance-work-mem "$BUILD_MWM" --parallel-workers "$WORKERS" \
        --sweep 10 --num-queries 100 ${REBUILD:+--rebuild} --no-save
  report_sizes
  start_server "$QUERY_SB" 2GB -c max_parallel_workers_per_gather=0
  prewarm_or_die
  say "IVFFlat probes sweep ($PROBE_SWEEP), ${DURATION}s per point"
  bench --index ivfflat $(index_args) --sweep "$PROBE_SWEEP" \
        --num-queries "$NUM_QUERIES" --duration "$DURATION"
}

phase_exact() {
  start_server "$QUERY_SB" 2GB -c max_parallel_workers_per_gather=0
  prewarm_or_die
  say "exact sequential-scan baseline ($EXACT_QUERIES queries; each scans the whole table)"
  bench --index none --num-queries "$EXACT_QUERIES"
}

phase_profile() {
  say "CPU profile + flamegraph at ef_search=$PROFILE_EF"
  if ! "$BIN/postgres" --version >/dev/null 2>&1; then die "postgres not built"; fi
  # profile.sh does its own prewarm + residency assert, so just make sure the
  # server is configured to hold everything.
  start_server "$QUERY_SB" 2GB -c max_parallel_workers_per_gather=0
  # --duration matters here more than anywhere else: a single pass over the
  # query set is dominated by the one-off cost of faulting the shared_buffers
  # mapping into the new backend's page table (measured: ~243k minor faults in
  # the first ~1000 queries, then exactly zero), so a short profile attributes
  # a quarter of its cycles to the kernel fault path and misrepresents steady
  # state. Profile a long-lived connection instead.
  NAME="${NAME:-hnsw-ef$PROFILE_EF}" \
    "$BENCH/profile.sh" --dataset "$DATASET" --index hnsw $(index_args) \
      --sweep "$PROFILE_EF" --num-queries "$NUM_QUERIES" \
      --duration "$DURATION" --no-save
  cat >&2 <<'MSG'
note: a flamegraph is only readable if Postgres and pgvector were built with
      frame pointers:  FRAME_POINTERS=1 ./build_postgres.sh
      and it needs  sudo sysctl -w kernel.perf_event_paranoid=1 kernel.kptr_restrict=0
MSG
}

# ---- main -------------------------------------------------------------------

[ -x "$PY" ] || die "no venv at $PY; create it with: python3 -m venv bench/.venv && bench/.venv/bin/pip install -r bench/requirements.txt"
[ -x "$BIN/pg_ctl" ] || die "Postgres not built; run ./build_postgres.sh first"
mkdir -p "$LOG_DIR"

PHASES=("$@")
[ ${#PHASES[@]} -eq 0 ] && PHASES=(data build query concurrency profile ivfflat exact)

printf 'dataset=%s port=%s m=%s ef_construction=%s workers=%s\n' \
  "$DATASET" "$PGPORT" "$M" "$EF_CONSTRUCTION" "$WORKERS" >&2
printf 'phases: %s\n' "${PHASES[*]}" >&2

START=$SECONDS
for p in "${PHASES[@]}"; do
  case "$p" in
    data|build|query|concurrency|ivfflat|exact|profile) "phase_$p" ;;
    *) die "unknown phase '$p' (want: data build query concurrency ivfflat exact profile)" ;;
  esac
done

say "done in $(( (SECONDS - START) / 60 ))m$(( (SECONDS - START) % 60 ))s; results in $LOG_DIR"
if [ "${KEEP_RUNNING:-0}" = 1 ]; then
  echo "server left running on port $PGPORT" >&2
else
  stop_server
  echo "server stopped" >&2
fi
