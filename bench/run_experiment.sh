#!/usr/bin/env bash
# Run the whole pgvector ANN experiment end to end and leave the numbers in
# bench/results/ (one JSON line per sweep point) plus bench/profiles/.
#
# Usage: bench/run_experiment.sh [phase ...]          (no phase = all of them)
#   phases: data build query concurrency profile build_prism compare ivfflat exact rabitq
#           pgvectorfast pvf_ablation prism_solo overhead
#   For the DBpedia-OpenAI comparison on CloudLab, source bench/cloudlab/dbpedia.env first.
#
# NOTE ON ORDER: ann_bench.py keeps at most one vector index per table unless
# --keep-other-indexes is passed. The ivfflat phase builds on its own copy of
# the table, so it leaves the HNSW index alone; the compare phase does drop the
# prism index, and says so.
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
#   EF_SWEEP, PROBE_SWEEP, NPROBE_SWEEP, CLIENTS   sweep points
#   NUM_QUERIES                        size of the query set (<=10000 for bigann)
#   DURATION                           seconds per sweep point (default 60; 0 = one pass)
#   EXACT_QUERIES                      queries for the exact baseline
#   KEEP_RUNNING=1                     leave the server up at the end
#   RABITQ_BITS, RQ_EF_SWEEP           rabitq phase: bit widths, ef_search sweep
#   BENCH_ARGS                         extra ann_bench.py args for the query phase,
#                                      e.g. "--set hnsw.rabitq_bits=4"
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
NPROBE_SWEEP="${NPROBE_SWEEP:-5,10,20,40,80,160,320}"   # prism.nprobe
CLIENTS="${CLIENTS:-4,8}"
NUM_QUERIES="${NUM_QUERIES:-1000}"
EXACT_QUERIES="${EXACT_QUERIES:-30}"     # each one scans the whole table
# Seconds per sweep point. A single pass over the query set is too short to be a
# trustworthy throughput measurement once there are several clients: 10k queries
# at 8 clients is ~3s, so startup costs and one-off stalls do not average out.
DURATION="${DURATION:-60}"
PROFILE_EF="${PROFILE_EF:-40}"
RABITQ_BITS="${RABITQ_BITS:-1,2,4}"
# Wider than EF_SWEEP: estimated distances cost recall, so matching plain
# HNSW's recall needs a larger ef_search (1000 is pgvector's maximum)
RQ_EF_SWEEP="${RQ_EF_SWEEP:-10,20,40,80,160,320,640,1000}"
LOG_DIR="${LOG_DIR:-$BENCH/results}"
# -----------------------------------------------------------------------------

PSQL=("$BIN/psql" -h localhost -p "$PGPORT" -U postgres -d postgres -X -q)
say() { printf '\n=== %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
# Postgres size setting (e.g. 28GB, 512MB) -> whole GB, rounded up.
to_gb() {
  local v="${1^^}"
  case "$v" in
    *TB) echo $(( ${v%TB} * 1024 )) ;;
    *GB) echo "${v%GB}" ;;
    *MB) echo $(( (${v%MB} + 1023) / 1024 )) ;;
    *KB) echo 1 ;;
    *) die "cannot parse size '$1'" ;;
  esac
}

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

  # shared_buffers plus maintenance_work_mem (the build's DSM arena) must fit
  # in RAM with room for backends, the client and the OS; otherwise "in
  # memory" is a promise the kernel cannot keep.
  local need_gb total_gb
  need_gb=$(( $(to_gb "$sb") + $(to_gb "$mwm") + 6 ))
  total_gb=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
  [ "$need_gb" -le "$total_gb" ] ||
    die "shared_buffers=$sb + maintenance_work_mem=$mwm + 6 GB headroom = ${need_gb} GB > ${total_gb} GB RAM"

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
  local t="${1:-$(dataset_table)}"
  say "prewarming $t + its TOAST table + all their indexes into shared_buffers"

  # Prewarming an already-resident pool re-reads every block for nothing, so
  # check first. The residency assert below still runs either way.
  if [ "$("${PSQL[@]}" -v t="$t" -At <<'SQL'
SELECT count(*) FROM pg_class c
LEFT JOIN pg_index i ON i.indexrelid = c.oid
LEFT JOIN pg_buffercache b
  ON b.relfilenode = pg_relation_filenode(c.oid)
 AND b.reldatabase = (SELECT oid FROM pg_database WHERE datname = current_database())
 AND b.relforknumber = 0
WHERE c.oid IN (
        SELECT :'t'::regclass::oid
        UNION ALL SELECT reltoastrelid FROM pg_class WHERE oid = :'t'::regclass AND reltoastrelid <> 0
        UNION ALL SELECT indexrelid FROM pg_index WHERE indrelid IN (
          :'t'::regclass, (SELECT reltoastrelid FROM pg_class WHERE oid = :'t'::regclass)))
  AND pg_relation_size(c.oid) > 0
GROUP BY c.oid
HAVING count(b.bufferid) <> pg_relation_size(c.oid) / current_setting('block_size')::int;
SQL
)" = "0" ]; then
    echo "already fully resident; skipping prewarm" >&2
  else
  "${PSQL[@]}" -v t="$t" >/dev/null <<'SQL'
SELECT pg_prewarm(c.oid) FROM pg_class c
LEFT JOIN pg_index i ON i.indexrelid = c.oid
WHERE c.oid IN (
        SELECT :'t'::regclass::oid
        UNION ALL SELECT reltoastrelid FROM pg_class WHERE oid = :'t'::regclass AND reltoastrelid <> 0
        UNION ALL SELECT indexrelid FROM pg_index WHERE indrelid IN (
          :'t'::regclass, (SELECT reltoastrelid FROM pg_class WHERE oid = :'t'::regclass)))
  AND pg_relation_size(c.oid) > 0;
SQL
  fi

  # Captured, then printed, rather than piped through `tee /dev/stderr`: when
  # stderr is redirected to a file (nohup ... > run.log 2>&1), tee reopens
  # /dev/stderr with O_TRUNC and wipes the run log -- which is how the HNSW
  # build-time lines of the first DBpedia runs were lost.
  local residency
  residency="$("${PSQL[@]}" -v t="$t" -At -F $'\t' <<'SQL'
SELECT c.relname,
       pg_relation_size(c.oid) / current_setting('block_size')::int AS blocks,
       count(b.bufferid) AS cached
FROM pg_class c
LEFT JOIN pg_index i ON i.indexrelid = c.oid
LEFT JOIN pg_buffercache b
  ON b.relfilenode = pg_relation_filenode(c.oid)
 AND b.reldatabase = (SELECT oid FROM pg_database WHERE datname = current_database())
 AND b.relforknumber = 0
WHERE c.oid IN (
        SELECT :'t'::regclass::oid
        UNION ALL SELECT reltoastrelid FROM pg_class WHERE oid = :'t'::regclass AND reltoastrelid <> 0
        UNION ALL SELECT indexrelid FROM pg_index WHERE indrelid IN (
          :'t'::regclass, (SELECT reltoastrelid FROM pg_class WHERE oid = :'t'::regclass)))
  AND pg_relation_size(c.oid) > 0
GROUP BY c.oid, c.relname
ORDER BY c.relname;
SQL
)"
  printf '%s\n' "$residency" >&2
  awk -F '\t' '$2 != $3 { bad = 1 } END { exit bad }' <<<"$residency" ||
    die "not all blocks are resident; raise QUERY_SB"
}

report_sizes() {
  "${PSQL[@]}" -v t="${1:-$(dataset_table)}" >&2 <<'SQL'
SELECT c.relname, pg_size_pretty(pg_relation_size(c.oid)) AS size,
       pg_relation_size(c.oid) / current_setting('block_size')::int AS blocks
FROM pg_class c
LEFT JOIN pg_index i ON i.indexrelid = c.oid
WHERE c.oid IN (
        SELECT :'t'::regclass::oid
        UNION ALL SELECT reltoastrelid FROM pg_class WHERE oid = :'t'::regclass AND reltoastrelid <> 0
        UNION ALL SELECT indexrelid FROM pg_index WHERE indrelid IN (
          :'t'::regclass, (SELECT reltoastrelid FROM pg_class WHERE oid = :'t'::regclass)))
  AND pg_relation_size(c.oid) > 0
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
  # shellcheck disable=SC2086  # BENCH_ARGS is split on purpose
  bench --index hnsw $(index_args) --sweep "$EF_SWEEP" \
        --num-queries "$NUM_QUERIES" --duration "$DURATION" ${BENCH_ARGS:-}
  report_sizes
}

# Build a PRISM index next to the HNSW one, with build resources. Separate from
# `compare` because a build needs maintenance_work_mem (it bounds the k-means
# sample, and therefore the nlist the index resolves to) while the sweeps need
# shared_buffers, and start_server restarts when those differ.
phase_build_prism() {
  start_server "$BUILD_SB" "$BUILD_MWM"
  say "building PRISM (auto nlist, $WORKERS workers)"
  "${PSQL[@]}" -c "CREATE EXTENSION IF NOT EXISTS pg_vectorsearch" >/dev/null
  bench --index prism --sweep 20 --num-queries 100 --no-save --keep-other-indexes \
        --maintenance-work-mem "$BUILD_MWM" --parallel-workers "$WORKERS" \
        ${REBUILD:+--rebuild}
  report_sizes
}

# HNSW and PRISM measured on the same table, same buffer pool, same window.
#
# ORDER MATTERS, and the two arms are not symmetric. pg_vectorsearch adds
# pgvector's operators to prism's operator families, so `embedding <-> $1` is a
# candidate for *either* index and the planner decides on cost. pgvector's
# vector_l2_ops family has no vec32 member, so the reverse is not true:
# `embedding::vec32 <-> $1::vec32` can only reach prism.
#
# So prism is forced by type and is immune to a cost flip -- it runs first, with
# both indexes in place. HNSW cannot be forced that way, and raising
# hnsw.ef_search raises its estimated cost until prism wins (measured: at
# ef_search=320 the planner switched to prism and reported an HNSW point 3.7x
# faster than ef_search=160). HNSW therefore runs second and must be the only
# ANN index, which is what ensure_index does by default.
#
# This phase consequently DROPS the prism index at the end. Re-run
# `build_prism` to get it back; it takes under three minutes.
phase_compare() {
  start_server "$QUERY_SB" 2GB -c max_parallel_workers_per_gather=0
  prewarm_or_die
  say "PRISM nprobe sweep ($NPROBE_SWEEP), 1 client, ${DURATION}s per point"
  bench --index prism --sweep "$NPROBE_SWEEP" \
        --num-queries "$NUM_QUERIES" --duration "$DURATION" --keep-other-indexes
  say "HNSW ef_search sweep ($EF_SWEEP), 1 client, ${DURATION}s per point"
  say "this drops the prism index so the planner cannot switch to it mid-sweep"
  bench --index hnsw $(index_args) --sweep "$EF_SWEEP" \
        --num-queries "$NUM_QUERIES" --duration "$DURATION"
  report_sizes
  echo "note: the prism index was dropped; run 'build_prism' to recreate it" >&2
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

# IVFFlat on its own copy of the table.
#
# HNSW and IVFFlat are both pgvector indexes on `vector`, reached by the same
# operator on the same operand types, so -- unlike prism -- there is no query
# spelling that reaches one and not the other. On a single table the planner
# decides on cost, and ivfflat.probes feeds that estimate, so the choice can
# flip part-way up a sweep. A second table sidesteps it: one ANN index per
# table, nothing to flip to, and the HNSW index is never touched.
#
# The copy is made server-side rather than reloaded through the client: a
# CREATE TABLE AS of 10 GB is minutes of sequential I/O against ~9 minutes of
# COPY from numpy. It stays a logged table so the IVFFlat build pays the same
# WAL cost the HNSW build did and the two build times stay comparable.
#
# Only one index is resident at a time, by design. This arm prewarms its own
# table and index (~20 GB) and evicts the HNSW pages; going back to HNSW
# re-prewarms. shared_buffers stays at QUERY_SB.
phase_ivfflat() {
  local src ivf
  src="$(dataset_table)"
  ivf="${src}_ivf"

  start_server "$BUILD_SB" "$BUILD_MWM"
  local n
  n="$("${PSQL[@]}" -At -c "SELECT count(*) FROM $ivf" 2>/dev/null || echo 0)"
  if [ "$n" != "$("${PSQL[@]}" -At -c "SELECT count(*) FROM $src")" ]; then
    say "copying $src -> $ivf (server-side)"
    "${PSQL[@]}" -c "DROP TABLE IF EXISTS $ivf" >/dev/null
    "${PSQL[@]}" -c "CREATE TABLE $ivf AS SELECT * FROM $src" >/dev/null
    "${PSQL[@]}" -c "ANALYZE $ivf" >/dev/null
  else
    echo "reusing $ivf ($n rows)" >&2
  fi

  say "building IVFFlat on $ivf"
  bench --index ivfflat $(index_args) --table "$ivf" \
        --maintenance-work-mem "$BUILD_MWM" --parallel-workers "$WORKERS" \
        --sweep 10 --num-queries 100 ${REBUILD:+--rebuild} --no-save
  report_sizes "$ivf"

  start_server "$QUERY_SB" 2GB -c max_parallel_workers_per_gather=0
  prewarm_or_die "$ivf"
  say "IVFFlat probes sweep ($PROBE_SWEEP), 1 client, ${DURATION}s per point"
  bench --index ivfflat $(index_args) --table "$ivf" --sweep "$PROBE_SWEEP" \
        --num-queries "$NUM_QUERIES" --duration "$DURATION"
  report_sizes "$ivf"
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

# HNSW with RaBitQ codes stored instead of vectors (rabitq_bits index option,
# awitten1/pgvector rabitq branch), against plain HNSW with the same binary.
#
# Each bit width gets its own copy of the table: every arm is an HNSW index on
# the same operator, so on one table the planner would pick between them on
# cost. The copies keep their indexes, so a rerun only re-measures.
#
# All builds happen first, then one restart into the query configuration and
# every arm back to back. Plain HNSW runs first and last as a drift check.
# Before each arm the whole buffer pool is evicted: the previous arm's pages
# carry high usage counts, so the clock sweep would evict the freshly
# prewarmed ones first and the residency assert would fail.
phase_rabitq() {
  local src t bits log_start
  src="$(dataset_table)"

  dsm_preflight
  start_server "$BUILD_SB" "$BUILD_MWM"
  for bits in ${RABITQ_BITS//,/ }; do
    t="${src}_rq${bits}"
    if [ "$("${PSQL[@]}" -At -c "SELECT count(*) FROM $t" 2>/dev/null || echo 0)" != \
         "$("${PSQL[@]}" -At -c "SELECT count(*) FROM $src")" ]; then
      say "copying $src -> $t (server-side)"
      "${PSQL[@]}" -c "DROP TABLE IF EXISTS $t" >/dev/null
      "${PSQL[@]}" -c "CREATE TABLE $t AS SELECT * FROM $src" >/dev/null
      "${PSQL[@]}" -c "ANALYZE $t" >/dev/null
    fi

    say "building HNSW with ${bits}-bit RaBitQ codes on $t ($WORKERS workers)"
    log_start=$(stat -c %s "$PGDATA_DIR/log")
    bench --index hnsw $(index_args) --rabitq-bits "$bits" --table "$t" \
          --maintenance-work-mem "$BUILD_MWM" --parallel-workers "$WORKERS" \
          --sweep 40 --num-queries 100 ${REBUILD:+--rebuild} --no-save
    if tail -c +$(( log_start + 1 )) "$PGDATA_DIR/log" | grep -qi 'no longer fits'; then
      echo "warning: the graph spilled out of maintenance_work_mem; the rest was built" >&2
      echo "         on disk with estimated distances. Raise BUILD_MWM." >&2
    fi
    report_sizes "$t"
  done

  start_server "$QUERY_SB" 2GB -c max_parallel_workers_per_gather=0
  for arm in 0 ${RABITQ_BITS//,/ } 0; do
    if [ "$arm" = 0 ]; then t="$src"; else t="${src}_rq${arm}"; fi
    "${PSQL[@]}" -c "SELECT pg_buffercache_evict_all()" >/dev/null
    prewarm_or_die "$t"
    say "HNSW rabitq_bits=$arm on $t: ef_search sweep ($RQ_EF_SWEEP), 1 client, ${DURATION}s per point"
    bench --index hnsw $(index_args) --table "$t" \
          $([ "$arm" != 0 ] && printf -- '--rabitq-bits %s' "$arm") \
          --sweep "$RQ_EF_SWEEP" --num-queries "$NUM_QUERIES" --duration "$DURATION"
  done
}

# Stock pgvector against pgvectorfast (pgvector 0.8.7 + the Calico buffer pool),
# on the same HNSW index: pgvectorfast keeps pgvector's on-disk format.
#
# Calico has to be in shared_preload_libraries, so every arm restarts the
# server -- stock arms too, so both pay the same cold-start cost. Arms repeat
# (PVF_ARMS="stock fast stock fast") so that restart-to-restart noise shows up
# as disagreement between rounds instead of passing for a difference. Each arm
# is set up by serve_arm, below.
PVF_ARMS="${PVF_ARMS:-stock fast stock fast}"
PVF_EF_SWEEP="${PVF_EF_SWEEP:-10,20,40,80,160,320,640,1000}"
PVF_CLIENTS="${PVF_CLIENTS:-1,8}"
STOCK_SO="${STOCK_SO:-}"                 # a stock pgvector vector.so (e.g. v0.8.7)
FAST_SO="${FAST_SO:-$ROOT/pgvectorfast/vectorfast.so}"

# Bring the query server up as one pgvector arm (stock | fast) on the shared
# HNSW index, prewarmed. Extra arguments are passed to start_server, e.g. a
# listen_addresses for a client on another machine. Every arm restarts the
# server, stock arms too, so both pay the same cold-start cost.
#
# The database's vector functions are bound to $libdir/vector, not
# $libdir/vectorfast, so the fast arm points vector.so at vectorfast.so with a
# symlink. dfmgr matches libraries by inode, so the backend maps only
# vectorfast.so, once. Requires a non-LTO Postgres (Calico refuses LTO builds).
serve_arm() {
  local arm="$1"; shift
  local lib preload=""
  lib="$("$BIN/pg_config" --pkglibdir)"
  stop_server
  rm -f "$lib/vector.so"
  case "$arm" in
    stock) install -m755 "$STOCK_SO" "$lib/vector.so" ;;
    fast)  install -m755 "$FAST_SO" "$lib/vectorfast.so"
           ln -s vectorfast.so "$lib/vector.so"
           preload="-c shared_preload_libraries=vectorfast" ;;
    *) die "unknown arm '$arm' (want stock or fast)" ;;
  esac
  # shellcheck disable=SC2086  # preload is split on purpose
  start_server "$QUERY_SB" 2GB -c max_parallel_workers_per_gather=0 $preload "$@"
  if [ "$arm" = fast ]; then
    "${PSQL[@]}" -c "CREATE EXTENSION IF NOT EXISTS calico" -c "CALL calico.activate()" >/dev/null
    [ "$("${PSQL[@]}" -At -c 'SELECT calico.status()')" = calico ] ||
      die "calico did not activate"
  fi
  prewarm_or_die
}

# Leave the server up as one arm (SERVE_ARM=stock|fast), reachable from a
# client on another machine at LISTEN_IP from CLIENT_NET. The client is then
# driven separately (bench/cloudlab/remote_client.sh); KEEP_RUNNING is forced.
#
# Trust auth on a private experiment network only: CLIENT_NET must be the
# CloudLab LAN, never a routable range.
LISTEN_IP="${LISTEN_IP:-}"
CLIENT_NET="${CLIENT_NET:-}"
phase_serve() {
  [ -n "$LISTEN_IP" ] && [ -n "$CLIENT_NET" ] || die "set LISTEN_IP and CLIENT_NET for serve"
  case "$CLIENT_NET" in
    10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*) ;;
    *) die "CLIENT_NET=$CLIENT_NET is not a private range; refusing trust auth" ;;
  esac
  local hba="$PGDATA_DIR/pg_hba.conf" line="host all postgres $CLIENT_NET trust"
  grep -qxF "$line" "$hba" || printf '%s\n' "$line" >> "$hba"
  serve_arm "${SERVE_ARM:?set SERVE_ARM=stock or fast}" -c "listen_addresses=localhost,$LISTEN_IP"
  "${PSQL[@]}" -c "SELECT pg_reload_conf()" >/dev/null
  # SERVE_TABLE (e.g. the PRISM copy) replaces the main table in the buffer pool.
  if [ -n "${SERVE_TABLE:-}" ]; then
    "${PSQL[@]}" -c "SELECT pg_buffercache_evict_all()" >/dev/null
    prewarm_or_die "$SERVE_TABLE"
  fi
  KEEP_RUNNING=1
  say "serving the $SERVE_ARM arm on $LISTEN_IP:$PGPORT for clients in $CLIENT_NET"
}

phase_pgvectorfast() {
  [ -f "$STOCK_SO" ] || die "set STOCK_SO to a stock pgvector vector.so"
  local lib; lib="$("$BIN/pg_config" --pkglibdir)"
  # A stock-only run (PVF_ARMS="stock stock", e.g. a cross-node baseline) does
  # not need pgvectorfast built at all.
  if [[ " $PVF_ARMS " == *" fast "* ]]; then
    [ -f "$FAST_SO" ] || die "no $FAST_SO; build pgvectorfast first"
    [ -f "$("$BIN/pg_config" --sharedir)/extension/calico.control" ] ||
      die "calico extension files not installed"
    # grep -c, not -q: -q exits early and pipefail then sees nm's SIGPIPE.
    [ "$(nm "$FAST_SO" | grep -c ' calico_')" -gt 0 ] || die "$FAST_SO has no calico symbols"
  fi
  [ "$(nm "$STOCK_SO" | grep -c ' calico_' || true)" -eq 0 ] || die "$STOCK_SO has calico symbols"

  local arm round=0 c
  for arm in $PVF_ARMS; do
    round=$(( round + 1 ))
    serve_arm "$arm"
    for c in ${PVF_CLIENTS//,/ }; do
      say "arm $round ($arm): HNSW ef_search sweep ($PVF_EF_SWEEP), $c client(s), ${DURATION}s per point"
      BENCH_LABEL="pvf-$arm-r$round" \
        bench --index hnsw $(index_args) --sweep "$PVF_EF_SWEEP" --clients "$c" \
              --num-queries "$NUM_QUERIES" --duration "$DURATION"
    done
  done
}

# Ablation of pgvectorfast's speedup over stock pgvector, one layer at a time,
# on the same HNSW index:
#   A0  stock pgvector (STOCK_SO), no preload
#   A1  vectorfast.so, NOT preloaded: Calico never installs, so this is the
#       rewritten HNSW search loop + SIMD kernels on Postgres's own buffer manager
#       (hnsw.enable_prefetch=off; its batching rides on Calico's lookups anyway)
#   A2  Calico preloaded + active, calico.optimistic_reads=off, prefetch off:
#       adds array page translation, still pinning and locking
#   A3  A2 + optimistic_reads=on: no pins or content locks on traversal
#   A4  A3 + hnsw.enable_prefetch=on: the full pgvectorfast
# The server restarts only when the library or preload changes (A0 | A1 | A2-4).
# If perf is allowed (perf_event_paranoid <= 1), each arm also gets a 1-client
# window at PVA_PERF_EF under `perf stat -I` and a flat `perf record`, saved to
# bench/profiles/ablation/.
PVA_ARMS="${PVA_ARMS:-A0 A1 A2 A3 A4}"
PVA_EF_SWEEP="${PVA_EF_SWEEP:-10,40,160}"
PVA_CLIENTS="${PVA_CLIENTS:-1,8}"
PVA_PERF_EF="${PVA_PERF_EF:-40}"
PVA_PERF_S="${PVA_PERF_S:-40}"

pva_perf_window() {
  local arm="$1"; shift
  local out="$BENCH/profiles/ablation" pm pstat prec
  if [ "$(cat /proc/sys/kernel/perf_event_paranoid)" -gt 1 ]; then
    echo "perf_event_paranoid > 1; skipping the perf window for $arm" >&2
    return
  fi
  mkdir -p "$out"
  pm="$(head -1 "$PGDATA_DIR/postmaster.pid")"
  say "arm $arm: perf window, ef_search=$PVA_PERF_EF, 1 client, ${PVA_PERF_S}s"
  perf stat -I 2000 -x, -e cycles,instructions,cache-misses,branch-misses \
    -p "$pm" -o "$out/$arm.stat.csv" 2>/dev/null &
  pstat=$!
  perf record -F 999 -p "$pm" -o "$out/$arm.perf.data" >/dev/null 2>&1 &
  prec=$!
  sleep 1
  BENCH_LABEL="pva-$arm-perf" \
    bench --index hnsw $(index_args) --sweep "$PVA_PERF_EF" --clients 1 \
          --num-queries "$NUM_QUERIES" --duration "$PVA_PERF_S" --no-save "$@" \
    | tee "$out/$arm.bench.txt"
  kill -INT "$pstat" "$prec" 2>/dev/null || true
  wait "$pstat" "$prec" 2>/dev/null || true
}

phase_pvf_ablation() {
  [ -f "$STOCK_SO" ] || die "set STOCK_SO to a stock pgvector vector.so"
  [ -f "$FAST_SO" ] || die "no $FAST_SO; build pgvectorfast first"
  local lib; lib="$("$BIN/pg_config" --pkglibdir)"
  [ "$(nm "$FAST_SO" | grep -c ' calico_')" -gt 0 ] || die "$FAST_SO has no calico symbols"
  [ "$(nm "$STOCK_SO" | grep -c ' calico_' || true)" -eq 0 ] || die "$STOCK_SO has calico symbols"

  local arm group cur="" c
  local -a sets
  for arm in $PVA_ARMS; do
    case "$arm" in
      A0) group=stock; sets=() ;;
      A1) group=fast-nopreload; sets=(--set hnsw.enable_prefetch=off) ;;
      A2) group=fast-calico; sets=(--set calico.optimistic_reads=off --set hnsw.enable_prefetch=off) ;;
      A3) group=fast-calico; sets=(--set calico.optimistic_reads=on --set hnsw.enable_prefetch=off) ;;
      A4) group=fast-calico; sets=(--set calico.optimistic_reads=on --set hnsw.enable_prefetch=on) ;;
      *) die "unknown arm '$arm' (want A0..A4)" ;;
    esac
    if [ "$group" != "$cur" ]; then
      stop_server
      rm -f "$lib/vector.so"
      if [ "$group" = stock ]; then
        install -m755 "$STOCK_SO" "$lib/vector.so"
      else
        install -m755 "$FAST_SO" "$lib/vectorfast.so"
        ln -s vectorfast.so "$lib/vector.so"
      fi
      if [ "$group" = fast-calico ]; then
        start_server "$QUERY_SB" 2GB -c max_parallel_workers_per_gather=0 \
                     -c shared_preload_libraries=vectorfast
        "${PSQL[@]}" -c "CREATE EXTENSION IF NOT EXISTS calico" -c "CALL calico.activate()" >/dev/null
        [ "$("${PSQL[@]}" -At -c 'SELECT calico.status()')" = calico ] || die "calico did not activate"
      else
        start_server "$QUERY_SB" 2GB -c max_parallel_workers_per_gather=0
        # Never call calico.status() under the stock library: it would dlopen
        # vectorfast.so next to vector.so and register the hnsw.* GUCs twice.
        if [ "$group" = fast-nopreload ]; then
          [ "$("${PSQL[@]}" -At -c 'SELECT calico.status()')" = not_preloaded ] ||
            die "expected calico to be inactive in $arm"
        fi
      fi
      prewarm_or_die
      cur="$group"
    fi
    for c in ${PVA_CLIENTS//,/ }; do
      say "arm $arm ($group ${sets[*]}): ef_search $PVA_EF_SWEEP, $c client(s), ${DURATION}s per point"
      BENCH_LABEL="pva-$arm" \
        bench --index hnsw $(index_args) --sweep "$PVA_EF_SWEEP" --clients "$c" \
              --num-queries "$NUM_QUERIES" --duration "$DURATION" "${sets[@]}"
    done
    pva_perf_window "$arm" "${sets[@]}"
  done
}

# Install STOCK_SO as the server's vector.so, replacing a pgvectorfast symlink
# if one is there. A preload of vectorfast cannot be undone without a restart,
# so the server is stopped first whenever the library actually changes.
install_stock_so() {
  [ -f "$STOCK_SO" ] || die "set STOCK_SO to a stock pgvector vector.so"
  [ "$(nm "$STOCK_SO" | grep -c ' calico_' || true)" -eq 0 ] || die "$STOCK_SO has calico symbols"
  local lib; lib="$("$BIN/pg_config" --pkglibdir)"
  if [ -L "$lib/vector.so" ] || ! cmp -s "$STOCK_SO" "$lib/vector.so"; then
    stop_server
    rm -f "$lib/vector.so"
    install -m755 "$STOCK_SO" "$lib/vector.so"
  fi
  [ "$(nm -D "$lib/vector.so" | grep -c calico_ || true)" -eq 0 ] ||
    die "$lib/vector.so still has calico symbols"
}

# PRISM on its own server-side copy of the table, with stock pgvector loaded.
#
# A separate table means there is never a second ANN index for the planner to
# switch to, and building or dropping indexes here cannot touch the HNSW index
# the pgvector/pgvectorfast arms share. Stock vector.so, so no Calico and no
# pgvectorfast code is anywhere in this arm's process.
phase_prism_solo() {
  local src t c
  src="$(dataset_table)"
  t="${src}_prism"
  install_stock_so

  start_server "$BUILD_SB" "$BUILD_MWM"
  "${PSQL[@]}" -c "CREATE EXTENSION IF NOT EXISTS pg_vectorsearch" >/dev/null
  if [ "$("${PSQL[@]}" -At -c "SELECT count(*) FROM $t" 2>/dev/null || echo 0)" != \
       "$("${PSQL[@]}" -At -c "SELECT count(*) FROM $src")" ]; then
    say "copying $src -> $t (server-side)"
    "${PSQL[@]}" -c "DROP TABLE IF EXISTS $t" >/dev/null
    "${PSQL[@]}" -c "CREATE TABLE $t AS SELECT * FROM $src" >/dev/null
    "${PSQL[@]}" -c "ANALYZE $t" >/dev/null
  fi
  say "building PRISM on $t (auto nlist, $WORKERS workers)"
  bench --index prism --table "$t" --sweep 20 --num-queries 100 --no-save \
        --maintenance-work-mem "$BUILD_MWM" --parallel-workers "$WORKERS" \
        ${REBUILD:+--rebuild}
  report_sizes "$t"

  start_server "$QUERY_SB" 2GB -c max_parallel_workers_per_gather=0
  "${PSQL[@]}" -c "SELECT pg_buffercache_evict_all()" >/dev/null
  prewarm_or_die "$t"
  for c in ${CLIENTS//,/ }; do
    say "PRISM nprobe sweep ($NPROBE_SWEEP) on $t, $c client(s), ${DURATION}s per point"
    BENCH_LABEL="prism-solo" \
      bench --index prism --table "$t" --sweep "$NPROBE_SWEEP" --clients "$c" \
            --num-queries "$NUM_QUERIES" --duration "$DURATION"
  done
}

# The floor every Postgres query pays before any index work: the same client,
# prepared statement and vector parameter as an ANN query, with no search (see
# RTT_QUERY in ann_bench.py). In-process engines (Ginger, FAISS) never pay it,
# so it is what to subtract when reading them against Postgres on one chart.
phase_overhead() {
  local c
  install_stock_so
  start_server "$QUERY_SB" 2GB -c max_parallel_workers_per_gather=0
  for c in ${CLIENTS//,/ }; do
    say "round-trip floor, $c client(s), ${DURATION}s"
    BENCH_LABEL="rtt" \
      bench --index rtt --clients "$c" --num-queries "$NUM_QUERIES" --duration "$DURATION"
  done
}

# ---- main -------------------------------------------------------------------

[ -x "$PY" ] || die "no venv at $PY; create it with: python3 -m venv bench/.venv && bench/.venv/bin/pip install -r bench/requirements.txt"
[ -x "$BIN/pg_ctl" ] || die "Postgres not built; run ./build_postgres.sh first"
mkdir -p "$LOG_DIR"

# Environment guards, opted into by bench/cloudlab/dbpedia.env. Both reset on a
# reboot, so they are checked on every run rather than trusted from setup.
if [ "${REQUIRE_PINNED:-0}" = 1 ]; then
  "$BENCH/cloudlab/pin_cpu.sh" check || die "CPU clock not pinned"
fi
if [ "${REQUIRE_NO_SWAP:-0}" = 1 ] && [ -n "$(tail -n +2 /proc/swaps)" ]; then
  die "swap is enabled; run 'sudo swapoff -a' so a run that does not fit in RAM fails instead of slowing down"
fi

PHASES=("$@")
[ ${#PHASES[@]} -eq 0 ] && PHASES=(data build query concurrency profile ivfflat exact)
# `build_prism` and `compare` are not in the default set: they need
# pg_vectorsearch installed, and `compare` drops the prism index as it runs.

printf 'dataset=%s port=%s m=%s ef_construction=%s workers=%s\n' \
  "$DATASET" "$PGPORT" "$M" "$EF_CONSTRUCTION" "$WORKERS" >&2
printf 'phases: %s\n' "${PHASES[*]}" >&2

START=$SECONDS
for p in "${PHASES[@]}"; do
  case "$p" in
    data|build|query|concurrency|ivfflat|exact|profile|compare|build_prism|rabitq|pgvectorfast|pvf_ablation|prism_solo|overhead|serve) "phase_$p" ;;
    *) die "unknown phase '$p' (want: data build query concurrency profile build_prism compare ivfflat exact rabitq pgvectorfast pvf_ablation prism_solo overhead serve)" ;;
  esac
done

say "done in $(( (SECONDS - START) / 60 ))m$(( (SECONDS - START) % 60 ))s; results in $LOG_DIR"
if [ "${KEEP_RUNNING:-0}" = 1 ]; then
  echo "server left running on port $PGPORT" >&2
else
  stop_server
  echo "server stopped" >&2
fi
