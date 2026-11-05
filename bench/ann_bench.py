#!/usr/bin/env python3
"""Benchmark approximate nearest neighbor queries in pgvector.

Loads a dataset into Postgres, builds an HNSW, IVFFlat or PRISM index, then
sweeps the search parameter (hnsw.ef_search / ivfflat.probes / prism.nprobe) and
reports recall@k, QPS and latency percentiles for each value. Results are
printed as a table and appended to bench/results/<dataset>.jsonl.

The loaded table and the index are reused across runs when they already match,
so repeated sweeps only pay for the queries. See bench/README.md for usage.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

import datasets

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"

DEFAULT_DSN = f"host=localhost port={os.environ.get('PGPORT', '5432')} user=postgres dbname=postgres"
DEFAULT_SWEEP = {"hnsw": "10,20,40,80,160,320", "ivfflat": "1,2,4,8,16,32,64",
                 "prism": "5,10,20,40,80,160,320", "none": "0"}
SEARCH_GUC = {"hnsw": "hnsw.ef_search", "ivfflat": "ivfflat.probes",
              "prism": "prism.nprobe", "none": None}
INDEX_AMS = ("hnsw", "ivfflat", "prism")

# PRISM declares its opclasses for its own vec32 type. The type is binary
# compatible with pgvector's vector, and the extension installs a WITHOUT
# FUNCTION cast between them, so an opclass FOR TYPE vec32 is valid on a
# vector column -- no rewrite and no copy.
PRISM_OPCLASS = {"euclidean": "vec32_l2_ops",
                 "angular": "vec32_cosine_ops",
                 "ip": "vec32_ip_ops"}


def order_by(ds, index: str) -> str:
    """The ORDER BY expression that reaches `index` and no other index.

    Both extensions spell the distance operator the same way, and
    pg_vectorsearch adds pgvector's operators to prism's operator families, so
    `embedding <-> $1` is a candidate for either index and the planner picks on
    cost -- which on this data means HNSW. Resolving the operator on vec32
    operands instead leaves only prism, because pgvector's vector_l2_ops family
    has no vec32 member. The casts are WITHOUT FUNCTION, so ::vec32 is a
    zero-cost relabel rather than a conversion.
    """
    if index == "prism":
        return f"embedding::vec32 {ds.operator} %s::vec32"
    return f"embedding {ds.operator} %s"


def main() -> None:
    args = parse_args()
    ds = datasets.load(args.dataset, seed=args.seed, num_queries=args.num_queries)
    ds.table_override = args.table
    if args.k > ds.neighbors.shape[1]:
        sys.exit(f"error: k={args.k} exceeds the {ds.neighbors.shape[1]} stored neighbors per query")
    test = ds.test[: args.num_queries]
    neighbors = ds.neighbors[: args.num_queries]
    log(f"dataset {ds.name}: {len(ds.train)} x {ds.dim}, {len(test)} queries, metric {ds.metric}")

    with connect(args.dsn) as conn:
        if args.index == "prism":
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_vectorsearch")
        load_table(conn, ds, args.reload)
        index_name, build = ensure_index(conn, ds, args)
        check_plan(conn, ds, args, index_name, test[0])
        server = conn.execute(
            "SELECT current_setting('server_version'), extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        extra = prism_meta(conn, index_name) if args.index == "prism" else {}

    meta = {
        "dataset": ds.name,
        "rows": len(ds.train),
        "dim": ds.dim,
        "metric": ds.metric,
        "num_queries": len(test),
        "k": args.k,
        "clients": args.clients,
        "duration_s": args.duration or None,
        "index": args.index,
        "build_params": build_params(args, len(ds.train)),
        **build,
        "postgres_version": server[0],
        "pgvector_version": server[1],
        "pgvector_git": pgvector_git_rev(),
        **extra,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    guc = SEARCH_GUC[args.index]
    sweep = [int(v) for v in args.sweep.split(",")]
    rows = []
    print_header(guc)
    for value in sweep:
        if args.index == "hnsw" and value < args.k:
            log(f"warning: ef_search={value} < k={args.k}; HNSW returns at most ef_search rows")
        with connect(args.dsn) as conn:
            check_plan(conn, ds, args, index_name, test[0], guc, value)
        ids, latencies_ns, wall_ns, executed = run_queries(
            args.dsn, ds, test, args.k, guc, value, args.clients, args.duration,
            index=args.index)
        result = {**meta, "search_param": guc, "search_value": value if guc else None,
                  **summarize(ids, neighbors, args.k, latencies_ns, wall_ns, executed)}
        rows.append(result)
        print_row(result)

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{ds.name}.jsonl"
        with out.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log(f"appended {len(rows)} rows to {out.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="synthetic:<n>x<d>[:<metric>] or an ann-benchmarks name such as sift-128-euclidean")
    p.add_argument("--index", choices=["hnsw", "ivfflat", "prism", "none"], default="hnsw",
                   help="index to benchmark; 'none' runs exact sequential scans (default: hnsw)")
    p.add_argument("--k", type=int, default=10, help="neighbors per query (default: 10)")
    p.add_argument("--sweep", help="comma-separated ef_search (hnsw), probes (ivfflat) or "
                   f"nprobe (prism) values (default: hnsw {DEFAULT_SWEEP['hnsw']}; "
                   f"ivfflat {DEFAULT_SWEEP['ivfflat']}; prism {DEFAULT_SWEEP['prism']})")

    b = p.add_argument_group("index build")
    b.add_argument("--m", type=int, default=16, help="HNSW m (default: 16)")
    b.add_argument("--ef-construction", type=int, default=64, help="HNSW ef_construction (default: 64)")
    b.add_argument("--lists", type=int,
                   help="IVFFlat lists (default: rows/1000 up to 1M rows, sqrt(rows) beyond)")
    b.add_argument("--maintenance-work-mem", default="2GB", help="for CREATE INDEX (default: 2GB)")
    b.add_argument("--parallel-workers", type=int, default=7,
                   help="max_parallel_maintenance_workers for CREATE INDEX (default: 7)")
    b.add_argument("--rebuild", action="store_true", help="rebuild the index even if it already exists")
    b.add_argument("--keep-other-indexes", action="store_true",
                   help="do not drop the table's other ANN indexes. Off by default because the "
                        "planner has to be left no choice but the index being measured; pass this "
                        "when two indexes must coexist and the query itself forces the choice "
                        "(see order_by). check_plan still verifies which index the plan uses")

    r = p.add_argument_group("run")
    r.add_argument("--clients", type=int, default=1, help="concurrent connections issuing queries (default: 1)")
    r.add_argument("--duration", type=float, default=0.0,
                   help="seconds to keep issuing queries per sweep point, cycling the query set "
                        "(default: 0, meaning one pass). Use this for throughput numbers: a single "
                        "pass over 10k queries is only a few seconds at high client counts")
    r.add_argument("--num-queries", type=int, default=1000,
                   help="queries to run; also the synthetic query set size (default: 1000)")
    r.add_argument("--seed", type=int, default=42, help="seed for synthetic data (default: 42)")
    r.add_argument("--table", help="use this table instead of the dataset's default. A second "
                   "copy of the same data can then carry a different index type, which is the "
                   "only way to measure two pgvector indexes independently: HNSW and IVFFlat "
                   "share an operator on the same operand types, so no query shape reaches one "
                   "and not the other, and on a single table the planner picks on cost")
    r.add_argument("--reload", action="store_true", help="drop and reload the table")
    r.add_argument("--dsn", default=DEFAULT_DSN, help=f"libpq connection string (default: {DEFAULT_DSN!r})")
    r.add_argument("--no-save", action="store_true", help="don't append results to bench/results/")

    args = p.parse_args()
    args.sweep = args.sweep or DEFAULT_SWEEP[args.index]
    if args.index == "none" and args.sweep != "0":
        p.error("--sweep has no effect with --index none")
    return args


# --- setup -------------------------------------------------------------------


def connect(dsn: str) -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("SET client_min_messages = warning")
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def load_table(conn: psycopg.Connection, ds: datasets.Dataset, reload: bool) -> None:
    table = ident(ds.table)
    exists = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (table,)).fetchone()[0]
    if exists and not reload:
        count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        if count == len(ds.train):
            return
        log(f"{table} has {count} rows, expected {len(ds.train)}; reloading")

    log(f"loading {len(ds.train)} rows into {table} ...")
    start = time.perf_counter()
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(f"CREATE TABLE {table} (id int4 NOT NULL, embedding vector({ds.dim}) NOT NULL)")
    with conn.cursor() as cur, cur.copy(f"COPY {table} (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
        copy.write(b"PGCOPY\n\xff\r\n\x00" + b"\x00" * 8)  # signature, flags, header extension length
        for start_row in range(0, len(ds.train), 50_000):
            copy.write(binary_copy_rows(ds.train[start_row : start_row + 50_000], start_row))
        copy.write(b"\xff\xff")  # end-of-data marker
    conn.execute(f"ANALYZE {table}")
    log(f"loaded in {time.perf_counter() - start:.1f}s")


def binary_copy_rows(vectors: np.ndarray, first_id: int) -> bytes:
    """Encode rows as COPY BINARY tuples (id int4, embedding vector) with numpy.

    A vector's binary form is int16 dim, int16 unused, then dim big-endian float4s.
    """
    n, d = vectors.shape
    row = np.dtype([
        ("nfields", ">i2"), ("id_len", ">i4"), ("id", ">i4"),
        ("vec_len", ">i4"), ("dim", ">i2"), ("unused", ">i2"), ("vec", ">f4", (d,)),
    ])
    rows = np.empty(n, dtype=row)
    rows["nfields"] = 2
    rows["id_len"] = 4
    rows["id"] = np.arange(first_id, first_id + n)
    rows["vec_len"] = 4 + 4 * d
    rows["dim"] = d
    rows["unused"] = 0
    rows["vec"] = vectors
    return rows.tobytes()


def build_params(args: argparse.Namespace, rows: int) -> dict:
    if args.index == "hnsw":
        return {"m": args.m, "ef_construction": args.ef_construction}
    if args.index == "ivfflat":
        lists = args.lists or max(1, rows // 1000 if rows <= 1_000_000 else int(np.sqrt(rows)))
        return {"lists": lists}
    # prism derives nlist, fan_out and the rest from the row count at build
    # time; prism_index_settings() reports what it resolved (see prism_meta).
    return {}


def ensure_index(conn: psycopg.Connection, ds: datasets.Dataset, args: argparse.Namespace):
    """Make sure the requested index exists, and return (name, build info).

    By default the table's other ANN indexes are dropped, so the planner cannot
    pick one of them instead of the one being measured. --keep-other-indexes
    skips that, for a run where two indexes coexist and order_by() is what
    forces the choice; check_plan then verifies the plan really used it.
    """
    table = ident(ds.table)
    params = build_params(args, len(ds.train))
    wanted = None
    if args.index != "none":
        suffix = "_".join(f"{k}{v}" for k, v in params.items())
        wanted = ident(f"{ds.table}_{args.index}_{suffix}" if suffix
                       else f"{ds.table}_{args.index}")

    existing = [r[0] for r in conn.execute(
        """SELECT i.relname FROM pg_index x
           JOIN pg_class i ON i.oid = x.indexrelid
           JOIN pg_am am ON am.oid = i.relam
           WHERE x.indrelid = %s::regclass AND am.amname = ANY(%s)""",
        (table, list(INDEX_AMS)),
    )]
    for name in existing:
        if name == wanted and not args.rebuild:
            continue
        if args.keep_other_indexes and name != wanted:
            continue
        log(f"dropping index {name}")
        conn.execute(f"DROP INDEX {name}")

    build = {"build_s": None, "index_bytes": None, "parallel_workers": None}
    if wanted is None:
        return None, build

    if wanted not in existing or args.rebuild:
        with_opts = ", ".join(f"{k} = {v}" for k, v in params.items())
        log(f"building {wanted} ...")
        conn.execute(f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
        conn.execute(f"SET max_parallel_maintenance_workers = {args.parallel_workers}")
        opclass = PRISM_OPCLASS[ds.metric] if args.index == "prism" else ds.opclass
        start = time.perf_counter()
        with_clause = f" WITH ({with_opts})" if with_opts else ""
        conn.execute(f"CREATE INDEX {wanted} ON {table} USING {args.index} "
                     f"(embedding {opclass}){with_clause}")
        build["build_s"] = round(time.perf_counter() - start, 3)
        build["parallel_workers"] = args.parallel_workers
        log(f"built in {build['build_s']:.1f}s")
    else:
        log(f"reusing index {wanted} (pass --rebuild to time a fresh build)")

    build["index_bytes"] = conn.execute("SELECT pg_relation_size(%s::regclass)", (wanted,)).fetchone()[0]
    return wanted, build


def check_plan(conn, ds, args, index_name, query_vec, guc=None, value=None) -> None:
    """Fail if the planner won't use the index we mean to benchmark.

    This has to be re-checked for every sweep point, with that point's GUC
    applied, because the search parameter feeds the index's own cost estimate.
    Raising hnsw.ef_search raises HNSW's estimated startup cost, so with another
    ANN index on the table the planner can switch to it part-way up a sweep --
    measured here at ef_search=320, where it quietly moved to the prism index
    and produced an HNSW row that was 3.7x faster than ef_search=160.
    """
    if guc:
        conn.execute(f"SET {guc} = {value}")
    plan = "\n".join(r[0] for r in conn.execute(
        f"EXPLAIN SELECT id FROM {ident(ds.table)} "
        f"ORDER BY {order_by(ds, args.index)} LIMIT {args.k}",
        (query_vec,),
    ))
    at = f" at {guc}={value}" if guc else ""
    if index_name and index_name not in plan:
        sys.exit(f"error: the query plan doesn't use {index_name}{at}:\n{plan}\n"
                 f"hint: another index on the table won on cost. Drop it (omit "
                 f"--keep-other-indexes) or force this one through the query shape.")
    if not index_name and "Index" in plan:
        sys.exit(f"error: expected an exact sequential scan:\n{plan}")


# --- queries -----------------------------------------------------------------


def run_queries(dsn, ds, test, k, guc, value, clients, duration=0.0, index="hnsw"):
    """Run the test queries, split across `clients` connections.

    With duration=0 each query runs exactly once. With duration > 0 every
    worker keeps cycling its slice until that many seconds have elapsed, which
    is how you get a measurement window that does not depend on the size of the
    query set: at 8 clients a 10k-query single pass is only ~3s of wall clock,
    too short for startup costs to amortize or for outliers to average out.
    Repeated passes re-issue identical queries, so recall is unaffected.

    Returns (result ids [q, k] padded with -1, per-query latencies in ns,
    wall time in ns from the first query start to the last query end, and the
    total number of queries executed).
    """
    query = f"SELECT id FROM {ident(ds.table)} ORDER BY {order_by(ds, index)} LIMIT {k}"
    slices = np.array_split(np.arange(len(test)), clients)
    jobs = [(dsn, query, guc, value, k, test[s], duration) for s in slices]

    if clients == 1:
        outputs = [query_worker(*jobs[0])]
    else:
        # Each worker connects and prepares, then waits so all start together
        with mp.Manager() as manager:
            barrier = manager.Barrier(clients)
            with mp.get_context("fork").Pool(clients) as pool:
                outputs = pool.starmap(query_worker, [(*job, barrier) for job in jobs])

    ids = np.concatenate([o[0] for o in outputs])
    latencies = np.concatenate([o[1] for o in outputs])
    wall = max(o[3] for o in outputs) - min(o[2] for o in outputs)
    executed = sum(len(o[1]) for o in outputs)
    return ids, latencies, wall, executed


def query_worker(dsn, query, guc, value, k, vectors, duration=0.0, barrier=None):
    ids = np.full((len(vectors), k), -1, dtype=np.int64)
    latencies = []
    with connect(dsn) as conn, conn.cursor() as cur:
        if guc:
            conn.execute(f"SET {guc} = {int(value)}")
        cur.execute(query, (vectors[0],), prepare=True)  # prepare before the clock starts
        cur.fetchall()
        if barrier is not None:
            barrier.wait()
        # monotonic_ns is system-wide on Linux, so starts/ends compare across workers
        start = time.monotonic_ns()
        deadline = start + int(duration * 1e9) if duration else 0
        # The deadline is checked between passes, never inside one, so at least
        # one full pass always runs and `duration` is a floor rather than a cap.
        # Breaking mid-pass leaves rows of ids[] unanswered while recall is
        # computed over the whole query set, which scales the reported recall by
        # however much of the set got through: ivfflat at probes=100 managed 492
        # of 1000 queries in 60s and reported 0.485 instead of ~0.987.
        while True:
            for i, vec in enumerate(vectors):
                t0 = time.perf_counter_ns()
                cur.execute(query, (vec,), prepare=True)
                rows = cur.fetchall()
                latencies.append(time.perf_counter_ns() - t0)
                # ids is overwritten on every pass; the queries are identical,
                # so the last pass is as good as the first for recall
                ids[i, : len(rows)] = [r[0] for r in rows]
            if not deadline or time.monotonic_ns() >= deadline:
                break
        end = time.monotonic_ns()
    return ids, np.asarray(latencies, dtype=np.int64), start, end


def summarize(ids, neighbors, k, latencies_ns, wall_ns, executed=None) -> dict:
    truth = neighbors[:, :k]
    answered = [i for i in range(len(ids)) if (ids[i] != -1).any()]
    if len(answered) != len(ids):
        sys.exit(f"error: only {len(answered)} of {len(ids)} queries returned rows; "
                 f"recall would be understated. This should be unreachable now that "
                 f"the duration deadline is checked between passes.")
    hits = sum(len(np.intersect1d(ids[i], truth[i])) for i in answered)
    ms = latencies_ns / 1e6
    # recall is over the distinct query set; QPS is over everything executed,
    # which differs from it once --duration makes workers cycle their slice
    n_exec = len(latencies_ns) if executed is None else executed
    return {
        "recall": round(hits / (len(ids) * k), 5),
        "qps": round(n_exec / (wall_ns / 1e9), 2),
        "queries_executed": n_exec,
        "latency_ms": {
            "mean": round(float(ms.mean()), 4),
            "p50": round(float(np.percentile(ms, 50)), 4),
            "p95": round(float(np.percentile(ms, 95)), 4),
            "p99": round(float(np.percentile(ms, 99)), 4),
        },
    }


# --- output and helpers -----------------------------------------------------


def print_header(guc) -> None:
    print(f"\n{guc or 'exact':>16} {'recall':>8} {'QPS':>10} {'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9}")


def print_row(r) -> None:
    lat = r["latency_ms"]
    value = r["search_value"] if r["search_value"] is not None else "-"
    print(f"{value:>16} {r['recall']:>8.4f} {r['qps']:>10.1f} {lat['p50']:>9.3f} {lat['p95']:>9.3f} {lat['p99']:>9.3f}",
          flush=True)


def ident(name: str) -> str:
    """Fit a generated name into Postgres's 63-byte identifier limit."""
    if len(name) <= 63:
        return name
    return name[:54] + "_" + hashlib.sha1(name.encode()).hexdigest()[:8]


def prism_meta(conn: psycopg.Connection, index_name: str) -> dict:
    """Extension build and the settings PRISM resolved for this index.

    Almost every prism knob defaults to automatic, so neither pg_class.reloptions
    nor pg_settings says what the index actually uses; prism_index_settings()
    reports the resolved values, which is what makes a run reproducible.
    """
    version, commit = conn.execute(
        "SELECT pg_vectorsearch_version(), pg_vectorsearch_git_commit()").fetchone()
    settings = {name: setting for name, setting, _source in conn.execute(
        "SELECT * FROM prism_index_settings(%s::regclass)", (index_name,))}
    return {"pg_vectorsearch_version": version,
            "pg_vectorsearch_git": commit,
            "prism_settings": settings}


def pgvector_git_rev() -> str | None:
    """Short SHA of the pgvector submodule, with -dirty if it has local edits."""
    def git(*cmd):
        return subprocess.run(["git", "-C", str(REPO_ROOT / "pgvector"), *cmd],
                              capture_output=True, text=True, check=True).stdout.strip()
    try:
        rev = git("rev-parse", "--short", "HEAD")
        return rev + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
