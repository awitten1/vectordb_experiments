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
import socket
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
                 "prism": "5,10,20,40,80,160,320", "none": "0", "rtt": "0",
                 "hnsw_halfvec": "10,20,40,80,160,320", "hnsw_bq": "10,20,40,80,160,320"}
SEARCH_GUC = {"hnsw": "hnsw.ef_search", "ivfflat": "ivfflat.probes",
              "prism": "prism.nprobe", "none": None, "rtt": None,
              "hnsw_halfvec": "hnsw.ef_search", "hnsw_bq": "hnsw.ef_search"}

# Quantized HNSW variants, built as pgvector expression indexes on the same
# `embedding` column (pgvector README: "Half-Precision Indexing" and "Binary
# Quantization"). Each serves only its own query expression, so they can sit on
# the table next to the plain HNSW index without the planner confusing them;
# check_plan still verifies it at every point.
#   hnsw_halfvec  the index stores float16 copies: half the bytes per element
#   hnsw_bq       the index stores 1 bit per dimension (sign) and is searched by
#                 Hamming distance; the query fetches ef_search candidates from
#                 it and re-sorts them by exact distance on the full vectors
HALFVEC_OPS = {"euclidean": "halfvec_l2_ops", "angular": "halfvec_cosine_ops", "ip": "halfvec_ip_ops"}

# --index rtt: the per-query cost of the Postgres boundary with no search at
# all. Same client, same prepared-statement path and the same vector parameter
# (parsed by vector_in on the server) as a real ANN query, but the plan is a
# Result node. Subtracting it from an ANN query's latency leaves the index's
# share, which is what an in-process library like Ginger or FAISS pays alone.
RTT_QUERY = "SELECT 0::bigint FROM (SELECT %s::vector AS v) s WHERE s.v IS NOT NULL"
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


def query_sql(ds, index: str, k: int, value) -> tuple[str, int]:
    """The benchmark query for `index` at search value `value`, and how many
    times the query vector is passed to it."""
    table = ident(ds.table)
    if index == "rtt":
        return RTT_QUERY, 1
    if index == "hnsw_halfvec":
        return (f"SELECT id FROM {table} ORDER BY embedding::halfvec({ds.dim}) {ds.operator} "
                f"%s::halfvec({ds.dim}) LIMIT {k}"), 1
    if index == "hnsw_bq":
        # ef_search candidates by Hamming distance, then exact re-ranking. HNSW
        # returns at most ef_search rows, so that is also the candidate count.
        cand = max(int(value), k)
        return (f"SELECT id FROM (SELECT id, embedding FROM {table} "
                f"ORDER BY binary_quantize(embedding)::bit({ds.dim}) <~> binary_quantize(%s::vector) "
                f"LIMIT {cand}) c ORDER BY embedding {ds.operator} %s LIMIT {k}"), 2
    return f"SELECT id FROM {table} ORDER BY {order_by(ds, index)} LIMIT {k}", 1


def main() -> None:
    args = parse_args()
    if os.environ.get("REQUIRE_NO_SWAP") == "1":
        require_no_swap()
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
        if args.index == "rtt":
            # No table and no index: only the boundary is measured.
            index_name, build = None, {"build_s": None, "index_bytes": None, "parallel_workers": None}
        else:
            load_table(conn, ds, args.reload)
            index_name, build = ensure_index(conn, ds, args)
            check_plan(conn, ds, args, index_name, test[0])
        server = conn.execute(
            "SELECT current_setting('server_version'), extversion, current_setting('shared_preload_libraries') FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        server_host = conn.info.host
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
        # The digest reads this machine's install, so it only describes the
        # server when the server is this machine.
        "vector_so": vector_so_digest() if server_is_local(server_host) else None,
        "server_host": server_host,
        "preload": server[2] or None,
        "label": os.environ.get("BENCH_LABEL") or None,
        "host": socket.gethostname(),
        "settings": EXTRA_SETTINGS or None,
        **extra,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    guc = SEARCH_GUC[args.index]
    sweep = [int(v) for v in args.sweep.split(",")]
    rows = []
    print_header(guc)
    for value in sweep:
        if args.index in ("hnsw", "hnsw_halfvec") and value < args.k:
            log(f"warning: ef_search={value} < k={args.k}; HNSW returns at most ef_search rows")
        if args.index != "rtt":
            with connect(args.dsn) as conn:
                check_plan(conn, ds, args, index_name, test[0], guc, value)
        vm_before = vmstat()
        ids, latencies_ns, wall_ns, executed = run_queries(
            args.dsn, ds, test, args.k, guc, value, args.clients, args.duration,
            index=args.index)
        mem = memory_check(args, ds, vm_before)
        result = {**meta, "search_param": guc, "search_value": value if guc else None,
                  **summarize(ids, neighbors, args.k, latencies_ns, wall_ns, executed),
                  "memory": mem}
        if args.index == "rtt":
            result["recall"] = None  # nothing was searched
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
    p.add_argument("--index", choices=["hnsw", "ivfflat", "prism", "none", "rtt", "hnsw_halfvec", "hnsw_bq"],
                   default="hnsw",
                   help="index to benchmark; 'none' runs exact sequential scans; 'rtt' measures "
                        "the query round trip with no search (default: hnsw)")
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
    b.add_argument("--rabitq-bits", type=int, default=0,
                   help="HNSW only, experimental (awitten1/pgvector rabitq branch): store RaBitQ "
                        "codes with this many bits per dimension instead of vectors (default: 0, off)")
    b.add_argument("--quantizer", choices=["tq4"],
                   help="HNSW only, experimental (pgvector_ntq, awitten1/pgvector hnsw-tq4-nav): "
                        "store 4-bit TurboQuant navigation codes in the graph instead of vectors")
    b.add_argument("--rerank", choices=["none", "ntq10", "ntq12", "exact"],
                   help="with --quantizer: what the final candidates are re-ranked from "
                        "(pgvector_ntq's default is ntq12)")
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
    r.add_argument("--set", action="append", default=[], metavar="GUC=VALUE",
                   help="extra setting for every connection, e.g. hnsw.rabitq_bits=4; "
                        "repeatable, and recorded in the results")

    args = p.parse_args()
    args.sweep = args.sweep or DEFAULT_SWEEP[args.index]
    for item in args.set:
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            p.error(f"--set expects GUC=VALUE, got {item!r}")
        EXTRA_SETTINGS[name.strip()] = value.strip()
    if args.rabitq_bits and args.index != "hnsw":
        p.error("--rabitq-bits only applies to --index hnsw")
    if args.quantizer and args.index != "hnsw":
        p.error("--quantizer only applies to --index hnsw")
    if args.rerank and not args.quantizer:
        p.error("--rerank needs --quantizer")
    if args.index in ("none", "rtt") and args.sweep != "0":
        p.error(f"--sweep has no effect with --index {args.index}")
    if args.index not in ("none", "rtt"):
        # When the vectors are TOASTed (anything over ~2 KB, e.g. 1536-d
        # float4), the heap is a few hundred pages of pointers and the planner
        # prices a seq scan + sort as nearly free: it does not cost detoasting
        # 6 KB per row. Measured: at prism.nprobe=80 on 100K x 1536 it chose a
        # seq scan costed at 4051. Disabling seq scans for index arms keeps the
        # benchmark on the index it means to measure; check_plan still proves
        # it at every point, and the setting is recorded in the results.
        EXTRA_SETTINGS.setdefault("enable_seqscan", "off")
    return args


# --- setup -------------------------------------------------------------------


# GUCs from --set, applied on every connection (query workers inherit this via fork)
EXTRA_SETTINGS: dict[str, str] = {}


def connect(dsn: str) -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("SET client_min_messages = warning")
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if EXTRA_SETTINGS:
        # Load the library first so its GUCs exist: a typo, or a setting the
        # installed vector.so doesn't define, then errors instead of silently
        # becoming a placeholder
        conn.execute("LOAD 'vector'")
        for name, value in EXTRA_SETTINGS.items():
            conn.execute("SELECT set_config(%s, %s, false)", (name, value))
    register_vector(conn)
    return conn


def protect_indexes() -> bool:
    """PROTECT_INDEXES=1 (set by bench/cloudlab/hn.env): never drop or rebuild an
    existing table or ANN index; fail instead. For indexes that take hours to
    build and are meant to be measured again under other server settings. A
    missing table or index is still created."""
    return os.environ.get("PROTECT_INDEXES") == "1"


def load_table(conn: psycopg.Connection, ds: datasets.Dataset, reload: bool) -> None:
    table = ident(ds.table)
    exists = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (table,)).fetchone()[0]
    if exists and not reload:
        count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        if count == len(ds.train):
            return
        if protect_indexes():
            sys.exit(f"error: {table} has {count} rows but the dataset has {len(ds.train)}; "
                     f"PROTECT_INDEXES=1 refuses to drop and reload it (and its indexes)")
        log(f"{table} has {count} rows, expected {len(ds.train)}; reloading")
    elif exists and protect_indexes():
        sys.exit(f"error: PROTECT_INDEXES=1 refuses --reload of {table}")

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


def binary_copy_rows(vectors: np.ndarray, first_id: int = 0, ids: np.ndarray | None = None) -> bytes:
    """Encode rows as COPY BINARY tuples (id int4, embedding vector) with numpy.

    The ids are first_id, first_id + 1, ..., or `ids` (one per vector) when
    given, e.g. to load rows in another order with their own ids. A vector's
    binary form is int16 dim, int16 unused, then dim big-endian float4s.
    """
    n, d = vectors.shape
    row = np.dtype([
        ("nfields", ">i2"), ("id_len", ">i4"), ("id", ">i4"),
        ("vec_len", ">i4"), ("dim", ">i2"), ("unused", ">i2"), ("vec", ">f4", (d,)),
    ])
    rows = np.empty(n, dtype=row)
    rows["nfields"] = 2
    rows["id_len"] = 4
    if ids is None:
        rows["id"] = np.arange(first_id, first_id + n)
    else:
        if len(ids) != n:
            raise ValueError(f"{len(ids)} ids for {n} vectors")
        rows["id"] = ids
    rows["vec_len"] = 4 + 4 * d
    rows["dim"] = d
    rows["unused"] = 0
    rows["vec"] = vectors
    return rows.tobytes()


def build_params(args: argparse.Namespace, rows: int) -> dict:
    if args.index.startswith("hnsw"):
        params = {"m": args.m, "ef_construction": args.ef_construction}
        if args.rabitq_bits:
            params["rabitq_bits"] = args.rabitq_bits
        # Part of the index name, so a quantized index is never mistaken for
        # (and reused as) the plain one with the same m / ef_construction.
        if args.quantizer:
            params["quantizer"] = args.quantizer
            params["rerank"] = args.rerank or "ntq12"
        return params
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
        # quantizer/rerank values are self-describing (tq4, ntq12): no key prefix.
        suffix = "_".join(f"{v}" if k in ("quantizer", "rerank") else f"{k}{v}" for k, v in params.items())
        raw = f"{ds.table}_{args.index}_{suffix}" if suffix else f"{ds.table}_{args.index}"
        # Postgres truncates identifiers past 63 bytes, and the truncated name
        # would never match `wanted`: the index would be rebuilt on every run.
        if len(raw) > 63:
            sys.exit(f"error: index name {raw!r} is {len(raw)} characters; Postgres keeps 63. "
                     f"Use a shorter --table.")
        wanted = ident(raw)

    existing = [r[0] for r in conn.execute(
        """SELECT i.relname FROM pg_index x
           JOIN pg_class i ON i.oid = x.indexrelid
           JOIN pg_am am ON am.oid = i.relam
           WHERE x.indrelid = %s::regclass AND am.amname = ANY(%s)""",
        (table, list(INDEX_AMS)),
    )]
    if protect_indexes():
        if args.rebuild and wanted in existing:
            sys.exit(f"error: PROTECT_INDEXES=1 refuses to rebuild {wanted}")
        # Keep every other index too; check_plan still verifies which one runs.
        args.keep_other_indexes = True
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
        with_opts = ", ".join(f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}"
                              for k, v in params.items())
        log(f"building {wanted} ...")
        conn.execute(f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
        conn.execute(f"SET max_parallel_maintenance_workers = {args.parallel_workers}")
        if args.index == "hnsw_halfvec":
            am, key = "hnsw", f"(embedding::halfvec({ds.dim})) {HALFVEC_OPS[ds.metric]}"
        elif args.index == "hnsw_bq":
            am, key = "hnsw", f"(binary_quantize(embedding)::bit({ds.dim})) bit_hamming_ops"
        else:
            opclass = PRISM_OPCLASS[ds.metric] if args.index == "prism" else ds.opclass
            am, key = args.index, f"embedding {opclass}"
        start = time.perf_counter()
        with_clause = f" WITH ({with_opts})" if with_opts else ""
        conn.execute(f"CREATE INDEX {wanted} ON {table} USING {am} ({key}){with_clause}")
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
    sql, nparams = query_sql(ds, args.index, args.k, value if value is not None else args.k)
    plan = "\n".join(r[0] for r in conn.execute("EXPLAIN " + sql, (query_vec,) * nparams))
    at = f" at {guc}={value}" if guc else ""
    if index_name and index_name not in plan:
        sys.exit(f"error: the query plan doesn't use {index_name}{at}:\n{plan}\n"
                 f"hint: another index on the table won on cost. Drop it (omit "
                 f"--keep-other-indexes) or force this one through the query shape.")
    if not index_name and "Index" in plan:
        sys.exit(f"error: expected an exact sequential scan:\n{plan}")


# --- memory guard ------------------------------------------------------------
#
# Every measurement here is meant to be fully in memory. prewarm_or_die checks
# that once, before a phase; these checks hold it for every sweep point, after
# its timing window, so a point that ran partly from disk fails instead of
# becoming a plausible-looking slow number.

VMSTAT_KEYS = ("pswpin", "pswpout", "pgmajfault")


def vmstat() -> dict | None:
    """The swap and major-fault counters from /proc/vmstat (None off Linux)."""
    try:
        with open("/proc/vmstat") as f:
            pairs = (line.split() for line in f)
            return {k: int(v) for k, v in pairs if k in VMSTAT_KEYS}
    except FileNotFoundError:
        return None


def require_no_swap() -> None:
    try:
        with open("/proc/swaps") as f:
            active = f.read().splitlines()[1:]
    except FileNotFoundError:
        return
    if active:
        sys.exit("error: swap is enabled, so a run that does not fit in RAM would slow down "
                 "silently instead of failing. Run `sudo swapoff -a` (bench/cloudlab/setup_node.sh "
                 "does) or unset REQUIRE_NO_SWAP.\n" + "\n".join(active))


def memory_check(args, ds, before: dict | None) -> dict:
    """Fail the run if the sweep point swapped or its relations left shared_buffers."""
    out: dict = {}
    after = vmstat()
    if before is not None and after is not None:
        out = {k: after[k] - before[k] for k in VMSTAT_KEYS}
        if out["pswpin"] or out["pswpout"]:
            sys.exit(f"error: the machine swapped during this sweep point ({out}); the data does "
                     f"not fit in memory alongside everything else running")
    if args.index != "rtt":
        with connect(args.dsn) as conn:
            res = residency(conn, ds.table)
        if res is not None:
            blocks, cached = res
            out["blocks"], out["cached"] = blocks, cached
            # --no-save points are the build phases' sanity queries, run on a
            # deliberately small build-time buffer pool; nothing they measure
            # is kept. Every saved point must have run fully resident.
            if cached < blocks and not args.no_save:
                sys.exit(f"error: {blocks - cached} of {blocks} blocks of {ds.table} and its indexes "
                         f"left shared_buffers during the sweep point; raise shared_buffers")
    return out


def residency(conn, table: str) -> tuple[int, int] | None:
    """(blocks, blocks in shared_buffers) over the table, its TOAST table and
    all their indexes, or None without pg_buffercache. Same relation set as
    run_experiment.sh's prewarm_or_die.

    TOAST matters here: a 1536-d vector is ~6 KB, past the ~2 KB TOAST
    threshold, so the heap holds only a pointer and the vectors themselves live
    in the TOAST relation -- which anything that reads vectors from the table
    (PRISM's rerank, an exact scan) depends on."""
    if not conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_buffercache'").fetchone():
        return None
    row = conn.execute(
        """SELECT coalesce(sum(blocks), 0)::bigint, coalesce(sum(cached), 0)::bigint FROM (
             SELECT pg_relation_size(c.oid) / current_setting('block_size')::int AS blocks,
                    (SELECT count(*) FROM pg_buffercache b
                      WHERE b.relfilenode = pg_relation_filenode(c.oid)
                        AND b.reldatabase = (SELECT oid FROM pg_database WHERE datname = current_database())
                        AND b.relforknumber = 0) AS cached
             FROM pg_class c
             WHERE c.oid IN (
                     SELECT %(t)s::regclass::oid
                     UNION ALL SELECT reltoastrelid FROM pg_class
                               WHERE oid = %(t)s::regclass AND reltoastrelid <> 0
                     UNION ALL SELECT indexrelid FROM pg_index WHERE indrelid IN (
                       %(t)s::regclass,
                       (SELECT reltoastrelid FROM pg_class WHERE oid = %(t)s::regclass)))
               AND pg_relation_size(c.oid) > 0) s""",
        {"t": table}).fetchone()
    return int(row[0]), int(row[1])


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
    query, nparams = query_sql(ds, index, k, value)
    slices = np.array_split(np.arange(len(test)), clients)
    jobs = [(dsn, query, guc, value, k, test[s], duration, nparams) for s in slices]

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


def query_worker(dsn, query, guc, value, k, vectors, duration=0.0, nparams=1, barrier=None):
    ids = np.full((len(vectors), k), -1, dtype=np.int64)
    latencies = []
    with connect(dsn) as conn, conn.cursor() as cur:
        if guc:
            conn.execute(f"SET {guc} = {int(value)}")
        cur.execute(query, (vectors[0],) * nparams, prepare=True)  # prepare before the clock starts
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
                cur.execute(query, (vec,) * nparams, prepare=True)
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
    recall = f"{r['recall']:>8.4f}" if r["recall"] is not None else f"{'-':>8}"
    print(f"{value:>16} {recall} {r['qps']:>10.1f} {lat['p50']:>9.3f} {lat['p95']:>9.3f} {lat['p99']:>9.3f}",
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


def server_is_local(host: str | None) -> bool:
    """True for a Unix socket or a loopback address: the client shares the server's machine."""
    return not host or host.startswith("/") or host in ("localhost", "127.0.0.1", "::1")


def vector_so_digest() -> str | None:
    """Short SHA-256 of the installed vector.so. The submodule SHA says what the
    source tree holds, which is not necessarily what the server loads."""
    try:
        pkglibdir = subprocess.run([str(REPO_ROOT / "postgres/install/bin/pg_config"), "--pkglibdir"],
                                   capture_output=True, text=True, check=True).stdout.strip()
        return hashlib.sha256(Path(pkglibdir, "vector.so").read_bytes()).hexdigest()[:12]
    except (OSError, subprocess.CalledProcessError):
        return None


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
