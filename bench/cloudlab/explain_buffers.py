#!/usr/bin/env python3
"""Per-query buffer accesses of the benchmark's HNSW query, from EXPLAIN.

For each hnsw.ef_search value, runs EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) on
the first N test queries (after one warm-up query) and reports the mean shared
buffer hits, reads and server-side execution time. With every block resident,
hits are page accesses: for pgvector HNSW roughly one per element visited plus
its neighbor tuple, which turns into bytes moved per query.

  bench/.venv/bin/python bench/cloudlab/explain_buffers.py \
      --dataset qdrant:dbpedia-openai-1M-1536-angular --ef 40,640 --n 100 --dsn ...
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import datasets  # noqa: E402
import psycopg  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--ef", default="40,640")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--index", choices=["hnsw", "hnsw_halfvec"], default="hnsw",
                   help="hnsw_halfvec: the query ann_bench.py sends to the halfvec expression index")
    p.add_argument("--table", help="table to query (default: the dataset's own)")
    p.add_argument("--dsn", default="host=localhost port=5433 user=postgres dbname=postgres")
    args = p.parse_args()

    ds = datasets.load(args.dataset, num_queries=args.n + 1)
    if args.index == "hnsw_halfvec":
        order = f"embedding::halfvec({ds.dim}) {ds.operator} %s::halfvec({ds.dim})"
    else:
        order = f"embedding {ds.operator} %s"
    table = args.table or ds.table
    sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT id FROM {table} ORDER BY {order} LIMIT {args.k}"
    with psycopg.connect(args.dsn, autocommit=True) as conn:
        register_vector(conn)
        conn.execute("SET enable_seqscan = off")
        print(f"{'ef_search':>10} {'hits/query':>11} {'reads':>7} {'exec ms':>8} {'MiB/query':>10}")
        for ef in (int(v) for v in args.ef.split(",")):
            conn.execute(f"SET hnsw.ef_search = {ef}")
            conn.execute(sql, (ds.test[0],)).fetchone()  # warm-up
            hits, reads, ms = [], [], []
            for q in ds.test[1 : args.n + 1]:
                plan = conn.execute(sql, (q,)).fetchone()[0][0]
                top = plan["Plan"]
                if "Index Scan" not in json.dumps(plan):
                    sys.exit(f"error: plan at ef_search={ef} is not an index scan")
                hits.append(top["Shared Hit Blocks"])
                reads.append(top["Shared Read Blocks"])
                ms.append(plan["Execution Time"])
            h = statistics.mean(hits)
            print(f"{ef:>10} {h:>11.0f} {statistics.mean(reads):>7.1f} {statistics.mean(ms):>8.2f} "
                  f"{h * 8192 / 2**20:>10.2f}", flush=True)


if __name__ == "__main__":
    main()
