#!/usr/bin/env python3
"""The benchmark's queries as a table, for searches run inside the server.

  gap_queries.py --dataset hackernews --table items_hn_pipnn_ro --index <index> \
      --ef 64 --script bench/profiles/hn-gap/batch.sql

hn_g5_gap.sh measures Postgres without the LAN and without one protocol
round trip and transaction per query: pgbench on the server runs a statement
that answers 100 queries, each a nested-loop rescan of the index. This script:
  1. creates hn_queries (qid int4, v vector(d)) with the first --num-queries
     test vectors, the ones ann_bench uses, unless it already holds them;
  2. writes the pgbench script (--script);
  3. checks with EXPLAIN that the statement is a nested loop over an index
     scan on --index, and fails otherwise;
  4. runs it for qid 0..99 at hnsw.ef_search = --ef and prints recall@10
     against the ground truth, which should match ann_bench's at that ef.
Connects to localhost on $PGPORT, like ann_bench.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # noqa: E402
from ann_bench import binary_copy_rows, connect, DEFAULT_DSN  # noqa: E402

BATCH = 100


def batch_sql(table: str, k: int, lo: str) -> str:
    return (f"SELECT count(*) FROM hn_queries q CROSS JOIN LATERAL "
            f"(SELECT id FROM {table} ORDER BY embedding <-> q.v LIMIT {k}) s "
            f"WHERE q.qid >= {lo} AND q.qid < {lo} + {BATCH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--ef", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--num-queries", type=int, default=1000)
    ap.add_argument("--script", required=True, help="write the pgbench script here")
    a = ap.parse_args()

    ds = datasets.load(a.dataset, num_queries=a.num_queries)
    q = np.ascontiguousarray(ds.test[: a.num_queries], dtype=np.float32)
    n, d = q.shape
    if n < BATCH:
        sys.exit(f"error: {n} queries, fewer than a batch of {BATCH}")
    conn = connect(DEFAULT_DSN)
    conn.autocommit = True
    have = conn.execute("SELECT to_regclass('hn_queries') IS NOT NULL").fetchone()[0]
    if have:
        cnt = conn.execute("SELECT count(*) FROM hn_queries").fetchone()[0]
        if cnt != n:
            sys.exit(f"error: hn_queries holds {cnt} rows, not {n}; drop it by hand to reload")
        print(f"hn_queries already holds {n} queries")
    else:
        conn.execute(f"CREATE TABLE hn_queries (qid int4 PRIMARY KEY, v vector({d}) NOT NULL)")
        with conn.cursor() as cur, cur.copy("COPY hn_queries (qid, v) FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.write(b"PGCOPY\n\xff\r\n\x00" + b"\x00" * 8)
            cp.write(binary_copy_rows(q, 0))
            cp.write(b"\xff\xff")
        conn.execute("ANALYZE hn_queries")
        print(f"loaded {n} queries into hn_queries")

    Path(a.script).parent.mkdir(parents=True, exist_ok=True)
    Path(a.script).write_text(f"\\set a random(0, {n - BATCH})\n{batch_sql(a.table, a.k, ':a')};\n")
    print(f"wrote {a.script}")

    conn.execute(f"SET hnsw.ef_search = {a.ef}")
    plan = "\n".join(r[0] for r in conn.execute("EXPLAIN " + batch_sql(a.table, a.k, "0")))
    print(plan)
    if "Nested Loop" not in plan or f"Index Scan using {a.index}" not in plan:
        sys.exit(f"error: the batch statement is not a nested loop over an index scan on {a.index}")

    rows = conn.execute(
        f"SELECT q.qid, s.id FROM hn_queries q CROSS JOIN LATERAL "
        f"(SELECT id FROM {a.table} ORDER BY embedding <-> q.v LIMIT {a.k}) s "
        f"WHERE q.qid < {BATCH}").fetchall()
    got = {}
    for qid, rid in rows:
        got.setdefault(qid, set()).add(rid)
    rec = np.mean([len(got.get(i, set()) & set(ds.neighbors[i][: a.k].tolist())) / a.k for i in range(BATCH)])
    print(f"recall@{a.k} of the batch statement, qid 0..{BATCH - 1}, ef_search {a.ef}: {rec:.4f}")


if __name__ == "__main__":
    main()
