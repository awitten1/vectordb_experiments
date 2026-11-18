"""Datasets for the ANN benchmark.

Two kinds of dataset names are supported:

  synthetic:<n>x<d>[:<metric>]   e.g. synthetic:100000x128, synthetic:18000000x128
      Clustered Gaussian vectors generated from a seed. Exact neighbors are
      computed by brute force in numpy and cached in bench/data/.

  bigann:<n>                      e.g. bigann:18000000
      The first <n> vectors of the BIGANN/SIFT1B base set (128-d SIFT
      descriptors, the same distribution as sift-128-euclidean). Only the
      needed prefix of the 98 GB gzip stream is downloaded; queries come from
      the official query file and exact neighbors are computed locally over
      the chosen prefix.

  qdrant:<name>                   e.g. qdrant:dbpedia-openai-1M-1536-angular
      A set from Qdrant's vector-db-benchmark (see QDRANT_SETS). Served under
      L2 after asserting unit norms; exact top-100 is recomputed locally and
      checked against the published top-10.

  <ann-benchmarks name>           e.g. sift-128-euclidean, glove-100-angular
      Downloaded once from ann-benchmarks.com into bench/data/. These files
      ship with precomputed exact neighbors.

Large synthetic sets are built to scale: the base vectors are generated in
chunks straight into a .npy file and then memory-mapped, so a 9 GB train set
never has to fit in the client's address space as a live array, and exact
neighbors are computed blockwise (see exact_neighbors).
"""

import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

DATA_DIR = Path(__file__).resolve().parent / "data"
ANN_BENCHMARKS_URL = "http://ann-benchmarks.com/{name}.hdf5"
BIGANN_BASE_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz"
BIGANN_QUERY_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz"

# bvecs record: int32 dimension, then that many uint8 components.
BVECS_DIM = 128
BVECS_REC = 4 + BVECS_DIM

# Number of exact neighbors stored per query (k must be <= this).
GT_K = 100

# Rows generated per RNG draw. Fixed so a given seed reproduces the same data
# regardless of how much memory the generating machine had.
GEN_CHUNK = 1_000_000

# Target size of the [queries, train_block] distance matrix in exact_neighbors.
GT_BLOCK_BYTES = 256 << 20

# metric -> (pgvector operator class, distance operator)
METRICS = {
    "euclidean": ("vector_l2_ops", "<->"),
    "angular": ("vector_cosine_ops", "<=>"),
    "ip": ("vector_ip_ops", "<#>"),
}


@dataclass
class Dataset:
    name: str
    train: np.ndarray  # float32 [n, d]; may be a read-only memmap
    test: np.ndarray  # float32 [q, d]
    neighbors: np.ndarray  # int [q, >=GT_K], ids are row indexes into train
    metric: str
    # Use a different table than the dataset's own. Lets a second copy of the
    # same data carry a different index, so two index types can be measured
    # without competing for the planner's choice on one table.
    table_override: str | None = None

    @property
    def dim(self) -> int:
        return self.train.shape[1]

    @property
    def table(self) -> str:
        """Postgres table name for this dataset."""
        if self.table_override:
            return self.table_override
        return "items_" + re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")

    @property
    def opclass(self) -> str:
        return METRICS[self.metric][0]

    @property
    def operator(self) -> str:
        return METRICS[self.metric][1]


def load(name: str, seed: int = 42, num_queries: int = 1000) -> Dataset:
    if name.startswith("synthetic:"):
        return _load_synthetic(name, seed, num_queries)
    if name.startswith("bigann:"):
        return _load_bigann(name, num_queries)
    if name.startswith("qdrant:"):
        return _load_qdrant(name, num_queries)
    return _load_ann_benchmarks(name)


# --- synthetic ---------------------------------------------------------------


def _load_synthetic(name: str, seed: int, num_queries: int) -> Dataset:
    m = re.fullmatch(r"synthetic:(\d+)x(\d+)(?::(\w+))?", name)
    if not m:
        raise ValueError(f"bad synthetic dataset spec {name!r}, expected synthetic:<n>x<d>[:<metric>]")
    n, d = int(m.group(1)), int(m.group(2))
    metric = m.group(3) or "euclidean"
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}, expected one of {list(METRICS)}")

    # The seed changes the data, so it is part of the name. The query set is
    # drawn from its own RNG stream, so asking for more queries later only
    # regenerates the queries and ground truth, not the (already loaded) base
    # vectors.
    full_name = f"synthetic_{n}x{d}_{metric}_s{seed}"
    train_path = DATA_DIR / f"{full_name}_train.npy"
    meta_path = DATA_DIR / f"{full_name}_meta.npz"

    if train_path.exists() and meta_path.exists():
        z = np.load(meta_path)
        if len(z["test"]) >= num_queries:
            train = np.load(train_path, mmap_mode="r")
            return Dataset(full_name, train, z["test"], z["neighbors"], metric)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    num_clusters = max(1, int(np.sqrt(n)))

    if not train_path.exists():
        print(f"generating {full_name} ({n * d * 4 / 1e9:.1f} GB) ...", file=sys.stderr)
        tmp = train_path.with_suffix(".npy.part")
        try:
            out = open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, d))
            # centers and the base vectors come off one stream, in that order
            rng = np.random.default_rng(seed)
            centers = rng.standard_normal((num_clusters, d), dtype=np.float32)  # overlapping clusters
            for start in range(0, n, GEN_CHUNK):
                count = min(GEN_CHUNK, n - start)
                labels = rng.integers(0, num_clusters, size=count)
                out[start : start + count] = centers[labels] + rng.standard_normal((count, d), dtype=np.float32)
            out.flush()
            del out
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.rename(train_path)
    else:
        rng = np.random.default_rng(seed)
        centers = rng.standard_normal((num_clusters, d), dtype=np.float32)

    train = np.load(train_path, mmap_mode="r")

    # Queries come from their own stream so they don't perturb the base vectors
    qrng = np.random.default_rng([seed, 1])
    labels = qrng.integers(0, num_clusters, size=num_queries)
    test = centers[labels] + qrng.standard_normal((num_queries, d), dtype=np.float32)

    print(f"computing exact neighbors for {num_queries} queries over {n} vectors ...", file=sys.stderr)
    neighbors = exact_neighbors(train, test, metric, min(GT_K, n))
    np.savez(meta_path, test=test, neighbors=neighbors)
    return Dataset(full_name, train, test, neighbors, metric)


# --- bigann (SIFT1B) ---------------------------------------------------------


def _bvecs_to_float32(buf: bytes, dim: int = BVECS_DIM) -> np.ndarray:
    """Decode a whole number of bvecs records into float32 [rows, dim]."""
    rec = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 4 + dim)
    stated = rec[:, :4].copy().view(np.int32).ravel()
    if stated.size and (stated != dim).any():
        raise ValueError(f"bvecs record with unexpected dimension {stated[stated != dim][0]}")
    return rec[:, 4:].astype(np.float32)


def _load_bigann(name: str, num_queries: int) -> Dataset:
    m = re.fullmatch(r"bigann:(\d+)", name)
    if not m:
        raise ValueError(f"bad bigann dataset spec {name!r}, expected bigann:<n>")
    n = int(m.group(1))
    if n > 1_000_000_000:
        raise ValueError(f"bigann has 1e9 vectors, asked for {n}")
    metric = "euclidean"  # SIFT descriptors are benchmarked under L2

    full_name = f"bigann_{n}"
    train_path = DATA_DIR / f"{full_name}_train.npy"
    meta_path = DATA_DIR / f"{full_name}_meta.npz"

    if train_path.exists() and meta_path.exists():
        z = np.load(meta_path)
        if len(z["test"]) >= num_queries:
            train = np.load(train_path, mmap_mode="r")
            return Dataset(full_name, train, z["test"], z["neighbors"], metric)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        want = n * BVECS_REC
        print(f"streaming first {n} bigann vectors ({n * BVECS_DIM * 4 / 1e9:.1f} GB as float32) "
              f"from {want / 1e9:.1f} GB of the gzip stream ...", file=sys.stderr)
        tmp = train_path.with_suffix(".npy.part")
        try:
            out = open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, BVECS_DIM))
            got = 0  # rows decoded so far
            with _bigann_stream(BIGANN_BASE_URL) as fh:
                carry = b""
                while got < n:
                    # read in whole-record multiples; keep any partial record
                    chunk = fh.read(GEN_CHUNK * BVECS_REC // 8)
                    if not chunk:
                        raise EOFError(f"stream ended after {got} of {n} vectors")
                    buf = carry + chunk
                    rows = min(len(buf) // BVECS_REC, n - got)
                    carry = buf[rows * BVECS_REC:]
                    if rows:
                        out[got : got + rows] = _bvecs_to_float32(buf[: rows * BVECS_REC])
                        got += rows
                        print(f"\r  {got * 100 / n:5.1f}%", end="", file=sys.stderr)
            print(file=sys.stderr)
            out.flush()
            del out
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.rename(train_path)

    train = np.load(train_path, mmap_mode="r")

    query_path = DATA_DIR / "bigann_query.bvecs.gz"
    if not query_path.exists():
        _download(BIGANN_QUERY_URL, query_path)
    import gzip
    with gzip.open(query_path, "rb") as fh:
        test = _bvecs_to_float32(fh.read())[:num_queries]

    # The official ground truth is for all 1e9 base vectors, so it does not
    # apply to a prefix; compute it over the prefix we actually loaded.
    print(f"computing exact neighbors for {len(test)} queries over {n} vectors ...", file=sys.stderr)
    neighbors = exact_neighbors(train, test, metric, min(GT_K, n))
    np.savez(meta_path, test=test, neighbors=neighbors)
    return Dataset(full_name, train, test, neighbors, metric)


def _bigann_stream(url: str):
    """Open the gzip base file as a decompressed, read-once binary stream."""
    import gzip
    return gzip.GzipFile(fileobj=urllib.request.urlopen(url), mode="rb")


def exact_neighbors(train: np.ndarray, test: np.ndarray, metric: str, k: int) -> np.ndarray:
    """Brute-force top-k neighbor ids for each query, sorted nearest first.

    Blocked over `train`, keeping a running top-k, so the [queries, block]
    distance matrix stays near GT_BLOCK_BYTES and `train` may be a memmap far
    larger than RAM. Distances are float32 BLAS (sgemm); for euclidean the
    constant |q|^2 term is dropped since it doesn't affect per-query ranking.
    """
    q = np.ascontiguousarray(test, dtype=np.float32)
    if metric == "angular":
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
    nq = len(q)
    block = max(1, int(GT_BLOCK_BYTES // (4 * nq)))

    best_d = np.full((nq, k), np.inf, dtype=np.float32)
    best_i = np.zeros((nq, k), dtype=np.int64)

    nblocks = (len(train) + block - 1) // block
    for bi, start in enumerate(range(0, len(train), block)):
        x = np.ascontiguousarray(train[start : start + block], dtype=np.float32)
        if metric == "angular":
            x = x / np.linalg.norm(x, axis=1, keepdims=True)

        # -q.x for angular/ip; |x|^2 - 2 q.x for euclidean
        dist = q @ x.T
        np.negative(dist, out=dist)
        if metric == "euclidean":
            dist *= 2.0
            dist += np.einsum("ij,ij->i", x, x)[None, :]

        # Reduce this block to its own top-k, then merge into the running best
        kb = min(k, dist.shape[1])
        part = np.argpartition(dist, kb - 1, axis=1)[:, :kb]
        cat_d = np.concatenate([best_d, np.take_along_axis(dist, part, axis=1)], axis=1)
        cat_i = np.concatenate([best_i, part + start], axis=1)
        sel = np.argpartition(cat_d, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(cat_d, sel, axis=1)
        best_i = np.take_along_axis(cat_i, sel, axis=1)

        if nblocks > 4 and (bi % 20 == 0 or bi == nblocks - 1):
            print(f"\r  {(bi + 1) * 100 / nblocks:5.1f}%", end="", file=sys.stderr)
    print(file=sys.stderr)

    order = np.argsort(best_d, axis=1, kind="stable")
    return np.take_along_axis(best_i, order, axis=1)


# --- Qdrant vector-db-benchmark sets -----------------------------------------

QDRANT_URL = "https://storage.googleapis.com/ann-filtered-benchmark/datasets/{file}"

# name -> tarball. Names follow qdrant/vector-db-benchmark datasets/datasets.json.
# Each tarball holds exactly vectors.npy (float32 [n, d]) and tests.jsonl (one
# {"query", "conditions", "closest_ids", "closest_scores"} object per line).
QDRANT_SETS = {
    "dbpedia-openai-1M-1536-angular": "dbpedia_openai_1M.tgz",
    "dbpedia-openai-100K-1536-angular": "dbpedia_openai_100K.tgz",
}
QDRANT_MEMBERS = {"vectors.npy", "tests.jsonl"}

# How far a vector's norm may stray from 1 before we refuse to treat cosine as L2.
UNIT_NORM_TOL = 1e-3


def _load_qdrant(name: str, num_queries: int) -> Dataset:
    """A Qdrant benchmark set, served under L2.

    These sets are published as cosine. OpenAI ada-002 embeddings are
    unit-normalized, and for unit vectors |q-x|^2 = 2 - 2 q.x, so L2, cosine
    and inner product produce the same ranking. Serving them as L2 lets every
    engine in the comparison compute the same metric (Ginger has no cosine);
    the unit norms are asserted here rather than assumed.

    The published ground truth is only the top 10 under cosine. Exact top-GT_K
    is recomputed under L2 and compared against it as a sanity check.
    """
    key = name.split(":", 1)[1]
    if key not in QDRANT_SETS:
        raise ValueError(f"unknown qdrant dataset {key!r}, expected one of {list(QDRANT_SETS)}")
    full_name = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    metric = "euclidean"
    train_path = DATA_DIR / f"{full_name}_train.npy"
    meta_path = DATA_DIR / f"{full_name}_meta.npz"

    if train_path.exists() and meta_path.exists():
        z = np.load(meta_path)
        if len(z["test"]) >= num_queries:
            train = np.load(train_path, mmap_mode="r")
            return Dataset(full_name, train, z["test"], z["neighbors"], metric)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    src = DATA_DIR / f"{full_name}_src"
    if not (src / "tests.jsonl").exists() or not (train_path.exists() or (src / "vectors.npy").exists()):
        tarball = DATA_DIR / QDRANT_SETS[key]
        if not tarball.exists():
            _download(QDRANT_URL.format(file=QDRANT_SETS[key]), tarball)
        _extract_qdrant(tarball, src)

    if not train_path.exists():
        # The published file is already a C-ordered <f4 .npy; check, then adopt it as is.
        v = np.load(src / "vectors.npy", mmap_mode="r")
        if v.dtype != np.float32 or v.ndim != 2 or not v.flags.c_contiguous:
            raise ValueError(f"{src}/vectors.npy: expected C-ordered float32 [n, d], got {v.dtype} {v.shape}")
        del v
        (src / "vectors.npy").rename(train_path)
    train = np.load(train_path, mmap_mode="r")

    import json
    queries, published = [], []
    with (src / "tests.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            if t.get("conditions"):
                raise ValueError(f"{key}: test {len(queries)} has filter conditions; this loader is unfiltered only")
            queries.append(t["query"])
            published.append(t["closest_ids"])
    if num_queries > len(queries):
        raise ValueError(f"{key} has {len(queries)} test queries, asked for {num_queries}")
    test = np.asarray(queries[:num_queries], dtype=np.float32)
    published = published[:num_queries]
    if test.shape[1] != train.shape[1]:
        raise ValueError(f"{key}: query dim {test.shape[1]} != base dim {train.shape[1]}")

    _assert_unit_norm(train, f"{key} base vectors")
    _assert_unit_norm(test, f"{key} queries")

    print(f"computing exact L2 neighbors for {len(test)} queries over {len(train)} vectors ...", file=sys.stderr)
    neighbors = exact_neighbors(train, test, metric, min(GT_K, len(train)))

    # Queries drawn from the base set would hand every engine a free hit at
    # distance 0. Report it either way; refuse rather than silently patch it.
    d0 = np.linalg.norm(np.asarray(train[neighbors[:, 0]], dtype=np.float32) - test, axis=1)
    dup = int((d0 < 1e-6).sum())
    print(f"  queries identical to a base vector: {dup} of {len(test)}", file=sys.stderr)
    if dup:
        raise ValueError(f"{key}: {dup} queries are also base vectors; the query set is not held out")

    # Published ids are row indexes into vectors.npy (the archive carries no id
    # column), so they are directly comparable.
    k = min(10, min(len(p) for p in published))
    overlap = np.mean([len(set(neighbors[i, :k].tolist()) & set(published[i][:k])) / k
                       for i in range(len(test))])
    print(f"  top-{k} agreement with the published ground truth: {overlap:.4f}", file=sys.stderr)
    if overlap < 0.99:
        print(f"  warning: published and recomputed neighbors disagree more than float32 ties explain",
              file=sys.stderr)

    np.savez(meta_path, test=test, neighbors=neighbors, published_top10_agreement=np.float64(overlap))
    return Dataset(full_name, train, test, neighbors, metric)


def _extract_qdrant(tarball: Path, dest: Path) -> None:
    """Extract exactly the expected members into a fresh directory."""
    import shutil
    import tarfile

    tmp = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(tarball, "r:gz") as tf:
        members = [m for m in tf.getmembers() if not m.isdir()]
        names = {Path(m.name).name for m in members}
        if names != QDRANT_MEMBERS or any(not m.isfile() for m in members):
            raise ValueError(f"{tarball.name}: expected regular files {sorted(QDRANT_MEMBERS)}, "
                             f"found {sorted(m.name for m in members)}")
        print(f"extracting {tarball.name} ...", file=sys.stderr)
        for m in members:
            # Flatten paths: never let a member name choose where it lands.
            with tf.extractfile(m) as fin, (tmp / Path(m.name).name).open("wb") as fout:
                shutil.copyfileobj(fin, fout, 1 << 24)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)


def _assert_unit_norm(x: np.ndarray, what: str) -> None:
    worst = 0.0
    for start in range(0, len(x), GEN_CHUNK):
        n = np.linalg.norm(np.asarray(x[start : start + GEN_CHUNK], dtype=np.float32), axis=1)
        worst = max(worst, float(np.abs(n - 1.0).max()))
    print(f"  {what}: max |norm - 1| = {worst:.2e}", file=sys.stderr)
    if worst > UNIT_NORM_TOL:
        raise ValueError(f"{what} are not unit-normalized (max |norm-1| = {worst:.3g}); "
                         f"L2 would not rank like cosine")


# --- ann-benchmarks ----------------------------------------------------------


def _load_ann_benchmarks(name: str) -> Dataset:
    import h5py

    path = DATA_DIR / f"{name}.hdf5"
    if not path.exists():
        _download(ANN_BENCHMARKS_URL.format(name=name), path)

    with h5py.File(path, "r") as f:
        metric = f.attrs["distance"]
        if isinstance(metric, bytes):
            metric = metric.decode()
        if metric == "dot":
            metric = "ip"
        if metric not in METRICS:
            raise ValueError(f"{name}: unsupported distance {metric!r}")
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int64)
    return Dataset(name, train, test, neighbors, metric)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}", file=sys.stderr)

    # ann-benchmarks.com returns 403 for urllib's default User-Agent
    req = urllib.request.Request(url, headers={"User-Agent": "vectordb-experiments-bench"})
    try:
        with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done * 100 / total:5.1f}% of {total / 1e6:.0f} MB", end="", file=sys.stderr)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print(file=sys.stderr)
    tmp.rename(dest)
