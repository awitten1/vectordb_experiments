"""Datasets for the ANN benchmark.

Two kinds of dataset names are supported:

  synthetic:<n>x<d>[:<metric>]   e.g. synthetic:100000x128, synthetic:50000x64:angular
      Clustered Gaussian vectors generated from a seed. Exact neighbors are
      computed by brute force in numpy and cached in bench/data/.

  <ann-benchmarks name>           e.g. sift-128-euclidean, glove-100-angular
      Downloaded once from ann-benchmarks.com into bench/data/. These files
      ship with precomputed exact neighbors.
"""

import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
ANN_BENCHMARKS_URL = "http://ann-benchmarks.com/{name}.hdf5"

# Number of exact neighbors stored per query (k must be <= this).
GT_K = 100

# metric -> (pgvector operator class, distance operator)
METRICS = {
    "euclidean": ("vector_l2_ops", "<->"),
    "angular": ("vector_cosine_ops", "<=>"),
    "ip": ("vector_ip_ops", "<#>"),
}


@dataclass
class Dataset:
    name: str
    train: np.ndarray  # float32 [n, d]
    test: np.ndarray  # float32 [q, d]
    neighbors: np.ndarray  # int [q, >=GT_K], ids are row indexes into train
    metric: str

    @property
    def dim(self) -> int:
        return self.train.shape[1]

    @property
    def table(self) -> str:
        """Postgres table name for this dataset."""
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
    # regenerates the queries, not the (already loaded) base vectors.
    full_name = f"synthetic_{n}x{d}_{metric}_s{seed}"
    cache = DATA_DIR / f"{full_name}.npz"
    if cache.exists():
        z = np.load(cache)
        if len(z["test"]) >= num_queries:
            return Dataset(full_name, z["train"], z["test"], z["neighbors"], metric)

    print(f"generating {full_name} ...", file=sys.stderr)
    rng = np.random.default_rng(seed)
    num_clusters = max(1, int(np.sqrt(n)))
    centers = rng.standard_normal((num_clusters, d), dtype=np.float32)  # overlapping clusters

    def sample(r: np.random.Generator, count: int) -> np.ndarray:
        labels = r.integers(0, num_clusters, size=count)
        return centers[labels] + r.standard_normal((count, d), dtype=np.float32)

    train = sample(rng, n)
    test = sample(np.random.default_rng([seed, 1]), num_queries)
    print(f"computing exact neighbors for {num_queries} queries ...", file=sys.stderr)
    neighbors = exact_neighbors(train, test, metric, min(GT_K, n))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, train=train, test=test, neighbors=neighbors)
    return Dataset(full_name, train, test, neighbors, metric)


def exact_neighbors(train: np.ndarray, test: np.ndarray, metric: str, k: int) -> np.ndarray:
    """Brute-force top-k neighbor ids for each query, sorted nearest first."""
    if metric == "angular":
        train = train / np.linalg.norm(train, axis=1, keepdims=True)
        test = test / np.linalg.norm(test, axis=1, keepdims=True)
    train_sq = (train.astype(np.float64) ** 2).sum(axis=1) if metric == "euclidean" else None

    # Chunk the queries so the [chunk, n] distance matrix stays around 256MB
    chunk = max(1, (1 << 25) // len(train))
    out = np.empty((len(test), k), dtype=np.int64)
    for start in range(0, len(test), chunk):
        q = test[start : start + chunk].astype(np.float64)
        dots = q @ train.T.astype(np.float64)
        if metric == "euclidean":
            # |q - x|^2 = |q|^2 - 2 q.x + |x|^2; |q|^2 is constant per row
            dist = train_sq[None, :] - 2 * dots
        else:
            # cosine distance on normalized vectors / negative inner product
            dist = -dots
        top = np.argpartition(dist, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(dist, top, axis=1), axis=1, kind="stable")
        out[start : start + chunk] = np.take_along_axis(top, order, axis=1)
    return out


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
