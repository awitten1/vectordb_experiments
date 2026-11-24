#!/usr/bin/env bash
# Build the two non-stock extensions against the Postgres from build_postgres.sh
# (run that first; it also builds and installs stock pgvector).
#
#   - keeps a copy of the stock pgvector vector.so at postgres/stock/vector.so,
#     which run_experiment.sh's pgvectorfast/prism phases swap back in
#   - builds pgvectorfast in-tree (pgvectorfast/vectorfast.so) WITHOUT
#     `make install`: its install would replace vector.control with one bound to
#     $libdir/vectorfast. The harness keeps the database bound to $libdir/vector
#     and swaps the library behind that name, so only Calico's control and SQL
#     files are installed here.
#   - builds and installs pg_vectorsearch (PRISM), unless SKIP_PRISM=1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PREFIX="$ROOT/postgres/install"
PG_CONFIG="$PREFIX/bin/pg_config"
[ -x "$PG_CONFIG" ] || { echo "error: run ./build_postgres.sh first" >&2; exit 1; }
LIB="$("$PG_CONFIG" --pkglibdir)"
EXT="$("$PG_CONFIG" --sharedir)/extension"
JOBS="$(getconf _NPROCESSORS_ONLN)"

# 1. Stock pgvector, as build_postgres.sh just installed it.
mkdir -p "$ROOT/postgres/stock"
if [ -L "$LIB/vector.so" ]; then
  echo "error: $LIB/vector.so is a symlink (a pgvectorfast arm is installed);" >&2
  echo "       re-run ./build_postgres.sh to restore stock pgvector first" >&2
  exit 1
fi
[ "$(nm -D "$LIB/vector.so" | grep -c calico_ || true)" -eq 0 ] ||
  { echo "error: $LIB/vector.so has calico symbols" >&2; exit 1; }
install -m755 "$LIB/vector.so" "$ROOT/postgres/stock/vector.so"
echo "stock pgvector: $ROOT/postgres/stock/vector.so"

# 2. pgvectorfast, in-tree.
make -C "$ROOT/pgvectorfast" clean PG_CONFIG="$PG_CONFIG" >/dev/null
make -C "$ROOT/pgvectorfast" -j"$JOBS" PG_CONFIG="$PG_CONFIG"
for f in calico.control sql/calico--0.1.sql sql/calico--0.2.sql sql/calico--0.1--0.2.sql; do
  install -m644 "$ROOT/pgvectorfast/$f" "$EXT/"
done
[ "$(nm "$ROOT/pgvectorfast/vectorfast.so" | grep -c ' calico_')" -gt 0 ] ||
  { echo "error: vectorfast.so has no calico symbols" >&2; exit 1; }
echo "pgvectorfast: $ROOT/pgvectorfast/vectorfast.so (calico files in $EXT)"

# 3. pg_vectorsearch.
if [ "${SKIP_PRISM:-0}" = 1 ]; then echo "pg_vectorsearch: skipped (SKIP_PRISM=1)"; exit 0; fi
cd "$ROOT/pg_vectorsearch"
if [ ! -d builddir ]; then
  meson setup builddir --buildtype=release -Dnative=true -Dpg_config="$PG_CONFIG"
fi
meson compile -C builddir
meson install -C builddir
echo "pg_vectorsearch installed into $PREFIX"
