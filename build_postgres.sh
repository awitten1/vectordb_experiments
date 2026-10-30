#!/usr/bin/env bash
# Build Postgres from source, then build pgvector against it.
# Everything is installed under ./postgres/install
set -euo pipefail

PG_VERSION="${PG_VERSION:-17.11}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/postgres/source"
BUILD="$SRC/build"
PREFIX="$ROOT/postgres/install"
JOBS="$(getconf _NPROCESSORS_ONLN)"

for tool in meson ninja; do
  if ! command -v "$tool" >/dev/null; then
    echo "error: $tool not found (macOS: brew install meson ninja, Debian/Ubuntu: apt install meson ninja-build)" >&2
    exit 1
  fi
done

# 1. Download and unpack the Postgres source release
if [ ! -d "$SRC" ]; then
  TARBALL="postgresql-$PG_VERSION.tar.gz"
  URL="https://ftp.postgresql.org/pub/source/v$PG_VERSION/$TARBALL"
  mkdir -p "$ROOT/postgres"
  cd "$ROOT/postgres"
  curl -fSL -o "$TARBALL" "$URL"
  curl -fsSL -o "$TARBALL.sha256" "$URL.sha256"
  # sha256sum on Linux, shasum on macOS
  if command -v sha256sum >/dev/null; then
    sha256sum -c "$TARBALL.sha256"
  else
    shasum -a 256 -c "$TARBALL.sha256"
  fi
  # Unpack into a temp dir first so an interrupted run doesn't leave a partial $SRC
  rm -rf source.tmp
  mkdir source.tmp
  tar -xzf "$TARBALL" -C source.tmp --strip-components=1
  mv source.tmp "$SRC"
  rm -f "$TARBALL" "$TARBALL.sha256"
fi

# 2. Build and install Postgres. Meson also writes $BUILD/compile_commands.json,
# which clangd finds for files under $SRC (pgvector uses ./compile_flags.txt)
if [ ! -d "$BUILD" ]; then
  meson setup "$BUILD" "$SRC" --prefix="$PREFIX" -Dicu=disabled
fi
ninja -C "$BUILD"
ninja -C "$BUILD" install

# 3. Build and install pgvector into that Postgres
cd "$ROOT/pgvector"
make -j"$JOBS" PG_CONFIG="$PREFIX/bin/pg_config"
make install PG_CONFIG="$PREFIX/bin/pg_config"

echo "Done. Postgres binaries are in $PREFIX/bin"
