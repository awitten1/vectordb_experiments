#!/usr/bin/env bash
# Build Postgres from source, then build pgvector against it.
# Everything is installed under ./postgres/install
set -euo pipefail

PG_VERSION="${PG_VERSION:-18.6}"
# FRAME_POINTERS=1 builds with frame pointers so `perf record --call-graph fp`
# gets complete stacks (costs ~1-2% throughput). Re-run without it to switch back.
FRAME_POINTERS="${FRAME_POINTERS:-0}"
FP_FLAGS=""
if [ "$FRAME_POINTERS" = 1 ]; then
  FP_FLAGS="-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
fi

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

# A stamp records which version $SRC holds. Without it, bumping PG_VERSION with
# a source tree already present is a silent no-op: the download is skipped and
# the old version gets rebuilt. Refuse rather than delete, because the tree may
# carry local patches (see bench/patches/).
STAMP="$SRC/.pg_version"
if [ -d "$SRC" ] && [ "$(cat "$STAMP" 2>/dev/null)" != "$PG_VERSION" ]; then
  echo "error: $SRC holds Postgres $(cat "$STAMP" 2>/dev/null || echo 'unknown'), not $PG_VERSION." >&2
  echo "       Remove it to switch versions: rm -rf $SRC $PREFIX" >&2
  echo "       Any local patches in that tree are lost, so save them first." >&2
  exit 1
fi

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
  echo "$PG_VERSION" > "$STAMP"
fi

# 2. Build and install Postgres. Meson also writes $BUILD/compile_commands.json,
# which clangd finds for files under $SRC (pgvector uses ./compile_flags.txt)
# PG_BLOCKSIZE=32 builds Postgres with 32 KB pages (meson's blocksize, in kB;
# Postgres's default is 8). It is fixed when the build tree is set up: asking
# an existing tree for another size is refused rather than switching it, since
# a data directory only works with the page size it was created with. Leave it
# unset to keep whatever the tree has.
if [ ! -d "$BUILD" ]; then
  meson setup "$BUILD" "$SRC" --prefix="$PREFIX" -Dicu=disabled ${PG_BLOCKSIZE:+-Dblocksize=$PG_BLOCKSIZE}
elif [ -n "${PG_BLOCKSIZE:-}" ]; then
  have="$(meson introspect "$BUILD" --buildoptions |
    python3 -c 'import json, sys; print(next(o["value"] for o in json.load(sys.stdin) if o["name"] == "blocksize"))')"
  if [ "$have" != "$PG_BLOCKSIZE" ]; then
    echo "error: $BUILD is set up for $have kB pages, not $PG_BLOCKSIZE." >&2
    echo "       Use a separate checkout for another page size." >&2
    exit 1
  fi
fi
# Applied on every run so FRAME_POINTERS can be toggled; ninja rebuilds if it changed
meson configure "$BUILD" -Dc_args="$FP_FLAGS"
ninja -C "$BUILD"
ninja -C "$BUILD" install

# 3. Build and install pgvector into that Postgres
cd "$ROOT/pgvector"
# make doesn't track flag changes, so always rebuild from clean (pgvector is small)
make clean PG_CONFIG="$PREFIX/bin/pg_config" >/dev/null
make -j"$JOBS" PG_CONFIG="$PREFIX/bin/pg_config" OPTFLAGS="-march=native $FP_FLAGS"
make install PG_CONFIG="$PREFIX/bin/pg_config"

echo "Done. Postgres binaries are in $PREFIX/bin"
