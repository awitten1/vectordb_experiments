#!/usr/bin/env bash
# Build Ginger from the submodule together with the benchmark driver in
# bench/ginger. Nothing is installed; the binary lands in bench/ginger/build.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/bench/ginger"
BUILD="$SRC/build"
JOBS="$(getconf _NPROCESSORS_ONLN)"

[ -f "$ROOT/ginger/CMakeLists.txt" ] || {
  echo "error: $ROOT/ginger is empty; run: git submodule update --init ginger" >&2
  exit 1
}
command -v cmake >/dev/null || { echo "error: cmake not found" >&2; exit 1; }
pkg-config --exists jemalloc || {
  echo "error: jemalloc not found (apt install libjemalloc-dev)" >&2
  exit 1
}

# Ginger FetchContent's its dependencies (abseil, protobuf, highway, oneTBB,
# KaMinPar, ...) on first configure, which takes a while and needs network.
cmake -B "$BUILD" "$SRC" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$JOBS" --target ginger_bench

echo
"$BUILD/ginger_bench" --version
echo "Driver: $BUILD/ginger_bench"
