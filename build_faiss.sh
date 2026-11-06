#!/usr/bin/env bash
# Build FAISS from the submodule, then the C++ benchmark driver against it.
# Everything is installed under ./faiss/install; nothing touches the system.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/faiss"
BUILD="$SRC/build"
PREFIX="$SRC/install"
DRIVER_SRC="$ROOT/bench/faiss"
DRIVER_BUILD="$DRIVER_SRC/build"
JOBS="$(getconf _NPROCESSORS_ONLN)"

[ -f "$SRC/CMakeLists.txt" ] || {
  echo "error: $SRC is empty; run: git submodule update --init faiss" >&2
  exit 1
}
command -v cmake >/dev/null || { echo "error: cmake not found" >&2; exit 1; }

# FAISS_OPT_LEVEL defaults to "generic", which would be an unfair baseline: both
# pgvector and pg_vectorsearch are built -march=native on this Zen 5 part. avx512
# builds generic + AVX-512 variants and dispatches at run time. Fall back to avx2
# if the CPU lacks AVX-512 so the script stays portable.
OPT_LEVEL="${FAISS_OPT_LEVEL:-}"
if [ -z "$OPT_LEVEL" ]; then
  if grep -qm1 avx512f /proc/cpuinfo 2>/dev/null; then OPT_LEVEL=avx512
  elif grep -qm1 avx2 /proc/cpuinfo 2>/dev/null; then OPT_LEVEL=avx2
  else OPT_LEVEL=generic
  fi
fi

# 1. FAISS itself.
#   GPU off:    there is an RTX 5060 Ti here but no CUDA toolkit, and Blackwell
#               needs 12.8+. CPU-only is also what the comparison calls for.
#   MKL off:    not installed; this falls back to OpenBLAS, which is.
#   PYTHON off: the driver is C++, so SWIG is not needed (and is not installed).
cmake -B "$BUILD" "$SRC" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_MKL=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_EXTRAS=OFF \
  -DFAISS_OPT_LEVEL="$OPT_LEVEL" \
  -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=OFF

TARGETS=(faiss)
case "$OPT_LEVEL" in
  avx512) TARGETS+=(faiss_avx2 faiss_avx512) ;;
  avx2)   TARGETS+=(faiss_avx2) ;;
esac
cmake --build "$BUILD" -j"$JOBS" --target "${TARGETS[@]}"
cmake --install "$BUILD"

# 2. The benchmark driver, linked against what we just installed.
cmake -B "$DRIVER_BUILD" "$DRIVER_SRC" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PREFIX"
cmake --build "$DRIVER_BUILD" -j"$JOBS"

echo
echo "Done. FAISS $OPT_LEVEL build in $PREFIX"
echo "Driver: $DRIVER_BUILD/faiss_bench"
