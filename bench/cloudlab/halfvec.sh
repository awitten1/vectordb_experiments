#!/usr/bin/env bash
# halfvec HNSW against float HNSW, on stock pgvector and pgvectorfast, for
# DBpedia-OpenAI 1M (1536-d) and SIFT1M (128-d), then perf profiles of both.
#
# Run from the workstation once both nodes are set up (setup_node.sh) and
# node-0 has a FRAME_POINTERS=1 Postgres with build_extensions.sh run.
#
# Per dataset:
#   1. node-0: load the table, build the float HNSW index and next to it the
#      halfvec expression index (same m=16, ef_construction=64); node-1 at the
#      same time materializes the dataset for its recall ground truth.
#   2. For each arm (stock, fast): serve it, then at 1, 8, 12 and 16 clients
#      run the float sweep and the halfvec sweep back to back from node-1
#      (labels fl-<arm>, hv-<arm>). Float is re-measured here so that every
#      comparison is between runs on the same two machines.
#   3. perf profiles (profile_pgvector.sh) of float and halfvec at the
#      dataset's high-recall ef_search, stock at 1/8/16 clients and
#      pgvectorfast at 8/16 (labels pf-fl-*, pf-hv-*).
#
# Steps are detached and resumable (see lib.sh).
#
#   bench/cloudlab/halfvec.sh            # both datasets
#   DATASETS=sift bench/cloudlab/halfvec.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATASETS="${DATASETS:-dbpedia sift}"
COUNTS="${COUNTS:-1 8 12 16}"

for d in $DATASETS; do
  export ENV_FILE="bench/cloudlab/$d.env"
  # shellcheck source=bench/cloudlab/lib.sh
  . "$HERE/lib.sh"
  require_measurement_state
  common='--m 16 --ef-construction 64 --keep-other-indexes --sweep "$PVF_EF_SWEEP"'

  # 1. Table, float HNSW, halfvec HNSW; the client's copy of the dataset.
  if ! step_done "$SERVER" "hv-$d-build"; then
    start_step "$SERVER" "hv-$d-build" "QUANT_KINDS=hnsw_halfvec bench/run_experiment.sh data build build_quant"
  fi
  if ! step_done "$CLIENT" "hv-$d-data"; then
    # The client has no Postgres build, so call the loader directly.
    # shellcheck disable=SC2016
    start_step "$CLIENT" "hv-$d-data" \
      'cd bench && .venv/bin/python -c "import sys, datasets; ds = datasets.load(sys.argv[1], num_queries=int(sys.argv[2])); print(ds.name, ds.train.shape)" "$DATASET" "$NUM_QUERIES"'
  fi
  wait_step "$SERVER" "hv-$d-build"
  wait_step "$CLIENT" "hv-$d-data"

  # 2. Sweeps.
  for arm in stock fast; do
    serve_step "hv-$d-serve-$arm" "$arm"
    for c in $COUNTS; do
      bench_step "$CLIENT" "hv-$d-$arm-c$c-float" "fl-$arm" "--index hnsw $common --clients $c"
      bench_step "$CLIENT" "hv-$d-$arm-c$c-half" "hv-$arm" "--index hnsw_halfvec $common --clients $c"
    done
  done

  # 3. Profiles at the ef_search that gives recall ~0.99 with float vectors.
  case "$d" in
    dbpedia) efs="40 640"; hi=640 ;;
    sift)    efs="40 160"; hi=160 ;;
  esac
  for kind in hnsw hnsw_halfvec; do
    short=$([ "$kind" = hnsw ] && echo fl || echo hv)
    PROF_INDEX=$kind PROF_LABEL="pf-$short" TAG="$short-$d" PROFILE_DIR="$d-$short" \
      EFS="$efs" FAST_EF=$hi "$HERE/profile_pgvector.sh"
  done
done

say "done; pull with bench/cloudlab/sync.sh pull"
