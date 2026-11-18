#!/usr/bin/env bash
# Recall/QPS sweeps at 12 and 16 concurrent clients (Postgres) or query threads
# (Ginger), with the Postgres client on node-1 over the private LAN so the
# server keeps all of node-0's cores. Complements the 8-client sweeps from
# remote_client.sh (pgvector, pgvectorfast) and run_ginger.sh (Ginger), and
# reruns PRISM at 8 with the remote client so all its counts match.
#
# Steps are detached and resumable (see lib.sh): re-run to continue. RUN names
# the server sessions; bump it (RUN=2) if node-0's server stopped since the
# markers were written, so the serve steps run again.
#
#   bench/cloudlab/scaling.sh
set -euo pipefail
# shellcheck source=bench/cloudlab/lib.sh
. "$(dirname "$0")/lib.sh"

COUNTS="${COUNTS:-12 16}"
RUN="${RUN:-1}"
PRISM_TABLE=items_dbpedia_openai_1m_1536_angular_prism

require_measurement_state

for arm in stock fast; do
  serve_step "sc-serve-$arm-r$RUN" "$arm"
  for c in $COUNTS; do
    bench_step "$CLIENT" "sc-$arm-c$c" "sc-$arm" "$HNSW_ARGS --clients $c"
  done
done

serve_step "sc-serve-prism-r$RUN" stock "$PRISM_TABLE"
for c in 8 $COUNTS; do
  # shellcheck disable=SC2016
  bench_step "$CLIENT" "sc-prism-c$c" sc-prism \
    "--index prism --table $PRISM_TABLE --sweep \"\$NPROBE_SWEEP\" --clients $c"
done
for c in $COUNTS; do
  bench_step "$CLIENT" "sc-rtt-c$c" sc-rtt "--index rtt --clients $c"
done
step "$SERVER" "sc-stop-r$RUN" "./run_postgres.sh stop"

# Ginger: node-1 alone, Postgres stopped everywhere. Indexes are reused.
step "$CLIENT" sc-ginger "THREADS=${COUNTS// /,} bench/run_ginger.sh recommended float-rerank"

say "done; pull with bench/cloudlab/sync.sh pull"
