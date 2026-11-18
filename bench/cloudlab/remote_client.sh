#!/usr/bin/env bash
# Measure what running the benchmark client on the server's own machine costs.
#
# Run from the workstation. For each pgvector arm (fast, then stock) the server
# comes up once on node-0, and the same 8-client HNSW sweep runs twice back to
# back on that warm server: once with the client on node-1 over the private
# LAN, once with it on node-0 itself. The order of the two flips between arms
# so slow drift cannot favor one placement. The round-trip floor (index=rtt) is
# then measured from both places on the stock arm.
#
# A remote client frees the server's cores but adds a network hop; the rtt rows
# measure that hop, so the two effects can be told apart.
#
# Steps are detached and resumable (see lib.sh): re-run to continue.
#
#   bench/cloudlab/remote_client.sh
set -euo pipefail
# shellcheck source=bench/cloudlab/lib.sh
. "$(dirname "$0")/lib.sh"

require_measurement_state

serve_step rc-serve-fast fast
bench_step "$CLIENT" rc-fast-remote rc-fast-remote "$HNSW_ARGS --clients 8"
bench_step "$SERVER" rc-fast-local rc-fast-local "$HNSW_ARGS --clients 8"

serve_step rc-serve-stock stock
bench_step "$SERVER" rc-stock-local rc-stock-local "$HNSW_ARGS --clients 8"
bench_step "$CLIENT" rc-stock-remote rc-stock-remote "$HNSW_ARGS --clients 8"

for c in 1 8; do
  bench_step "$CLIENT" "rc-rtt-c$c-remote" rc-rtt-remote "--index rtt --clients $c"
  bench_step "$SERVER" "rc-rtt-c$c-local" rc-rtt-local "--index rtt --clients $c"
done

step "$SERVER" rc-stop "./run_postgres.sh stop"
say "done; pull with bench/cloudlab/sync.sh pull"
