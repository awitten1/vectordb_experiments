#!/usr/bin/env bash
# Profile everything running on this node during a steady-state benchmark
# window. Meant for the server node while the client runs elsewhere, so every
# counter belongs to Postgres.
#
#   bench/cloudlab/perf_window.sh <name> [delay_s] [window_s]
#
# After <delay_s> of warm-up it takes three back-to-back windows of <window_s>,
# each its own perf session so counters are never multiplexed against the
# call-graph sampling:
#   <name>.stat.txt     core counters + DRAM traffic from the IMC (uncore) PMUs
#   <name>.topdown.txt  Top-Down levels 1 and 2 (where the cycles go)
#   <name>.perf.data    system-wide call graphs (frame pointers), 499 Hz
# Output goes to bench/profiles/pgvector/. Needs kernel.perf_event_paranoid <= 0
# (system-wide and uncore events) and a FRAME_POINTERS=1 build for usable stacks.
set -euo pipefail

NAME="$1"
DELAY="${2:-10}"
WIN="${3:-20}"
OUT="$(cd "$(dirname "$0")/../.." && pwd)/bench/profiles/pgvector"
mkdir -p "$OUT"

[ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 0 ] ||
  { echo "error: run: sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0" >&2; exit 1; }

IMC=""
for d in /sys/bus/event_source/devices/uncore_imc_*; do
  n="$(basename "$d")"
  IMC+="${IMC:+,}$n/cas_count_read/,$n/cas_count_write/"
done

sleep "$DELAY"
perf stat -a -o "$OUT/$NAME.stat.txt" \
  -e cycles,instructions,ref-cycles,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,branch-misses,context-switches,task-clock \
  -e "$IMC" -- sleep "$WIN"
perf stat -a -o "$OUT/$NAME.topdown.txt" -M TopdownL1,TopdownL2 -- sleep "$WIN"
perf record -a -g --call-graph fp -F 499 -o "$OUT/$NAME.perf.data" -- sleep "$WIN" 2>"$OUT/$NAME.record.log"
echo "profiles in $OUT/$NAME.*"
