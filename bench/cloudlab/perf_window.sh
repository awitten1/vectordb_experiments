#!/usr/bin/env bash
# Profile everything running on this node during a steady-state benchmark
# window. Meant for the server node while the client runs elsewhere, so every
# counter belongs to Postgres.
#
#   bench/cloudlab/perf_window.sh <name> [delay_s] [window_s]
#
# After <delay_s> of warm-up it takes four back-to-back windows of <window_s>,
# each its own perf session so counters are never multiplexed against the
# call-graph sampling:
#   <name>.stat.txt     core counters + DRAM traffic from the IMC (uncore) PMUs
#   <name>.topdown.txt  Top-Down levels 1 and 2 (where the cycles go)
#   <name>.perf.data    system-wide call graphs, 499 Hz (CALLGRAPH: fp, the
#                       default, which wants a FRAME_POINTERS=1 build; lbr, last
#                       branch records, user space only, no rebuild; none, flat)
#   <name>.tlb.txt      page-table walks: completed walks and cycles spent walking
# Output goes to bench/profiles/$PROFILE_DIR (default pgvector). Needs kernel.perf_event_paranoid <= 0
# (system-wide and uncore events) and a FRAME_POINTERS=1 build for usable stacks.
set -euo pipefail

NAME="$1"
DELAY="${2:-10}"
WIN="${3:-20}"
OUT="$(cd "$(dirname "$0")/../.." && pwd)/bench/profiles/${PROFILE_DIR:-pgvector}"
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
# Not every CPU/perf pairing has the level-2 metrics; fall back to level 1.
perf stat -a -o "$OUT/$NAME.topdown.txt" -M TopdownL1,TopdownL2 -- sleep "$WIN" ||
  perf stat -a -o "$OUT/$NAME.topdown.txt" -M TopdownL1 -- sleep "$WIN" ||
  echo "Top-Down metrics unavailable" > "$OUT/$NAME.topdown.txt"
case "${CALLGRAPH:-fp}" in
  none) CG=() ;;
  *) CG=(--call-graph "${CALLGRAPH:-fp}") ;;
esac
perf record -a "${CG[@]}" -F 499 -o "$OUT/$NAME.perf.data" -- sleep "$WIN" 2>"$OUT/$NAME.record.log"
perf stat -a -o "$OUT/$NAME.tlb.txt" \
  -e cycles,ref-cycles,dtlb_load_misses.miss_causes_a_walk,dtlb_load_misses.walk_completed,dtlb_load_misses.walk_duration,dtlb_load_misses.stlb_hit \
  -- sleep "$WIN"
echo "profiles in $OUT/$NAME.*"
