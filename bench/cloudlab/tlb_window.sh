#!/usr/bin/env bash
# Attribute the server's dTLB misses during a steady-state benchmark window.
# Meant for the server node while the client runs elsewhere, so every counter
# belongs to Postgres.
#
#   PERIOD=2003 PROFILE_DIR=hn-tlb bench/cloudlab/tlb_window.sh <name> [delay_s] [window_s]
#
# After <delay_s> of warm-up, back-to-back windows of <window_s>, each its own
# perf session with at most four programmable events, meant to avoid
# multiplexing (Haswell has four counters per hyperthread). In practice perf
# still multiplexed: the NMI watchdog takes the fixed cycles counter, so
# cycles lands on a programmable one (~83% coverage), and the 0xD0-0xD3 load
# events are scheduled exclusively across hyperthread siblings (a Haswell
# erratum workaround), ~60%. trav_window.sh turns the watchdog off.
#   <name>.tlbstat1..4.csv  system-wide counts (perf stat -x,):
#     1 load walks started, completed on 4 KB pages, on 2 MB pages, walk cycles
#     2 STLB hits on 4 KB / 2 MB pages, PDE-cache misses, retired loads that
#       missed the STLB
#     3 where the walker's page-table reads were served: L1, L2, L3, DRAM
#     4 the same walk counts for stores
#   <name>.stlb.data        PEBS samples of retired loads and stores that
#                           missed the STLB, one per PERIOD, with the exact
#                           instruction and the data address (perf record -d)
#   <name>.stlb.txt.gz      those samples as text (postgres only)
#   <name>.stlb.callers.txt who calls the functions that miss (LBR stacks, if
#                           the CPU allows them together with PEBS)
#   <name>.maps/<pid>.{start,end}  /proc/<pid>/maps of every postgres process
#                           around the sampling window, to place the addresses
#   <name>.smaps.txt        smaps of the busiest backend (page size per mapping)
# Output goes to bench/profiles/$PROFILE_DIR (default tlb). Needs
# kernel.perf_event_paranoid <= 0.
set -euo pipefail

NAME="$1"
DELAY="${2:-20}"
WIN="${3:-10}"
PERIOD="${PERIOD:-2003}"
OUT="$(cd "$(dirname "$0")/../.." && pwd)/bench/profiles/${PROFILE_DIR:-tlb}"
mkdir -p "$OUT/$NAME.maps"

[ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 0 ] ||
  { echo "error: run: sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0" >&2; exit 1; }

# ev <perf name> <event=..,umask=..>: the named event if this perf knows it,
# else the same Haswell encoding as a raw event under that name.
ev() {
  if perf stat -e "$1" -- true >/dev/null 2>&1; then printf '%s' "$1"
  else printf 'cpu/%s,name=%s/' "$2" "$1"; fi
}
G1="$(ev dtlb_load_misses.miss_causes_a_walk event=0x08,umask=0x01),$(ev dtlb_load_misses.walk_completed_4k event=0x08,umask=0x02),$(ev dtlb_load_misses.walk_completed_2m_4m event=0x08,umask=0x04),$(ev dtlb_load_misses.walk_duration event=0x08,umask=0x10)"
G2="$(ev dtlb_load_misses.stlb_hit_4k event=0x08,umask=0x20),$(ev dtlb_load_misses.stlb_hit_2m event=0x08,umask=0x40),$(ev dtlb_load_misses.pde_cache_miss event=0x08,umask=0x80),$(ev mem_uops_retired.stlb_miss_loads event=0xd0,umask=0x11)"
G3="$(ev page_walker_loads.dtlb_l1 event=0xbc,umask=0x11),$(ev page_walker_loads.dtlb_l2 event=0xbc,umask=0x12),$(ev page_walker_loads.dtlb_l3 event=0xbc,umask=0x14),$(ev page_walker_loads.dtlb_memory event=0xbc,umask=0x18)"
G4="$(ev dtlb_store_misses.miss_causes_a_walk event=0x49,umask=0x01),$(ev dtlb_store_misses.walk_completed_4k event=0x49,umask=0x02),$(ev dtlb_store_misses.walk_completed_2m_4m event=0x49,umask=0x04),$(ev dtlb_store_misses.walk_duration event=0x49,umask=0x10)"

snap() {  # snap <start|end>
  local p
  for p in $(pgrep -x postgres); do cat "/proc/$p/maps" > "$OUT/$NAME.maps/$p.$1" 2>/dev/null || true; done
}

echo "period=$PERIOD window=$WIN" > "$OUT/$NAME.params"
sleep "$DELAY"
snap start
# The client's backends started with this run, so lifetime %CPU picks a busy one.
busy="$(ps -C postgres -o pid=,pcpu= --sort=-pcpu | awk 'NR == 1 {print $1}')"
cat "/proc/$busy/smaps" > "$OUT/$NAME.smaps.txt" 2>/dev/null || true
cp /proc/meminfo "$OUT/$NAME.meminfo.txt"
echo "busiest backend: $busy" > "$OUT/$NAME.busy.txt"

i=0
for g in "$G1" "$G2" "$G3" "$G4"; do
  i=$(( i + 1 ))
  perf stat -a -x, -o "$OUT/$NAME.tlbstat$i.csv" -e cycles,instructions -e "$g" -- sleep "$WIN" ||
    echo "counter group $i failed: $g" | tee -a "$OUT/$NAME.errors.txt"
done

# PEBS: precise retired loads/stores that missed the STLB, with data addresses.
REC=(-a -d -c "$PERIOD"
     -e "cpu/event=0xd0,umask=0x11,name=stlb_miss_loads/pp"
     -e "cpu/event=0xd0,umask=0x12,name=stlb_miss_stores/pp")
if perf record "${REC[@]}" --call-graph lbr -o "$OUT/$NAME.stlb.data" -- sleep "$WIN" 2>"$OUT/$NAME.stlb.log"; then
  echo "call graph: lbr" >> "$OUT/$NAME.stlb.log"
else
  echo "lbr call graph refused with PEBS; recording without stacks" >> "$OUT/$NAME.stlb.log"
  perf record "${REC[@]}" -o "$OUT/$NAME.stlb.data" -- sleep "$WIN" 2>>"$OUT/$NAME.stlb.log"
fi
snap end

perf script -i "$OUT/$NAME.stlb.data" --comms postgres --hide-call-graph \
  -F comm,pid,event,addr,ip,sym,dso 2>/dev/null | gzip > "$OUT/$NAME.stlb.txt.gz"
perf report -i "$OUT/$NAME.stlb.data" --comms postgres --no-children --sort sym --stdio \
  -g caller,0.5,callee,function,percent --percent-limit 2 > "$OUT/$NAME.stlb.callers.txt" 2>/dev/null || true
echo "samples: $(zcat "$OUT/$NAME.stlb.txt.gz" | wc -l); profiles in $OUT/$NAME.*"
