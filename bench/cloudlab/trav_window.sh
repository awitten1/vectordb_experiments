#!/usr/bin/env bash
# Where pgvectorfast's HNSW traversal spends its time, during a steady-state
# benchmark window on the server (the client runs on another node).
#
#   PERIOD_CYC=500009 PERIOD_LAT=2003 PROFILE_DIR=hn-trav bench/cloudlab/trav_window.sh <name> [delay_s] [window_s]
#
# After <delay_s> of warm-up, back-to-back windows of <window_s>:
#   <name>.qps.txt          queries/s over the counter and sampling windows,
#                           from pg_stat_database.xact_commit (one query per
#                           transaction), since the probe window below slows
#                           queries down and the client's average includes it
#   <name>.travstat1..5.csv system-wide counts (perf stat -x,), four events per
#                           group on the programmable counters; cycles and
#                           instructions on the fixed ones. The NMI watchdog
#                           holds a counter, so it is off for these windows
#                           and restored afterwards. Groups 1 and 2 still
#                           multiplex (~40-60%): Linux schedules the 0xD1-0xD3
#                           events exclusively across hyperthread siblings on
#                           Haswell (erratum workaround), so they are scaled.
#     1 retired loads that missed L1, L2, L3, and that hit a fill buffer
#       (the line was already on its way: a late prefetch or a second miss)
#     2 retired loads served by local DRAM, remote DRAM, L3 (no snoop), and
#       loads that hit a software prefetch still in flight
#     3 L1D misses outstanding per cycle (sum), cycles stalled with an L2 miss
#       pending, cycles with any load pending, L1D line replacements
#     4 cycles with an L1D miss outstanding, demand data reads sent off core,
#       demand reads outstanding (sum per cycle), cycles with one outstanding
#     5 cycles stalled with an L1D miss pending, stalled with any load pending,
#       cycles with nothing executing, resource stalls
#   <name>.cyc.txt.gz       precise cycle samples (cycles:pp, Haswell's PEBS
#                           gives the eventing IP), postgres only: ip, sym, dso
#   <name>.lat.txt.gz       PEBS load-latency samples (loads over 50 cycles):
#                           ip, data address, latency, data source
#   <name>.probes.csv       function-entry counts from uprobes on the served
#                           vectorfast.so (hops, layers, slow paths, queries)
#   <name>.maps/<pid>.{start,end}, <name>.smaps.txt, <name>.busy.txt
# Output goes to bench/profiles/$PROFILE_DIR (default trav). Needs
# kernel.perf_event_paranoid <= 0 and passwordless sudo (watchdog, uprobes).
set -euo pipefail

NAME="$1"
DELAY="${2:-30}"
WIN="${3:-10}"
PERIOD_CYC="${PERIOD_CYC:-500009}"
PERIOD_LAT="${PERIOD_LAT:-2003}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/bench/profiles/${PROFILE_DIR:-trav}"
LIB="$ROOT/postgres/install/lib/x86_64-linux-gnu/vectorfast.so"
PSQL=("$ROOT/postgres/install/bin/psql" -X -At -p "${PGPORT:-5432}" -U postgres -d postgres)
mkdir -p "$OUT/$NAME.maps"

[ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 0 ] ||
  { echo "error: run: sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0" >&2; exit 1; }

WD="$(cat /proc/sys/kernel/nmi_watchdog)"
cleanup() {
  sudo -n sysctl -qw kernel.nmi_watchdog="$WD" || true
  sudo -n perf probe -q -d 'probe_vectorfast:*' >/dev/null 2>&1 || true
}
trap cleanup EXIT

ev() {  # ev <perf name> <raw terms>: the named event if perf knows it, else the raw encoding
  if perf stat -e "$1" -- true >/dev/null 2>&1; then printf '%s' "$1"
  else printf 'cpu/%s,name=%s/' "$2" "$1"; fi
}
G1="$(ev mem_load_uops_retired.l1_miss event=0xd1,umask=0x08),$(ev mem_load_uops_retired.l2_miss event=0xd1,umask=0x10),$(ev mem_load_uops_retired.l3_miss event=0xd1,umask=0x20),$(ev mem_load_uops_retired.hit_lfb event=0xd1,umask=0x40)"
G2="$(ev mem_load_uops_l3_miss_retired.local_dram event=0xd3,umask=0x01),$(ev mem_load_uops_l3_miss_retired.remote_dram event=0xd3,umask=0x04),$(ev mem_load_uops_l3_hit_retired.xsnp_none event=0xd2,umask=0x08),$(ev load_hit_pre.sw_pf event=0x4c,umask=0x01)"
G3="$(ev l1d_pend_miss.pending event=0x48,umask=0x01),$(ev cycle_activity.stalls_l2_pending event=0xa3,umask=0x05,cmask=5),$(ev cycle_activity.cycles_ldm_pending event=0xa3,umask=0x02,cmask=2),$(ev l1d.replacement event=0x51,umask=0x01)"
G4="$(ev l1d_pend_miss.pending_cycles event=0x48,umask=0x01,cmask=1),$(ev offcore_requests.demand_data_rd event=0xb0,umask=0x01),$(ev offcore_requests_outstanding.demand_data_rd event=0x60,umask=0x01),$(ev offcore_requests_outstanding.cycles_with_demand_data_rd event=0x60,umask=0x01,cmask=1)"
G5="$(ev cycle_activity.stalls_l1d_pending event=0xa3,umask=0x0c,cmask=12),$(ev cycle_activity.stalls_ldm_pending event=0xa3,umask=0x06,cmask=6),$(ev cycle_activity.cycles_no_execute event=0xa3,umask=0x04,cmask=4),$(ev resource_stalls.any event=0xa2,umask=0x01)"

snap() {  # snap <start|end>
  local p
  for p in $(pgrep -x postgres); do cat "/proc/$p/maps" > "$OUT/$NAME.maps/$p.$1" 2>/dev/null || true; done
}
xacts() { "${PSQL[@]}" -c "SELECT sum(xact_commit) FROM pg_stat_database"; }

echo "period_cyc=$PERIOD_CYC period_lat=$PERIOD_LAT window=$WIN" > "$OUT/$NAME.params"
sleep "$DELAY"
snap start
busy="$(ps -C postgres -o pid=,pcpu= --sort=-pcpu | awk 'NR == 1 {print $1}')"
cat "/proc/$busy/smaps" > "$OUT/$NAME.smaps.txt" 2>/dev/null || true
echo "busiest backend: $busy" > "$OUT/$NAME.busy.txt"

x0="$(xacts)"; t0="$(date +%s.%N)"
sudo -n sysctl -qw kernel.nmi_watchdog=0
i=0
for g in "$G1" "$G2" "$G3" "$G4" "$G5"; do
  i=$(( i + 1 ))
  perf stat -a -x, -o "$OUT/$NAME.travstat$i.csv" -e cycles,instructions -e "$g" -- sleep "$WIN" ||
    echo "counter group $i failed: $g" | tee -a "$OUT/$NAME.errors.txt"
done
sudo -n sysctl -qw kernel.nmi_watchdog="$WD"

perf record -a -c "$PERIOD_CYC" -e cycles:pp -o "$OUT/$NAME.cyc.data" -- sleep "$WIN" 2>"$OUT/$NAME.cyc.log"
LAT="cpu/mem-loads,ldlat=50/P"
perf record -e "$LAT" -- true >/dev/null 2>&1 || LAT="cpu/event=0xcd,umask=0x01,ldlat=50,name=load_latency/pp"
perf record -a -W -d -c "$PERIOD_LAT" -e "$LAT" -o "$OUT/$NAME.lat.data" -- sleep "$WIN" 2>"$OUT/$NAME.lat.log"
x1="$(xacts)"; t1="$(date +%s.%N)"
snap end
awk -v a="$x0" -v b="$x1" -v s="$t0" -v e="$t1" 'BEGIN {printf "qps=%.1f xacts=%d seconds=%.1f\n", (b - a) / (e - s), b - a, e - s}' > "$OUT/$NAME.qps.txt"

# Function-entry counts. uprobes trap into the kernel on every hit, so this
# window is slow; it only gives ratios (hops per query, and so on).
probes=()
for f in hnswrescan HnswSearchLayerUpperDiskFast HnswSearchLayerDiskFastCore HnswLoadUnvisitedRaw \
         HnswLoadNeighborTidsAndHashSlow HnswLoadUnvisitedMissingOrPinSlow HnswLoadFastCandSlow \
         HnswEvalAsyncUnvisitedBatchFast HnswTqRerankFast hnsw_fast_visited_grow CalicoLookupAndPin; do
  if sudo -n perf probe -q -x "$LIB" -a "$f" >/dev/null 2>>"$OUT/$NAME.probes.log"; then probes+=(-e "probe_vectorfast:$f")
  else echo "no probe for $f" >> "$OUT/$NAME.probes.log"; fi
done
if [ "${#probes[@]}" -gt 0 ]; then
  sudo -n perf stat -a -x, -o "$OUT/$NAME.probes.csv" "${probes[@]}" -- sleep "$WIN" || echo "probe window failed" >> "$OUT/$NAME.probes.log"
  sudo -n chown "$(id -u):$(id -g)" "$OUT/$NAME.probes.csv" 2>/dev/null || true
fi

F="comm,pid,ip,sym,dso"
perf script -i "$OUT/$NAME.cyc.data" --comms postgres -F "$F" 2>/dev/null | gzip > "$OUT/$NAME.cyc.txt.gz"
if ! perf script -i "$OUT/$NAME.lat.data" --comms postgres -F "$F,addr,weight,data_src" 2>"$OUT/$NAME.lat.script.log" | gzip > "$OUT/$NAME.lat.txt.gz" ||
   [ "$(zcat "$OUT/$NAME.lat.txt.gz" | head -c1 | wc -c)" -eq 0 ]; then
  echo "perf script refused weight/data_src; falling back to perf report" >> "$OUT/$NAME.lat.script.log"
  perf script -i "$OUT/$NAME.lat.data" --comms postgres -F "$F,addr" 2>/dev/null | gzip > "$OUT/$NAME.lat.txt.gz"
fi
perf report -i "$OUT/$NAME.lat.data" --comms postgres --mem-mode --no-children --stdio \
  --sort mem,sym --percent-limit 0.5 > "$OUT/$NAME.lat.memreport.txt" 2>/dev/null || true
echo "cycle samples: $(zcat "$OUT/$NAME.cyc.txt.gz" | wc -l); latency samples: $(zcat "$OUT/$NAME.lat.txt.gz" | wc -l); $(cat "$OUT/$NAME.qps.txt")"
