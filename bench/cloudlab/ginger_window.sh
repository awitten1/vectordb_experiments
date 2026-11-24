#!/usr/bin/env bash
# What a running ginger_bench spends its time on, during a steady-state window
# (the counterpart of trav_window.sh for Ginger, which runs in-process on the
# client node).
#
#   PERIOD_CYC=500009 PERIOD_LAT=2003 PROFILE_DIR=hn-ginger \
#     bench/cloudlab/ginger_window.sh <name> <progress file> [delay_s] [window_s]
#
# Waits until ginger_bench has answered its first queries (the progress file,
# written by ginger_bench under GINGER_PROGRESS, counts them), then after
# <delay_s> of warm-up records back-to-back windows of <window_s>:
#   <name>.qps.txt          queries/s over the counter and sampling windows,
#                           from the progress file
#   <name>.travstat1..5.csv the cache counter groups of trav_window.sh
#   <name>.tlbstat.csv      load page walks: started, completed for 4 KB and
#                           2 MB pages, cycles walking
#   <name>.cyc.txt.gz       precise cycle samples (cycles:pp), ginger_bench only
#   <name>.lat.txt.gz       PEBS load-latency samples (loads over 50 cycles)
#   <name>.smaps.txt, <name>.smaps_rollup.txt, <name>.thp.txt
# The NMI watchdog is off for the counter windows and restored afterwards.
# Needs kernel.perf_event_paranoid <= 0 and passwordless sudo (watchdog).
set -euo pipefail

NAME="$1"
PROGRESS="$2"
DELAY="${3:-20}"
WIN="${4:-10}"
PERIOD_CYC="${PERIOD_CYC:-500009}"
PERIOD_LAT="${PERIOD_LAT:-2003}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/bench/profiles/${PROFILE_DIR:-ginger}"
COMM=ginger_bench
mkdir -p "$OUT"

[ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 0 ] ||
  { echo "error: run: sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0" >&2; exit 1; }

WD="$(cat /proc/sys/kernel/nmi_watchdog)"
cleanup() { sudo -n sysctl -qw kernel.nmi_watchdog="$WD" || true; }
trap cleanup EXIT

# Wait (up to 15 minutes: the index load reads ~18 GB) for queries to start.
for _ in $(seq 900); do
  [ -s "$PROGRESS" ] && [ "$(awk '{print $1}' "$PROGRESS")" -gt 0 ] && break
  sleep 1
done
[ -s "$PROGRESS" ] && [ "$(awk '{print $1}' "$PROGRESS")" -gt 0 ] ||
  { echo "error: ginger_bench answered no queries within 15 minutes" >&2; exit 1; }
sleep "$DELAY"
pid="$(pgrep -x "$COMM" | head -1)"
[ -n "$pid" ] || { echo "error: no $COMM process" >&2; exit 1; }

ev() {  # ev <perf name> <raw terms>: the named event if perf knows it, else the raw encoding
  if perf stat -e "$1" -- true >/dev/null 2>&1; then printf '%s' "$1"
  else printf 'cpu/%s,name=%s/' "$2" "$1"; fi
}
G1="$(ev mem_load_uops_retired.l1_miss event=0xd1,umask=0x08),$(ev mem_load_uops_retired.l2_miss event=0xd1,umask=0x10),$(ev mem_load_uops_retired.l3_miss event=0xd1,umask=0x20),$(ev mem_load_uops_retired.hit_lfb event=0xd1,umask=0x40)"
G2="$(ev mem_load_uops_l3_miss_retired.local_dram event=0xd3,umask=0x01),$(ev mem_load_uops_l3_miss_retired.remote_dram event=0xd3,umask=0x04),$(ev mem_load_uops_l3_hit_retired.xsnp_none event=0xd2,umask=0x08),$(ev load_hit_pre.sw_pf event=0x4c,umask=0x01)"
G3="$(ev l1d_pend_miss.pending event=0x48,umask=0x01),$(ev cycle_activity.stalls_l2_pending event=0xa3,umask=0x05,cmask=5),$(ev cycle_activity.cycles_ldm_pending event=0xa3,umask=0x02,cmask=2),$(ev l1d.replacement event=0x51,umask=0x01)"
G4="$(ev l1d_pend_miss.pending_cycles event=0x48,umask=0x01,cmask=1),$(ev offcore_requests.demand_data_rd event=0xb0,umask=0x01),$(ev offcore_requests_outstanding.demand_data_rd event=0x60,umask=0x01),$(ev offcore_requests_outstanding.cycles_with_demand_data_rd event=0x60,umask=0x01,cmask=1)"
G5="$(ev cycle_activity.stalls_l1d_pending event=0xa3,umask=0x0c,cmask=12),$(ev cycle_activity.stalls_ldm_pending event=0xa3,umask=0x06,cmask=6),$(ev cycle_activity.cycles_no_execute event=0xa3,umask=0x04,cmask=4),$(ev resource_stalls.any event=0xa2,umask=0x01)"
GT="$(ev dtlb_load_misses.miss_causes_a_walk event=0x08,umask=0x01),$(ev dtlb_load_misses.walk_completed_4k event=0x08,umask=0x02),$(ev dtlb_load_misses.walk_completed_2m_4m event=0x08,umask=0x04),$(ev dtlb_load_misses.walk_duration event=0x08,umask=0x10)"

count() { awk '{print $1, $2}' "$PROGRESS"; }

echo "period_cyc=$PERIOD_CYC period_lat=$PERIOD_LAT window=$WIN pid=$pid" > "$OUT/$NAME.params"
cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag > "$OUT/$NAME.thp.txt" 2>/dev/null || true
cat "/proc/$pid/smaps_rollup" > "$OUT/$NAME.smaps_rollup.txt" 2>/dev/null || true
cat "/proc/$pid/smaps" > "$OUT/$NAME.smaps.txt" 2>/dev/null || true

read -r q0 t0 < <(count)
sudo -n sysctl -qw kernel.nmi_watchdog=0
i=0
for g in "$G1" "$G2" "$G3" "$G4" "$G5"; do
  i=$(( i + 1 ))
  perf stat -a -x, -o "$OUT/$NAME.travstat$i.csv" -e cycles,instructions -e "$g" -- sleep "$WIN" ||
    echo "counter group $i failed: $g" | tee -a "$OUT/$NAME.errors.txt"
done
perf stat -a -x, -o "$OUT/$NAME.tlbstat.csv" -e cycles,instructions -e "$GT" -- sleep "$WIN" ||
  echo "TLB group failed: $GT" | tee -a "$OUT/$NAME.errors.txt"
sudo -n sysctl -qw kernel.nmi_watchdog="$WD"

perf record -a -c "$PERIOD_CYC" -e cycles:pp -o "$OUT/$NAME.cyc.data" -- sleep "$WIN" 2>"$OUT/$NAME.cyc.log"
LAT="cpu/mem-loads,ldlat=50/P"
perf record -e "$LAT" -- true >/dev/null 2>&1 || LAT="cpu/event=0xcd,umask=0x01,ldlat=50,name=load_latency/pp"
perf record -a -W -d -c "$PERIOD_LAT" -e "$LAT" -o "$OUT/$NAME.lat.data" -- sleep "$WIN" 2>"$OUT/$NAME.lat.log"
read -r q1 t1 < <(count)
awk -v a="$q0" -v b="$q1" -v s="$t0" -v e="$t1" 'BEGIN {printf "qps=%.1f queries=%d seconds=%.1f\n", (b - a) / (e - s), b - a, e - s}' > "$OUT/$NAME.qps.txt"
kill -0 "$pid" 2>/dev/null || echo "warning: $COMM exited during the windows; QPS covers a partial window" >> "$OUT/$NAME.qps.txt"

F="comm,pid,ip,sym,dso"
perf script -i "$OUT/$NAME.cyc.data" --comms "$COMM" -F "$F" 2>/dev/null | gzip > "$OUT/$NAME.cyc.txt.gz"
perf script -i "$OUT/$NAME.lat.data" --comms "$COMM" -F "$F,addr,weight,data_src" 2>"$OUT/$NAME.lat.script.log" | gzip > "$OUT/$NAME.lat.txt.gz" || true
perf report -i "$OUT/$NAME.lat.data" --comms "$COMM" --mem-mode --no-children --stdio \
  --sort mem,sym --percent-limit 0.5 > "$OUT/$NAME.lat.memreport.txt" 2>/dev/null || true
echo "cycle samples: $(zcat "$OUT/$NAME.cyc.txt.gz" | wc -l); latency samples: $(zcat "$OUT/$NAME.lat.txt.gz" | wc -l); $(cat "$OUT/$NAME.qps.txt")"
