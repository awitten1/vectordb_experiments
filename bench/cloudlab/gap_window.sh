#!/usr/bin/env bash
# What a node spends its CPU on while a search engine is under steady load:
# the counterpart of trav_window.sh / ginger_window.sh for the c220g5 nodes
# (Skylake-SP; event names that perf knows on this CPU, no Haswell encodings).
# The same script serves Postgres on the server and Ginger on the client node.
#
#   QPS_SRC=xact|progress:<file> [QPS_MULT=1] [C2C=1] PROFILE_DIR=hn-gap \
#     bench/cloudlab/gap_window.sh <name> [delay_s] [window_s]
#
# QPS_SRC says how queries are counted:
#   xact             pg_stat_database.xact_commit of database postgres (one
#                    query per transaction; QPS_MULT=100 when each answers
#                    100). Backends flush these counts about once a second,
#                    so use the span of all windows, not one 10 s window.
#   progress:<file>  ginger_bench's GINGER_PROGRESS file ("<answered> <time>")
# It waits until queries are being answered (at least 20 a second), then
# <delay_s> (default 20) of warm-up, then back-to-back windows of <window_s>
# (default 10), each bracketed by a query count and a /proc/stat snapshot:
#   stat1  cycles, instructions, ref-cycles, context switches, migrations,
#          and DRAM reads/writes (IMC CAS counts) and UPI data flits
#          (cross-socket traffic) from the uncore
#   stat2  retired loads that missed L1, L2, L3, and that hit a fill buffer
#   stat3  retired L3-miss loads served by local DRAM, remote DRAM, a remote
#          cache with the line modified (HITM); memory-ordering machine clears
#   stat4  cycles stalled with an L3 miss / any memory load pending, cycles
#          with a page walk active
#   stat5  the TopdownL1 metrics on their own (with other events they would
#          multiplex): retiring, bad speculation, frontend and backend bound
#   record cycles:pp with frame-pointer call stacks, system wide; folded on
#          the node into <name>.folded.gz ("comm;frame;...;leaf count", root
#          first) plus a flat <name>.flat.txt, so analysis needs no binaries
#   c2c    (C2C=1) perf c2c: the cache lines most contended between cores
# Writes bench/profiles/$PROFILE_DIR/<name>.{windows.csv,stat*.csv,folded.gz,
# flat.txt,c2c.txt.gz,params}. windows.csv has one row per window: name,
# start and end time, queries answered, and the /proc/stat cpu line at each
# end. The counters are system wide: everything on the node is included.
# The NMI watchdog is off for the counter windows and restored afterwards.
# Needs kernel.perf_event_paranoid <= 0 and kptr_restrict 0 (set by the
# driver's preflight) and passwordless sudo.
set -euo pipefail

NAME="$1"
DELAY="${2:-20}"
WIN="${3:-10}"
QPS_SRC="${QPS_SRC:-xact}"
QPS_MULT="${QPS_MULT:-1}"
PERIOD="${PERIOD:-2000003}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/bench/profiles/${PROFILE_DIR:-gap}"
# Counted from template1, so the counting itself is not a transaction of the
# benchmark's database
PSQL=("$ROOT/postgres/install/bin/psql" -X -At -p "${PGPORT:-5432}" -U postgres -d template1)
mkdir -p "$OUT"

[ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 0 ] ||
  { echo "error: run: sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0" >&2; exit 1; }

WD="$(cat /proc/sys/kernel/nmi_watchdog)"
cleanup() { sudo -n sysctl -qw kernel.nmi_watchdog="$WD" || true; }
trap cleanup EXIT

answered() {  # queries answered so far
  case "$QPS_SRC" in
    xact) echo $(( $("${PSQL[@]}" -c "SELECT xact_commit FROM pg_stat_database WHERE datname = 'postgres'") * QPS_MULT )) ;;
    progress:*) awk '{print $1}' "${QPS_SRC#progress:}" 2>/dev/null || echo 0 ;;
    *) echo "error: QPS_SRC must be xact or progress:<file>" >&2; exit 1 ;;
  esac
}
cpuline() { head -1 /proc/stat | cut -d' ' -f3- ; }

# Wait for load: at least 20 queries a second over 2 s, for up to 15 minutes.
for i in $(seq 450); do
  a="$(answered)"; sleep 2; b="$(answered)"
  [ $(( b - a )) -ge 40 ] && break
  [ "$i" -lt 450 ] || { echo "error: no load seen in 15 minutes" >&2; exit 1; }
done
echo "load seen: $(( (b - a) / 2 )) queries/s; warming up ${DELAY}s"
sleep "$DELAY"

{ echo "period=$PERIOD window=$WIN qps_src=$QPS_SRC qps_mult=$QPS_MULT"; uname -r; perf --version
  lscpu | grep -E "Model name|^CPU\(s\)|Thread"; } > "$OUT/$NAME.params"
echo "window,start,end,q0,q1,cpu0,cpu1" > "$OUT/$NAME.windows.csv"
win() {  # win <window name> <command...>: run it, bracketed by counts
  local w="$1" q0 q1 t0 t1 c0 c1; shift
  q0="$(answered)"; c0="$(cpuline)"; t0="$(date +%s.%N)"
  "$@" || echo "window $w failed: $*" | tee -a "$OUT/$NAME.errors.txt"
  t1="$(date +%s.%N)"; c1="$(cpuline)"; q1="$(answered)"
  echo "$w,$t0,$t1,$q0,$q1,$c0,$c1" >> "$OUT/$NAME.windows.csv"
  awk -v a="$q0" -v b="$q1" -v s="$t0" -v e="$t1" -v w="$w" \
    'BEGIN {printf "%-7s %.1f s, %.0f queries/s\n", w, e - s, (b - a) / (e - s)}'
}

UNC="uncore_imc/cas_count_read/,uncore_imc/cas_count_write/"
perf stat -e unc_upi_txl_flits.all_data -a -- true >/dev/null 2>&1 && UNC="$UNC,unc_upi_txl_flits.all_data"
S2="mem_load_retired.l1_miss,mem_load_retired.l2_miss,mem_load_retired.l3_miss,mem_load_retired.fb_hit"
S3="mem_load_l3_miss_retired.local_dram,mem_load_l3_miss_retired.remote_dram,mem_load_l3_miss_retired.remote_hitm,machine_clears.memory_ordering"
S4="cycle_activity.stalls_l3_miss,cycle_activity.stalls_mem_any,dtlb_load_misses.walk_active"

sudo -n sysctl -qw kernel.nmi_watchdog=0
win stat1 perf stat -a -x, -o "$OUT/$NAME.stat1.csv" \
  -e cycles,instructions,ref-cycles,context-switches,cpu-migrations -e "$UNC" -- sleep "$WIN"
win stat2 perf stat -a -x, -o "$OUT/$NAME.stat2.csv" -e cycles,instructions -e "$S2" -- sleep "$WIN"
win stat3 perf stat -a -x, -o "$OUT/$NAME.stat3.csv" -e cycles,instructions -e "$S3" -- sleep "$WIN"
win stat4 perf stat -a -x, -o "$OUT/$NAME.stat4.csv" -e cycles,instructions -e "$S4" -- sleep "$WIN"
win stat5 perf stat -a -x, -o "$OUT/$NAME.stat5.csv" -M TopdownL1 -- sleep "$WIN"
sudo -n sysctl -qw kernel.nmi_watchdog="$WD"

win record perf record -q -a -g -c "$PERIOD" -e cycles:pp -o "$OUT/$NAME.cyc.data" -- sleep "$WIN"
if [ "${C2C:-0}" = 1 ]; then
  win c2c perf c2c record -a -o "$OUT/$NAME.c2c.data" -- sleep "$WIN"
fi

# Fold the call stacks here, where the binaries are: one line per distinct
# stack, root first, prefixed by the command name.
perf script -i "$OUT/$NAME.cyc.data" -F comm,ip,sym,dso 2>/dev/null | python3 -c '
import sys, collections
folded = collections.Counter(); comm = None; frames = []
def flush():
    if comm is not None:
        folded[";".join([comm] + frames[::-1])] += 1
for line in sys.stdin:
    s = line.strip()
    if not s:
        flush(); comm = None; frames = []
    elif comm is None:
        comm = s.split()[0] if s.split() else "?"
    else:
        p = s.split(None, 1)
        rest = p[1] if len(p) > 1 else "[unknown]"
        sym, _, dso = rest.rpartition(" (")
        sym = (sym or rest).split("+0x")[0].replace(";", ":")
        dso = dso.rstrip(")").rsplit("/", 1)[-1]
        frames.append(f"{sym}@{dso}")
flush()
for k, v in folded.most_common():
    print(k, v)
' | gzip > "$OUT/$NAME.folded.gz"
perf report -i "$OUT/$NAME.cyc.data" --stdio --no-children -g none --sort comm,dso,sym \
  --percent-limit 0.05 2>/dev/null > "$OUT/$NAME.flat.txt" || true
rm -f "$OUT/$NAME.cyc.data"
if [ "${C2C:-0}" = 1 ]; then
  perf c2c report -i "$OUT/$NAME.c2c.data" --stdio --full-symbols -NN 2>/dev/null | gzip > "$OUT/$NAME.c2c.txt.gz" || true
  rm -f "$OUT/$NAME.c2c.data"
fi
echo "folded stacks: $(zcat "$OUT/$NAME.folded.gz" | wc -l) distinct, $(zcat "$OUT/$NAME.folded.gz" | awk '{s += $NF} END {print s}') samples"
ls -la "$OUT" | grep -F "$NAME."
