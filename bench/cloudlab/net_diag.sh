#!/usr/bin/env bash
# Where a node's time goes while a LAN benchmark runs, per CPU: is one core (or
# a few) saturated by network interrupts while the node as a whole looks idle?
# And, on the Postgres server, what are the backends waiting on?
#
#   NIC=enp94s0f0np0 [PGWAIT=1] PROFILE_DIR=hn-netdiag bench/cloudlab/net_diag.sh <name> [delay_s] [window_s]
#
# After <delay_s> (default 40), over one window of <window_s> (default 20):
#   <name>.stat0/.stat1          /proc/stat (every CPU) at the start and end
#   <name>.irq0/.irq1            /proc/interrupts
#   <name>.softirq0/.softirq1    /proc/softirqs
#   <name>.softnet0/.softnet1    /proc/net/softnet_stat (per CPU: processed,
#                                dropped, time squeezed)
#   <name>.ethS0/.ethS1          ethtool -S <NIC> (per-queue packet counters)
#   <name>.waits.txt             with PGWAIT=1: what client backends were
#                                doing, sampled every 100 ms (state, wait event)
# and once: <name>.nic.txt with ethtool -l / -c / -k, the NIC's IRQs and their
# CPU affinity, RPS/XPS masks, irqbalance, and the kernel's network sysctls.
# Writes bench/profiles/$PROFILE_DIR/. Read-only: changes no setting.
set -euo pipefail

NAME="$1"
DELAY="${2:-40}"
WIN="${3:-20}"
NIC="${NIC:?set NIC to the interface the benchmark uses}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/bench/profiles/${PROFILE_DIR:-netdiag}"
mkdir -p "$OUT"

{
  echo "## ethtool -l"; ethtool -l "$NIC" 2>&1 || true
  echo "## ethtool -c"; ethtool -c "$NIC" 2>&1 || true
  echo "## ethtool -k (offloads)"; ethtool -k "$NIC" 2>&1 | grep -E "^(generic-receive|generic-segmentation|tcp-segmentation|large-receive|rx-checksumming|tx-checksumming)" || true
  echo "## driver"; ethtool -i "$NIC" 2>&1 || true
  echo "## NIC IRQs: irq, affinity, name"
  grep -E "$NIC|mlx|i40e|ice|bnxt" /proc/interrupts | awk '{sub(":", "", $1); print $1, $NF}' | while read -r irq name; do
    echo "$irq $(cat /proc/irq/"$irq"/smp_affinity_list 2>/dev/null) $name"
  done
  echo "## RPS / XPS"
  for q in /sys/class/net/"$NIC"/queues/*; do
    printf '%s rps=%s xps=%s\n' "$(basename "$q")" "$(cat "$q"/rps_cpus 2>/dev/null)" "$(cat "$q"/xps_cpus 2>/dev/null)"
  done
  echo "## irqbalance"; systemctl is-active irqbalance 2>&1 || true
  echo "## sysctls"; sysctl net.core.busy_poll net.core.busy_read net.core.netdev_budget net.core.rps_sock_flow_entries \
    net.ipv4.tcp_low_latency 2>&1 || true
  echo "## CPUs"; lscpu | grep -E "^CPU\(s\)|NUMA node[0-9]"
  echo "## NIC NUMA node"; cat /sys/class/net/"$NIC"/device/numa_node 2>/dev/null || true
} > "$OUT/$NAME.nic.txt"

sleep "$DELAY"
snap() {  # snap <0|1>
  cat /proc/stat > "$OUT/$NAME.stat$1"
  cat /proc/interrupts > "$OUT/$NAME.irq$1"
  cat /proc/softirqs > "$OUT/$NAME.softirq$1"
  cat /proc/net/softnet_stat > "$OUT/$NAME.softnet$1"
  ethtool -S "$NIC" > "$OUT/$NAME.ethS$1" 2>/dev/null || true
  date +%s.%N > "$OUT/$NAME.t$1"
}
snap 0
if [ "${PGWAIT:-0}" = 1 ]; then
  PSQL=("$ROOT/postgres/install/bin/psql" -X -At -p "${PGPORT:-5432}" -U postgres -d template1)
  end=$(( $(date +%s) + WIN ))
  : > "$OUT/$NAME.waits.txt"
  while [ "$(date +%s)" -lt "$end" ]; do
    "${PSQL[@]}" -c "SELECT coalesce(state, '-') || ' ' || coalesce(wait_event_type, 'CPU') || ':' || coalesce(wait_event, '-')
                     FROM pg_stat_activity WHERE backend_type = 'client backend' AND datname = 'postgres'" \
      >> "$OUT/$NAME.waits.txt" || true
    sleep 0.1
  done
else
  sleep "$WIN"
fi
snap 1
echo "window done: $(cat "$OUT/$NAME.t0") .. $(cat "$OUT/$NAME.t1")"
[ -s "$OUT/$NAME.waits.txt" ] && sort "$OUT/$NAME.waits.txt" | uniq -c | sort -rn | head -12 || true
