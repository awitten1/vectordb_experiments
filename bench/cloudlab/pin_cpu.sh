#!/usr/bin/env bash
# Pin the CPU clock so measurements are stable and identical across nodes.
#
#   bench/cloudlab/pin_cpu.sh          apply (needs sudo)
#   bench/cloudlab/pin_cpu.sh check    exit 1 unless the pinning is in place
#
# On the m510s (Xeon D-1548, intel_pstate in passive mode) the defaults are the
# performance governor with turbo on: busy cores run at up to 2.6 GHz, but the
# turbo bin depends on how many cores are active and on thermals, so 1-thread
# and 8-thread runs see different clocks and two "identical" nodes can differ.
# So: turbo off, and min = max = the 2.0 GHz base clock on every core.
#
# Idle states deeper than C1 are disabled too (exit latency > 2 us, i.e. C1E
# and below: C1E also drops the clock to its minimum while halted). A
# single-client Postgres query is a ping-pong between the client and the
# backend, each sleeping while the other works, so every query would pay the
# wake-up latency; an in-process engine like Ginger never sleeps and would not.
#
# None of this survives a reboot; the runners call `check` and refuse to start.
set -euo pipefail

FREQ_KHZ="${FREQ_KHZ:-2000000}"
MAX_IDLE_LATENCY_US="${MAX_IDLE_LATENCY_US:-2}"
CPU=/sys/devices/system/cpu

check() {
  local bad=0 c f
  if [ -f $CPU/intel_pstate/no_turbo ] && [ "$(cat $CPU/intel_pstate/no_turbo)" != 1 ]; then
    echo "turbo is on" >&2; bad=1
  fi
  for c in $CPU/cpu[0-9]*/cpufreq; do
    for f in scaling_min_freq scaling_max_freq; do
      [ "$(cat "$c/$f")" = "$FREQ_KHZ" ] || { echo "$c/$f = $(cat "$c/$f"), want $FREQ_KHZ" >&2; bad=1; }
    done
  done
  for s in $CPU/cpu[0-9]*/cpuidle/state*; do
    [ -d "$s" ] || continue
    if [ "$(cat "$s/latency")" -gt "$MAX_IDLE_LATENCY_US" ] && [ "$(cat "$s/disable")" != 1 ]; then
      echo "$s ($(cat "$s/name"), ${MAX_IDLE_LATENCY_US}us+) is enabled" >&2; bad=1
    fi
  done
  if [ "$bad" = 1 ]; then
    echo "error: CPU frequency is not pinned; run bench/cloudlab/pin_cpu.sh" >&2
    return 1
  fi
}

if [ "${1:-}" = check ]; then
  check
  exit
fi

[ -f $CPU/intel_pstate/no_turbo ] && echo 1 | sudo tee $CPU/intel_pstate/no_turbo >/dev/null
for c in $CPU/cpu[0-9]*/cpufreq; do
  echo performance | sudo tee "$c/scaling_governor" >/dev/null
  # Order matters when moving the window: raise max before min, lower min before max.
  echo "$FREQ_KHZ" | sudo tee "$c/scaling_max_freq" >/dev/null
  echo "$FREQ_KHZ" | sudo tee "$c/scaling_min_freq" >/dev/null
  echo "$FREQ_KHZ" | sudo tee "$c/scaling_max_freq" >/dev/null
done
for s in $CPU/cpu[0-9]*/cpuidle/state*; do
  [ -d "$s" ] || continue
  if [ "$(cat "$s/latency")" -gt "$MAX_IDLE_LATENCY_US" ]; then
    echo 1 | sudo tee "$s/disable" >/dev/null
  fi
done
check
echo "CPU pinned at $((FREQ_KHZ / 1000)) MHz, turbo off, idle states > ${MAX_IDLE_LATENCY_US}us disabled" >&2
