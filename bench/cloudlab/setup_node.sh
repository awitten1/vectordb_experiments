#!/usr/bin/env bash
# Prepare a fresh CloudLab node (m510 or c220g2, Ubuntu 24.04) for the benchmarks. Idempotent:
# every step checks before it acts, so re-running is safe.
#
# Usage (on the node, from the repo root): bench/cloudlab/setup_node.sh
#
# On a node with no /mydata yet, create it first, before the repo is there:
#   ssh <node> 'bash -s -- disk' < bench/cloudlab/setup_node.sh
#
# The repo is expected at /mydata/vectordb_experiments; push it there from the
# workstation with bench/cloudlab/sync.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
say() { printf '\n=== %s\n' "$*" >&2; }
MODE="${1:-all}"  # "disk": only create /mydata

# 1. Disk. / is a 64 GB partition; the rest of the NVMe (~166 GB) is
#    unallocated, and the dataset + PGDATA + WAL + Ginger index want more room
#    than / leaves. CloudLab's own tool carves it into /mydata.
#    The m510 image already has an empty partition 4 with a type GUID set, which
#    mkextrafs refuses without -f; refuse ourselves if it holds a filesystem.
#
#    Other node types (e.g. c220g2: / on a 1.1 TB HDD, plus a blank 480 GB
#    SATA SSD) get /mydata on a whole blank SSD instead: Postgres, the dataset
#    and the build all want flash, and the HDD would make loads and builds
#    I/O-bound. Only a non-rotational disk with no partitions, no filesystem
#    and no mount is ever formatted.
if ! mountpoint -q /mydata; then
  if [ -b /dev/nvme0n1p4 ]; then
    say "creating /mydata from the unallocated NVMe space"
    [ -z "$(lsblk -no FSTYPE /dev/nvme0n1p4 2>/dev/null)" ] ||
      { echo "error: /dev/nvme0n1p4 already has a filesystem; mount it by hand" >&2; exit 1; }
    sudo mkdir -p /mydata
    sudo /usr/local/etc/emulab/mkextrafs.pl -f /mydata
  else
    ssd=""
    while read -r name rota type; do
      [ "$type" = disk ] && [ "$rota" = 0 ] || continue
      # Blank: the disk is its only lsblk row (no partitions) and has no FS or mount.
      [ "$(lsblk -no NAME "$name" | wc -l)" = 1 ] || continue
      [ -z "$(lsblk -no FSTYPE,MOUNTPOINT "$name" | tr -d '[:space:]')" ] || continue
      ssd="$name"; break
    done < <(lsblk -dnpo NAME,ROTA,TYPE)
    [ -n "$ssd" ] || { echo "error: no blank non-rotational disk for /mydata; refusing to use an HDD" >&2; lsblk >&2; exit 1; }
    say "creating /mydata on the blank SSD $ssd ($(lsblk -dno SIZE "$ssd"))"
    # Initialise inode tables and journal now, not lazily in the background
    # during measurements.
    sudo mkfs.ext4 -q -F -L mydata -E lazy_itable_init=0,lazy_journal_init=0 "$ssd"
    sudo mkdir -p /mydata
    sudo mount -o noatime "$ssd" /mydata
  fi
  sudo chown "$(id -u):$(id -g)" /mydata
fi
if [ "$MODE" = disk ]; then df -h /mydata; exit 0; fi

# 2. Packages.
PKGS=(build-essential meson ninja-build cmake pkg-config bison flex
      libreadline-dev zlib1g-dev libicu-dev python3-venv python3-dev
      libcapstone-dev libjemalloc-dev libopenblas-dev git rsync
      linux-tools-common "linux-tools-$(uname -r)" numactl)
missing=()
for p in "${PKGS[@]}"; do
  dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'install ok installed' || missing+=("$p")
done
if [ "${#missing[@]}" -gt 0 ]; then
  say "installing: ${missing[*]}"
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
fi

# 3. CPU: turbo off, every core pinned at the base clock, deep C-states off
#    (see pin_cpu.sh for why each one matters).
#    The base clock: 2.0 GHz on the m510's Xeon D-1548, 2.6 GHz on the
#    c220g2's E5-2660 v3 (hn.env's FREQ_KHZ; the runners check against it).
if [ -z "${FREQ_KHZ:-}" ] && lscpu | grep -q 'E5-2660 v3'; then export FREQ_KHZ=2600000; fi
say "pinning the CPU clock"
"$ROOT/bench/cloudlab/pin_cpu.sh"

# 3b. Memory: no swap. The comparison is defined as fully in-memory, and with
#     swap on, a run that does not fit degrades silently instead of failing.
#     With it off, it either fits or is OOM-killed. Not persistent across a
#     reboot; the runners refuse to start if swap is back.
if [ -n "$(tail -n +2 /proc/swaps)" ]; then
  say "disabling swap"
  sudo swapoff -a
fi

# 4. Python environment for ann_bench.py and datasets.py.
if [ ! -x "$ROOT/bench/.venv/bin/python" ]; then
  say "creating bench/.venv"
  python3 -m venv "$ROOT/bench/.venv"
fi
"$ROOT/bench/.venv/bin/pip" install -q -r "$ROOT/bench/requirements.txt"

# 5. Record the environment, so node-to-node differences are visible later.
mkdir -p "$ROOT/bench/results"
ENV_OUT="$ROOT/bench/results/env-$(hostname -s).txt"
{
  echo "# captured $(date -u +%FT%TZ)"
  hostname -f
  uname -a
  lscpu
  echo "--- governors"; sort /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | uniq -c
  echo "--- no_turbo"; cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo "n/a"
  echo "--- scaling min/max kHz"; sort /sys/devices/system/cpu/cpu*/cpufreq/scaling_{min,max}_freq | uniq -c
  echo "--- idle states (name latency disabled)"
  for s in /sys/devices/system/cpu/cpu0/cpuidle/state*; do
    echo "$(cat "$s/name") $(cat "$s/latency") $(cat "$s/disable")"
  done
  echo "--- swap"; cat /proc/swaps
  echo "--- driver"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver
  echo "--- memory"; free -g
  echo "--- numa"; numactl -H
  echo "--- THP"; cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/shmem_enabled
  echo "--- disks"; lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
  echo "--- compilers"; gcc --version | head -1; cmake --version | head -1; meson --version
} > "$ENV_OUT"
say "environment recorded in $ENV_OUT"
