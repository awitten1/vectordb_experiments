#!/usr/bin/env bash
# Symbolize every perf data file under bench/profiles on this node into plain
# text that can be read anywhere, before the machine (and its binaries) go.
#
#   bench/cloudlab/perf_export.sh [profiles dir]     # default bench/profiles
#
# 1. Registers every saved copy of the profiled libraries and binaries with
#    perf's build-id cache (vectorfast.so / vector.so builds under /mydata,
#    the per-study copies in bench/profiles, postgres, ginger_bench), so each
#    recording finds the exact binary it sampled even if it was rebuilt since.
# 2. For each *.data, writes beside it into <name>.export/:
#      report.txt      flat profile: samples by comm, dso, symbol (kernel too)
#      callgraph.txt   for recordings with call stacks: caller-based call graph
#      folded.gz       for recordings with call stacks: one line per unique
#                      stack, "comm;outer;...;inner count" (flame graphs:
#                      speedscope, Firefox Profiler, flamegraph.pl)
#      script.txt.gz   for recordings without call stacks: every sample, all
#                      fields perf has (ip, sym+off, dso, addr, weight,
#                      data_src where recorded), every process
#      symbols.txt     share of samples resolved to a symbol, by dso, and the
#                      build-ids perf could not find
#      header.txt      the recording's header (command line, events, host)
# Needs perf, python3 and passwordless sudo (kptr_restrict for kernel symbols).
set -uo pipefail

DIR="${1:-bench/profiles}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
sudo -n sysctl -qw kernel.kptr_restrict=0 kernel.perf_event_paranoid=-1 || true

echo "== registering binaries with the build-id cache"
{
  find /mydata -xdev \( -name 'vectorfast*.so' -o -name 'vector.so' -o -name 'vectorfast_ntq_*.so' \) -type f 2>/dev/null
  find "$DIR" -name 'ginger_bench' -type f 2>/dev/null
  ls postgres/install/bin/postgres bench/ginger/build/ginger_bench 2>/dev/null
} | sort -u | while read -r f; do
  perf buildid-cache -a "$f" 2>/dev/null && echo "  $f"
done

COLLAPSE='
import sys, collections
# perf script with call stacks: a header line per sample, then one frame per
# line ("addr sym+off (dso)"), innermost first, then a blank line.
counts = collections.Counter()
comm, frames = None, []
def flush():
    if comm is not None:
        counts[";".join([comm] + frames[::-1])] += 1
for line in sys.stdin:
    line = line.rstrip("\n")
    if not line.strip():
        flush(); comm, frames = None, []
        continue
    if not line[0].isspace():
        flush(); comm, frames = line.split()[0], []
        continue
    parts = line.split(None, 1)
    if len(parts) < 2:
        continue
    rest = parts[1]
    sym = rest.rsplit(" (", 1)[0] if " (" in rest else rest
    sym = sym.split("+0x")[0] if "+0x" in sym else sym
    frames.append(sym.replace(";", ":") or "[unknown]")
flush()
for stack, n in counts.most_common():
    print(stack, n)
'

find "$DIR" -name '*.data' -type f | sort | while read -r f; do
  out="${f%.data}.export"
  if [ -s "$out/report.txt" ]; then echo "== $f: done before, skipping"; continue; fi
  mkdir -p "$out"
  echo "== $f ($(du -h "$f" | cut -f1))"
  perf report -i "$f" --header-only > "$out/header.txt" 2>/dev/null
  has_stacks=0
  perf report -i "$f" --header-only 2>/dev/null | grep -q "CALLCHAIN" && has_stacks=1
  # Flat profile, every comm, children off.
  timeout 1800 perf report -i "$f" --stdio --no-children --sort comm,dso,sym --percent-limit 0.01 \
    --no-inline > "$out/report.txt" 2>"$out/report.log"
  # Resolution: samples per dso, and the share left as unknown symbols.
  timeout 1800 perf report -i "$f" --stdio --no-children --sort dso --percent-limit 0 \
    > "$out/symbols.txt" 2>/dev/null
  {
    echo
    echo "# samples whose symbol is unknown, as a share of all samples:"
    grep -E "^\s+[0-9.]+%" "$out/report.txt" | grep "\[\.\] 0x\|\[unknown\]" |
      awk '{gsub("%","",$1); s += $1} END {printf "%.2f%%\n", s}'
    echo "# build-ids with hits that perf could not find a binary for:"
    perf buildid-list -i "$f" --with-hits 2>/dev/null | while read -r id path; do
      [ -e "$HOME/.debug/.build-id/${id:0:2}/${id:2}" ] && continue
      [ -e "$path" ] && [ "$(perf buildid-list -i "$path" 2>/dev/null)" = "$id" ] && continue
      echo "  $id $path"
    done
  } >> "$out/symbols.txt"
  if [ "$has_stacks" = 1 ]; then
    timeout 1800 perf report -i "$f" --stdio --children --sort sym -g caller,0.5,callee,function,percent \
      --percent-limit 0.5 --no-inline > "$out/callgraph.txt" 2>/dev/null
    timeout 3600 perf script -i "$f" --no-inline -F comm,ip,sym,symoff,dso 2>/dev/null |
      python3 -c "$COLLAPSE" | gzip > "$out/folded.gz"
  else
    fields=comm,pid,tid,cpu,time,event,ip,sym,symoff,dso
    for extra in addr weight data_src; do
      perf script -i "$f" -F "$fields,$extra" 2>/dev/null | head -1 | grep -q . && fields="$fields,$extra"
    done
    timeout 3600 perf script -i "$f" -F "$fields" 2>/dev/null | gzip > "$out/script.txt.gz"
  fi
  ls -s "$out" | sed 's/^/    /'
done
echo "== done"
