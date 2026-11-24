#!/usr/bin/env bash
# Why pgvector_ntq's tq4 navigation codes give no speedup on Hacker News 27.4M:
# profile stock pgvector, pgvectorfast and pgvector_ntq under the same steady
# load, count their buffer accesses, and microbenchmark the two distance
# kernels with Postgres stopped. Run from the workstation after hn_tq4.sh:
#
#   caffeinate -i bench/cloudlab/hn_profile_tq4.sh
#
# Steps are detached and resumable (see lib.sh); bump RUN to redo them all.
#   0. enable perf on the server (perf_event_paranoid=-1, kptr_restrict=0) and
#      pick the call-graph mode: LBR call stacks if the CPU and kernel give
#      them (no frame-pointer rebuild needed), else flat profiles
#   1. per arm (stock, fast, ntq): serve it (prewarmed, residency and NUMA
#      interleave checked), then at 1 and 40 clients and ef_search EF the
#      client runs on the client node for 130 s while perf_window.sh profiles
#      the server (30 s warm-up, then four 20 s windows: counters, Top-Down,
#      call graph, TLB walks); for stock and ntq, per-query buffer accesses
#      from EXPLAIN (ANALYZE, BUFFERS)
#   2. stop the server; disassemble the two distance kernels as built, and run
#      tq_kernel_bench.c (in-cache and random access, 4 KB and huge pages)
#   3. summarize on the server (perf_summary.py, perf_buckets.py) and pull the
#      text outputs into bench/profiles/hn-tq4/
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"

RUN="${RUN:-1}"
T="hnprof-r$RUN"
EF="${EF:-160}"
CLIENTS_LIST="${CLIENTS_LIST:-1 40}"
ARMS="${ARMS:-stock fast ntq}"
PD=hn-tq4
OUT="bench/profiles/$PD"
CG_FILE="$STEPS/$T-callgraph"

require_measurement_state

# shellcheck disable=SC2016
step "$SERVER" "$T-perf-enable" \
  'sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0 && perf --version &&
   if perf record -a --call-graph lbr -o /mydata/lbr-test.data -- sleep 1 >/dev/null 2>&1; then
     echo lbr > '"$CG_FILE"'; else echo none > '"$CG_FILE"'; fi; rm -f /mydata/lbr-test.data;
   echo "call graphs: $(cat '"$CG_FILE"')"; mkdir -p '"$OUT"

# profile_pair <arm> <clients>: the client on the client node for 130 s, and
# perf_window.sh on the server during it.
profile_pair() {
  local arm="$1" c="$2" name="hn-$1-c$2-ef$EF" extra=""
  # shellcheck disable=SC2016  # NTQ_* expand on the node, from hn.env
  [ "$arm" = ntq ] && extra='--table "$NTQ_TABLE" $NTQ_ARGS'
  if step_done "$CLIENT" "$T-client-$name" && step_done "$SERVER" "$T-perf-$name"; then
    say "$name: already done, skipping"; return
  fi
  start_step "$CLIENT" "$T-client-$name" \
    "BENCH_LABEL=prof-$arm bench/.venv/bin/python bench/ann_bench.py --dataset \"\$DATASET\" --index hnsw --m 16 --ef-construction 64 $extra --sweep $EF --clients $c --num-queries \"\$NUM_QUERIES\" --duration 130 $REMOTE_DSN"
  start_step "$SERVER" "$T-perf-$name" \
    "CALLGRAPH=\$(cat $CG_FILE) PROFILE_DIR=$PD bench/cloudlab/perf_window.sh $name 30 20"
  wait_step "$SERVER" "$T-perf-$name"
  wait_step "$CLIENT" "$T-client-$name"
}

for arm in $ARMS; do
  serve_step "$T-serve-$arm" "$arm"
  # shellcheck disable=SC2016
  step "$SERVER" "$T-numa-$arm" \
    '[ -n "${NUMA_CMD:-}" ] || exit 0; n=$(grep -c " interleave" /proc/$(head -1 postgres/data/postmaster.pid)/numa_maps); echo "postmaster mappings with interleave policy: $n"; [ "$n" -gt 0 ]'
  for c in $CLIENTS_LIST; do
    profile_pair "$arm" "$c"
  done
  case "$arm" in
    stock) table='' ;;
    # shellcheck disable=SC2016  # expands on the node, from hn.env
    ntq)   table='--table "$NTQ_TABLE"' ;;
    *)     continue ;;
  esac
  step "$SERVER" "$T-explain-$arm" \
    "set -o pipefail; bench/.venv/bin/python bench/cloudlab/explain_buffers.py --dataset \"\$DATASET\" $table --ef 10,40,160,640 --n 100 | tee $OUT/explain-$arm.txt"
done

step "$SERVER" "$T-stop" "./run_postgres.sh stop"

# The kernels as the extensions run them: TqNavDot is static and may be
# inlined into HnswTqCodeDistance; pgvector's L2 may carry a clone suffix.
# shellcheck disable=SC2016
step "$SERVER" "$T-disasm" 'set -o pipefail
  dis() { objdump -d --no-show-raw-insn "$1" | awk -v pat="$2" "/^[0-9a-f]+ <.*>:\$/ {p = (\$2 ~ pat)} p"; }
  { for f in pgvector_ntq/vector.so:HnswTqCodeDistance\|TqNavDot\|TqRefineDot \
             postgres/stock/vector.so:VectorL2SquaredDistance\|vector_l2_squared_distance; do
      lib=${f%%:*}; pat=${f#*:}
      echo "##### $lib: $pat"
      dis "$lib" "$pat"
    done; } > '"$OUT"'/disasm.txt
  for f in pgvector_ntq/vector.so:HnswTqCodeDistance\|TqNavDot postgres/stock/vector.so:L2SquaredDistance\|vector_l2_squared_distance; do
    lib=${f%%:*}; pat=${f#*:}; body=$(dis "$lib" "$pat")
    printf "%-26s %-24s insns %4d  ymm %4d  scalar-ss %4d  gather %d\n" "$lib" "$pat" \
      "$(grep -cE "^ +[0-9a-f]+:" <<<"$body")" "$(grep -c ymm <<<"$body")" \
      "$(grep -cE "\bv?(add|mul|sub|fmadd[0-9]*|fnmadd[0-9]*)ss\b" <<<"$body")" "$(grep -c gather <<<"$body")"
  done | tee '"$OUT"'/disasm-summary.txt'

step "$SERVER" "$T-kernel" \
  "set -o pipefail; gcc -O2 -march=native -ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math -ffp-contract=fast -o /mydata/tq_kernel_bench bench/cloudlab/tq_kernel_bench.c -lm && \${NUMA_CMD:-} /mydata/tq_kernel_bench 8000000 4000000 | tee $OUT/kernel.txt"

# The client's result rows carry the QPS the summaries divide by.
ssh_to "$CLIENT" "cat $DEST/bench/results/hackernews_minilm_384.jsonl" |
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$SERVER.$DOMAIN" "cat > $DEST/$OUT/client-results.jsonl"
step "$SERVER" "$T-summary" \
  "set -o pipefail; python3 bench/cloudlab/perf_summary.py $OUT $OUT/client-results.jsonl | tee $OUT/summary.md &&
   python3 bench/cloudlab/perf_buckets.py --top 30 $OUT $OUT/client-results.jsonl | tee $OUT/buckets.md"

for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "mkdir -p $DEST/bench/results/logs && cp $STEPS/$T-*.log $DEST/bench/results/logs/ 2>/dev/null; true"
done
mkdir -p "$HERE/../profiles/$PD"
rsync -az --exclude '*.perf.data' -e "ssh ${SSH_OPTS[*]}" \
  "$REMOTE_USER@$SERVER.$DOMAIN:$DEST/$OUT/" "$HERE/../profiles/$PD/"
say "done; profiles in bench/profiles/$PD (perf.data files stay on $SERVER); results: sync.sh pull"
