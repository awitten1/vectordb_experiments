#!/usr/bin/env bash
# The best Postgres arm (pgvectorfast + PiPNN + tq4 on 32 KB pages, reordered,
# rerank capped at 100 candidates: label g5hp32kro-rr10-pgvectorfast-tq4) with
# the 4+8 bit rerank distance computed by a vector kernel instead of the scalar
# loop (pgvectorfast_ntq: HnswTqDistance4k; the setting hnsw.rerank_kernel
# picks auto / scalar / avx2 / avx512, auto being the widest one built).
# Measured exactly as hn_g5_rr.sh measured the scalar version: huge pages,
# QUERY_SB, 1 and 60 clients over the LAN, the same sweep, hnsw.rerank_limit=100.
# Label g5hp32kro-rr10simd-pgvectorfast-tq4.
#
# No index rebuild: only the server's library changes.
#   0. pgvectorfast_ntq's changed sources and the self-test to the 32 KB
#      checkout; bench/cloudlab/tq_rerank_bench.c
#   1. rebuild vectorfast.so (the previous one kept as vectorfast.so.rr10);
#      the codec self-test built with -march=native must pass (each vector
#      kernel against the scalar loop, the checksum of HEAD) and the
#      micro-benchmark gives each kernel's time per distance
#   2. huge pages: reserve, serve, check (SHOW hnsw.rerank_kernel = auto, and
#      SET hnsw.rerank_kernel = avx512 is accepted)
#   3. sweeps at 1 and 60 clients with the default kernel (AVX-512 here)
#   4. the kernels against each other in the same server session, at ef 64
#      and 320, 1 and 60 clients: avx2 and scalar (labels ...-kavx2, -kscalar).
#      Heavy 512-bit floating point can lower a Skylake core's clock, so the
#      narrower kernel may win under load.
#   5. results, logs, CPU traces and saturation report into bench/profiles/hn-g5-simd/
#
# Run from the workstation (no sync.sh push: its --delete removes builds that
# exist only on the server):
#   caffeinate -i bench/cloudlab/hn_g5_simd.sh
# Steps are detached and resumable (lib.sh); bump RUN to redo them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=bench/cloudlab/nodes-wisc.sh
. "$HERE/nodes-wisc.sh"
export ENV_FILE=bench/cloudlab/hn.env
export SMOKE=0
# shellcheck source=bench/cloudlab/lib.sh
. "$HERE/lib.sh"
# shellcheck source=bench/cloudlab/hp_lib.sh
. "$HERE/hp_lib.sh"

RUN="${RUN:-1}"
CLIENTS=60
RR=100
T="hng5-simd-r$RUN"
LABEL=g5hp32kro-rr10simd-pgvectorfast-tq4
CHECKSUM=251cc2ee3b2c0245     # tq_selftest on the codec before this change
D32=/mydata/vectordb_experiments_32k
NTQ32="$D32/pgvectorfast_ntq/vectorfast.so"
TABLE=items_hn_pipnn_ro
TQ_ARGS="--index hnsw --table $TABLE --m \$PIPNN_M --ef-construction 64 --quantizer tq4 --rerank ntq12 --set hnsw.rerank_limit=$RR"
PD=bench/profiles/hn-g5-simd

ENVS8="$ENVS"
ENVS32="${ENVS/cd $DEST /cd $D32 }"
[ "$ENVS32" != "$ENVS8" ] || { echo "error: could not point ENVS at $D32" >&2; exit 1; }
on32() { local save="$ENVS"; ENVS="$ENVS32"; "$@"; ENVS="$save"; }
NOPG='if pgrep -x postgres >/dev/null; then echo "error: a Postgres is running"; exit 1; fi'

require_measurement_state

# ---- 0. The changed sources ------------------------------------------------------
( cd "$HERE/../.." && rsync -azR -e "ssh ${SSH_OPTS[*]}" \
    pgvectorfast_ntq/src/hnsw.c pgvectorfast_ntq/src/hnsw.h pgvectorfast_ntq/src/hnswtqam.c \
    pgvectorfast_ntq/src/hnswtq.c pgvectorfast_ntq/src/hnswtq.h pgvectorfast_ntq/tools/tq_check/tq_selftest.c \
    bench/cloudlab/tq_rerank_bench.c "$REMOTE_USER@$SERVER.$DOMAIN:$D32/" )

# ---- 1. vectorfast.so with the vector kernels; self-test; micro-benchmark ----------
# shellcheck disable=SC2016  # expanded on the node
on32 step "$SERVER" "$T-build" "$NOPG"'
  set -eo pipefail
  PGC=$PWD/postgres/install/bin/pg_config
  so=pgvectorfast_ntq/vectorfast.so
  [ -f $so.rr10 ] || cp -p $so $so.rr10
  make -C pgvectorfast_ntq clean PG_CONFIG=$PGC > /dev/null
  make -C pgvectorfast_ntq -j"$(nproc)" PG_CONFIG=$PGC 2>&1 | grep -E " error|error:|warning:" || true
  st=$(strings $so)
  for s in hnsw.rerank_kernel hnsw.rerank_limit "quantizer \"tq4\""; do
    grep -q "$s" <<< "$st" || { echo "error: no $s in $so"; exit 1; }
  done
  [ "$(nm $so | grep -c " calico_")" -gt 0 ] || { echo "error: no calico symbols in $so"; exit 1; }
  grep -qE "^#define BLCKSZ 32768$" "$($PGC --includedir-server)/pg_config.h" ||
    { echo "error: this Postgres is not built with 32 KB pages"; exit 1; }
  n=$(objdump -d $so | grep -c "vgatherdps.*zmm" || true)
  echo "AVX-512 gathers in $so: $n"; [ "$n" -gt 0 ] || { echo "error: no AVX-512 gather in $so"; exit 1; }
  t=/tmp/hng5-simd; mkdir -p $t
  cc -O2 -ffp-contract=off -march=native -Ipgvectorfast_ntq/src pgvectorfast_ntq/src/hnswtq.c     pgvectorfast_ntq/tools/tq_check/tq_selftest.c -lm -o $t/tq_selftest
  $t/tq_selftest | tee $t/selftest.out | tail -4
  grep -q "avx2 built, avx512 built" $t/selftest.out || { echo "error: the self-test has no AVX-512 kernel"; exit 1; }
  grep -q "^checksum: '"$CHECKSUM"'$" $t/selftest.out || { echo "error: the codec checksum changed"; exit 1; }
  grep -q "^OK$" $t/selftest.out || { echo "error: the self-test failed"; exit 1; }
  cc -O2 -ffp-contract=off -march=native -Ipgvectorfast_ntq/src pgvectorfast_ntq/src/hnswtq.c     bench/cloudlab/tq_rerank_bench.c -lm -o $t/tq_rerank_bench
  numactl --cpunodebind=0 --membind=0 $t/tq_rerank_bench 384
  ls -la $so $so.rr10; echo "built $so with the vector rerank kernels"'

# ---- 2-4. Huge pages; 1 and 60 clients; the kernels against each other -------------------------------------------------
step "$CLIENT" "$T-client-idle" \
  'if pgrep -x ginger_bench >/dev/null; then echo "error: a ginger_bench is running"; exit 1; fi; echo idle'
ssh_to "$SERVER" "if [ -f $STEPS/nic-$NIC.pid ] && kill -0 \$(cat $STEPS/nic-$NIC.pid) 2>/dev/null; then echo 'nic monitor running'; else nohup sh -c 'while :; do echo \"\$(date +%s) \$(cat /sys/class/net/$NIC/carrier 2>/dev/null) \$(cat /sys/class/net/$NIC/statistics/rx_crc_errors) \$(cat /sys/class/net/$NIC/statistics/rx_packets)\"; sleep 2; done' >> $STEPS/$T-nic.txt 2>&1 < /dev/null & echo \$! > $STEPS/nic-$NIC.pid; echo nic monitor started; fi"
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "if [ -f $STEPS/$T-cpu.pid ] && kill -0 \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; then echo 'cpu sampler running'; else nohup sh -c 'while :; do echo \"\$(date +%s) \$(head -1 /proc/stat)\"; sleep 2; done' >> $STEPS/$T-cpu.txt 2>&1 < /dev/null & echo \$! > $STEPS/$T-cpu.pid; echo cpu sampler started; fi"
done

on32 step "$SERVER" "$T-hp-reserve" "$HP_RESERVE"
on32 step "$SERVER" "$T-hp-serve" \
  "EXTRA_PG_OPTS='-c huge_pages=on' FAST_SO=$NTQ32 SERVE_ARM=fast SERVE_TABLE=$TABLE LISTEN_IP=$SERVER_IP CLIENT_NET=$CLIENT_NET bench/run_experiment.sh serve"
# shellcheck disable=SC2016  # expanded on the node
on32 step "$SERVER" "$T-hp-check" 'HP_MIN_GB=50; '"$HP_CHECK"'
  b=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW block_size")
  echo "block_size: $b"; [ "$b" = 32768 ] || { echo "error: not a 32 KB cluster"; exit 1; }
  r=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SHOW hnsw.rerank_kernel")
  echo "hnsw.rerank_kernel default: $r"; [ "$r" = auto ] || { echo "error: the served library has no hnsw.rerank_kernel"; exit 1; }
  r=$(postgres/install/bin/psql -X -At -p $PGPORT -U postgres -d postgres -c "SET hnsw.rerank_kernel = avx512" -c "SHOW hnsw.rerank_kernel" | tail -1)
  echo "hnsw.rerank_kernel after SET avx512: $r"; [ "$r" = avx512 ] || { echo "error: the served library has no AVX-512 kernel"; exit 1; }'
bench_step "$CLIENT" "$T-tq4-c1" "$LABEL" "$TQ_ARGS --sweep \"\$EF_SWEEP\" --clients 1"
bench_step "$CLIENT" "$T-tq4-c$CLIENTS" "$LABEL" "$TQ_ARGS --sweep \"\$EF_SWEEP\" --clients $CLIENTS"
# The kernels against each other, same server session
for k in avx2 scalar; do
  bench_step "$CLIENT" "$T-k$k-c1" "$LABEL-k$k" "$TQ_ARGS --set hnsw.rerank_kernel=$k --sweep 64,320 --clients 1"
  bench_step "$CLIENT" "$T-k$k-c$CLIENTS" "$LABEL-k$k" "$TQ_ARGS --set hnsw.rerank_kernel=$k --sweep 64,320 --clients $CLIENTS"
done
on32 step "$SERVER" "$T-hp-release" "$HP_RELEASE"

# ---- 5. Results ---------------------------------------------------------------------
for n in "$SERVER" "$CLIENT"; do
  ssh_to "$n" "kill \$(cat $STEPS/$T-cpu.pid) 2>/dev/null; rm -f $STEPS/$T-cpu.pid; true"
done
ssh_to "$SERVER" "kill \$(cat $STEPS/nic-$NIC.pid) 2>/dev/null; rm -f $STEPS/nic-$NIC.pid; true"
LPD="$HERE/../../$PD"
mkdir -p "$LPD/sat"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-cpu.txt" "$LPD/server-cpu.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$CLIENT.$DOMAIN:$STEPS/$T-cpu.txt" "$LPD/client-cpu.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-nic.txt" "$LPD/nic.txt"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$SERVER.$DOMAIN:$STEPS/$T-*.log" "$LPD/"
ssh_to "$CLIENT" "grep -E '\"label\": \"$LABEL\"' $DEST/bench/results/hackernews_minilm_384.jsonl" > "$LPD/results.jsonl"
ssh_to "$CLIENT" "grep -E '\"label\": \"$LABEL-k' $DEST/bench/results/hackernews_minilm_384.jsonl" > "$LPD/kernels.jsonl"
ssh_to "$SERVER" "cat /tmp/hng5-simd/selftest.out" > "$LPD/selftest.out"
python3 - "$LPD/results.jsonl" "$CLIENTS" > "$LPD/sat/results.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    if json.loads(line).get("clients") == int(sys.argv[2]):
        print(line, end="")
PY
cp "$LPD/server-cpu.txt" "$LPD/client-cpu.txt" "$LPD/nic.txt" "$LPD/sat/"
python3 "$HERE/saturation_report.py" "$LPD/sat" | tee "$LPD/saturation.md"
say "done; label $LABEL in $PD/results.jsonl"
