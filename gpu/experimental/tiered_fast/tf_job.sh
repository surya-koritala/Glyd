#!/usr/bin/env bash
# tf_job.sh: the tiered-fast harness on this GPU (A100: bin/harness_80, H100/GH200: bin/harness_90, L4 rehearsal: bin/harness_89).
# Needs only the NVIDIA driver (ncu optional). Downloads nothing, sets nothing (clocks are recorded, not changed).
#   ./tf_job.sh          the whole job (about 25 min on an A100/H100)
#   QUICK=1 ./tf_job.sh  without the ncu step and the long runs (about 8 min)
cd "$(dirname "$0")" || exit 1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
case $CC in 80) H=./bin/harness_80;; 90) H=./bin/harness_90;; 89) H=./bin/harness_89;; *) echo "no binary for compute capability $CC"; exit 1;; esac
OUT=results-$(hostname)-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT/ncu"
exec > >(tee "$OUT/job.log") 2>&1
echo "== $(date -u) $(hostname) cc $CC binary $H"
nvidia-smi -q > "$OUT/nvidia-smi-q.txt" 2>&1
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit,memory.total --format=csv | tee "$OUT/gpu.csv"
SECONDS=0

# variants (tokens of the harness: n = the library's 12-bit kernel, l = its T2 kernel (the library's T1 kernel always runs); oOPTuUNRwNWbMB = the
# prototype step kernels: OPT bits as in v2.cuh's header (552 = flagged hot/cold decode, 568 = the same decoding two steps together, 33320/16936 = with
# the group table 16/32 times over), UNR 1 one step 3 a pair, NW warps a block, MB blocks an SM; a trailing s = swapped MMA operands, up to 8 tokens)
V1="n,l,o40u1w4,o552u1w4b4s,o568u3w4b4s,o536u3w4b4s,o552u1w2b8s,o520u1w2b8s,o552u1w1b16s,o152u3w4b4s,o168u1w2b8s,o33320u1w4b4s,o33336u3w4b4s,o16936u1w4b2s,o16952u3w4b2s,"
VN="n,l,o40u1w4,o552u1w4b4,o568u3w4b4,o552u1w2b8,o33320u1w4b4,o16936u1w4b2,"
BEST="${BEST:-o568u3w4b4s}"        # for ncu (a prototype kernel)

run() {  # run NAME PACK M ITERS VARIANTS [ENV=VALUE...]
  local name=$1 pack=$2 M=$3 it=$4 vars=$5; shift 5
  echo "-- $name  (${SECONDS}s)"
  env "$@" timeout 400 "$H" "packs/$pack" "$M" "$it" "$vars" > "$OUT/$name.txt" 2>&1 || echo "   failed or timed out: $name"
}

# 1. the real regime: weights streaming from HBM (u10: 12288x4096 up_proj of Qwen3-8B layer 10, 68 MB tiered / 75 MB 12-bit)
run stream_u10_m1  u10 1  100 "$V1" COLD=1 ROUNDS=7
run stream_u10_m8  u10 8  100 "$V1" COLD=1 ROUNDS=5
run stream_u10_m16 u10 16 100 "$VN" COLD=1 ROUNDS=5
[ -z "$QUICK" ] && run stream_u10_m32 u10 32 100 "n,l,o8u1," COLD=1 ROUNDS=5
[ -z "$QUICK" ] && run stream_u10_m64 u10 64 100 "n,l,o8u1," COLD=1 ROUNDS=5
# 2. L2-resident (compute-bound): o10 = o_proj 4096x4096
run hot_o10_m1  o10 1  150 "$V1" ROUNDS=7
run hot_o10_m8  o10 8  150 "$V1" ROUNDS=5
run hot_o10_m16 o10 16 150 "$VN" ROUNDS=5
[ -z "$QUICK" ] && run hot_o10_m32 o10 32 100 "n,l,o8u1," ROUNDS=5
[ -z "$QUICK" ] && run hot_o10_m64 o10 64 100 "n,l,o8u1," ROUNDS=5
# 2b. a heavy-tailed tensor (g1: 2048x4096 of Qwen3-8B layer 1 gate_proj; every step takes the general decode): exactness of the cold path on this GPU
run heavy_g1_m1 g1 1 100 "l,o552u1w4b4s,o552u1w2b8s,o568u3w4b4s,o536u3w4b4s," ROUNDS=5
# 2c. a tensor with 3.6% cold steps among hot ones (m2: Qwen3-4B layer 28 q_proj 4096x2560): the switch between the two decodes inside one product
run mixed_m2_m1 m2 1 100 "l,o552u1w4b4s,o552u1w2b8s,o568u3w4b4s,o536u3w4b4s," ROUNDS=5
run mixed_m2_m8 m2 8 100 "l,o552u1w4b4s,o552u1w2b8s,o568u3w4b4s,o536u3w4b4s," ROUNDS=5
run mixed_m2_m16 m2 16 100 "l,o552u1w4b4,o568u3w4b4," ROUNDS=5
# 2d. a small matrix (c2: 2048x4096, 62 steps an SM on an H100) and forced cold fractions (every 20th / 5th step flagged cold on o10: the general decode
# takes them; the product must not change; the time gives the cost of a cold step)
run small_c2_m1 c2 1 200 "n,l,o552u1w4b4s,o568u3w4b4s,o552u1w2b8s," ROUNDS=7
run small_c2_m8 c2 8 200 "n,l,o552u1w4b4s,o568u3w4b4s,o552u1w2b8s," ROUNDS=5
run cold5pc_o10_m1 o10 1 150 "l,o552u1w4b4s,o568u3w4b4s," ROUNDS=5 FLAGEVERY=20
run cold20pc_o10_m1 o10 1 150 "l,o552u1w4b4s,o568u3w4b4s," ROUNDS=5 FLAGEVERY=5
# 3. prompts: the fused prompt kernel of each layout
if [ -z "$QUICK" ]; then
  for M in 128 256 1024; do run big_o10_m$M o10 $M 20 "" ROUNDS=3; done
fi

# 4. ncu (needs counters: sudo): T1, T2 and 12-bit library kernels and the best prototype, one launch each, 1 token, o10
if [ -z "$QUICK" ]; then
  NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null | tail -1)
  if [ -n "$NCU" ]; then
    SUDO=""; [ "$(id -u)" != 0 ] && SUDO="sudo -n -E env PATH=$PATH"
    ncu_one() {  # ncu_one NAME 'kernel regex' VARIANTS
      echo "-- ncu $1  (${SECONDS}s)"
      ROUNDS=1 $SUDO timeout 600 "$NCU" --cache-control none --clock-control none --kernel-name-base demangled -k "regex:$2" \
        --launch-skip 2 --launch-count 1 --set full -f -o "$OUT/ncu/$1" "$H" packs/o10 1 1 "$3" > "$OUT/ncu/$1.log" 2>&1 || echo "   ncu failed: $1"
      $SUDO "$NCU" -i "$OUT/ncu/$1.ncu-rep" --page raw --csv > "$OUT/ncu/$1.raw" 2>/dev/null
      $SUDO chown "$(id -u)" "$OUT/ncu/$1".* 2>/dev/null
    }
    ncu_one t1_lib  'mma_gemm_kernel<Tiered, \(int\)1>' "l,"
    ncu_one t2_lib  'mma_gemm_kernel<Tiered2, \(int\)1>' "l,"
    ncu_one mma12   'mma_gemm_kernel<Nib, \(int\)1>' "n,"
    ncu_one best    'step_kernel' "$BEST,"
  else
    echo "ncu not found: skipped"
  fi
fi
echo "== done in ${SECONDS}s"
command -v python3 > /dev/null && python3 summarize.py "$OUT" | tee "$OUT/summary.txt"
echo "results in $OUT (send the directory back: tar czf $OUT.tgz $OUT)"
