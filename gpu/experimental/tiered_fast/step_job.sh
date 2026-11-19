#!/usr/bin/env bash
# step_job.sh: the library's step kernels on this GPU (A100: bin/harness_80, H100/GH200: bin/harness_90, L4 rehearsal: bin/harness_89), built from
# harness_step.cu. Needs only the NVIDIA driver. Downloads nothing, sets nothing (clocks are recorded, not changed).
#   ./step_job.sh          the whole job (about 10 min on an A100/H100, most of it ncu; 3 min without it: QUICK=1)
#   QUICK=1 ./step_job.sh  the gate's numbers alone, no ncu
cd "$(dirname "$0")" || exit 1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
case $CC in 80) H=./bin/harness_80;; 90) H=./bin/harness_90;; 89) H=./bin/harness_89;; *) echo "no binary for compute capability $CC"; exit 1;; esac
OUT=results-$(hostname)-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
exec > >(tee "$OUT/job.log") 2>&1
echo "== $(date -u) $(hostname) cc $CC binary $H"
nvidia-smi -q > "$OUT/nvidia-smi-q.txt" 2>&1
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit,memory.total --format=csv | tee "$OUT/gpu.csv"
SECONDS=0

# variants: s = the library's product of the tiered layout (the shipped kernel), n = the 12-bit layout's, o = the tiered one's 8-warp kernel (before the
# step kernel), rn rs = the routes at 17-64 tokens
V1="s,n,o,"
VN="s,n,o,"

run() {  # run NAME PACK M ITERS VARIANTS [ENV=VALUE...]
  local name=$1 pack=$2 M=$3 it=$4 vars=$5; shift 5
  echo "-- $name  (${SECONDS}s)"
  env "$@" timeout 400 "$H" "packs/$pack" "$M" "$it" "$vars" > "$OUT/$name.txt" 2>&1 || echo "   failed or timed out: $name"
}

# 1. the real regime: weights streaming from HBM (u10: 12288x4096 up_proj of Qwen3-8B layer 10, 68 MB tiered / 75 MB 12-bit, three copies in turn)
run stream_u10_m1  u10 1  100 "$V1" COLD=1 ROUNDS=9
run stream_u10_m8  u10 8  100 "$V1" COLD=1 ROUNDS=9
run stream_u10_m16 u10 16 100 "$VN" COLD=1 ROUNDS=7
if [ -z "$QUICK" ]; then
  run stream_u10_m2 u10 2 100 "$V1" COLD=1 ROUNDS=7
  run stream_u10_m4 u10 4 100 "$V1" COLD=1 ROUNDS=7
  run stream_u10_m32 u10 32 100 "s,n,rs,rn," COLD=1 ROUNDS=5
  run stream_u10_m64 u10 64 100 "s,n,rs,rn," COLD=1 ROUNDS=5
fi
# 2. L2-resident (compute-bound): o10 = o_proj 4096x4096
run hot_o10_m1  o10 1  150 "$V1" ROUNDS=9
run hot_o10_m8  o10 8  150 "$V1" ROUNDS=9
run hot_o10_m16 o10 16 150 "$VN" ROUNDS=7
if [ -z "$QUICK" ]; then
  run hot_o10_m2 o10 2 150 "$V1" ROUNDS=7
  run hot_o10_m4 o10 4 150 "$V1" ROUNDS=7
  run hot_o10_m32 o10 32 100 "s,n,rs,rn," ROUNDS=5
  run hot_o10_m64 o10 64 100 "s,n,rs,rn," ROUNDS=5
fi
# 3. small matrices (c1: 1024x4096, c2: 2048x4096: 62 steps an SM on an H100)
run small_c1_m1 c1 1 200 "$V1" ROUNDS=7
run small_c2_m1 c2 1 200 "$V1" ROUNDS=7
run small_c2_m8 c2 8 200 "$V1" ROUNDS=7
# 4. exactness of the general decode on this GPU: a heavy-tailed tensor (g1: 2048x4096 of Qwen3-8B layer 1 gate_proj, nearly every step takes it), a tensor with
# some steps flagged among the others (m2: Qwen3-4B layer 28 q_proj 4096x2560), and the flag set on every step, every 5th, every 64th of o10: the product must be
# the first's bits whatever the share, and the time says what such a step costs
run heavy_g1_m1 g1 1 100 "s,o," ROUNDS=5
run heavy_g1_m8 g1 8 100 "s,o," ROUNDS=5
run mixed_m2_m1 m2 1 100 "s,o," ROUNDS=5
run mixed_m2_m8 m2 8 100 "s,o," ROUNDS=5
run mixed_m2_m16 m2 16 100 "s,o," ROUNDS=5
run flags_o10_m1 o10 1 150 "s,o," ROUNDS=5 FLAGEVERY=1,5,64
run flags_o10_m8 o10 8 150 "s,o," ROUNDS=5 FLAGEVERY=1,5,64
run flags_o10_m16 o10 16 150 "s,o," ROUNDS=5 FLAGEVERY=1,5,64
# 5. ncu (needs counters: sudo), one launch each, o10: the step kernel at 1, 8 and 16 tokens and the 12-bit layout's at 1 and 8 (pipe use, issue and the stalls)
if [ -z "$QUICK" ]; then
  mkdir -p "$OUT/ncu"
  NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null | tail -1)
  if [ -n "$NCU" ]; then
    SUDO=""; [ "$(id -u)" != 0 ] && SUDO="sudo -n -E env PATH=$PATH"
    ncu_one() {  # ncu_one NAME 'kernel regex' TOKENS VARIANT
      echo "-- ncu $1  (${SECONDS}s)"
      ROUNDS=1 $SUDO timeout 600 "$NCU" --cache-control none --clock-control none --kernel-name-base demangled -k "regex:$2" \
        --launch-skip 2 --launch-count 1 --set full -f -o "$OUT/ncu/$1" "$H" packs/o10 "$3" 1 "$4" > "$OUT/ncu/$1.log" 2>&1 || echo "   ncu failed: $1"
      $SUDO "$NCU" -i "$OUT/ncu/$1.ncu-rep" --page raw --csv > "$OUT/ncu/$1.raw" 2>/dev/null
      $SUDO chown "$(id -u)" "$OUT/ncu/$1".* 2>/dev/null
    }
    ncu_one step_m1 'step_kernel' 1 "s,"
    ncu_one step_m8 'step_kernel' 8 "s,"
    ncu_one step_m16 'step_kernel' 16 "s,"
    ncu_one mma12_m1 'mma_gemm_kernel<Nib, \(int\)1' 1 "n,"
    ncu_one mma12_m8 'mma_gemm_kernel<Nib, \(int\)1' 8 "n,"
  else
    echo "ncu not found: skipped"
  fi
fi
echo "== done in ${SECONDS}s"
command -v python3 > /dev/null && python3 summarize_step.py "$OUT" | tee "$OUT/summary.txt"
echo "results in $OUT (send the directory back: tar czf $OUT.tgz $OUT)"
