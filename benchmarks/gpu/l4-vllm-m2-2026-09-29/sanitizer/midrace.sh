#!/bin/bash
# racecheck's hazards in mma12_mid_kernel: the library again with -lineinfo (sm_89), racecheck on the MID route alone
# with every report and its source lines; then midstress.py (the MID route 2000 times a shape and M). Under the lock.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; V=$W/venv/bin; O=$W/m3prep/midrace; L=$W/lib-v0251-li; mkdir -p $O $L
source ~/gpuenv/cuda.sh
export PYTHONSAFEPATH=1 PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp GLYD_GPU_LIB=$L/libglyd_gpu_cuda13.so
CU=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)
F="-O3 -lineinfo -std=c++20 --expt-relaxed-constexpr -isystem $CU/include -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode arch=compute_89,code=sm_89 -Xcompiler -fPIC,-fvisibility=hidden"
echo "== $(date -u +%T) build with -lineinfo"
[ -f $L/libglyd_gpu_cuda13.so ] || flock ~/.glyd-box.lock bash -c "nvcc $F -c -o $L/glyd_gpu.o $W/glyd/gpu/glyd_gpu.cu && nvcc $F -shared -Xlinker --exclude-libs,ALL -cudart static -L$CU/lib -L$CU/lib64 -o $L/libglyd_gpu_cuda13.so $L/glyd_gpu.o" > $O/build.txt 2>&1; echo "exit $?"
echo "== $(date -u +%T) racecheck, the MID route (12-bit, 17 and 64 tokens), every report"
(cd $O && OPCHECK_SHAPES="2048,2048" OPCHECK_MS="17,64" PYTORCH_NO_CUDA_MEMORY_CACHING=1 flock ~/.glyd-box.lock timeout 1200 /usr/local/cuda-13.0/bin/compute-sanitizer --tool racecheck --racecheck-report all --print-limit 40 --show-backtrace device $V/python $W/opcheck.py) > $O/racecheck.txt 2>&1; echo "exit $?"
grep -E "RACECHECK SUMMARY|ERROR SUMMARY" $O/racecheck.txt
echo "== $(date -u +%T) midstress.py"
(cd $O && flock ~/.glyd-box.lock timeout 1200 $V/python $W/midstress.py 2000) > $O/midstress.txt 2>&1; echo "exit $?"
cat $O/midstress.txt
