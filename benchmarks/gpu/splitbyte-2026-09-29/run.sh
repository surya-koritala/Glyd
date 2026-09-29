#!/bin/bash
# Split byte on an RTX 4080 SUPER (Ryzen 9 7950X3D), the box shared with other jobs: every step under the box-wide lock
# (flock ~/.glyd-box.lock) once the GPU is idle, one at a time, on 8 CPUs (taskset) at most. Each step's log starts
# with the box's state as it began (load averages, the busiest processes). Work in $W: this tree in $W/src, main's
# (origin/main) in $W/main, each with its library in lib/; logs in $W/logs.
#   W=~/p9sb bash run.sh
set -u
W=${W:-~/p9sb}
cd $W && source ~/gpuenv/cuda.sh
export HF_HOME=~/p1/hf HF_HUB_OFFLINE=1 TMPDIR=$W/tmp TORCH_EXTENSIONS_DIR=$W/torch_ext CUDA_CACHE_PATH=~/p1/nvcache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_JOBS=8 TOKENIZERS_PARALLELISM=false
LOCK=~/.glyd-box.lock L=$W/logs PY=~/pypi-venv/bin/python T="taskset -c 0-7"
LIB=$W/src/lib/libglyd_gpu_cuda13.so MLIB=$W/main/lib/libglyd_gpu_cuda13.so B=$W/src/benchmarks/gpu/splitbyte-2026-09-29
CUDA=$(dirname "$(dirname "$(command -v nvcc)")")
mkdir -p $L $W/tmp $W/sass
# run LOG 'CMD': CMD under the lock once the GPU is idle, its output appended to LOG after a line of the box's state as it
# began: the time, the load averages, the three busiest processes (%CPU and name)
run() {
    flock $LOCK bash -c "until [ -z \"\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)\" ]; do sleep 20; done
        { echo \"# \$(date '+%F %T'): load \$(cut -d' ' -f1-3 /proc/loadavg); busiest: \$(ps -eo pcpu=,comm= --sort=-pcpu | head -n 3 | tr -s ' ' | tr '\n' ';')\"; $2; } >> $L/$1 2>&1"
}
QUIET="grep -v 'Loading weights\|Writing model\|Fetching\|^\[transformers\]'"

# the builds: this tree's library and main's (gpu/build_lib.sh: sm_80, 86, 89, 90a, 100, 120 and compute_80), where not
# built yet
[ -f $LIB ] && [ -f $MLIB ] || run build.txt "time $T bash $W/src/gpu/build_lib.sh $W/src/lib && time $T bash $W/main/gpu/build_lib.sh $W/main/lib"

# SASS: each architecture's kernels from the two libraries, compared (sass_diff.py)
for a in 80 86 89 90a; do
    for side in main src; do
        cuobjdump -sass -arch sm_$a $W/$side/lib/libglyd_gpu_cuda13.so > $W/sass/$side-sm_$a.sass
        cuobjdump -res-usage -arch sm_$a $W/$side/lib/libglyd_gpu_cuda13.so > $W/sass/$side-sm_$a.res
    done
    $PY $B/sass_diff.py $W/sass/main-sm_$a.sass $W/sass/src-sm_$a.sass $W/sass/main-sm_$a.res $W/sass/src-sm_$a.res > $L/sass-sm_$a.txt 2>&1
done

export GLYD_GPU_LIB=$LIB PYTHONPATH=$W/src/bindings/python
# the checks: the self-test, check_capi (the JIT build against the library, bit for bit), test_gpu.py, check_api
run selftest.txt "cd $W/src/gpu && $T $PY glyd_gpu.py; echo exit \$?"
run check_capi.txt "cd $W/src/gpu && $T $PY check_capi.py $LIB; echo exit \$?"
run test_gpu.txt "cd $W/src/bindings/python && $T $PY test_gpu.py 2>&1 | $QUIET; echo exit \${PIPESTATUS[0]}"
run check_api-dense.txt "cd $W/src/gpu && $T $PY check_api.py Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B 2>&1 | $QUIET; echo exit \${PIPESTATUS[0]}"
run check_api-moe.txt "cd $W/src/gpu && $T $PY check_api.py ibm-granite/granite-3.1-3b-a800m-instruct 2>&1 | $QUIET; echo exit \${PIPESTATUS[0]}"

# split byte against main's 12-bit layout in the same kernels (xcheck.py), then models end to end in the 12-bit layout
# (e2e12.py: main's package and library, then this tree's, then main's again; the lines compared)
run xcheck.txt "cd $B && $T $PY xcheck.py $W/main $MLIB Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B ibm-granite/granite-3.1-3b-a800m-instruct; echo exit \$?"
E2E="Qwen/Qwen3-1.7B Qwen/Qwen3-4B-Instruct-2507 ibm-granite/granite-3.1-3b-a800m-instruct"
for side in main src main2; do
    tree=$W/${side%2}
    run e2e12-$side.txt "cd $B && GLYD_COMPILE=0 GLYD_GPU_LIB=$tree/lib/libglyd_gpu_cuda13.so PYTHONPATH=$tree/bindings/python $T $PY e2e12.py $E2E 2>&1 | $QUIET; echo exit \${PIPESTATUS[0]}"
done
{ grep -v '^#' $L/e2e12-main.txt | cmp -s - <(grep -v '^#' $L/e2e12-main2.txt) && echo "main twice: the same lines" || echo "main twice: DIFFERS";
  grep -v '^#' $L/e2e12-main.txt | cmp -s - <(grep -v '^#' $L/e2e12-src.txt) && echo "main and split byte: the same lines ($(grep -c logits $L/e2e12-src.txt) logits, $(grep -c 'tokens:' $L/e2e12-src.txt) generations)" || echo "main and split byte: DIFFER"; } > $L/e2e12-compare.txt

# one layer's products and decodes, main's 12-bit layout against split byte (layer.py), two runs a model
for round in 1 2; do
    for m in Qwen/Qwen3-8B Qwen/Qwen3-4B-Instruct-2507; do
        run layer-$(basename $m)-run$round.txt "cd $B && $T $PY layer.py $W/main $MLIB $m; echo exit \$?"
    done
done

# The grid prompt kernels (an A10's, and any GPU's but GeForce Ada and the A100) on this GPU: main's and this tree's
# glyd_gpu.cu built for sm_89 with the library's test for GeForce by name never matching ("GeForce-not"), so this GPU
# takes the path of an Ada that is not GeForce (an L40S): mma_gemm_big's variants 1 and 2 the grid kernels, its routes
# by compute capability alone; xcheck.py's matrices through them.
F="-O3 -std=c++20 --expt-relaxed-constexpr -isystem $CUDA/include -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode arch=compute_89,code=sm_89 -Xcompiler -fPIC,-fvisibility=hidden"
for side in main src; do
    mkdir -p $W/grid/$side
    sed 's/"GeForce")/"GeForce-not")/' $W/$side/gpu/glyd_gpu.cu > $W/grid/$side/glyd_gpu.cu
    cp $W/$side/gpu/glyd_gpu.h $W/grid/$side/
    run grid-build.txt "cd $W/grid/$side && grep -c 'GeForce-not' glyd_gpu.cu && time $T nvcc $F -c -o glyd_gpu.o glyd_gpu.cu && nvcc $F -shared -Xlinker --exclude-libs,ALL -cudart static -L$CUDA/lib -o libglyd_gpu_cuda13.so glyd_gpu.o"
done
run xcheck-grid.txt "cd $B && GLYD_GPU_LIB=$W/grid/src/libglyd_gpu_cuda13.so $T $PY xcheck.py $W/main $W/grid/main/libglyd_gpu_cuda13.so; echo exit \$?"
echo "all done $(date '+%F %T')" >> $L/run.txt
