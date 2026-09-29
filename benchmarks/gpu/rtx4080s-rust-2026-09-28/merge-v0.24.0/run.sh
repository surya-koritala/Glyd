#!/bin/bash
# rust-gpu merged with v0.24.0 (main 44393a9) on the box: the library and the crate built, the crate's tests (with the
# library and the GPU), the self-test, check_capi, test_gpu.py, check_api dense (Qwen3-0.6B, Qwen3-1.7B) and MoE
# (granite-3.1-3b-a800m-instruct). One at a time, each under the box's lock once the GPU is idle, on 8 CPUs.
set -u
W=~/p9rust2; cd $W && source ~/gpuenv/cuda.sh
export HF_HOME=~/p1/hf HF_HUB_OFFLINE=1 TMPDIR=$W/tmp TORCH_EXTENSIONS_DIR=$W/torch_ext CUDA_CACHE_PATH=~/p1/nvcache
export PATH=~/.cargo/bin:$PATH CARGO_TARGET_DIR=$W/target CARGO_BUILD_JOBS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_JOBS=8 TORCHINDUCTOR_COMPILE_THREADS=8
LOCK=~/.glyd-box.lock L=$W/logs PY=~/pypi-venv/bin/python LIB=$W/lib/libglyd_gpu_cuda13.so
mkdir -p $L $W/tmp
run() {
    flock $LOCK bash -c "until [ -z \"\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)\" ]; do sleep 20; done
        { echo \"# \$(date '+%F %T'): load \$(cut -d' ' -f1-3 /proc/loadavg); busiest: \$(ps -eo pcpu=,comm= --sort=-pcpu | head -n 3 | tr -s ' ' | tr '\n' ';')\"; $2; } >> $L/$1 2>&1"
}
run build.txt "bash $W/src/gpu/build_lib.sh $W/lib 2>&1 | tail -n 3 && cd $W/src && cargo build --release -p glyd-gpu --bins --examples -j 8 2>&1 | tail -n 1"
export GLYD_GPU_LIB=$LIB PYTHONPATH=$W/src/bindings/python
run cargo_test.txt "cd $W/src && cargo test --release -p glyd-gpu -j 8 -- --test-threads 1 2>&1 | grep -E 'test |test result'; echo exit \${PIPESTATUS[0]}"
run selftest.txt "cd $W/src/gpu && taskset -c 0-7 $PY glyd_gpu.py; echo exit \$?"
run check_capi.txt "cd $W/src/gpu && taskset -c 0-7 $PY check_capi.py $LIB; echo exit \$?"
run test_gpu.txt "cd $W/src/bindings/python && taskset -c 0-7 $PY test_gpu.py 2>&1 | grep -v 'Loading weights\|Writing model\|^\[transformers\]'; echo exit \${PIPESTATUS[0]}"
run check_api-dense.txt "cd $W/src/gpu && taskset -c 0-7 $PY check_api.py Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B 2>&1 | grep -v 'Loading weights\|Writing model\|Fetching\|^\[transformers\]'; echo exit \${PIPESTATUS[0]}"
run check_api-moe.txt "cd $W/src/gpu && taskset -c 0-7 $PY check_api.py ibm-granite/granite-3.1-3b-a800m-instruct 2>&1 | grep -v 'Loading weights\|Writing model\|Fetching\|^\[transformers\]'; echo exit \${PIPESTATUS[0]}"
echo "all done $(date '+%F %T')" >> $L/run.txt
