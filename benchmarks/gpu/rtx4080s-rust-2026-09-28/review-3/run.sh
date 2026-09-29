#!/bin/bash
# Review 3's fixes on the box: the library built, check_capi and test_gpu.py, and a malformed route variable at the
# package's import. One at a time, each under the box's lock once the GPU is idle, on 8 CPUs.
set -u
W=~/p9rust2; cd $W && source ~/gpuenv/cuda.sh
export HF_HOME=~/p1/hf HF_HUB_OFFLINE=1 TMPDIR=$W/tmp TORCH_EXTENSIONS_DIR=$W/torch_ext CUDA_CACHE_PATH=~/p1/nvcache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_JOBS=8 TORCHINDUCTOR_COMPILE_THREADS=8
LOCK=~/.glyd-box.lock L=$W/logs PY=~/pypi-venv/bin/python LIB=$W/lib/libglyd_gpu_cuda13.so
mkdir -p $L $W/tmp
run() {
    flock $LOCK bash -c "until [ -z \"\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)\" ]; do sleep 20; done
        { echo \"# \$(date '+%F %T'): load \$(cut -d' ' -f1-3 /proc/loadavg); busiest: \$(ps -eo pcpu=,comm= --sort=-pcpu | head -n 3 | tr -s ' ' | tr '\n' ';')\"; $2; } >> $L/$1 2>&1"
}
run build.txt "bash $W/src/gpu/build_lib.sh $W/lib 2>&1 | tail -n 3"
export GLYD_GPU_LIB=$LIB PYTHONPATH=$W/src/bindings/python
run check_capi.txt "cd $W/src/gpu && taskset -c 0-7 $PY check_capi.py $LIB; echo exit \$?"
run test_gpu.txt "cd $W/src/bindings/python && taskset -c 0-7 $PY test_gpu.py 2>&1 | grep -v 'Loading weights\|Writing model\|^\[transformers\]'; echo exit \${PIPESTATUS[0]}"
run route_env.txt "cd $W && for v in 1e3 '' 2k 512; do GLYD_WG_MAX=\"\$v\" $PY -c 'import glyd.gpu.model as gm; print(\"imported\")' 2>&1 | tail -n 1 | sed \"s|^|GLYD_WG_MAX='\$v': |\"; done"
echo "all done $(date '+%F %T')" >> $L/run.txt
