#!/bin/bash
# test_gpu.py (and the Python tests of the files' checks) from ~/p9rust2/src2 against the job's library, under the box's
# lock once the GPU is idle; logs in ~/p9rust2/logs2.
W=~/p9rust2; source ~/gpuenv/cuda.sh
export HF_HOME=~/p1/hf HF_HUB_OFFLINE=1 TMPDIR=$W/tmp TORCH_EXTENSIONS_DIR=$W/torch_ext CUDA_CACHE_PATH=~/p1/nvcache OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_JOBS=8
export GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda13.so PYTHONPATH=$W/src2/bindings/python
mkdir -p $W/logs2 $W/tmp
flock ~/.glyd-box.lock bash -c 'until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 20; done
  cd ~/p9rust2/src2/bindings/python && date && taskset -c 0-7 ~/pypi-venv/bin/python test_gpu.py 2>&1 | grep -v "Loading weights\|Writing model\|^\[transformers\]"; echo exit ${PIPESTATUS[0]}' > $W/logs2/test_gpu.txt 2>&1
