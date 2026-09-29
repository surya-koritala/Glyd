#!/bin/bash
# main's glyd (0.23, origin/main 151d146) loading and verifying the branch's glyd-v1 save of Qwen3-0.6B, whose glyd.json
# carries "tensors": the key ignored. Under the box-wide lock once the GPU is idle.
W=~/p9rust2; source ~/gpuenv/cuda.sh
export HF_HOME=~/p1/hf HF_HUB_OFFLINE=1 TMPDIR=$W/tmp TORCH_EXTENSIONS_DIR=$W/torch_ext CUDA_CACHE_PATH=~/p1/nvcache OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export GLYD_GPU_LIB=$W/main/lib/libglyd_gpu_cuda13.so PYTHONPATH=$W/main/bindings/python
mkdir -p $W/logs2
until [ -f $W/out/Qwen3-0.6B-mma/glyd.json ] && grep -q "Qwen3-0.6B mma12, cuda:0" $W/logs/verify.txt; do sleep 30; done
flock ~/.glyd-box.lock bash -c 'until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 20; done
cd ~/p9rust2 && ~/pypi-venv/bin/python -c "
import glyd, glyd.gpu as gg, json
p = \"/home/surya-koritala/p9rust2/out/Qwen3-0.6B-mma\"
m = json.load(open(p + \"/glyd.json\"))
print(\"glyd\", glyd.__version__, \"from\", glyd.__file__)
print(p, m[\"format\"], \"tensors\" in m, len(m[\"tensors\"]), \"sha256 saved as they are\")
model = gg.from_pretrained(p, verify=True)
print(\"loaded and verified by main: \", model.config.quantization_config.verified, \"tensors\")
" 2>&1 | grep -v "Loading weights"; echo exit ${PIPESTATUS[0]}' > $W/logs2/main-reads-new-save.txt 2>&1
