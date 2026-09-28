# v0.22.0's final check on the box (RTX 4080 SUPER 16 GB): the CI-built wheel in a fresh venv, then the checks and the
# numbers, bf16 against Glyd where bf16 fits (Qwen3-4B-Instruct-2507, granite MoE), Qwen3-8B Glyd alone. Everything in
# ~/p3final; models from ~/p1/hf (nothing new downloaded but the small check models if missing). DONE at the end.
set -x
H=~/p3final; R=$H/results; mkdir -p $R $H/t $H/src
export HF_HOME=~/p1/hf TMPDIR=~/p1/tmp TORCH_EXTENSIONS_DIR=~/p1/torch_ext CUDA_CACHE_PATH=~/p1/nvcache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/gpuenv/cuda.sh > /dev/null 2>&1  # nvcc for the JIT build (the checks call $PY explicitly)
{ nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv; nproc; lscpu | grep "Model name"; free -g | head -2; } > $R/machine.txt 2>&1
UV=$(command -v uv || echo $H/uvd/uv); [ -x "$UV" ] || { mkdir -p $H/uvd && curl -sL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar xz --strip-components=1 -C $H/uvd; UV=$H/uvd/uv; }
W=$(ls $H/glyd-*manylinux_2_28_x86_64.whl)
$UV venv -q --python 3.12 $H/wv
$UV pip install --python $H/wv/bin/python "glyd[gpu] @ file://$W" ninja > $R/install.txt 2>&1
PY=$H/wv/bin/python
$PY -c "import torch, transformers, accelerate, glyd, glyd.gpu; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(), 'transformers', transformers.__version__, glyd.__version__)" >> $R/install.txt 2>&1
tar -C $H/src -xf $H/glyd-src.tar
LIB=$($PY -c "import glyd.gpu, os, torch; print(os.path.join(os.path.dirname(glyd.gpu.__file__), 'libglyd_gpu_cuda%s.so' % torch.version.cuda.split('.')[0]))")
run() { local name=$1 min=$2; shift 2; ( cd $H/src/gpu && timeout ${min}m "$@" ) > $R/$name.txt 2>&1; echo "exit $?" >> $R/$name.txt; }
cp $H/src/gpu/check_api.py $H/src/bindings/python/test_gpu.py $H/src/bindings/python/test_gpu_site.json $H/t/
( cd $H/t && timeout 40m $PY check_api.py Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B ibm-granite/granite-3.1-3b-a800m-instruct ) > $R/check_api.txt 2>&1; echo "exit $?" >> $R/check_api.txt
run check_capi 30 $PY check_capi.py $LIB
run selftest 20 $PY glyd_gpu.py
( cd $H/t && timeout 30m $PY test_gpu.py ) > $R/test_gpu.txt 2>&1; echo "exit $?" >> $R/test_gpu.txt
M4=Qwen/Qwen3-4B-Instruct-2507
run e2e-4b 30 env GLYD_GPU_LIB=$LIB $PY e2e.py $M4 --format auto --fused --merge --baseline --tokens 64 --batch 1,8,32,64
run e2e-4b-profile 40 env GLYD_GPU_LIB=$LIB $PY e2e.py $M4 --format auto --fused --merge --baseline --tokens 32 --batch 1,8,32,64 --profile 16 --prefill 128,512,1024,2048,4096
run e2e-4b-mma12 40 env GLYD_GPU_LIB=$LIB $PY e2e.py $M4 --format mma12 --fused --merge --tokens 32 --batch 1,8 --prefill 128,512,1024,2048,4096
run e2e-8b 30 env GLYD_GPU_LIB=$LIB $PY e2e.py Qwen/Qwen3-8B --format auto --fused --merge --tokens 64 --batch 1,8,32 --prefill 128,512,2048
run e2e-granite 30 env GLYD_GPU_LIB=$LIB $PY e2e.py ibm-granite/granite-3.1-3b-a800m-instruct --format auto --fused --merge --baseline --tokens 64 --batch 1,8 --prompts --profile 16 --prefill 128,512,2048
for w in bf16 glyd; do
  ( cd $H && timeout 15m $PY $H/gen_eager.py $M4 $w 1 128 2>&1 | grep tokens/s | tail -1 ) >> $R/eager.txt
  ( cd $H && COMPILE=1 timeout 20m $PY $H/gen_eager.py $M4 $w 1 128 2>&1 | grep -E "tokens/s|Error" | tail -1 ) >> $R/compiled.txt
done
touch $R/DONE
