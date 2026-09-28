# v0.22.0's final check on one GPU: the CI-built wheel installed as a user installs it, then every check and the numbers
# the release states, bf16 against Glyd on the same GPU. Needs ~/glyd-*manylinux_2_28_x86_64.whl and ~/glyd-src.tar
# (git archive of the tested ref), and ~/gen_eager.py. Results in ~/results; DONE at the end. Each step has its own timeout and exit line.
set -x
R=~/results; mkdir -p $R
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{ nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv; nproc; lscpu | grep "Model name"; free -g | head -2; ls -d /usr/local/cuda*; } > $R/machine.txt 2>&1
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
cv=$(nvidia-smi | grep -o "CUDA Version: [0-9]*" | grep -o "[0-9]*$")
mkdir -p ~/uvd && curl -sL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar xz --strip-components=1 -C ~/uvd
UV=~/uvd/uv; W=$(ls ~/glyd-*manylinux_2_28_x86_64.whl)
$UV venv -q --python 3.12 ~/wv
if [ "${cv:-0}" -lt 13 ]; then $UV pip install --python ~/wv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch > $R/install.txt 2>&1; fi
$UV pip install --python ~/wv/bin/python "glyd[gpu] @ file://$W" hf_transfer ninja >> $R/install.txt 2>&1
PY=~/wv/bin/python; HF=~/wv/bin/hf
$PY -c "import torch, transformers, accelerate, glyd, glyd.gpu; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), 'transformers', transformers.__version__, glyd.__version__)" >> $R/install.txt 2>&1
mkdir -p ~/src && tar -C ~/src -xf ~/glyd-src.tar
LIB=$($PY -c "import glyd.gpu, os, torch; print(os.path.join(os.path.dirname(glyd.gpu.__file__), 'libglyd_gpu_cuda%s.so' % torch.version.cuda.split('.')[0]))")
export PATH=~/wv/bin:/usr/local/cuda/bin:$PATH  # ninja (the JIT build) is in the venv
run() { local name=$1 min=$2; shift 2; ( cd ~/src/gpu && timeout ${min}m "$@" ) > $R/$name.txt 2>&1; echo "exit $?" >> $R/$name.txt; }

# correctness: the wheel's API, its library against a JIT build here, the self-test against fp32, the package's tests
mkdir -p ~/t && cp ~/src/gpu/check_api.py ~/t/  # run apart from the source tree: the wheel's glyd, not ~/src's
( cd ~/t && timeout 40m $PY check_api.py Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B ) > $R/check_api-dense.txt 2>&1; echo "exit $?" >> $R/check_api-dense.txt
( cd ~/t && timeout 40m $PY check_api.py ibm-granite/granite-3.1-3b-a800m-instruct ) > $R/check_api-moe.txt 2>&1; echo "exit $?" >> $R/check_api-moe.txt
run check_capi 30 $PY check_capi.py $LIB
run selftest 20 $PY glyd_gpu.py
( cd ~/src/bindings/python && GLYD_GPU_LIB=$LIB timeout 30m $PY test_gpu.py ) > $R/test_gpu.txt 2>&1; echo "exit $?" >> $R/test_gpu.txt  # from the tree: its import test copies the package beside it
run check_models 40 env GLYD_GPU_LIB=$LIB $PY check_models.py Qwen/Qwen3-8B

# speed, Qwen3-8B: tokens/s in fresh processes first, then GPU time a step and prompts (a profiler slows what follows it)
B=1,8,32,64; [ "${MEM%.*}" -ge 40000 ] && B=1,8,32,64,128
run e2e-8b 30 env GLYD_GPU_LIB=$LIB $PY e2e.py Qwen/Qwen3-8B --format auto --fused --merge --baseline --tokens 64 --batch $B
run e2e-8b-profile 40 env GLYD_GPU_LIB=$LIB $PY e2e.py Qwen/Qwen3-8B --format auto --fused --merge --baseline --tokens 32 --batch $B --profile 16 --prefill 128,512,1024,2048,4096
for w in bf16 glyd; do
  ( cd ~ && timeout 15m $PY ~/gen_eager.py Qwen/Qwen3-8B $w 1 128 ) > $R/eager-$w.txt 2>&1; echo "exit $?" >> $R/eager-$w.txt
  ( cd ~ && COMPILE=1 timeout 20m $PY ~/gen_eager.py Qwen/Qwen3-8B $w 1 128 ) > $R/compiled-$w.txt 2>&1; echo "exit $?" >> $R/compiled-$w.txt
done

# by GPU: the larger models each can hold, and a mixture of experts
case "$GPU" in
  *H100*|*H200*|*GH200*)
    run e2e-32b 50 env GLYD_GPU_LIB=$LIB $PY e2e.py Qwen/Qwen3-32B --format auto --fused --merge --baseline --tokens 32 --batch 1,8,32,64 --profile 16 --prefill 128,512,2048
    run e2e-30b-a3b 50 env GLYD_GPU_LIB=$LIB $PY e2e.py Qwen/Qwen3-30B-A3B --format auto --fused --merge --baseline --tokens 64 --batch 1,8 --prompts --profile 16 --prefill 128,512,2048
    # the wheel's CUDA 12 library (a torch built for CUDA 12) on Hopper: its 192- and 256-token kernels had never run here
    $UV venv -q --python 3.12 ~/wv12 && $UV pip install --python ~/wv12/bin/python --index-url https://download.pytorch.org/whl/cu128 torch > $R/install-cu12.txt 2>&1
    $UV pip install --python ~/wv12/bin/python "glyd[gpu] @ file://$W" hf_transfer >> $R/install-cu12.txt 2>&1
    LIB12=$(~/wv12/bin/python -c "import glyd.gpu, os; print(os.path.join(os.path.dirname(glyd.gpu.__file__), 'libglyd_gpu_cuda12.so'))")
    ( cd ~/t && timeout 30m ~/wv12/bin/python check_api.py Qwen/Qwen3-1.7B ) > $R/check_api-cu12.txt 2>&1; echo "exit $?" >> $R/check_api-cu12.txt
    run e2e-8b-cu12 30 env GLYD_GPU_LIB=$LIB12 ~/wv12/bin/python e2e.py Qwen/Qwen3-8B --format auto --fused --merge --tokens 32 --batch 1,32,64 --prefill 129,192,256,384,512 ;;
  *A100*|*A800*)
    run e2e-14b 50 env GLYD_GPU_LIB=$LIB $PY e2e.py Qwen/Qwen3-14B --format auto --fused --merge --baseline --tokens 32 --batch 1,8,32,64,128 --profile 16 --prefill 128,512,2048 ;;
  *RTX\ PRO\ 6000*|*Blackwell*)
    run e2e-32b 50 env GLYD_GPU_LIB=$LIB $PY e2e.py Qwen/Qwen3-32B --format auto --fused --merge --baseline --tokens 32 --batch 1,8,32,64 --profile 16 --prefill 128,512,2048
    run e2e-30b-a3b 50 env GLYD_GPU_LIB=$LIB $PY e2e.py Qwen/Qwen3-30B-A3B --format auto --fused --merge --baseline --tokens 64 --batch 1,8 --prompts --profile 16 --prefill 128,512,2048 ;;
  *A10*)
    run e2e-olmoe 40 env GLYD_GPU_LIB=$LIB $PY e2e.py allenai/OLMoE-1B-7B-0924 --format auto --fused --merge --baseline --tokens 64 --batch 1,8 --prompts --profile 16 --prefill 128,512,2048 ;;
esac
touch $R/DONE
