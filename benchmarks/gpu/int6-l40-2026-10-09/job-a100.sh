# A100: the KL study of the six-bit layout as packed (bf16 scale) beside FP8; meanwhile the library and the engine of the int6 branch;
# then examples/int6 on Ampere, Qwen3.8-27B's speed (Int6, compact, vLLM's FP8 checkpoint) and glyd-engine run --lossy. ~/a/steps.txt.
set -u
cd ~/a; P=~/a/steps.txt; : > $P
step() { local n=$1; shift; local t=$(date +%s); "$@" > ~/a/$n.log 2>&1; echo "$n: rc=$? $(( $(date +%s) - t )) s" >> $P; }
export PATH=/usr/local/cuda-12.8/bin:$PATH HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false
step venv bash -c 'python3 -m venv ~/a/venv && ~/a/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128 && ~/a/venv/bin/pip install -q "transformers==5.18.0" "accelerate>=1.1" "safetensors>=0.8" "huggingface_hub>=1.5,<2" pyarrow flash-linear-attention'
step selftest bash -c 'cd ~/a/k && ~/a/venv/bin/python kld_study.py selftest'
step download bash -c '~/a/venv/bin/hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --local-dir ~/a/bf16 > /dev/null && ~/a/venv/bin/hf download Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a --local-dir ~/a/fp8 > /dev/null'
# the build on the CPU while the study holds the GPU
( curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > ~/a/rustup.log 2>&1; . ~/.cargo/env
  s=$(date +%s); (cd ~/a/src && bash gpu/build_lib.sh ~/a/lib > ~/a/lib.log 2>&1); echo "lib: rc=$? $(( $(date +%s) - s )) s" >> $P
  s=$(date +%s); (cd ~/a/src && cargo build --release -p glyd-engine > ~/a/engine.log 2>&1 && cargo build --release -p glyd-gpu --example int6 >> ~/a/engine.log 2>&1); echo "engine: rc=$? $(( $(date +%s) - s )) s" >> $P
  s=$(date +%s); (curl -LsSf https://getglyd.com/install.sh | sh > ~/a/install.log 2>&1); echo "install: rc=$? $(( $(date +%s) - s )) s" >> $P ) &
cd ~/a/k
step ref ~/a/venv/bin/python kld_study.py ref ~/a/bf16 ~/a/out
step score ~/a/venv/bin/python kld_study.py score ~/a/bf16 ~/a/out bf16 int:6:32:absmax:bf16 int:6:32:absmax frac:3
step fp8 ~/a/venv/bin/python kld_study.py score ~/a/bf16 ~/a/out fp8 --fp8-dir ~/a/fp8
wait
. ~/.cargo/env
LIB=$(ls ~/a/lib/libglyd_gpu_cuda*.so | head -1); E=~/a/src/target/release/glyd-engine
step int6-check env GLYD_GPU_LIB=$LIB ~/a/src/target/release/examples/int6
step bench-i6 env GLYD_GPU_LIB=$LIB $E bench ~/a/bf16 --layout i6 --prompt 1024 --new 256
step bench-a env GLYD_GPU_LIB=$LIB $E bench ~/a/bf16 --layout a --prompt 1024 --new 256
step profile-i6 env GLYD_GPU_LIB=$LIB $E bench ~/a/bf16 --layout i6 --prompt 1024 --new 64 --profile
step run-lossy env GLYD_GPU_LIB=$LIB $E run ~/a/bf16 --lossy -p "In two sentences: what is the capital of Australia, and why was it chosen?"
step vllm-fp8 env VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 REF_KW='{"language_model_only": true, "max_num_seqs": 16}' ~/.local/share/uv/tools/glyd/bin/python ~/a/ref_speed.py ~/a/fp8 ~/a/vllm-fp8.json bf16 1024 256 5
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > ~/a/gpu.txt
echo done >> $P
