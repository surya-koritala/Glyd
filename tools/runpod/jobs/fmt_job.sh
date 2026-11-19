#!/usr/bin/env bash
# Lossless room in 2026 models' released formats: glyd-gpu 0.28.0 from PyPI (its packer, bit-checked), glyd 0.28.0's CLI from the release, fmt_measure.py.
set -u
mkdir -p ~/results; exec > >(tee -a ~/results/job.log) 2>&1
echo "$(date -u +%T) start"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > ~/results/gpu.txt; nproc >> ~/results/gpu.txt
python3 -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())" | tee -a ~/results/gpu.txt
python3 -m venv --system-site-packages ~/v && PY=~/v/bin/python && $PY -m pip install -q --no-deps "glyd-gpu==0.28.0" && $PY -m pip install -q numpy zstandard || { echo "pip failed"; touch ~/results/DONE; exit 1; }
$PY -c "import torch; from glyd_gpu import kernels as g; w=torch.randn(256,512,dtype=torch.bfloat16,device='cuda'); q=g.pack_mma(w); assert torch.equal(g.mma_unpack(q).view(torch.int16), w.view(torch.int16)); print('glyd_gpu ok', q.nbytes())" | tee -a ~/results/gpu.txt
cd ~ && B=glyd-v0.28.0-linux-x86_64.tar.gz && curl -fsSL -O https://github.com/surya-koritala/Glyd/releases/download/v0.28.0/$B && curl -fsSL -O https://github.com/surya-koritala/Glyd/releases/download/v0.28.0/$B.sha256 \
  && sha256sum -c $B.sha256 && mkdir -p glydcli && tar -xzf $B -C glydcli && export PATH="$(dirname "$(find ~/glydcli -name glyd -type f | head -1)"):$PATH" && glyd --version | tee -a ~/results/gpu.txt || echo "glyd CLI not available: coders will be zstd only"
$PY ~/fmt_measure.py run ~/results \
  Qwen/Qwen3.8-27B Qwen/Qwen3.8-27B-FP8 google/gemma-4-26B-A4B-it \
  zai-org/GLM-5.3-Flash-BF16 zai-org/GLM-5.3-Flash MiniMaxAI/MiniMax-M3 MiniMaxAI/MiniMax-M3-MXFP8 \
  nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
  deepseek-ai/DeepSeek-V4.1-Flash moonshotai/Kimi-K3 mistralai/Mistral-Small-4-119B-2603 XiaomiMiMo/MiMo-V2.6-Pro-RL
echo "$(date -u +%T) done"; touch ~/results/DONE
