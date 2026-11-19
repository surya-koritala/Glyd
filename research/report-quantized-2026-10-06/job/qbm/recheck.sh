#!/usr/bin/env bash
# qbm: after job.sh ends, hold DONE back and run Qwen3.8-27B-NVFP4's stock eager and Glyd exact eager again (its first stock eager run met
# FlashInfer's first JIT build of the NVFP4 GEMM failing), from the checkpoint kept on disk; then DONE.
R=$HOME/results; W=$HOME/w
until [ -f "$R/DONE" ]; do sleep 1; done; rm -f "$R/DONE"
echo "$(date -u +%T) recheck: nvidia/Qwen3.8-27B-NVFP4 stock-eager and glyd-exact again" >> "$R/steps.txt"
PY=$W/venv/bin/python; MM=$("$PY" -c "import torch; print(torch.version.cuda)")
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp PYTHONDONTWRITEBYTECODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH LIBRARY_PATH=$CU/lib64:$CU/lib PYTHONPATH=$W/src/glyd-gpu-py GLYD_GPU_LIB=$(ls $W/lib/libglyd_gpu_cuda*.so) VLLM_CACHE_ROOT=$W/vllmcache GLYD_CACHE=$W/glydcache
M=$(ls -d $HOME/keep/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/*/)
timeout -k 30 1500 "$PY" "$W/src/gpu/vllm/quant_parts.py" "$M" --out "$R/nvidia_Qwen3.8-27B-NVFP4-recheck" --steps stock-eager,glyd-exact --max-num-seqs 64 > "$R/nvidia_Qwen3.8-27B-NVFP4-recheck.txt" 2>&1
echo "$(date -u +%T) recheck: exit $? | $(grep -E '^(EXACT|stock|glyd)' "$R/nvidia_Qwen3.8-27B-NVFP4-recheck.txt" | cut -c1-400 | tr '\n' '|')" >> "$R/steps.txt"
touch "$R/DONE"
