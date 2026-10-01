source /work/home/glyd-env/bin/activate
export PYTHONPATH=/work/geforce
echo $$ > /work/logs/server.pid
exec env VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3-8B --quantization glyd --enforce-eager \
  --max-model-len 8192 --gpu-memory-utilization 0.622 --host 127.0.0.1 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
