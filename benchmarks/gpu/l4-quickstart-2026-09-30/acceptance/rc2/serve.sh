source /work/home/glyd-env/bin/activate
export PYTHONPATH=/work/geforce
echo $$ > /work/logs/server.pid
exec env vllm serve Qwen/Qwen3-8B --quantization glyd --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.622 --host 127.0.0.1
