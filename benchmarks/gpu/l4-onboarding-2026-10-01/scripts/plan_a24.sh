# CP2: Glyd as a user runs it, on the L4 with no CUDA toolkit: the whole GPU, then a 16 GB card's memory (an RTX 4080 SUPER with a desktop), then 8 GB, then no gcc,
# then the whole GPU again for the servers (Open WebUI, eager against compiled) and the small models the constants are calibrated on.
WHEEL="/onb/wheels/glyd-0.26.0rc2-py3-none-manylinux_2_28_x86_64.whl[vllm]"
COMMON="-e PORT=8100 -e GLYD_SPEC=$WHEEL"

mark A1-full-install-doctor-run
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-8B -e STAGES="install freeze doctor run"
mark A1b-full-run-no-think
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-8B -e RUN_FLAGS="--no-think" -e PROMPT="Say hello in five words." -e STAGES="run"

mark B1-16gb-run-serve
hog 14827
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-8B -e PYTHONPATH=/geforce -e STAGES="doctor run serve http api stop"
mark B1b-16gb-run-default-allocator
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-8B -e PYTHONPATH=/geforce -e PYTORCH_CUDA_ALLOC_CONF= -e STAGES="run"

mark B2-16gb-bigprompt
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-8B -e PYTHONPATH=/geforce -e STAGES="bigprompt"

mark C1-8gb-refusal
hog 7800
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-8B -e STAGES="doctor runfail"
mark C2-8gb-4b
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-4B -e STAGES="runfail"
mark C2b-8gb-4b-forced
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-4B -e RUN_EXTRA="-- --max-model-len 3072" -e STAGES="run"
mark C3-8gb-1.7b
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-1.7B -e STAGES="run"
hog stop

mark D1-no-gcc
dock $COMMON -e MODEL=Qwen/Qwen3-0.6B -e STAGES="doctor runfail"
mark D2-no-gcc-zig
dock $COMMON -e MODEL=Qwen/Qwen3-0.6B -e STAGES="zig"

mark A2-full-serve-auto-openwebui
# Open WebUI 0.11.4 on the host (docker, host network), as the README gives it; it finds the server when that is up
docker rm -f owui >/dev/null 2>&1
docker run -d --name owui --network=host -e PORT=3000 -e HOST=127.0.0.1 -e OPENAI_API_BASE_URL=http://127.0.0.1:8100/v1 -e OPENAI_API_KEY=none -e WEBUI_AUTH=False \
  -e ENABLE_PERSISTENT_CONFIG=False -e ENABLE_OLLAMA_API=False ghcr.io/open-webui/open-webui:v0.11.4 > /dev/null
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-8B -e STAGES="serve http api webui bench stop" -e BENCH_N="1 4 8"
docker logs owui 2>&1 | grep -iE "error|traceback|auto tool" | head -5
docker rm -f owui >/dev/null 2>&1

mark A3-full-serve-eager
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-8B -e STAGES="serve bench stop" -e SERVE_FLAGS="-- --enforce-eager" -e BENCH_N="1 4 8"

mark A4-full-calibration-models
for M in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B; do
  dock $COMMON -e GCC=1 -e MODEL=$M -e STAGES="run"
done
