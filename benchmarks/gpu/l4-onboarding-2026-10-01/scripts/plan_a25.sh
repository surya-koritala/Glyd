# CP2 again, with what the first pass changed: eager always, PyTorch's default allocator, the memory constants as measured with the fixed plugin,
# the memory check counting a context given after --, and the installer's own ziglang where there is no gcc.
WHEEL="/onb/wheels/glyd-0.26.0rc2-py3-none-manylinux_2_28_x86_64.whl[vllm]"
COMMON="-e PORT=8100 -e GLYD_SPEC=$WHEEL -e LOCAL_INSTALL=1"

mark A1-full-install-doctor-run
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-8B -e STAGES="install freeze doctor run"

mark B1-16gb-run-serve
hog 14827
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-8B -e PYTHONPATH=/geforce -e STAGES="doctor run serve http api stop"
mark B2-16gb-bigprompt
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-8B -e PYTHONPATH=/geforce -e STAGES="bigprompt"

mark C1-8gb-refusal
hog 7800
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-8B -e STAGES="doctor runfail"
mark C2-8gb-4b-forced-context
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-4B -e RUN_EXTRA="-- --max-model-len 2048" -e STAGES="run"
mark C3-8gb-1.7b
dock $COMMON -e GCC=1 -e MODEL=Qwen/Qwen3-1.7B -e STAGES="run"
hog stop

mark D1-no-gcc-install-doctor-run
dock $COMMON -e MODEL=Qwen/Qwen3-0.6B -e STAGES="install freeze doctor run"
