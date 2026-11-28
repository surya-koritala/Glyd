# GLY-25 / GLY-81 fresh-pod check (v030 at 47d88184): `run Qwen/Qwen3.8-27B:linnet` downloads glyd/Qwen3.8-27B-linnet (no bf16), the second start is from the cache;
# `serve ...:linnet` answers /v1/models and one chat; a model with no glyd repo (Qwen3.5-0.8B) falls back to bf16 + pack.
set -u
R=~/results; mkdir -p $R ~/v; cd ~/v; tar -xf ~/src.tar
P=$R/steps.txt; : > $P
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; echo "$n: rc=$? $(( $(date +%s) - t )) s" >> $P; }
export PATH=/usr/local/cuda/bin:$PATH HF_XET_HIGH_PERFORMANCE=1
step rustup bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'
. ~/.cargo/env
step lib bash -c 'cd ~/v/src && bash gpu/build_lib.sh ~/v/lib'
export GLYD_GPU_LIB=$(ls ~/v/lib/libglyd_gpu_cuda*.so | head -1)
step build bash -c 'cd ~/v/src && cargo build --release -p glyd-engine'
E=~/v/src/target/release/glyd-engine
Q="Name three primes above 100."
step run27-linnet-first $E run Qwen/Qwen3.8-27B:linnet --no-think --temp 0 -p "$Q"
step run27-linnet-second $E run Qwen/Qwen3.8-27B:linnet --no-think --temp 0 -p "$Q"
du -sh ~/.cache/huggingface/hub/models--* > $R/hf-cache.txt 2>&1
( $E serve Qwen/Qwen3.8-27B:linnet --port 8000 > $R/serve.log 2>&1 & echo $! > ~/v/serve.pid )
t=$(date +%s); until curl -sf localhost:8000/health > /dev/null || [ $(( $(date +%s) - t )) -gt 600 ]; do sleep 3; done; echo "serve-up: $(( $(date +%s) - t )) s" >> $P
curl -s localhost:8000/v1/models > $R/serve-models.json
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"Name three primes above 100."}],"max_tokens":200,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' > $R/serve-chat.json; echo "serve-chat: rc=$?" >> $P
kill $(cat ~/v/serve.pid); sleep 5
step run08-linnet-fallback $E run Qwen/Qwen3.5-0.8B:linnet --no-think --temp 0 -p "$Q"
nvidia-smi > $R/gpu-final.txt; touch $R/DONE
