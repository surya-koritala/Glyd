#!/usr/bin/env bash
# One model card's task benchmarks on one pod (rp_run.sh contract: results in ~/results, ~/results/DONE at the end). Uploaded beside it: src.tar (git archive, prefix src/)
# and eval.env (MODEL, REV, LEVEL, SHARD i/n, LIMIT 0 = all; or PACKED_S3=s3://BUCKET/PREFIX/, a packed glyd-v7 directory in the private bucket, with s3.env's short-lived
# keys uploaded beside it: served as it is, e.g. Nemotron 3 Super at Hummingbird, which has no Hub repo yet). Builds the engine, starts `glyd-engine serve ~/v/m:LEVEL` (what `glyd serve` runs) on localhost and runs Inspect AI's
# gsm8k and mmlu_pro against it (eval_run.py). Writes evals.json, logs/ (Inspect's), serve.log, steps.txt, gpu.txt (rp_run.sh).
set -u
R=~/results; mkdir -p $R ~/v; cd ~/v; tar -xf ~/src.tar
. ~/eval.env
P=$R/steps.txt; : > $P
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; echo "$n: rc=$? $(( $(date +%s) - t )) s" >> $P; }
trap 'kill $SERVER 2> /dev/null; rm -f ~/s3.env; touch $R/DONE' EXIT
export PATH=/usr/local/cuda/bin:$PATH HF_XET_HIGH_PERFORMANCE=1
step rustup bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'
. ~/.cargo/env
# (--system-site-packages: PEP 668 keeps pip out of the system python; the pins are Inspect's and its task package's)
step venv bash -c 'python3 -m venv --system-site-packages ~/v/venv && ~/v/venv/bin/pip install -q "huggingface_hub>=1.5,<2" hf_xet "inspect-ai==0.3.278" "inspect-evals==0.24.0" openai '"${PACKED_S3:+awscli}"
if [ -n "${PACKED_S3:-}" ]; then   # (the keys in this subshell alone, never printed; 64 requests at once)
  ( A=~/v/venv/bin/aws; $A configure set default.s3.max_concurrent_requests 64; $A configure set default.s3.multipart_chunksize 64MB
    ( set +x; set -a; . ~/s3.env; set +a; $A s3 cp --recursive --only-show-errors "$PACKED_S3" ~/v/m ) > $R/download.log 2>&1; echo "download: rc=$? (from the bucket)" >> $P; rm -f ~/s3.env ) &
else
( ~/v/venv/bin/hf download "$MODEL" --revision "$REV" --local-dir ~/v/m > $R/download.log 2>&1; echo "download: rc=$?" >> $P ) &
fi
if [ "$LEVEL" = bf16 ]; then   # (the baseline: the source model in bf16 on stock vLLM, the same harness; the engine's lossless path runs Qwen3 only so far)
  step vllm bash -c 'python3 -m venv ~/v/vllm && ~/v/vllm/bin/pip install -q vllm ninja && ~/v/vllm/bin/vllm --version'
  wait
  export PATH=~/v/vllm/bin:$PATH   # (flashinfer builds its sampler with the venv's ninja on the first request)
  ~/v/vllm/bin/vllm serve ~/v/m --port 8000 --max-model-len 6144 --max-num-seqs 8 > $R/serve.log 2>&1 & SERVER=$!
else
step lib bash -c 'cd ~/v/src && bash gpu/build_lib.sh ~/v/lib'
export GLYD_GPU_LIB=$(ls ~/v/lib/libglyd_gpu_cuda*.so | head -1)
step build bash -c 'cd ~/v/src && cargo build --release -p glyd-engine'
wait
M=~/v/m:$LEVEL; [ -n "${PACKED_S3:-}" ] && M=~/v/m   # (a packed directory names its own level)
~/v/src/target/release/glyd-engine serve $M --port 8000 --context 6144 --users ${USERS:-8} > $R/serve.log 2>&1 & SERVER=$!
fi
for _ in $(seq 1 240); do curl -sf localhost:8000/health > /dev/null && break; kill -0 $SERVER 2> /dev/null || break; sleep 5; done
curl -sf localhost:8000/health > /dev/null || { echo "serve: not healthy" >> $P; exit 1; }
NAME=$(curl -s localhost:8000/v1/models | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
export GLYD_API_KEY=none   # (the server has no key; the client wants one set)
TASKS=${TASKS:-} ~/v/venv/bin/python ~/v/src/tools/cards/eval_run.py http://localhost:8000/v1 "$NAME" "$SHARD" "$LIMIT" $R > $R/eval.log 2>&1; echo "eval: rc=$?" >> $P
nvidia-smi > $R/gpu-final.txt
