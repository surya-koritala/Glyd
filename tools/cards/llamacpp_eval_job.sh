#!/usr/bin/env bash
# llama.cpp against Glyd on the same card (Linear GLY-98; rp_run.sh contract: results in ~/results, ~/results/DONE at the end): Nemotron 3 Super as Unsloth's GGUF
# (unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF at 036038fb, UD-Q2_K_XL: 54.7 GB, the smallest of its usual picks; Glyd's hummingbird is 48.7 GB) on llama.cpp built
# here for this GPU, the experts of the first N layers in the computer's memory (--n-cpu-moe N, the usual way to run a MoE larger than the card), N the least that loads:
#   1. speed   llama-server with each N of NS from the most experts on the CPU down, the first that fails to load ends the sweep; for each, the prompt of
#              tools/cards/nh_offload_job.sh's generate (raw completion, 256 tokens, greedy, the end of text ignored) twice, the second's timings kept
#   2. evals   the fastest N, 4 slots of 8,192 tokens: tools/cards/eval_run.py (Inspect AI's gsm8k and mmlu_pro, non-thinking, temperature 0) on SHARD of LIMIT
#              questions a task, the questions Glyd's own run of the same shard had (benchmarks/cards/2026-10-10/nemotron-3-hummingbird-evals)
# Uploaded beside it: src.tar (git archive, prefix src/). Results: speed.txt (one line a N), evals.json, logs/ (Inspect's), server-N.log, steps.txt, gpu.txt.
#   NAME=llamacpp-nh MAX_MIN=120 RP_GPUS=1 RP_DISK=150 UPLOAD="$W/src.tar" bash tools/runpod/rp_class.sh 4090 tools/cards/llamacpp_eval_job.sh
set -u
R=~/results; mkdir -p $R ~/v; cd ~/v; tar -xf ~/src.tar
P=$R/steps.txt; : > $P
REPO=unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF; REV=036038fb30334a2d56a146c6f0d4871ab5edccbb; QUANT=UD-Q2_K_XL
SHARD=${SHARD:-1/2}; LIMIT=${LIMIT:-150}; NS=${NS:-"88 64 56 48 40"}
PROMPT="Explain, step by step, how a hash table handles collisions, then compare open addressing with chaining, and say when each one is the better choice for a program that reads far more than it writes."
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; local rc=$?; echo "$n: rc=$rc $(( $(date +%s) - t )) s" >> $P; return $rc; }
SERVER=; trap 'kill $SERVER 2> /dev/null; touch $R/DONE' EXIT
export PATH=/usr/local/cuda/bin:$PATH HF_XET_HIGH_PERFORMANCE=1
nproc > $R/cpus.txt; free -g > $R/free.txt; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $R/card.txt
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
step venv bash -c 'python3 -m venv --system-site-packages ~/v/venv && for i in 1 2 3; do ~/v/venv/bin/pip install -q --timeout 60 "huggingface_hub>=1.5,<2" hf_xet "inspect-ai==0.3.278" "inspect-evals==0.24.0" openai && exit 0; sleep 30; done; exit 1' || exit 1
( step download ~/v/venv/bin/hf download $REPO --revision $REV --include "$QUANT/*" --local-dir ~/v/gguf ) &
DL=$!
step build bash -c "git clone -q --depth 1 https://github.com/ggml-org/llama.cpp ~/v/llama.cpp && cd ~/v/llama.cpp && git log -1 --format='llama.cpp %h %cd' > $R/llamacpp.txt && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DLLAMA_CURL=OFF > /dev/null && cmake --build build -j $(nproc) --target llama-server" || exit 1
wait $DL || { echo "download: failed" >> $P; exit 1; }
M=$(ls ~/v/gguf/$QUANT/*00001-of-*.gguf 2> /dev/null | head -1); [ -n "$M" ] || M=$(ls ~/v/gguf/$QUANT/*.gguf | head -1)
du -sb ~/v/gguf/$QUANT > $R/size.txt
S=~/v/llama.cpp/build/bin/llama-server
up() { for _ in $(seq 1 120); do curl -sf localhost:8000/health > /dev/null && return 0; kill -0 $SERVER 2> /dev/null || return 1; sleep 5; done; return 1; }
# 1. speed: the least N (most experts on the GPU) that loads, and its decode speed on the prompt
: > $R/speed.txt; BEST=; BEST_TPS=0
for n in $NS; do
    $S -m "$M" -ngl 99 --n-cpu-moe $n -fa on -c 4096 --jinja --port 8000 > $R/server-$n.log 2>&1 & SERVER=$!
    if ! up; then kill $SERVER 2> /dev/null; wait $SERVER 2> /dev/null; echo "n-cpu-moe $n: does not load (server-$n.log)" >> $R/speed.txt; break; fi
    tps=
    for i in 1 2; do
        tps=$(curl -s localhost:8000/completion -H 'Content-Type: application/json' -d "{\"prompt\": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$PROMPT"), \"n_predict\": 256, \"temperature\": 0, \"ignore_eos\": true, \"cache_prompt\": false}" | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print("%.2f %.1f %d" % (t["predicted_per_second"], t["prompt_per_second"], t["predicted_n"]))')
    done
    echo "n-cpu-moe $n: decode, prefill tok/s, tokens: $tps; $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $R/speed.txt
    d=${tps%% *}; awk -v a="${d:-0}" -v b="$BEST_TPS" 'BEGIN{exit !(a > b)}' && { BEST=$n; BEST_TPS=$d; }
    kill $SERVER 2> /dev/null; wait $SERVER 2> /dev/null; SERVER=
done
echo "speed: best n-cpu-moe ${BEST:-none}, $BEST_TPS decode tok/s" >> $P
[ -n "$BEST" ] || exit 1
# 2. evals at the fastest N (4 slots, 8,192 tokens each: a 10-shot GSM8K prompt and 3,072 new tokens fit)
for n in $BEST $((BEST + 8)); do   # (the 4 slots' KV and states are more than the speed runs had: a step back if they do not fit)
    $S -m "$M" -ngl 99 --n-cpu-moe $n -fa on -c 32768 -np 4 --jinja --port 8000 > $R/server-eval.log 2>&1 & SERVER=$!
    up && break; kill $SERVER 2> /dev/null; wait $SERVER 2> /dev/null; SERVER=
done
[ -n "$SERVER" ] || { echo "serve: not healthy at n-cpu-moe $BEST or $((BEST + 8))" >> $P; exit 1; }
echo "evals at n-cpu-moe $n" >> $P
NAME=$(curl -s localhost:8000/v1/models | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
export GLYD_API_KEY=none
~/v/venv/bin/python ~/v/src/tools/cards/eval_run.py http://localhost:8000/v1 "$NAME" "$SHARD" "$LIMIT" $R > $R/eval.log 2>&1; echo "eval: rc=$?" >> $P
nvidia-smi > $R/gpu-final.txt
