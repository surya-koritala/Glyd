#!/bin/bash
# In the CUDA 13 container on 8 GPUs, branch tp-fewer-exchanges: the gathers' low-latency kind (tp_gather_ll) against the regular one (GLYD_TP_LL=0).
# 1. tpcheck on real GPUs (random Kimi-K3 and DeepSeek-V4, every matrix sliced; 2 and 8 GPUs); 2. tpgather, each kind, 2/4/8 GPUs; 3. DeepSeek-V4-Flash
# tpbench across 2/4/8, each kind, and split on 2, and `serve` with no switch (across all 8 by default now); 4. GLM-5.3-Flash across 8 (GLYD_TP=1; its
# split is in ../kimi-k3-h100x8-2026-10-21); 5. Kimi-K3 `serve` with no switch (across 8 by default): the 512-token story (GLYD_KIMI_STATS), chat, tools,
# the pass key.
W=$HOME/w; O=$W/out; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc/glyd-engine; chmod +x $E
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat && export LD_LIBRARY_PATH=$HOME/compat:${LD_LIBRARY_PATH:-}
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl procps numactl) > $O/apt.txt 2>&1
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total,driver_version --format=csv > $O/machine.txt; nproc > $O/cpus.txt; grep -m1 'model name' /proc/cpuinfo >> $O/cpus.txt
for i in $(seq 1 12); do  # (802: the fabric still coming up; 100: the GPU not yet in the container)
  $E kimirandom $W/toy --layers 8 > /dev/null 2>&1; $E kimicheck $W/toy --tokens 4 > $O/probe.txt 2>&1
  grep -qiE "CUDA error (802|100)" $O/probe.txt || break
  sleep 20
done
if grep -qiE "CUDA error|insufficient|no device" $O/probe.txt; then echo FAIL-CUDA > $O/FAIL; touch $O/DONE; exit 1; fi
echo "$(date +%T) checks" >> $O/steps.txt
# 1. the same logits to the bit as one GPU, over the real links
GLYD_TP_MIN_MB=0 timeout 600 $E tpcheck $W/toy --tp 0,1 > $O/tpcheck-resident.txt 2>&1
GLYD_TP_MIN_MB=0 timeout 600 $E tpcheck $W/toy --tp 0,1 --chunk 8 --tokens 40 > $O/tpcheck-chunks.txt 2>&1
GLYD_TP_MIN_MB=0 timeout 900 $E tpcheck $W/toy --tp 0,1,2,3,4,5,6,7 --chunk 8 --tokens 40 > $O/tpcheck-chunks-8.txt 2>&1
for m in copy cpu; do GLYD_TP_MIN_MB=0 GLYD_KIMI_PLACES=4 GLYD_KIMI_HOST=$m timeout 600 $E tpcheck $W/toy --tp 0,1 --chunk 8 --tokens 40 > $O/tpcheck-cache-$m.txt 2>&1; done
$E dsv4random $W/dstoy > /dev/null 2>&1
GLYD_TP_MIN_MB=0 GLYD_V4_F8_MIN=0 timeout 600 $E tpcheck $W/dstoy --tp 0,1 > $O/tpcheck-ds.txt 2>&1
GLYD_TP_MIN_MB=0 GLYD_V4_F8_MIN=0 timeout 900 $E tpcheck $W/dstoy --tp 0,1,2,3,4,5,6,7 --chunk 16 --tokens 40 > $O/tpcheck-ds-chunks-8.txt 2>&1
# 2. a gather's own time, each kind
for g in 0,1 0,1,2,3 0,1,2,3,4,5,6,7; do
  k=$(echo $g | tr -cd , | wc -c)
  timeout 300 $E tpgather --tp $g > $O/tpgather-ll-$k.txt 2>&1
  GLYD_TP_LL=0 timeout 300 $E tpgather --tp $g > $O/tpgather-reg-$k.txt 2>&1
done
# 2b. the host's FP4 experts (Kimi-K3's misses the host computes): a call's time by pairs and threads, spread and pinned, and each socket alone
lscpu > $O/lscpu.txt; numactl -H > $O/numa.txt 2>&1
timeout 900 $E fp4bench --pairs 1,2,4,16 --threads 16,32,64,120,0 --reps 10 > $O/fp4bench.txt 2>&1
timeout 900 $E fp4bench --pairs 1,2,4,16 --threads 32,64,120,0 --reps 10 --pin > $O/fp4bench-pin.txt 2>&1
timeout 600 numactl -N 0 -m 0 $E fp4bench --pairs 1,4,16 --threads 32,60,120 --reps 10 > $O/fp4bench-node0.txt 2>&1
wait_for() { for i in $(seq 1 540); do grep -q "^$1 exit" $O/download.txt 2>/dev/null && return 0; sleep 20; done; return 1; }
snap() { ls -d $HOME/.cache/huggingface/hub/models--$(echo $1 | sed 's#/#--#')/snapshots/* | head -1; }
serve_check() {
  local tag=$1 reqs=$2; shift 2
  local t0=$(date +%s)
  $E serve "$@" --port 8100 > $O/$tag-serve.log 2>&1 &
  local sp=$!
  for i in $(seq 1 1080); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8100/health)" = 200 ] && break; kill -0 $sp 2>/dev/null || break; sleep 5; done
  echo "$(date +%T) $tag up (or gone) after $(( $(date +%s) - t0 )) s" >> $O/steps.txt
  for r in $reqs; do
    local t=$(date +%s)
    curl -s -m 2400 localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d @$HOME/dc/req/$r.json > $O/$tag-$r.json
    echo "$(date +%T) $tag $r done in $(( $(date +%s) - t )) s" >> $O/steps.txt
  done
  # (the last one, Kimi-K3, not waited for: a process with 1.35 TB pinned takes ~10 min to end, and the box is terminated)
  kill $sp; [ "$tag" = kimi ] || wait $sp 2>/dev/null
}
# 3. DeepSeek-V4-Flash
DR=deepseek-ai/DeepSeek-V4-Flash
if wait_for $DR; then
  echo "$(date +%T) deepseek" >> $O/steps.txt
  D=$(snap $DR); NL=$(grep -o '"num_hidden_layers": *[0-9]*' $D/config.json | grep -o '[0-9]*$')
  timeout 900 $E tpbench $D --tp 0,1 --split $(( NL / 2 )) --devices 0,1 --steps 256 > $O/tpbench-ds-2.txt 2>&1
  GLYD_TP_LL=0 timeout 900 $E tpbench $D --tp 0,1 --steps 256 > $O/tpbench-ds-2-reg.txt 2>&1
  for g in 0,1,2,3 0,1,2,3,4,5,6,7; do
    k=$(echo $g | tr -cd , | wc -c | awk '{print $1+1}')
    timeout 900 $E tpbench $D --tp $g --steps 256 > $O/tpbench-ds-$k.txt 2>&1
    GLYD_TP_LL=0 timeout 900 $E tpbench $D --tp $g --steps 256 > $O/tpbench-ds-$k-reg.txt 2>&1
  done
  serve_check ds-default "long tools ds_canonical" $DR
else echo "no DeepSeek download" > $O/ds-FAIL; fi
# 4. GLM-5.3-Flash across the 8
GR=zai-org/GLM-5.3-Flash
if wait_for $GR; then
  echo "$(date +%T) glm" >> $O/steps.txt
  (export GLYD_TP=1; serve_check glm-tp8 "long tools" $GR)
else echo "no GLM download" > $O/glm-FAIL; fi
# 5. Kimi-K3, across the 8 by default
KR=moonshotai/Kimi-K3
if wait_for $KR; then
  echo "$(date +%T) kimi" >> $O/steps.txt
  vmstat -t 10 > $O/vmstat-kimi.txt 2>&1 &
  VM=$!
  export GLYD_KIMI_STATS=1 GLYD_MLA_TILE=512
  serve_check kimi "chat long tools passkey" $KR --context 32768
  unset GLYD_KIMI_STATS GLYD_MLA_TILE
  kill $VM
else echo "no Kimi download" > $O/kimi-FAIL; fi
touch $O/DONE
