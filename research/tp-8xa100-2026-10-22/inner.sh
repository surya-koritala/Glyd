#!/bin/bash
# In the CUDA 13 container on 8 GPUs: branch gly-tp. First the gathers across real GPUs (tpcheck on a random Kimi-K3, two GPUs, its matrices resident and in
# the expert cache), then Kimi-K3 with every layer across all 8 GPUs (GLYD_TP=1), then split by layers (main's way, with #223 and #224's 12-bit layout and load):
# the load, a 512-token story for the speed (GLYD_KIMI_STATS: each 64 tokens' time and the cache's counts), chat, tools, a 2k-token pass key.
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
# the gathers over the GPUs' links: one GPU against two, every matrix of the random model sliced; then its experts in the cache, copies alone and the host alone
GLYD_TP_MIN_MB=0 timeout 600 $E tpcheck $W/toy --tp 0,1 > $O/tpcheck-resident.txt 2>&1
GLYD_TP_MIN_MB=0 timeout 600 $E tpcheck $W/toy --tp 0,1 --chunk 8 --tokens 40 > $O/tpcheck-chunks.txt 2>&1
for m in copy cpu; do GLYD_TP_MIN_MB=0 GLYD_KIMI_PLACES=4 GLYD_KIMI_HOST=$m timeout 600 $E tpcheck $W/toy --tp 0,1 --chunk 8 --tokens 40 > $O/tpcheck-cache-$m.txt 2>&1; done
# the same for a random DeepSeek-V4, every matrix in the FP8 format
$E dsv4random $W/dstoy > /dev/null 2>&1
GLYD_TP_MIN_MB=0 GLYD_V4_F8_MIN=0 timeout 600 $E tpcheck $W/dstoy --tp 0,1 > $O/tpcheck-ds.txt 2>&1
GLYD_TP_MIN_MB=0 GLYD_V4_F8_MIN=0 timeout 600 $E tpcheck $W/dstoy --tp 0,1 --chunk 16 --tokens 40 > $O/tpcheck-ds-chunks.txt 2>&1
wait_for() { for i in $(seq 1 540); do grep -q "^$1 exit" $O/download.txt 2>/dev/null && return 0; sleep 20; done; return 1; }
serve_check() {
  local tag=$1 reqs=$2; shift 2
  local t0=$(date +%s)
  "$E" serve "$@" --port 8100 > $O/$tag-serve.log 2>&1 &
  local sp=$!
  for i in $(seq 1 1080); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8100/health)" = 200 ] && break; kill -0 $sp 2>/dev/null || break; sleep 5; done
  echo "$(date +%T) $tag up (or gone) after $(( $(date +%s) - t0 )) s" >> $O/steps.txt
  for r in $reqs; do local t=$(date +%s); curl -s -m 2400 localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d @$HOME/dc/req/$r.json > $O/$tag-$r.json; echo "$(date +%T) $tag $r done in $(( $(date +%s) - t )) s" >> $O/steps.txt; done
  kill $sp; wait $sp 2>/dev/null
}
# DeepSeek-V4-Flash (while Kimi-K3 downloads): split by layers on 2 GPUs as today, then every layer across 2, 4 and 8
DR=deepseek-ai/DeepSeek-V4-Flash
if wait_for $DR; then
  NL=$(grep -ho '"num_hidden_layers": *[0-9]*' $HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/*/config.json | grep -o '[0-9]*$' | head -1)
  serve_check ds-split2 "long tools ds_canonical" $DR --split $(( NL / 2 )) --devices 0,1
  for g in 0,1 0,1,2,3 0,1,2,3,4,5,6,7; do (export GLYD_TP=1 GLYD_TP_GPUS=$g; serve_check ds-tp$(echo $g | tr -cd , | wc -c | awk '{print $1+1}') "long tools ds_canonical" $DR); done
else echo "no DeepSeek download" > $O/ds-FAIL; fi
KR=moonshotai/Kimi-K3
if wait_for $KR; then
  vmstat -t 10 > $O/vmstat-kimi.txt 2>&1 &
  VM=$!
  export GLYD_KIMI_STATS=1 GLYD_MLA_TILE=512
  (export GLYD_TP=1; serve_check tp "chat long tools toolsreq passkey" $KR --context 32768)
  serve_check split "chat long tools passkey" $KR --context 32768
  unset GLYD_KIMI_STATS GLYD_MLA_TILE
  kill $VM
else echo "no Kimi download" > $O/kimi-FAIL; fi
touch $O/DONE
