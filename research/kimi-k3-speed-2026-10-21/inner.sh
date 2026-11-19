#!/bin/bash
# In the CUDA 13 container on 8 GPUs (GLY-330): Kimi-K3's decode with the missed experts computed on the host (PR #219, the expert cache's auto mode),
# the cache's report every 64 tokens (GLYD_KIMI_STATS), a 512-token story for the speed; the disk's reads during the load (vmstat: the load's time).
# Then DeepSeek-V4-Flash's tool calls on its own encoding's example (GLY-354).
W=$HOME/w; O=$W/out; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc/glyd-engine; chmod +x $E
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat && export LD_LIBRARY_PATH=$HOME/compat:${LD_LIBRARY_PATH:-}
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl procps) > $O/apt.txt 2>&1
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total,driver_version --format=csv > $O/machine.txt; nproc > $O/cpus.txt; grep -m1 'model name' /proc/cpuinfo >> $O/cpus.txt; grep -o 'avx512f\|avx2' /proc/cpuinfo | sort -u >> $O/cpus.txt
for i in $(seq 1 12); do  # (802: the fabric still coming up; 100: the GPU not yet in the container)
  $E kimirandom $W/toy --layers 8 > /dev/null 2>&1; $E kimicheck $W/toy --tokens 4 > $O/probe.txt 2>&1
  grep -qiE "CUDA error (802|100)" $O/probe.txt || break
  sleep 20
done
if grep -qiE "CUDA error|insufficient|no device" $O/probe.txt; then echo FAIL-CUDA > $O/FAIL; touch $O/DONE; exit 1; fi
wait_for() { for i in $(seq 1 540); do grep -q "^$1 exit" $O/download.txt 2>/dev/null && return 0; sleep 20; done; return 1; }
serve_check() {
  local tag=$1 reqs=$2; shift 2
  "$E" serve "$@" --port 8100 > $O/$tag-serve.log 2>&1 &
  local sp=$!
  for i in $(seq 1 1080); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8100/health)" = 200 ] && break; kill -0 $sp 2>/dev/null || break; sleep 5; done
  echo "$(date +%T) $tag up (or gone)" >> $O/steps.txt
  for r in $reqs; do curl -s -m 2400 localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d @$HOME/dc/req/$r.json > $O/$tag-$r.json; echo "$(date +%T) $tag $r done" >> $O/steps.txt; done
  kill $sp; wait $sp 2>/dev/null
}
KR=moonshotai/Kimi-K3
if wait_for $KR; then
  vmstat -t 10 > $O/vmstat-kimi.txt 2>&1 &
  VM=$!
  export GLYD_KIMI_STATS=1
  serve_check kimi "chat long tools toolsreq" $KR
  unset GLYD_KIMI_STATS
  kill $VM
else echo "no Kimi download" > $O/kimi-FAIL; fi
DR=deepseek-ai/DeepSeek-V4-Flash
if wait_for $DR; then serve_check ds "ds_canonical tools_think" $DR; else echo "no DeepSeek download" > $O/ds-FAIL; fi
touch $O/DONE
