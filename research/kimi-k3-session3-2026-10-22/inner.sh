#!/bin/bash
# In the CUDA 13 container on 8 GPUs (GLY-330): main after #219 and #220. Kimi-K3: the load with the experts packed straight into pinned memory (#220),
# decode with the host's crew of 64 pinned threads (#219) and the early guess of each layer's experts (#221), the cache's report every 64 tokens
# (GLYD_KIMI_STATS, with each window's time), a 512-token story for the speed,
# tools, and a 2k-token pass key in tiles of 512 (tiled latent attention, GLY-357). DeepSeek-V4-Flash: its own tools example and our tool prompts
# after the fix of prompt chunks past 64 rows (GLY-354).
W=$HOME/w; O=$W/out; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc/glyd-engine; chmod +x $E
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat && export LD_LIBRARY_PATH=$HOME/compat:${LD_LIBRARY_PATH:-}
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl procps numactl) > $O/apt.txt 2>&1
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total,driver_version --format=csv > $O/machine.txt; nproc > $O/cpus.txt; grep -m1 'model name' /proc/cpuinfo >> $O/cpus.txt; grep -o 'avx512f\|avx2' /proc/cpuinfo | sort -u >> $O/cpus.txt
numactl -H > $O/numa.txt 2>&1
for i in $(seq 1 12); do  # (802: the fabric still coming up; 100: the GPU not yet in the container)
  $E kimirandom $W/toy --layers 8 > /dev/null 2>&1; $E kimicheck $W/toy --tokens 4 > $O/probe.txt 2>&1
  grep -qiE "CUDA error (802|100)" $O/probe.txt || break
  sleep 20
done
if grep -qiE "CUDA error|insufficient|no device" $O/probe.txt; then echo FAIL-CUDA > $O/FAIL; touch $O/DONE; exit 1; fi
# (the host's FP4 experts: threads, pinned, and NUMA local against remote memory, while the downloads run; the 7J13 did 0.48 ms a pair at 3 pairs on 64 threads, the PC 0.29)
$E fp4bench --pairs 1,3 --threads 16,32,64,120 --pin > $O/fp4bench.txt 2>&1
numactl --cpunodebind=0 --membind=0 $E fp4bench --pairs 1,3 --threads 16,32,60 --pin > $O/fp4bench-local.txt 2>&1
numactl --cpunodebind=0 --membind=1 $E fp4bench --pairs 1,3 --threads 16,32,60 --pin > $O/fp4bench-remote.txt 2>&1
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
# (DeepSeek first, while Kimi-K3's 1.45 TB downloads)
DR=deepseek-ai/DeepSeek-V4-Flash
if wait_for $DR; then serve_check ds "ds_canonical tools_think tools toolsreq long" $DR; else echo "no DeepSeek download" > $O/ds-FAIL; fi
KR=moonshotai/Kimi-K3
if wait_for $KR; then
  vmstat -t 10 > $O/vmstat-kimi.txt 2>&1 &
  VM=$!
  # (the early guess's copies a layer, 0, 1, 2 in turn for each 64 tokens: the story's windows compare them)
  # (GLYD_MLA_TILE=512: the pass key's 2,000 tokens in four tiles of the latent attention; a prompt past one tile's 20,000 would take an hour at the cache's prompt speed)
  export GLYD_KIMI_STATS=1 GLYD_KIMI_EARLY=0,1,2 GLYD_MLA_TILE=512
  serve_check kimi "chat long tools toolsreq passkey" $KR --context 32768
  unset GLYD_KIMI_STATS GLYD_KIMI_EARLY GLYD_MLA_TILE
  kill $VM
else echo "no Kimi download" > $O/kimi-FAIL; fi
touch $O/DONE
