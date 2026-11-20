#!/bin/bash
# In the CUDA 13 container on 8x A100 (branch a100-run: main after #239, with #230, #241, #242): 1. tpcheck over the real links on 2 GPUs (random Kimi-K3,
# every matrix sliced; the low-latency gathers and the regular ones; the expert cache's device path in copy mode); 2. tpgather, each kind, 2/4/8 GPUs;
# 3. Kimi-K3 `serve` across the 8 (the default) on the device path: chat, the 2,472-token pass key (its prompt's routing traced: GLYD_KIMI_TRACE), the
# 512-token story (the decode's speed; its found share said at the next prompt: GLYD_KIMI_STATS), tools, chat again.
W=$HOME/w; O=$W/out; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc/glyd-engine; chmod +x $E
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat && export LD_LIBRARY_PATH=$HOME/compat:${LD_LIBRARY_PATH:-}
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl procps numactl) > $O/apt.txt 2>&1
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total,driver_version --format=csv > $O/machine.txt; nproc > $O/cpus.txt
for i in $(seq 1 12); do  # (802: the fabric still coming up; 100: the GPU not yet in the container)
  $E kimirandom $W/toy --layers 8 > /dev/null 2>&1; $E kimicheck $W/toy --tokens 4 > $O/probe.txt 2>&1
  grep -qiE "CUDA error (802|100)" $O/probe.txt || break
  sleep 20
done
if grep -qiE "CUDA error|insufficient|no device" $O/probe.txt; then echo FAIL-CUDA > $O/FAIL; touch $O/DONE; exit 1; fi
echo "$(date +%T) checks" >> $O/steps.txt
# 1. the same logits to the bit as one GPU, over NVLink (the toy's latent of 128 slices on 2 GPUs alone)
GLYD_TP_MIN_MB=0 timeout 600 $E tpcheck $W/toy --tp 0,1 --chunk 8 --tokens 40 > $O/tpcheck-ll.txt 2>&1
GLYD_TP_LL=0 GLYD_TP_MIN_MB=0 timeout 600 $E tpcheck $W/toy --tp 0,1 --chunk 8 --tokens 40 > $O/tpcheck-reg.txt 2>&1
GLYD_TP_MIN_MB=0 GLYD_KIMI_PLACES=4 GLYD_KIMI_HOST=copy GLYD_KIMI_DEVICE=1 GLYD_KIMI_STATS=1 timeout 600 $E tpcheck $W/toy --tp 0,1 --chunk 8 --tokens 40 > $O/tpcheck-device.txt 2>&1
# 2. a gather's own time, each kind
for g in 0,1 0,1,2,3 0,1,2,3,4,5,6,7; do
  k=$(echo $g | tr -cd , | wc -c | awk '{print $1+1}')
  timeout 300 $E tpgather --tp $g > $O/tpgather-ll-$k.txt 2>&1
  GLYD_TP_LL=0 timeout 300 $E tpgather --tp $g > $O/tpgather-reg-$k.txt 2>&1
done
echo "$(date +%T) checks done" >> $O/steps.txt
wait_for() { for i in $(seq 1 540); do grep -q "^$1 exit" $O/download.txt 2>/dev/null && return 0; sleep 20; done; return 1; }
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
  # (not waited for: a process with 1.35 TB pinned takes ~10 min to end, and the box is terminated)
  kill $sp
}
# 3. Kimi-K3 across the 8, the device path
KR=moonshotai/Kimi-K3
if wait_for $KR; then
  echo "$(date +%T) kimi" >> $O/steps.txt
  vmstat -t 10 > $O/vmstat-kimi.txt 2>&1 &
  VM=$!
  export GLYD_KIMI_DEVICE=1 GLYD_KIMI_STATS=1 GLYD_KIMI_TRACE=$O/kimi-trace.txt GLYD_MLA_TILE=512
  serve_check kimi "chat passkey long tools chat2" $KR --context 32768
  kill $VM
else echo "no Kimi download" > $O/kimi-FAIL; fi
touch $O/DONE
