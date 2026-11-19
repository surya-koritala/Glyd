#!/bin/bash
# In the CUDA 13 container on the 8x H100: Kimi-K3 (2.8T, 896 MXFP4 experts a layer) through `glyd serve` (chat, stream, tools, tool_choice required) and `glyd run`,
# lossless, on the dynamic engine: the weights besides the experts over the 8 GPUs, each GPU's room left as places for its layers' experts, every expert in host memory.
# Then DeepSeek-V4-Flash's and GLM-5.3-Flash's tool calls on real weights (DSV4's plain tools with a system prompt: DSML or JSON; GLM's format, PR #213).
# inner2.sh (17:57 EDT): the session again from Kimi-K3's serve, with the attention's shared-memory fix; the first serve's requests failed on a launch (out/).
W=$HOME/w; O=$W/out2; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc/glyd-engine; chmod +x $E
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat && export LD_LIBRARY_PATH=$HOME/compat:${LD_LIBRARY_PATH:-}
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl) > $O/apt.txt 2>&1
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total,driver_version --format=csv > $O/machine.txt
for i in $(seq 1 12); do  # (802: the fabric still coming up; 100: the GPU not yet in the container)
  $E kimirandom $W/toy --layers 4 > /dev/null 2>&1; $E kimicheck $W/toy --tokens 4 > $O/probe.txt 2>&1
  grep -qiE "CUDA error (802|100)" $O/probe.txt || break
  sleep 20
done
if grep -qiE "CUDA error|insufficient|no device" $O/probe.txt; then echo FAIL-CUDA > $O/FAIL; touch $O/DONE; exit 1; fi
step() { local out=${@: -1}; echo "$(date +%T) start $out" >> $O/steps.txt; timeout 5400 "${@:1:$#-1}" > $out 2>&1; echo "$(date +%T) end $out exit $?" >> $O/steps.txt; }
wait_for() { for i in $(seq 1 540); do grep -q "^$1 exit" $W/out/download.txt 2>/dev/null && return 0; sleep 20; done; return 1; }
serve_check() {
  local tag=$1; shift
  "$E" serve "$@" --port 8100 > $O/$tag-serve.log 2>&1 &
  local sp=$!
  for i in $(seq 1 1080); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8100/health)" = 200 ] && break; kill -0 $sp 2>/dev/null || break; sleep 5; done
  echo "$(date +%T) serve up (or gone)" >> $O/steps.txt
  for r in chat stream tools toolsreq tools_sys; do curl -s -m 1800 localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d @$HOME/dc/req/$r.json > $O/$tag-$r.json; echo "$(date +%T) $r done" >> $O/steps.txt; done
  curl -s localhost:8100/v1/glyd/info > $O/$tag-info.json
  kill $sp; wait $sp 2>/dev/null
}
Q="What is the capital of France? Answer in one sentence."
KR=moonshotai/Kimi-K3
if wait_for $KR; then
  serve_check kimi $KR
else echo "no Kimi download" > $O/kimi-FAIL; fi
for m in deepseek-ai/DeepSeek-V4-Flash:ds zai-org/GLM-5.3-Flash:glm; do
  if wait_for ${m%%:*}; then serve_check ${m##*:} ${m%%:*}; else echo "no ${m%%:*} download" > $O/${m##*:}-FAIL; fi
done
# (GLY-356: the engine's own download on this link, its connections added while they help; 7 GB)
step $E hubslice moonshotai/Kimi-K3 $W/k1 --layers 1 --prefix language_model.model. $O/slice-ramp.txt
touch $O/DONE
