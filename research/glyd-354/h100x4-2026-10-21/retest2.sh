#!/bin/bash
# The combined fix (#210, #211, #212): GLM-5.3-Flash's tools requests, then DeepSeek-V4-Flash's tools request with thinking on.
O=$HOME/w/out; exec > $O/retest2.log 2>&1; set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc/glyd-engine-fix
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat3 && export LD_LIBRARY_PATH=$HOME/compat3:${LD_LIBRARY_PATH:-}
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl) > /dev/null 2>&1
one() {
  local tag=$1 model=$2; shift 2
  "$E" serve $model --port 8100 > $O/$tag-serve.log 2>&1 &
  local sp=$!
  for i in $(seq 1 360); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8100/health)" = 200 ] && break; kill -0 $sp 2>/dev/null || break; sleep 5; done
  for r in "$@"; do curl -s -m 900 localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d @$HOME/dc/req/$r.json > $O/$tag-$r.json; done
  kill $sp; wait $sp 2>/dev/null
}
one glm2 zai-org/GLM-5.3-Flash tools toolsreq
one ds3 deepseek-ai/DeepSeek-V4-Flash tools_think
touch $O/RETEST2-DONE
