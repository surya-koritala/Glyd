#!/bin/bash
# DeepSeek-V4-Flash's tools requests again with #210's build (the grouped o low rank in parts of 64 rows), in the CUDA 13 container.
O=$HOME/w/out; exec > $O/retest.log 2>&1; set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc/glyd-engine-210
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat2 && export LD_LIBRARY_PATH=$HOME/compat2:${LD_LIBRARY_PATH:-}
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl) > /dev/null 2>&1
"$E" serve deepseek-ai/DeepSeek-V4-Flash --port 8100 > $O/ds2-serve.log 2>&1 &
sp=$!
for i in $(seq 1 360); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8100/health)" = 200 ] && break; kill -0 $sp 2>/dev/null || break; sleep 5; done
for r in chat tools toolsreq stream; do curl -s -m 900 localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d @$HOME/dc/req/$r.json > $O/ds2-$r.json; done
kill $sp; wait $sp 2>/dev/null
touch $O/RETEST-DONE
