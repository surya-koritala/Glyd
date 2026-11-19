#!/bin/bash
# inner3.sh (19:00 EDT): GLM-5.3-Flash's tool requests again (the library's workspace now sized for the latent attentions' matrices too: they failed on
# prompts past 64 rows), and DeepSeek-V4-Flash's tools request with each token's log-probability (what it writes where it gives no call).
W=$HOME/w; O=$W/out3; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc/glyd-engine
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat3 && export LD_LIBRARY_PATH=$HOME/compat3:${LD_LIBRARY_PATH:-}
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl) > $O/apt.txt 2>&1
serve() { "$E" serve "$1" --port 8100 > $O/$2-serve.log 2>&1 & SP=$!; for i in $(seq 1 360); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8100/health)" = 200 ] && break; kill -0 $SP 2>/dev/null || break; sleep 5; done; echo "$(date +%T) $2 up" >> $O/steps.txt; }
req() { curl -s -m 900 localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d @$HOME/dc/req/$1.json > $O/$2-$1.json; echo "$(date +%T) $2 $1" >> $O/steps.txt; }
serve zai-org/GLM-5.3-Flash glm
for r in tools toolsreq tools_sys stream; do req $r glm; done
kill $SP; wait $SP
serve deepseek-ai/DeepSeek-V4-Flash ds
for r in tools_lp tools_sys_lp; do req $r ds; done
kill $SP; wait $SP
touch $O/DONE
