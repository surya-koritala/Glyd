#!/bin/bash
# inner4.sh (19:10 EDT): GLM-5.3-Flash's required call again (a forced call now closes the thinking its template opens), and DeepSeek-V4-Flash's raw
# replies to the tools prompts the server makes (rendered on the side, sent to /v1/completions with the special tokens kept), and its tools with thinking.
W=$HOME/w; O=$W/out4; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc/glyd-engine
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat4 && export LD_LIBRARY_PATH=$HOME/compat4:${LD_LIBRARY_PATH:-}
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl) > $O/apt.txt 2>&1
serve() { "$E" serve "$1" --port 8100 > $O/$2-serve.log 2>&1 & SP=$!; for i in $(seq 1 360); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8100/health)" = 200 ] && break; kill -0 $SP 2>/dev/null || break; sleep 5; done; echo "$(date +%T) $2 up" >> $O/steps.txt; }
req() { curl -s -m 900 localhost:8100/v1/$3 -H 'Content-Type: application/json' -d @$HOME/dc/req/$1.json > $O/$2-$1.json; echo "$(date +%T) $2 $1" >> $O/steps.txt; }
serve zai-org/GLM-5.3-Flash glm
for r in toolsreq tools; do req $r glm chat/completions; done
kill $SP; wait $SP
serve deepseek-ai/DeepSeek-V4-Flash ds
for r in ds_tools_raw ds_tools_sys_raw; do req $r ds completions; done
req tools_think ds chat/completions
kill $SP; wait $SP
touch $O/DONE
