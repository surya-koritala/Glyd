#!/bin/bash
# In the CUDA 13 container after inner.sh: the newer build (~/dc2: gathers of a few vectors one block; the shared expert's workspace in either layout;
# Kimi-K3's load without the upfront drop, FP4 packs 10x faster). The gathers' own time on 2, 4, 8 GPUs; DeepSeek-V4-Flash's decode steps across 2, 4,
# 8 GPUs against split by layers on 2; then Kimi-K3 across 8 again: tools and the pass key (which failed in the first run), the story.
W=$HOME/w; O=$W/out2; mkdir -p $O; cd $W
exec > $O/extra.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc2/glyd-engine; chmod +x $E
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat2 && export LD_LIBRARY_PATH=$HOME/compat2:${LD_LIBRARY_PATH:-}
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl procps) > $O/apt.txt 2>&1
for g in 0,1 0,1,2,3 0,1,2,3,4,5,6,7; do timeout 300 $E tpgather --tp $g > $O/tpgather-$(echo $g | tr -cd , | wc -c).txt 2>&1; done
DR=$(ls -d $HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/*/ | head -1)
NL=$(grep -ho '"num_hidden_layers": *[0-9]*' $DR/config.json | grep -o '[0-9]*$' | head -1)
timeout 900 $E tpbench $DR --tp 0,1 --split $(( NL / 2 )) --devices 0,1 --steps 256 > $O/tpbench-ds-2.txt 2>&1
timeout 900 $E tpbench $DR --tp 0,1,2,3 --steps 256 > $O/tpbench-ds-4.txt 2>&1
timeout 900 $E tpbench $DR --tp 0,1,2,3,4,5,6,7 --steps 256 > $O/tpbench-ds-8.txt 2>&1
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
export GLYD_KIMI_STATS=1 GLYD_MLA_TILE=512 GLYD_TP=1
serve_check tp2 "tools passkey long" moonshotai/Kimi-K3 --context 32768
touch $O/DONE
