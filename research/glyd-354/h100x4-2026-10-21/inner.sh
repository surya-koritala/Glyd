#!/bin/bash
# In the CUDA 13 container on the 4x H100. GLY-354: DeepSeek-V4-Flash, Nemotron 3.5 Lightning and Nemotron 3 Super (hummingbird) through `glyd run` and
# `glyd serve` (chat, stream, tools, tool_choice required) on the dynamic engine; Lightning's and Super's dynamic engine against the hand-written one; Super's
# expert cache on both. GLY-351: DeepSeek's score and greedy text, the same build twice, on two GPUs and four, plain and coded FP8. GLY-318: its decode speed.
# GLM-5.3-Flash last: its FP8 matrices coded to fit (GLY-349).
W=$HOME/w; O=$W/out; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc/glyd-engine; chmod +x $E
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat && export LD_LIBRARY_PATH=$HOME/compat:${LD_LIBRARY_PATH:-}
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl) > $O/apt.txt 2>&1
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total,driver_version --format=csv > $O/machine.txt
for i in $(seq 1 12); do  # (802: the fabric still coming up; 100: the GPU not yet in the container)
  $E f8moecheck --tokens 33 > $O/probe.txt 2>&1
  grep -qiE "CUDA error (802|100)" $O/probe.txt || break
  sleep 20
done
if grep -qiE "CUDA error|insufficient|no device" $O/probe.txt; then echo FAIL-CUDA > $O/FAIL; touch $O/DONE; exit 1; fi
step() { local out=${@: -1}; echo "$(date +%T) start $out" >> $O/steps.txt; timeout 2400 "${@:1:$#-1}" > $out 2>&1; echo "$(date +%T) end $out exit $?" >> $O/steps.txt; }
wait_for() { for i in $(seq 1 360); do grep -q "^$1 exit" $O/download.txt 2>/dev/null && return 0; sleep 20; done; return 1; }
snap() { ls -d $HOME/.cache/huggingface/hub/models--${1//\//--}/snapshots/*/ | head -1; }
# `serve MODEL` on port 8100: the four requests of dc/req, the info; then stopped
serve_check() {
  local tag=$1; shift
  "$E" serve "$@" --port 8100 > $O/$tag-serve.log 2>&1 &
  local sp=$!
  for i in $(seq 1 360); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8100/health)" = 200 ] && break; kill -0 $sp 2>/dev/null || break; sleep 5; done
  for r in chat stream tools toolsreq; do curl -s -m 900 localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d @$HOME/dc/req/$r.json > $O/$tag-$r.json; done
  curl -s localhost:8100/v1/glyd/info > $O/$tag-info.json
  kill $sp; wait $sp 2>/dev/null
}
Q="What is the capital of France? Answer in one sentence."

# ---- DeepSeek-V4-Flash
DSR=deepseek-ai/DeepSeek-V4-Flash
if wait_for $DSR; then
  DS=$(snap $DSR)
  # (GLY-351: the same build twice on two GPUs, then four GPUs, then coded FP8; every token's log-probability. GLY-318: 256 tokens' decode on two and four)
  step $E dsv4gen $DS --score $HOME/dc/score-text.txt --out $O/ds-score2-r1.lp --split 22 --devices 0,1 $O/ds-score2-r1.txt
  step $E dsv4gen $DS --score $HOME/dc/score-text.txt --out $O/ds-score2-r2.lp --split 22 --devices 0,1 $O/ds-score2-r2.txt
  step $E dsv4gen $DS --score $HOME/dc/score-text.txt --out $O/ds-score4.lp --split 11,22,33 --devices 0,1,2,3 $O/ds-score4.txt
  step env GLYD_F8H_PLAIN=0 $E dsv4gen $DS --score $HOME/dc/score-text.txt --out $O/ds-score2-coded.lp --split 22 --devices 0,1 $O/ds-score2-coded.txt
  md5sum $O/ds-score*.lp > $O/ds-score-md5.txt
  step $E dsv4gen $DS --prompt "The capital of France is" --tokens 256 --split 22 --devices 0,1 $O/ds-paris2.txt
  step $E dsv4gen $DS --prompt "The capital of France is" --tokens 256 --split 11,22,33 --devices 0,1,2,3 $O/ds-paris4.txt
  step $E run $DSR --prompt "$Q" --temp 0 $O/ds-run.txt
  serve_check ds $DSR
else echo "no DeepSeek download" > $O/ds-FAIL; fi

# ---- Nemotron 3.5 Lightning (penguin): the hand-written engine against the dynamic one, then run and serve
LTR=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
if wait_for $LTR; then
  LT=$(snap $LTR)
  step $E check $LT $HOME/dc/prompts-nh.json $O/lt-old.json --layout mma $O/lt-check-old.txt
  step $E check $LT $HOME/dc/prompts-nh.json $O/lt-new.json --layout mma --dynamic $O/lt-check-new.txt
  cmp $O/lt-old.json $O/lt-new.json > $O/lt-cmp.txt 2>&1; echo "cmp exit $?" >> $O/lt-cmp.txt
  step $E run $LTR --prompt "$Q" --temp 0 $O/lt-run.txt
  serve_check lt $LTR
else echo "no Lightning download" > $O/lt-FAIL; fi

# ---- Nemotron 3 Super at hummingbird: the dynamic engine (now `run`'s) against the hand-written one, every expert on the GPU and with an 8 GiB cache
SUR=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:hummingbird
if wait_for glyd/nemotron-3-hummingbird; then
  for v in 1 0; do
    step env GLYD_NH_DYNAMIC=$v $E run $SUR --prompt "$Q" --temp 0 $O/su-run-dyn$v.txt
    step env GLYD_NH_DYNAMIC=$v $E run $SUR --prompt "$Q" --temp 0 --expert-cache 8 $O/su-cache-dyn$v.txt
  done
  serve_check su $SUR
  serve_check su-cache $SUR --expert-cache 8
else echo "no Super download" > $O/su-FAIL; fi

# ---- GLM-5.3-Flash (FP8): plain packs do not fit 4x 80 GB, so `run` holds the FP8 matrices coded (the fit's choice, GLY-349); run and serve
GLR=zai-org/GLM-5.3-Flash
if wait_for $GLR; then
  step $E run $GLR --prompt "$Q" --temp 0 $O/glm-run.txt
  nvidia-smi --query-gpu=memory.used --format=csv,noheader > $O/glm-mem-idle.txt
  serve_check glm $GLR
else echo "no GLM download" > $O/glm-FAIL; fi
du -sh $HOME/.cache/huggingface/hub/* > $O/disk.txt 2>&1
touch $O/DONE
