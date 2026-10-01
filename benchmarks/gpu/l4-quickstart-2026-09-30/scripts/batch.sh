#!/bin/bash
# one hold of the box lock: the packers' bench; the fixed tree at 16 GB (three chunkings) and 24 GB; the sampler (off, with precompiled
# kernels, the owner's case); the reasoning parser; then acceptance.sh on rc2 and on the tree's wheel
export DK=./dkg.sh
cd ~/accept
OUT=~/accept/batch1.out; W=~/accept/work/logs
export HF_HOME=$HOME/hf PATH=$HOME/accept/bin:$PATH
: > $OUT
stamp() { echo "=========== $* ($(date -u +%T))" >> $OUT; }
summ() { N=$1
  grep -E "OOM warnings|Model loading|Available KV|GPU KV|memwatch: most|Free memory on device" $W/$N.out | sort -u | cut -c1-230 >> $OUT
  grep -E "with OOM" $W/$N.log | sed -E "s/.*(memory allocation failed|memory mapping failed).*/\1/" | sort | uniq -c >> $OUT
  grep -E "probe\] (at the end of loading: [0-9]+ live|cached)" $W/$N.log | cut -c1-300 >> $OUT
  grep -E "peak reserved" $W/$N.log | tail -1 | cut -c1-260 >> $OUT
  [ -f $W/$N.after ] && { echo "--- after:" >> $OUT; cat $W/$N.after >> $OUT; }; }
P=/work/fixpkg:/work/probe
OFF="-e VLLM_USE_FLASHINFER_SAMPLER=0"
BENCH="python /work/chatbench.py http://127.0.0.1:8000/v1 Qwen/Qwen3-8B 5"
EXTRA="--enable-auto-tool-choice --tool-call-parser hermes"

stamp packbench
./dkg.sh -e PYTHONPATH=/work/fixpkg -- -c ". /work/env-rc2/bin/activate; python /work/packbench.py" > $W/packbench.out 2>&1; cat $W/packbench.out >> $OUT

stamp "fix-a: the fixed tree, 16 GB, hist 4M pack 2M"; ./rv.sh env-rc2 fix-a 15016 0.622 $OFF -e PYTHONPATH=$P > /dev/null 2>&1; summ fix-a
stamp "fix-b: chunks 2M / 1M"; ./rv.sh env-rc2 fix-b 15016 0.622 $OFF -e PYTHONPATH=$P -e GLYD_PROBE_CHUNKS=2097152,1048576 > /dev/null 2>&1; summ fix-b
stamp "fix-c: chunks 4M / 4M (the trim alone)"; ./rv.sh env-rc2 fix-c 15016 0.622 $OFF -e PYTHONPATH=$P -e GLYD_PROBE_CHUNKS=4194304,4194304 > /dev/null 2>&1; summ fix-c
stamp "fix-24: no hog, 0.88"; ./rv.sh env-rc2 fix-24 99999 0.88 $OFF -e PYTHONPATH=$P > /dev/null 2>&1; summ fix-24

stamp "sampler off (PyTorch + Triton), 16 GB: tokens/s"; ./rv.sh env-rc2 s-off 15016 0.622 $OFF -e PYTHONPATH=$P -e "AFTER=$BENCH" > /dev/null 2>&1; summ s-off
stamp "sampler on, flashinfer-jit-cache, no nvcc, 16 GB: tokens/s"; ./rv.sh env-fi s-fi 15016 0.622 -e PYTHONPATH=$P -e "AFTER=$BENCH" > /dev/null 2>&1; summ s-fi
stamp "sampler_check (FlashInfer against PyTorch's draws)"; ./dkg.sh -e PYTHONPATH=/work/fixpkg -- -c ". /work/env-fi/bin/activate; python /work/sampler_check.py" > $W/sampler_check.out 2>&1; grep -v -E "^(INFO|WARNING)" $W/sampler_check.out | tail -12 >> $OUT
stamp "the owner's case with the fixed plugin: sampler on, no nvcc, no precompiled kernels"; ./rv.sh env-rc2 s-nvcc 15016 0.622 -e PYTHONPATH=$P > /dev/null 2>&1; summ s-nvcc
grep -E "glyd: no nvcc|Could not find nvcc" $W/s-nvcc.log | cut -c1-420 | head -3 >> $OUT

stamp "reasoning: hermes only"; ./rv.sh env-rc2 r-none 15016 0.622 $OFF -e PYTHONPATH=$P -e "SERVE_EXTRA=$EXTRA" -e "AFTER=python /work/reason_probe.py" > /dev/null 2>&1; summ r-none
stamp "reasoning: hermes and --reasoning-parser qwen3"; ./rv.sh env-rc2 r-qwen3 15016 0.622 $OFF -e PYTHONPATH=$P -e "SERVE_EXTRA=$EXTRA --reasoning-parser qwen3" -e "AFTER=python /work/reason_probe.py" > /dev/null 2>&1; summ r-qwen3

cd ~/accept/repo/gpu/vllm
stamp "acceptance.sh on rc2, the owner's command (both failures expected)"
bash acceptance.sh --version 0.26.0rc2 --card 4080s --webui none --work ~/accept/acc-rc2 --command "vllm serve Qwen/Qwen3-8B --quantization glyd --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.88 --host 127.0.0.1" > ~/accept/acc-rc2.out 2>&1
cat ~/accept/acc-rc2.out | cut -c1-400 >> $OUT
stamp "acceptance.sh on the fixed wheel, the README's command, both Open WebUI routes"
bash acceptance.sh --wheel $HOME/accept/wheels/glyd-0.25.1-py3-none-manylinux_2_28_x86_64.whl --card 4080s --work ~/accept/acc-fix > ~/accept/acc-fix.out 2>&1
cat ~/accept/acc-fix.out | cut -c1-400 >> $OUT
cd ~/accept && mkdir -p checks
stamp "check_vllm.py --brief with the fixed plugin (overlay), no hog: Qwen3-1.7B (dense), then granite-3.1-3b-a800m-instruct (a mixture of experts)"
export PYTHONPATH=$HOME/accept/work/fixpkg HF_HUB_OFFLINE=1 HF_HOME=$HOME/hf
unset LD_LIBRARY_PATH; [ -f $HOME/gpuenv/cuda.sh ] && source $HOME/gpuenv/cuda.sh
timeout 1500 ~/quick/venv/bin/python check_vllm.py --brief --out ~/accept/checks/dense Qwen/Qwen3-1.7B > ~/accept/checks/dense.log 2>&1
tail -25 ~/accept/checks/dense.log | cut -c1-220 >> $OUT
timeout 1800 ~/quick/venv/bin/python check_vllm.py --brief --out ~/accept/checks/moe ibm-granite/granite-3.1-3b-a800m-instruct > ~/accept/checks/moe.log 2>&1
tail -25 ~/accept/checks/moe.log | cut -c1-220 >> $OUT
echo "done $(date -u +%T)" >> $OUT
cd ~/accept
unset PYTHONPATH
stamp "RERUN with the corrected overlay (release Python, C API 7): fix-a"; ./rv.sh env-rc2 fix-a2 15016 0.622 $OFF -e PYTHONPATH=$P > /dev/null 2>&1; summ fix-a2
stamp "RERUN fix-b: chunks 2M / 1M"; ./rv.sh env-rc2 fix-b2 15016 0.622 $OFF -e PYTHONPATH=$P -e GLYD_PROBE_CHUNKS=2097152,1048576 > /dev/null 2>&1; summ fix-b2
stamp "RERUN fix-c: chunks 4M / 4M (the trim alone)"; ./rv.sh env-rc2 fix-c2 15016 0.622 $OFF -e PYTHONPATH=$P -e GLYD_PROBE_CHUNKS=4194304,4194304 > /dev/null 2>&1; summ fix-c2
echo "rerun done $(date -u +%T)" >> $OUT
cd ~/accept/repo/gpu/vllm
export HF_HOME=$HOME/hf PATH=$HOME/accept/bin:$PATH
unset PYTHONPATH
stamp "acceptance.sh on rc2, the owner's command (both failures expected), work dir ~/accept/work"
bash acceptance.sh --version 0.26.0rc2 --card 4080s --webui none --work ~/accept/work --command "vllm serve Qwen/Qwen3-8B --quantization glyd --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.88 --host 127.0.0.1" > ~/accept/acc-rc2.out 2>&1
mkdir -p ~/accept/acc-rc2-logs && cp -a ~/accept/work/logs/server.log ~/accept/work/logs/install.log ~/accept/work/logs/freeze.txt ~/accept/work/logs/hog.log ~/accept/work/summary.txt ~/accept/work/serve.sh ~/accept/acc-rc2-logs/ 2>/dev/null
cat ~/accept/acc-rc2.out | cut -c1-400 >> $OUT
stamp "acceptance.sh on the fixed wheel (merged tree), the README's command, both Open WebUI routes"
bash acceptance.sh --wheel $HOME/accept/wheels/glyd-0.25.1-py3-none-manylinux_2_28_x86_64.whl --card 4080s --work ~/accept/work > ~/accept/acc-fix.out 2>&1
mkdir -p ~/accept/acc-fix-logs && cp -a ~/accept/work/logs/server.log ~/accept/work/logs/install.log ~/accept/work/logs/freeze.txt ~/accept/work/logs/hog.log ~/accept/work/logs/chat.log ~/accept/work/logs/webui-*.log ~/accept/work/logs/webui-*.check.log ~/accept/work/summary.txt ~/accept/work/serve.sh ~/accept/acc-fix-logs/ 2>/dev/null
cat ~/accept/acc-fix.out | cut -c1-400 >> $OUT
echo "acceptance reruns done $(date -u +%T)" >> $OUT
cd ~/accept
BENCH="python /work/chatbench.py http://127.0.0.1:8000/v1 Qwen/Qwen3-8B 5"
stamp "rc2's plugin as published, sampler off, 16 GB: tokens/s (the fix's speed against it)"; ./rv.sh env-rc2 s-rc2 15016 0.622 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e "AFTER=$BENCH" > /dev/null 2>&1; summ s-rc2
echo "s-rc2 done $(date -u +%T)" >> $OUT
cd ~/accept
stamp "browser session: the README's server command and Open WebUI (uvx), held up until ~/accept/work/browser.done (25 minutes at most)"
rm -f ~/accept/work/browser.done
./rv.sh env-rc2 browser 15016 0.622 $OFF -e PYTHONPATH=$P -e "SERVE_EXTRA=$EXTRA --reasoning-parser qwen3" -e "AFTER=bash /work/browser_hold.sh" > /dev/null 2>&1; summ browser
echo "browser session done $(date -u +%T)" >> $OUT
cd ~/accept
stamp "rc2's plugin as published, no hog, 0.88 (24 GB-class: warnings?), sampler off"; ./rv.sh env-rc2 rc2-24 99999 0.88 $OFF > /dev/null 2>&1; summ rc2-24
echo "rc2-24 done $(date -u +%T)" >> $OUT
cd ~/accept/repo/gpu/vllm
export HF_HOME=$HOME/hf PATH=$HOME/accept/bin:$PATH
unset PYTHONPATH
stamp "acceptance.sh on the fixed wheel again, the README's command as it now is (with --reasoning-parser qwen3), held 25 minutes for the browser"
bash acceptance.sh --wheel $HOME/accept/wheels/glyd-0.25.1-py3-none-manylinux_2_28_x86_64.whl --card 4080s --hold 25 --work ~/accept/work > ~/accept/acc-fix2.out 2>&1
mkdir -p ~/accept/acc-fix2-logs && cp -a ~/accept/work/logs/server.log ~/accept/work/logs/install.log ~/accept/work/logs/freeze.txt ~/accept/work/logs/hog.log ~/accept/work/logs/chat.log ~/accept/work/logs/webui-*.log ~/accept/work/logs/webui-*.check.log ~/accept/work/summary.txt ~/accept/work/serve.sh ~/accept/acc-fix2-logs/ 2>/dev/null
cat ~/accept/acc-fix2.out | cut -c1-400 >> $OUT
echo "acceptance with browser hold done $(date -u +%T)" >> $OUT
cd ~/accept
unset PYTHONPATH
BENCH="python /work/chatbench.py http://127.0.0.1:8000/v1 Qwen/Qwen3-8B 5"
stamp "the fixed tree, sampler off, 16 GB again with Triton's cache warm (s-off was the first start, with its kernels compiled in the 8-user burst): tokens/s"; ./rv.sh env-rc2 s-off2 15016 0.622 $OFF -e PYTHONPATH=$P -e "AFTER=$BENCH" > /dev/null 2>&1; summ s-off2
stamp "FlashInfer's jit-cache again, the second start"; ./rv.sh env-fi s-fi2 15016 0.622 -e PYTHONPATH=$P -e "AFTER=$BENCH" > /dev/null 2>&1; summ s-fi2
echo "s-off2 done $(date -u +%T)" >> $OUT
