#!/usr/bin/env bash
# generate()'s long-prompt gap on the L4 (gap.py: Glyd compiled and not, bf16), then decode-ahead again with its scratch
# buffer sized for it (route_e2e.py: Qwen3-8B, decoded against ahead), the tree l4-routes; under the lock
[ -e ~/.glyd-busy ] || { touch ~/.glyd-busy; trap "rm -f ~/.glyd-busy" EXIT; }
exec flock $HOME/.glyd-box.lock bash -c '
set -u
cd ~/gapwork && R=~/gapwork/results && mkdir -p $R/log && unset LD_LIBRARY_PATH && source ~/gpuenv/cuda.sh
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TZ=UTC
W=~/gapwork/w && rm -rf $W && mkdir -p $W && tar -C $W -xf src.tar && cat $W/COMMIT
(cd $W/gpu && timeout 900 bash build_lib.sh $W/lib) > $R/log/build_lib.txt 2>&1; echo "build_lib exit $?"
export GLYD_GPU_LIB=$(ls $W/lib/libglyd_gpu_cuda*.so | head -1) PYTHONPATH=$W/bindings/python
d=$(python -c "from huggingface_hub import snapshot_download as s; print(s(\"Qwen/Qwen3-8B\", allow_patterns=[\"*.json\", \"*.safetensors\", \"*.txt\", \"tokenizer*\"]))")
for m in "glyd --compile 1" "glyd --compile 0" "bf16"; do
  n=$(echo $m | tr -d " -"); timeout 1200 python -u gap.py $d --mode $m > $R/gap-$n.txt 2>&1; echo "gap $n exit $?"; grep -E "tokens:|profiled" $R/gap-$n.txt | cut -c1-230
done
for mode in mma mma12; do
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active --format=csv,noheader -lms 100 > $R/smi-Qwen3-8B-$mode.csv 2> /dev/null & S=$!
  timeout 1500 python -u route_e2e.py $d --mode glyd --layout $mode --routes decoded,ahead --lengths 896,1024,2048,4096,8192 --out $R/route-Qwen3-8B-$mode.json > $R/log/route-Qwen3-8B-$mode.txt 2>&1; echo "route $mode exit $?"; kill $S
  grep tokens, $R/log/route-Qwen3-8B-$mode.txt
done
touch $R/DONE
'
