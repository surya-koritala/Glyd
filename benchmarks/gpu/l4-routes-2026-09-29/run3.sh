#!/usr/bin/env bash
# The L4 routes (l4-routes): check_capi (the JIT build and the library, bit for bit; the routes pinned) and test_gpu.py
# on the L4, then the prompt passes by the library's own routes (default) against bf16, both layouts, both models.
[ -e ~/.glyd-busy ] || { touch ~/.glyd-busy; trap "rm -f ~/.glyd-busy" EXIT; }
exec flock $HOME/.glyd-box.lock bash -c '
set -u
cd ~/l4routes && R=~/l4routes/results-new && mkdir -p $R/log && unset LD_LIBRARY_PATH && source ~/gpuenv/cuda.sh
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TZ=UTC
W=~/l4routes/wnew && rm -rf $W && mkdir -p $W && tar -C $W -xf l4_src_new.tar && cat $W/COMMIT
(cd $W/gpu && timeout 900 bash build_lib.sh $W/lib) > $R/log/build_lib.txt 2>&1; echo "build_lib exit $?"
export GLYD_GPU_LIB=$(ls $W/lib/libglyd_gpu_cuda*.so | head -1) PYTHONPATH=$W/bindings/python
(cd $W/gpu && MAX_JOBS=8 timeout 1500 python -u check_capi.py $GLYD_GPU_LIB) > $R/check_capi.txt 2>&1; echo "check_capi exit $?" | tee -a $R/check_capi.txt
(cd $W/bindings/python && timeout 1200 python -u test_gpu.py) > $R/test_gpu.txt 2>&1; echo "test_gpu exit $?" | tee -a $R/test_gpu.txt
(cd $W/glyd-gpu && source ~/.cargo/env 2>/dev/null; timeout 900 cargo test --release -- --test-threads 1) > $R/cargo_test.txt 2>&1; echo "cargo test exit $?" | tee -a $R/cargo_test.txt
echo "$(cat $W/COMMIT)" > $R/COMMIT
for m in Qwen/Qwen3-8B Qwen/Qwen3-4B-Instruct-2507; do
  d=$(python -c "from huggingface_hub import snapshot_download as s; print(s(\"$m\", allow_patterns=[\"*.json\", \"*.safetensors\", \"*.txt\", \"tokenizer*\"]))"); n=$(basename $m)
  for mode in mma mma12 bf16; do
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active --format=csv,noheader -lms 100 > $R/smi-$n-$mode.csv 2> /dev/null & S=$!
    if [ $mode = bf16 ]; then a="--mode bf16"; else a="--mode glyd --layout $mode --routes default"; fi
    timeout 1800 python -u ~/l4routes/route_e2e.py $d $a --out $R/route-$n-$mode.json > $R/log/route-$n-$mode.txt 2>&1; echo "$n $mode exit $?"; kill $S
  done
done
python ~/l4routes/route_summary.py $R > $R/summary.txt 2> $R/log/summary.txt
touch $R/DONE
'
