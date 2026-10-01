# usage: run1.sh ENV NAME FREE_MIB UTIL [extra vllm serve args...]   (inside the container)
ENVDIR=$1; NAME=$2; FREE=$3; UTIL=$4; shift 4
export HOME=/work HF_HOME=/hf
. /work/$ENVDIR/bin/activate
L=/work/logs/$NAME.log
python /work/hog.py $FREE > /work/logs/$NAME.hog 2>&1 &
HOG=$!
for i in $(seq 1 60); do grep -q "hog:" /work/logs/$NAME.hog && break; sleep 1; done
cat /work/logs/$NAME.hog
echo "\$ vllm serve Qwen/Qwen3-8B --quantization glyd --enforce-eager --max-model-len 8192 --gpu-memory-utilization $UTIL --host 127.0.0.1 $*" > $L
env | grep -E "^(PATH|CUDA|VLLM|PYTORCH|GLYD|PYTHONPATH|HF_)" >> $L
vllm serve Qwen/Qwen3-8B --quantization glyd --enforce-eager --max-model-len 8192 --gpu-memory-utilization $UTIL --host 127.0.0.1 "$@" $SERVE_EXTRA >> $L 2>&1 &
SRV=$!
t0=$(date +%s)
for i in $(seq 1 400); do
  grep -q "Application startup complete" $L && { echo "up after $(( $(date +%s) - t0 )) s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "exited after $(( $(date +%s) - t0 )) s"; break; }
  sleep 3
done
[ -n "$AFTER" ] && grep -q "Application startup complete" $L && { echo "== $AFTER"; bash -c "$AFTER" > /work/logs/$NAME.after 2>&1; cat /work/logs/$NAME.after | head -60; }
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
kill $HOG
echo "OOM warnings: $(grep -c "with OOM" $L)"
grep -E "Model loading took|Available KV|GPU KV cache size|Free memory on device|Traceback|RuntimeError|nvcc" $L | cut -c1-260 | head
