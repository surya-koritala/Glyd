# The second pass on the same instance, once the first pass (run.sh) was done: the models whose count
# changed with sizes.py's rule for fused experts and every projection (gpu/sizes.py at c914521),
# measured again.
source ~/gpuenv/cuda.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=~/results
for m in google/gemma-4-26B-A4B-it Qwen/Qwen3.8-27B meta-models/Muse-Glimmer-30B Qwen/Qwen3-Next-80B-A3B-Instruct unsloth/Llama-4-Scout-17B-16E-Instruct; do
  n=${m#*/}; d=~/models/$n
  [ -f $d/config.json ] || timeout 3600 hf download $m --local-dir $d > $R/download2-$n.txt 2>&1
  (cd ~/fix && timeout 2400 python sizes.py $d >> $R/sizes2.txt 2>> $R/sizes2-err.txt)
  rm -rf $d
done
