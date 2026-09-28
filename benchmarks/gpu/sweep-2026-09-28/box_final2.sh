# The steps box_final.sh got wrong, again: check_api dense and MoE in separate processes (16 GB), test_gpu.py from the
# source tree (its import test copies the package beside it) with the wheel's library, gen_eager with its whole output kept.
set -x
H=~/p3final; R=$H/results
export HF_HOME=~/p1/hf TMPDIR=~/p1/tmp TORCH_EXTENSIONS_DIR=~/p1/torch_ext CUDA_CACHE_PATH=~/p1/nvcache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/gpuenv/cuda.sh > /dev/null 2>&1
PY=$H/wv/bin/python
LIB=$($PY -c "import glyd.gpu, os, torch; print(os.path.join(os.path.dirname(glyd.gpu.__file__), 'libglyd_gpu_cuda%s.so' % torch.version.cuda.split('.')[0]))")
( cd $H/t && timeout 40m $PY check_api.py Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B ) > $R/check_api-dense.txt 2>&1; echo "exit $?" >> $R/check_api-dense.txt
( cd $H/t && timeout 40m $PY check_api.py ibm-granite/granite-3.1-3b-a800m-instruct ) > $R/check_api-moe.txt 2>&1; echo "exit $?" >> $R/check_api-moe.txt
( cd $H/src/bindings/python && GLYD_GPU_LIB=$LIB timeout 30m $PY test_gpu.py ) > $R/test_gpu-src.txt 2>&1; echo "exit $?" >> $R/test_gpu-src.txt
for w in bf16 glyd; do
  ( cd $H && timeout 15m $PY $H/gen_eager.py Qwen/Qwen3-4B-Instruct-2507 $w 1 128 ) > $R/eager-$w.txt 2>&1; echo "exit $?" >> $R/eager-$w.txt
  ( cd $H && COMPILE=1 timeout 20m $PY $H/gen_eager.py Qwen/Qwen3-4B-Instruct-2507 $w 1 128 ) > $R/compiled-$w.txt 2>&1; echo "exit $?" >> $R/compiled-$w.txt
done
touch $R/DONE2
