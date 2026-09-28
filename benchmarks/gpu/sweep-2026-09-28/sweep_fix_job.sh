# After sweep_job.sh's first version on a machine: the steps it got wrong, again. ninja on PATH (the JIT build),
# test_gpu.py from the source tree (its import test copies the package beside it), gen_eager with its whole output kept.
set -x
R=~/results; PY=~/wv/bin/python
[ -f ~/gpuenv/cuda.sh ] && source ~/gpuenv/cuda.sh > /dev/null 2>&1
export PATH=~/wv/bin:$PATH:/usr/local/cuda/bin HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LIB=$($PY -c "import glyd.gpu, os, torch; print(os.path.join(os.path.dirname(glyd.gpu.__file__), 'libglyd_gpu_cuda%s.so' % torch.version.cuda.split('.')[0]))")
{ which ninja nvcc; nvcc --version | tail -2; } > $R/fix-env.txt 2>&1
( cd ~/src/gpu && timeout 30m $PY check_capi.py $LIB ) > $R/check_capi.txt 2>&1; echo "exit $?" >> $R/check_capi.txt
( cd ~/src/gpu && timeout 20m $PY glyd_gpu.py ) > $R/selftest.txt 2>&1; echo "exit $?" >> $R/selftest.txt
( cd ~/src/bindings/python && GLYD_GPU_LIB=$LIB timeout 30m $PY test_gpu.py ) > $R/test_gpu.txt 2>&1; echo "exit $?" >> $R/test_gpu.txt
for w in bf16 glyd; do ( cd ~ && COMPILE=1 timeout 20m $PY ~/gen_eager.py Qwen/Qwen3-8B $w 1 128 ) > $R/compiled-$w.txt 2>&1; echo "exit $?" >> $R/compiled-$w.txt; done
touch $R/DONE2
