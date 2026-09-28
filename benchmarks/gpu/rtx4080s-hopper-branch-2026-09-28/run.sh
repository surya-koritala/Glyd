#!/bin/bash
# Review #4 on the RTX 4080 SUPER (sm_89): gpu-hopper (~/p2hopper/br) against main (~/p2hopper/main): the self-test,
# check_capi with main's library against this build, mma_gemm_mid at 17-64 tokens on Qwen2.5-7B's shapes and mma_gemm
# at 1-16 on Qwen3-8B's layers 2 and 12, each build twice in turn. Only when no other process holds the GPU.
source ~/gpuenv/cuda.sh
export HF_HOME=~/p1/hf TMPDIR=~/p1/tmp CUDA_CACHE_PATH=~/p1/nvcache
B=~/p2hopper/br/gpu; A=~/p2hopper/main/gpu; F=~/p2hopper/logs; mkdir -p $F ~/p1/tmp
st() { echo "$1 $(date -u +%H:%M:%S)" >> $F/status; }
free() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; }
(time bash $A/build_lib.sh ~/p2hopper/mainlib) > $F/build_lib-main.txt 2>&1 &
(cd $B && TORCH_EXTENSIONS_DIR=~/p1/torch_ext/p2hopper-br GLYD_GPU_ARCH=sm_89 python -c "import glyd_gpu" > $F/jit-br.txt 2>&1) &
(cd $A && TORCH_EXTENSIONS_DIR=~/p1/torch_ext/p2hopper-main GLYD_GPU_ARCH=sm_89 python -c "import glyd_gpu" > $F/jit-main.txt 2>&1) &
wait; st built
until free; do sleep 10; done; st free
cd $B
TORCH_EXTENSIONS_DIR=~/p1/torch_ext/p2hopper-br timeout 1200 python glyd_gpu.py > $F/selftest.txt 2>&1; st "selftest $?"
until free; do sleep 10; done
TORCH_EXTENSIONS_DIR=~/p1/torch_ext/p2hopper-br timeout 1200 python check_capi.py ~/p2hopper/mainlib/libglyd_gpu_cuda13.so > $F/check_capi-main-lib.txt 2>&1; st "capi $?"
Q=$(ls -d ~/p1/hf/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
for r in 1 2; do
  for v in main br; do
    until free; do sleep 10; done
    D=$([ $v = br ] && echo $B || echo $A)
    GLYD_SRC=$D TORCH_EXTENSIONS_DIR=~/p1/torch_ext/p2hopper-$v python ~/p2hopper/t_mid.py 17,24,32,48,64 > $F/mid-$v-$r.txt 2>&1
    for L in 2 12; do
      GLYD_SRC=$D TORCH_EXTENSIONS_DIR=~/p1/torch_ext/p2hopper-$v python ~/p2hopper/kbench.py $Q $L 1,8,16 mma12 > $F/step-$v-l$L-$r.txt 2>&1
    done
  done
done
st done
