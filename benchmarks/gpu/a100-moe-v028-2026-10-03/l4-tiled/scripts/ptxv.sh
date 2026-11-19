#!/bin/bash
# ptxv.sh NAME TARBALL ARCH: ptxas -v of glyd_gpu.cu for one arch (CPU only), the lines of the kernels given in KER (default mma_gemm_big)
N=$1; T=$2; A=${3:-sm_80}; D=$HOME/dev/ptx-$N
rm -rf $D && mkdir -p $D/src && tar -C $D/src -xf $T
. $HOME/gpuenv/cuda.sh 2> /dev/null || true
CU=$(dirname "$(command -v nvcc)")/..
CC=${A#sm_}; CC=${CC%a}
nvcc -O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode "arch=compute_$CC$( [ "$A" = sm_90a ] && echo a ),code=$A" -Xptxas -v -c -o $D/o.o $D/src/gpu/glyd_gpu.cu > $D/ptxas.txt 2>&1
echo "done $N $A $(date -u +%T)" >> $D/ptxas.txt
