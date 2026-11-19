#!/bin/bash
# devbuild.sh NAME: ~/dev/NAME/tree.tar (gpu/ and glyd-gpu-py/) -> ~/dev/NAME/src and the development library for this GPU (the job's flags) in ~/dev/NAME/lib; CPU only
set -e
N=${1:?name}; D=$HOME/dev/$N
rm -rf "$D/src" "$D/lib" && mkdir -p "$D/src" "$D/lib" && tar -C "$D/src" -xf "$D/tree.tar"
. $HOME/gpuenv/cuda.sh 2> /dev/null || true
CU=$(dirname "$(command -v nvcc)")/..
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); ARCH=${CC/./}
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode "arch=compute_$ARCH,code=sm_$ARCH" ${EXTRA_GENCODE:-} -Xcompiler -fPIC,-fvisibility=hidden)
t=$(date +%s)
nvcc "${F[@]}" ${EXTRA_NVCC:-} -c -o "$D/lib/glyd_gpu.o" "$D/src/gpu/glyd_gpu.cu"
nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$D/lib/libglyd_gpu_cuda$MAJOR.so" "$D/lib/glyd_gpu.o"
echo "built $D/lib/libglyd_gpu_cuda$MAJOR.so (sm_$ARCH) in $(( $(date +%s) - t )) s"
