#!/bin/bash
# The decode fix as merged (FINAL: the 12-bit whole and experts' decodes in ORDER 3's load order, the decode ahead in
# load()'s, no GLYD_DEC_ORDER) in SASS, with no GPU: in nvidia/cuda:13.0.3-devel-rockylinux8 (as sass.sh; here arm64,
# on a Mac), /work holding base/ (the tree it was merged onto: git archive of l4-routes), fix/ (git archive 73b9560: the
# orders a choice) and final/ (this tree), each gpu/ with a COMMIT file:
#   docker run --rm --platform linux/arm64 -v "$PWD/work:/work" nvidia/cuda:13.0.3-devel-rockylinux8 bash /work/final/benchmarks/gpu/decode-fix-2026-09-29/sass_final.sh
#   then, where Python is:  python3 sass_order.py --final work/out/base-A.sass work/out/fix-A.sass work/out/final-A.sass
# Each tree's glyd_gpu.cu for build_lib.sh's architectures with its flags, J at a time.
set -u
cd /work
mkdir -p out
[ -f /opt/rh/gcc-toolset-13/enable ] || dnf install -y -q --setopt=install_weak_deps=False gcc-toolset-13-gcc-c++ > out/dnf.txt 2>&1
source /opt/rh/gcc-toolset-13/enable
CU=/usr/local/cuda
F="-O3 -std=c++20 --expt-relaxed-constexpr -isystem $CU/include -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -Xcompiler -fPIC,-fvisibility=hidden"
echo "== $(nvcc --version | tail -1); $(gcc --version | head -1); $(uname -m); base $(cat base/COMMIT), fix $(cat fix/COMMIT), final $(cat final/COMMIT)" | tee out/sass-final.txt
for a in 80 86 89 90a 100 120; do for side in base fix final; do
  while [ "$(jobs -r | wc -l)" -ge "${J:-4}" ]; do sleep 2; done
  ( nvcc $F -gencode arch=compute_$a,code=sm_$a -cubin -o out/$side-$a.cubin $side/gpu/glyd_gpu.cu > out/$side-$a.build.txt 2>&1
    echo "exit $?" >> out/$side-$a.build.txt
    cuobjdump -sass out/$side-$a.cubin > out/$side-$a.sass ) &
done; done
wait
for a in 80 86 89 90a 100 120; do for side in base fix final; do
  echo "$side sm_$a: build $(tail -1 out/$side-$a.build.txt), warnings $(grep -c -i warning out/$side-$a.build.txt)" | tee -a out/sass-final.txt
done; done
