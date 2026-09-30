#!/bin/bash
# The 12-bit decode kernel's load orders in SASS, with no GPU: in nvidia/cuda:13.0.3-devel-rockylinux8 (the release's
# build image; here arm64, on a Mac), /work holding main's tree (main/: git archive origin/main), v0.25.0's (rel/: git
# archive of rust-splitbyte at bfb2c4e) and this one's (fix/: git archive HEAD), each gpu/ with a COMMIT file:
#   docker run --rm --platform linux/arm64 -v "$PWD/work:/work" nvidia/cuda:13.0.3-devel-rockylinux8 bash /work/fix/benchmarks/gpu/decode-fix-2026-09-29/sass.sh
# Each tree's glyd_gpu.cu for sm_90a (Hopper) and sm_86 (an A10) with build_lib.sh's flags and GCC 13; then
# sass_order.py: v0.25.0's kernels against this tree's (all the same instructions, its ORDER 1-3 decode kernels
# added), and every 12-bit decode kernel's loads, shuffles, calls and stores in order, with its registers. Output in
# /work/out (sass.txt).
set -u
cd /work
mkdir -p out
exec > >(tee out/sass.txt) 2>&1
[ -f /opt/rh/gcc-toolset-13/enable ] || dnf install -y -q --setopt=install_weak_deps=False gcc-toolset-13-gcc-c++ python3.11 > out/dnf.txt 2>&1
source /opt/rh/gcc-toolset-13/enable
CU=/usr/local/cuda
D=fix/benchmarks/gpu/decode-fix-2026-09-29
F="-O3 -std=c++20 --expt-relaxed-constexpr -isystem $CU/include -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -Xcompiler -fPIC,-fvisibility=hidden"
echo "== $(nvcc --version | tail -1); $(gcc --version | head -1); $(uname -m); main $(cat main/COMMIT), v0.25.0 $(cat rel/COMMIT), this tree $(cat fix/COMMIT)"
for side in main rel fix; do
  for a in 90a 86; do
    ( nvcc $F -gencode arch=compute_$a,code=sm_$a -cubin -o out/$side-$a.cubin $side/gpu/glyd_gpu.cu > out/$side-$a.build.txt 2>&1
      echo "exit $?" >> out/$side-$a.build.txt
      cuobjdump -sass out/$side-$a.cubin > out/$side-$a.sass
      cuobjdump -res-usage out/$side-$a.cubin > out/$side-$a.res ) &
  done
done
wait
for side in main rel fix; do for a in 90a 86; do
  echo "$side sm_$a: build $(tail -1 out/$side-$a.build.txt), warnings $(grep -c -i warning out/$side-$a.build.txt)"
done; done
for a in 90a 86; do
  echo "## sm_$a"
  python3.11 $D/sass_order.py --same out/rel-$a.sass out/fix-$a.sass
  python3.11 $D/sass_order.py out/main-$a.sass out/main-$a.res out/rel-$a.sass out/rel-$a.res out/fix-$a.sass out/fix-$a.res
done
