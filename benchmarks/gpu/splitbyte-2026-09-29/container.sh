#!/bin/bash
# The split-byte kernels compiled and their SASS reviewed with no GPU: in nvidia/cuda:13.0.3-devel-rockylinux8 (the
# release's build image; here arm64, on a Mac), with /work holding main's tree (main/, git archive origin/main) and
# this one's (src/, git archive HEAD), both with gpu/ and bindings/python:
#   docker run --rm --platform linux/arm64 -v "$PWD/work:/work" nvidia/cuda:13.0.3-devel-rockylinux8 bash /work/src/benchmarks/gpu/splitbyte-2026-09-29/container.sh
# (1) Each library for sm_80, 86, 89 and 90a with build_lib.sh's flags and GCC 13 as the release builds it (compute_80's
# PTX and Blackwell left out), and main's again with its exception loop kept to an entry a pass as this tree's is
# (main-once/: the baseline for the schedule). (2) Each architecture's SASS compared by sass_diff.py: main -> this tree,
# main -> main-once, main-once -> this tree. (3) This tree's JIT source (the pybind module) compiled as PyTorch 2.14's
# extension build compiles it, against the headers of its CUDA 13 wheel, its SASS against the library's, and linked
# with every symbol resolved (-z defs; libpython linked for the check). Logs in /work/out.
set -u
cd /work
mkdir -p out
exec > >(tee out/container.txt) 2>&1
dnf install -y -q --setopt=install_weak_deps=False gcc-toolset-13-gcc-c++ gcc-toolset-13-binutils python3.11 python3.11-devel python3.11-pip time > out/dnf.txt 2>&1
source /opt/rh/gcc-toolset-13/enable
CU=/usr/local/cuda
S=src/benchmarks/gpu/splitbyte-2026-09-29
echo "== $(date -u +%T) $(nvcc --version | tail -1); $(gcc --version | head -1); $(uname -m), $(nproc) CPUs; main $(cat main/COMMIT 2>/dev/null), this tree $(cat src/COMMIT 2>/dev/null)"
rm -rf main-once && cp -R main main-once
sed -i '/static __device__ __forceinline__ void patch(At at, int e0, int e1, int lane, uint32_t ew\[8\]) {/a #pragma unroll 1' main-once/gpu/glyd_gpu.cu
echo "main-once: $(( $(grep -c '^#pragma unroll 1$' main-once/gpu/glyd_gpu.cu) - $(grep -c '^#pragma unroll 1$' main/gpu/glyd_gpu.cu) )) pragma added (main's Nib::patch)"
GEN="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90a,code=sm_90a"
F="-O3 -std=c++20 --expt-relaxed-constexpr -isystem $CU/include -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ $GEN"
for side in src main main-once; do
  mkdir -p out/$side
  /usr/bin/time -f "%e s, %M KB peak" nvcc $F -Xcompiler -fPIC,-fvisibility=hidden --threads 4 -c -o out/$side/glyd_gpu.o $side/gpu/glyd_gpu.cu > out/$side/build.txt 2>&1
  e=$?
  nvcc $F -Xcompiler -fPIC,-fvisibility=hidden -shared -Xlinker --exclude-libs,ALL -cudart static -L$CU/lib64 -o out/$side/libglyd_gpu_cuda13.so out/$side/glyd_gpu.o >> out/$side/build.txt 2>&1
  echo "== $(date -u +%T) $side: compile exit $e, link exit $? ($(tail -1 out/$side/build.txt)); warnings and errors: $(grep -c -i -E 'warning|error' out/$side/build.txt)"
  for a in 80 86 89 90a; do
    cuobjdump -sass -arch sm_$a out/$side/libglyd_gpu_cuda13.so > out/$side/sm_$a.sass
    cuobjdump -res-usage -arch sm_$a out/$side/libglyd_gpu_cuda13.so > out/$side/sm_$a.res
  done
done
for pair in "main src" "main main-once" "main-once src"; do
  set -- $pair
  for a in 80 86 89 90a; do
    echo "## sm_$a"
    python3.11 $S/sass_diff.py out/$1/sm_$a.sass out/$2/sm_$a.sass out/$1/sm_$a.res out/$2/sm_$a.res
  done > out/sass-$1-to-$2.txt 2>&1
  echo "== SASS, $1 -> $2:"; grep -E "^## |kernels; the|kernels of the 12-bit" out/sass-$1-to-$2.txt | sed 's/^/   /'
done
echo "== $(date -u +%T) the JIT source, as PyTorch 2.14's extension build compiles it"
[ -d /opt/torchcu ] || { python3.11 -m pip download -q --no-deps --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.0" -d /tmp/whl &&
  python3.11 -m zipfile -e /tmp/whl/torch-2.14.0+cu130-*.whl /opt/torchcu; }
TI=/opt/torchcu/torch/include TL=/opt/torchcu/torch/lib
JF="-O3 -std=c++20 --expt-relaxed-constexpr -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ $GEN -I$TI -I$TI/torch/csrc/api/include -isystem $CU/include -I/usr/include/python3.11 -DTORCH_EXTENSION_NAME=glyd_gpu -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1"
/usr/bin/time -f "%e s, %M KB peak" nvcc $JF -Xcompiler -fPIC --threads 1 -c -o out/src/glyd_gpu_jit.o src/gpu/glyd_gpu.cu > out/src/jit.txt 2>&1
e=$?
g++ -shared -o out/src/glyd_gpu_jit.so out/src/glyd_gpu_jit.o -L$TL -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L$CU/lib64 -lcudart -lpython3.11 -Wl,-z,defs -Wl,--allow-shlib-undefined >> out/src/jit.txt 2>&1
echo "compile exit $e, link exit $? ($(grep -E '^[0-9.]+ s, ' out/src/jit.txt)); warnings not from torch's headers: $(grep -i 'warning #' out/src/jit.txt | grep -v -c '/torch/include/')"
for a in 80 86 89 90a; do
  cuobjdump -sass -arch sm_$a out/src/glyd_gpu_jit.o > out/src/jit_sm_$a.sass
  python3.11 - out/src/sm_$a.sass out/src/jit_sm_$a.sass sm_$a <<'PY'
import re, sys
def ops(p):
    fns, name = {}, None
    for line in open(p):
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            name = m.group(1)
            fns[name] = []
            continue
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if m and name:
            fns[name].append(m.group(1))
    return fns
a, b = ops(sys.argv[1]), ops(sys.argv[2])
print(f"   {sys.argv[3]}: the JIT build's {len(b)} kernels, the library's {len(a)}: {sum(a[n] == b.get(n) for n in a)} the same instructions")
PY
done
echo "== $(date -u +%T) done"
