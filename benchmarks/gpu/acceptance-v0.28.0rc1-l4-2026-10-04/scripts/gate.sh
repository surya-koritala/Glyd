#!/bin/bash
# the wheel gate (glyd-gpu-py/check_wheel.py of private release-0.28.0-gpu be9f9292 = tag v0.28.0rc1, the commit the wheels were built from) on the two glyd-gpu wheels
# as PyPI serves them (sha256 checked against PyPI's), with cuobjdump (CUDA 13.4.92), readelf, nm and c++filt
export PATH=$HOME/harden-tools/cuobjdump/nvidia/cu13/bin:$PATH
G=/opt/dlami/nvme/rel028/gate; rm -rf $G; mkdir -p $G && cd $G && tar -xzf /tmp/acc028-gate-src.tgz
cuobjdump --version | tail -2; command -v cuobjdump c++filt readelf nm
rcs=
while read -r w fn url sha size; do
  curl -fsSL -o $fn "$url" < /dev/null || { echo "FAIL download $fn"; exit 1; }
  got=$(sha256sum $fn | cut -d' ' -f1); sz=$(stat -c %s $fn)
  [ "$got" = "$sha" ] && [ "$sz" = "$size" ] && echo "sha256 ok  $fn $sz bytes $got (PyPI's)" || { echo "FAIL sha256 of $fn: $got $sz, PyPI says $sha $size"; exit 1; }
done < /tmp/wheels.list
for a in x86_64 aarch64; do
  fn=glyd_gpu-0.28.0rc1-cp310-abi3-manylinux_2_28_$a.whl
  echo "== python3 glyd-gpu-py/check_wheel.py $fn --version 0.28.0rc1 --platform manylinux_2_28_$a --source glyd-gpu-py/glyd_gpu --header glyd-gpu-py/include/glyd_gpu.h --license gpu/LICENSE --summary"
  python3 glyd-gpu-py/check_wheel.py $fn --version 0.28.0rc1 --platform manylinux_2_28_$a --source glyd-gpu-py/glyd_gpu --header glyd-gpu-py/include/glyd_gpu.h --license gpu/LICENSE --summary
  rc=$?; echo "== $a: exit $rc"; rcs="$rcs $a=$rc"
done
echo "GATE DONE:$rcs"
