#!/bin/bash
# glyd's wheel from release-0.26.0 (c956e54) on the dev L4, as the release workflow builds it: the GPU library by
# gpu/build_lib.sh (CUDA 13, every architecture), the codec by cargo, then python -m build; under the box's lock.
set -e
Q=~/quick; cd $Q
rm -rf src wheels && mkdir -p src wheels && tar -C src -xf src.tar
cd src
echo "== $(date -u +%T) build_lib.sh"; time bash gpu/build_lib.sh $Q/src/lib > $Q/build_lib.txt 2>&1; tail -1 $Q/build_lib.txt
echo "== $(date -u +%T) cargo"; time ~/.cargo/bin/cargo build --release --workspace > $Q/cargo.txt 2>&1; tail -1 $Q/cargo.txt
cp target/release/libglyd_store.* bindings/python/glyd/ 2>/dev/null || true
cp lib/libglyd_gpu_cuda13.so bindings/python/glyd/gpu/
cp LICENSE COPYING bindings/python/ && cp glyd-store/LICENSE bindings/python/LICENSE-glyd-store && cp gpu/LICENSE bindings/python/LICENSE-glyd-gpu
python3 -m venv $Q/buildenv && $Q/buildenv/bin/pip install --quiet build wheel
$Q/buildenv/bin/python -m build --wheel --outdir $Q/wheels bindings/python > $Q/build_wheel.txt 2>&1; tail -2 $Q/build_wheel.txt
$Q/buildenv/bin/python -m wheel tags --python-tag py3 --abi-tag none --platform-tag manylinux_2_28_x86_64 --remove $Q/wheels/*.whl
ls -la $Q/wheels; unzip -l $Q/wheels/*.whl | grep -E "\.so|vllm_(plugin|entry)|entry_points" 
echo "== $(date -u +%T) built"
