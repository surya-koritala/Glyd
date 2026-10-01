# the wheel under test: bindings/python of release-0.26.0 (889f4e3) with this branch's changes applied and its version strings set to
# 0.25.1 (the file name the batch's step had), in $HOME/accept/bp-wheel.tar, over rc2's libraries (nothing in the tree's C code changed)
set -e
export PATH=$HOME/accept/bin:$PATH
S=$HOME/accept/work/env-rc2/lib/python3.12/site-packages
B=$HOME/accept/wheelbuild2
rm -rf $B && mkdir -p $B
tar -C $B -xf $HOME/accept/bp-wheel.tar 2>/dev/null
cp $S/glyd/libglyd_store.so $B/python/glyd/
cp $S/glyd/gpu/libglyd_gpu_cuda12.so $S/glyd/gpu/libglyd_gpu_cuda13.so $B/python/glyd/gpu/
cp $S/glyd-*.dist-info/licenses/* $B/python/
cd $B/python && uv build --wheel --out-dir $B/out . 2>&1 | tail -1
uvx --from wheel wheel tags --python-tag py3 --abi-tag none --platform-tag manylinux_2_28_x86_64 --remove $B/out/glyd-*-py3-none-any.whl 2>&1 | tail -1
cp $B/out/glyd-0.25.1-py3-none-manylinux_2_28_x86_64.whl $HOME/accept/wheels/glyd-0.25.1-py3-none-manylinux_2_28_x86_64.whl
unzip -p $HOME/accept/wheels/glyd-0.25.1-py3-none-manylinux_2_28_x86_64.whl glyd/gpu/_lib.py | grep -n "^API_VERSION"
unzip -p $HOME/accept/wheels/glyd-0.25.1-py3-none-manylinux_2_28_x86_64.whl glyd/gpu/vllm_plugin.py | grep -c "_trim"
ls -la $HOME/accept/wheels
tail -4 ~/accept/batch1.out | cut -c1-160
