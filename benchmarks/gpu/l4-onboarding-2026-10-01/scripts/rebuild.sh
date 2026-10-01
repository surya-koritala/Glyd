#!/bin/bash
# The latest onboarding branch as a wheel, as release.yml builds it (the CUDA 13 library from release-0.26.0's build), in ~/onb/wheels.
set -e
cd ~/onb/src && git pull -q && git log -1 --format="%h %s" | cut -c1-100
cp ~/quick/src/lib/libglyd_gpu_cuda13.so bindings/python/glyd/gpu/
cp LICENSE COPYING bindings/python/ && cp glyd-store/LICENSE bindings/python/LICENSE-glyd-store && cp gpu/LICENSE bindings/python/LICENSE-glyd-gpu
rm -rf ~/onb/wheels bindings/python/build bindings/python/*.egg-info
~/quick/buildenv/bin/python -m build --wheel --outdir ~/onb/wheels bindings/python > ~/onb/build.txt 2>&1 || { tail -20 ~/onb/build.txt; exit 1; }
~/quick/buildenv/bin/python -m wheel tags --python-tag py3 --abi-tag none --platform-tag manylinux_2_28_x86_64 --remove ~/onb/wheels/*.whl
ls ~/onb/wheels
