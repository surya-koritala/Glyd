#!/bin/bash
# devrun.sh NAME CMD...: CMD (python moe_layer.py ... or others) in ~/dev/NAME/src/gpu with that tree's library and sources; the caller holds the box's lock
N=${1:?name}; shift; D=$HOME/dev/$N
unset LD_LIBRARY_PATH
export HF_HOME=${HF_HOME:-$HOME/hf} HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$D/src/glyd-gpu-py GLYD_GPU_LIB=$D/lib/libglyd_gpu_cuda13.so
cd "$D/src/gpu" && exec $HOME/v027/venv/bin/python -u "$@"
