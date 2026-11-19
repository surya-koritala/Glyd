#!/bin/bash
# In the CUDA 13 container: Kimi-K3's first 4 layers sliced from the Hub (0-2 delta attention, 3 latent attention; 0 dense, 1-3 its 896 MXFP4 experts),
# the dynamic engine against the CPU reference (src/kimi_ref.rs) with the expert cache (40 GB holds 128 or 16 places a layer, not the 3 x 15.7 GB of experts).
W=$HOME/w; O=$W/out; mkdir -p $O; cd $W
exec > $O/session.log 2>&1
set -x
export GLYD_DOWNLOAD_THREADS=64; export GLYD_GPU_LIB=$HOME/dc/libglyd_gpu_cuda13.so; E=$HOME/dc/glyd-engine; chmod +x $E
[ -d /usr/local/cuda/compat ] && cp -a /usr/local/cuda/compat $HOME/compat && export LD_LIBRARY_PATH=$HOME/compat:${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total,driver_version --format=csv > $O/machine.txt
step() { local out=${@: -1}; echo "$(date +%T) start $out" >> $O/steps.txt; timeout 3600 "${@:1:$#-1}" > $out 2>&1; echo "$(date +%T) end $out exit $?" >> $O/steps.txt; }
step $E hubslice moonshotai/Kimi-K3 $W/k4 --layers 4 --prefix language_model.model. $O/slice.txt
step env GLYD_KIMI_PLACES=128 $E kimilayers $W/k4 --tokens 6 $O/layers.txt
step env GLYD_KIMI_PLACES=128 $E kimicheck $W/k4 --tokens 16 $O/check-128.txt
step env GLYD_KIMI_PLACES=16 $E kimicheck $W/k4 --tokens 8 --no-graph $O/check-16.txt
touch $O/DONE
