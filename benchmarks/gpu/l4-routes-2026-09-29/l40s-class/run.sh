#!/usr/bin/env bash
# the L40S route change (l4-routes) checked on the L4: check_capi (its routes pinned, an L40S made as one), test_gpu.py, the crate
[ -e ~/.glyd-busy ] || { touch ~/.glyd-busy; trap "rm -f ~/.glyd-busy" EXIT; }
exec flock $HOME/.glyd-box.lock bash -c '
set -u
cd ~/l40schk && R=~/l40schk/results && mkdir -p $R && unset LD_LIBRARY_PATH && source ~/gpuenv/cuda.sh
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=~/l40schk/w && rm -rf $W && mkdir -p $W && tar -C $W -xf src.tar && cp $W/COMMIT $R/COMMIT
(cd $W/gpu && timeout 900 bash build_lib.sh $W/lib) > $R/build_lib.txt 2>&1; echo "build_lib exit $?"
export GLYD_GPU_LIB=$(ls $W/lib/libglyd_gpu_cuda*.so | head -1) PYTHONPATH=$W/bindings/python
(cd $W/gpu && MAX_JOBS=8 timeout 1500 python -u check_capi.py $GLYD_GPU_LIB) > $R/check_capi.txt 2>&1; echo "check_capi exit $?" | tee -a $R/check_capi.txt
(cd $W/bindings/python && timeout 1200 python -u test_gpu.py) > $R/test_gpu.txt 2>&1; echo "test_gpu exit $?" | tee -a $R/test_gpu.txt
(cd $W/glyd-gpu && source ~/.cargo/env 2>/dev/null; timeout 900 cargo test --release -- --test-threads 1) > $R/cargo_test.txt 2>&1; echo "cargo test exit $?" | tee -a $R/cargo_test.txt
touch $R/DONE
'
