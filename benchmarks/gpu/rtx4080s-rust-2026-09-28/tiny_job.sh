#!/bin/bash
# The tiny checkpoints (tiny.py: Llama tied and not, Qwen2, Mistral, Granite, Qwen3 odd and untied, GraniteMoe)
# packed by Python (src2's package, on the GPU) and by glyd-gpu built from src2, both layouts, every file's sha256
# compared; glyd-gpu verify and python -m glyd.gpu verify of Rust's. Each GPU or heavy step under the box-wide lock.
W=~/p9rust2; source ~/gpuenv/cuda.sh
export HF_HOME=~/p1/hf HF_HUB_OFFLINE=1 TMPDIR=$W/tmp TORCH_EXTENSIONS_DIR=$W/torch_ext CUDA_CACHE_PATH=~/p1/nvcache
export PATH=~/.cargo/bin:$PATH CARGO_TARGET_DIR=$W/target2 CARGO_BUILD_JOBS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_JOBS=8
export GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda13.so PYTHONPATH=$W/src2/bindings/python
LOCK=~/.glyd-box.lock L=$W/logs2/tiny.txt BIN=$W/target2/release/glyd-gpu PY=~/pypi-venv/bin/python
idle='until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 20; done'
mkdir -p $W/logs2 $W/tmp
flock $LOCK bash -c "cd $W/src2 && cargo build --release -p glyd-gpu -j 8 2>&1 | tail -n 1 && cargo test --release -p glyd-gpu -j 8 2>&1 | grep 'test result' && rm -rf $W/tiny && taskset -c 0-7 $PY ~/p9rust2/tiny.py" > $L 2>&1
cd $W && rm -rf tinyout && mkdir -p tinyout
for d in tiny/*/; do
    n=$(basename $d)
    for layout in mma mma12; do
        flock $LOCK bash -c "$idle; cd $W && $PY -m glyd.gpu pack tiny/$n tinyout/py-$n-$layout --layout $layout 2>&1 | grep packed" >> $L
        flock $LOCK bash -c "cd $W && $BIN pack tiny/$n tinyout/rs-$n-$layout --layout $layout --threads 8 2>&1 | grep -v 'GB of bf16'" >> $L
        (cd tinyout/py-$n-$layout && sha256sum *) > tinyout/py.sha256; (cd tinyout/rs-$n-$layout && sha256sum *) > tinyout/rs.sha256
        if diff tinyout/py.sha256 tinyout/rs.sha256 > /dev/null; then echo "$n $layout: every file byte-identical to Python's ($(wc -l < tinyout/py.sha256) files)" >> $L; else echo "$n $layout: DIFFERS" >> $L; diff tinyout/py.sha256 tinyout/rs.sha256 >> $L; fi
        $BIN verify tinyout/rs-$n-$layout --threads 2 >> $L 2>&1
        flock $LOCK bash -c "$idle; cd $W && $PY -m glyd.gpu verify tinyout/rs-$n-$layout 2>&1 | grep -v 'Loading weights'" >> $L
    done
done
echo done >> $L
