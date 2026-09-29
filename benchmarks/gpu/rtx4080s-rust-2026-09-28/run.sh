#!/bin/bash
# rust-gpu on an RTX 4080 SUPER (Ryzen 9 7950X3D), the box shared with other jobs: every step under the box-wide lock
# (flock ~/.glyd-box.lock) once the GPU is idle, one at a time, at most 8 threads. Each step's log starts with the box's
# state as it began (load averages, the busiest processes): a step is contended where a process not the job's was busy
# (the 1-minute load past 2 with the job's own step not yet begun). Work in $W (the branch's tree in $W/src, main's
# package and library in $W/main), logs in $W/logs, the saves in $W/out.
#   W=~/p9rust2 bash run.sh
set -u
W=${W:-~/p9rust2}
cd $W && source ~/gpuenv/cuda.sh
export HF_HOME=~/p1/hf HF_HUB_OFFLINE=1 TMPDIR=$W/tmp TORCH_EXTENSIONS_DIR=$W/torch_ext CUDA_CACHE_PATH=~/p1/nvcache
export PATH=~/.cargo/bin:$PATH CARGO_TARGET_DIR=$W/target CARGO_BUILD_JOBS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_JOBS=8
LOCK=~/.glyd-box.lock L=$W/logs PY=~/pypi-venv/bin/python
LIB=$W/lib/libglyd_gpu_cuda13.so BIN=$W/target/release/glyd-gpu
mkdir -p $L $W/out $W/tmp
# run LOG 'CMD': CMD under the lock once the GPU is idle, its output appended to LOG after a line of the box's state as it
# began: the time, the load averages, the three busiest processes (%CPU and name)
run() {
    flock $LOCK bash -c "until [ -z \"\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)\" ]; do sleep 20; done
        { echo \"# \$(date '+%F %T'): load \$(cut -d' ' -f1-3 /proc/loadavg); busiest: \$(ps -eo pcpu=,comm= --sort=-pcpu | head -n 3 | tr -s ' ' | tr '\n' ';')\"; $2; } >> $L/$1 2>&1"
}

# the builds: the branch's library and crate, main's library
run build.txt "time bash $W/src/gpu/build_lib.sh $W/lib && time bash $W/main/gpu/build_lib.sh $W/main/lib && cd $W/src && cargo build --release -p glyd-gpu -p glyd --bins --examples -j 8 2>&1 | tail -n 3 && cargo test --release -p glyd-gpu -j 8 2>&1 | grep 'test result'"
export GLYD_GPU_LIB=$LIB PYTHONPATH=$W/src/bindings/python

# the checks: the self-test, check_capi (the JIT build against the library, bit for bit), test_gpu.py, check_api
run selftest.txt "cd $W/src/gpu && $PY glyd_gpu.py; echo exit \$?"
run check_capi.txt "cd $W/src/gpu && $PY check_capi.py $LIB; echo exit \$?"
run test_gpu.txt "cd $W/src/bindings/python && $PY test_gpu.py 2>&1 | grep -v 'Loading weights\|Writing model'; echo exit \$?"
run check_api-dense.txt "cd $W/src/gpu && $PY check_api.py Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B 2>&1 | grep -v 'Loading weights\|Writing model\|Fetching'; echo exit \$?"

# pack: Python's save of each model in each layout (on the GPU, once), then glyd pack's, three rounds on 8 threads,
# every file's sha256 against Python's; verify on the CPU (three rounds) and on the GPU (once); the 12-bit layout's
# load from the bf16 checkpoint, the tiered save and the 12-bit save, three rounds, fresh processes
for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B ibm-granite/granite-3.1-3b-a800m-instruct Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B; do
    n=$(basename $m)
    for layout in mma mma12; do
        py=$W/out/py-$n-$layout rs=$W/out/$n-$layout
        rm -rf $py
        run pack-python.txt "/usr/bin/time -f '%e s wall' $PY -m glyd.gpu pack $m $py --layout $layout 2>&1 | grep -v 'Loading weights\|Fetching'; (cd $py && sha256sum * > ../py-$n-$layout.sha256)"
        rm -rf $py
        for round in 1 2 3; do
            rm -rf $rs
            run pack.txt "echo 'round $round: $m $layout'; /usr/bin/time -f '%e s wall, %M KB peak RSS, %P CPU' $BIN pack $m $rs --layout $layout --threads 8; (cd $rs && sha256sum * > ../$n-$layout.sha256); if cmp -s $W/out/py-$n-$layout.sha256 $W/out/$n-$layout.sha256; then echo \"$n $layout: every file byte-identical to Python's (\$(wc -l < $W/out/$n-$layout.sha256) files)\"; else echo \"$n $layout: DIFFERS\"; diff $W/out/py-$n-$layout.sha256 $W/out/$n-$layout.sha256; fi"
        done
        for round in 1 2 3; do
            run verify.txt "echo 'round $round: $n $layout, cpu'; /usr/bin/time -f '%e s wall, %M KB peak RSS' $BIN verify $rs --threads 8"
        done
        run verify.txt "echo '$n $layout, cuda:0'; /usr/bin/time -f '%e s wall, %M KB peak RSS' $BIN verify $rs --device cuda:0"
    done
    case $n in granite*|Qwen3-4B*|Qwen3-8B)
        for round in 1 2 3; do
            for path in $m $W/out/$n-mma $W/out/$n-mma12; do
                run load.txt "echo 'round $round'; $PY $W/src/benchmarks/gpu/rtx4080s-rust-2026-09-28/load_time.py $path mma12 2>&1 | grep 'layout mma12'"
            done
        done;;
    esac
    [ $n = Qwen3-0.6B ] || rm -rf $W/out/$n-mma $W/out/$n-mma12
done

# unpack.c against the branch's library, on the saved Qwen3-0.6B (tiered): its first pack, then a merged one
SRC=$(ls -d ~/p1/hf/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ | head -1)
run unpack_c.txt "cd $W/src/gpu && gcc -O2 -Wall -I . -I \$CUDA_HOME/include examples/unpack.c -o $W/unpack -L $W/lib -lglyd_gpu_cuda13 -L \$CUDA_HOME/lib64 -lcudart -Wl,-rpath,$W/lib:\$CUDA_HOME/lib && cd $W/out && $W/unpack Qwen3-0.6B-mma $SRC && $W/unpack Qwen3-0.6B-mma $SRC model.layers.0.self_attn.q_proj; echo exit \$?"

# generate() eager, main's package and library against the branch's in turn (the order swapped each round), fresh
# processes: Qwen3-1.7B and Qwen3-4B-Instruct-2507, both layouts, 1, 8 and 32 sequences
for round in 1 2 3 4; do
    sides="main branch"; [ $((round % 2)) = 0 ] && sides="branch main"
    for m in Qwen/Qwen3-1.7B Qwen/Qwen3-4B-Instruct-2507; do
        for layout in mma mma12; do
            for side in $sides; do
                if [ $side = main ]; then pp=$W/main/bindings/python lib=$W/main/lib/libglyd_gpu_cuda13.so; else pp=$W/src/bindings/python lib=$LIB; fi
                run generate.txt "GLYD_GPU_LIB=$lib PYTHONPATH=$pp $PY $W/src/benchmarks/gpu/rtx4080s-rust-2026-09-28/gen.py $m $layout 1,8,32 128 2>&1 | grep 'tokens/s' | sed 's|^|round $round $side $(basename $m) |'"
            done
        done
    done
done
echo "all done $(date '+%F %T')" >> $L/run.txt
