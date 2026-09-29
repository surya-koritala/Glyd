#!/bin/bash
# v0.25.0's Rust-against-Python check, and the release tree's self-test and xcheck, on the AWS dev machine: a
# g6.4xlarge, an NVIDIA L4 (Ada, sm_89), 16 vCPUs (8 cores of an AMD EPYC 7R13), 60 GB, CUDA 13. W/src is the release
# tree (rust-splitbyte, its COMMIT), W/main main's (origin/main db8e7b0: the 12-bit layout before split byte, C API 4)
# for xcheck. Another job shares the machine: every step runs under its lock (flock ~/.glyd-box.lock) once the GPU is
# idle, one at a time, each log's step after a line of the machine's state as it began (the load averages, the three
# busiest processes by ps's %CPU).
#   build.txt       the release library (gpu/build_lib.sh: every architecture), main's and two patched ones for sm_89
#                   alone (build-*.txt), the glyd-gpu crate (its binary, examples and tests)
#   selftest.txt    the self-test (gpu/glyd_gpu.py) with the release library
#   xcheck.txt      xcheck.py, synthetic: split byte against main's 12-bit layout in every entry point the L4 takes
#   xcheck-geforce.txt  the same through both trees built with the library's GeForce test matching the L4's name
#                   ("L4" for "GeForce"): GeForce Ada's prompt kernels (stream-K) and routes, which the L4 takes
#                   nowhere else
#   rust-tests.txt  the crate's tests with the library and the GPU
#   test_gpu.txt    test_gpu.py
#   pack-python.txt python -m glyd.gpu pack of each model in each layout (on the GPU): the reference
#   pack.txt        glyd-gpu pack, three rounds a model and layout on 16 threads: time, GB/s of bf16, peak RSS, and every
#                   file's sha256 against Python's
#   verify.txt      glyd-gpu verify of each Rust save, three rounds on the CPU (16 threads) and one on the GPU, and
#                   python -m glyd.gpu verify of it
#   examples.txt    the crate's examples on Qwen3-0.6B's tiered save: unpack.rs (a pack, a merged one), linear.rs
#   tiny.txt        tiny.py's checkpoints packed by both in both layouts, every file's sha256; each Rust save verified
#                   by both
#   summary.txt     what each log's steps ended with
#   W=~/rust-rc bash run.sh      (HF_HOME: the five models)
set -u
export W=${W:-~/rust-rc} PY=python T=16 HF_HUB_OFFLINE=1
export TMPDIR=$W/tmp TORCH_EXTENSIONS_DIR=$W/torch_ext PATH=~/.cargo/bin:$PATH CARGO_TARGET_DIR=$W/target
export GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda13.so PYTHONPATH=$W/src/bindings/python BIN=$W/target/release/glyd-gpu
export L=$W/logs O=$W/out
LOCK=~/.glyd-box.lock
mkdir -p $L $O $TMPDIR
touch ~/.glyd-busy
trap 'rm -f ~/.glyd-busy' EXIT
# run LOG STEP [ARGS]: the step (a function below, or a program) under the lock once the GPU is idle, its output
# appended to LOG after the machine's state
run() {
    local log=$1
    shift
    flock $LOCK bash -c 'until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 20; done
        echo "# $(date "+%F %T"): load $(cut -d" " -f1-3 /proc/loadavg); busiest: $(ps -eo pcpu=,comm= --sort=-pcpu | head -n 3 | tr -s " " | tr "\n" ";")"
        "$@"' _ "$@" >> $L/$log 2>&1
}
export NVFLAGS="-O3 -std=c++20 --expt-relaxed-constexpr -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode arch=compute_89,code=sm_89 -Xcompiler -fPIC,-fvisibility=hidden"
b89() {  # TREE: its gpu/glyd_gpu.cu for sm_89 alone, as TREE/lib/libglyd_gpu_cuda13.so (build_lib.sh's flags)
    mkdir -p $1/lib && nvcc $NVFLAGS -isystem $CUDA_HOME/include -c -o $1/lib/glyd_gpu.o $1/gpu/glyd_gpu.cu &&
        nvcc $NVFLAGS -shared -Xlinker --exclude-libs,ALL -cudart static -L$CUDA_HOME/lib -o $1/lib/libglyd_gpu_cuda13.so $1/lib/glyd_gpu.o
}
builds() {
    for side in src main; do  # the GeForce test matching the L4: "L4" for "GeForce" in geforce_ada and gpu_class
        mkdir -p $W/gf/$side/gpu && cp $W/$side/gpu/glyd_gpu.h $W/gf/$side/gpu/ && sed 's/"GeForce")/"L4")/' $W/$side/gpu/glyd_gpu.cu > $W/gf/$side/gpu/glyd_gpu.cu
        echo "gf/$side: $(grep -c '"L4")' $W/gf/$side/gpu/glyd_gpu.cu) of the GeForce tests patched"
    done
    local p=()
    (time bash $W/src/gpu/build_lib.sh $W/lib) > $L/build-release.txt 2>&1 & p+=($!)
    (time b89 $W/main) > $L/build-main.txt 2>&1 & p+=($!)
    (time b89 $W/gf/src) > $L/build-gf-src.txt 2>&1 & p+=($!)
    (time b89 $W/gf/main) > $L/build-gf-main.txt 2>&1 & p+=($!)
    (cd $W/src/glyd-gpu && time cargo build --release --bins --examples 2>&1 | tail -n 2 && cargo test --release --no-run 2>&1 | tail -n 1) > $L/build-crate.txt 2>&1 & p+=($!)
    local e=0
    for i in "${p[@]}"; do wait $i || e=1; done
    for b in release main gf-src gf-main crate; do echo "build-$b: $(grep -m1 real $L/build-$b.txt)"; done
    ls -la $W/lib/libglyd_gpu_cuda13.so $W/main/lib/libglyd_gpu_cuda13.so $W/gf/src/lib/libglyd_gpu_cuda13.so $W/gf/main/lib/libglyd_gpu_cuda13.so $BIN
    echo "exit $e"
}
selftest() { cd $W/src/gpu && $PY glyd_gpu.py; echo "exit $?"; }
xcheck() { cd $W/src/benchmarks/gpu/splitbyte-2026-09-29 && $PY -u xcheck.py $W/main $W/main/lib/libglyd_gpu_cuda13.so; echo "exit $?"; }
xcheck_geforce() {
    cd $W/src/benchmarks/gpu/splitbyte-2026-09-29 && GLYD_GPU_LIB=$W/gf/src/lib/libglyd_gpu_cuda13.so $PY -u xcheck.py $W/main $W/gf/main/lib/libglyd_gpu_cuda13.so
    echo "exit $?"
}
rust_tests() { cd $W/src/glyd-gpu && cargo test --release -- --test-threads 1 2>&1 | grep -E '^test |test result|panicked'; echo "exit ${PIPESTATUS[0]}"; }
test_gpu() { cd $W/src/bindings/python && $PY test_gpu.py 2>&1 | grep -v 'Loading weights\|Writing model'; echo "exit ${PIPESTATUS[0]}"; }
# MODEL LAYOUT: Python's save's files' sha256 (the save itself removed)
pack_py() {
    local n=$(basename $1)
    echo "$1 $2:"
    rm -rf $O/py
    /usr/bin/time -f '%e s wall' $PY -m glyd.gpu pack $1 $O/py --layout $2 2>&1 | grep -v 'Loading weights\|Fetching'
    local e=${PIPESTATUS[0]}
    (cd $O/py && sha256sum *) > $O/py-$n-$2.sha256
    rm -rf $O/py
    echo "exit $e"
}
# MODEL LAYOUT ROUND: glyd-gpu pack, its files' sha256 against Python's
pack_rs() {
    local n=$(basename $1) rs=$O/$(basename $1)-$2
    echo "round $3: $1 $2"
    rm -rf $rs
    /usr/bin/time -f '%e s wall, %M KB peak RSS, %P CPU' $BIN pack $1 $rs --layout $2 --threads $T
    local e=$?
    (cd $rs && sha256sum *) > $O/$n-$2.sha256
    if cmp -s $O/py-$n-$2.sha256 $O/$n-$2.sha256; then
        echo "$n $2: every file byte-identical to Python's ($(wc -l < $O/$n-$2.sha256) files)"
    else
        echo "$n $2: DIFFERS"
        diff $O/py-$n-$2.sha256 $O/$n-$2.sha256
        e=1
    fi
    echo "exit $e"
}
# SAVE HOW [ROUND]: glyd-gpu verify on the CPU (cpu) or the GPU (cuda:0), or python -m glyd.gpu verify (python)
verify() {
    echo "${3:+round $3: }$(basename $1), $2"
    case $2 in
        cpu) /usr/bin/time -f '%e s wall, %M KB peak RSS' $BIN verify $1 --threads $T ;;
        cuda:0) /usr/bin/time -f '%e s wall, %M KB peak RSS' $BIN verify $1 --device cuda:0 ;;
        python) /usr/bin/time -f '%e s wall' $PY -m glyd.gpu verify $1 ;;
    esac
    echo "exit $?"
}
examples() {
    local snap=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ | head -1)
    cd $W/src/glyd-gpu && cargo run -q --release --example unpack -- $O/Qwen3-0.6B-mma $snap &&
        cargo run -q --release --example unpack -- $O/Qwen3-0.6B-mma $snap model.layers.0.self_attn.q_proj &&
        cargo run -q --release --example linear -- $O/Qwen3-0.6B-mma model.layers.0.mlp.gate_proj
    echo "exit $?"
}
tiny_make() { rm -rf $W/tiny && $PY $W/tiny.py $W/tiny; echo "exit $?"; }
# CHECKPOINT LAYOUT: both packs, every file's sha256; the Rust save verified by both
tiny_one() {
    local n=$(basename $1) e=0
    rm -rf $O/tpy $O/trs
    $PY -m glyd.gpu pack $1 $O/tpy --layout $2 2>&1 | grep packed || e=1
    $BIN pack $1 $O/trs --layout $2 --threads $T 2>&1 | grep -v 'GB of bf16' || e=1
    (cd $O/tpy && sha256sum *) > $O/tpy.sha256
    (cd $O/trs && sha256sum *) > $O/trs.sha256
    if cmp -s $O/tpy.sha256 $O/trs.sha256; then echo "$n $2: every file byte-identical to Python's ($(wc -l < $O/trs.sha256) files)"; else echo "$n $2: DIFFERS"; diff $O/tpy.sha256 $O/trs.sha256; e=1; fi
    $BIN verify $O/trs --threads 2 || e=1
    $PY -m glyd.gpu verify $O/trs 2>&1 | grep -v 'Loading weights' || e=1
    echo "exit $e"
}
export -f b89 builds selftest xcheck xcheck_geforce rust_tests test_gpu pack_py pack_rs verify examples tiny_make tiny_one

echo "# $(date '+%F %T') $(cat $W/src/COMMIT): $(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,power.limit --format=csv,noheader); $(nproc) CPUs, $(lscpu | sed -n 's/^Model name: *//p'); $(free -g | awk '/Mem:/{print $2}') GB" > $L/run.txt
run build.txt builds
run selftest.txt selftest
run xcheck.txt xcheck
run xcheck-geforce.txt xcheck_geforce
run rust-tests.txt rust_tests
run test_gpu.txt test_gpu
for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B ibm-granite/granite-3.1-3b-a800m-instruct Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B; do
    n=$(basename $m)
    for layout in mma mma12; do
        run pack-python.txt pack_py $m $layout
        for round in 1 2 3; do run pack.txt pack_rs $m $layout $round; done
        for round in 1 2 3; do run verify.txt verify $O/$n-$layout cpu $round; done
        run verify.txt verify $O/$n-$layout cuda:0
        run verify.txt verify $O/$n-$layout python
        [ $n = Qwen3-0.6B ] && [ $layout = mma ] || rm -rf $O/$n-$layout
    done
done
run examples.txt examples
rm -rf $O/Qwen3-0.6B-mma
run tiny.txt tiny_make
for d in $W/tiny/*/; do for layout in mma mma12; do run tiny.txt tiny_one $d $layout; done; done
{ for f in $L/*.txt; do
    [ $f = $L/summary.txt ] || [ $f = $L/run.txt ] && continue
    echo "$(basename $f): $(grep -c '^exit 0$' $f) steps exit 0, $(grep -c '^exit [^0]' $f) not; $(grep -c 'byte-identical' $f) byte-identical, $(grep -c 'DIFFERS' $f) differ"
  done
  echo "all done $(date '+%F %T')"; } > $L/summary.txt
cat $L/summary.txt
