#!/usr/bin/env bash
# The CPU half of the sparse experts offload (src/cpu_experts.rs; Linear GLY-130) on a pod's processor, the GPU idle (rp_run.sh contract: results in ~/results, ~/results/DONE at the end). Uploaded beside it: src.tar (git archive, prefix src/).
# The engine is built without nvcc (no kernels: the CPU code needs none), then, each a step in steps.txt with its log:
#   cpu       what the processor is: the model, its flags (avx2, fma, avx512f ...), the threads, the memory, the cgroup's CPU cap
#   unit      `cargo test --release -p glyd-engine cpu_experts`: every kernel this processor runs (scalar, AVX2, AVX-512) against the scalar one bit for bit, and against the f64 pair, at 4, 5 and 6 bits
#   bench4    `glyd-engine cpuexperts`: Nemotron 3 Super's experts at four bits, the best kernel on all the threads: streaming read GB/s, pairs a second and GB/s read, a decode token's ms for 100, 230 and 400 missed experts
#   avx2      the same with --kernel avx2, where the best kernel is AVX-512 (the two compared on one processor)
#   threads   the same at 1, 2, 4, 8, 16 and 32 threads (those the pod has) and at the count the program picks (the cgroup's CPU cap), for 230 missed experts: how the work scales
#   bench5, bench6   five and six bits
# ~/results: steps.txt, summary.txt (made last), cpu.txt, one .log a step. Optional environment: JOB_MAX_MIN (default 45; a step that would not end in time is skipped).
#   NAME=cpu-experts MAX_MIN=60 UPLOAD="$W/src.tar" bash tools/runpod/rp_class.sh 4090 tools/cards/cpu_experts_job.sh        # $0.89/h; about 30 minutes
set -u
R=~/results; mkdir -p $R ~/v; cd ~/v; tar -xf ~/src.tar
P=$R/steps.txt; : > $P
T0=$(date +%s); DEADLINE=$((T0 + ${JOB_MAX_MIN:-45} * 60))
# a step: its name, the seconds it may take at most, the command (its output in NAME.log, its time and exit code in steps.txt); skipped where too little time is left
step() {
    local n=$1 secs=$2 left=$((DEADLINE - $(date +%s))); shift 2
    [ "$left" -gt 60 ] || { echo "$n: skipped, $left s left" >> $P; return 1; }
    [ "$secs" -le "$left" ] || secs=$left
    local t=$(date +%s); timeout "$secs" "$@" > $R/$n.log 2>&1; local rc=$?
    echo "$n: rc=$rc $(( $(date +%s) - t )) s (job minute $(( ($(date +%s) - T0) / 60 )))" >> $P
    return $rc
}
trap 'touch $R/DONE' EXIT
export RUST_BACKTRACE=1
# the CUDA toolkit out of the way: with no nvcc the engine builds in minutes, without its kernels
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v cuda | paste -sd: -); unset CUDA_HOME CUDA_PATH NVCC
{
    echo "--- lscpu"; lscpu
    echo "--- flags"; grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx|avx2|fma|avx512[a-z_0-9]*|bmi2|f16c|sse4_1|ssse3|amx[a-z_0-9]*)$' | sort -u | tr '\n' ' '; echo
    echo "--- threads"; nproc
    echo "--- memory"; free -g
    echo "--- cgroup cpu.max"; cat /sys/fs/cgroup/cpu.max 2> /dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2> /dev/null
    echo "--- numa"; (numactl --hardware 2> /dev/null || lscpu | grep -i numa)
} > $R/cpu.txt 2>&1
step rustup 300 bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1'
. ~/.cargo/env
cd ~/v/src
step unit 1800 cargo test --release -p glyd-engine cpu_experts -- --nocapture --test-threads=1
step build 1500 cargo build --release -p glyd-engine
E=~/v/src/target/release/glyd-engine
FLAGS=$(cat $R/cpu.txt)
N=$(nproc)
step bench4 900 $E cpuexperts --bits 4
case "$FLAGS" in *avx512f*) step avx2 900 $E cpuexperts --bits 4 --kernel avx2;; esac
step threads_all 600 $E cpuexperts --bits 4 --misses 230 --reps 3 --tokens 20
for t in 1 2 4 8 16 32; do
    [ "$t" -le "$N" ] && step threads$t 600 $E cpuexperts --bits 4 --threads $t --misses 230 --reps 3 --tokens 20
done
step bench5 900 $E cpuexperts --bits 5 --misses 230
step bench6 900 $E cpuexperts --bits 6 --misses 230
{
    echo "CPU half of the sparse experts (GLY-130) on $(grep -m1 'Model name' $R/cpu.txt | sed 's/.*: *//'), $N threads; tree $(gzip -dc ~/src.tar 2> /dev/null | git get-tar-commit-id 2> /dev/null)"
    echo "--- steps"; cat $P
    echo "--- flags"; sed -n '/^--- flags/,/^--- threads/p' $R/cpu.txt | sed '1d;$d'
    echo "--- unit tests (every kernel against the scalar one and f64)"; grep -E '^(test |running|test result|bits |worst error|scalar:|neon:|avx2:|avx512:)' $R/unit.log 2> /dev/null
    for s in bench4 avx2 threads_all threads1 threads2 threads4 threads8 threads16 threads32 bench5 bench6; do
        [ -f $R/$s.log ] && { echo "--- $s"; grep -E '^(.* threads, kernel|experts of|streaming read|check:|one call|a decode token)' $R/$s.log; }
    done
} > $R/summary.txt 2>&1
