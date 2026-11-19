# 0.29.4's GPU check (rp_run.sh contract: results in ~/results, ~/results/DONE at the end). Branch int5-swift from src.tar:
# library + engine built, crate tests, examples/int5 and int6, Qwen3.8-27B: bench i5/i6, run :swift / :kestrel / --lossy, :wren refused.
set -u
R=~/results; mkdir -p $R ~/v; cd ~/v; tar -xf ~/src.tar
P=$R/steps.txt; : > $P
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; echo "$n: rc=$? $(( $(date +%s) - t )) s" >> $P; }
export PATH=/usr/local/cuda/bin:$PATH HF_XET_HIGH_PERFORMANCE=1
ls -d /usr/local/cuda* >> $P 2>&1; nvcc --version | tail -2 >> $P 2>&1
step rustup bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'
. ~/.cargo/env
step hfvenv bash -c 'python3 -m venv ~/v/venv && ~/v/venv/bin/pip install -q "huggingface_hub>=1.5,<2" hf_xet'
( ~/v/venv/bin/hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --local-dir ~/v/bf16 > $R/download.log 2>&1; echo "download: rc=$?" >> $P ) &
step lib bash -c 'cd ~/v/src && bash gpu/build_lib.sh ~/v/lib'
LIB=$(ls ~/v/lib/libglyd_gpu_cuda*.so | head -1)
step build bash -c 'cd ~/v/src && cargo build --release -p glyd-engine && cargo build --release -p glyd-gpu --example int5 --example int6 && cargo test --release -p glyd-gpu'
step int5-check env GLYD_GPU_LIB=$LIB ~/v/src/target/release/examples/int5
step int6-check env GLYD_GPU_LIB=$LIB ~/v/src/target/release/examples/int6
wait
E=~/v/src/target/release/glyd-engine
step bench-i5 env GLYD_GPU_LIB=$LIB $E bench ~/v/bf16 --layout i5 --prompt 1024 --new 256
step bench-i6 env GLYD_GPU_LIB=$LIB $E bench ~/v/bf16 --layout i6 --prompt 1024 --new 256
step run-swift env GLYD_GPU_LIB=$LIB $E run ~/v/bf16:swift -p "In two sentences: what is the capital of Australia, and why was it chosen?"
step run-kestrel env GLYD_GPU_LIB=$LIB $E run ~/v/bf16:kestrel --no-think -p "Name three primes above 100."
step run-lossy env GLYD_GPU_LIB=$LIB $E run ~/v/bf16 --lossy --no-think -p "What is 17 times 23?"
step tag-wren env GLYD_GPU_LIB=$LIB $E run ~/v/bf16:wren -p "hi"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > $R/gpu-final.txt
touch $R/DONE
