# Six-bit block codes on this pod: the library and the engine built from the branch, the crate's tests, examples/int6 (GPU decode and
# products against the packer and the tiered layout, bit for bit), then Qwen3.8-27B with --layout i6 against --layout a: speed, a
# profile, a generation. Steps to ~/i/steps.txt.
set -u
cd ~/i; P=~/i/steps.txt; : > $P
step() { local n=$1; shift; local t=$(date +%s); "$@" > ~/i/$n.log 2>&1; echo "$n: rc=$? $(( $(date +%s) - t )) s" >> $P; }
export PATH=/usr/local/cuda/bin:$PATH
step rustup bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'
. ~/.cargo/env
step hfvenv bash -c 'python3 -m venv ~/i/venv && ~/i/venv/bin/pip install -q "huggingface_hub>=1.5,<2" hf_xet'
( HF_XET_HIGH_PERFORMANCE=1 ~/i/venv/bin/hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --local-dir ~/i/bf16 > ~/i/download.log 2>&1; echo "download: rc=$?" >> $P ) &
step lib bash -c 'cd ~/i/src && GLYD_GPU_NVCC_THREADS=0 bash gpu/build_lib.sh ~/i/lib'
LIB=$(ls ~/i/lib/libglyd_gpu_cuda*.so | head -1)
step crate-tests bash -c 'cd ~/i/src && cargo test --release -p glyd-gpu int6'
step int6-check env GLYD_GPU_LIB=$LIB bash -c 'cd ~/i/src && cargo run --release -p glyd-gpu --example int6'
step engine bash -c 'cd ~/i/src && cargo build --release -p glyd-engine'
wait
E=~/i/src/target/release/glyd-engine
step bench-i6 env GLYD_GPU_LIB=$LIB $E bench ~/i/bf16 --layout i6 --prompt 1024 --new 256
step bench-a env GLYD_GPU_LIB=$LIB $E bench ~/i/bf16 --layout a --prompt 1024 --new 256
step profile-i6 env GLYD_GPU_LIB=$LIB $E bench ~/i/bf16 --layout i6 --prompt 1024 --new 64 --profile
step generate-i6 env GLYD_GPU_LIB=$LIB $E generate ~/i/bf16 "The three primary colours of light are" --new 32 --layout i6
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > ~/i/gpu.txt
echo done >> $P
