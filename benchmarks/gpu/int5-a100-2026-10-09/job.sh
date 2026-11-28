# Int5 (branch int5-linnet) on this A100: the library and the engine built, examples/int5 (the GPU's decode and products against the
# packer), then, once the fresh-install check is done, Qwen3.8-27B with --layout i5 and i6 (its checkpoint from that check's download).
set -u
cd ~/i5; P=~/i5/steps.txt; : > $P
step() { local n=$1; shift; local t=$(date +%s); "$@" > ~/i5/$n.log 2>&1; echo "$n: rc=$? $(( $(date +%s) - t )) s" >> $P; }
export PATH=/usr/local/cuda-12.8/bin:$PATH
step rustup bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'
. ~/.cargo/env
step lib bash -c 'cd ~/i5/src && bash gpu/build_lib.sh ~/i5/lib'
LIB=$(ls ~/i5/lib/libglyd_gpu_cuda*.so | head -1)
step build bash -c 'cd ~/i5/src && cargo build --release -p glyd-engine && cargo build --release -p glyd-gpu --example int5 && cargo test --release -p glyd-gpu int5'
step int5-check env GLYD_GPU_LIB=$LIB ~/i5/src/target/release/examples/int5
until grep -q '^done' /root/fresh3/results.txt 2>/dev/null; do sleep 15; done
M=$(ls -d /root/fresh3/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ | head -1)
E=~/i5/src/target/release/glyd-engine
step bench-i5 env GLYD_GPU_LIB=$LIB $E bench $M --layout i5 --prompt 1024 --new 256
step bench-i6 env GLYD_GPU_LIB=$LIB $E bench $M --layout i6 --prompt 1024 --new 256
step generate-i5 env GLYD_GPU_LIB=$LIB $E generate $M "The three primary colours of light are" --new 32 --layout i5
echo done >> $P
