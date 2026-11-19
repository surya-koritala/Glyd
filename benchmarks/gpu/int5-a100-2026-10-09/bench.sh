# After the fresh-install check: Qwen3.8-27B with --layout i5 and i6 on this A100, its checkpoint from that check download.
set -u
P=~/i5/steps.txt
step() { local n=$1; shift; local t=$(date +%s); "$@" > ~/i5/$n.log 2>&1; echo "$n: rc=$? $(( $(date +%s) - t )) s" >> $P; }
until grep -q "^done" /root/fresh3/results.txt 2>/dev/null; do sleep 15; done
M=$(ls -d /root/fresh3/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ | head -1)
LIB=$(ls ~/i5/lib/libglyd_gpu_cuda*.so | head -1); E=~/i5/src/target/release/glyd-engine
step bench-i5 env GLYD_GPU_LIB=$LIB $E bench $M --layout i5 --prompt 1024 --new 256
step bench-i6 env GLYD_GPU_LIB=$LIB $E bench $M --layout i6 --prompt 1024 --new 256
step generate-i5 env GLYD_GPU_LIB=$LIB $E generate $M "The three primary colours of light are" --new 32 --layout i5
echo done2 >> $P
