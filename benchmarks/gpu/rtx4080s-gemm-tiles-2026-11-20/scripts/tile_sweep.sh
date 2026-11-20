cd ~/glyd-scratch/ttft; exec 9> ~/glyd-scratch/gpu.lock; flock 9
while docker ps --format "{{.Image}}" | grep -q nvidia/cuda; do sleep 5; done
export HF_HOME=$HOME/glyd-demo/hf HF_HUB_OFFLINE=1 LD_LIBRARY_PATH=$PWD/out GLYD_GPU_LIB=$PWD/out/libglyd_gpu_cuda13.so GLYD_ENGINE_NO_PIPE=1
for p in 300 600 1000 1500 1800 2048 4000; do printf "prompt %s:" $p; for t in gemm3_256x128_s3 gemm3_128x128_s4 gemm3_128x128_b64_s3 gemm3_128x256_s3; do r=$(GLYD_ENGINE_TILE=$t ./out/glyd-engine-new bench Qwen/Qwen3-8B --prompt $p --new 2 --ctx 8192 --chunk 2048 --reps 3 2>&1 | grep -E "^prefill ms" | sed -E 's/.*: [0-9.]+, [0-9.]+, ([0-9.]+)/\1/'); printf " %s=%s" ${t#gemm3_} $r; done; echo; done
