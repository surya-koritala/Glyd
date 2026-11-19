#!/bin/bash
# one hold of the box's lock: the baseline tree's big-step products on the L4, as the A100 runs them (mma12): the real layer and a compute-bound proxy, then ncu on the tiled kernel
N=${1:-base}; O=$HOME/dev/out/$N; mkdir -p $O
R=$HOME/dev/devrun.sh
echo "== $(date -u +%T) real layer (Qwen3-30B-A3B layer 0, 128 experts)"
bash $R $N moe_layer.py --hf Qwen/Qwen3-30B-A3B --layer 0 --layout mma12 --tokens "384 768 1152 2048 4096" --bf16 --variants "new:;rows17:GLYD_GPU_MOE_ROWS=17;big96:GLYD_GPU_MOE_BIG=96;big192:GLYD_GPU_MOE_BIG=192" --json $O/real.json 2>&1 | grep -E "^T=|Error|FAILED" | cut -c1-250
echo "== $(date -u +%T) proxy (128 experts, hidden 512, intermediate 192: the weights in the L2 cache, the products bound by their own work)"
bash $R $N moe_layer.py --synthetic 128,8,512,192 --layout mma12 --tokens "384 768 1152 2048 4096 8192" --bf16 --variants "new:;big192:GLYD_GPU_MOE_BIG=192;rows17:GLYD_GPU_MOE_ROWS=17" --json $O/proxy.json 2>&1 | grep -E "^T=|Error|FAILED" | cut -c1-250
echo "== $(date -u +%T) ncu: the tiled kernel (gate+up, then down) on the real layer at 2,048 tokens"
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu | sort -V | tail -1)
D=$HOME/dev/$N
sudo -n env "PATH=$PATH" "HOME=$HOME" "PYTHONPATH=$D/src/glyd-gpu-py" PYTHONDONTWRITEBYTECODE=1 "GLYD_GPU_LIB=$D/lib/libglyd_gpu_cuda13.so" "HF_HOME=$HOME/hf" HF_HUB_OFFLINE=1 \
  bash -c "cd $D/src/gpu && timeout 300 $NCU --set full -k regex:mma_gemm_big_kernel -c 2 -f -o $O/big-2048 $HOME/v027/venv/bin/python -u moe_layer.py --hf Qwen/Qwen3-30B-A3B --layer 0 --layout mma12 --tokens 2048 --check --no-time --variants new:" > $O/ncu-2048.txt 2>&1
sudo -n chown "$(id -u):$(id -g)" $O/big-2048.ncu-rep 2> /dev/null
$NCU -i $O/big-2048.ncu-rep --page raw --csv > $O/raw-2048.csv 2> /dev/null
python3 $HOME/dev/ncu_summary.py $O/raw-2048.csv 2>&1 | head -60
echo "== $(date -u +%T) done"
