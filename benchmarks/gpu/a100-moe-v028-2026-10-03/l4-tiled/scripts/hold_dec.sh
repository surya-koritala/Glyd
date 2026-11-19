#!/bin/bash
# one hold's tail: ncu of the decode kernel (gate+up) and the rows kernel (down) on the real layer at 232 tokens, the epilogue-slimmed tree
O=$HOME/dev/out/epi; mkdir -p $O
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu | sort -V | tail -1)
D=$HOME/dev/epil2
sudo -n env "PATH=$PATH" "HOME=$HOME" "PYTHONPATH=$D/src/glyd-gpu-py" PYTHONDONTWRITEBYTECODE=1 "GLYD_GPU_LIB=$D/lib/libglyd_gpu_cuda13.so" "HF_HOME=$HOME/hf" HF_HUB_OFFLINE=1 REPS=1 GLYD_GPU_MOE_ROWS=17 \
  bash -c "cd $D/src/gpu && timeout 300 $NCU --set full -k regex:mma_moe_ -c 2 -f -o $O/dec232 $HOME/v027/venv/bin/python -u moe_layer.py --hf Qwen/Qwen3-30B-A3B --layer 0 --layout mma12 --tokens 232 --check --no-time --variants new:" > $O/ncu-dec232.txt 2>&1
sudo -n chown "$(id -u):$(id -g)" $O/dec232.ncu-rep 2> /dev/null
$NCU -i $O/dec232.ncu-rep --page raw --csv > $O/raw-dec232.csv 2> /dev/null
python3 $HOME/dev/ncu_summary.py $O/raw-dec232.csv 2>&1 | grep -E "launch|duration|tensor bf16|hmma|issue slots|warp instr|local|warps active|stall|DRAM thr|registers" | cut -c1-150
echo "== $(date -u +%T) done"
