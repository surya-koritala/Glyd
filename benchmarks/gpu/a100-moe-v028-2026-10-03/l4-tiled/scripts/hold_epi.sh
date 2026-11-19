#!/bin/bash
# one hold of the box's lock: the tiled kernel with its epilogues shrunk (epil2) against the tree before them (tm3), the same settings, the two in turn twice; ncu of the products; the checks
O=$HOME/dev/out/epi; mkdir -p $O
R=$HOME/dev/devrun.sh
f() { grep -E "^T=|Error|FAILED|Traceback" | cut -c1-250; }
V="t128:GLYD_GPU_MOE_BIG=0;t128ns:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_SKIP=0;t256:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=256"
for round in 1 2; do
  for N in tm3 epil2; do
    echo "== $(date -u +%T) $N, round $round: the compute-bound layer (Qwen3-30B-A3B's shapes, 6 experts in the L2 cache)"
    bash $R $N moe_layer.py --synthetic 6,2,2048,768 --layout mma12 --tokens "384 768 1536 3072" --variants "$V" --json $O/e6-$N-$round.json 2>&1 | f
    echo "== $(date -u +%T) $N, round $round: the real layer (128 experts from memory), the built-in settings and the tiled kernel always"
    bash $R $N moe_layer.py --hf Qwen/Qwen3-30B-A3B --layer 0 --layout mma12 --tokens "384 1152 2048" --variants "new:;t128:GLYD_GPU_MOE_BIG=0" --json $O/real-$N-$round.json 2>&1 | f
  done
done
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu | sort -V | tail -1)
D=$HOME/dev/epil2
prof() {  # NAME REGEX ARGS...: ncu of the first 2 launches that match
  local name=$1 k=$2; shift 2
  sudo -n env "PATH=$PATH" "HOME=$HOME" "PYTHONPATH=$D/src/glyd-gpu-py" PYTHONDONTWRITEBYTECODE=1 "GLYD_GPU_LIB=$D/lib/libglyd_gpu_cuda13.so" "HF_HOME=$HOME/hf" HF_HUB_OFFLINE=1 REPS=1 \
    bash -c "cd $D/src/gpu && timeout 300 $NCU --set full -k regex:$k -c 2 -f -o $O/$name $HOME/v027/venv/bin/python -u moe_layer.py $* --check --no-time --variants new:" > $O/ncu-$name.txt 2>&1
  sudo -n chown "$(id -u):$(id -g)" $O/$name.ncu-rep 2> /dev/null
  $NCU -i $O/$name.ncu-rep --page raw --csv > $O/raw-$name.csv 2> /dev/null
  echo "-- $name"; python3 $HOME/dev/ncu_summary.py $O/raw-$name.csv 2>&1 | grep -E "launch|duration|tensor bf16|hmma|issue slots|warp instr|local|warps active|stall|DRAM thr" | cut -c1-150
}
echo "== $(date -u +%T) ncu: the tiled kernel, gate+up and down, compute-bound (6 experts, 1,536 tokens) and the real layer at 2,048"
GLYD_GPU_MOE_BIG=0 prof e6 mma_gemm_big_kernel --synthetic 6,2,2048,768 --layout mma12 --tokens 1536
GLYD_GPU_MOE_BIG=0 prof real2048 mma_gemm_big_kernel --hf Qwen/Qwen3-30B-A3B --layer 0 --layout mma12 --tokens 2048
echo "== $(date -u +%T) ncu: the decode kernel (gate+up) and the rows kernel (down) on the real layer at 232 tokens"
GLYD_GPU_MOE_ROWS=17 prof dec232 'mma_moe_(rows_)?kernel' --hf Qwen/Qwen3-30B-A3B --layer 0 --layout mma12 --tokens 232
echo "== $(date -u +%T) checks: the real layer, every path"
bash $R epil2 moe_layer.py --hf Qwen/Qwen3-30B-A3B --layer 0 --layout mma12 --tokens "17 24 48 232 384 768 1152 2048 4096" --check --no-time --variants "new:;noskip:GLYD_GPU_BIG_SKIP=0;a100:GLYD_GPU_MOE_ROWS=17,GLYD_GPU_MOE_BIG_GU=84,GLYD_GPU_MOE_BIG_DN=48;dn192:GLYD_GPU_MOE_BIG_DN=192;gu64:GLYD_GPU_MOE_BIG_GU=64;tiled:GLYD_GPU_MOE_BIG=0;tm256:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=256" 2>&1 | grep -E "checked|FAILED|Error" | cut -c1-200 > $O/check-real.txt; grep -c checked $O/check-real.txt; grep -E "FAILED|Error" $O/check-real.txt | head -5
echo "== $(date -u +%T) checks: skewed routing, exceptions by the dozen, an odd small layer, gelu"
bash $R epil2 moe_layer.py --hf Qwen/Qwen3-30B-A3B --layer 0 --layout mma12 --skew 1.5 --tokens "64 232 1152 4096" --check --no-time --variants "new:;tiled:GLYD_GPU_MOE_BIG=0;noskip:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_SKIP=0;tm256:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=256" 2>&1 | grep -E "checked|FAILED|Error" | cut -c1-200 > $O/check-skew.txt; grep -c checked $O/check-skew.txt; grep -E "FAILED|Error" $O/check-skew.txt | head -3
bash $R epil2 moe_layer.py --synthetic 64,8,2048,768 --wild 0.02 --layout mma12 --tokens "17 232 1152 4096" --check --no-time --variants "new:;tiled:GLYD_GPU_MOE_BIG=0;tm256:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=256" 2>&1 | grep -E "checked|FAILED|Error" | cut -c1-200 > $O/check-wild.txt; grep -c checked $O/check-wild.txt; grep -E "FAILED|Error" $O/check-wild.txt | head -3
bash $R epil2 moe_layer.py --synthetic 9,3,256,192 --layout mma12 --tokens "1 5 17 40 100 333 1000" --check --no-time --variants "new:;tiled:GLYD_GPU_MOE_BIG=0;tm256:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=256;a100:GLYD_GPU_MOE_ROWS=17,GLYD_GPU_MOE_BIG_GU=84,GLYD_GPU_MOE_BIG_DN=48" 2>&1 | grep -E "checked|FAILED|Error" | cut -c1-200 > $O/check-odd.txt; grep -c checked $O/check-odd.txt; grep -E "FAILED|Error" $O/check-odd.txt | head -3
bash $R epil2 moe_layer.py --synthetic 8,2,2048,768 --gelu --layout mma12 --tokens "1 17 64 300 1200" --check --no-time --variants "new:;tiled:GLYD_GPU_MOE_BIG=0;tm256:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=256" 2>&1 | grep -E "checked|FAILED|Error" | cut -c1-200 > $O/check-gelu.txt; grep -c checked $O/check-gelu.txt; grep -E "FAILED|Error" $O/check-gelu.txt | head -3
echo "== $(date -u +%T) done"
