#!/bin/bash
# one hold of the box's lock: the tiled kernel's tile of 256 tokens (GLYD_GPU_BIG_TM) and its m-tile skipping, on the L4 (the A100's code), then every path's checks
N=${1:-tm}; O=$HOME/dev/out/$N; mkdir -p $O
R=$HOME/dev/devrun.sh
f() { grep -E "^T=|Error|FAILED|Traceback" | cut -c1-250; }
V="t128:GLYD_GPU_MOE_BIG=0;t128ns:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_SKIP=0;t256:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=256;t256ns:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=256,GLYD_GPU_BIG_SKIP=0"
echo "== $(date -u +%T) compute-bound: Qwen3-30B-A3B's shapes (hidden 2048, intermediate 768, 8 of 128 experts a token), 6 experts so that the weights stay in the L2 cache (pairs an expert = tokens / 3 x 2)"
bash $R $N moe_layer.py --synthetic 6,2,2048,768 --layout mma12 --tokens "192 384 768 1536 3072" --bf16 --variants "$V" --json $O/e6.json 2>&1 | f
echo "== $(date -u +%T) the same with skewed routing (--skew 1.5: some experts have many pairs, some few)"
bash $R $N moe_layer.py --synthetic 6,2,2048,768 --layout mma12 --skew 1.5 --tokens "768 1536 3072" --bf16 --variants "$V" --json $O/e6skew.json 2>&1 | f
echo "== $(date -u +%T) the real layer (128 experts, the weights from memory), the tile of 128 and of 256, skipping"
bash $R $N moe_layer.py --hf Qwen/Qwen3-30B-A3B --layer 0 --layout mma12 --tokens "384 1152 2048" --bf16 --variants "$V" --json $O/real.json 2>&1 | f
echo "== $(date -u +%T) ncu: gate+up and down of the tile of 128 and of 256, compute-bound (6 experts, 1,536 tokens)"
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu | sort -V | tail -1)
D=$HOME/dev/$N
for tm in 128 256; do
  sudo -n env "PATH=$PATH" "HOME=$HOME" "PYTHONPATH=$D/src/glyd-gpu-py" PYTHONDONTWRITEBYTECODE=1 "GLYD_GPU_LIB=$D/lib/libglyd_gpu_cuda13.so" "HF_HOME=$HOME/hf" HF_HUB_OFFLINE=1 REPS=1 GLYD_GPU_MOE_BIG=0 GLYD_GPU_BIG_TM=$tm \
    bash -c "cd $D/src/gpu && timeout 300 $NCU --set full -k regex:mma_gemm_big_kernel -c 2 -f -o $O/e6-tm$tm $HOME/v027/venv/bin/python -u moe_layer.py --synthetic 6,2,2048,768 --layout mma12 --tokens 1536 --check --no-time --variants new:" > $O/ncu-tm$tm.txt 2>&1
  sudo -n chown "$(id -u):$(id -g)" $O/e6-tm$tm.ncu-rep 2> /dev/null
  $NCU -i $O/e6-tm$tm.ncu-rep --page raw --csv > $O/raw-tm$tm.csv 2> /dev/null
  echo "-- tile of $tm"; python3 $HOME/dev/ncu_summary.py $O/raw-tm$tm.csv 2>&1 | head -44
done
echo "== $(date -u +%T) checks: the real layer, every path"
bash $R $N moe_layer.py --hf Qwen/Qwen3-30B-A3B --layer 0 --layout mma12 --tokens "17 24 48 232 384 768 1152 2048 4096" --check --no-time --variants "new:;noskip:GLYD_GPU_BIG_SKIP=0;a100:GLYD_GPU_MOE_ROWS=17,GLYD_GPU_MOE_BIG_GU=84,GLYD_GPU_MOE_BIG_DN=48;dn192:GLYD_GPU_MOE_BIG_DN=192;gu64:GLYD_GPU_MOE_BIG_GU=64;tiled:GLYD_GPU_MOE_BIG=0;tm256:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=256;tm256a:GLYD_GPU_MOE_BIG_GU=84,GLYD_GPU_MOE_BIG_DN=48,GLYD_GPU_BIG_TM=256" 2>&1 | grep -E "checked|FAILED|Error" | cut -c1-200 > $O/check-real.txt; grep -c checked $O/check-real.txt; grep -E "FAILED|Error" $O/check-real.txt | head -5
echo "== $(date -u +%T) checks: skewed routing, exceptions by the dozen, an odd small layer"
bash $R $N moe_layer.py --hf Qwen/Qwen3-30B-A3B --layer 0 --layout mma12 --skew 1.5 --tokens "64 232 1152 4096" --check --no-time --variants "new:;tiled:GLYD_GPU_MOE_BIG=0;noskip:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_SKIP=0;tm256:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=256" 2>&1 | grep -E "checked|FAILED|Error" | cut -c1-200 > $O/check-skew.txt; grep -c checked $O/check-skew.txt; grep -E "FAILED|Error" $O/check-skew.txt | head -3
bash $R $N moe_layer.py --synthetic 64,8,2048,768 --wild 0.02 --layout mma12 --tokens "17 232 1152 4096" --check --no-time --variants "new:;tiled:GLYD_GPU_MOE_BIG=0;tm256:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=256" 2>&1 | grep -E "checked|FAILED|Error" | cut -c1-200 > $O/check-wild.txt; grep -c checked $O/check-wild.txt; grep -E "FAILED|Error" $O/check-wild.txt | head -3
bash $R $N moe_layer.py --synthetic 9,3,256,192 --layout mma12 --tokens "1 5 17 40 100 333 1000" --check --no-time --variants "new:;tiled:GLYD_GPU_MOE_BIG=0;tm256:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=256;a100:GLYD_GPU_MOE_ROWS=17,GLYD_GPU_MOE_BIG_GU=84,GLYD_GPU_MOE_BIG_DN=48" 2>&1 | grep -E "checked|FAILED|Error" | cut -c1-200 > $O/check-odd.txt; grep -c checked $O/check-odd.txt; grep -E "FAILED|Error" $O/check-odd.txt | head -3
echo "== $(date -u +%T) done"
