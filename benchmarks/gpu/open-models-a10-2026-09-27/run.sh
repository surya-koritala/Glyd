#!/usr/bin/env bash
# On the instance: every queued model's Linear weights packed in both layouts and unpacked bit for bit
# (gpu/sizes.py), the class transformers loads it as (no weights read), and bf16 against Glyd for the
# two that fit this GPU in bf16. Models download one ahead and are deleted after.
set -x
source ~/gpuenv/cuda.sh
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=$HOME; R=$W/results; mkdir -p $R $W/models; cd $W/glyd/gpu
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"; df -h $W | tail -1; } > $R/machine.txt 2>&1
python -c "import glyd_gpu" > $R/build.txt 2>&1 || { echo BUILD FAILED > $R/FAILED; touch $R/DONE; exit 1; }
[ -f $W/enwik8 ] || { curl -sL http://mattmahoney.net/dc/enwik8.zip -o $W/enwik8.zip && python -c "import zipfile; zipfile.ZipFile('$W/enwik8.zip').extractall('$W')"; }
M="unsloth/Llama-3.2-3B-Instruct Qwen/Qwen3-4B-Instruct-2507 unsloth/gemma-3-12b-it google/gemma-4-26B-A4B-it Qwen/Qwen3.8-27B meta-models/Muse-Glimmer-30B Qwen/Qwen3-Next-80B-A3B-Instruct zai-org/GLM-4.5-Air unsloth/Llama-4-Scout-17B-16E-Instruct"
dl() { local n=${1#*/}; [ -f $W/models/$n/config.json ] || timeout 3600 hf download $1 --local-dir $W/models/$n > $R/download-$n.txt 2>&1; }
set -- $M
dl $1 &
for m in $M; do
  n=${m#*/}; d=$W/models/$n
  wait
  shift; [ $# -gt 0 ] && { dl $1 & }
  echo "$(date -u +%H:%M:%S) $n downloaded: $(du -sh $d | cut -f1)" >> $R/progress.txt
  timeout 300 python - "$d" >> $R/loaders.txt 2>&1 <<'PY'
import sys, torch, transformers as t
d = sys.argv[1]; c = t.AutoConfig.from_pretrained(d)
for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
    try:
        with torch.device("meta"):
            m = getattr(t, name).from_config(c)
        print(d.split("/")[-1], name, "->", type(m).__name__)
        break
    except Exception as e:
        print(d.split("/")[-1], name, "fails:", type(e).__name__, str(e)[:160])
PY
  timeout 1800 python sizes.py $d >> $R/sizes.txt 2>> $R/sizes-err.txt
  case $n in Llama-3.2-3B-Instruct|Qwen3-4B-Instruct-2507)
    timeout 1800 python e2e.py $d --format auto --fused --baseline --merge --batch 1,8,32 --tokens 64 --ppl $W/enwik8 --mmlu 300 --smi $R/smi-$n > $R/quality-$n.txt 2>&1;;
  esac
  echo "$(date -u +%H:%M:%S) $n done" >> $R/progress.txt
  rm -rf $d
done
touch $R/DONE
