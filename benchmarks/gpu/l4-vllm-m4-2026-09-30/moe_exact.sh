#!/bin/bash
# Exact mode's experts on the L4, first: granite-3.1-3b-a800m-instruct eager, bf16 against Glyd exact (the routed experts
# decoded, then vLLM's Triton MoE kernel): tokens and logprobs bit for bit? Under the box's lock.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; PY=$W/venv/bin/python; O=$W/m4/smoke; mkdir -p $O
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp VLLM_ENABLE_V1_MULTIPROCESSING=0
M=ibm-granite/granite-3.1-3b-a800m-instruct
echo "== $(date -u +%T) glyd exact eager"
(cd $O && VLLM_CACHE_ROOT=$(mktemp -d -p $TMPDIR) GLYD_EXACT=1 timeout 900 $PY $W/glyd/gpu/vllm/check_vllm.py --child "{\"model\": \"$M\", \"quantization\": \"glyd\", \"eager\": true}" $O/glyd-exact-eager.json) > $O/glyd-exact-eager.log 2>&1; echo "exit $?"
$PY - $O $W/glyd/gpu/vllm <<'PY'
import json, sys
sys.path.insert(0, sys.argv[2])
from check_vllm import compare
L = lambda n: json.load(open(f"{sys.argv[1]}/{n}.json"))
r = L("glyd-exact-eager")
print(r.get("error", "")[:800] or ("against bf16 eager: " + str(compare(r, L("bf16-eager")))))
PY
