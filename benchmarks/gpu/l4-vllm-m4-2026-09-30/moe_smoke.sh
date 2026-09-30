#!/bin/bash
# The MoE path's first runs on the L4, under the box's lock: granite-3.1-3b-a800m-instruct through check_vllm.py's child,
# bf16 eager, then Glyd tiered eager with GLYD_VERIFY=1 and its layers checked (the experts' products among them).
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; PY=$W/venv/bin/python; O=$W/m4/smoke; mkdir -p $O
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib-v0251/libglyd_gpu_cuda13.so
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp VLLM_ENABLE_V1_MULTIPROCESSING=0
M=ibm-granite/granite-3.1-3b-a800m-instruct
one() { echo "== $(date -u +%T) $1"; (cd $O && env VLLM_CACHE_ROOT=$(mktemp -d -p $TMPDIR) "${@:3}" timeout 900 $PY $W/glyd/gpu/vllm/check_vllm.py --child "$2" $O/$1.json) > $O/$1.log 2>&1; echo "$1 exit $?"; }
one bf16-eager "{\"model\": \"$M\", \"eager\": true}"
one glyd-eager "{\"model\": \"$M\", \"quantization\": \"glyd\", \"eager\": true, \"layers\": true}" GLYD_LAYOUT=mma GLYD_VERIFY=1
one glyd-graphs "{\"model\": \"$M\", \"quantization\": \"glyd\", \"layers\": true}" GLYD_LAYOUT=mma12
$PY - $O $W/glyd/gpu/vllm <<'PY'
import json, sys
sys.path.insert(0, sys.argv[2])
from check_vllm import compare
L = lambda n: json.load(open(f"{sys.argv[1]}/{n}.json"))
b = L("bf16-eager")
for n in ("glyd-eager", "glyd-graphs"):
    r = L(n)
    if "error" in r:
        print(n, "error:", r["error"][:600]); continue
    print(n, "layers", r.get("layers"), "\n  against bf16 eager", compare(r, b), "\n  first tokens", r["tokens"][0][:12], b["tokens"][0][:12])
PY
