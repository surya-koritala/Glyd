#!/bin/bash
# Compiled with CUDA graphs (vLLM's default), inductor deterministic, each on an empty compile cache: bf16 twice, Glyd
# fused twice, Glyd exact (its refusal lifted for this test); the check's 8 prompts and continuation (Qwen3-1.7B).
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; PY=$W/venv/bin/python; O=$W/m2/dbg4; mkdir -p $O
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=${GLYD_GPU_LIB:-$W/lib/libglyd_gpu_cuda13.so}
export PYTHONPATH=$W/glyd/bindings/python TMPDIR=$W/tmp VLLM_ENABLE_V1_MULTIPROCESSING=0 GLYD_LAYOUT=mma
spec() { $PY -c "import json,sys; d=json.load(open('$W/m2/check-Qwen3-1.7B/Qwen3-1.7B-bf16.json')); print(json.dumps(dict(model='Qwen/Qwen3-1.7B', long_ids=d['long_ids'], compilation_config={'inductor_compile_config': {'deterministic': True, 'combo_kernels': True, 'benchmark_combo_kernel': False}}, **json.loads(sys.argv[1]))))" "$1"; }
one() {  # NAME SPEC [VAR=VALUE...]
  local n=$1 s=$2; shift 2
  echo "== $(date -u +%T) $n"
  (cd $O && env VLLM_CACHE_ROOT=$(mktemp -d -p $TMPDIR) "$@" timeout 900 $PY $W/dbg_exact.py "$(spec "$s")" $O/$n.json $W/glyd/gpu/vllm) > $O/$n.log 2>&1; echo "$n exit $?"
}
one bf16A '{}'
one bf16B '{}'
one glydA '{"quantization": "glyd"}'
one glydB '{"quantization": "glyd"}'
one exact '{"quantization": "glyd"}' GLYD_EXACT=1
$PY - $O $W/glyd/gpu/vllm <<'PY' | tee $O/summary.txt
import json, sys
sys.path.insert(0, sys.argv[2])
from check_vllm import compare
L = lambda n: json.load(open(f"{sys.argv[1]}/{n}.json"))
for a, b in (("bf16A", "bf16B"), ("glydA", "glydB"), ("exact", "bf16A"), ("exact", "bf16B"), ("glydA", "bf16A")):
    try:
        c = compare(L(a), L(b))
        print(f"{a} against {b}: prompts bit for bit {c['bit_identical']} of 8, continuation bit for bit {c['long_bit_identical']}, top-1 {c['long_top1']:.4f}, mean |d| {c['long_mean_abs_dlogprob']:.2e}")
    except Exception as e:
        print(f"{a} against {b}: {e!r}")
PY
