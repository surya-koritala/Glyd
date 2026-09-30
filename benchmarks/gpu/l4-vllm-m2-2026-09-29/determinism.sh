#!/bin/bash
# Is glyd in vLLM the same from one process to the next? Qwen3-1.7B, the tiered layout: eager twice (no compile), and
# compiled on two empty caches, then again on the first (its graph loaded); bf16 compiled on two empty caches beside it.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
set -u
W=~/vllm-work; PY=$W/venv/bin/python; O=$W/m2/determinism; mkdir -p $O
export HF_HOME=~/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONSAFEPATH=1 GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda13.so VLLM_ENABLE_V1_MULTIPROCESSING=0 TMPDIR=$W/tmp
C=$W/check_vllm_child.py
LONG=$W/m2/check-Qwen3-1.7B/Qwen3-1.7B-bf16.json
spec() { $PY -c "import json,sys; d=json.load(open('$LONG')); print(json.dumps(dict(model='Qwen/Qwen3-1.7B', long_ids=d['long_ids'], **json.loads(sys.argv[1]))))" "$1"; }
one() {  # NAME CACHE SPEC [VAR=VALUE...]
  local n=$1 c=$2 s=$3; shift 3
  echo "== $(date -u +%T) $n"
  (cd $O && env VLLM_CACHE_ROOT=$c "$@" timeout 900 $PY $W/glyd/gpu/vllm/check_vllm.py --child "$(spec "$s")" $O/$n.json) > $O/$n.log 2>&1; echo "$n exit $?"
}
A=$(mktemp -d -p $W/tmp) B=$(mktemp -d -p $W/tmp) D=$(mktemp -d -p $W/tmp) E=$(mktemp -d -p $W/tmp)
one eager1 $D '{"quantization": "glyd", "eager": true}' GLYD_LAYOUT=mma
one eager2 $E '{"quantization": "glyd", "eager": true}' GLYD_LAYOUT=mma
one compiledA $A '{"quantization": "glyd"}' GLYD_LAYOUT=mma
one compiledB $B '{"quantization": "glyd"}' GLYD_LAYOUT=mma
one compiledA-again $A '{"quantization": "glyd"}' GLYD_LAYOUT=mma
F=$(mktemp -d -p $W/tmp) G=$(mktemp -d -p $W/tmp)
one bf16A $F '{}'
one bf16B $G '{}'
$PY - $O <<'PY'
import json, sys
O = sys.argv[1]
L = lambda n: json.load(open(f"{O}/{n}.json"))
for a, b in (("eager1", "eager2"), ("compiledA", "compiledB"), ("compiledA", "compiledA-again"), ("bf16A", "bf16B")):
    try:
        x, y = L(a), L(b)
        same = sum(p == q and lp == lq for p, q, lp, lq in zip(x["tokens"], y["tokens"], x["logprobs"], y["logprobs"]))
        print(f"{a} against {b}: prompts bit for bit {same} of 8; continuation bit for bit {x['long_logprobs'] == y['long_logprobs']}")
    except Exception as e:
        print(f"{a} against {b}: {e!r}")
PY
rm -rf $A $B $D $E $F $G
