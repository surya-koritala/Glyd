#!/bin/bash
# Is glyd in vLLM the same from one process to the next, compiled? Qwen3-1.7B: glyd tiered compiled twice on one
# compile cache (the second loads the first's graph), with CUDA graphs and without them (torch.compile alone); bf16
# likewise beside it. Run under the box's lock, from m2_extra.sh (its environment).
set -u
W=~/vllm-work; PY=$W/venv/bin/python; O=$W/m2/repro; mkdir -p $O
for i in $(seq 1 180); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 10; done  # a GPU of its own
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "another process is on the GPU: not run"; exit 1; }
LONG=$W/m2/check-Qwen3-1.7B/Qwen3-1.7B-bf16.json
spec() { $PY -c "import json,sys; d=json.load(open('$LONG')); print(json.dumps(dict(model='Qwen/Qwen3-1.7B', long_ids=d['long_ids'], **json.loads(sys.argv[1]))))" "$1"; }
one() {  # NAME CACHE SPEC [VAR=VALUE...]
  local n=$1 c=$2 s=$3; shift 3
  echo "== $(date -u +%T) $n"
  (cd $O && env VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CACHE_ROOT=$c "$@" timeout 900 $PY $W/glyd/gpu/vllm/check_vllm.py --child "$(spec "$s")" $O/$n.json) > $O/$n.log 2>&1; echo "$n exit $?"
}
A=$(mktemp -d -p $TMPDIR) B=$(mktemp -d -p $TMPDIR) C=$(mktemp -d -p $TMPDIR) D=$(mktemp -d -p $TMPDIR)
NG='"compilation_config": {"cudagraph_mode": "NONE"}'
one glyd-graphs $A '{"quantization": "glyd"}' GLYD_LAYOUT=mma
one glyd-graphs-again $A '{"quantization": "glyd"}' GLYD_LAYOUT=mma
one glyd-nographs $B "{\"quantization\": \"glyd\", $NG}" GLYD_LAYOUT=mma
one glyd-nographs-again $B "{\"quantization\": \"glyd\", $NG}" GLYD_LAYOUT=mma
one bf16-graphs $C '{}'
one bf16-graphs-again $C '{}'
one bf16-nographs $D "{$NG}"
one bf16-nographs-again $D "{$NG}"
$PY - $O <<'PY' | tee $O/summary.txt
import json, sys
O = sys.argv[1]
L = lambda n: json.load(open(f"{O}/{n}.json"))
for a in ("glyd-graphs", "glyd-nographs", "bf16-graphs", "bf16-nographs"):
    try:
        x, y = L(a), L(a + "-again")
        same = sum(p == q and lp == lq for p, q, lp, lq in zip(x["tokens"], y["tokens"], x["logprobs"], y["logprobs"]))
        print(f"{a} against {a}-again (one compile cache, its graph loaded): prompts bit for bit {same} of 8; continuation bit for bit {x['long_logprobs'] == y['long_logprobs']}")
    except Exception as e:
        print(f"{a}: {e!r}")
PY
rm -rf $A $B $C $D
touch $O/done
