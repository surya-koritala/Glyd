#!/bin/bash
# The planner's predictions (src/plan.rs, P2c) against runs they were NOT fitted on, on glyd-pc's RTX 4080 SUPER. Run by the M2 handoff's wrapper:
#   handoff/scripts/gpu-run.sh g3 scripts/planner_heldout_check.sh g3.tar.gz      (g3.tar.gz: `git archive` of the branch; it lands as /work/g3.tar.gz)
# The constants were fitted on bench --loads 1k:1 of gemma-4-12B at wren and kestrel and Qwen3.5-0.8B at kestrel and wren (in-sample). Held out here: the swift
# level of both, both at 4k and 8k positions, and prompts of 4k and 8k tokens. Every prediction is made and saved before any measurement (the cache is new, so no
# run observed refines it), then each case is measured and its error printed. Its own cache (/cache/glyd-g: rates/ and plans/), the shared packs read through a
# link; the swift packs it makes are deleted after each case, and the cache at the end. Needs 25 GB free on / for gemma's swift pack (it is skipped otherwise).
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive CARGO_HOME=/cache/cargo RUSTUP_HOME=/cache/rustup HF_HOME=/cache/hf HF_HUB_OFFLINE=1
export GLYD_ENGINE_ARCHS=89 NVCC=/usr/local/cuda/bin/nvcc CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential pkg-config curl git ca-certificates python3 > /dev/null
export PATH="$CARGO_HOME/bin:$PATH"
df -h / | tail -1
rm -rf /work/g3 && mkdir -p /work/g3 && tar -xzf /work/g3.tar.gz -C /work/g3 && find /work/g3 -type f -exec touch {} +
(cd /work/g3 && CARGO_TARGET_DIR=/cache/target-g bash scripts/ci_gpu_build.sh > /work/build-g3.log 2>&1); echo "build g3 rc=$? $(date -u +%H:%M:%S)"; tail -2 /work/build-g3.log
E=/cache/target-g/release/glyd-engine; export GLYD_GPU_LIB=$(ls -t /cache/target-g/gpu-*/libglyd_gpu_cuda*.so | head -1)
export GLYD_CACHE=/cache/glyd-g
rm -rf /cache/glyd-g && mkdir -p /cache/glyd-g && ln -s /cache/glyd/engine /cache/glyd-g/engine
cd /work

echo "=== 1. the rate probe: fit on the GPU's first use, then again"
for i in 1 2; do s=$(date +%s%N); gpu "$E" fit google/gemma-4-12B-it:wren --json > /work/g3-fit$i.json 2> /work/g3-fit$i.err; echo "fit $i rc=$? $(( ($(date +%s%N) - s) / 1000000 )) ms"; done
cat /cache/glyd-g/rates/*.json; echo

# the held-out cases: name, model, bench load, positions a decode step reads (the prompt and half of the 128 tokens made)
CASES="gemma-swift-1k google/gemma-4-12B-it:swift 1k:1 1088
q08-swift-1k Qwen/Qwen3.5-0.8B:swift 1k:1 1088
gemma-wren-4k google/gemma-4-12B-it:wren 4k:1 4160
gemma-wren-8k google/gemma-4-12B-it:wren 8k:1 8256
q08-kestrel-4k Qwen/Qwen3.5-0.8B:kestrel 4k:1 4160
q08-kestrel-8k Qwen/Qwen3.5-0.8B:kestrel 8k:1 8256"

echo "=== 2. every prediction, saved before anything is measured $(date -u +%H:%M:%S)"
echo "$CASES" | while read -r name model load pos; do
  GLYD_FIT_POSITIONS=$pos gpu "$E" fit "$model" --json > /work/g3-pred-$name.json 2> /work/g3-pred-$name.err
  python3 -c "import json,sys;v=json.load(open('/work/g3-pred-$name.json'));print('$name', 'level', v.get('level'), 'positions', v.get('positions'), 'decode', v.get('tokens_per_second'), (v.get('tokens_per_second_p10'), v.get('tokens_per_second_p90')), 'prefill', v.get('prefill_tokens_per_second'))" 2>&1 | tail -1
done
ls -la --time-style=+%H:%M:%S /work/g3-pred-*.json | awk '{print $6, $7}'

echo "=== 3. each case measured (bench, 3 runs, the median), then its error $(date -u +%H:%M:%S)"
echo "$CASES" | while read -r name model load pos; do
  touch /work/g3-$name.start
  case "$model" in *gemma*swift*)
    avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); echo "disk free ${avail}G"
    if [ "${avail:-0}" -lt 25 ]; then echo "$name: skipped, under 25 GB free for its pack"; continue; fi ;;
  esac
  gpu "$E" bench "$model" --loads "$load" --runs 3 --warmup 1 --no-draft --label "g3 $name" --out /work/g3-bench-$name.json > /work/g3-bench-$name.log 2>&1; echo "$name bench rc=$?"
  for d in $(find /cache/glyd/engine -mindepth 1 -maxdepth 1 -type d -newer /work/g3-$name.start \( -name 'google--gemma-4-12B-it*-i5' -o -name 'Qwen--Qwen3.5-0.8B*-i5' \)); do du -sh "$d"; rm -rf "$d"; echo "removed $d"; done
done
python3 - "$CASES" <<'PY'
import json, os, sys
out = open('/work/g3-heldout.tsv', 'w')
head = 'case\tlevel\tpositions\tdecode_predicted\tdecode_measured\tdecode_error_pct\tprompt_tokens\tprefill_predicted\tprefill_measured\tprefill_error_pct'
print(head); out.write(head + '\n')
for line in sys.argv[1].strip().splitlines():
    name = line.split()[0]
    try:
        p = json.load(open(f'/work/g3-pred-{name}.json'))
        b = json.load(open(f'/work/g3-bench-{name}.json'))
    except Exception as e:
        print(name, 'not measured:', e); continue
    c = b['cells'][0]
    med = lambda k: (c.get(k) or {}).get('median')
    dm, pm, dp, pp = med('decode_tps_per_user'), med('prefill_tps'), p.get('tokens_per_second'), p.get('prefill_tokens_per_second')
    err = lambda pr, me: None if not pr or not me else round((pr - me) / me * 100, 1)
    row = [name, p.get('level'), p.get('positions'), dp, dm and round(dm, 1), err(dp, dm), c.get('prompt_tokens'), pp, pm and round(pm), err(pp, pm)]
    s = '\t'.join(str(x) for x in row); print(s); out.write(s + '\n')
PY

echo "=== 4. the fitted cases again (in-sample, re-measured with this build), and fit refined by the runs observed above"
for M in google/gemma-4-12B-it:wren Qwen/Qwen3.5-0.8B; do
  tag=$(echo "$M" | tr '/:' '__')
  gpu "$E" bench "$M" --loads 1k:1 --runs 3 --warmup 1 --no-draft --label "g3 fitted $M" --out /work/g3-bench-fitted-$tag.json > /work/g3-bench-fitted-$tag.log 2>&1; echo "$M bench rc=$?"
  tail -2 /work/g3-bench-fitted-$tag.log | cut -c1-200
  gpu "$E" fit "$M" 2>&1 | grep -E "^GPU|^Plan|^\(" | cut -c1-260
done
python3 -c "
import json,glob
for f in glob.glob('/cache/glyd-g/plans/*.json'):
    v=json.load(open(f)); o=v.get('observed',[])
    print(f.split('/')[-1], len(o), [round(s.get('decode_ratio',0),3) for s in o][:12], [round(s.get('prefill_ratio',0),3) for s in o if 'prefill_ratio' in s][:6])
"
rm /cache/glyd-g/engine && rm -rf /cache/glyd-g
df -h / | tail -1
echo G3-DONE
