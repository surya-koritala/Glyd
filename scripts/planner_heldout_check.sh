#!/bin/bash
# The planner's predictions (src/plan.rs, P2c) against runs they were NOT fitted on:   scripts/planner_heldout_check.sh OUT_DIR
# Run from the repository's root on a Linux machine with an NVIDIA GPU, the CUDA toolkit (NVCC, or nvcc on the PATH), cargo and python3. The shipped constants
# are an RTX 4080 SUPER's, which is the GPU this is for (GLYD_ENGINE_ARCHS=89). The models are read from the Hugging Face cache, offline.
# The constants were fitted on bench --loads 1k:1 of gemma-4-12B at wren and kestrel and Qwen3.5-0.8B at kestrel and wren (in-sample). Held out here: the swift
# level of both, both at 4k and 8k positions, and prompts of 4k and 8k tokens. Every prediction is made and saved before any measurement (the cache is new, so no
# run observed refines it), then each case is measured and its error printed. It is a measurement: no error is held to a bound, and it ends 0 whatever they are.
# OUT_DIR gets every prediction, bench result and log, and heldout.tsv. The engine's cache is a new one (OUT_DIR/cache: rates/ and plans/), deleted at the end.
# PACKS, where set, is an existing cache's engine directory, read through a link so that the models it holds are not packed again; the swift packs made are
# deleted after each case. Needs 25 GB free where the packs are made for gemma's swift pack (the case is skipped otherwise).
set -uo pipefail
OUT=${1:?usage: scripts/planner_heldout_check.sh OUT_DIR}
mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd)
export HF_HUB_OFFLINE=1 GLYD_ENGINE_ARCHS="${GLYD_ENGINE_ARCHS:-89}" NVCC="${NVCC:-$(command -v nvcc)}" CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target}"
export CARGO_PROFILE_RELEASE_LTO="${CARGO_PROFILE_RELEASE_LTO:-false}" CARGO_PROFILE_RELEASE_CODEGEN_UNITS="${CARGO_PROFILE_RELEASE_CODEGEN_UNITS:-16}"
bash scripts/ci_gpu_build.sh > "$OUT/build.log" 2>&1; echo "build rc=$? $(date -u +%H:%M:%S)"; tail -2 "$OUT/build.log"
E=$CARGO_TARGET_DIR/release/glyd-engine
export GLYD_CACHE=$OUT/cache
rm -rf "$GLYD_CACHE" && mkdir -p "$GLYD_CACHE"
if [ -n "${PACKS:-}" ]; then ln -s "$PACKS" "$GLYD_CACHE/engine"; else mkdir -p "$GLYD_CACHE/engine"; fi
PK=${PACKS:-$GLYD_CACHE/engine}
df -h "$PK" | tail -1

echo "=== 1. the rate probe: fit on the GPU's first use, then again"
for i in 1 2; do s=$(date +%s%N); "$E" fit google/gemma-4-12B-it:wren --json > "$OUT/fit$i.json" 2> "$OUT/fit$i.err"; echo "fit $i rc=$? $(( ($(date +%s%N) - s) / 1000000 )) ms"; done
cat "$GLYD_CACHE"/rates/*.json; echo

# the held-out cases: name, model, bench load, positions a decode step reads (the prompt and half of the 128 tokens made)
CASES="gemma-swift-1k google/gemma-4-12B-it:swift 1k:1 1088
q08-swift-1k Qwen/Qwen3.5-0.8B:swift 1k:1 1088
gemma-wren-4k google/gemma-4-12B-it:wren 4k:1 4160
gemma-wren-8k google/gemma-4-12B-it:wren 8k:1 8256
q08-kestrel-4k Qwen/Qwen3.5-0.8B:kestrel 4k:1 4160
q08-kestrel-8k Qwen/Qwen3.5-0.8B:kestrel 8k:1 8256"

echo "=== 2. every prediction, saved before anything is measured $(date -u +%H:%M:%S)"
echo "$CASES" | while read -r name model load pos; do
  GLYD_FIT_POSITIONS=$pos "$E" fit "$model" --json > "$OUT/pred-$name.json" 2> "$OUT/pred-$name.err"
  python3 -c "import json,sys;v=json.load(open(sys.argv[1]));print(sys.argv[2], 'level', v.get('level'), 'positions', v.get('positions'), 'decode', v.get('tokens_per_second'), (v.get('tokens_per_second_p10'), v.get('tokens_per_second_p90')), 'prefill', v.get('prefill_tokens_per_second'))" "$OUT/pred-$name.json" "$name" 2>&1 | tail -1
done
ls -la --time-style=+%H:%M:%S "$OUT"/pred-*.json | awk '{print $6, $7}'

echo "=== 3. each case measured (bench, 3 runs, the median), then its error $(date -u +%H:%M:%S)"
echo "$CASES" | while read -r name model load pos; do
  touch "$OUT/$name.start"
  case "$model" in *gemma*swift*)
    avail=$(df --output=avail -BG "$PK" | tail -1 | tr -dc 0-9); echo "disk free ${avail}G"
    if [ "${avail:-0}" -lt 25 ]; then echo "$name: skipped, under 25 GB free for its pack"; continue; fi ;;
  esac
  "$E" bench "$model" --loads "$load" --runs 3 --warmup 1 --no-draft --label "held out $name" --out "$OUT/bench-$name.json" > "$OUT/bench-$name.log" 2>&1; echo "$name bench rc=$?"
  for d in $(find "$PK/" -mindepth 1 -maxdepth 1 -type d -newer "$OUT/$name.start" \( -name 'google--gemma-4-12B-it*-i5' -o -name 'Qwen--Qwen3.5-0.8B*-i5' \)); do du -sh "$d"; rm -rf "$d"; echo "removed $d"; done
done
python3 - "$CASES" "$OUT" <<'PY'
import json, os, sys
dir = sys.argv[2]
out = open(f'{dir}/heldout.tsv', 'w')
head = 'case\tlevel\tpositions\tdecode_predicted\tdecode_measured\tdecode_error_pct\tprompt_tokens\tprefill_predicted\tprefill_measured\tprefill_error_pct'
print(head); out.write(head + '\n')
for line in sys.argv[1].strip().splitlines():
    name = line.split()[0]
    try:
        p = json.load(open(f'{dir}/pred-{name}.json'))
        b = json.load(open(f'{dir}/bench-{name}.json'))
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
  "$E" bench "$M" --loads 1k:1 --runs 3 --warmup 1 --no-draft --label "fitted $M" --out "$OUT/bench-fitted-$tag.json" > "$OUT/bench-fitted-$tag.log" 2>&1; echo "$M bench rc=$?"
  tail -2 "$OUT/bench-fitted-$tag.log" | cut -c1-200
  "$E" fit "$M" 2>&1 | grep -E "^GPU|^Plan|^\(" | cut -c1-260
done
python3 - "$GLYD_CACHE" <<'PY'
import glob, json, sys
for f in glob.glob(f'{sys.argv[1]}/plans/*.json'):
    v = json.load(open(f)); o = v.get('observed', [])
    print(f.split('/')[-1], len(o), [round(s.get('decode_ratio', 0), 3) for s in o][:12], [round(s.get('prefill_ratio', 0), 3) for s in o if 'prefill_ratio' in s][:6])
PY
[ -L "$GLYD_CACHE/engine" ] && rm "$GLYD_CACHE/engine"
rm -rf "$GLYD_CACHE"
df -h "$PK" 2> /dev/null | tail -1
echo "held-out measurement done: $OUT/heldout.tsv"
