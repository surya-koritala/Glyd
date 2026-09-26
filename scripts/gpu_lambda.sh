#!/usr/bin/env bash
# Model weights compressed in GPU memory, measured on one Lambda Cloud GPU
# instance, launched for the run only: bf16 and Glyd side by side on the
# same GPU (gpu/e2e.py, gpu/gemm.py), results to
# benchmarks/gpu/lambda-<type>/. The instance is terminated when the script
# exits for any reason, and after MAX_MIN minutes whatever happens; at the
# end the account's instances are listed to show none of ours is left.
#   scripts/gpu_lambda.sh [git-ref]
#   HOST=user@ip SSH_KEY=~/.ssh/key RW=glyd-dry scripts/gpu_lambda.sh
#     (the same run on a machine you already have: no launch, no API)
#   IP=a.b.c.d SSH_KEY=~/.ssh/glyd-lambda scripts/gpu_lambda.sh
#     (an instance launched from the console: run on it, and with the API
#     key, terminate it at the end and at the cap)
# Env: LAMBDA_KEY_FILE (~/.lambda/api_key), TYPE (gpu_1x_h100_sxm5),
#      REGION (the first with capacity), MAX_MIN (75),
#      MODELS ("Qwen2.5-7B-Instruct Qwen2.5-32B-Instruct"), BATCH (1,8,32,64),
#      RW (the remote work directory, relative to home: . on Lambda).
set -euo pipefail
REF="${1:-HEAD}"
KEY_FILE="${LAMBDA_KEY_FILE:-$HOME/.lambda/api_key}"
TYPE="${TYPE:-gpu_1x_h100_sxm5}"
MAX_MIN="${MAX_MIN:-75}"
MODELS="${MODELS:-Qwen2.5-7B-Instruct Qwen2.5-32B-Instruct}"
BATCH="${BATCH:-1,8,32,64}"
RW="${RW:-.}"
API=https://cloud.lambda.ai/api/v1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/benchmarks/gpu/lambda-$TYPE"
RUN_ID="glyd-gpu-$(date +%Y%m%d-%H%M%S)"
TMP="$(mktemp -d)"
ID=""; KEY_ID=""; PRICE=0; START=$(date +%s)

# What runs on the instance: everything in \$HOME/$RW, results in its results/.
run_remote() {
    cat <<EOF
set -x
W=\$HOME/$RW
mkdir -p \$W/results \$W/models
R=\$W/results
{ echo "commit: $(git -C "$ROOT" rev-parse --short "$REF")"; date -u; nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv; nproc; free -g | head -2; } > \$R/machine.txt
bash \$W/glyd/gpu/setup_env.sh \$W/gpuenv > \$R/setup.txt 2>&1 || { echo "SETUP FAILED" > \$R/FAILED; touch \$R/DONE; exit 1; }
source \$W/gpuenv/cuda.sh
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd \$W/glyd/gpu
python -c "import glyd_gpu" > \$R/build.txt 2>&1 &
for m in $MODELS; do
  [ -f \$W/models/\$m/config.json ] || (hf download Qwen/\$m --local-dir \$W/models/\$m || huggingface-cli download Qwen/\$m --local-dir \$W/models/\$m) > \$R/download-\$m.txt 2>&1 &
done
[ -f \$W/enwik8 ] || (curl -sL http://mattmahoney.net/dc/enwik8.zip -o \$W/enwik8.zip && python -c "import zipfile; zipfile.ZipFile('\$W/enwik8.zip').extractall('\$W')") &
wait
du -sh \$W/models/* >> \$R/machine.txt
# The GPU's clocks, power and temperature every 5 s while the tests run.
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 5 > \$R/clocks.csv 2>&1 &
CLK=\$!
# Every product checked and timed first (a failure there: the models without bulk copies); every step bounded.
first=\$(echo $MODELS | cut -d' ' -f1)
timeout 900 python gemm.py \$W/models/\$first 1,16,64,256,2048 > \$R/gemm-\$first.txt 2>&1 || { echo "gemm.py failed: GLYD_GPU_TMA=0" >> \$R/machine.txt; export GLYD_GPU_TMA=0; }
for m in $MODELS; do
  timeout 1500 python e2e.py \$W/models/\$m --format mma --fused --baseline --tokens 64 --batch $BATCH --prefill 64,128,512,2048 --ppl \$W/enwik8 --profile 16 > \$R/e2e-\$m.txt 2>&1
done
kill \$CLK
touch \$R/DONE
EOF
}

# Upload, start, follow, fetch: the same for a Lambda instance and for HOST.
run_on() {
    "${SSH[@]}" "${H[@]}" "mkdir -p $RW/glyd && rm -rf $RW/results $RW/run.log && tar -C $RW/glyd -xf -" < "$TMP/glyd.tar"
    run_remote | "${SSH[@]}" "${H[@]}" "cat > $RW/run.sh"
    "${SSH[@]}" "${H[@]}" "cd $RW && nohup bash run.sh > run.log 2>&1 < /dev/null &"
    LIVE="$TMP/results"; mkdir -p "$LIVE"
    while true; do
        sleep 60
        "${SSH[@]}" "${H[@]}" "tar -C $RW/results -cf - . 2>/dev/null; true" | tar -C "$LIVE" -xf - 2> /dev/null || true
        "${SSH[@]}" "${H[@]}" cat $RW/run.log > "$LIVE/run.log" 2> /dev/null || true
        [ -f "$LIVE/DONE" ] && break
        echo "$(( ($(date +%s) - START) / 60 )) min: $(ls "$LIVE" | tr '\n' ' ')"
    done
    rm -rf "$OUT"; mkdir -p "$OUT"; cp -R "$LIVE"/. "$OUT"/
    grep -h -E "tokens/s|prefill|perplexity|choice|packed" "$OUT"/e2e-*.txt 2> /dev/null
}

if [ -n "${HOST:-}" ]; then
    OUT="$ROOT/benchmarks/gpu/dry-$(echo "$HOST" | tr -c 'a-zA-Z0-9.\n' _)"
    SSH=(ssh -i "${SSH_KEY:-$HOME/.ssh/id_ed25519}" -o ConnectTimeout=10 -o ServerAliveInterval=60)
    H=("$HOST")
    git -C "$ROOT" archive --format=tar -o "$TMP/glyd.tar" "$REF" gpu
    run_on
    rm -rf "$TMP"
    exit
fi

if [ -n "${IP:-}" ] && [ ! -s "$KEY_FILE" ]; then
    # Launched from the console, no API key here: run on it; the console terminates it.
    SSH=(ssh -i "${SSH_KEY:-$HOME/.ssh/glyd-lambda}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=60)
    H=(ubuntu@"$IP")
    until "${SSH[@]}" "${H[@]}" true 2> /dev/null; do sleep 5; done
    git -C "$ROOT" archive --format=tar -o "$TMP/glyd.tar" "$REF" gpu
    run_on
    rm -rf "$TMP"
    echo "done: TERMINATE THE INSTANCE IN THE CONSOLE NOW"
    exit
fi

[ -s "$KEY_FILE" ] || { echo "no API key at $KEY_FILE"; exit 1; }
printf 'Authorization: Bearer %s\n' "$(cat "$KEY_FILE")" > "$TMP/auth"; chmod 600 "$TMP/auth"
api() { local m=$1 p=$2; shift 2; curl -sS --fail-with-body -X "$m" -H @"$TMP/auth" -H "Content-Type: application/json" "$API$p" "$@"; }
py() { python3 -c "$1"; }

cleanup() {
    set +e
    # Whatever carries this run's name, even if its launch's reply was lost.
    for i in $(api GET /instances | py "import json,sys; print(' '.join(i['id'] for i in json.load(sys.stdin)['data'] if i.get('name') == '$RUN_ID' and i['id'] != '$ID'))" 2> /dev/null); do
        echo "cleanup: terminating $i (by name)"
        api POST /instance-operations/terminate -d "{\"instance_ids\": [\"$i\"]}" > /dev/null
    done
    if [ -n "$ID" ]; then
        echo "cleanup: terminating $ID"
        api POST /instance-operations/terminate -d "{\"instance_ids\": [\"$ID\"]}" > /dev/null
        for _ in $(seq 60); do
            s=$(api GET "/instances/$ID" | py 'import json,sys; print(json.load(sys.stdin)["data"]["status"])' 2>/dev/null)
            [ "$s" = "terminated" ] || [ -z "$s" ] && break
            sleep 10
        done
    fi
    [ -n "$KEY_ID" ] && { api DELETE "/ssh-keys/$KEY_ID" > /dev/null 2>&1 || api DELETE "/ssh-keys/$RUN_ID" > /dev/null 2>&1; }
    left=$(api GET /instances | py 'import json,sys; print(sum(1 for i in json.load(sys.stdin)["data"] if i.get("name", "").startswith("glyd-gpu") and i["status"] != "terminated"))' 2>/dev/null)
    mins=$(( ($(date +%s) - START + 59) / 60 ))
    echo "cleanup: our instances still running: ${left:-unknown}; ${mins} min, about \$$(py "print(f'{$PRICE / 100 * $mins / 60:.2f}')")"
    [ -n "${WATCH:-}" ] && kill "$WATCH" 2> /dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT
trap 'exit 1' INT TERM HUP

if [ -n "${IP:-}" ]; then
    # Launched from the console: find it by its address, then run, cap and terminate as for our own.
    read -r ID PRICE < <(api GET /instances | py "
import json,sys
i = next(i for i in json.load(sys.stdin)['data'] if i.get('ip') == '$IP')
print(i['id'], (i.get('instance_type') or {}).get('price_cents_per_hour', 0))")
    echo "instance $ID at $IP; cap $MAX_MIN min"
    ( sleep $((MAX_MIN * 60)); echo "time cap reached"; kill -TERM $$ ) &
    WATCH=$!
    command -v caffeinate > /dev/null && caffeinate -i -w $$ &
    SSH=(ssh -i "${SSH_KEY:-$HOME/.ssh/glyd-lambda}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=60)
    H=(ubuntu@"$IP")
    until "${SSH[@]}" "${H[@]}" true 2> /dev/null; do sleep 5; done
    git -C "$ROOT" archive --format=tar -o "$TMP/glyd.tar" "$REF" gpu
    run_on
    echo "done: $IP"
    exit
fi

# Price and a region with capacity for the type.
api GET /instance-types > "$TMP/types.json"
read -r REGION_FOUND PRICE < <(py "
import json
d = json.load(open('$TMP/types.json'))['data']['$TYPE']
regions = [r['name'] for r in d['regions_with_capacity_available']]
print(regions[0] if regions else '-', d['instance_type']['price_cents_per_hour'])")
REGION="${REGION:-$REGION_FOUND}"
[ "$REGION" != "-" ] || { echo "no capacity for $TYPE now"; exit 1; }
echo "$TYPE in $REGION at \$$(py "print($PRICE / 100)")/hour; cap $MAX_MIN min"

# A key for this run only.
ssh-keygen -q -t ed25519 -N "" -f "$TMP/key" -C "$RUN_ID"
KEY_ID=$(api POST /ssh-keys -d "{\"name\": \"$RUN_ID\", \"public_key\": \"$(cat "$TMP/key.pub")\"}" | py 'import json,sys; print(json.load(sys.stdin)["data"]["id"])')
SSH=(ssh -i "$TMP/key" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=60)

git -C "$ROOT" archive --format=tar -o "$TMP/glyd.tar" "$REF" gpu
ID=$(api POST /instance-operations/launch -d "{\"region_name\": \"$REGION\", \"instance_type_name\": \"$TYPE\", \"ssh_key_names\": [\"$RUN_ID\"], \"name\": \"$RUN_ID\"}" | py 'import json,sys; print(json.load(sys.stdin)["data"]["instance_ids"][0])')
START=$(date +%s)
echo "launched $ID"
# Whatever happens here: terminate at the cap (a watcher of its own; the Mac kept awake meanwhile).
( sleep $((MAX_MIN * 60)); echo "time cap reached"; kill -TERM $$ ) &
WATCH=$!
command -v caffeinate > /dev/null && caffeinate -i -w $$ &

IP=""
until [ -n "$IP" ]; do
    sleep 10
    s=""; IP=""
    read -r s IP < <(api GET "/instances/$ID" | py 'import json,sys; d = json.load(sys.stdin)["data"]; print(d["status"], d.get("ip") or "")' 2> /dev/null) || true
    [ "$s" = "active" ] || IP=""
    [ "$s" = "terminated" ] || [ "$s" = "unhealthy" ] && { echo "instance $s"; exit 1; }
done
H=(ubuntu@"$IP")
until "${SSH[@]}" "${H[@]}" true 2> /dev/null; do sleep 5; done
echo "active at $IP after $(( $(date +%s) - START )) s"
run_on
echo "done: $RUN_ID"
