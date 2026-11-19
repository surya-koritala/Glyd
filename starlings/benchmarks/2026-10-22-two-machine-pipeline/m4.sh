#!/bin/bash
# GLY-188 on two machines (M4): coordinator + stage 0 on pod C, stage 1 on pod G, enrolled
# through the CLI as in the daemon test. Usage: m4.sh STEP [ARGS]; each step prints what the
# next step needs. Everything binds to the podnet1 addresses.
set -euo pipefail
# Records and policy files must be private (0600) for the CLI to read them.
umask 077
M=/opt/m4
S=$M/starlings
T=$M/pipeline_tool
R=/workspace/m4run
MODEL=/workspace/.cache/huggingface/hub/models--glyd--Qwen3.5-9B-swift/snapshots/073a2132abeaa41c119678b9687092871521eb4d
C_IP=10.0.26.59
G_IP=10.1.54.61
admin() { sleep 0.3; "$S" --state-dir "$@"; }
word() { grep -o "$1[^ ;]*" | head -1 | cut -d= -f2; }

case "$1" in
c-init) # coordinator daemon, then stage 0 and the client enrolled on this pod
    mkdir -p "$R"
    "$S" --state-dir $R/coord cluster init --listen $C_IP:9741
    nohup "$S" --state-dir $R/coord daemon > $R/coord-enroll.log 2>&1 &
    sleep 3
    for pair in stage0:worker client:client; do
        name=${pair%%:*}; role=${pair##*:}
        admin $R/coord cluster invite $role $R/coord/$name.json
        out=$("$S" --state-dir $R/$name cluster prepare $R/coord/$name.json)
        admin $R/coord cluster approve "$(echo "$out" | word node=)" "$(echo "$out" | word fingerprint=)"
        sleep 0.3
        "$S" --state-dir $R/$name cluster join $R/coord/$name.json
    done
    ;;
c-invite) # an invitation for stage 1, as base64 for pod G
    admin $R/coord cluster invite worker $R/coord/stage1.json
    base64 -w0 $R/coord/stage1.json; echo
    ;;
g-prepare) # $2: the invitation as base64
    mkdir -p "$R" && mkdir -m 700 -p $R/inv
    echo "$2" | base64 -d > $R/inv/stage1.json
    chmod 600 $R/inv/stage1.json
    "$S" --state-dir $R/stage1 cluster prepare $R/inv/stage1.json
    ;;
c-approve) # $2 node $3 fingerprint
    admin $R/coord cluster approve "$2" "$3"
    ;;
g-join)
    "$S" --state-dir $R/stage1 cluster join $R/inv/stage1.json
    ;;
describe) # $2 stage name, $3 first layer, $4 end layer, $5 this pod's IP: an approved stage route
    name=$2
    lib=$(sha256sum $M/libglyd_gpu_cuda13.so | cut -d' ' -f1)
    engine=$(sha256sum $M/glyd-engine | cut -d' ' -f1)
    python3 - "$R/$name/stage.json" "$3" "$4" "$5" "$lib" "$engine" "$MODEL" <<'EOF'
import json, sys
path, start, end, ip, lib, engine, model = sys.argv[1:]
json.dump({"version": 1, "listen": f"{ip}:9801", "data_listen": f"{ip}:9802",
    "executable": "/opt/m4/glyd-engine", "executable_sha256": engine, "model": model,
    "layers": [int(start), int(end)], "context_tokens": 1024, "chunk_tokens": 32,
    "max_prompt_tokens": 768, "max_output_tokens": 256, "descriptor": "0" * 63 + "1",
    "execution_profile": "glyd-range", "boundary_profile": "qwen35-residual-mlp-bf16",
    "backend_id": "glyd-rangeworker", "device_id": "cuda:0", "engine_bytes_limit": str(15 << 30),
    "environment": {"GLYD_GPU_LIB": "/opt/m4/libglyd_gpu_cuda13.so"},
    "gpu_library_sha256": lib}, open(path, "w"))
EOF
    described=$("$T" describe $R/$name/stage.json $R/$name/engine)
    python3 - "$R/$name/stage.json" "$described" <<'EOF'
import json, sys
path, described = sys.argv[1:]
stage = json.load(open(path)); stage["descriptor"] = json.loads(described)["descriptor"]
json.dump(stage, open(path, "w"))
EOF
    echo "$described"
    ;;
c-policy) # $2 model digest, $3 stage 1 node: permissions, a client token, then the plan and route
    model=$2
    coord=$(python3 -c "import json; print(json.load(open('$R/coord/config.json'))['node'])")
    cluster=$(python3 -c "import json; print(json.load(open('$R/coord/config.json'))['cluster'])")
    stage0=$(python3 -c "import json; print(json.load(open('$R/stage0/config.json'))['node'])")
    client=$(python3 -c "import json; print(json.load(open('$R/client/config.json'))['node'])")
    for pair in $coord:control $stage0:infer $3:infer $client:infer; do
        echo "[{\"model\":\"$model\",\"action\":\"${pair##*:}\"}]" > $R/coord/scopes.json
        chmod 600 $R/coord/scopes.json
        admin $R/coord cluster permissions ${pair%%:*} $R/coord/scopes.json
    done
    admin $R/coord cluster token $client $R/coord/scopes.json 3600 $R/coord/token.json
    sleep 1
    admin $R/coord cluster stop
    sleep 3
    python3 - "$R/coord/plan-draft.json" "$cluster" "$model" "$stage0" "$3" \
        "$(sha256sum $M/glyd-engine | cut -d' ' -f1)" <<'EOF'
import json, sys, uuid
path, cluster, model, s0, s1, build = sys.argv[1:]
def stage(i, node, start, end):
    return {"index": i, "node_id": node, "device_id": "cuda:0", "backend_id": "glyd-rangeworker",
        "backend_build_digest": build, "layer_start": start, "layer_end_exclusive": end,
        "owns_embedding": i == 0, "owns_final_norm_and_head": i == 1, "memory_pool_id": "cuda:0",
        "resident_bytes": str(6 << 30), "state_budget_bytes": str(256 << 20),
        "scratch_budget_bytes": str(64 << 20), "safety_reserve_bytes": str(512 << 20)}
json.dump({"kind": "PlacementPlan", "protocol_version": 1, "fixture": False,
    "plan_id": str(uuid.uuid4()), "plan_digest": "0" * 64, "cluster_id": cluster,
    "authority_epoch": "1", "manifest_digest": model, "mode": "layer_pipeline",
    "model_layer_count": 32, "execution_profile": "glyd-range",
    "boundary_profile": "qwen35-residual-mlp-bf16", "context_limit": 1024, "max_sessions": 1,
    "stages": [stage(0, s0, 0, 16), stage(1, s1, 16, 32)],
    "created_at": "2026-10-22T12:00:00-04:00", "expires_at": "2026-10-22T12:01:00-04:00"},
    open(path, "w"))
EOF
    "$T" plan $R/coord/plan-draft.json > $R/coord/plan.json
    python3 - "$R/coord/plan.json" "$R/coord/pipeline.json" "$C_IP" "$G_IP" <<'EOF'
import json, sys
plan, path, c, g = sys.argv[1:]
json.dump({"version": 1, "listen": "127.0.0.1:9800", "plan": json.load(open(plan)),
    "stages": [{"endpoint": f"{c}:9801", "data": f"{c}:9802"},
               {"endpoint": f"{g}:9801", "data": f"{g}:9802"}],
    "chunk_tokens": 32, "max_output_tokens": 256, "lifetime_ms": 60000,
    "stop_tokens": [248044, 248046]}, open(path, "w"))
EOF
    echo "plan $(jq -r .plan_digest $R/coord/plan.json)"
    ;;
start) # $2 coord|stage0|stage1: a daemon with its approved record
    case $2 in
    coord) file=$R/coord/pipeline.json; flag=--pipeline; kind=pipeline-digest ;;
    *) file=$R/$2/stage.json; flag=--stage; kind=stage-digest ;;
    esac
    digest=$("$S" --state-dir $R/$2 model $kind $file | jq -r .configuration)
    nohup "$S" --state-dir $R/$2 daemon $flag $file --approve "$digest" > $R/$2.log 2>&1 &
    sleep 2
    echo "$2 started, approved $digest"
    ;;
service) # $2 daemon: its service report
    admin $R/$2 cluster service
    ;;
resources) # coordinator's view of node $2
    admin $R/coord cluster resources "$2" | sed 's/^RESOURCES: //' | jq -c '.sample.capabilities.state'
    ;;
chat) # $2 name $3 text $4 max_tokens $5 temperature $6 top_p $7 seed: one streamed request
    bearer=$(python3 -c "import json; print(''.join(f'{b:02x}' for b in json.load(open('$R/coord/token.json'))['secret']))")
    model=$(jq -r .plan.manifest_digest $R/coord/pipeline.json)
    body=$(jq -cn --arg m "$model" --arg t "$3" --argjson n "$4" --argjson T "$5" --argjson p "$6" --argjson s "$7" \
        '{model:$m, messages:[{role:"user",content:$t}], max_tokens:$n, temperature:$T, top_p:$p, seed:$s, stream:true}')
    start=$(date +%s.%N)
    curl -sN -H "Authorization: Bearer $bearer" -H 'Content-Type: application/json' \
        --data "$body" http://127.0.0.1:9800/v1/chat/completions > $R/chat-$2.sse
    end=$(date +%s.%N)
    echo "client_ms $(python3 -c "print(round(($end - $start) * 1000, 1))")"
    grep -c '^data: {' $R/chat-$2.sse || true
    tail -c 300 $R/chat-$2.sse; echo
    grep PIPELINE_REPORT $R/coord.log | tail -1
    ;;
stop) # $2 daemon
    admin $R/$2 cluster stop || true
    ;;
reference) # $2 name $3 text $4 max_tokens $5 temperature $6 top_p $7 seed: the whole model here
    "$T" reference $R/stage0/stage.json $R/stage0/engine "$3" "$4" "$5" "$6" "$7" 248044,248046 \
        | tee $R/reference-$2.json
    ;;
esac
