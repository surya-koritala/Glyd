#!/usr/bin/env bash
# Nemotron 3 Super on the engine, the whole model: packed from the real checkpoint at the level the card holds and run on that one card (Linear GLY-132; rp_run.sh contract: results in ~/results, ~/results/DONE at the end).
# nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 at 2dc98e2a (247 GB). The level follows the card: 100 GB or more (the H200 141 GB) packs at linnet (--layout i5: five-bit codes, 83.5 GB), less (the A100 80 GB)
# at wren (--layout i4: the routed experts four-bit, every other matrix six-bit, 70.4 GB by the plan: the done-when of GLY-132, Nemotron on an 80 GB card at 4 bits). Each step is a line of steps.txt with its log;
# a step that fails does not stop the ones that can still run:
#   1. download   the checkpoint (the hub client, in the background beside the builds; the job stops if the speed projects past 45 minutes)
#   1b. moecheck  glyd-engine moecheck relu2-22of512 relu2-rows-22of512: the library's grouped products (the experts' second matrices in the block layout: the dense product) and the sparse product (in the row layout:
#                 src/moe_rows.rs) over experts in each layout (Int6, Int5, Int4) against the CPU reference, on Nemotron's layer: about two minutes, before the pack
#   1c. topkcheck glyd-engine topkcheck: the sampling kernels (a row's 64 largest logits, the sum of its exponentials) against the host's, on rows of every shape and at the two vocabularies' sizes: seconds, before the pack (GLY-139)
#   2. pack       glyd-engine pack MODEL ~/v/packed --layout $LAYOUT (src/pack.rs: plan, from its description: the packs and the tensors kept; the experts' down matrices in the row layout, so that every step below
#                 but 9 runs the sparse product): time, bytes, shards, the layouts of the packs
#   3. verify     glyd-gpu verify --device cuda:0: every pack decoded by the GPU library, in its own layout, to the sha256 glyd.json holds
#   4. generate   glyd-engine generate: 128 tokens by greedy steps (the decode graph): the engine's own speed, and the memory it reports
#   5. logits     160 tokens of a text (the checkpoint's tokenizer; 64 prefilled, then teacher-forced steps, as a CUDA graph and not) through the whole packed model (nhtoy), against tools/nemotron_ref.py full:
#                 the bf16 checkpoint run layer by layer on the GPU. Top-1 token and KL divergence at each of 97 positions: what the level costs on this text
#   6. layers     layers 0 and 40 (Mamba-2), 1 and 37 (mixture of experts), 7 and 36 (attention) of the packed directory through the engine's own code (nhlayers) against tools/nemotron_ref.py's reference
#                 on the weights the packs hold (nhdecode writes them as a mini checkpoint): every stage of every case (the prompt whole and in pieces, single steps, a CUDA graph's replays)
#   7. run        glyd-engine run on the packed directory: one answer (--prompt), then a chat of three prompts from stdin (each answer's tokens a second)
#   7b. sampled   glyd-engine sampletime: one prompt, 256 tokens by greedy steps, by sampled steps (the checkpoint's own sampling: temperature 1, top-p 0.95) drawn from the 64 largest logits the GPU makes, and by the same sampled steps with the
#                 row read to the host (GLY-139: a sampled step took twice a greedy one): tokens a second of each, how many draws read the row, and that the two sampled replies are the same tokens (the step fails if not)
#   8. serve      glyd-engine serve: three chat completions, and tools/runpod/speed1.py (time to first token and decode tokens a second: a prompt of about 1,000 words, 256 new tokens, the median of 3)
#   9. dense      the same model packed again with the experts' down matrices in the block layout (GLYD_ENGINE_DENSE_DOWN=1, a development build's variable), which runs the library's dense product, on the same card: generate,
#                 the whole model's logits against the bf16 checkpoint as step 5 (the same 160 tokens), serve and speed1.py as step 8 (decode tokens a second for one user: 96.6 on this card dense, nh-run-check-2026-10-10);
#                 then the sparse directory's generate with the blocks a pair's neurons are cut in forced to 2 and to 4 (GLYD_ENGINE_SPARSE_SPLIT; the default is 8 for a step's 22 pairs). Left out where the time left
#                 is short of another pack (the A100's pack takes an hour)
#   10. layer     glyd-engine moecheck nemotron3 --speed --no-check: the layer's time at the real shape, dense and sparse, at 1 to 2,048 tokens, on activations that are 92% zeros
# Uploaded beside it: src.tar (git archive of the tree, prefix src/). Results: summary.txt (made last), steps.txt, one .log a step, smi.csv (nvidia-smi each second), pack.txt, layers/ (spec.json), logits/ and
# logits-dense/ (toy.txt, ids.txt, the engine's tokens), speed.json and speed-dense.json, gpu.txt (rp_run.sh).
#
#   cd <the tree at the commit to test> && W=$(mktemp -d) && git archive --format=tar.gz --prefix=src/ HEAD > $W/src.tar && \
#   NAME=nh-run-check-wren MAX_MIN=150 RP_DISK=450 UPLOAD="$W/src.tar" bash tools/runpod/rp_class.sh a100 tools/cards/nh_run_check_job.sh      # wren on an A100 80 GB: $1.79/h; about 110 minutes, $3.3 (the cap, 150 minutes: $4.48)
#   NAME=nh-run-check MAX_MIN=150 RP_DISK=520 UPLOAD="$W/src.tar" bash tools/runpod/rp_class.sh h200 tools/cards/nh_run_check_job.sh            # linnet on an H200: $5.29/h; about 65 minutes, $5.8 with step 9 (the cap: $13.23)
# (The wren line's estimate is the old job's, steps 1 to 8; step 9 does not fit its time. The H200's: the 34 minutes of nh-run-check-2026-10-10, two minutes of moecheck, a second pack of 13 minutes, the dense model's
# generate, logits and speed 8, two forced splits 3, the layer's speed 5: neither launched. The pack, bound by the CPUs, is the longest step.) (rp_guard.sh deletes glyd-* pods older than its MAX_AGE: MAX_AGE=160 bash tools/runpod/rp_guard.sh & first.)
# Results come to $GLYD_RUNS/results-$NAME. RP_DISK: the checkpoint 247 GB, the packed model 84 GB (71 at wren) and, in step 9, the dense one, the decoded layers 12 GB, the builds and the environment 30 GB. Linnet's weights are 83.5 GB packed (77.8 GiB)
# and a run takes about 80 GiB of the card; at kestrel (--layout i6) 98.6 GB and about 94 GiB; wren's 70.4 GB (65.6 GiB) are 67.8 GiB with 4,096 tokens of context by glyd-engine's plan (an 80 GB card reports 79 GiB free).
set -u
R=~/results; mkdir -p $R ~/v; cd ~/v; tar -xf ~/src.tar
P=$R/steps.txt; : > $P
T0=$(date +%s)
MAX_MIN=${JOB_MAX_MIN:-150}   # (the launcher's cap, MAX_MIN=150 in the launch line above, counts the pod's start too: the job starts no step in its last 8 minutes)
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; local rc=$?; LAST_S=$(( $(date +%s) - t )); echo "$n: rc=$rc $LAST_S s (job minute $(( ($(date +%s) - T0) / 60 )))" >> $P; return $rc; }
late() { [ $(( ($(date +%s) - T0) / 60 )) -ge $((MAX_MIN - 8)) ]; }
fin() { [ -z "${SMI:-}" ] || kill $SMI 2> /dev/null; [ -z "${SERVE:-}" ] || kill $SERVE 2> /dev/null; touch $R/DONE; }   # (not `kill 0`, which is the whole process group)
trap fin EXIT
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN   # (the repo is ungated: nothing here reads or sends a token)
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_XET_HIGH_PERFORMANCE=1 HF_HOME=$HOME/v/hf PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1 RUST_BACKTRACE=1
REPO=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16; REV=2dc98e2afe4face0e4ce40972a915c45368bd34a
SRC=~/v/src; E=$SRC/glyd-engine; REF=$E/tools/nemotron_ref.py; M=~/v/m; PK=~/v/packed
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
# the level the card holds: linnet on 100 GB or more (an H200: 143,771 MiB), wren on less (an A100 80 GB: 81,920 MiB); LV is the bits the reference script's --level names
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${MEM:-0}" -ge 100000 ]; then LAYOUT=i5; LV=5; LEVEL=linnet; else LAYOUT=i4; LV=4; LEVEL=wren; fi
# the CPUs this container really has (a cgroup quota, not the host's cores): torch's thread pools and the packer's threads are sized by it
cpus() { local q p c=${CGROOT:-/sys/fs/cgroup}; { read -r q p < $c/cpu.max; } 2> /dev/null && [ "$q" != max ] && { echo $(( (q + p - 1) / p )); return; }
         { read -r q < $c/cpu/cpu.cfs_quota_us; read -r p < $c/cpu/cpu.cfs_period_us; } 2> /dev/null && [ "${q:--1}" -gt 0 ] && { echo $(( (q + p - 1) / p )); return; }; nproc; }
CPUS=$(cpus); export OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS
# the pack is bound by the CPUs (about 5.7 MB/s of bf16 a CPU: Qwen3.6-27B's 53.8 GB took 29 min on 7): 241 GB takes about 59 min on 12, past the time left on fewer
[ "$CPUS" -ge 12 ] || { echo "cpus: $CPUS, too few to pack 241 GB in time: stopped (launch again for another host)" >> $P; exit 1; }
{ TZ=EST5EDT date; nvidia-smi; nvcc --version | tail -2; echo "arch sm_$ARCH, $CPUS CPUs of $(nproc)"; free -g | head -2; df -h ~ | tail -1; cat $SRC/COMMIT 2> /dev/null; } > $R/machine.txt 2>&1
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 1 > $R/smi.csv 2>&1 &
SMI=$!

# the hub client first: the checkpoint downloads (the pinned revision, 247 GB) while torch installs and the library and the engine build, at the same time (every step has a time limit)
step venv bash -c 'python3 -m venv --system-site-packages ~/v/venv && ~/v/venv/bin/pip install -q "huggingface_hub>=1.5,<2" hf_xet' || exit 1
dl() { for i in 1 2 3; do timeout 1800 ~/v/venv/bin/hf download "$REPO" --revision "$REV" --local-dir $M && return 0; done; return 1; }   # (a stalled attempt is cut after 30 min and resumes)
step download dl &
DL=$!
( step pyenv timeout -k 10 1500 bash -c '( ~/v/venv/bin/python -c "import torch" 2> /dev/null || ~/v/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128 ) && ~/v/venv/bin/pip install -q numpy safetensors tokenizers' ) &
PYENV=$!
step rustup timeout -k 10 600 bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1'
. ~/.cargo/env
( step lib timeout -k 10 2400 bash -c "cd $SRC && bash gpu/build_lib.sh ~/v/lib" ) &
LIB=$!
[ -n "$ARCH" ] && export GLYD_ENGINE_ARCHS=$ARCH   # (this GPU alone)
step engine timeout -k 10 2400 bash -c "cd $SRC && cargo build --release -p glyd-engine -p glyd-gpu --bins"
B=$SRC/target/release/glyd-engine; G=$SRC/target/release/glyd-gpu; PY=~/v/venv/bin/python
wait $PYENV
wait $LIB
export GLYD_GPU_LIB=$(ls ~/v/lib/libglyd_gpu_cuda*.so 2> /dev/null | head -1)
[ -x $B ] && [ -x $G ] || { echo "no engine or no glyd-gpu: see engine.log" >> $P; exit 1; }
[ -n "$GLYD_GPU_LIB" ] || { echo "no library: see lib.log" >> $P; exit 1; }
step versions bash -c "$PY -c 'import torch; print(\"torch\", torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'; $B --version"
# the second product dense (the library's grouped products) and sparse (the row layout: src/moe_rows.rs) over each layout's experts (Int6, Int5, Int4) against the CPU reference on Nemotron's layer (relu squared, 22 of
# 512), two minutes of the GPU's, while the weights download: a product that is wrong is known before the hour of packing
late || step moecheck timeout -k 10 900 $B moecheck relu2-22of512 relu2-rows-22of512
late || step topkcheck timeout -k 10 300 $B topkcheck
# a slow host stops here, not at the cap: 8 min in, the weights must project to be on disk 45 min after the start
while kill -0 $DL 2> /dev/null && [ $(( $(date +%s) - T0 )) -lt 480 ]; do sleep 15; done
if kill -0 $DL 2> /dev/null; then
    t=$(( $(date +%s) - T0 )); b=$(du -sbc $M ~/v/hf 2> /dev/null | tail -1 | cut -f1)
    [ $(( ${b:-0} / (t + 1) * 2700 )) -ge 247000000000 ] || { echo "download: $(( ${b:-0} / 1000000 / (t + 1) )) MB/s after $(( t / 60 )) min, too slow for the weights in 45 min: stopped" >> $P; exit 1; }
fi
wait $DL || { echo "download: failed" >> $P; exit 1; }
df -h ~ > $R/disk-after-download.txt 2>&1

# 2. the pack: the whole checkpoint at the card's level ($LEVEL, --layout $LAYOUT; at wren the experts in four-bit codes and every other matrix in six-bit ones), the plan of src/pack.rs
late || step pack timeout -k 10 4200 $B pack $M $PK --layout $LAYOUT --threads $CPUS
PACK_S=${LAST_S:-4200}
if [ -f $PK/glyd.json ]; then
    { du -sb $PK; ls $PK | head -40; $PY -I -c "import json,sys,collections; m=json.load(open(sys.argv[1])); print(m['format'], m['layout'], m.get('level'), len(m['packs']), 'packs', dict(collections.Counter(e['layout'] for e in m['packs'].values())), len(m['tensors']), 'tensors saved as they are')" $PK/glyd.json; } > $R/pack.txt 2>&1
else
    echo "pack: no glyd.json: the steps that need the packed directory are skipped" >> $P
fi

if [ -f $PK/glyd.json ]; then
    # 3. every pack decoded by the GPU library against glyd.json's sha256
    late || step verify timeout -k 10 2400 $G verify $PK --device cuda:0 --threads $CPUS
    # 4. the engine's own decode speed (greedy steps, the CUDA graph) and its memory
    late || step generate timeout -k 10 1500 $B generate $PK "The capital of France is" --new 128 --ctx 4096
    # 5. the whole model's logits on 160 tokens of a text against the bf16 checkpoint
    late || {
        step ids timeout -k 10 300 $PY -I $REF ids $M ~/v/logits --prefill 64 --total 160 --level $LV
        step logits-run timeout -k 10 3000 $B nhtoy $PK ~/v/logits
        step logits-check timeout -k 10 3300 $PY -I $REF full $M ~/v/logits
        mkdir -p $R/logits; cp ~/v/logits/toy.txt ~/v/logits/ids.txt ~/v/logits/toy_*_tokens.txt $R/logits/ 2> /dev/null
    }
    # 6. layers of the packed directory against the reference on the weights the packs hold
    late || {
        step decode timeout -k 10 1500 $B nhdecode $PK ~/v/dec --layers 0,1,7,36,37,40 --tokens 333
        ln -sf ~/v/dec/x.bin $PK/x.bin   # (nhlayers reads its input beside the packed model's files)
        step layers-gen timeout -k 10 300 $PY -I $REF layers gen ~/v/dec ~/v/io --level $LV
        step layers-run timeout -k 10 1800 $B nhlayers $PK ~/v/io
        step layers-check timeout -k 10 2700 $PY -I $REF layers check ~/v/dec ~/v/io
        mkdir -p $R/layers; cp ~/v/io/spec.json $R/layers/ 2> /dev/null
    }
    # 7. the CLI: one answer, then a chat of three prompts (each answer's tokens a second)
    late || step run-prompt timeout -k 10 1500 $B run $PK --prompt "What is the capital of France? Answer in one short sentence." --no-think
    printf '%s\n' "What is the capital of France? Answer in one short sentence." "Write a Python function that returns the n-th Fibonacci number, with a docstring." "Explain in two sentences why the sky is blue." "/bye" > $R/chat-in.txt
    late || step run-chat timeout -k 10 1800 bash -c "$B run $PK --chat --no-think < $R/chat-in.txt"
    # 7b. sampled decode against greedy on one prompt: the sampled steps' candidates are the GPU's (nhtoy, step 5, has checked them against the row), and the speed of each way, the host's included, is measured in one load
    late || step sampled timeout -k 10 1500 $B sampletime $PK "The capital of France is" --new 256 --ctx 4096
    # 8. the API: three completions, and one user's speed
    late || {
        cat > ~/v/serve_chat.py <<'PY'
import json, time, urllib.request
for q in ["What is the capital of France? Answer in one short sentence.", "Write a Python function that returns the n-th Fibonacci number.", "Explain in two sentences why the sky is blue."]:
    body = {"model": "nemotron", "messages": [{"role": "user", "content": q}], "max_tokens": 200, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    print("Q:", q, "\nA:", r["choices"][0]["message"]["content"], "\n(%d tokens in %.1f s)\n" % (r["usage"]["completion_tokens"], time.time() - t))
PY
        $B serve $PK --port 8000 > $R/serve.log 2>&1 &
        SERVE=$!
        for _ in $(seq 1 200); do curl -sf http://127.0.0.1:8000/health > /dev/null && break; kill -0 $SERVE 2> /dev/null || break; sleep 6; done
        if curl -sf http://127.0.0.1:8000/health > /dev/null; then
            step serve-chat timeout -k 10 900 $PY -I ~/v/serve_chat.py
            step speed timeout -k 10 1500 $PY -I $SRC/tools/runpod/speed1.py http://127.0.0.1:8000 nemotron $R/speed.json --rounds 3 --prompt-tokens 1024 --new 256
            nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > $R/serve-mem.txt 2>&1
        else
            echo "serve: not up (serve.log)" >> $P
        fi
        kill $SERVE 2> /dev/null; wait $SERVE 2> /dev/null
    }
fi

# 9. the same card with the experts' down matrices dense: the model packed again with them in the block layout (GLYD_ENGINE_DENSE_DOWN=1), which runs the library's dense product; its generate, its logits against
# the bf16 checkpoint (the sparse directory's ids and prefill) and its speed as step 8 measures it; then the sparse directory with the blocks a pair's neurons are cut in forced (2 and 4; the default is 8 for 22 pairs)
PKD=~/v/packed-dense
if [ -f $PK/glyd.json ]; then
    if [ $(( MAX_MIN * 60 - ($(date +%s) - T0) )) -gt $(( PACK_S * 12 / 10 + 1500 )) ]; then
        step pack-dense env GLYD_ENGINE_DENSE_DOWN=1 timeout -k 10 4200 $B pack $M $PKD --layout $LAYOUT --threads $CPUS
        if [ -f $PKD/glyd.json ]; then
            $PY -I -c "import json,sys,collections; m=json.load(open(sys.argv[1])); print(m['format'], m['layout'], len(m['packs']), 'packs', dict(collections.Counter(e['layout'] for e in m['packs'].values())))" $PKD/glyd.json > $R/pack-dense.txt 2>&1
            late || step generate-dense timeout -k 10 1500 $B generate $PKD "The capital of France is" --new 128 --ctx 4096
            late || {
                mkdir -p ~/v/logits-dense && cp ~/v/logits/ids.txt ~/v/logits/toy.txt ~/v/logits-dense/
                step logits-run-dense timeout -k 10 3000 $B nhtoy $PKD ~/v/logits-dense
                step logits-check-dense timeout -k 10 3300 $PY -I $REF full $M ~/v/logits-dense
                mkdir -p $R/logits-dense; cp ~/v/logits-dense/toy.txt ~/v/logits-dense/ids.txt ~/v/logits-dense/toy_*_tokens.txt $R/logits-dense/ 2> /dev/null
            }
            late || {
                $B serve $PKD --port 8000 > $R/serve-dense.log 2>&1 &
                SERVE=$!
                for _ in $(seq 1 200); do curl -sf http://127.0.0.1:8000/health > /dev/null && break; kill -0 $SERVE 2> /dev/null || break; sleep 6; done
                if curl -sf http://127.0.0.1:8000/health > /dev/null; then
                    step speed-dense timeout -k 10 1500 $PY -I $SRC/tools/runpod/speed1.py http://127.0.0.1:8000 nemotron $R/speed-dense.json --rounds 3 --prompt-tokens 1024 --new 256
                else
                    echo "serve-dense: not up (serve-dense.log)" >> $P
                fi
                kill $SERVE 2> /dev/null; wait $SERVE 2> /dev/null
            }
        fi
        for n in 2 4; do
            late || step generate-split$n env GLYD_ENGINE_SPARSE_SPLIT=$n $B generate $PK "The capital of France is" --new 128 --ctx 4096
        done
    else
        echo "dense: not run, the time left is short of another pack ($(( PACK_S / 60 )) min)" >> $P
    fi
fi

# 10. the layer at the real shape, dense and sparse (random packs of arbitrary bytes, first matrices whose activations are 92% zeros): the GPU is free
late || step layer-speed timeout -k 10 1500 $B moecheck nemotron3 --speed --no-check

# the summary: every check's verdict, the speeds, the memory
{
    echo "Nemotron 3 Super at $LEVEL ($LAYOUT) on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), tree $(cat $SRC/COMMIT 2> /dev/null)"
    echo "--- steps"; cat $P
    echo "--- moecheck (the second product dense and sparse, over Int6, Int5 and Int4 experts, against the CPU reference)"; grep -E "^(ok|FAIL|all|FAILED)" $R/moecheck.log 2> /dev/null
    echo "--- topkcheck (the sampling kernels: a row's 64 largest logits and the sum of its exponentials, against the host's)"; grep -E "kernel|DIFFERENT" $R/topkcheck.log 2> /dev/null
    echo "--- the pack (and the layouts of its packs)"; cat $R/pack.log $R/pack.txt 2> /dev/null
    echo "--- verify (every pack decoded by the GPU library against glyd.json)"; tail -3 $R/verify.log 2> /dev/null
    echo "--- generate: greedy steps, the decode graph (the experts' down matrices in the row layout: the sparse product)"; grep -E "loaded in|prompt tokens" $R/generate.log 2> /dev/null; tail -3 $R/generate.log 2> /dev/null
    echo "--- generate: the same, the blocks a pair's neurons are cut in forced to 2 and to 4, and the dense directory"; grep -H -E "prompt tokens" $R/generate-split2.log $R/generate-split4.log $R/generate-dense.log 2> /dev/null; cat $R/pack-dense.txt 2> /dev/null
    echo "--- the whole model against the bf16 checkpoint (top-1, KL; the sampler's candidates against the row)"; grep -E "^(ok|FAIL|full)|position|toy|prefill|candidates" $R/logits-run.log $R/logits-check.log 2> /dev/null
    echo "--- the same, the dense directory"; grep -E "^(ok|FAIL|full)|position|toy|prefill|candidates" $R/logits-run-dense.log $R/logits-check-dense.log 2> /dev/null
    echo "--- layers of the packed directory against the reference on the weights the packs hold: the verdicts (FAIL lines first)"; grep -E "FAIL|^layers" $R/layers-check.log 2> /dev/null
    echo "--- layers: time in the layer by case"; grep -E "^case " $R/layers-run.log 2> /dev/null
    echo "--- run: one answer"; cat $R/run-prompt.log 2> /dev/null
    echo "--- run: a chat of three prompts"; cat $R/run-chat.log 2> /dev/null
    echo "--- sampled decode against greedy (sampletime): tokens a second by greedy steps, by sampled steps from the GPU's candidates, and from the row on the host"; cat $R/sampled.log 2> /dev/null
    echo "--- serve: three completions"; cat $R/serve-chat.log 2> /dev/null
    echo "--- serve: one user (speed1.py); GPU memory used and total"; cat $R/speed.json $R/serve-mem.txt 2> /dev/null
    echo "--- serve: decode tokens a second for one user, the experts' down matrices sparse and dense, on this card"
    $PY -I -c "import json,sys; a,b=(json.load(open(f)) for f in sys.argv[1:]); x,y=a['decode_tok_s_median'],b['decode_tok_s_median']; print('sparse %.1f, dense %.1f (x%.2f); first token %.0f ms against %.0f ms' % (x,y,x/y,a['ttft_ms_median'],b['ttft_ms_median']))" $R/speed.json $R/speed-dense.json 2>&1 | tail -1
    echo "--- the layer at the real shape, dense and sparse"; grep -E "^(nemotron3|FAIL)|^  T=" $R/layer-speed.log 2> /dev/null
    echo "--- GPU memory used at most in the job (MiB)"; awk -F', ' '$2 + 0 > m { m = $2 + 0 } END { print m }' $R/smi.csv 2> /dev/null
} > $R/summary.txt 2>&1
cat $R/summary.txt | head -250
nvidia-smi > $R/gpu-final.txt 2>&1
