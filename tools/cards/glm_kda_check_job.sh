#!/usr/bin/env bash
# GLM-5.3-Flash's Kimi delta attention, hyper-connection (mHC), dense MLP and mixture-of-experts layers on a GPU, against PyTorch (Linear GLY-133; rp_run.sh contract: results in ~/results, ~/results/DONE at the end; one A100 80 GB pod).
# Builds the library and the engine from the tarball, reads three layers of zai-org/GLM-5.3-Flash (pinned revision eb9eb208) from the Hub by HTTP range (layers 0 and 1: Kimi delta attention and a dense MLP; layer 44: Kimi delta
# attention and its 288 experts; about 8.4 GB in all, no whole shard, no token), and runs, each against tools/glm_ref.py's reference:
#   1. selftest   the reference against transformers 5.19.0's own functions, modules and model (CPU): its delta-rule functions, the hyper-connection, the Kimi delta mixer, the MLP, the router and the experts, the decoder layer, a
#                 random tiny model whole, and that model loaded by from_pretrained with its FP8 matrices dequantised
#   2. emu        kernels/glm.cu on the host (tools/glm_emu.cpp, any computer) against the reference
#   3. kernels    glyd-engine glmkernels: the kernels on random inputs, at a small shape and the model's (64 heads, streams of 4,096, 2,048 tokens: the chunked delta rule's time and the recurrence's)
#   4. tiny       for each level (6: six-bit codes, 5: five-bit, 4: four-bit) a random tiny GLM (glm_ref.py tiny: no Hub read; four layers of Kimi delta attention, a dense MLP whose matrices are FP8, then experts, shared experts
#                 and a router; the experts and the shared experts on the grid of the layouts the engine holds them in, so that its packs hold them exactly), then
#                 layers  glyd-engine glmlayers: all four layers through the engine's own code, both halves of each: the prompt in one call and in odd pieces, the delta rule token by token, single steps one by one and replayed
#                         from CUDA graphs, a prompt then steps; every stage (mHC's weights, the collapse, q k v, the convolution, the gates, the delta rule, the gated norm, the streams after each half, the MLP's stages, the
#                         router's choices and weights against the reference's, the experts and the shared expert)
#                 toy     glyd-engine glmtoy on the whole tiny model: prefill in two chunks and decode steps, as a graph and not, the logits of every step (the median position within 5e-2 of the reference's largest logit,
#                         the top-1 token the reference's at 98% of the positions)
#   5. fp8        the FP8 matrices of layers 1 and 44 (the dense MLP, the shared expert, one expert) as Checkpoint::read_bf16 reads them, against transformers' Fp8Dequantize, bit for bit
#   6. real       the reference against transformers' modules on the three real layers' weights (CPU); glmdecode (the layer 44's experts packed at six bits and decoded, the weights the engine multiplies by) and glyd-engine
#                 glmlayers on layers 0, 1 and 44 with their real weights through the engine's own code, the stages checked on the decoded experts
# The latent attention layers (11 of the 45) are the next round: nothing here runs the model whole. Uploaded beside it: src.tar (git archive of the tree, prefix src/). Results: summary.txt (made last), steps.txt, one .log a
# step, emu/, kernels-*/, tiny*/, real/, gpu.txt (rp_run.sh).
#
#   cd <the tree at the commit to test> && W=$(mktemp -d) && git archive --format=tar.gz --prefix=src/ HEAD > $W/src.tar && \
#   NAME=glm-kda-check MAX_MIN=75 UPLOAD="$W/src.tar" bash tools/runpod/rp_class.sh a100 tools/cards/glm_kda_check_job.sh        # $1.79/h; about 55 minutes, $1.7 (the cap, 75 minutes: $2.24); the estimate is the other jobs' steps added up, the job not launched
# (rp_guard.sh deletes glyd-* pods older than its MAX_AGE: MAX_AGE=100 bash tools/runpod/rp_guard.sh & first.) Results come to $GLYD_RUNS/results-glm-kda-check.
set -u
R=~/results; mkdir -p $R ~/v; cd ~/v; tar -xf ~/src.tar
P=$R/steps.txt; : > $P
T0=$(date +%s)
MAX_MIN=${JOB_MAX_MIN:-75}   # (the launcher's cap, MAX_MIN=75 in the launch line above, counts the pod's start too: the job stops starting steps 8 minutes before its own)
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; local rc=$?; echo "$n: rc=$rc $(( $(date +%s) - t )) s (job minute $(( ($(date +%s) - T0) / 60 )))" >> $P; return $rc; }
late() { [ $(( ($(date +%s) - T0) / 60 )) -ge $((MAX_MIN - 8)) ]; }
trap 'touch $R/DONE' EXIT
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN   # (the repo is ungated: nothing here reads or sends a token)
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1 RUST_BACKTRACE=1
SRC=~/v/src; E=$SRC/glyd-engine; REF=$E/tools/glm_ref.py
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
# the CPUs this container really has (a cgroup quota, not the host's cores): torch's thread pools and the packer's threads are sized by it
cpus() { local q p c=${CGROOT:-/sys/fs/cgroup}; { read -r q p < $c/cpu.max; } 2> /dev/null && [ "$q" != max ] && { echo $(( (q + p - 1) / p )); return; }
         { read -r q < $c/cpu/cpu.cfs_quota_us; read -r p < $c/cpu/cpu.cfs_period_us; } 2> /dev/null && [ "${q:--1}" -gt 0 ] && { echo $(( (q + p - 1) / p )); return; }; nproc; }
CPUS=$(cpus); export OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS
# the layer of 288 experts is packed twice (the engine's load, glmdecode) from 7.3 GB of FP8, bound by the CPUs, beside the builds: on fewer than 8 the steps do not fit the cap
[ "$CPUS" -ge 8 ] || { echo "cpus: $CPUS, too few to pack a layer of 288 experts and build in time: stopped (launch again for another host)" >> $P; exit 1; }
{ TZ=EST5EDT date; nvidia-smi; nvcc --version | tail -2; echo "arch sm_$ARCH, $CPUS CPUs of $(nproc)"; free -g | head -2; df -h ~ | tail -1; cat $SRC/COMMIT 2> /dev/null; } > $R/machine.txt 2>&1

# the Python environment (torch is the template's where it has one), the Rust toolchain, the library and the engine, at the same time (every step has a time limit: a kernel that hangs ends its step, not the job)
( step pyenv timeout -k 10 1500 bash -c 'python3 -m venv --system-site-packages ~/v/venv && ( ~/v/venv/bin/python -c "import torch" 2> /dev/null || ~/v/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128 ) && ~/v/venv/bin/pip install -q numpy safetensors "transformers==5.19.0"' ) &
PYENV=$!
step rustup timeout -k 10 600 bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1'
. ~/.cargo/env
( step lib timeout -k 10 2400 bash -c "cd $SRC && bash gpu/build_lib.sh ~/v/lib" ) &
LIB=$!
[ -n "$ARCH" ] && export GLYD_ENGINE_ARCHS=$ARCH   # (this GPU alone)
step engine timeout -k 10 2400 bash -c "cd $SRC && cargo build --release -p glyd-engine"
wait $PYENV
PY=~/v/venv/bin/python
B=$SRC/target/release/glyd-engine
step versions bash -c "$PY -c 'import torch, transformers; print(\"torch\", torch.__version__, torch.version.cuda, \"transformers\", transformers.__version__, torch.cuda.get_device_name())'; $B --version"
# the real layers, by range from the Hub, in the background (7.3 GB of them are the 288 experts of layer 44): they are needed from step 5 on
( step fetch timeout -k 10 1800 $PY -I $REF fetch ~/v/real --layers 0,1,44 --tokens 333 --seed 1 ) &
FETCH=$!
step selftest timeout -k 10 900 $PY -I $REF selftest

# 2. the kernels on the host (any computer: threads and barriers; 100 tokens, 2 heads, streams of 256)
step emu-build timeout -k 10 600 c++ -std=c++20 -O1 -pthread -fno-strict-aliasing -o ~/v/glm_emu $E/tools/glm_emu.cpp
step emu-gen timeout -k 10 300 $PY -I $REF kernels gen ~/v/k-emu --heads 2 --n 100 --split 45 --d 256 --seed 1
step emu-run timeout -k 10 1200 ~/v/glm_emu ~/v/k-emu/in ~/v/k-emu/out
step emu-check timeout -k 10 600 $PY -I $REF kernels check ~/v/k-emu
wait $LIB
export GLYD_GPU_LIB=$(ls ~/v/lib/libglyd_gpu_cuda*.so 2> /dev/null | head -1)
[ -x $B ] || { echo "no engine: see engine.log" >> $P; exit 1; }
[ -n "$GLYD_GPU_LIB" ] || echo "no library (see lib.log): the kernel checks need none, the layers and the toy models do" >> $P

# 3. the kernels on random inputs: a small shape, and the model's (64 heads of 128 channels, streams of 4,096, a prompt of 2,048 tokens in two calls)
for cfg in "small:--heads 4 --n 150 --split 70 --d 512 --seed 1" "model:--heads 64 --n 2048 --split 1000 --d 4096 --seed 2"; do
    n=${cfg%%:*}; a=${cfg#*:}
    late && { echo "kernels-$n: skipped, late" >> $P; continue; }
    step kernels-$n-gen timeout -k 10 900 $PY -I $REF kernels gen ~/v/k-$n $a
    step kernels-$n-run timeout -k 10 900 $B glmkernels ~/v/k-$n/in ~/v/k-$n/out
    step kernels-$n-check timeout -k 10 1500 $PY -I $REF kernels check ~/v/k-$n
    mkdir -p $R/kernels-$n; cp ~/v/k-$n/in/*.txt $R/kernels-$n/ 2> /dev/null
done

# 4. the random tiny model at each level: its four layers by stage, then the whole model
for lv in 6 5 4; do
    late && { echo "tiny$lv: skipped, late" >> $P; continue; }
    T=~/v/tiny$lv
    step tiny$lv-make timeout -k 10 300 $PY -I $REF tiny $T --level $lv
    step tiny$lv-layers-gen timeout -k 10 300 $PY -I $REF layers gen $T $T/io --level $lv
    step tiny$lv-layers-run timeout -k 10 900 $B glmlayers $T $T/io
    step tiny$lv-layers-check timeout -k 10 900 $PY -I $REF layers check $T $T/io
    step tiny$lv-toy-run timeout -k 10 900 $B glmtoy $T/toy $T/toy
    step tiny$lv-toy-check timeout -k 10 600 $PY -I $REF toy $T
    mkdir -p $R/tiny$lv; cp $T/io/spec.json $R/tiny$lv/ 2> /dev/null; cp $T/toy/toy.txt $R/tiny$lv/ 2> /dev/null
done

# 5 and 6. the real layers (the fetch above is waited for here)
wait $FETCH
M=~/v/real
if [ -f $M/x.bin ] && ! late; then
    # the FP8 matrices as the engine reads them against transformers' dequantisation: bit for bit (the dense MLP of layer 1, the shared expert and one expert of layer 44)
    step fp8-engine timeout -k 10 900 $B glmdequant $M ~/v/fp8out --match layers.1.mlp.,layers.44.mlp.shared,layers.44.mlp.experts.7.
    step fp8-check timeout -k 10 900 $PY -I $REF fp8 check $M ~/v/fp8out
    # the reference against transformers' modules on the real weights
    step real-vs-transformers timeout -k 10 1200 $PY -I $REF real $M
    # the layer of experts as the engine holds it at six bits, decoded: what its products multiply by
    step real-decode timeout -k 10 1800 $B glmdecode $M ~/v/dec6 --layers 44 --layout i6
fi
if [ -f $M/x.bin ] && [ -d ~/v/dec6 ] && ! late && [ -n "$GLYD_GPU_LIB" ]; then
    step real-layers-gen timeout -k 10 300 $PY -I $REF layers gen $M $M/io --level 6
    step real-layers-run timeout -k 10 2400 $B glmlayers $M $M/io
    step real-layers-check timeout -k 10 1800 $PY -I $REF layers check $M $M/io --decoded ~/v/dec6
    mkdir -p $R/real; cp $M/io/spec.json $R/real/ 2> /dev/null; cp $M/config.json $R/ 2> /dev/null
fi

# the summary: every check's verdict, the kernels' and layers' times
{
    echo "GLM-5.3-Flash's Kimi delta attention, mHC, MLP and experts on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), tree $(cat $SRC/COMMIT 2> /dev/null)"
    echo "--- steps"; cat $P
    echo "--- selftest (the reference against transformers)"; grep -E "^(ok|FAIL|selftest)" $R/selftest.log 2> /dev/null
    echo "--- kernels/glm.cu on the host"; grep -E "^(ok|FAIL|kernels)" $R/emu-check.log 2> /dev/null
    for n in small model; do echo "--- kernels on the GPU, $n (times of the delta rule's cases are the engine's own)"; grep -E "^(ok|FAIL|kernels|delta rule case|streams)" $R/kernels-$n-run.log $R/kernels-$n-check.log 2> /dev/null; done
    for lv in 6 5 4; do
        echo "--- tiny model, level $lv: layers by stage (verdicts, FAIL lines first)"; grep -E "FAIL|^layers" $R/tiny$lv-layers-check.log 2> /dev/null
        echo "--- tiny model, level $lv: time in the layer by case"; grep -E "^case " $R/tiny$lv-layers-run.log 2> /dev/null
        echo "--- tiny model, level $lv: the whole model"; grep -E "^(graph|eager):|^(ok|FAIL|toy)|top-1" $R/tiny$lv-toy-run.log $R/tiny$lv-toy-check.log 2> /dev/null
    done
    echo "--- FP8 matrices: the engine's bf16 against transformers' Fp8Dequantize"; grep -E "^(ok|FAIL)" $R/fp8-check.log 2> /dev/null
    echo "--- real layers: the reference against transformers' modules"; grep -E "^(ok|FAIL|real)" $R/real-vs-transformers.log 2> /dev/null
    echo "--- real layers through the engine: the verdicts (FAIL lines first)"; grep -E "FAIL|^layers" $R/real-layers-check.log 2> /dev/null
    echo "--- real layers: time in the layer by case"; grep -E "^case " $R/real-layers-run.log 2> /dev/null
    echo "--- real layers: all"; cat $R/real-layers-check.log 2> /dev/null
} > $R/summary.txt 2>&1
cat $R/summary.txt | head -200
nvidia-smi > $R/gpu-final.txt 2>&1
