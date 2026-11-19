#!/usr/bin/env bash
# GLM-5.3-Flash on one GPU of this computer, against tools/glm_ref.py (Linear GLY-133): the kernels, the layers by stage, the whole tiny model (with the engine's routers' and indexers' choices run in the reference), the
# draft verification and the MTP head, and, where the reference's real layers are given, the checkpoint's layers one at a time. Not a pod job: it runs where it is started, a step after another, and stops at the first
# step that fails. Needs a python with torch, numpy, safetensors and transformers 5.19.0 (PY), the engine built with nvcc in the same tree (ENGINE) and its GPU library (GLYD_GPU_LIB).
#
#   PY=~/venv/bin/python ENGINE=target/release/glyd-engine GLYD_GPU_LIB=lib/libglyd_gpu_cuda13.so WORK=~/glm-check bash tools/cards/glm_pc_check.sh
#
# GPU_RUN: a prefix for every step that uses the GPU (a shared machine: GPU_RUN="flock ~/gpu.lock"). LEVELS: the bits the tiny model's experts are held in (default "6 5 4"). SNAPSHOT, REF: the real checkpoint's directory
# and the reference of research/glm-data's ref128 (a directory with final_ref128.safetensors and layers/ref128_layerNN.safetensors); with both, the real model's layers RANGE (FIRST:END, default 0:8) run one at a time at the
# first of LEVELS, and the real MTP head on the reference's hidden states; they are 6 GB of experts a layer on the GPU and read 7 GB of FP8 from the checkpoint.
set -eu
PY=${PY:?PY: a python with torch and transformers 5.19.0}
ENGINE=${ENGINE:?ENGINE: the glyd-engine binary}
WORK=${WORK:-$PWD/glm-check}
LEVELS=${LEVELS:-"6 5 4"}
GPU_RUN=${GPU_RUN:-}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
REF=${REF:-}
SNAPSHOT=${SNAPSHOT:-}
REF_PY="$PY -I $HERE/glyd-engine/tools/glm_ref.py"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1   # (the checkpoint is ungated: nothing reads or sends a token)
mkdir -p "$WORK"
step() { local n=$1; shift; echo "== $n"; "$@" 2>&1 | grep -v '^\[transformers\]' | tee "$WORK/$n.log"; test "${PIPESTATUS[0]}" -eq 0 || { echo "step $n failed (see $WORK/$n.log)"; exit 1; }; }

# the reference against transformers' own functions, modules and model (CPU)
step selftest $REF_PY selftest

# the kernels on random inputs: a small shape, the model's (64 heads, 2,048 tokens; the latent attention at 2,600 rows with the indexer choosing among 650 pools) and an edge one (latents of 192, no tail: the first rows pick no token)
for shape in "small:--heads 4 --n 150 --split 70 --d 512 --seed 1 --mla small" "model:--heads 64 --n 2048 --split 1000 --d 4096 --seed 2 --mla model" "edge:--heads 4 --n 150 --split 70 --d 512 --seed 4 --mla edge"; do
    n=${shape%%:*}; a=${shape#*:}
    step kernels-$n-gen $REF_PY kernels gen "$WORK/k-$n" $a
    step kernels-$n-run $GPU_RUN "$ENGINE" glmkernels "$WORK/k-$n/in" "$WORK/k-$n/out"
    step kernels-$n-check $REF_PY kernels check "$WORK/k-$n"
done

# the random tiny model (eight layers in the real model's pattern, the first MTP layer after them) at each level: its layers by stage, the whole model, a verified draft, the MTP head
for lv in $LEVELS; do
    T=$WORK/tiny$lv
    step tiny$lv-make $REF_PY tiny "$T" --level $lv
    step tiny$lv-layers-gen $REF_PY layers gen "$T" "$T/io" --level $lv
    step tiny$lv-layers-run $GPU_RUN "$ENGINE" glmlayers "$T" "$T/io"
    step tiny$lv-layers-check $REF_PY layers check "$T" "$T/io"
    step tiny$lv-toy-run $GPU_RUN "$ENGINE" glmtoy "$T/toy" "$T/toy"
    step tiny$lv-toy-check $REF_PY toy "$T"
    step tiny$lv-spec $GPU_RUN "$ENGINE" glmspec "$T/toy" "$T/toy"
    step tiny$lv-mtp-run $GPU_RUN "$ENGINE" glmmtp "$T/toy" "$T/toy"
    step tiny$lv-mtp-check $REF_PY mtp "$T"
done

# a decode step of the real layers, stage by stage in a graph's replay (four real layers by range: the Hub, no token)
step real-fetch $REF_PY fetch "$WORK/real" --layers 0,3,4,43 --experts 16 --tokens 333 --seed 1
step real-gen $REF_PY layers gen "$WORK/real" "$WORK/real/io" --level 6
step real-run $GPU_RUN "$ENGINE" glmlayers "$WORK/real" "$WORK/real/io"
step real-decode $GPU_RUN "$ENGINE" glmdecode "$WORK/real" "$WORK/dec6" --layers 3,4,43 --layout i6
step real-check $REF_PY layers check "$WORK/real" "$WORK/real/io" --decoded "$WORK/dec6"
step real-prof $GPU_RUN "$ENGINE" glmprof "$WORK/real" --layers 0,3,4,43 --ctx 2600 --reps 30

# the checkpoint's layers one at a time, and its MTP head, against the reference of the real model
if [ -n "$SNAPSHOT" ] && [ -n "$REF" ]; then
    lv=${LEVELS%% *}; RANGE=${RANGE:-0:8}
    step chain-run $GPU_RUN "$ENGINE" glmreal "$SNAPSHOT" "$REF" "$WORK/chain" --layers $RANGE --layout i$lv
    step chain-check $REF_PY chain "$WORK/chain" "$REF" --layers $RANGE
    step mtpreal-run $GPU_RUN "$ENGINE" glmmtpreal "$SNAPSHOT" "$REF" "$WORK/mtpreal" --layout i$lv
    step mtpreal-check $REF_PY mtpreal "$WORK/mtpreal" "$REF"
fi
echo "glm_pc_check: all steps ok"
