#!/usr/bin/env bash
# The pre-packed saves, made and checked on one GPU pod (an 80 GB GPU: the largest save is packed on the GPU whole).
# For each model, in turn (the next one's download beside it): the bf16 checkpoint downloaded, `python -m glyd_gpu pack`
# (the tiered layout, glyd-v4; every pack checked against its weights as it is made), check.py (the save loaded as saved
# with verify, one chat answer, sizes), `glyd verify` (the CLI pip installs); then source and save deleted
# unless KEEP=1. In ~: this, check.py (card.py and upload.py with UPLOAD_ORG), and src.tar (glyd_gpu/ with
# libglyd_gpu_cuda{12,13}.so beside it) or, without it, glyd-gpu from PyPI (GLYD_GPU, default GLYD_TAG's version).
# UPLOAD_ORG=glyd: each save that passes check.py and glyd verify uploaded (upload.py; the token from publish.sh).
#   MODELS="Qwen/Qwen3.5-4B ..." bash job.sh        results in ~/results, ~/results/DONE at the end
set -u
R=$HOME/results; W=$HOME/w; mkdir -p "$R" "$W/save"
exec > >(tee -a "$R/job.log") 2>&1
trap 'touch "$R/DONE"' EXIT
MODELS=${MODELS:-"Qwen/Qwen3.5-4B google/gemma-4-E4B-it Qwen/Qwen3.5-9B google/gemma-4-12B-it Qwen/Qwen3.5-27B Qwen/Qwen3.8-27B google/gemma-4-31B-it"}
GLYD_TAG=${GLYD_TAG:-v0.29.0}
T0=$(date +%s)
step() { echo "== $(date -u +%T) (+$(( $(date +%s) - T0 )) s) $*"; }

step machine
{ nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv; nproc; free -g | head -2; df -h "$HOME" | tail -1; } | tee "$R/machine.txt"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN PYTHONPATH GLYD_GPU_LIB
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$W/hf HF_XET_HIGH_PERFORMANCE=1  # (public models: no token read or sent)

step install
python3 -m venv --system-site-packages "$W/venv" && . "$W/venv/bin/activate" || { echo "no venv"; exit 1; }  # (the image's torch; its system pip is externally managed)
python -c "import torch" 2> /dev/null || pip install -q torch 2>&1 | tail -3
pip install -q "transformers>=5.17,<6" "accelerate>=1.1" "safetensors>=0.8" "huggingface_hub[hf_xet]>=1.5,<2" "glyd==${GLYD_TAG#v}" 2>&1 | tail -3
if [ -f "$HOME/src.tar" ]; then  # a tree's package with its libraries; else the release as pip installs it
  mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/src.tar" && export PYTHONPATH=$W/src
else
  pip install -q "glyd-gpu==${GLYD_GPU:-${GLYD_TAG#v}}" 2>&1 | tail -3
fi
python -c "import torch, transformers, huggingface_hub, glyd_gpu; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'hub', huggingface_hub.__version__, 'glyd_gpu', glyd_gpu.__version__, glyd_gpu.__file__)" > "$R/versions.txt" || { echo "the packages do not import"; exit 1; }
GLYD=$(command -v glyd); echo "glyd CLI: $GLYD ($("$GLYD" --version 2>&1 | head -1)); glyd-gpu beside it: $(ls "$(dirname "$GLYD")"/glyd-gpu 2> /dev/null || echo none)" >> "$R/versions.txt"
command -v hf > /dev/null || { echo "no hf CLI"; exit 1; }
cat "$R/versions.txt"

src_dir() { python -c "from huggingface_hub import snapshot_download as s; print(s('$1', local_files_only=True))"; }
fetch() { hf download "$1" > "$R/dl-${1##*/}.log" 2>&1; }

if [ -n "${UPLOAD_ORG:-}" ]; then  # the token, once, from a FIFO that publish.sh writes over ssh: held in this shell only, never in a file
  mkfifo -m 600 "$HOME/hf_token"; exec 3<> "$HOME/hf_token"; read -t 1800 -r HF_UP <&3; exec 3>&-; rm -f "$HOME/hf_token"
  [ -n "${HF_UP:-}" ] || { echo "no token within 30 minutes: nothing uploaded"; UPLOAD_ORG=; }
fi
set -- $MODELS
fetch "$1" & DL=$!
while [ $# -gt 0 ]; do
  M=$1; N=${M##*/}; shift
  wait "$DL"; rc=$?
  [ $# -gt 0 ] && { fetch "$1" & DL=$!; }
  step "$M: downloaded (exit $rc)"
  [ "$rc" = 0 ] || { echo "$M: download failed: $(tail -3 "$R/dl-$N.log")"; continue; }
  SRC=$(src_dir "$M"); OUT=$W/save/$N; rm -rf "$OUT"
  step "$M: pack"
  timeout 2700 python -m glyd_gpu pack "$M" "$OUT" > "$R/pack-$N.log" 2>&1; rc=$?
  grep -E "packed and checked|Error|error:" "$R/pack-$N.log" | tail -4
  if [ "$rc" = 0 ]; then
    cp "$OUT/glyd.json" "$R/glyd-$N.json"; cp "$OUT/config.json" "$R/config-$N.json"; ls -la "$OUT" > "$R/files-$N.txt"
    step "$M: check (verify, chat, sizes)"
    timeout 1800 python "$HOME/check.py" "$SRC" "$OUT" "$R/check-$N.json" > "$R/check-$N.log" 2>&1 || echo "$M: check failed: $(tail -5 "$R/check-$N.log")"
    tail -3 "$R/check-$N.log"
    if [ -n "${GLYD:-}" ]; then
      step "$M: glyd verify (the CLI)"
      timeout 1800 "$GLYD" verify "$OUT" > "$R/cli-verify-$N.txt" 2>&1; echo "exit $?" >> "$R/cli-verify-$N.txt"; tail -2 "$R/cli-verify-$N.txt"
    fi
    if [ -n "${UPLOAD_ORG:-}" ] && grep -q '"answer"' "$R/check-$N.json" 2> /dev/null && grep -q "^exit 0" "$R/cli-verify-$N.txt" 2> /dev/null; then
      step "$M: upload to $UPLOAD_ORG/$N"
      python "$HOME/card.py" "$R/check-$N.json" "${ON:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), $(date -u +%F)}" ${SERVE:+--serve} > "$R/README-$N.md"
      LIC=$SRC/LICENSE; [ -f "$LIC" ] || { curl -fsSL https://www.apache.org/licenses/LICENSE-2.0.txt -o "$W/LICENSE-2.0.txt"; LIC=$W/LICENSE-2.0.txt; }
      HF_TOKEN=$HF_UP python "$HOME/upload.py" "$OUT" "$UPLOAD_ORG/$N" "$R/README-$N.md" "$LIC" ${PUBLIC:+--public} 2>&1 | tail -3
    fi
  else
    echo "$M: pack failed (exit $rc): $(grep -v '^\s' "$R/pack-$N.log" | tail -4)"
  fi
  [ "${KEEP:-0}" = 1 ] || rm -rf "$OUT" "$HF_HOME/hub/models--${M//\//--}"
  df -h "$HOME" | tail -1
done
step "all done"
