#!/usr/bin/env bash
# Saves of 2026 models before and after the table packs (glyd_gpu of v0.29.0, then with format.py, hf.py and _cli.py of
# the branch save-coverage-2026), on one GPU pod. Each model: `python -m glyd_gpu pack` (verify on: every pack checked as
# it is made), check.py on the save (loaded as saved with verify, one greedy answer, GPU bytes, sizes), for both; for
# the branch also `python -m glyd_gpu verify` on its save, its load of the v0.29.0 save, and the source loaded as a bf16
# checkpoint packed as it loads; v0.29.0 reading the branch's glyd.json (refused by the format). First the branch's GPU
# tests of saves. Results in ~/results, ~/results/DONE at the end.
set -u
R=$HOME/results; W=$HOME/w; mkdir -p "$R" "$W"
exec > >(tee -a "$R/job.log") 2>&1
trap 'touch "$R/DONE"' EXIT
MODELS=${MODELS:-"google/gemma-4-E2B-it google/gemma-4-E4B-it Qwen/Qwen3.5-9B google/gemma-4-12B-it"}
T0=$(date +%s)
step() { echo "== $(date -u +%T) (+$(( $(date +%s) - T0 )) s) $*"; }

step machine
{ nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv; nproc; free -g | head -2; df -h "$HOME" | tail -1; } | tee "$R/machine.txt"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN PYTHONPATH GLYD_GPU_LIB
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$W/hf HF_XET_HIGH_PERFORMANCE=1  # (public models: no token read or sent)

step install
python3 -m venv --system-site-packages "$W/venv" && . "$W/venv/bin/activate" || { echo "no venv"; exit 1; }
python -c "import torch" 2> /dev/null || pip install -q torch 2>&1 | tail -3
pip install -q "transformers>=5.17,<6" "accelerate>=1.1" "safetensors>=0.8" "huggingface_hub[hf_xet]>=1.5,<2" 2>&1 | tail -3
mkdir -p "$W/before" "$W/after" && tar -C "$W/before" -xf "$HOME/before.tar" && tar -C "$W/after" -xf "$HOME/after.tar" || { echo "no packages"; exit 1; }
for side in before after; do
  PYTHONPATH=$W/$side python -c "import torch, transformers, glyd_gpu; print('$side', 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'glyd_gpu', glyd_gpu.__version__, glyd_gpu.__file__)" >> "$R/versions.txt" || { echo "$side does not import"; exit 1; }
done
cat "$R/versions.txt"
command -v hf > /dev/null || { echo "no hf CLI"; exit 1; }

step "the branch's GPU tests of saves"
(cd "$W/after/tests" && PYTHONPATH=$W/after timeout 1200 python -c "
import test_gpu as t
for n in ('test_manifest_and_names', 'test_check_files', 'test_tables_saved_packed', 'test_cli_pack_and_verify', 'test_packed_weight_view'):
    getattr(t, n)(); print(n, 'ok', flush=True)") > "$R/tests.txt" 2>&1; echo "exit $?" >> "$R/tests.txt"
tail -8 "$R/tests.txt"

src_dir() { python -c "from huggingface_hub import snapshot_download as s; print(s('$1', local_files_only=True))"; }
fetch() { hf download "$1" > "$R/dl-${1##*/}.log" 2>&1; }
set -- $MODELS
fetch "$1" & DL=$!
while [ $# -gt 0 ]; do
  M=$1; N=${M##*/}; shift
  wait "$DL"; rc=$?
  [ $# -gt 0 ] && { fetch "$1" & DL=$!; }
  step "$M: downloaded (exit $rc)"
  [ "$rc" = 0 ] || { echo "$M: download failed: $(tail -3 "$R/dl-$N.log")"; continue; }
  SRC=$(src_dir "$M")
  for side in before after; do
    OUT=$W/save-$side; rm -rf "$OUT"
    step "$M: pack ($side)"
    PYTHONPATH=$W/$side timeout 2700 python "$HOME/measure.py" "$R/$N-$side-pack.json" pack "$SRC" "$OUT" > "$R/$N-$side-pack.log" 2>&1; echo "exit $?"; grep -E "packed and checked|Error|error:" "$R/$N-$side-pack.log" | tail -3; cat "$R/$N-$side-pack.json" 2> /dev/null; echo
    [ -f "$OUT/glyd.json" ] || continue
    cp "$OUT/glyd.json" "$R/$N-$side-glyd.json"; ls -la "$OUT" > "$R/$N-$side-files.txt"
    step "$M: check ($side save, $side code)"
    PYTHONPATH=$W/$side timeout 1800 python "$HOME/check.py" "$SRC" "$OUT" "$R/$N-$side-check.json" > "$R/$N-$side-check.log" 2>&1 || echo "check failed: $(tail -5 "$R/$N-$side-check.log")"
    tail -2 "$R/$N-$side-check.log"
  done
  if [ -f "$W/save-after/glyd.json" ]; then
    step "$M: verify (branch)"
    PYTHONPATH=$W/after timeout 1800 python "$HOME/measure.py" "$R/$N-after-verify.json" verify "$W/save-after" > "$R/$N-after-verify.log" 2>&1; echo "exit $?"; tail -1 "$R/$N-after-verify.log"
    step "$M: v0.29.0 reads the branch's glyd.json"
    PYTHONPATH=$W/before python -c "from glyd_gpu import format as f; f.read_manifest('$W/save-after'); print('read')" > "$R/$N-before-reads-after.txt" 2>&1; tail -1 "$R/$N-before-reads-after.txt"
  fi
  if [ -f "$W/save-before/glyd.json" ]; then
    step "$M: the branch loads the v0.29.0 save"
    PYTHONPATH=$W/after timeout 1800 python "$HOME/check.py" "$SRC" "$W/save-before" "$R/$N-after-loads-before.json" > "$R/$N-after-loads-before.log" 2>&1 || echo "failed: $(tail -5 "$R/$N-after-loads-before.log")"
    tail -1 "$R/$N-after-loads-before.log"
  fi
  rm -rf "$W/save-before" "$W/save-after"
  step "$M: the bf16 checkpoint packed as it loads (branch)"
  PYTHONPATH=$W/after timeout 1800 python "$HOME/check.py" "$SRC" "$SRC" "$R/$N-source-check.json" > "$R/$N-source-check.log" 2>&1 || echo "failed: $(tail -5 "$R/$N-source-check.log")"
  tail -1 "$R/$N-source-check.log"
  rm -rf "$HF_HOME/hub/models--${M//\//--}"
  df -h "$HOME" | tail -1
done
step "all done"
