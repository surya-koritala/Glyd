#!/usr/bin/env bash
# Run inside the pinned CI preparation image.
set -euo pipefail
SOURCE=${1:?source root}
OUT=${2:?artifact output}
MODEL=${3:?registered model key}
PROFILE=${4:?screen or full}
HERE=$SOURCE/research/quant-benchmark
mkdir -p "$OUT/bundle/research"
BUNDLE=$OUT/bundle
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN PYTHONPATH LD_LIBRARY_PATH
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=/tmp/benchmark-hf TOKENIZERS_PARALLELISM=false
python3.12 "$HERE/environment.py" --write "$BUNDLE/base-versions.json"
bash "$SOURCE/research/lossy-2026/quality_env.sh" /tmp/benchmark-build-env "$OUT/reference-environment.json"
PY=/tmp/benchmark-build-env/bin/python
"$PY" -m pip install -r "$HERE/requirements-preparation.txt"
"$PY" -m pip check
"$PY" "$HERE/test_protocol.py"
"$PY" "$HERE/check_adapter.py"
"$PY" "$HERE/prepare.py" --model "$MODEL" --profile "$PROFILE" --out "$BUNDLE/prepared"
"$PY" "$HERE/lock_environment.py" "$BUNDLE/environment"
cp -R "$HERE" "$BUNDLE/research/quant-benchmark"
mkdir -p "$BUNDLE/research/lossy-2026"
for name in matched_quality.py qwen38-inventory.json qwen38-config.json; do
    cp "$SOURCE/research/lossy-2026/$name" "$BUNDLE/research/lossy-2026/$name"
done
printf '%s\n' "$GITHUB_SHA" > "$BUNDLE/COMMIT"
# Prove that the paid pod needs neither package resolution nor a build.
bash "$BUNDLE/research/quant-benchmark/pod_environment.sh" "$BUNDLE" /tmp/benchmark-verify-env --cpu-only
/tmp/benchmark-verify-env/bin/python "$BUNDLE/research/quant-benchmark/test_protocol.py"
/tmp/benchmark-verify-env/bin/python "$BUNDLE/research/quant-benchmark/check_adapter.py"
python3.12 - "$BUNDLE" "$OUT" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]);out=Path(sys.argv[2])
files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for directory in sorted(root.rglob('__pycache__'),reverse=True):
    import shutil
    shutil.rmtree(directory)
(root/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n' for p in files))
m=json.loads((root/'prepared/manifest.json').read_text())
(out/'preparation-summary.json').write_text(json.dumps({k:m[k] for k in ['schema','profile','model_key','source_commit','suite_sha256','request_manifest_sha256','question_count','group_counts','work']},indent=2)+'\n')
PY
tar -C "$BUNDLE" -cf "$OUT/quant-benchmark.tar" .
(cd "$OUT" && sha256sum quant-benchmark.tar > quant-benchmark.tar.sha256)
