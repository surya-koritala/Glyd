#!/bin/bash
# a clean pip resolve of glyd[vllm] at 0.27.0 and 0.28.0rc1 (a dry run: nothing is installed), and the differences
export PIP_CACHE_DIR=/opt/dlami/nvme/rel028/pipcache TMPDIR=/opt/dlami/nvme/rel028/tmp PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p $PIP_CACHE_DIR $TMPDIR /opt/dlami/nvme/rel028/resolve && cd /opt/dlami/nvme/rel028/resolve
rm -rf rv && python3 -m venv rv && . rv/bin/activate && pip install -q -U pip && pip --version && python --version && uname -m
for v in 0.27.0 0.28.0rc1; do
  pip install --dry-run --ignore-installed --report r-$v.json -q "glyd[vllm]==$v" > r-$v.txt 2>&1; echo "pip resolve $v: rc=$? ($(wc -l < r-$v.txt) lines of output)"
done
python - <<'PYEOF'
import json
def pins(v):
    r = json.load(open(f'r-{v}.json'))
    return {i['metadata']['name'].lower().replace('_', '-'): i['metadata']['version'] for i in r['install']}, r
(a, ra), (b, rb) = pins('0.27.0'), pins('0.28.0rc1')
out = [f"pip {rb['pip_version']}, CPython {rb['environment']['python_full_version']}, {rb['environment']['platform_machine']}; a dry run of pip install \"glyd[vllm]==V\" in a clean virtual environment"]
out.append(f"glyd[vllm]==0.27.0: {len(a)} packages; glyd[vllm]==0.28.0rc1: {len(b)} packages")
diff = {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
for k, (p, q) in diff.items():
    out.append(f"  differs from 0.27.0: {k}: {p} -> {q}")
out.append("  only glyd and glyd-gpu differ" if set(diff) <= {"glyd", "glyd-gpu"} else "  OTHER PACKAGES DIFFER")
open('diff.txt', 'w').write("\n".join(out) + "\n")
print("\n".join(out))
PYEOF
echo RESOLVE DONE
