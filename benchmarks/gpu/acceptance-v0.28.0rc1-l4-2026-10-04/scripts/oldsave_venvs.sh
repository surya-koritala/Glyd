#!/bin/bash
# the two environments of the old-save check: glyd[gpu]==0.27.0 and glyd[gpu]==0.28.0rc1 from PyPI, each in a virtual environment of its own, with the package versions install.sh lists as constraints (CPU and network only)
export PIP_CACHE_DIR=/opt/dlami/nvme/rel028/pipcache TMPDIR=/opt/dlami/nvme/rel028/tmp PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONPATH LD_LIBRARY_PATH GLYD_GPU_LIB
O=/opt/dlami/nvme/rel028/oldsave; mkdir -p $PIP_CACHE_DIR $TMPDIR $O && cd $O
for v in 0.27.0 0.28.0rc1; do
  [ $v = 0.28.0rc1 ] && d=v028rc1 || d=v027
  rm -rf $d && python3 -m venv $d && $d/bin/pip install -q -U pip && $d/bin/pip install "glyd[gpu]==$v" -c /tmp/constraints.txt > pip-$d.txt 2>&1; echo "pip install glyd[gpu]==$v: rc=$? ($(tail -1 pip-$d.txt | cut -c1-150))"
  $d/bin/pip freeze > freeze-$d.txt; grep -iE '^(glyd|glyd-gpu|torch|transformers|safetensors|huggingface-hub|accelerate)==' freeze-$d.txt | tr '\n' ' '; echo
done
diff <(grep -viE '^glyd(-gpu)?==' freeze-v027.txt) <(grep -viE '^glyd(-gpu)?==' freeze-v028rc1.txt) > freeze.diff && echo "the two environments differ in glyd and glyd-gpu alone" || { echo "OTHER PACKAGES DIFFER:"; cat freeze.diff; }
echo VENVS DONE
