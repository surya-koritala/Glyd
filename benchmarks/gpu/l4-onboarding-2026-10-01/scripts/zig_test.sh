#!/bin/bash
# In a container with no C compiler: does Triton build its launcher with ziglang's `zig cc` as $CC (no sudo, pip-installable)?
export PATH="$HOME/.local/bin:$PATH"
T=$(uv tool dir)/glyd/bin/python
echo "gcc: $(command -v gcc || echo none)  cc: $(command -v cc || echo none)"
echo "--- glyd doctor (no compiler)"; glyd doctor | grep -E "C compiler|Ready|Not ready"
echo "--- triton's own build without CC"
$T - <<'PY' 2>&1 | tail -3
from triton.backends.nvidia.driver import CudaUtils
try:
    CudaUtils(); print("built without a compiler?!")
except Exception as e:
    print(type(e).__name__, str(e)[:160])
PY
echo "--- install ziglang into the tool env"
uv pip install --python $T ziglang 2>&1 | tail -2
printf '#!/bin/sh\nexec %s -m ziglang cc "$@"\n' "$T" > /root/zigcc; chmod +x /root/zigcc
/root/zigcc --version 2>&1 | head -2
echo "--- triton's build with CC=/root/zigcc"
CC=/root/zigcc $T - <<'PY' 2>&1 | tail -6
import time
t = time.time()
from triton.backends.nvidia.driver import CudaUtils
u = CudaUtils()
print("built and loaded CudaUtils with zig cc in %.1f s" % (time.time() - t), u)
PY
