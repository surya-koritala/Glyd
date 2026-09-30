"""Round 1 (v0.25.0): this GPU's code and routes as the library gives them on the real device, against what its code
should route (glyd_gpu.cu's route_for at v0.25.0, the defaults: no GLYD_* route variable set). check_capi checks every
code's routes against 0.24's rule by names it makes up; this one reads the device's own name and class (an A10's 2086,
an H100's 90, an A100's 80, GeForce Ada's 1089, an L4's 89), through the library and the package alike.
    python r1_routes.py      (with GLYD_GPU_LIB and PYTHONPATH at the branch's library and package)"""
import os
import sys

import torch
import glyd.gpu.kernels as g
from glyd.gpu import _lib, model as gm

assert not any(os.environ.get(v) for v in gm.ROUTE_ENV), "a GLYD_* route variable is set: the defaults are checked"
# the library loaded first, as the kernels' first call loads it (the package at 975bbb1's _lib.gpu() read its functions
# before anything had: KeyError 'gpu'; since, it loads the library itself)
assert g.lib() is _lib, f"no prebuilt library loaded (GLYD_GPU_LIB={os.environ.get('GLYD_GPU_LIB')!r})"
name, cc = torch.cuda.get_device_name(), torch.cuda.get_device_capability()
lib, py = _lib.gpu(), gm.gpu_code(cc, name)
print(f"{name}, compute capability {cc[0]}.{cc[1]}: the library's code {lib}, the package's {py}")
R = "DGMWBA"  # DECODE, GEMM, MID, WG, BIG, AHEAD
w = (torch.randn(512, 1024, device="cuda") * 0.02).bfloat16()  # K a multiple of 64
got = {}
for layout, p in (("tiered", g.pack_mma(w)), ("12-bit", g.pack_mma12(w))):
    runs, M = [], 1
    while M <= 5000:
        r, last = g.route(p, lib, M)
        runs.append(f"{M}-{min(last, 5000)} {R[r]}")
        M = last + 1
    got[layout] = ", ".join(runs)
    print(f"  {layout}: {got[layout]}")
# route_for's runs at 1-5000 tokens by code, the defaults (GLYD_WG_MAX 1024, GLYD_MID_MIN 17, GLYD_DEC_MIN unset)
want = {
    90: ("1-64 G, 65-5000 D", "1-16 G, 17-1024 W, 1025-5000 D"),  # Hopper: wgmma to 1024, then decoded for cuBLAS
    80: ("1-64 G, 65-5000 B", "1-16 G, 17-128 M, 129-768 B, 769-5000 D"),  # A100: its mid kernel to 128, decoded from 769
    2086: ("1-64 G, 65-511 B, 512-5000 A", "1-16 G, 17-64 M, 65-639 B, 640-5000 A"),  # an A10: decoded ahead from 512 / 640
    86: ("1-64 G, 65-5000 B", "1-16 G, 17-64 M, 65-5000 B"),  # an A10G, A40, RTX A6000: fused throughout
    1089: ("1-64 G, 65-512 B, 513-5000 A", "1-16 G, 17-64 M, 65-1792 B, 1793-5000 A"),  # GeForce Ada
    89: ("1-64 G, 65-5000 B", "1-16 G, 17-64 M, 65-5000 B"),  # an L4, L40S, RTX 6000 Ada
}
ok = lib == py
if lib in want:
    ok &= (got["tiered"], got["12-bit"]) == want[lib]
    print(f"code {lib}: {'the routes route_for gives it' if ok else 'OTHER ROUTES than route_for gives it: ' + ' | '.join(want[lib])}")
else:
    print(f"code {lib}: no routes written down for it here (printed only)")
print("routes: " + ("as expected" if ok else "DIFFER"))
sys.exit(0 if ok else 1)
