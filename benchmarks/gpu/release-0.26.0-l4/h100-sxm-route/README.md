# The route SPLIT on an H100 SXM, checked on the AWS dev L4 and on the host, 2026-10-01

The extension of the route SPLIT to an H100 SXM (the class `GLYD_GPU_H100`, an H100 SXM's code 7090, 132 SMs; the commit
"gpu, glyd.gpu, glyd-gpu: the route SPLIT on an H100 SXM as measured") has no H100 behind these checks: they run its rule
through the library's route code, the packages' host code and the L4's fallback path. The measurement the rule rests
on is `../../option2-2026-09-29/h100-sxm-measure`.

`h100_l4_checks.sh` (here) on a fresh clone of origin's release-0.26.0 (889f4e3) with the extension's code (the commit's
`gpu/glyd_gpu.cu`, `gpu/glyd_gpu.h`, `gpu/check_capi.py`, the glyd package's `kernels.py` and `model.py`, `test_gpu.py`
and the glyd-gpu crate's `lib.rs`) applied, the library built for sm_89 with build_lib.sh's flags, under the box's lock
(queued behind other jobs; the run itself took 7 minutes). The L4's own code (3089) takes the routes without SPLIT: its
fallback path.

- **check_capi.py** (`check_capi.txt`): 7,268 calls through both hosts bit for bit identical, 260,052 routes as v0.25.1's
  rule, the route SPLIT's rule pinned on 62,328 routes (57,876 before this extension: the new code 7090 among the 14 codes),
  asked for and not, its decode bit for bit on 1-200 SMs, `GLinear` by the route as on an A100 (a split of 12 + 46 SMs
  here), and on this GPU's own code today's route with no ring (10 checks). "NVIDIA H100 80GB HBM3" is class H100 there
  and "NVIDIA H100 NVL" none.
- **test_gpu.py whole** (`test_gpu.txt`): all 24 tests, among them the classes' test (the C++ class function compiled
  and compared with `model.gpu_code` over the H100's names and their neighbours), the header test, the route pins on 7090
  and 90, `Split.measured` by name and SM count (an A100 SXM4's 108, a GH200's and an H100 SXM's 132 take the route; a MIG
  slice, an H100 PCIe's 114, an H200, an H100 NVL and other counts do not), and the split stress (1,536 products).
- **split_stress.py --quick** (`split_stress.txt`): 11,904 products, 0 failures.
- **The glyd-gpu crate** (`cargo_test.txt`): 15 tests passed, among them the header's defines (`GLYD_GPU_H100` 7000).
- **The host route check** (`host_routes.py`, `host_routes.txt`; run on the Mac, no GPU): the library's route code
  (glyd_gpu.cu's, compiled with the host's C++ compiler) against check_capi.py's rule over 10 compute capabilities (7.0
  to 12.0) with each of the 8 classes (none, GeForce, A10, L4, L40S, PCIe, GH200, H100), with and without the SPLIT flag,
  both layouts, 8 shapes and 2,831 token counts (0-10,000 and the lengths the rules end at, to 2^40), in 5 environments
  (none, `GLYD_SPLIT_MIN=-1`, the SPLIT variables moved, `GLYD_DEC_MIN`, `GLYD_WG_MIN`, `GLYD_WG_MAX` and `GLYD_MID_MIN`
  moved): 7,247,360 routes in each, 36,236,800 in all, 0 differing from the rule, and each route's last token count
  checked at its last and one past it.

`machine-short.txt` has the GPU and the tree, `steps.txt` and `job.log` the run, `log/build.txt` the library's build.
