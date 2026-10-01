# v0.26.0's release candidate on the AWS dev L4, 2026-09-30

`l4_checks.sh` on release-0.26.0 at cd42c44 (the code of the branch's head: what followed is the CHANGELOG and
benchmark logs), the library built for sm_89 with build_lib.sh's flags, under the box's lock:

- check_capi.py: 7,268 calls through both hosts bit for bit, 260,052 routes as the rule, the route SPLIT's 57,876
  pinned and its decode bit for bit on 1-200 SMs; test_gpu.py whole: 24 tests; split_stress.py --quick: 11,904
  products, 0 failures; the glyd-gpu crate's tests: 15.
- check_vllm.py --quick (vLLM 0.30's venv, this tree's package and library, C API 7): Qwen3-8B, both layouts, every
  check passed; granite-3.1-3b-a800m-instruct, every check passed.
- respond.py, Qwen3-0.6B, its four modes (2 repeats for the time to first token, 1 for the rest): every
  configuration ran, each mode's tokens the same call to call.
- `h100-sxm-route/`: the route SPLIT's extension to an H100 SXM, 2026-10-01 (889f4e3 and the extension's code): check_capi.py,
  test_gpu.py whole, split_stress.py --quick and the crate's tests on the L4, and the host route check (36,236,800 routes).
