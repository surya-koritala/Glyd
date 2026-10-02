# gpu-hopper2 as committed, on an H100 SXM, 2026-09-28

One unattended job, `h100_val.sh` (its files are beside the logs). The GPU was an H100 80GB HBM3 (SXM, 700 W). The
source was e889502's gpu/ and bindings/python; step (e) also used 479139a's glyd_gpu.cu, an earlier build.
`summary.txt` has the tables, and each step's output is beside it.

- **(a) Builds.** The library for CUDA 13.0 and for CUDA 12.8.93, sm_90a (`build-*.txt`, with `-Xptxas -v`). CUDA 12
  came from NVIDIA's 12.8.2 redistributables, sha256-checked, since the image had CUDA 13 alone.
  - Both builds: no C7515, no spills. The disassembly is in `sass-*.txt` and `log/hgmma-*.txt`.
- **(b) Checks.**
  - The full self-test (`glyd_gpu.py`) passed through both libraries (`selftest-*.txt`): 9 matrices, (8960, 128)
    among them, with products at up to 600 and up to 2100 tokens.
  - `val_check.py` ran products on the same matrices at 1-2100 tokens, with bias and without: 342 products. In
    both libraries each is within 1e-2 of fp32 and the same every run, and CUDA 12's are CUDA 13's bit for bit
    (`check-*.json`).
  - The 304 products on the earlier run's matrices match that run's outputs bit for bit wherever both took the same
    path: 288 against 3add84a's library, 16 against an earlier library's. The earlier run is
    benchmarks/gpu/h100-hopper2-cu12-2026-09-28; `prev-check-cu13-*.json` here are copies of its
    check-cu13-base.json and check-cu13-fixn1.json.
- **(c) Per layer** (`layer.txt`). Qwen3-8B's q, k, v, o, gate_up and down and Qwen3-14B's o and q, k, v, from layer
  10. Lengths 17, 128, 129, 256, 512 and 1024 tokens, against cuBLAS, through both libraries, in two rounds.
  - Qwen3-8B's layer took 0.94 / 1.16 / 1.16 / 1.24 / 1.43 / 1.38x cuBLAS's time through the CUDA 13 library, and
    0.94 / 1.16 / 1.16 / 1.24 / 1.42 / 1.37x through the CUDA 12 one.
- **(d) e2e** (`e2e-*.txt`), Qwen3-8B, one forward pass:
  - 1024 tokens: 45.0 ms against bf16's 36.9 (the CUDA 12 library: 45.2 against 37.0).
  - 256 tokens: 30.9 ms against 28.3 (30.5 against 27.9).
  - `--exact`: logits bit-identical to bf16's, 8 of 8 tokens, through both libraries.
- **(e) 479139a's library** on (8960, 128), `e-479139a.txt`.
  - At 129 tokens it left rows 8448-8703 unwritten (error 0.94): found in review, fixed in the committed code.
  - Its next calls in the same process, at 160 and 256 tokens, came out within 1e-2.
