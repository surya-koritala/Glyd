# The 12-bit layout's unpack of a whole matrix, 2026-09-29

v0.25.0's unpack of a whole matrix (for cuBLAS and exact mode) takes 3.0-5.8% longer than the 12-bit layout's before it
on an H100 SXM and 1.0-1.6% on an A10 (benchmarks/gpu/splitbyte-2026-09-29, rc/). v0.25.1 changes it on Hopper alone;
every other GPU keeps v0.25.0's.

## Results

A GH200 (gh200/): layer 10 (q, k, v and gate, up merged), the unpack of a whole matrix over main's (the 12-bit layout
before it), the median of 21, two runs:

| model | main, us | v0.25.0 | v0.25.1 |
| :-- | --: | --: | --: |
| Qwen3-8B | 283-285 | 1.035-1.037 | 0.964-0.975 |
| Qwen3-14B | 513-514 | 1.051-1.052 | 0.970 |
| Qwen3-32B | 723 | 1.059-1.060 | 0.966 |

v0.25.1's unpack moves 2.32-2.46 TB/s (the packed weights read and the bf16 matrix written), v0.25.0's 2.14-2.30. The
v0.25.1 column was measured in 73b9560's build, which has the same instructions (`sass-final.txt`).

An L4 (l4-head/, the release's code): the unpack of a whole matrix took 1.002-1.003 of main's (v0.25.0's 1.001 in the
same run); the L4 keeps v0.25.0's.

Every build unpacks to the weights' bits (the self-test and xcheck.py, 3205 calls the same bits as main's library on
the L4; `dec_e2e.py`'s sha256 lines the same in every build: 49 lines, 7 outputs on the GH200; 21 lines, 3 outputs on
the L4).

End to end, exact mode on Qwen3-8B on the L4, the GPU the bound (about 155 ms a token): main's, v0.25.0's and fd24666's
libraries took 155.0, 154.9 and 155.5 ms a token, 421.6-424.1 ms for 1024-token prompts and 1544-1548 ms for 4096-token:
all within about 1%. On the GH200 eager `generate()` is bound by its host (the Grace CPU: bf16 itself, eager, took 42 ms
a token at batch 1 against 8.3 compiled), so its step times differed between processes by more than the change: the
same kernel through the same Python took 53.02 ms in one process and 56.31 in another.

## Files

| file | what |
| :--- | :--- |
| `sass.sh`, `sass.txt`, `sass_order.py`, `sass_final.sh`, `sass-final.txt` | with no GPU (NVIDIA's `cuda:13.0.3-devel-rockylinux8` image, arm64, GCC 13): the builds' disassembly compared, for sm_80, 86, 89, 90a, 100 and 120 |
| `dec_time.py MAIN_TREE MAIN_LIB REL_LIB MODEL [--final LIB]` | layer 10's matrices (q, k, v and gate, up merged) unpacked by main's library, v0.25.0's, 73b9560's and (--final) v0.25.1's in one process: each output the weights' bits, then each call timed alone after an L2 flush, the median of 21 |
| `dec_e2e.py MODEL` | a model loaded as a user loads it, eager: exact mode's steps and long prompts, each output's sha256 |
| `dec_prof.py` | for Nsight Compute: one matrix unpacked once |
| `h100_dec.sh`, `dec_summary.py` | the unattended job (Hopper; an A10 by the same script): builds, the self-test and xcheck.py, dec_time.py, the profile, dec_e2e.py, and the summary |
| `gh200/` | h100_dec.sh on a GH200 (Lambda, aarch64 Grace host; 73b9560, bfb2c4e and db8e7b0): summary.txt; respond/: bf16 Qwen3-8B, eager and compiled, in the same session (respond.py, branch respond-bench) |
| `l4_dec.sh`, `l4/`, `l4-head/` | the AWS dev L4 (sm_89). l4/ (fd24666): the self-test, xcheck.py, check_capi.py, test_gpu.py and cargo test; dec_time.py with --final; dec_e2e.py, Qwen3-8B exact (the GPU the bound) and Qwen3-0.6B exact steps (the host the bound): l4/summary.txt, l4/host/summary.txt. l4-head/ (dc490e4, the release's code): the library as build_lib.sh builds it, the same checks, dec_time.py |
