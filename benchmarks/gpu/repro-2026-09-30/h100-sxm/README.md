# Whether generate()'s numbers repeat on an H100 SXM (2026-10-01)

`diag_job.sh` (this directory's parent) on one NVIDIA H100 SXM, Qwen3-8B: two identical `generate()` calls in one
process, every `F.linear` and `scaled_dot_product_attention` call hashed in order, for bf16 eager and for Glyd's exact
mode, as on the GH200 (`../../respond-2026-09-29/gh200-v0.25.1`), where tokens did not repeat at 8 and 32 sequences.
The job was the fifth of the session's seven (354 s, every run exit 0; `../../vllm-m6-h100-2026-10-01/session/`).

## Setup

- **Machine:** NVIDIA H100 80GB HBM3, the SXM5 (compute 9.0, sm_90a, power limit 700 W), driver 580.126.20. Xeon Platinum
  8480+ (26 CPUs), x86_64. Lambda `gpu_1x_h100_sxm5`. The hostname in `machine.txt` (the instance's address) is written
  `host`.
- **Software:** torch 2.14.1+cu130, CUDA 13.0, cuDNN 9.24.0 (92400), transformers 5.18.0, in `~/gpuenv`; the library built
  in the job for sm_90a.
- **Tree: 8b86631 (respond-bench, v0.25.1), not the release candidate.** The job's source tarball (`diag_src.tar`) was
  the one made for the GH200 validation session; the session's other jobs were rebuilt from release-0.26.0 at 7fe66a2,
  and this one was not. `diag.py` and `diag_job.sh` are the same files in release-0.26.0; the library and the glyd
  package are v0.25.1's.
- **Run:** Qwen3-8B, 128 prompt tokens and 32 new, greedy, eager, 1, 8 and 32 sequences (copies of one prompt): bf16 and
  exact at 32, 1 and 8 sequences; at 32 with torch's deterministic algorithms (`--det`) and with the attention held to its
  math backend; then bf16 alone at 32 with the efficient, flash and cuDNN attention backends held and with `--blas cublas`
  and `--blas cublaslt` (cuBLAS and cuBLASLt for torch's matmuls).
  `diag.py` hashes each call's input and output words on the GPU (and a prompt's linear weights), and names the first call
  of run B whose output differs from run A's, with whether its inputs did, and its CUDA kernels from a profiled third call.
  The runs' JSON (every call's hashes, 2.8-4.2 MB each, 48 MB in all) is not in the repository; `run-*.txt`,
  `compare-*.txt` and `summary.txt` are.

## Whether two identical calls repeat (`run-*.txt`, `summary.txt`)

| Run | Mode | Sequences | Attention, matmuls | Tokens of the two calls | Calls with the same inputs and another output | First call to differ |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- |
| `b32` | bf16 | 32 | default | differ | 4 of 154 attention calls, 0 of 1,105 matmuls | attention, step 1 (cuDNN's) |
| `b32` | exact | 32 | default | differ | 4 of 154 attention calls, 0 of 1,105 matmuls | attention, step 1 (cuDNN's) |
| `b8` | bf16 | 8 | default | differ | 3 of 288 attention calls, 0 of 2,042 matmuls | attention, step 5 (cuDNN's) |
| `b8` | exact | 8 | default | differ | 3 of 271 attention calls, 0 of 1,923 matmuls | attention, step 5 (cuDNN's) |
| `b1` | bf16, exact | 1 | default | the same | none | none |
| `b32-det` | bf16, exact | 32 | default, deterministic algorithms | differ | 4 attention calls | attention, step 1 (cuDNN's) |
| `b32-math` | bf16, exact | 32 | math | the same | none | none |
| `b32-efficient` | bf16 | 32 | efficient | the same | none | none |
| `b32-flash` | bf16 | 32 | flash | the same | none | none |
| `b32-cudnn` | bf16 | 32 | cuDNN | differ | 4 attention calls | attention, step 1 (cuDNN's) |
| `b32-cublas` | bf16 | 32 | default, cuBLAS | differ | 4 attention calls | attention, step 1 (cuDNN's) |
| `b32-cublaslt` | bf16 | 32 | default, cuBLASLt | differ | 2 attention calls | attention, step 1 (cuDNN's) |

(Each run had 9,248 calls: 8,096 linear and 1,152 attention; the counts above are the calls whose inputs were the same
in the two calls, which the first difference leaves few of. The matmul counts are read from the runs' JSON.)

The kernel of the first differing call in every run that differed is
`cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_2_16x128x128_1x4x1_cga1x1x1_kernel0_0`, PyTorch's cuDNN
attention (the default backend for these shapes here), called with the same inputs.

## Exact against bf16, call by call (`compare-*.txt`)

bf16 eager's run A against exact's run A, in each configuration:

| Run | Tokens | Linear calls with the same inputs | Of them with another output | The prompt's linear weights |
| :--- | :--- | ---: | ---: | :--- |
| `b1` | the same | 8,096 | 0 | 253 of 253 the same words |
| `b32-math` | the same | 8,096 | 0 | 253 of 253 the same words |
| `b32` | differ | 1,105 | 0 | 253 of 253 the same words |
| `b32-det` | differ | 1,105 | 0 | 253 of 253 the same words |
| `b8` | differ | 1,923 | 0 | 253 of 253 the same words |

## What it shows

- **What does not repeat on the H100 SXM is cuDNN's attention kernel**, at 8 and 32 sequences, for bf16 and for exact
  alike. Of the calls whose inputs were the same in the two calls, only attention's gave another output; no matmul
  did, bf16's or exact's.
- **Where the attention is held to the math backend, or flash or efficient, nothing differed** (at 32 sequences; flash
  and efficient bf16 alone), and with the math backend exact's tokens were bf16 eager's. At 1 sequence nothing differed
  in either mode, and exact's tokens were bf16 eager's.
- **torch's deterministic algorithms, cuBLAS and cuBLASLt** did not change it: the cuDNN kernel still differed (its
  first call in every one of those runs).
- **At 8 and 32 sequences with the default attention bf16 eager's own tokens do not repeat from one call to the next**,
  so exact's cannot be compared with them there; wherever bf16 eager's calls repeat (1 sequence, and the math backend at 32),
  exact's tokens were its tokens.
- **On the L4** (`../l4-check`) every run repeated, and the cuDNN backend gave other tokens than the rest, repeatably.
- **Not run:** the same job on a GH200 (so whether its earlier non-repetition is this kernel is not shown here),
  compiled generation, and the candidate's tree.

## Files

`run-*.txt` (each run's line), `compare-*.txt` (bf16 against exact), `summary.txt` (every run's line, the comparisons and the
steps), `steps.txt`, `job.log`, `machine.txt`, `machine-short.txt`, `env.txt`, `log/`.
