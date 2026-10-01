# Whether generate()'s numbers repeat on a GPU, 2026-09-30

On a GH200 (benchmarks/gpu/respond-2026-09-29/gh200-v0.25.1, and the v0.25.0 run before it), greedy tokens did not
repeat from one call to the next in one process at 8 and 32 sequences: bf16 eager's own (transformers' model, no Glyd
code in the process) and every other mode's; nor compiled at one sequence (bf16 compiled and Glyd's default). The
eager calls at one sequence repeated, in every mode, and there Glyd exact's tokens were bf16 eager's in every
configuration. On an L4 every mode's calls repeated in every configuration.

`diag.py` finds which operation does not repeat: two identical `generate()` calls (A, B) in one process, every
`F.linear` and `scaled_dot_product_attention` call hashed on the GPU in order (its inputs' and output's words, at the
prompt's forward each linear's weight too; its pointers mod 256, strides, stream); the first call where B's output
differs from A's, with whether its inputs did; a third call under the profiler gives each call's CUDA kernels.
`--compare` sets bf16 eager's run A against exact's call by call: the linear calls whose inputs are the same and whose
outputs are not are what exact itself would change. `diag_job.sh`: the unattended job (15 minutes at most), Qwen3-8B,
bf16 eager and exact at 32, 1 and 8 sequences, then with torch's deterministic algorithms, with the attention held to
its math backend, and bf16 alone with each other attention backend and each BLAS library.

l4-check/: diag_job.sh on the AWS dev L4 with Qwen3-0.6B (a check that it runs), and backends/: bf16 and exact at 8
sequences with each attention backend held. On the L4 every run repeated call for call; exact's linear outputs were
bf16's in every call whose inputs were the same (1576-6304 calls a run), its prompt's weights bf16's words (197 of
197). The cuDNN attention backend gave other tokens than the rest (repeatably, bf16 and exact alike).

h100-sxm/: diag_job.sh on an H100 SXM (2026-10-01, tree 8b86631), Qwen3-8B. At 8 and 32 sequences, bf16 and exact alike, the
first call to differ between the two identical `generate()` calls was PyTorch's cuDNN attention kernel, called with the
same inputs (4 of 154 attention calls with the same inputs at 32 sequences; none of the 1,105 matmuls); at 1 sequence
nothing differed, nor with the attention held to the math backend (bf16 and exact: exact's tokens bf16 eager's) or to the
flash or efficient one (bf16), and torch's deterministic algorithms, cuBLAS and cuBLASLt changed nothing. Its README has
every run and the comparisons.
