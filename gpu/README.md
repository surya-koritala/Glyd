# Glyd weights on the GPU

A bf16 model's weights held compressed in VRAM, decoded on the GPU bit
for bit. A bf16 weight is a sign, 8 exponent bits and 7 mantissa bits;
in a trained model the sign and mantissa are noise, and the exponent
carries about 2.6 bits of information. Both formats keep the sign and
mantissa byte as it is and code the exponent:

| Format | Exponent | Bits a weight (Qwen2.5) | Decode |
| :--- | :--- | ---: | :--- |
| `huffman` (`pack`) | per-tensor prefix code read by counting leading zeros (as short as Huffman's on every tensor measured), 32 streams a tile | 10.88 | a lane decodes its stream in turn |
| `fast` (`pack_fast`) | 3-bit code into the tensor's 7 most common exponents, an escape to the exponent itself | 11.25 | bit operations, every weight in parallel |
| `mma` (`pack_mma`) | `fast`'s codes into the tensor's densest run of 7 exponents, laid out in the order the tensor cores take their operand | 11.25 | in registers, straight into the tensor cores' operands |

The floor for any code that sees each tensor's exponents on their own
is about 10.6 bits a weight.

For one-token steps (generation) the product is fused: `fast_gemv` and
`gemv` read the packed weights, decode them in registers and multiply,
never writing bf16 out. Otherwise a matrix is decoded into one scratch
buffer and multiplied by PyTorch (`unpack`, `fast_unpack`).

## Measured

RTX 4080 SUPER (16 GB), PyTorch 2.14, CUDA 13.0; Qwen2.5-7B-Instruct,
greedy (`e2e.py MODEL --format mma --fused --baseline`): generating for
1 to 48 sequences at once, a prompt's forward pass, and perplexity on
Wikipedia text (enwik8 from its 10th MB, 200 windows of 64 tokens):

| | bf16 | Glyd `mma` |
| :--- | ---: | ---: |
| Peak VRAM | 15.25 GB | **11.04 GB** |
| 1 sequence | 43.2 tokens/s | **55.5** (1.28x) |
| 4 sequences | 167.6 | **216.7** (1.29x) |
| 8 sequences | 331.4 | **418.3** (1.26x) |
| 16 sequences | 649.6 | **809.4** (1.25x) |
| 32 sequences | 1148.8 | **1516.6** (1.32x) |
| 48 sequences | 1664.5 | **2014.5** (1.21x) |
| Prompt of 16 tokens | 24 ms | **19 ms** |
| Prompt of 64 tokens | 27 ms | **22 ms** |
| Prompt of 128 tokens | 29 ms | 60 ms |
| Prompt of 512 tokens | 79 ms | 111 ms |
| Perplexity | 17.0015 | 17.0052 |

The weights are the model's to the bit; the product sums in another
order than cuBLAS, which moves the logits by a rounding: the next token
chosen is bf16's 98.13% of the time. bf16 against itself, two windows a
pass instead of one: perplexity 17.0153, the same next token 98.33% of
the time.

`mma_gemm` (1 to 64 tokens): a warp step is 1024 weights, 64 rows by
16 columns, one 1408-byte run in the order `mma.sync.m16n8k16` takes its
B operand, so a lane's 32 weights arrive in four loads and are decoded
in registers straight into its fragments: codes spread to exponent
bytes by two products, a warp scan finds each lane's escapes, one byte
permute puts them in place for 4 weights, and a pair of bf16s is one
permute and one rotate (each weight's byte carries its pair's other
sign). The steps are split evenly over the blocks (stream-K), a row
block's parts added in a fixed order by its last block: the same result
every run. On 7B's matrices it reads the packed weights at 95% of the
bandwidth bf16's product reaches, 1.27-1.34x faster than bf16 at one
token, 1.16-1.36x at 64. Longer prompts decode each matrix and use
PyTorch's matmul, and there bf16 is still faster.

The other formats, 128 new tokens, one sequence:

| | Peak VRAM | Tokens/s | Tokens as bf16's |
| :--- | ---: | ---: | ---: |
| bf16 | 15.25 GB | 43.3 | |
| `fast`, fused | 11.05 GB | 55.1 | 128 of 128 |
| `huffman`, fused | 10.60 GB | 52.0 | 54 of 128 |
| `fast`, decoded then PyTorch's matmul | 11.05 GB | 18.3 | 128 of 128 |

The fused `fast` product on 7B's matrices (`shapes.py`): 662–695 GB/s of
packed weights, 1.29–1.48x bf16's matrix-vector time (the small key and
value projections, which sit in L2, 0.66x). The fused `huffman` product:
1.17–1.46x (575–690 GB/s of packed weights). Profiled (Nsight Compute),
its integer pipe had been the limit (83% busy, 44 instructions a
weight): the reader is two words and a funnel shift, a code's rank is its
value plus a stored base (taken mod 32 by the shuffle), a weight's float
is assembled by byte permutes around an exponent kept in a float's place,
and the next stream word is asked for a word ahead. Rows longer than a
tile are split across tiles, their parts added in fp32 by the last tile
of the row. Tried and dropped: two interleaved streams a lane, a table
giving up to three codes a lookup. The fused product sums in
another order than cuBLAS, as any two GEMM kernels do; the decoded path
multiplies with PyTorch's own kernel and gives bf16's logits bit for bit
where a matrix is decoded whole (Qwen2.5-0.5B: logits and 128 tokens
identical; the 7B output layer is decoded in blocks to cap the scratch).

Several tokens at once in the fast format: `fast_gemm` reads it,
decodes each 64-weight step of a 64-row tile into shared memory once for
all the tile's tokens and multiplies on the tensor cores; K is split
across blocks where W has few rows, the parts added in a fixed order (the
same result every run). `e2e.py` uses it for prompts of up to 64 tokens;
longer ones decode each matrix and use PyTorch's matmul. A prompt's
forward pass, Qwen2.5-7B (`--prefill`):

| Prompt | bf16 | Glyd `fast` |
| ---: | ---: | ---: |
| 16 tokens | 24 ms | 35 ms (was 61) |
| 64 tokens | 27 ms | 40 ms (was 61) |
| 128 tokens | 29 ms | 61 ms |
| 512 tokens | 79 ms | 113 ms |
| 2048 tokens | 301 ms | 337 ms |

Past 64 tokens the fused kernel is not yet faster than decoding the matrix
and multiplying (profiled: its loads queue up and stall, at 29%
occupancy); a GEMM at cuBLAS's efficiency is what closes that.

## Running

Needs PyTorch with CUDA and nvcc (the extension builds on first import):

    python check.py model.safetensors         # every tensor packed, unpacked, compared; speeds
    python shapes.py MODEL_DIR                # fused product against bf16, one layer's matrices
    python gemm.py MODEL_DIR 1,16,64          # several tokens: every product against bf16, one layer's matrices
    python e2e.py MODEL_DIR --format mma --fused --baseline [--batch 1,8,32] [--prefill 16,64] [--ppl TEXT]
